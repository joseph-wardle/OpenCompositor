//! One view of a graph: its navigation and the gesture in progress.

use std::collections::HashSet;

use crate::connection_edit::{ConnectionEdit, Verdict};
use crate::geometry::{self, Part, Rect};
use crate::graph::{AddStamp, Connection, Graph, GroupId, NodeId, PinRef, Point, Side};
use crate::input::{Button, InputEvent, Key, Modifiers, PhysicalPoint};

const MIN_ZOOM: f32 = 0.01;
const MAX_ZOOM: f32 = 4.0;
/// The zoom factor for one wheel notch.
const ZOOM_STEP: f32 = 1.2;
/// How far, in logical pixels along either axis, the pointer moves before a press becomes a drag.
const DRAG_THRESHOLD: f32 = 3.0;
/// The fraction of the view that framed nodes fill.
const FRAME_FILL: f32 = 0.9;

/// The part of the render target a view draws into and takes input from.
///
/// The rectangle must lie within the render target and is drawn only if it has an area; hosts
/// clamp it when the target is resized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Physical pixels per logical pixel.
    pub scale_factor: f32,
}

impl Viewport {
    fn contains(&self, point: PhysicalPoint) -> bool {
        (self.x as f32..(self.x + self.width) as f32).contains(&point.x)
            && (self.y as f32..(self.y + self.height) as f32).contains(&point.y)
    }

    fn centre(&self) -> PhysicalPoint {
        PhysicalPoint {
            x: self.x as f32 + self.width as f32 / 2.0,
            y: self.y as f32 + self.height as f32 / 2.0,
        }
    }
}

/// What a view asks of the host after an event.
///
/// The host applies the requests to its document and then to the [`Graph`], or ignores them to
/// reject them, before the view draws again. A completed gesture produces one response, which
/// is one undo step; what it replaces is still in the graph until the host applies it.
#[must_use]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Response {
    /// New committed positions for the moved nodes. Empty if nothing moved.
    pub positions: Vec<(NodeId, Point)>,
    /// The new selection, if it selects different nodes. Empty deselects all.
    pub selection: Option<Vec<NodeId>>,
    /// A group to fold or unfold, from a click on its header.
    pub fold: Option<Fold>,
    /// Connections to remove and add, from a released gesture the host accepted: a wire drag whose
    /// last proposal it accepted, or the wires a cut crossed that it accepted cutting. The host
    /// applies it as it judged it, displaced connections included.
    pub connections: Option<ConnectionEdit>,
    /// The view used the event: the host should redraw and should not also treat it as its own
    /// shortcut.
    pub handled: bool,
}

/// A request to fold or unfold one group. The host applies it by updating the node's record
/// with the group's `folded` set as requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fold {
    pub node: NodeId,
    pub group: GroupId,
    /// The state requested, which differs from the record's when the click was released.
    pub folded: bool,
}

/// One view of a [`Graph`], with its own navigation and gesture.
///
/// A view belongs to one graph: showing another means another view.
#[derive(Debug)]
pub struct View {
    viewport: Viewport,
    /// The graph point at the viewport's centre.
    centre: Point,
    zoom: f32,
    /// The last pointer position, which gestures follow even after the pointer leaves.
    pointer: Option<PhysicalPoint>,
    pointer_in_target: bool,
    space_held: bool,
    y_held: bool,
    gesture: Option<Gesture>,
    /// The side of a pin a press at the pointer would grab, while no gesture is in progress.
    hover: Option<(PinRef, Side)>,
}

#[derive(Debug)]
enum Gesture {
    Pan {
        button: Button,
        last: PhysicalPoint,
    },
    /// The left button is down and has not yet moved past the drag threshold.
    Press {
        at: PhysicalPoint,
        from: Point,
        node: Option<NodeId>,
        /// The group whose header the press hit, if it was unmodified. Released there, it folds
        /// or unfolds the group and leaves the selection alone.
        header: Option<GroupId>,
        /// The side of a pin the press hit. Dragged, it starts a wire or lifts the wires into an
        /// input; released, it is a click on the node.
        port: Option<(PinRef, Side)>,
        combine: Combine,
        since: AddStamp,
    },
    Move {
        from: Point,
        nodes: HashSet<NodeId>,
        selection: SelectionEdit,
    },
    BoxSelect {
        from: Point,
        combine: Combine,
    },
    Wire(WireDrag),
    Cut(Cut),
}

/// Wires dragged by their free end: a new wire from one side of a pin toward the other side of
/// another, or every wire into an input, lifted off it.
#[derive(Debug)]
struct WireDrag {
    pressed: PinRef,
    side: Side,
    /// When the press found `pressed`, so that a node added since under its ID is not mistaken
    /// for it, nor a wire from one for a lifted wire.
    since: AddStamp,
    dragged: Dragged,
    /// Where releasing would drop the free end.
    target: Option<Target>,
    proposal: Option<Proposal>,
}

#[derive(Debug)]
enum Dragged {
    /// A wire from the pressed side.
    New,
    /// The wires into the pressed input, by the outputs they come from, which stay attached. Never
    /// empty: the drag ends when the last of them goes.
    Lifted(Vec<PinRef>),
}

/// Where a wire drag's free end would land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// The side of a pin the free end would join.
    Pin(PinRef),
    /// Empty space in the view, where lifted wires are removed and a new wire goes nowhere.
    Space,
}

/// A normalised edit and the host's verdict on it.
#[derive(Debug)]
struct Proposal {
    edit: ConnectionEdit,
    verdict: Verdict,
}

/// The host's judge, as the view's gestures call it.
type Judge<'a> = dyn FnMut(&ConnectionEdit) -> Verdict + 'a;

/// How a view draws its wire drag: as the host would apply it, or as the committed graph with a
/// loose wire to the pointer.
#[derive(Debug)]
pub(crate) enum WirePreview<'a> {
    Accepted {
        edit: &'a ConnectionEdit,
        displaced: &'a [Connection],
    },
    /// There is no proposal, or the host refused it, for `reason` if it gave one. A loose wire
    /// runs from each pin in `from`, on `side`, to the pointer.
    Loose {
        from: &'a [PinRef],
        side: Side,
        pointer: PhysicalPoint,
        refusal: Option<&'a str>,
    },
}

/// A left drag with Y held, cutting the wires its stroke crosses where this view shows them.
#[derive(Debug)]
struct Cut {
    /// In graph units, from the press, so that zooming during the stroke changes nothing it has
    /// crossed.
    stroke: Vec<Point>,
    /// Every wire the stroke has crossed, once each, in order, with when it was crossed, so that
    /// a wire from a node added since under the same ID is not mistaken for it.
    crossed: Vec<(Connection, AddStamp)>,
    /// The edit removing the crossed wires the host accepted cutting, if it accepted any.
    accepted: Option<Proposal>,
    /// The host's reason for refusing the last wire it judged, if it refused it and gave one.
    refusal: Option<String>,
}

/// How a view draws its cut: the committed graph, with the stroke and the wires it would cut.
#[derive(Debug)]
pub(crate) struct CutPreview<'a> {
    pub(crate) stroke: &'a [Point],
    /// The crossed wires the host accepted cutting, and the others it would remove with them.
    pub(crate) cut: &'a [Connection],
    pub(crate) displaced: &'a [Connection],
    pub(crate) pointer: PhysicalPoint,
    pub(crate) refusal: Option<&'a str>,
}

/// How a click or box combines its nodes with the current selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    Replace,
    Add,
    Remove,
}

impl Combine {
    fn from_modifiers(modifiers: Modifiers) -> Combine {
        if modifiers.shift {
            Combine::Add
        } else if modifiers.ctrl {
            Combine::Remove
        } else {
            Combine::Replace
        }
    }
}

#[derive(Debug)]
struct SelectionEdit {
    combine: Combine,
    nodes: Vec<NodeId>,
    /// When `nodes` were found. Any of them gone or replaced since are skipped.
    since: AddStamp,
}

impl SelectionEdit {
    fn click(node: Option<NodeId>, combine: Combine, since: AddStamp) -> SelectionEdit {
        SelectionEdit {
            combine,
            nodes: node.into_iter().collect(),
            since,
        }
    }

    /// What a press shows before release: its click, except that pressing an already selected
    /// node changes nothing until release, so that a drag moves the whole selection.
    fn press(
        graph: &Graph,
        node: Option<NodeId>,
        combine: Combine,
        since: AddStamp,
    ) -> SelectionEdit {
        let combine = match node {
            Some(id) if combine == Combine::Replace && graph.selection().contains(&id) => {
                Combine::Add
            }
            _ => combine,
        };
        SelectionEdit::click(node, combine, since)
    }

    /// The graph's current selection with this edit applied.
    fn apply(&self, graph: &Graph) -> Vec<NodeId> {
        let nodes = self
            .nodes
            .iter()
            .copied()
            .filter(|&id| graph.has_node_since(id, self.since));
        match self.combine {
            Combine::Replace => nodes.collect(),
            Combine::Add => {
                let mut selection = graph.selection().to_vec();
                let mut selected: HashSet<_> = selection.iter().copied().collect();
                selection.extend(nodes.filter(|&id| selected.insert(id)));
                selection
            }
            Combine::Remove => {
                let removed: HashSet<_> = nodes.collect();
                graph
                    .selection()
                    .iter()
                    .copied()
                    .filter(|id| !removed.contains(id))
                    .collect()
            }
        }
    }
}

impl WireDrag {
    fn new(graph: &Graph, pressed: PinRef, side: Side, since: AddStamp) -> WireDrag {
        let lifted: Vec<_> = match side {
            Side::Output => Vec::new(),
            Side::Input => graph
                .connections()
                .iter()
                .filter(|connection| connection.to == pressed)
                .map(|connection| connection.from)
                .collect(),
        };
        WireDrag {
            pressed,
            side,
            since,
            dragged: if lifted.is_empty() {
                Dragged::New
            } else {
                Dragged::Lifted(lifted)
            },
            target: None,
            proposal: None,
        }
    }

    /// The side of a pin the free end joins.
    fn free_side(&self) -> Side {
        match self.dragged {
            Dragged::New => self.side.opposite(),
            Dragged::Lifted(_) => Side::Input,
        }
    }

    /// The pins the dragged wires stay attached to, all on the returned side.
    fn attached(&self) -> (&[PinRef], Side) {
        match &self.dragged {
            Dragged::New => (std::slice::from_ref(&self.pressed), self.side),
            Dragged::Lifted(outputs) => (outputs, Side::Output),
        }
    }

    /// Forgets lifted wires the graph no longer has, and returns whether the drag still holds
    /// any wire: the pressed side and, for a lift, at least one of its wires must remain.
    fn still_holds(&mut self, graph: &Graph) -> bool {
        if !graph.has_port_since(self.pressed, self.side, self.since) {
            return false;
        }
        let Dragged::Lifted(outputs) = &mut self.dragged else {
            return true;
        };
        let input = self.pressed;
        outputs.retain(|&from| {
            graph.has_node_since(from.node, self.since)
                && graph.contains_connection(Connection { from, to: input })
        });
        !outputs.is_empty()
    }

    /// The edit dropping the free end at `target`, if it would change the graph. Lifted wires all
    /// move or are all removed.
    fn edit(&self, graph: &Graph, target: Target) -> Option<ConnectionEdit> {
        let edit = match (&self.dragged, target) {
            (Dragged::New, Target::Space) => return None,
            (Dragged::New, Target::Pin(target)) => {
                let connection = match self.side {
                    Side::Output => Connection {
                        from: self.pressed,
                        to: target,
                    },
                    Side::Input => Connection {
                        from: target,
                        to: self.pressed,
                    },
                };
                ConnectionEdit {
                    remove: Vec::new(),
                    add: vec![connection],
                }
            }
            (Dragged::Lifted(outputs), target) => {
                let into = |to| outputs.iter().map(move |&from| Connection { from, to });
                ConnectionEdit {
                    remove: into(self.pressed).collect(),
                    add: match target {
                        Target::Pin(target) => into(target).collect(),
                        Target::Space => Vec::new(),
                    },
                }
            }
        };
        edit.normalised(graph)
    }

    /// Makes `edit` the proposal, asking the host about it only if it differs from the last.
    fn propose(&mut self, edit: Option<ConnectionEdit>, judge: &mut Judge<'_>) {
        if self.proposal.as_ref().map(|proposal| &proposal.edit) != edit.as_ref() {
            self.proposal = edit.map(|edit| Proposal {
                verdict: judge(&edit),
                edit,
            });
        }
    }
}

impl Cut {
    fn new(from: Point) -> Cut {
        Cut {
            stroke: vec![from],
            crossed: Vec::new(),
            accepted: None,
            refusal: None,
        }
    }

    /// Extends the stroke to `to`, and asks the host about each wire it newly crosses inside
    /// `visible` and outside every node, where the wire is seen.
    fn extend(&mut self, graph: &Graph, to: Point, visible: Rect, judge: &mut Judge<'_>) {
        let from = *self.stroke.last().expect("a stroke starts at the press");
        if to == from {
            return;
        }
        self.stroke.push(to);
        let now = graph.add_stamp();
        for &connection in graph.connections() {
            let Some(control) = committed_wire(graph, connection) else {
                continue;
            };
            let seen = geometry::wire_crossings(control, from, to)
                .any(|point| visible.contains(point) && graph.node_at(point).is_none());
            if seen
                && !self
                    .crossed
                    .iter()
                    .any(|&(crossed, _)| crossed == connection)
            {
                self.crossed.push((connection, now));
                self.judge(graph, connection, judge);
            }
        }
    }

    /// Asks the host about cutting `connection` together with the wires it already accepted.
    /// Refused, it stays uncut and the others stay cut.
    fn judge(&mut self, graph: &Graph, connection: Connection, judge: &mut Judge<'_>) {
        let mut edit = self
            .accepted
            .as_ref()
            .map_or_else(ConnectionEdit::default, |accepted| accepted.edit.clone());
        edit.remove.push(connection);
        let Some(edit) = edit.normalised(graph) else {
            return;
        };
        match judge(&edit) {
            verdict @ Verdict::Accept { .. } => {
                self.accepted = Some(Proposal { edit, verdict });
                self.refusal = None;
            }
            Verdict::Refuse { reason } => self.refusal = reason,
        }
    }

    /// Asks the host about every crossed wire again, in the order crossed.
    fn judge_again(&mut self, graph: &Graph, judge: &mut Judge<'_>) {
        self.accepted = None;
        self.refusal = None;
        for index in 0..self.crossed.len() {
            let (connection, _) = self.crossed[index];
            self.judge(graph, connection, judge);
        }
    }

    /// Forgets crossed wires the graph no longer has and, if there were any, asks about the rest
    /// again.
    fn keep_up(&mut self, graph: &Graph, judge: &mut Judge<'_>) {
        if self.forget_lost(graph) {
            self.judge_again(graph, judge);
        }
    }

    /// Forgets crossed wires the graph no longer has, and returns whether there were any.
    fn forget_lost(&mut self, graph: &Graph) -> bool {
        let before = self.crossed.len();
        self.crossed.retain(|&(connection, since)| {
            graph.has_node_since(connection.from.node, since)
                && graph.has_node_since(connection.to.node, since)
                && graph.contains_connection(connection)
        });
        self.crossed.len() != before
    }
}

/// A connection's wire, with its nodes where they are committed.
fn committed_wire(graph: &Graph, connection: Connection) -> Option<[Point; 4]> {
    let end = |pin: PinRef, side| {
        let position = graph.node(pin.node)?.position;
        graph.node_layout(pin.node)?.port(position, pin.pin, side)
    };
    let output = end(connection.from, Side::Output)?;
    let input = end(connection.to, Side::Input)?;
    Some(geometry::wire(output, input))
}

impl View {
    /// A view at 100% zoom with the graph origin at its centre.
    pub fn new(viewport: Viewport) -> View {
        View {
            viewport,
            centre: Point::default(),
            zoom: 1.0,
            pointer: None,
            pointer_in_target: false,
            space_held: false,
            y_held: false,
            gesture: None,
            hover: None,
        }
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    /// Keeps the graph point at the view's centre in place.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
    }

    /// Whether a gesture is in progress. The host should keep sending pointer events while one is,
    /// even from outside the window.
    pub fn has_active_gesture(&self) -> bool {
        self.gesture.is_some()
    }

    /// Discards the gesture in progress without a response, and forgets held keys. Call it when
    /// the window loses focus.
    pub fn cancel_gesture(&mut self) {
        self.gesture = None;
        self.space_held = false;
        self.y_held = false;
    }

    /// Handles one event against the graph as it is now.
    ///
    /// While a gesture proposes a [`ConnectionEdit`], the view asks `judge` whether the host would
    /// accept it, once each time the proposal changes, and previews the verdict. The verdict is
    /// advisory: on release the view requests the edit only if the last verdict accepted it, and
    /// the host decides again when applying it. A cut asks about each wire it crosses together
    /// with those the host already accepted cutting, so that one wire the host keeps leaves the
    /// rest of the cut alone.
    pub fn handle_input(
        &mut self,
        graph: &Graph,
        event: InputEvent,
        mut judge: impl FnMut(&ConnectionEdit) -> Verdict,
    ) -> Response {
        // Keys the view does not use leave the hover alone, so that a change to the graph since
        // the last event never reports a host shortcut as handled.
        let pointer_event = !matches!(
            event,
            InputEvent::KeyPressed(_) | InputEvent::KeyReleased(_)
        );
        let mut response = match event {
            InputEvent::PointerMoved { position } => {
                self.pointer_moved(graph, position, &mut judge)
            }
            InputEvent::PointerPressed {
                position,
                button,
                modifiers,
            } => self.pointer_pressed(graph, position, button, modifiers),
            InputEvent::PointerReleased { position, button } => {
                self.pointer_released(graph, position, button, &mut judge)
            }
            InputEvent::Scrolled { position, lines } => self.scrolled(position, lines),
            InputEvent::KeyPressed(key) => self.key_pressed(graph, key),
            InputEvent::KeyReleased(key) => {
                match key {
                    Key::Space => self.space_held = false,
                    Key::Y => self.y_held = false,
                    Key::Escape | Key::H | Key::G => {}
                }
                Response::default()
            }
            InputEvent::PointerLeft => {
                self.pointer_in_target = false;
                Response::default()
            }
        };
        response.handled |= self.follow_graph(graph, &mut judge);
        if pointer_event || response.handled {
            let hover = self.side_under_pointer(graph);
            if hover != self.hover {
                self.hover = hover;
                response.handled = true;
            }
        }
        response
    }

    /// Brings the view up to date with the graph and with what the host would accept: asks the
    /// host again about the gesture's proposal, and finds the port under the pointer again.
    ///
    /// Call it for every view after changing the graph, or the policy the judge applies, other
    /// than through an event the view is handling. A stationary pointer sends no events, so the
    /// view cannot notice the change itself.
    pub fn refresh(&mut self, graph: &Graph, mut judge: impl FnMut(&ConnectionEdit) -> Verdict) {
        match &mut self.gesture {
            Some(Gesture::Wire(drag)) => {
                // Forgetting the target makes the host judge the proposal again.
                drag.target = None;
                drag.proposal = None;
            }
            Some(Gesture::Cut(cut)) => {
                cut.forget_lost(graph);
                cut.judge_again(graph, &mut judge);
            }
            _ => {}
        }
        self.follow_graph(graph, &mut judge);
        self.hover = self.side_under_pointer(graph);
    }

    /// The graph point at the viewport's centre.
    pub(crate) fn centre(&self) -> Point {
        self.centre
    }

    /// Where a node's top-left is drawn: its committed position, offset while it is being dragged.
    pub(crate) fn drawn_position(&self, graph: &Graph, id: NodeId) -> Option<Point> {
        let position = graph.node(id)?.position;
        if let Some(Gesture::Move {
            from,
            nodes,
            selection,
        }) = &self.gesture
            && nodes.contains(&id)
            && graph.has_node_since(id, selection.since)
        {
            let offset = self.drag_offset(*from);
            return Some(Point {
                x: position.x + offset.x,
                y: position.y + offset.y,
            });
        }
        Some(position)
    }

    /// The selection as drawn: the graph's, with the gesture's selection preview applied.
    pub(crate) fn drawn_selection(&self, graph: &Graph) -> Vec<NodeId> {
        match &self.gesture {
            None | Some(Gesture::Pan { .. } | Gesture::Wire(_) | Gesture::Cut(_)) => {
                graph.selection().to_vec()
            }
            Some(Gesture::Press {
                header: Some(_), ..
            }) => graph.selection().to_vec(),
            Some(Gesture::Press {
                node,
                combine,
                since,
                ..
            }) => SelectionEdit::press(graph, *node, *combine, *since).apply(graph),
            Some(Gesture::Move { selection, .. }) => selection.apply(graph),
            Some(Gesture::BoxSelect { from, combine }) => {
                self.box_selection(graph, *from, *combine).apply(graph)
            }
        }
    }

    /// The side of a pin a press at the pointer would grab, or a wire drag's target, drawn
    /// highlighted.
    pub(crate) fn hovered_side(&self) -> Option<(PinRef, Side)> {
        match &self.gesture {
            Some(Gesture::Wire(drag)) => match drag.target {
                Some(Target::Pin(pin)) => Some((pin, drag.free_side())),
                Some(Target::Space) | None => None,
            },
            _ => self.hover,
        }
    }

    pub(crate) fn wire_preview(&self) -> Option<WirePreview<'_>> {
        let Some(Gesture::Wire(drag)) = &self.gesture else {
            return None;
        };
        let (attached, side) = drag.attached();
        Some(match &drag.proposal {
            Some(Proposal {
                edit,
                verdict: Verdict::Accept { displaced },
            }) => WirePreview::Accepted { edit, displaced },
            proposal => WirePreview::Loose {
                from: attached,
                side,
                pointer: self.pointer.expect("a wire drag follows a press"),
                refusal: match proposal {
                    Some(Proposal {
                        verdict: Verdict::Refuse { reason },
                        ..
                    }) => reason.as_deref(),
                    _ => None,
                },
            },
        })
    }

    pub(crate) fn cut_preview(&self) -> Option<CutPreview<'_>> {
        let Some(Gesture::Cut(cut)) = &self.gesture else {
            return None;
        };
        let (cut_wires, displaced): (&[Connection], &[Connection]) = match &cut.accepted {
            Some(Proposal {
                edit,
                verdict: Verdict::Accept { displaced },
            }) => (&edit.remove, displaced),
            _ => (&[], &[]),
        };
        Some(CutPreview {
            stroke: &cut.stroke,
            cut: cut_wires,
            displaced,
            pointer: self.pointer.expect("a cut follows a press"),
            refusal: cut.refusal.as_deref(),
        })
    }

    /// The part of the graph the viewport shows.
    pub(crate) fn visible_rect(&self) -> Rect {
        let Viewport {
            x,
            y,
            width,
            height,
            ..
        } = self.viewport;
        let corner = |x: u32, y: u32| {
            self.to_graph(PhysicalPoint {
                x: x as f32,
                y: y as f32,
            })
        };
        Rect::from_corners(corner(x, y), corner(x + width, y + height))
    }

    /// The box being dragged out to select nodes.
    pub(crate) fn drawn_box(&self) -> Option<Rect> {
        let Some(Gesture::BoxSelect { from, .. }) = self.gesture else {
            return None;
        };
        let to = self.to_graph(self.pointer.expect("a box selection follows a press"));
        Some(Rect::from_corners(from, to))
    }

    fn pointer_moved(
        &mut self,
        graph: &Graph,
        position: PhysicalPoint,
        judge: &mut Judge<'_>,
    ) -> Response {
        self.track_pointer(position);
        self.gesture = match self.gesture.take() {
            None => return Response::default(),
            Some(Gesture::Cut(mut cut)) => {
                cut.extend(graph, self.to_graph(position), self.visible_rect(), judge);
                Some(Gesture::Cut(cut))
            }
            Some(Gesture::Pan { button, last }) => {
                let per_unit = self.physical_per_unit();
                self.centre.x -= (position.x - last.x) / per_unit;
                self.centre.y -= (position.y - last.y) / per_unit;
                Some(Gesture::Pan {
                    button,
                    last: position,
                })
            }
            // Modifiers do not change a port drag. The view follows the wire after the event.
            Some(Gesture::Press {
                at,
                port: Some((pressed, side)),
                since,
                ..
            }) if self.beyond_drag_threshold(at, position) => {
                Some(Gesture::Wire(WireDrag::new(graph, pressed, side, since)))
            }
            // Modified drags on a node will move it with its upstream or downstream nodes, so for
            // now they stay clicks.
            Some(Gesture::Press {
                at,
                from,
                node,
                combine,
                since,
                ..
            }) if self.beyond_drag_threshold(at, position)
                && (node.is_none() || combine == Combine::Replace) =>
            {
                Some(match node {
                    None => Gesture::BoxSelect { from, combine },
                    Some(_) => {
                        let selection = SelectionEdit::press(graph, node, combine, since);
                        Gesture::Move {
                            from,
                            nodes: selection.apply(graph).into_iter().collect(),
                            selection,
                        }
                    }
                })
            }
            // Moves and box selections follow the pointer when drawn.
            gesture => gesture,
        };
        Response {
            handled: true,
            ..Response::default()
        }
    }

    fn pointer_pressed(
        &mut self,
        graph: &Graph,
        position: PhysicalPoint,
        button: Button,
        modifiers: Modifiers,
    ) -> Response {
        self.track_pointer(position);
        // A press during a gesture means its release was lost.
        let cancelled = self.gesture.take().is_some();
        if !self.viewport.contains(position) {
            return Response {
                handled: cancelled,
                ..Response::default()
            };
        }
        self.gesture = Some(if button == Button::Middle || self.space_held {
            Gesture::Pan {
                button,
                last: position,
            }
        } else if self.y_held {
            Gesture::Cut(Cut::new(self.to_graph(position)))
        } else {
            let from = self.to_graph(position);
            let hit = graph.hit(from, self.port_reach());
            let combine = Combine::from_modifiers(modifiers);
            // Modified clicks on a header select, as they do on the rest of the node.
            let header = match hit {
                Some((_, Part::Header(group))) if combine == Combine::Replace => Some(group),
                _ => None,
            };
            let port = match hit {
                Some((node, Part::Port(pin, side))) => Some((PinRef { node, pin }, side)),
                _ => None,
            };
            Gesture::Press {
                at: position,
                from,
                node: hit.map(|(id, _)| id),
                header,
                port,
                combine,
                since: graph.add_stamp(),
            }
        });
        Response {
            handled: true,
            ..Response::default()
        }
    }

    fn pointer_released(
        &mut self,
        graph: &Graph,
        position: PhysicalPoint,
        button: Button,
        judge: &mut Judge<'_>,
    ) -> Response {
        self.track_pointer(position);
        let Some(gesture) = self.gesture.take_if(|gesture| gesture.button() == button) else {
            return Response {
                handled: self.gesture.is_some(),
                ..Response::default()
            };
        };
        let mut fold = None;
        let mut connections = None;
        let (positions, after) = match gesture {
            Gesture::Pan { .. } => (Vec::new(), None),
            // Released anywhere, a cut removes what the host accepted cutting.
            Gesture::Cut(mut cut) => {
                cut.extend(graph, self.to_graph(position), self.visible_rect(), judge);
                cut.keep_up(graph, judge);
                connections = cut.accepted.map(|accepted| accepted.edit);
                (Vec::new(), None)
            }
            Gesture::Wire(mut drag) => {
                // The graph may have changed since the last event without a refresh.
                if drag.still_holds(graph) {
                    let edit = self
                        .drop_under_pointer(graph, drag.free_side())
                        .and_then(|target| drag.edit(graph, target));
                    drag.propose(edit, judge);
                    connections = drag
                        .proposal
                        .filter(|proposal| matches!(proposal.verdict, Verdict::Accept { .. }))
                        .map(|proposal| proposal.edit);
                }
                (Vec::new(), None)
            }
            Gesture::Press {
                node,
                header: Some(group),
                since,
                ..
            } => {
                // The host may have removed the node or the group since the press.
                fold = node
                    .filter(|&id| graph.has_node_since(id, since))
                    .and_then(|id| {
                        let group = graph.node(id)?.group(group)?;
                        Some(Fold {
                            node: id,
                            group: group.id,
                            folded: !group.folded,
                        })
                    });
                (Vec::new(), None)
            }
            Gesture::Press {
                node,
                combine,
                since,
                ..
            } => {
                let click = SelectionEdit::click(node, combine, since);
                (Vec::new(), Some(click.apply(graph)))
            }
            Gesture::Move {
                from,
                nodes,
                selection,
            } => {
                let offset = self.drag_offset(from);
                let positions = if offset == Point::default() {
                    Vec::new()
                } else {
                    graph
                        .nodes()
                        .filter(|&(id, _)| {
                            nodes.contains(&id) && graph.has_node_since(id, selection.since)
                        })
                        .map(|(id, record)| {
                            let position = Point {
                                x: record.position.x + offset.x,
                                y: record.position.y + offset.y,
                            };
                            (id, position)
                        })
                        .collect()
                };
                (positions, Some(selection.apply(graph)))
            }
            Gesture::BoxSelect { from, combine } => {
                let edit = self.box_selection(graph, from, combine);
                (Vec::new(), Some(edit.apply(graph)))
            }
        };
        Response {
            positions,
            selection: after.filter(|after| selects_other_nodes(graph, after)),
            fold,
            connections,
            handled: true,
        }
    }

    /// Zooms about the pointer, keeping the graph point under it in place.
    fn scrolled(&mut self, position: PhysicalPoint, lines: f32) -> Response {
        self.track_pointer(position);
        if !self.viewport.contains(position) {
            return Response::default();
        }
        let anchor = self.to_graph(position);
        self.zoom = (self.zoom * ZOOM_STEP.powf(lines)).clamp(MIN_ZOOM, MAX_ZOOM);
        let centre = self.viewport.centre();
        let per_unit = self.physical_per_unit();
        self.centre = Point {
            x: anchor.x - (position.x - centre.x) / per_unit,
            y: anchor.y - (position.y - centre.y) / per_unit,
        };
        Response {
            handled: true,
            ..Response::default()
        }
    }

    fn track_pointer(&mut self, position: PhysicalPoint) {
        self.pointer = Some(position);
        self.pointer_in_target = true;
    }

    fn key_pressed(&mut self, graph: &Graph, key: Key) -> Response {
        let pointer_inside = self.pointer_in_view().is_some();
        let handled = match key {
            Key::Space => {
                self.space_held = true;
                pointer_inside
            }
            Key::Y => {
                self.y_held = true;
                pointer_inside
            }
            Key::Escape => self.gesture.take().is_some(),
            Key::H if pointer_inside => {
                self.frame(graph, graph.nodes().map(|(id, _)| id));
                true
            }
            Key::G if pointer_inside => {
                self.frame(graph, graph.selection().iter().copied());
                true
            }
            Key::H | Key::G => false,
        };
        Response {
            handled,
            ..Response::default()
        }
    }

    /// Centres the nodes and zooms to fit them. Does nothing if there are none.
    fn frame(&mut self, graph: &Graph, nodes: impl Iterator<Item = NodeId>) {
        let Some(bounds) = nodes
            .filter_map(|id| graph.node_rect(id))
            .reduce(|a, b| Rect {
                min: Point {
                    x: a.min.x.min(b.min.x),
                    y: a.min.y.min(b.min.y),
                },
                max: Point {
                    x: a.max.x.max(b.max.x),
                    y: a.max.y.max(b.max.y),
                },
            })
        else {
            return;
        };
        let scale = self.viewport.scale_factor;
        let fit_width = self.viewport.width as f32 / scale / (bounds.max.x - bounds.min.x);
        let fit_height = self.viewport.height as f32 / scale / (bounds.max.y - bounds.min.y);
        self.zoom = (FRAME_FILL * fit_width.min(fit_height)).clamp(MIN_ZOOM, MAX_ZOOM);
        self.centre = Point {
            x: (bounds.min.x + bounds.max.x) / 2.0,
            y: (bounds.min.y + bounds.max.y) / 2.0,
        };
    }

    fn box_selection(&self, graph: &Graph, from: Point, combine: Combine) -> SelectionEdit {
        let to = self.to_graph(self.pointer.expect("a box selection follows a press"));
        SelectionEdit {
            combine,
            nodes: graph.nodes_touching(Rect::from_corners(from, to)).collect(),
            since: graph.add_stamp(),
        }
    }

    /// How far the pointer has moved, in graph units, since it was at `from`.
    fn drag_offset(&self, from: Point) -> Point {
        let to = self.to_graph(self.pointer.expect("a drag follows a press"));
        Point {
            x: to.x - from.x,
            y: to.y - from.y,
        }
    }

    fn beyond_drag_threshold(&self, from: PhysicalPoint, to: PhysicalPoint) -> bool {
        let threshold = DRAG_THRESHOLD * self.viewport.scale_factor;
        (to.x - from.x).abs() > threshold || (to.y - from.y).abs() > threshold
    }

    /// The side of a pin a press at the pointer would grab, if no gesture is in progress.
    fn side_under_pointer(&self, graph: &Graph) -> Option<(PinRef, Side)> {
        if self.gesture.is_some() {
            return None;
        }
        self.port_under_pointer(graph)
    }

    /// Where a wire end joining `side` would land at the pointer: on that side of the pin under
    /// it, or in empty space. Elsewhere on a node, or outside this view, it lands nowhere.
    fn drop_under_pointer(&self, graph: &Graph, side: Side) -> Option<Target> {
        let pointer = self.pointer_in_view()?;
        match graph.hit(self.to_graph(pointer), self.port_reach()) {
            None => Some(Target::Space),
            Some((node, Part::Port(pin, hit))) if hit == side => {
                Some(Target::Pin(PinRef { node, pin }))
            }
            Some(_) => None,
        }
    }

    /// The side of a pin under the pointer, if the pointer is in this view.
    fn port_under_pointer(&self, graph: &Graph) -> Option<(PinRef, Side)> {
        let pointer = self.pointer_in_view()?;
        match graph.hit(self.to_graph(pointer), self.port_reach())? {
            (node, Part::Port(pin, side)) => Some((PinRef { node, pin }, side)),
            (_, Part::Body | Part::Header(_)) => None,
        }
    }

    fn pointer_in_view(&self) -> Option<PhysicalPoint> {
        self.pointer
            .filter(|&pointer| self.pointer_in_target && self.viewport.contains(pointer))
    }

    /// Brings the gesture up to date with the graph and the pointer. Returns whether it cancelled
    /// a wire drag.
    fn follow_graph(&mut self, graph: &Graph, judge: &mut Judge<'_>) -> bool {
        if let Some(Gesture::Cut(cut)) = &mut self.gesture {
            cut.keep_up(graph, judge);
            return false;
        }
        self.follow_wire(graph, judge)
    }

    /// Moves a wire drag's target to where the pointer is, proposing the edit dropping it there,
    /// or cancels the drag if it no longer holds a wire. Returns whether it cancelled the drag.
    fn follow_wire(&mut self, graph: &Graph, judge: &mut Judge<'_>) -> bool {
        let Some(Gesture::Wire(mut drag)) = self
            .gesture
            .take_if(|gesture| matches!(gesture, Gesture::Wire(_)))
        else {
            return false;
        };
        if !drag.still_holds(graph) {
            return true;
        }
        let target = self.drop_under_pointer(graph, drag.free_side());
        if target != drag.target {
            drag.target = target;
            let edit = target.and_then(|target| drag.edit(graph, target));
            drag.propose(edit, judge);
        }
        self.gesture = Some(Gesture::Wire(drag));
        false
    }

    /// How far past a node's outline its ports are hit, in graph units.
    fn port_reach(&self) -> f32 {
        geometry::port_reach(self.logical_per_unit())
    }

    pub(crate) fn physical_per_unit(&self) -> f32 {
        self.viewport.scale_factor * self.zoom
    }

    pub(crate) fn logical_per_unit(&self) -> f32 {
        self.zoom
    }

    pub(crate) fn to_graph(&self, point: PhysicalPoint) -> Point {
        let centre = self.viewport.centre();
        let per_unit = self.physical_per_unit();
        Point {
            x: self.centre.x + (point.x - centre.x) / per_unit,
            y: self.centre.y + (point.y - centre.y) / per_unit,
        }
    }

    pub(crate) fn to_physical(&self, point: Point) -> PhysicalPoint {
        let centre = self.viewport.centre();
        let per_unit = self.physical_per_unit();
        PhysicalPoint {
            x: centre.x + (point.x - self.centre.x) * per_unit,
            y: centre.y + (point.y - self.centre.y) * per_unit,
        }
    }
}

/// Whether `after` selects different nodes from the graph's selection. Order alone is no change.
fn selects_other_nodes(graph: &Graph, after: &[NodeId]) -> bool {
    let current: HashSet<_> = graph.selection().iter().collect();
    after.len() != current.len() || after.iter().any(|id| !current.contains(id))
}

impl Gesture {
    fn button(&self) -> Button {
        match self {
            Gesture::Pan { button, .. } => *button,
            Gesture::Press { .. }
            | Gesture::Move { .. }
            | Gesture::BoxSelect { .. }
            | Gesture::Wire(_)
            | Gesture::Cut(_) => Button::Left,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection_edit::accept_all;
    use crate::geometry::{BOTTOM_PADDING, MIN_WIDTH, TITLE_HEIGHT};
    use crate::graph::{Color, Direction, Group, NodeRecord, Pin, PinId, Row};

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    /// 800 × 600 physical pixels at scale 1, so graph (0, 0) starts at physical (400, 300).
    const WHOLE: Viewport = Viewport {
        x: 0,
        y: 0,
        width: 800,
        height: 600,
        scale_factor: 1.0,
    };

    fn at(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    fn px(x: f32, y: f32) -> PhysicalPoint {
        PhysicalPoint { x, y }
    }

    /// The size of an untitled node with no rows: 100 × 30.
    const SIZE: Point = Point {
        x: MIN_WIDTH,
        y: TITLE_HEIGHT + BOTTOM_PADDING,
    };

    /// The top-left of an untitled node with no rows centred on (x, y).
    fn centred(x: f32, y: f32) -> Point {
        at(x - SIZE.x / 2.0, y - SIZE.y / 2.0)
    }

    fn untitled(position: Point) -> NodeRecord {
        NodeRecord {
            position,
            title: String::new(),
            color: GREY,
            rows: Vec::new(),
        }
    }

    /// Nodes 1, 2 and 3 centred at graph x = 0, 200 and 400, each 100 × 30.
    fn three_nodes() -> Graph {
        let mut graph = Graph::default();
        for id in 1..=3 {
            let record = untitled(centred((id - 1) as f32 * 200.0, 0.0));
            graph.add_node(NodeId(id), record).unwrap();
        }
        graph
    }

    fn press(x: f32, y: f32) -> InputEvent {
        press_with(x, y, Modifiers::default())
    }

    fn press_with(x: f32, y: f32, modifiers: Modifiers) -> InputEvent {
        InputEvent::PointerPressed {
            position: px(x, y),
            button: Button::Left,
            modifiers,
        }
    }

    fn move_to(x: f32, y: f32) -> InputEvent {
        InputEvent::PointerMoved { position: px(x, y) }
    }

    fn release(x: f32, y: f32) -> InputEvent {
        InputEvent::PointerReleased {
            position: px(x, y),
            button: Button::Left,
        }
    }

    const SHIFT: Modifiers = Modifiers {
        shift: true,
        ctrl: false,
    };
    const CTRL: Modifiers = Modifiers {
        shift: false,
        ctrl: true,
    };
    const SHIFT_CTRL: Modifiers = Modifiers {
        shift: true,
        ctrl: true,
    };

    /// Sends the events in order and returns the last response.
    fn send(view: &mut View, graph: &Graph, events: &[InputEvent]) -> Response {
        let mut last = Response::default();
        for &event in events {
            last = view.handle_input(graph, event, accept_all);
        }
        last
    }

    /// Applies a response the way a host that accepts everything would.
    fn accept(graph: &mut Graph, response: Response) {
        graph.set_node_positions(&response.positions).unwrap();
        if let Some(selection) = response.selection {
            graph.set_selection(&selection).unwrap();
        }
    }

    fn ids(ids: &[u64]) -> Vec<NodeId> {
        ids.iter().map(|&id| NodeId(id)).collect()
    }

    #[test]
    fn clicks_select_and_modified_clicks_add_and_remove() {
        let mut graph = three_nodes();
        let mut view = View::new(WHOLE);

        let response = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), release(400.0, 300.0)],
        );
        assert_eq!(response.selection, Some(ids(&[1])));
        assert_eq!(response.positions, []);
        accept(&mut graph, response);

        let click = [press_with(600.0, 300.0, SHIFT), release(600.0, 300.0)];
        let response = send(&mut view, &graph, &click);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[1, 2]));

        let click = [press_with(400.0, 300.0, CTRL), release(400.0, 300.0)];
        let response = send(&mut view, &graph, &click);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[2]));

        let click = [press_with(780.0, 300.0, SHIFT_CTRL), release(780.0, 300.0)];
        let response = send(&mut view, &graph, &click);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[2, 3]), "Shift wins over Ctrl");

        let click = [press(500.0, 300.0), release(500.0, 300.0)];
        let response = send(&mut view, &graph, &click);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), []);

        let response = send(&mut view, &graph, &click);
        assert_eq!(
            response.selection, None,
            "an unchanged selection is no request"
        );
    }

    #[test]
    fn pressing_a_selected_node_keeps_the_selection_until_a_release_without_drag() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[1, 2])).unwrap();
        let mut view = View::new(WHOLE);

        let _ = view.handle_input(&graph, press(400.0, 300.0), accept_all);
        assert_eq!(view.drawn_selection(&graph), ids(&[1, 2]));

        let response = view.handle_input(&graph, release(400.0, 300.0), accept_all);
        assert_eq!(response.selection, Some(ids(&[1])));
    }

    #[test]
    fn dragging_moves_the_selection_by_one_offset_and_requests_absolute_positions() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[1, 2])).unwrap();
        let mut view = View::new(WHOLE);

        let _ = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), move_to(430.0, 340.0)],
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(30.0, 40.0))
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(2)),
            Some(centred(230.0, 40.0))
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(3)),
            Some(centred(400.0, 0.0))
        );
        assert_eq!(graph.node(NodeId(1)).unwrap().position, centred(0.0, 0.0));

        let response = view.handle_input(&graph, release(430.0, 340.0), accept_all);
        assert_eq!(
            response.positions,
            [
                (NodeId(1), centred(30.0, 40.0)),
                (NodeId(2), centred(230.0, 40.0))
            ]
        );
        assert_eq!(response.selection, None);

        // Ignoring the response rejects it.
        assert!(!view.has_active_gesture());
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
    }

    #[test]
    fn a_drag_back_to_its_start_requests_no_move() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);

        let drag = [
            press(400.0, 300.0),
            move_to(450.0, 300.0),
            move_to(400.0, 300.0),
            release(400.0, 300.0),
        ];
        let response = send(&mut view, &graph, &drag);
        assert_eq!(response.positions, []);
        assert_eq!(response.selection, Some(ids(&[1])));
    }

    #[test]
    fn dragging_an_unselected_node_moves_and_selects_only_it() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[1])).unwrap();
        let mut view = View::new(WHOLE);

        let response = send(
            &mut view,
            &graph,
            &[
                press(600.0, 300.0),
                move_to(600.0, 320.0),
                release(600.0, 320.0),
            ],
        );
        assert_eq!(response.positions, [(NodeId(2), centred(200.0, 20.0))]);
        assert_eq!(response.selection, Some(ids(&[2])));
    }

    #[test]
    fn modified_drags_on_a_node_act_as_clicks() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);

        let drag = [
            press_with(400.0, 300.0, SHIFT),
            move_to(450.0, 300.0),
            release(450.0, 300.0),
        ];
        let response = send(&mut view, &graph, &drag);
        assert_eq!(response.positions, []);
        assert_eq!(response.selection, Some(ids(&[1])));
    }

    #[test]
    fn a_press_becomes_a_drag_beyond_three_logical_pixels_on_either_axis() {
        let graph = three_nodes();
        let mut view = View::new(Viewport {
            scale_factor: 2.0,
            ..WHOLE
        });

        let _ = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), move_to(406.0, 294.0)],
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
        let response = view.handle_input(&graph, release(406.0, 294.0), accept_all);
        assert_eq!(response.positions, []);

        let _ = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), move_to(400.0, 307.0)],
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 3.5))
        );
    }

    #[test]
    fn cancelling_discards_previews_and_reports_nothing_but_keeps_navigation() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[3])).unwrap();
        let mut view = View::new(WHOLE);

        let _ = send(
            &mut view,
            &graph,
            &[
                press(400.0, 300.0),
                move_to(450.0, 300.0),
                InputEvent::Scrolled {
                    position: px(450.0, 300.0),
                    lines: 1.0,
                },
            ],
        );
        assert_eq!(view.drawn_selection(&graph), ids(&[1]));

        let response = view.handle_input(&graph, InputEvent::KeyPressed(Key::Escape), accept_all);
        assert_eq!(
            response,
            Response {
                handled: true,
                ..Response::default()
            }
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
        assert_eq!(view.drawn_selection(&graph), ids(&[3]));
        assert_eq!(view.zoom, 1.2);

        let response = view.handle_input(&graph, release(450.0, 300.0), accept_all);
        assert_eq!(response, Response::default());
    }

    #[test]
    fn focus_loss_forgets_a_held_space() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);

        let _ = send(
            &mut view,
            &graph,
            &[
                InputEvent::KeyPressed(Key::Space),
                press(500.0, 300.0),
                move_to(510.0, 300.0),
            ],
        );
        assert_eq!(view.centre, at(-10.0, 0.0), "Space and the left button pan");

        view.cancel_gesture();
        let response = send(
            &mut view,
            &graph,
            &[press(500.0, 300.0), move_to(510.0, 300.0)],
        );
        assert!(response.handled);
        assert_eq!(
            view.centre,
            at(-10.0, 0.0),
            "without Space the drag box-selects"
        );
    }

    #[test]
    fn a_press_during_a_gesture_cancels_it_first() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);

        let _ = send(
            &mut view,
            &graph,
            &[
                press(400.0, 300.0),
                move_to(450.0, 300.0),
                InputEvent::PointerPressed {
                    position: px(450.0, 300.0),
                    button: Button::Middle,
                    modifiers: Modifiers::default(),
                },
                move_to(440.0, 300.0),
            ],
        );
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
        assert_eq!(view.centre, at(10.0, 0.0));

        let response = view.handle_input(&graph, release(440.0, 300.0), accept_all);
        assert_eq!(
            response.positions,
            [],
            "the left release no longer ends anything"
        );
    }

    #[test]
    fn a_press_outside_the_view_cancels_its_gesture() {
        let graph = three_nodes();
        let mut view = View::new(Viewport {
            width: 400,
            ..WHOLE
        });

        let _ = send(
            &mut view,
            &graph,
            &[press(200.0, 300.0), move_to(250.0, 300.0)],
        );
        let middle_release = InputEvent::PointerReleased {
            position: px(250.0, 300.0),
            button: Button::Middle,
        };
        let response = view.handle_input(&graph, middle_release, accept_all);
        assert!(response.handled && view.has_active_gesture());

        let response = view.handle_input(&graph, press(600.0, 300.0), accept_all);
        assert!(response.handled && !view.has_active_gesture());
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
        let response = view.handle_input(&graph, release(600.0, 300.0), accept_all);
        assert_eq!(response, Response::default());
    }

    #[test]
    fn zoom_keeps_the_graph_point_under_the_pointer() {
        let graph = three_nodes();
        let mut view = View::new(Viewport {
            scale_factor: 2.0,
            ..WHOLE
        });
        let pointer = px(700.0, 100.0);
        let under = view.to_graph(pointer);

        for lines in [1.0, 2.5, -4.0] {
            let scroll = InputEvent::Scrolled {
                position: pointer,
                lines,
            };
            let response = view.handle_input(&graph, scroll, accept_all);
            assert!(response.handled);
            let now = view.to_graph(pointer);
            assert!((now.x - under.x).abs() < 1e-3 && (now.y - under.y).abs() < 1e-3);
        }
        assert!((view.zoom - 1.2f32.powf(-0.5)).abs() < 1e-6);
    }

    #[test]
    fn a_scroll_outside_the_view_is_ignored_even_during_a_gesture() {
        let graph = three_nodes();
        let mut view = View::new(Viewport {
            width: 400,
            ..WHOLE
        });

        let _ = send(
            &mut view,
            &graph,
            &[press(200.0, 300.0), move_to(250.0, 300.0)],
        );
        let scroll = InputEvent::Scrolled {
            position: px(600.0, 300.0),
            lines: 1.0,
        };
        assert!(!view.handle_input(&graph, scroll, accept_all).handled);
        assert_eq!(view.zoom, 1.0);
    }

    #[test]
    fn framing_fits_the_nodes_in_logical_pixels() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[3])).unwrap();
        // 400 × 300 logical pixels.
        let mut view = View::new(Viewport {
            scale_factor: 2.0,
            ..WHOLE
        });
        let _ = view.handle_input(&graph, move_to(10.0, 10.0), accept_all);

        // All three span 500 × 30 units, so the width limits the zoom to 0.9 × 400 / 500.
        let response = view.handle_input(&graph, InputEvent::KeyPressed(Key::H), accept_all);
        assert!(response.handled);
        assert_eq!(view.centre, at(200.0, 0.0));
        assert!((view.zoom - 0.72).abs() < 1e-6);

        // Node 3 alone fits at 0.9 × 400 / 100.
        let _ = view.handle_input(&graph, InputEvent::KeyPressed(Key::G), accept_all);
        assert_eq!(view.centre, at(400.0, 0.0));
        assert!((view.zoom - 3.6).abs() < 1e-6);

        // A wide view would fit it at 7.2×, beyond the maximum.
        view.set_viewport(Viewport {
            scale_factor: 1.0,
            ..WHOLE
        });
        let _ = view.handle_input(&graph, InputEvent::KeyPressed(Key::G), accept_all);
        assert_eq!(view.zoom, MAX_ZOOM);

        // A node a million units away would fit only below the minimum.
        graph
            .add_node(NodeId(4), untitled(centred(1_000_000.0, 0.0)))
            .unwrap();
        let _ = view.handle_input(&graph, InputEvent::KeyPressed(Key::H), accept_all);
        assert_eq!(view.zoom, MIN_ZOOM);
    }

    #[test]
    fn keys_act_only_while_the_pointer_is_in_the_target() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);
        let _ = view.handle_input(&graph, move_to(10.0, 10.0), accept_all);
        let _ = view.handle_input(&graph, InputEvent::PointerLeft, accept_all);

        for key in [Key::H, Key::G, Key::Space] {
            let response = view.handle_input(&graph, InputEvent::KeyPressed(key), accept_all);
            assert!(!response.handled, "{key:?}");
        }
        assert_eq!(view.centre, at(0.0, 0.0));

        let _ = view.handle_input(&graph, move_to(10.0, 10.0), accept_all);
        let response = view.handle_input(&graph, InputEvent::KeyPressed(Key::H), accept_all);
        assert!(response.handled);
        assert_eq!(view.centre, at(200.0, 0.0));
    }

    #[test]
    fn leaving_the_target_does_not_end_a_gesture() {
        let graph = three_nodes();
        let mut view = View::new(WHOLE);
        let _ = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), move_to(420.0, 300.0)],
        );
        let _ = view.handle_input(&graph, InputEvent::PointerLeft, accept_all);
        assert_eq!(
            view.drawn_position(&graph, NodeId(1)),
            Some(centred(20.0, 0.0))
        );

        let response = send(
            &mut view,
            &graph,
            &[move_to(900.0, 300.0), release(900.0, 300.0)],
        );
        assert_eq!(response.positions, [(NodeId(1), centred(500.0, 0.0))]);

        let _ = send(
            &mut view,
            &graph,
            &[press(100.0, 100.0), move_to(150.0, 150.0)],
        );
        let _ = view.handle_input(&graph, InputEvent::PointerLeft, accept_all);
        assert!(view.drawn_box().is_some());
    }

    #[test]
    fn a_box_selects_what_it_touches_and_modifiers_add_or_remove() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[3])).unwrap();
        let mut view = View::new(WHOLE);

        // From empty space above node 1 to inside node 2.
        let _ = send(
            &mut view,
            &graph,
            &[press(350.0, 250.0), move_to(570.0, 300.0)],
        );
        assert_eq!(view.drawn_selection(&graph), ids(&[1, 2]));
        let response = view.handle_input(&graph, release(570.0, 300.0), accept_all);
        assert_eq!(response.selection, Some(ids(&[1, 2])));
        accept(&mut graph, response);

        let shift_box = [
            press_with(750.0, 250.0, SHIFT),
            move_to(790.0, 300.0),
            release(790.0, 300.0),
        ];
        let response = send(&mut view, &graph, &shift_box);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[1, 2, 3]));

        let ctrl_box = [
            press_with(350.0, 250.0, CTRL),
            move_to(570.0, 300.0),
            release(570.0, 300.0),
        ];
        let response = send(&mut view, &graph, &ctrl_box);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[3]));

        let shift_ctrl_box = [
            press_with(350.0, 250.0, SHIFT_CTRL),
            move_to(570.0, 300.0),
            release(570.0, 300.0),
        ];
        let response = send(&mut view, &graph, &shift_ctrl_box);
        accept(&mut graph, response);
        assert_eq!(graph.selection(), ids(&[3, 1, 2]), "Shift wins over Ctrl");
    }

    #[test]
    fn reselecting_the_same_nodes_in_another_order_is_no_request() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[2, 1])).unwrap();
        let mut view = View::new(WHOLE);

        let response = send(
            &mut view,
            &graph,
            &[
                press(350.0, 250.0),
                move_to(570.0, 300.0),
                release(570.0, 300.0),
            ],
        );
        assert_eq!(response.selection, None);
        assert!(response.handled);
    }

    #[test]
    fn selection_requests_start_from_the_selection_at_release() {
        let mut graph = three_nodes();
        let mut view = View::new(WHOLE);

        let _ = view.handle_input(&graph, press_with(400.0, 300.0, SHIFT), accept_all);
        graph.set_selection(&ids(&[3])).unwrap();
        let response = view.handle_input(&graph, release(400.0, 300.0), accept_all);
        assert_eq!(response.selection, Some(ids(&[3, 1])));
    }

    #[test]
    fn a_node_deleted_during_a_drag_leaves_the_drag() {
        let mut graph = three_nodes();
        graph.set_selection(&ids(&[1, 2])).unwrap();
        let mut view = View::new(WHOLE);

        let _ = send(
            &mut view,
            &graph,
            &[press(400.0, 300.0), move_to(400.0, 350.0)],
        );
        graph.remove_node(NodeId(2)).unwrap();
        assert_eq!(view.drawn_position(&graph, NodeId(2)), None);
        assert_eq!(view.drawn_selection(&graph), ids(&[1]));

        let response = view.handle_input(&graph, release(400.0, 350.0), accept_all);
        assert_eq!(response.positions, [(NodeId(1), centred(0.0, 50.0))]);
    }

    #[test]
    fn a_reused_id_does_not_inherit_a_gesture() {
        let mut graph = three_nodes();
        let mut view = View::new(WHOLE);

        // Pressing node 2 previews it as the only selection and will drag it.
        let _ = send(
            &mut view,
            &graph,
            &[press(600.0, 300.0), move_to(600.0, 350.0)],
        );
        graph.remove_node(NodeId(2)).unwrap();
        let returned = untitled(centred(200.0, 0.0));
        graph.add_node(NodeId(2), returned).unwrap();

        assert_eq!(
            view.drawn_position(&graph, NodeId(2)),
            Some(centred(200.0, 0.0))
        );
        assert_eq!(view.drawn_selection(&graph), []);
        let response = view.handle_input(&graph, release(600.0, 350.0), accept_all);
        assert_eq!(response.positions, []);
        assert_eq!(response.selection, None);
    }

    #[test]
    fn hovering_shows_the_side_a_press_would_grab_and_a_press_there_grabs_the_node() {
        let mut graph = Graph::default();
        let mut record = untitled(at(0.0, 0.0));
        record.rows = vec![Row::Pin(Pin {
            id: PinId(1),
            label: "Size".to_owned(),
            direction: Direction::Both,
        })];
        graph.add_node(NodeId(1), record).unwrap();
        let mut view = View::new(WHOLE);
        let size = |side| {
            let pin = PinRef {
                node: NodeId(1),
                pin: PinId(1),
            };
            Some((pin, side))
        };
        // The row's middle, below the 24-unit title.
        let y = 334.0;

        assert!(
            view.handle_input(&graph, move_to(490.0, y), accept_all)
                .handled
        );
        assert_eq!(view.hovered_side(), size(Side::Output), "past the label");
        assert!(
            !view
                .handle_input(&graph, move_to(491.0, y), accept_all)
                .handled
        );
        let _ = view.handle_input(&graph, move_to(402.0, y), accept_all);
        assert_eq!(view.hovered_side(), size(Side::Input));
        let _ = view.handle_input(&graph, move_to(395.0, y), accept_all);
        assert_eq!(
            view.hovered_side(),
            size(Side::Input),
            "a reach past the edge"
        );
        assert!(
            view.handle_input(&graph, move_to(450.0, 312.0), accept_all)
                .handled
        );
        assert_eq!(view.hovered_side(), None, "the title");

        let _ = send(&mut view, &graph, &[move_to(490.0, y), press(490.0, y)]);
        assert_eq!(view.hovered_side(), None, "hidden during a gesture");
        let response = view.handle_input(&graph, release(490.0, y), accept_all);
        assert_eq!(response.selection, Some(ids(&[1])));
        assert_eq!(view.hovered_side(), size(Side::Output));
        assert!(
            view.handle_input(&graph, InputEvent::PointerLeft, accept_all)
                .handled
        );
        assert_eq!(view.hovered_side(), None);

        // Zooming in about a point just past the edge brings the reach within it.
        let _ = view.handle_input(&graph, move_to(395.0, y), accept_all);
        assert_eq!(view.hovered_side(), size(Side::Input));
        let zoom_in = InputEvent::Scrolled {
            position: px(395.0, y),
            lines: 8.0,
        };
        let _ = view.handle_input(&graph, zoom_in, accept_all);
        assert_eq!(view.hovered_side(), None);
    }

    /// Node 1 centred at the origin as in [`three_nodes`], with group 7's header as its first row
    /// and one input inside the group, and node 2 as before.
    fn grouped(folded: bool) -> Graph {
        let mut graph = three_nodes();
        let mut record = graph.node(NodeId(1)).unwrap().clone();
        record.rows = vec![Row::Group(Group {
            id: GroupId(7),
            label: "Options".to_owned(),
            folded,
            pins: vec![Pin {
                id: PinId(1),
                label: "Size".to_owned(),
                direction: Direction::Input,
            }],
        })];
        graph.update_node(NodeId(1), record).unwrap();
        graph
    }

    /// Over node 1's group header, whose row spans graph y 9 to 29 below its 24-unit title.
    const HEADER: (f32, f32) = (400.0, 320.0);

    fn set_folded(graph: &mut Graph, fold: Fold) {
        let mut record = graph.node(fold.node).unwrap().clone();
        for row in &mut record.rows {
            if let Row::Group(group) = row
                && group.id == fold.group
            {
                group.folded = fold.folded;
            }
        }
        graph.update_node(fold.node, record).unwrap();
    }

    #[test]
    fn a_click_on_a_header_requests_one_fold_and_keeps_the_selection() {
        let mut graph = grouped(false);
        graph.set_selection(&ids(&[2])).unwrap();
        let mut view = View::new(WHOLE);
        let (x, y) = HEADER;

        let _ = view.handle_input(&graph, press(x, y), accept_all);
        assert_eq!(view.drawn_selection(&graph), ids(&[2]));
        let response = view.handle_input(&graph, release(x, y), accept_all);
        let fold = Fold {
            node: NodeId(1),
            group: GroupId(7),
            folded: true,
        };
        assert_eq!(
            response,
            Response {
                fold: Some(fold),
                handled: true,
                ..Response::default()
            }
        );

        set_folded(&mut graph, fold);
        let response = send(&mut view, &graph, &[press(x, y), release(x, y)]);
        assert_eq!(
            response.fold,
            Some(Fold {
                folded: false,
                ..fold
            })
        );

        let response = send(
            &mut view,
            &graph,
            &[press(x, y - 15.0), release(x, y - 15.0)],
        );
        assert_eq!(response.fold, None, "the title is not a header");
        assert_eq!(response.selection, Some(ids(&[1])));
    }

    #[test]
    fn modified_clicks_and_drags_on_a_header_act_as_on_the_node() {
        let mut graph = grouped(false);
        graph.set_selection(&ids(&[2])).unwrap();
        let mut view = View::new(WHOLE);
        let (x, y) = HEADER;

        let shift_click = [press_with(x, y, SHIFT), release(x, y)];
        let response = send(&mut view, &graph, &shift_click);
        assert_eq!(response.fold, None);
        assert_eq!(response.selection, Some(ids(&[2, 1])));

        let drag = [press(x, y), move_to(x + 20.0, y), release(x + 20.0, y)];
        let response = send(&mut view, &graph, &drag);
        assert_eq!(response.fold, None);
        assert_eq!(response.positions, [(NodeId(1), centred(20.0, 0.0))]);
        assert_eq!(response.selection, Some(ids(&[1])));
    }

    #[test]
    fn a_header_click_requests_nothing_if_its_group_or_node_went_or_it_was_cancelled() {
        let mut graph = grouped(false);
        let mut view = View::new(WHOLE);
        let (x, y) = HEADER;
        let nothing = Response {
            handled: true,
            ..Response::default()
        };

        let _ = view.handle_input(&graph, press(x, y), accept_all);
        let mut ungrouped = graph.node(NodeId(1)).unwrap().clone();
        ungrouped.rows.clear();
        graph.update_node(NodeId(1), ungrouped).unwrap();
        assert_eq!(
            view.handle_input(&graph, release(x, y), accept_all),
            nothing
        );

        let mut graph = grouped(false);
        let _ = view.handle_input(&graph, press(x, y), accept_all);
        let record = graph.node(NodeId(1)).unwrap().clone();
        graph.remove_node(NodeId(1)).unwrap();
        graph.add_node(NodeId(1), record).unwrap();
        assert_eq!(
            view.handle_input(&graph, release(x, y), accept_all),
            nothing
        );

        let _ = view.handle_input(&graph, press(x, y), accept_all);
        let _ = view.handle_input(&graph, InputEvent::KeyPressed(Key::Escape), accept_all);
        assert_eq!(
            view.handle_input(&graph, release(x, y), accept_all),
            Response::default()
        );
    }

    /// Sends the event to every view and accepts each response, as a host would.
    fn broadcast(graph: &mut Graph, views: &mut [View], event: InputEvent) {
        for view in views {
            let response = view.handle_input(graph, event, accept_all);
            accept(graph, response);
        }
    }

    #[test]
    fn two_views_share_commits_but_not_navigation_or_previews() {
        let mut graph = three_nodes();
        let left = View::new(Viewport {
            width: 400,
            ..WHOLE
        });
        let right = View::new(Viewport {
            x: 400,
            width: 400,
            ..WHOLE
        });
        let mut views = [left, right];

        // The left view's centre is physical (200, 300), where node 1 is.
        broadcast(&mut graph, &mut views, press(200.0, 300.0));
        broadcast(&mut graph, &mut views, move_to(220.0, 300.0));
        let [left, right] = &views;
        assert!(left.has_active_gesture() && !right.has_active_gesture());
        assert_eq!(
            left.drawn_position(&graph, NodeId(1)),
            Some(centred(20.0, 0.0))
        );
        assert_eq!(
            right.drawn_position(&graph, NodeId(1)),
            Some(centred(0.0, 0.0))
        );
        assert_eq!(left.drawn_selection(&graph), ids(&[1]));
        assert_eq!(right.drawn_selection(&graph), []);

        // Released over the right view, which ignores it.
        broadcast(&mut graph, &mut views, release(620.0, 300.0));
        let [_, right] = &views;
        assert_eq!(graph.node(NodeId(1)).unwrap().position, centred(420.0, 0.0));
        assert_eq!(
            right.drawn_position(&graph, NodeId(1)),
            Some(centred(420.0, 0.0))
        );
        assert_eq!(right.drawn_selection(&graph), ids(&[1]));

        let zoom_out = InputEvent::Scrolled {
            position: px(600.0, 300.0),
            lines: -1.0,
        };
        broadcast(&mut graph, &mut views, zoom_out);
        let [left, right] = &views;
        assert_eq!((left.zoom, right.zoom), (1.0, 1.0 / 1.2));

        broadcast(&mut graph, &mut views, InputEvent::KeyPressed(Key::H));
        let [left, right] = &views;
        assert_eq!(left.centre, at(0.0, 0.0), "H acts only under the pointer");
        assert_eq!(right.centre, at(310.0, 0.0));
    }

    fn pin_ref(node: u64, pin: u64) -> PinRef {
        PinRef {
            node: NodeId(node),
            pin: PinId(pin),
        }
    }

    fn pin_row(id: u64, direction: Direction) -> Row {
        Row::Pin(Pin {
            id: PinId(id),
            label: String::new(),
            direction,
        })
    }

    /// Node 1 at the origin with output 1, its port at physical (500, 334), and node 2 at
    /// (200, 0) with inputs 1 and 2, their ports at physical (600, 334) and (600, 354).
    fn unwired() -> Graph {
        let mut graph = Graph::default();
        let mut source = untitled(at(0.0, 0.0));
        source.rows = vec![pin_row(1, Direction::Output)];
        graph.add_node(NodeId(1), source).unwrap();
        let mut target = untitled(at(200.0, 0.0));
        target.rows = vec![pin_row(1, Direction::Input), pin_row(2, Direction::Input)];
        graph.add_node(NodeId(2), target).unwrap();
        graph
    }

    const OUTPUT: (f32, f32) = (500.0, 334.0);
    const INPUT_1: (f32, f32) = (600.0, 334.0);
    const INPUT_2: (f32, f32) = (600.0, 354.0);

    fn into(input: u64) -> ConnectionEdit {
        ConnectionEdit {
            remove: Vec::new(),
            add: vec![Connection {
                from: pin_ref(1, 1),
                to: pin_ref(2, input),
            }],
        }
    }

    /// A host's connection policy that refuses wires into one pin and removing one wire, and
    /// records what it is asked.
    #[derive(Default)]
    struct Policy {
        refused: Option<PinRef>,
        kept: Option<Connection>,
        asked: Vec<ConnectionEdit>,
    }

    impl Policy {
        fn judge(&mut self) -> impl FnMut(&ConnectionEdit) -> Verdict + '_ {
            |edit| {
                self.asked.push(edit.clone());
                if edit.add.iter().any(|c| Some(c.to) == self.refused) {
                    Verdict::Refuse {
                        reason: Some("Locked".to_owned()),
                    }
                } else if edit.remove.iter().any(|&c| Some(c) == self.kept) {
                    Verdict::Refuse {
                        reason: Some("Kept".to_owned()),
                    }
                } else {
                    Verdict::Accept {
                        displaced: Vec::new(),
                    }
                }
            }
        }
    }

    /// Sends the events in order to a view judged by `policy` and returns the last response.
    fn send_judged(
        view: &mut View,
        graph: &Graph,
        policy: &mut Policy,
        events: &[InputEvent],
    ) -> Response {
        let mut last = Response::default();
        for &event in events {
            last = view.handle_input(graph, event, policy.judge());
        }
        last
    }

    /// The edit a wire drag previews as accepted, or the refusal it shows on a loose wire.
    fn preview(view: &View) -> Option<Result<ConnectionEdit, Option<String>>> {
        match view.wire_preview()? {
            WirePreview::Accepted { edit, .. } => Some(Ok(edit.clone())),
            WirePreview::Loose { refusal, .. } => Some(Err(refusal.map(str::to_owned))),
        }
    }

    #[test]
    fn a_drag_from_an_output_proposes_a_wire_to_the_input_under_the_pointer() {
        let graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = OUTPUT;

        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(x + 2.0, y)],
        );
        assert_eq!(preview(&view), None, "a press within the threshold");
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x + 4.0, y)]);
        assert_eq!(preview(&view), Some(Err(None)), "a loose wire");
        assert_eq!(view.drawn_selection(&graph), []);

        let (x, y) = INPUT_1;
        let over_input = [move_to(x, y), move_to(x + 1.0, y + 1.0)];
        let _ = send_judged(&mut view, &graph, &mut policy, &over_input);
        assert_eq!(preview(&view), Some(Ok(into(1))));
        assert_eq!(view.hovered_side(), Some((pin_ref(2, 1), Side::Input)));
        assert_eq!(policy.asked, [into(1)], "asked once per target");

        let response = send_judged(&mut view, &graph, &mut policy, &[release(x, y)]);
        assert_eq!(
            response,
            Response {
                connections: Some(into(1)),
                handled: true,
                ..Response::default()
            }
        );
        assert_eq!(policy.asked, [into(1)]);
    }

    #[test]
    fn a_drag_from_an_empty_input_proposes_a_wire_from_the_output_under_the_pointer() {
        let graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = INPUT_2;

        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(x - 10.0, y)],
        );
        let (x, y) = INPUT_1;
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, y)]);
        assert_eq!(preview(&view), Some(Err(None)), "an input joins no input");
        let (x, y) = OUTPUT;
        let response = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[move_to(x, y), release(x, y)],
        );
        assert_eq!(response.connections, Some(into(2)));
        assert_eq!(policy.asked, [into(2)]);
    }

    #[test]
    fn a_click_on_a_port_selects_its_node_and_a_modified_drag_still_draws_a_wire() {
        let graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = OUTPUT;

        let click = [press(x, y), move_to(x + 3.0, y), release(x + 3.0, y)];
        let response = send_judged(&mut view, &graph, &mut policy, &click);
        assert_eq!(response.selection, Some(ids(&[1])));
        assert_eq!(response.connections, None);

        let (to_x, to_y) = INPUT_1;
        let shift_drag = [
            press_with(x, y, SHIFT),
            move_to(to_x, to_y),
            release(to_x, to_y),
        ];
        let response = send_judged(&mut view, &graph, &mut policy, &shift_drag);
        assert_eq!(response.connections, Some(into(1)));
        assert_eq!(response.selection, None);
    }

    #[test]
    fn a_refused_or_existing_wire_shows_why_and_requests_nothing() {
        let mut graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy {
            refused: Some(pin_ref(2, 1)),
            ..Policy::default()
        };
        let (x, y) = OUTPUT;
        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(x + 10.0, y)],
        );

        let (x, y) = INPUT_1;
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, y)]);
        assert_eq!(preview(&view), Some(Err(Some("Locked".to_owned()))));
        let response = send_judged(&mut view, &graph, &mut policy, &[release(x, y)]);
        assert_eq!(response.connections, None);

        graph.add_connection(into(2).add[0]).unwrap();
        let (x, y) = OUTPUT;
        let (to_x, to_y) = INPUT_2;
        let reconnect = [press(x, y), move_to(to_x, to_y), release(to_x, to_y)];
        let response = send_judged(&mut view, &graph, &mut policy, &reconnect);
        assert_eq!(response.connections, None);
        assert_eq!(policy.asked, [into(1)], "an edit that changes nothing");
    }

    #[test]
    fn a_wire_drag_cancelled_or_released_off_a_target_requests_nothing() {
        let graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = OUTPUT;
        let (to_x, to_y) = INPUT_1;
        let drag = [press(x, y), move_to(to_x, to_y)];
        let nothing = Response {
            handled: true,
            ..Response::default()
        };

        let mut over_body = drag.to_vec();
        over_body.extend([move_to(650.0, 320.0), release(650.0, 320.0)]);
        let response = send_judged(&mut view, &graph, &mut policy, &over_body);
        assert_eq!(response, nothing);

        let mut outside = drag.to_vec();
        outside.extend([move_to(850.0, y), release(850.0, y)]);
        let response = send_judged(&mut view, &graph, &mut policy, &outside);
        assert_eq!(response, nothing, "outside the viewport");

        let middle = InputEvent::PointerPressed {
            position: px(to_x, to_y),
            button: Button::Middle,
            modifiers: Modifiers::default(),
        };
        for cancel in [InputEvent::KeyPressed(Key::Escape), middle] {
            let _ = send_judged(&mut view, &graph, &mut policy, &drag);
            let _ = send_judged(&mut view, &graph, &mut policy, &[cancel]);
            assert!(view.wire_preview().is_none());
            let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
            assert_eq!(response.connections, None);
        }

        let _ = send_judged(&mut view, &graph, &mut policy, &drag);
        view.cancel_gesture();
        let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
        assert_eq!(
            response, nothing,
            "after focus loss, with the port ringed again"
        );
    }

    #[test]
    fn zooming_mid_drag_finds_the_target_again() {
        let graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = OUTPUT;
        let (to_x, to_y) = INPUT_1;
        // Just past node 2's edge, within a port's reach only at low zoom.
        let near = [press(x, y), move_to(to_x - 5.0, to_y)];
        let _ = send_judged(&mut view, &graph, &mut policy, &near);
        assert_eq!(preview(&view), Some(Ok(into(1))));

        let zoom_in = InputEvent::Scrolled {
            position: px(to_x - 5.0, to_y),
            lines: 8.0,
        };
        let _ = send_judged(&mut view, &graph, &mut policy, &[zoom_in]);
        assert_eq!(preview(&view), Some(Err(None)));
        let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x - 5.0, to_y)]);
        assert_eq!(response.connections, None);
    }

    #[test]
    fn a_dual_role_row_starts_a_wire_from_the_side_pressed_whatever_the_zoom() {
        let mut graph = unwired();
        for (node, pin) in [(1, 1), (2, 2)] {
            let mut record = graph.node(NodeId(node)).unwrap().clone();
            record.rows[pin - 1] = pin_row(pin as u64, Direction::Both);
            graph.update_node(NodeId(node), record).unwrap();
        }
        let wire = |from: (u64, u64), to: (u64, u64)| ConnectionEdit {
            remove: Vec::new(),
            add: vec![Connection {
                from: pin_ref(from.0, from.1),
                to: pin_ref(to.0, to.1),
            }],
        };
        let out_of_1 = wire((1, 1), (2, 1));
        let into_1 = wire((2, 2), (1, 1));
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let mut drag = |view: &mut View, (x, y), (to_x, to_y)| {
            let events = [press(x, y), move_to(to_x, to_y), release(to_x, to_y)];
            send_judged(view, &graph, &mut policy, &events).connections
        };

        // Node 1's sides at physical x 400 and 500, node 2's second row's output side at 700.
        assert_eq!(drag(&mut view, OUTPUT, INPUT_1), Some(out_of_1.clone()));
        assert_eq!(
            drag(&mut view, (400.0, 334.0), (700.0, 354.0)),
            Some(into_1.clone())
        );

        // Two notches out about node 1's input side scale every distance from it by 1 / 1.44.
        let zoom_out = InputEvent::Scrolled {
            position: px(400.0, 334.0),
            lines: -2.0,
        };
        let _ = view.handle_input(&graph, zoom_out, accept_all);
        let scaled = |x: f32, y: f32| (400.0 + (x - 400.0) / 1.44, 334.0 + (y - 334.0) / 1.44);
        let (output, input_1) = (scaled(500.0, 334.0), scaled(600.0, 334.0));
        assert_eq!(drag(&mut view, output, input_1), Some(out_of_1));
        let output_2 = scaled(700.0, 354.0);
        assert_eq!(drag(&mut view, (400.0, 334.0), output_2), Some(into_1));
    }

    #[test]
    fn a_wire_drag_ends_when_its_pressed_side_goes() {
        let changes: [fn(&mut Graph); 3] = [
            |graph| graph.remove_node(NodeId(1)).unwrap(),
            |graph| {
                let record = graph.node(NodeId(1)).unwrap().clone();
                graph.remove_node(NodeId(1)).unwrap();
                graph.add_node(NodeId(1), record).unwrap();
            },
            |graph| {
                let mut record = graph.node(NodeId(1)).unwrap().clone();
                record.rows = vec![pin_row(1, Direction::Input)];
                graph.update_node(NodeId(1), record).unwrap();
            },
        ];
        for change in changes {
            let mut graph = unwired();
            let mut view = View::new(WHOLE);
            let mut policy = Policy::default();
            let (x, y) = OUTPUT;
            let (to_x, to_y) = INPUT_1;
            let drag = [press(x, y), move_to(to_x, to_y)];
            let _ = send_judged(&mut view, &graph, &mut policy, &drag);

            change(&mut graph);
            let response = send_judged(&mut view, &graph, &mut policy, &[move_to(to_x, to_y)]);
            assert!(response.handled && !view.has_active_gesture());
            let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
            assert_eq!(response, Response::default());
        }
    }

    #[test]
    fn a_refresh_asks_again_and_a_release_rechecks_an_unrefreshed_graph() {
        let mut graph = unwired();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = OUTPUT;
        let (to_x, to_y) = INPUT_1;
        let drag = [press(x, y), move_to(to_x, to_y)];
        let _ = send_judged(&mut view, &graph, &mut policy, &drag);
        assert_eq!(preview(&view), Some(Ok(into(1))));

        policy.refused = Some(pin_ref(2, 1));
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(to_x, to_y)]);
        assert_eq!(preview(&view), Some(Ok(into(1))), "stale until refreshed");
        view.refresh(&graph, policy.judge());
        assert_eq!(preview(&view), Some(Err(Some("Locked".to_owned()))));
        assert_eq!(policy.asked, [into(1), into(1)]);
        let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
        assert_eq!(response.connections, None);

        policy.refused = None;
        let _ = send_judged(&mut view, &graph, &mut policy, &drag);
        graph.add_connection(into(1).add[0]).unwrap();
        let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
        assert_eq!(response.connections, None, "the wire exists by release");
    }

    #[test]
    fn a_refresh_finds_the_hovered_port_again() {
        let mut graph = unwired();
        let mut view = View::new(WHOLE);
        let (x, y) = OUTPUT;
        let _ = view.handle_input(&graph, move_to(x, y), accept_all);
        assert_eq!(view.hovered_side(), Some((pin_ref(1, 1), Side::Output)));

        graph
            .set_node_positions(&[(NodeId(1), at(0.0, 100.0))])
            .unwrap();
        view.refresh(&graph, accept_all);
        assert_eq!(view.hovered_side(), None);
    }

    /// [`unwired`] plus node 3 below node 1 with output 1, its port at physical (500, 394), and
    /// wires from both outputs into node 2's input 1.
    fn lifting() -> Graph {
        let mut graph = unwired();
        let mut source = untitled(at(0.0, 60.0));
        source.rows = vec![pin_row(1, Direction::Output)];
        graph.add_node(NodeId(3), source).unwrap();
        for wire in lifted(1) {
            graph.add_connection(wire).unwrap();
        }
        graph
    }

    const OUTPUT_3: (f32, f32) = (500.0, 394.0);
    /// Over no node, well inside the view.
    const SPACE: (f32, f32) = (700.0, 500.0);

    /// The wires [`lifting`] lifts, as they would be into node 2's `input`.
    fn lifted(input: u64) -> [Connection; 2] {
        [1, 3].map(|node| Connection {
            from: pin_ref(node, 1),
            to: pin_ref(2, input),
        })
    }

    fn moved(input: u64) -> ConnectionEdit {
        ConnectionEdit {
            remove: lifted(1).to_vec(),
            add: lifted(input).to_vec(),
        }
    }

    fn removed() -> ConnectionEdit {
        ConnectionEdit {
            remove: lifted(1).to_vec(),
            add: Vec::new(),
        }
    }

    #[test]
    fn a_drag_from_an_occupied_input_lifts_all_its_wires_onto_another_input() {
        let graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = INPUT_1;

        let click = [press(x, y), release(x, y)];
        let response = send_judged(&mut view, &graph, &mut policy, &click);
        assert_eq!(
            response.selection,
            Some(ids(&[2])),
            "a click never disconnects"
        );
        assert_eq!(response.connections, None);

        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(x + 4.0, y)],
        );
        let Some(WirePreview::Loose { from, side, .. }) = view.wire_preview() else {
            panic!("dropping back where they were changes nothing");
        };
        assert_eq!(
            (from, side),
            (&[pin_ref(1, 1), pin_ref(3, 1)][..], Side::Output)
        );

        let (x, y) = INPUT_2;
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, y)]);
        assert_eq!(preview(&view), Some(Ok(moved(2))));
        assert_eq!(view.hovered_side(), Some((pin_ref(2, 2), Side::Input)));
        let (x, y) = OUTPUT_3;
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, y)]);
        assert_eq!(
            preview(&view),
            Some(Err(None)),
            "a lifted end joins no output"
        );
        let (x, y) = INPUT_2;
        let response = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[move_to(x, y), release(x, y)],
        );
        assert_eq!(response.connections, Some(moved(2)));
        assert_eq!(policy.asked, [moved(2), moved(2)]);

        let (to_x, to_y) = INPUT_1;
        let drop_back = [
            press(to_x, to_y),
            move_to(x, y),
            move_to(to_x, to_y),
            release(to_x, to_y),
        ];
        let response = send_judged(&mut view, &graph, &mut policy, &drop_back);
        assert_eq!(response.connections, None);
    }

    #[test]
    fn lifted_wires_are_removed_only_over_empty_space_in_the_view() {
        let graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = INPUT_1;
        let (space_x, space_y) = SPACE;
        let lift = [press(x, y), move_to(space_x, space_y)];
        let nothing = Response {
            handled: true,
            ..Response::default()
        };

        let _ = send_judged(&mut view, &graph, &mut policy, &lift);
        assert_eq!(preview(&view), Some(Ok(removed())));
        assert_eq!(view.hovered_side(), None);
        let response = send_judged(&mut view, &graph, &mut policy, &[release(space_x, space_y)]);
        assert_eq!(response.connections, Some(removed()));

        for (x, y) in [(650.0, 320.0), (850.0, 500.0)] {
            let mut released = lift.to_vec();
            released.extend([move_to(x, y), release(x, y)]);
            let response = send_judged(&mut view, &graph, &mut policy, &released);
            assert_eq!(response, nothing, "over a node body or outside the view");
        }

        let mut left = lift.to_vec();
        left.push(InputEvent::PointerLeft);
        let _ = send_judged(&mut view, &graph, &mut policy, &left);
        assert_eq!(preview(&view), Some(Err(None)), "outside the window");

        let middle = InputEvent::PointerPressed {
            position: px(space_x, space_y),
            button: Button::Middle,
            modifiers: Modifiers::default(),
        };
        for cancel in [InputEvent::KeyPressed(Key::Escape), middle] {
            let _ = send_judged(&mut view, &graph, &mut policy, &lift);
            let _ = send_judged(&mut view, &graph, &mut policy, &[cancel]);
            let response =
                send_judged(&mut view, &graph, &mut policy, &[release(space_x, space_y)]);
            assert_eq!(response.connections, None);
        }
        let _ = send_judged(&mut view, &graph, &mut policy, &lift);
        view.cancel_gesture();
        let response = send_judged(&mut view, &graph, &mut policy, &[release(space_x, space_y)]);
        assert_eq!(response.connections, None);
    }

    #[test]
    fn a_lift_is_judged_all_or_nothing() {
        let graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy {
            kept: Some(lifted(1)[1]),
            ..Policy::default()
        };
        let (x, y) = INPUT_1;
        let (space_x, space_y) = SPACE;
        let lift = [press(x, y), move_to(space_x, space_y)];

        let _ = send_judged(&mut view, &graph, &mut policy, &lift);
        assert_eq!(preview(&view), Some(Err(Some("Kept".to_owned()))));
        let response = send_judged(&mut view, &graph, &mut policy, &[release(space_x, space_y)]);
        assert_eq!(response.connections, None);
        assert_eq!(policy.asked, [removed()]);
    }

    #[test]
    fn a_lift_lets_go_of_wires_the_host_removes_and_ends_with_the_last() {
        let mut graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let (x, y) = INPUT_1;
        let (space_x, space_y) = SPACE;
        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(space_x, space_y)],
        );

        // Node 3 goes and returns under its ID with the same wire, which is not the lifted one.
        let record = graph.node(NodeId(3)).unwrap().clone();
        graph.remove_node(NodeId(3)).unwrap();
        graph.add_node(NodeId(3), record).unwrap();
        graph.add_connection(lifted(1)[1]).unwrap();
        view.refresh(&graph, policy.judge());
        let only_1 = ConnectionEdit {
            remove: vec![lifted(1)[0]],
            add: Vec::new(),
        };
        assert_eq!(preview(&view), Some(Ok(only_1.clone())));

        let (to_x, to_y) = INPUT_2;
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(to_x, to_y)]);
        graph.remove_connection(lifted(1)[0]).unwrap();
        let response = send_judged(&mut view, &graph, &mut policy, &[release(to_x, to_y)]);
        assert_eq!(
            response,
            Response {
                handled: true,
                ..Response::default()
            },
            "unrefreshed, the release finds none left"
        );

        let mut graph = lifting();
        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, y), move_to(space_x, space_y)],
        );
        graph.remove_connection(lifted(1)[0]).unwrap();
        let response = send_judged(&mut view, &graph, &mut policy, &[move_to(to_x, to_y)]);
        assert_eq!(
            preview(&view),
            Some(Ok(ConnectionEdit {
                remove: vec![lifted(1)[1]],
                add: vec![lifted(2)[1]],
            }))
        );
        assert!(response.handled);
        graph.remove_connection(lifted(1)[1]).unwrap();
        let response = send_judged(&mut view, &graph, &mut policy, &[move_to(to_x, to_y)]);
        assert!(response.handled && !view.has_active_gesture());
    }

    /// What a cut would remove and the reason for its last refusal.
    fn cut(view: &View) -> Option<(Vec<Connection>, Option<String>)> {
        let preview = view.cut_preview()?;
        Some((preview.cut.to_vec(), preview.refusal.map(str::to_owned)))
    }

    fn cutting(wires: &[Connection]) -> ConnectionEdit {
        ConnectionEdit {
            remove: wires.to_vec(),
            add: Vec::new(),
        }
    }

    /// A stroke between the outputs and node 2 in [`lifting`], down across both wires: the upper
    /// at physical y = 334 and the lower at 364, where it is half way.
    const STROKE: [(f32, f32); 2] = [(550.0, 300.0), (550.0, 420.0)];

    #[test]
    fn a_y_drag_cuts_each_wire_it_crosses_once_whatever_the_zoom() {
        let graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let [upper, lower] = lifted(1);
        let [(x, top), (_, bottom)] = STROKE;

        let hold_y = [move_to(x, top), InputEvent::KeyPressed(Key::Y)];
        let response = send_judged(&mut view, &graph, &mut policy, &hold_y);
        assert!(response.handled);
        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[press(x, top), move_to(x, 340.0)],
        );
        assert_eq!(cut(&view), Some((vec![upper], None)));
        assert_eq!(view.hovered_side(), None);

        // Zoomed out about the pointer, the lower wire's middle rises to y = 360.
        let zoom_out = InputEvent::Scrolled {
            position: px(x, 340.0),
            lines: -1.0,
        };
        let released_y = InputEvent::KeyReleased(Key::Y);
        let _ = send_judged(&mut view, &graph, &mut policy, &[zoom_out, released_y]);
        assert_eq!(cut(&view), Some((vec![upper], None)));
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, 362.0)]);
        assert_eq!(cut(&view), Some((vec![upper, lower], None)));

        let back = [move_to(x, top), move_to(x, bottom), release(x, bottom)];
        let response = send_judged(&mut view, &graph, &mut policy, &back);
        assert_eq!(response.connections, Some(cutting(&[upper, lower])));
        assert_eq!(policy.asked, [cutting(&[upper]), cutting(&[upper, lower])]);

        let without_y = [press(x, top), move_to(x, bottom), release(x, bottom)];
        let response = send_judged(&mut view, &graph, &mut policy, &without_y);
        assert_eq!(response.connections, None);
        assert_eq!(policy.asked.len(), 2);
    }

    #[test]
    fn a_cut_leaves_out_the_wires_the_host_keeps() {
        let graph = lifting();
        let mut view = View::new(WHOLE);
        let [upper, lower] = lifted(1);
        let mut policy = Policy {
            kept: Some(upper),
            ..Policy::default()
        };
        let [(x, top), (_, bottom)] = STROKE;

        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[
                InputEvent::KeyPressed(Key::Y),
                press(x, top),
                move_to(x, 340.0),
            ],
        );
        assert_eq!(cut(&view), Some((vec![], Some("Kept".to_owned()))));
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, bottom)]);
        assert_eq!(cut(&view), Some((vec![lower], None)));
        let response = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[move_to(x, top), release(x, top)],
        );
        assert_eq!(response.connections, Some(cutting(&[lower])));
        assert_eq!(policy.asked, [cutting(&[upper]), cutting(&[lower])]);

        policy = Policy {
            kept: Some(lower),
            ..Policy::default()
        };
        let stroke = [press(x, top), move_to(x, bottom), release(x, bottom)];
        let response = send_judged(&mut view, &graph, &mut policy, &stroke);
        assert_eq!(response.connections, Some(cutting(&[upper])));
        assert_eq!(policy.asked, [cutting(&[upper]), cutting(&[upper, lower])]);
    }

    #[test]
    fn a_cut_crosses_only_wires_seen_in_the_view_and_outside_nodes() {
        let mut graph = lifting();
        // Over the upper wire's middle, from physical x = 530 to 630.
        graph
            .add_node(NodeId(4), untitled(at(130.0, 20.0)))
            .unwrap();
        // Down to physical y = 350 with the graph where it was, above the lower wire's middle.
        let mut view = View::new(Viewport {
            height: 350,
            ..WHOLE
        });
        view.centre = at(0.0, -125.0);
        let mut policy = Policy::default();
        let [(x, top), (_, bottom)] = STROKE;

        let _ = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[
                InputEvent::KeyPressed(Key::Y),
                press(x, top),
                move_to(x, bottom),
            ],
        );
        assert_eq!(cut(&view), Some((vec![], None)));
        // Back up left of node 4.
        let response = send_judged(
            &mut view,
            &graph,
            &mut policy,
            &[
                move_to(520.0, bottom),
                move_to(520.0, top),
                release(520.0, top),
            ],
        );
        let [upper, _] = lifted(1);
        assert_eq!(response.connections, Some(cutting(&[upper])));
        assert_eq!(policy.asked, [cutting(&[upper])]);
    }

    #[test]
    fn one_stroke_cuts_every_wire_into_a_folded_group() {
        let mut graph = unwired();
        let mut target = graph.node(NodeId(2)).unwrap().clone();
        let pin = |id| Pin {
            id: PinId(id),
            label: String::new(),
            direction: Direction::Input,
        };
        target.rows = vec![Row::Group(Group {
            id: GroupId(1),
            label: String::new(),
            folded: true,
            pins: vec![pin(1), pin(2)],
        })];
        graph.update_node(NodeId(2), target).unwrap();
        let members = [1, 2].map(|input| Connection {
            from: pin_ref(1, 1),
            to: pin_ref(2, input),
        });
        for wire in members {
            graph.add_connection(wire).unwrap();
        }
        let mut view = View::new(WHOLE);
        let [(x, top), (_, bottom)] = STROKE;

        let response = send(
            &mut view,
            &graph,
            &[
                InputEvent::KeyPressed(Key::Y),
                press(x, top),
                move_to(x, 320.0),
                release(x, bottom),
            ],
        );
        assert_eq!(
            response.connections,
            Some(cutting(&members)),
            "crossed between the last move and the release"
        );
    }

    #[test]
    fn a_cut_ends_without_a_request_when_cancelled_or_left_with_nothing() {
        let mut graph = lifting();
        let mut view = View::new(WHOLE);
        let mut policy = Policy::default();
        let [upper, lower] = lifted(1);
        let [(x, top), (_, bottom)] = STROKE;
        let stroke = [press(x, top), move_to(x, bottom)];
        let hold_y = InputEvent::KeyPressed(Key::Y);
        let _ = send_judged(&mut view, &graph, &mut policy, &[hold_y]);

        let middle = InputEvent::PointerPressed {
            position: px(x, bottom),
            button: Button::Middle,
            modifiers: Modifiers::default(),
        };
        for cancel in [InputEvent::KeyPressed(Key::Escape), middle] {
            let _ = send_judged(&mut view, &graph, &mut policy, &stroke);
            let _ = send_judged(&mut view, &graph, &mut policy, &[cancel]);
            let response = send_judged(&mut view, &graph, &mut policy, &[release(x, bottom)]);
            assert_eq!(response.connections, None);
        }
        let _ = send_judged(&mut view, &graph, &mut policy, &stroke);
        view.cancel_gesture();
        let mut forgot_y = stroke.to_vec();
        forgot_y.push(release(x, bottom));
        let response = send_judged(&mut view, &graph, &mut policy, &forgot_y);
        assert_eq!(response.connections, None, "focus loss forgets Y");

        let _ = send_judged(&mut view, &graph, &mut policy, &[hold_y]);
        let _ = send_judged(&mut view, &graph, &mut policy, &stroke);
        // Node 3 goes and returns under its ID with the same wire, which was not crossed.
        let record = graph.node(NodeId(3)).unwrap().clone();
        graph.remove_node(NodeId(3)).unwrap();
        graph.add_node(NodeId(3), record).unwrap();
        graph.add_connection(lower).unwrap();
        policy.asked.clear();
        view.refresh(&graph, policy.judge());
        assert_eq!(cut(&view), Some((vec![upper], None)));
        assert_eq!(policy.asked, [cutting(&[upper])]);

        graph.remove_connection(upper).unwrap();
        let response = send_judged(&mut view, &graph, &mut policy, &[release(x, bottom)]);
        assert_eq!(
            response,
            Response {
                handled: true,
                ..Response::default()
            },
            "unrefreshed, the release finds nothing left to cut"
        );

        let _ = send_judged(&mut view, &graph, &mut policy, &stroke);
        assert_eq!(cut(&view), Some((vec![lower], None)));
        graph.remove_connection(lower).unwrap();
        let _ = send_judged(&mut view, &graph, &mut policy, &[move_to(x, bottom)]);
        assert_eq!(
            cut(&view),
            Some((vec![], None)),
            "unrefreshed, the next event"
        );
    }
}
