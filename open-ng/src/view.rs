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
    /// Connections to remove and add, from a released gesture whose last proposal the host
    /// accepted. The host applies it as it judged it, displaced connections included.
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
        /// The side of a pin the press hit. Dragged, it starts a wire; released, it is a click on
        /// the node.
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
}

/// A new wire dragged from one side of a pin toward the other side of another.
#[derive(Debug)]
struct WireDrag {
    origin: PinRef,
    side: Side,
    /// When the press found `origin`, so that a node added since under its ID is not mistaken
    /// for it.
    since: AddStamp,
    /// The pin under the pointer with the side opposite `side`.
    target: Option<PinRef>,
    proposal: Option<Proposal>,
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
    /// There is no proposal, or the host refused it, for `reason` if it gave one.
    Loose {
        from: PinRef,
        side: Side,
        pointer: PhysicalPoint,
        refusal: Option<&'a str>,
    },
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
    /// The edit connecting the origin to `target`, if it would change the graph.
    fn edit(&self, graph: &Graph, target: PinRef) -> Option<ConnectionEdit> {
        let connection = match self.side {
            Side::Output => Connection {
                from: self.origin,
                to: target,
            },
            Side::Input => Connection {
                from: target,
                to: self.origin,
            },
        };
        let edit = ConnectionEdit {
            remove: Vec::new(),
            add: vec![connection],
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
    }

    /// Handles one event against the graph as it is now.
    ///
    /// While a gesture proposes a [`ConnectionEdit`], the view asks `judge` whether the host would
    /// accept it, once each time the proposal changes, and previews the verdict. The verdict is
    /// advisory: on release the view requests the edit only if the last verdict accepted it, and
    /// the host decides again when applying it.
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
            InputEvent::PointerMoved { position } => self.pointer_moved(graph, position),
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
                if key == Key::Space {
                    self.space_held = false;
                }
                Response::default()
            }
            InputEvent::PointerLeft => {
                self.pointer_in_target = false;
                Response::default()
            }
        };
        response.handled |= self.follow_wire(graph, &mut judge);
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
        if let Some(Gesture::Wire(drag)) = &mut self.gesture {
            // Forgetting the target makes the host judge the proposal again.
            drag.target = None;
            drag.proposal = None;
        }
        self.follow_wire(graph, &mut judge);
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
            None | Some(Gesture::Pan { .. } | Gesture::Wire(_)) => graph.selection().to_vec(),
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
            Some(Gesture::Wire(drag)) => drag.target.map(|pin| (pin, drag.side.opposite())),
            _ => self.hover,
        }
    }

    pub(crate) fn wire_preview(&self) -> Option<WirePreview<'_>> {
        let Some(Gesture::Wire(drag)) = &self.gesture else {
            return None;
        };
        Some(match &drag.proposal {
            Some(Proposal {
                edit,
                verdict: Verdict::Accept { displaced },
            }) => WirePreview::Accepted { edit, displaced },
            proposal => WirePreview::Loose {
                from: drag.origin,
                side: drag.side,
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

    fn pointer_moved(&mut self, graph: &Graph, position: PhysicalPoint) -> Response {
        self.track_pointer(position);
        self.gesture = match self.gesture.take() {
            None => return Response::default(),
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
                port: Some((origin, side)),
                since,
                ..
            }) if self.beyond_drag_threshold(at, position) && starts_wire(graph, origin, side) => {
                Some(Gesture::Wire(WireDrag {
                    origin,
                    side,
                    since,
                    target: None,
                    proposal: None,
                }))
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
            Gesture::Wire(mut drag) => {
                // The graph may have changed since the last event without a refresh.
                if graph.has_port_since(drag.origin, drag.side, drag.since) {
                    let edit = self
                        .wire_target(graph, drag.side)
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
        let pointer_inside =
            self.pointer_in_target && self.pointer.is_some_and(|p| self.viewport.contains(p));
        let handled = match key {
            Key::Space => {
                self.space_held = true;
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

    /// The pin whose side opposite `side` is under the pointer, which a wire from `side` would
    /// join.
    fn wire_target(&self, graph: &Graph, side: Side) -> Option<PinRef> {
        self.port_under_pointer(graph)
            .filter(|&(_, target)| target == side.opposite())
            .map(|(pin, _)| pin)
    }

    /// The side of a pin under the pointer, if the pointer is in this view.
    fn port_under_pointer(&self, graph: &Graph) -> Option<(PinRef, Side)> {
        if !self.pointer_in_target {
            return None;
        }
        let pointer = self.pointer.filter(|&p| self.viewport.contains(p))?;
        match graph.hit(self.to_graph(pointer), self.port_reach())? {
            (node, Part::Port(pin, side)) => Some((PinRef { node, pin }, side)),
            (_, Part::Body | Part::Header(_)) => None,
        }
    }

    /// Moves a wire drag's target to the port under the pointer, proposing the new connection,
    /// or cancels the drag if its origin is gone. Returns whether it cancelled the drag.
    fn follow_wire(&mut self, graph: &Graph, judge: &mut Judge<'_>) -> bool {
        let Some(Gesture::Wire(mut drag)) = self
            .gesture
            .take_if(|gesture| matches!(gesture, Gesture::Wire(_)))
        else {
            return false;
        };
        if !graph.has_port_since(drag.origin, drag.side, drag.since) {
            return true;
        }
        let target = self.wire_target(graph, drag.side);
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

/// Whether a drag from this side of a pin starts a new wire. An occupied input's wires are
/// lifted rather than joined by another; lifting is not built, so a drag there moves the node.
fn starts_wire(graph: &Graph, pin: PinRef, side: Side) -> bool {
    side == Side::Output || !graph.connections().iter().any(|c| c.to == pin)
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
            | Gesture::Wire(_) => Button::Left,
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

    /// A host's connection policy that refuses wires into one pin and records what it is asked.
    #[derive(Default)]
    struct Policy {
        refused: Option<PinRef>,
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
}
