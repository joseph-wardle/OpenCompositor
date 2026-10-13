//! The host's document: the graph both views show, what the host knows beyond it, the rules it
//! applies to edits, and its undo history.
//!
//! Moves snap to a 10-unit grid. The Write node starts locked: it cannot move and its wires
//! cannot change. Inputs take one wire, and a new one replaces the old, except two kinds of
//! input that take several, modelled as hosts commonly do. The contact sheet's Images input
//! takes any number of wires, in an order only the host keeps. The merge always ends with a
//! spare input; connecting it makes it the next foreground and adds another spare. Wires that
//! would make a cycle are refused.

use std::collections::HashMap;

use open_ng::{
    Color, Connection, ConnectionEdit, Direction, Fold, Graph, Group, GroupId, NodeId, NodeRecord,
    Pin, PinId, PinRef, Point, Response, Row, Verdict,
};

const GRID: f32 = 10.0;
const MERGE: NodeId = NodeId(3);
const WRITE: NodeId = NodeId(5);
const SHEET_IMAGES: PinRef = PinRef {
    node: NodeId(6),
    pin: PinId(1),
};
const SPARE_LABEL: &str = "Add foreground";
/// Every change after `plan` has checked an edit.
const PLANNED: &str = "a planned change";

#[derive(Debug)]
pub struct Document {
    graph: Graph,
    /// The sources of each input that takes several wires, in the order the host combines them.
    /// The graph's connection order is only the order wires are drawn in, and undoing a removal
    /// adds the wire back at its end.
    sources: HashMap<PinRef, Vec<PinRef>>,
    write_locked: bool,
    undo: Vec<Step>,
    redo: Vec<Step>,
}

/// One response as applied.
#[derive(Debug)]
struct Step {
    before: State,
    after: State,
    connections: Option<Effect>,
}

/// The positions and selection a step changes.
#[derive(Debug, Default)]
struct State {
    positions: Vec<(NodeId, Point)>,
    selection: Option<Vec<NodeId>>,
}

/// Everything a connection edit changes, worked out before anything changes, so that once it is
/// planned no change to the graph can fail and none is left half done.
#[derive(Debug)]
struct Effect {
    removed: Vec<Connection>,
    /// The wires the additions replace, which the edit does not list.
    displaced: Vec<Connection>,
    /// The edit's additions the graph does not have yet.
    added: Vec<Connection>,
    records: Vec<(NodeId, Change<NodeRecord>)>,
    sources: Vec<(PinRef, Change<Vec<PinRef>>)>,
}

#[derive(Debug)]
struct Change<T> {
    before: T,
    after: T,
}

impl Document {
    pub fn new() -> Document {
        let graph = network();
        let sheet_sources = graph
            .connections()
            .iter()
            .filter(|c| c.to == SHEET_IMAGES)
            .map(|c| c.from)
            .collect();
        Document {
            graph,
            sources: HashMap::from([(SHEET_IMAGES, sheet_sources)]),
            write_locked: true,
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Whether the host would apply a connection edit now, and what else it would remove.
    pub fn judge(&self, edit: &ConnectionEdit) -> Verdict {
        match self.plan(edit) {
            Ok(effect) => Verdict::Accept {
                displaced: effect.map_or_else(Vec::new, |effect| effect.displaced),
            },
            Err(reason) => Verdict::Refuse {
                reason: Some(reason),
            },
        }
    }

    /// Applies what the host accepts of a response as one undo step, and returns whether the
    /// graph changed. A fold is applied outside the history, as Blender keeps panel toggles out
    /// of undo. A connection edit the host refuses changes nothing, and the reason is returned.
    pub fn apply(&mut self, response: Response) -> Result<bool, String> {
        let connections = match &response.connections {
            Some(edit) => self.plan(edit)?,
            None => None,
        };
        let mut changed = false;
        if let Some(fold) = response.fold {
            self.fold(fold);
            changed = true;
        }
        let mut step = Step {
            before: State::default(),
            after: State::default(),
            connections,
        };
        if let Some(selection) = response.selection {
            step.before.selection = Some(self.graph.selection().to_vec());
            step.after.selection = Some(selection);
        }
        for (id, position) in response.positions {
            let current = self
                .graph
                .node(id)
                .expect("views request only nodes that exist")
                .position;
            let snapped = Point {
                x: (position.x / GRID).round() * GRID,
                y: (position.y / GRID).round() * GRID,
            };
            let locked = self.write_locked && id == WRITE;
            if !locked && snapped != current {
                step.before.positions.push((id, current));
                step.after.positions.push((id, snapped));
            }
        }
        if step.connections.is_none()
            && step.after.positions.is_empty()
            && step.after.selection.is_none()
        {
            return Ok(changed);
        }
        step.redo(&mut self.graph, &mut self.sources);
        self.undo.push(step);
        self.redo.clear();
        Ok(true)
    }

    /// Returns whether the graph changed.
    pub fn undo(&mut self) -> bool {
        let Some(step) = self.undo.pop() else {
            return false;
        };
        step.undo(&mut self.graph, &mut self.sources);
        self.redo.push(step);
        true
    }

    /// Returns whether the graph changed.
    pub fn redo(&mut self) -> bool {
        let Some(step) = self.redo.pop() else {
            return false;
        };
        step.redo(&mut self.graph, &mut self.sources);
        self.undo.push(step);
        true
    }

    /// A change of policy, not of the document, so it stays out of the history.
    pub fn toggle_write_lock(&mut self) {
        self.write_locked = !self.write_locked;
    }

    /// The judge's verdict and the applied edit both come from here, so the preview shows what
    /// applying would do. An edit that would change nothing has no effect.
    fn plan(&self, edit: &ConnectionEdit) -> Result<Option<Effect>, String> {
        let graph = &self.graph;
        // Views propose only normalised edits, but scripts may not.
        let Some(edit) = edit.clone().normalised(graph) else {
            return Ok(None);
        };
        let added: Vec<_> = (edit.add.iter().copied())
            .filter(|&c| !graph.contains_connection(c))
            .collect();
        if !added.iter().all(|&c| can_wire(graph, c)) {
            return Err("No such output and input".to_owned());
        }

        let mut displaced = Vec::new();
        for (index, kept) in edit.add.iter().enumerate() {
            if self.sources.contains_key(&kept.to) {
                continue;
            }
            if edit.add[..index].iter().any(|other| other.to == kept.to) {
                return Err(format!("{} takes one wire", label(graph, kept.to)));
            }
            let occupants = graph.connections().iter();
            displaced.extend(
                occupants.filter(|c| {
                    c.to == kept.to && !edit.remove.contains(c) && !edit.add.contains(c)
                }),
            );
        }
        let touches_write = |c: &Connection| c.from.node == WRITE || c.to.node == WRITE;
        let changed = || edit.remove.iter().chain(&displaced).chain(&added);
        if self.write_locked && changed().any(touches_write) {
            return Err("Write is locked".to_owned());
        }
        let removed: Vec<_> = edit.remove.iter().chain(&displaced).copied().collect();
        if makes_cycle(graph, &removed, &added) {
            return Err("That would make a cycle".to_owned());
        }
        Ok(Some(Effect {
            records: self.spare_connected(&added),
            sources: self.source_changes(&removed, &added),
            removed: edit.remove,
            displaced,
            added,
        }))
    }

    /// If an addition connects the merge's spare input, the spare becomes the next foreground
    /// and a new spare under a fresh ID follows it, in the same step, so an undo leaves no spare
    /// behind and a redo brings back the same IDs.
    fn spare_connected(&self, added: &[Connection]) -> Vec<(NodeId, Change<NodeRecord>)> {
        let record = self.graph.node(MERGE).expect("the merge is never removed");
        let Some(Row::Pin(spare)) = record.rows.last() else {
            return Vec::new();
        };
        let spare = PinRef {
            node: MERGE,
            pin: spare.id,
        };
        if !added.iter().any(|c| c.to == spare) {
            return Vec::new();
        }
        // The background, every foreground and the spare.
        let inputs = record
            .pins()
            .filter(|pin| pin.direction == Direction::Input)
            .count();
        let fresh = record.pins().map(|pin| pin.id.0).max().unwrap_or(0) + 1;
        let mut after = record.clone();
        if let Some(Row::Pin(spare)) = after.rows.last_mut() {
            spare.label = format!("Foreground {}", inputs - 1);
        }
        after.rows.push(Row::Pin(Pin {
            id: PinId(fresh),
            label: SPARE_LABEL.to_owned(),
            direction: Direction::Input,
        }));
        let change = Change {
            before: record.clone(),
            after,
        };
        vec![(MERGE, change)]
    }

    /// New sources keep their order and join the end.
    fn source_changes(
        &self,
        removed: &[Connection],
        added: &[Connection],
    ) -> Vec<(PinRef, Change<Vec<PinRef>>)> {
        let mut changes = Vec::new();
        for (&input, before) in &self.sources {
            let kept = before
                .iter()
                .copied()
                .filter(|&from| !removed.contains(&Connection { from, to: input }));
            let joined = added.iter().filter(|c| c.to == input).map(|c| c.from);
            let after: Vec<_> = kept.chain(joined).collect();
            if after != *before {
                let before = before.clone();
                changes.push((input, Change { before, after }));
            }
        }
        changes
    }

    fn fold(&mut self, fold: Fold) {
        let mut record = self
            .graph
            .node(fold.node)
            .expect("views request only nodes that exist")
            .clone();
        for row in &mut record.rows {
            if let Row::Group(group) = row
                && group.id == fold.group
            {
                group.folded = fold.folded;
            }
        }
        self.graph
            .update_node(fold.node, record)
            .expect("the same pins and groups");
    }
}

impl Step {
    fn redo(&self, graph: &mut Graph, sources: &mut HashMap<PinRef, Vec<PinRef>>) {
        if let Some(effect) = &self.connections {
            effect.redo(graph, sources);
        }
        self.after.restore(graph);
    }

    fn undo(&self, graph: &mut Graph, sources: &mut HashMap<PinRef, Vec<PinRef>>) {
        self.before.restore(graph);
        if let Some(effect) = &self.connections {
            effect.undo(graph, sources);
        }
    }
}

impl State {
    fn restore(&self, graph: &mut Graph) {
        graph
            .set_node_positions(&self.positions)
            .expect("nodes are never removed");
        if let Some(selection) = &self.selection {
            graph
                .set_selection(selection)
                .expect("nodes are never removed");
        }
    }
}

impl Effect {
    fn redo(&self, graph: &mut Graph, sources: &mut HashMap<PinRef, Vec<PinRef>>) {
        for &connection in self.removed.iter().chain(&self.displaced) {
            graph.remove_connection(connection).expect(PLANNED);
        }
        for (id, record) in &self.records {
            graph.update_node(*id, record.after.clone()).expect(PLANNED);
        }
        for &connection in &self.added {
            graph.add_connection(connection).expect(PLANNED);
        }
        for (input, order) in &self.sources {
            sources.insert(*input, order.after.clone());
        }
    }

    /// The old records still have every pin a removed wire used, so restoring them removes no
    /// wire that should stay.
    fn undo(&self, graph: &mut Graph, sources: &mut HashMap<PinRef, Vec<PinRef>>) {
        for &connection in &self.added {
            graph.remove_connection(connection).expect(PLANNED);
        }
        for (id, record) in &self.records {
            graph
                .update_node(*id, record.before.clone())
                .expect(PLANNED);
        }
        for &connection in self.removed.iter().chain(&self.displaced) {
            graph.add_connection(connection).expect(PLANNED);
        }
        for (input, order) in &self.sources {
            sources.insert(*input, order.before.clone());
        }
    }
}

/// Whether both pins exist, `from` with an output side and `to` with an input side. Views propose
/// only such wires, but scripts may not.
fn can_wire(graph: &Graph, connection: Connection) -> bool {
    let direction = |pin: PinRef| Some(graph.node(pin.node)?.pin(pin.pin)?.direction);
    matches!(
        direction(connection.from),
        Some(Direction::Output | Direction::Both)
    ) && matches!(
        direction(connection.to),
        Some(Direction::Input | Direction::Both)
    )
}

fn label(graph: &Graph, pin: PinRef) -> &str {
    graph
        .node(pin.node)
        .and_then(|record| record.pin(pin.pin))
        .map_or("", |pin| &pin.label)
}

/// Whether the graph would have a cycle after the edit. It searches every wire from each
/// addition, which is fine at the example's size.
fn makes_cycle(graph: &Graph, removed: &[Connection], added: &[Connection]) -> bool {
    let wires: Vec<_> = (graph.connections().iter())
        .filter(|c| !removed.contains(c))
        .chain(added)
        .collect();
    added.iter().any(|addition| {
        let mut unvisited = vec![addition.to.node];
        let mut visited = Vec::new();
        while let Some(node) = unvisited.pop() {
            if node == addition.from.node {
                return true;
            }
            if !visited.contains(&node) {
                visited.push(node);
                let next = wires.iter().filter(|w| w.from.node == node);
                unvisited.extend(next.map(|w| w.to.node));
            }
        }
        false
    })
}

/// Two sources merged, blurred and written, from left to right around the graph origin where
/// new views are centred, and laid out on a contact sheet. Every node's first input is pin 1
/// and its output pin 2. The blur has four more pins in two groups, one folded; its size both
/// receives and supplies a value.
fn network() -> Graph {
    let pin = |id, label: &str, direction| {
        Row::Pin(Pin {
            id: PinId(id),
            label: label.to_owned(),
            direction,
        })
    };
    let input = |label| pin(1, label, Direction::Input);
    let output = pin(2, "Image", Direction::Output);
    let node = |title: &str, x, y, (r, g, b), rows| NodeRecord {
        position: Point { x, y },
        title: title.to_owned(),
        color: Color { r, g, b },
        rows,
    };
    let source = (70, 120, 170);
    let mut graph = Graph::default();
    let nodes = [
        node("Read", -340.0, -90.0, source, vec![output.clone()]),
        node("Read", -340.0, 20.0, source, vec![output.clone()]),
        node(
            "Merge",
            -180.0,
            -40.0,
            (150, 110, 60),
            vec![
                output.clone(),
                pin(3, "Background", Direction::Input),
                input("Foreground"),
                pin(4, SPARE_LABEL, Direction::Input),
            ],
        ),
        node(
            "Directional Blur",
            -20.0,
            -40.0,
            (90, 140, 90),
            vec![
                input("Image"),
                output.clone(),
                Row::Group(Group {
                    id: GroupId(1),
                    label: "Options".to_owned(),
                    folded: false,
                    pins: vec![
                        Pin {
                            id: PinId(3),
                            label: "Size".to_owned(),
                            direction: Direction::Both,
                        },
                        Pin {
                            id: PinId(4),
                            label: "Mask".to_owned(),
                            direction: Direction::Input,
                        },
                    ],
                }),
                Row::Group(Group {
                    id: GroupId(2),
                    label: "Advanced".to_owned(),
                    folded: true,
                    pins: vec![
                        Pin {
                            id: PinId(5),
                            label: "Edge handling".to_owned(),
                            direction: Direction::Input,
                        },
                        Pin {
                            id: PinId(6),
                            label: "Premultiplied alpha".to_owned(),
                            direction: Direction::Input,
                        },
                    ],
                }),
            ],
        ),
        node("Write", 140.0, -40.0, (150, 70, 70), vec![input("Image")]),
        node(
            "Contact Sheet",
            -180.0,
            90.0,
            (110, 90, 150),
            vec![input("Images"), output],
        ),
    ];
    for (id, record) in (1..).map(NodeId).zip(nodes) {
        graph.add_node(id, record).expect("a new node ID");
    }
    let wires = [
        (1, 3, 1),
        (2, 3, 3),
        (3, 4, 1),
        (4, 5, 1),
        (1, 6, 1),
        (2, 6, 1),
    ];
    for (from, to, input) in wires {
        let connection = Connection {
            from: PinRef {
                node: NodeId(from),
                pin: PinId(2),
            },
            to: PinRef {
                node: NodeId(to),
                pin: PinId(input),
            },
        };
        graph
            .add_connection(connection)
            .expect("an output and an input that exist");
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(node: u64, pin: u64) -> PinRef {
        PinRef {
            node: NodeId(node),
            pin: PinId(pin),
        }
    }

    /// From a node's output, pin 2, to another node's pin.
    fn wire(from: u64, to: u64, input: u64) -> Connection {
        Connection {
            from: pin(from, 2),
            to: pin(to, input),
        }
    }

    fn edit(remove: &[Connection], add: &[Connection]) -> ConnectionEdit {
        ConnectionEdit {
            remove: remove.to_vec(),
            add: add.to_vec(),
        }
    }

    fn connect(edit: ConnectionEdit) -> Response {
        Response {
            connections: Some(edit),
            ..Response::default()
        }
    }

    fn accepted(displaced: &[Connection]) -> Verdict {
        Verdict::Accept {
            displaced: displaced.to_vec(),
        }
    }

    fn refused(reason: &str) -> Verdict {
        Verdict::Refuse {
            reason: Some(reason.to_owned()),
        }
    }

    fn merge_inputs(document: &Document) -> Vec<(u64, &str)> {
        let record = document.graph.node(MERGE).unwrap();
        let inputs = record.pins().filter(|p| p.direction == Direction::Input);
        inputs.map(|p| (p.id.0, p.label.as_str())).collect()
    }

    #[test]
    fn a_wire_into_an_occupied_input_replaces_its_wire_until_undone() {
        let mut document = Document::new();
        let (old, new) = (wire(1, 3, 1), wire(2, 3, 1));
        let replace = edit(&[], &[new]);
        assert_eq!(document.judge(&replace), accepted(&[old]));

        assert_eq!(document.apply(connect(replace)), Ok(true));
        assert!(document.graph.contains_connection(new));
        assert!(!document.graph.contains_connection(old));
        assert!(document.undo());
        assert!(document.graph.contains_connection(old));
        assert!(!document.graph.contains_connection(new));
        assert!(document.redo());
        assert!(document.graph.contains_connection(new));
        assert!(!document.graph.contains_connection(old));

        let swap_back = edit(&[new], &[old]);
        assert_eq!(document.judge(&swap_back), accepted(&[]), "already removed");
        assert_eq!(document.apply(connect(swap_back)), Ok(true));
        assert!(document.graph.contains_connection(old));
    }

    #[test]
    fn a_bundle_is_refused_by_an_input_that_takes_one_wire() {
        let mut document = Document::new();
        let lifted = [wire(1, 6, 1), wire(2, 6, 1)];
        let onto_mask = edit(&lifted, &[wire(1, 4, 4), wire(2, 4, 4)]);
        assert_eq!(document.judge(&onto_mask), refused("Mask takes one wire"));

        let before = document.graph.connections().to_vec();
        let refusal = document.apply(connect(onto_mask));
        assert_eq!(refusal, Err("Mask takes one wire".to_owned()));
        assert_eq!(document.graph.connections(), before);
        assert!(!document.undo(), "no step");
    }

    #[test]
    fn connecting_the_spare_input_adds_the_next_under_a_fresh_id_until_undone() {
        let mut document = Document::new();
        let original = [(3, "Background"), (1, "Foreground"), (4, SPARE_LABEL)];
        let connected = [
            (3, "Background"),
            (1, "Foreground"),
            (4, "Foreground 2"),
            (5, SPARE_LABEL),
        ];
        let into_spare = wire(1, 3, 4);
        assert_eq!(document.judge(&edit(&[], &[into_spare])), accepted(&[]));

        assert_eq!(document.apply(connect(edit(&[], &[into_spare]))), Ok(true));
        assert_eq!(merge_inputs(&document), connected);
        assert!(document.graph.contains_connection(into_spare));
        assert!(document.undo());
        assert_eq!(merge_inputs(&document), original);
        assert!(!document.graph.contains_connection(into_spare));
        assert!(document.redo());
        assert_eq!(merge_inputs(&document), connected);
        assert!(document.graph.contains_connection(into_spare));

        let into_next = wire(2, 3, 5);
        assert_eq!(document.apply(connect(edit(&[], &[into_next]))), Ok(true));
        let inputs = merge_inputs(&document);
        assert_eq!(inputs[3..], [(5, "Foreground 3"), (6, SPARE_LABEL)]);
    }

    #[test]
    fn undo_puts_a_removed_source_back_in_its_place() {
        let mut document = Document::new();
        let sources = |document: &Document| document.sources[&SHEET_IMAGES].clone();
        let first = wire(1, 6, 1);
        assert_eq!(sources(&document), [pin(1, 2), pin(2, 2)]);

        let cut = edit(&[first], &[]);
        assert_eq!(document.judge(&cut), accepted(&[]));
        assert_eq!(document.apply(connect(cut)), Ok(true));
        assert_eq!(sources(&document), [pin(2, 2)]);
        let more = edit(&[], &[wire(3, 6, 1), wire(4, 6, 1)]);
        assert_eq!(document.judge(&more), accepted(&[]), "takes several");
        assert_eq!(document.apply(connect(more)), Ok(true));
        assert_eq!(sources(&document), [pin(2, 2), pin(3, 2), pin(4, 2)]);

        assert!(document.undo());
        assert!(document.undo());
        assert_eq!(sources(&document), [pin(1, 2), pin(2, 2)]);
        let drawn_last = document.graph.connections().last();
        assert_eq!(
            drawn_last,
            Some(&first),
            "the graph's order is only drawing"
        );
        assert!(document.redo());
        assert_eq!(sources(&document), [pin(2, 2)]);
    }

    #[test]
    fn the_write_lock_refuses_its_wires_and_moves_until_toggled() {
        let mut document = Document::new();
        let cut = edit(&[wire(4, 5, 1)], &[]);
        let replace = edit(&[], &[wire(3, 5, 1)]);
        let moved = Response {
            positions: vec![(WRITE, Point { x: 200.0, y: 0.0 })],
            ..Response::default()
        };
        assert_eq!(document.judge(&cut), refused("Write is locked"));
        assert_eq!(document.judge(&replace), refused("Write is locked"));
        assert_eq!(document.apply(moved.clone()), Ok(false));

        document.toggle_write_lock();
        assert_eq!(document.judge(&cut), accepted(&[]));
        assert_eq!(document.judge(&replace), accepted(&[wire(4, 5, 1)]));
        assert_eq!(document.apply(moved), Ok(true));
    }

    #[test]
    fn a_wire_that_would_make_a_cycle_is_refused() {
        let document = Document::new();
        let size_to_mask = Connection {
            from: pin(4, 3),
            to: pin(4, 4),
        };
        let cycle = refused("That would make a cycle");
        assert_eq!(document.judge(&edit(&[], &[size_to_mask])), cycle);
        let blur_into_merge = edit(&[], &[wire(4, 3, 4)]);
        assert_eq!(document.judge(&blur_into_merge), cycle);
        let after_unwiring = edit(&[wire(3, 4, 1)], &[wire(4, 3, 4)]);
        assert_eq!(document.judge(&after_unwiring), accepted(&[]));
    }

    #[test]
    fn a_bundle_keeps_a_wire_its_target_already_has() {
        let mut document = Document::new();
        let sheet = [wire(1, 6, 1), wire(2, 6, 1)];
        let onto_foreground = edit(&sheet, &[wire(1, 3, 1), wire(2, 3, 1)]);
        let refusal = refused("Foreground takes one wire");
        assert_eq!(
            document.judge(&onto_foreground),
            refusal,
            "not Read 1 displaced"
        );

        let foreground = wire(1, 3, 1);
        let onto_sheet = edit(&[foreground], &[sheet[0]]);
        assert_eq!(document.judge(&onto_sheet), accepted(&[]));
        assert_eq!(document.apply(connect(onto_sheet)), Ok(true));
        assert!(document.graph.contains_connection(sheet[0]));
        assert_eq!(document.sources[&SHEET_IMAGES], [pin(1, 2), pin(2, 2)]);
        assert!(document.undo());
        assert!(document.graph.contains_connection(foreground));
        assert!(document.graph.contains_connection(sheet[0]));
    }

    #[test]
    fn a_script_edit_is_normalised_and_its_wires_checked() {
        let mut document = Document::new();
        let gone = wire(1, 4, 4);
        let new = wire(3, 6, 1);
        let repeated = edit(&[gone, wire(1, 6, 1), wire(1, 6, 1)], &[new, new]);
        assert_eq!(document.apply(connect(repeated)), Ok(true));
        assert_eq!(document.sources[&SHEET_IMAGES], [pin(2, 2), pin(3, 2)]);
        assert!(document.undo());
        assert_eq!(document.sources[&SHEET_IMAGES], [pin(1, 2), pin(2, 2)]);

        let unchanged = edit(&[gone], &[wire(2, 6, 1)]);
        assert_eq!(document.apply(connect(unchanged)), Ok(false));
        let no_such_pin = edit(&[], &[wire(1, 4, 9)]);
        assert!(document.apply(connect(no_such_pin)).is_err());
        let steps = (document.undo.len(), document.redo.len());
        assert_eq!(steps, (0, 1), "neither added a step");
    }
}
