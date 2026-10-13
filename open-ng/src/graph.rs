use std::collections::HashSet;
use std::fmt;

use ab_glyph::FontRef;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::geometry::NodeLayout;
use crate::text::bundled_font;

/// Host-supplied node identity. Unique among the nodes currently in a [`Graph`]; an ID may be
/// reused after its node is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// A position in graph units: one unit is one logical pixel at 100% zoom, with y pointing down.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// An sRGB-encoded color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Host-chosen pin identity, unique within its node across both directions and all groups.
/// Connections refer to pins by it, so renaming, reordering or regrouping a pin keeps its wires.
/// Pins added at run time, such as a merge's extra inputs, need fresh IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PinId(pub u64);

/// Host-chosen group identity, unique among the groups of its node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(pub u64);

/// Which way data flows through a pin. An input's port is on the node's left edge, an output's
/// on its right, and a pin that both receives and supplies a value has one on each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
    Both,
}

impl Direction {
    pub(crate) fn has(self, side: Side) -> bool {
        matches!(
            (self, side),
            (Direction::Both, _)
                | (Direction::Input, Side::Input)
                | (Direction::Output, Side::Output)
        )
    }
}

/// One of a pin's ports. A connection leaves its `from` pin's output side and enters its `to`
/// pin's input side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Side {
    Input,
    Output,
}

impl Side {
    pub(crate) fn opposite(self) -> Side {
        match self {
            Side::Input => Side::Output,
            Side::Output => Side::Input,
        }
    }
}

/// One row with a label and a port on each side its direction has. Labels may repeat.
#[derive(Clone, Debug, PartialEq)]
pub struct Pin {
    pub id: PinId,
    pub label: String,
    pub direction: Direction,
}

/// A header row followed by its pins, of any directions in any order. Groups do not nest. A
/// folded group hides its pins; their wires end at the header instead.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub id: GroupId,
    pub label: String,
    pub folded: bool,
    pub pins: Vec<Pin>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    Pin(Pin),
    Group(Group),
}

/// What the editor needs to draw and interact with one node.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeRecord {
    /// The top-left corner of the node. It grows and shrinks down and to the right, so its title
    /// and inputs stay put.
    pub position: Point,
    pub title: String,
    pub color: Color,
    /// From top to bottom below the title.
    pub rows: Vec<Row>,
}

impl NodeRecord {
    /// Every pin, grouped or not, from top to bottom.
    pub fn pins(&self) -> impl Iterator<Item = &Pin> {
        self.rows.iter().flat_map(|row| match row {
            Row::Pin(pin) => std::slice::from_ref(pin),
            Row::Group(group) => &group.pins[..],
        })
    }

    /// The pin with this ID, grouped or not.
    pub fn pin(&self, id: PinId) -> Option<&Pin> {
        self.pins().find(|pin| pin.id == id)
    }

    pub(crate) fn group(&self, id: GroupId) -> Option<&Group> {
        self.rows.iter().find_map(|row| match row {
            Row::Group(group) if group.id == id => Some(group),
            _ => None,
        })
    }
}

/// A pin of a node in the graph. It names no side: a connection's `from` always uses the output
/// side and its `to` the input side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PinRef {
    pub node: NodeId,
    pub pin: PinId,
}

/// A wire from a pin's output side to a pin's input side. A connection is identified by its
/// endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Connection {
    pub from: PinRef,
    pub to: PinRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNode(NodeId),
    MissingNode(NodeId),
    /// A record gives two pins this ID.
    DuplicatePin(PinRef),
    /// A record gives two groups this ID.
    DuplicateGroup {
        node: NodeId,
        group: GroupId,
    },
    MissingPin(PinRef),
    /// A connection must run from a pin with an output side to a pin with an input side.
    NotOutputToInput(Connection),
    MissingConnection(Connection),
    RepeatedInSelection(NodeId),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::DuplicateNode(id) => write!(f, "node {} already exists", id.0),
            GraphError::MissingNode(id) => write!(f, "node {} does not exist", id.0),
            GraphError::DuplicatePin(pin) => {
                write!(f, "node {} has more than one pin {}", pin.node.0, pin.pin.0)
            }
            GraphError::DuplicateGroup { node, group } => {
                write!(f, "node {} has more than one group {}", node.0, group.0)
            }
            GraphError::MissingPin(pin) => {
                write!(f, "node {} has no pin {}", pin.node.0, pin.pin.0)
            }
            GraphError::NotOutputToInput(c) => write!(
                f,
                "node {} pin {} to node {} pin {} is not from an output to an input",
                c.from.node.0, c.from.pin.0, c.to.node.0, c.to.pin.0
            ),
            GraphError::MissingConnection(c) => write!(
                f,
                "no connection from node {} pin {} to node {} pin {}",
                c.from.node.0, c.from.pin.0, c.to.node.0, c.to.pin.0
            ),
            GraphError::RepeatedInSelection(id) => {
                write!(f, "node {} appears more than once in the selection", id.0)
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// The presentation of one host network, shared by every view of it.
///
/// The host keeps it in step with its own document. Each method validates its whole input
/// first and leaves the graph unchanged on error.
#[derive(Debug)]
pub struct Graph {
    /// In the order nodes were added. Later nodes draw above earlier ones, and selected nodes
    /// above unselected ones.
    nodes: Vec<Entry>,
    // Fx rather than the standard hasher, here and for pins: every prepared frame looks up both
    // ends of every connection, and with thousands of them SipHash doubled the time to prepare.
    index_of: FxHashMap<NodeId, usize>,
    connections: Vec<Connection>,
    /// The same connections, so that adding one need not search them all.
    connection_set: FxHashSet<Connection>,
    selection: Vec<NodeId>,
    /// How many nodes have ever been added.
    add_count: u64,
    /// Measures titles and labels when a record is laid out.
    font: FontRef<'static>,
}

/// The graph's add count at one moment. Views take one when a gesture starts, so a node added
/// since, even under a reused ID, is not mistaken for one the gesture refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AddStamp(u64);

#[derive(Debug)]
struct Entry {
    id: NodeId,
    record: NodeRecord,
    layout: NodeLayout,
    added: AddStamp,
}

impl Default for Graph {
    fn default() -> Graph {
        Graph {
            nodes: Vec::new(),
            index_of: FxHashMap::default(),
            connections: Vec::new(),
            connection_set: FxHashSet::default(),
            selection: Vec::new(),
            add_count: 0,
            font: bundled_font(),
        }
    }
}

impl Graph {
    /// Adds a node above all existing nodes.
    pub fn add_node(&mut self, id: NodeId, record: NodeRecord) -> Result<(), GraphError> {
        if self.index_of.contains_key(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        check_ids(id, &record)?;
        self.add_count += 1;
        self.index_of.insert(id, self.nodes.len());
        self.nodes.push(Entry {
            id,
            layout: NodeLayout::new(&self.font, &record),
            record,
            added: AddStamp(self.add_count),
        });
        Ok(())
    }

    /// Replaces a node's record. Wires follow their pins wherever the record now places them;
    /// connections to pins it no longer has, or that lost the side the connection uses, are
    /// removed. To undo an update, restore the old record and then the removed connections.
    pub fn update_node(&mut self, id: NodeId, record: NodeRecord) -> Result<(), GraphError> {
        let index = self.index(id)?;
        check_ids(id, &record)?;
        let has = |pin: PinRef, side: Side| {
            pin.node != id || record.pin(pin.pin).is_some_and(|p| p.direction.has(side))
        };
        self.retain_connections(|c| has(c.from, Side::Output) && has(c.to, Side::Input));
        let entry = &mut self.nodes[index];
        entry.layout = NodeLayout::new(&self.font, &record);
        entry.record = record;
        Ok(())
    }

    /// Removes a node with its connections and its place in the selection. The other nodes keep
    /// their draw order.
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), GraphError> {
        let index = self.index(id)?;
        self.nodes.remove(index);
        self.index_of.remove(&id);
        for later in &self.nodes[index..] {
            *self
                .index_of
                .get_mut(&later.id)
                .expect("every node is indexed") -= 1;
        }
        self.retain_connections(|c| c.from.node != id && c.to.node != id);
        self.selection.retain(|&selected| selected != id);
        Ok(())
    }

    /// Moves several nodes at once. If an ID appears more than once, its last position wins.
    pub fn set_node_positions(&mut self, positions: &[(NodeId, Point)]) -> Result<(), GraphError> {
        for &(id, _) in positions {
            self.index(id)?;
        }
        for &(id, position) in positions {
            let index = self.index_of[&id];
            self.nodes[index].record.position = position;
        }
        Ok(())
    }

    /// Adds a connection from an existing pin with an output side to an existing pin with an
    /// input side. Adding an existing connection does nothing.
    pub fn add_connection(&mut self, connection: Connection) -> Result<(), GraphError> {
        let from = self.pin(connection.from)?;
        let to = self.pin(connection.to)?;
        if !from.direction.has(Side::Output) || !to.direction.has(Side::Input) {
            return Err(GraphError::NotOutputToInput(connection));
        }
        if self.connection_set.insert(connection) {
            self.connections.push(connection);
        }
        Ok(())
    }

    pub fn remove_connection(&mut self, connection: Connection) -> Result<(), GraphError> {
        if !self.connection_set.remove(&connection) {
            return Err(GraphError::MissingConnection(connection));
        }
        let index = self
            .connections
            .iter()
            .position(|&c| c == connection)
            .expect("the set and the list hold the same connections");
        self.connections.remove(index);
        Ok(())
    }

    /// Replaces the selection. Order is kept; the last node is the most recently selected.
    pub fn set_selection(&mut self, ids: &[NodeId]) -> Result<(), GraphError> {
        let mut seen = HashSet::with_capacity(ids.len());
        for &id in ids {
            self.index(id)?;
            if !seen.insert(id) {
                return Err(GraphError::RepeatedInSelection(id));
            }
        }
        self.selection.clear();
        self.selection.extend_from_slice(ids);
        Ok(())
    }

    pub fn node(&self, id: NodeId) -> Option<&NodeRecord> {
        let &index = self.index_of.get(&id)?;
        Some(&self.nodes[index].record)
    }

    /// All nodes in the order they were added. Later nodes draw above earlier ones, and selected
    /// nodes above unselected ones.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &NodeRecord)> {
        self.nodes.iter().map(|entry| (entry.id, &entry.record))
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn contains_connection(&self, connection: Connection) -> bool {
        self.connection_set.contains(&connection)
    }

    pub fn selection(&self) -> &[NodeId] {
        &self.selection
    }

    pub(crate) fn node_layout(&self, id: NodeId) -> Option<&NodeLayout> {
        let &index = self.index_of.get(&id)?;
        Some(&self.nodes[index].layout)
    }

    /// All nodes with their layouts, in the order they were added.
    pub(crate) fn laid_out_nodes(
        &self,
    ) -> impl DoubleEndedIterator<Item = (NodeId, &NodeRecord, &NodeLayout)> {
        self.nodes
            .iter()
            .map(|entry| (entry.id, &entry.record, &entry.layout))
    }

    /// All nodes with their layouts from bottom to top: the unselected nodes in the order they
    /// were added, then the selected ones in that order. Deriving the order from the selection
    /// keeps the nodes being worked on on top without a stored stacking order.
    pub(crate) fn nodes_in_draw_order<'a>(
        &'a self,
        selected: &'a HashSet<NodeId>,
    ) -> impl Iterator<Item = (NodeId, &'a NodeRecord, &'a NodeLayout)> {
        let unselected = self
            .laid_out_nodes()
            .filter(|(id, _, _)| !selected.contains(id));
        let selected = self
            .laid_out_nodes()
            .filter(|(id, _, _)| selected.contains(id));
        unselected.chain(selected)
    }

    pub(crate) fn add_stamp(&self) -> AddStamp {
        AddStamp(self.add_count)
    }

    /// Whether `id` names a node that has been in the graph since `stamp` was taken.
    pub(crate) fn has_node_since(&self, id: NodeId, stamp: AddStamp) -> bool {
        self.index_of
            .get(&id)
            .is_some_and(|&index| self.nodes[index].added <= stamp)
    }

    /// Whether `pin` is on a node that has been in the graph since `stamp` was taken and still
    /// has that side.
    pub(crate) fn has_port_since(&self, pin: PinRef, side: Side, stamp: AddStamp) -> bool {
        self.has_node_since(pin.node, stamp)
            && self.pin(pin).is_ok_and(|pin| pin.direction.has(side))
    }

    fn retain_connections(&mut self, mut keep: impl FnMut(&Connection) -> bool) {
        let set = &mut self.connection_set;
        self.connections.retain(|connection| {
            let kept = keep(connection);
            if !kept {
                set.remove(connection);
            }
            kept
        });
    }

    fn index(&self, id: NodeId) -> Result<usize, GraphError> {
        self.index_of
            .get(&id)
            .copied()
            .ok_or(GraphError::MissingNode(id))
    }

    fn pin(&self, pin: PinRef) -> Result<&Pin, GraphError> {
        let record = self
            .node(pin.node)
            .ok_or(GraphError::MissingNode(pin.node))?;
        record.pin(pin.pin).ok_or(GraphError::MissingPin(pin))
    }
}

/// Checks that the record's pin IDs, and separately its group IDs, are unique.
fn check_ids(node: NodeId, record: &NodeRecord) -> Result<(), GraphError> {
    let mut pins = HashSet::new();
    if let Some(pin) = record.pins().find(|pin| !pins.insert(pin.id)) {
        return Err(GraphError::DuplicatePin(PinRef { node, pin: pin.id }));
    }
    let mut groups = HashSet::new();
    for row in &record.rows {
        if let Row::Group(group) = row
            && !groups.insert(group.id)
        {
            return Err(GraphError::DuplicateGroup {
                node,
                group: group.id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    fn pin(id: u64, label: &str, direction: Direction) -> Pin {
        Pin {
            id: PinId(id),
            label: label.to_owned(),
            direction,
        }
    }

    /// A node with input pin 1 and output pin 2.
    fn node(x: f32) -> NodeRecord {
        NodeRecord {
            position: Point { x, y: 0.0 },
            title: "Blur".to_owned(),
            color: GREY,
            rows: vec![
                Row::Pin(pin(1, "Image", Direction::Input)),
                Row::Pin(pin(2, "Image", Direction::Output)),
            ],
        }
    }

    fn pin_ref(node: u64, pin: u64) -> PinRef {
        PinRef {
            node: NodeId(node),
            pin: PinId(pin),
        }
    }

    /// From one node's output pin 2 to another's input pin 1.
    fn wire(from: u64, to: u64) -> Connection {
        Connection {
            from: pin_ref(from, 2),
            to: pin_ref(to, 1),
        }
    }

    /// A chain 1 → 2 → 3, all selected.
    fn chain() -> Graph {
        let mut graph = Graph::default();
        for id in 1..=3 {
            graph.add_node(NodeId(id), node(id as f32)).unwrap();
        }
        graph.add_connection(wire(1, 2)).unwrap();
        graph.add_connection(wire(2, 3)).unwrap();
        graph
            .set_selection(&[NodeId(1), NodeId(2), NodeId(3)])
            .unwrap();
        graph
    }

    fn ids(graph: &Graph) -> Vec<u64> {
        graph.nodes().map(|(id, _)| id.0).collect()
    }

    #[test]
    fn removing_a_node_drops_its_connections_and_selection_and_keeps_draw_order() {
        let mut graph = chain();
        graph.remove_node(NodeId(2)).unwrap();

        assert_eq!(ids(&graph), [1, 3]);
        assert!(graph.connections().is_empty());
        assert_eq!(graph.selection(), [NodeId(1), NodeId(3)]);
        assert_eq!(graph.node(NodeId(3)).unwrap().position.x, 3.0);
    }

    #[test]
    fn a_reused_id_returns_as_a_new_node_on_top() {
        let mut graph = chain();
        graph.remove_node(NodeId(1)).unwrap();
        graph.add_node(NodeId(1), node(10.0)).unwrap();

        assert_eq!(ids(&graph), [2, 3, 1]);
        assert_eq!(graph.connections(), [wire(2, 3)]);
        graph.add_connection(wire(1, 2)).unwrap();
        assert_eq!(graph.connections(), [wire(2, 3), wire(1, 2)]);
        assert_eq!(graph.selection(), [NodeId(2), NodeId(3)]);
        assert_eq!(
            graph.add_node(NodeId(1), node(0.0)),
            Err(GraphError::DuplicateNode(NodeId(1)))
        );
    }

    #[test]
    fn connections_run_from_existing_outputs_to_existing_inputs_and_are_added_once() {
        let mut graph = chain();
        let missing_pin = Connection {
            from: pin_ref(1, 2),
            to: pin_ref(3, 9),
        };
        let backwards = Connection {
            from: pin_ref(3, 1),
            to: pin_ref(1, 2),
        };
        let output_to_output = Connection {
            from: pin_ref(1, 2),
            to: pin_ref(3, 2),
        };

        assert_eq!(
            graph.add_connection(missing_pin),
            Err(GraphError::MissingPin(pin_ref(3, 9)))
        );
        assert_eq!(
            graph.add_connection(wire(1, 9)),
            Err(GraphError::MissingNode(NodeId(9)))
        );
        for wrong_way in [backwards, output_to_output] {
            assert_eq!(
                graph.add_connection(wrong_way),
                Err(GraphError::NotOutputToInput(wrong_way))
            );
        }
        graph.add_connection(wire(1, 2)).unwrap();
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);
        assert!(graph.contains_connection(wire(1, 2)));

        graph.remove_connection(wire(1, 2)).unwrap();
        assert!(!graph.contains_connection(wire(1, 2)));
        assert_eq!(
            graph.remove_connection(wire(1, 2)),
            Err(GraphError::MissingConnection(wire(1, 2)))
        );
        assert_eq!(graph.connections(), [wire(2, 3)]);
    }

    #[test]
    fn renaming_reordering_regrouping_and_inserting_pins_keeps_their_wires() {
        let mut graph = chain();
        let mut changed = node(2.0);
        changed.rows = vec![
            Row::Pin(pin(2, "Result", Direction::Output)),
            Row::Pin(pin(3, "Mask", Direction::Input)),
            Row::Group(Group {
                id: GroupId(1),
                label: "Sources".to_owned(),
                folded: true,
                pins: vec![pin(1, "Source", Direction::Input)],
            }),
        ];

        graph.update_node(NodeId(2), changed.clone()).unwrap();
        assert_eq!(graph.node(NodeId(2)), Some(&changed));
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);
    }

    #[test]
    fn removing_or_reversing_a_pin_drops_its_connections() {
        let mut graph = chain();
        let mut without_input = node(2.0);
        without_input.rows.remove(0);
        graph.update_node(NodeId(2), without_input).unwrap();
        assert_eq!(graph.connections(), [wire(2, 3)]);

        let mut reversed_output = node(2.0);
        reversed_output.rows[1] = Row::Pin(pin(2, "Image", Direction::Input));
        graph.update_node(NodeId(2), reversed_output).unwrap();
        assert_eq!(graph.connections(), []);

        // Undone by restoring the record and then the connections.
        graph.update_node(NodeId(2), node(2.0)).unwrap();
        graph.add_connection(wire(1, 2)).unwrap();
        graph.add_connection(wire(2, 3)).unwrap();
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);
    }

    #[test]
    fn a_pin_with_both_sides_keeps_the_wires_of_the_side_it_still_has() {
        let mut graph = chain();
        let with_image = |direction| {
            let mut record = node(2.0);
            record.rows[0] = Row::Pin(pin(1, "Image", direction));
            record
        };
        // Node 2's pin 1 supplies its value as well as receiving one.
        let supplied = Connection {
            from: pin_ref(2, 1),
            to: pin_ref(3, 1),
        };
        graph
            .update_node(NodeId(2), with_image(Direction::Both))
            .unwrap();
        graph.add_connection(supplied).unwrap();
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3), supplied]);

        graph
            .update_node(NodeId(2), with_image(Direction::Input))
            .unwrap();
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);
        assert_eq!(
            graph.add_connection(supplied),
            Err(GraphError::NotOutputToInput(supplied))
        );

        graph
            .update_node(NodeId(2), with_image(Direction::Both))
            .unwrap();
        graph.add_connection(supplied).unwrap();
        graph
            .update_node(NodeId(2), with_image(Direction::Output))
            .unwrap();
        assert_eq!(graph.connections(), [wire(2, 3), supplied]);
    }

    #[test]
    fn a_record_with_repeated_ids_changes_nothing() {
        let mut graph = chain();
        let group = |id: u64, pins: Vec<Pin>| {
            Row::Group(Group {
                id: GroupId(id),
                label: "Options".to_owned(),
                folded: false,
                pins,
            })
        };
        let mut repeated_pin = node(50.0);
        repeated_pin
            .rows
            .push(group(1, vec![pin(1, "Mask", Direction::Input)]));
        let mut repeated_group = node(50.0);
        repeated_group
            .rows
            .extend([group(1, vec![]), group(1, vec![])]);

        assert_eq!(
            graph.update_node(NodeId(2), repeated_pin.clone()),
            Err(GraphError::DuplicatePin(pin_ref(2, 1)))
        );
        assert_eq!(
            graph.update_node(NodeId(2), repeated_group.clone()),
            Err(GraphError::DuplicateGroup {
                node: NodeId(2),
                group: GroupId(1)
            })
        );
        assert_eq!(
            graph.add_node(NodeId(4), repeated_pin),
            Err(GraphError::DuplicatePin(pin_ref(4, 1)))
        );
        assert_eq!(graph.node(NodeId(2)), Some(&node(2.0)));
        assert_eq!(graph.node(NodeId(4)), None);
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);

        // Labels may repeat, and pins and groups have separate IDs.
        let mut shared = node(50.0);
        shared
            .rows
            .push(group(1, vec![pin(3, "Image", Direction::Input)]));
        graph.update_node(NodeId(2), shared).unwrap();
    }

    #[test]
    fn a_position_batch_with_an_unknown_node_moves_nothing() {
        let mut graph = chain();
        let moved = Point { x: 40.0, y: 20.0 };

        assert_eq!(
            graph.set_node_positions(&[(NodeId(1), moved), (NodeId(9), moved)]),
            Err(GraphError::MissingNode(NodeId(9)))
        );
        assert_eq!(graph.node(NodeId(1)).unwrap().position.x, 1.0);

        graph.set_node_positions(&[(NodeId(1), moved)]).unwrap();
        assert_eq!(graph.node(NodeId(1)).unwrap().position, moved);
    }

    #[test]
    fn selection_keeps_order_and_rejects_unknown_or_repeated_nodes() {
        let mut graph = chain();

        assert_eq!(
            graph.set_selection(&[NodeId(3), NodeId(9)]),
            Err(GraphError::MissingNode(NodeId(9)))
        );
        assert_eq!(
            graph.set_selection(&[NodeId(3), NodeId(1), NodeId(3)]),
            Err(GraphError::RepeatedInSelection(NodeId(3)))
        );
        assert_eq!(graph.selection(), [NodeId(1), NodeId(2), NodeId(3)]);

        graph.set_selection(&[NodeId(3), NodeId(1)]).unwrap();
        assert_eq!(graph.selection(), [NodeId(3), NodeId(1)]);
    }
}
