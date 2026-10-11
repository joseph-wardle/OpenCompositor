use std::collections::{HashMap, HashSet};
use std::fmt;

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

/// The side of a node a port attaches to. Wires leave a port along this side's outward normal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Top,
    Bottom,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Port {
    pub side: Side,
}

/// What the editor needs to draw and interact with one node.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeRecord {
    /// The centre of the node.
    pub position: Point,
    pub color: Color,
    pub inputs: Vec<Port>,
    pub outputs: Vec<Port>,
}

impl NodeRecord {
    /// A node with its inputs along the top and its outputs along the bottom.
    pub fn vertical(position: Point, color: Color, inputs: usize, outputs: usize) -> Self {
        NodeRecord {
            position,
            color,
            inputs: vec![Port { side: Side::Top }; inputs],
            outputs: vec![Port { side: Side::Bottom }; outputs],
        }
    }
}

/// An output port, identified by its index in the node's output list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputRef {
    pub node: NodeId,
    pub index: usize,
}

/// An input port, identified by its index in the node's input list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputRef {
    pub node: NodeId,
    pub index: usize,
}

/// A wire from an output port to an input port. A connection is identified by its endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Connection {
    pub from: OutputRef,
    pub to: InputRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNode(NodeId),
    MissingNode(NodeId),
    MissingOutput(OutputRef),
    MissingInput(InputRef),
    MissingConnection(Connection),
    /// Ports are identified by index, so changing how many a node has would silently re-point
    /// its connections.
    PortCountChanged(NodeId),
    RepeatedInSelection(NodeId),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::DuplicateNode(id) => write!(f, "node {} already exists", id.0),
            GraphError::MissingNode(id) => write!(f, "node {} does not exist", id.0),
            GraphError::MissingOutput(port) => {
                write!(f, "node {} has no output {}", port.node.0, port.index)
            }
            GraphError::MissingInput(port) => {
                write!(f, "node {} has no input {}", port.node.0, port.index)
            }
            GraphError::MissingConnection(c) => write!(
                f,
                "no connection from node {} output {} to node {} input {}",
                c.from.node.0, c.from.index, c.to.node.0, c.to.index
            ),
            GraphError::PortCountChanged(id) => {
                write!(f, "node {} cannot change its number of ports", id.0)
            }
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
#[derive(Debug, Default)]
pub struct Graph {
    /// In draw order: later nodes are drawn above earlier ones.
    nodes: Vec<Entry>,
    index_of: HashMap<NodeId, usize>,
    connections: Vec<Connection>,
    selection: Vec<NodeId>,
    /// How many nodes have ever been added.
    add_count: u64,
}

/// The graph's add count at one moment. Views take one when a gesture starts, so a node added
/// since, even under a reused ID, is not mistaken for one the gesture refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct AddStamp(u64);

#[derive(Debug)]
struct Entry {
    id: NodeId,
    record: NodeRecord,
    added: AddStamp,
}

impl Graph {
    /// Adds a node above all existing nodes.
    pub fn add_node(&mut self, id: NodeId, record: NodeRecord) -> Result<(), GraphError> {
        if self.index_of.contains_key(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        self.add_count += 1;
        self.index_of.insert(id, self.nodes.len());
        self.nodes.push(Entry {
            id,
            record,
            added: AddStamp(self.add_count),
        });
        Ok(())
    }

    /// Replaces a node's record. The number of inputs and outputs must not change.
    pub fn update_node(&mut self, id: NodeId, record: NodeRecord) -> Result<(), GraphError> {
        let index = self.index(id)?;
        let current = &mut self.nodes[index].record;
        if record.inputs.len() != current.inputs.len()
            || record.outputs.len() != current.outputs.len()
        {
            return Err(GraphError::PortCountChanged(id));
        }
        *current = record;
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
        self.connections
            .retain(|c| c.from.node != id && c.to.node != id);
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

    /// Adds a connection between existing ports. Adding an existing connection does nothing.
    pub fn add_connection(&mut self, connection: Connection) -> Result<(), GraphError> {
        let from = self.node_record(connection.from.node)?;
        if connection.from.index >= from.outputs.len() {
            return Err(GraphError::MissingOutput(connection.from));
        }
        let to = self.node_record(connection.to.node)?;
        if connection.to.index >= to.inputs.len() {
            return Err(GraphError::MissingInput(connection.to));
        }
        if !self.connections.contains(&connection) {
            self.connections.push(connection);
        }
        Ok(())
    }

    pub fn remove_connection(&mut self, connection: Connection) -> Result<(), GraphError> {
        let index = self
            .connections
            .iter()
            .position(|&c| c == connection)
            .ok_or(GraphError::MissingConnection(connection))?;
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

    /// All nodes in draw order, bottom first.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &NodeRecord)> {
        self.nodes.iter().map(|entry| (entry.id, &entry.record))
    }

    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    pub fn selection(&self) -> &[NodeId] {
        &self.selection
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

    fn index(&self, id: NodeId) -> Result<usize, GraphError> {
        self.index_of
            .get(&id)
            .copied()
            .ok_or(GraphError::MissingNode(id))
    }

    fn node_record(&self, id: NodeId) -> Result<&NodeRecord, GraphError> {
        self.node(id).ok_or(GraphError::MissingNode(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    fn node(x: f32) -> NodeRecord {
        NodeRecord::vertical(Point { x, y: 0.0 }, GREY, 1, 1)
    }

    fn wire(from: u64, to: u64) -> Connection {
        Connection {
            from: OutputRef {
                node: NodeId(from),
                index: 0,
            },
            to: InputRef {
                node: NodeId(to),
                index: 0,
            },
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
        assert_eq!(graph.selection(), [NodeId(2), NodeId(3)]);
        assert_eq!(
            graph.add_node(NodeId(1), node(0.0)),
            Err(GraphError::DuplicateNode(NodeId(1)))
        );
    }

    #[test]
    fn connections_need_existing_ports_and_are_added_once() {
        let mut graph = chain();
        let past_last_output = Connection {
            from: OutputRef {
                node: NodeId(1),
                index: 1,
            },
            to: InputRef {
                node: NodeId(3),
                index: 0,
            },
        };
        let past_last_input = Connection {
            from: OutputRef {
                node: NodeId(1),
                index: 0,
            },
            to: InputRef {
                node: NodeId(3),
                index: 1,
            },
        };

        assert_eq!(
            graph.add_connection(past_last_output),
            Err(GraphError::MissingOutput(past_last_output.from))
        );
        assert_eq!(
            graph.add_connection(past_last_input),
            Err(GraphError::MissingInput(past_last_input.to))
        );
        assert_eq!(
            graph.add_connection(wire(1, 9)),
            Err(GraphError::MissingNode(NodeId(9)))
        );
        graph.add_connection(wire(1, 2)).unwrap();
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);

        graph.remove_connection(wire(1, 2)).unwrap();
        assert_eq!(
            graph.remove_connection(wire(1, 2)),
            Err(GraphError::MissingConnection(wire(1, 2)))
        );
        assert_eq!(graph.connections(), [wire(2, 3)]);
    }

    #[test]
    fn update_rejects_a_port_count_change_and_keeps_the_old_record() {
        let mut graph = chain();
        let wider = NodeRecord::vertical(Point { x: 50.0, y: 0.0 }, GREY, 2, 1);

        assert_eq!(
            graph.update_node(NodeId(2), wider),
            Err(GraphError::PortCountChanged(NodeId(2)))
        );
        let more_outputs = NodeRecord::vertical(Point { x: 50.0, y: 0.0 }, GREY, 1, 2);
        assert_eq!(
            graph.update_node(NodeId(2), more_outputs),
            Err(GraphError::PortCountChanged(NodeId(2)))
        );
        assert_eq!(graph.node(NodeId(2)), Some(&node(2.0)));

        let mut side_input = node(50.0);
        side_input.inputs[0].side = Side::Left;
        graph.update_node(NodeId(2), side_input.clone()).unwrap();
        assert_eq!(graph.node(NodeId(2)), Some(&side_input));
        assert_eq!(graph.connections(), [wire(1, 2), wire(2, 3)]);
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
