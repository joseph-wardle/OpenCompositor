//! The connection edits a view proposes to its host and the host's answers.

use rustc_hash::FxHashSet;

use crate::graph::{Connection, Graph};

/// Connections to remove and connections to have after, as one edit.
///
/// A view shows its host the same edit twice: while a gesture is in progress, as a proposal for
/// the host to judge (see [`View::handle_input`](crate::View::handle_input)), and once released,
/// if the host's last verdict accepted it, as
/// [`Response::connections`](crate::Response::connections). Either way it is
/// [normalised](ConnectionEdit::normalised).
///
/// An addition the graph already has is a wire the edit keeps: dropping a bundle of lifted wires
/// onto an input already wired from one of their outputs keeps that wire, and the host must not
/// displace it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConnectionEdit {
    pub remove: Vec<Connection>,
    pub add: Vec<Connection>,
}

/// The host's answer to a proposed [`ConnectionEdit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The host would apply the edit and also remove `displaced`, connections in neither of its
    /// lists, such as the wire a new one replaces on an input that takes only one. The view hides
    /// them in its preview.
    Accept { displaced: Vec<Connection> },
    /// The host would apply none of the edit. The view shows the reason, if any, beside the
    /// pointer.
    Refuse { reason: Option<String> },
}

impl ConnectionEdit {
    /// The edit without what would change nothing, or `None` if it would change nothing. No
    /// connection appears twice or in both lists, and every removal is in the graph. A connection
    /// in both lists stays, which is what dropping a lifted wire back where it was proposes.
    ///
    /// Views propose only normalised edits. A host applying edits from elsewhere, such as a
    /// script, can normalise them first.
    pub fn normalised(mut self, graph: &Graph) -> Option<ConnectionEdit> {
        let added: FxHashSet<_> = self.add.iter().copied().collect();
        let mut seen = FxHashSet::default();
        self.remove.retain(|&connection| {
            !added.contains(&connection)
                && graph.contains_connection(connection)
                && seen.insert(connection)
        });
        seen.clear();
        self.add.retain(|&connection| seen.insert(connection));
        let changes = !self.remove.is_empty()
            || (self.add.iter()).any(|&connection| !graph.contains_connection(connection));
        changes.then_some(self)
    }
}

/// A judge for tests whose gestures propose no connection edits or whose edits are all
/// acceptable.
#[cfg(test)]
pub(crate) fn accept_all(_: &ConnectionEdit) -> Verdict {
    Verdict::Accept {
        displaced: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Color, Direction, NodeId, NodeRecord, Pin, PinId, PinRef, Point, Row};

    /// From one node's output pin 2 to another's input pin 1.
    fn wire(from: u64, to: u64) -> Connection {
        let pin = |node, pin| PinRef {
            node: NodeId(node),
            pin: PinId(pin),
        };
        Connection {
            from: pin(from, 2),
            to: pin(to, 1),
        }
    }

    /// Four nodes with input pin 1 and output pin 2, wired 1 → 2 → 3.
    fn chain() -> Graph {
        let pin = |id, direction| {
            Row::Pin(Pin {
                id: PinId(id),
                label: String::new(),
                direction,
            })
        };
        let mut graph = Graph::default();
        for id in 1..=4 {
            let record = NodeRecord {
                position: Point::default(),
                title: String::new(),
                color: Color { r: 0, g: 0, b: 0 },
                rows: vec![pin(1, Direction::Input), pin(2, Direction::Output)],
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        graph.add_connection(wire(1, 2)).unwrap();
        graph.add_connection(wire(2, 3)).unwrap();
        graph
    }

    fn edit(remove: &[Connection], add: &[Connection]) -> ConnectionEdit {
        ConnectionEdit {
            remove: remove.to_vec(),
            add: add.to_vec(),
        }
    }

    #[test]
    fn edits_that_change_nothing_normalise_to_none() {
        let graph = chain();
        let dropped_back = edit(&[wire(1, 2)], &[wire(1, 2)]);
        assert_eq!(dropped_back.normalised(&graph), None);
        let reconnected = edit(&[], &[wire(2, 3), wire(2, 3)]);
        assert_eq!(reconnected.normalised(&graph), None);
        let already_gone = edit(&[wire(3, 4)], &[]);
        assert_eq!(already_gone.normalised(&graph), None);
    }

    #[test]
    fn normalising_keeps_each_change_and_kept_wire_once_in_order() {
        let graph = chain();
        // Wires 1 → 2 and 2 → 3 lifted and dropped onto node 3's input, which 2 → 3 already
        // feeds.
        let bundle = edit(
            &[wire(1, 2), wire(2, 3), wire(1, 2), wire(3, 4)],
            &[wire(1, 3), wire(2, 3), wire(1, 3)],
        );
        assert_eq!(
            bundle.normalised(&graph),
            Some(edit(&[wire(1, 2)], &[wire(1, 3), wire(2, 3)]))
        );
    }
}
