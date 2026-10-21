//! How the document appears in OpenNG, and what OpenNG's connection edits mean to it.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use open_ng::{
    Color, Connection, ConnectionEdit, Direction, Graph, NodeRecord, Pin, PinId, PinRef, Row,
    Verdict,
};

use crate::document::{
    Assignment, Changes, ConnectionError, Document, InputSlot, MAX_SLOTS, Node, NodeId, NodeKind,
};

/// Every node's one output. An input's pin is derived from its declaration index and, for a
/// repeating input, its slot, so pins are never stored and a slot's pin is its own for the
/// node's life: a version that reorders or removes inputs would need pins of its own that are
/// never reused.
const OUTPUT_PIN: PinId = PinId(0);

const READER_COLOR: Color = Color {
    r: 70,
    g: 110,
    b: 150,
};
const EXPOSURE_COLOR: Color = Color {
    r: 90,
    g: 130,
    b: 80,
};
const MERGE_COLOR: Color = Color {
    r: 150,
    g: 105,
    b: 60,
};
const FAILED_COLOR: Color = Color {
    r: 160,
    g: 60,
    b: 60,
};
/// A node is as wide as its title, so a longer file name is shortened in the middle until OpenNG
/// elides titles itself.
const MAX_NAME_CHARS: usize = 32;

/// Brings the graph in line with the document and the nodes shown failed, for the nodes and
/// selection `changes` names only. Returns whether the graph changed.
pub(crate) fn present(
    graph: &mut Graph,
    document: &Document,
    failing: &BTreeSet<NodeId>,
    changes: &Changes,
) -> bool {
    let mut changed = false;
    let mut moved = Vec::new();
    for &id in &changes.nodes {
        match (document.node(id), graph.node(id.into())) {
            (None, Some(_)) => {
                graph.remove_node(id.into()).expect("the node is shown");
                changed = true;
            }
            (None, None) => {}
            (Some(node), None) => {
                graph
                    .add_node(id.into(), record(document, failing, id, node))
                    .expect("the node is not shown");
                changed = true;
            }
            (Some(node), Some(shown)) => {
                let record = record(document, failing, id, node);
                // Updating a record lays the node out again; moving it does not.
                if record.title == shown.title
                    && record.color == shown.color
                    && record.rows == shown.rows
                {
                    if record.position != shown.position {
                        moved.push((id.into(), record.position));
                    }
                } else {
                    graph
                        .update_node(id.into(), record)
                        .expect("the node is shown");
                    changed = true;
                }
            }
        }
    }
    graph
        .set_node_positions(&moved)
        .expect("moved nodes are shown");
    // Every node is shown now, so each changed node's wires can be made to match its inputs.
    let wanted: HashSet<Connection> = changes
        .nodes
        .iter()
        .filter_map(|&id| Some((id, document.node(id)?)))
        .flat_map(|(id, node)| wires_into(id, node))
        .collect();
    let unwanted: Vec<Connection> = graph
        .connections()
        .iter()
        .filter(|wire| changes.nodes.contains(&wire.to.node.into()) && !wanted.contains(wire))
        .copied()
        .collect();
    for &wire in &unwanted {
        graph.remove_connection(wire).expect("the wire is shown");
        changed = true;
    }
    for &wire in &wanted {
        if !graph.contains_connection(wire) {
            graph.add_connection(wire).expect("both pins are shown");
            changed = true;
        }
    }
    if changes.selection {
        let selection: Vec<_> = document.selection().iter().map(|&id| id.into()).collect();
        graph
            .set_selection(&selection)
            .expect("selected nodes are shown, once each");
    }
    changed || !moved.is_empty() || changes.selection
}

pub(crate) fn record(
    document: &Document,
    failing: &BTreeSet<NodeId>,
    id: NodeId,
    node: &Node,
) -> NodeRecord {
    let mut title = shorten(&node.name());
    let color = match node.kind {
        NodeKind::Reader { .. } => READER_COLOR,
        NodeKind::Exposure { .. } => EXPOSURE_COLOR,
        NodeKind::Merge { .. } => MERGE_COLOR,
    };
    // Slot numbers go in the title, and a failure in the colour, until OpenNG has badges.
    // Nothing shows an evaluation in progress: the title would widen the node while it lasted, and the
    // viewer's caption says it.
    let slots: Vec<_> = document
        .slots_marking(id)
        .map(|slot| slot.digit().to_string())
        .collect();
    if !slots.is_empty() {
        title = format!("{title} [{}]", slots.join(" "));
    }
    NodeRecord {
        position: node.position,
        title,
        color: if failing.contains(&id) {
            FAILED_COLOR
        } else {
            color
        },
        rows: std::iter::once(Pin {
            id: OUTPUT_PIN,
            label: "Image".to_owned(),
            direction: Direction::Output,
        })
        .chain(input_pins(node))
        .map(Row::Pin)
        .collect(),
    }
}

/// The node's input pins: one per input, and for a repeating input one per slot up to the last
/// connected one, empty slots included, then one free slot, while there is one.
fn input_pins(node: &Node) -> impl Iterator<Item = Pin> {
    (node.kind.inputs().iter().enumerate()).flat_map(|(index, input)| {
        let slots = if input.repeating {
            let last = node.slots(input.name).last().map_or(0, |(slot, _)| slot);
            1..=last.saturating_add(1).min(MAX_SLOTS)
        } else {
            0..=0
        };
        slots.map(move |slot| Pin {
            id: input_pin(index, slot),
            label: InputSlot {
                input: input.label,
                slot,
            }
            .to_string(),
            direction: Direction::Input,
        })
    })
}

fn input_pin(index: usize, slot: u32) -> PinId {
    PinId(u64::from(slot) << 32 | (index as u64 + 1))
}

/// The input and slot a pin shows on a node of this kind, named as the kind declares the input.
/// Whether the input has the slot is the document's to say.
fn input_at(kind: &NodeKind, pin: PinId) -> Option<InputSlot<'static>> {
    let index = usize::try_from((pin.0 & u64::from(u32::MAX)).checked_sub(1)?).ok()?;
    Some(InputSlot {
        input: kind.inputs().get(index)?.name,
        slot: (pin.0 >> 32) as u32,
    })
}

/// The wires a node's connected inputs show.
pub(crate) fn wires_into(id: NodeId, node: &Node) -> impl Iterator<Item = Connection> {
    node.inputs()
        .map(move |(input, source)| wire(id, &node.kind, input, source))
}

fn wire(consumer: NodeId, kind: &NodeKind, input: InputSlot<'_>, source: NodeId) -> Connection {
    let index = kind
        .inputs()
        .iter()
        .position(|declared| declared.name == input.input)
        .expect("a connected input is declared");
    Connection {
        from: PinRef {
            node: source.into(),
            pin: OUTPUT_PIN,
        },
        to: PinRef {
            node: consumer.into(),
            pin: input_pin(index, input.slot),
        },
    }
}

/// What a connection edit from OpenNG assigns: each removed wire's input disconnected, unless the
/// edit also wires that input, and each added wire's input connected. Refused if a pin names no
/// input, or if two added wires enter one input.
pub(crate) fn assignments(
    document: &Document,
    edit: &ConnectionEdit,
) -> Result<Vec<Assignment<'static>>, ConnectionError> {
    let input = |pin: PinRef| {
        let consumer = NodeId::from(pin.node);
        let node = document
            .node(consumer)
            .ok_or(ConnectionError::MissingNode(consumer))?;
        let input = input_at(&node.kind, pin.pin)
            .ok_or_else(|| ConnectionError::UnknownInput(format!("pin {}", pin.pin.0)))?;
        Ok((consumer, input))
    };
    let mut sources = BTreeMap::new();
    for wire in &edit.remove {
        sources.insert(input(wire.to)?, None);
    }
    for wire in &edit.add {
        // A node's only pin with an output side is its output, so the source is its node.
        let source = Some(NodeId::from(wire.from.node));
        if sources.insert(input(wire.to)?, source).flatten().is_some() {
            return Err(ConnectionError::InputAssignedTwice);
        }
    }
    Ok(sources
        .into_iter()
        .map(|((consumer, input), source)| Assignment {
            consumer,
            input,
            source,
        })
        .collect())
}

/// Whether the document would take a proposed connection edit, and which wires it would replace.
pub(crate) fn judge(document: &Document, edit: &ConnectionEdit) -> Verdict {
    let assignments = match assignments(document, edit) {
        Ok(assignments) => assignments,
        Err(error) => return refuse(error),
    };
    if let Err(error) = document.assign_inputs(&assignments) {
        return refuse(error);
    }
    let displaced = assignments
        .iter()
        .filter(|assignment| assignment.source.is_some())
        .filter_map(|assignment| {
            let node = document.node(assignment.consumer)?;
            let current = node.input(assignment.input)?;
            let wire = wire(assignment.consumer, &node.kind, assignment.input, current);
            (Some(current) != assignment.source && !edit.remove.contains(&wire)).then_some(wire)
        })
        .collect();
    Verdict::Accept { displaced }
}

fn refuse(error: ConnectionError) -> Verdict {
    Verdict::Refuse {
        reason: Some(error.to_string()),
    }
}

/// Keeps the start of a long name, often the shot, and the end, the version, frame and extension,
/// either side of an ellipsis.
fn shorten(name: &str) -> String {
    let length = name.chars().count();
    if length <= MAX_NAME_CHARS {
        return name.to_owned();
    }
    let start = (MAX_NAME_CHARS - 1) / 2;
    let end = MAX_NAME_CHARS - 1 - start;
    let mut shortened: String = name.chars().take(start).collect();
    shortened.push('…');
    shortened.extend(name.chars().skip(length - end));
    shortened
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_file_names_lose_their_middle() {
        let fits = "a".repeat(MAX_NAME_CHARS);
        assert_eq!(shorten(&fits), fits);

        let shortened = shorten("ABC_seq020_sh0140_plate_bg01_v012.1001.exr");
        assert_eq!(shortened, "ABC_seq020_sh01…01_v012.1001.exr");
        assert_eq!(shortened.chars().count(), MAX_NAME_CHARS);

        // Counted in characters, not bytes.
        let accented = "é".repeat(MAX_NAME_CHARS + 1);
        assert_eq!(shorten(&accented).chars().count(), MAX_NAME_CHARS);
    }
}
