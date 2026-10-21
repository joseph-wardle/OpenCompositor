use std::path::Path;

use super::*;
use crate::parameter::Value;

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

/// A document of one Reader, with ID 0, and `exposures` Exposures, with IDs from 1.
fn document_of(exposures: u64) -> Document {
    let mut reader = Node::new(
        NodeKind::Reader {
            version: ReaderVersion::V1,
        },
        ORIGIN,
    );
    reader
        .set_value("file", Value::File(Path::new("a.exr").into()))
        .unwrap();
    let exposure = NodeKind::Exposure {
        version: ExposureVersion::V1,
    };
    let mut document = Document::default();
    document.nodes.insert(NodeId(0), Arc::new(reader));
    for id in 1..=exposures {
        document
            .nodes
            .insert(NodeId(id), Arc::new(Node::new(exposure.clone(), ORIGIN)));
    }
    document
}

fn connect(consumer: u64, source: u64) -> Assignment<'static> {
    Assignment {
        consumer: NodeId(consumer),
        input: "source".into(),
        source: Some(NodeId(source)),
    }
}

fn disconnect(consumer: u64) -> Assignment<'static> {
    Assignment {
        consumer: NodeId(consumer),
        input: "source".into(),
        source: None,
    }
}

fn source_of(document: &Document, consumer: u64) -> Option<NodeId> {
    document.node(NodeId(consumer)).unwrap().input("source")
}

#[test]
fn a_connection_replaces_the_input_it_assigns() {
    let document = document_of(2).assign_inputs(&[connect(1, 0)]).unwrap();
    let document = document.assign_inputs(&[connect(1, 2)]).unwrap();
    assert_eq!(source_of(&document, 1), Some(NodeId(2)));
    let document = document.assign_inputs(&[disconnect(1)]).unwrap();
    assert_eq!(source_of(&document, 1), None);
}

#[test]
fn a_batch_is_judged_on_its_result() {
    // 1 takes 2. Turning the wire around would be a cycle while it stands, but this batch also
    // removes it, in whichever order.
    let document = document_of(2).assign_inputs(&[connect(1, 2)]).unwrap();
    let document = document
        .assign_inputs(&[connect(2, 1), disconnect(1)])
        .unwrap();
    assert_eq!(source_of(&document, 1), None);
    assert_eq!(source_of(&document, 2), Some(NodeId(1)));
}

#[test]
fn refused_edits_name_their_reason() {
    // 1 takes 2, which takes 3. 4 is unconnected.
    let document = document_of(4)
        .assign_inputs(&[connect(1, 2), connect(2, 3)])
        .unwrap();
    let refusals = [
        (vec![connect(9, 0)], ConnectionError::MissingNode(NodeId(9))),
        (vec![connect(1, 9)], ConnectionError::MissingNode(NodeId(9))),
        (vec![disconnect(9)], ConnectionError::MissingNode(NodeId(9))),
        (
            vec![Assignment {
                consumer: NodeId(1),
                input: "mask".into(),
                source: Some(NodeId(0)),
            }],
            ConnectionError::UnknownInput("mask".to_owned()),
        ),
        (
            vec![Assignment {
                consumer: NodeId(0),
                input: "source".into(),
                source: Some(NodeId(1)),
            }],
            ConnectionError::UnknownInput("source".to_owned()),
        ),
        (vec![connect(1, 1)], ConnectionError::SelfConnection),
        (vec![connect(3, 1)], ConnectionError::Cycle),
        (vec![connect(3, 2)], ConnectionError::Cycle),
        (
            vec![connect(4, 0), connect(4, 1)],
            ConnectionError::InputAssignedTwice,
        ),
        // A valid assignment does not carry a refused one in the same batch.
        (vec![connect(4, 0), connect(3, 1)], ConnectionError::Cycle),
    ];
    for (assignments, reason) in refusals {
        assert_eq!(
            document.assign_inputs(&assignments).err(),
            Some(reason),
            "{assignments:?}"
        );
    }
}

#[test]
fn an_edit_that_changes_nothing_shares_every_node() {
    let before = document_of(2).assign_inputs(&[connect(1, 0)]).unwrap();
    let after = before
        .assign_inputs(&[connect(1, 0), disconnect(2)])
        .unwrap();
    assert!(before.changes_to(&after).is_empty());
}

#[test]
fn deleting_nodes_clears_what_named_them() {
    // 1 and 2 take 0, and 3 takes 2.
    let mut before = document_of(3)
        .assign_inputs(&[connect(1, 0), connect(2, 0), connect(3, 2)])
        .unwrap();
    before.selection = [NodeId(3), NodeId(0), NodeId(1)].into();
    before.marks[1] = Some(NodeId(0));
    before.marks[2] = Some(NodeId(3));

    let after = before.without_nodes(&[NodeId(0), NodeId(2)]);

    assert_eq!(
        after.nodes().map(|(id, _)| id).collect::<Vec<_>>(),
        [NodeId(1), NodeId(3)]
    );
    assert_eq!(source_of(&after, 1), None);
    assert_eq!(source_of(&after, 3), None);
    assert_eq!(after.selection(), [NodeId(3), NodeId(1)]);
    assert_eq!(after.marks[1], None);
    assert_eq!(after.marks[2], Some(NodeId(3)));
}

#[test]
fn what_is_missing_is_found_up_the_chain() {
    // 3 takes 2, which takes nothing; 1 takes the Reader and is not upstream of 3.
    let document = document_of(3)
        .assign_inputs(&[connect(3, 2), connect(1, 0)])
        .unwrap();
    let names = |document: &Document, id: u64| -> Vec<(u64, &str)> {
        (missing(document, NodeId(id)).into_iter())
            .map(|(node, missing)| match missing {
                Missing::Input(input) => (node.0, input.name),
                Missing::File(parameter) => (node.0, parameter.name),
            })
            .collect()
    };

    assert_eq!(names(&document, 3), [(2, "source")]);
    assert_eq!(names(&document, 2), [(2, "source")]);
    assert_eq!(names(&document, 1), []);
    assert_eq!(names(&document, 0), []);
    assert_eq!(names(&document, 9), []);

    // A Reader without a file is reported where it is, as an unconnected input is.
    let mut unset = document.clone();
    let reader = unset.nodes.get_mut(&NodeId(0)).unwrap();
    Arc::make_mut(reader).reset("file").unwrap();
    assert_eq!(names(&unset, 1), [(0, "file")]);
    assert_eq!(names(&unset, 0), [(0, "file")]);
}

/// A document of two Readers, with IDs 0 and 1, and a Merge with ID 2.
fn document_with_merge() -> Document {
    let mut document = document_of(0);
    let mut reader = Node::new(
        NodeKind::Reader {
            version: ReaderVersion::V1,
        },
        ORIGIN,
    );
    reader
        .set_value("file", Value::File(Path::new("b.exr").into()))
        .unwrap();
    document.nodes.insert(NodeId(1), Arc::new(reader));
    let merge = NodeKind::Merge {
        version: MergeVersion::V1,
    };
    document
        .nodes
        .insert(NodeId(2), Arc::new(Node::new(merge, ORIGIN)));
    document
}

fn layer(consumer: u64, slot: u32, source: u64) -> Assignment<'static> {
    Assignment {
        consumer: NodeId(consumer),
        input: ("layer", slot).into(),
        source: Some(NodeId(source)),
    }
}

#[test]
fn a_repeating_input_takes_a_wire_per_slot_and_each_slot_stays_where_it_is() {
    let document = document_with_merge()
        .assign_inputs(&[layer(2, 1, 0), layer(2, 3, 1)])
        .unwrap();
    let merge = document.node(NodeId(2)).unwrap();
    assert_eq!(
        merge.slots("layer").collect::<Vec<_>>(),
        [(1, NodeId(0)), (3, NodeId(1))]
    );
    assert_eq!(merge.input(("layer", 2)), None);
    assert_eq!(merge.input("background"), None);
    assert_eq!(
        (merge.inputs().map(|(input, _)| input.to_string())).collect::<Vec<_>>(),
        ["layer 1", "layer 3"]
    );

    // Disconnecting the lower slot leaves the higher one where it is.
    let document = document
        .assign_inputs(&[Assignment {
            consumer: NodeId(2),
            input: ("layer", 1).into(),
            source: None,
        }])
        .unwrap();
    let merge = document.node(NodeId(2)).unwrap();
    assert_eq!(merge.slots("layer").collect::<Vec<_>>(), [(3, NodeId(1))]);
    // Only a repeating input has slots.
    assert_eq!(merge.slots("background").next(), None);
}

#[test]
fn an_input_has_only_the_slots_its_kind_declares() {
    let document = document_with_merge();
    let unknown_slot = |input, slot| ConnectionError::UnknownSlot { input, slot };
    let refusals = [
        ("background", 1, unknown_slot("background", 1)),
        ("layer", 0, unknown_slot("layer", 0)),
        ("layer", MAX_SLOTS + 1, unknown_slot("layer", MAX_SLOTS + 1)),
        ("mask", 1, ConnectionError::UnknownInput("mask".to_owned())),
        ("mask", 0, ConnectionError::UnknownInput("mask".to_owned())),
    ];
    for (input, slot, reason) in refusals {
        let assignment = Assignment {
            consumer: NodeId(2),
            input: (input, slot).into(),
            source: Some(NodeId(0)),
        };
        assert_eq!(
            document.assign_inputs(&[assignment]).err(),
            Some(reason),
            "{input} {slot}"
        );
    }
    assert!(document.assign_inputs(&[layer(2, MAX_SLOTS, 0)]).is_ok());
}

#[test]
fn a_merge_is_never_missing_an_input_but_reports_what_is_missing_through_its_slots() {
    let document = document_with_merge();
    assert!(missing(&document, NodeId(2)).is_empty());
    let document = document.assign_inputs(&[layer(2, 2, 0)]).unwrap();
    assert!(missing(&document, NodeId(2)).is_empty());

    let mut unset = document.clone();
    let reader = unset.nodes.get_mut(&NodeId(0)).unwrap();
    Arc::make_mut(reader).reset("file").unwrap();
    assert_eq!(
        missing(&unset, NodeId(2)),
        [(NodeId(0), Missing::File(&READER_V1[0]))]
    );
}
