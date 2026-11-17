use std::path::Path;

use super::*;
use crate::parameter::Value;

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

/// A network of one Reader, with ID 0, and `exposures` Exposures, with IDs from 1.
fn network_of(exposures: u64) -> Network {
    let mut reader = Node::new(NodeKind::Reader, "Reader", ORIGIN);
    reader
        .set_value("file", Value::File(Path::new("a.exr").into()))
        .unwrap();
    let exposure = NodeKind::Exposure;
    let mut network = Network::default();
    network.nodes.insert(NodeId(0), Arc::new(reader));
    for id in 1..=exposures {
        network.nodes.insert(
            NodeId(id),
            Arc::new(Node::new(exposure.clone(), "Exposure", ORIGIN)),
        );
    }
    network
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

fn source_of(network: &Network, consumer: u64) -> Option<NodeId> {
    network.node(NodeId(consumer)).unwrap().input("source")
}

#[test]
fn a_connection_replaces_the_input_it_assigns() {
    let network = network_of(2).assign_inputs(&[connect(1, 0)]).unwrap();
    let network = network.assign_inputs(&[connect(1, 2)]).unwrap();
    assert_eq!(source_of(&network, 1), Some(NodeId(2)));
    let network = network.assign_inputs(&[disconnect(1)]).unwrap();
    assert_eq!(source_of(&network, 1), None);
}

#[test]
fn a_batch_is_judged_on_its_result() {
    // 1 takes 2. Turning the wire around would be a cycle while it stands, but this batch also
    // removes it, in whichever order.
    let network = network_of(2).assign_inputs(&[connect(1, 2)]).unwrap();
    let network = network
        .assign_inputs(&[connect(2, 1), disconnect(1)])
        .unwrap();
    assert_eq!(source_of(&network, 1), None);
    assert_eq!(source_of(&network, 2), Some(NodeId(1)));
}

#[test]
fn refused_edits_name_their_reason() {
    // 1 takes 2, which takes 3. 4 is unconnected.
    let network = network_of(4)
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
            network.assign_inputs(&assignments).err(),
            Some(reason),
            "{assignments:?}"
        );
    }
}

#[test]
fn an_edit_that_changes_nothing_shares_every_node() {
    let before = network_of(2).assign_inputs(&[connect(1, 0)]).unwrap();
    let after = before
        .assign_inputs(&[connect(1, 0), disconnect(2)])
        .unwrap();
    assert!(before.changes_to(&after).is_empty());
}

#[test]
fn what_is_missing_is_found_up_the_chain() {
    // 3 takes 2, which takes nothing; 1 takes the Reader and is not upstream of 3.
    let network = network_of(3)
        .assign_inputs(&[connect(3, 2), connect(1, 0)])
        .unwrap();
    let names = |network: &Network, id: u64| -> Vec<(u64, &str)> {
        (missing(network, NodeId(id), 1.0).into_iter())
            .map(|(node, missing)| match missing {
                Missing::Input(input) => (node.0, input.name),
                Missing::File(parameter) => (node.0, parameter.name),
                Missing::Refused(problem) => panic!("refused: {problem}"),
            })
            .collect()
    };

    assert_eq!(names(&network, 3), [(2, "source")]);
    assert_eq!(names(&network, 2), [(2, "source")]);
    assert_eq!(names(&network, 1), []);
    assert_eq!(names(&network, 0), []);
    assert_eq!(names(&network, 9), []);

    // A Reader without a file is reported where it is, as an unconnected input is.
    let mut unset = network.clone();
    let reader = unset.nodes.get_mut(&NodeId(0)).unwrap();
    Arc::make_mut(reader).reset("file").unwrap();
    assert_eq!(names(&unset, 1), [(0, "file")]);
    assert_eq!(names(&unset, 0), [(0, "file")]);
}

/// A network of two Readers, with IDs 0 and 1, and a Merge with ID 2.
fn network_with_merge() -> Network {
    let mut network = network_of(0);
    let mut reader = Node::new(NodeKind::Reader, "Reader", ORIGIN);
    reader
        .set_value("file", Value::File(Path::new("b.exr").into()))
        .unwrap();
    network.nodes.insert(NodeId(1), Arc::new(reader));
    let merge = NodeKind::Merge;
    network
        .nodes
        .insert(NodeId(2), Arc::new(Node::new(merge, "Merge", ORIGIN)));
    network
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
    let network = network_with_merge()
        .assign_inputs(&[layer(2, 1, 0), layer(2, 3, 1)])
        .unwrap();
    let merge = network.node(NodeId(2)).unwrap();
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
    let network = network
        .assign_inputs(&[Assignment {
            consumer: NodeId(2),
            input: ("layer", 1).into(),
            source: None,
        }])
        .unwrap();
    let merge = network.node(NodeId(2)).unwrap();
    assert_eq!(merge.slots("layer").collect::<Vec<_>>(), [(3, NodeId(1))]);
    // Only a repeating input has slots.
    assert_eq!(merge.slots("background").next(), None);
}

#[test]
fn an_input_has_only_the_slots_its_kind_declares() {
    let network = network_with_merge();
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
            network.assign_inputs(&[assignment]).err(),
            Some(reason),
            "{input} {slot}"
        );
    }
    assert!(network.assign_inputs(&[layer(2, MAX_SLOTS, 0)]).is_ok());
}

#[test]
fn a_merge_is_never_missing_an_input_but_reports_what_is_missing_through_its_slots() {
    let network = network_with_merge();
    assert!(missing(&network, NodeId(2), 1.0).is_empty());
    let network = network.assign_inputs(&[layer(2, 2, 0)]).unwrap();
    assert!(missing(&network, NodeId(2), 1.0).is_empty());

    let mut unset = network.clone();
    let reader = unset.nodes.get_mut(&NodeId(0)).unwrap();
    Arc::make_mut(reader).reset("file").unwrap();
    assert_eq!(
        missing(&unset, NodeId(2), 1.0),
        [(NodeId(0), Missing::File(&READER_PARAMETERS[0]))]
    );
}
