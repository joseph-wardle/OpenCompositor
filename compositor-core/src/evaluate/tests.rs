use open_ng::Point;

use super::*;
use crate::document::Assignment;

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

/// A key at most this deep is compared and dropped on a thread with this stack.
const DEEP_CHAIN: u64 = 10_000;
const DEEP_STACK: usize = 64 << 20;

fn reader(path: &str) -> Node {
    Node::new(
        NodeKind::Reader {
            path: Path::new(path).into(),
        },
        ORIGIN,
    )
}

fn exposure(stops: Option<f64>) -> Node {
    let mut node = Node::new(
        NodeKind::Exposure {
            version: ExposureVersion::V1,
        },
        ORIGIN,
    );
    if let Some(stops) = stops {
        node.set_value("stops", Value::Float(stops)).unwrap();
    }
    node
}

/// A document of these nodes, with IDs from 1, each Exposure taking the image of the node before
/// it.
fn chain(nodes: impl IntoIterator<Item = Node>) -> Document {
    let mut document = Document::default();
    let mut previous = None;
    let mut assignments = Vec::new();
    for (id, node) in (1..).map(NodeId).zip(nodes) {
        if let (NodeKind::Exposure { .. }, Some(source)) = (&node.kind, previous) {
            assignments.push(Assignment {
                consumer: id,
                input: "source",
                source: Some(source),
            });
        }
        document.nodes.insert(id, Arc::new(node));
        previous = Some(id);
    }
    // From the end of the chain, so that each cycle check finds its source unconnected.
    for assignment in assignments.iter().rev() {
        document = document.assign_inputs(&[*assignment]).unwrap();
    }
    document
}

fn key(document: &Document, id: u64) -> ResultKey {
    result_key(document, &Reloads::default(), NodeId(id)).unwrap()
}

#[test]
fn keys_name_what_decides_the_image_and_nothing_else() {
    let document = chain([reader("a.exr"), exposure(Some(1.0))]);
    let mut elsewhere = chain([reader("b.exr"), reader("a.exr"), exposure(Some(1.0))]);
    let moved = Arc::make_mut(elsewhere.nodes.get_mut(&NodeId(3)).unwrap());
    moved.position = Point { x: 50.0, y: 20.0 };
    assert_eq!(key(&document, 2), key(&elsewhere, 3));
    assert_eq!(key(&document, 1), key(&elsewhere, 2));

    let different = [
        (chain([reader("b.exr"), exposure(Some(1.0))]), 2),
        (chain([reader("a.exr"), exposure(Some(2.0))]), 2),
        (
            chain([reader("a.exr"), exposure(Some(1.0)), exposure(None)]),
            3,
        ),
    ];
    for (other, id) in &different {
        assert_ne!(key(&document, 2), key(other, *id));
    }
    assert_ne!(key(&document, 1), key(&document, 2));
}

#[test]
fn a_value_equal_to_its_default_keys_as_the_default() {
    let unset = chain([reader("a.exr"), exposure(None)]);
    let zero = chain([reader("a.exr"), exposure(Some(0.0))]);
    let negative_zero = chain([reader("a.exr"), exposure(Some(-0.0))]);
    assert_eq!(key(&unset, 2), key(&zero, 2));
    assert_eq!(key(&unset, 2), key(&negative_zero, 2));
}

#[test]
fn an_unconnected_input_keys_apart_from_any_connection() {
    let unconnected = chain([exposure(None)]);
    let connected = chain([reader("a.exr"), exposure(None)]);
    assert_eq!(key(&unconnected, 1), key(&chain([exposure(None)]), 1));
    assert_ne!(key(&unconnected, 1), key(&connected, 2));
}

#[test]
fn reloading_a_path_changes_the_key_of_everything_downstream_of_it() {
    let document = chain([reader("a.exr"), exposure(None), reader("b.exr")]);
    let mut reloads = Reloads::default();
    let before = |id| result_key(&document, &Reloads::default(), NodeId(id));

    reloads.reload(Path::new("a.exr").into());
    let after = |id| result_key(&document, &reloads, NodeId(id));
    assert_ne!(after(1), before(1));
    assert_ne!(after(2), before(2));
    assert_eq!(after(3), before(3));
    assert_eq!(reloads.count(Path::new("a.exr")), 1);
}

#[test]
fn a_missing_node_has_no_key() {
    let document = chain([reader("a.exr")]);
    assert_eq!(result_key(&document, &Reloads::default(), NodeId(2)), None);
}

#[test]
fn a_deep_chain_keys_without_recursion() {
    let nodes = std::iter::once(reader("a.exr"))
        .chain((1..DEEP_CHAIN).map(|stops| exposure(Some(stops as f64 % 64.0))));
    let document = chain(nodes);

    // Built on the test's own stack, which recursion this deep would overflow.
    let first = key(&document, DEEP_CHAIN);
    let second = key(&document, DEEP_CHAIN);
    let shallower = key(&document, DEEP_CHAIN - 1);

    std::thread::Builder::new()
        .stack_size(DEEP_STACK)
        .spawn(move || {
            assert!(first == second);
            assert!(first != shallower);
        })
        .unwrap()
        .join()
        .unwrap();
}
