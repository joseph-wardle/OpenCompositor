use std::collections::HashSet;
use std::path::Path;

use open_ng::Point;

use super::*;
use crate::document::{ExposureVersion, NodeKind};

fn exposure() -> Node {
    Node::new(
        NodeKind::Exposure {
            version: ExposureVersion::V1,
        },
        Point { x: 0.0, y: 0.0 },
    )
}

fn stops(node: &Node) -> Option<Value> {
    resolve(node).get("stops")
}

/// Every built-in node type at every version.
fn built_in() -> [NodeKind; 2] {
    [
        NodeKind::Reader {
            path: Path::new("a.exr").into(),
        },
        NodeKind::Exposure {
            version: ExposureVersion::V1,
        },
    ]
}

#[test]
fn built_in_descriptions_are_consistent() {
    for kind in built_in() {
        let parameters = kind.parameters();
        let names: HashSet<_> = parameters.iter().map(|parameter| parameter.name).collect();
        assert_eq!(names.len(), parameters.len(), "{kind:?}: names repeat");
        for parameter in parameters {
            let ParameterKind::Float(float) = &parameter.kind;
            let limits = &float.limits;
            assert!(
                limits.start().is_finite() && limits.end().is_finite() && !limits.is_empty(),
                "{}",
                parameter.name
            );
            assert!(limits.contains(&float.default), "{}", parameter.name);
            assert!(
                !float.slider.is_empty()
                    && limits.contains(float.slider.start())
                    && limits.contains(float.slider.end()),
                "{}",
                parameter.name
            );
        }
    }
}

#[test]
fn an_unset_value_resolves_to_its_default() {
    let node = exposure();
    assert_eq!(node.authored_value("stops"), None);
    assert_eq!(stops(&node), Some(Value::Float(0.0)));
}

#[test]
fn a_value_set_to_its_default_stays_set_until_reset() {
    let mut node = exposure();
    node.set_value("stops", Value::Float(1.5)).unwrap();
    assert_eq!(stops(&node), Some(Value::Float(1.5)));

    node.set_value("stops", Value::Float(0.0)).unwrap();
    assert_eq!(node.authored_value("stops"), Some(Value::Float(0.0)));
    let set_to_default = resolve(&node);

    node.reset("stops").unwrap();
    assert_eq!(node.authored_value("stops"), None);
    assert_eq!(resolve(&node), set_to_default);
    assert_eq!(node, exposure());
}

#[test]
fn values_the_description_does_not_allow_are_refused() {
    let mut node = exposure();
    node.set_value("stops", Value::Float(2.0)).unwrap();
    let limits = -64.0..=64.0;

    let refusals = [
        ("stops", f64::NAN, ValueError::NotFinite),
        ("stops", f64::INFINITY, ValueError::NotFinite),
        ("stops", 64.5, ValueError::OutOfLimits(limits.clone())),
        ("stops", -100.0, ValueError::OutOfLimits(limits)),
        ("Stops", 1.0, ValueError::UnknownParameter),
    ];
    for (name, value, error) in refusals {
        assert_eq!(
            node.set_value(name, Value::Float(value)),
            Err(error),
            "{name} {value}"
        );
    }
    assert_eq!(node.reset("gain"), Err(ValueError::UnknownParameter));
    assert_eq!(stops(&node), Some(Value::Float(2.0)));

    // A typed value may go beyond the slider, up to the limits.
    node.set_value("stops", Value::Float(64.0)).unwrap();
    assert_eq!(stops(&node), Some(Value::Float(64.0)));
}

#[test]
fn a_reader_has_no_parameters_to_set() {
    let mut reader = Node::new(
        NodeKind::Reader {
            path: Path::new("a.exr").into(),
        },
        Point { x: 0.0, y: 0.0 },
    );
    assert_eq!(
        reader.set_value("stops", Value::Float(1.0)),
        Err(ValueError::UnknownParameter)
    );
    assert_eq!(resolve(&reader), Resolved { values: Vec::new() });
}

#[test]
fn negative_zero_is_stored_as_zero() {
    let mut node = exposure();
    node.set_value("stops", Value::Float(-0.0)).unwrap();
    let Value::Float(stored) = node.authored_value("stops").unwrap();
    assert_eq!(stored.to_bits(), 0.0f64.to_bits());
}
