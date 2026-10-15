use std::collections::HashSet;

use super::*;
use crate::document::{ExposureVersion, NodeKind};

/// Every built-in node type at every version.
fn built_in() -> [NodeKind; 2] {
    [
        NodeKind::Reader {
            path: "a.exr".into(),
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
