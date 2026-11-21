//! Composes its layers over its background, each layer in slot order over the result so far, then
//! dissolves that stack by its Mix against the base, the lowest connected input.

use open_ng::Color;

use crate::nodes::{NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "merge",
    label: "Merge",
    type_name: "OcMerge",
    step: "Create Merge",
    color: Color {
        r: 150,
        g: 105,
        b: 60,
    },
    parameters: &[Parameter {
        name: "mix",
        label: "Mix",
        kind: ParameterKind::Float(FloatParameter {
            default: 1.0,
            slider: 0.0..=1.0,
            limits: 0.0..=1.0,
        }),
    }],
    // Neither is required: the base is the lowest connected input, whichever that is, and nothing
    // connected gives the empty image.
    inputs: &[
        NodeInput {
            name: "background",
            label: "Background",
            repeating: false,
            required: false,
        },
        NodeInput {
            name: "layer",
            label: "Layer",
            repeating: true,
            required: false,
        },
    ],
};
