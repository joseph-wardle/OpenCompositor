//! Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied pixels
//! that equals unpremultiplying, scaling and premultiplying again.

use open_ng::Color;

use crate::nodes::{NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "exposure",
    label: "Exposure",
    type_name: "OcExposure",
    step: "Create Exposure",
    color: Color {
        r: 90,
        g: 130,
        b: 80,
    },
    parameters: &[Parameter {
        name: "stops",
        label: "Stops",
        kind: ParameterKind::Float(FloatParameter {
            default: 0.0,
            slider: -6.0..=6.0,
            // Keeps the gain, two to these powers, many orders of magnitude inside f32's range.
            limits: -64.0..=64.0,
        }),
    }],
    inputs: &[NodeInput {
        name: "source",
        label: "Source",
        repeating: false,
        required: true,
    }],
};
