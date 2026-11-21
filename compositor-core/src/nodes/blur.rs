//! Blurs its source by its Size, a radius in pixels, every channel as it is, so that
//! premultiplied edges fade without an unpremultiply; the data window grows by the reach.

use open_ng::Color;

use crate::nodes::{NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "blur",
    label: "Blur",
    type_name: "OcBlur",
    step: "Create Blur",
    color: Color {
        r: 100,
        g: 90,
        b: 140,
    },
    parameters: &[Parameter {
        name: "size",
        label: "Size",
        kind: ParameterKind::Float(FloatParameter {
            // A new Blur changes nothing until its size is set.
            default: 0.0,
            slider: 0.0..=100.0,
            limits: 0.0..=4096.0,
        }),
    }],
    inputs: &[NodeInput {
        name: "source",
        label: "Source",
        repeating: false,
        required: true,
    }],
};
