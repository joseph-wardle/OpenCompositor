//! Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied pixels
//! that equals unpremultiplying, scaling and premultiplying again.

use open_ng::Color;
use opencompositor_kernels::Exposure;

use crate::image::Channel;
use crate::nodes::pointwise::ChannelMap;
use crate::nodes::{Expects, Makes, NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind, Resolved};

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
        expects: Expects::Either,
    }],
    makes: Makes::AsInputs,
};

/// R, G and B each through the gain, computed here, on the host, so that both devices agree on
/// it; A is the source's.
pub(crate) fn maps(values: &Resolved) -> Vec<ChannelMap<Exposure, 1, 1>> {
    let stops = values.float("stops").expect("Stops has a default");
    // Exact for whole stops, and far inside f32's range at the limits.
    let op = Exposure {
        gain: stops.exp2() as f32,
    };
    let colour = [Channel::R, Channel::G, Channel::B];
    Vec::from(colour.map(|channel| ChannelMap {
        op,
        reads: [channel],
        writes: [channel],
    }))
}
