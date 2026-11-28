//! Multiplies R, G and B by A and leaves A: a source whose colour is straight, as a key or a
//! straight-alpha file holds it, made premultiplied, as every image Merge and Blur take is.

use open_ng::Color;
use opencompositor_kernels::Premult;

use crate::image::Channel;
use crate::nodes::pointwise::ChannelMap;
use crate::nodes::{Expects, Makes, NodeInput, NodeType};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "premult",
    label: "Premult",
    type_name: "OcPremult",
    step: "Create Premult",
    color: Color {
        r: 60,
        g: 120,
        b: 130,
    },
    parameters: &[],
    inputs: &[NodeInput {
        name: "source",
        label: "Source",
        repeating: false,
        required: true,
        expects: Expects::Straight,
    }],
    makes: Makes::Premultiplied,
};

/// R, G and B each with A through the product; A is the source's.
pub(crate) const MAPS: [ChannelMap<Premult, 2, 1>; 3] = [
    by_alpha(Channel::R),
    by_alpha(Channel::G),
    by_alpha(Channel::B),
];

const fn by_alpha(channel: Channel) -> ChannelMap<Premult, 2, 1> {
    ChannelMap {
        op: Premult,
        reads: [channel, Channel::A],
        writes: [channel],
    }
}
