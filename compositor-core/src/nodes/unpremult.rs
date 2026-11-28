//! Divides R, G and B by A and leaves A: a premultiplied source made straight, for an operation
//! on colour alone. Where A is zero, of either sign, the colour is left as it is.

use open_ng::Color;
use opencompositor_kernels::Unpremult;

use crate::image::Channel;
use crate::nodes::pointwise::ChannelMap;
use crate::nodes::{Expects, Makes, NodeInput, NodeType};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "unpremult",
    label: "Unpremult",
    type_name: "OcUnpremult",
    step: "Create Unpremult",
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
        expects: Expects::Premultiplied,
    }],
    makes: Makes::Straight,
};

/// R, G and B each with A through the quotient; A is the source's.
pub(crate) const MAPS: [ChannelMap<Unpremult, 2, 1>; 3] = [
    by_alpha(Channel::R),
    by_alpha(Channel::G),
    by_alpha(Channel::B),
];

const fn by_alpha(channel: Channel) -> ChannelMap<Unpremult, 2, 1> {
    ChannelMap {
        op: Unpremult,
        reads: [channel, Channel::A],
        writes: [channel],
    }
}
