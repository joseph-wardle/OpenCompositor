//! The node types: what each offers to be set, the images it takes, and how it is created.
//! Each node's file says what it computes and holds its description.

pub(crate) mod blur;
pub(crate) mod exposure;
pub(crate) mod merge;
pub(crate) mod pointwise;
pub(crate) mod premult;
mod reader;
pub(crate) mod unpremult;

use open_ng::Color;

use crate::network::InputSlot;
use crate::parameter::Parameter;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Reader,
    Exposure,
    Premult,
    Unpremult,
    Merge,
    Blur,
}

impl NodeKind {
    /// Everything that describes a node of this kind.
    pub const fn node_type(&self) -> &'static NodeType {
        match self {
            NodeKind::Reader => &reader::NODE_TYPE,
            NodeKind::Exposure => &exposure::NODE_TYPE,
            NodeKind::Premult => &premult::NODE_TYPE,
            NodeKind::Unpremult => &unpremult::NODE_TYPE,
            NodeKind::Merge => &merge::NODE_TYPE,
            NodeKind::Blur => &blur::NODE_TYPE,
        }
    }

    pub(crate) fn of_type(type_name: &str) -> Option<NodeKind> {
        (NODE_KINDS.iter())
            .find(|kind| kind.node_type().type_name == type_name)
            .cloned()
    }
}

/// What a node kind is to the artist, to scripts and to the stage.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeType {
    /// Unchanged across labels: scripts create nodes by it.
    pub name: &'static str,
    /// What the artist calls the type.
    pub label: &'static str,
    /// The prim type a node of this kind is authored as. Types, parameters and defaults change in
    /// place until 1.0; after it, a breaking change takes USD's suffix, as `OcExposure_1`.
    pub type_name: &'static str,
    /// What creating one names in the undo history.
    pub(crate) step: &'static str,
    /// The node's colour in the graph.
    pub(crate) color: Color,
    /// What the node offers to be set.
    pub parameters: &'static [Parameter],
    /// The images the node takes.
    pub inputs: &'static [NodeInput],
}

const NODE_KINDS: &[NodeKind] = &[
    NodeKind::Reader,
    NodeKind::Exposure,
    NodeKind::Premult,
    NodeKind::Unpremult,
    NodeKind::Merge,
    NodeKind::Blur,
];

/// Every node kind that can be created, in menu order.
pub fn node_kinds() -> &'static [NodeKind] {
    NODE_KINDS
}

const _: () = {
    let mut kind = 0;
    while kind < NODE_KINDS.len() {
        let inputs = NODE_KINDS[kind].node_type().inputs;
        let mut index = 0;
        while index < inputs.len() {
            assert!(
                !(inputs[index].repeating && inputs[index].required),
                "a repeating input is never required"
            );
            // A result's key lists a node's inputs in one run, which says where the stack starts
            // only while nothing follows it.
            assert!(
                !inputs[index].repeating || index + 1 == inputs.len(),
                "only the last input repeats"
            );
            index += 1;
        }
        kind += 1;
    }
};

/// One image a node type takes, from another node's output, or for a repeating input a numbered
/// series of them.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeInput {
    /// Unchanged across labels: connections and scripts address it, and it names the input's
    /// attribute, `inputs:` and the name, or for a repeating input's slot the name and its number.
    pub name: &'static str,
    pub label: &'static str,
    /// Whether the input is a stack rather than one image: a series of numbered slots, each
    /// taking an image, of which the node receives the connected ones in slot order. An empty
    /// slot is no part of the result, so a node to which a slot's number matters needs another
    /// kind of input.
    pub repeating: bool,
    /// Whether the node reports the input as missing while it takes no image, rather than making
    /// do without it. Never set on a repeating input: no node type asks for at least one slot.
    pub required: bool,
}

impl NodeInput {
    /// The slot a wire takes when the input is named alone: the one slot of an input that does
    /// not repeat, or a repeating input's first.
    pub fn first_slot(&self) -> InputSlot<'static> {
        InputSlot {
            input: self.name,
            slot: u32::from(self.repeating),
        }
    }
}
