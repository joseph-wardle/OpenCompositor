//! The node types: what each offers to be set, the images it takes, and how it is created.

use crate::network::InputSlot;
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Reads the image file its File names, or for a sequence the file numbered the frame less its
    /// Offset, holding the nearest end of its range, from First to Last, outside it. Without a
    /// file it gives the empty image.
    Reader,
    /// Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied
    /// pixels that equals unpremultiplying, scaling and premultiplying again.
    Exposure,
    /// Composes its layers over its background, each layer in slot order over the result so far,
    /// then dissolves that stack by its Mix against the base, the lowest connected input.
    Merge,
    /// Blurs its source by its Size, a radius in pixels, every channel as it is, so that
    /// premultiplied edges fade without an unpremultiply; the data window grows by the reach.
    Blur,
}

impl NodeKind {
    /// What the node offers to be set.
    pub fn parameters(&self) -> &'static [Parameter] {
        match self {
            NodeKind::Reader => READER_PARAMETERS,
            NodeKind::Exposure => EXPOSURE_PARAMETERS,
            NodeKind::Merge => MERGE_PARAMETERS,
            NodeKind::Blur => BLUR_PARAMETERS,
        }
    }

    /// What the artist calls the node's type.
    pub fn label(&self) -> &'static str {
        match self {
            NodeKind::Reader => "Reader",
            NodeKind::Exposure => "Exposure",
            NodeKind::Merge => "Merge",
            NodeKind::Blur => "Blur",
        }
    }

    /// The images the node takes.
    pub fn inputs(&self) -> &'static [NodeInput] {
        match self {
            NodeKind::Reader => &[],
            NodeKind::Exposure => EXPOSURE_INPUTS,
            NodeKind::Merge => MERGE_INPUTS,
            NodeKind::Blur => BLUR_INPUTS,
        }
    }

    /// The prim type a node of this kind is authored as. Types, parameters and defaults change in
    /// place until 1.0; after it, a breaking change takes USD's suffix, as `OcExposure_1`.
    pub fn type_name(&self) -> &'static str {
        match self {
            NodeKind::Reader => "OcReader",
            NodeKind::Exposure => "OcExposure",
            NodeKind::Merge => "OcMerge",
            NodeKind::Blur => "OcBlur",
        }
    }

    pub(crate) fn of_type(type_name: &str) -> Option<NodeKind> {
        (NODE_TYPES.iter())
            .map(|node_type| node_type.kind.clone())
            .find(|kind| kind.type_name() == type_name)
    }
}

/// A node type the artist and scripts can create.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeType {
    /// Unchanged across labels: scripts create nodes by it.
    pub name: &'static str,
    /// What creating one names in the undo history.
    pub(crate) step: &'static str,
    pub(crate) kind: NodeKind,
}

const NODE_TYPES: &[NodeType] = &[
    NodeType {
        name: "reader",
        step: "Create Reader",
        kind: NodeKind::Reader,
    },
    NodeType {
        name: "exposure",
        step: "Create Exposure",
        kind: NodeKind::Exposure,
    },
    NodeType {
        name: "merge",
        step: "Create Merge",
        kind: NodeKind::Merge,
    },
    NodeType {
        name: "blur",
        step: "Create Blur",
        kind: NodeKind::Blur,
    },
];

impl NodeType {
    pub fn label(&self) -> &'static str {
        self.kind.label()
    }
}

/// Every node type that can be created, in menu order.
pub fn node_types() -> &'static [NodeType] {
    NODE_TYPES
}

/// One image a node type takes, from another node's output, or for a repeating input a numbered
/// series of them.
#[derive(Clone, Debug, PartialEq)]
pub struct NodeInput {
    /// Unchanged across labels: connections and scripts address it, and it names the input's
    /// attribute, `inputs:` and the name, or for a repeating input's slot the name and its number.
    pub name: &'static str,
    pub label: &'static str,
    /// Whether the input is a series of slots, each taking an image, rather than one image.
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

const READER_PARAMETERS: &[Parameter] = &[
    Parameter {
        name: "file",
        label: "File",
        kind: ParameterKind::File,
    },
    // Unset until the artist sets them, rather than filled in by the scan: a scan finishes on its
    // own thread, and an edit arriving then would land outside undo's order.
    Parameter {
        name: "first",
        label: "First",
        kind: ParameterKind::Frame { default: None },
    },
    Parameter {
        name: "last",
        label: "Last",
        kind: ParameterKind::Frame { default: None },
    },
    // Added to the file's frame, as Nuke's frame offset is: at 10, file frame 1 shows at 11.
    Parameter {
        name: "offset",
        label: "Offset",
        kind: ParameterKind::Frame { default: Some(0.0) },
    },
];

const EXPOSURE_PARAMETERS: &[Parameter] = &[Parameter {
    name: "stops",
    label: "Stops",
    kind: ParameterKind::Float(FloatParameter {
        default: 0.0,
        slider: -6.0..=6.0,
        // Keeps the gain, two to these powers, many orders of magnitude inside f32's range.
        limits: -64.0..=64.0,
    }),
}];

const EXPOSURE_INPUTS: &[NodeInput] = &[NodeInput {
    name: "source",
    label: "Source",
    repeating: false,
    required: true,
}];

const MERGE_PARAMETERS: &[Parameter] = &[Parameter {
    name: "mix",
    label: "Mix",
    kind: ParameterKind::Float(FloatParameter {
        default: 1.0,
        slider: 0.0..=1.0,
        limits: 0.0..=1.0,
    }),
}];

// Neither is required: the base is the lowest connected input, whichever that is, and nothing
// connected gives the empty image.
const MERGE_INPUTS: &[NodeInput] = &[
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
];

const BLUR_PARAMETERS: &[Parameter] = &[Parameter {
    name: "size",
    label: "Size",
    kind: ParameterKind::Float(FloatParameter {
        // A new Blur changes nothing until its size is set.
        default: 0.0,
        slider: 0.0..=100.0,
        limits: 0.0..=4096.0,
    }),
}];

const BLUR_INPUTS: &[NodeInput] = &[NodeInput {
    name: "source",
    label: "Source",
    repeating: false,
    required: true,
}];

const _: () = {
    let tables = [EXPOSURE_INPUTS, MERGE_INPUTS, BLUR_INPUTS];
    let mut table = 0;
    while table < tables.len() {
        let mut index = 0;
        while index < tables[table].len() {
            let input = &tables[table][index];
            assert!(
                !(input.repeating && input.required),
                "a repeating input is never required"
            );
            index += 1;
        }
        table += 1;
    }
};
