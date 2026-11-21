//! Reads the image file its File names, or for a sequence the file numbered the frame less its
//! Offset, holding the nearest end of its range, from First to Last, outside it. Without a file
//! it gives the empty image.

use open_ng::Color;

use crate::nodes::NodeType;
use crate::parameter::{Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "reader",
    label: "Reader",
    type_name: "OcReader",
    step: "Create Reader",
    color: Color {
        r: 70,
        g: 110,
        b: 150,
    },
    parameters: &[
        Parameter {
            name: "file",
            label: "File",
            kind: ParameterKind::File,
        },
        // Unset until the artist sets them, rather than filled in by the scan: a scan finishes on
        // its own thread, and an edit arriving then would land outside undo's order.
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
    ],
    inputs: &[],
};
