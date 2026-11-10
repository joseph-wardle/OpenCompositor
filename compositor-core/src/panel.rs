//! What the parameter panel shows: one node's parameters, their values, and which were set.

use crate::document::{Document, NodeId};
use crate::files::{Files, scanned_end};
use crate::parameter::{Parameter, Value, resolve};

/// One node's parameters as the panel shows them.
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    pub node: NodeId,
    pub name: String,
    pub type_label: &'static str,
    /// Whether the panel stays on this node whatever is selected.
    pub pinned: bool,
    /// In description order.
    pub rows: Vec<Row>,
}

/// One parameter of the panel's node.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub parameter: &'static Parameter,
    /// What evaluation reads at the frame: the authored value or the default, or what the node
    /// decides for an unset value without a default, such as a sequence's first frame found. None
    /// for a file not set, or for what the node cannot decide yet.
    pub value: Option<Value>,
    pub state: ValueState,
}

/// Whether a value was set, which the panel shows apart from what it resolves to: a value set to
/// its default is still set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueState {
    Authored,
    Absent,
}

impl Panel {
    /// The node's panel at a frame, or `None` if the document has no such node.
    pub(crate) fn of(
        document: &Document,
        files: &Files,
        node: NodeId,
        pinned: bool,
        frame: f64,
    ) -> Option<Panel> {
        let shown = document.node(node)?;
        let resolved = resolve(shown, frame);
        let rows = (shown.kind.parameters().iter())
            .map(|parameter| Row {
                parameter,
                value: (resolved.get(parameter.name).cloned())
                    .or_else(|| scanned_end(&resolved, files, parameter.name).map(Value::Float)),
                state: match shown.authored_value(parameter.name) {
                    Some(_) => ValueState::Authored,
                    None => ValueState::Absent,
                },
            })
            .collect();
        Some(Panel {
            node,
            name: shown.name(),
            type_label: shown.kind.label(),
            pinned,
            rows,
        })
    }
}
