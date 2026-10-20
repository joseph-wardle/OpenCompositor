//! What the parameter panel shows: one node's parameters, their values, and which were set.

use crate::document::{Document, NodeId};
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
    /// What evaluation reads: the authored value or the default, or none for a file not set.
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
    /// The node's panel, or `None` if the document has no such node.
    pub(crate) fn of(document: &Document, node: NodeId, pinned: bool) -> Option<Panel> {
        let shown = document.node(node)?;
        let resolved = resolve(shown);
        let rows = (shown.kind.parameters().iter())
            .map(|parameter| Row {
                parameter,
                value: resolved.get(parameter.name).cloned(),
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
