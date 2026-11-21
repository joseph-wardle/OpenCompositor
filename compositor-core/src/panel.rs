//! What the parameter panel shows: one node's parameters, their values, and which were set.

use crate::files::scanned_end;
use crate::network::NodeId;
use crate::parameter::{Parameter, ParameterKind, Value, resolve};
use crate::session::{KeyState, Property, RefusedConnection, Session};

/// One node's parameters as the panel shows them.
#[derive(Clone, Debug, PartialEq)]
pub struct Panel {
    pub node: NodeId,
    pub name: String,
    pub type_label: &'static str,
    /// Whether the panel stays on this node whatever is selected.
    pub pinned: bool,
    /// Why the stage's definition of the node was refused, if it was.
    pub problem: Option<String>,
    /// In description order.
    pub rows: Vec<Row>,
    /// The frames at which any parameter has a time sample, whichever file authors it,
    /// ascending and each once.
    pub key_frames: Vec<f64>,
    /// The declared inputs whose connections were refused.
    pub refused: Vec<RefusedConnection>,
    /// What else the node's prim authors, by name.
    pub properties: Vec<Property>,
}

/// One parameter of the panel's node.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub parameter: &'static Parameter,
    /// The authored value as written, a relative file unanchored; otherwise what evaluation reads
    /// at the frame: an animated value there, the default, or what the node decides for an unset value without a default,
    /// such as a sequence's first frame found. None for a file not set, or for what the node
    /// cannot decide yet.
    pub value: Option<Value>,
    pub state: ValueState,
    /// How the parameter stands in time at the frame. `None` for one that takes no keys, or is
    /// animated in a way this version does not read.
    pub keys: Option<KeyState>,
    /// A number's type as authored, if other than `double`: `float`, `half` or `int`.
    pub number: Option<&'static str>,
}

/// Whether a value or animation was set, which the panel shows apart from what it resolves to: a
/// value set to its default is still set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueState {
    Authored,
    Absent,
}

impl Panel {
    /// The node's panel at a frame, or `None` if the network has no such node.
    pub(crate) fn of(session: &Session, node: NodeId, pinned: bool) -> Option<Panel> {
        let (files, frame) = (session.files(), session.frame());
        let shown = session.network().node(node)?;
        let resolved = resolve(shown, frame);
        let rows = (shown.kind.node_type().parameters.iter())
            .map(|parameter| Row {
                parameter,
                value: (shown.authored_value(parameter.name).cloned())
                    .or_else(|| resolved.get(parameter.name).cloned())
                    .or_else(|| scanned_end(&resolved, files, parameter.name).map(Value::Float)),
                state: if shown.authored_value(parameter.name).is_some()
                    || shown.samples(parameter.name).is_some()
                {
                    ValueState::Authored
                } else {
                    ValueState::Absent
                },
                keys: session.key_state(node, parameter.name).ok(),
                number: match parameter.kind {
                    ParameterKind::Float(_) | ParameterKind::Frame { .. } => {
                        session.number_type(node, parameter.name)
                    }
                    ParameterKind::File => None,
                },
            })
            .collect();
        let mut key_frames: Vec<f64> = (shown.kind.node_type().parameters.iter())
            .filter_map(|parameter| shown.samples(parameter.name))
            .flat_map(|samples| samples.frames())
            .collect();
        key_frames.sort_by(f64::total_cmp);
        key_frames.dedup();
        Some(Panel {
            node,
            key_frames,
            name: shown.name().to_owned(),
            type_label: shown.kind.node_type().label,
            pinned,
            problem: shown.problem().map(str::to_owned),
            rows,
            refused: session.refused_connections(node),
            properties: session.other_properties(node),
        })
    }
}
