//! Computing a node's image on the CPU, without a window or a GPU.

use std::fmt;
use std::path::PathBuf;

use crate::document::{Document, NodeId, NodeKind};
use crate::image::Image;
use crate::read::{ReadError, read_image};

/// Computes a node's image from the document as it stands.
///
/// This is the headless path and the Reader's CPU implementation, not yet a general operator
/// contract: requests for time, region and channels arrive with the first operator that has an
/// input.
pub fn evaluate(document: &Document, id: NodeId) -> Result<Image, EvalError> {
    let node = document.node(id).ok_or(EvalError::MissingNode(id))?;
    match &node.kind {
        NodeKind::Reader { path } => read_image(path).map_err(|error| EvalError::Read {
            path: path.clone(),
            error,
        }),
        NodeKind::Exposure { .. } => Err(EvalError::NoInput(id)),
    }
}

#[derive(Debug)]
pub enum EvalError {
    MissingNode(NodeId),
    /// The node needs an input it does not have. Until connections arrive, no node has one.
    NoInput(NodeId),
    Read {
        path: PathBuf,
        error: ReadError,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::MissingNode(id) => write!(f, "node {} no longer exists", id.0),
            EvalError::NoInput(id) => write!(f, "node {} has no input", id.0),
            EvalError::Read { path, error } => write!(f, "{}: {error}", path.display()),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EvalError::MissingNode(_) | EvalError::NoInput(_) => None,
            EvalError::Read { error, .. } => Some(error),
        }
    }
}
