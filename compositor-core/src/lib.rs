//! OpenCompositor without its Qt shell: the document, its history and its presentation in OpenNG,
//! all usable and testable without a window or a GPU.

mod document;
mod presentation;
mod session;

pub use document::{Changes, Document, Node, NodeId, NodeKind, Slot};
pub use session::{EditError, Session};
