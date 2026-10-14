//! OpenCompositor without its Qt shell: the document, its history and its presentation in OpenNG,
//! evaluation, and the application the shell dispatches events to, all usable and testable without
//! a window or a GPU.

mod app;
mod document;
mod evaluate;
mod gpu;
mod image;
mod loader;
mod presentation;
mod read;
mod session;

pub use app::{App, Event, FileIntent, Input, Key, Pane, Request};
pub use document::{Changes, Document, Node, NodeId, NodeKind, Slot};
pub use evaluate::{EvalError, evaluate};
pub use gpu::{NativeWindow, PaneSize};
pub use image::{
    Chromaticities, Cicp, ExrChannel, ExrCompression, ExrPart, ExrSource, Image, PngColorType,
    PngSource, SampleType, Source, Warning, Window,
};
pub use loader::{LoadRequest, Loaded};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session};
