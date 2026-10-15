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
mod viewer;

pub use app::{App, Command, Event, FileIntent, Input, Key, Pane, Request};
pub use document::{Changes, Document, Node, NodeId, NodeKind, Slot};
pub use evaluate::{EvalError, evaluate};
pub use gpu::{FrameCheck, NativeWindow, PaneSize};
pub use image::{
    Chromaticities, Cicp, ExrChannel, ExrCompression, ExrPart, ExrSource, Image, ImageInfo,
    PngColorType, PngSource, SampleType, Source, Warning, Window,
};
pub use loader::{Decoded, LoadError, LoadRequest, Loaded};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session, Viewed};
pub use viewer::{Placement, Viewer};
