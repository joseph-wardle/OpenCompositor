//! OpenCompositor without its Qt shell: the document, its history and its presentation in OpenNG,
//! and evaluation, all usable and testable without a window or a GPU.

mod document;
mod evaluate;
mod image;
mod presentation;
mod read;
mod session;

pub use document::{Changes, Document, Node, NodeId, NodeKind, Slot};
pub use evaluate::{EvalError, evaluate};
pub use image::{
    Chromaticities, Cicp, ExrChannel, ExrCompression, ExrPart, ExrSource, Image, PngColorType,
    PngSource, SampleType, Source, Warning, Window,
};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session};
