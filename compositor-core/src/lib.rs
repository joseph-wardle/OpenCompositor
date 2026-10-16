//! OpenCompositor without its Qt shell: the document, its history and its presentation in OpenNG,
//! evaluation, and the application the shell dispatches events to, all usable and testable without
//! a window or a GPU.

mod app;
mod document;
mod evaluate;
mod gpu;
mod image;
mod parameter;
mod presentation;
mod read;
mod session;
mod viewer;
mod worker;

pub use app::{App, Command, Event, FileIntent, Input, Key, Pane, Request};
pub use document::{
    Assignment, Changes, ConnectionError, Document, ExposureVersion, Node, NodeId, NodeInput,
    NodeKind, Slot, unconnected_inputs,
};
pub use evaluate::{EvalError, Evaluator, Reloads, ResultKey, result_key};
pub use gpu::{FrameCheck, NativeWindow, PaneSize};
pub use image::{
    Channel, Chromaticities, Cicp, ExrChannel, ExrCompression, ExrPart, ExrSource, Image,
    ImageInfo, PngColorType, PngSource, SampleType, Source, Warning, Window,
};
pub use parameter::{
    FloatParameter, Parameter, ParameterKind, Resolved, Value, ValueError, resolve,
};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session};
pub use viewer::{Placement, Viewer};
pub use worker::{Evaluated, ViewerError, ViewerImage};
