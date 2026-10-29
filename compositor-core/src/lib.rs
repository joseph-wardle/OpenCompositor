//! OpenCompositor without its Qt shell: the document, its history and its presentation in OpenNG,
//! evaluation, and the application the shell dispatches events to, all usable and testable without
//! a window or a GPU.

mod app;
mod document;
mod evaluate;
#[cfg(test)]
mod fixtures;
mod gpu;
mod image;
mod kernels;
mod panel;
mod parameter;
mod presentation;
mod read;
mod session;
mod viewer;
mod worker;

/// What `benches/kernels.rs` measures. The benchmark is a crate of its own, so this is the one
/// place the pool and the CPU kernels are reachable from outside; nothing else is public here.
#[doc(hidden)]
pub mod bench {
    pub use crate::evaluate::Pool;
    pub use crate::kernels::cpu::{Level, Stopped, fold, line, map};
}

pub use app::{Anchor, App, Command, Event, FileIntent, Input, Key, Notice, Pane, Request};
pub use document::{
    Assignment, BlurVersion, Changes, ConnectionError, Document, ExposureVersion, InputSlot,
    MAX_SLOTS, MergeVersion, Missing, Node, NodeId, NodeInput, NodeKind, NodeType, ReaderVersion,
    Slot, missing, node_types,
};
pub use evaluate::{
    EvalError, Evaluator, GpuEvaluation, GpuImage, Recorded, Reloads, ResultKey, result_key,
};
pub use gpu::{FrameCheck, FrameTiming, NativeWindow, PaneSize};
pub use image::{Channel, Chromaticities, Image, ImageInfo, Oversize, Plane, Warning, Window};
pub use panel::{Panel, Row, ValueState};
pub use parameter::{
    FloatParameter, Parameter, ParameterKind, Resolved, Value, ValueError, resolve,
};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session};
pub use viewer::{EXPOSURE_STEP, MAX_EXPOSURE, Placement, Viewer};
pub use worker::{Evaluated, EvaluationDevice, StagingTiming, Timing, ViewerError, ViewerImage};
