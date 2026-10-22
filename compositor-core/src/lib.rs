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
// The GPU half is used by the worker once it dispatches; until then only the tests do.
#[cfg_attr(not(test), expect(dead_code))]
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
    pub use crate::evaluate::blur::{BandScratch, band_passes};
    pub use crate::evaluate::{Columns, Pool};
    pub use crate::kernels::cpu::{
        BLUR_BAND, BLUR_ROWS, BoxScratch, Level, blur_band, blur_rows, exposure, mix, over,
    };
}

pub use app::{Anchor, App, Command, Event, FileIntent, Input, Key, Notice, Pane, Request};
pub use document::{
    Assignment, BlurVersion, Changes, ConnectionError, Document, ExposureVersion, InputSlot,
    MAX_SLOTS, MergeVersion, Missing, Node, NodeId, NodeInput, NodeKind, NodeType, ReaderVersion,
    Slot, missing, node_types,
};
pub use evaluate::{EvalError, Evaluator, Reloads, ResultKey, result_key};
pub use gpu::{FrameCheck, NativeWindow, PaneSize};
pub use image::{Channel, Chromaticities, Image, ImageInfo, Oversize, Plane, Warning, Window};
pub use panel::{Panel, Row, ValueState};
pub use parameter::{
    FloatParameter, Parameter, ParameterKind, Resolved, Value, ValueError, resolve,
};
pub use read::{MAX_PIXELS, ReadError, Refusal, read_image};
pub use session::{EditError, Session};
pub use viewer::{Placement, Viewer};
pub use worker::{Evaluated, StagingTiming, Timing, ViewerError, ViewerImage};
