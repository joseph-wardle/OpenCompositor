//! The execution primitives every operation is written against: a map, a fold of a stack of
//! layers and a windowed line reduction. An operation is a value implementing one of their
//! traits; each primitive has one CPU implementation, in compositor-core, and one GPU
//! implementation, the generic functions of [`crate::gpu`], whose entry points only name an
//! operation.

use core::ops::{Add, Mul};

/// A dispatch's parameters, which the host writes as their bytes into the storage buffer the
/// entry point reads.
///
/// # Safety
///
/// Only for `#[repr(C)]` structs whose fields are four-byte values, or arrays or such structs
/// of them: they have no padding, so every byte is initialised, and a storage buffer's offsets
/// are the struct's.
pub unsafe trait Params: Copy + Sync {}

/// Where an input's plane lies in the output's, in the output's texels: zero outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Placement {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Placement {
    /// Whether the plane covers the output's texel `(x, y)`.
    #[inline(always)]
    pub fn covers(&self, x: i32, y: i32) -> bool {
        let (x, y) = (x - self.x, y - self.y);
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }
}

// SAFETY: `#[repr(C)]`, four-byte fields.
unsafe impl Params for Placement {}

/// An operation each of whose `O` output samples is a function of its `I` inputs' samples at
/// the same position.
pub trait Map<const I: usize, const O: usize>: Params {
    /// Its GPU entry point.
    const ENTRY: &'static str;

    /// The outputs at one position. An input whose placement does not cover it gives zero.
    fn apply(&self, samples: [f32; I]) -> [f32; O];
}

/// A map's parameters: where each input lies in the output, the output's size and the
/// operation's own. The placements come first, which keeps every field four-byte aligned.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MapParams<M, const I: usize> {
    pub places: [Placement; I],
    pub width: u32,
    pub height: u32,
    pub op: M,
}

// SAFETY: `#[repr(C)]`, four-byte fields, `Placement`s and an operation that is `Params`.
unsafe impl<M: Params, const I: usize> Params for MapParams<M, I> {}

/// A map over a stack: at each position, the base's `O` samples folded through the samples of
/// each layer that covers it, a layer being `G` planes over one window, then finished against
/// the base's.
pub trait Fold<const G: usize, const O: usize>: Params {
    /// Its GPU entry points: one layer onto the stack below it, and the last layer with the
    /// finish.
    const STEP: &'static str;
    const LAST: &'static str;

    /// The stack after a layer whose samples are `layer`.
    fn step(&self, below: [f32; O], layer: [f32; G]) -> [f32; O];

    fn finish(&self, base: [f32; O], stack: [f32; O]) -> [f32; O];
}

/// A fold's parameters for one layer: where the base, the stack below and the layer lie in
/// the output, the output's size and the operation's own.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FoldParams<F> {
    pub base: Placement,
    pub below: Placement,
    pub layer: Placement,
    pub width: u32,
    pub height: u32,
    pub op: F,
}

// SAFETY: `#[repr(C)]`, four-byte fields, `Placement`s and an operation that is `Params`.
unsafe impl<F: Params> Params for FoldParams<F> {}

/// The arithmetic a line reduction accumulates in.
pub trait Real: Copy + Send + Add<Output = Self> + Mul<Output = Self> {
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    /// `self * b + c` with one rounding.
    fn mul_add(self, b: Self, c: Self) -> Self;
}

impl Real for f32 {
    #[inline(always)]
    fn from_f32(v: f32) -> f32 {
        v
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn mul_add(self, b: f32, c: f32) -> f32 {
        crate::mul_add(self, b, c)
    }
}

impl Real for f64 {
    #[inline(always)]
    fn from_f32(v: f32) -> f64 {
        f64::from(v)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn mul_add(self, b: f64, c: f64) -> f64 {
        #[cfg(target_arch = "spirv")]
        {
            libm::fma(self, b, c)
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            f64::mul_add(self, b, c)
        }
    }
}

/// An operation each of whose outputs along a line reduces a window of `2 t + 1` input samples,
/// zero beyond the line's ends: the core, every sample but the window's two ends, by `combine`
/// from `identity` in order, then `finish` with the ends.
pub trait LineReduce: Params {
    /// What the CPU accumulates in, the definition's arithmetic. The GPU, which lacks f64,
    /// always accumulates in f32.
    type Acc: Real;

    /// Its GPU entry points: each output's window reduced directly, and by blocks.
    const DIRECT: &'static str;
    const SCAN: &'static str;

    /// The window's half-width, at least 1.
    fn t(&self) -> u32;

    /// How far output 0's window starts before input 0: `t` keeps the line's length, `2 t`
    /// grows it by `2 t`.
    fn shift(&self) -> u32;

    fn identity<R: Real>(&self) -> R;
    fn combine<R: Real>(&self, a: R, b: R) -> R;
    fn finish<R: Real>(&self, core: R, first: R, last: R) -> f32;

    /// How much longer each pass's output line is than its input.
    #[inline(always)]
    fn growth(&self) -> u32 {
        assert!(
            self.shift() >= self.t(),
            "a window starting at or before its output"
        );
        2 * (self.shift() - self.t())
    }
}

/// The axis a line reduction runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Axis {
    /// Each row over itself.
    Rows = 0,
    /// Each column over itself.
    Columns = 1,
}

/// A line reduction's parameters: the output's size, the axis as [`Axis`]'s discriminant, the
/// input's length along it and the operation's own.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LineParams<L> {
    pub width: u32,
    pub height: u32,
    pub axis: u32,
    pub len: u32,
    pub op: L,
}

// SAFETY: `#[repr(C)]`, four-byte fields and an operation that is `Params`.
unsafe impl<L: Params> Params for LineParams<L> {}

/// The map's, the fold's and the direct line reduction's workgroup is this many invocations on
/// each side; their `threads` must say the same.
pub const WORKGROUP: u32 = 16;

/// Line reductions whose half-width is at most this are reduced directly on the GPU, each
/// output its own window; wider ones by the scan's blocks.
///
/// Direct sums cost `2 t + 1` taps per output, the scan about the same at any width until its
/// workgroup holds a single line, and the two crossed between `t = 5` and `6` on an RTX 4070 Ti
/// SUPER at 4K.
pub const DIRECT_T: u32 = 5;

/// Invocations in a scan's workgroup, the most wgpu's default limits allow; the scan's
/// `threads` must say the same.
pub const SCAN_THREADS: u32 = 256;

/// The longest block the scan holds in workgroup memory: the widest window it reduces has
/// `2 t - 1` below this. Its entry points declare the array's length as this.
///
/// Blur's largest Size, 4096, has `t = 1365`. With the scan's totals the workgroup holds
/// 13,056 bytes, inside the 16,384 that Vulkan guarantees and wgpu's default limits ask for.
pub const SCAN_BLOCK: u32 = 2752;

/// How many adjacent lines a scan's workgroup reduces for half-width `t`: the most, a power of
/// two up to [`SCAN_THREADS`], whose blocks fit the workgroup's memory together. Its invocations
/// take the lines in turn, so neighbouring invocations read neighbouring lines, which along
/// columns are neighbouring texels.
#[inline(always)]
pub fn scan_lines(t: u32) -> u32 {
    let l = 2 * t - 1;
    let mut lines = SCAN_THREADS;
    while lines > 1 && lines * l > SCAN_BLOCK {
        lines /= 2;
    }
    lines
}
