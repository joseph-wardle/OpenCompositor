//! The primitives' GPU halves and the entry points, one per operation, that instantiate them:
//! rust-gpu needs a `#[spirv]` function for each entry and cannot generate them generically.
//! Every plane is an R32Float texture; invocations past an output's edge return, since an
//! image store outside the texture is discarded only under robust image access.

use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::{IVec2, UVec3, Vec4};
use spirv_std::{Image, spirv};

use crate::{Fold, FoldParams, LineParams, LineReduce, Map, MapParams, Placement};

type Input = Image!(2D, type = f32, sampled);
type Output = Image!(2D, format = r32f, sampled = false);

fn texel(image: &Input, at: IVec2) -> f32 {
    let texel: Vec4 = image.fetch(at);
    texel.x
}

/// The sample of a plane placed by `place` at the output's texel `at`, and whether the plane
/// covers it: zero where it does not. A fetch outside a texture is defined only under robust
/// image access, so it is never made.
fn placed(image: &Input, at: IVec2, place: &Placement) -> (f32, bool) {
    if place.covers(at.x, at.y) {
        (texel(image, at - IVec2::new(place.x, place.y)), true)
    } else {
        (0.0, false)
    }
}

/// A map of one input to one output, one invocation per texel.
fn map_1_1<M: Map<1, 1>>(id: UVec3, params: &MapParams<M, 1>, a: &Input, output: &Output) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let at = IVec2::new(id.x as i32, id.y as i32);
    let (a, inside) = placed(a, at, &params.places[0]);
    let [v] = params.op.apply([a], [inside]);
    // SAFETY: `at` is inside the output, which the host sizes to the params, and the output is
    // bound as R32Float, the format this image type declares.
    unsafe { output.write(at, v) };
}

type Planes<'a> = (&'a Input, &'a Input, &'a Input, &'a Input);
type Outputs<'a> = (&'a Output, &'a Output, &'a Output, &'a Output);

/// Four planes' samples at the output's texel `at`, placed by `place`, and whether they cover
/// it.
fn placed_4(planes: Planes, at: IVec2, place: &Placement) -> ([f32; 4], bool) {
    let (a, inside) = placed(planes.0, at, place);
    let (b, _) = placed(planes.1, at, place);
    let (c, _) = placed(planes.2, at, place);
    let (d, _) = placed(planes.3, at, place);
    ([a, b, c, d], inside)
}

/// The stack at the output's texel `at` after a layer of four planes: the stack below where
/// the layer is not.
fn folded_4<F: Fold<4, 4>>(
    at: IVec2,
    params: &FoldParams<F>,
    below: Planes,
    layer: Planes,
) -> [f32; 4] {
    let (below, _) = placed_4(below, at, &params.below);
    let (layer, inside) = placed_4(layer, at, &params.layer);
    if inside {
        params.op.step(below, layer)
    } else {
        below
    }
}

/// Writes four samples to four planes at `at`.
///
/// # Safety
///
/// `at` is inside each output, and each is bound as R32Float.
unsafe fn write_4(outputs: Outputs, at: IVec2, v: [f32; 4]) {
    // SAFETY: as the caller ensures.
    unsafe {
        outputs.0.write(at, v[0]);
        outputs.1.write(at, v[1]);
        outputs.2.write(at, v[2]);
        outputs.3.write(at, v[3]);
    }
}

/// One layer of four planes onto the stack below it, one invocation per texel.
fn fold_4_step<F: Fold<4, 4>>(
    id: UVec3,
    params: &FoldParams<F>,
    below: Planes,
    layer: Planes,
    outputs: Outputs,
) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let at = IVec2::new(id.x as i32, id.y as i32);
    let stack = folded_4(at, params, below, layer);
    // SAFETY: `at` is inside the outputs, which the host sizes to the params, and each is bound
    // as R32Float.
    unsafe { write_4(outputs, at, stack) };
}

/// The last layer of four planes onto the stack below it, finished against the base.
fn fold_4_last<F: Fold<4, 4>>(
    id: UVec3,
    params: &FoldParams<F>,
    base: Planes,
    below: Planes,
    layer: Planes,
    outputs: Outputs,
) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let at = IVec2::new(id.x as i32, id.y as i32);
    let stack = folded_4(at, params, below, layer);
    let (base, _) = placed_4(base, at, &params.base);
    // SAFETY: as for `fold_4_step`.
    unsafe { write_4(outputs, at, params.op.finish(base, stack)) };
}

/// The texel `position` along line `line` of a plane, the line a row for `axis` 0 and a column
/// otherwise.
fn along(axis: u32, line: i32, position: i32) -> IVec2 {
    if axis == 0 {
        IVec2::new(position, line)
    } else {
        IVec2::new(line, position)
    }
}

/// Where a pass reads along its lines: the axis, the input's length along it and the windows'
/// shift, read out of the parameters once rather than from the storage buffer at every sample.
#[derive(Clone, Copy)]
struct Reach {
    axis: u32,
    len: i32,
    shift: i32,
}

impl Reach {
    fn of<L: LineReduce>(params: &LineParams<L>) -> Reach {
        Reach {
            axis: params.axis,
            len: params.len as i32,
            shift: params.op.shift() as i32,
        }
    }
}

/// The input along a line as the windows read it: sample `i - shift`, zero beyond the input's
/// ends.
fn padded(input: &Input, reach: Reach, line: i32, i: i32) -> f32 {
    let position = i - reach.shift;
    if position < 0 || position >= reach.len {
        0.0
    } else {
        texel(input, along(reach.axis, line, position))
    }
}

/// One pass of a line reduction, each output its own window: output `o`'s is `padded[o ..= o +
/// 2 t]`, reduced in f32 in order.
fn line_direct<L: LineReduce>(id: UVec3, params: &LineParams<L>, input: &Input, output: &Output) {
    if id.x >= params.width || id.y >= params.height {
        return;
    }
    let at = IVec2::new(id.x as i32, id.y as i32);
    let (o, line) = if params.axis == 0 {
        (at.x, at.y)
    } else {
        (at.y, at.x)
    };
    let op = &params.op;
    let reach = Reach::of(params);
    let t = op.t() as i32;
    let mut core = op.identity::<f32>();
    for k in 1..2 * t {
        core = op.combine(core, padded(input, reach, line, o + k));
    }
    let first = padded(input, reach, line, o);
    let last = padded(input, reach, line, o + 2 * t);
    // SAFETY: `at` is inside the output, which the host sizes to the params, and the output is
    // bound as R32Float.
    unsafe { output.write(at, op.finish(core, first, last)) };
}

/// One pass of a line reduction for wide windows, a workgroup per block of `l = 2 t - 1` padded
/// samples of one line: the core of the window starting at offset `k` of block `j` is that
/// block's suffix from `k` and block `j + 1`'s first `k` samples. Each invocation reduces a run
/// of the block's offsets in order, and one invocation carries the runs' totals across the
/// workgroup, so every core combines exactly its window's samples, as a direct reduction would.
///
/// Dispatched as (blocks, lines): block `j` writes outputs `j l - 1 .. (j + 1) l - 1`.
fn line_scan<L: LineReduce>(
    group: UVec3,
    local: UVec3,
    params: &LineParams<L>,
    input: &Input,
    output: &Output,
    block: &mut [f32; 2752],
    totals: &mut [f32; 128],
) {
    // A copy out of the storage buffer, whose fields the loops would otherwise reread; it
    // slowed the direct pass, so only here.
    let params = *params;
    let params = &params;
    let op = &params.op;
    let reach = Reach::of(params);
    let threads = crate::SCAN_THREADS as i32;
    let t = op.t() as i32;
    let l = 2 * t - 1;
    let extent = if params.axis == 0 {
        params.width
    } else {
        params.height
    } as i32;
    let line = group.y as i32;
    let start = group.x as i32 * l;
    let i = local.x as i32;
    // This invocation's run of offsets.
    let run = (l + threads - 1) / threads;
    let first = (i * run).min(l);
    let last = (first + run).min(l);

    let mut k = i;
    while k < l {
        block[k as usize] = padded(input, reach, line, start + k);
        k += threads;
    }
    workgroup_memory_barrier_with_group_sync();
    let mut suffix = op.identity::<f32>();
    let mut k = last - 1;
    while k >= first {
        suffix = op.combine(suffix, block[k as usize]);
        k -= 1;
    }
    let mut prefix = op.identity::<f32>();
    for k in first..last {
        prefix = op.combine(prefix, padded(input, reach, line, start + l + k));
    }
    totals[i as usize] = suffix;
    totals[(threads + i) as usize] = prefix;
    workgroup_memory_barrier_with_group_sync();
    if i == 0 {
        // Each run's suffix total becomes the runs' after it, and its prefix total the runs'
        // before it.
        let mut after = op.identity::<f32>();
        let mut m = threads - 1;
        while m >= 0 {
            let own = totals[m as usize];
            totals[m as usize] = after;
            after = op.combine(after, own);
            m -= 1;
        }
        let mut before = op.identity::<f32>();
        for m in 0..threads {
            let own = totals[(threads + m) as usize];
            totals[(threads + m) as usize] = before;
            before = op.combine(before, own);
        }
    }
    workgroup_memory_barrier_with_group_sync();
    // The block's suffixes over this run, in place.
    let mut suffix = totals[i as usize];
    let mut k = last - 1;
    while k >= first {
        suffix = op.combine(suffix, block[k as usize]);
        block[k as usize] = suffix;
        k -= 1;
    }
    let mut prefix = totals[(threads + i) as usize];
    for k in first..last {
        // The window of output `o = start + k - 1` ends at `o + 2 t = start + l + k`, the next
        // block's sample that enters the prefix after it.
        let entering = padded(input, reach, line, start + l + k);
        let o = start + k - 1;
        if o >= 0 && o < extent {
            let core = op.combine(block[k as usize], prefix);
            let first = padded(input, reach, line, o);
            // SAFETY: `o` is inside the output along the axis and `line` across it, which the
            // host dispatches no further than, and the output is R32Float.
            unsafe {
                output.write(
                    along(params.axis, line, o),
                    op.finish(core, first, entering),
                )
            };
        }
        prefix = op.combine(prefix, entering);
    }
}

#[spirv(compute(threads(16, 16)))] // WORKGROUP
pub fn exposure(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &MapParams<
        crate::Exposure,
        1,
    >,
    #[spirv(descriptor_set = 0, binding = 1)] input: &Input,
    #[spirv(descriptor_set = 0, binding = 2)] output: &Output,
) {
    map_1_1(id, params, input, output);
}

#[spirv(compute(threads(16, 16)))] // WORKGROUP
pub fn merge_step(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &FoldParams<crate::Merge>,
    #[spirv(descriptor_set = 0, binding = 1)] below_r: &Input,
    #[spirv(descriptor_set = 0, binding = 2)] below_g: &Input,
    #[spirv(descriptor_set = 0, binding = 3)] below_b: &Input,
    #[spirv(descriptor_set = 0, binding = 4)] below_a: &Input,
    #[spirv(descriptor_set = 0, binding = 5)] layer_r: &Input,
    #[spirv(descriptor_set = 0, binding = 6)] layer_g: &Input,
    #[spirv(descriptor_set = 0, binding = 7)] layer_b: &Input,
    #[spirv(descriptor_set = 0, binding = 8)] layer_a: &Input,
    #[spirv(descriptor_set = 0, binding = 9)] r: &Output,
    #[spirv(descriptor_set = 0, binding = 10)] g: &Output,
    #[spirv(descriptor_set = 0, binding = 11)] b: &Output,
    #[spirv(descriptor_set = 0, binding = 12)] a: &Output,
) {
    let below = (below_r, below_g, below_b, below_a);
    let layer = (layer_r, layer_g, layer_b, layer_a);
    fold_4_step(id, params, below, layer, (r, g, b, a));
}

#[spirv(compute(threads(16, 16)))] // WORKGROUP
pub fn merge_last(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &FoldParams<crate::Merge>,
    #[spirv(descriptor_set = 0, binding = 1)] background_r: &Input,
    #[spirv(descriptor_set = 0, binding = 2)] background_g: &Input,
    #[spirv(descriptor_set = 0, binding = 3)] background_b: &Input,
    #[spirv(descriptor_set = 0, binding = 4)] background_a: &Input,
    #[spirv(descriptor_set = 0, binding = 5)] below_r: &Input,
    #[spirv(descriptor_set = 0, binding = 6)] below_g: &Input,
    #[spirv(descriptor_set = 0, binding = 7)] below_b: &Input,
    #[spirv(descriptor_set = 0, binding = 8)] below_a: &Input,
    #[spirv(descriptor_set = 0, binding = 9)] layer_r: &Input,
    #[spirv(descriptor_set = 0, binding = 10)] layer_g: &Input,
    #[spirv(descriptor_set = 0, binding = 11)] layer_b: &Input,
    #[spirv(descriptor_set = 0, binding = 12)] layer_a: &Input,
    #[spirv(descriptor_set = 0, binding = 13)] r: &Output,
    #[spirv(descriptor_set = 0, binding = 14)] g: &Output,
    #[spirv(descriptor_set = 0, binding = 15)] b: &Output,
    #[spirv(descriptor_set = 0, binding = 16)] a: &Output,
) {
    let background = (background_r, background_g, background_b, background_a);
    let below = (below_r, below_g, below_b, below_a);
    let layer = (layer_r, layer_g, layer_b, layer_a);
    fold_4_last(id, params, background, below, layer, (r, g, b, a));
}

#[spirv(compute(threads(16, 16)))] // WORKGROUP
pub fn blur_direct(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &LineParams<crate::BoxSum>,
    #[spirv(descriptor_set = 0, binding = 1)] input: &Input,
    #[spirv(descriptor_set = 0, binding = 2)] output: &Output,
) {
    line_direct(id, params, input, output);
}

#[spirv(compute(threads(64)))] // SCAN_THREADS
pub fn blur_scan(
    #[spirv(workgroup_id)] group: UVec3,
    #[spirv(local_invocation_id)] local: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] params: &LineParams<crate::BoxSum>,
    #[spirv(descriptor_set = 0, binding = 1)] input: &Input,
    #[spirv(descriptor_set = 0, binding = 2)] output: &Output,
    #[spirv(workgroup)] block: &mut [f32; 2752], // SCAN_BLOCK
    #[spirv(workgroup)] totals: &mut [f32; 128], // 2 SCAN_THREADS
) {
    line_scan(group, local, params, input, output, block, totals);
}
