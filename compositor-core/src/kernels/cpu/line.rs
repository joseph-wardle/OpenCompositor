//! The line reduction's CPU half: passes of an operation along a plane's rows or columns, lines
//! of up to sixteen lanes so that the reductions, serial along a line, run as one vector across
//! lanes.

use std::sync::atomic::{AtomicBool, Ordering};

use opencompositor_kernels::{Axis, LineReduce, Real};

use super::at_level;
use crate::evaluate::Pool;

/// Rows per chunk of a line reduction along rows, the lanes its block reductions run over: the
/// reductions are serial along a row, so the rows of a chunk are interleaved and reduced as one
/// vector.
pub const LINE_ROWS: usize = 16;

/// Columns per band of a line reduction along columns: one cache line of samples, the lanes of
/// its reductions.
pub const LINE_BAND: usize = 16;

/// A single row's windows of up to this half-width are reduced directly, every output its own
/// window, which vectorises across the outputs of the row; wider ones by blocks, the rows
/// interleaved as lanes. Measured on the workstation for Blur: direct sums cost a tap each,
/// block sums about seven. Lines of lanes always take blocks: across 16 lanes the direct sums
/// were slower at every window, by a sixth at one tap each side and more than twice as slow
/// from three.
const DIRECT_T: usize = 6;

/// Asked to stop, a line reduction left some of its output unwritten.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

/// `passes` passes of `op` along `axis` of `input`, a plane of `width` by `height`, into
/// `output`, which each pass lengthens along the axis by the operation's growth. Along rows a
/// chunk of rows at a time, along columns a band of columns at a time, the band's columns the
/// lanes of one line. Each chunk and band asks `stop` first, and once told leaves the rest
/// undone.
#[expect(
    clippy::too_many_arguments,
    reason = "a plane, its size and its output"
)]
pub fn line<L: LineReduce>(
    pool: &Pool,
    op: &L,
    axis: Axis,
    passes: u32,
    input: &[f32],
    (width, height): (usize, usize),
    output: &mut [f32],
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<(), Stopped> {
    assert_eq!(input.len(), width * height, "the input is width by height");
    let growth = (op.growth() * passes) as usize;
    let stopped = AtomicBool::new(false);
    match axis {
        Axis::Rows => {
            let grown = width + growth;
            assert_eq!(output.len(), grown * height, "the output's rows are grown");
            pool.fill_rows_with(
                [output],
                grown,
                LINE_ROWS,
                LineScratch::default,
                |scratch, level, first_row, [dst]| {
                    if stop() {
                        stopped.store(true, Ordering::Relaxed);
                        return;
                    }
                    let rows = dst.len() / grown;
                    let src = &input[first_row * width..(first_row + rows) * width];
                    at_level(
                        level,
                        #[inline(always)]
                        || line_rows(op, passes, src, width, dst, scratch),
                    );
                },
            );
        }
        Axis::Columns => {
            let grown = height + growth;
            assert_eq!(
                output.len(),
                width * grown,
                "the output's columns are grown"
            );
            pool.fill_columns(
                output,
                width,
                LINE_BAND,
                LineScratch::default,
                |scratch, level, mut band| {
                    if stop() {
                        stopped.store(true, Ordering::Relaxed);
                        return;
                    }
                    let columns = band.columns();
                    let lanes = columns.len();
                    scratch.lanes.resize(height * LINE_BAND, 0.0);
                    let (gathered, _) = scratch.lanes.as_chunks_mut::<LINE_BAND>();
                    for (y, row) in gathered.iter_mut().enumerate() {
                        row[..lanes].copy_from_slice(
                            &input[y * width + columns.start..y * width + columns.end],
                        );
                        // The last band's missing lanes, which may hold an earlier band's.
                        row[lanes..].fill(0.0);
                    }
                    scratch.results.resize(grown * LINE_BAND, 0.0);
                    let LineScratch {
                        lanes: gathered,
                        results,
                        passes: pads,
                    } = scratch;
                    at_level(
                        level,
                        #[inline(always)]
                        || line_passes::<L, LINE_BAND>(op, passes, gathered, results, pads),
                    );
                    let (results, _) = scratch.results.as_chunks::<LINE_BAND>();
                    for (y, row) in results.iter().enumerate() {
                        band.write_row(y, &row[..lanes]);
                    }
                },
            );
        }
    }
    if stopped.load(Ordering::Relaxed) {
        Err(Stopped)
    } else {
        Ok(())
    }
}

/// The most bytes one thread's temporaries hold after [`line`] over lines of `len` samples,
/// along either axis: the lines as lanes and their results, the two padded lines the passes
/// alternate between, and one block's accumulators.
pub(crate) fn line_scratch_bytes<L: LineReduce>(op: &L, passes: u32, len: usize) -> u64 {
    let lanes = LINE_ROWS.max(LINE_BAND) as u64;
    let t = u64::from(op.t());
    let len = len as u64;
    let grown = len + u64::from(op.growth()) * u64::from(passes);
    let samples = len + grown + 2 * (grown + 2 * t);
    lanes * (samples * 4 + 2 * t * size_of::<L::Acc>() as u64)
}

/// A thread's temporaries, kept from call to call: the passes' own, and the lines interleaved
/// as lanes with their results. Empty until the first call sizes it.
struct LineScratch<A> {
    passes: PassScratch<A>,
    lanes: Vec<f32>,
    results: Vec<f32>,
}

impl<A> Default for LineScratch<A> {
    fn default() -> LineScratch<A> {
        LineScratch {
            passes: PassScratch {
                pads: [Vec::new(), Vec::new()],
                suffixes: Vec::new(),
            },
            lanes: Vec::new(),
            results: Vec::new(),
        }
    }
}

/// The passes' temporaries: the zero-padded lines they read, alternately, and one block's
/// suffixes.
struct PassScratch<A> {
    pads: [Vec<f32>; 2],
    suffixes: Vec<A>,
}

/// The passes along up to [`LINE_ROWS`] rows of `width` samples in `src`, into the same rows of
/// `dst`, each grown by the passes.
#[inline(always)]
fn line_rows<L: LineReduce>(
    op: &L,
    passes: u32,
    src: &[f32],
    width: usize,
    dst: &mut [f32],
    scratch: &mut LineScratch<L::Acc>,
) {
    let rows = src.len().checked_div(width).unwrap_or(0);
    assert!(rows <= LINE_ROWS, "at most LINE_ROWS rows");
    let grown = dst.len() / rows.max(1);
    if op.t() as usize <= DIRECT_T {
        for (src, dst) in src.chunks_exact(width).zip(dst.chunks_exact_mut(grown)) {
            line_passes::<L, 1>(op, passes, src, dst, &mut scratch.passes);
        }
        return;
    }
    // The chunk's rows as lanes, a column of the chunk per element; rows past the chunk's last
    // are lanes of zeros and their results are dropped. Sixteen columns at a time, so that the
    // lanes each row writes stay in cache until the last row has: across the whole width,
    // every write missed it, which cost a tenth of Blur's rows pass on the workstation.
    scratch.lanes.clear();
    scratch.lanes.resize(width * LINE_ROWS, 0.0);
    let (lanes, _) = scratch.lanes.as_chunks_mut::<LINE_ROWS>();
    for (x, tile) in lanes.chunks_mut(LINE_ROWS).enumerate() {
        for (row, src) in src.chunks_exact(width).enumerate() {
            for (lane, &sample) in tile.iter_mut().zip(&src[x * LINE_ROWS..]) {
                lane[row] = sample;
            }
        }
    }
    scratch.results.clear();
    scratch.results.resize(grown * LINE_ROWS, 0.0);
    line_passes::<L, LINE_ROWS>(
        op,
        passes,
        &scratch.lanes,
        &mut scratch.results,
        &mut scratch.passes,
    );
    let (results, _) = scratch.results.as_chunks::<LINE_ROWS>();
    for (x, tile) in results.chunks(LINE_ROWS).enumerate() {
        for (row, dst) in dst.chunks_exact_mut(grown).enumerate() {
            for (d, lane) in dst[x * LINE_ROWS..].iter_mut().zip(tile) {
                *d = lane[row];
            }
        }
    }
}

/// `passes` passes over one line of `n` elements of `LANES` lanes in `src`, into the `n +
/// passes growth` elements of `dst`: each pass from a zero-padded copy of its input, the window
/// of output `o` that copy's `o ..= o + 2 t`, into the middle of the next pass's copy, the last
/// into `dst`.
#[inline(always)]
fn line_passes<L: LineReduce, const LANES: usize>(
    op: &L,
    passes: u32,
    src: &[f32],
    dst: &mut [f32],
    scratch: &mut PassScratch<L::Acc>,
) {
    let t = op.t() as usize;
    let shift = op.shift() as usize;
    let growth = op.growth() as usize;
    let [a, b] = &mut scratch.pads;
    let mut n = src.len() / LANES;
    padded::<LANES>(a, n + growth + 2 * t, shift, n).copy_from_slice(src);
    for pass in 0..passes {
        let (z, next) = if pass % 2 == 0 {
            (&*a, &mut *b)
        } else {
            (&*b, &mut *a)
        };
        let n_out = n + growth;
        let out = if pass + 1 == passes {
            &mut dst[..]
        } else {
            padded::<LANES>(next, n_out + growth + 2 * t, shift, n_out)
        };
        window_pass::<L, LANES>(
            op,
            &z[..(n_out + 2 * t) * LANES],
            out,
            &mut scratch.suffixes,
        );
        n = n_out;
    }
}

/// `pad` as a line of `len` elements, zero but for the `n` from element `shift`, which it
/// returns for the caller to fill.
#[inline(always)]
fn padded<const LANES: usize>(
    pad: &mut Vec<f32>,
    len: usize,
    shift: usize,
    n: usize,
) -> &mut [f32] {
    pad.resize(len * LANES, 0.0);
    pad[..shift * LANES].fill(0.0);
    pad[(shift + n) * LANES..].fill(0.0);
    &mut pad[shift * LANES..(shift + n) * LANES]
}

/// One pass: output `o` reduces the window `z[o ..= o + 2 t]`, in the operation's accumulator.
/// The order is the same at every SIMD level, so the result is too.
#[inline(always)]
fn window_pass<L: LineReduce, const LANES: usize>(
    op: &L,
    z: &[f32],
    out: &mut [f32],
    suffixes: &mut Vec<L::Acc>,
) {
    if LANES > 1 {
        return block_pass::<L, LANES>(op, z, out, suffixes);
    }
    match op.t() {
        1 => direct_pass::<L, LANES, 1>(op, z, out),
        2 => direct_pass::<L, LANES, 2>(op, z, out),
        3 => direct_pass::<L, LANES, 3>(op, z, out),
        4 => direct_pass::<L, LANES, 4>(op, z, out),
        5 => direct_pass::<L, LANES, 5>(op, z, out),
        6 => direct_pass::<L, LANES, 6>(op, z, out),
        _ => block_pass::<L, LANES>(op, z, out, suffixes),
    }
}

// DIRECT_T is the last half-width reduced directly.
const _: () = assert!(DIRECT_T == 6);

/// Each output its own window of `2 T + 1` samples, the half-width fixed so the loop unrolls.
#[inline(always)]
fn direct_pass<L: LineReduce, const LANES: usize, const T: usize>(
    op: &L,
    z: &[f32],
    out: &mut [f32],
) {
    let (z, _) = z.as_chunks::<LANES>();
    let (out, _) = out.as_chunks_mut::<LANES>();
    for (o, out) in out.iter_mut().enumerate() {
        let window: &[[f32; LANES]] = &z[o..o + 2 * T + 1];
        for lane in 0..LANES {
            let mut core = op.identity::<L::Acc>();
            for tap in &window[1..2 * T] {
                core = op.combine(core, L::Acc::from_f32(tap[lane]));
            }
            let first = L::Acc::from_f32(window[0][lane]);
            let last = L::Acc::from_f32(window[2 * T][lane]);
            out[lane] = op.finish(core, first, last);
        }
    }
}

/// The window's core, every sample but the ends, by blocks: the line in blocks of the core's
/// length, the core starting at offset `k` of a block is that block's suffix from `k` and the
/// next block's prefix up to `k - 1`. `suffixes` holds one block's suffixes; the prefix
/// accumulates as the outputs go. Each core combines exactly its window's samples, so an
/// infinity or NaN reaches exactly the outputs whose window holds it, as a direct reduction's
/// would.
#[inline(always)]
fn block_pass<L: LineReduce, const LANES: usize>(
    op: &L,
    z: &[f32],
    out: &mut [f32],
    suffixes: &mut Vec<L::Acc>,
) {
    let t = op.t() as usize;
    let l = 2 * t - 1;
    let (z, _) = z.as_chunks::<LANES>();
    let (out, _) = out.as_chunks_mut::<LANES>();
    let n_z = z.len();
    let n_out = out.len();
    let identity = op.identity::<L::Acc>();
    suffixes.clear();
    suffixes.resize(l * LANES, identity);
    let (suffixes, _) = suffixes.as_chunks_mut::<LANES>();
    let finish = |o: usize, core: [L::Acc; LANES], out: &mut [f32; LANES]| {
        let (first, last) = (z[o], z[o + 2 * t]);
        for lane in 0..LANES {
            let (first, last) = (L::Acc::from_f32(first[lane]), L::Acc::from_f32(last[lane]));
            out[lane] = op.finish(core[lane], first, last);
        }
    };
    // Output `o` has the core `z[o + 1 ..= o + l]`: the cores start from 1 to `n_out`.
    let mut start = 0;
    while start <= n_out {
        let end = (start + l).min(n_z);
        let mut acc = [identity; LANES];
        for (suffix, sample) in suffixes[..end - start].iter_mut().zip(&z[start..end]).rev() {
            for lane in 0..LANES {
                acc[lane] = op.combine(acc[lane], L::Acc::from_f32(sample[lane]));
            }
            *suffix = acc;
        }
        // The core starting at the block's start is the whole block; none starts at 0.
        if start > 0 {
            finish(start - 1, suffixes[0], &mut out[start - 1]);
        }
        let mut prefix = [identity; LANES];
        for k in 1..l.min(n_out - start + 1) {
            let o = start + k - 1;
            let entering = z[start + k + l - 1];
            let mut core = suffixes[k];
            for lane in 0..LANES {
                prefix[lane] = op.combine(prefix[lane], L::Acc::from_f32(entering[lane]));
                core[lane] = op.combine(core[lane], prefix[lane]);
            }
            finish(o, core, &mut out[o]);
        }
        start += l;
    }
}

#[cfg(test)]
mod tests {
    use opencompositor_kernels::{BoxPass, BoxSum, flush, ulp_distance};

    use super::super::{Kind, Level};
    use super::*;

    /// The CPU's K for Blur's three passes along one axis, each rounding to f32 once, half an
    /// f32 unit of its result's magnitude; the node's six take 8.
    const K: f64 = 3.0;

    /// Pseudo-random scene values of both signs over forty stops.
    fn values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let bits = (state >> 32) as u32;
                let magnitude = (f64::from(bits & 0xffff) / 65536.0 * 40.0 - 20.0).exp2() as f32;
                if bits & 0x1_0000 != 0 {
                    -magnitude
                } else {
                    magnitude
                }
            })
            .collect()
    }

    /// Lines of `len` samples: the values, then the values with a spike at f32's limit and its
    /// negation next to it, with an infinity of each sign, and with a NaN, each where the line
    /// has the room.
    fn lines(len: usize) -> Vec<Vec<f32>> {
        let base = values(0x2545_f491_4f6c_dd1d, len);
        let mut hdr = base.clone();
        let mut inf = base.clone();
        let mut nan = base.clone();
        let put = |line: &mut Vec<f32>, at: usize, value: f32| {
            if at < len {
                line[at] = value;
            }
        };
        put(&mut hdr, len / 3, 3.0e38);
        put(&mut hdr, len / 3 + 1, -3.0e38);
        put(&mut hdr, len / 2, 1.0e30);
        put(&mut inf, len / 8, f32::INFINITY);
        put(&mut inf, 2 * len / 3, f32::NEG_INFINITY);
        put(&mut nan, len / 4, f32::NAN);
        vec![base, hdr, inf, nan]
    }

    /// One pass of Blur's definition: the direct weighted sum in f64 of the window around each
    /// output, taps in order, the line transparent black beyond its ends.
    fn direct_pass(pass: BoxPass, line: &[f32]) -> Vec<f32> {
        let t = pass.tmax() as usize;
        let weights: Vec<f64> = pass.weights().map(f64::from).collect();
        (0..line.len() + 2 * t)
            .map(|o| {
                let mut acc = 0.0f64;
                for i in 0..=2 * t {
                    let weight = weights[i.abs_diff(t)];
                    let sample = (o + i).checked_sub(2 * t).and_then(|x| line.get(x));
                    acc += weight * f64::from(sample.copied().unwrap_or(0.0));
                }
                flush(acc as f32)
            })
            .collect()
    }

    fn reference(pass: BoxPass, line: &[f32]) -> Vec<f32> {
        let mut line = line.to_vec();
        for _ in 0..BoxPass::PASSES {
            line = direct_pass(pass, &line);
        }
        line
    }

    /// The definition's result over `line` and the bound beside it: the passes over the
    /// magnitudes.
    struct Expected {
        result: Vec<f32>,
        bound: Vec<f32>,
    }

    fn expected(pass: BoxPass, line: &[f32]) -> Expected {
        let magnitudes: Vec<f32> = line.iter().map(|v| v.abs()).collect();
        Expected {
            result: reference(pass, line),
            bound: reference(pass, &magnitudes),
        }
    }

    /// `actual` against the definition: the same class where the definition is not finite,
    /// else within K of 2⁻²⁴ times the bound, plus one `MIN_POSITIVE` for flushing.
    fn assert_within_bound(expected: &Expected, actual: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.result.len(), "{what}: length");
        for (o, ((&a, &e), &m)) in actual
            .iter()
            .zip(&expected.result)
            .zip(&expected.bound)
            .enumerate()
        {
            if !e.is_finite() {
                assert!(
                    e.is_nan() && a.is_nan() || a == e,
                    "{what}: output {o} is {a:?}, not {e:?}"
                );
                continue;
            }
            let allowed = K * f64::from(m) * 2f64.powi(-24) + f64::from(f32::MIN_POSITIVE);
            let error = (f64::from(a) - f64::from(e)).abs();
            assert!(
                error <= allowed,
                "{what}: output {o} is {a:e}, {} ULP from {e:e}, {error:e} over a bound of {allowed:e}",
                ulp_distance(a, e)
            );
        }
    }

    const SIZES: [f64; 12] = [
        1.6, 2.0, 3.0, 4.5, 6.0, 10.0, 19.5, 19.6, 22.5, 31.4, 100.0, 1000.0,
    ];

    /// Lengths to try at a size: every small one where the window is small, so that the block
    /// boundaries fall everywhere, and a few where the reference is slow.
    fn lengths(pass: BoxPass) -> Vec<usize> {
        if pass.tmax() <= 17 {
            (1..=33).chain([179]).collect()
        } else {
            vec![1, 2, 3, 33, 100, 179]
        }
    }

    /// The line rotated by `by`: what row `by` of a chunk and lane `by` of a band hold.
    fn rotated(line: &[f32], by: usize) -> Vec<f32> {
        line.iter()
            .cycle()
            .skip(by)
            .take(line.len())
            .copied()
            .collect()
    }

    /// A chunk of rows through Blur's three passes at `level`.
    fn blur_rows(level: Level, pass: BoxPass, src: &[f32], width: usize, dst: &mut [f32]) {
        let mut scratch = LineScratch::default();
        let op = BoxSum::new(pass);
        at_level(
            level,
            #[inline(always)]
            || line_rows(&op, BoxPass::PASSES, src, width, dst, &mut scratch),
        );
    }

    /// A band of columns, gathered row by row, through Blur's three passes at `level`.
    fn blur_band(level: Level, pass: BoxPass, src: &[f32], dst: &mut [f32]) {
        let mut scratch = LineScratch::default();
        let op = BoxSum::new(pass);
        at_level(
            level,
            #[inline(always)]
            || {
                line_passes::<BoxSum, LINE_BAND>(
                    &op,
                    BoxPass::PASSES,
                    src,
                    dst,
                    &mut scratch.passes,
                )
            },
        );
    }

    /// Every size's passes along rows of every length, in chunks of every row count, and along
    /// bands of every length, against the definition, or a skip when this machine lacks the
    /// level.
    fn passes_match_the_definition(level: Level) {
        let detected = Level::detect();
        if level > detected {
            println!("skipped {level:?}: this machine detects {detected:?}");
            return;
        }
        for size in SIZES {
            let pass = BoxPass::new(size);
            let t = pass.tmax() as usize;
            for len in lengths(pass) {
                for line in lines(len) {
                    let rotations: Vec<Vec<f32>> =
                        (0..LINE_ROWS).map(|by| rotated(&line, by)).collect();
                    let expectations: Vec<Expected> =
                        rotations.iter().map(|line| expected(pass, line)).collect();
                    for rows in [1, 2, LINE_ROWS - 1, LINE_ROWS] {
                        let src: Vec<f32> = rotations[..rows].concat();
                        let mut dst = vec![0.0f32; rows * (len + 6 * t)];
                        blur_rows(level, pass, &src, len, &mut dst);
                        for (row, dst) in dst.chunks_exact(len + 6 * t).enumerate() {
                            let what =
                                format!("{level:?} size {size}, {rows} rows of {len}, row {row}");
                            assert_within_bound(&expectations[row], dst, &what);
                        }
                    }
                    let mut src = vec![0.0f32; len * LINE_BAND];
                    for (y, row) in src.as_chunks_mut::<LINE_BAND>().0.iter_mut().enumerate() {
                        for (lane, sample) in row.iter_mut().enumerate() {
                            *sample = rotations[lane][y];
                        }
                    }
                    let mut dst = vec![0.0f32; (len + 6 * t) * LINE_BAND];
                    blur_band(level, pass, &src, &mut dst);
                    for lane in 0..LINE_BAND {
                        let actual: Vec<f32> = (0..len + 6 * t)
                            .map(|y| dst[y * LINE_BAND + lane])
                            .collect();
                        let what =
                            format!("{level:?} size {size}, band of {len} rows, lane {lane}");
                        assert_within_bound(&expectations[lane], &actual, &what);
                    }
                }
            }
        }
    }

    #[test]
    fn baseline_matches_the_definition() {
        passes_match_the_definition(Level(Kind::Baseline));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_matches_the_definition() {
        passes_match_the_definition(Level(Kind::V3));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_matches_the_definition() {
        passes_match_the_definition(Level(Kind::V4));
    }

    /// The result is one sequence of f64 operations, so every level this machine has gives the
    /// baseline's bits.
    #[test]
    fn levels_agree_bit_for_bit() {
        let detected = Level::detect();
        let baseline = Level(Kind::Baseline);
        if detected == baseline {
            println!("skipped: this machine detects only {baseline:?}");
            return;
        }
        for size in SIZES {
            let pass = BoxPass::new(size);
            let t = pass.tmax() as usize;
            let width = 179;
            let src: Vec<f32> = lines(width).concat();
            let rows = src.len() / width;
            let mut at_baseline = vec![0.0f32; rows * (width + 6 * t)];
            let mut at_detected = at_baseline.clone();
            blur_rows(baseline, pass, &src, width, &mut at_baseline);
            blur_rows(detected, pass, &src, width, &mut at_detected);
            assert!(
                bits(&at_baseline) == bits(&at_detected),
                "rows at size {size}"
            );
            let band: Vec<f32> = lines(16 * 61).concat()[..61 * LINE_BAND].to_vec();
            let mut at_baseline = vec![0.0f32; (61 + 6 * t) * LINE_BAND];
            let mut at_detected = at_baseline.clone();
            blur_band(baseline, pass, &band, &mut at_baseline);
            blur_band(detected, pass, &band, &mut at_detected);
            assert!(
                bits(&at_baseline) == bits(&at_detected),
                "band at size {size}"
            );
        }
    }

    fn bits(plane: &[f32]) -> Vec<u32> {
        plane.iter().map(|v| v.to_bits()).collect()
    }

    /// Both axes through the pool against the passes over each line alone, the columns of a
    /// width no band divides, and a stop on the first ask leaving the output unwritten.
    #[test]
    fn the_line_reduction_runs_each_line_through_the_pool() {
        let pool = Pool::new();
        let pass = BoxPass::new(10.0);
        let op = BoxSum::new(pass);
        let growth = 6 * pass.tmax() as usize;
        let (width, height) = (37, 21);
        let input = values(7, width * height);
        let never = || false;

        let mut rows = vec![f32::NAN; (width + growth) * height];
        line(
            &pool,
            &op,
            Axis::Rows,
            3,
            &input,
            (width, height),
            &mut rows,
            &never,
        )
        .unwrap();
        for (y, row) in rows.chunks_exact(width + growth).enumerate() {
            let expected = expected(pass, &input[y * width..(y + 1) * width]);
            assert_within_bound(&expected, row, &format!("row {y}"));
        }

        let mut columns = vec![f32::NAN; width * (height + growth)];
        line(
            &pool,
            &op,
            Axis::Columns,
            3,
            &input,
            (width, height),
            &mut columns,
            &never,
        )
        .unwrap();
        for x in 0..width {
            let column: Vec<f32> = (0..height).map(|y| input[y * width + x]).collect();
            let actual: Vec<f32> = (0..height + growth)
                .map(|y| columns[y * width + x])
                .collect();
            assert_within_bound(&expected(pass, &column), &actual, &format!("column {x}"));
        }

        let mut stopped = vec![f32::NAN; (width + growth) * height];
        let always = || true;
        let result = line(
            &pool,
            &op,
            Axis::Rows,
            3,
            &input,
            (width, height),
            &mut stopped,
            &always,
        );
        assert_eq!(result, Err(Stopped));
        assert!(stopped.iter().all(|v| v.is_nan()), "nothing written");
    }

    #[test]
    #[should_panic(expected = "grown")]
    fn an_output_of_the_wrong_length_is_refused() {
        let pass = BoxPass::new(3.0);
        line(
            &Pool::new(),
            &BoxSum::new(pass),
            Axis::Rows,
            3,
            &[1.0; 8],
            (8, 1),
            &mut [0.0; 8],
            &|| false,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "at most LINE_ROWS")]
    fn too_many_rows_are_refused() {
        let pass = BoxPass::new(3.0);
        let src = [1.0; 2 * (LINE_ROWS + 1)];
        let mut dst = [0.0; (LINE_ROWS + 1) * 8];
        blur_rows(Level::detect(), pass, &src, 2, &mut dst);
    }
}
