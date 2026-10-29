//! The primitives' CPU halves: the map and the fold here, the line reduction in [`line`], each
//! generic over the operation, run over the evaluator's pool and compiled once per x86-64
//! feature level so the evaluator can use the vector width the machine has without being built
//! for it.
//!
//! [`at_level`]'s wrappers are the only place `target_feature` appears. Each enables exactly one
//! level's features over the same body, so the body vectorises three ways while the per-sample
//! arithmetic stays the kernel crate's, bit for bit at every level.

use std::ops::Range;

use opencompositor_kernels::{Fold, Map};

use crate::evaluate::Pool;
use crate::image::{Overlap, Window};

mod line;

#[cfg(test)]
pub(crate) use line::LINE_ROWS;
pub(crate) use line::line_scratch_bytes;
pub use line::{Stopped, line};

/// The x86-64 feature level the kernels run at. Only [`Level::detect`] makes one, after checking
/// every feature its wrapper enables, so holding a level proves this machine supports it. Other
/// architectures have only the baseline, the plain build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Level(Kind);

// Private so that no caller can name a level the machine lacks: `at_level`'s safety rests on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// The build's own target: SSE2 on x86-64, four lanes of f32.
    Baseline,
    /// x86-64-v3, AVX2 and FMA: eight lanes.
    #[cfg(target_arch = "x86_64")]
    V3,
    /// x86-64-v4, AVX-512: sixteen lanes.
    #[cfg(target_arch = "x86_64")]
    V4,
}

impl Level {
    /// The widest level this machine supports, checked feature by feature against the lists
    /// the wrappers enable.
    pub(crate) fn detect() -> Level {
        #[cfg(target_arch = "x86_64")]
        {
            let v3 = is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("bmi1")
                && is_x86_feature_detected!("bmi2")
                && is_x86_feature_detected!("lzcnt")
                && is_x86_feature_detected!("movbe")
                && is_x86_feature_detected!("f16c");
            let v4 = v3
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512cd")
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vl");
            if v4 {
                return Level(Kind::V4);
            }
            if v3 {
                return Level(Kind::V3);
            }
        }
        Level(Kind::Baseline)
    }
}

/// Runs `body` in the wrapper that enables `level`'s features, so that the code inlined into it
/// vectorises for the level. The feature lists are x86-64-v3 and v4 as the psABI defines them,
/// and `Level::detect` checks the same lists.
///
/// A closure is not inlined into the wrapper by default: callers mark theirs
/// `#[inline(always)]`, or the wrapper only calls a baseline copy, which still passes every
/// numerical test and silently runs without the level's vectors. The kernels benchmark's fold
/// and Blur gates fail without them; Exposure is memory-bound either way, so its gate cannot
/// tell, and a costlier map would need a gate of its own.
#[inline(always)]
fn at_level<R>(level: Level, body: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,bmi1,bmi2,lzcnt,movbe,f16c")]
    fn v3<R>(body: impl FnOnce() -> R) -> R {
        body()
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,fma,bmi1,bmi2,lzcnt,movbe,f16c"
    )]
    fn v4<R>(body: impl FnOnce() -> R) -> R {
        body()
    }
    match level.0 {
        Kind::Baseline => body(),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `level` came from `Level::detect` on this machine, which found every feature
        // the wrapper enables.
        Kind::V3 => unsafe { v3(body) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: as above.
        Kind::V4 => unsafe { v4(body) },
    }
}

/// `op` over `inputs`, each a plane over its window, into `outputs`, planes over `output`: each
/// row of the output in spans over which every input either covers it or does not, each span
/// one loop over its samples. An input is zero where its window does not reach.
pub fn map<M: Map<I, O>, const I: usize, const O: usize>(
    pool: &Pool,
    op: &M,
    inputs: [(&[f32], Window); I],
    output: Window,
    outputs: [&mut [f32]; O],
) {
    let width = output.width as usize;
    for (plane, window) in inputs {
        assert_eq!(
            plane.len() as u64,
            window.area(),
            "an input over its window"
        );
    }
    assert_eq!(
        outputs[0].len() as u64,
        output.area(),
        "outputs over the output"
    );
    let overlaps = inputs.map(|(_, window)| Overlap::of(window, output));
    let zeros = vec![0.0f32; width];
    pool.fill_rows(outputs, width, |level, first_row, mut chunk| {
        let mut cuts = Vec::with_capacity(2 * I + 2);
        let rows = chunk[0].len() / width;
        for (index, y) in (first_row..first_row + rows).enumerate() {
            let spans = overlaps.each_ref().map(|overlap| overlap.row(y));
            cut(
                spans.iter().flatten().map(|(_, columns)| columns),
                width,
                &mut cuts,
            );
            for pair in cuts.windows(2) {
                let columns = pair[0]..pair[1];
                let src: [&[f32]; I] = std::array::from_fn(|k| {
                    samples(inputs[k].0, &spans[k], columns.clone(), &zeros)
                });
                let dst =
                    (chunk.each_mut()).map(|plane| &mut plane[index * width..][columns.clone()]);
                map_span(level, op, src, dst);
            }
        }
    });
}

/// `op` over a stack of `layers` on `base`, each planes over a window, into `outputs`, planes
/// over `output`: each row of the output in spans over which the base and each layer either
/// cover it or do not, each span folded a covering layer at a time into the outputs, then
/// finished against the base. The base is zero where its window does not reach.
pub fn fold<F: Fold<G, O>, const G: usize, const O: usize>(
    pool: &Pool,
    op: &F,
    base: ([&[f32]; O], Window),
    layers: &[([&[f32]; G], Window)],
    output: Window,
    outputs: [&mut [f32]; O],
) {
    let width = output.width as usize;
    for (planes, window) in std::iter::once((&base.0[..], base.1))
        .chain(layers.iter().map(|(planes, window)| (&planes[..], *window)))
    {
        for plane in planes {
            assert_eq!(
                plane.len() as u64,
                window.area(),
                "an input over its window"
            );
        }
    }
    assert_eq!(
        outputs[0].len() as u64,
        output.area(),
        "outputs over the output"
    );
    let base_overlap = Overlap::of(base.1, output);
    let overlaps: Vec<Overlap> = (layers.iter())
        .map(|(_, window)| Overlap::of(*window, output))
        .collect();
    let zeros = vec![0.0f32; width];
    pool.fill_rows(outputs, width, |level, first_row, mut chunk| {
        let mut cuts = Vec::with_capacity(2 * layers.len() + 4);
        let mut spans = Vec::with_capacity(layers.len());
        let mut sources: Vec<[&[f32]; G]> = Vec::with_capacity(layers.len());
        let rows = chunk[0].len() / width;
        for (index, y) in (first_row..first_row + rows).enumerate() {
            let base_span = base_overlap.row(y);
            spans.clear();
            spans.extend(overlaps.iter().map(|overlap| overlap.row(y)));
            let all = base_span.iter().chain(spans.iter().flatten());
            cut(all.map(|(_, columns)| columns), width, &mut cuts);
            for pair in cuts.windows(2) {
                let columns = pair[0]..pair[1];
                let below =
                    (base.0).map(|plane| samples(plane, &base_span, columns.clone(), &zeros));
                sources.clear();
                for ((planes, _), span) in layers.iter().zip(&spans) {
                    if covers(span, columns.start) {
                        sources.push(
                            planes.map(|plane| samples(plane, span, columns.clone(), &zeros)),
                        );
                    }
                }
                let dst =
                    (chunk.each_mut()).map(|plane| &mut plane[index * width..][columns.clone()]);
                at_level(
                    level,
                    #[inline(always)]
                    || fold_samples(op, below, &sources, dst),
                );
            }
        }
    });
}

/// A plane's span of an output row, as [`Overlap::row`] gives it: the plane's samples and the
/// output's columns they lie in.
type Span = Option<(Range<usize>, Range<usize>)>;

/// Whether `span` covers the output's column `x`.
fn covers(span: &Span, x: usize) -> bool {
    span.as_ref()
        .is_some_and(|(_, columns)| columns.contains(&x))
}

/// A plane's samples for the output's `columns` of a row, by its span in the row, which covers
/// all of them or none: zeros where it covers none.
fn samples<'a>(
    plane: &'a [f32],
    span: &Span,
    columns: Range<usize>,
    zeros: &'a [f32],
) -> &'a [f32] {
    match span {
        Some((input, covered)) if covered.contains(&columns.start) => {
            &plane[input.start + (columns.start - covered.start)..][..columns.len()]
        }
        _ => &zeros[..columns.len()],
    }
}

/// Into `cuts`, the distinct column boundaries of a row of `width` that `spans` make, from 0 to
/// `width`: between consecutive cuts, each span covers every column or none.
fn cut<'a>(spans: impl Iterator<Item = &'a Range<usize>>, width: usize, cuts: &mut Vec<usize>) {
    cuts.clear();
    cuts.push(0);
    for span in spans {
        cuts.extend([span.start, span.end]);
    }
    cuts.push(width);
    cuts.sort_unstable();
    cuts.dedup();
}

/// One span of a map at `level` on the caller's thread: the map's kernel without the pool,
/// which the kernels benchmark times alone.
#[inline(always)]
pub fn map_span<M: Map<I, O>, const I: usize, const O: usize>(
    level: Level,
    op: &M,
    src: [&[f32]; I],
    dst: [&mut [f32]; O],
) {
    at_level(
        level,
        #[inline(always)]
        || map_samples(op, src, dst),
    );
}

/// One span of a map: each input's samples, zero where it does not cover the span.
#[inline(always)]
fn map_samples<M: Map<I, O>, const I: usize, const O: usize>(
    op: &M,
    src: [&[f32]; I],
    dst: [&mut [f32]; O],
) {
    let len = dst[0].len();
    let src = src.map(|src| &src[..len]);
    let mut dst = dst.map(|dst| &mut dst[..len]);
    for i in 0..len {
        let samples = std::array::from_fn(|k| src[k][i]);
        let results = op.apply(samples);
        for (dst, result) in dst.iter_mut().zip(results) {
            dst[i] = result;
        }
    }
}

/// One span of a fold: the base's samples, then each layer's that covers the span, then the
/// finish.
#[inline(always)]
fn fold_samples<F: Fold<G, O>, const G: usize, const O: usize>(
    op: &F,
    base: [&[f32]; O],
    layers: &[[&[f32]; G]],
    dst: [&mut [f32]; O],
) {
    let len = dst[0].len();
    let base = base.map(|base| &base[..len]);
    let mut dst = dst.map(|dst| &mut dst[..len]);
    for (dst, base) in dst.iter_mut().zip(base) {
        dst.copy_from_slice(base);
    }
    for planes in layers {
        let planes = planes.map(|plane| &plane[..len]);
        for i in 0..len {
            let below = std::array::from_fn(|k| dst[k][i]);
            let stack = op.step(below, std::array::from_fn(|k| planes[k][i]));
            for (dst, sample) in dst.iter_mut().zip(stack) {
                dst[i] = sample;
            }
        }
    }
    for i in 0..len {
        let stack = std::array::from_fn(|k| dst[k][i]);
        let result = op.finish(std::array::from_fn(|k| base[k][i]), stack);
        for (dst, sample) in dst.iter_mut().zip(result) {
            dst[i] = sample;
        }
    }
}

#[cfg(test)]
mod tests {
    use opencompositor_kernels::{Exposure, Merge};

    use super::*;

    /// Gains across the stop range, including one that rounds.
    const GAINS: [f32; 6] = [1.0, 0.5, 1.259_921_1, 2.0, 1.8446744e19, 5.421011e-20];

    /// `len` samples: every class a plane can hold or a kernel must pass through, then
    /// pseudo-random bit patterns, so that the remainder loops at every vector width see each.
    fn samples(len: usize) -> Vec<f32> {
        const CLASSES: [f32; 13] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.18,
            f32::MIN_POSITIVE,
            -1e-39,
            3.0e38,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        (0..len)
            .map(|i| {
                CLASSES.get(i).copied().unwrap_or_else(|| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    f32::from_bits((state >> 32) as u32)
                })
            })
            .collect()
    }

    /// Whether this machine has `level`, printing the skip when it has not.
    fn supported(level: Level) -> bool {
        let detected = Level::detect();
        if level > detected {
            println!("skipped {level:?}: this machine detects {detected:?}");
            return false;
        }
        println!("cpu kernel tests at {level:?}");
        true
    }

    /// Exposure through the map, one span of every length, against the kernel crate's
    /// function bit for bit.
    fn exposure_matches_the_kernel_crate(level: Level) {
        // A signalling NaN, which no multiply returns: a sample left unwritten fails below.
        let unwritten = f32::from_bits(0x7f80_0001);
        for len in (0..=70).chain([255, 1000, 4097]) {
            let src = samples(len);
            for gain in GAINS {
                let mut dst = vec![unwritten; len];
                let op = Exposure { gain };
                map_span(level, &op, [&src], [&mut dst]);
                for (i, (&s, &d)) in src.iter().zip(&dst).enumerate() {
                    let expected = opencompositor_kernels::exposure(s, gain);
                    assert_eq!(
                        d.to_bits(),
                        expected.to_bits(),
                        "{level:?}, {len} samples: sample {i} ({s:e}) by {gain:e} gave {d:e}, \
                         not {expected:e}"
                    );
                }
            }
        }
    }

    /// Merge through the fold, one span of four planes of every length with one layer and with
    /// two, against over and then mix from the kernel crate bit for bit, and Mix 1 as the
    /// stack itself.
    fn merge_matches_the_kernel_crate(level: Level) {
        for len in (0..=70).chain([255, 1000, 4097]) {
            let planes = |seeds: [usize; 4]| seeds.map(|seed| samples(len + seed)[seed..].to_vec());
            let base = planes([5, 6, 7, 8]);
            let first = planes([1, 2, 3, 4]);
            let second = planes([9, 10, 11, 12]);
            let base_planes = base.each_ref().map(Vec::as_slice);
            let first_planes = first.each_ref().map(Vec::as_slice);
            let second_planes = second.each_ref().map(Vec::as_slice);
            for mix in [0.3, 1.0] {
                for layers in [&[first_planes][..], &[first_planes, second_planes]] {
                    let mut dst: [Vec<f32>; 4] = [(); 4].map(|()| vec![0.0; len]);
                    let op = Merge { mix };
                    at_level(
                        level,
                        #[inline(always)]
                        || {
                            fold_samples(
                                &op,
                                base_planes,
                                layers,
                                dst.each_mut().map(Vec::as_mut_slice),
                            )
                        },
                    );
                    for i in 0..len {
                        let mut stack = base_planes.map(|plane| plane[i]);
                        for layer in layers {
                            let alpha = layer[3][i];
                            for (channel, sample) in stack.iter_mut().enumerate() {
                                *sample =
                                    opencompositor_kernels::over(*sample, layer[channel][i], alpha);
                            }
                        }
                        for channel in 0..4 {
                            let expected = if mix == 1.0 {
                                stack[channel]
                            } else {
                                opencompositor_kernels::mix(
                                    base_planes[channel][i],
                                    stack[channel],
                                    mix,
                                )
                            };
                            assert_eq!(
                                dst[channel][i].to_bits(),
                                expected.to_bits(),
                                "{level:?}, {len} samples, {} layers at Mix {mix}: sample {i} of plane {channel}",
                                layers.len()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn baseline_matches_the_kernel_crate() {
        let level = Level(Kind::Baseline);
        exposure_matches_the_kernel_crate(level);
        merge_matches_the_kernel_crate(level);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_matches_the_kernel_crate() {
        let level = Level(Kind::V3);
        if supported(level) {
            exposure_matches_the_kernel_crate(level);
            merge_matches_the_kernel_crate(level);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_matches_the_kernel_crate() {
        let level = Level(Kind::V4);
        if supported(level) {
            exposure_matches_the_kernel_crate(level);
            merge_matches_the_kernel_crate(level);
        }
    }

    fn window(x: i32, y: i32, width: u32, height: u32) -> Window {
        Window {
            x,
            y,
            width,
            height,
        }
    }

    /// The map over the pool with an input offset in the output, reaching past it on two
    /// sides: each output sample is Exposure of the input's sample at the same position, and
    /// zero where the input is not.
    #[test]
    fn the_map_places_its_inputs() {
        let pool = Pool::new();
        let input = window(3, -2, 9, 6);
        let output = window(0, 0, 10, 7);
        let src: Vec<f32> = (0..input.area()).map(|i| i as f32 + 1.0).collect();
        let mut dst = vec![f32::NAN; output.area() as usize];
        map(
            &pool,
            &Exposure { gain: 2.0 },
            [(&src, input)],
            output,
            [&mut dst],
        );
        for y in 0..7 {
            for x in 0..10 {
                let expected = if input.contains(x, y) {
                    let at = (y - input.y) as u32 * input.width + (x - input.x) as u32;
                    2.0 * src[at as usize]
                } else {
                    0.0
                };
                assert_eq!(dst[(y * 10 + x) as usize], expected, "({x}, {y})");
            }
        }
    }

    /// The fold over the pool with two layers in different parts of the output: each texel is
    /// the base, overed by each layer whose window holds it, in order, and the base is zero
    /// outside its own window.
    #[test]
    fn the_fold_places_its_layers() {
        let pool = Pool::new();
        let base_window = window(0, 0, 6, 4);
        let first_window = window(4, 2, 5, 5);
        let second_window = window(-2, 1, 4, 3);
        let output = window(-2, 0, 11, 7);
        let plane = |window: Window, value: f32| vec![value; window.area() as usize];
        let base = [0.1, 0.2, 0.3, 0.4].map(|v| plane(base_window, v));
        let first = [0.5, 0.25, 0.125, 0.5].map(|v| plane(first_window, v));
        let second = [0.0625, 0.75, 1.0, 1.0].map(|v| plane(second_window, v));
        let mut dst: [Vec<f32>; 4] = [(); 4].map(|()| vec![f32::NAN; output.area() as usize]);
        fold(
            &pool,
            &Merge { mix: 1.0 },
            (base.each_ref().map(Vec::as_slice), base_window),
            &[
                (first.each_ref().map(Vec::as_slice), first_window),
                (second.each_ref().map(Vec::as_slice), second_window),
            ],
            output,
            dst.each_mut().map(Vec::as_mut_slice),
        );
        for y in 0..7 {
            for x in -2..9 {
                let at = |window: Window, planes: &[Vec<f32>; 4]| {
                    window
                        .contains(x, y)
                        .then(|| planes.each_ref().map(|plane| plane[0]))
                };
                let mut expected = at(base_window, &base).unwrap_or([0.0; 4]);
                for layer in [at(first_window, &first), at(second_window, &second)]
                    .into_iter()
                    .flatten()
                {
                    for channel in 0..4 {
                        expected[channel] = opencompositor_kernels::over(
                            expected[channel],
                            layer[channel],
                            layer[3],
                        );
                    }
                }
                let index = (y * 11 + x + 2) as usize;
                for channel in 0..4 {
                    assert_eq!(
                        dst[channel][index], expected[channel],
                        "({x}, {y}) plane {channel}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "an input over its window")]
    fn an_input_not_over_its_window_is_refused() {
        let pool = Pool::new();
        let output = window(0, 0, 4, 2);
        map(
            &pool,
            &Exposure { gain: 1.0 },
            [(&[1.0; 7], output)],
            output,
            [&mut [0.0; 8]],
        );
    }

    #[test]
    #[should_panic(expected = "one length")]
    fn outputs_of_different_lengths_are_refused() {
        let pool = Pool::new();
        let output = window(0, 0, 4, 2);
        let planes = [[1.0; 8]; 4];
        let mut dst = [[0.0; 8]; 3];
        let [r, g, b] = &mut dst;
        fold(
            &pool,
            &Merge { mix: 1.0 },
            (planes.each_ref().map(|plane| &plane[..]), output),
            &[(planes.each_ref().map(|plane| &plane[..]), output)],
            output,
            [r, g, b, &mut [0.0; 7]],
        );
    }
}
