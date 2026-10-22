//! Blur's box passes on the CPU: each pass a window sum over a line, lines of up to sixteen
//! lanes so that the sums, serial along a line, run as one vector across lanes.

use opencompositor_kernels::BoxPass;

use super::{Kind, Level, kernels};

at_levels! {
    /// Blur's three box passes along rows: up to [`BLUR_ROWS`] rows of `width` samples in
    /// `src`, contiguous, into the same rows of `width + 2 reach` samples in `dst`, each row
    /// the three passes of `pass` over itself with transparent black beyond its ends. `scratch`
    /// is this thread's, reused call to call.
    blur_rows => blur_rows_samples(
        pass: BoxPass,
        src: &[f32],
        width: usize,
        dst: &mut [f32],
        scratch: &mut BoxScratch,
    )
}

at_levels! {
    /// Blur's three box passes along columns: a band of [`BLUR_BAND`] columns gathered row by
    /// row in `src`, into the band of `rows + 2 reach` rows in `dst`, each column the three
    /// passes of `pass` over itself with transparent black beyond its ends.
    blur_band => blur_band_samples(
        pass: BoxPass,
        src: &[f32],
        dst: &mut [f32],
        scratch: &mut BoxScratch,
    )
}

/// Rows per call of [`blur_rows`], the lanes its window sums run over: the sums are serial
/// along a row, so the rows of a chunk are interleaved and summed as one vector.
pub const BLUR_ROWS: usize = 16;

/// Columns per call of [`blur_band`]: one cache line of samples, the lanes of its sums.
pub const BLUR_BAND: usize = 16;

/// Windows of up to this many taps each side are summed directly, every output its own sum
/// over the taps, which vectorises across the outputs of a row; wider ones go through block
/// sums. Measured on the workstation: direct sums cost a tap each, block sums about seven.
const DIRECT_TAPS: usize = 6;

/// A thread's temporaries, kept from call to call: the passes' own, and the rows interleaved
/// as lanes with their results. Empty until the first call sizes it.
#[derive(Debug, Default)]
pub struct BoxScratch {
    passes: PassScratch,
    lanes: Vec<f32>,
    results: Vec<f32>,
}

/// The passes' temporaries: the three zero-padded lines they read, and one block's suffix sums.
#[derive(Debug, Default)]
struct PassScratch {
    pads: [Vec<f32>; 3],
    suffixes: Vec<f64>,
}

#[inline(always)]
fn blur_rows_samples(
    pass: BoxPass,
    src: &[f32],
    width: usize,
    dst: &mut [f32],
    scratch: &mut BoxScratch,
) {
    let t = pass.tmax() as usize;
    let rows = src.len().checked_div(width).unwrap_or(0);
    assert!(rows <= BLUR_ROWS, "more rows than blur_rows takes");
    assert_eq!(src.len(), rows * width, "whole rows");
    let grown = width + 6 * t;
    assert_eq!(
        dst.len(),
        rows * grown,
        "blur_rows' output rows are the input's grown"
    );
    if t <= DIRECT_TAPS {
        for (src, dst) in src.chunks_exact(width).zip(dst.chunks_exact_mut(grown)) {
            box_passes::<1>(pass, src, dst, &mut scratch.passes);
        }
        return;
    }
    // The chunk's rows as lanes, a column of the chunk per element; rows past the chunk's
    // last are lanes of zeros and their results are dropped.
    scratch.lanes.clear();
    scratch.lanes.resize(width * BLUR_ROWS, 0.0);
    let (lanes, _) = scratch.lanes.as_chunks_mut::<BLUR_ROWS>();
    for (row, src) in src.chunks_exact(width).enumerate() {
        for (lane, &sample) in lanes.iter_mut().zip(src) {
            lane[row] = sample;
        }
    }
    scratch.results.clear();
    scratch.results.resize(grown * BLUR_ROWS, 0.0);
    box_passes::<BLUR_ROWS>(
        pass,
        &scratch.lanes,
        &mut scratch.results,
        &mut scratch.passes,
    );
    let (results, _) = scratch.results.as_chunks::<BLUR_ROWS>();
    for (row, dst) in dst.chunks_exact_mut(grown).enumerate() {
        for (d, lane) in dst.iter_mut().zip(results) {
            *d = lane[row];
        }
    }
}

#[inline(always)]
fn blur_band_samples(pass: BoxPass, src: &[f32], dst: &mut [f32], scratch: &mut BoxScratch) {
    let t = pass.tmax() as usize;
    assert_eq!(src.len() % BLUR_BAND, 0, "whole rows of the band");
    assert_eq!(
        dst.len(),
        src.len() + 6 * t * BLUR_BAND,
        "blur_band's output rows are the input's grown"
    );
    box_passes::<BLUR_BAND>(pass, src, dst, &mut scratch.passes);
}

/// The three passes over one line of `n` elements of `L` lanes in `src`, into the `n + 6 tmax`
/// elements of `dst`: each pass from a zero-padded copy of its input into the middle of the
/// next pass's, the last into `dst`.
#[inline(always)]
fn box_passes<const L: usize>(
    pass: BoxPass,
    src: &[f32],
    dst: &mut [f32],
    scratch: &mut PassScratch,
) {
    let t = pass.tmax() as usize;
    assert!(t >= 1, "Blur's passes are for sizes above 1.5");
    let n = src.len() / L;
    let [a, b, c] = &mut scratch.pads;
    let a = padded::<L>(a, n, t);
    a[2 * t * L..(2 * t + n) * L].copy_from_slice(src);
    let b = padded::<L>(b, n + 2 * t, t);
    window_sums::<L>(
        pass,
        a,
        &mut b[2 * t * L..(4 * t + n) * L],
        &mut scratch.suffixes,
    );
    let c = padded::<L>(c, n + 4 * t, t);
    window_sums::<L>(
        pass,
        b,
        &mut c[2 * t * L..(6 * t + n) * L],
        &mut scratch.suffixes,
    );
    window_sums::<L>(pass, c, dst, &mut scratch.suffixes);
}

/// `pad` as a line of `n` elements with `2 t` zero elements each side, the middle left for the
/// caller to fill.
#[inline(always)]
fn padded<const L: usize>(pad: &mut Vec<f32>, n: usize, t: usize) -> &mut [f32] {
    pad.resize((n + 4 * t) * L, 0.0);
    pad[..2 * t * L].fill(0.0);
    pad[(2 * t + n) * L..].fill(0.0);
    pad
}

/// One pass: output `o` is the weighted sum of the window `z[o ..= o + 2 tmax]`, the inner
/// weight on every tap but the two ends. The arithmetic is f64 adds and products of an f32
/// sample by an f32 weight, which are exact; one order at every level, so the result is too.
#[inline(always)]
fn window_sums<const L: usize>(pass: BoxPass, z: &[f32], out: &mut [f32], suffixes: &mut Vec<f64>) {
    match pass.tmax() {
        1 => direct_sums::<L, 1>(pass, z, out),
        2 => direct_sums::<L, 2>(pass, z, out),
        3 => direct_sums::<L, 3>(pass, z, out),
        4 => direct_sums::<L, 4>(pass, z, out),
        5 => direct_sums::<L, 5>(pass, z, out),
        6 => direct_sums::<L, 6>(pass, z, out),
        _ => block_sums::<L>(pass, z, out, suffixes),
    }
}

// DIRECT_TAPS is the last tap count summed directly.
const _: () = assert!(DIRECT_TAPS == 6);

/// Each output its own sum over the `2 T + 1` taps, the tap count fixed so the sum unrolls.
#[inline(always)]
fn direct_sums<const L: usize, const T: usize>(pass: BoxPass, z: &[f32], out: &mut [f32]) {
    let inner = f64::from(pass.inner_weight());
    let edge = f64::from(pass.edge_weight());
    let (z, _) = z.as_chunks::<L>();
    let (out, _) = out.as_chunks_mut::<L>();
    for (o, out) in out.iter_mut().enumerate() {
        let window: &[[f32; L]] = &z[o..o + 2 * T + 1];
        let mut acc = [0.0f64; L];
        for lane in 0..L {
            acc[lane] = edge * f64::from(window[0][lane]);
        }
        for tap in &window[1..2 * T] {
            for lane in 0..L {
                acc[lane] += inner * f64::from(tap[lane]);
            }
        }
        for lane in 0..L {
            out[lane] = store(acc[lane] + edge * f64::from(window[2 * T][lane]));
        }
    }
}

/// The window's core, every tap but the ends, as block sums: the line in blocks of the core's
/// length, the core starting at offset `k` of a block is that block's suffix from `k` plus the
/// next block's prefix up to `k - 1`. `suffixes` holds one block's suffix sums; the prefix
/// accumulates as the outputs go. Each sum adds exactly the window's samples, so an infinity
/// or NaN reaches exactly the outputs whose window holds it, as a direct sum's would.
#[inline(always)]
fn block_sums<const L: usize>(pass: BoxPass, z: &[f32], out: &mut [f32], suffixes: &mut Vec<f64>) {
    let t = pass.tmax() as usize;
    let l = 2 * t - 1;
    let inner = f64::from(pass.inner_weight());
    let edge = f64::from(pass.edge_weight());
    let (z, _) = z.as_chunks::<L>();
    let (out, _) = out.as_chunks_mut::<L>();
    let n_z = z.len();
    let n_out = out.len();
    suffixes.clear();
    suffixes.resize(l * L, 0.0);
    let (suffixes, _) = suffixes.as_chunks_mut::<L>();
    let sum = |o: usize, core: [f64; L], out: &mut [f32; L]| {
        let (left, right) = (z[o], z[o + 2 * t]);
        for lane in 0..L {
            let ends = f64::from(left[lane]) + f64::from(right[lane]);
            out[lane] = store(edge * ends + inner * core[lane]);
        }
    };
    // Output `o` has the core `z[o + 1 ..= o + l]`: the cores start from 1 to `n_out`.
    let mut start = 0;
    while start <= n_out {
        let end = (start + l).min(n_z);
        let mut acc = [0.0f64; L];
        for (suffix, sample) in suffixes[..end - start].iter_mut().zip(&z[start..end]).rev() {
            for lane in 0..L {
                acc[lane] += f64::from(sample[lane]);
            }
            *suffix = acc;
        }
        // The core starting at the block's start is the whole block; none starts at 0.
        if start > 0 {
            sum(start - 1, suffixes[0], &mut out[start - 1]);
        }
        let mut prefix = [0.0f64; L];
        for k in 1..l.min(n_out - start + 1) {
            let o = start + k - 1;
            let entering = z[start + k + l - 1];
            let mut core = suffixes[k];
            for lane in 0..L {
                prefix[lane] += f64::from(entering[lane]);
                core[lane] += prefix[lane];
            }
            sum(o, core, &mut out[o]);
        }
        start += l;
    }
}

/// A pass's result as the next reads it or the plane holds it: f32, subnormals flushed.
#[inline(always)]
fn store(v: f64) -> f32 {
    kernels::flush(v as f32)
}

#[cfg(test)]
mod tests {
    use opencompositor_kernels::ulp_distance;

    use super::*;

    /// The CPU's K, the roundings of the three passes, each half an f32 unit of its result's
    /// magnitude; the node's six take 8.
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

    /// One pass of the definition: the direct weighted sum in f64 of the window around each
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
                kernels::flush(acc as f32)
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

    /// Lengths to try at a size: every small one where the window is small, so that the
    /// block boundaries fall everywhere, and a few where the reference is slow.
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

    /// Every size's passes along rows of every length, in chunks of every row count, and
    /// along bands of every length, against the definition, or a skip when this machine lacks
    /// the level.
    fn passes_match_the_definition(level: Level) {
        let detected = Level::detect();
        if level > detected {
            println!("skipped {level:?}: this machine detects {detected:?}");
            return;
        }
        let mut scratch = BoxScratch::default();
        for size in SIZES {
            let pass = BoxPass::new(size);
            let t = pass.tmax() as usize;
            for len in lengths(pass) {
                for line in lines(len) {
                    let rotations: Vec<Vec<f32>> =
                        (0..BLUR_ROWS).map(|by| rotated(&line, by)).collect();
                    let expectations: Vec<Expected> =
                        rotations.iter().map(|line| expected(pass, line)).collect();
                    for rows in [1, 2, BLUR_ROWS - 1, BLUR_ROWS] {
                        let src: Vec<f32> = rotations[..rows].concat();
                        let mut dst = vec![0.0f32; rows * (len + 6 * t)];
                        blur_rows(level, pass, &src, len, &mut dst, &mut scratch);
                        for (row, dst) in dst.chunks_exact(len + 6 * t).enumerate() {
                            let what =
                                format!("{level:?} size {size}, {rows} rows of {len}, row {row}");
                            assert_within_bound(&expectations[row], dst, &what);
                        }
                    }
                    let mut src = vec![0.0f32; len * BLUR_BAND];
                    for (y, row) in src.as_chunks_mut::<BLUR_BAND>().0.iter_mut().enumerate() {
                        for (lane, sample) in row.iter_mut().enumerate() {
                            *sample = rotations[lane][y];
                        }
                    }
                    let mut dst = vec![0.0f32; (len + 6 * t) * BLUR_BAND];
                    blur_band(level, pass, &src, &mut dst, &mut scratch);
                    for lane in 0..BLUR_BAND {
                        let actual: Vec<f32> = (0..len + 6 * t)
                            .map(|y| dst[y * BLUR_BAND + lane])
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

    /// The result is one sequence of f64 operations, so every level this machine has gives
    /// the baseline's bits.
    #[test]
    fn levels_agree_bit_for_bit() {
        let detected = Level::detect();
        let baseline = Level(Kind::Baseline);
        if detected == baseline {
            println!("skipped: this machine detects only {baseline:?}");
            return;
        }
        let mut scratch = BoxScratch::default();
        for size in SIZES {
            let pass = BoxPass::new(size);
            let t = pass.tmax() as usize;
            let width = 179;
            let src: Vec<f32> = lines(width).concat();
            let rows = src.len() / width;
            let mut at_baseline = vec![0.0f32; rows * (width + 6 * t)];
            let mut at_detected = at_baseline.clone();
            blur_rows(baseline, pass, &src, width, &mut at_baseline, &mut scratch);
            blur_rows(detected, pass, &src, width, &mut at_detected, &mut scratch);
            assert!(
                bits(&at_baseline) == bits(&at_detected),
                "rows at size {size}"
            );
            let band: Vec<f32> = lines(16 * 61).concat()[..61 * BLUR_BAND].to_vec();
            let mut at_baseline = vec![0.0f32; (61 + 6 * t) * BLUR_BAND];
            let mut at_detected = at_baseline.clone();
            blur_band(baseline, pass, &band, &mut at_baseline, &mut scratch);
            blur_band(detected, pass, &band, &mut at_detected, &mut scratch);
            assert!(
                bits(&at_baseline) == bits(&at_detected),
                "band at size {size}"
            );
        }
    }

    fn bits(plane: &[f32]) -> Vec<u32> {
        plane.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    #[should_panic(expected = "grown")]
    fn rows_of_the_wrong_length_are_refused() {
        let pass = BoxPass::new(3.0);
        blur_rows(
            Level::detect(),
            pass,
            &[1.0; 8],
            8,
            &mut [0.0; 8],
            &mut BoxScratch::default(),
        );
    }

    #[test]
    #[should_panic(expected = "more rows")]
    fn too_many_rows_are_refused() {
        let pass = BoxPass::new(3.0);
        let src = [1.0; 2 * (BLUR_ROWS + 1)];
        let mut dst = [0.0; (BLUR_ROWS + 1) * 8];
        blur_rows(
            Level::detect(),
            pass,
            &src,
            2,
            &mut dst,
            &mut BoxScratch::default(),
        );
    }

    #[test]
    #[should_panic(expected = "grown")]
    fn a_band_of_the_wrong_length_is_refused() {
        let pass = BoxPass::new(3.0);
        let src = [1.0; 4 * BLUR_BAND];
        let mut dst = [0.0; 4 * BLUR_BAND];
        blur_band(
            Level::detect(),
            pass,
            &src,
            &mut dst,
            &mut BoxScratch::default(),
        );
    }
}
