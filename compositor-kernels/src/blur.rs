//! Blur's definition: three box passes along each axis, each a direct weighted sum of the taps
//! within its half-width, a third of the size, with pixel-coverage weights.

use crate::{LineReduce, Params, Real, flush};

/// One of Blur's box passes, for a size in pixels: the taps and their weights that the CPU
/// kernel sums, the host uploads for the GPU entry point and the tests' reference convolves.
///
/// Pixel `i` covers `[i - 0.5, i + 0.5]` and the box `[-h, h]`, with `h` a third of the size,
/// so a tap's weight is its pixel's overlap with the box: 1 inside, a fraction at the edge,
/// which fades the outermost tap in as the size grows, so that the result is continuous in
/// size. The weights are symmetric and held for taps `0..=tmax` only, `tmax` being the
/// outermost tap whose overlap is positive; the taps beyond are excluded rather than weighted
/// zero, since `0 × ∞` is NaN. A box no wider than the centre pixel, sizes up to 1.5, is that
/// pixel alone.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxPass {
    half_width: f64,
    tmax: u32,
    /// The sum of every tap's overlap as f32 holds it, the weights' normaliser.
    sum: f64,
}

impl BoxPass {
    /// How many passes Blur runs along each axis.
    pub const PASSES: u32 = 3;

    /// The pass for a size, which the Size parameter keeps finite and at least 0.
    pub fn new(size: f64) -> BoxPass {
        let half_width = size.max(0.0) / 3.0;
        // The outermost tap whose pixel starts inside the box, `i - 0.5 < h`: the integer part
        // of `h + 0.5`, one less where that is whole and the pixel starts on the edge.
        let whole = (half_width + 0.5) as u32;
        let tmax = if f64::from(whole) == half_width + 0.5 {
            whole - 1
        } else {
            whole
        };
        let mut pass = BoxPass {
            half_width,
            tmax,
            sum: 0.0,
        };
        pass.sum = (1..=tmax).fold(f64::from(pass.overlap(0)), |sum, tap| {
            sum + 2.0 * f64::from(pass.overlap(tap))
        });
        pass
    }

    /// The outermost tap each side of the centre with a positive weight.
    pub fn tmax(self) -> u32 {
        self.tmax
    }

    /// How far a pixel's result reaches into its input along each axis, and how far the data
    /// window grows each side: three passes each reaching `tmax`. Never derived from the
    /// size's ceiling, which this exceeds by one at some sizes.
    pub fn reach(self) -> u32 {
        Self::PASSES.saturating_mul(self.tmax)
    }

    /// The weights of taps `0..=tmax`: each the tap's overlap with the box, divided by the sum
    /// of every tap's overlap as f32 holds it, so that the `2 tmax + 1` weights sum to 1 within
    /// f32 rounding and a flat image stays flat. Every weight is positive and normal. Every tap
    /// but the outermost has [`BoxPass::inner_weight`], and the outermost
    /// [`BoxPass::edge_weight`].
    pub fn weights(self) -> impl Iterator<Item = f32> {
        (0..=self.tmax).map(move |tap| self.weight(tap))
    }

    /// The weight of every tap inside the outermost: those pixels lie wholly in the box. With
    /// no tap inside, this is the centre's weight, 1.
    pub fn inner_weight(self) -> f32 {
        self.weight(0)
    }

    /// The weight of the outermost tap each side, the pixel the box ends in: the fraction of
    /// it the box covers, over the sum. The centre's own weight, 1, when it is the only tap.
    pub fn edge_weight(self) -> f32 {
        self.weight(self.tmax)
    }

    fn weight(self, tap: u32) -> f32 {
        if self.tmax == 0 {
            1.0
        } else {
            (f64::from(self.overlap(tap)) / self.sum) as f32
        }
    }

    /// The overlap of a tap's pixel with the box, exact in f64 and rounded once.
    fn overlap(self, tap: u32) -> f32 {
        let overlap = if tap == 0 {
            2.0 * self.half_width.min(0.5)
        } else {
            (self.half_width - (f64::from(tap) - 0.5)).min(1.0)
        };
        overlap as f32
    }
}

/// One of Blur's box passes as a line reduction: the window's core summed, then weighted with
/// the two ends, `edge (first + last) + inner core` fused, the line grown by `2 tmax`. The CPU
/// sums in f64, whose products and sums of f32 samples round far below the f32 each pass
/// stores; the GPU in f32.
///
/// Blur's input range on the GPU: magnitudes within `f32::MAX / max(2 tmax - 1, 2)`, where the
/// GPU's unweighted f32 sums of a window cannot overflow. Beyond it the GPU may give an infinity,
/// or NaN where infinities of both signs meet, where the CPU's result is finite. A pass's output
/// stays within its input's magnitude, so the range holds for all of Blur's passes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct BoxSum {
    tmax: u32,
    inner_weight: f32,
    edge_weight: f32,
}

impl BoxSum {
    /// The reduction for `pass`, which must have at least one tap each side: a pass of the
    /// centre alone is the identity, and Blur skips it.
    pub fn new(pass: BoxPass) -> BoxSum {
        assert!(pass.tmax() >= 1, "a box pass with taps beside the centre");
        BoxSum {
            tmax: pass.tmax(),
            inner_weight: pass.inner_weight(),
            edge_weight: pass.edge_weight(),
        }
    }
}

// SAFETY: `#[repr(C)]`, four-byte fields.
unsafe impl Params for BoxSum {}

impl LineReduce for BoxSum {
    type Acc = f64;

    const DIRECT: &'static str = "gpu::blur_direct";
    const SCAN: &'static str = "gpu::blur_scan";

    #[inline(always)]
    fn t(&self) -> u32 {
        self.tmax
    }

    #[inline(always)]
    fn shift(&self) -> u32 {
        2 * self.tmax
    }

    #[inline(always)]
    fn identity<R: Real>(&self) -> R {
        R::from_f32(0.0)
    }

    #[inline(always)]
    fn combine<R: Real>(&self, a: R, b: R) -> R {
        a + b
    }

    #[inline(always)]
    fn finish<R: Real>(&self, core: R, first: R, last: R) -> f32 {
        let edge = R::from_f32(self.edge_weight);
        let inner = R::from_f32(self.inner_weight);
        flush(edge.mul_add(first + last, inner * core).to_f32())
    }
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;

    fn weights(size: f64) -> Vec<f32> {
        BoxPass::new(size).weights().collect()
    }

    /// The weights over taps `-tmax..=tmax`.
    fn symmetric(size: f64) -> Vec<f64> {
        let half: Vec<f64> = weights(size).into_iter().map(f64::from).collect();
        let mut all: Vec<f64> = half[1..].iter().rev().copied().collect();
        all.extend(&half);
        all
    }

    /// The taps of the three passes composed, by convolving the pass with itself twice.
    fn composed(size: f64) -> Vec<f64> {
        let pass = symmetric(size);
        let mut kernel = pass.clone();
        for _ in 1..BoxPass::PASSES {
            let mut next = std::vec![0.0; kernel.len() + pass.len() - 1];
            for (i, a) in kernel.iter().enumerate() {
                for (j, b) in pass.iter().enumerate() {
                    next[i + j] += a * b;
                }
            }
            kernel = next;
        }
        kernel
    }

    fn standard_deviation(kernel: &[f64]) -> f64 {
        let centre = (kernel.len() / 2) as f64;
        kernel
            .iter()
            .enumerate()
            .map(|(i, w)| w * (i as f64 - centre).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn weights_are_the_pixels_overlap_with_the_box() {
        assert_eq!(weights(0.0), [1.0]);
        assert_eq!(weights(1.0), [1.0]);
        assert_eq!(weights(1.5), [1.0]);
        // Half-width 1: the centre pixel, and half of each neighbour.
        assert_eq!(weights(3.0), [0.5, 0.25]);
        // Half-width 1.5: three whole pixels.
        assert_eq!(weights(4.5), [1.0 / 3.0, 1.0 / 3.0]);
        // Half-width 2: three whole pixels and half of the next each side.
        assert_eq!(weights(6.0), [0.25, 0.25, 0.125]);
        // Negative and NaN sizes, which the parameter never gives, are the centre pixel.
        assert_eq!(weights(-2.0), [1.0]);
        assert_eq!(weights(f64::NAN), [1.0]);
    }

    /// The weights against the box drawn again from intervals: each pixel's overlap is the
    /// length of its intersection with the box, over every pixel near it, normalised in f64.
    /// The weights round three times, the overlaps, their sum and the quotient, each within
    /// half a unit.
    #[test]
    fn weights_are_each_pixels_share_of_the_box() {
        let sizes = (1..=40960).map(|tenths| f64::from(tenths) / 10.0);
        for size in sizes.chain([1.5 + 1e-9, 4.5 + 1e-9, 4096.0 - 1e-9]) {
            let half_width = size / 3.0;
            let overlap =
                |tap: f64| ((tap + 0.5).min(half_width) - (tap - 0.5).max(-half_width)).max(0.0);
            let beyond = half_width.ceil() + 1.0;
            let total: f64 = (-beyond as i64..=beyond as i64)
                .map(|tap| overlap(tap as f64))
                .sum();
            let expected: Vec<f64> = (0..)
                .map(|tap| overlap(f64::from(tap)) / total)
                .take_while(|share| *share > 0.0)
                .collect();
            let actual = weights(size);
            assert_eq!(actual.len(), expected.len(), "size {size}");
            for (tap, (&weight, &share)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(weight) - share).abs() <= 1.5 * f64::from(f32::EPSILON) * share,
                    "size {size}, tap {tap}: {weight} for {share}"
                );
            }
        }
    }

    #[test]
    fn inner_and_edge_weights_are_the_taps() {
        for size in [0.0, 1.5, 3.0, 4.5, 6.0, 31.4, 100.0, 4096.0] {
            let pass = BoxPass::new(size);
            let weights = weights(size);
            let t = pass.tmax() as usize;
            assert!(
                weights[..t].iter().all(|w| *w == pass.inner_weight()),
                "size {size}"
            );
            assert_eq!(weights[t], pass.edge_weight(), "size {size}");
        }
        assert_eq!(BoxPass::new(1.0).inner_weight(), 1.0);
        assert_eq!(BoxPass::new(1.0).edge_weight(), 1.0);
    }

    #[test]
    fn weights_sum_to_one_and_are_all_positive() {
        let sizes = (0..=40960).map(|tenths| f64::from(tenths) / 10.0);
        for size in sizes.chain([1.5 + 1e-9, 4096.0 - 1e-9]) {
            let pass = BoxPass::new(size);
            let weights: Vec<f32> = pass.weights().collect();
            assert_eq!(weights.len() as u32, pass.tmax() + 1, "size {size}");
            for weight in &weights {
                assert!(
                    weight.is_normal() && *weight > 0.0,
                    "size {size}: {weight:?}"
                );
            }
            let sum = (1..weights.len()).fold(f64::from(weights[0]), |sum, tap| {
                sum + 2.0 * f64::from(weights[tap])
            });
            assert!(
                (sum - 1.0).abs() <= f64::from(f32::EPSILON),
                "size {size}: {sum}"
            );
        }
    }

    #[test]
    fn reach_grows_in_threes_and_exceeds_the_ceiling_of_some_sizes() {
        let reach = |size: f64| BoxPass::new(size).reach();
        assert_eq!(reach(0.0), 0);
        assert_eq!(reach(1.5), 0);
        assert_eq!(reach(1.5 + 1e-9), 3);
        assert_eq!(reach(2.0), 3);
        assert_eq!(reach(3.0), 3);
        assert_eq!(reach(4.0), 3);
        assert_eq!(reach(4.5), 3);
        assert_eq!(reach(4.5 + 1e-9), 6);
        assert_eq!(reach(5.0), 6);
        assert_eq!(reach(4096.0), 4095);
        // Within 1.5 pixels of the size, reached below where a pixel starts on the box's edge,
        // so at most one past its ceiling; and never shrinking with the size.
        let mut previous = 0;
        for hundredths in 0..=409_600 {
            let size = f64::from(hundredths) / 100.0;
            let reach = reach(size);
            assert!(reach >= previous, "size {size}");
            assert!(
                f64::from(reach) >= size - 1.5 && f64::from(reach) < size + 1.5,
                "size {size}: reach {reach}"
            );
            previous = reach;
        }
    }

    #[test]
    fn weights_are_continuous_in_size() {
        // The weights change by at most 2 / h in total per unit of half-width, and the
        // half-width by a third of the size; a step of a hundredth stays well inside that.
        let step = 0.01;
        let mut previous = symmetric(0.0);
        for hundredths in 1..=3000 {
            let size = f64::from(hundredths) * step;
            let current = symmetric(size);
            let pad = (current.len() - previous.len()) / 2;
            let distance: f64 = current
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    let before = i
                        .checked_sub(pad)
                        .and_then(|i| previous.get(i))
                        .copied()
                        .unwrap_or(0.0);
                    (w - before).abs()
                })
                .sum();
            assert!(distance <= 0.02, "size {size}: {distance}");
            previous = current;
        }
    }

    #[test]
    fn composed_kernel_spans_the_reach_and_narrows_small_sizes() {
        for size in [2.0, 3.0, 10.0, 30.0, 100.0] {
            let pass = BoxPass::new(size);
            let kernel = composed(size);
            assert_eq!(kernel.len() as u32, 2 * pass.reach() + 1);
            assert!(kernel[0] > 0.0 && kernel[kernel.len() - 1] > 0.0);
            assert!((kernel.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        }
        // The standard deviation is near a third of the size only well above size 10: the
        // pixel's own width widens small boxes.
        assert!((standard_deviation(&composed(3.0)) - 1.2247).abs() < 1e-3);
        assert!(standard_deviation(&composed(3.0)) > 1.2);
        assert!((standard_deviation(&composed(10.0)) / (10.0 / 3.0) - 1.0).abs() < 0.01);
        assert!((standard_deviation(&composed(30.0)) / 10.0 - 1.0).abs() < 0.003);
    }
}
