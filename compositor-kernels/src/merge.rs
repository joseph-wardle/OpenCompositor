//! Merge: a stack of premultiplied layers over a background, dissolved against it by Mix.

use crate::{Fold, Params, flush, mul_add};

/// One channel of a premultiplied layer over a premultiplied background, `background * (1 -
/// alpha) + layer` fused: Merge's per-pixel operation, the same for every channel, alpha
/// included. Where the driver fuses it matches this bit for bit; where it splits, within 1 ULP.
#[inline(always)]
pub fn over(background: f32, layer: f32, alpha: f32) -> f32 {
    flush(mul_add(background, 1.0 - alpha, layer))
}

/// A dissolve by `mix` from the background to the stack composed over it, `(1 - mix) *
/// background + mix * over` as two products and a fused add, within 2 ULP where the driver
/// splits. Mix 0 and 1 are the evaluator's special cases: a product by zero turns an infinite
/// operand into NaN, where the artist asked for one side unchanged.
#[inline(always)]
pub fn mix(background: f32, over: f32, mix: f32) -> f32 {
    flush(mul_add(mix, over, (1.0 - mix) * background))
}

/// Merge: each layer's four planes over the stack below it where the layer is, then the stack
/// dissolved against the background by Mix. The fold always has a layer: a background alone,
/// and Mix 0, are the evaluator's to return unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Merge {
    pub mix: f32,
}

// SAFETY: `#[repr(C)]`, four-byte fields.
unsafe impl Params for Merge {}

impl Fold<4, 4> for Merge {
    const STEP: &'static str = "gpu::merge_step";
    const LAST: &'static str = "gpu::merge_last";

    /// Alpha composes as the colours do: the layer's alpha is the sample composed and the
    /// coverage both.
    #[inline(always)]
    fn step(&self, below: [f32; 4], [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        [
            over(below[0], r, a),
            over(below[1], g, a),
            over(below[2], b, a),
            over(below[3], a, a),
        ]
    }

    /// Mix 1 is the stack as composed: the dissolve's product by zero would make NaN of an
    /// infinite background.
    #[inline(always)]
    fn finish(&self, background: [f32; 4], stack: [f32; 4]) -> [f32; 4] {
        if self.mix == 1.0 {
            stack
        } else {
            let m = self.mix;
            [
                mix(background[0], stack[0], m),
                mix(background[1], stack[1], m),
                mix(background[2], stack[2], m),
                mix(background[3], stack[3], m),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_exact, inputs};
    use crate::ulp_distance;

    /// `over` is one fused multiply-add: against the product and sum taken exactly in f64 and
    /// rounded once, which differs from a fused f32 result only by double rounding, rarer than
    /// one in a million of these inputs and never by more than 1 ULP.
    #[test]
    fn over_is_the_fused_product_and_sum() {
        for background in inputs() {
            for (layer, alpha) in [(0.25, 0.5), (1.0, 1.0), (-3.0e38, 0.75), (0.0, 0.0)] {
                let actual = over(background, layer, alpha);
                let exact = f64::from(background) * f64::from(1.0 - alpha) + f64::from(layer);
                assert!(
                    ulp_distance(actual, flush(exact as f32)) <= 1,
                    "{background:?} under {layer:?} at alpha {alpha:?} gave {actual:?}"
                );
            }
        }
    }

    #[test]
    fn over_special_values() {
        // Nothing over a background leaves it, but for a negative zero, which the add makes
        // positive, and an opaque layer replaces it.
        for background in inputs() {
            if background.is_finite() {
                assert_eq!(ulp_distance(over(background, 0.0, 0.0), background), 0);
                assert_eq!(over(background, 0.5, 1.0).to_bits(), 0.5f32.to_bits());
            }
        }
        assert_exact(over(1.0, 0.25, 0.5), 0.75);
        // Alpha composes as every other channel: a half-transparent layer over an opaque one.
        assert_exact(over(1.0, 0.5, 0.5), 1.0);
        assert_exact(over(f32::INFINITY, 1.0, 0.5), f32::INFINITY);
        assert_exact(over(1.0, f32::NEG_INFINITY, 0.5), f32::NEG_INFINITY);
        assert!(over(f32::INFINITY, 1.0, 1.0).is_nan());
        assert!(over(f32::NAN, 0.0, 0.0).is_nan());
        assert!(over(1.0, 0.0, f32::NAN).is_nan());
        assert_exact(over(f32::MAX, f32::MAX, 0.0), f32::INFINITY);
        assert_exact(over(f32::MIN_POSITIVE, 0.0, 0.5), 0.0);
        assert_exact(over(-f32::MIN_POSITIVE, 0.0, 0.5), -0.0);
    }

    #[test]
    fn mix_dissolves_between_its_operands() {
        assert_exact(mix(1.0, 3.0, 0.5), 2.0);
        assert_exact(mix(1.0, 3.0, 0.25), 1.5);
        assert_exact(mix(-2.0, 2.0, 0.75), 1.0);
        assert_exact(mix(f32::INFINITY, 1.0, 0.5), f32::INFINITY);
        // The evaluator never asks for the endpoints: they would turn an infinity into NaN.
        assert!(mix(f32::INFINITY, 1.0, 1.0).is_nan());
        assert!(mix(1.0, f32::INFINITY, 0.0).is_nan());
        assert!(mix(f32::NAN, 1.0, 0.5).is_nan());
        assert_exact(mix(f32::MIN_POSITIVE, 0.0, 0.5), 0.0);
    }
}
