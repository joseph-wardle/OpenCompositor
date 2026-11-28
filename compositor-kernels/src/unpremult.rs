//! Unpremult: each colour sample divided by the alpha at its position.

use crate::primitives::map_entry;
use crate::{Map, Params, flush};

/// Divides one colour sample by its alpha, Unpremult's whole per-pixel operation, leaving the
/// sample where the alpha is zero of either sign, which would make it infinite or NaN.
///
/// The CPU's division is correctly rounded. Vulkan allows the GPU's 2.5 ULP, and only for a
/// divisor of magnitude 2⁻¹²⁶ to 2¹²⁶, which a nonzero alpha a plane holds meets below 2¹²⁶.
#[inline(always)]
pub fn unpremult(v: f32, alpha: f32) -> f32 {
    if alpha == 0.0 { v } else { flush(v / alpha) }
}

/// Unpremult: a colour sample and the alpha at its position to their quotient, each colour
/// plane with the alpha plane through the map. It has no parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Unpremult;

// SAFETY: `#[repr(C)]`, no fields.
unsafe impl Params for Unpremult {}

impl Map<2, 1> for Unpremult {
    const ENTRY: &'static str = ENTRY;

    #[inline(always)]
    fn apply(&self, [v, alpha]: [f32; 2]) -> [f32; 1] {
        [unpremult(v, alpha)]
    }
}

map_entry!(2 => 1, const Unpremult);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_exact, inputs};

    /// A correctly rounded reference independent of the kernel: the quotient rounded to f64
    /// and then to f32 is the quotient rounded once to f32, since f64 carries more than twice
    /// f32's precision plus two bits.
    fn reference(v: f32, alpha: f32) -> f32 {
        flush((f64::from(v) / f64::from(alpha)) as f32)
    }

    #[test]
    fn unpremult_is_the_correctly_rounded_quotient() {
        for v in inputs() {
            for alpha in inputs().into_iter().filter(|&alpha| alpha != 0.0) {
                assert_exact(unpremult(v, alpha), reference(v, alpha));
            }
        }
    }

    #[test]
    fn unpremult_special_values() {
        assert_exact(unpremult(0.75, 1.0), 0.75);
        assert_exact(unpremult(0.375, 0.5), 0.75);
        // Alpha zero of either sign leaves the colour, whatever it is.
        assert_exact(unpremult(0.75, 0.0), 0.75);
        assert_exact(unpremult(-0.75, -0.0), -0.75);
        assert_exact(unpremult(-0.0, 0.0), -0.0);
        assert_exact(unpremult(f32::INFINITY, 0.0), f32::INFINITY);
        assert!(unpremult(f32::NAN, 0.0).is_nan());
        // Alpha outside [0, 1] is carried, not clamped.
        assert_exact(unpremult(0.75, 1.5), 0.5);
        assert_exact(unpremult(0.5, -0.5), -1.0);
        assert_exact(unpremult(0.5, f32::INFINITY), 0.0);
        assert!(unpremult(0.5, f32::NAN).is_nan());
        assert_exact(unpremult(1.0, f32::MIN_POSITIVE), 2.0f32.powi(126));
        assert_exact(unpremult(f32::MAX, 0.5), f32::INFINITY);
        // A subnormal result is flushed, keeping its sign.
        assert_exact(unpremult(f32::MIN_POSITIVE, 2.0), 0.0);
        assert_exact(unpremult(-f32::MIN_POSITIVE, 2.0), -0.0);
    }
}
