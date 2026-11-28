//! Premult: each colour sample times the alpha at its position.

use crate::primitives::map_entry;
use crate::{Map, Params, flush};

/// Multiplies one colour sample by its alpha, Premult's whole per-pixel operation. The single
/// multiply is correctly rounded on both devices, so the GPU result matches this bit for bit.
#[inline(always)]
pub fn premult(v: f32, alpha: f32) -> f32 {
    flush(v * alpha)
}

/// Premult: a colour sample and the alpha at its position to their product, each colour plane
/// with the alpha plane through the map. It has no parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Premult;

// SAFETY: `#[repr(C)]`, no fields.
unsafe impl Params for Premult {}

impl Map<2, 1> for Premult {
    const ENTRY: &'static str = ENTRY;

    #[inline(always)]
    fn apply(&self, [v, alpha]: [f32; 2]) -> [f32; 1] {
        [premult(v, alpha)]
    }
}

map_entry!(2 => 1, const Premult);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_exact, inputs};

    /// A correctly rounded reference independent of the kernel: the product of two f32s is
    /// exact in f64, so rounding it once to f32 is what IEEE asks of the multiply.
    fn reference(v: f32, alpha: f32) -> f32 {
        flush((f64::from(v) * f64::from(alpha)) as f32)
    }

    #[test]
    fn premult_is_the_correctly_rounded_product() {
        for v in inputs() {
            for alpha in inputs() {
                assert_exact(premult(v, alpha), reference(v, alpha));
            }
        }
    }

    #[test]
    fn premult_special_values() {
        assert_exact(premult(0.75, 1.0), 0.75);
        assert_exact(premult(0.75, 0.0), 0.0);
        assert_exact(premult(-0.75, 0.0), -0.0);
        assert_exact(premult(0.75, -0.0), -0.0);
        // Alpha outside [0, 1] is carried, not clamped.
        assert_exact(premult(0.5, 1.5), 0.75);
        assert_exact(premult(0.5, -0.5), -0.25);
        assert_exact(premult(f32::MAX, 2.0), f32::INFINITY);
        assert!(premult(f32::INFINITY, 0.0).is_nan());
        assert!(premult(0.5, f32::NAN).is_nan());
        // A subnormal result is flushed, keeping its sign.
        assert_exact(premult(f32::MIN_POSITIVE, 0.5), 0.0);
        assert_exact(premult(-f32::MIN_POSITIVE, 0.5), -0.0);
    }
}
