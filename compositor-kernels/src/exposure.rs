//! Exposure: each colour sample times a gain.

use crate::{Map, Params, flush};

/// Multiplies one sample by a gain, Exposure's whole per-pixel operation.
///
/// `gain` is the host's `exp2(stops)`, computed once per evaluation, since the GPU's `exp2` is
/// approximate. The single multiply is correctly rounded on both devices, so the GPU result
/// matches this bit for bit.
#[inline(always)]
pub fn exposure(v: f32, gain: f32) -> f32 {
    flush(v * gain)
}

/// Exposure: one sample times the gain, each colour plane through the map.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Exposure {
    pub gain: f32,
}

// SAFETY: `#[repr(C)]`, four-byte fields.
unsafe impl Params for Exposure {}

impl Map<1, 1> for Exposure {
    const ENTRY: &'static str = "gpu::exposure";

    #[inline(always)]
    fn apply(&self, [v]: [f32; 1]) -> [f32; 1] {
        [exposure(v, self.gain)]
    }
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;
    use crate::tests::{assert_exact, inputs};

    /// A correctly rounded reference independent of the kernel: the product of two f32s is
    /// exact in f64, so rounding it once to f32 is what IEEE asks of the multiply.
    fn reference(v: f32, gain: f32) -> f32 {
        flush((f64::from(v) * f64::from(gain)) as f32)
    }

    /// Gains the host computes: `exp2(stops)` in f64, rounded once to f32.
    fn gains() -> Vec<f32> {
        (-40..=40)
            .map(|quarter_stops| (f64::from(quarter_stops) / 4.0).exp2() as f32)
            .collect()
    }

    #[test]
    fn exposure_is_the_correctly_rounded_product() {
        for v in inputs() {
            for gain in gains() {
                assert_exact(exposure(v, gain), reference(v, gain));
            }
        }
    }

    #[test]
    fn exposure_special_values() {
        assert_exact(exposure(0.0, 2.0), 0.0);
        assert_exact(exposure(-0.0, 2.0), -0.0);
        assert_exact(exposure(f32::MAX, 2.0), f32::INFINITY);
        assert_exact(exposure(-f32::MAX, 2.0), f32::NEG_INFINITY);
        assert_exact(exposure(f32::INFINITY, 0.5), f32::INFINITY);
        assert_exact(exposure(f32::NEG_INFINITY, 0.5), f32::NEG_INFINITY);
        assert!(exposure(f32::INFINITY, 0.0).is_nan());
        assert!(exposure(f32::NAN, 2.0).is_nan());
        // Above f16's largest value, unclamped.
        assert_exact(exposure(65504.0, 2.0), 131008.0);
        // A subnormal result is flushed, keeping its sign.
        assert_exact(exposure(f32::MIN_POSITIVE, 0.5), 0.0);
        assert_exact(exposure(-f32::MIN_POSITIVE, 0.5), -0.0);
    }
}
