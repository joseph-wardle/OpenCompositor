//! Each operation OpenCompositor computes, written once against the execution primitives in
//! [`primitives`], with the per-sample functions they apply and, compiled to SPIR-V, the GPU
//! halves of the primitives in the `gpu` module.
//!
//! Each operation's CPU result is its definition; its GPU result must land within the
//! operation's ULP bound of that result, measured by [`ulp_distance`]. Nothing here owns an
//! image, allocates or spawns threads: the primitives' CPU halves and the GPU dispatch live in
//! compositor-core, and the entry points are compiled to SPIR-V by the builder in `shaders/`.
#![no_std]

// For `mul_add`, which `core` has only unstably.
#[cfg(not(target_arch = "spirv"))]
extern crate std;

mod blur;
#[cfg(target_arch = "spirv")]
pub mod gpu;
pub mod primitives;

pub use blur::{BoxPass, BoxSum};
pub use primitives::{
    Axis, DIRECT_T, Fold, FoldParams, LineParams, LineReduce, Map, MapParams, Params, Placement,
    Real, SCAN_BLOCK, SCAN_THREADS, WORKGROUP,
};

/// `a * b + c` with one rounding: std's `mul_add` on the host, and on SPIR-V libm's `fmaf`,
/// which rust-gpu lowers to the GLSL `Fma` instruction. The host result is always fused, so the
/// CPU definition is one sequence of roundings at every level; a GPU driver may still split
/// `Fma` into a multiply and an add, which each operation's bound allows for.
#[inline(always)]
pub fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "spirv")]
    {
        libm::fmaf(a, b, c)
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        a.mul_add(b, c)
    }
}

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

/// Multiplies one sample by a gain, Exposure's whole per-pixel operation.
///
/// `gain` is the host's `exp2(stops)`, computed once per evaluation, since the GPU's `exp2` is
/// approximate. The single multiply is correctly rounded on both devices, so the GPU result
/// matches this bit for bit.
#[inline(always)]
pub fn exposure(v: f32, gain: f32) -> f32 {
    flush(v * gain)
}

/// A kernel result as it may be stored: a subnormal becomes zero with its sign, everything
/// else passes through. Planes never hold subnormals, so every kernel's last step is this.
///
/// It compares bits rather than magnitudes so that no flush-to-zero or denormals-are-zero
/// mode on either device can change what it does.
#[inline(always)]
pub fn flush(v: f32) -> f32 {
    let bits = v.to_bits();
    if (bits & !SIGN) < f32::MIN_POSITIVE.to_bits() {
        f32::from_bits(bits & SIGN)
    } else {
        v
    }
}

const SIGN: u32 = 1 << 31;

/// The GPU entry points, as the builder in `shaders/` compiled them for Vulkan 1.2. Rebuild it
/// whenever the `gpu` module or a function it calls changes; CI rebuilds and compares.
#[cfg(not(target_arch = "spirv"))]
pub const SPIRV: &[u8] = include_bytes!("../kernels.spv");

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

/// How far apart two results are, in representable values: the distance every per-operation
/// bound is stated in.
///
/// Both zeros count as one value, any two NaNs are equal, and each infinity is one step past
/// the largest finite magnitude. A NaN against a number is `u32::MAX`, outside every bound.
pub fn ulp_distance(a: f32, b: f32) -> u32 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => return 0,
        (false, false) => {}
        _ => return u32::MAX,
    }
    // Sign-magnitude bits as a signed integer: monotone in value, both zeros at 0.
    let ordered = |v: f32| -> i64 {
        let bits = v.to_bits();
        let magnitude = i64::from(bits & !SIGN);
        if bits & SIGN == 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    // The largest distance is between the infinities, 0xff00_0000.
    (ordered(a) - ordered(b)).unsigned_abs() as u32
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;

    const SUBNORMAL_MIN: f32 = f32::from_bits(1);
    const SUBNORMAL_MAX: f32 = f32::from_bits(0x007f_ffff);

    /// Exposure's bound of 0 ULP, and the exact sign of every zero, which the distance folds.
    fn assert_exposure(actual: f32, expected: f32) {
        assert_eq!(
            ulp_distance(actual, expected),
            0,
            "{actual:?} != {expected:?}"
        );
        if expected == 0.0 {
            assert_eq!(actual.to_bits(), expected.to_bits(), "zero sign {actual:?}");
        }
    }

    /// A correctly rounded reference independent of the kernel: the product of two f32s is
    /// exact in f64, so rounding it once to f32 is what IEEE asks of the multiply.
    fn reference(v: f32, gain: f32) -> f32 {
        flush((f64::from(v) * f64::from(gain)) as f32)
    }

    /// Supported inputs: zero, or normal in magnitude, with the extremes of the plane invariant.
    fn inputs() -> Vec<f32> {
        let mut values = std::vec![
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX
        ];
        for exponent in (-100..=127).step_by(7) {
            for mantissa in [0x0000_0000, 0x0000_0001, 0x0055_5555, 0x007f_ffff] {
                let magnitude = f32::from_bits(((exponent + 127) as u32) << 23 | mantissa);
                values.push(magnitude);
                values.push(-magnitude);
            }
        }
        values
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
                assert_exposure(exposure(v, gain), reference(v, gain));
            }
        }
    }

    #[test]
    fn exposure_special_values() {
        assert_exposure(exposure(0.0, 2.0), 0.0);
        assert_exposure(exposure(-0.0, 2.0), -0.0);
        assert_exposure(exposure(f32::MAX, 2.0), f32::INFINITY);
        assert_exposure(exposure(-f32::MAX, 2.0), f32::NEG_INFINITY);
        assert_exposure(exposure(f32::INFINITY, 0.5), f32::INFINITY);
        assert_exposure(exposure(f32::NEG_INFINITY, 0.5), f32::NEG_INFINITY);
        assert!(exposure(f32::INFINITY, 0.0).is_nan());
        assert!(exposure(f32::NAN, 2.0).is_nan());
        // Above f16's largest value, unclamped.
        assert_exposure(exposure(65504.0, 2.0), 131008.0);
        // A subnormal result is flushed, keeping its sign.
        assert_exposure(exposure(f32::MIN_POSITIVE, 0.5), 0.0);
        assert_exposure(exposure(-f32::MIN_POSITIVE, 0.5), -0.0);
    }

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
        assert_exposure(over(1.0, 0.25, 0.5), 0.75);
        // Alpha composes as every other channel: a half-transparent layer over an opaque one.
        assert_exposure(over(1.0, 0.5, 0.5), 1.0);
        assert_exposure(over(f32::INFINITY, 1.0, 0.5), f32::INFINITY);
        assert_exposure(over(1.0, f32::NEG_INFINITY, 0.5), f32::NEG_INFINITY);
        assert!(over(f32::INFINITY, 1.0, 1.0).is_nan());
        assert!(over(f32::NAN, 0.0, 0.0).is_nan());
        assert!(over(1.0, 0.0, f32::NAN).is_nan());
        assert_exposure(over(f32::MAX, f32::MAX, 0.0), f32::INFINITY);
        assert_exposure(over(f32::MIN_POSITIVE, 0.0, 0.5), 0.0);
        assert_exposure(over(-f32::MIN_POSITIVE, 0.0, 0.5), -0.0);
    }

    #[test]
    fn mix_dissolves_between_its_operands() {
        assert_exposure(mix(1.0, 3.0, 0.5), 2.0);
        assert_exposure(mix(1.0, 3.0, 0.25), 1.5);
        assert_exposure(mix(-2.0, 2.0, 0.75), 1.0);
        assert_exposure(mix(f32::INFINITY, 1.0, 0.5), f32::INFINITY);
        // The evaluator never asks for the endpoints: they would turn an infinity into NaN.
        assert!(mix(f32::INFINITY, 1.0, 1.0).is_nan());
        assert!(mix(1.0, f32::INFINITY, 0.0).is_nan());
        assert!(mix(f32::NAN, 1.0, 0.5).is_nan());
        assert_exposure(mix(f32::MIN_POSITIVE, 0.0, 0.5), 0.0);
    }

    #[test]
    fn flush_zeroes_only_subnormals() {
        for subnormal in [SUBNORMAL_MIN, SUBNORMAL_MAX, f32::from_bits(0x0040_0000)] {
            assert_eq!(flush(subnormal).to_bits(), 0.0f32.to_bits());
            assert_eq!(flush(-subnormal).to_bits(), (-0.0f32).to_bits());
        }
        for kept in [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.5,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(flush(kept).to_bits(), kept.to_bits());
        }
        assert!(flush(f32::NAN).is_nan());
        assert!(flush(-f32::NAN).is_nan());
    }

    #[test]
    fn ulp_distance_counts_representable_values() {
        assert_eq!(ulp_distance(1.0, 1.0), 0);
        assert_eq!(ulp_distance(1.0, 1.0 + f32::EPSILON), 1);
        assert_eq!(ulp_distance(1.0 + f32::EPSILON, 1.0), 1);
        assert_eq!(ulp_distance(1.0, 1.0 + 3.0 * f32::EPSILON), 3);
        assert_eq!(ulp_distance(0.0, -0.0), 0);
        assert_eq!(
            ulp_distance(f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
            2 * 0x0080_0000
        );
        assert_eq!(ulp_distance(f32::MAX, f32::INFINITY), 1);
        assert_eq!(ulp_distance(f32::NEG_INFINITY, f32::INFINITY), 0xff00_0000);
        assert_eq!(ulp_distance(f32::NAN, f32::NAN), 0);
        assert_eq!(ulp_distance(f32::NAN, -f32::NAN), 0);
        assert_eq!(ulp_distance(f32::NAN, 1.0), u32::MAX);
        assert_eq!(ulp_distance(f32::INFINITY, f32::NAN), u32::MAX);
    }
}
