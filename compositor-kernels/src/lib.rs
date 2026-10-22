//! Per-pixel functions shared by OpenCompositor's CPU and GPU kernels, and the GPU entry points
//! built on them.
//!
//! Each operation's CPU result is its definition; its GPU entry point must land within the
//! operation's ULP bound of that result, measured by [`ulp_distance`]. Nothing here owns an
//! image, allocates or spawns threads: the CPU shells and the GPU dispatch live in
//! compositor-core, and the entry points are compiled to SPIR-V by the builder in `shaders/`.
#![no_std]

// For `mul_add`, which `core` has only unstably.
#[cfg(not(target_arch = "spirv"))]
extern crate std;

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

/// Exposure's entry point in [`SPIRV`]: rust-gpu names each one by its path in this crate.
pub const EXPOSURE_ENTRY: &str = "gpu::exposure";

/// Over's entry point in [`SPIRV`].
pub const OVER_ENTRY: &str = "gpu::over";

/// The entry points' workgroup is this many invocations on each side; their `threads` must say
/// the same, since the host dispatches `ceil(width / WORKGROUP)` by `ceil(height / WORKGROUP)`.
pub const WORKGROUP: u32 = 16;

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

/// Exposure's dispatch parameters, as the entry point reads them from its uniform buffer.
///
/// The host writes exactly these bytes, in this order.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct ExposureParams {
    pub width: u32,
    pub height: u32,
    pub gain: f32,
}

impl ExposureParams {
    /// The uniform buffer's contents.
    pub fn bytes(self) -> [u8; 12] {
        let mut bytes = [0; 12];
        bytes[0..4].copy_from_slice(&self.width.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.height.to_ne_bytes());
        bytes[8..12].copy_from_slice(&self.gain.to_ne_bytes());
        bytes
    }
}

// The bytes the host writes are the struct the shader reads, with no padding.
const _: () = assert!(size_of::<ExposureParams>() == 12);

/// Over's dispatch parameters, as the entry point reads them from its uniform buffer: the
/// planes' size, which is all it needs.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct OverParams {
    pub width: u32,
    pub height: u32,
}

impl OverParams {
    /// The uniform buffer's contents.
    pub fn bytes(self) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[0..4].copy_from_slice(&self.width.to_ne_bytes());
        bytes[4..8].copy_from_slice(&self.height.to_ne_bytes());
        bytes
    }
}

const _: () = assert!(size_of::<OverParams>() == 8);

/// The GPU entry points, one invocation per texel of one plane.
#[cfg(target_arch = "spirv")]
pub mod gpu {
    use spirv_std::glam::{IVec2, UVec3, Vec4};
    use spirv_std::{Image, spirv};

    use crate::{ExposureParams, OverParams};

    type Input = Image!(2D, type = f32, sampled);
    type Output = Image!(2D, format = r32f, sampled = false);

    /// Exposure over one R32Float plane. Dispatched once per distinct colour plane, with the
    /// grid rounded up to whole 16 by 16 workgroups; invocations past the edge return, since
    /// an image store outside the texture is discarded only under robust image access.
    #[spirv(compute(threads(16, 16)))] // WORKGROUP
    pub fn exposure(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &ExposureParams,
        #[spirv(descriptor_set = 0, binding = 1)] input: &Input,
        #[spirv(descriptor_set = 0, binding = 2)] output: &Output,
    ) {
        if id.x >= params.width || id.y >= params.height {
            return;
        }
        let at = IVec2::new(id.x as i32, id.y as i32);
        let texel: Vec4 = input.fetch(at);
        // SAFETY: `at` is inside both textures, which the host sizes to the params' width and
        // height, and the output is bound as R32Float, the format this image type declares.
        unsafe { output.write(at, crate::exposure(texel.x, params.gain)) };
    }

    /// One plane of a layer over the same plane of the background, with the layer's alpha
    /// plane, into the output: dispatched once per plane, alpha's with the alpha plane bound
    /// as the layer too, over planes of one size.
    #[spirv(compute(threads(16, 16)))] // WORKGROUP
    pub fn over(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(uniform, descriptor_set = 0, binding = 0)] params: &OverParams,
        #[spirv(descriptor_set = 0, binding = 1)] background: &Input,
        #[spirv(descriptor_set = 0, binding = 2)] layer: &Input,
        #[spirv(descriptor_set = 0, binding = 3)] alpha: &Input,
        #[spirv(descriptor_set = 0, binding = 4)] output: &Output,
    ) {
        if id.x >= params.width || id.y >= params.height {
            return;
        }
        let at = IVec2::new(id.x as i32, id.y as i32);
        let background: Vec4 = background.fetch(at);
        let layer: Vec4 = layer.fetch(at);
        let alpha: Vec4 = alpha.fetch(at);
        // SAFETY: as for `exposure`: `at` is inside every texture, and the output is R32Float.
        unsafe { output.write(at, crate::over(background.x, layer.x, alpha.x)) };
    }
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
