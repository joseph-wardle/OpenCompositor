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
mod exposure;
#[cfg(target_arch = "spirv")]
pub mod gpu;
mod merge;
pub mod primitives;

pub use blur::{BoxPass, BoxSum};
pub use exposure::{Exposure, exposure};
pub use merge::{Merge, mix, over};
pub use primitives::{
    Axis, DIRECT_T, Fold, FoldParams, LineParams, LineReduce, Map, MapParams, Params, Placement,
    Real, SCAN_BLOCK, SCAN_MIN_LINES, SCAN_THREADS, WORKGROUP, scan_lines, scan_segment,
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

    /// A bound of 0 ULP, and the exact sign of every zero, which the distance folds.
    pub(crate) fn assert_exact(actual: f32, expected: f32) {
        assert_eq!(
            ulp_distance(actual, expected),
            0,
            "{actual:?} != {expected:?}"
        );
        if expected == 0.0 {
            assert_eq!(actual.to_bits(), expected.to_bits(), "zero sign {actual:?}");
        }
    }

    /// Supported inputs: zero, or normal in magnitude, with the extremes of the plane invariant.
    pub(crate) fn inputs() -> Vec<f32> {
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
