//! The CPU kernels: each operation over a slice of samples, compiled once per x86-64 feature
//! level so the evaluator can use the vector width the machine has without being built for it.
//!
//! The wrappers are the only place `target_feature` appears. Each enables exactly one level's
//! features over the same body, so the body vectorises three ways while the per-sample
//! arithmetic stays the kernel crate's, bit for bit at every level.

use opencompositor_kernels as kernels;

/// The x86-64 feature level the kernels run at. Only [`Level::detect`] makes one, after checking
/// every feature its wrapper enables, so holding a level proves this machine supports it. Other
/// architectures have only the baseline, the plain build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Level(Kind);

// Private so that no caller can name a level the machine lacks: the kernels' safety rests on it.
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

/// Defines an operation at a level: `$name` takes the level and the arguments and runs
/// `$body`, an `#[inline(always)]` function of the same arguments, in the wrapper that enables
/// the level's features, so the body vectorises once per level.
///
/// The feature lists are x86-64-v3 and v4 as the psABI defines them, and `Level::detect` checks
/// the same lists.
macro_rules! at_levels {
    ($(#[$doc:meta])* $name:ident => $body:ident($($arg:ident: $ty:ty),* $(,)?)) => {
        $(#[$doc])*
        pub fn $name(level: Level, $($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "avx2,fma,bmi1,bmi2,lzcnt,movbe,f16c")]
            fn v3($($arg: $ty),*) {
                $body($($arg),*);
            }
            #[cfg(target_arch = "x86_64")]
            #[target_feature(
                enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,fma,bmi1,bmi2,lzcnt,movbe,f16c"
            )]
            fn v4($($arg: $ty),*) {
                $body($($arg),*);
            }
            match level.0 {
                Kind::Baseline => $body($($arg),*),
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `level` came from `Level::detect` on this machine, which found every
                // feature the wrapper enables.
                Kind::V3 => unsafe { v3($($arg),*) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: as above.
                Kind::V4 => unsafe { v4($($arg),*) },
            }
        }
    };
}

at_levels! {
    /// Exposure by `gain` from `src` into `dst`, two slices of one length: each sample of `dst`
    /// becomes [`kernels::exposure`] of the sample at the same index in `src`.
    exposure => exposure_samples(src: &[f32], dst: &mut [f32], gain: f32)
}

#[inline(always)]
fn exposure_samples(src: &[f32], dst: &mut [f32], gain: f32) {
    assert_eq!(src.len(), dst.len(), "Exposure's slices differ in length");
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = kernels::exposure(s, gain);
    }
}

at_levels! {
    /// A layer over the background in `dst`, in place: each sample of each plane of `dst`
    /// becomes [`kernels::over`] of itself, the layer's sample at the same index and the
    /// layer's alpha there. The planes are R, G, B and A, all of one length.
    over => over_samples(layer: [&[f32]; 4], dst: [&mut [f32]; 4])
}

#[inline(always)]
fn over_samples(layer: [&[f32]; 4], dst: [&mut [f32]; 4]) {
    let [lr, lg, lb, la] = layer;
    let [dr, dg, db, da] = dst;
    let len = la.len();
    assert!(
        [
            lr.len(),
            lg.len(),
            lb.len(),
            dr.len(),
            dg.len(),
            db.len(),
            da.len()
        ]
        .iter()
        .all(|&n| n == len),
        "over's planes differ in length"
    );
    // Alpha composes as the colours do, from the layer's own alpha; it is written last, since
    // the colours read it.
    for (d, (&l, &a)) in dr.iter_mut().zip(lr.iter().zip(la)) {
        *d = kernels::over(*d, l, a);
    }
    for (d, (&l, &a)) in dg.iter_mut().zip(lg.iter().zip(la)) {
        *d = kernels::over(*d, l, a);
    }
    for (d, (&l, &a)) in db.iter_mut().zip(lb.iter().zip(la)) {
        *d = kernels::over(*d, l, a);
    }
    for (d, &a) in da.iter_mut().zip(la) {
        *d = kernels::over(*d, a, a);
    }
}

at_levels! {
    /// The stack in `dst` dissolved against `base` by `mix`, in place: each sample of each plane
    /// of `dst` becomes [`kernels::mix`] of the base's sample at the same index, itself and
    /// `mix`. The planes are R, G, B and A, all of one length.
    mix => mix_samples(base: [&[f32]; 4], dst: [&mut [f32]; 4], mix: f32)
}

#[inline(always)]
fn mix_samples(base: [&[f32]; 4], dst: [&mut [f32]; 4], mix: f32) {
    for (base, dst) in base.into_iter().zip(dst) {
        assert_eq!(base.len(), dst.len(), "mix's planes differ in length");
        for (d, &b) in dst.iter_mut().zip(base) {
            *d = kernels::mix(b, *d, mix);
        }
    }
}

#[cfg(test)]
mod tests {
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

    /// Every sample at every length against the kernel crate's definition, bit for bit, or
    /// a skip when this machine lacks the level.
    fn exposure_matches_the_kernel_crate(level: Level) {
        let detected = Level::detect();
        if level > detected {
            println!("skipped {level:?}: this machine detects {detected:?}");
            return;
        }
        println!("cpu kernel tests at {level:?}");
        // A signalling NaN, which no multiply returns: a sample left unwritten fails below.
        let unwritten = f32::from_bits(0x7f80_0001);
        for len in (0..=70).chain([255, 1000, 4097]) {
            let src = samples(len);
            for gain in GAINS {
                let mut dst = vec![unwritten; len];
                exposure(level, &src, &mut dst, gain);
                for (i, (&s, &d)) in src.iter().zip(&dst).enumerate() {
                    let expected = kernels::exposure(s, gain);
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

    /// Four planes of samples at every length through `over` and then `mix`, in place, against
    /// the kernel crate's functions bit for bit, or a skip when this machine lacks the level.
    fn over_and_mix_match_the_kernel_crate(level: Level) {
        let detected = Level::detect();
        if level > detected {
            println!("skipped {level:?}: this machine detects {detected:?}");
            return;
        }
        for len in (0..=70).chain([255, 1000, 4097]) {
            let layer: [Vec<f32>; 4] = [1, 2, 3, 4].map(|seed| samples(len + seed));
            let background: [Vec<f32>; 4] = [5, 6, 7, 8].map(|seed| samples(len + seed));
            let layer: [&[f32]; 4] = layer
                .each_ref()
                .map(|plane| &plane[seed_skip(plane, len)..]);
            let background: [&[f32]; 4] = background
                .each_ref()
                .map(|plane| &plane[seed_skip(plane, len)..]);
            let mut dst: [Vec<f32>; 4] = background.map(<[f32]>::to_vec);

            over(level, layer, dst.each_mut().map(Vec::as_mut_slice));
            for channel in 0..4 {
                for i in 0..len {
                    let expected =
                        kernels::over(background[channel][i], layer[channel][i], layer[3][i]);
                    assert_eq!(
                        dst[channel][i].to_bits(),
                        expected.to_bits(),
                        "{level:?}, {len} samples: over at {i} of plane {channel}"
                    );
                }
            }

            let composed: [Vec<f32>; 4] = dst.clone();
            mix(
                level,
                background,
                dst.each_mut().map(Vec::as_mut_slice),
                0.3,
            );
            for channel in 0..4 {
                for i in 0..len {
                    let expected = kernels::mix(background[channel][i], composed[channel][i], 0.3);
                    assert_eq!(
                        dst[channel][i].to_bits(),
                        expected.to_bits(),
                        "{level:?}, {len} samples: mix at {i} of plane {channel}"
                    );
                }
            }
        }
    }

    /// Where a plane of `len + seed` samples starts, so that planes of one length differ.
    fn seed_skip(plane: &[f32], len: usize) -> usize {
        plane.len() - len
    }

    #[test]
    fn baseline_matches_the_kernel_crate() {
        exposure_matches_the_kernel_crate(Level(Kind::Baseline));
        over_and_mix_match_the_kernel_crate(Level(Kind::Baseline));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_matches_the_kernel_crate() {
        exposure_matches_the_kernel_crate(Level(Kind::V3));
        over_and_mix_match_the_kernel_crate(Level(Kind::V3));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_matches_the_kernel_crate() {
        exposure_matches_the_kernel_crate(Level(Kind::V4));
        over_and_mix_match_the_kernel_crate(Level(Kind::V4));
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn slices_of_different_lengths_are_refused() {
        exposure(Level::detect(), &[1.0; 8], &mut [0.0; 7], 1.0);
    }

    #[test]
    #[should_panic(expected = "differ in length")]
    fn planes_of_different_lengths_are_refused() {
        let layer = [1.0; 8];
        let mut short = [0.0; 7];
        let mut dst = [[0.0; 8]; 3];
        let [r, g, b] = &mut dst;
        over(Level::detect(), [&layer; 4], [r, g, b, &mut short]);
    }
}
