//! The CPU kernels' throughput on this machine, as `cargo bench -p opencompositor-core` runs
//! it. Exposure: the kernel alone on one thread, the evaluator's pool over its chunking with
//! reused and with fresh outputs, and a copy of the same planes through the same pool and
//! chunking, the reference a pointwise kernel is held to. Merge's over and mix: each in place
//! over four planes through the pool, held to the copy by the bytes each moves. It runs on the
//! workstation before each PR to `main`, never in CI: each gate is a fraction of the copy
//! measured in the same run, so it holds on any machine and never blesses a number.

use std::time::{Duration, Instant};

use opencompositor_core::bench::{Pool, exposure, mix, over};

/// A 4K plate. Its three colour planes go through; alpha is shared unchanged, so neither
/// Exposure nor the copy touches one.
const WIDTH: usize = 4096;
const HEIGHT: usize = 2160;
const PLANES: usize = 3;
/// Timed runs per column; the median is reported.
const RUNS: usize = 21;
/// The least of the copy's throughput Exposure over the pool with reused outputs must reach. A
/// pointwise kernel is memory-bound, so the copy is what the machine allows and the rest is the
/// kernel's own cost.
const GATE: f64 = 0.8;

fn main() {
    if cfg!(debug_assertions) {
        println!("kernels benchmark: skipped, it measures an optimized build (cargo bench)");
        return;
    }
    let pool = Pool::new();
    let level = pool.level();
    let area = WIDTH * HEIGHT;
    // Half a stop, as the evaluator derives it from Stops.
    let gain = 0.5f64.exp2() as f32;
    let inputs: Vec<Vec<f32>> = (1..=PLANES as u32).map(|seed| plate(seed, area)).collect();
    let mut outputs: Vec<Vec<f32>> = (0..PLANES).map(|_| vec![0.0; area]).collect();

    let alone = median(|| {
        for (src, dst) in inputs.iter().zip(&mut outputs) {
            exposure(level, src, dst, gain);
        }
    });
    let reused = median(|| {
        for (src, dst) in inputs.iter().zip(&mut outputs) {
            pool.run(WIDTH, src, dst, |level, src, dst| {
                exposure(level, src, dst, gain)
            });
        }
    });
    for (src, dst) in inputs.iter().zip(&outputs) {
        let expected = src
            .iter()
            .map(|&v| opencompositor_kernels::exposure(v, gain).to_bits());
        assert!(
            dst.iter().map(|v| v.to_bits()).eq(expected),
            "the pool wrote Exposure"
        );
    }
    let fresh = median(|| {
        for src in &inputs {
            // Allocated as the evaluator allocates a plane and dropped as one it did not reuse.
            let mut dst = vec![0.0; area];
            pool.run(WIDTH, src, &mut dst, |level, src, dst| {
                exposure(level, src, dst, gain)
            });
        }
    });
    let copy = median(|| {
        for (src, dst) in inputs.iter().zip(&mut outputs) {
            pool.run(WIDTH, src, dst, |_, src, dst| dst.copy_from_slice(src));
        }
    });
    for (src, dst) in inputs.iter().zip(&outputs) {
        assert!(src == dst, "the pool copied");
    }

    let bytes = (2 * PLANES * area * size_of::<f32>()) as f64;
    println!(
        "Exposure on {WIDTH}x{HEIGHT} RGB, {} threads at {level:?}, median of {RUNS} runs",
        pool.threads()
    );
    for (what, time) in [
        ("kernel alone, one thread", alone),
        ("pool, reused outputs", reused),
        ("pool, fresh outputs", fresh),
        ("copy, reused outputs", copy),
    ] {
        let ms = time.as_secs_f64() * 1e3;
        let throughput = bytes / time.as_secs_f64() / 1e9;
        println!("  {what:<26}{ms:>8.2} ms{throughput:>8.1} GB/s");
    }
    let ratio = copy.as_secs_f64() / reused.as_secs_f64();
    let verdict = if ratio >= GATE { "pass" } else { "FAIL" };
    println!("  Exposure at {ratio:.2} of the copy's throughput, gate {GATE}: {verdict}");
    let mut failed = ratio < GATE;

    // Merge's kernels, in place over four planes: over reads the layer's four planes and the
    // four it writes, and mix the base's four and the four it writes, so each moves twelve
    // planes' bytes where the copy moves two per plane.
    let layer: Vec<Vec<f32>> = (11..=14).map(|seed| plate(seed, area)).collect();
    let base: Vec<Vec<f32>> = (21..=24).map(|seed| plate(seed, area)).collect();
    let mut stack: Vec<Vec<f32>> = base.clone();
    let layer_planes: [&[f32]; 4] = [&layer[0], &layer[1], &layer[2], &layer[3]];
    let base_planes: [&[f32]; 4] = [&base[0], &base[1], &base[2], &base[3]];
    let over_pool = median(|| {
        let [r, g, b, a] = &mut stack[..] else {
            unreachable!("four planes");
        };
        pool.fill_rows([r, g, b, a], WIDTH, |level, first_row, chunk| {
            let rows = first_row * WIDTH..first_row * WIDTH + chunk[0].len();
            over(level, layer_planes.map(|plane| &plane[rows.clone()]), chunk);
        });
    });
    let mix_pool = median(|| {
        let [r, g, b, a] = &mut stack[..] else {
            unreachable!("four planes");
        };
        pool.fill_rows([r, g, b, a], WIDTH, |level, first_row, chunk| {
            let rows = first_row * WIDTH..first_row * WIDTH + chunk[0].len();
            mix(
                level,
                base_planes.map(|plane| &plane[rows.clone()]),
                chunk,
                0.5,
            );
        });
    });
    let copy_throughput = bytes / copy.as_secs_f64() / 1e9;
    let merge_bytes = (12 * area * size_of::<f32>()) as f64;
    println!("Merge on {WIDTH}x{HEIGHT} RGBA, in place over four planes, median of {RUNS} runs");
    for (what, time) in [("over, pool", over_pool), ("mix, pool", mix_pool)] {
        let ms = time.as_secs_f64() * 1e3;
        let throughput = merge_bytes / time.as_secs_f64() / 1e9;
        println!("  {what:<26}{ms:>8.2} ms{throughput:>8.1} GB/s");
        let ratio = throughput / copy_throughput;
        let verdict = if ratio >= GATE { "pass" } else { "FAIL" };
        let (name, _) = what.split_once(',').expect("a name before the comma");
        println!(
            "  {name} at {ratio:.2} of the copy's throughput by bytes moved, gate {GATE}: {verdict}"
        );
        failed |= ratio < GATE;
    }
    if failed {
        std::process::exit(1);
    }
}

/// The median over [`RUNS`] of what `run` takes, after one run to warm up.
fn median(mut run: impl FnMut()) -> Duration {
    run();
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    times.sort();
    times[RUNS / 2]
}

/// A plane of `area` scene-linear values in [0, 16) from a linear congruential sequence: varied,
/// never subnormal, and the same on every run.
fn plate(seed: u32, area: usize) -> Vec<f32> {
    let mut state = seed;
    (0..area)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 20) as f32
        })
        .collect()
}
