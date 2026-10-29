use std::path::Path;
use std::time::{Duration, Instant};

use opencompositor_kernels::ulp_distance;

use super::*;
use crate::document::{InputSlot, Node, NodeKind, ReaderVersion};
use crate::evaluate::tests::{blur, chain, exposure, graph, key, merge, reader, write_png};
use crate::fixtures::{directory, write_exr};
use crate::kernels::headless;

/// Waits for the GPU to complete evaluation's submissions by polling without blocking, as the
/// worker will, failing after a generous while.
fn wait_until_idle(gpu: &GpuEvaluation) {
    let start = Instant::now();
    while !gpu.idle() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the GPU completes evaluation's work"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn run(evaluator: &mut Evaluator, gpu: &mut GpuEvaluation, key: &ResultKey) -> GpuImage {
    evaluator
        .evaluate_on(gpu, key, || false)
        .expect("nothing stops it")
        .expect("the request runs on the GPU")
}

/// The channel's values, read back once evaluation's work is complete.
fn read_back(gpu: &GpuEvaluation, image: &GpuImage, channel: Channel) -> Vec<f32> {
    wait_until_idle(gpu);
    let texture = image
        .texture(channel)
        .expect("a texture over the data window");
    headless::readback(&gpu.device, &gpu.queue, texture)
}

/// Values for every sample of an image of `count` samples: the special values Exposure must
/// carry exactly, and normal magnitudes of both signs over the whole range. `seed` rotates
/// them, so that each channel differs.
fn special_values(count: usize, seed: usize) -> Vec<f32> {
    const SPECIAL: [f32; 10] = [
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        65504.0,
        -1.0,
    ];
    (0..count)
        .map(|i| {
            let i = i + seed;
            if i.is_multiple_of(3) {
                SPECIAL[(i / 3) % SPECIAL.len()]
            } else {
                let magnitude = ((i % 1700) as f32 * 0.15 - 126.0).exp2();
                if i.is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                }
            }
        })
        .collect()
}

/// An EXR whose data window is offset from its display window to negative coordinates and
/// whose rows are no multiple of the copy alignment, holding the special values in every
/// channel.
fn special_exr(path: &Path) {
    let (width, height) = (37, 19);
    let channels = ["R", "G", "B", "A"]
        .into_iter()
        .zip(0..)
        .map(|(name, seed)| (name, special_values(width * height, seed)))
        .collect();
    write_exr(
        path,
        ((0, 0), (32, 16)),
        (-3, -2),
        (width, height),
        channels,
    );
}

#[test]
fn exposure_on_the_gpu_is_the_cpu_result_over_whole_images() {
    let path = directory("exposure").join("special.exr");
    special_exr(&path);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let mut cpu = Evaluator::new();
    // Whole and fractional stops, and one that flushes the smallest values to zero.
    for stops in [1.0, -2.5, 0.75, -20.0] {
        let key = key(&chain([reader(&path), exposure(Some(stops))]), 2);
        let expected = cpu.evaluate(&key, || false).unwrap().unwrap();
        let image = run(&mut evaluator, &mut gpu, &key);
        assert_eq!(image.info().data_window, expected.data_window());
        assert_eq!(image.info().display_window, expected.display_window());
        for channel in Channel::ALL {
            let actual = read_back(&gpu, &image, channel);
            let expected = expected.plane(channel);
            assert_eq!(actual.len(), expected.len());
            for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let what = format!("{stops} stops, {channel:?} sample {i}");
                assert_eq!(
                    ulp_distance(actual, expected),
                    0,
                    "{what}: {actual:?}, not {expected:?}"
                );
                if expected == 0.0 {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{what}: zero sign");
                }
            }
        }
    }
}

#[test]
fn exposure_on_the_gpu_shares_alpha_and_keeps_a_grey_image_one_plane() {
    let directory = directory("sharing");
    let grey = directory.join("grey.png");
    crate::evaluate::tests::write_grey_png(&grey, 200);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let document = chain([reader(&grey), exposure(Some(1.0))]);

    let read = run(&mut evaluator, &mut gpu, &key(&document, 1));
    let exposed = run(&mut evaluator, &mut gpu, &key(&document, 2));

    let texture = |image: &GpuImage, channel| image.texture(channel).unwrap() as *const _;
    assert_eq!(texture(&read, Channel::A), texture(&exposed, Channel::A));
    assert_eq!(texture(&exposed, Channel::R), texture(&exposed, Channel::G));
    assert_eq!(texture(&exposed, Channel::R), texture(&exposed, Channel::B));
    assert_ne!(texture(&exposed, Channel::R), texture(&read, Channel::R));
}

#[test]
fn a_stops_edit_uploads_nothing_and_reuses_the_released_output() {
    let path = directory("stops_edit").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let [one, two] = [1.0, 2.0].map(|stops| key(&chain([reader(&path), exposure(Some(stops))]), 2));
    let first = run(&mut evaluator, &mut gpu, &one);
    let uploaded = gpu.uploaded_bytes();
    assert!(uploaded > 0);
    let before = read_back(&gpu, &first, Channel::R);
    let first_output = first.texture(Channel::R).unwrap().clone();
    drop(first);

    let second = run(&mut evaluator, &mut gpu, &two);

    assert_eq!(
        gpu.uploaded_bytes(),
        uploaded,
        "a warm edit uploads nothing"
    );
    assert_eq!(second.texture(Channel::R), Some(&first_output));
    let after = read_back(&gpu, &second, Channel::R);
    assert_eq!(after, before.iter().map(|v| v * 2.0).collect::<Vec<_>>());
}

#[test]
fn a_released_texture_held_by_an_image_is_not_reused() {
    let path = directory("held").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let [one, two] = [1.0, 2.0].map(|stops| key(&chain([reader(&path), exposure(Some(stops))]), 2));
    let shown = run(&mut evaluator, &mut gpu, &one);
    let before = read_back(&gpu, &shown, Channel::R);

    let second = run(&mut evaluator, &mut gpu, &two);

    assert_ne!(second.texture(Channel::R), shown.texture(Channel::R));
    assert_eq!(read_back(&gpu, &shown, Channel::R), before);
}

#[test]
fn a_new_generation_uploads_again_without_decoding() {
    let path = directory("generation").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);
    let before = {
        let (device, queue) = headless::device();
        let mut gpu = GpuEvaluation::new(&device, &queue);
        let image = run(&mut evaluator, &mut gpu, &key);
        read_back(&gpu, &image, Channel::R)
    };
    // A file read again would be black.
    write_png(&path, 0);

    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let image = run(&mut evaluator, &mut gpu, &key);

    assert!(
        gpu.uploaded_bytes() > 0,
        "the new device has its own upload"
    );
    assert_eq!(read_back(&gpu, &image, Channel::R), before);
}

/// Values for Merge whose every product and sum is exact, so that a driver splitting the fused
/// multiply-add cannot differ from the CPU: small dyadic magnitudes of both signs, with the
/// non-finite classes, alpha among 0, ½, ¼ and 1. `seed` rotates them, so that each input
/// differs.
fn exact_merge_values(count: usize, seed: usize, alpha: bool) -> Vec<f32> {
    const COLOUR: [f32; 9] = [
        0.0,
        -0.0,
        0.5,
        0.25,
        1.0,
        2.0,
        -0.75,
        f32::INFINITY,
        f32::NAN,
    ];
    const ALPHA: [f32; 4] = [0.0, 0.5, 0.25, 1.0];
    (0..count)
        .map(|i| {
            let i = i * 7 + seed;
            if alpha {
                ALPHA[i % ALPHA.len()]
            } else {
                COLOUR[i % COLOUR.len()]
            }
        })
        .collect()
}

/// An EXR over `origin` and `size` of Merge's exact values, its display window the first's.
fn merge_exr(path: &Path, origin: (i32, i32), size: (usize, usize), seed: usize) {
    let count = size.0 * size.1;
    let channels = ["R", "G", "B", "A"]
        .into_iter()
        .zip(0..)
        .map(|(name, channel)| {
            let values = exact_merge_values(count, seed + channel, name == "A");
            (name, values)
        })
        .collect();
    write_exr(path, ((0, 0), (32, 16)), origin, size, channels);
}

/// A Reader with no file, whose image is empty.
fn reader_without_a_file() -> Node {
    Node::new(
        NodeKind::Reader {
            version: ReaderVersion::V1,
        },
        open_ng::Point { x: 0.0, y: 0.0 },
    )
}

/// Merge on the GPU against the CPU's result over whole images, bit for bit: offset inputs
/// whose union is larger than each, two layers through the stack between them, an empty
/// background and an empty layer, at Mix 1 and two dissolves.
#[test]
fn merge_on_the_gpu_is_the_cpu_result_over_whole_images() {
    let directory = directory("merge");
    let [background, first, second] =
        ["background", "first", "second"].map(|name| directory.join(format!("{name}.exr")));
    merge_exr(&background, (-3, -2), (37, 19), 0);
    merge_exr(&first, (5, 4), (23, 29), 3);
    merge_exr(&second, (-9, 10), (17, 6), 5);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let mut cpu = Evaluator::new();
    // Nodes 1 to 3 read the files, 4 reads nothing and 5 merges.
    let layer = |slot| ("layer", slot).into();
    let stacks: [&[(u64, InputSlot<'static>, u64)]; 4] = [
        &[(5, "background".into(), 1), (5, layer(1), 2)],
        &[
            (5, "background".into(), 1),
            (5, layer(1), 2),
            (5, layer(2), 3),
        ],
        &[(5, "background".into(), 4), (5, layer(1), 2)],
        &[(5, "background".into(), 1), (5, layer(1), 4)],
    ];
    for (stack, wires) in stacks.iter().enumerate() {
        for mix in [1.0, 0.5, 0.25] {
            let nodes = [
                reader(&background),
                reader(&first),
                reader(&second),
                reader_without_a_file(),
                merge(Some(mix)),
            ];
            let key = key(&graph(nodes, wires), 5);
            let expected = cpu.evaluate(&key, || false).unwrap().unwrap();
            let image = run(&mut evaluator, &mut gpu, &key);
            assert_eq!(image.info().data_window, expected.data_window());
            assert_eq!(image.info().display_window, expected.display_window());
            for channel in Channel::ALL {
                let actual = read_back(&gpu, &image, channel);
                let expected = expected.plane(channel);
                assert_eq!(actual.len(), expected.len());
                for (i, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                    let what = format!("stack {stack} at Mix {mix}, {channel:?} sample {i}");
                    assert_eq!(
                        ulp_distance(actual, expected),
                        0,
                        "{what}: {actual:?}, not {expected:?}"
                    );
                    if expected == 0.0 {
                        assert_eq!(actual.to_bits(), expected.to_bits(), "{what}: zero sign");
                    }
                }
            }
        }
    }
}

/// The base alone and Mix 0 are the base itself, as on the CPU: its textures, nothing computed.
#[test]
fn merge_of_the_base_alone_or_at_mix_0_is_the_base_itself() {
    let path = directory("merge_base").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let alone = graph(
        [reader(&path), merge(Some(0.5))],
        &[(2, "background".into(), 1)],
    );
    let mix_0 = graph(
        [reader(&path), merge(Some(0.0))],
        &[(2, "background".into(), 1), (2, ("layer", 1).into(), 1)],
    );
    let read = run(&mut evaluator, &mut gpu, &key(&alone, 1));
    for document in [alone, mix_0] {
        let merged = run(&mut evaluator, &mut gpu, &key(&document, 2));
        for channel in Channel::ALL {
            assert!(std::ptr::eq(
                read.texture(channel).unwrap(),
                merged.texture(channel).unwrap()
            ));
        }
    }
}

#[test]
fn an_operation_without_a_gpu_path_is_refused_before_anything_is_recorded() {
    let path = directory("refused").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let blurred = graph([reader(&path), blur(Some(4.0))], &[(2, "source".into(), 1)]);
    let requested = key(&blurred, 2);

    let error = evaluator
        .evaluate_on(&mut gpu, &requested, || false)
        .expect("nothing stops it")
        .expect_err("refused");

    assert!(matches!(error, EvalError::NoGpuPath { .. }), "{error:?}");
    assert!(*error.origin() == requested);
    assert_eq!(gpu.uploaded_bytes(), 0);
    assert_eq!(gpu.submitted, 0);
}

#[test]
fn a_stopped_request_submits_and_retains_nothing() {
    let path = directory("stopped").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);
    let asked = std::cell::Cell::new(0);
    // Stops before the second node, once the Reader's upload is recorded.
    let stop = || {
        asked.set(asked.get() + 1);
        asked.get() > 1
    };

    assert!(evaluator.evaluate_on(&mut gpu, &key, stop).is_none());

    assert_eq!(gpu.submitted, 0);
    assert!(gpu.retained.is_empty());
}

#[test]
fn a_reader_without_a_file_gives_the_empty_image() {
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let document = chain([reader_without_a_file(), exposure(Some(1.0))]);

    let image = run(&mut evaluator, &mut gpu, &key(&document, 2));

    assert!(image.info().data_window.is_empty());
    assert!(image.texture(Channel::R).is_none());
}

#[test]
fn reading_for_the_gpu_reads_what_the_request_uploads_and_nothing_uploaded() {
    let path = directory("read_for_gpu").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    // Retaining nothing on the CPU, so that only the GPU table can spare a read.
    let mut evaluator = Evaluator::with_cap(0);
    let [one, two] = [1.0, 2.0].map(|stops| key(&chain([reader(&path), exposure(Some(stops))]), 2));
    run(&mut evaluator, &mut gpu, &one);
    std::fs::remove_file(&path).unwrap();

    let uploaded = gpu.retained_keys();
    assert!(matches!(
        evaluator.read_for_gpu(&two, &uploaded, || false),
        Some(Ok(()))
    ));
    let read = evaluator.read_for_gpu(&two, &[], || false);
    assert!(
        matches!(read, Some(Err(EvalError::Read { .. }))),
        "{read:?}"
    );
}

#[test]
fn a_request_read_for_the_gpu_decodes_nothing_when_recorded() {
    let path = directory("read_then_record").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);

    assert!(matches!(
        evaluator.read_for_gpu(&key, &[], || false),
        Some(Ok(()))
    ));
    std::fs::remove_file(&path).unwrap();

    let image = run(&mut evaluator, &mut gpu, &key);
    assert_eq!(read_back(&gpu, &image, Channel::R), vec![2.0; 2]);
}

#[test]
fn a_recorded_request_dropped_unsubmitted_submits_and_retains_nothing() {
    let path = directory("recorded_dropped").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);

    let recorded = evaluator
        .record_on(&mut gpu, &key, || false)
        .unwrap()
        .unwrap();
    drop(recorded);

    assert_eq!(gpu.submitted, 0);
    assert!(gpu.retained_keys().is_empty());
}
