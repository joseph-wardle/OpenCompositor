use std::path::Path;
use std::time::{Duration, Instant};

use opencompositor_kernels::ulp_distance;

use super::*;
use crate::document::{InputSlot, Node, NodeKind, ReaderVersion};
use crate::evaluate::tests::{blur, chain, exposure, graph, key, merge, reader, write_png};
use crate::fixtures::{directory, write_exr};
use crate::kernels::headless;

/// Tests' stand-in for the device running out of memory, which lavapipe never reports: with a
/// cap, a plane made while the planes alive would exceed it refuses the attempt that made it, as
/// a failed creation's error scope would. A plane counts as alive until nothing holds it.
#[derive(Debug, Default)]
pub(super) struct Shortage {
    cap: Option<u64>,
    made: Vec<std::sync::Weak<wgpu::Texture>>,
    refused: bool,
}

impl Shortage {
    pub(super) fn made(&mut self, texture: &Arc<wgpu::Texture>) {
        self.made.retain(|plane| plane.strong_count() > 0);
        let alive: u64 = (self.made.iter())
            .filter_map(std::sync::Weak::upgrade)
            .map(|plane| texture_bytes(&plane))
            .sum();
        if self
            .cap
            .is_some_and(|cap| alive + texture_bytes(texture) > cap)
        {
            self.refused = true;
        } else {
            self.made.push(Arc::downgrade(texture));
        }
    }

    pub(super) fn take_refusal(&mut self) -> bool {
        std::mem::take(&mut self.refused)
    }
}

fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    let size = texture.size();
    u64::from(size.width) * u64::from(size.height) * size_of::<f32>() as u64
}

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
fn a_released_texture_held_by_an_image_is_reused_only_once_let_go() {
    let path = directory("held").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let [one, two, three] =
        [1.0, 2.0, 3.0].map(|stops| key(&chain([reader(&path), exposure(Some(stops))]), 2));
    // Held, as the viewer holds the image it shows.
    let shown = run(&mut evaluator, &mut gpu, &one);
    let before = read_back(&gpu, &shown, Channel::R);
    let shown_output = shown.texture(Channel::R).unwrap().clone();

    let second = run(&mut evaluator, &mut gpu, &two);

    assert_ne!(second.texture(Channel::R), Some(&shown_output));
    assert_eq!(read_back(&gpu, &shown, Channel::R), before);
    drop((shown, second));

    let third = run(&mut evaluator, &mut gpu, &three);
    assert_eq!(third.texture(Channel::R), Some(&shown_output));
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

/// A document of a Reader, node 1, and a Blur of it at `size`, node 2.
fn blurred(reader: Node, size: f64) -> crate::document::Document {
    graph([reader, blur(Some(size))], &[(2, "source".into(), 1)])
}

/// Values for one channel of a Blur's source, `channel` in `Channel::ALL`'s order: zeros and
/// normal magnitudes of both signs over sixty stops, then, where the plane has the room, in R a
/// spike near f32's limit beside its negation, in G an infinity of each sign and in B a NaN, so
/// that most of every blur stays finite and bounded by its own magnitudes rather than a spike's.
fn blur_values(count: usize, channel: usize) -> Vec<f32> {
    let mut values: Vec<f32> = (0..count)
        .map(|i| {
            let i = i + 7 * channel;
            if i.is_multiple_of(5) {
                0.0
            } else {
                let magnitude = ((i % 800) as f32 * 0.0731 - 30.0).exp2();
                if i.is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                }
            }
        })
        .collect();
    let mut put = |at: usize, value: f32| {
        if at < count {
            values[at] = value;
        }
    };
    match channel {
        0 => {
            put(count / 3, 3.0e38);
            put(count / 3 + 1, -3.0e38);
        }
        1 => {
            put(count / 8, f32::INFINITY);
            put(2 * count / 3, f32::NEG_INFINITY);
        }
        2 => put(count / 4, f32::NAN),
        _ => {}
    }
    values
}

/// Blur's definition over the grown window of one channel of `source`: the composed
/// two-dimensional kernel summed in f64 over the source, transparent black beyond its data
/// window, and the same sum over the magnitudes, which bounds the error of any computation that
/// rounds its intermediates. The kernel is the outer product of the composed line kernel with
/// itself, so each output is summed along the source's longer side and then across it, in f64,
/// whose rounding is far below f32's; every weight is positive, so the order leaves the
/// non-finite classes as one sum would.
struct BlurDefinition {
    kernel: Vec<f64>,
    rows_first: bool,
    /// The source's length across the first sums, and the grown window's along them.
    across: usize,
    along: usize,
    /// The first sums: `across` lines of `along` values and magnitudes.
    lines: Vec<(f64, f64)>,
}

impl BlurDefinition {
    fn new(pass: BoxPass, source: &Image, channel: Channel) -> BlurDefinition {
        let kernel = crate::evaluate::tests::composed(pass);
        let window = source.data_window();
        let (width, height) = (window.width as usize, window.height as usize);
        let rows_first = width >= height;
        let (len, across) = if rows_first {
            (width, height)
        } else {
            (height, width)
        };
        let along = len + kernel.len() - 1;
        let mut lines = Vec::with_capacity(across * along);
        for j in 0..across {
            let sample = |i: usize| {
                let (x, y) = if rows_first { (i, j) } else { (j, i) };
                f64::from(source.value(channel, window.x + x as i32, window.y + y as i32))
            };
            lines.extend((0..along).map(|o| weighted_sum(&kernel, o, len, sample)));
        }
        BlurDefinition {
            kernel,
            rows_first,
            across,
            along,
            lines,
        }
    }

    /// The value and the magnitude at `(x, y)` from the grown window's corner.
    fn at(&self, x: usize, y: usize) -> (f64, f64) {
        let (along, across) = if self.rows_first { (x, y) } else { (y, x) };
        let line = |j: usize| self.lines[j * self.along + along];
        let value = weighted_sum(&self.kernel, across, self.across, |j| line(j).0);
        let magnitude = weighted_sum(&self.kernel, across, self.across, |j| line(j).1).0;
        (value.0, magnitude)
    }
}

/// Output `o` of `kernel` along a line of `len` samples, sample `i` lying `i + 2 reach - o`
/// along the kernel, and the same sum over the samples' magnitudes.
fn weighted_sum(kernel: &[f64], o: usize, len: usize, sample: impl Fn(usize) -> f64) -> (f64, f64) {
    let span = kernel.len() - 1;
    let (mut value, mut magnitude) = (0.0, 0.0);
    for i in o.saturating_sub(span)..(o + 1).min(len) {
        let (weight, v) = (kernel[i + span - o], sample(i));
        value += weight * v;
        magnitude += weight * v.abs();
    }
    (value, magnitude)
}

/// The GPU's K for Blur, under the quality ceiling: within 2⁻¹⁶ of the magnitude sum, a
/// thirty-second of a half-float step. The test prints the factor the adapter needed.
const BLUR_GPU_K: f64 = 256.0;

/// Every output of a Blur on the GPU against its definition: a non-finite result by IEEE class,
/// and a finite one within K units of 2⁻²⁴ of the same blur of the source's magnitudes, plus one
/// `MIN_POSITIVE` for each of the six passes' flushes. Returns the K it needed.
fn assert_gpu_blur_matches_the_definition(
    gpu: &GpuEvaluation,
    size: f64,
    source: &Image,
    output: &GpuImage,
    what: &str,
) -> f64 {
    let pass = BoxPass::new(size);
    let window = output.info().data_window;
    assert_eq!(window, source.data_window().grow(pass.reach()).unwrap());
    assert_eq!(output.info().display_window, source.display_window());
    assert_eq!(output.info().warnings, source.warnings());
    let flushes = 6.0 * f64::from(f32::MIN_POSITIVE);
    let mut worst: f64 = 0.0;
    let mut checked: Vec<&wgpu::Texture> = Vec::new();
    for channel in Channel::ALL {
        let texture = output.texture(channel).unwrap();
        if checked.contains(&texture) {
            continue;
        }
        checked.push(texture);
        let definition = BlurDefinition::new(pass, source, channel);
        let actual = read_back(gpu, output, channel);
        assert_eq!(actual.len() as u64, window.area());
        for (i, &actual) in actual.iter().enumerate() {
            let (x, y) = (i % window.width as usize, i / window.width as usize);
            let (expected, magnitude) = definition.at(x, y);
            let place = || format!("{what} {channel:?} at ({x}, {y}): {actual} for {expected}");
            if expected.is_nan() || actual.is_nan() {
                assert!(expected.is_nan() && actual.is_nan(), "{}", place());
            } else if expected.is_infinite() || actual.is_infinite() {
                assert!(f64::from(actual) == expected, "{}", place());
            } else {
                let difference = (f64::from(actual) - expected).abs();
                let unit = magnitude * 2f64.powi(-24);
                assert!(
                    difference <= BLUR_GPU_K * unit + flushes,
                    "{}, off by {difference} with {magnitude} summed",
                    place()
                );
                if unit > 0.0 {
                    worst = worst.max((difference - flushes) / unit);
                }
            }
        }
    }
    worst
}

/// Blur on the GPU against its definition over whole images, every output within Blur's GPU
/// bound and the non-finite ones by class, never beyond the reach: offset EXRs of every class
/// and a line longer than the scan's blocks along each axis, at sizes from one tap each side,
/// either side of a tap count's change, and either side of the change from direct sums to the
/// scan's blocks; then a grey PNG up to the largest Size, where only so small a source fits the
/// device's textures. The test prints the K it needed.
#[test]
fn blur_on_the_gpu_is_within_its_bound_of_the_composed_kernel() {
    let directory = directory("blur");
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let mut cpu = Evaluator::new();
    let mut worst: f64 = 0.0;
    let mut check = |path: &Path, sizes: &[f64]| {
        let read = key(&chain([reader(path)]), 1);
        let source = cpu.evaluate(&read, || false).unwrap().unwrap();
        for &size in sizes {
            let requested = key(&blurred(reader(path), size), 2);
            let output = run(&mut evaluator, &mut gpu, &requested);
            let what = format!("{} at size {size}", path.display());
            let needed =
                assert_gpu_blur_matches_the_definition(&gpu, size, &source, &output, &what);
            worst = worst.max(needed);
        }
    };
    // Tap counts change after 1.5 + 3 k, and the scan takes over after 181.5.
    let after = |size: f64| size + 1e-9;
    let sizes = [
        after(1.5),
        2.0,
        3.0,
        4.5,
        after(4.5),
        10.0,
        31.4,
        100.0,
        181.5,
        after(181.5),
    ];
    let shapes = [(1, 1), (16, 16), (33, 17), (257, 3), (2800, 2), (2, 2800)];
    for (index, (width, height)) in shapes.into_iter().enumerate() {
        let path = directory.join(format!("{width}x{height}.exr"));
        let channels = ["R", "G", "B", "A"]
            .into_iter()
            .zip(0..)
            .map(|(name, channel)| (name, blur_values(width * height, channel)))
            .collect();
        let origin = (index as i32 - 3, 2 - index as i32);
        write_exr(&path, ((0, 0), (32, 16)), origin, (width, height), channels);
        check(&path, &sizes);
    }
    let grey = directory.join("grey.png");
    crate::evaluate::tests::write_grey_png(&grey, 200);
    check(&grey, &[1000.0, 1201.5, after(1201.5), 4096.0]);
    println!(
        "Blur on the GPU: K {} at most, under {BLUR_GPU_K}",
        worst.ceil()
    );
}

/// As on the CPU: a plane the channels share is blurred once, and a size whose reach is zero,
/// or an empty source, gives the source itself.
#[test]
fn blur_on_the_gpu_keeps_a_grey_image_one_plane_and_shares_a_source_it_leaves() {
    let grey = directory("blur_sharing").join("grey.png");
    crate::evaluate::tests::write_grey_png(&grey, 200);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let texture = |image: &GpuImage, channel| image.texture(channel).unwrap() as *const _;

    let grey_blurred = run(
        &mut evaluator,
        &mut gpu,
        &key(&blurred(reader(&grey), 3.0), 2),
    );
    assert_eq!(
        texture(&grey_blurred, Channel::R),
        texture(&grey_blurred, Channel::G)
    );
    assert_eq!(
        texture(&grey_blurred, Channel::R),
        texture(&grey_blurred, Channel::B)
    );
    assert_ne!(
        texture(&grey_blurred, Channel::R),
        texture(&grey_blurred, Channel::A)
    );

    let read = run(&mut evaluator, &mut gpu, &key(&chain([reader(&grey)]), 1));
    for size in [0.0, 1.0, 1.5] {
        let unchanged = run(
            &mut evaluator,
            &mut gpu,
            &key(&blurred(reader(&grey), size), 2),
        );
        for channel in Channel::ALL {
            assert_eq!(texture(&unchanged, channel), texture(&read, channel));
        }
    }
    let empty = blurred(reader_without_a_file(), 10.0);
    let image = run(&mut evaluator, &mut gpu, &key(&empty, 2));
    assert!(image.info().data_window.is_empty());
}

/// A Size edit that keeps the reach uploads nothing and writes the previous result's released
/// textures.
#[test]
fn a_size_edit_uploads_nothing_and_reuses_the_released_outputs() {
    let path = directory("size_edit").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    // Three taps each side at both.
    let [one, two] = [10.0, 10.5].map(|size| key(&blurred(reader(&path), size), 2));
    let first = run(&mut evaluator, &mut gpu, &one);
    let uploaded = gpu.uploaded_bytes();
    let outputs: Vec<wgpu::Texture> = Channel::ALL
        .map(|channel| first.texture(channel).unwrap().clone())
        .to_vec();
    drop(first);

    let second = run(&mut evaluator, &mut gpu, &two);

    assert_eq!(gpu.uploaded_bytes(), uploaded);
    for channel in Channel::ALL {
        assert!(
            outputs.contains(second.texture(channel).unwrap()),
            "{channel:?}"
        );
    }
}

/// The outputs, the scratch plane and the passes' intermediate planes count toward the cap on
/// one node's allocations, which fails the request before anything is recorded.
#[test]
fn a_blur_over_the_cap_fails_before_recording() {
    let path = directory("blur_cap").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let requested = key(&blurred(reader(&path), 4096.0), 2);

    let error = evaluator
        .evaluate_on(&mut gpu, &requested, || false)
        .expect("nothing stops it")
        .expect_err("over the cap");

    // Four distinct planes over the 2 × 1 source grown by 4095 each side, the rows' scratch,
    // and the planes after the first and second passes along each axis.
    let (width, height) = (8192u64, 8191u64);
    let intermediates = (2 + 2730) + (2 + 5460) + width * (1 + 2730) + width * (1 + 5460);
    let bytes = 4 * (4 * width * height + width + intermediates);
    assert!(
        matches!(&error, EvalError::OverCap { origin, bytes: needed, .. }
            if *origin == requested && *needed == bytes),
        "{error:?}"
    );
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

#[test]
fn a_request_out_of_gpu_memory_releases_what_it_does_not_read_and_fits_on_its_retry() {
    let path = directory("out_of_memory_retry").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let document = chain([reader(&path), exposure(Some(1.0)), exposure(Some(2.0))]);
    drop(run(&mut evaluator, &mut gpu, &key(&document, 2)));
    // The upload's four planes and the first Exposure's three are alive; the second Exposure's
    // three fit only once the upload's colour planes, which it does not read, are released.
    let plane = 2 * size_of::<f32>() as u64;
    assert_eq!(gpu.retained_bytes(), 7 * plane);
    gpu.shortage.cap = Some(7 * plane);

    let image = run(&mut evaluator, &mut gpu, &key(&document, 3));

    assert_eq!(
        gpu.retained_keys().len(),
        2,
        "the two Exposures, not the upload"
    );
    assert_eq!(gpu.retained_bytes(), 7 * plane);
    let expected = Evaluator::new()
        .evaluate(&key(&document, 3), || false)
        .unwrap()
        .unwrap();
    for channel in Channel::ALL {
        assert_eq!(
            read_back(&gpu, &image, channel),
            expected.plane(channel).to_vec()
        );
    }
}

#[test]
fn a_request_out_of_gpu_memory_after_its_retry_fails_and_submits_nothing() {
    let path = directory("out_of_memory").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);
    gpu.shortage.cap = Some(0);

    let failed = evaluator.evaluate_on(&mut gpu, &key, || false).unwrap();

    let Err(EvalError::OutOfGpuMemory { origin }) = failed else {
        panic!("out of GPU memory, not {failed:?}");
    };
    assert!(Arc::ptr_eq(&origin.0, &key.0));
    assert_eq!(gpu.submitted, 0);
    assert!(gpu.retained_keys().is_empty());
    gpu.shortage.cap = None;
    run(&mut evaluator, &mut gpu, &key);
}

#[test]
fn a_request_wgpu_rejects_fails_and_retains_nothing() {
    let path = directory("rejected").join("image.png");
    write_png(&path, 255);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let mut evaluator = Evaluator::new();
    let key = key(&chain([reader(&path), exposure(Some(1.0))]), 2);
    let mut recorded = evaluator
        .record_on(&mut gpu, &key, || false)
        .unwrap()
        .unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    recorded
        .encoder()
        .copy_buffer_to_buffer(&buffer, 0, &buffer, 0, 8);

    let rejected = gpu.submit(recorded);

    assert!(matches!(rejected, Err(EvalError::GpuValidation { .. })));
    assert!(gpu.retained_keys().is_empty());
    run(&mut evaluator, &mut gpu, &key);
}
