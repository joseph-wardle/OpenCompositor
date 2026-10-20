use open_ng::Point;

use super::*;
use crate::document::Assignment;
use crate::fixtures::{self, directory};
use crate::image::{Channel, Window};

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

/// A key at most this deep is compared and dropped on a thread with a `DEEP_STACK`.
const DEEP_CHAIN: u64 = 10_000;

fn reader(path: impl AsRef<Path>) -> Node {
    Node::new(
        NodeKind::Reader {
            path: path.as_ref().into(),
        },
        ORIGIN,
    )
}

fn exposure(stops: Option<f64>) -> Node {
    let mut node = Node::new(
        NodeKind::Exposure {
            version: ExposureVersion::V1,
        },
        ORIGIN,
    );
    if let Some(stops) = stops {
        node.set_value("stops", Value::Float(stops)).unwrap();
    }
    node
}

/// A 2 × 1 opaque PNG whose red is one 8-bit code and green and blue lower ones, replacing any
/// file at `path`.
fn write_png(path: &Path, code: u8) {
    let pixel = [code, code / 2, code / 4, 255];
    write(path, png::ColorType::Rgba, &[pixel, pixel].concat());
}

/// A 2 × 1 opaque grey PNG of one 8-bit code.
fn write_grey_png(path: &Path, code: u8) {
    write(
        path,
        png::ColorType::GrayscaleAlpha,
        &[code, 255, code, 255],
    );
}

fn write(path: &Path, color: png::ColorType, data: &[u8]) {
    fixtures::write_png(path, 2, 1, color, data);
}

/// A document of these nodes, with IDs from 1, each Exposure taking the image of the node before
/// it.
fn chain(nodes: impl IntoIterator<Item = Node>) -> Document {
    let mut document = Document::default();
    let mut previous = None;
    let mut assignments = Vec::new();
    for (id, node) in (1..).map(NodeId).zip(nodes) {
        if let (NodeKind::Exposure { .. }, Some(source)) = (&node.kind, previous) {
            assignments.push(Assignment {
                consumer: id,
                input: "source",
                source: Some(source),
            });
        }
        document.nodes.insert(id, Arc::new(node));
        previous = Some(id);
    }
    // From the end of the chain, so that each cycle check finds its source unconnected.
    for assignment in assignments.iter().rev() {
        document = document.assign_inputs(&[*assignment]).unwrap();
    }
    document
}

fn key(document: &Document, id: u64) -> ResultKey {
    result_key(document, &Reloads::default(), NodeId(id)).unwrap()
}

/// An image of these planes over `width` × `height`, as a reader would make it.
fn image(width: u32, height: u32, planes: [Arc<[f32]>; 4]) -> Image {
    let window = Window {
        x: 0,
        y: 0,
        width,
        height,
    };
    Image::new(window, window, planes, Vec::new())
}

/// `len` finite values of both signs, from tiny to huge, different for each `seed`.
fn finite_plane(len: usize, seed: u32) -> Arc<[f32]> {
    let mut state = seed.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            (unit - 0.5) * 2f32.powi((state & 0x3f) as i32 - 32)
        })
        .collect()
}

#[test]
fn keys_name_what_decides_the_image_and_nothing_else() {
    let document = chain([reader("a.exr"), exposure(Some(1.0))]);
    let mut elsewhere = chain([reader("b.exr"), reader("a.exr"), exposure(Some(1.0))]);
    let moved = Arc::make_mut(elsewhere.nodes.get_mut(&NodeId(3)).unwrap());
    moved.position = Point { x: 50.0, y: 20.0 };
    assert_eq!(key(&document, 2), key(&elsewhere, 3));
    assert_eq!(key(&document, 1), key(&elsewhere, 2));

    let different = [
        (chain([reader("b.exr"), exposure(Some(1.0))]), 2),
        (chain([reader("a.exr"), exposure(Some(2.0))]), 2),
        (
            chain([reader("a.exr"), exposure(Some(1.0)), exposure(None)]),
            3,
        ),
    ];
    for (other, id) in &different {
        assert_ne!(key(&document, 2), key(other, *id));
    }
    assert_ne!(key(&document, 1), key(&document, 2));
}

#[test]
fn a_value_equal_to_its_default_keys_as_the_default() {
    let unset = chain([reader("a.exr"), exposure(None)]);
    let zero = chain([reader("a.exr"), exposure(Some(0.0))]);
    let negative_zero = chain([reader("a.exr"), exposure(Some(-0.0))]);
    assert_eq!(key(&unset, 2), key(&zero, 2));
    assert_eq!(key(&unset, 2), key(&negative_zero, 2));
}

#[test]
fn an_unconnected_input_keys_apart_from_any_connection() {
    let unconnected = chain([exposure(None)]);
    let connected = chain([reader("a.exr"), exposure(None)]);
    assert_eq!(key(&unconnected, 1), key(&chain([exposure(None)]), 1));
    assert_ne!(key(&unconnected, 1), key(&connected, 2));
}

#[test]
fn reloading_a_path_changes_the_key_of_everything_downstream_of_it() {
    let document = chain([reader("a.exr"), exposure(None), reader("b.exr")]);
    let mut reloads = Reloads::default();
    let before = |id| result_key(&document, &Reloads::default(), NodeId(id));

    reloads.reload(Path::new("a.exr").into());
    let after = |id| result_key(&document, &reloads, NodeId(id));
    assert_ne!(after(1), before(1));
    assert_ne!(after(2), before(2));
    assert_eq!(after(3), before(3));
    assert_eq!(reloads.count(Path::new("a.exr")), 1);
}

#[test]
fn a_missing_node_has_no_key() {
    let document = chain([reader("a.exr")]);
    assert_eq!(result_key(&document, &Reloads::default(), NodeId(2)), None);
}

#[test]
fn a_deep_chain_keys_without_recursion() {
    let nodes = std::iter::once(reader("a.exr"))
        .chain((1..DEEP_CHAIN).map(|stops| exposure(Some(stops as f64 % 64.0))));
    let document = chain(nodes);

    // Built on the test's own stack, which recursion this deep would overflow.
    let first = key(&document, DEEP_CHAIN);
    let second = key(&document, DEEP_CHAIN);
    let shallower = key(&document, DEEP_CHAIN - 1);

    std::thread::Builder::new()
        .stack_size(DEEP_STACK)
        .spawn(move || {
            assert!(first == second);
            assert!(first != shallower);
        })
        .unwrap()
        .join()
        .unwrap();
}

fn run(evaluator: &mut Evaluator, key: &ResultKey) -> Result<Image, EvalError> {
    evaluator.evaluate(key, || false).expect("nothing stops it")
}

fn red(image: &Image) -> f32 {
    image.value(Channel::R, 0, 0)
}

/// A Reader of a file, and an Exposure of each of `stops` on it, the keys of the second
/// and third nodes.
fn exposure_keys(path: &Path, stops: [f64; 2]) -> [ResultKey; 2] {
    stops.map(|stops| key(&chain([reader(path), exposure(Some(stops))]), 2))
}

#[test]
fn exposure_scales_colour_and_shares_alpha() {
    let path = directory("exposure").join("image.png");
    write_png(&path, 255);
    let document = chain([reader(&path), exposure(Some(-2.0))]);
    let mut evaluator = Evaluator::new();

    let read = run(&mut evaluator, &key(&document, 1)).unwrap();
    let exposed = run(&mut evaluator, &key(&document, 2)).unwrap();

    for channel in [Channel::R, Channel::G, Channel::B] {
        let expected: Vec<f32> = read.plane(channel).iter().map(|v| v * 0.25).collect();
        assert_eq!(**exposed.plane(channel), *expected);
    }
    assert!(Arc::ptr_eq(
        read.plane(Channel::A),
        exposed.plane(Channel::A)
    ));
    assert_eq!(exposed.data_window(), read.data_window());
}

#[test]
fn exposure_keeps_a_grey_image_one_plane() {
    let path = directory("grey").join("grey.png");
    write_grey_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [exposed, _] = exposure_keys(&path, [1.0, 1.0]);

    let image = run(&mut evaluator, &exposed).unwrap();

    assert!(Arc::ptr_eq(
        image.plane(Channel::R),
        image.plane(Channel::G)
    ));
    assert!(Arc::ptr_eq(
        image.plane(Channel::R),
        image.plane(Channel::B)
    ));
    assert_eq!(red(&image), 2.0);
}

#[test]
fn an_unconnected_input_gives_the_empty_image() {
    let mut evaluator = Evaluator::new();
    let image = run(&mut evaluator, &key(&chain([exposure(Some(3.0))]), 1)).unwrap();
    assert_eq!(image.display_window().width, 0);
    assert_eq!(image.data_window().height, 0);
    assert!(image.plane(Channel::R).is_empty());
}

#[test]
fn a_stops_edit_reuses_the_read_and_reload_reads_again() {
    let path = directory("stops_edit").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [one, two] = exposure_keys(&path, [1.0, 2.0]);
    let before = red(&run(&mut evaluator, &one).unwrap());

    // A re-read would see black.
    write_png(&path, 0);
    assert_eq!(red(&run(&mut evaluator, &two).unwrap()), before * 2.0);

    let mut reloads = Reloads::default();
    reloads.reload(path.as_path().into());
    let document = chain([reader(&path), exposure(Some(2.0))]);
    let reloaded = result_key(&document, &reloads, NodeId(2)).unwrap();
    assert_eq!(red(&run(&mut evaluator, &reloaded).unwrap()), 0.0);
}

#[test]
fn viewing_an_unrelated_chain_releases_the_old_one() {
    let directory = directory("release");
    let (a, b) = (directory.join("a.png"), directory.join("b.png"));
    write_png(&a, 255);
    write_png(&b, 255);
    let mut evaluator = Evaluator::new();
    let [exposed, _] = exposure_keys(&a, [1.0, 1.0]);
    let before = red(&run(&mut evaluator, &exposed).unwrap());

    run(&mut evaluator, &key(&chain([reader(&b)]), 1)).unwrap();

    write_png(&a, 0);
    let after = red(&run(&mut evaluator, &exposed).unwrap());
    assert_ne!(after, before);
}

#[test]
fn releasing_everything_reads_again() {
    let path = directory("release-all").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [exposed, _] = exposure_keys(&path, [1.0, 1.0]);
    run(&mut evaluator, &exposed).unwrap();

    evaluator.release_all();

    write_png(&path, 0);
    assert_eq!(red(&run(&mut evaluator, &exposed).unwrap()), 0.0);
}

#[test]
fn stopping_keeps_what_was_completed() {
    let path = directory("stopping").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [exposed, _] = exposure_keys(&path, [1.0, 1.0]);
    let asked = std::cell::Cell::new(0);
    let stop = || {
        asked.set(asked.get() + 1);
        asked.get() > 1
    };

    assert!(evaluator.evaluate(&exposed, stop).is_none());
    assert_eq!(asked.get(), 2);

    write_png(&path, 0);
    assert_ne!(red(&run(&mut evaluator, &exposed).unwrap()), 0.0);
}

#[test]
fn the_cap_keeps_readers_first_and_never_a_result_without_its_inputs() {
    let path = directory("cap").join("image.png");
    write_png(&path, 255);
    let [one, two] = exposure_keys(&path, [1.0, 2.0]);
    let image_bytes = 4 * 2 * 4;

    // Room for the read, not for the Exposure's three new planes.
    let mut evaluator = Evaluator::with_cap(image_bytes);
    let before = red(&run(&mut evaluator, &one).unwrap());
    assert_eq!(evaluator.retained_bytes(), image_bytes);
    write_png(&path, 0);
    assert_eq!(red(&run(&mut evaluator, &two).unwrap()), before * 2.0);

    // No room for the read, so none for what is computed from it either.
    write_png(&path, 255);
    let mut evaluator = Evaluator::with_cap(image_bytes - 1);
    run(&mut evaluator, &one).unwrap();
    write_png(&path, 0);
    assert_eq!(red(&run(&mut evaluator, &two).unwrap()), 0.0);
}

#[test]
fn a_plane_shared_with_an_input_is_counted_once() {
    let path = directory("counting").join("image.png");
    write_png(&path, 255);
    let [exposed, _] = exposure_keys(&path, [1.0, 1.0]);

    // The Exposure shares the read's alpha, so seven planes hold both.
    let mut evaluator = Evaluator::with_cap(7 * 2 * 4);
    run(&mut evaluator, &exposed).unwrap();

    assert_eq!(evaluator.retained.len(), 2);
    assert_eq!(evaluator.retained_bytes(), 7 * 2 * 4);
}

#[test]
fn a_failure_carries_the_key_of_the_node_it_happened_at() {
    let missing = directory("failure").join("missing.png");
    let document = chain([reader(&missing), exposure(Some(1.0))]);
    let mut evaluator = Evaluator::new();

    let error = run(&mut evaluator, &key(&document, 2)).unwrap_err();

    assert_eq!(*error.origin(), key(&document, 1));
    assert!(
        matches!(&error, EvalError::Read { error: ReadError::Io(io), .. } if io.kind() == std::io::ErrorKind::NotFound),
        "{error:?}"
    );
    assert!(error.to_string().contains("missing.png"), "{error}");
}

#[test]
fn a_panicking_operation_fails_its_node() {
    // An Exposure whose key says it is connected, given no image, panics.
    let exposure = key(&chain([reader("a.png"), exposure(None)]), 2);
    let evaluator = Evaluator::new();
    let error = evaluator
        .compute(&exposure, &[], &mut Vec::new())
        .unwrap_err();
    assert!(matches!(error, EvalError::Panicked { origin } if origin == exposure));
}

/// Exposure through the pool's chunking at every width across a SIMD tail and at heights on both
/// sides of a chunk: every output sample is the shared function's bit for bit, and every one is
/// written, since the outputs start as NaN and the inputs are finite.
#[test]
fn exposure_on_the_pool_writes_every_sample_as_the_shared_function_does() {
    let evaluator = Evaluator::new();
    let exposure = key(&chain([reader("a.png"), exposure(Some(1.5))]), 2);
    let gain = 1.5_f64.exp2() as f32;
    for height in [1, 7, 8, 9, 17] {
        for width in 1..=33 {
            let len = (width * height) as usize;
            let input = image(
                width,
                height,
                [1, 2, 3, 4].map(|c| finite_plane(len, c * 64 + width)),
            );
            let nans = || -> Arc<[f32]> { std::iter::repeat_n(f32::NAN, len).collect() };
            let mut released = vec![nans(), nans(), nans()];

            let output = evaluator
                .compute(&exposure, &[&input], &mut released)
                .unwrap();

            for channel in [Channel::R, Channel::G, Channel::B] {
                let (src, dst) = (input.plane(channel), output.plane(channel));
                for (index, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
                    let expected = opencompositor_kernels::exposure(s, gain);
                    assert_eq!(
                        d.to_bits(),
                        expected.to_bits(),
                        "{width}x{height} {channel:?} at {index}"
                    );
                }
            }
            assert!(
                released.is_empty(),
                "{width}x{height} left a released plane unused"
            );
            assert!(Arc::ptr_eq(
                output.plane(Channel::A),
                input.plane(Channel::A)
            ));
        }
    }
}

#[test]
fn exposure_never_writes_its_input_even_where_alpha_is_a_colour_plane() {
    // One plane for all four channels, which an in-place write would show through A.
    let plane = finite_plane(12, 7);
    let before: Vec<u32> = plane.iter().map(|v| v.to_bits()).collect();
    let input = image(4, 3, [(); 4].map(|()| plane.clone()));
    let evaluator = Evaluator::new();
    let exposure = key(&chain([reader("a.png"), exposure(Some(2.0))]), 2);

    let output = evaluator
        .compute(&exposure, &[&input], &mut Vec::new())
        .unwrap();

    assert!(Arc::ptr_eq(output.plane(Channel::A), &plane));
    assert_eq!(
        plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        before
    );
    assert!(!Arc::ptr_eq(output.plane(Channel::R), &plane));
    assert!(Arc::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::G)
    ));
    assert!(Arc::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::B)
    ));
    assert_eq!(output.value(Channel::R, 1, 2), plane[9] * 4.0);
}

/// The addresses of the R, G and B planes.
fn colour_addresses(image: &Image) -> BTreeSet<usize> {
    [Channel::R, Channel::G, Channel::B]
        .map(|channel| plane_address(image.plane(channel)))
        .into()
}

#[test]
fn a_request_reuses_the_planes_of_the_results_it_releases() {
    let path = directory("reuse").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [one, two] = exposure_keys(&path, [1.0, 2.0]);
    let first = colour_addresses(&run(&mut evaluator, &one).unwrap());

    // The first result is now the evaluator's alone, and the second request releases it.
    let second = run(&mut evaluator, &two).unwrap();

    assert_eq!(colour_addresses(&second), first);
    assert_eq!(red(&second), 4.0);
}

#[test]
fn a_released_plane_held_elsewhere_is_not_reused() {
    let path = directory("held").join("image.png");
    write_png(&path, 255);
    let mut evaluator = Evaluator::new();
    let [one, two] = exposure_keys(&path, [1.0, 2.0]);
    let first = run(&mut evaluator, &one).unwrap();

    let second = run(&mut evaluator, &two).unwrap();

    assert!(colour_addresses(&first).is_disjoint(&colour_addresses(&second)));
    assert_eq!(red(&first) * 2.0, red(&second));
}

#[test]
fn the_pool_has_a_thread_per_physical_core_the_process_may_use() {
    let available = std::thread::available_parallelism().unwrap().get();
    let threads = pool_threads();
    assert!((1..=available).contains(&threads));
    println!("evaluator pool: {threads} threads at {:?}", Level::detect());
}

#[cfg(target_os = "linux")]
#[test]
fn cpu_lists_name_ranges_and_single_cpus() {
    assert_eq!(
        cpus_in("0-3,16-19,32\n"),
        Some(vec![0, 1, 2, 3, 16, 17, 18, 19, 32])
    );
    assert_eq!(cpus_in(""), Some(Vec::new()));
    assert_eq!(cpus_in("0-x"), None);
}

#[test]
fn a_deep_chain_evaluates_without_recursion() {
    let path = directory("deep").join("image.png");
    write_png(&path, 255);
    let nodes = |top: f64| {
        std::iter::once(reader(&path))
            .chain((1..DEEP_CHAIN - 1).map(|_| exposure(None)))
            .chain([exposure(Some(top))])
    };
    let (first, second) = (
        key(&chain(nodes(0.0)), DEEP_CHAIN),
        key(&chain(nodes(1.0)), DEEP_CHAIN),
    );
    let mut evaluator = Evaluator::new();

    let before = red(&run(&mut evaluator, &first).unwrap());
    write_png(&path, 0);
    assert_eq!(red(&run(&mut evaluator, &second).unwrap()), before * 2.0);
    assert_eq!(evaluator.retained.len(), DEEP_CHAIN as usize);

    std::thread::Builder::new()
        .stack_size(DEEP_STACK)
        .spawn(move || drop((evaluator, first, second)))
        .unwrap()
        .join()
        .unwrap();
}
