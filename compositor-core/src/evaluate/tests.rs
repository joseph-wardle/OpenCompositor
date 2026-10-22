use open_ng::Point;

use super::*;
use crate::document::{Assignment, ReaderVersion};
use crate::fixtures::{self, directory};
use crate::image::{Channel, Warning, Window};
use crate::parameter::Value;

const ORIGIN: Point = Point { x: 0.0, y: 0.0 };

/// A key at most this deep is compared and dropped on a thread with a `DEEP_STACK`.
const DEEP_CHAIN: u64 = 10_000;

fn reader(path: impl AsRef<Path>) -> Node {
    let mut node = Node::new(
        NodeKind::Reader {
            version: ReaderVersion::V1,
        },
        ORIGIN,
    );
    node.set_value("file", Value::File(path.as_ref().into()))
        .unwrap();
    node
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

fn merge(mix: Option<f64>) -> Node {
    let mut node = Node::new(
        NodeKind::Merge {
            version: MergeVersion::V1,
        },
        ORIGIN,
    );
    if let Some(mix) = mix {
        node.set_value("mix", Value::Float(mix)).unwrap();
    }
    node
}

/// A document of these nodes, with IDs from 1, and these wires, each a consumer, its input and
/// the source.
fn graph(
    nodes: impl IntoIterator<Item = Node>,
    wires: &[(u64, InputSlot<'static>, u64)],
) -> Document {
    let mut document = Document::default();
    for (id, node) in (1..).map(NodeId).zip(nodes) {
        document.nodes.insert(id, Arc::new(node));
    }
    let assignments: Vec<Assignment> = wires
        .iter()
        .map(|&(consumer, input, source)| Assignment {
            consumer: NodeId(consumer),
            input,
            source: Some(NodeId(source)),
        })
        .collect();
    document.assign_inputs(&assignments).unwrap()
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
                input: "source".into(),
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

fn window(x: i32, y: i32, width: u32, height: u32) -> Window {
    Window {
        x,
        y,
        width,
        height,
    }
}

/// An image of one value per channel over a data window, with its own display window and
/// warnings.
fn flat(display: Window, data: Window, values: [f32; 4], warnings: Vec<Warning>) -> Image {
    let len = data.area() as usize;
    let planes = values.map(|value| std::iter::repeat_n(value, len).collect());
    Image::new(display, data, planes, warnings)
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
fn an_unconnected_input_or_a_reader_without_a_file_gives_the_empty_image() {
    let without_file = Node::new(
        NodeKind::Reader {
            version: ReaderVersion::V1,
        },
        ORIGIN,
    );
    let documents = [chain([exposure(Some(3.0))]), chain([without_file])];
    let mut evaluator = Evaluator::new();
    for document in documents {
        let image = run(&mut evaluator, &key(&document, 1)).unwrap();
        assert_eq!(image.display_window().width, 0);
        assert_eq!(image.data_window().height, 0);
        assert!(image.plane(Channel::R).is_empty());
    }
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

#[test]
fn merge_keys_name_the_connected_layers_in_slot_order() {
    let nodes = |mix| [reader("a.exr"), reader("b.exr"), merge(mix)];
    let slot = |slot| ("layer", slot).into();
    let slots_1_and_2 = graph(nodes(None), &[(3, slot(1), 1), (3, slot(2), 2)]);
    let slots_1_and_3 = graph(nodes(None), &[(3, slot(1), 1), (3, slot(3), 2)]);
    let reversed = graph(nodes(None), &[(3, slot(1), 2), (3, slot(2), 1)]);
    let as_background = graph(nodes(None), &[(3, "background".into(), 1), (3, slot(1), 2)]);
    let mixed = graph(nodes(Some(0.5)), &[(3, slot(1), 1), (3, slot(2), 2)]);
    let full = graph(nodes(Some(1.0)), &[(3, slot(1), 1), (3, slot(2), 2)]);

    // An empty slot composes as the empty image, so where the layers sit is not in the key.
    assert_eq!(key(&slots_1_and_2, 3), key(&slots_1_and_3, 3));
    assert_eq!(key(&slots_1_and_2, 3), key(&full, 3));
    assert_ne!(key(&slots_1_and_2, 3), key(&reversed, 3));
    assert_ne!(key(&slots_1_and_2, 3), key(&as_background, 3));
    assert_ne!(key(&slots_1_and_2, 3), key(&mixed, 3));
}

/// The key of a Merge of a background and one layer at `mix`, for computing from given images.
fn merge_key(mix: f64) -> ResultKey {
    let document = graph(
        [reader("a.png"), reader("b.png"), merge(Some(mix))],
        &[(3, "background".into(), 1), (3, ("layer", 1).into(), 2)],
    );
    key(&document, 3)
}

fn values(image: &Image, x: i32, y: i32) -> [f32; 4] {
    Channel::ALL.map(|channel| image.value(channel, x, y))
}

#[test]
fn merge_composes_offset_layers_over_the_base_within_their_union() {
    let evaluator = Evaluator::new();
    let base = flat(
        window(0, 0, 4, 4),
        window(0, 0, 2, 2),
        [1.0, 0.5, 0.25, 1.0],
        vec![Warning::IccProfile],
    );
    let layer = flat(
        window(10, 10, 1, 1),
        window(1, 1, 2, 2),
        [0.5, 0.5, 0.0, 0.5],
        vec![Warning::PixelAspect(2.0)],
    );

    let full = evaluator
        .compute(&merge_key(1.0), &[&base, &layer], &mut Vec::new())
        .unwrap();

    assert_eq!(full.display_window(), base.display_window());
    assert_eq!(full.data_window(), window(0, 0, 3, 3));
    assert_eq!(values(&full, 0, 0), [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(values(&full, 1, 1), [1.0, 0.75, 0.125, 1.0]);
    assert_eq!(values(&full, 2, 2), [0.5, 0.5, 0.0, 0.5]);
    assert_eq!(values(&full, 2, 0), [0.0; 4]);
    assert_eq!(
        full.warnings(),
        [Warning::IccProfile, Warning::PixelAspect(2.0)]
    );

    let half = evaluator
        .compute(&merge_key(0.5), &[&base, &layer], &mut Vec::new())
        .unwrap();

    assert_eq!(values(&half, 0, 0), [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(values(&half, 1, 1), [1.0, 0.625, 0.1875, 1.0]);
    assert_eq!(values(&half, 2, 2), [0.25, 0.25, 0.0, 0.25]);
    assert_eq!(values(&half, 2, 0), [0.0; 4]);

    // One file by two branches warns once.
    let diamond = evaluator
        .compute(&merge_key(1.0), &[&layer, &base, &layer], &mut Vec::new())
        .unwrap();
    assert_eq!(
        diamond.warnings(),
        [Warning::PixelAspect(2.0), Warning::IccProfile]
    );
}

#[test]
fn a_base_alone_or_mix_zero_is_the_base_shared_and_nothing_connected_is_empty() {
    let evaluator = Evaluator::new();
    let base = flat(
        window(0, 0, 4, 4),
        window(1, 1, 2, 2),
        [1.0, 0.5, 0.25, 1.0],
        Vec::new(),
    );
    let layer = flat(
        window(0, 0, 1, 1),
        window(5, 5, 2, 2),
        [0.5, 0.5, 0.0, 0.5],
        Vec::new(),
    );
    let shares_base = |image: &Image| {
        assert_eq!(image.data_window(), base.data_window());
        assert_eq!(image.display_window(), base.display_window());
        for channel in Channel::ALL {
            assert!(Arc::ptr_eq(image.plane(channel), base.plane(channel)));
        }
    };

    let alone = evaluator
        .compute(&merge_key(1.0), &[&base], &mut Vec::new())
        .unwrap();
    shares_base(&alone);
    let zero = evaluator
        .compute(&merge_key(0.0), &[&base, &layer], &mut Vec::new())
        .unwrap();
    shares_base(&zero);

    let unconnected = key(&graph([merge(None)], &[]), 1);
    let empty = evaluator
        .compute(&unconnected, &[], &mut Vec::new())
        .unwrap();
    assert!(empty.data_window().is_empty());
    assert!(empty.display_window().is_empty());
}

/// A Mix the kernel would receive as 0 or 1 is that special case, whatever its f64 value: the
/// products by zero that the cases avoid are the kernel's, in f32.
#[test]
fn mix_is_special_cased_as_the_kernel_receives_it() {
    let evaluator = Evaluator::new();
    let one = window(0, 0, 1, 1);
    let base = flat(one, one, [f32::INFINITY, 1.0, 1.0, 1.0], Vec::new());
    let layer = flat(one, one, [0.5, 0.5, 0.5, 0.5], Vec::new());
    let compute = |mix| {
        evaluator
            .compute(&merge_key(mix), &[&base, &layer], &mut Vec::new())
            .unwrap()
    };

    let nearly_one = compute(1.0 - 1e-10);
    assert_eq!(
        values(&nearly_one, 0, 0),
        [f32::INFINITY, 1.0, 1.0, 1.0],
        "the stack as composed, not dissolved against the base"
    );
    let nearly_zero = compute(1e-50);
    assert!(Arc::ptr_eq(
        nearly_zero.plane(Channel::R),
        base.plane(Channel::R)
    ));
}

/// Merge through the pool's chunking at every width across a SIMD tail, the layer offset by one
/// pixel from the base so that the union has rows and columns of each alone and of both: every
/// output sample is the shared functions' bit for bit, and every one is written, since the
/// outputs start as NaN and the inputs are finite.
#[test]
fn merge_on_the_pool_writes_every_sample_as_the_shared_functions_do() {
    let evaluator = Evaluator::new();
    let mix = merge_key(0.3);
    for height in [1, 7, 8, 9, 17] {
        for width in 1..=33 {
            let len = (width * height) as usize;
            let make = |x, y, seed| {
                let data = window(x, y, width, height);
                let planes = [1, 2, 3, 4].map(|c| finite_plane(len, c * 64 + width + seed));
                Image::new(window(0, 0, 1, 1), data, planes, Vec::new())
            };
            let base = make(0, 0, 0);
            let layer = make(1, 1, 1000);
            let union = window(0, 0, width + 1, height + 1);
            let nans =
                || -> Arc<[f32]> { std::iter::repeat_n(f32::NAN, union.area() as usize).collect() };
            let mut released = vec![nans(), nans(), nans(), nans()];

            let output = evaluator
                .compute(&mix, &[&base, &layer], &mut released)
                .unwrap();

            assert_eq!(output.data_window(), union);
            assert!(
                released.is_empty(),
                "{width}x{height} left a released plane unused"
            );
            for y in 0..union.height as i32 {
                for x in 0..union.width as i32 {
                    let alpha = layer.value(Channel::A, x, y);
                    for channel in Channel::ALL {
                        let b = base.value(channel, x, y);
                        let l = layer.value(channel, x, y);
                        let over = opencompositor_kernels::over(b, l, alpha);
                        let expected = opencompositor_kernels::mix(b, over, 0.3);
                        assert_eq!(
                            output.value(channel, x, y).to_bits(),
                            expected.to_bits(),
                            "{width}x{height} {channel:?} at {x}, {y}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_merge_too_large_for_an_image_or_the_cap_fails_before_allocating() {
    let evaluator = Evaluator::new();
    let key = merge_key(1.0);
    let pixel = |x, y| flat(window(0, 0, 1, 1), window(x, y, 1, 1), [1.0; 4], Vec::new());

    let far_apart = evaluator
        .compute(
            &key,
            &[&pixel(i32::MIN, 0), &pixel(i32::MAX, 0)],
            &mut Vec::new(),
        )
        .unwrap_err();
    assert!(
        matches!(&far_apart, EvalError::Oversize { origin, oversize }
            if *origin == key && *oversize == Oversize { width: 1 << 32, height: 1 }),
        "{far_apart:?}"
    );
    assert_eq!(
        far_apart.to_string(),
        "Merge: the data window would be 4294967296 × 1 pixels, more than an image can hold"
    );

    let over_cap = evaluator
        .compute(&key, &[&pixel(0, 0), &pixel(20000, 20000)], &mut Vec::new())
        .unwrap_err();
    let bytes = 20001 * 20001 * 16;
    assert!(
        matches!(&over_cap, EvalError::OverCap { origin, window, bytes: needed }
            if *origin == key && *window == self::window(0, 0, 20001, 20001) && *needed == bytes),
        "{over_cap:?}"
    );
    assert_eq!(
        over_cap.to_string(),
        "Merge: needs 6104 MiB of planes for a 20001 × 20001 data window, over the 1024 MiB limit"
    );
}

/// The addresses of every plane.
fn plane_addresses(image: &Image) -> BTreeSet<usize> {
    image.planes().iter().map(plane_address).collect()
}

/// One Reader through two Exposures into a Merge. Editing the layer's Exposure recomputes it and
/// the Merge, nothing else: the file is not read again, the background's planes stay where they
/// are, and the new results take the released planes of the old. Viewing the background then
/// returns it retained and releases the rest.
#[test]
fn editing_one_branch_of_a_diamond_recomputes_only_that_branch_and_reuses_its_planes() {
    let path = directory("diamond").join("image.png");
    write_png(&path, 255);
    let diamond = |stops: f64| {
        graph(
            [
                reader(&path),
                exposure(Some(1.0)),
                exposure(Some(stops)),
                merge(Some(0.5)),
            ],
            &[
                (2, "source".into(), 1),
                (3, "source".into(), 1),
                (4, "background".into(), 2),
                (4, ("layer", 1).into(), 3),
            ],
        )
    };
    let mut evaluator = Evaluator::new();
    let retained = |evaluator: &Evaluator, key: &ResultKey| -> Image {
        (evaluator.retained.iter())
            .find(|retained| retained.key == *key)
            .unwrap_or_else(|| panic!("{key:?} is retained"))
            .image
            .clone()
    };

    let first = run(&mut evaluator, &key(&diamond(2.0), 4)).unwrap();
    assert_eq!(red(&first), 3.0);
    assert_eq!(evaluator.retained.len(), 4);
    let background = plane_addresses(&retained(&evaluator, &key(&diamond(2.0), 2)));
    let mut old = colour_addresses(&retained(&evaluator, &key(&diamond(2.0), 3)));
    old.extend(plane_addresses(&first));
    assert_eq!(old.len(), 7);
    drop(first);

    // A re-read would see black.
    write_png(&path, 0);
    let second = run(&mut evaluator, &key(&diamond(3.0), 4)).unwrap();

    assert_eq!(red(&second), 5.0);
    assert_eq!(evaluator.retained.len(), 4);
    assert_eq!(
        plane_addresses(&retained(&evaluator, &key(&diamond(3.0), 2))),
        background
    );
    let mut new = colour_addresses(&retained(&evaluator, &key(&diamond(3.0), 3)));
    new.extend(plane_addresses(&second));
    assert_eq!(new, old);

    let viewed = run(&mut evaluator, &key(&diamond(3.0), 2)).unwrap();
    assert_eq!(red(&viewed), 2.0);
    assert_eq!(plane_addresses(&viewed), background);
    assert_eq!(evaluator.retained.len(), 2);
}
