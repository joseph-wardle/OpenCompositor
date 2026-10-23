use std::sync::atomic::{AtomicUsize, Ordering};

use open_ng::Point;
use opencompositor_kernels::BoxPass;

use super::*;
use crate::document::{Assignment, BlurVersion, ReaderVersion};
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

fn blur(size: Option<f64>) -> Node {
    let mut node = Node::new(
        NodeKind::Blur {
            version: BlurVersion::V1,
        },
        ORIGIN,
    );
    if let Some(size) = size {
        node.set_value("size", Value::Float(size)).unwrap();
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
fn image(width: u32, height: u32, planes: [Plane; 4]) -> Image {
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
fn finite_plane(len: usize, seed: u32) -> Plane {
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

impl Evaluator {
    /// One node computed from given images, with nothing stopping it.
    fn compute_unstopped(
        &self,
        key: &ResultKey,
        inputs: &[&Image],
        released: &mut Vec<Plane>,
    ) -> Result<Image, EvalError> {
        self.compute(key, inputs, released, &|| false)
            .map(|image| image.expect("nothing stops it"))
    }
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
    assert!(Plane::ptr_eq(
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

    assert!(Plane::ptr_eq(
        image.plane(Channel::R),
        image.plane(Channel::G)
    ));
    assert!(Plane::ptr_eq(
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
    let asked = AtomicUsize::new(0);
    let stop = || asked.fetch_add(1, Ordering::Relaxed) + 1 > 1;

    assert!(evaluator.evaluate(&exposed, stop).is_none());
    assert_eq!(asked.load(Ordering::Relaxed), 2);

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
        .compute_unstopped(&exposure, &[], &mut Vec::new())
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
            let nans = || -> Plane { std::iter::repeat_n(f32::NAN, len).collect() };
            let mut released = vec![nans(), nans(), nans()];

            let output = evaluator
                .compute_unstopped(&exposure, &[&input], &mut released)
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
            assert!(Plane::ptr_eq(
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
        .compute_unstopped(&exposure, &[&input], &mut Vec::new())
        .unwrap();

    assert!(Plane::ptr_eq(output.plane(Channel::A), &plane));
    assert_eq!(
        plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        before
    );
    assert!(!Plane::ptr_eq(output.plane(Channel::R), &plane));
    assert!(Plane::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::G)
    ));
    assert!(Plane::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::B)
    ));
    assert_eq!(output.value(Channel::R, 1, 2), plane[9] * 4.0);
}

/// The addresses of the R, G and B planes.
fn colour_addresses(image: &Image) -> BTreeSet<usize> {
    [Channel::R, Channel::G, Channel::B]
        .map(|channel| image.plane(channel).address())
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
        .compute_unstopped(&merge_key(1.0), &[&base, &layer], &mut Vec::new())
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
        .compute_unstopped(&merge_key(0.5), &[&base, &layer], &mut Vec::new())
        .unwrap();

    assert_eq!(values(&half, 0, 0), [1.0, 0.5, 0.25, 1.0]);
    assert_eq!(values(&half, 1, 1), [1.0, 0.625, 0.1875, 1.0]);
    assert_eq!(values(&half, 2, 2), [0.25, 0.25, 0.0, 0.25]);
    assert_eq!(values(&half, 2, 0), [0.0; 4]);

    // One file by two branches warns once.
    let diamond = evaluator
        .compute_unstopped(&merge_key(1.0), &[&layer, &base, &layer], &mut Vec::new())
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
            assert!(Plane::ptr_eq(image.plane(channel), base.plane(channel)));
        }
    };

    let alone = evaluator
        .compute_unstopped(&merge_key(1.0), &[&base], &mut Vec::new())
        .unwrap();
    shares_base(&alone);
    let zero = evaluator
        .compute_unstopped(&merge_key(0.0), &[&base, &layer], &mut Vec::new())
        .unwrap();
    shares_base(&zero);

    let unconnected = key(&graph([merge(None)], &[]), 1);
    let empty = evaluator
        .compute_unstopped(&unconnected, &[], &mut Vec::new())
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
            .compute_unstopped(&merge_key(mix), &[&base, &layer], &mut Vec::new())
            .unwrap()
    };

    let nearly_one = compute(1.0 - 1e-10);
    assert_eq!(
        values(&nearly_one, 0, 0),
        [f32::INFINITY, 1.0, 1.0, 1.0],
        "the stack as composed, not dissolved against the base"
    );
    let nearly_zero = compute(1e-50);
    assert!(Plane::ptr_eq(
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
                || -> Plane { std::iter::repeat_n(f32::NAN, union.area() as usize).collect() };
            let mut released = vec![nans(), nans(), nans(), nans()];

            let output = evaluator
                .compute_unstopped(&mix, &[&base, &layer], &mut released)
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
        .compute_unstopped(
            &key,
            &[&pixel(i32::MIN, 0), &pixel(i32::MAX, 0)],
            &mut Vec::new(),
        )
        .unwrap_err();
    assert!(
        matches!(&far_apart, EvalError::Oversize { origin, oversize }
            if *origin == key
                && *oversize == Oversize { x: i64::from(i32::MIN), y: 0, width: 1 << 32, height: 1 }),
        "{far_apart:?}"
    );
    assert_eq!(
        far_apart.to_string(),
        "Merge: the data window would be 4294967296 × 1 pixels, more than an image can hold"
    );

    let over_cap = evaluator
        .compute_unstopped(&key, &[&pixel(0, 0), &pixel(20000, 20000)], &mut Vec::new())
        .unwrap_err();
    let bytes = 20001 * 20001 * 16;
    assert!(
        matches!(&over_cap, EvalError::OverCap { origin, window, bytes: needed }
            if *origin == key && *window == self::window(0, 0, 20001, 20001) && *needed == bytes),
        "{over_cap:?}"
    );
    assert_eq!(
        over_cap.to_string(),
        "Merge: needs 6104 MiB of memory for a 20001 × 20001 data window, over the 1024 MiB limit"
    );
}

/// The addresses of every plane.
fn plane_addresses(image: &Image) -> BTreeSet<usize> {
    image.planes().iter().map(Plane::address).collect()
}

/// One Reader through two branches into a Merge, an Exposure as the background and an Exposure
/// then a Blur as the layer. Editing the layer's Exposure recomputes it, the Blur and the Merge,
/// nothing else: the file is not read again, the background's planes stay where they are, and
/// the new results take the released planes of the old. Viewing the background then returns it
/// retained and releases the rest.
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
                blur(Some(2.0)),
            ],
            &[
                (2, "source".into(), 1),
                (3, "source".into(), 1),
                (5, "source".into(), 3),
                (4, "background".into(), 2),
                (4, ("layer", 1).into(), 5),
            ],
        )
    };
    // The layer's value and alpha at the origin are those of its 2 × 1 source, scaled by the
    // blur's mass over its two pixels; over the opaque base and dissolved by half, the red is
    // the base's 2 plus that mass times half the layer's excess over it.
    let kernel = composed(BoxPass::new(2.0));
    let reach = kernel.len() / 2;
    let mass = (kernel[reach] + kernel[reach + 1]) * kernel[reach];
    let red_at = |layer_red: f64| (2.0 + mass * (layer_red - 2.0) / 2.0) as f32;
    let mut evaluator = Evaluator::new();
    let retained = |evaluator: &Evaluator, key: &ResultKey| -> Image {
        (evaluator.retained.iter())
            .find(|retained| retained.key == *key)
            .unwrap_or_else(|| panic!("{key:?} is retained"))
            .image
            .clone()
    };

    let first = run(&mut evaluator, &key(&diamond(2.0), 4)).unwrap();
    assert!((red(&first) - red_at(4.0)).abs() < 1e-5, "{}", red(&first));
    assert_eq!(evaluator.retained.len(), 5);
    let background = plane_addresses(&retained(&evaluator, &key(&diamond(2.0), 2)));
    let mut old = colour_addresses(&retained(&evaluator, &key(&diamond(2.0), 3)));
    old.extend(plane_addresses(&retained(
        &evaluator,
        &key(&diamond(2.0), 5),
    )));
    old.extend(plane_addresses(&first));
    assert_eq!(old.len(), 11);
    drop(first);

    // A re-read would see black.
    write_png(&path, 0);
    let second = run(&mut evaluator, &key(&diamond(3.0), 4)).unwrap();

    assert!(
        (red(&second) - red_at(8.0)).abs() < 1e-5,
        "{}",
        red(&second)
    );
    assert_eq!(evaluator.retained.len(), 5);
    assert_eq!(
        plane_addresses(&retained(&evaluator, &key(&diamond(3.0), 2))),
        background
    );
    let mut new = colour_addresses(&retained(&evaluator, &key(&diamond(3.0), 3)));
    new.extend(plane_addresses(&retained(
        &evaluator,
        &key(&diamond(3.0), 5),
    )));
    new.extend(plane_addresses(&second));
    assert_eq!(new, old);

    let viewed = run(&mut evaluator, &key(&diamond(3.0), 2)).unwrap();
    assert_eq!(red(&viewed), 2.0);
    assert_eq!(plane_addresses(&viewed), background);
    assert_eq!(evaluator.retained.len(), 2);
}

#[test]
fn blur_keys_name_the_size_and_the_source() {
    let blurred = |size| graph([reader("a.exr"), blur(size)], &[(2, "source".into(), 1)]);
    assert_eq!(key(&blurred(None), 2), key(&blurred(Some(0.0)), 2));
    assert_ne!(key(&blurred(Some(1.0)), 2), key(&blurred(Some(2.0)), 2));
    assert_ne!(key(&blurred(Some(1.0)), 2), key(&blurred(None), 2));
    let unconnected = graph([reader("a.exr"), blur(Some(1.0))], &[]);
    assert_ne!(key(&unconnected, 2), key(&blurred(Some(1.0)), 2));
    assert_ne!(key(&unconnected, 2), key(&unconnected, 1));
}

/// The key of a Blur of a Reader at `size`, for computing from a given image.
fn blur_key(size: f64) -> ResultKey {
    let document = graph(
        [reader("a.png"), blur(Some(size))],
        &[(2, "source".into(), 1)],
    );
    key(&document, 2)
}

/// Blur's one-dimensional kernel along either axis: the three passes' weights convolved
/// together, `2 reach + 1` long, in f64.
fn composed(pass: BoxPass) -> Vec<f64> {
    let t = pass.tmax() as usize;
    let weights: Vec<f64> = pass.weights().map(f64::from).collect();
    let line: Vec<f64> = (0..=2 * t).map(|i| weights[i.abs_diff(t)]).collect();
    let mut kernel = vec![1.0];
    for _ in 0..BoxPass::PASSES {
        let mut next = vec![0.0; kernel.len() + line.len() - 1];
        for (i, k) in kernel.iter().enumerate() {
            for (j, l) in line.iter().enumerate() {
                next[i + j] += k * l;
            }
        }
        kernel = next;
    }
    kernel
}

/// What Blur's definition gives at a pixel: the composed two-dimensional kernel, the outer
/// product of [`composed`] with itself, summed explicitly in f64 over the source, transparent
/// black beyond its data window, and the same sum over the source's magnitudes, which bounds
/// the error of any computation that rounds its intermediates.
fn blur_reference(kernel: &[f64], source: &Image, channel: Channel, x: i32, y: i32) -> (f64, f64) {
    let reach = (kernel.len() / 2) as i32;
    let (mut result, mut magnitude) = (0.0, 0.0);
    for (j, kj) in kernel.iter().enumerate() {
        for (i, ki) in kernel.iter().enumerate() {
            let value =
                f64::from(source.value(channel, x + i as i32 - reach, y + j as i32 - reach));
            let weight = ki * kj;
            result += weight * value;
            magnitude += weight * value.abs();
        }
    }
    (result, magnitude)
}

/// Blur's CPU K: six f32 roundings, one per pass, each within half a unit of its result's
/// magnitude, and the rest of the arithmetic in f64.
const BLUR_K: f64 = 8.0;

/// Every output of a Blur against its definition: a non-finite result by IEEE class, and a
/// finite one within K units of 2⁻²⁴ times the same blur of the source's magnitudes, plus one
/// `MIN_POSITIVE` for a flushed intermediate.
fn assert_blur_matches_the_definition(pass: BoxPass, source: &Image, output: &Image, what: &str) {
    let kernel = composed(pass);
    assert_eq!(
        output.data_window(),
        source.data_window().grow(pass.reach()).unwrap()
    );
    let window = output.data_window();
    for channel in Channel::ALL {
        for y in window.y..window.y + window.height as i32 {
            for x in window.x..window.x + window.width as i32 {
                let actual = output.value(channel, x, y);
                let (expected, magnitude) = blur_reference(&kernel, source, channel, x, y);
                let place = format!("{what} {channel:?} at ({x}, {y}): {actual} for {expected}");
                if expected.is_nan() || actual.is_nan() {
                    assert!(expected.is_nan() && actual.is_nan(), "{place}");
                } else if expected.is_infinite() || actual.is_infinite() {
                    assert!(f64::from(actual) == expected, "{place}");
                } else {
                    let bound = BLUR_K * magnitude * 2f64.powi(-24) + f64::from(f32::MIN_POSITIVE);
                    assert!(
                        (f64::from(actual) - expected).abs() <= bound,
                        "{place}, off by {} against {bound}",
                        (f64::from(actual) - expected).abs()
                    );
                }
            }
        }
    }
}

/// A source over a data window away from the origin, with a display window of its own and a
/// warning: finite values of both signs over all magnitudes in every plane, then, where the
/// plane has the room, a spike at f32's limit with its negation beside it and a lesser one in
/// G, an infinity of each sign in B, and a NaN in A.
fn blur_source(width: u32, height: u32) -> Image {
    let len = (width * height) as usize;
    let mut planes: [Vec<f32>; 4] =
        [1, 2, 3, 4].map(|c| finite_plane(len, c * 64 + width + height).to_vec());
    let mut put = |plane: usize, at: usize, value: f32| {
        if at < len {
            planes[plane][at] = value;
        }
    };
    put(1, len / 3, 3.0e38);
    put(1, len / 3 + 1, -3.0e38);
    put(1, len / 2, 1.0e30);
    put(2, len / 8, f32::INFINITY);
    put(2, 2 * len / 3, f32::NEG_INFINITY);
    put(3, len / 4, f32::NAN);
    Image::new(
        window(0, 0, 8, 8),
        window(-3, 5, width, height),
        planes.map(Plane::from),
        vec![Warning::IccProfile],
    )
}

/// Blur through the pool's row chunks and column bands, at sizes with the direct and the block
/// sums along rows and shapes on both sides of a chunk and a band: every output is the
/// definition's within Blur's bound, with its classes, and every one is written, since the
/// outputs and the scratch plane start as NaN. The display window and the warnings stay the
/// source's.
#[test]
fn blur_on_the_pool_matches_the_composed_kernel_and_writes_every_sample() {
    let evaluator = Evaluator::new();
    for &(width, height) in &[(1, 1), (16, 16), (17, 17), (33, 9), (9, 33)] {
        let source = blur_source(width, height);
        for size in [1.6, 3.0, 10.0, 19.6, 22.5] {
            let pass = BoxPass::new(size);
            let reach = pass.reach();
            let grown = source.data_window().grow(reach).unwrap();
            let scratch_len = ((width + 2 * reach) * height) as usize;
            let nans = |len: usize| -> Plane { std::iter::repeat_n(f32::NAN, len).collect() };
            let mut released: Vec<Plane> = (0..4)
                .map(|_| nans(grown.area() as usize))
                .chain([nans(scratch_len)])
                .collect();

            let output = evaluator
                .compute_unstopped(&blur_key(size), &[&source], &mut released)
                .unwrap();

            let what = format!("{width}x{height} at size {size}");
            assert_blur_matches_the_definition(pass, &source, &output, &what);
            assert_eq!(output.display_window(), source.display_window());
            assert_eq!(output.warnings(), source.warnings());
            // The scratch plane came back, and nothing else was left.
            assert_eq!(released.len(), 1, "{what}");
            assert_eq!(released[0].len(), scratch_len, "{what}");
        }
    }
}

/// Dragging Blur's Size grows its window a little every few units. Its first output planes have
/// a quarter to spare and serve the drag up and back down, each edit taking the planes the last
/// one released: the scratch plane, taken after the outputs, never takes one of them.
#[test]
fn dragging_blur_size_reuses_its_first_output_planes() {
    let evaluator = Evaluator::new();
    let source = blur_source(90, 160);
    let addresses = |image: &Image| image.planes().each_ref().map(Plane::address);
    let mut previous = evaluator
        .compute_unstopped(&blur_key(10.0), &[&source], &mut Vec::new())
        .unwrap();
    let first = addresses(&previous);
    for size in [10.5, 11.0, 12.5, 14.0, 13.0, 11.5, 10.0] {
        let (_, planes) = previous.into_parts();
        let mut released = Vec::from(planes);
        let image = evaluator
            .compute_unstopped(&blur_key(size), &[&source], &mut released)
            .unwrap();
        let mut reused = addresses(&image);
        reused.sort_unstable();
        let mut expected = first;
        expected.sort_unstable();
        assert_eq!(reused, expected, "at size {size}");
        previous = image;
    }
}

#[test]
fn a_blur_whose_reach_is_zero_shares_its_source_and_takes_no_planes() {
    let evaluator = Evaluator::new();
    let source = blur_source(5, 4);
    for size in [0.0, 1.0, 1.5] {
        let mut released = vec![finite_plane(20, 1)];
        let output = evaluator
            .compute_unstopped(&blur_key(size), &[&source], &mut released)
            .unwrap();
        assert_eq!(output.data_window(), source.data_window());
        for channel in Channel::ALL {
            assert!(Plane::ptr_eq(output.plane(channel), source.plane(channel)));
        }
        assert_eq!(released.len(), 1);
    }
    let empty = evaluator
        .compute_unstopped(&blur_key(10.0), &[&Image::empty()], &mut Vec::new())
        .unwrap();
    assert!(empty.data_window().is_empty());
}

#[test]
fn blur_keeps_a_grey_image_one_plane() {
    let evaluator = Evaluator::new();
    let grey = finite_plane(12, 7);
    let alpha = finite_plane(12, 8);
    let source = image(4, 3, [grey.clone(), grey.clone(), grey, alpha]);

    let output = evaluator
        .compute_unstopped(&blur_key(3.0), &[&source], &mut Vec::new())
        .unwrap();

    assert!(Plane::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::G)
    ));
    assert!(Plane::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::B)
    ));
    assert!(!Plane::ptr_eq(
        output.plane(Channel::R),
        output.plane(Channel::A)
    ));
}

#[test]
fn a_blur_too_large_for_an_image_or_the_cap_fails_before_allocating() {
    let evaluator = Evaluator::new();
    let key = blur_key(4096.0);
    let reach = BoxPass::new(4096.0).reach();
    assert_eq!(reach, 4095);
    let pixel = |x, y| flat(window(0, 0, 1, 1), window(x, y, 1, 1), [1.0; 4], Vec::new());

    let at_the_edge = evaluator
        .compute_unstopped(&key, &[&pixel(i32::MIN, 0)], &mut Vec::new())
        .unwrap_err();
    assert!(
        matches!(&at_the_edge, EvalError::Oversize { origin, oversize }
        if *origin == key
            && *oversize == Oversize {
                x: i64::from(i32::MIN) - 4095,
                y: -4095,
                width: 8191,
                height: 8191
            }),
        "{at_the_edge:?}"
    );
    assert_eq!(
        at_the_edge.to_string(),
        "Blur: the data window would start at (-2147487743, -4095), beyond image coordinates"
    );

    let source = flat(
        window(0, 0, 1, 1),
        window(0, 0, 100, 100),
        [1.0; 4],
        Vec::new(),
    );
    let over_cap = evaluator
        .compute_unstopped(&key, &[&source], &mut Vec::new())
        .unwrap_err();
    // The outputs with a quarter to spare, the scratch plane, and each thread's temporaries.
    let len = 8290 * 8290;
    let threads = evaluator.pool.threads() as u64;
    let bytes = (len + len / 4) * 16
        + 8290 * 100 * 4
        + threads * blur::thread_scratch_bytes(BoxPass::new(4096.0), window(0, 0, 100, 100));
    assert!(
        matches!(&over_cap, EvalError::OverCap { origin, window, bytes: needed }
            if *origin == key && *window == self::window(-4095, -4095, 8290, 8290) && *needed == bytes),
        "{over_cap:?}"
    );
    assert_eq!(
        over_cap.to_string(),
        format!(
            "Blur: needs {} MiB of memory for a 8290 × 8290 data window, over the 1024 MiB limit",
            bytes >> 20
        )
    );
}

/// Blur asks before each chunk of rows and each band of columns, and once told to stop leaves
/// the rest undone: the request gives nothing, and what was computed before it stays retained.
#[test]
fn a_stopped_blur_leaves_the_rest_undone() {
    let evaluator = Evaluator::new();
    let source = blur_source(40, 40);
    let asked = AtomicUsize::new(0);
    // The third ask stops it.
    let stop = || asked.fetch_add(1, Ordering::Relaxed) >= 2;
    let mut released = Vec::new();

    let stopped = evaluator
        .compute(&blur_key(3.0), &[&source], &mut released, &stop)
        .unwrap();

    assert!(stopped.is_none());
    // Every chunk of the first plane's rows asked, and no band or later plane did.
    assert_eq!(
        asked.load(Ordering::Relaxed),
        40_usize.div_ceil(cpu::BLUR_ROWS)
    );
    assert_eq!(released.len(), 1, "the scratch plane came back");

    let path = directory("blur_stop").join("image.png");
    write_png(&path, 255);
    let document = graph([reader(&path), blur(Some(3.0))], &[(2, "source".into(), 1)]);
    let mut evaluator = Evaluator::new();
    let blurred = key(&document, 2);
    // Asked before the Reader, before the Blur, and at its first chunk.
    asked.store(0, Ordering::Relaxed);
    assert!(evaluator.evaluate(&blurred, stop).is_none());
    assert_eq!(asked.load(Ordering::Relaxed), 3);
    assert_eq!(evaluator.retained.len(), 1);

    // A re-read would see black.
    write_png(&path, 0);
    assert_ne!(red(&run(&mut evaluator, &blurred).unwrap()), 0.0);
}

/// The pool's column bands at every width across a band and a SIMD tail, into a plane of NaN:
/// every sample is written by the band of its column, once.
#[test]
fn column_bands_write_every_sample_of_their_own_columns() {
    let pool = Pool::new();
    for height in [1, 3] {
        for width in 1..=33 {
            let mut plane = vec![f32::NAN; width * height];
            pool.fill_columns(
                &mut plane,
                width,
                16,
                || (),
                |(), _, mut band| {
                    let columns = band.columns();
                    for y in 0..height {
                        let row: Vec<f32> =
                            columns.clone().map(|x| (y * width + x) as f32).collect();
                        band.write_row(y, &row);
                    }
                },
            );
            let expected: Vec<f32> = (0..width * height).map(|i| i as f32).collect();
            assert_eq!(plane, expected, "{width}x{height}");
        }
    }
}
