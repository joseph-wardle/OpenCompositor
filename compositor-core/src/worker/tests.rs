use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use open_ng::Point;

use super::*;
use crate::evaluate::result_key;
use crate::fixtures::{self, directory};
use crate::gpu::{noop_device, noop_device_pair};
use crate::read::{ReadError, read_image};
use crate::session::Session;

/// The key of a Reader of the path.
fn key(path: impl Into<PathBuf>) -> ResultKey {
    let mut session = Session::new();
    let changes = session.create_readers(vec![path.into()], Point::default(), None);
    let id = *changes.nodes.first().unwrap();
    result_key(session.document(), session.reloads(), id).unwrap()
}

/// A request for the node with the key, as the worker cares only about which node it is.
fn on(node: u64, key: ResultKey) -> Wanted {
    Wanted {
        node: NodeId(node),
        key,
        number: 0,
    }
}

fn missing(name: &str) -> ResultKey {
    key(format!("/nonexistent/{name}.exr"))
}

/// A white, opaque PNG.
fn write_png(path: &Path, width: u32, height: u32) {
    let white = vec![255; width as usize * height as usize * 4];
    fixtures::write_png(path, width, height, png::ColorType::Rgba, &white);
}

fn worker() -> (Worker, Receiver<Evaluated>) {
    let (delivered, deliveries) = mpsc::channel();
    let worker = Worker::start(EvaluationDevice::Cpu, move |evaluated| {
        delivered.send(evaluated).unwrap();
    });
    (worker, deliveries)
}

/// Connects the worker to the device and its queue as the generation.
fn connect(worker: &Worker, (device, queue): (wgpu::Device, wgpu::Queue), generation: u64) {
    worker.connect(device, queue, generation);
}

/// The staged planes of a CPU session's image.
fn staged(evaluated: Evaluated) -> Pixels {
    match evaluated.result.expect("an image").pixels {
        Some(pixels) if pixels.staged.is_some() => pixels,
        pixels => panic!("staged planes, not {pixels:?}"),
    }
}

/// The staging buffer staged planes are copied from.
fn buffer(pixels: &Pixels) -> &wgpu::Buffer {
    pixels.staged.as_ref().expect("staged").buffer()
}

fn receive(deliveries: &Receiver<Evaluated>) -> Evaluated {
    deliveries
        .recv_timeout(Duration::from_secs(10))
        .expect("the worker delivers")
}

/// Waits until `done` holds of the mailbox, checking it each time the state changes, failing
/// after a generous while.
fn wait_until(worker: &Worker, what: &str, done: impl Fn(&MailboxState) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut state = worker.mailbox.lock();
    while !done(&state) {
        let left = (deadline.checked_duration_since(Instant::now()))
            .unwrap_or_else(|| panic!("the worker never {what}"));
        state = (worker.mailbox.changed.wait_timeout(state, left))
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .0;
    }
}

/// Waits until the worker has started the last request.
fn wait_until_started(worker: &Worker) {
    wait_until(worker, "started", |state| state.request.is_none());
}

/// Waits until the worker has started the last request and finished with it, once it is past
/// evaluating: nothing it does for that request remains to be delivered.
fn wait_until_done_after_evaluating(worker: &Worker) {
    wait_until(worker, "finished", |state| {
        state.request.is_none() && !state.in_gpu_phase
    });
}

#[test]
fn a_request_replaces_any_the_worker_has_not_started() {
    let (delivered, deliveries) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    // Each delivery waits for the test, holding the worker as a long evaluation would.
    let worker = Worker::start(EvaluationDevice::Cpu, move |evaluated| {
        delivered.send(evaluated.wanted.key).unwrap();
        released.lock().unwrap().recv().unwrap();
    });

    worker.request(Some(on(1, missing("first"))));
    assert_eq!(deliveries.recv().unwrap(), missing("first"));
    worker.request(Some(on(1, missing("second"))));
    worker.request(Some(on(1, missing("third"))));
    release.send(()).unwrap();
    assert_eq!(deliveries.recv().unwrap(), missing("third"));
    release.send(()).unwrap();
    drop(worker);

    assert!(
        deliveries.try_recv().is_err(),
        "the superseded request was read"
    );
}

/// Drops the worker on another thread, returning a receiver for the drop's return.
fn drop_elsewhere(worker: Worker) -> Receiver<()> {
    let (dropped, drop_returned) = mpsc::channel();
    std::thread::spawn(move || {
        drop(worker);
        dropped.send(()).unwrap();
    });
    drop_returned
}

#[test]
fn dropping_the_worker_waits_for_the_delivery_in_progress() {
    let (started, delivering) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    let worker = Worker::start(EvaluationDevice::Cpu, move |_| {
        started.send(()).unwrap();
        released.lock().unwrap().recv().unwrap();
    });
    worker.request(Some(on(1, missing("missing"))));
    delivering.recv().unwrap();

    let drop_returned = drop_elsewhere(worker);

    assert!(
        drop_returned
            .recv_timeout(Duration::from_millis(200))
            .is_err(),
        "returned during the delivery"
    );
    release.send(()).unwrap();
    drop_returned.recv_timeout(Duration::from_secs(10)).unwrap();
}

#[cfg(unix)]
#[test]
fn dropping_the_worker_does_not_wait_for_an_evaluation() {
    use std::os::unix::fs::OpenOptionsExt;

    // Reading a named pipe waits for a writer, and then for what it writes, as a long decode
    // would.
    let pipe = directory("decode-in-progress").join("pipe.exr");
    let made = std::process::Command::new("mkfifo").arg(&pipe).status();
    assert!(made.unwrap().success());
    let (worker, deliveries) = worker();
    worker.request(Some(on(1, key(&pipe))));
    // Opening for writing without blocking fails until the worker has the pipe open for
    // reading; once it succeeds, the worker waits on the read for as long as this end is open.
    let deadline = Instant::now() + Duration::from_secs(10);
    let writer = loop {
        let opened = File::options()
            .write(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&pipe);
        match opened {
            Ok(writer) => break writer,
            Err(error) if error.raw_os_error() == Some(libc::ENXIO) => {
                assert!(
                    Instant::now() < deadline,
                    "the worker never opened the pipe"
                );
                std::thread::yield_now();
            }
            Err(error) => panic!("{error}"),
        }
    };

    let drop_returned = drop_elsewhere(worker);

    drop_returned.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(writer);
    assert!(
        deliveries.recv_timeout(Duration::from_millis(200)).is_err(),
        "delivered after the drop"
    );
}

#[test]
fn a_failure_is_delivered_with_its_key() {
    let (worker, deliveries) = worker();

    worker.request(Some(on(1, missing("missing"))));
    let evaluated = receive(&deliveries);

    assert_eq!(evaluated.wanted.key, missing("missing"));
    assert!(matches!(
        evaluated.result,
        Err(ViewerError::Eval(EvalError::Read {
            error: ReadError::Io(_),
            ..
        }))
    ));
}

#[test]
fn without_a_gpu_an_image_is_evaluated_but_not_staged() {
    let directory = directory("without-gpu");
    let path = directory.join("white.png");
    write_png(&path, 3, 2);
    let (worker, deliveries) = worker();

    worker.request(Some(on(1, key(&path))));

    let image = receive(&deliveries).result.unwrap();
    assert_eq!(image.info.data_window.width, 3);
    assert!(image.pixels.is_none());
}

#[test]
fn connecting_a_gpu_delivers_the_last_key_again_with_its_pixels() {
    let directory = directory("connect");
    let path = directory.join("white.png");
    write_png(&path, 3, 2);
    let (worker, deliveries) = worker();
    worker.request(Some(on(1, key(&path))));
    assert!(receive(&deliveries).result.unwrap().pixels.is_none());

    connect(&worker, noop_device(None), 1);

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, key(&path));
    assert!(evaluated.result.unwrap().pixels.is_some());
}

#[test]
fn asking_for_nothing_releases_everything() {
    let directory = directory("release");
    let path = directory.join("image.png");
    write_png(&path, 3, 2);
    let (worker, deliveries) = worker();
    let width = |evaluated: Evaluated| evaluated.result.unwrap().info.data_window.width;
    worker.request(Some(on(1, key(&path))));
    assert_eq!(width(receive(&deliveries)), 3);

    // Retained, a replaced file is not read again.
    write_png(&path, 5, 1);
    worker.request(Some(on(1, key(&path))));
    assert_eq!(width(receive(&deliveries)), 3);

    worker.request(None);
    wait_until_started(&worker);
    worker.request(Some(on(1, key(&path))));
    assert_eq!(width(receive(&deliveries)), 5);
}

#[test]
fn the_staging_buffer_waits_for_its_return_and_fits_the_images_staged() {
    let directory = directory("staging");
    let [small, other, tall] =
        ["small.png", "other.png", "tall.png"].map(|name| directory.join(name));
    write_png(&small, 3, 2);
    write_png(&other, 5, 1);
    // One pixel wide, which pads each row of its four planes to 256 bytes: over twice what the
    // first buffer holds.
    let (_, first_height) = FIRST_STAGING_SIZE;
    write_png(&tall, 1, 2 * (row_bytes(4096) / 256) * first_height + 1);
    let (worker, deliveries) = worker();
    connect(&worker, noop_device(None), 1);

    worker.request(Some(on(1, key(&small))));
    let first = staged(receive(&deliveries));
    let copies = first.staged.as_ref().unwrap();
    assert_eq!(copies.bytes_per_row, 256);
    // Four planes, one after another.
    let offsets: Vec<u64> = copies.copies.iter().map(|&(_, offset)| offset).collect();
    assert_eq!(offsets, [0, 512, 1024, 1536]);
    assert_eq!(
        first.planes[0].size(),
        wgpu::Extent3d {
            width: 3,
            height: 2,
            depth_or_array_layers: 1
        }
    );
    let first_buffer = buffer(&first).clone();

    // Held by the GUI, the buffer is not the worker's to write: it waits in its GPU phase,
    // which an image only leaves staged, and stages into that buffer once it returns.
    worker.request(Some(on(1, key(&other))));
    wait_until_finishing(&worker);
    assert!(
        deliveries.try_recv().is_err(),
        "staged while the buffer was held"
    );
    drop(first);
    let second = staged(receive(&deliveries));
    assert_eq!(buffer(&second), &first_buffer);
    drop(second);

    worker.request(Some(on(1, key(&tall))));
    let third = staged(receive(&deliveries));
    assert_ne!(buffer(&third), &first_buffer);
    assert!(buffer(&third).size() > first_buffer.size());
    // When it shrinks again is `StagingSizing`'s to test.
}

#[test]
fn alternating_large_and_small_images_keeps_one_staging_buffer() {
    let (first_width, first_height) = FIRST_STAGING_SIZE;
    let mut current = staging_size(first_width, first_height, 4);
    let uhd = staging_size(3840, 2160, 4);
    let eight_k = staging_size(8192, 4320, 4);
    let mut sizing = StagingSizing::default();
    let mut replacements = Vec::new();

    for _ in 0..10 {
        for needed in [uhd, eight_k] {
            if let Some(size) = sizing.replacement(current, needed, u64::MAX) {
                replacements.push(size);
                current = size;
            }
        }
    }
    assert_eq!(replacements, [eight_k + eight_k / 4]);

    // A run of small images gives the memory back.
    for _ in 0..IMAGES_BEFORE_SHRINKING {
        if let Some(size) = sizing.replacement(current, uhd, u64::MAX) {
            current = size;
        }
    }
    assert_eq!(current, staging_size(first_width, first_height, 4));
}

/// The quarter to spare stops at the GPU's largest buffer, as long as the image fits in it.
#[test]
fn a_growing_staging_buffer_stays_within_the_gpus_limit() {
    let uhd = staging_size(3840, 2160, 4);
    let eight_k = staging_size(8192, 4320, 4);
    let mut sizing = StagingSizing::default();
    assert_eq!(sizing.first(uhd, uhd), uhd);
    assert_eq!(sizing.replacement(uhd, eight_k, eight_k), Some(eight_k));
    let grown = eight_k + 1;
    assert_eq!(
        sizing.replacement(eight_k, grown, u64::MAX),
        Some(grown + grown / 4)
    );
}

#[test]
fn an_image_larger_than_the_gpu_allows_is_read_but_cannot_be_shown() {
    let directory = directory("too-large");
    let path = directory.join("wide.png");
    write_png(&path, 3, 2);
    let (worker, deliveries) = worker();
    let limits = wgpu::Limits {
        max_texture_dimension_2d: 2,
        ..wgpu::Limits::downlevel_defaults()
    };
    connect(&worker, noop_device(Some(limits)), 1);

    worker.request(Some(on(1, key(&path))));

    let error = receive(&deliveries).result.unwrap_err();
    assert!(matches!(error, ViewerError::CannotShow(_)), "{error}");
}

/// Waits until the worker has started the last request and is staging or delivering it.
fn wait_until_finishing(worker: &Worker) {
    wait_until(worker, "began staging", |state| {
        state.request.is_none() && state.in_gpu_phase
    });
}

#[test]
fn a_newer_request_for_another_node_ends_the_wait_for_the_staging_buffer() {
    let directory = directory("staging-wait");
    let [first, second] = ["first.png", "second.png"].map(|name| directory.join(name));
    write_png(&first, 3, 2);
    write_png(&second, 3, 2);
    let (worker, deliveries) = worker();
    connect(&worker, noop_device(None), 1);
    worker.request(Some(on(1, key(&first))));
    let held = staged(receive(&deliveries));

    worker.request(Some(on(1, key(&second))));
    wait_until_finishing(&worker);
    // A failure needs no buffer, so it arrives while the buffer is still held.
    worker.request(Some(on(2, missing("missing"))));

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, missing("missing"));
    assert!(evaluated.result.is_err());
    drop(held);
    wait_until_done_after_evaluating(&worker);
    assert!(
        deliveries.try_recv().is_err(),
        "the abandoned image was delivered"
    );
}

#[test]
fn a_newer_request_for_the_same_node_lets_the_started_one_be_delivered() {
    let directory = directory("same-node");
    let [first, second, third] =
        ["first.png", "second.png", "third.png"].map(|name| directory.join(name));
    for path in [&first, &second, &third] {
        write_png(path, 3, 2);
    }
    let (worker, deliveries) = worker();
    connect(&worker, noop_device(None), 1);
    worker.request(Some(on(1, key(&first))));
    let held = staged(receive(&deliveries));

    // Waiting for the buffer the test holds when the edit goes on.
    worker.request(Some(on(1, key(&second))));
    wait_until_finishing(&worker);
    worker.request(Some(on(1, key(&third))));
    drop(held);

    let staged = |evaluated: Evaluated| {
        assert!(evaluated.result.unwrap().pixels.is_some());
        evaluated.wanted.key
    };
    assert_eq!(staged(receive(&deliveries)), key(&second));
    assert_eq!(staged(receive(&deliveries)), key(&third));
}

/// Two white images in a directory of the test's own.
fn two_images(name: &str) -> [PathBuf; 2] {
    let directory = directory(name);
    let paths = ["first.png", "second.png"].map(|name| directory.join(name));
    for path in &paths {
        write_png(path, 3, 2);
    }
    paths
}

#[test]
fn a_staging_buffer_returned_after_its_device_was_replaced_is_dropped() {
    let [first, second] = two_images("replaced-return");
    let (worker, deliveries) = worker();
    let [replaced, replacement] = noop_device_pair();
    connect(&worker, replaced, 1);
    worker.request(Some(on(1, key(&first))));
    let old = staged(receive(&deliveries));

    connect(&worker, replacement, 2);
    let new = staged(receive(&deliveries));
    let returned = buffer(&new).clone();
    drop(new);
    drop(old);

    worker.request(Some(on(1, key(&second))));
    let staged = staged(receive(&deliveries));
    assert_eq!(buffer(&staged), &returned);
}

#[test]
fn an_image_finishing_for_a_replaced_device_is_staged_again_for_the_new_one() {
    let [first, second] = two_images("replaced-finishing");
    let (worker, deliveries) = worker();
    connect(&worker, noop_device(None), 1);
    worker.request(Some(on(1, key(&first))));
    let held = staged(receive(&deliveries));

    // Waiting for the buffer the test holds when the device is replaced.
    worker.request(Some(on(1, key(&second))));
    wait_until_finishing(&worker);
    connect(&worker, noop_device(None), 2);

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, key(&second));
    assert!(evaluated.result.unwrap().pixels.is_some());
    drop(held);
}

#[test]
fn a_device_lost_while_staging_fails_the_image_and_the_next_device_shows_it() {
    let [first, second] = two_images("lost-staging");
    let (worker, deliveries) = worker();
    let (device, queue) = noop_device(None);
    worker.connect(device.clone(), queue, 1);
    worker.request(Some(on(1, key(&first))));
    let held = staged(receive(&deliveries));

    device.destroy();
    // Returned unmapped, the buffer must be mapped again on the lost device.
    drop(held);
    worker.request(Some(on(1, key(&second))));
    let error = receive(&deliveries).result.unwrap_err();
    assert!(matches!(error, ViewerError::CannotShow(_)), "{error}");

    connect(&worker, noop_device(None), 2);
    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, key(&second));
    assert!(evaluated.result.unwrap().pixels.is_some());
}

#[test]
fn disconnecting_drops_the_image_finishing_and_delivers_it_again_without_pixels() {
    let [first, second] = two_images("disconnect");
    let (worker, deliveries) = worker();
    connect(&worker, noop_device(None), 1);
    worker.request(Some(on(1, key(&first))));
    let held = staged(receive(&deliveries));

    // Waiting for the buffer the test holds when the device is lost.
    worker.request(Some(on(1, key(&second))));
    wait_until_finishing(&worker);
    worker.disconnect(Instant::now());

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, key(&second));
    assert!(evaluated.result.unwrap().pixels.is_none());
    drop(held);
    wait_until_done_after_evaluating(&worker);
    assert!(
        deliveries.try_recv().is_err(),
        "the image finishing for the lost device was delivered"
    );
}

#[test]
fn every_row_of_every_plane_is_staged_where_its_row_number_says() {
    // More rows than one chunk of the pool's, and a count no chunk divides.
    let (width, height) = (3, 21);
    let plane =
        |first: usize| -> Vec<f32> { (first..first + width * height).map(|i| i as f32).collect() };
    let planes = [plane(0), plane(1000)];
    let floats_per_row = row_bytes(width as u32) as usize / 4;
    let mut out = vec![f32::NAN; floats_per_row * height * 2];

    copy_rows(
        &Pool::new(),
        &[&planes[0], &planes[1]],
        width,
        &mut out,
        floats_per_row,
    );

    for (index, plane) in planes.iter().enumerate() {
        for row in 0..height {
            let staged = &out[(index * height + row) * floats_per_row..][..width];
            assert_eq!(
                staged,
                &plane[row * width..][..width],
                "plane {index} row {row}"
            );
        }
    }
}

#[test]
fn staged_planes_keep_every_value_and_leave_the_row_padding() {
    // Two rows of one pixel; a NaN with a payload, an infinity and a value beyond f16's range.
    let payload = f32::from_bits(0x7fc0_1234);
    let plane = [payload, 1e6];
    let other = [f32::NEG_INFINITY, -0.0];
    let padding = 7.0;
    let mut out = vec![padding; 4 * 64];

    copy_rows(&Pool::new(), &[&plane, &other], 1, &mut out, 64);

    let staged: Vec<u32> = [0, 64, 128, 192].map(|start| out[start].to_bits()).to_vec();
    let expected: Vec<u32> = [payload, 1e6, f32::NEG_INFINITY, -0.0]
        .map(f32::to_bits)
        .to_vec();
    assert_eq!(staged, expected);
    let padded = (0..4).flat_map(|row| out[row * 64 + 1..(row + 1) * 64].iter());
    assert!(padded.into_iter().all(|&value| value == padding));
}

#[test]
fn a_file_is_staged_bottom_row_first() {
    // A 16 × 8 data window at (-7, 7) in a 10 × 10 display window, red each pixel's x and green
    // its row counted down from row 9, written by OpenEXR.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/tiled_zip.exr");
    let (info, planes) = read_image(&path).unwrap().into_parts();
    let window = info.data_window;
    let (width, height) = (window.width as usize, window.height as usize);
    let floats_per_row = row_bytes(window.width) as usize / 4;
    let mut out = vec![0.0; floats_per_row * height * 2];

    copy_rows(
        &Pool::new(),
        &[&planes[0][..], &planes[1][..]],
        width,
        &mut out,
        floats_per_row,
    );

    for (row, y) in (window.y..).take(height).enumerate() {
        for (column, x) in (window.x..).take(width).enumerate() {
            let red = out[row * floats_per_row + column];
            let green = out[(height + row) * floats_per_row + column];
            assert_eq!([red, green], [x as f32, (9 - y) as f32], "({x}, {y})");
        }
    }
}

/// A GPU session's worker, connected to a device on the adapter the kernel tests use.
fn gpu_worker() -> (Worker, Receiver<Evaluated>, wgpu::Device, wgpu::Queue) {
    let (delivered, deliveries) = mpsc::channel();
    let worker = Worker::start(EvaluationDevice::Gpu, move |evaluated| {
        delivered.send(evaluated).unwrap();
    });
    let (device, queue) = crate::kernels::headless::device();
    worker.connect(device.clone(), queue.clone(), 1);
    (worker, deliveries, device, queue)
}

/// The planes a GPU session's request wrote, and their device's generation.
fn evaluated_planes(evaluated: Evaluated) -> ([Arc<wgpu::Texture>; 4], u64) {
    match evaluated.result.expect("an image").pixels {
        Some(Pixels {
            planes,
            device_generation,
            staged: None,
        }) => (planes, device_generation),
        pixels => panic!("evaluated planes, not {pixels:?}"),
    }
}

/// Reader → Exposure of the path, at the stops.
fn exposed(path: &Path, stops: f64) -> ResultKey {
    use crate::evaluate::tests::{chain, exposure, key, reader};
    key(&chain([reader(path), exposure(Some(stops))]), 2)
}

/// Waits until the worker has left its GPU phase.
fn wait_until_out_of_gpu_phase(worker: &Worker) {
    wait_until(worker, "left its GPU phase", |state| !state.in_gpu_phase);
}

/// The keys retained on the connected device.
fn retained_on_gpu(worker: &Worker) -> Vec<ResultKey> {
    wait_until_out_of_gpu_phase(worker);
    worker.mailbox.retained_on_gpu()
}

#[test]
fn a_gpu_session_delivers_the_planes_its_request_wrote() {
    let path = directory("gpu-session").join("image.png");
    write_png(&path, 3, 2);
    let (worker, deliveries, device, queue) = gpu_worker();

    worker.request(Some(on(1, exposed(&path, -1.0))));

    let evaluated = receive(&deliveries);
    assert!(evaluated.timing.read.is_some());
    let (planes, generation) = evaluated_planes(evaluated);
    assert_eq!(generation, 1);
    // White, opaque, and half as bright; alpha is not exposed.
    let values = planes.each_ref().map(|plane| {
        assert_eq!((plane.width(), plane.height()), (3, 2));
        crate::kernels::headless::readback(&device, &queue, plane)
    });
    assert_eq!(values, [0.5, 0.5, 0.5, 1.0].map(|value| vec![value; 6]));
}

#[test]
fn a_gpu_session_evaluates_nothing_until_a_device_is_connected() {
    let path = directory("gpu-unconnected").join("image.png");
    write_png(&path, 3, 2);
    let (delivered, deliveries) = mpsc::channel();
    let worker = Worker::start(EvaluationDevice::Gpu, move |evaluated| {
        delivered.send(evaluated).unwrap();
    });

    worker.request(Some(on(1, exposed(&path, 1.0))));
    assert!(
        deliveries.recv_timeout(Duration::from_millis(200)).is_err(),
        "evaluated without a device"
    );
    let (device, queue) = crate::kernels::headless::device();
    worker.connect(device, queue, 1);

    let (_, generation) = evaluated_planes(receive(&deliveries));
    assert_eq!(generation, 1);
}

#[test]
fn a_replaced_device_drops_its_state_and_the_new_one_shows_the_request_again() {
    let path = directory("gpu-replaced").join("image.png");
    write_png(&path, 3, 2);
    let (worker, deliveries, _, _) = gpu_worker();
    let requested = exposed(&path, 1.0);
    worker.request(Some(on(1, requested.clone())));
    let (first, _) = evaluated_planes(receive(&deliveries));
    assert!(!retained_on_gpu(&worker).is_empty());

    let (device, queue) = crate::kernels::headless::device();
    worker.connect(device, queue, 2);

    let evaluated = receive(&deliveries);
    assert_eq!(evaluated.wanted.key, requested);
    let (second, generation) = evaluated_planes(evaluated);
    assert_eq!(generation, 2);
    assert!(!Arc::ptr_eq(&first[0], &second[0]));
    // The first device's planes are the test's alone: its state dropped with it.
    assert_eq!(Arc::strong_count(&first[0]), 1);
}

#[test]
fn asking_for_nothing_releases_everything_on_the_gpu() {
    let path = directory("gpu-release").join("image.png");
    write_png(&path, 3, 2);
    let (worker, deliveries, _, _) = gpu_worker();
    worker.request(Some(on(1, exposed(&path, 1.0))));
    drop(evaluated_planes(receive(&deliveries)));
    assert_eq!(retained_on_gpu(&worker).len(), 2);

    worker.request(None);

    // The evaluation is the phase's while it lasts, so retained keys are read outside one.
    wait_until(&worker, "released everything", |state| {
        let retained = (state.connection.as_ref())
            .and_then(|connection| connection.evaluation.as_ref())
            .map_or(0, |evaluation| evaluation.retained_keys().len());
        state.request.is_none() && !state.in_gpu_phase && retained == 0
    });
}

#[test]
fn dropping_a_gpu_session_worker_waits_for_its_gpu_phase() {
    let path = directory("gpu-drop").join("image.png");
    write_png(&path, 3, 2);
    let (started, delivering) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    let worker = Worker::start(EvaluationDevice::Gpu, move |evaluated| {
        started.send(evaluated).unwrap();
        released.lock().unwrap().recv().unwrap();
    });
    let (device, queue) = crate::kernels::headless::device();
    worker.connect(device, queue, 1);
    worker.request(Some(on(1, exposed(&path, 1.0))));
    let evaluated = delivering.recv().unwrap();

    let drop_returned = drop_elsewhere(worker);
    assert!(
        drop_returned
            .recv_timeout(Duration::from_millis(200))
            .is_err(),
        "the drop returned during the GPU phase"
    );
    release.send(()).unwrap();
    drop_returned
        .recv_timeout(Duration::from_secs(10))
        .expect("the drop returns once the phase ends");

    // The worker made the planes in its phase and dropped its own with the connection, on the
    // dropping thread.
    let (planes, _) = evaluated_planes(evaluated);
    assert_eq!(Arc::strong_count(&planes[0]), 1);
}

#[test]
fn moving_a_gpu_session_to_the_cpu_drops_its_gpu_state_and_stages_the_request_again() {
    let path = directory("gpu-to-cpu").join("image.png");
    write_png(&path, 3, 2);
    let (worker, deliveries, _, _) = gpu_worker();
    worker.request(Some(on(1, exposed(&path, 1.0))));
    evaluated_planes(receive(&deliveries));
    assert!(!retained_on_gpu(&worker).is_empty());

    worker.evaluate_on(EvaluationDevice::Cpu);

    let again = receive(&deliveries);
    assert_eq!(again.wanted.key, exposed(&path, 1.0));
    staged(again);
    assert!(retained_on_gpu(&worker).is_empty());
    worker.evaluate_on(EvaluationDevice::Gpu);
    evaluated_planes(receive(&deliveries));
}

#[test]
fn a_loss_counts_as_during_evaluation_only_until_the_gpu_completed_its_work() {
    let path = directory("gpu-loss-during").join("image.png");
    write_png(&path, 3, 2);
    for lost_during in [true, false] {
        let (worker, deliveries, device, queue) = gpu_worker();
        worker.request(Some(on(1, exposed(&path, 1.0))));
        evaluated_planes(receive(&deliveries));
        wait_until_out_of_gpu_phase(&worker);
        let state = worker.mailbox.lock();
        let submission = state.last_submission.as_ref().unwrap();
        let completed = submission.completed.expect("seen complete");
        let lost_at = if lost_during {
            submission.at
        } else {
            completed + Duration::from_millis(1)
        };
        drop(state);

        assert_eq!(worker.disconnect(lost_at), lost_during);
        // Without a submission on the device connected now, a loss is not during evaluation.
        connect(&worker, (device, queue), 2);
        assert!(!worker.disconnect(Instant::now()));
    }
}
