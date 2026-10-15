use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use super::*;
use crate::gpu::noop_device;

fn request(name: &str) -> LoadRequest {
    LoadRequest {
        node: NodeId(1),
        path: PathBuf::from(format!("/nonexistent/{name}.exr")),
        reload: 0,
    }
}

fn request_for(path: &Path) -> LoadRequest {
    LoadRequest {
        node: NodeId(1),
        path: path.to_owned(),
        reload: 0,
    }
}

/// A directory of this test's own, emptied first.
fn directory(name: &str) -> PathBuf {
    let directory = std::env::temp_dir()
        .join(format!("opencompositor-loader-{}", std::process::id()))
        .join(name);
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A white, opaque PNG.
fn write_png(path: &Path, width: u32, height: u32) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), width, height);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&vec![255; width as usize * height as usize * 4])
        .unwrap();
    writer.finish().unwrap();
}

fn loader() -> (Loader, Receiver<Loaded>) {
    let (delivered, deliveries) = mpsc::channel();
    let loader = Loader::start(move |loaded| delivered.send(loaded).unwrap());
    (loader, deliveries)
}

fn receive(deliveries: &Receiver<Loaded>) -> Loaded {
    deliveries
        .recv_timeout(Duration::from_secs(10))
        .expect("the loader delivers")
}

#[test]
fn a_request_replaces_any_the_worker_has_not_started() {
    let (delivered, deliveries) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    // Each delivery waits for the test, holding the worker as a long read would.
    let loader = Loader::start(move |loaded| {
        delivered.send(loaded.request).unwrap();
        released.lock().unwrap().recv().unwrap();
    });

    loader.request(request("first"));
    assert_eq!(deliveries.recv().unwrap(), request("first"));
    loader.request(request("second"));
    loader.request(request("third"));
    release.send(()).unwrap();
    assert_eq!(deliveries.recv().unwrap(), request("third"));
    release.send(()).unwrap();
    drop(loader);

    assert!(
        deliveries.try_recv().is_err(),
        "the superseded request was read"
    );
}

/// Drops the loader on another thread, returning a receiver for the drop's return.
fn drop_elsewhere(loader: Loader) -> Receiver<()> {
    let (dropped, drop_returned) = mpsc::channel();
    std::thread::spawn(move || {
        drop(loader);
        dropped.send(()).unwrap();
    });
    drop_returned
}

#[test]
fn dropping_the_loader_waits_for_the_delivery_in_progress() {
    let (started, delivering) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    let loader = Loader::start(move |_| {
        started.send(()).unwrap();
        released.lock().unwrap().recv().unwrap();
    });
    loader.request(request("missing"));
    delivering.recv().unwrap();

    let drop_returned = drop_elsewhere(loader);

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
fn dropping_the_loader_does_not_wait_for_a_decode() {
    use std::os::unix::fs::OpenOptionsExt;

    // Reading a named pipe waits for a writer, and then for what it writes, as a long decode
    // would.
    let pipe = directory("decode-in-progress").join("pipe.exr");
    let made = std::process::Command::new("mkfifo").arg(&pipe).status();
    assert!(made.unwrap().success());
    let (loader, deliveries) = loader();
    loader.request(request_for(&pipe));
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

    let drop_returned = drop_elsewhere(loader);

    drop_returned.recv_timeout(Duration::from_secs(1)).unwrap();
    drop(writer);
    assert!(
        deliveries.recv_timeout(Duration::from_millis(200)).is_err(),
        "delivered after the drop"
    );
}

#[test]
fn a_failed_read_is_delivered_with_its_request() {
    let (loader, deliveries) = loader();

    loader.request(request("missing"));
    let loaded = receive(&deliveries);

    assert_eq!(loaded.request, request("missing"));
    assert!(matches!(
        loaded.result,
        Err(LoadError::Read(ReadError::Io(_)))
    ));
}

#[test]
fn without_a_gpu_an_image_is_read_but_not_staged() {
    let directory = directory("without-gpu");
    let path = directory.join("white.png");
    write_png(&path, 3, 2);
    let (loader, deliveries) = loader();

    loader.request(request_for(&path));

    let decoded = receive(&deliveries).result.unwrap();
    assert_eq!(decoded.info.data_window.width, 3);
    assert!(decoded.pixels.is_none());
}

#[test]
fn the_staging_buffer_waits_for_its_return_and_fits_the_images_staged() {
    let directory = directory("staging");
    let [small, other, tall] =
        ["small.png", "other.png", "tall.png"].map(|name| directory.join(name));
    write_png(&small, 3, 2);
    write_png(&other, 5, 1);
    // One pixel wide, which pads each row to 256 bytes: over twice what the first buffer holds.
    let (_, first_height) = FIRST_STAGING_SIZE;
    write_png(&tall, 1, 2 * (row_bytes(4096) / 256) * first_height + 1);
    let (loader, deliveries) = loader();
    loader.connect(noop_device(None));

    loader.request(request_for(&small));
    let first = receive(&deliveries).result.unwrap().pixels.unwrap();
    assert_eq!(first.bytes_per_row, 256);
    assert_eq!(
        first.texture.size(),
        wgpu::Extent3d {
            width: 3,
            height: 2,
            depth_or_array_layers: 1
        }
    );
    let buffer = first.buffer().clone();

    // Held by the GUI, the buffer is not the worker's to write.
    loader.request(request_for(&other));
    assert!(
        deliveries.recv_timeout(Duration::from_millis(200)).is_err(),
        "staged while the buffer was held"
    );
    drop(first);
    let second = receive(&deliveries).result.unwrap().pixels.unwrap();
    assert_eq!(second.buffer(), &buffer);
    drop(second);

    loader.request(request_for(&tall));
    let third = receive(&deliveries).result.unwrap().pixels.unwrap();
    assert_ne!(third.buffer(), &buffer);
    assert!(third.buffer().size() > buffer.size());
    let grown = third.buffer().clone();
    drop(third);

    // Under half the grown buffer, which goes back to the first size after enough such loads.
    for _ in 1..LOADS_BEFORE_SHRINKING {
        loader.request(request_for(&small));
        let kept = receive(&deliveries).result.unwrap().pixels.unwrap();
        assert_eq!(kept.buffer(), &grown);
    }
    loader.request(request_for(&small));
    let shrunk = receive(&deliveries).result.unwrap().pixels.unwrap();
    assert_eq!(shrunk.buffer().size(), buffer.size());
}

#[test]
fn alternating_large_and_small_images_keeps_one_staging_buffer() {
    let (first_width, first_height) = FIRST_STAGING_SIZE;
    let mut current = staging_size(first_width, first_height);
    let uhd = staging_size(3840, 2160);
    let eight_k = staging_size(8192, 4320);
    let mut sizing = StagingSizing::default();
    let mut replacements = Vec::new();

    for _ in 0..10 {
        for needed in [uhd, eight_k] {
            if let Some(size) = sizing.replacement(current, needed) {
                replacements.push(size);
                current = size;
            }
        }
    }
    assert_eq!(replacements, [eight_k]);

    // A run of small images gives the memory back.
    for _ in 0..LOADS_BEFORE_SHRINKING {
        if let Some(size) = sizing.replacement(current, uhd) {
            current = size;
        }
    }
    assert_eq!(current, staging_size(first_width, first_height));
}

#[test]
fn an_image_larger_than_the_gpu_allows_is_read_but_cannot_be_shown() {
    let directory = directory("too-large");
    let path = directory.join("wide.png");
    write_png(&path, 3, 2);
    let (loader, deliveries) = loader();
    loader.connect(noop_device(Some(wgpu::Limits {
        max_texture_dimension_2d: 2,
        ..wgpu::Limits::downlevel_defaults()
    })));

    loader.request(request_for(&path));

    let error = receive(&deliveries).result.unwrap_err();
    assert!(matches!(error, LoadError::CannotShow(_)), "{error}");
}

#[test]
fn staged_rows_are_half_floats_padded_to_the_copy_alignment() {
    // Two rows of one pixel each; the second has a NaN and a value beyond f16's range.
    let pixels = [0.5, 1.0, -2.0, 0.25, f32::NAN, 1e6, 0.0, 1.0];
    let padding = f16::from_f32(7.0);
    let mut out = vec![padding; 2 * 128];

    convert_rows(&pixels, 1, &mut out, 128);

    let row = |index: usize| out[index * 128..index * 128 + 4].to_vec();
    assert_eq!(row(0), [0.5, 1.0, -2.0, 0.25].map(f16::from_f32));
    let second = row(1);
    assert!(second[0].is_nan());
    assert_eq!(second[1], f16::INFINITY);
    assert_eq!(second[2..], [0.0, 1.0].map(f16::from_f32));
    assert!(out[4..128].iter().all(|&half| half == padding));
}
