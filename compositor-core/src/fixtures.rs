//! Files the unit tests read, written by each test for itself.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// A directory of the calling test's own, emptied first. Tests run in parallel, so it is named
/// after the test, which libtest names each test's thread for.
pub(crate) fn directory(name: &str) -> PathBuf {
    let test = std::thread::current()
        .name()
        .expect("a test's thread is named")
        .replace("::", "-");
    let directory = std::env::temp_dir()
        .join(format!("opencompositor-{}", std::process::id()))
        .join(test)
        .join(name);
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A PNG of these 8-bit samples, replacing any file at `path`.
pub(crate) fn write_png(path: &Path, width: u32, height: u32, color: png::ColorType, data: &[u8]) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), width, height);
    encoder.set_color(color);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
    writer.finish().unwrap();
}

/// A single-part EXR of 32-bit float channels over a data window at `origin` and the display
/// window `display`, replacing any file at `path`. Each channel is its name and its values over
/// the data window, top row first as EXR stores them.
pub(crate) fn write_exr(
    path: &Path,
    display: ((i32, i32), (usize, usize)),
    origin: (i32, i32),
    size: (usize, usize),
    channels: Vec<(&str, Vec<f32>)>,
) {
    use exr::prelude::{
        AnyChannel, AnyChannels, Encoding, FlatSamples, Image, ImageAttributes, IntegerBounds,
        Layer, LayerAttributes, WritableImage,
    };
    let channels = channels
        .into_iter()
        .map(|(name, values)| AnyChannel::new(name, FlatSamples::F32(values)))
        .collect();
    let layer = Layer::new(
        size,
        LayerAttributes {
            layer_position: exr::math::Vec2(origin.0, origin.1),
            ..LayerAttributes::default()
        },
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(channels),
    );
    let attributes = ImageAttributes::new(IntegerBounds::new(display.0, display.1));
    Image::from_layers(attributes, vec![layer])
        .write()
        .to_file(path)
        .expect("an EXR writes");
}
