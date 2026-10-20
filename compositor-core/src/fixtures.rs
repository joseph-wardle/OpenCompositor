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
