//! The compositor without a window: a document built by the same edits the interface makes,
//! evaluated to pixels.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use exr::prelude::{
    AnyChannel, AnyChannels, Encoding, FlatSamples, ImageAttributes, IntegerBounds, Layer,
    LayerAttributes, Vec2, WritableImage,
};
use open_ng::Point;
use opencompositor_core::{EvalError, NodeId, ReadError, Session, Window, evaluate};

/// A directory of this test's own, emptied first.
fn directory(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A 2 × 1 EXR stored at (-1, -1) in a 4 × 4 frame: red, then green at half coverage.
fn write_exr(path: &Path) {
    let attributes = LayerAttributes {
        layer_position: Vec2(-1, -1),
        ..LayerAttributes::default()
    };
    let channel =
        |name, samples: [f32; 2]| AnyChannel::new(name, FlatSamples::F32(samples.to_vec()));
    let channels = [
        channel("R", [1.0, 0.0]),
        channel("G", [0.0, 0.5]),
        channel("B", [0.0, 0.0]),
        channel("A", [1.0, 0.5]),
    ];
    let layer = Layer::new(
        (2, 1),
        attributes,
        Encoding::SMALL_LOSSLESS,
        AnyChannels::sort(channels.into_iter().collect()),
    );
    let display = ImageAttributes::new(IntegerBounds::new((0, 0), (4, 4)));
    exr::image::Image::from_layers(display, vec![layer])
        .write()
        .to_file(path)
        .unwrap();
}

/// A 1 × 2 PNG: opaque white above transparent black.
fn write_png(path: &Path) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), 1, 2);
    encoder.set_color(png::ColorType::Rgba);
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&[255, 255, 255, 255, 0, 0, 0, 0])
        .unwrap();
    writer.finish().unwrap();
}

fn reader(session: &mut Session, path: PathBuf) -> NodeId {
    let changes = session.create_readers(vec![path], Point { x: 0.0, y: 0.0 });
    *changes.nodes.first().unwrap()
}

#[test]
fn a_reader_built_by_editing_evaluates_to_its_files_pixels() {
    let directory = directory("reader_pixels");
    let (plate, matte) = (directory.join("plate.exr"), directory.join("matte.png"));
    write_exr(&plate);
    write_png(&matte);
    let mut session = Session::new();

    let id = reader(&mut session, plate);
    let image = evaluate(session.document(), id).unwrap();
    assert_eq!(
        image.display_window(),
        Window {
            x: 0,
            y: 0,
            width: 4,
            height: 4
        }
    );
    assert_eq!(
        image.data_window(),
        Window {
            x: -1,
            y: -1,
            width: 2,
            height: 1
        }
    );
    assert_eq!(image.pixel(-1, -1), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(image.pixel(0, -1), [0.0, 0.5, 0.0, 0.5]);
    assert_eq!(image.pixel(1, 1), [0.0; 4]);

    session.set_reader_path(id, matte).unwrap();
    let image = evaluate(session.document(), id).unwrap();
    assert_eq!(
        image.data_window(),
        Window {
            x: 0,
            y: 0,
            width: 1,
            height: 2
        }
    );
    assert_eq!(image.pixel(0, 0), [1.0; 4]);
    assert_eq!(image.pixel(0, 1), [0.0; 4]);

    session.undo();
    let image = evaluate(session.document(), id).unwrap();
    assert_eq!(image.pixel(-1, -1), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn evaluation_reports_what_went_wrong() {
    let directory = directory("reader_errors");
    let text = directory.join("notes.exr");
    std::fs::write(&text, "not an image, whatever its name").unwrap();
    let missing = directory.join("missing.exr");
    let mut session = Session::new();

    let id = reader(&mut session, text.clone());
    match evaluate(session.document(), id) {
        Err(EvalError::Read {
            path,
            error: ReadError::UnknownFormat,
        }) => assert_eq!(path, text),
        other => panic!("expected an unknown format, got {other:?}"),
    }

    session.set_reader_path(id, missing).unwrap();
    let error = evaluate(session.document(), id).unwrap_err();
    assert!(
        matches!(&error, EvalError::Read { error: ReadError::Io(io), .. } if io.kind() == std::io::ErrorKind::NotFound),
        "{error:?}"
    );
    assert!(error.to_string().contains("missing.exr"), "{error}");

    session.undo();
    session.undo();
    assert!(
        matches!(evaluate(session.document(), id), Err(EvalError::MissingNode(missing)) if missing == id)
    );
}
