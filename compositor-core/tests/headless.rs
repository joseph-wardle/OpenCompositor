//! The compositor without a window: a document built by the same edits the interface makes,
//! evaluated to pixels from files on disk. What each format reads as, and how edits undo, are
//! the readers' and the session's own tests.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use open_ng::Point;
use opencompositor_core::{
    Channel, EvalError, ExposureVersion, NodeId, NodeKind, ReadError, Session, evaluate,
};

/// A directory of this test's own, emptied first.
fn directory(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A 1 × 1 opaque grey PNG of one 8-bit code, which reads exactly at 0 and 255.
fn write_png(path: &Path, code: u8) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), 1, 1);
    encoder.set_color(png::ColorType::Grayscale);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&[code]).unwrap();
    writer.finish().unwrap();
}

fn reader(session: &mut Session, path: PathBuf) -> NodeId {
    let changes = session.create_readers(vec![path], Point { x: 0.0, y: 0.0 });
    *changes.nodes.first().unwrap()
}

#[test]
fn a_reader_evaluates_to_the_file_it_points_at() {
    let directory = directory("reader_pixels");
    let (white, black) = (directory.join("white.png"), directory.join("black.png"));
    write_png(&white, 255);
    write_png(&black, 0);
    let mut session = Session::new();

    let id = reader(&mut session, white);
    let image = evaluate(session.document(), id).unwrap();
    assert_eq!(Channel::ALL.map(|c| image.value(c, 0, 0)), [1.0; 4]);

    session.set_reader_path(id, black).unwrap();
    let image = evaluate(session.document(), id).unwrap();
    assert_eq!(
        Channel::ALL.map(|c| image.value(c, 0, 0)),
        [0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn evaluation_reports_what_went_wrong() {
    let missing = directory("reader_errors").join("missing.exr");
    let mut session = Session::new();

    let id = reader(&mut session, missing);
    let error = evaluate(session.document(), id).unwrap_err();
    assert!(
        matches!(&error, EvalError::Read { error: ReadError::Io(io), .. } if io.kind() == std::io::ErrorKind::NotFound),
        "{error:?}"
    );
    assert!(error.to_string().contains("missing.exr"), "{error}");

    session.delete_nodes(&[id]).unwrap();
    assert!(
        matches!(evaluate(session.document(), id), Err(EvalError::MissingNode(missing)) if missing == id)
    );
}

#[test]
fn an_exposure_node_reports_its_missing_input_then_that_it_is_not_computed() {
    let mut session = Session::new();
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        Point { x: 0.0, y: 0.0 },
    );
    let id = *changes.nodes.first().unwrap();

    let error = evaluate(session.document(), id).unwrap_err();
    assert!(
        matches!(error, EvalError::NoInput { node, input: "source" } if node == id),
        "{error:?}"
    );

    let source = reader(&mut session, directory("reader_errors").join("missing.exr"));
    session.connect(id, "source", source).unwrap();
    // The Exposure does not read its input, so its source's failure does not reach it yet.
    let error = evaluate(session.document(), id).unwrap_err();
    assert!(
        matches!(error, EvalError::NotImplemented(node) if node == id),
        "{error:?}"
    );
}
