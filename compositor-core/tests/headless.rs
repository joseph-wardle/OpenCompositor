//! The compositor without a window: a document built by the same edits the interface makes,
//! evaluated to pixels from files on disk. What each format reads as, and how edits undo, are
//! the readers' and the session's own tests.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use open_ng::Point;
use opencompositor_core::{
    Channel, EvalError, Evaluator, ExposureVersion, Image, NodeId, NodeKind, ReadError, Session,
    Value, result_key,
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

/// The node's image as the session's document and reloads describe it.
fn evaluate(evaluator: &mut Evaluator, session: &Session, id: NodeId) -> Result<Image, EvalError> {
    let key = result_key(session.document(), session.reloads(), id).expect("the node exists");
    evaluator
        .evaluate(&key, || false)
        .expect("nothing stops it")
}

fn pixel(image: &Image) -> [f32; 4] {
    Channel::ALL.map(|c| image.value(c, 0, 0))
}

fn create_exposure(session: &mut Session) -> NodeId {
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        Point { x: 0.0, y: 0.0 },
    );
    *changes.nodes.first().unwrap()
}

#[test]
fn a_reader_evaluates_to_the_file_it_points_at() {
    let directory = directory("reader_pixels");
    let (white, black) = (directory.join("white.png"), directory.join("black.png"));
    write_png(&white, 255);
    write_png(&black, 0);
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();

    let id = reader(&mut session, white);
    let image = evaluate(&mut evaluator, &session, id).unwrap();
    assert_eq!(pixel(&image), [1.0; 4]);

    session.set_reader_path(id, black).unwrap();
    let image = evaluate(&mut evaluator, &session, id).unwrap();
    assert_eq!(pixel(&image), [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn evaluation_reports_what_went_wrong_and_where() {
    let missing = directory("reader_errors").join("missing.exr");
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let id = reader(&mut session, missing);
    let exposure = create_exposure(&mut session);
    session.connect(exposure, "source", id).unwrap();

    let error = evaluate(&mut evaluator, &session, exposure).unwrap_err();
    assert!(
        matches!(&error, EvalError::Read { error: ReadError::Io(io), .. } if io.kind() == std::io::ErrorKind::NotFound),
        "{error:?}"
    );
    assert_eq!(
        Some(error.origin()),
        result_key(session.document(), session.reloads(), id).as_ref()
    );
    assert!(error.to_string().contains("missing.exr"), "{error}");

    session.delete_nodes(&[id]).unwrap();
    assert_eq!(result_key(session.document(), session.reloads(), id), None);
}

#[test]
fn an_exposure_node_scales_its_source_by_its_stops() {
    let white = directory("exposure").join("white.png");
    write_png(&white, 255);
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let exposure = create_exposure(&mut session);

    let image = evaluate(&mut evaluator, &session, exposure).unwrap();
    assert_eq!(image.data_window().width, 0);

    let source = reader(&mut session, white);
    session.connect(exposure, "source", source).unwrap();
    session.begin_edit(exposure, "Stops").unwrap();
    for (stops, expected) in [(1.0, 2.0), (-2.0, 0.25)] {
        session.edit_value("stops", Value::Float(stops)).unwrap();
        let image = evaluate(&mut evaluator, &session, exposure).unwrap();
        assert_eq!(pixel(&image), [expected, expected, expected, 1.0]);
    }
    session.finish_edit();
}
