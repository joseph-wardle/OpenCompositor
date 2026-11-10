//! The compositor without a window: a document built by the same edits the interface makes,
//! evaluated to pixels from files on disk.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use open_ng::Point;
use opencompositor_core::{
    Channel, EvalError, Evaluator, Image, NodeId, ReadError, Reading, Scan, Session, Value,
    readings, result_key,
};

/// The frames the node's Readers read at a frame that `which` picks out, as held or missing.
fn frames_read(
    session: &Session,
    node: NodeId,
    frame: f64,
    which: impl Fn(&Reading) -> Option<i64>,
) -> Vec<(NodeId, i64)> {
    (readings(session.document(), session.files(), node, frame).into_iter())
        .filter_map(|(id, reading)| Some((id, which(&reading)?)))
        .collect()
}

/// A directory of this test's own, emptied first.
fn directory(name: &str) -> PathBuf {
    let directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// A 1 × 1 opaque grey PNG of one 8-bit code, which reads exactly at 0 and 255.
fn write_png(path: &Path, code: u8) {
    write_grey(path, 1, code);
}

/// A `width` × 1 opaque grey PNG of one 8-bit code.
fn write_grey(path: &Path, width: u32, code: u8) {
    let mut encoder = png::Encoder::new(BufWriter::new(File::create(path).unwrap()), width, 1);
    encoder.set_color(png::ColorType::Grayscale);
    let mut writer = encoder.write_header().unwrap();
    writer
        .write_image_data(&vec![code; width as usize])
        .unwrap();
    writer.finish().unwrap();
}

fn reader(session: &mut Session, path: PathBuf) -> NodeId {
    let changes = session.create_readers(vec![path], Point { x: 0.0, y: 0.0 }, None);
    *changes.nodes.first().unwrap()
}

/// The node's image as the session's document and reloads describe it.
fn evaluate(evaluator: &mut Evaluator, session: &Session, id: NodeId) -> Result<Image, EvalError> {
    let key = session.result_key(id).expect("the node exists");
    evaluator
        .evaluate(&key, || false)
        .expect("nothing stops it")
}

fn pixel(image: &Image) -> [f32; 4] {
    Channel::ALL.map(|c| image.value(c, 0, 0))
}

fn create_exposure(session: &mut Session) -> NodeId {
    let changes = session
        .create_node("exposure", Point { x: 0.0, y: 0.0 }, None)
        .unwrap();
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

    session
        .set_value(id, "file", Value::File(black.into()))
        .unwrap();
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
    assert_eq!(Some(error.origin()), session.result_key(id).as_ref());
    assert!(error.to_string().contains("missing.exr"), "{error}");

    session.delete_nodes(&[id]).unwrap();
    assert_eq!(session.result_key(id), None);
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

#[test]
fn a_still_image_is_read_once_for_every_frame_until_something_upstream_changes() {
    let white = directory("still").join("white.png");
    write_png(&white, 255);
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let source = reader(&mut session, white.clone());
    let exposure = create_exposure(&mut session);
    session.connect(exposure, "source", source).unwrap();
    evaluate(&mut evaluator, &session, exposure).unwrap();

    // Gone from disk, the file is not read again at other frames: their keys are the same.
    std::fs::remove_file(&white).unwrap();
    for frame in [2.0, -40.0, 1001.0] {
        session.set_frame(frame).unwrap();
        let image = evaluate(&mut evaluator, &session, exposure).unwrap();
        assert_eq!(pixel(&image), [1.0; 4]);
    }

    // An edit downstream of the Reader computes again from what it read.
    session
        .set_value(exposure, "stops", Value::Float(1.0))
        .unwrap();
    let image = evaluate(&mut evaluator, &session, exposure).unwrap();
    assert_eq!(pixel(&image), [2.0, 2.0, 2.0, 1.0]);

    // Reloading changes the Reader's key at every frame, so it reads the file again.
    assert!(session.reload(&[source]).sequences.is_empty());
    let error = evaluate(&mut evaluator, &session, exposure).unwrap_err();
    assert!(matches!(error, EvalError::Read { .. }), "{error:?}");
}

/// Writes a frame of a sequence as an image whose width tells which frame it is.
fn write_frame(directory: &Path, frame: i64) {
    write_frame_wide(directory, frame, width(frame));
}

fn write_frame_wide(directory: &Path, frame: i64, width: u32) {
    write_grey(&directory.join(format!("plate.{frame:04}.png")), width, 0);
}

fn width(frame: i64) -> u32 {
    (frame + 10) as u32
}

/// Records a scan of every sequence the session reads that has none, as the app does off the GUI
/// thread.
fn scan_unscanned(session: &mut Session) {
    for sequence in session.unscanned() {
        let scan = Scan::of(&sequence).unwrap();
        session.record_scan(&sequence, scan);
    }
}

/// The width of the node's image at a frame, which names the frame of a sequence read.
fn width_at(evaluator: &mut Evaluator, session: &mut Session, id: NodeId, frame: f64) -> u32 {
    session.set_frame(frame).unwrap();
    evaluate(evaluator, session, id)
        .unwrap()
        .data_window()
        .width
}

fn key_at(session: &mut Session, id: NodeId, frame: f64) -> opencompositor_core::ResultKey {
    session.set_frame(frame).unwrap();
    result_key(session.document(), session.files(), id, frame).unwrap()
}

#[test]
fn a_sequence_reads_the_frame_its_offset_maps_to_and_holds_its_ends() {
    let directory = directory("sequence");
    for frame in [-1, 0, 1, 2, 4] {
        write_frame(&directory, frame);
    }
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let id = reader(&mut session, directory.join("plate.%04d.png"));

    // Until scanned, each frame reads its own file, with nothing held.
    session.set_frame(10.0).unwrap();
    let error = evaluate(&mut evaluator, &session, id).unwrap_err();
    assert!(error.to_string().contains("plate.0010.png"), "{error}");
    assert_eq!(session.unscanned().len(), 1);
    scan_unscanned(&mut session);
    assert!(session.unscanned().is_empty());

    for (frame, read) in [
        (-5.0, -1),
        (-1.0, -1),
        (0.0, 0),
        (2.0, 2),
        (4.0, 4),
        (10.0, 4),
    ] {
        let shown = width_at(&mut evaluator, &mut session, id, frame);
        assert_eq!(shown, width(read), "{frame}");
        let holds = frames_read(&session, id, frame, |reading| reading.held());
        let expected = if frame == read as f64 {
            vec![]
        } else {
            vec![(id, read)]
        };
        assert_eq!(holds, expected, "{frame}");
    }
    // A held frame is the end's image.
    assert_eq!(
        key_at(&mut session, id, 10.0),
        key_at(&mut session, id, 4.0)
    );

    // Inside the range, a frame the scan did not find is that frame's error.
    session.set_frame(3.0).unwrap();
    let missed = frames_read(&session, id, 3.0, |reading| reading.missing());
    assert_eq!(missed, [(id, 3)]);
    let error = evaluate(&mut evaluator, &session, id).unwrap_err();
    assert!(
        matches!(&error, EvalError::Read { error: ReadError::Io(io), .. } if io.kind() == std::io::ErrorKind::NotFound),
        "{error:?}"
    );
    assert!(error.to_string().contains("plate.0003.png"), "{error}");

    // File frame 1 shows at frame 11 with an offset of 10.
    session.set_value(id, "offset", Value::Float(10.0)).unwrap();
    assert_eq!(width_at(&mut evaluator, &mut session, id, 11.0), width(1));
    assert_eq!(width_at(&mut evaluator, &mut session, id, 0.0), width(-1));
}

#[test]
fn first_and_last_trim_the_range_until_reset_to_follow_the_scan() {
    let directory = directory("trimmed");
    for frame in 1..=5 {
        write_frame(&directory, frame);
    }
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let id = reader(&mut session, directory.join("plate.####.png"));
    scan_unscanned(&mut session);

    session.set_value(id, "first", Value::Float(2.0)).unwrap();
    session.set_value(id, "last", Value::Float(4.0)).unwrap();
    assert_eq!(width_at(&mut evaluator, &mut session, id, 1.0), width(2));
    assert_eq!(width_at(&mut evaluator, &mut session, id, 5.0), width(4));

    session.reset_value(id, "last").unwrap();
    assert_eq!(width_at(&mut evaluator, &mut session, id, 5.0), width(5));

    // Set before the frames found, First makes the frames between missing.
    session.set_value(id, "first", Value::Float(0.0)).unwrap();
    let missed = frames_read(&session, id, -3.0, |reading| reading.missing());
    assert_eq!(missed, [(id, 0)]);
}

#[test]
fn reload_files_rescans_and_reads_again_only_the_frames_that_changed() {
    let directory = directory("reloaded");
    for frame in [1, 2, 5] {
        write_frame(&directory, frame);
    }
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let id = reader(&mut session, directory.join("plate.####.png"));
    scan_unscanned(&mut session);
    let before = [1.0, 2.0, 5.0].map(|frame| key_at(&mut session, id, frame));
    assert_eq!(width_at(&mut evaluator, &mut session, id, 1.0), width(1));

    write_frame_wide(&directory, 1, 40);
    std::fs::remove_file(directory.join("plate.0002.png")).unwrap();
    write_frame(&directory, 3);
    // Unseen until reloaded: keys never look at the disk.
    assert_eq!(key_at(&mut session, id, 1.0), before[0]);
    assert_eq!(width_at(&mut evaluator, &mut session, id, 1.0), width(1));

    let reload = session.reload(&[id]);
    assert!(reload.stills.is_empty());
    for sequence in reload.sequences {
        session.record_scan(&sequence, Scan::of(&sequence).unwrap());
    }

    let after = [1.0, 2.0, 5.0].map(|frame| key_at(&mut session, id, frame));
    assert_ne!(after[0], before[0]);
    assert_ne!(after[1], before[1]);
    assert_eq!(after[2], before[2]);
    assert_eq!(width_at(&mut evaluator, &mut session, id, 1.0), 40);
    assert_eq!(width_at(&mut evaluator, &mut session, id, 3.0), width(3));
    let missed = frames_read(&session, id, 2.0, |reading| reading.missing());
    assert_eq!(missed, [(id, 2)]);
}

#[test]
fn a_read_stopped_part_way_keeps_nothing_and_reads_whole_when_asked_again() {
    use exr::prelude::{
        Encoding, Image as ExrImage, Layer, LayerAttributes, SpecificChannels, WritableImage, f16,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    let path = directory("stopped_read").join("lines.exr");
    // A chunk per line.
    let lines = Layer::new(
        (8, 64),
        LayerAttributes::default(),
        Encoding::UNCOMPRESSED,
        SpecificChannels::rgba(|_| (f16::ONE, f16::ONE, f16::ONE, f16::ONE)),
    );
    ExrImage::from_layer(lines).write().to_file(&path).unwrap();
    let mut session = Session::new();
    let mut evaluator = Evaluator::new();
    let id = reader(&mut session, path);
    let key = result_key(session.document(), session.files(), id, 1.0).unwrap();

    // Asked once before the node, then before each chunk.
    let asked = AtomicUsize::new(0);
    let stopped = evaluator.evaluate(&key, || asked.fetch_add(1, Ordering::Relaxed) >= 4);
    assert!(stopped.is_none());
    assert!(asked.load(Ordering::Relaxed) < 64);

    let asked = AtomicUsize::new(0);
    let image = evaluator
        .evaluate(&key, || {
            asked.fetch_add(1, Ordering::Relaxed);
            false
        })
        .unwrap()
        .unwrap();
    assert_eq!(pixel(&image), [1.0; 4]);
    assert!(asked.load(Ordering::Relaxed) > 64, "read again, not kept");
}
