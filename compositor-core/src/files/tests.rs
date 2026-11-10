use super::*;
use crate::fixtures::directory;

fn path(path: &str) -> Arc<Path> {
    Path::new(path).into()
}

fn pattern(written: &str) -> Pattern {
    Pattern::parse(&path(written)).expect("a pattern")
}

#[test]
fn a_pattern_writes_each_frame_zero_padded_as_printf_does() {
    let plate = pattern("shots/plate.####.exr");
    let written = [1, -1, 0, 12345, -12345].map(|frame| plate.frame_path(frame));
    let expected = [
        "shots/plate.0001.exr",
        "shots/plate.-001.exr",
        "shots/plate.0000.exr",
        "shots/plate.12345.exr",
        "shots/plate.-12345.exr",
    ];
    assert_eq!(written, expected.map(PathBuf::from));
    for frame in [1, -1, 0, 12345, -12345] {
        assert_eq!(plate.frame_of(&plate.frame_path(frame)), Some(frame));
    }
    for other in [
        "plate.00001.exr",
        "plate.+001.exr",
        "plate.1.exr",
        "plate.0001.exr.tmp",
        "plate.####.exr",
        "take.0001.exr",
    ] {
        assert_eq!(plate.frame_of(Path::new(other)), None, "{other}");
    }

    // The last run is the frame; the rest of the name is kept as written.
    let last = pattern("v##/a##_b###.exr");
    assert_eq!(last.frame_path(7), Path::new("v##/a##_b007.exr"));
    assert_eq!(Pattern::parse(&path("shots#/plate.exr")), None);
}

#[test]
fn a_printf_frame_number_is_written_as_hashes() {
    let written = |name: &str| with_hashes(path(name));
    assert_eq!(written("a/plate.%04d.exr"), path("a/plate.####.exr"));
    assert_eq!(written("plate.%d.exr"), path("plate.#.exr"));
    assert_eq!(written("%02d_plate.%03d.exr"), path("%02d_plate.###.exr"));
    for kept in [
        "plate.%4d.exr",
        "plate.%0d.exr",
        "100%.exr",
        "a%04d/plate.exr",
    ] {
        assert_eq!(written(kept), path(kept), "{kept}");
    }
}

#[test]
fn a_file_may_be_a_frame_of_the_sequence_its_last_digits_number() {
    let candidate = |name| Pattern::candidate(Path::new(name)).map(|p| p.path().clone());
    assert_eq!(
        candidate("renders/beauty_part_0001.exr"),
        Some(path("renders/beauty_part_####.exr"))
    );
    assert_eq!(candidate("v2_shot_010.exr"), Some(path("v2_shot_###.exr")));
    assert_eq!(candidate("plate.exr"), None);
}

fn touch(directory: &Path, name: &str, bytes: usize) {
    std::fs::write(directory.join(name), vec![0; bytes]).unwrap();
}

#[test]
fn a_scan_finds_the_patterns_frames_and_nothing_else() {
    let directory = directory("scan");
    for name in [
        "plate.-002.exr",
        "plate.-001.exr",
        "plate.0001.exr",
        "plate.0002.exr",
        "plate.0005.exr",
        "plate.00003.exr",
        "plate.0004.exr.tmp",
        "other.0003.exr",
    ] {
        touch(&directory, name, 1);
    }
    std::fs::create_dir(directory.join("plate.0006.exr")).unwrap();
    let plate = Pattern::parse(&directory.join("plate.####.exr").into()).unwrap();

    let scan = Scan::of(&plate).unwrap();
    assert_eq!(scan.frames().collect::<Vec<_>>(), [-2, -1, 1, 2, 5]);
    assert_eq!(scan.range(), Some((-2, 5)));
    assert!(!scan.contains(3));

    let nowhere = Pattern::parse(&directory.join("gone/plate.####.exr").into()).unwrap();
    let error = Scan::of(&nowhere).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}

#[test]
fn a_rescan_reloads_each_frame_changed_new_or_gone() {
    let directory = directory("rescan");
    for name in ["plate.0001.exr", "plate.0002.exr", "plate.0005.exr"] {
        touch(&directory, name, 1);
    }
    let plate = Pattern::parse(&directory.join("plate.####.exr").into()).unwrap();
    let mut files = Files::default();
    assert_eq!(files.record(&plate, Scan::of(&plate).unwrap()), Some(0));
    assert_eq!(files.record(&plate, Scan::of(&plate).unwrap()), None);

    // A size changes, whatever the filesystem's clock resolution.
    touch(&directory, "plate.0001.exr", 2);
    std::fs::remove_file(directory.join("plate.0002.exr")).unwrap();
    touch(&directory, "plate.0003.exr", 1);
    assert_eq!(files.record(&plate, Scan::of(&plate).unwrap()), Some(3));

    let reloads = [1, 2, 3, 5].map(|frame| files.reloads(&plate.frame_path(frame)));
    assert_eq!(reloads, [1, 1, 1, 0]);
    assert_eq!(files.scan(plate.path()).unwrap().range(), Some((1, 5)));
}
