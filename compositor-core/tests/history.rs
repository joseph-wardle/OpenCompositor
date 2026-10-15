//! What one undo step keeps alive. A step is the whole document from before it, sharing every
//! node the edit left alone, so it retains only the nodes the edit replaced and the path of the
//! persistent map down to them. Cloning a node must cost in proportion to its parameter count,
//! never its payloads: a value that can grow is shared on its own, not cloned with the node.
//!
//! Bytes are counted by this binary's allocator, per thread, so tests running alongside on other
//! threads do not disturb a measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::path::PathBuf;

use open_ng::{Point, Response};
use opencompositor_core::{ExposureVersion, NodeKind, Session, Value};

struct Counting;

thread_local! {
    /// Bytes this thread has allocated and not yet freed.
    static LIVE: Cell<isize> = const { Cell::new(0) };
}

fn count(bytes: isize) {
    // Never fails while the thread runs; during its teardown the count no longer matters.
    let _ = LIVE.try_with(|live| live.set(live.get() + bytes));
}

// SAFETY: every call forwards to the system allocator unchanged; counting only reads the layout.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size() as isize);
        // SAFETY: the caller's contract for `alloc` is the system allocator's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count(-(layout.size() as isize));
        // SAFETY: `ptr` came from the system allocator with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size as isize - layout.size() as isize);
        // SAFETY: `ptr` came from the system allocator with this layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The bytes `edit` leaves allocated: the step it records, with whatever else it keeps.
fn retained_by(edit: impl FnOnce()) -> isize {
    let before = LIVE.with(Cell::get);
    edit();
    LIVE.with(Cell::get) - before
}

/// A session of `readers` Readers, each with a path of `path_length` bytes, created as one
/// step so the history's own vector stays well within its capacity.
fn session_of(readers: usize, path_length: usize) -> Session {
    let mut session = Session::new();
    let paths = (0..readers)
        .map(|index| PathBuf::from(format!("{index:0>path_length$}.exr")))
        .collect();
    session.create_readers(paths, Point { x: 0.0, y: 0.0 });
    session
}

/// What a single-node step may retain in a document of 10,000 nodes: the node and its values,
/// with a map path that deepens only logarithmically as documents grow.
const STEP_BUDGET: isize = 2048;

#[test]
fn a_value_edit_retains_its_node_and_map_path_alone() {
    let mut session = session_of(10_000, 16);
    let changes = session.create_node(
        NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        },
        Point { x: 0.0, y: 0.0 },
    );
    let exposure = *changes.nodes.first().expect("one node created");

    let retained = retained_by(|| {
        session.begin_edit(exposure, "Set Stops").unwrap();
        for stops in 1..=100 {
            session
                .set_value("stops", Value::Float(stops as f64 / 10.0))
                .unwrap();
        }
        session.finish_edit();
    });
    assert!(retained <= STEP_BUDGET, "a drag retained {retained} bytes");
}

#[test]
fn moving_a_reader_retains_no_copy_of_its_path() {
    let retained = [16, 32 * 1024].map(|path_length| {
        let mut session = session_of(100, path_length);
        let reader = session.document().selection()[0];
        retained_by(|| {
            session.apply_response(&Response {
                positions: vec![(reader.into(), Point { x: 5.0, y: 5.0 })],
                ..Response::default()
            });
        })
    });
    assert_eq!(retained[0], retained[1], "a longer path retained more");
    assert!(
        retained[0] <= STEP_BUDGET,
        "a move retained {} bytes",
        retained[0]
    );
}
