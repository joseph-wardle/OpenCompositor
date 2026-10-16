//! `--lifecycle-check IMAGE`: drives the window through what breaks GPU panes embedded in docks,
//! then quits. Rerun it in a nested compositor after every Qt upgrade, because the shell relies on
//! Qt's private Wayland API.
//!
//! It exits 0 if every step ran; each pane a step made draw had a surface and a frame the GPU
//! finished by the next step; the image was shown while a dialog was open, in the caption and, where
//! the surface can be read back, at the viewer's centre; both panes had a surface at the end; and
//! the GPU reported no uncaptured error. It exits 1 if a check failed, and otherwise as a crash or
//! protocol error does. Frames are checked as rendered, not as presented: that needs the display,
//! as a screenshot.
//!
//! IMAGE must not be black at its centre.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::pin::Pin;

use open_ng::{Modifiers, Point};
use opencompositor_core::{Event, FileIntent, Key, Pane, Request, Slot};

use crate::qt::ffi;

/// Each step, when it runs in milliseconds from the start, and the panes it makes draw: each of
/// those has a surface and has finished a new frame by the next step.
const STEPS: [(i32, &str, &[Pane]); 13] = [
    (1000, "resize the window", &[Pane::Graph, Pane::Viewer]),
    (2000, "float the viewer", &[Pane::Viewer]),
    (3000, "dock the viewer again", &[Pane::Viewer]),
    (4000, "tab the viewer behind the graph", &[]),
    (4500, "redraw the hidden viewer", &[]),
    (5000, "show the viewer tab", &[Pane::Viewer]),
    (6000, "float the graph", &[Pane::Graph]),
    (7000, "dock the graph again", &[Pane::Graph]),
    (8000, "close the viewer", &[]),
    (9000, "reopen the viewer", &[Pane::Viewer]),
    (
        10_000,
        "read the image, view it, and open a dialog while it loads",
        &[Pane::Viewer],
    ),
    (15_000, "close the dialog", &[]),
    (16_000, "quit", &[]),
];

thread_local! {
    static IMAGE: RefCell<PathBuf> = RefCell::default();
    static FAILURES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// Each pane's finished frames when the last step began.
    static FINISHED: Cell<[u64; Pane::COUNT]> = const { Cell::new([0; Pane::COUNT]) };
}

pub fn start(image: PathBuf) {
    IMAGE.with(|cell| *cell.borrow_mut() = image);
    for (index, &(at, ..)) in STEPS.iter().enumerate() {
        ffi::single_shot(at, step, index);
    }
}

fn step(index: usize) {
    let finished = Pane::ALL.map(|pane| crate::frame_check(pane).map_or(0, |check| check.finished));
    eprintln!(
        "lifecycle check: {} (frames finished: graph {}, viewer {})",
        STEPS[index].1, finished[0], finished[1]
    );
    let before = FINISHED.with(|cell| cell.replace(finished));
    let (after, panes) = match index.checked_sub(1) {
        Some(previous) => (STEPS[previous].1, STEPS[previous].2),
        None => ("starting", &Pane::ALL[..]),
    };
    for &pane in panes {
        if !crate::has_surface(pane) {
            fail(format!("the {pane:?} pane had no surface after {after:?}"));
        } else if finished[pane.index()] <= before[pane.index()] {
            fail(format!(
                "the {pane:?} pane finished no frame after {after:?}"
            ));
        }
    }
    let widgets = crate::widgets();
    match index {
        // SAFETY: as for `with_dock`; the reference ends with the call.
        0 => unsafe { Pin::new_unchecked(&mut *widgets.window) }.resize(1100, 700),
        1 => with_dock(Pane::Viewer, |dock| dock.set_floating(true)),
        2 => with_dock(Pane::Viewer, |dock| dock.set_floating(false)),
        3 => {
            let [graph, viewer] =
                [Pane::Graph, Pane::Viewer].map(|pane| widgets.docks[pane.index()]);
            // SAFETY: as for `with_dock`.
            unsafe { ffi::tabify(widgets.window, viewer, graph) };
            with_dock(Pane::Graph, |dock| dock.raise());
        }
        4 => crate::perform(Request::Redraw(Pane::Viewer)),
        5 => with_dock(Pane::Viewer, |dock| dock.raise()),
        6 => with_dock(Pane::Graph, |dock| dock.set_floating(true)),
        7 => with_dock(Pane::Graph, |dock| dock.set_floating(false)),
        8 => {
            with_dock(Pane::Viewer, |dock| dock.close());
        }
        9 => with_dock(Pane::Viewer, |dock| dock.show()),
        10 => {
            let at = Point::default();
            let image = IMAGE.with(|cell| cell.borrow().clone());
            crate::dispatch(Event::FilesChosen {
                intent: FileIntent::CreateReaders { at },
                paths: vec![image],
            });
            crate::dispatch(Event::KeyPressed {
                key: Key::Slot(Slot::new(1).expect("1 is a slot")),
                modifiers: Modifiers {
                    shift: false,
                    ctrl: true,
                },
            });
            // Returns once the next step closes the dialog.
            crate::perform(Request::ChooseFiles(FileIntent::CreateReaders { at }));
        }
        11 => {
            // SAFETY: the caption lives as long as the viewer's dock, as for `with_dock`.
            let caption = String::from(&unsafe { ffi::caption(widgets.caption) });
            if !caption.contains(" × ") {
                fail(format!(
                    "the image was not shown with the dialog open: {caption:?}"
                ));
            }
            match crate::frame_check(Pane::Viewer).and_then(|check| check.centre) {
                Some([0, 0, 0, 255]) => fail(format!(
                    "the viewer's centre was black with the image shown: {caption:?}"
                )),
                Some(_) => {}
                None => eprintln!(
                    "lifecycle check: the viewer's surface cannot be read back, so its pixels go unchecked"
                ),
            }
            ffi::close_modal_dialog();
        }
        _ => {
            for pane in Pane::ALL {
                if !crate::has_surface(pane) {
                    fail(format!("the {pane:?} pane has no surface"));
                }
            }
            match crate::gpu_errors() {
                0 => {}
                errors => fail(format!("the GPU reported {errors} uncaptured errors")),
            }
            let failures = FAILURES.with(|cell| cell.take());
            for failure in &failures {
                eprintln!("lifecycle check failed: {failure}");
            }
            ffi::exit(i32::from(!failures.is_empty()));
        }
    }
}

/// Calls `use_dock` with the pane's dock. The reference ends with it.
fn with_dock<R>(pane: Pane, use_dock: impl FnOnce(Pin<&mut ffi::QDockWidget>) -> R) -> R {
    let dock = crate::widgets().docks[pane.index()];
    // SAFETY: the window and docks live until the main window is destroyed after the event loop,
    // which ends at the last step at the earliest. Steps run from timers, so no other reference to
    // the dock is live.
    use_dock(unsafe { Pin::new_unchecked(&mut *dock) })
}

fn fail(failure: String) {
    FAILURES.with(|cell| cell.borrow_mut().push(failure));
}
