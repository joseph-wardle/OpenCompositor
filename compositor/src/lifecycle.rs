//! `--lifecycle-check IMAGE`: drives the window through what breaks GPU panes embedded in docks,
//! then quits. Rerun it in a nested compositor after every Qt upgrade, because the shell relies on
//! Qt's private Wayland API.
//!
//! It exits 0 if every step ran, a pane floated, docked or reopened had a surface again by the
//! next step, the image loaded while a dialog was open, and both panes had a surface at the end;
//! 1 if a check failed; and otherwise as a crash or protocol error does.

use std::cell::RefCell;
use std::path::PathBuf;
use std::pin::Pin;

use open_ng::Point;
use opencompositor_core::{Event, FileIntent, Pane, Request};

use crate::qt::ffi;

/// Each step and when it runs, in milliseconds from the start.
const STEPS: [(i32, &str); 13] = [
    (1000, "resize the window"),
    (2000, "float the viewer"),
    (3000, "dock the viewer again"),
    (4000, "tab the viewer behind the graph"),
    (4500, "redraw the hidden viewer"),
    (5000, "show the viewer tab"),
    (6000, "float the graph"),
    (7000, "dock the graph again"),
    (8000, "close the viewer"),
    (9000, "reopen the viewer"),
    (10_000, "read the image and open a dialog while it loads"),
    (15_000, "close the dialog"),
    (16_000, "quit"),
];

thread_local! {
    static IMAGE: RefCell<PathBuf> = RefCell::default();
    static FAILURES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

pub fn start(image: PathBuf) {
    IMAGE.with(|cell| *cell.borrow_mut() = image);
    for (index, &(at, _)) in STEPS.iter().enumerate() {
        ffi::single_shot(at, step, index);
    }
}

fn step(index: usize) {
    eprintln!("lifecycle check: {}", STEPS[index].1);
    let moved = match index {
        2 | 3 | 10 => Some(Pane::Viewer),
        7 | 8 => Some(Pane::Graph),
        _ => None,
    };
    if let Some(pane) = moved
        && !crate::has_surface(pane)
    {
        fail(format!(
            "the {pane:?} pane had no surface after the step before {:?}",
            STEPS[index].1
        ));
    }
    let widgets = crate::widgets();
    match index {
        // SAFETY: as for `dock`.
        0 => unsafe { Pin::new_unchecked(&mut *widgets.window) }.resize(1100, 700),
        1 => dock(Pane::Viewer).set_floating(true),
        2 => dock(Pane::Viewer).set_floating(false),
        3 => {
            let [graph, viewer] =
                [Pane::Graph, Pane::Viewer].map(|pane| widgets.docks[pane.index()]);
            // SAFETY: as for `dock`.
            unsafe { ffi::tabify(widgets.window, viewer, graph) };
            dock(Pane::Graph).raise();
        }
        4 => crate::perform(Request::Redraw(Pane::Viewer)),
        5 => dock(Pane::Viewer).raise(),
        6 => dock(Pane::Graph).set_floating(true),
        7 => dock(Pane::Graph).set_floating(false),
        8 => {
            dock(Pane::Viewer).close();
        }
        9 => dock(Pane::Viewer).show(),
        10 => {
            let at = Point::default();
            let image = IMAGE.with(|cell| cell.borrow().clone());
            crate::dispatch(Event::FilesChosen {
                intent: FileIntent::CreateReaders { at },
                paths: vec![image],
            });
            // Returns once the next step closes the dialog.
            crate::perform(Request::ChooseFiles(FileIntent::CreateReaders { at }));
        }
        11 => {
            // SAFETY: as for `dock`.
            let status = String::from(&unsafe { ffi::status(widgets.window) });
            if !status.contains(" × ") {
                fail(format!(
                    "the image had not loaded with the dialog open: {status:?}"
                ));
            }
            ffi::close_modal_dialog();
        }
        _ => {
            for pane in Pane::ALL {
                if !crate::has_surface(pane) {
                    fail(format!("the {pane:?} pane has no surface"));
                }
            }
            let failures = FAILURES.with(|cell| cell.take());
            for failure in &failures {
                eprintln!("lifecycle check failed: {failure}");
            }
            ffi::exit(i32::from(!failures.is_empty()));
        }
    }
}

fn dock(pane: Pane) -> Pin<&'static mut ffi::QDockWidget> {
    // SAFETY: the window and docks live until the main window is destroyed after the event loop,
    // which ends at the last step at the earliest.
    unsafe { Pin::new_unchecked(&mut *crate::widgets().docks[pane.index()]) }
}

fn fail(failure: String) {
    FAILURES.with(|cell| cell.borrow_mut().push(failure));
}
