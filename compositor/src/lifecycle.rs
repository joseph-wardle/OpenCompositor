//! `--lifecycle-check IMAGE`: drives the window through what breaks GPU panes embedded in docks,
//! then quits. Rerun it in a nested compositor after every Qt upgrade, because the shell relies on
//! Qt's private Wayland API.
//!
//! It exits 0 if every step ran; each pane a step made draw had a surface and a frame the GPU
//! finished by the next step; the image was shown while a dialog was open, in the caption and, where
//! the surface can be read back, at the viewer's centre; Tab over the graph opened the creation
//! menu, and Escape closed it with no step; Exposure chosen in it was inserted after the Reader as
//! one step, leaving the Reader marked; an edit from a script moved the panel's slider without an
//! edit of its own; the panel survived its dock floated, closed and reopened; Reader chosen in the
//! menu and its dialog cancelled created nothing; after the device was destroyed, a new one
//! drew both panes, with the same centre and caption, within the rebuild's budget; both panes had a
//! surface at the end; and the GPU reported no uncaptured error. It exits 1 if a check failed, and
//! otherwise as a crash or protocol error does. Frames are checked as rendered, not as presented:
//! that needs the display, as a screenshot.
//!
//! `destroy` stands in for a driver reset, which the check never causes: a reset may differ.
//!
//! IMAGE must not be black at its centre.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::pin::Pin;
use std::time::{Duration, Instant};

use cxx_qt_lib::QString;
use open_ng::{Modifiers, Point};
use opencompositor_core::{
    Anchor, Event, FileIntent, FrameCheck, Input, Key, NodeId, NodeKind, Pane, Request, Slot, Value,
};

use crate::keys::qt_key;
use crate::qt::ffi;

/// Each step, when it runs in milliseconds from the start, and the panes it makes draw: each of
/// those has a surface and has finished a new frame by the next step.
const STEPS: [(i32, &str, &[Pane]); 23] = [
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
    (16_000, "press Tab over the graph", &[]),
    (16_500, "press Escape in the creation menu", &[]),
    (
        17_000,
        "choose Exposure in the creation menu",
        &[Pane::Graph],
    ),
    (17_500, "edit Stops from a script", &[]),
    (18_000, "float the parameter panel", &[]),
    (18_500, "close the parameter panel", &[]),
    (19_000, "reopen the parameter panel", &[]),
    (19_500, "choose Reader in the creation menu", &[]),
    (20_500, "cancel the Reader's dialog", &[]),
    (
        21_000,
        "destroy the GPU device",
        &[Pane::Graph, Pane::Viewer],
    ),
    (22_000, "quit", &[]),
];

/// The longest an attempt to make a device may block the GUI thread, warm, on the workstation.
/// Configuring the panes' swapchains is most of it and varies most from run to run.
/// Hypothesis, untested: after a real reset the same adapter returns, so its shader cache is warm.
const REBUILD_BUDGET: Duration = Duration::from_millis(350);
/// The longest from the loss to the viewer's first frame showing the image again, re-uploaded and
/// re-evaluated, in a GPU session on the workstation's desktop.
const TOTAL_RECOVERY_BUDGET: Duration = Duration::from_millis(700);

thread_local! {
    static IMAGE: RefCell<PathBuf> = RefCell::default();
    static FAILURES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// What each pane's frames proved when the last step began.
    static CHECKED: Cell<[Option<FrameCheck>; Pane::COUNT]> = const { Cell::new([None; Pane::COUNT]) };
    /// What each pane's frames proved on the device the check destroyed.
    static DESTROYED: Cell<Option<[Option<FrameCheck>; Pane::COUNT]>> = const { Cell::new(None) };
    /// When the check destroyed the device, and how long until the viewer's first frame on a
    /// new one finished showing the image again.
    static DESTROYED_AT: Cell<Option<Instant>> = const { Cell::new(None) };
    static RECOVERED_IN: Cell<Option<Duration>> = const { Cell::new(None) };
    /// The longest an attempt to make a device took.
    static LONGEST_ATTEMPT: Cell<Option<Duration>> = const { Cell::new(None) };
    /// How long retiring the lost device took, which has no budget yet.
    static RETIREMENT: Cell<Option<Duration>> = const { Cell::new(None) };
    /// Where Stops' slider was after the scripted edit.
    static STOPS_TICK: Cell<Option<i32>> = const { Cell::new(None) };
}

pub fn start(image: PathBuf) {
    IMAGE.with(|cell| *cell.borrow_mut() = image);
    for (index, &(at, ..)) in STEPS.iter().enumerate() {
        ffi::single_shot(at, step, index);
    }
}

/// Records how long an attempt to make a GPU device blocked the GUI thread.
pub fn attempt_took(duration: Duration) {
    LONGEST_ATTEMPT.with(|cell| cell.set(cell.get().max(Some(duration))));
}

/// Records how long retiring a lost device blocked the GUI thread.
pub fn retirement_took(duration: Duration) {
    RETIREMENT.with(|cell| cell.set(Some(duration)));
}

fn step(index: usize) {
    let checks = Pane::ALL.map(crate::frame_check);
    let finished = checks.map(|check| check.map_or(0, |check| check.finished));
    eprintln!(
        "lifecycle check: {} (frames finished: graph {}, viewer {})",
        STEPS[index].1, finished[0], finished[1]
    );
    let before = CHECKED.with(|cell| cell.replace(checks));
    let (after, panes) = match index.checked_sub(1) {
        Some(previous) => (STEPS[previous].1, STEPS[previous].2),
        None => ("starting", &Pane::ALL[..]),
    };
    for &pane in panes {
        let [before, now] = [before, checks].map(|checks| checks[pane.index()]);
        if !crate::has_surface(pane) {
            fail(format!("the {pane:?} pane had no surface after {after:?}"));
        } else if new_frames(before, now) == 0 {
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
                intent: FileIntent::CreateReaders(Anchor { at, current: None }),
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
            crate::perform(Request::ChooseFiles(FileIntent::CreateReaders(Anchor {
                at,
                current: None,
            })));
        }
        11 => {
            // SAFETY: the caption lives as long as the viewer's dock, as for `with_dock`.
            let caption = String::from(&unsafe { ffi::label_text(widgets.caption) });
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
        12 => press_tab_over_the_graph(),
        13 => {
            if !ffi::popup_open() {
                fail("Tab over the graph opened no menu".to_owned());
            }
            press_key(qt_key::ESCAPE, "");
        }
        14 => {
            if ffi::popup_open() {
                fail("Escape left the creation menu open".to_owned());
            }
            check_undo_label("Mark", "after the menu was dismissed");
            choose_in_menu("exp");
        }
        15 => {
            check_undo_label("Create Exposure", "after Exposure was chosen");
            let exposure = check_inserted_exposure();
            if let Some(exposure) = exposure {
                edit_stops(exposure);
            }
        }
        16 => with_panel_dock(|dock| dock.set_floating(true)),
        17 => {
            with_panel_dock(|dock| dock.close());
        }
        18 => with_panel_dock(|dock| dock.show()),
        19 => {
            let tick = stops_slider().map(|slider| slider.value());
            if tick.is_none() || tick != STOPS_TICK.with(Cell::get) {
                fail(format!(
                    "the panel showed Stops' slider at {tick:?} after its dock was reopened"
                ));
            }
            with_panel_dock(|mut dock| {
                dock.as_mut().set_floating(false);
            });
            // Returns once the next step cancels the dialog.
            choose_in_menu("read");
        }
        20 => ffi::close_modal_dialog(),
        21 => {
            check_undo_label("Stops", "after the Reader's dialog was cancelled");
            let nodes = crate::read_app(|app| app.session().document().nodes().count());
            if nodes != 2 {
                fail(format!(
                    "{nodes} nodes after the Reader's dialog was cancelled, not 2"
                ));
            }
            DESTROYED.with(|cell| cell.set(Some(checks)));
            DESTROYED_AT.with(|cell| cell.set(Some(Instant::now())));
            crate::destroy_device();
            ffi::single_shot(1, watch_recovery, 0);
        }
        _ => {
            check_recovery(checks);
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

fn press_tab_over_the_graph() {
    crate::dispatch(Event::Input {
        pane: Pane::Graph,
        input: Input::PointerEntered,
    });
    press_key(qt_key::TAB, "");
}

/// Opens the creation menu, types the text and presses Enter. The type is created once the event
/// loop runs.
fn choose_in_menu(text: &str) {
    press_tab_over_the_graph();
    for character in text.chars() {
        let key = character.to_ascii_uppercase() as i32;
        press_key(key, &character.to_string());
    }
    press_key(qt_key::RETURN, "");
}

fn press_key(key: i32, text: &str) {
    // SAFETY: the window lives until it is destroyed after the event loop.
    unsafe { ffi::press_key(crate::widgets().window, key, &QString::from(text)) };
}

fn check_undo_label(expected: &str, when: &str) {
    let label = crate::read_app(|app| app.session().undo_label());
    if label != Some(expected) {
        fail(format!("the step to undo was {label:?} {when}"));
    }
}

/// Checks that one Exposure is selected, inserted after the Reader slot 1 still marks, and
/// returns it.
fn check_inserted_exposure() -> Option<NodeId> {
    let slot = Slot::new(1).expect("1 is a slot");
    let (selection, source, marked) = crate::read_app(|app| {
        let document = app.session().document();
        let &[exposure] = document.selection() else {
            return (document.selection().to_vec(), None, None);
        };
        let source = (document.node(exposure))
            .filter(|node| matches!(node.kind, NodeKind::Exposure { .. }))
            .and_then(|node| node.input("source"));
        (vec![exposure], source, document.mark(slot))
    });
    match (selection.as_slice(), source) {
        (&[exposure], Some(source)) if marked == Some(source) => Some(exposure),
        _ => {
            fail(format!(
                "after Exposure was chosen, {selection:?} was selected, taking {source:?}, and \
                 slot 1 marked {marked:?}"
            ));
            None
        }
    }
}

/// Sets Stops as a script does, and checks that the panel's slider followed without an edit.
fn edit_stops(exposure: NodeId) {
    let before = stops_slider().map(|slider| slider.value());
    let edited = crate::edit(|session| {
        let edited = session.set_value(exposure, "stops", Value::Float(2.0));
        (edited, session.version())
    });
    let version = crate::read_app(|app| app.session().version());
    let after = stops_slider().map(|slider| slider.value());
    STOPS_TICK.with(|cell| cell.set(after));
    match edited {
        (Err(error), _) => fail(format!("the scripted edit failed: {error}")),
        (Ok(_), edited) if edited != version => {
            fail("refreshing the panel after a scripted edit edited the document".to_owned())
        }
        _ if before.is_none() || after <= before => fail(format!(
            "the panel's Stops slider went from {before:?} to {after:?} for 2 stops"
        )),
        _ => check_undo_label("Stops", "after the scripted edit"),
    }
}

/// Stops' slider in the panel, which lives until the panel next rebuilds its rows.
fn stops_slider() -> Option<&'static ffi::QSlider> {
    // SAFETY: the panel lives as long as its dock; the slider until a rebuild, which the checks
    // that read it do not cause.
    unsafe { ffi::panel_slider(crate::widgets().panel, &QString::from("stops")).as_ref() }
}

/// Frames finished between the checks, counting all of a new device's.
fn new_frames(before: Option<FrameCheck>, now: Option<FrameCheck>) -> u64 {
    match (before, now) {
        (Some(before), Some(now)) if before.generation == now.generation => {
            now.finished - before.finished
        }
        (_, now) => now.map_or(0, |now| now.finished),
    }
}

/// Records when the viewer's last finished frame is on a new device and has the centre it had
/// before the loss, so shows the image again, re-evaluated in a GPU session; until then looks
/// again each millisecond.
fn watch_recovery(_: usize) {
    let (Some(destroyed), Some(at)) = (DESTROYED.with(Cell::get), DESTROYED_AT.with(Cell::get))
    else {
        return;
    };
    let before = destroyed[Pane::Viewer.index()];
    let now = crate::frame_check(Pane::Viewer);
    let recovered = match (before, now) {
        (Some(before), Some(now)) => {
            now.generation > before.generation
                && before.centre.is_some()
                && now.centre == before.centre
        }
        _ => false,
    };
    if recovered {
        RECOVERED_IN.with(|cell| cell.set(Some(at.elapsed())));
    } else {
        ffi::single_shot(1, watch_recovery, 0);
    }
}

/// Checks that a new device replaced the destroyed one: both panes drew on it with the same
/// centre as before, the viewer's not black, the GPU label is gone, and making it kept within the
/// budget.
fn check_recovery(checks: [Option<FrameCheck>; Pane::COUNT]) {
    let Some(destroyed) = DESTROYED.with(Cell::get) else {
        fail("the device was not destroyed".to_owned());
        return;
    };
    for pane in Pane::ALL {
        let [before, after] = [destroyed, checks].map(|checks| checks[pane.index()]);
        let Some(before) = before else {
            fail(format!(
                "the {pane:?} pane drew nothing before the device was destroyed"
            ));
            continue;
        };
        if after.map_or(0, |after| after.generation) <= before.generation {
            fail(format!(
                "the {pane:?} pane drew on no new device after the loss"
            ));
            continue;
        }
        match (before.centre, after.and_then(|after| after.centre)) {
            (Some(_), Some([0, 0, 0, 255])) if pane == Pane::Viewer => {
                fail("the viewer's centre was black on the new device".to_owned())
            }
            (Some(before), Some(after)) if before != after => fail(format!(
                "the {pane:?} pane's centre was {before:?} before the loss and {after:?} after"
            )),
            (Some(_), None) => fail(format!(
                "the {pane:?} pane's centre was not read on the new device"
            )),
            _ => {}
        }
    }
    let widgets = crate::widgets();
    // SAFETY: the caption lives as long as the viewer's dock, and the GPU label as long as the
    // window, as for `with_dock`.
    let [caption, gpu_label] = [widgets.caption, widgets.gpu_label]
        .map(|label| String::from(&unsafe { ffi::label_text(label) }));
    if !caption.contains(" × ") || caption.contains("updating") {
        fail(format!(
            "the image was not shown again after the loss: {caption:?}"
        ));
    }
    if !gpu_label.is_empty() {
        fail(format!(
            "the GPU label stayed after the loss: {gpu_label:?}"
        ));
    }
    if let Some(retirement) = RETIREMENT.with(Cell::get) {
        eprintln!("lifecycle check: retiring the lost device took {retirement:?}");
    }
    match RECOVERED_IN.with(Cell::get) {
        None => fail("the viewer did not show the image again on a new device".to_owned()),
        Some(recovered) => {
            eprintln!(
                "lifecycle check: from the loss to the viewer's first frame showing the image \
                 again took {recovered:?} ({} evaluation)",
                crate::evaluation_device()
            );
            if recovered > TOTAL_RECOVERY_BUDGET {
                fail(format!(
                    "recovery took {recovered:?}, over {TOTAL_RECOVERY_BUDGET:?}"
                ));
            }
        }
    }
    match LONGEST_ATTEMPT.with(Cell::get) {
        None => fail("no attempt made a device after the loss".to_owned()),
        Some(longest) => {
            eprintln!("lifecycle check: the longest attempt to make a device took {longest:?}");
            if longest > REBUILD_BUDGET {
                fail(format!(
                    "making a device blocked the GUI thread for {longest:?}, over {REBUILD_BUDGET:?}"
                ));
            }
        }
    }
}

/// Calls `use_dock` with the parameter panel's dock. The reference ends with it.
fn with_panel_dock<R>(use_dock: impl FnOnce(Pin<&mut ffi::QDockWidget>) -> R) -> R {
    // SAFETY: as for `with_dock`; the window owns the panel's dock.
    unsafe {
        let dock = ffi::panel_dock(crate::widgets().panel);
        use_dock(Pin::new_unchecked(&mut *dock))
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
