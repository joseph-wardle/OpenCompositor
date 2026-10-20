//! `--drag-check IMAGE`: drags the parameter panel's Stops slider on a Reader → Exposure chain of
//! IMAGE with synthetic mouse events: a press at the groove's start, a move one pixel along it each
//! refresh of the viewer's screen, and a release at its end. It reports how the viewer kept up. The
//! worker evaluates the latest request and discards the rest, so an edit's latency ends at the
//! first viewer frame submitted that samples a result from its request or a later one: a
//! superseded edit is answered by the frame showing a newer value, as the artist sees it. The check
//! reports the latencies over every edit, the final edit's own, how many edits' own values were
//! shown, how long the GUI thread took to dispatch each mouse event from the slider to the requests
//! performed, the stages of the worker's path for the edits whose result it delivered, and the
//! process's resident memory. Then it checks that the drag undoes as one step and that a drag
//! cancelled with Escape leaves the value as it was. It exits 0 if those held and the latencies'
//! p95 and the final edit's met the budget, and 1 otherwise.
//!
//! Run it in a release build on the workstation, and twice, since the first run's shader and
//! file caches are cold. A frame counts when the GUI thread submits it, not when the display
//! shows it, which needs the display.

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use cxx_qt_lib::QString;
use open_ng::Point;
use opencompositor_core::{Changes, Command, Event, NodeId, Pane, Slot, Timing};

use crate::qt::ffi;

/// `QEvent::Type` values.
const MOUSE_BUTTON_PRESS: i32 = 2;
const MOUSE_BUTTON_RELEASE: i32 = 3;
const MOUSE_MOVE: i32 = 5;

/// What the latencies' p95 and the final edit's latency must meet in a CPU session on the
/// workstation.
const BUDGET: Duration = Duration::from_millis(50);

/// How often the check looks for the image before the drag and for the final edit's frame after,
/// and for how long at most.
const POLL: Duration = Duration::from_millis(100);
const LONGEST_WAIT: Duration = Duration::from_secs(60);

struct Check {
    image: PathBuf,
    exposure: Option<NodeId>,
    /// The panel's Stops slider, and how many pixels its drag covers.
    slider: *mut ffi::QSlider,
    span: i32,
    /// Between moves: the viewer's refresh period, to Qt's millisecond timers.
    period: i32,
    edits: Vec<Edit>,
    /// How long each mouse event took the GUI thread.
    dispatches: Vec<Duration>,
}

struct Edit {
    /// The worker's request for it.
    request: u64,
    at: Instant,
    /// When its result arrived on the GUI thread, with the worker's record; `None` if the worker
    /// discarded it.
    delivered: Option<(Instant, Timing)>,
    /// When the first viewer frame sampling its result, or a later one, was submitted.
    framed: Option<Instant>,
    /// Whether a frame sampled its own result.
    shown: bool,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

pub fn start(image: PathBuf) {
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            image,
            exposure: None,
            slider: std::ptr::null_mut(),
            span: 0,
            period: 0,
            edits: Vec::new(),
            dispatches: Vec::new(),
        })
    });
    ffi::single_shot(1000, setup, 0);
}

/// Records a result's arrival on the GUI thread, if the drag check is running and an edit asked
/// for it.
pub fn delivered(request: u64, timing: Timing, at: Instant) {
    with_edit(request, |edit| edit.delivered = Some((at, timing)));
}

/// Records the viewer's last presented frame as the first for every edit its result answers,
/// which is each edit it is from or supersedes, if the drag check is running. A draw that
/// presented nothing leaves the last frame, whose edits are already recorded.
pub fn viewer_frame_drawn() {
    let Some(shown) = crate::presented_request() else {
        return;
    };
    let now = Instant::now();
    CHECK.with(|cell| {
        let mut check = cell.borrow_mut();
        let Some(check) = check.as_mut() else {
            return;
        };
        for edit in check.edits.iter_mut().filter(|edit| edit.request <= shown) {
            edit.framed.get_or_insert(now);
            edit.shown |= edit.request == shown;
        }
    });
}

fn with_edit(request: u64, use_edit: impl FnOnce(&mut Edit)) {
    CHECK.with(|cell| {
        let mut check = cell.borrow_mut();
        let Some(check) = check.as_mut() else {
            return;
        };
        if let Some(edit) = check.edits.iter_mut().find(|edit| edit.request == request) {
            use_edit(edit);
        }
    });
}

fn with<R>(use_check: impl FnOnce(&mut Check) -> R) -> R {
    CHECK.with(|cell| use_check(cell.borrow_mut().as_mut().expect("the check has started")))
}

fn poll_in(tries: usize) -> i32 {
    POLL.as_millis() as i32 * tries as i32
}

/// Makes the chain, which selects the Exposure for the panel, and marks it in the viewer's slot,
/// which asks the worker for it.
fn setup(_: usize) {
    let image = with(|check| check.image.clone());
    let exposure = crate::edit(|session| {
        let reader = created(session.create_readers(vec![image], Point::default()));
        let node = created(
            (session.create_node("exposure", Point::default(), Some(reader)))
                .expect("Exposure is a node type and the Reader exists"),
        );
        session
            .set_mark(Slot::new(1).expect("1 is a slot"), Some(node))
            .expect("the node exists");
        node
    });
    with(|check| check.exposure = Some(exposure));
    ffi::single_shot(poll_in(1), wait_for_image, 1);
}

/// The one node a change created.
fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

fn wait_for_image(tries: usize) {
    if crate::presented_request().is_some() {
        begin();
    } else if POLL * tries as u32 >= LONGEST_WAIT {
        report(vec![format!(
            "the image was not shown within {LONGEST_WAIT:?}: {}",
            caption()
        )]);
    } else {
        ffi::single_shot(poll_in(1), wait_for_image, tries + 1);
    }
}

/// Presses the slider at the start of its groove and schedules the first move a refresh away.
fn begin() {
    // SAFETY: the panel lives as long as its dock, until the main window is destroyed after the
    // event loop; the slider until the panel shows another node, which nothing selects here.
    let slider = unsafe { ffi::panel_slider(crate::widgets().panel, &QString::from("stops")) };
    if slider.is_null() {
        return report(vec!["the panel shows no Stops slider".to_owned()]);
    }
    // SAFETY: as above.
    let span = unsafe { ffi::slider_span(slider) };
    if span < 1 {
        return report(vec![format!("the Stops slider spans {span} pixels")]);
    }
    let refresh =
        crate::with_pane_window(Pane::Viewer, |window| crate::pace::refresh_period(&window));
    let period = ((refresh.as_micros() + 500) / 1000).max(1) as i32;
    with(|check| {
        check.slider = slider;
        check.span = span;
        check.period = period;
    });
    send(MOUSE_BUTTON_PRESS, 0);
    ffi::single_shot(period, step, 1);
}

/// Sends the slider a mouse event, timing it, and records the edit if it asked for a new result.
fn send(kind: i32, pixel: i32) {
    let slider = with(|check| check.slider);
    let request = crate::latest_request();
    let at = Instant::now();
    // SAFETY: as in `begin`.
    unsafe { ffi::drag_slider(slider, kind, pixel) };
    let took = at.elapsed();
    let latest = crate::latest_request();
    with(|check| {
        check.dispatches.push(took);
        if latest != request {
            check.edits.push(Edit {
                request: latest,
                at,
                delivered: None,
                framed: None,
                shown: false,
            });
        }
    });
}

/// One move a pixel along, and the next a refresh later, until the end of the groove, where the
/// slider is released.
fn step(pixel: usize) {
    let (span, period) = with(|check| (check.span, check.period));
    let pixel = pixel as i32;
    send(MOUSE_MOVE, pixel);
    if pixel < span {
        ffi::single_shot(period, step, pixel as usize + 1);
    } else {
        send(MOUSE_BUTTON_RELEASE, span);
        ffi::single_shot(poll_in(1), wait_for_final, 1);
    }
}

fn wait_for_final(tries: usize) {
    let framed = with(|check| check.edits.last().is_some_and(|edit| edit.framed.is_some()));
    if framed {
        report(check_steps());
    } else if POLL * tries as u32 >= LONGEST_WAIT {
        report(vec![format!(
            "the final edit reached no frame within {LONGEST_WAIT:?}: {}",
            caption()
        )]);
    } else {
        ffi::single_shot(poll_in(1), wait_for_final, tries + 1);
    }
}

/// Checks that the drag was one step, which one undo removes, and that a drag cancelled with
/// Escape changes nothing. Returns the failures.
fn check_steps() -> Vec<String> {
    let (exposure, slider, span) =
        with(|check| (check.exposure.expect("set up"), check.slider, check.span));
    let state = || {
        crate::read_session(|session| {
            let stops = (session.document().node(exposure))
                .and_then(|node| node.authored_value("stops").cloned());
            (stops, session.undo_label(), session.editing().is_some())
        })
    };
    let mut failures = Vec::new();
    let dragged = state();
    if dragged.1 != Some("Stops") || dragged.2 {
        failures.push(format!(
            "after the release, undo would undo {:?}, with an edit in progress: {}",
            dragged.1, dragged.2
        ));
    }
    let middle = span / 2;
    for (kind, pixel) in [
        (MOUSE_BUTTON_PRESS, middle),
        (MOUSE_MOVE, (middle + 10).min(span)),
    ] {
        // SAFETY: as in `begin`.
        unsafe { ffi::drag_slider(slider, kind, pixel) };
    }
    let moved = state().0;
    // SAFETY: as in `begin`.
    unsafe {
        ffi::press_escape(slider);
        ffi::drag_slider(slider, MOUSE_BUTTON_RELEASE, middle);
    }
    let cancelled = state();
    if moved == dragged.0 || cancelled != dragged {
        failures.push(format!(
            "a drag cancelled with Escape moved Stops from {:?} to {moved:?} and left {:?}, then \
             undo would undo {:?}, with an edit in progress: {}",
            dragged.0, cancelled.0, cancelled.1, cancelled.2
        ));
    }
    crate::dispatch(Event::Command(Command::Undo));
    let undone = state().0;
    if undone.is_some() {
        failures.push(format!("one undo after the drag left Stops at {undone:?}"));
    }
    failures
}

fn caption() -> String {
    // SAFETY: the caption lives as long as the viewer's dock, until the main window is destroyed
    // after the event loop.
    String::from(&unsafe { ffi::label_text(crate::widgets().caption) })
}

/// Prints what the drag showed and exits: 1 with any failure, else 0.
fn report(mut failures: Vec<String>) {
    let check = CHECK
        .with(|cell| cell.borrow_mut().take())
        .expect("the check has started");
    let edits = &check.edits;
    let latency = |edit: &Edit| Some(edit.framed? - edit.at);
    // Every edit once the final one is framed, since that frame answers them all.
    let mut latencies: Vec<Duration> = edits.iter().filter_map(latency).collect();
    latencies.sort();
    let final_latency = edits.last().and_then(latency);
    let shown = edits.iter().filter(|edit| edit.shown).count();
    let mut spacing: Vec<Duration> = edits
        .windows(2)
        .map(|pair| pair[1].at - pair[0].at)
        .collect();
    spacing.sort();
    let mut dispatches = check.dispatches.clone();
    dispatches.sort();
    eprintln!("drag check: {}", caption());
    if let (Some(first), Some(last)) = (edits.first(), edits.last()) {
        eprintln!(
            "drag check: {} edits {:.1?} apart (asked {} ms) over {:.1?}",
            edits.len(),
            percentile(&spacing, 0.5),
            check.period,
            last.at - first.at
        );
    }
    eprintln!(
        "drag check: {} mouse events over {} pixels took the GUI thread p50 {:.1?}, p95 {:.1?}, max {:.1?}",
        dispatches.len(),
        check.span,
        percentile(&dispatches, 0.5),
        percentile(&dispatches, 0.95),
        percentile(&dispatches, 1.0),
    );
    eprintln!(
        "drag check: edit → frame p50 {:.1?}, p95 {:.1?}, max {:.1?}; the final edit {}; {} of {} edits' own values were shown",
        percentile(&latencies, 0.5),
        percentile(&latencies, 0.95),
        percentile(&latencies, 1.0),
        final_latency.map_or_else(|| "never".to_owned(), |latency| format!("{latency:.1?}")),
        shown,
        edits.len(),
    );
    type Stage = (
        &'static str,
        fn(&Edit, &Timing, Instant) -> Option<Duration>,
    );
    let stages: [Stage; 7] = [
        ("worker wait", |edit, timing, _| {
            Some(timing.started - edit.at)
        }),
        ("evaluate", |_, timing, _| {
            Some(timing.evaluated - timing.started)
        }),
        ("staging wait", |_, timing, _| {
            Some(timing.staging.mapped? - timing.evaluated)
        }),
        ("f16 conversion", |_, timing, _| {
            Some(timing.staging.converted? - timing.staging.mapped?)
        }),
        ("texture", |_, timing, _| {
            Some(timing.staging.staged? - timing.staging.converted?)
        }),
        ("delivery", |_, timing, received| {
            Some(received - timing.delivered)
        }),
        ("frame", |edit, _, received| Some(edit.framed? - received)),
    ];
    let delivered: Vec<(&Edit, Instant, Timing)> = edits
        .iter()
        .filter_map(|edit| {
            let (received, timing) = edit.delivered?;
            Some((edit, received, timing))
        })
        .collect();
    let medians: Vec<String> = stages
        .iter()
        .map(|(name, stage)| {
            let mut values: Vec<Duration> = delivered
                .iter()
                .filter_map(|(edit, received, timing)| stage(edit, timing, *received))
                .collect();
            values.sort();
            format!("{name} {:.1?}", percentile(&values, 0.5))
        })
        .collect();
    eprintln!(
        "drag check: stage medians of the {} edits whose result was delivered: {}",
        delivered.len(),
        medians.join(", ")
    );
    eprintln!("drag check: resident memory {}", resident_memory());
    if let Some(final_latency) = final_latency {
        let p95 = percentile(&latencies, 0.95);
        if p95 > BUDGET {
            failures.push(format!(
                "edit → frame p95 {p95:.1?}, over the {BUDGET:?} budget"
            ));
        }
        if final_latency > BUDGET {
            failures.push(format!(
                "the final edit {final_latency:.1?}, over the {BUDGET:?} budget"
            ));
        }
    }
    for failure in &failures {
        eprintln!("drag check failed: {failure}");
    }
    ffi::exit(i32::from(!failures.is_empty()));
}

/// The value at the quantile of sorted durations, or zero of none.
fn percentile(sorted: &[Duration], quantile: f64) -> Duration {
    match sorted.len() {
        0 => Duration::ZERO,
        len => sorted[((len - 1) as f64 * quantile).round() as usize],
    }
}

/// The process's resident set from /proc, as Linux reports it.
fn resident_memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .map_or_else(|| "unknown".to_owned(), |rss| rss.trim().to_owned())
}
