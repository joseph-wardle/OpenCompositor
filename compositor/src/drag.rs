//! `--drag-check IMAGE`: drags Exposure's Stops on a Reader → Exposure chain of IMAGE, one edit
//! per refresh of the viewer's screen, and reports how the viewer kept up. The worker evaluates the
//! latest request and discards the rest, so no one number says it: the check reports how many
//! edits reached a frame, the latency from each such edit to the first frame that sampled its
//! result, the final edit's latency, the stages of the worker's path, and the process's resident
//! memory. It exits 0 if the latencies' p95 met the budget and 1 otherwise.
//!
//! Run it in a release build on the workstation, and twice, since the first run's shader and
//! file caches are cold. A frame counts when the GUI thread submits it, not when the display
//! shows it, which needs the display.

use std::cell::RefCell;
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use open_ng::Point;
use opencompositor_core::{
    Changes, ExposureVersion, NodeId, NodeKind, Pane, Session, Slot, Timing, Value,
};

use crate::qt::ffi;

/// How many Stops edits the drag makes, spread over this range.
const EDITS: usize = 200;
const STOPS: RangeInclusive<f64> = -3.0..=3.0;

/// The p95 from an edit to the frame that first sampled its result, in a CPU session on the
/// workstation: the kernel branch's gate.
const BUDGET: Duration = Duration::from_millis(50);

/// How often the check looks for the image before the drag and for the final edit's frame after,
/// and for how long at most.
const POLL: Duration = Duration::from_millis(100);
const LONGEST_WAIT: Duration = Duration::from_secs(60);

struct Check {
    image: PathBuf,
    exposure: Option<NodeId>,
    /// Between edits: the viewer's refresh period, to Qt's millisecond timers.
    period: i32,
    edits: Vec<Edit>,
}

struct Edit {
    /// The worker's request for it.
    request: u64,
    at: Instant,
    /// When its result arrived on the GUI thread, with the worker's record; `None` if the worker
    /// discarded it.
    delivered: Option<(Instant, Timing)>,
    /// When the first viewer frame sampling its result was submitted.
    framed: Option<Instant>,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

pub fn start(image: PathBuf) {
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            image,
            exposure: None,
            period: 0,
            edits: Vec::new(),
        })
    });
    ffi::single_shot(1000, setup, 0);
}

/// Records a result's arrival on the GUI thread, if the drag check is running and an edit asked
/// for it.
pub fn delivered(request: u64, timing: Timing, at: Instant) {
    with_edit(request, |edit| edit.delivered = Some((at, timing)));
}

/// Records that the viewer submitted a frame, the first for the result it shows if the drag
/// check is running and an edit asked for it.
pub fn viewer_frame_submitted() {
    let Some(shown) = crate::shown_request() else {
        return;
    };
    with_edit(shown, |edit| {
        edit.framed.get_or_insert_with(Instant::now);
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

/// Makes the chain and marks it in the viewer's slot, which asks the worker for it.
fn setup(_: usize) {
    let image = with(|check| check.image.clone());
    let mut exposure = None;
    crate::edit(|session| {
        let reader = created(session.create_readers(vec![image], Point::default()));
        let kind = NodeKind::Exposure {
            version: ExposureVersion::CURRENT,
        };
        let node = created(session.create_node(kind, Point { x: 0.0, y: 150.0 }));
        session
            .connect(node, "source", reader)
            .expect("both nodes exist");
        session
            .set_mark(Slot::new(1).expect("1 is a slot"), Some(node))
            .expect("the node exists");
        exposure = Some(node);
    });
    with(|check| check.exposure = exposure);
    ffi::single_shot(poll_in(1), wait_for_image, 1);
}

/// The one node a change created.
fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

fn wait_for_image(tries: usize) {
    if crate::shown_request().is_some() {
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

/// Begins the edit and schedules its first step a refresh away.
fn begin() {
    let exposure = with(|check| check.exposure.expect("set up"));
    crate::edit(|session| {
        session
            .begin_edit(exposure, "Stops")
            .expect("the node exists");
    });
    let refresh =
        crate::with_pane_window(Pane::Viewer, |window| crate::pace::refresh_period(&window));
    let period = ((refresh.as_micros() + 500) / 1000).max(1) as i32;
    with(|check| check.period = period);
    ffi::single_shot(period, step, 0);
}

/// One Stops edit, and the next a refresh later, until the last, which finishes the edit.
fn step(index: usize) {
    let stops = STOPS.start() + (STOPS.end() - STOPS.start()) * (index + 1) as f64 / EDITS as f64;
    let at = Instant::now();
    crate::edit(|session| {
        session
            .set_value("stops", Value::Float(stops))
            .expect("Stops takes the value");
    });
    let request = crate::last_request();
    let period = with(|check| {
        check.edits.push(Edit {
            request,
            at,
            delivered: None,
            framed: None,
        });
        check.period
    });
    if index + 1 < EDITS {
        ffi::single_shot(period, step, index + 1);
    } else {
        crate::edit(Session::finish_edit);
        ffi::single_shot(poll_in(1), wait_for_final, 1);
    }
}

fn wait_for_final(tries: usize) {
    let framed = with(|check| check.edits.last().is_some_and(|edit| edit.framed.is_some()));
    if framed {
        report(Vec::new());
    } else if POLL * tries as u32 >= LONGEST_WAIT {
        report(vec![format!(
            "the final edit reached no frame within {LONGEST_WAIT:?}: {}",
            caption()
        )]);
    } else {
        ffi::single_shot(poll_in(1), wait_for_final, tries + 1);
    }
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
    let framed: Vec<&Edit> = edits.iter().filter(|edit| edit.framed.is_some()).collect();
    let latency = |edit: &Edit| edit.framed.expect("framed") - edit.at;
    let mut latencies: Vec<Duration> = framed.iter().map(|edit| latency(edit)).collect();
    latencies.sort();
    let mut spacing: Vec<Duration> = edits
        .windows(2)
        .map(|pair| pair[1].at - pair[0].at)
        .collect();
    spacing.sort();
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
        "drag check: {} of {} edits reached a frame; edit → frame p50 {:.1?}, p95 {:.1?}, max {:.1?}; the final edit {}",
        framed.len(),
        edits.len(),
        percentile(&latencies, 0.5),
        percentile(&latencies, 0.95),
        percentile(&latencies, 1.0),
        match edits.last() {
            Some(edit) if edit.framed.is_some() => format!("{:.1?}", latency(edit)),
            _ => "never".to_owned(),
        }
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
    let medians: Vec<String> = stages
        .iter()
        .map(|(name, stage)| {
            let mut values: Vec<Duration> = framed
                .iter()
                .filter_map(|edit| {
                    let (received, timing) = edit.delivered?;
                    stage(edit, &timing, received)
                })
                .collect();
            values.sort();
            format!("{name} {:.1?}", percentile(&values, 0.5))
        })
        .collect();
    eprintln!(
        "drag check: stage medians of the edits that reached a frame: {}",
        medians.join(", ")
    );
    eprintln!("drag check: resident memory {}", resident_memory());
    if latencies.is_empty() {
        failures.push("no edit reached a frame".to_owned());
    } else if percentile(&latencies, 0.95) > BUDGET {
        failures.push(format!(
            "edit → frame p95 {:.1?}, over the {BUDGET:?} budget",
            percentile(&latencies, 0.95)
        ));
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
