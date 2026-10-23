//! `--drag-check IMAGE [CHAIN]`: drags a parameter panel slider with synthetic mouse events, a
//! press at the groove's start, a move one pixel along it each refresh of the viewer's screen,
//! and a release at its end, on a chain: `exposure`, Stops on Reader → Exposure of IMAGE,
//! `merge`, Mix on Reader → Merge with IMAGE as the background and as one layer, or `blur`, Size
//! on Reader → Blur of IMAGE. Without CHAIN it runs each chain in a child process of its own and
//! fails if any failed. It reports how the viewer kept up with each. The worker evaluates the
//! latest request and discards the rest, so an edit's latency ends at the first viewer frame
//! submitted that samples a result from its request or a later one: a superseded edit is
//! answered by the frame showing a newer value, as the artist sees it. For each chain the check
//! reports the latencies over every edit, the final edit's own, how many edits' own values were
//! shown, how long the GUI thread took to dispatch each mouse event from the slider to the
//! requests performed, the stages of the worker's path for the edits whose result it delivered,
//! the same stages for the delivered edits whose data window differed from the one delivered
//! before, which Blur's growing reach makes of some edits, and the process's resident memory.
//! Then it checks that the drag undoes as one step and that a drag cancelled with Escape leaves
//! the value as it was. It exits 0 if those held and the chain's latencies' p95 and final edit
//! met the chain's budget, and 1 otherwise.
//!
//! Run it in a release build on the workstation, and twice, since the first run's shader and
//! file caches are cold. A frame counts when the GUI thread submits it, not when the display
//! shows it, which needs the display.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cxx_qt_lib::QString;
use open_ng::Point;
use opencompositor_core::{Changes, Command, Event, NodeId, Pane, Slot, Timing, Window};

use crate::qt::ffi;

/// `QEvent::Type` values.
const MOUSE_BUTTON_PRESS: i32 = 2;
const MOUSE_BUTTON_RELEASE: i32 = 3;
const MOUSE_MOVE: i32 = 5;

/// How often the check looks for the image before the drag and for the final edit's frame after,
/// and for how long at most.
const POLL: Duration = Duration::from_millis(100);
const LONGEST_WAIT: Duration = Duration::from_secs(60);

/// A chain the check drags a slider on, from its Reader to the node whose panel has the slider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Exposure,
    Merge,
    Blur,
}

/// The chains, in the order the check runs them.
const CHAINS: [Chain; 3] = [Chain::Exposure, Chain::Merge, Chain::Blur];

impl Chain {
    /// The chain named on the command line by its node type.
    pub fn from_name(name: &str) -> Option<Chain> {
        CHAINS.into_iter().find(|chain| chain.type_name() == name)
    }

    /// The node type after the Reader, which the chain is named for.
    fn type_name(self) -> &'static str {
        match self {
            Chain::Exposure => "exposure",
            Chain::Merge => "merge",
            Chain::Blur => "blur",
        }
    }

    /// The parameter dragged, and its label, which names the edit's step.
    fn parameter(self) -> (&'static str, &'static str) {
        match self {
            Chain::Exposure => ("stops", "Stops"),
            Chain::Merge => ("mix", "Mix"),
            Chain::Blur => ("size", "Size"),
        }
    }

    /// What the latencies' p95 and the final edit's latency must meet in a CPU session on the
    /// workstation. Blur's evaluation takes about as long as the gap between slider events, so
    /// edits wait behind the one running; its budget is set from that, measured.
    fn budget(self) -> Duration {
        match self {
            Chain::Exposure | Chain::Merge => Duration::from_millis(50),
            Chain::Blur => Duration::from_millis(80),
        }
    }
}

struct Check {
    image: PathBuf,
    chain: Chain,
    /// The chain's nodes: the Reader and the node whose slider is dragged.
    nodes: Vec<NodeId>,
    /// The request marking the chain's node made, which the viewer must show before the drag.
    marked: u64,
    /// The panel's slider, and how many pixels its drag covers.
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
    /// When its result arrived on the GUI thread, with the worker's record and the image's data
    /// window, `None` for a failure; `None` if the worker discarded it.
    delivered: Option<(Instant, Timing, Option<Window>)>,
    /// When the first viewer frame sampling its result, or a later one, was submitted.
    framed: Option<Instant>,
    /// Whether a frame sampled its own result.
    shown: bool,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

/// Runs every chain, each as a child process of this binary dragging that chain alone, and
/// exits: 1 if any child did not exit 0, else 0. One process per chain because a chain run after
/// another in the same process showed its f16 conversion several times slower, whichever chain
/// it was; the hypothesis is that the staging buffer, recreated for the second chain, lands in a
/// slower heap. Either way the chains are only comparable from equal starts.
pub fn run_chains(image: &Path) -> ! {
    let binary = std::env::current_exe().expect("the binary knows its own path");
    let mut failed = false;
    for chain in CHAINS {
        let status = std::process::Command::new(&binary)
            .arg("--drag-check")
            .arg(image)
            .arg(chain.type_name())
            .status();
        match status {
            Ok(status) if status.success() => {}
            Ok(status) => {
                eprintln!("drag check ({chain:?}): exited with {status}");
                failed = true;
            }
            Err(error) => {
                eprintln!(
                    "drag check ({chain:?}): could not run {}: {error}",
                    binary.display()
                );
                failed = true;
            }
        }
    }
    std::process::exit(i32::from(failed))
}

pub fn start(image: PathBuf, chain: Chain) {
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            image,
            chain,
            nodes: Vec::new(),
            marked: 0,
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
pub fn delivered(request: u64, timing: Timing, window: Option<Window>, at: Instant) {
    with_edit(request, |edit| edit.delivered = Some((at, timing, window)));
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

/// Makes the chain, which selects its last node for the panel, and marks that node in the
/// viewer's slot, which asks the worker for it.
fn setup(_: usize) {
    let (image, chain) = with(|check| (check.image.clone(), check.chain));
    let nodes = crate::edit(|session| {
        let reader = created(session.create_readers(vec![image], Point::default(), None));
        let node = created(
            (session.create_node(chain.type_name(), Point::default(), Some(reader)))
                .expect("the type exists and the Reader exists"),
        );
        if chain == Chain::Merge {
            session
                .connect(node, ("layer", 1), reader)
                .expect("a Merge takes a layer");
        }
        session
            .set_mark(Slot::new(1).expect("1 is a slot"), Some(node))
            .expect("the node exists");
        vec![reader, node]
    });
    let marked = crate::latest_request();
    with(|check| {
        check.nodes = nodes;
        check.marked = marked;
    });
    ffi::single_shot(poll_in(1), wait_for_image, 1);
}

/// The one node a change created.
fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

fn wait_for_image(tries: usize) {
    let marked = with(|check| check.marked);
    if crate::presented_request().is_some_and(|shown| shown >= marked) {
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
    let (parameter, label) = with(|check| check.chain.parameter());
    // SAFETY: the panel lives as long as its dock, until the main window is destroyed after the
    // event loop; the slider until the panel shows another node, which nothing selects here.
    let slider = unsafe { ffi::panel_slider(crate::widgets().panel, &QString::from(parameter)) };
    if slider.is_null() {
        return report(vec![format!("the panel shows no {label} slider")]);
    }
    // SAFETY: as above.
    let span = unsafe { ffi::slider_span(slider) };
    if span < 1 {
        return report(vec![format!("the {label} slider spans {span} pixels")]);
    }
    let refresh =
        crate::with_pane_window(Pane::Viewer, |window| crate::pace::refresh_period(&window));
    let period = ((refresh.as_micros() + 500) / 1000).max(1) as i32;
    with(|check| {
        check.slider = slider;
        check.span = span;
        check.period = period;
    });
    if let Some(failure) = input_blocked() {
        return report(vec![failure]);
    }
    send(MOUSE_BUTTON_PRESS, 0);
    ffi::single_shot(period, step, 1);
}

/// Names a modal dialog or popup that would swallow the check's mouse and keys. Most likely a
/// key typed into the check's window opened it, such as R for Read Files.
fn input_blocked() -> Option<String> {
    let open = if ffi::modal_open() {
        "a modal dialog"
    } else if ffi::popup_open() {
        "a popup"
    } else {
        return None;
    };
    Some(format!(
        "{open} opened during the check and would swallow its input; was a key typed into the \
         window?"
    ))
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
    if let Some(failure) = input_blocked() {
        return report(vec![failure]);
    }
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
    let (node, slider, span, (parameter, label)) = with(|check| {
        let node = *check.nodes.last().expect("set up");
        (node, check.slider, check.span, check.chain.parameter())
    });
    let state = || {
        crate::read_app(|app| {
            let session = app.session();
            let value = (session.document().node(node))
                .and_then(|node| node.authored_value(parameter).cloned());
            (value, session.undo_label(), session.editing().is_some())
        })
    };
    if let Some(failure) = input_blocked() {
        return vec![failure];
    }
    let mut failures = Vec::new();
    let dragged = state();
    if dragged.1 != Some(label) || dragged.2 {
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
    if let Some(failure) = input_blocked() {
        failures.push(failure);
        return failures;
    }
    // SAFETY: as in `begin`.
    unsafe {
        let window = crate::widgets().window;
        ffi::press_key(window, crate::keys::qt_key::ESCAPE, &QString::default());
        ffi::drag_slider(slider, MOUSE_BUTTON_RELEASE, middle);
    }
    let cancelled = state();
    if moved == dragged.0 || cancelled != dragged {
        failures.push(format!(
            "a drag cancelled with Escape moved {label} from {:?} to {moved:?} and left {:?}, \
             then undo would undo {:?}, with an edit in progress: {}",
            dragged.0, cancelled.0, cancelled.1, cancelled.2
        ));
    }
    crate::dispatch(Event::Command(Command::Undo));
    let undone = state().0;
    if undone.is_some() {
        failures.push(format!(
            "one undo after the drag left {label} at {undone:?}"
        ));
    }
    failures
}

fn caption() -> String {
    // SAFETY: the caption lives as long as the viewer's dock, until the main window is destroyed
    // after the event loop.
    String::from(&unsafe { ffi::label_text(crate::widgets().caption) })
}

/// Prints what the chain's drag showed and exits: 1 with any failure, else 0.
fn report(mut failures: Vec<String>) {
    let check = CHECK
        .with(|cell| cell.borrow_mut().take())
        .expect("the check has started");
    let name = format!("drag check ({:?})", check.chain);
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
    eprintln!("{name}: {}", caption());
    if let (Some(first), Some(last)) = (edits.first(), edits.last()) {
        eprintln!(
            "{name}: {} edits {:.1?} apart (asked {} ms) over {:.1?}",
            edits.len(),
            percentile(&spacing, 0.5),
            check.period,
            last.at - first.at
        );
    }
    eprintln!(
        "{name}: {} mouse events over {} pixels took the GUI thread p50 {:.1?}, p95 {:.1?}, max {:.1?}",
        dispatches.len(),
        check.span,
        percentile(&dispatches, 0.5),
        percentile(&dispatches, 0.95),
        percentile(&dispatches, 1.0),
    );
    eprintln!(
        "{name}: edit → frame p50 {:.1?}, p95 {:.1?}, max {:.1?}; the final edit {}; {} of {} edits' own values were shown",
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
    let delivered: Vec<(&Edit, Instant, Timing, Option<Window>)> = edits
        .iter()
        .filter_map(|edit| {
            let (received, timing, window) = edit.delivered?;
            Some((edit, received, timing, window))
        })
        .collect();
    let medians = |selected: &dyn Fn(usize) -> bool| -> String {
        let medians: Vec<String> = stages
            .iter()
            .map(|(name, stage)| {
                let mut values: Vec<Duration> = delivered
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| selected(*index))
                    .filter_map(|(_, (edit, received, timing, _))| stage(edit, timing, *received))
                    .collect();
                values.sort();
                format!("{name} {:.1?}", percentile(&values, 0.5))
            })
            .collect();
        medians.join(", ")
    };
    eprintln!(
        "{name}: stage medians of the {} edits whose result was delivered: {}",
        delivered.len(),
        medians(&|_| true)
    );
    // Blur's reach grows every few units of Size, and with it the data window, so those edits'
    // planes and staging must fit a larger image than the one before.
    let changed: Vec<bool> = delivered
        .iter()
        .scan(None, |previous, (_, _, _, window)| {
            let changed = previous.is_some_and(|previous| window.is_some_and(|w| w != previous));
            *previous = window.or(*previous);
            Some(changed)
        })
        .collect();
    let changes = changed.iter().filter(|&&changed| changed).count();
    eprintln!(
        "{name}: stage medians of the {changes} of them whose data window changed: {}",
        medians(&|index| changed[index])
    );
    eprintln!("{name}: resident memory {}", resident_memory());
    if let Some(final_latency) = final_latency {
        let p95 = percentile(&latencies, 0.95);
        let budget = check.chain.budget();
        if p95 > budget {
            failures.push(format!(
                "edit → frame p95 {p95:.1?}, over the {budget:?} budget"
            ));
        }
        if final_latency > budget {
            failures.push(format!(
                "the final edit {final_latency:.1?}, over the {budget:?} budget"
            ));
        }
    }
    for failure in &failures {
        eprintln!("{name} failed: {failure}");
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
