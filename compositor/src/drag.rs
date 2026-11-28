//! `--drag-check IMAGE [TYPE [PARAMETER]] [keyed]`: drags a parameter panel slider with synthetic
//! mouse events, one pixel each refresh of the viewer's screen, on Reader → a node of TYPE, the
//! Reader of IMAGE feeding each of the node's inputs. PARAMETER is a Float parameter of the type,
//! by default its first. A type with no Float parameter gets Reader → Exposure → the node instead,
//! dragging the Exposure's Stops. Without TYPE it runs every type that takes an image, each in a
//! child process of its own, and fails if any failed. `keyed` gives the dragged parameter keys
//! first, so that the drag sets a key at the frame.
//!
//! The worker evaluates the latest request and discards the rest, so an edit's latency ends at
//! the first viewer frame that samples a result from its request or a later one: a superseded
//! edit is answered by the frame showing a newer value, as the artist sees it. In a GPU session
//! (unless `--cpu` comes first) that end is the GPU's completion of the frame, since a frame
//! queued behind the worker's compute completes only after it; in a CPU session it is the GUI
//! thread's presentation, since when the display shows it would need the display.
//!
//! It fails unless the latencies' p95 and the final edit's meet the chain's budget, the GUI's
//! frames meet theirs (in a GPU session, also acquired → completed within two refreshes), the
//! drag undoes as one step, and a drag cancelled with Escape leaves the value as it was. Run it
//! twice, since the first run's shader and file caches are cold.

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cxx_qt_lib::QString;
use open_ng::Point;
use opencompositor_core::{
    Changes, Command, EvaluationDevice, Event, FloatParameter, FrameTiming, NodeId, NodeKind, Pane,
    Parameter, ParameterKind, Session, Slot, Timing, Value, Window, node_kinds, resolve,
};

use crate::check::{Durations, Seen, finish, gui_frames, resident_memory};
use crate::qt::ffi;

/// `QEvent::Type` values.
const MOUSE_BUTTON_PRESS: i32 = 2;
const MOUSE_BUTTON_RELEASE: i32 = 3;
const MOUSE_MOVE: i32 = 5;

const POLL: Duration = Duration::from_millis(100);
const LONGEST_WAIT: Duration = Duration::from_secs(60);

/// What a kind's drag needs beyond its description. A kind not named takes the defaults; one that
/// fails them needs its budget decided here, not raised to pass.
struct Tuning {
    /// The dragged parameter's values shown before the drag, the last its starting value.
    warm_up: &'static [f64],
    /// What the latencies' p95 and the final edit's must meet on the workstation in a CPU
    /// session. A GPU session's budget is two refreshes for every kind.
    cpu_budget: Duration,
}

fn tuning(kind: &NodeKind) -> Tuning {
    match kind {
        NodeKind::Blur => Tuning {
            // One Size summed directly on the GPU and one by blocks, either side of `DIRECT_T`, so
            // that the drag's pipelines are built.
            warm_up: &[3.0, 30.0],
            // Blur's CPU evaluation takes about as long as the gap between slider events, so
            // edits wait behind the one running; its budget is set from that, measured.
            cpu_budget: Duration::from_millis(70),
        },
        _ => Tuning {
            warm_up: &[],
            cpu_budget: Duration::from_millis(50),
        },
    }
}

/// What the check drags: a slider in the panel of a node of `kind`, or of the Exposure feeding it.
#[derive(Clone, Debug)]
pub struct Drag {
    kind: NodeKind,
    dragged: Dragged,
}

#[derive(Clone, Copy, Debug)]
enum Dragged {
    /// One of the node's own Float parameters.
    Own(&'static Parameter),
    /// The Stops of an Exposure between the Reader and the node, which has no Float parameter.
    UpstreamStops,
}

impl Drag {
    /// The drag named on the command line: a type that takes an image, by its name, and one of
    /// its Float parameters, by default its first. Refused with the choices.
    pub fn parse(type_name: &str, parameter: Option<&str>) -> Result<Drag, String> {
        let Some(kind) = (draggable_kinds()).find(|kind| kind.node_type().name == type_name) else {
            let names: Vec<&str> = draggable_kinds()
                .map(|kind| kind.node_type().name)
                .collect();
            return Err(format!(
                "no type {type_name:?} takes an image; the types: {}",
                names.join(", ")
            ));
        };
        let Some(name) = parameter else {
            return Ok(Drag::of(kind.clone()));
        };
        let Some(parameter) = floats(kind).find(|parameter| parameter.name == name) else {
            let names: Vec<&str> = floats(kind).map(|parameter| parameter.name).collect();
            return Err(format!(
                "{} has no Float parameter {name:?}; its Float parameters: {}",
                kind.node_type().label,
                if names.is_empty() {
                    "none".to_owned()
                } else {
                    names.join(", ")
                }
            ));
        };
        Ok(Drag {
            kind: kind.clone(),
            dragged: Dragged::Own(parameter),
        })
    }

    /// The default drag on `kind`: its first Float parameter, or without one the Exposure's Stops.
    fn of(kind: NodeKind) -> Drag {
        let dragged = floats(&kind)
            .next()
            .map_or(Dragged::UpstreamStops, Dragged::Own);
        Drag { kind, dragged }
    }

    /// The parameter whose slider is dragged, whose label names the edit's step.
    fn parameter(&self) -> &'static Parameter {
        match self.dragged {
            Dragged::Own(parameter) => parameter,
            Dragged::UpstreamStops => floats(&NodeKind::Exposure)
                .next()
                .expect("Exposure has its Stops"),
        }
    }

    fn float(&self) -> &'static FloatParameter {
        match &self.parameter().kind {
            ParameterKind::Float(float) => float,
            _ => unreachable!("a drag is of a Float parameter"),
        }
    }

    /// What the latencies' p95 and the final edit's must meet on the workstation.
    fn budget(&self, device: EvaluationDevice, refresh: Duration) -> Duration {
        match device {
            EvaluationDevice::Cpu => tuning(&self.kind).cpu_budget,
            EvaluationDevice::Gpu => refresh * 2,
        }
    }
}

/// As the reports name it: "Merge, Mix", or "Premult, Exposure's Stops".
impl fmt::Display for Drag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = self.kind.node_type().label;
        let parameter = self.parameter().label;
        match self.dragged {
            Dragged::Own(_) => write!(f, "{node}, {parameter}"),
            Dragged::UpstreamStops => write!(f, "{node}, Exposure's {parameter}"),
        }
    }
}

/// The kinds a drag can be on, in menu order: every one that takes an image.
fn draggable_kinds() -> impl Iterator<Item = &'static NodeKind> {
    (node_kinds().iter()).filter(|kind| !kind.node_type().inputs.is_empty())
}

fn floats(kind: &NodeKind) -> impl Iterator<Item = &'static Parameter> {
    (kind.node_type().parameters.iter())
        .filter(|parameter| matches!(parameter.kind, ParameterKind::Float(_)))
}

struct Check {
    image: PathBuf,
    drag: Drag,
    /// Whether the dragged parameter has keys, so that the drag sets one at the frame.
    keyed: bool,
    device: EvaluationDevice,
    /// The chain's nodes, from the Reader, the last the node whose slider is dragged.
    nodes: Vec<NodeId>,
    /// The dragged parameter's authored value before the drag, or keyed its value at the frame,
    /// which one undo restores.
    before: Option<Value>,
    /// The request marking the chain's node made, which the viewer must show before the drag.
    marked: u64,
    marked_at: Option<Instant>,
    /// How many images of the chain the viewer has shown before the drag: the first, then one
    /// for each warm-up value after the first.
    warmed_up: usize,
    first_frame: Option<FrameTiming>,
    /// The panel's slider, and how many pixels its drag covers.
    slider: *mut ffi::QSlider,
    span: i32,
    /// The viewer's refresh period, and the same in Qt's milliseconds, between moves.
    refresh: Duration,
    period: i32,
    edits: Vec<Edit>,
    /// How long each mouse event took the GUI thread.
    dispatches: Vec<Duration>,
    /// The last frame recorded, of any pane.
    last_frame: u64,
    /// Every pane's frames from the drag's start.
    frames: Vec<FrameTiming>,
    dragging: bool,
}

impl Check {
    /// From the edit to the first frame showing its result or a later one: to the frame's
    /// presentation in a CPU session, to its completion in a GPU session. `None` until then.
    fn latency(&self, edit: &Edit) -> Option<Duration> {
        let frame = &edit.answer.as_ref()?.frame;
        let end = match self.device {
            EvaluationDevice::Cpu => frame.presented,
            EvaluationDevice::Gpu => *frame.completed.get()?,
        };
        Some(end - edit.at)
    }
}

struct Edit {
    request: u64,
    at: Instant,
    /// When its result arrived on the GUI thread, with the worker's record and the data window
    /// (`None` for a failure); `None` if the worker discarded it.
    delivered: Option<(Instant, Timing, Option<Window>)>,
    /// The first viewer frame sampling its result or a later one.
    answer: Option<Answer>,
}

impl Edit {
    /// Whether a frame sampled its own result.
    fn shown(&self) -> bool {
        self.answer
            .as_ref()
            .is_some_and(|answer| answer.request == self.request)
    }
}

struct Answer {
    /// The request whose result the frame sampled.
    request: u64,
    frame: FrameTiming,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

/// Runs each draggable kind's default drag in a child process of its own, named on its command
/// line so that the command reruns it, and exits 1 if any failed, else 0. A chain run after
/// another in the same process stages several times slower (hypothesis: the recreated staging
/// buffer lands in a slower heap), so chains are only comparable from equal starts.
pub fn run_all(image: &Path, device: EvaluationDevice, keyed: bool) -> ! {
    let binary = std::env::current_exe().expect("the binary knows its own path");
    let mut failed = false;
    for drag in draggable_kinds().cloned().map(Drag::of) {
        let mut command = std::process::Command::new(&binary);
        if device == EvaluationDevice::Cpu {
            command.arg("--cpu");
        }
        command
            .arg("--drag-check")
            .arg(image)
            .arg(drag.kind.node_type().name);
        if let Dragged::Own(parameter) = drag.dragged {
            command.arg(parameter.name);
        }
        if keyed {
            command.arg("keyed");
        }
        let status = command.status();
        match status {
            Ok(status) if status.success() => {}
            Ok(status) => {
                eprintln!("drag check ({drag}): exited with {status}");
                failed = true;
            }
            Err(error) => {
                eprintln!(
                    "drag check ({drag}): could not run {}: {error}",
                    binary.display()
                );
                failed = true;
            }
        }
    }
    std::process::exit(i32::from(failed))
}

pub fn start(image: PathBuf, drag: Drag, keyed: bool) {
    let device = crate::evaluation_device();
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            image,
            drag,
            keyed,
            device,
            nodes: Vec::new(),
            before: None,
            marked: 0,
            marked_at: None,
            warmed_up: 0,
            first_frame: None,
            slider: std::ptr::null_mut(),
            span: 0,
            refresh: Duration::ZERO,
            period: 0,
            edits: Vec::new(),
            dispatches: Vec::new(),
            last_frame: 0,
            frames: Vec::new(),
            dragging: false,
        })
    });
    if device == EvaluationDevice::Gpu {
        ffi::single_shot(1, poll_gpu, 0);
    }
    ffi::single_shot(1000, setup, 0);
}

/// Polls the GPU every millisecond while the check runs, so that a frame's completion is
/// learned as it happens rather than at the next submit.
fn poll_gpu(_: usize) {
    if CHECK.with(|cell| cell.borrow().is_none()) {
        return;
    }
    crate::poll_gpu();
    ffi::single_shot(1, poll_gpu, 0);
}

pub fn observe(seen: &Seen) {
    match seen {
        Seen::Frame(pane, frame) => frame_drawn(*pane, frame),
        Seen::Delivered {
            job,
            timing,
            window,
            received,
            ..
        } => with_edit(*job, |edit| {
            edit.delivered = Some((*received, *timing, *window))
        }),
        _ => {}
    }
}

/// A viewer frame is the first for every edit its result answers, which is each edit it is from
/// or supersedes.
fn frame_drawn(pane: Pane, frame: &FrameTiming) {
    let shown = crate::presented_job();
    CHECK.with(|cell| {
        let mut check = cell.borrow_mut();
        let Some(check) = check.as_mut() else {
            return;
        };
        // A draw that presented nothing leaves the pane's last frame, recorded already.
        if frame.number <= check.last_frame {
            return;
        }
        check.last_frame = frame.number;
        if let (Pane::Viewer, Some(shown)) = (pane, shown) {
            if shown >= check.marked && check.marked_at.is_some() && check.first_frame.is_none() {
                check.first_frame = Some(frame.clone());
            }
            for edit in check.edits.iter_mut().filter(|edit| edit.request <= shown) {
                edit.answer.get_or_insert_with(|| Answer {
                    request: shown,
                    frame: frame.clone(),
                });
            }
        }
        if check.dragging {
            check.frames.push(frame.clone());
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

/// Makes the chain, the dragged node last so that its creation selects it for the panel, and
/// marks the node of the checked type in the viewer's slot.
fn setup(_: usize) {
    let (image, drag, keyed) = with(|check| (check.image.clone(), check.drag.clone(), check.keyed));
    let nodes = crate::edit(|session| {
        let reader = created(session.create_readers(vec![image], Point::default(), None));
        let node_type = drag.kind.node_type();
        let node = created(
            (session.create_node(node_type.name, Point::default(), Some(reader)))
                .expect("the type exists and the Reader exists"),
        );
        // Creation connected the first input.
        for input in &node_type.inputs[1..] {
            session
                .connect(node, input.first_slot(), reader)
                .expect("the node takes the input");
        }
        let mut nodes = vec![reader, node];
        if let Dragged::UpstreamStops = drag.dragged {
            // Inserted after the Reader, it takes over every input the Reader fed.
            // The changes also name the nodes it rewired.
            let exposure = NodeKind::Exposure.node_type().name;
            let changes = (session.create_node(exposure, Point::default(), Some(reader)))
                .expect("the type exists and the Reader exists");
            let exposure = (changes.nodes.iter())
                .find(|id| !nodes.contains(id))
                .expect("one node was created");
            nodes.push(*exposure);
        }
        let dragged = *nodes.last().expect("the chain has nodes");
        let parameter = drag.parameter().name;
        if keyed {
            // From the slider's middle, since the drag ends at its maximum, and a drag back to
            // the key's value is no step.
            let slider = &drag.float().slider;
            let middle = (slider.start() + slider.end()) / 2.0;
            session
                .set_value(dragged, parameter, Value::Float(middle))
                .expect("the slider is within the limits");
            // A key here and one ten frames on, so that the drag keys this frame.
            let frame = session.frame();
            for at in [frame + 10.0, frame] {
                session.set_frame(at).expect("the frame is finite");
                (session.toggle_key(dragged, parameter)).expect("the parameter takes keys");
            }
        }
        if let Some(&first) = tuning(&drag.kind).warm_up.first() {
            session
                .set_value(dragged, parameter, Value::Float(first))
                .expect("the warm-up is within the limits");
        }
        session
            .set_mark(Slot::new(1).expect("1 is a slot"), Some(node))
            .expect("the node exists");
        nodes
    });
    let marked = crate::latest_job();
    with(|check| {
        check.nodes = nodes;
        check.marked = marked;
        check.marked_at = Some(Instant::now());
    });
    ffi::single_shot(poll_in(1), wait_for_image, 1);
}

/// The parameter's authored value, or keyed its value at the session's frame, which is where a
/// drag's key goes.
fn value(session: &Session, node: NodeId, parameter: &str, keyed: bool) -> Option<Value> {
    let node = session.network().node(node)?;
    if keyed {
        resolve(node, session.frame()).get(parameter).cloned()
    } else {
        node.authored_value(parameter).cloned()
    }
}

fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

fn wait_for_image(tries: usize) {
    let marked = with(|check| check.marked);
    if crate::presented_job().is_some_and(|shown| shown >= marked) {
        let (drag, node, keyed, shown) = with(|check| {
            check.warmed_up += 1;
            let node = *check.nodes.last().expect("set up");
            (check.drag.clone(), node, check.keyed, check.warmed_up)
        });
        let parameter = drag.parameter().name;
        let warm_up = tuning(&drag.kind).warm_up;
        if shown < warm_up.len() {
            crate::edit(|session| {
                session
                    .set_value(node, parameter, Value::Float(warm_up[shown]))
                    .expect("the warm-up is within the limits")
            });
            with(|check| check.marked = crate::latest_job());
            ffi::single_shot(poll_in(1), wait_for_image, 1);
            return;
        }
        let before = crate::read_app(|app| value(app.session(), node, parameter, keyed));
        with(|check| check.before = before);
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

fn begin() {
    let dragged = with(|check| check.drag.parameter());
    let (parameter, label) = (dragged.name, dragged.label);
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
        check.refresh = refresh;
        check.period = period;
        check.dragging = true;
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

/// Sends the slider a mouse event, timing it, and records an edit if it asked for a new result.
fn send(kind: i32, pixel: i32) {
    let slider = with(|check| check.slider);
    let request = crate::latest_job();
    let at = Instant::now();
    // SAFETY: as in `begin`.
    unsafe { ffi::drag_slider(slider, kind, pixel) };
    let took = at.elapsed();
    let latest = crate::latest_job();
    with(|check| {
        check.dispatches.push(took);
        if latest != request {
            check.edits.push(Edit {
                request: latest,
                at,
                delivered: None,
                answer: None,
            });
        }
    });
}

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
    let framed = with(|check| {
        let last = check.edits.last();
        last.is_some_and(|edit| check.latency(edit).is_some())
    });
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

/// Checks that one undo removes the drag and that a drag cancelled with Escape changes nothing.
fn check_steps() -> Vec<String> {
    let (node, before, slider, span, dragged, keyed) = with(|check| {
        let node = *check.nodes.last().expect("set up");
        let before = check.before.clone();
        (
            node,
            before,
            check.slider,
            check.span,
            check.drag.parameter(),
            check.keyed,
        )
    });
    let (parameter, label) = (dragged.name, dragged.label);
    let state = || {
        crate::read_app(|app| {
            let session = app.session();
            let value = value(session, node, parameter, keyed);
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
        ffi::press_key(
            window,
            crate::keys::qt_key::ESCAPE,
            &QString::default(),
            false,
        );
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
    if undone != before {
        failures.push(format!(
            "one undo after the drag left {label} at {undone:?}, not {before:?}"
        ));
    }
    failures
}

fn caption() -> String {
    // SAFETY: the caption lives as long as the viewer's dock, until the main window is destroyed
    // after the event loop.
    String::from(&unsafe { ffi::label_text(crate::widgets().caption) })
}

fn report(mut failures: Vec<String>) {
    let check = CHECK
        .with(|cell| cell.borrow_mut().take())
        .expect("the check has started");
    let keyed = if check.keyed { ", keyed" } else { "" };
    let name = format!("drag check ({}{keyed})", check.drag);
    let gpu = check.device == EvaluationDevice::Gpu;
    let budget = check.drag.budget(check.device, check.refresh);
    let edits = &check.edits;
    let latency = |edit: &Edit| check.latency(edit);
    // Every edit once the final one is framed, since that frame answers them all.
    let latencies: Durations = edits.iter().filter_map(latency).collect();
    let final_latency = edits.last().and_then(latency);
    let shown = edits.iter().filter(|edit| edit.shown()).count();
    let spacing: Durations = edits
        .windows(2)
        .map(|pair| pair[1].at - pair[0].at)
        .collect();
    let dispatches: Durations = check.dispatches.iter().copied().collect();
    eprintln!("{name}: {}", caption());
    let end = if gpu {
        "the GPU completes"
    } else {
        "the GUI thread presents"
    };
    eprintln!(
        "{name}: evaluated on the {}, an edit's latency ending when {end} its first frame; \
         budget {budget:.1?}",
        check.device
    );
    if let (Some(marked_at), Some(first)) = (check.marked_at, &check.first_frame) {
        let end = match first.completed.get().filter(|_| gpu) {
            Some(&completed) => format!("completed {:.1?}", completed - marked_at),
            None => format!("presented {:.1?}", first.presented - marked_at),
        };
        eprintln!(
            "{name}: the first image, cold: marked → frame submitted {:.1?}, {end}",
            first.submitted - marked_at
        );
    }
    if let (Some(first), Some(last)) = (edits.first(), edits.last()) {
        eprintln!(
            "{name}: {} edits {:.1?} apart (asked {} ms) over {:.1?}",
            edits.len(),
            spacing.p(0.5),
            check.period,
            last.at - first.at
        );
    }
    eprintln!(
        "{name}: {} mouse events over {} pixels took the GUI thread {dispatches}",
        dispatches.len(),
        check.span,
    );
    eprintln!(
        "{name}: edit → frame {latencies}; the final edit {}; {} of {} edits' own values were shown",
        final_latency.map_or_else(|| "never".to_owned(), |latency| format!("{latency:.1?}")),
        shown,
        edits.len(),
    );
    type Stage = (
        &'static str,
        fn(&Edit, &Timing, Instant) -> Option<Duration>,
    );
    let stages: [Stage; 10] = [
        ("worker wait", |edit, timing, _| {
            Some(timing.started - edit.at)
        }),
        ("read", |_, timing, _| Some(timing.read? - timing.started)),
        ("evaluate", |_, timing, _| {
            Some(timing.evaluated - timing.read.unwrap_or(timing.started))
        }),
        ("staging wait", |_, timing, _| {
            Some(timing.staging.mapped? - timing.evaluated)
        }),
        ("planes written", |_, timing, _| {
            Some(timing.staging.written? - timing.staging.mapped?)
        }),
        ("textures", |_, timing, _| {
            Some(timing.staging.staged? - timing.staging.written?)
        }),
        ("delivery", |_, timing, received| {
            Some(received - timing.delivered)
        }),
        ("frame", |edit, _, received| {
            Some(edit.answer.as_ref()?.frame.submitted - received)
        }),
        ("present", |edit, _, _| {
            let frame = &edit.answer.as_ref()?.frame;
            Some(frame.presented - frame.submitted)
        }),
        ("GPU completion", |edit, _, _| {
            let frame = &edit.answer.as_ref()?.frame;
            Some(*frame.completed.get()? - frame.submitted)
        }),
    ];
    // The stages on each session's path to the end of a latency, which they sum to for an edit
    // whose own result was shown.
    let stages: Vec<&Stage> = (stages.iter())
        .filter(|(name, _)| *name != if gpu { "present" } else { "GPU completion" })
        .collect();
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
            .filter_map(|(name, stage)| {
                let values: Durations = delivered
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| selected(*index))
                    .filter_map(|(_, (edit, received, timing, _))| stage(edit, timing, *received))
                    .collect();
                // A stage the evaluation does not pass through, such as staging on the GPU, has
                // no values.
                (!values.is_empty()).then(|| format!("{name} {:.1?}", values.p(0.5)))
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
    // The edits that set the latencies the budget judges, with every stage of the result that
    // answered each, which for a superseded edit is a later edit's.
    let p95 = latencies.p(0.95);
    let longest_latency = latencies.max();
    let mut attributed: Vec<(Vec<&str>, &Edit)> = Vec::new();
    for (role, edit) in [
        ("p95", edits.iter().find(|edit| latency(edit) == Some(p95))),
        (
            "longest",
            (edits.iter()).find(|edit| latency(edit) == Some(longest_latency)),
        ),
        ("final", edits.last()),
    ] {
        let Some(edit) = edit.filter(|edit| latency(edit).is_some()) else {
            continue;
        };
        match attributed
            .iter_mut()
            .find(|(_, e)| e.request == edit.request)
        {
            Some((roles, _)) => roles.push(role),
            None => attributed.push((vec![role], edit)),
        }
    }
    for (roles, edit) in attributed {
        let answer = edit
            .answer
            .as_ref()
            .expect("an edit with a latency was answered");
        let answering = edits.iter().find(|e| e.request == answer.request);
        let by = if answer.request == edit.request {
            "its own result".to_owned()
        } else {
            let later = answering.map_or_else(String::new, |answering| {
                format!(", edited {:.1?} later", answering.at - edit.at)
            });
            format!("request {}'s result{later}", answer.request)
        };
        let stages = answering
            .and_then(|answering| {
                let (received, timing, _) = answering.delivered.as_ref()?;
                let stages: Vec<String> = (stages.iter())
                    .filter_map(|(name, stage)| {
                        Some(format!(
                            "{name} {:.1?}",
                            stage(answering, timing, *received)?
                        ))
                    })
                    .collect();
                Some(stages.join(", "))
            })
            .unwrap_or_else(|| "its stages were not recorded".to_owned());
        eprintln!(
            "{name}: the {} edit, request {}: {:.1?}, answered by {by}: {stages}",
            roles.join(" and "),
            edit.request,
            latency(edit).expect("filtered"),
        );
    }
    let gui = gui_frames(&check.frames, gpu, &mut failures);
    eprintln!("{name}: during the drag, of either pane, {gui}");
    if gpu {
        let to_completion: Durations = (check.frames.iter())
            .filter_map(|frame| Some(*frame.completed.get()? - frame.acquired))
            .collect();
        let longest = to_completion.max();
        eprintln!(
            "{name}: {} of them completed, acquired → completed p50 {:.1?}, p99 {:.1?}, max \
             {longest:.1?}",
            to_completion.len(),
            to_completion.p(0.5),
            to_completion.p(0.99),
        );
        if longest > check.refresh * 2 {
            failures.push(format!(
                "a frame acquired → completed {longest:.1?}, over two refreshes, {:.1?}",
                check.refresh * 2
            ));
        }
    }
    eprintln!("{name}: resident memory {}", resident_memory());
    if let Some(final_latency) = final_latency {
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
    finish(&name, &failures);
}
