//! The interactive compositor: a session, the GPU that draws its panes, and the evaluator's
//! worker, changed only by [`App::dispatch`]. What only the shell can do comes back as [`Request`]s.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use open_ng::{Button, InputEvent, Modifiers, PhysicalPoint, Point};

use crate::document::{Missing, NodeId, Slot, missing};
use crate::evaluate::{EvalError, ResultKey, result_key};
use crate::files::{Pattern, Scan, held, input_range};
use crate::gpu::{
    Drawn, FrameCheck, FrameTiming, Gpu, GpuMemory, NativeWindow, PaneSize, PixelRead, Probe,
    ViewerFrame,
};
use crate::image::{ImageInfo, Window};
use crate::panel::Panel;
use crate::parameter::Value;
use crate::playback::{
    Clock, Direction, FrameMark, FrameRange, Frames, MAX_MARKED_FRAMES, PlaybackMode, Timebar,
};
use crate::session::Session;
use crate::viewer::{EXPOSURE_STEP, Zoom};
use crate::viewer_cache::ViewerCopy;
use crate::worker::{Evaluated, EvaluationDevice, ViewerError, Wanted, Worker};

/// A part of the window that the GPU draws. Panes are docked, floated and closed independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    Graph,
    Viewer,
}

impl Pane {
    pub const COUNT: usize = 2;
    pub const ALL: [Pane; Pane::COUNT] = [Pane::Graph, Pane::Viewer];

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Something that happened in the shell.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "a result is dispatched once and never stored, where a box would allocate per result"
)]
pub enum Event {
    /// The pane became visible and must be drawn. A pane without a native window asks for one.
    Exposed(Pane),
    /// The pane's native window, after [`Request::AttachSurface`].
    SurfaceCreated {
        pane: Pane,
        window: NativeWindow,
        size: PaneSize,
    },
    /// The pane's native window is about to be destroyed. Its surface is dropped before
    /// `dispatch` returns.
    SurfaceDestroyed(Pane),
    /// The pane changed size or scale.
    Resized {
        pane: Pane,
        size: PaneSize,
    },
    /// The time to draw a pane, after [`Request::Redraw`].
    Draw(Pane),
    Input {
        pane: Pane,
        input: Input,
    },
    /// A key pressed anywhere in the window, unless a text field, a dialog or a menu has the
    /// keyboard. Repeats are left out.
    KeyPressed {
        key: Key,
        modifiers: Modifiers,
    },
    /// A key released anywhere in the window, so that no key is left held.
    KeyReleased(Key),
    Command(Command),
    /// The type chosen in the menu a [`Request::ShowCreationMenu`] opened, by its name in
    /// [`node_types`](crate::node_types). A Reader asks for its files first.
    NodeTypeChosen {
        type_name: &'static str,
        anchor: Anchor,
    },
    /// The pane's gesture can no longer expect its button's release, as when the pane lost the
    /// keyboard to another pane or a dialog, or the window lost it to another application.
    FocusLost(Pane),
    /// The files chosen in the dialog a [`Request::ChooseFiles`] opened: none if it was cancelled.
    /// A file that may be a frame of a sequence waits for a scan to say, and is then read as the
    /// sequence if it has other frames.
    FilesChosen {
        intent: FileIntent,
        paths: Vec<PathBuf>,
    },
    /// The artist asked for the Reader's files to be read again, from its panel: Reload files.
    ReloadFiles(NodeId),
    /// A scan of a sequence finished on its own thread.
    Scanned {
        sequence: Pattern,
        scan: io::Result<Scan>,
    },
    /// An evaluation finished on the evaluator's thread.
    Evaluated(Evaluated),
    /// The evaluator's thread made a copy of an image it evaluated, for the viewer cache.
    Copied(ViewerCopy),
    /// The GPU device of the generation was lost, when wgpu reported it.
    DeviceLost {
        generation: u64,
        at: Instant,
    },
    /// The time for the attempt, after [`Request::RetryDeviceCreation`].
    RetryDeviceCreation(u64),
    /// The time to collect the pixel read in flight, after [`Request::CollectPixelRead`].
    CollectPixelRead,
    /// The artist set the viewer's exposure, in stops, in the viewer's own control.
    ViewerExposureSet(f32),
    /// The time to advance playback, after [`Request::Advance`], with the time it is.
    Advance(Instant),
    /// The artist went to a frame, typing it or scrubbing the timebar. It stops playback.
    FrameSet(f64),
    /// The artist set the timeline's first and last frames, as one step.
    FrameRangeSet {
        first: f64,
        last: f64,
    },
    /// The artist set the rate the timeline plays at, in frames per second, as one step.
    FrameRateSet(f64),
    /// The artist chose the mode playback plays in, from now on and at once if playing.
    PlaybackModeSet(PlaybackMode),
    /// The artist chose whether playback wraps round at the range's end.
    LoopingSet(bool),
    /// The artist chose which frames the timebar spans and playback plays through.
    FrameRangeChosen(FrameRange),
}

/// What another thread, or wgpu inside a call, tells the GUI thread through the waker
/// [`App::new`] takes. The shell dispatches it as an [`Event`].
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "moved once to the GUI thread and never stored, where a box would allocate per result"
)]
pub enum Notice {
    Evaluated(Evaluated),
    Copied(ViewerCopy),
    DeviceLost {
        generation: u64,
        at: Instant,
    },
    Scanned {
        sequence: Pattern,
        scan: io::Result<Scan>,
    },
}

impl From<Notice> for Event {
    fn from(notice: Notice) -> Event {
        match notice {
            Notice::Evaluated(evaluated) => Event::Evaluated(evaluated),
            Notice::Copied(copy) => Event::Copied(copy),
            Notice::DeviceLost { generation, at } => Event::DeviceLost { generation, at },
            Notice::Scanned { sequence, scan } => Event::Scanned { sequence, scan },
        }
    }
}

/// Hands a [`Notice`] to the GUI thread, from any thread.
#[derive(Clone)]
pub(crate) struct Waker(Arc<dyn Fn(Notice) + Send + Sync>);

impl Waker {
    pub(crate) fn wake(&self, notice: Notice) {
        (self.0)(notice);
    }
}

#[cfg(test)]
impl Waker {
    pub(crate) fn channel() -> (Waker, std::sync::mpsc::Receiver<Notice>) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let waker = Waker(Arc::new(move |notice| {
            let _ = sender.send(notice);
        }));
        (waker, receiver)
    }
}

impl fmt::Debug for Waker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Waker")
    }
}

/// What the shell does for [`App::dispatch`], after it returns.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Dispatch [`Event::Draw`] for the pane when the display is next ready.
    Redraw(Pane),
    /// Dispatch [`Event::SurfaceCreated`] with the pane's native window.
    AttachSurface(Pane),
    /// Open a file dialog, then dispatch [`Event::FilesChosen`].
    ChooseFiles(FileIntent),
    /// Show the text in the status bar for a few seconds, as the answer to what was just done.
    ShowStatus(String),
    /// Show the text in the status bar until the next status replaces it, for what stays wrong.
    ShowError(String),
    /// Describe under the viewer what it shows, or why it shows nothing.
    ShowViewerCaption(String),
    /// Show under the caption the shown image's working values at the pixel under the pointer,
    /// or nothing with an empty text.
    ShowReadout(String),
    /// Show the viewer's exposure, in stops, in its control, without dispatching
    /// [`Event::ViewerExposureSet`]. It starts at 0.
    ShowViewerExposure(f32),
    /// Name in the Edit menu what undo and redo would undo and redo, and disable either with
    /// `None`. Both start disabled.
    ShowUndoLabels {
        undo: Option<&'static str>,
        redo: Option<&'static str>,
    },
    /// Open the node creation menu at the pointer, then dispatch [`Event::NodeTypeChosen`] with
    /// the anchor if a type is chosen.
    ShowCreationMenu(Anchor),
    /// Show the node's parameters in the parameter panel, or that no node is selected with
    /// `None`. The panel starts empty.
    ShowPanel(Option<Panel>),
    /// Show the text in the status bar's permanent GPU label, or hide the label with `None`.
    /// It changes when the panes stop or start drawing, and when a GPU error is counted.
    ShowGpuLabel(Option<String>),
    /// Dispatch [`Event::RetryDeviceCreation`] with the attempt after the delay. An attempt not
    /// the last one requested is ignored, so dispatching every one is safe.
    RetryDeviceCreation { attempt: u64, after: Duration },
    /// Dispatch [`Event::CollectPixelRead`] after the delay.
    CollectPixelRead { after: Duration },
    /// Dispatch [`Event::Advance`] after the delay, with the time it is then. The app asks for one
    /// at a time.
    Advance { after: Duration },
    /// Show the timebar under the viewer as given. It starts empty.
    ShowTimebar(Timebar),
}

/// A window-wide command, from a menu or its shortcut, whichever pane has the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// A Reader for each file chosen, placed where the pointer is over the graph, or below the
    /// current node.
    ReadFiles,
    /// Keeps the parameter panel on the node it shows, whatever is selected, until unpinned or
    /// the node is deleted.
    PinPanel,
    /// Has the parameter panel show the current node, the last selected, again.
    UnpinPanel,
    /// Reads the selected Readers' files again, for every Reader of them, when they are next
    /// viewed, at once if they are: Reload files on each.
    Reload,
    Undo,
    Redo,
    /// Views what the slot marks.
    ViewSlot(Slot),
    /// Marks the one selected node in the slot, and views it.
    MarkSlot(Slot),
    /// Centres the node the slot marks in the graph.
    FrameMark(Slot),
    /// Fits the viewed image's display window to the viewer.
    FitImage,
    /// Shows the viewed image one pixel to a physical pixel.
    ActualSize,
    /// Raises the viewer's exposure by half a stop.
    ExposureUp,
    /// Lowers the viewer's exposure by half a stop.
    ExposureDown,
    /// Sets the viewer's exposure back to 0.
    ResetExposure,
    /// Evaluates on the device from now on, also after the app moved evaluation off the GPU.
    EvaluateOn(EvaluationDevice),
    /// Plays the timeline that way in the mode chosen, or stops if it plays that way already.
    Play(Direction),
    /// Stops playback, staying on the frame.
    Stop,
    /// Stops playback and goes one frame that way, round to the other end when looping.
    Step(Direction),
    /// Stops playback and goes to the timeline's first frame.
    FirstFrame,
    /// Stops playback and goes to the timeline's last frame.
    LastFrame,
}

/// The keys the window uses. The pane under the pointer receives all but the slot keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Space,
    Escape,
    E,
    G,
    H,
    Y,
    /// Delete or Backspace: deletes the selected nodes while the pointer is over the graph.
    Delete,
    /// Opens the node creation menu while the pointer is over the graph.
    Tab,
    /// Over the viewer, the arrows step a frame, and J, K and L play backwards, stop and play
    /// forwards.
    Left,
    Right,
    J,
    K,
    L,
    /// A key of the digit row, by its position rather than the character it types, so that
    /// layouts that type digits with Shift, as AZERTY does, use the same keys. Alone it views
    /// the slot, with Ctrl it marks the selected node there, with Shift it frames that node.
    Slot(Slot),
}

impl Key {
    fn in_graph(self) -> Option<open_ng::Key> {
        Some(match self {
            Key::Space => open_ng::Key::Space,
            Key::Escape => open_ng::Key::Escape,
            Key::G => open_ng::Key::G,
            Key::H => open_ng::Key::H,
            Key::Y => open_ng::Key::Y,
            Key::E
            | Key::Delete
            | Key::Tab
            | Key::Left
            | Key::Right
            | Key::J
            | Key::K
            | Key::L
            | Key::Slot(_) => return None,
        })
    }
}

/// What files chosen in a dialog are for. A dialog is modal, but its target can still change while
/// it is open, so the choice is checked again when it arrives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FileIntent {
    /// A Reader for each file, stacked downwards from the anchor.
    CreateReaders(Anchor),
    /// The first file chosen for a node's file parameter.
    SetFile {
        node: NodeId,
        parameter: &'static str,
    },
}

impl FileIntent {
    pub fn allows_several(self) -> bool {
        matches!(self, FileIntent::CreateReaders(_))
    }
}

/// Files chosen in a dialog, waiting for the scans that say which are frames of a sequence.
#[derive(Debug)]
struct Choice {
    intent: FileIntent,
    /// Each file chosen, with the sequence it may be a frame of until a scan decides.
    files: Vec<(PathBuf, Option<Pattern>)>,
}

/// Where a new node goes, as it was when the artist asked for it, so that a menu or dialog taking
/// the keyboard and pointer in between changes nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    /// The pointer over the graph, or the middle of the graph view if it was elsewhere.
    pub at: Point,
    /// The node a new node is inserted after or placed below: the last selected.
    pub current: Option<NodeId>,
}

/// Pointer input to a pane, in physical pixels from its top-left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Input {
    PointerEntered,
    PointerLeft,
    PointerMoved {
        position: PhysicalPoint,
    },
    PointerPressed {
        position: PhysicalPoint,
        button: Button,
        modifiers: Modifiers,
    },
    PointerReleased {
        position: PhysicalPoint,
        button: Button,
    },
    /// Wheel notches; positive scrolls up.
    Scrolled {
        position: PhysicalPoint,
        lines: f32,
    },
}

/// What the viewer shows, as [`App::displayed`] describes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Displayed {
    /// The node the image is of, which the viewer may no longer mark.
    pub node: NodeId,
    /// The frame the image is of.
    pub frame: f64,
    /// The session's frame, which the viewer is following.
    pub requested: f64,
    /// Whether the frame failed, and shows no image.
    pub failed: bool,
    /// How many times the viewer cache's copy shown was halved, or `None` if the image shown was
    /// evaluated.
    pub cached: Option<u32>,
}

/// What the viewer shows.
#[derive(Debug)]
enum Shown {
    Nothing,
    Image {
        wanted: Wanted,
        info: ImageInfo,
    },
    /// A copy from the viewer cache.
    Copy {
        wanted: Wanted,
        copy: Arc<ViewerCopy>,
    },
    /// An empty frame, never an older image.
    Failed {
        wanted: Wanted,
        error: String,
    },
}

impl Shown {
    /// The request whose result is shown.
    fn wanted(&self) -> Option<&Wanted> {
        match self {
            Shown::Nothing => None,
            Shown::Image { wanted, .. }
            | Shown::Copy { wanted, .. }
            | Shown::Failed { wanted, .. } => Some(wanted),
        }
    }

    /// The image the viewer frames: none for an empty display window, which has no size to fit.
    fn framed(&self) -> Option<&ImageInfo> {
        let info = match self {
            Shown::Image { info, .. } => info,
            Shown::Copy { copy, .. } => &copy.info,
            Shown::Nothing | Shown::Failed { .. } => return None,
        };
        (!info.display_window.is_empty()).then_some(info)
    }

    fn display_window(&self) -> Option<Window> {
        self.framed().map(|info| info.display_window)
    }
}

/// Waited for before the first attempt after a loss, so that the shell paints the GPU label
/// before the attempt blocks the GUI thread (a hypothesis, untested).
const FIRST_RETRY: Duration = Duration::from_millis(100);
/// Waited for after a failed attempt, doubling up to the longest wait.
const RETRY_AFTER_FAILURE: Duration = Duration::from_secs(1);
/// The longest wait between attempts, and how long a device must work before its loss starts
/// again from the first wait.
const LONGEST_RETRY: Duration = Duration::from_secs(30);

/// Waited for between looks at a pixel read in flight. Its copy is tiny, but waits behind the
/// evaluation's work on the queue; and nothing else polls an idle device, which a wait for the
/// map callback alone would need.
const PIXEL_READ_POLL: Duration = Duration::from_millis(4);

/// Whether the panes have a device to draw on.
#[derive(Debug)]
enum Drawing {
    /// The panes draw on a device, or none has been needed yet.
    Ready { recovered: Option<Recovered> },
    /// No device: the attempt numbered `attempt` makes one after `delay`. Editing continues
    /// unseen, since the panes are not redrawn: a fallback drawn by Qt is not worth its cost.
    Unavailable { attempt: u64, delay: Duration },
}

/// Losses of the device while evaluation had work on its GPU, since the artist last chose the
/// GPU, after which evaluation moves to the CPU: likely its own work lost them. A heuristic: a
/// display driver that fails during evaluation moves it too.
const LOSSES_BEFORE_LEAVING_THE_GPU: u32 = 2;

/// Why the app moved evaluation off the GPU the artist chose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeftGpu {
    OutOfMemory,
    Lost,
}

impl fmt::Display for LeftGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LeftGpu::OutOfMemory => "the GPU ran out of memory",
            LeftGpu::Lost => "the GPU device was lost twice during evaluation",
        })
    }
}

/// A device made after a loss: when, and after what wait.
#[derive(Clone, Copy, Debug)]
struct Recovered {
    at: Instant,
    delay: Duration,
}

#[derive(Debug)]
pub struct App {
    session: Session,
    gpu: Gpu,
    /// Where the worker evaluates: where the artist chose, or the CPU once the GPU failed it.
    device: EvaluationDevice,
    /// Why evaluation left the GPU, until the artist next chooses a device.
    left_gpu: Option<LeftGpu>,
    /// Losses while evaluation had work on the GPU since the artist last chose it.
    losses_during_evaluation: u32,
    /// Nothing is delivered once the app has dropped.
    worker: Worker,
    /// Hands each scan's result to the GUI thread.
    waker: Waker,
    /// Sequences being scanned on threads of their own, by path, each with whether it is to be
    /// scanned again when its scan returns, as Reload files asks while one runs.
    scanning: HashMap<Arc<Path>, bool>,
    /// Sequences whose scan failed, which are not scanned again until reloaded.
    unscannable: BTreeSet<Arc<Path>>,
    /// The session's version when the app last looked for sequences to scan.
    scans_followed: Option<u64>,
    /// Files chosen, in the order chosen, waiting for scans.
    choices: Vec<Choice>,
    /// What the viewer last wanted, which the worker was asked for unless it was paused.
    sent: Option<Wanted>,
    /// Whether the worker was paused instead, while the viewer shows the cached copy of `sent`.
    paused: bool,
    /// The session's version and the viewed node when the viewer last followed them.
    followed: Option<(u64, Option<NodeId>)>,
    /// How many requests the worker was sent, which numbers them.
    requests_sent: u64,
    drawing: Drawing,
    /// How many attempts to make a device were requested, which numbers them.
    attempts_requested: u64,
    /// The pane under the pointer, which receives keys whichever pane has the keyboard.
    hovered: Option<Pane>,
    /// The pointer's last position over the graph pane.
    graph_pointer: Option<PhysicalPoint>,
    shown: Shown,
    /// The request whose image the viewer's last presented frame showed.
    presented: Option<u64>,
    /// Images given to the GPU to show, which numbers them for their pixel reads.
    images_shown: u64,
    /// The pointer's last position over the viewer pane.
    viewer_pointer: Option<PhysicalPoint>,
    /// The image pixel the pointer last left the viewer from, which the readout stays on.
    left_pixel: Option<[i32; 2]>,
    /// The probe last asked of the GPU.
    probed: Option<Probe>,
    /// The last pixel read back, shown until a newer read arrives.
    last_read: Option<LastRead>,
    /// The readout as last requested.
    readout: String,
    /// The viewer's exposure as its control last showed it.
    exposure_shown: f32,
    /// The viewer's caption as last requested.
    caption: String,
    /// The Edit menu's undo and redo labels as last requested.
    undo_labels: [Option<&'static str>; 2],
    /// The status bar's GPU label as last requested.
    gpu_label: Option<String>,
    /// The node the parameter panel stays on instead of the current node.
    pinned: Option<NodeId>,
    /// The session's version and the pin when the panel last followed them.
    panel_followed: Option<(u64, Option<NodeId>)>,
    /// The parameter panel as last requested.
    panel: Option<Panel>,
    /// Playback in progress.
    clock: Option<Clock>,
    /// The mode playback plays in.
    playback_mode: PlaybackMode,
    /// Whether playback wraps round at the range's end.
    looping: bool,
    /// Which frames the timebar spans and playback plays through.
    frame_range: FrameRange,
    /// Whether a [`Request::Advance`] is waiting to be dispatched.
    advance_requested: bool,
    /// The keys of the frames the worker was last asked to prepare.
    ahead_sent: Vec<ResultKey>,
    /// What is known of each frame of the viewed node.
    marks: Marks,
    /// The timebar as last requested.
    timebar: Option<Timebar>,
}

/// How many frames ahead of playback's the worker prepares, at most: a provisional depth, which
/// bounds the work done for frames a stop then makes unneeded.
const LOOK_AHEAD: i64 = 24;

/// The viewed node's key at each frame of the timeline, and what the timebar shows of each.
#[derive(Debug, Default)]
struct Marks {
    /// What the keys were built from: the session's content version, the viewed node and the
    /// timeline's frames.
    built_for: Option<(u64, Option<NodeId>, Frames)>,
    keys: Vec<Option<ResultKey>>,
    marks: Vec<FrameMark>,
    /// What the marks' `cached` was learned from: the viewer cache's changes, and the zoom.
    cached_for: Option<(u64, Zoom)>,
}

impl App {
    /// An app evaluating on `device`. `wake` receives each [`Notice`] on the thread that raised
    /// it, which may hold wgpu's locks, and must hand it to the GUI thread to dispatch.
    pub fn new(device: EvaluationDevice, wake: impl Fn(Notice) + Send + Sync + 'static) -> App {
        let waker = Waker(Arc::new(wake));
        let deliver = waker.clone();
        App {
            session: Session::new(),
            gpu: Gpu::new(waker.clone()),
            device,
            left_gpu: None,
            losses_during_evaluation: 0,
            worker: Worker::start(
                device,
                {
                    let deliver = deliver.clone();
                    move |evaluated| deliver.wake(Notice::Evaluated(evaluated))
                },
                move |copy| deliver.wake(Notice::Copied(copy)),
            ),
            waker,
            scanning: HashMap::new(),
            unscannable: BTreeSet::new(),
            scans_followed: None,
            choices: Vec::new(),
            sent: None,
            paused: false,
            followed: None,
            requests_sent: 0,
            drawing: Drawing::Ready { recovered: None },
            attempts_requested: 0,
            hovered: None,
            graph_pointer: None,
            shown: Shown::Nothing,
            presented: None,
            images_shown: 0,
            viewer_pointer: None,
            left_pixel: None,
            probed: None,
            last_read: None,
            readout: String::new(),
            exposure_shown: 0.0,
            caption: String::new(),
            undo_labels: [None, None],
            gpu_label: None,
            pinned: None,
            panel_followed: None,
            panel: None,
            clock: None,
            playback_mode: PlaybackMode::default(),
            looping: true,
            frame_range: FrameRange::default(),
            advance_requested: false,
            ahead_sent: Vec::new(),
            marks: Marks::default(),
            timebar: None,
        }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Edits the session, as the parameter panel does, then follows the change as
    /// [`App::dispatch`] does. Returns the requests, and what the edit returned.
    pub fn edit<R>(&mut self, edit: impl FnOnce(&mut Session) -> R) -> (Vec<Request>, R) {
        let edited = edit(&mut self.session);
        let requests = self.after_change(vec![Request::Redraw(Pane::Graph)]);
        (requests, edited)
    }

    /// The number of the latest request to the worker, whose result shows the session as it is,
    /// for the drag check.
    pub fn latest_request(&self) -> u64 {
        self.requests_sent
    }

    /// Plays the timeline from the session's frame, that way, in the mode, from the first
    /// [`advance`](App::advance) on. A frame outside the timeline plays from the end it plays
    /// from.
    pub fn play(&mut self, direction: Direction, mode: PlaybackMode) -> Vec<Request> {
        self.playback_mode = mode;
        let requests = self.start_playing(direction);
        self.after_change(requests)
    }

    /// Stops playback, staying on the frame.
    pub fn stop(&mut self) -> Vec<Request> {
        self.dispatch(Event::Command(Command::Stop))
    }

    /// Moves playback's clock to `now`, as the shell does when [`Request::Advance`] asks.
    pub fn advance(&mut self, now: Instant) -> Vec<Request> {
        self.dispatch(Event::Advance(now))
    }

    /// Goes to a frame, stopping playback, as the artist does in the timebar.
    pub fn set_frame(&mut self, frame: f64) -> Vec<Request> {
        self.dispatch(Event::FrameSet(frame))
    }

    /// Which way and how playback plays, if it does.
    pub fn playing(&self) -> Option<(Direction, PlaybackMode)> {
        let clock = self.clock.as_ref()?;
        Some((clock.direction(), clock.mode()))
    }

    /// The timebar as last shown.
    pub fn timebar(&self) -> Option<&Timebar> {
        self.timebar.as_ref()
    }

    pub fn dispatch(&mut self, event: Event) -> Vec<Request> {
        let requests = match event {
            // Not while the pane has no surface: after a loss, only the retry makes a device.
            Event::Exposed(pane) if !self.gpu.has_window(pane) => {
                vec![Request::AttachSurface(pane)]
            }
            Event::Exposed(pane) | Event::Draw(pane) => self.draw(pane),
            Event::SurfaceCreated { pane, window, size } => {
                self.surface_created(pane, window, size)
            }
            Event::SurfaceDestroyed(pane) => {
                self.gpu.detach(pane);
                Vec::new()
            }
            Event::Resized { pane, size } => {
                self.gpu.resize(pane, size);
                self.resize(pane, size);
                vec![Request::Redraw(pane)]
            }
            Event::Input { pane, input } => self.input(pane, input),
            Event::KeyPressed { key, modifiers } => self.key_pressed(key, modifiers),
            Event::KeyReleased(key) => self.key_released(key),
            Event::Command(command) => self.command(command),
            Event::NodeTypeChosen { type_name, anchor } => self.node_type_chosen(type_name, anchor),
            // Another pane's focus loss can arrive after a press has given the graph the
            // keyboard, so it must not cancel the gesture that press began.
            Event::FocusLost(Pane::Graph) => {
                self.session.cancel_graph_gesture();
                vec![Request::Redraw(Pane::Graph)]
            }
            Event::FocusLost(Pane::Viewer) => {
                self.session.viewer_mut().cancel();
                Vec::new()
            }
            Event::FilesChosen { intent, paths } => self.files_chosen(intent, paths),
            Event::ReloadFiles(node) => self.reload(&[node]),
            Event::Scanned { sequence, scan } => self.scanned(sequence, scan),
            Event::Evaluated(evaluated) => self.evaluated(evaluated),
            Event::Copied(copy) => {
                self.session.viewer_cache_mut().insert(copy);
                // Look-ahead may have prepared the frame the viewer waits for.
                let mut requests = Vec::new();
                if !self.paused && !self.shows_sent() {
                    self.show_cached(&mut requests);
                }
                requests
            }
            Event::DeviceLost { generation, at } => self.device_lost(generation, at),
            Event::RetryDeviceCreation(attempt) => match self.drawing {
                Drawing::Unavailable { attempt: last, .. } if attempt == last => {
                    let mut requests = self.make_device();
                    for pane in Pane::ALL {
                        if self.gpu.has_surface(pane) {
                            requests.push(Request::Redraw(pane));
                        }
                    }
                    requests
                }
                _ => Vec::new(),
            },
            Event::CollectPixelRead => {
                if let Some(read) = self.gpu.take_pixel_read()
                    && let Some(info) = self.shown.framed()
                {
                    self.last_read = Some(LastRead::new(read, info.data_window));
                }
                self.poll_pixel_read()
            }
            Event::ViewerExposureSet(stops) => {
                self.session.viewer_mut().set_exposure(stops);
                // The control shows what the artist typed, which echoing could rewrite.
                self.exposure_shown = self.session.viewer().exposure();
                vec![Request::Redraw(Pane::Viewer)]
            }
            Event::Advance(now) => self.advance_clock(now),
            Event::FrameSet(frame) => {
                self.clock = None;
                self.go_to(frame)
            }
            Event::FrameRangeSet { first, last } => {
                match self.session.set_frame_range(first, last) {
                    Ok(_) => Vec::new(),
                    Err(error) => vec![Request::ShowStatus(format!(
                        "Cannot set the frame range: {error}"
                    ))],
                }
            }
            Event::FrameRateSet(rate) => match self.session.set_frame_rate(rate) {
                Ok(_) => Vec::new(),
                Err(error) => vec![Request::ShowStatus(format!(
                    "Cannot set the frame rate: {error}"
                ))],
            },
            Event::PlaybackModeSet(mode) => {
                self.playback_mode = mode;
                match &self.clock {
                    Some(clock) if clock.mode() != mode => {
                        let direction = clock.direction();
                        self.start_playing(direction)
                    }
                    _ => Vec::new(),
                }
            }
            Event::LoopingSet(looping) => {
                self.looping = looping;
                Vec::new()
            }
            Event::FrameRangeChosen(range) => {
                self.frame_range = range;
                Vec::new()
            }
        };
        self.after_change(requests)
    }

    fn surface_created(
        &mut self,
        pane: Pane,
        window: NativeWindow,
        size: PaneSize,
    ) -> Vec<Request> {
        if let Err(error) = self.gpu.attach(pane, window, size) {
            return vec![Request::ShowError(error)];
        }
        let mut requests = Vec::new();
        match self.drawing {
            Drawing::Ready { .. } if !self.gpu.has_device() => {
                requests = self.make_device();
            }
            Drawing::Ready { .. } => {}
            // The new window may be what the last attempt lacked, so the next one does not wait
            // out the backoff.
            Drawing::Unavailable { .. } => {
                requests.push(self.request_attempt(FIRST_RETRY));
            }
        }
        self.resize(pane, size);
        requests.extend(self.draw(pane));
        requests
    }

    /// Follows whatever the event or edit changed: the viewer, then the panel and the labels the
    /// shell shows.
    fn after_change(&mut self, mut requests: Vec<Request>) -> Vec<Request> {
        self.scan_unscanned();
        self.follow_marks();
        self.follow_viewer(&mut requests);
        self.follow_playback(&mut requests);
        self.follow_look_ahead();
        self.follow_timebar(&mut requests);
        self.follow_readout(&mut requests);
        let exposure = self.session.viewer().exposure();
        if exposure != self.exposure_shown {
            self.exposure_shown = exposure;
            requests.push(Request::ShowViewerExposure(exposure));
        }
        // These go first, since performing another request may dispatch again and show newer
        // ones.
        self.follow_panel(&mut requests);
        let undo_labels = [self.session.undo_label(), self.session.redo_label()];
        if undo_labels != self.undo_labels {
            self.undo_labels = undo_labels;
            let [undo, redo] = undo_labels;
            requests.insert(0, Request::ShowUndoLabels { undo, redo });
        }
        let gpu_label = self.gpu_label();
        if gpu_label != self.gpu_label {
            self.gpu_label.clone_from(&gpu_label);
            requests.insert(0, Request::ShowGpuLabel(gpu_label));
        }
        requests
    }

    /// Keeps the panel on the pinned node, or the current node: shows it when it changed. A pin
    /// on a deleted node is dropped, so undoing the delete does not pin it again.
    fn follow_panel(&mut self, requests: &mut Vec<Request>) {
        let document = self.session.document();
        if self
            .pinned
            .is_some_and(|pinned| document.node(pinned).is_none())
        {
            self.pinned = None;
        }
        // Building the panel clones the node's values, so not on every pointer move.
        let followed = Some((self.session.version(), self.pinned));
        if followed == self.panel_followed {
            return;
        }
        self.panel_followed = followed;
        let (files, frame) = (self.session.files(), self.session.frame());
        let panel = match self.pinned {
            Some(pinned) => Panel::of(document, files, pinned, true, frame),
            None => (document.selection().last())
                .and_then(|&current| Panel::of(document, files, current, false, frame)),
        };
        if panel != self.panel {
            self.panel.clone_from(&panel);
            requests.insert(0, Request::ShowPanel(panel));
        }
    }

    /// The parameter panel as last shown.
    pub fn panel(&self) -> Option<&Panel> {
        self.panel.as_ref()
    }

    /// What the viewer shows, if anything: the frame of the image, which may not be the frame
    /// asked for while the asked-for one is evaluated.
    pub fn displayed(&self) -> Option<Displayed> {
        let wanted = self.shown.wanted()?;
        Some(Displayed {
            node: wanted.node,
            frame: wanted.frame,
            requested: self.session.frame(),
            failed: matches!(self.shown, Shown::Failed { .. }),
            cached: match &self.shown {
                Shown::Copy { copy, .. } => Some(copy.halvings()),
                Shown::Nothing | Shown::Image { .. } | Shown::Failed { .. } => None,
            },
        })
    }

    /// Whether the panes cannot draw, and how many GPU errors there were, which the status bar
    /// shows for as long as either holds.
    fn gpu_label(&self) -> Option<String> {
        let unavailable = matches!(self.drawing, Drawing::Unavailable { .. });
        let label = match (unavailable, self.gpu.errors()) {
            (false, 0) => return None,
            (true, 0) => GPU_UNAVAILABLE.to_owned(),
            (false, errors) => gpu_errors(errors),
            (true, errors) => format!("{GPU_UNAVAILABLE}. {}", gpu_errors(errors)),
        };
        Some(label)
    }

    /// Retires the lost device, which stops the panes drawing, and requests the first attempt to
    /// replace it. A device lost before it worked for the longest wait counts as a failed
    /// attempt, so that a device lost again at once is not rebuilt at once forever. Evaluation
    /// stays on the GPU, since nothing could show a CPU result meanwhile, unless this loss makes
    /// too many while it had work there.
    fn device_lost(&mut self, generation: u64, at: Instant) -> Vec<Request> {
        if !self.gpu.retire_lost_device(generation) {
            return Vec::new();
        }
        // Read from the lost device's copy of the image, which the next device shows anew.
        self.last_read = None;
        self.probed = None;
        if self.worker.disconnect(at) && self.device == EvaluationDevice::Gpu {
            self.losses_during_evaluation += 1;
            if self.losses_during_evaluation >= LOSSES_BEFORE_LEAVING_THE_GPU {
                self.leave_gpu(LeftGpu::Lost);
            }
        }
        let delay = match self.drawing {
            Drawing::Ready {
                recovered: Some(recovered),
            } if recovered.at.elapsed() < LONGEST_RETRY => after_failure(recovered.delay),
            Drawing::Ready { .. } | Drawing::Unavailable { .. } => FIRST_RETRY,
        };
        vec![self.request_attempt(delay)]
    }

    /// Moves evaluation to the CPU, which evaluates the last request again, saying why.
    fn leave_gpu(&mut self, why: LeftGpu) {
        self.left_gpu = Some(why);
        self.evaluate_on(EvaluationDevice::Cpu);
    }

    /// Evaluates on the device from the next request. The viewer cache's copies are dropped,
    /// and a copy shown is evaluated anew, since results never mix devices.
    fn evaluate_on(&mut self, device: EvaluationDevice) {
        self.device = device;
        self.session.viewer_cache_mut().clear();
        self.worker.evaluate_on(device);
        if self.paused {
            self.resume();
        }
    }

    /// Makes a device for every pane with a native window, which the worker then uses. On
    /// failure, requests the next attempt. Why the first attempt failed is shown, and why a
    /// later one did is logged, so that retrying does not churn the status bar.
    fn make_device(&mut self) -> Vec<Request> {
        match self.gpu.make_device() {
            Ok(new) => {
                (self.worker).connect(new.device, new.queue, new.lost, new.generation);
                // The paused worker shows nothing again, so the copy is shown anew.
                if let Shown::Copy { copy, .. } = &self.shown {
                    self.images_shown += 1;
                    self.gpu.show_copy(copy, self.images_shown);
                }
                let mut requests = Vec::new();
                if !new.pane_errors.is_empty() {
                    requests.push(Request::ShowError(new.pane_errors.join("; ")));
                }
                if let Drawing::Unavailable { delay, .. } = self.drawing {
                    let at = Instant::now();
                    self.drawing = Drawing::Ready {
                        recovered: Some(Recovered { at, delay }),
                    };
                }
                requests
            }
            Err(error) => match self.drawing {
                Drawing::Ready { .. } => vec![
                    Request::ShowError(error),
                    self.request_attempt(RETRY_AFTER_FAILURE),
                ],
                Drawing::Unavailable { delay, .. } => {
                    eprintln!("the GPU device could not be made again: {error}");
                    vec![self.request_attempt(after_failure(delay))]
                }
            },
        }
    }

    /// Requests an attempt to make a device after the delay, replacing any requested before.
    fn request_attempt(&mut self, delay: Duration) -> Request {
        self.attempts_requested += 1;
        let attempt = self.attempts_requested;
        self.drawing = Drawing::Unavailable { attempt, delay };
        Request::RetryDeviceCreation {
            attempt,
            after: delay,
        }
    }

    fn draw(&mut self, pane: Pane) -> Vec<Request> {
        let viewer = self.session.viewer();
        let frame = self.shown.framed().map(|info| {
            let (data_window, pixels_per_texel) = match &self.shown {
                Shown::Copy { copy, .. } => {
                    let window = copy.window;
                    let halvings = copy.halvings();
                    let covered = Window {
                        x: window.x << halvings,
                        y: window.y << halvings,
                        width: window.width << halvings,
                        height: window.height << halvings,
                    };
                    (covered, 1 << halvings)
                }
                _ => (info.data_window, 1),
            };
            ViewerFrame {
                placement: viewer.placement(info.display_window),
                display_window: info.display_window,
                data_window,
                scale_factor: viewer.scale_factor(),
                gain: viewer.gain(),
                pixels_per_texel,
            }
        });
        match self
            .gpu
            .draw(pane, self.session.graph(), self.session.graph_view(), frame)
        {
            Drawn::Presented => {
                if pane == Pane::Viewer {
                    self.presented = match &self.shown {
                        Shown::Image { wanted, .. } | Shown::Copy { wanted, .. } => {
                            Some(wanted.number)
                        }
                        Shown::Nothing | Shown::Failed { .. } => None,
                    };
                }
                Vec::new()
            }
            Drawn::Skipped => Vec::new(),
            Drawn::Again => vec![Request::Redraw(pane)],
        }
    }

    fn resize(&mut self, pane: Pane, size: PaneSize) {
        match pane {
            Pane::Graph => self.session.resize_graph_view(size.viewport()),
            Pane::Viewer => self.session.viewer_mut().resize(size),
        }
    }

    fn input(&mut self, pane: Pane, input: Input) -> Vec<Request> {
        match input {
            Input::PointerEntered => self.hovered = Some(pane),
            Input::PointerLeft if self.hovered == Some(pane) => self.hovered = None,
            _ => {}
        }
        match pane {
            Pane::Graph => self.graph_input(input),
            Pane::Viewer => self.viewer_input(input),
        }
    }

    fn graph_input(&mut self, input: Input) -> Vec<Request> {
        let event = match input {
            Input::PointerEntered => return Vec::new(),
            Input::PointerLeft => {
                self.graph_pointer = None;
                InputEvent::PointerLeft
            }
            Input::PointerMoved { position } => {
                self.graph_pointer = Some(position);
                InputEvent::PointerMoved { position }
            }
            Input::PointerPressed {
                position,
                button,
                modifiers,
            } => InputEvent::PointerPressed {
                position,
                button,
                modifiers,
            },
            Input::PointerReleased { position, button } => {
                InputEvent::PointerReleased { position, button }
            }
            Input::Scrolled { position, lines } => InputEvent::Scrolled { position, lines },
        };
        redraw_if(Pane::Graph, self.session.graph_input(event))
    }

    fn viewer_input(&mut self, input: Input) -> Vec<Request> {
        match input {
            Input::PointerMoved { position }
            | Input::PointerPressed { position, .. }
            | Input::PointerReleased { position, .. }
            | Input::Scrolled { position, .. } => self.viewer_pointer = Some(position),
            Input::PointerLeft => {
                self.left_pixel = self.pointed().map(|(pixel, _)| pixel);
                self.viewer_pointer = None;
            }
            Input::PointerEntered => {}
        }
        let display = self.shown.display_window();
        let viewer = self.session.viewer_mut();
        let moved = match input {
            Input::PointerPressed {
                position, button, ..
            } => {
                viewer.pressed(position, button);
                false
            }
            Input::PointerReleased { button, .. } => {
                viewer.released(button);
                false
            }
            Input::PointerMoved { position } => {
                display.is_some_and(|display| viewer.pointer_moved(position, display))
            }
            Input::Scrolled { position, lines } => match display {
                Some(display) => {
                    viewer.scrolled(position, lines, display);
                    true
                }
                None => false,
            },
            Input::PointerEntered | Input::PointerLeft => false,
        };
        redraw_if(Pane::Viewer, moved)
    }

    fn key_pressed(&mut self, key: Key, modifiers: Modifiers) -> Vec<Request> {
        if let Key::Slot(slot) = key {
            return self.slot_key(slot, modifiers);
        }
        match (self.hovered, key.in_graph()) {
            (Some(Pane::Graph), _) if key == Key::Delete => self.delete_selected(),
            (Some(Pane::Graph), _) if key == Key::Tab => {
                vec![Request::ShowCreationMenu(self.anchor())]
            }
            (Some(Pane::Graph), Some(key)) => redraw_if(
                Pane::Graph,
                self.session.graph_input(InputEvent::KeyPressed(key)),
            ),
            (Some(Pane::Viewer), _) => self.viewer_key(key, modifiers),
            _ => Vec::new(),
        }
    }

    fn delete_selected(&mut self) -> Vec<Request> {
        let selection = self.session.document().selection().to_vec();
        if selection.is_empty() {
            return vec![Request::ShowStatus(
                "Select nodes to delete them".to_owned(),
            )];
        }
        let deleted = self.session.delete_nodes(&selection);
        debug_assert!(deleted.is_ok(), "selected nodes exist");
        vec![Request::Redraw(Pane::Graph)]
    }

    fn viewer_key(&mut self, key: Key, modifiers: Modifiers) -> Vec<Request> {
        match key {
            Key::Space => {
                self.session.viewer_mut().hold_space(true);
                Vec::new()
            }
            Key::H if modifiers.shift => self.command(Command::ActualSize),
            Key::H => self.command(Command::FitImage),
            Key::E if modifiers.ctrl => self.command(Command::ResetExposure),
            Key::E if modifiers.shift => self.command(Command::ExposureDown),
            Key::E => self.command(Command::ExposureUp),
            Key::Left => self.command(Command::Step(Direction::Backward)),
            Key::Right => self.command(Command::Step(Direction::Forward)),
            Key::J => self.command(Command::Play(Direction::Backward)),
            Key::K => self.command(Command::Stop),
            Key::L => self.command(Command::Play(Direction::Forward)),
            _ => Vec::new(),
        }
    }

    /// A release goes to both panes whichever is under the pointer, so that a key held across
    /// panes is not left held.
    fn key_released(&mut self, key: Key) -> Vec<Request> {
        if key == Key::Space {
            self.session.viewer_mut().hold_space(false);
        }
        match key.in_graph() {
            Some(key) => redraw_if(
                Pane::Graph,
                self.session.graph_input(InputEvent::KeyReleased(key)),
            ),
            None => Vec::new(),
        }
    }

    fn slot_key(&mut self, slot: Slot, modifiers: Modifiers) -> Vec<Request> {
        match (modifiers.ctrl, modifiers.shift) {
            (false, false) => self.command(Command::ViewSlot(slot)),
            (true, false) => self.command(Command::MarkSlot(slot)),
            (false, true) => self.command(Command::FrameMark(slot)),
            (true, true) => Vec::new(),
        }
    }

    fn command(&mut self, command: Command) -> Vec<Request> {
        match command {
            Command::ReadFiles => vec![Request::ChooseFiles(FileIntent::CreateReaders(
                self.anchor(),
            ))],
            Command::PinPanel => match &self.panel {
                Some(panel) => {
                    self.pinned = Some(panel.node);
                    Vec::new()
                }
                None => vec![Request::ShowStatus("Select a node to pin it".to_owned())],
            },
            Command::UnpinPanel => {
                self.pinned = None;
                Vec::new()
            }
            Command::Reload => {
                let selection = self.session.document().selection().to_vec();
                self.reload(&selection)
            }
            Command::Undo => redraw_if(Pane::Graph, !self.session.undo().is_empty()),
            Command::Redo => redraw_if(Pane::Graph, !self.session.redo().is_empty()),
            Command::ViewSlot(slot) => {
                self.session.viewer_mut().show_slot(slot);
                vec![Request::Redraw(Pane::Viewer)]
            }
            Command::MarkSlot(slot) => match self.session.document().selection() {
                &[id] => {
                    let changes = self.session.set_mark(slot, Some(id));
                    debug_assert!(changes.is_ok(), "a selected node exists");
                    self.session.viewer_mut().show_slot(slot);
                    vec![Request::Redraw(Pane::Graph), Request::Redraw(Pane::Viewer)]
                }
                _ => vec![Request::ShowStatus("Select one node to mark it".to_owned())],
            },
            Command::FrameMark(slot) => {
                if self.session.frame_mark(slot) {
                    vec![Request::Redraw(Pane::Graph)]
                } else {
                    vec![Request::ShowStatus(format!(
                        "Slot {} marks nothing",
                        slot.digit()
                    ))]
                }
            }
            Command::FitImage | Command::ActualSize => {
                let Some(display) = self.shown.display_window() else {
                    return vec![Request::ShowStatus("The viewer shows no image".to_owned())];
                };
                let viewer = self.session.viewer_mut();
                if command == Command::FitImage {
                    viewer.fit();
                } else {
                    viewer.actual_size(display);
                }
                vec![Request::Redraw(Pane::Viewer)]
            }
            Command::ExposureUp | Command::ExposureDown | Command::ResetExposure => {
                let viewer = self.session.viewer_mut();
                let stops = match command {
                    Command::ExposureUp => viewer.exposure() + EXPOSURE_STEP,
                    Command::ExposureDown => viewer.exposure() - EXPOSURE_STEP,
                    _ => 0.0,
                };
                viewer.set_exposure(stops);
                vec![Request::Redraw(Pane::Viewer)]
            }
            Command::EvaluateOn(device) => {
                self.left_gpu = None;
                self.losses_during_evaluation = 0;
                if device != self.device {
                    self.evaluate_on(device);
                }
                Vec::new()
            }
            Command::Play(direction) => {
                if self
                    .playing()
                    .is_some_and(|(playing, _)| playing == direction)
                {
                    self.clock = None;
                    return Vec::new();
                }
                self.start_playing(direction)
            }
            Command::Stop => {
                self.clock = None;
                Vec::new()
            }
            Command::Step(direction) => {
                self.clock = None;
                let frame = self.session.frame();
                let frames = self.frames();
                let stepped = match frames.index(frame) {
                    Some(index) => match frames.step(index, 1, direction, self.looping) {
                        Ok(index) | Err(index) => frames.frame(index),
                    },
                    None => match direction {
                        Direction::Forward => frame.floor() + 1.0,
                        Direction::Backward => frame.ceil() - 1.0,
                    },
                };
                self.go_to(stepped)
            }
            Command::FirstFrame | Command::LastFrame => {
                self.clock = None;
                let (first, last, _) = self.range();
                self.go_to(match command {
                    Command::FirstFrame => first,
                    _ => last,
                })
            }
        }
    }

    /// Goes to a frame, or says why not.
    fn go_to(&mut self, frame: f64) -> Vec<Request> {
        match self.session.set_frame(frame) {
            Ok(()) => Vec::new(),
            Err(error) => vec![Request::ShowStatus(format!(
                "Cannot go to the frame: {error}"
            ))],
        }
    }

    /// Starts playing that way in the mode chosen, from the next advance, asked for at once.
    fn start_playing(&mut self, direction: Direction) -> Vec<Request> {
        self.clock = Some(Clock::new(direction, self.playback_mode));
        self.request_advance(Duration::ZERO).into_iter().collect()
    }

    /// Asks for an advance after the delay, unless one is waiting already.
    fn request_advance(&mut self, after: Duration) -> Option<Request> {
        if self.advance_requested {
            return None;
        }
        self.advance_requested = true;
        Some(Request::Advance { after })
    }

    /// Moves playback's clock to `now`, and the session to its frame. Asks for the next advance
    /// when the clock knows when; in every-frame mode, following the viewer asks once the frame
    /// is shown.
    fn advance_clock(&mut self, now: Instant) -> Vec<Request> {
        self.advance_requested = false;
        let frames = self.frames();
        let Some(clock) = &mut self.clock else {
            return Vec::new();
        };
        let tick = clock.advance(
            now,
            self.session.frame(),
            frames,
            self.session.document().timeline().rate(),
            self.looping,
        );
        if tick.stopped {
            self.clock = None;
        }
        let mut requests = match tick.frame {
            Some(frame) => self.go_to(frame),
            None => Vec::new(),
        };
        if let Some(next) = tick.next {
            let after = next.saturating_duration_since(Instant::now());
            requests.extend(self.request_advance(after));
        }
        requests
    }

    /// Tells the clock when its frame is shown, and in every-frame mode asks for the advance
    /// that steps once it is due.
    fn follow_playback(&mut self, requests: &mut Vec<Request>) {
        let shown = self.shows_sent()
            && (self.shown.wanted()).is_some_and(|shown| shown.frame == self.session.frame());
        let Some(clock) = &mut self.clock else {
            return;
        };
        if shown {
            clock.frame_shown(Instant::now());
        }
        if clock.mode() == PlaybackMode::EveryFrame && clock.is_shown() {
            requests.extend(self.request_advance(Duration::ZERO));
        }
    }

    /// Rebuilds the viewed node's key at every frame of the timeline, and what the timebar marks
    /// of each, when the document, the files, the viewed node or the timeline changed, but not
    /// during an edit in progress, which would rebuild them for every value a drag passes; and
    /// which the viewer cache serves when it or the zoom changed.
    fn follow_marks(&mut self) {
        let session = &self.session;
        let (version, viewed, frames) = self.marks_for();
        let built_for = Some((version, viewed, frames));
        let marks = &mut self.marks;
        if marks.built_for != built_for && session.editing().is_none() {
            marks.built_for = built_for;
            marks.keys.clear();
            marks.marks.clear();
            marks.cached_for = None;
            if let Some(node) = viewed
                && frames.count() <= MAX_MARKED_FRAMES
            {
                let (document, files) = (session.document(), session.files());
                for index in 0..frames.count() {
                    let frame = frames.frame(index);
                    marks.keys.push(result_key(document, files, node, frame));
                    let lacks = missing(document, files, node, frame);
                    marks.marks.push(FrameMark {
                        cached: false,
                        missing: (lacks.iter()).any(|(_, what)| matches!(what, Missing::Frame(_))),
                        held: !held(document, files, node, frame).is_empty(),
                    });
                }
            }
        }
        let cache = session.viewer_cache();
        let zoom = session.viewer().zoom();
        let cached_for = Some((cache.changes(), zoom));
        if marks.cached_for != cached_for {
            marks.cached_for = cached_for;
            for (mark, key) in marks.marks.iter_mut().zip(&marks.keys) {
                mark.cached = key.as_ref().is_some_and(|key| cache.serves(key, zoom));
            }
        }
    }

    /// What the marks are built from as the session stands: its content version, the viewed node
    /// and the range's frames.
    fn marks_for(&self) -> (u64, Option<NodeId>, Frames) {
        let session = &self.session;
        (session.content_version(), session.viewed(), self.frames())
    }

    /// The first and last frame of the range chosen, and with [`FrameRange::Input`] whether the
    /// viewed node's was found, without which it is the timeline's.
    fn range(&self) -> (f64, f64, bool) {
        let session = &self.session;
        let timeline = session.document().timeline();
        let input = match (self.frame_range, session.viewed()) {
            (FrameRange::Input, Some(node)) => {
                input_range(session.document(), session.files(), node, session.frame())
            }
            _ => None,
        };
        match input {
            Some((first, last)) => (first, last, true),
            None => (timeline.first(), timeline.last(), false),
        }
    }

    /// The frames of the range chosen, which playback steps through.
    fn frames(&self) -> Frames {
        let (first, last, _) = self.range();
        Frames::between(first, last)
    }

    /// Asks the worker to prepare the frames playback shows next that the viewer cache cannot
    /// serve, while playing; and for nothing otherwise.
    fn follow_look_ahead(&mut self) {
        let ahead = self.look_ahead();
        if (ahead.iter().map(|wanted| &wanted.key)).ne(self.ahead_sent.iter()) {
            self.ahead_sent = ahead.iter().map(|wanted| wanted.key.clone()).collect();
            self.worker.look_ahead(ahead);
        }
    }

    /// The frames after the session's that playback shows next, in order, up to the look-ahead
    /// depth, with neither a copy to serve them nor a request for them. None while the viewer
    /// cache is full: the copy of a frame ahead would drop the least recently requested, which is
    /// the next frame not yet played, and preparing that would drop the one after, each frame
    /// prepared again on the device as playback reaches it.
    fn look_ahead(&self) -> Vec<Wanted> {
        let (Some(clock), Some(node)) = (&self.clock, self.session.viewed()) else {
            return Vec::new();
        };
        let session = &self.session;
        if !session.viewer_cache().has_room() {
            return Vec::new();
        }
        let frames = self.frames();
        let Some(current) = frames.index(session.frame()) else {
            return Vec::new();
        };
        let (cache, zoom) = (session.viewer_cache(), session.viewer().zoom());
        // The marks' keys, unless an edit in progress has left them behind.
        let marked = (self.marks.built_for == Some(self.marks_for())).then_some(&self.marks.keys);
        let mut ahead: Vec<Wanted> = Vec::new();
        for steps in 1..=LOOK_AHEAD.min(frames.count() - 1) {
            let Ok(index) = frames.step(current, steps, clock.direction(), self.looping) else {
                break;
            };
            let frame = frames.frame(index);
            let key = match marked.and_then(|keys| keys.get(index as usize)) {
                Some(key) => key.clone(),
                None => result_key(session.document(), session.files(), node, frame),
            };
            let Some(key) = key else {
                continue;
            };
            let requested = (self.sent.as_ref()).is_some_and(|sent| sent.key == key);
            if requested || cache.serves(&key, zoom) || ahead.iter().any(|wanted| wanted.key == key)
            {
                continue;
            }
            ahead.push(Wanted {
                node,
                key,
                frame,
                number: self.requests_sent,
                copy: Some(zoom),
            });
        }
        ahead
    }

    /// Shows the timebar again when anything it shows changed.
    fn follow_timebar(&mut self, requests: &mut Vec<Request>) {
        let (first, last, input_found) = self.range();
        let clock = self.clock.as_ref();
        let timebar = Timebar {
            first,
            last,
            range: self.frame_range,
            input_found,
            rate: self.session.document().timeline().rate(),
            frame: self.session.frame(),
            playing: clock.map(Clock::direction),
            mode: self.playback_mode,
            looping: self.looping,
            achieved: (clock.and_then(Clock::achieved_rate))
                .map(|rate| (rate * 10.0).round() / 10.0),
            drops: clock.map_or(0, Clock::drops),
            marks: self.marks.marks.clone(),
        };
        if self.timebar.as_ref() != Some(&timebar) {
            self.timebar = Some(timebar.clone());
            requests.insert(0, Request::ShowTimebar(timebar));
        }
    }

    fn anchor(&self) -> Anchor {
        let view = self.session.graph_view();
        let viewport = view.viewport();
        let pointer = self.graph_pointer.unwrap_or(PhysicalPoint {
            x: viewport.width as f32 / 2.0,
            y: viewport.height as f32 / 2.0,
        });
        Anchor {
            at: view.to_graph(pointer),
            current: self.session.document().selection().last().copied(),
        }
    }

    fn node_type_chosen(&mut self, type_name: &'static str, anchor: Anchor) -> Vec<Request> {
        // A Reader without a file is no use to the artist, so the dialog comes first, and
        // cancelling it creates nothing.
        if type_name == "reader" {
            return vec![Request::ChooseFiles(FileIntent::CreateReaders(anchor))];
        }
        match (self.session).create_node(type_name, anchor.at, anchor.current) {
            Ok(_) => vec![Request::Redraw(Pane::Graph)],
            Err(error) => vec![Request::ShowStatus(format!(
                "Cannot create the node: {error}"
            ))],
        }
    }

    /// Reload files on each Reader: stills at once, sequences once rescanned.
    fn reload(&mut self, nodes: &[NodeId]) -> Vec<Request> {
        let reload = self.session.reload(nodes);
        if reload.is_empty() {
            return vec![Request::ShowStatus("Select Readers to reload".to_owned())];
        }
        for sequence in reload.sequences {
            self.unscannable.remove(sequence.path());
            self.scan(sequence);
        }
        vec![Request::Redraw(Pane::Graph)]
    }

    /// Scans a sequence on a thread of its own, or again once the scan running returns, whose
    /// listing may predate a change.
    fn scan(&mut self, sequence: Pattern) {
        if let Some(again) = self.scanning.get_mut(sequence.path()) {
            *again = true;
            return;
        }
        self.scanning.insert(sequence.path().clone(), false);
        let waker = self.waker.clone();
        std::thread::Builder::new()
            .name("scan".to_owned())
            .spawn(move || {
                let scan = Scan::of(&sequence);
                waker.wake(Notice::Scanned { sequence, scan });
            })
            .expect("the system starts a thread");
    }

    /// Scans each sequence the document reads that has none, such as one typed into a File, once.
    fn scan_unscanned(&mut self) {
        let version = Some(self.session.version());
        if version == self.scans_followed {
            return;
        }
        self.scans_followed = version;
        for sequence in self.session.unscanned() {
            if !self.scanning.contains_key(sequence.path())
                && !self.unscannable.contains(sequence.path())
            {
                self.scan(sequence);
            }
        }
    }

    /// Records a finished scan, says how many frames it found changed or why it failed, and
    /// decides the files chosen that waited for it.
    fn scanned(&mut self, sequence: Pattern, scan: io::Result<Scan>) -> Vec<Request> {
        let again = self.scanning.remove(sequence.path()).unwrap_or(false);
        let mut requests = Vec::new();
        match scan {
            Ok(scan) => {
                self.unscannable.remove(sequence.path());
                if let Some(reloaded) = self.session.record_scan(&sequence, scan) {
                    requests.push(Request::Redraw(Pane::Graph));
                    if reloaded > 0 {
                        requests.push(Request::ShowStatus(format!(
                            "{}: {reloaded} {} changed on disk",
                            sequence.path().display(),
                            if reloaded == 1 { "frame" } else { "frames" }
                        )));
                    }
                }
            }
            Err(error) => {
                self.unscannable.insert(sequence.path().clone());
                requests.push(Request::ShowError(format!(
                    "Cannot scan {}: {error}",
                    sequence.directory().display()
                )));
            }
        }
        requests.extend(self.decide_choices());
        if again {
            self.scan(sequence);
        }
        requests
    }

    fn files_chosen(&mut self, intent: FileIntent, paths: Vec<PathBuf>) -> Vec<Request> {
        if paths.is_empty() {
            return Vec::new();
        }
        let files: Vec<(PathBuf, Option<Pattern>)> = (paths.into_iter())
            .map(|path| {
                let sequence = Pattern::candidate(&path);
                (path, sequence)
            })
            .collect();
        for sequence in files.iter().filter_map(|(_, sequence)| sequence.clone()) {
            if self.session.files().scan(sequence.path()).is_none() {
                self.scan(sequence);
            }
        }
        self.choices.push(Choice { intent, files });
        self.decide_choices()
    }

    /// Decides each file chosen whose sequence has been scanned, or failed to be: the sequence if
    /// the scan found it with another frame, otherwise the file. Then applies the choices with
    /// every file decided, in the order chosen, until one is still waiting.
    fn decide_choices(&mut self) -> Vec<Request> {
        let files = self.session.files();
        for choice in &mut self.choices {
            for (path, candidate) in &mut choice.files {
                let Some(sequence) = candidate.as_ref() else {
                    continue;
                };
                if self.scanning.contains_key(sequence.path()) {
                    continue;
                }
                let scan = files.scan(sequence.path());
                let is_sequence = scan.is_some_and(|scan| {
                    sequence
                        .frame_of(path)
                        .is_some_and(|frame| scan.contains(frame) && scan.frames().nth(1).is_some())
                });
                if is_sequence {
                    *path = sequence.path().to_path_buf();
                }
                *candidate = None;
            }
        }
        let mut requests = Vec::new();
        while let Some(choice) = self.choices.first() {
            if choice
                .files
                .iter()
                .any(|(_, candidate)| candidate.is_some())
            {
                break;
            }
            let choice = self.choices.remove(0);
            let paths = choice.files.into_iter().map(|(path, _)| path).collect();
            requests.extend(self.apply_choice(choice.intent, paths));
        }
        requests
    }

    fn apply_choice(&mut self, intent: FileIntent, paths: Vec<PathBuf>) -> Vec<Request> {
        let Some(first) = paths.first().cloned() else {
            return Vec::new();
        };
        match intent {
            FileIntent::CreateReaders(Anchor { at, current }) => {
                self.session.create_readers(paths, at, current);
            }
            FileIntent::SetFile { node, parameter } => {
                let file = Value::File(first.into());
                if let Err(error) = self.session.set_value(node, parameter, file) {
                    return vec![Request::ShowStatus(format!(
                        "Cannot change the file: {error}"
                    ))];
                }
            }
        }
        vec![Request::Redraw(Pane::Graph)]
    }

    /// Shows a finished evaluation of the node the viewer marks, even one an edit has superseded
    /// since, so that an edit faster than evaluation keeps the viewer updating. It is not shown if
    /// the image shown came from a later request, and how it ended is recorded only if it is
    /// shown or failed. A failure is shown only while its key is still wanted, so that an edit in
    /// progress never blanks the viewer. What is not shown is dropped, which returns its staging
    /// buffer to the worker.
    fn evaluated(&mut self, evaluated: Evaluated) -> Vec<Request> {
        let Evaluated {
            mut wanted, result, ..
        } = evaluated;
        // Not the node's failure, nor shown: the CPU evaluates the request again, and the last
        // frame stays until it does. One evaluated on the GPU before a move to the CPU may
        // still arrive.
        if let Err(ViewerError::Eval(EvalError::OutOfGpuMemory { .. })) = result {
            if self.device == EvaluationDevice::Gpu {
                self.leave_gpu(LeftGpu::OutOfMemory);
            }
            return Vec::new();
        }
        // Not the node's failure either: the device's lost callback has the next device show the
        // request again.
        if let Err(ViewerError::Eval(EvalError::GpuLost { .. })) = result {
            return Vec::new();
        }
        if let Err(ViewerError::Eval(error @ EvalError::GpuValidation { .. })) = &result {
            self.gpu.count_error(error);
        }
        // Not only later: connecting the GPU delivers the request shown again, with its pixels.
        let not_older = (self.shown.wanted()).is_none_or(|shown| wanted.number >= shown.number);
        if self.session.viewed() != Some(wanted.node) || !not_older {
            return Vec::new();
        }
        // What was sent is current: the viewer follows every change before this runs.
        let current = (self.sent.as_ref()).is_some_and(|sent| sent.key == wanted.key);
        // A current image is the current frame's, whichever frame asked for it.
        if let Some(sent) = self.sent.as_ref().filter(|_| current) {
            wanted.frame = sent.frame;
        }
        let graph_changed = match &result {
            Err(ViewerError::Eval(error)) => self.session.record_failure(error.origin()),
            // What the GPU cannot hold is the viewer's limitation, not the node's failure.
            Ok(_) | Err(ViewerError::CannotShow(_)) => self.session.record_success(&wanted.key),
        };
        if result.is_err() && !current {
            return redraw_if(Pane::Graph, graph_changed);
        }
        self.shown = match result {
            Ok(image) => {
                self.images_shown += 1;
                match image.pixels {
                    Some(pixels) => {
                        self.gpu
                            .show_image(pixels, image.info.data_window, self.images_shown)
                    }
                    None => self.gpu.clear_image(),
                }
                Shown::Image {
                    wanted,
                    info: image.info,
                }
            }
            Err(error) => {
                self.gpu.clear_image();
                Shown::Failed {
                    wanted,
                    error: error.to_string(),
                }
            }
        };
        let mut requests = vec![Request::Redraw(Pane::Viewer)];
        if graph_changed {
            requests.push(Request::Redraw(Pane::Graph));
        }
        requests
    }

    /// The node the viewer's slot marks, and its key at the current frame.
    fn wanted(&self) -> Option<(NodeId, ResultKey)> {
        let node = self.session.viewed()?;
        let session = &self.session;
        let key = result_key(session.document(), session.files(), node, session.frame())?;
        Some((node, key))
    }

    /// Keeps the viewer on what its slot marks: asks the worker for it, shows nothing if the slot
    /// marks nothing, and describes the result.
    fn follow_viewer(&mut self, requests: &mut Vec<Request>) {
        // Building a key walks the viewed node's whole upstream, so not on every pointer move.
        let followed = Some((self.session.version(), self.session.viewed()));
        if followed != self.followed {
            self.followed = followed;
            self.send_wanted(requests);
        }
        if self.paused && self.copy_falls_short() {
            self.resume();
        }
        let caption = self.caption();
        if caption != self.caption {
            self.caption.clone_from(&caption);
            requests.push(Request::ShowViewerCaption(caption));
        }
    }

    fn send_wanted(&mut self, requests: &mut Vec<Request>) {
        let wanted = self.wanted();
        if wanted.is_none() && !matches!(self.shown, Shown::Nothing) {
            self.shown = Shown::Nothing;
            self.gpu.clear_image();
            requests.push(Request::Redraw(Pane::Viewer));
        }
        // Compared with what was sent, not with what is shown: a key already shown is sent again
        // so that the worker releases what the last request retained, and stops working on it.
        // Nothing is sent too, which releases everything.
        let frame = self.session.frame();
        let sent = (self.sent.as_ref()).map(|sent| (sent.node, &sent.key));
        if wanted.as_ref().map(|(node, key)| (*node, key)) != sent {
            self.requests_sent += 1;
            let number = self.requests_sent;
            let Some((node, key)) = wanted else {
                self.worker.request(None);
                self.sent = None;
                self.paused = false;
                return;
            };
            let mut wanted = Wanted {
                node,
                key,
                frame,
                number,
                copy: None,
            };
            // A hit is shown at once, and the worker stops rather than evaluate it.
            self.sent = Some(wanted.clone());
            if !self.show_cached(requests) {
                wanted.copy = self.copy_wanted();
                self.worker.request(Some(wanted.clone()));
                self.paused = false;
                self.sent = Some(wanted);
            }
        } else if let Some(sent) = &mut self.sent {
            // Another frame with an equal key is not asked for: its image is the same, and what
            // is shown of the key is that frame's.
            sent.frame = frame;
            if let Shown::Image { wanted, .. }
            | Shown::Copy { wanted, .. }
            | Shown::Failed { wanted, .. } = &mut self.shown
                && wanted.key == sent.key
            {
                wanted.frame = frame;
            }
        }
    }

    /// Shows the viewer cache's copy of what was sent, if it holds one the viewer can show at its
    /// zoom, and pauses the worker rather than have it evaluate the image. Returns whether it did.
    fn show_cached(&mut self, requests: &mut Vec<Request>) -> bool {
        let Some(sent) = &self.sent else {
            return false;
        };
        let zoom = self.session.viewer().zoom();
        let Some(copy) = self.session.viewer_cache_mut().get(&sent.key, zoom) else {
            return false;
        };
        self.worker.pause();
        self.paused = true;
        self.images_shown += 1;
        self.gpu.show_copy(&copy, self.images_shown);
        self.shown = Shown::Copy {
            wanted: sent.clone(),
            copy,
        };
        requests.push(Request::Redraw(Pane::Viewer));
        true
    }

    /// Whether the viewer shows the result of what was last sent, an image, a copy or a failure.
    fn shows_sent(&self) -> bool {
        match (&self.sent, self.shown.wanted()) {
            (Some(sent), Some(shown)) => shown.key == sent.key,
            _ => false,
        }
    }

    /// Whether the cached copy shown falls short of the image: the readout points at one of its
    /// pixels, whose working values only the image has, or the viewer draws it larger than its
    /// texels, as after zooming in.
    fn copy_falls_short(&self) -> bool {
        let Shown::Copy { copy, .. } = &self.shown else {
            return false;
        };
        let zoom = self.session.viewer().zoom();
        let halvings = zoom.halvings(copy.info.display_window);
        let coarse = halvings.is_none_or(|halvings| copy.halvings() > halvings);
        coarse || matches!(self.pointed(), Some((_, Some(_))))
    }

    /// Asks the paused worker for what the viewer wants after all, under the same number, with a
    /// copy for the viewer cache if it has none to serve the viewer's zoom and no edit is in
    /// progress.
    fn resume(&mut self) {
        self.paused = false;
        let Some(sent) = &self.sent else {
            return;
        };
        let zoom = self.session.viewer().zoom();
        let served = self.session.viewer_cache().serves(&sent.key, zoom);
        let wanted = Wanted {
            copy: self.copy_wanted().filter(|_| !served),
            ..sent.clone()
        };
        self.worker.request(Some(wanted));
    }

    /// The zoom to copy the viewed result at for the viewer cache, unless an edit is in
    /// progress: each value a drag passes is stale by the next, and copying it cost the GPU
    /// session's GUI frames their budget.
    fn copy_wanted(&self) -> Option<Zoom> {
        (self.session.editing().is_none()).then(|| self.session.viewer().zoom())
    }

    /// Keeps the readout on the shown image's pixel under the pointer, or the one it left the
    /// viewer from: asks the GPU to read it, and describes the last read until a newer arrives.
    fn follow_readout(&mut self, requests: &mut Vec<Request>) {
        let pointed = self.pointed();
        if let Some((_, Some(probe))) = pointed
            && self.probed != Some(probe)
        {
            self.probed = Some(probe);
            let reading = self.gpu.reading_pixel();
            self.gpu.read_pixel(probe);
            // A read already in flight polls until the one queued behind it is done.
            if !reading {
                requests.extend(self.poll_pixel_read());
            }
        }
        let readout = match pointed {
            None => String::new(),
            Some(([x, y], None)) => {
                format!("x {x} y {y} · R 0 G 0 B 0 A 0, outside the data window")
            }
            Some((pixel, Some(probe))) => match self.last_read {
                Some(read) if read.image == probe.image && read.pixel == pixel => read.describe(),
                // Labelled with the pixel it read, which may be another, or of an older image.
                Some(read) => format!("{} · reading", read.describe()),
                None => format!("x {} y {}", pixel[0], pixel[1]),
            },
        };
        if readout != self.readout {
            self.readout.clone_from(&readout);
            requests.push(Request::ShowReadout(readout));
        }
    }

    /// Asks to collect the pixel read in flight once the GPU may have done it.
    fn poll_pixel_read(&self) -> Vec<Request> {
        if self.gpu.reading_pixel() {
            vec![Request::CollectPixelRead {
                after: PIXEL_READ_POLL,
            }]
        } else {
            Vec::new()
        }
    }

    /// The shown image's pixel the readout describes, in image pixels with y up, and the probe
    /// that reads it, or `None` for a pixel outside the data window, whose values are zero.
    fn pointed(&self) -> Option<([i32; 2], Option<Probe>)> {
        let info = self.shown.framed()?;
        let [x, y] = match self.viewer_pointer {
            // The pixel drawn at the centre of the physical pixel the pointer is in, where the
            // viewer's shader samples.
            Some(pointer) => {
                let centre = PhysicalPoint {
                    x: pointer.x.floor() + 0.5,
                    y: pointer.y.floor() + 0.5,
                };
                let placement = self.session.viewer().placement(info.display_window);
                placement.to_image(centre).map(|value| value.floor() as i32)
            }
            None => self.left_pixel?,
        };
        let window = info.data_window;
        let probe = window.contains(x, y).then(|| Probe {
            image: self.images_shown,
            texel: [(x - window.x) as u32, (y - window.y) as u32],
        });
        Some(([x, y], probe))
    }

    /// Names the viewed node from the document as it stands, then says what is shown of it, what
    /// upstream of it is not connected, and the device evaluating it. Called once the wanted key
    /// has been sent.
    fn caption(&self) -> String {
        let slot = self.session.viewer().slot().digit();
        let document = self.session.document();
        let Some(viewed) = self.session.viewed() else {
            // The next step, for an artist who has not found the menus yet.
            return if document.nodes().next().is_none() {
                format!(
                    "Slot {slot}: nothing marked. Read files with R, then mark one with Ctrl+{slot}"
                )
            } else {
                format!("Slot {slot}: nothing marked. Ctrl+{slot} marks the selected node")
            };
        };
        let node = document.node(viewed).expect("marks name existing nodes");
        let current = self.shows_sent();
        let mut details = Vec::new();
        match &self.shown {
            Shown::Image { info, .. } if current => {
                let window = info.display_window;
                if !window.is_empty() {
                    details.push(format!("{} × {}", window.width, window.height));
                }
                details.extend(info.warnings.iter().map(ToString::to_string));
            }
            Shown::Copy { copy, .. } if current => {
                let window = copy.info.display_window;
                if !window.is_empty() {
                    details.push(format!("{} × {}", window.width, window.height));
                }
                details.extend(copy.info.warnings.iter().map(ToString::to_string));
                details.push(match copy.halvings() {
                    0 => "cached".to_owned(),
                    halvings => format!("cached at 1/{}", 1u32 << halvings),
                });
            }
            Shown::Failed { error, .. } if current => details.push(error.clone()),
            // The previous image stays until then, and says which frame it is of.
            Shown::Image { wanted, .. } | Shown::Copy { wanted, .. }
                if wanted.frame != self.session.frame() =>
            {
                let frame = self.session.frame();
                details.push(format!("frame {} shown, {frame} updating…", wanted.frame));
            }
            _ => details.push("updating…".to_owned()),
        }
        let (files, frame) = (self.session.files(), self.session.frame());
        let name = |id| document.node(id).expect("upstream nodes exist").name();
        for (id, what) in missing(document, files, viewed, frame) {
            let (label, lack) = match what {
                Missing::Input(input) => (input.label, "is not connected"),
                Missing::File(parameter) => (parameter.label, "is not set"),
                // Reading it fails, and the failure names the file.
                Missing::Frame(_) => continue,
            };
            details.push(if id == viewed {
                format!("{label} {lack}")
            } else {
                format!("{label} of {} {lack}", name(id))
            });
        }
        for (id, read) in held(document, files, viewed, frame) {
            details.push(if id == viewed {
                format!("holds frame {read}")
            } else {
                format!("{} holds frame {read}", name(id))
            });
        }
        if details.is_empty() {
            details.push("empty".to_owned());
        }
        let device = match self.left_gpu {
            Some(why) => format!("{} ({why})", self.device),
            None => self.device.to_string(),
        };
        format!(
            "Slot {slot} · {} · {} · {device}",
            node.name(),
            details.join("; ")
        )
    }
}

/// What the shell's checks read and do, which the app itself has no use for.
#[doc(hidden)]
impl App {
    pub fn has_surface(&self, pane: Pane) -> bool {
        self.gpu.has_surface(pane)
    }

    /// Has every frame from now on record what it proves, for the lifecycle check. Each frame
    /// then waits for the GPU. Call it before the first surface is attached, so that every
    /// surface is made readable where it allows.
    pub fn check_frames(&mut self) {
        self.gpu.check_frames();
    }

    /// Has every frame from now on record when it passed each stage, for the drag check.
    pub fn time_frames(&mut self) {
        self.gpu.time_frames();
    }

    /// The pane's last frame since [`App::time_frames`]; `None` before one.
    pub fn frame_timing(&self, pane: Pane) -> Option<FrameTiming> {
        self.gpu.frame_timing(pane)
    }

    /// Runs the callbacks of the GPU's completed work without waiting for any, so that the drag
    /// check learns when frames complete while nothing else polls.
    pub fn poll_gpu(&self) {
        self.gpu.poll();
    }

    /// The device evaluating the session.
    pub fn evaluation_device(&self) -> EvaluationDevice {
        self.device
    }

    /// What the pane's frames have proved since [`App::check_frames`]; `None` before it.
    pub fn frame_check(&self, pane: Pane) -> Option<FrameCheck> {
        self.gpu.frame_check(pane)
    }

    /// How many GPU errors there were this session. Each is a bug, logged on standard error
    /// up to a limit, after which drawing continues.
    pub fn gpu_errors(&self) -> u64 {
        self.gpu.errors()
    }

    /// What the GPU's allocator holds, for the playback check.
    pub fn gpu_memory(&self) -> Option<GpuMemory> {
        self.gpu.memory()
    }

    /// Destroys the GPU device for the lifecycle check. Its loss then arrives as a driver's
    /// would, as [`Notice::DeviceLost`].
    pub fn destroy_device(&mut self) {
        self.gpu.destroy_device();
    }

    /// The number of the request whose image the viewer's last presented frame showed, for the
    /// drag check.
    pub fn presented_request(&self) -> Option<u64> {
        self.presented
    }
}

/// A pixel read back, at its pixel in image pixels, which a later image's data window would
/// not give from its texel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LastRead {
    image: u64,
    pixel: [i32; 2],
    values: [f32; 4],
}

impl LastRead {
    fn new(read: PixelRead, data_window: Window) -> LastRead {
        let [x, y] = read.probe.texel.map(|texel| texel as i32);
        LastRead {
            image: read.probe.image,
            pixel: [data_window.x + x, data_window.y + y],
            values: read.values,
        }
    }

    fn describe(&self) -> String {
        let [x, y] = self.pixel;
        let values = (["R", "G", "B", "A"].iter().zip(self.values))
            .map(|(channel, value)| format!("{channel} {}", working_value(value)))
            .collect::<Vec<_>>()
            .join(" ");
        format!("x {x} y {y} · {values}")
    }
}

const GPU_UNAVAILABLE: &str = "GPU unavailable: the panes draw again once it recovers";

/// Names the session's GPU errors, which are bugs, for the status bar.
fn gpu_errors(errors: u64) -> String {
    let noun = if errors == 1 { "error" } else { "errors" };
    format!("{errors} GPU {noun}, logged on standard error")
}

/// A working value for the readout: four decimals at most, without trailing zeros, or with an
/// exponent where that would hide it.
fn working_value(value: f32) -> String {
    let magnitude = value.abs();
    if value == 0.0 {
        "0".to_owned()
    } else if !value.is_finite() || (1e-4..1e6).contains(&magnitude) {
        let fixed = format!("{value:.4}");
        if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            fixed
        }
    } else {
        format!("{value:.3e}")
    }
}

/// The wait after a failed attempt that waited `delay`.
fn after_failure(delay: Duration) -> Duration {
    (delay * 2).clamp(RETRY_AFTER_FAILURE, LONGEST_RETRY)
}

fn redraw_if(pane: Pane, needed: bool) -> Vec<Request> {
    if needed {
        vec![Request::Redraw(pane)]
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests;
