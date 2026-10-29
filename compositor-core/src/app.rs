//! The interactive compositor: a session, the GPU that draws its panes, and the evaluator's
//! worker, changed only by [`App::dispatch`]. What only the shell can do comes back as [`Request`]s.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use open_ng::{Button, InputEvent, Modifiers, PhysicalPoint, Point};

use crate::document::{Missing, NodeId, Slot, missing};
use crate::evaluate::{EvalError, ResultKey, result_key};
use crate::gpu::{
    Drawn, FrameCheck, FrameTiming, Gpu, NativeWindow, PaneSize, PixelRead, Probe, ViewerFrame,
};
use crate::image::{ImageInfo, Window};
use crate::panel::Panel;
use crate::parameter::Value;
use crate::session::Session;
use crate::viewer::EXPOSURE_STEP;
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
    FilesChosen {
        intent: FileIntent,
        paths: Vec<PathBuf>,
    },
    /// An evaluation finished on the evaluator's thread.
    Evaluated(Evaluated),
    /// The GPU device of the generation was lost, when wgpu reported it.
    DeviceLost {
        generation: u64,
        at: Instant,
    },
    /// The time for the attempt, after [`Request::RetryDeviceCreation`].
    RetryDeviceCreation(u64),
    /// The artist set the viewer's exposure, in stops, in the viewer's own control.
    ViewerExposureSet(f32),
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
    DeviceLost { generation: u64, at: Instant },
}

impl From<Notice> for Event {
    fn from(notice: Notice) -> Event {
        match notice {
            Notice::Evaluated(evaluated) => Event::Evaluated(evaluated),
            Notice::DeviceLost { generation, at } => Event::DeviceLost { generation, at },
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
    /// viewed, at once if they are.
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
            Key::E | Key::Delete | Key::Tab | Key::Slot(_) => return None,
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

/// What the viewer shows.
#[derive(Debug)]
enum Shown {
    Nothing,
    Image {
        wanted: Wanted,
        info: ImageInfo,
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
            Shown::Image { wanted, .. } | Shown::Failed { wanted, .. } => Some(wanted),
        }
    }

    /// The image the viewer frames: none for an empty display window, which has no size to fit.
    fn framed(&self) -> Option<&ImageInfo> {
        match self {
            Shown::Image { info, .. } if !info.display_window.is_empty() => Some(info),
            Shown::Image { .. } | Shown::Nothing | Shown::Failed { .. } => None,
        }
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
    /// What the worker was last asked for.
    sent: Option<Wanted>,
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
    /// The probe last asked of the GPU.
    probed: Option<Probe>,
    /// The last pixel read back, shown while its image is.
    pixel_read: Option<PixelRead>,
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
}

impl App {
    /// An app evaluating on `device`. `wake` receives each [`Notice`] on the thread that raised
    /// it, which may hold wgpu's locks, and must hand it to the GUI thread to dispatch.
    pub fn new(device: EvaluationDevice, wake: impl Fn(Notice) + Send + Sync + 'static) -> App {
        let waker = Waker(Arc::new(wake));
        let deliver = waker.clone();
        App {
            session: Session::new(),
            gpu: Gpu::new(waker),
            device,
            left_gpu: None,
            losses_during_evaluation: 0,
            worker: Worker::start(device, move |evaluated| {
                deliver.wake(Notice::Evaluated(evaluated));
            }),
            sent: None,
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
            probed: None,
            pixel_read: None,
            readout: String::new(),
            exposure_shown: 0.0,
            caption: String::new(),
            undo_labels: [None, None],
            gpu_label: None,
            pinned: None,
            panel_followed: None,
            panel: None,
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
            Event::Evaluated(evaluated) => self.evaluated(evaluated),
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
            Event::ViewerExposureSet(stops) => {
                self.session.viewer_mut().set_exposure(stops);
                // The control shows what the artist typed, which echoing could rewrite.
                self.exposure_shown = self.session.viewer().exposure();
                vec![Request::Redraw(Pane::Viewer)]
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
        self.follow_viewer(&mut requests);
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
        let panel = match self.pinned {
            Some(pinned) => Panel::of(document, pinned, true),
            None => (document.selection().last())
                .and_then(|&current| Panel::of(document, current, false)),
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
        self.pixel_read = None;
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
        self.device = EvaluationDevice::Cpu;
        self.left_gpu = Some(why);
        self.worker.evaluate_on(EvaluationDevice::Cpu);
    }

    /// Makes a device for every pane with a native window, which the worker then uses. On
    /// failure, requests the next attempt. Why the first attempt failed is shown, and why a
    /// later one did is logged, so that retrying does not churn the status bar.
    fn make_device(&mut self) -> Vec<Request> {
        match self.gpu.make_device() {
            Ok(new) => {
                self.worker.connect(new.device, new.queue, new.generation);
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
        let frame = self.shown.framed().map(|info| ViewerFrame {
            placement: viewer.placement(info.display_window),
            display_window: info.display_window,
            data_window: info.data_window,
            scale_factor: viewer.scale_factor(),
            gain: viewer.gain(),
        });
        match self
            .gpu
            .draw(pane, self.session.graph(), self.session.graph_view(), frame)
        {
            Drawn::Presented => {
                if pane == Pane::Viewer {
                    self.presented = match &self.shown {
                        Shown::Image { wanted, .. } => Some(wanted.number),
                        Shown::Nothing | Shown::Failed { .. } => None,
                    };
                    // The frame tick on which a pixel read in flight is collected.
                    if let Some(read) = self.gpu.take_pixel_read() {
                        self.pixel_read = Some(read);
                    }
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
            Input::PointerLeft => self.viewer_pointer = None,
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
                if self.session.reload(&selection) {
                    vec![Request::Redraw(Pane::Graph)]
                } else {
                    vec![Request::ShowStatus("Select Readers to reload".to_owned())]
                }
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
                self.device = device;
                self.left_gpu = None;
                self.losses_during_evaluation = 0;
                self.worker.evaluate_on(device);
                Vec::new()
            }
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

    fn files_chosen(&mut self, intent: FileIntent, paths: Vec<PathBuf>) -> Vec<Request> {
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
        let Evaluated { wanted, result, .. } = evaluated;
        // Not the node's failure, nor shown: the CPU evaluates the request again, and the last
        // frame stays until it does. One evaluated on the GPU before a move to the CPU may
        // still arrive.
        if let Err(ViewerError::Eval(EvalError::OutOfGpuMemory { .. })) = result {
            if self.device == EvaluationDevice::Gpu {
                self.leave_gpu(LeftGpu::OutOfMemory);
            }
            return Vec::new();
        }
        // Not only later: connecting the GPU delivers the request shown again, with its pixels.
        let not_older = (self.shown.wanted()).is_none_or(|shown| wanted.number >= shown.number);
        if self.session.viewed() != Some(wanted.node) || !not_older {
            return Vec::new();
        }
        // What was sent is current: the viewer follows every change before this runs.
        let current = (self.sent.as_ref()).is_some_and(|sent| sent.key == wanted.key);
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
                    Some(pixels) => self.gpu.show_image(pixels, self.images_shown),
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

    /// The node the viewer's slot marks, and its key.
    fn wanted(&self) -> Option<(NodeId, ResultKey)> {
        let node = self.session.viewed()?;
        let key = result_key(self.session.document(), self.session.reloads(), node)?;
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
        let sent = (self.sent.as_ref()).map(|sent| (sent.node, &sent.key));
        if wanted.as_ref().map(|(node, key)| (*node, key)) != sent {
            self.requests_sent += 1;
            let number = self.requests_sent;
            let wanted = wanted.map(|(node, key)| Wanted { node, key, number });
            self.worker.request(wanted.clone());
            self.sent = wanted;
        }
    }

    /// Keeps the readout on the shown image's pixel under the pointer: asks the GPU to read it,
    /// collects the read on the viewer's frames while one is in flight, and describes it.
    fn follow_readout(&mut self, requests: &mut Vec<Request>) {
        let pointed = self.pointed();
        if let Some((_, Some(probe))) = pointed
            && self.probed != Some(probe)
        {
            self.probed = Some(probe);
            self.gpu.read_pixel(probe);
        }
        if self.gpu.reading_pixel() {
            requests.push(Request::Redraw(Pane::Viewer));
        }
        let readout = match pointed {
            None => String::new(),
            Some(([x, y], None)) => {
                format!("x {x} y {y} · R 0 G 0 B 0 A 0, outside the data window")
            }
            Some(([x, y], Some(probe))) => match (self.pixel_read, self.shown.framed()) {
                // Labelled with the pixel it read, which the pointer may have left.
                (Some(read), Some(info)) if read.probe.image == probe.image => {
                    let [texel_x, texel_y] = read.probe.texel.map(|texel| texel as i32);
                    let values = (["R", "G", "B", "A"].iter().zip(read.values))
                        .map(|(channel, value)| format!("{channel} {}", working_value(value)))
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!(
                        "x {} y {} · {values}",
                        info.data_window.x + texel_x,
                        info.data_window.y + texel_y
                    )
                }
                _ => format!("x {x} y {y}"),
            },
        };
        if readout != self.readout {
            self.readout.clone_from(&readout);
            requests.push(Request::ShowReadout(readout));
        }
    }

    /// The shown image's pixel under the pointer, in image pixels with y up, and the probe that
    /// reads it, or `None` for a pixel outside the data window, whose values are zero.
    fn pointed(&self) -> Option<([i32; 2], Option<Probe>)> {
        let pointer = self.viewer_pointer?;
        let info = self.shown.framed()?;
        let placement = self.session.viewer().placement(info.display_window);
        let [x, y] = placement
            .to_image(pointer)
            .map(|value| value.floor() as i32);
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
        let current = match (&self.sent, self.shown.wanted()) {
            (Some(sent), Some(shown)) => shown.key == sent.key,
            _ => false,
        };
        let mut details = Vec::new();
        match &self.shown {
            Shown::Image { info, .. } if current => {
                let window = info.display_window;
                if !window.is_empty() {
                    details.push(format!("{} × {}", window.width, window.height));
                }
                details.extend(info.warnings.iter().map(ToString::to_string));
            }
            Shown::Failed { error, .. } if current => details.push(error.clone()),
            // The previous image stays until then.
            _ => details.push("updating…".to_owned()),
        }
        for (id, what) in missing(document, viewed) {
            let (label, lack) = match what {
                Missing::Input(input) => (input.label, "is not connected"),
                Missing::File(parameter) => (parameter.label, "is not set"),
            };
            details.push(if id == viewed {
                format!("{label} {lack}")
            } else {
                let upstream = document.node(id).expect("inputs name existing nodes");
                format!("{label} of {} {lack}", upstream.name())
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

    /// How many GPU errors this session went uncaptured. Each is a bug, logged on standard error
    /// up to a limit, after which drawing continues.
    pub fn gpu_errors(&self) -> u64 {
        self.gpu.errors()
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
