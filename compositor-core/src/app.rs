//! The interactive compositor: a session, the GPU that draws its panes, and the evaluator's
//! worker, changed only by [`App::dispatch`]. What only the shell can do comes back as [`Request`]s.

use std::path::PathBuf;

use open_ng::{Button, InputEvent, Modifiers, PhysicalPoint, Point};

use crate::document::{ExposureVersion, Node, NodeId, NodeKind, Slot, unconnected_inputs};
use crate::evaluate::{ResultKey, result_key};
use crate::gpu::{FrameCheck, Gpu, NativeWindow, PaneSize, ViewerFrame};
use crate::image::{ImageInfo, Window};
use crate::session::Session;
use crate::worker::{Evaluated, ViewerError, Wanted, Worker};

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
pub enum Event {
    /// The pane became visible and must be drawn. A pane without a surface asks for one.
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
}

/// A window-wide command, from a menu or its shortcut, whichever pane has the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// A Reader for each file chosen, placed where the pointer is over the graph.
    ReadFiles,
    /// An Exposure node where the pointer is over the graph, selected alone.
    CreateExposure,
    /// A new file for the one selected Reader.
    ChangeReaderFile,
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
}

/// The keys the window uses. The pane under the pointer receives all but the slot keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Space,
    Escape,
    G,
    H,
    Y,
    /// Delete or Backspace: deletes the selected nodes while the pointer is over the graph.
    Delete,
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
            Key::Delete | Key::Slot(_) => return None,
        })
    }
}

/// What files chosen in a dialog are for. A dialog is modal, but its target can still change while
/// it is open, so the choice is checked again when it arrives.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FileIntent {
    /// A Reader for each file, stacked downwards from a graph point.
    CreateReaders {
        at: Point,
    },
    SetReaderPath(NodeId),
}

impl FileIntent {
    pub fn allows_several(self) -> bool {
        matches!(self, FileIntent::CreateReaders { .. })
    }
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
            Shown::Image { info, .. } if !is_empty(info.display_window) => Some(info),
            Shown::Image { .. } | Shown::Nothing | Shown::Failed { .. } => None,
        }
    }

    fn display_window(&self) -> Option<Window> {
        self.framed().map(|info| info.display_window)
    }
}

#[derive(Debug)]
pub struct App {
    session: Session,
    gpu: Gpu,
    /// Nothing is delivered once the app has dropped.
    worker: Worker,
    /// What the worker was last asked for.
    sent: Option<Wanted>,
    /// How many requests the worker was sent, which numbers them.
    requests_sent: u64,
    /// The pane under the pointer, which receives keys whichever pane has the keyboard.
    hovered: Option<Pane>,
    /// The pointer's last position over the graph pane.
    graph_pointer: Option<PhysicalPoint>,
    shown: Shown,
    /// The viewer's caption as last requested.
    caption: String,
}

impl App {
    /// `deliver` receives each finished evaluation on the evaluator's thread, and must hand it to
    /// the GUI thread as an [`Event::Evaluated`].
    pub fn new(deliver: impl Fn(Evaluated) + Send + 'static) -> App {
        App {
            session: Session::new(),
            gpu: Gpu::new(),
            worker: Worker::start(deliver),
            sent: None,
            requests_sent: 0,
            hovered: None,
            graph_pointer: None,
            shown: Shown::Nothing,
            caption: String::new(),
        }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn has_surface(&self, pane: Pane) -> bool {
        self.gpu.has_surface(pane)
    }

    /// Has every frame from now on record what it proves, for the shell's lifecycle check. Each
    /// frame then waits for the GPU. Call it before the first surface is attached, so that every
    /// surface is made readable where it allows.
    pub fn check_frames(&mut self) {
        self.gpu.check_frames();
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

    pub fn dispatch(&mut self, event: Event) -> Vec<Request> {
        let mut requests = match event {
            Event::Exposed(pane) if !self.gpu.has_surface(pane) => {
                vec![Request::AttachSurface(pane)]
            }
            Event::Exposed(pane) | Event::Draw(pane) => self.draw(pane),
            Event::SurfaceCreated { pane, window, size } => {
                match self.gpu.attach(pane, window, size) {
                    Ok(device) => {
                        if let Some(device) = device {
                            self.worker.connect(device, self.gpu.device_generation());
                        }
                    }
                    Err(error) => return vec![Request::ShowError(error)],
                }
                self.resize(pane, size);
                self.draw(pane)
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
        };
        self.follow_viewer(&mut requests);
        requests
    }

    fn draw(&mut self, pane: Pane) -> Vec<Request> {
        let frame = self.shown.framed().map(|info| ViewerFrame {
            placement: self.session.viewer().placement(info.display_window),
            display_window: info.display_window,
            data_window: info.data_window,
            scale_factor: self.session.viewer().scale_factor(),
        });
        let again = self
            .gpu
            .draw(pane, self.session.graph(), self.session.graph_view(), frame);
        redraw_if(pane, again)
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
            Command::ReadFiles => vec![Request::ChooseFiles(FileIntent::CreateReaders {
                at: self.graph_point_at_pointer(),
            })],
            Command::CreateExposure => {
                let kind = NodeKind::Exposure {
                    version: ExposureVersion::CURRENT,
                };
                self.session
                    .create_node(kind, self.graph_point_at_pointer());
                vec![Request::Redraw(Pane::Graph)]
            }
            Command::ChangeReaderFile => match self.session.document().selection() {
                &[id] => vec![Request::ChooseFiles(FileIntent::SetReaderPath(id))],
                _ => vec![Request::ShowStatus(
                    "Select one Reader to change its file".to_owned(),
                )],
            },
            Command::Reload => {
                if self.session.reload_selected() {
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
        }
    }

    /// Where the pointer is over the graph, or the middle of the graph view if it is elsewhere.
    fn graph_point_at_pointer(&self) -> Point {
        let view = self.session.graph_view();
        let viewport = view.viewport();
        let pointer = self.graph_pointer.unwrap_or(PhysicalPoint {
            x: viewport.width as f32 / 2.0,
            y: viewport.height as f32 / 2.0,
        });
        view.to_graph(pointer)
    }

    fn files_chosen(&mut self, intent: FileIntent, paths: Vec<PathBuf>) -> Vec<Request> {
        let Some(first) = paths.first().cloned() else {
            return Vec::new();
        };
        match intent {
            FileIntent::CreateReaders { at } => {
                self.session.create_readers(paths, at);
            }
            FileIntent::SetReaderPath(node) => {
                if let Err(error) = self.session.set_reader_path(node, first) {
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
        let Evaluated { wanted, result } = evaluated;
        // Not only later: connecting the GPU delivers the request shown again, with its pixels.
        let not_older = (self.shown.wanted()).is_none_or(|shown| wanted.number >= shown.number);
        if self.session.viewed() != Some(wanted.node) || !not_older {
            return Vec::new();
        }
        let current = self.wanted().is_some_and(|(_, key)| key == wanted.key);
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
                match image.pixels {
                    Some(staged) => self.gpu.show_image(staged),
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
        let caption = self.caption();
        if caption != self.caption {
            self.caption.clone_from(&caption);
            requests.push(Request::ShowViewerCaption(caption));
        }
    }

    /// Names the viewed node from the document as it stands, then says what is shown of it and
    /// what upstream of it is not connected. Called once the wanted key has been sent.
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
                if !is_empty(window) {
                    details.push(format!("{} × {}", window.width, window.height));
                }
                details.extend(info.warnings.iter().map(ToString::to_string));
            }
            Shown::Failed { error, .. } if current => details.push(error.clone()),
            // The previous image stays until then.
            _ => details.push("updating…".to_owned()),
        }
        for (id, input) in unconnected_inputs(document, viewed) {
            details.push(if id == viewed {
                format!("{} is not connected", input.label)
            } else {
                let upstream = document.node(id).expect("inputs name existing nodes");
                format!(
                    "{} of {} is not connected",
                    input.label,
                    node_name(upstream)
                )
            });
        }
        if details.is_empty() {
            details.push("empty".to_owned());
        }
        format!("Slot {slot} · {} · {}", node_name(node), details.join("; "))
    }
}

fn redraw_if(pane: Pane, needed: bool) -> Vec<Request> {
    if needed {
        vec![Request::Redraw(pane)]
    } else {
        Vec::new()
    }
}

/// A Reader's file name, or the node's kind.
fn node_name(node: &Node) -> String {
    match &node.kind {
        NodeKind::Reader { path } => path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned(),
        NodeKind::Exposure { .. } => "Exposure".to_owned(),
    }
}

fn is_empty(window: Window) -> bool {
    window.width == 0 || window.height == 0
}

#[cfg(test)]
mod tests;
