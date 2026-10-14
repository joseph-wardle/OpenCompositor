//! The interactive compositor: a session, the GPU that draws its panes, and the loader, changed
//! only by [`App::dispatch`]. What only the shell can do comes back as [`Request`]s.

use std::path::{Path, PathBuf};

use open_ng::{Button, InputEvent, Modifiers, PhysicalPoint, Point};

use crate::document::{NodeId, Slot};
use crate::gpu::{Gpu, NativeWindow, PaneSize, ViewerFrame};
use crate::image::{ImageInfo, Window};
use crate::loader::{LoadRequest, Loaded, Loader};
use crate::session::{Session, Viewed};

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
    /// A read finished on the loader's thread.
    Loaded(Loaded),
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
    ShowStatus(String),
    /// Describe under the viewer what it shows, or why it shows nothing.
    ShowViewerCaption(String),
}

/// A window-wide command, from a menu or its shortcut, whichever pane has the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// A Reader for each file chosen, placed where the pointer is over the graph.
    ReadFiles,
    /// A new file for the one selected Reader.
    ChangeReaderFile,
    /// Reads the selected Readers' files again when they are next viewed, at once if they are.
    Reload,
    Undo,
    Redo,
}

impl Command {
    pub const ALL: [Command; 5] = [
        Command::ReadFiles,
        Command::ChangeReaderFile,
        Command::Reload,
        Command::Undo,
        Command::Redo,
    ];
}

/// The keys the window uses. The pane under the pointer receives all but the slot keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Space,
    Escape,
    G,
    H,
    Y,
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
            Key::Slot(_) => return None,
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
    /// Its pixels are on the GPU if there was one when it was read.
    Image {
        request: LoadRequest,
        info: ImageInfo,
    },
    /// An empty frame, never an older image.
    Failed {
        request: LoadRequest,
        error: String,
    },
}

impl Shown {
    fn request(&self) -> Option<&LoadRequest> {
        match self {
            Shown::Nothing => None,
            Shown::Image { request, .. } | Shown::Failed { request, .. } => Some(request),
        }
    }

    fn display_window(&self) -> Option<Window> {
        match self {
            Shown::Image { info, .. } => Some(info.display_window),
            Shown::Nothing | Shown::Failed { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct App {
    session: Session,
    gpu: Gpu,
    /// Nothing is delivered once the app has dropped.
    loader: Loader,
    /// The pane under the pointer, which receives keys whichever pane has the keyboard.
    hovered: Option<Pane>,
    /// The pointer's last position over the graph pane.
    graph_pointer: Option<PhysicalPoint>,
    viewer_scale_factor: f32,
    shown: Shown,
    /// The viewer's caption as last requested.
    caption: String,
}

impl App {
    /// `deliver` receives each finished read on the loader's thread, and must hand it to the GUI
    /// thread as an [`Event::Loaded`].
    pub fn new(deliver: impl Fn(Loaded) + Send + 'static) -> App {
        App {
            session: Session::new(),
            gpu: Gpu::new(),
            loader: Loader::start(deliver),
            hovered: None,
            graph_pointer: None,
            viewer_scale_factor: 1.0,
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
                            self.loader.connect(device);
                        }
                    }
                    Err(error) => return vec![Request::ShowStatus(error)],
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
            Event::Loaded(loaded) => self.loaded(loaded),
        };
        self.follow_viewer(&mut requests);
        requests
    }

    fn draw(&mut self, pane: Pane) -> Vec<Request> {
        let frame = match &self.shown {
            Shown::Image { info, .. } => Some(ViewerFrame {
                placement: self.session.viewer().placement(info.display_window),
                display_window: info.display_window,
                data_window: info.data_window,
                scale_factor: self.viewer_scale_factor,
            }),
            Shown::Nothing | Shown::Failed { .. } => None,
        };
        let again = self
            .gpu
            .draw(pane, self.session.graph(), self.session.graph_view(), frame);
        redraw_if(pane, again)
    }

    fn resize(&mut self, pane: Pane, size: PaneSize) {
        match pane {
            Pane::Graph => self.session.resize_graph_view(size.viewport()),
            Pane::Viewer => {
                self.session.viewer_mut().resize(size);
                self.viewer_scale_factor = size.scale_factor;
            }
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
            (Some(Pane::Graph), Some(key)) => redraw_if(
                Pane::Graph,
                self.session.graph_input(InputEvent::KeyPressed(key)),
            ),
            (Some(Pane::Viewer), _) => self.viewer_key(key, modifiers),
            _ => Vec::new(),
        }
    }

    fn viewer_key(&mut self, key: Key, modifiers: Modifiers) -> Vec<Request> {
        let display = self.shown.display_window();
        let viewer = self.session.viewer_mut();
        match (key, display) {
            (Key::Space, _) => {
                viewer.hold_space(true);
                Vec::new()
            }
            (Key::H, Some(display)) => {
                if modifiers.shift {
                    viewer.actual_size(display);
                } else {
                    viewer.fit();
                }
                vec![Request::Redraw(Pane::Viewer)]
            }
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
            (false, false) => {
                self.session.viewer_mut().show_slot(slot);
                vec![Request::Redraw(Pane::Viewer)]
            }
            (true, false) => match self.session.document().selection() {
                &[id] => {
                    let changes = self.session.set_mark(slot, Some(id));
                    debug_assert!(changes.is_ok(), "a selected node exists");
                    self.session.viewer_mut().show_slot(slot);
                    vec![Request::Redraw(Pane::Graph), Request::Redraw(Pane::Viewer)]
                }
                _ => vec![Request::ShowStatus("Select one node to mark it".to_owned())],
            },
            (false, true) => {
                if self.session.frame_mark(slot) {
                    vec![Request::Redraw(Pane::Graph)]
                } else {
                    vec![Request::ShowStatus(format!(
                        "Slot {} marks nothing",
                        slot.digit()
                    ))]
                }
            }
            (true, true) => Vec::new(),
        }
    }

    fn command(&mut self, command: Command) -> Vec<Request> {
        match command {
            Command::ReadFiles => vec![Request::ChooseFiles(FileIntent::CreateReaders {
                at: self.graph_point_at_pointer(),
            })],
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

    /// Shows a finished read if the viewer still wants it. Otherwise it is dropped, which returns
    /// its staging buffer to the loader.
    fn loaded(&mut self, loaded: Loaded) -> Vec<Request> {
        if self.session.reading() != Some(&loaded.request) {
            return Vec::new();
        }
        let Loaded { request, result } = loaded;
        let failed = result.is_err();
        self.shown = match result {
            Ok(decoded) => {
                match decoded.pixels {
                    Some(staged) => self.gpu.show_image(staged),
                    None => self.gpu.clear_image(),
                }
                Shown::Image {
                    request,
                    info: decoded.info,
                }
            }
            Err(error) => {
                self.gpu.clear_image();
                Shown::Failed {
                    request,
                    error: error.to_string(),
                }
            }
        };
        let mut requests = vec![Request::Redraw(Pane::Viewer)];
        if self.session.finish_reading(failed) {
            requests.push(Request::Redraw(Pane::Graph));
        }
        requests
    }

    /// Keeps the viewer on what its slot leads to: reads it unless it is shown or being read,
    /// shows nothing if the slot leads nowhere, and describes the result.
    fn follow_viewer(&mut self, requests: &mut Vec<Request>) {
        let wanted = match self.session.viewed() {
            Viewed::Node(request) => Some(request),
            Viewed::Unmarked | Viewed::Missing(_) => None,
        };
        if wanted.is_none() && !matches!(self.shown, Shown::Nothing) {
            self.shown = Shown::Nothing;
            self.gpu.clear_image();
            requests.push(Request::Redraw(Pane::Viewer));
        }
        let read = wanted.filter(|wanted| Some(wanted) != self.shown.request());
        if read.as_ref() != self.session.reading() {
            if let Some(request) = &read {
                self.loader.request(request.clone());
            }
            if self.session.start_reading(read) {
                requests.push(Request::Redraw(Pane::Graph));
            }
        }
        let caption = self.caption();
        if caption != self.caption {
            self.caption.clone_from(&caption);
            requests.push(Request::ShowViewerCaption(caption));
        }
    }

    fn caption(&self) -> String {
        let slot = self.session.viewer().slot().digit();
        let request = match self.session.viewed() {
            Viewed::Unmarked => return format!("Slot {slot}: nothing marked"),
            Viewed::Missing(_) => {
                return format!("Slot {slot}: the node it marked no longer exists");
            }
            Viewed::Node(request) => request,
        };
        let name = file_name(&request.path);
        if self.session.reading() == Some(&request) {
            return match &self.shown {
                Shown::Image { request: shown, .. } if shown.node != request.node => format!(
                    "Slot {slot} · {name} · reading… (showing {} until then)",
                    file_name(&shown.path)
                ),
                _ => format!("Slot {slot} · {name} · reading…"),
            };
        }
        match &self.shown {
            Shown::Image {
                request: shown,
                info,
            } if *shown == request => {
                let window = info.display_window;
                let mut caption = format!(
                    "Slot {slot} · {name} · {} × {}",
                    window.width, window.height
                );
                for warning in &info.warnings {
                    caption.push_str("; ");
                    caption.push_str(&warning.to_string());
                }
                caption
            }
            Shown::Failed {
                request: shown,
                error,
            } if *shown == request => format!("Slot {slot} · {name} · {error}"),
            // Nothing is being read once loading stops.
            _ => format!("Slot {slot} · {name}"),
        }
    }
}

fn redraw_if(pane: Pane, needed: bool) -> Vec<Request> {
    if needed {
        vec![Request::Redraw(pane)]
    } else {
        Vec::new()
    }
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests;
