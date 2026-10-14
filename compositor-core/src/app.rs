//! The interactive compositor: a session, the GPU that draws its panes, and the loader, changed
//! only by [`App::dispatch`]. What only the shell can do comes back as [`Request`]s.

use std::path::{Path, PathBuf};

use open_ng::{Button, InputEvent, Key, Modifiers, PhysicalPoint, Point};

use crate::document::NodeId;
use crate::gpu::{Gpu, NativeWindow, PaneSize};
use crate::loader::{LoadRequest, Loaded, Loader};
use crate::session::Session;

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
    Command(Command),
    /// The pane lost the keyboard, as to another pane, a dialog or another application.
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
}

/// A window-wide command, from a menu or its shortcut, whichever pane has the keyboard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    /// A Reader for each file chosen, placed where the pointer is over the graph.
    ReadFiles,
    /// A new file for the one selected Reader.
    ChangeReaderFile,
    Undo,
    Redo,
}

impl Command {
    pub const ALL: [Command; 4] = [
        Command::ReadFiles,
        Command::ChangeReaderFile,
        Command::Undo,
        Command::Redo,
    ];
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

/// Input to a pane, in physical pixels from its top-left. The shell leaves out repeated keys.
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
    /// Only the keys the graph view uses.
    KeyPressed(Key),
    KeyReleased(Key),
}

#[derive(Debug)]
pub struct App {
    session: Session,
    gpu: Gpu,
    /// `None` once stopping.
    loader: Option<Loader>,
    /// The pane under the pointer, which receives keys whichever pane has the keyboard.
    hovered: Option<Pane>,
    /// The pointer's last position over the graph pane.
    graph_pointer: Option<PhysicalPoint>,
}

impl App {
    /// `deliver` receives each finished read on the loader's thread, and must hand it to the GUI
    /// thread as an [`Event::Loaded`].
    pub fn new(deliver: impl Fn(Loaded) + Send + 'static) -> App {
        App {
            session: Session::new(),
            gpu: Gpu::new(),
            loader: Some(Loader::start(deliver)),
            hovered: None,
            graph_pointer: None,
        }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn has_surface(&self, pane: Pane) -> bool {
        self.gpu.has_surface(pane)
    }

    /// Stops the loader without waiting for it. The read in progress, if any, may still be
    /// delivered; nothing after it is.
    pub fn stop_loading(&mut self) {
        self.loader = None;
    }

    pub fn dispatch(&mut self, event: Event) -> Vec<Request> {
        match event {
            Event::Exposed(pane) if !self.gpu.has_surface(pane) => {
                vec![Request::AttachSurface(pane)]
            }
            Event::Exposed(pane) | Event::Draw(pane) => self.draw(pane),
            Event::SurfaceCreated { pane, window, size } => {
                if let Err(error) = self.gpu.attach(pane, window, size) {
                    return vec![Request::ShowStatus(error)];
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
            Event::Command(command) => self.command(command),
            // Another pane's focus loss can arrive after a press has given the graph the
            // keyboard, so it must not cancel the gesture that press began.
            Event::FocusLost(Pane::Graph) => {
                self.session.cancel_graph_gesture();
                vec![Request::Redraw(Pane::Graph)]
            }
            Event::FocusLost(Pane::Viewer) => Vec::new(),
            Event::FilesChosen { intent, paths } => self.files_chosen(intent, paths),
            Event::Loaded(loaded) => vec![Request::ShowStatus(describe(&loaded))],
        }
    }

    fn draw(&mut self, pane: Pane) -> Vec<Request> {
        let again = self
            .gpu
            .draw(pane, self.session.graph(), self.session.graph_view());
        redraw_if(pane, again)
    }

    fn resize(&mut self, pane: Pane, size: PaneSize) {
        if pane == Pane::Graph {
            self.session.resize_graph_view(size.viewport());
        }
    }

    fn input(&mut self, pane: Pane, input: Input) -> Vec<Request> {
        let graph_event = match input {
            Input::PointerEntered => {
                self.hovered = Some(pane);
                return Vec::new();
            }
            Input::PointerLeft => {
                if self.hovered == Some(pane) {
                    self.hovered = None;
                }
                if pane == Pane::Graph {
                    self.graph_pointer = None;
                }
                InputEvent::PointerLeft
            }
            Input::PointerMoved { position } => {
                if pane == Pane::Graph {
                    self.graph_pointer = Some(position);
                }
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
            Input::KeyPressed(key) => match self.hovered {
                Some(Pane::Graph) => InputEvent::KeyPressed(key),
                _ => return Vec::new(),
            },
            // A release goes to the view whichever pane is under the pointer, so that a key held
            // across panes is not left held.
            Input::KeyReleased(key) => InputEvent::KeyReleased(key),
        };
        let is_key = matches!(
            graph_event,
            InputEvent::KeyPressed(_) | InputEvent::KeyReleased(_)
        );
        if pane != Pane::Graph && !is_key {
            return Vec::new();
        }
        redraw_if(Pane::Graph, self.session.graph_input(graph_event))
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
        let node = match intent {
            FileIntent::CreateReaders { at } => {
                let changes = self.session.create_readers(paths, at);
                *changes.nodes.first().expect("a Reader for each file")
            }
            FileIntent::SetReaderPath(node) => {
                if let Err(error) = self.session.set_reader_path(node, first.clone()) {
                    return vec![Request::ShowStatus(format!(
                        "Cannot change the file: {error}"
                    ))];
                }
                node
            }
        };
        let mut requests = vec![Request::Redraw(Pane::Graph)];
        requests.extend(self.load(node, first));
        requests
    }

    /// Reads the node's file. For now every new or changed Reader is read, to show its status.
    fn load(&mut self, node: NodeId, path: PathBuf) -> Option<Request> {
        let loader = self.loader.as_ref()?;
        let status = format!("Reading {}…", file_name(&path));
        loader.request(LoadRequest { node, path });
        Some(Request::ShowStatus(status))
    }
}

fn redraw_if(pane: Pane, needed: bool) -> Vec<Request> {
    if needed {
        vec![Request::Redraw(pane)]
    } else {
        Vec::new()
    }
}

fn describe(loaded: &Loaded) -> String {
    let name = file_name(&loaded.request.path);
    match &loaded.result {
        Ok(image) => {
            let window = image.display_window();
            let mut status = format!("{name}: {} × {}", window.width, window.height);
            for warning in image.warnings() {
                status.push_str("; ");
                status.push_str(&warning.to_string());
            }
            status
        }
        Err(error) => format!("{name}: {error}"),
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
