//! OpenCompositor's Qt shell. Qt owns the windows, docks, dialogs and event loop; the core's
//! [`App`] owns everything else. Everything here runs on Qt's GUI thread.

mod keys;
mod lifecycle;
mod pane;
mod qt;

use std::cell::{OnceCell, RefCell};
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::pin::Pin;

use cxx_qt::{QMetaObjectConnectionGuard, Threading};
use cxx_qt_lib::{QList, QString};
use opencompositor_core::{App, Command, Event, FileIntent, Pane, Request};

use crate::qt::ffi;

/// `Qt::DockWidgetArea` values.
const TOP_DOCK_AREA: i32 = 4;
const BOTTOM_DOCK_AREA: i32 = 8;

thread_local! {
    /// Borrowed only while the app handles an event, during which nothing calls Qt: Qt can call
    /// back into the shell synchronously, as when floating a dock destroys a pane's surface.
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    static WIDGETS: OnceCell<Widgets> = const { OnceCell::new() };
    /// Each pane's connection to Qt Wayland's `surfaceRoleDestroyed`.
    static SURFACE_WATCHES: RefCell<[Option<QMetaObjectConnectionGuard>; Pane::COUNT]> =
        RefCell::new(Default::default());
}

/// The Qt objects the shell acts on. Docks are hidden when closed, not deleted, so all of them
/// live until the main window is destroyed after the event loop.
#[derive(Clone, Copy, Debug)]
pub struct Widgets {
    pub window: *mut ffi::QMainWindow,
    pub docks: [*mut ffi::QDockWidget; Pane::COUNT],
    pub panes: [*mut qt::Pane; Pane::COUNT],
    /// Under the viewer.
    pub caption: *mut ffi::QLabel,
}

fn main() {
    let lifecycle_check = match &std::env::args_os().skip(1).collect::<Vec<_>>()[..] {
        [] => None,
        [flag, image] if flag == "--lifecycle-check" => Some(PathBuf::from(image)),
        _ => {
            eprintln!("usage: opencompositor [--lifecycle-check IMAGE]");
            std::process::exit(2);
        }
    };
    // The shim calls Qt's private API, which has no ABI guarantee across Qt releases, while the
    // dynamic loader accepts any Qt exporting the unversioned Qt_6_PRIVATE_API symbols.
    let (compiled, running) = (ffi::qt_compiled_version(), ffi::qt_runtime_version());
    if compiled != running {
        eprintln!(
            "opencompositor was built against Qt {compiled} but loaded Qt {running}; rebuild it"
        );
        std::process::exit(2);
    }

    ffi::new_application();
    let waker = ffi::new_waker();
    // SAFETY: the waker is live. Its thread handle outlives it: queueing fails once it is deleted.
    let waker_thread = unsafe { &*waker }.qt_thread();
    let app = App::new(move |loaded| {
        // Fails only once the waker is deleted, when a read outlasts the event loop.
        let _ = waker_thread.queue(move |_| dispatch(Event::Loaded(loaded)));
    });
    APP.with(|cell| *cell.borrow_mut() = Some(app));

    let mut window = ffi::new_main_window();
    window
        .pin_mut()
        .set_window_title(&QString::from("OpenCompositor"));
    window.pin_mut().set_dock_nesting_enabled(true);
    window.pin_mut().resize(1400, 900);
    let window_ptr = window.as_mut_ptr();
    // SAFETY: the window is live; it owns the menus.
    let [file_menu, edit_menu, window_menu] = ["&File", "&Edit", "&Window"]
        .map(|title| unsafe { ffi::new_menu(window_ptr, &QString::from(title)) });
    for (index, command) in Command::ALL.into_iter().enumerate() {
        let (menu, text, shortcut) = match command {
            Command::ReadFiles => (file_menu, "&Read Files…", "R"),
            Command::ChangeReaderFile => (file_menu, "&Change the Reader's File…", "Shift+R"),
            Command::Reload => (file_menu, "Re&load the Selected Readers", ""),
            Command::Undo => (edit_menu, "&Undo", "Ctrl+Z"),
            Command::Redo => (edit_menu, "&Redo", "Ctrl+Shift+Z"),
        };
        let [text, shortcut] = [text, shortcut].map(QString::from);
        // SAFETY: the menu is live; it owns the action.
        unsafe { ffi::new_action(menu, &text, &shortcut, run_command, index) };
    }
    let docks = Pane::ALL.map(|pane| {
        let (title, area) = match pane {
            Pane::Graph => ("Node Graph", BOTTOM_DOCK_AREA),
            Pane::Viewer => ("Viewer", TOP_DOCK_AREA),
        };
        // SAFETY: the window and menu are live; the window owns the dock.
        unsafe { ffi::new_dock(window_ptr, window_menu, &QString::from(title), area) }
    });
    let panes = Pane::ALL.map(|pane| ffi::new_pane(pane.index()));
    let [graph_dock, viewer_dock] = Pane::ALL.map(|pane| docks[pane.index()]);
    // SAFETY: the docks are live; each owns its pane, and the viewer's its caption, from here on.
    let caption = unsafe {
        ffi::container(panes[Pane::Graph.index()], graph_dock);
        ffi::container_with_caption(panes[Pane::Viewer.index()], viewer_dock)
    };
    WIDGETS.with(|cell| {
        cell.set(Widgets {
            window: window_ptr,
            docks,
            panes,
            caption,
        })
        .expect("widgets are made once")
    });
    keys::start();
    window.pin_mut().show();
    if let Some(image) = lifecycle_check {
        lifecycle::start(image);
    }
    let status = ffi::exec();

    APP.with(|cell| cell.borrow_mut().as_mut().map(App::stop_loading));
    // SAFETY: nothing else deletes the waker. Results queued to it are dropped without running.
    unsafe { ffi::delete_waker(waker) };
    // Destroying the window destroys the panes' native windows, whose events drop the surfaces.
    drop(window);
    // The device goes after every surface.
    drop(APP.with(|cell| cell.borrow_mut().take()));
    std::process::exit(status);
}

/// Hands an event to the app, then does what it asks.
pub fn dispatch(event: Event) {
    let requests = APP.with(|cell| match cell.borrow_mut().as_mut() {
        Some(app) => app.dispatch(event),
        None => Vec::new(),
    });
    for request in requests {
        perform(request);
    }
}

fn run_command(index: usize) {
    dispatch(Event::Command(Command::ALL[index]));
}

pub fn perform(request: Request) {
    let widgets = widgets();
    match request {
        Request::Redraw(pane) => pane_window(pane).request_update(),
        Request::AttachSurface(pane) => {
            let size = pane::size(&pane_window(pane));
            if let Some(window) = pane::native_window(pane_window(pane), pane) {
                dispatch(Event::SurfaceCreated { pane, window, size });
            }
        }
        Request::ChooseFiles(intent) => {
            let title = match intent {
                FileIntent::CreateReaders { .. } => "Read Files",
                FileIntent::SetReaderPath(_) => "Change the Reader's File",
            };
            let several = intent.allows_several();
            // SAFETY: the window is live.
            let chosen =
                unsafe { ffi::choose_files(widgets.window, &QString::from(title), several) };
            let paths = QList::from(&chosen).iter().map(path).collect();
            dispatch(Event::FilesChosen { intent, paths });
        }
        // SAFETY: the window is live.
        Request::ShowStatus(text) => unsafe {
            ffi::show_status(widgets.window, &QString::from(&text))
        },
        // SAFETY: the caption lives as long as the viewer's dock.
        Request::ShowViewerCaption(text) => unsafe {
            ffi::set_caption(widgets.caption, &QString::from(&text))
        },
    }
}

fn path(name: &QString) -> PathBuf {
    let bytes = ffi::encode_name(name);
    PathBuf::from(OsString::from_vec(bytes.as_slice().to_vec()))
}

/// Whether the pane has a surface to draw to.
pub fn has_surface(pane: Pane) -> bool {
    APP.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|app| app.has_surface(pane))
    })
}

pub fn widgets() -> Widgets {
    WIDGETS.with(|cell| *cell.get().expect("widgets are made before the event loop"))
}

fn pane_window(pane: Pane) -> Pin<&'static mut qt::Pane> {
    let window = widgets().panes[pane.index()];
    // SAFETY: a pane lives as long as its dock, which lives until the main window is destroyed,
    // and no event that runs after that performs requests. Each use ends before the next begins.
    unsafe { Pin::new_unchecked(&mut *window) }
}

/// Follows Qt Wayland's `surfaceRoleDestroyed` for the pane's current window, replacing the
/// previous connection, whose window Qt may since have replaced.
pub fn watch_surface(pane: Pane, wayland: Pin<&mut qt::QWaylandWindow>) {
    let connection =
        wayland.on_surface_role_destroyed(move |_| dispatch(Event::SurfaceDestroyed(pane)));
    let previous = SURFACE_WATCHES.with(|cell| cell.borrow_mut()[pane.index()].replace(connection));
    drop(previous);
}
