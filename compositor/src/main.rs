//! OpenCompositor's Qt shell. Qt owns the windows, docks, dialogs and event loop; the core's
//! [`App`] owns everything else. Everything here runs on Qt's GUI thread.

mod keys;
mod lifecycle;
mod pace;
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
        waker_thread
            .queue(move |_| dispatch(Event::Loaded(loaded)))
            .expect("the waker outlives the app, whose loader delivers nothing once dropped");
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

    // The app goes before the window: its surfaces before the panes' native windows, and its
    // requests with it, so that the events destroying the window dispatch to nothing. Its loader
    // delivers nothing after it.
    drop(APP.with(|cell| cell.borrow_mut().take()));
    // SAFETY: nothing else deletes the waker. Results queued to it are dropped without running.
    unsafe { ffi::delete_waker(waker) };
    drop(window);
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
        Request::Redraw(pane) => pace::redraw(pane),
        Request::AttachSurface(pane) => {
            let attached = with_pane_window(pane, |mut window| {
                let size = pane::size(&window);
                pane::native_window(window.as_mut(), pane).map(|native| (native, size))
            });
            if let Some((window, size)) = attached {
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

/// Calls `use_window` with the pane's window. The reference ends with it.
fn with_pane_window<R>(pane: Pane, use_window: impl FnOnce(Pin<&mut qt::Pane>) -> R) -> R {
    let window = widgets().panes[pane.index()];
    // SAFETY: a pane lives as long as its dock, until the main window is destroyed after the app,
    // and requests come only from the app or the lifecycle check's steps in the event loop. The
    // reference may overlap the one Qt passes to the pane's `event` further up the stack, since
    // requests are performed while dispatching its events, but `Pane` is a zero-sized opaque type:
    // neither covers memory that Rust reads or writes, and Qt reaches the window only from C++.
    use_window(unsafe { Pin::new_unchecked(&mut *window) })
}

/// Follows Qt Wayland's `surfaceRoleDestroyed` for the pane's current window, replacing the
/// previous connection, whose window Qt may since have replaced.
pub fn watch_surface(pane: Pane, wayland: Pin<&mut qt::QWaylandWindow>) {
    let connection =
        wayland.on_surface_role_destroyed(move |_| dispatch(Event::SurfaceDestroyed(pane)));
    let previous = SURFACE_WATCHES.with(|cell| cell.borrow_mut()[pane.index()].replace(connection));
    drop(previous);
}
