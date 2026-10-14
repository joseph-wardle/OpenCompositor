//! OpenCompositor's Qt shell. Qt owns the windows, docks, dialogs and event loop; the core's
//! [`App`] owns everything else. Everything here runs on Qt's GUI thread.

mod lifecycle;
mod pane;
mod qt;

use std::cell::{OnceCell, RefCell};
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::pin::Pin;

use cxx_qt::{QMetaObjectConnectionGuard, Threading};
use cxx_qt_lib::{QList, QString};
use opencompositor_core::{App, Event, FileIntent, Pane, Request};

use crate::qt::ffi;

/// The exit status after a panic in a callback from Qt, as for a panic in `main`.
const PANIC_EXIT_STATUS: i32 = 101;

/// `Qt::DockWidgetArea` values.
const TOP_DOCK_AREA: i32 = 4;
const BOTTOM_DOCK_AREA: i32 = 8;

thread_local! {
    /// Borrowed only while the app handles an event, during which nothing calls Qt: Qt can call
    /// back into the shell synchronously, as when floating a dock destroys a pane's surface.
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    static WIDGETS: OnceCell<Widgets> = const { OnceCell::new() };
    /// Each pane's connection to Qt Wayland's `surfaceDestroyed`.
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
    // SAFETY: the waker is deleted only after the loader has stopped.
    let waker_thread = unsafe { &*waker }.qt_thread();
    let app = App::new(move |loaded| {
        // Queueing fails only once the waker is deleted, after the loader has stopped.
        let _ = waker_thread.queue(move |_| guarded((), || dispatch(Event::Loaded(loaded))));
    });
    APP.with(|cell| *cell.borrow_mut() = Some(app));

    let mut window = ffi::new_main_window();
    window
        .pin_mut()
        .set_window_title(&QString::from("OpenCompositor"));
    window.pin_mut().set_dock_nesting_enabled(true);
    window.pin_mut().resize(1400, 900);
    let window_ptr = window.as_mut_ptr();
    let docks = Pane::ALL.map(|pane| {
        let (title, area) = match pane {
            Pane::Graph => ("Node Graph", BOTTOM_DOCK_AREA),
            Pane::Viewer => ("Viewer", TOP_DOCK_AREA),
        };
        // SAFETY: the window is live; it owns the dock.
        unsafe { ffi::new_dock(window_ptr, &QString::from(title), area) }
    });
    let panes = Pane::ALL.map(|pane| {
        let window = ffi::new_pane(pane.index());
        // SAFETY: the dock is live; it owns the pane from here on.
        unsafe { ffi::container(window, docks[pane.index()]) };
        window
    });
    WIDGETS.with(|cell| {
        cell.set(Widgets {
            window: window_ptr,
            docks,
            panes,
        })
        .expect("widgets are made once")
    });
    window.pin_mut().show();
    if let Some(image) = lifecycle_check {
        lifecycle::start(image);
    }
    let status = ffi::exec();

    APP.with(|cell| cell.borrow_mut().as_mut().map(App::stop_loading));
    // SAFETY: the loader has stopped, so nothing queues to the waker any more. Results it queued
    // are dropped without running.
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

/// Follows Qt Wayland's `surfaceDestroyed` for the pane's current window, replacing the previous
/// connection, whose window Qt may since have replaced.
pub fn watch_surface(pane: Pane, wayland: Pin<&mut qt::QWaylandWindow>) {
    let connection = wayland.on_surface_destroyed(move |_| {
        guarded((), || dispatch(Event::SurfaceDestroyed(pane)));
    });
    let previous = SURFACE_WATCHES.with(|cell| cell.borrow_mut()[pane.index()].replace(connection));
    drop(previous);
}

/// Runs a callback from Qt. A panic unwinding into C++ would abort, so it is caught here and ends
/// the event loop instead; the panic hook has already reported it.
pub fn guarded<R>(on_panic: R, callback: impl FnOnce() -> R) -> R {
    std::panic::catch_unwind(AssertUnwindSafe(callback)).unwrap_or_else(|_| {
        ffi::exit(PANIC_EXIT_STATUS);
        on_panic
    })
}
