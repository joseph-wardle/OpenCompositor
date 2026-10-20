//! OpenCompositor's Qt shell. Qt owns the windows, docks, dialogs and event loop; the core's
//! [`App`] owns everything else. Everything here runs on Qt's GUI thread.

mod creation;
mod drag;
mod keys;
mod lifecycle;
mod pace;
mod pane;
mod panel;
mod qt;

use std::cell::{OnceCell, RefCell};
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Instant;

use cxx_qt::{QMetaObjectConnectionGuard, Threading};
use cxx_qt_lib::{QList, QString};
use opencompositor_core::{
    App, Command, Event, FileIntent, FrameCheck, Notice, Pane, Request, Session, Slot,
};

use crate::qt::ffi;

/// A `Qt::DockWidgetArea` value.
const TOP_DOCK_AREA: i32 = 4;

thread_local! {
    /// Borrowed only while the app handles an event, during which nothing calls Qt: Qt can call
    /// back into the shell synchronously, as when floating a dock destroys a pane's surface.
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    static WIDGETS: OnceCell<Widgets> = const { OnceCell::new() };
    /// The menus' commands, by the index each item passes to `run_command`.
    static COMMANDS: OnceCell<Vec<Command>> = const { OnceCell::new() };
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
    pub panel: *mut ffi::ParameterPanel,
    pub creation_menu: *mut ffi::CreationMenu,
    /// Under the viewer.
    pub caption: *mut ffi::QLabel,
    /// At the status bar's right, while the GPU is unavailable or has reported errors.
    pub gpu_label: *mut ffi::QLabel,
    pub undo: *mut ffi::QAction,
    pub redo: *mut ffi::QAction,
}

/// A scripted check of the shell, from the command line.
enum Check {
    Lifecycle(PathBuf),
    Drag(PathBuf),
}

fn main() {
    let check = match &std::env::args_os().skip(1).collect::<Vec<_>>()[..] {
        [] => None,
        [flag, image] if flag == "--lifecycle-check" => Some(Check::Lifecycle(image.into())),
        [flag, image] if flag == "--drag-check" => Some(Check::Drag(image.into())),
        _ => {
            eprintln!("usage: opencompositor [--lifecycle-check IMAGE | --drag-check IMAGE]");
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
    let mut app = App::new(move |notice| {
        // Fails only once the waker is deleted at exit, when nothing is left to tell.
        let _ = waker_thread.queue(move |_| receive(notice));
    });
    if matches!(check, Some(Check::Lifecycle(_))) {
        app.check_frames();
    }
    APP.with(|cell| *cell.borrow_mut() = Some(app));

    let mut window = ffi::new_main_window();
    window
        .pin_mut()
        .set_window_title(&QString::from("OpenCompositor"));
    window.pin_mut().set_dock_nesting_enabled(true);
    window.pin_mut().resize(1400, 900);
    let window_ptr = window.as_mut_ptr();
    // SAFETY: the window is live; it owns the menus.
    let [file_menu, edit_menu, node_menu, viewer_menu, window_menu] =
        ["&File", "&Edit", "&Node", "&Viewer", "&Window"]
            .map(|title| unsafe { ffi::new_menu(window_ptr, &QString::from(title)) });
    let mut commands = Vec::new();
    let mut add = |menu, text: &str, shortcut: &str, command| {
        let [text, shortcut] = [text, shortcut].map(QString::from);
        // SAFETY: the menu is live; it owns the action.
        let action =
            unsafe { ffi::new_action(menu, &text, &shortcut, run_command, commands.len()) };
        commands.push(command);
        action
    };
    add(file_menu, "&Read Files…", "R", Command::ReadFiles);
    add(
        file_menu,
        "Re&load the Selected Readers",
        "",
        Command::Reload,
    );
    let undo = add(edit_menu, "&Undo", "Ctrl+Z", Command::Undo);
    let redo = add(edit_menu, "&Redo", "Ctrl+Shift+Z", Command::Redo);
    // SAFETY: the node menu is live; it owns the note.
    unsafe { ffi::add_note(node_menu, &QString::from("Create a Node\tTab")) };
    // The key filter takes the digit row by position and H for the pane under the pointer, so
    // these items only name their keys, after a tab, rather than binding shortcuts.
    let slot_menus = [
        ("&View Slot", "", Command::ViewSlot as fn(Slot) -> Command),
        (
            "&Mark the Selected Node in Slot",
            "Ctrl+",
            Command::MarkSlot,
        ),
        ("&Frame the Node in Slot", "Shift+", Command::FrameMark),
    ];
    for (title, modifier, command) in slot_menus {
        // SAFETY: the viewer menu is live; it owns the submenu.
        let menu = unsafe { ffi::new_submenu(viewer_menu, &QString::from(title)) };
        for digit in (1..=9).chain([0]) {
            let slot = Slot::new(digit).expect("a digit");
            add(
                menu,
                &format!("Slot {digit}\t{modifier}{digit}"),
                "",
                command(slot),
            );
        }
    }
    // SAFETY: the viewer menu is live.
    unsafe { ffi::add_separator(viewer_menu) };
    add(viewer_menu, "&Fit the Image\tH", "", Command::FitImage);
    add(
        viewer_menu,
        "&Actual Size\tShift+H",
        "",
        Command::ActualSize,
    );
    // SAFETY: the viewer menu is live; it owns the note.
    unsafe { ffi::add_note(viewer_menu, &QString::from("Pan\tSpace+Drag, Middle-Drag")) };
    COMMANDS.with(|cell| cell.set(commands).expect("menus are made once"));
    // In one area, which the parameter panel's dock then divides.
    let docks = Pane::ALL.map(|pane| {
        let title = match pane {
            Pane::Graph => "Node Graph",
            Pane::Viewer => "Viewer",
        };
        // SAFETY: the window and menu are live; the window owns the dock.
        unsafe {
            ffi::new_dock(
                window_ptr,
                window_menu,
                &QString::from(title),
                TOP_DOCK_AREA,
            )
        }
    });
    let panes = Pane::ALL.map(|pane| ffi::new_pane(pane.index()));
    let [graph_dock, viewer_dock] = Pane::ALL.map(|pane| docks[pane.index()]);
    // SAFETY: the window, menu and docks are live.
    let panel = unsafe {
        ffi::new_parameter_panel(
            window_ptr,
            window_menu,
            viewer_dock,
            graph_dock,
            panel::gesture,
        )
    };
    // SAFETY: the docks are live; each owns its pane, and the viewer's its caption, from here on.
    let caption = unsafe {
        ffi::container(panes[Pane::Graph.index()], graph_dock);
        ffi::container_with_caption(panes[Pane::Viewer.index()], viewer_dock)
    };
    // SAFETY: the window is live; it owns the menu.
    let creation_menu = unsafe { ffi::new_creation_menu(window_ptr, creation::chosen) };
    // SAFETY: the window is live; it owns the label.
    let gpu_label = unsafe { ffi::new_permanent_status(window_ptr) };
    WIDGETS.with(|cell| {
        cell.set(Widgets {
            window: window_ptr,
            docks,
            panes,
            panel,
            creation_menu,
            caption,
            gpu_label,
            undo,
            redo,
        })
        .expect("widgets are made once")
    });
    show_undo_labels([None, None]);
    keys::start();
    window.pin_mut().show();
    match check {
        Some(Check::Lifecycle(image)) => lifecycle::start(image),
        Some(Check::Drag(image)) => drag::start(image),
        None => {}
    }
    let status = ffi::exec();

    // The app goes before the window: its surfaces before the panes' native windows, and its
    // requests with it, so that the events destroying the window dispatch to nothing. Its worker
    // delivers nothing after it.
    drop(APP.with(|cell| cell.borrow_mut().take()));
    // SAFETY: nothing else deletes the waker. Results queued to it are dropped without running.
    unsafe { ffi::delete_waker(waker) };
    drop(window);
    std::process::exit(status);
}

/// Hands an event to the app, then does what it asks.
pub fn dispatch(event: Event) {
    let draws_viewer = matches!(event, Event::Draw(Pane::Viewer));
    let Some(requests) = APP.with(|cell| cell.borrow_mut().as_mut().map(|app| app.dispatch(event)))
    else {
        return;
    };
    if draws_viewer {
        drag::viewer_frame_drawn();
    }
    for request in requests {
        perform(request);
    }
}

/// Edits the session, then does what the app asks. Returns what the edit returned.
pub fn edit<R>(edit: impl FnOnce(&mut Session) -> R) -> R {
    let (requests, edited) = APP.with(|cell| {
        let mut app = cell.borrow_mut();
        app.as_mut()
            .expect("the panel and the drag check edit while the app runs")
            .edit(edit)
    });
    for request in requests {
        perform(request);
    }
    edited
}

/// Reads the app: its session in the checks, its panel for a gesture on it.
pub fn read_app<R>(read: impl FnOnce(&App) -> R) -> R {
    APP.with(|cell| read(cell.borrow().as_ref().expect("the app runs")))
}

/// Names in the Edit menu what undo and redo would do, and disables either when there is nothing.
fn show_undo_labels(labels: [Option<&str>; 2]) {
    let widgets = widgets();
    for ((action, verb), label) in [(widgets.undo, "&Undo"), (widgets.redo, "&Redo")]
        .into_iter()
        .zip(labels)
    {
        let text = match label {
            Some(label) => format!("{verb} {label}"),
            None => verb.to_owned(),
        };
        // SAFETY: the Edit menu owns the action, and lives as long as the window.
        unsafe { ffi::set_action(action, &QString::from(&text), label.is_some()) };
    }
}

fn run_command(index: usize) {
    let command =
        COMMANDS.with(|cell| cell.get().expect("menus are made before the event loop")[index]);
    dispatch(Event::Command(command));
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
                FileIntent::CreateReaders(_) => "Read Files",
                FileIntent::SetFile { .. } => "Choose a File",
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
        // SAFETY: the window is live.
        Request::ShowError(text) => unsafe {
            ffi::show_error(widgets.window, &QString::from(&text))
        },
        // SAFETY: the caption lives as long as the viewer's dock.
        Request::ShowViewerCaption(text) => unsafe {
            ffi::set_caption(widgets.caption, &QString::from(&text))
        },
        Request::ShowUndoLabels { undo, redo } => show_undo_labels([undo, redo]),
        Request::ShowCreationMenu(anchor) => creation::show(anchor),
        Request::ShowPanel(shown) => panel::show(shown.as_ref()),
        // SAFETY: the window owns the label, and lives as long as it.
        Request::ShowGpuLabel(text) => unsafe {
            let text = QString::from(text.as_deref().unwrap_or_default());
            ffi::set_permanent_status(widgets.gpu_label, &text)
        },
        Request::RetryDeviceCreation { attempt, after } => {
            let milliseconds = i32::try_from(after.as_millis()).unwrap_or(i32::MAX);
            ffi::single_shot(milliseconds, retry_device_creation, attempt as usize);
        }
    }
}

/// Dispatches a notice from another thread. Retiring a lost device blocks the GUI thread, so
/// the lifecycle check reports how long; the drag check records each result's arrival.
fn receive(notice: Notice) {
    let lost = matches!(notice, Notice::DeviceLost(_));
    let evaluated = match &notice {
        Notice::Evaluated(evaluated) => Some((evaluated.request(), evaluated.timing)),
        Notice::DeviceLost(_) => None,
    };
    let started = Instant::now();
    dispatch(Event::from(notice));
    if lost {
        lifecycle::retirement_took(started.elapsed());
    }
    if let Some((request, timing)) = evaluated {
        drag::delivered(request, timing, started);
    }
}

/// Every attempt is dispatched; the app ignores all but the last. One whose timer fires after
/// the app has dropped dispatches to nothing.
fn retry_device_creation(attempt: usize) {
    let started = Instant::now();
    dispatch(Event::RetryDeviceCreation(attempt as u64));
    lifecycle::attempt_took(started.elapsed());
}

pub fn path(name: &QString) -> PathBuf {
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

/// What the pane's frames have proved, in the lifecycle check.
pub fn frame_check(pane: Pane) -> Option<FrameCheck> {
    APP.with(|cell| cell.borrow().as_ref()?.frame_check(pane))
}

/// Destroys the GPU device, in the lifecycle check.
pub fn destroy_device() {
    APP.with(|cell| {
        if let Some(app) = cell.borrow_mut().as_mut() {
            app.destroy_device();
        }
    });
}

/// How many GPU errors this session went uncaptured.
pub fn gpu_errors() -> u64 {
    APP.with(|cell| cell.borrow().as_ref().map_or(0, |app| app.gpu_errors()))
}

/// The number of the latest request to the worker, in the drag check.
pub fn latest_request() -> u64 {
    APP.with(|cell| cell.borrow().as_ref().map_or(0, App::latest_request))
}

/// The number of the request whose image the viewer last presented, in the drag check.
pub fn presented_request() -> Option<u64> {
    APP.with(|cell| cell.borrow().as_ref()?.presented_request())
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
    // Rust reads the pane's own state with `rust()` alone; a `rust_mut()` held across a dispatch
    // would put a real `&mut PaneRust` under one held further up.
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
