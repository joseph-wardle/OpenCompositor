//! Every Qt declaration the shell uses: Qt's own headers where `cxx` can bind them, and
//! `cpp/shim.h` one-liners where it cannot.

#[cxx_qt::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("cxx-qt-lib/qstring.h");
        type QString = cxx_qt_lib::QString;
        include!("cxx-qt-lib/qstringlist.h");
        type QStringList = cxx_qt_lib::QStringList;
        include!("cxx-qt-lib/qbytearray.h");
        type QByteArray = cxx_qt_lib::QByteArray;
        include!(<QtCore/QEvent>);
        /// Opaque: the shim reads its fields.
        #[allow(missing_debug_implementations)]
        type QEvent;
        include!(<QtWidgets/QMenu>);
        #[allow(missing_debug_implementations)]
        type QMenu;
        include!(<QtWidgets/QLabel>);
        #[allow(missing_debug_implementations)]
        type QLabel;
    }

    unsafe extern "C++Qt" {
        include!(<QtWidgets/QMainWindow>);
        include!(<QtWidgets/QDockWidget>);
        include!(<QtWidgets/QWidget>);
        include!(<QtGui/QWindow>);
        include!(<QtGui/QScreen>);

        #[qobject]
        type QWidget;
        #[qobject]
        type QMainWindow;
        #[qobject]
        type QDockWidget;
        #[qobject]
        type QWindow;
        #[qobject]
        type QScreen;

        fn resize(self: Pin<&mut QMainWindow>, width: i32, height: i32);
        fn show(self: Pin<&mut QMainWindow>);
        #[cxx_name = "setWindowTitle"]
        fn set_window_title(self: Pin<&mut QMainWindow>, title: &QString);
        #[cxx_name = "setDockNestingEnabled"]
        fn set_dock_nesting_enabled(self: Pin<&mut QMainWindow>, enabled: bool);

        #[cxx_name = "setFloating"]
        fn set_floating(self: Pin<&mut QDockWidget>, floating: bool);
        fn raise(self: Pin<&mut QDockWidget>);
        fn show(self: Pin<&mut QDockWidget>);
        fn close(self: Pin<&mut QDockWidget>) -> bool;

        #[cxx_name = "refreshRate"]
        fn refresh_rate(self: &QScreen) -> f64;
    }

    // Qt Wayland's per-window object (private QPA API), for its native surface.
    unsafe extern "C++Qt" {
        include!(<qpa/qplatformwindow_p.h>);
        #[qobject]
        #[namespace = "QNativeInterface::Private"]
        type QWaylandWindow;

        /// Emitted when Qt Wayland unmaps the window and before it destroys the window's
        /// `wl_surface`, including when floating a dock recreates it without a platform surface
        /// event. (`surfaceDestroyed` comes only after the `wl_surface` is gone.)
        #[qsignal]
        #[cxx_name = "surfaceRoleDestroyed"]
        fn surface_role_destroyed(self: Pin<&mut QWaylandWindow>);
    }

    // A pane: a native window the GPU draws, inside a dock.
    extern "RustQt" {
        #[qobject]
        #[base = QWindow]
        type Pane = super::PaneRust;
    }

    // QWindow's constructor takes a QWindow parent, not CXX-Qt's default QObject one.
    impl cxx_qt::Constructor<(usize,), NewArguments = (usize,)> for Pane {}

    unsafe extern "RustQt" {
        #[cxx_override]
        unsafe fn event(self: Pin<&mut Pane>, event: *mut QEvent) -> bool;

        #[inherit]
        #[cxx_name = "event"]
        unsafe fn base_event(self: Pin<&mut Pane>, event: *mut QEvent) -> bool;

        #[inherit]
        #[cxx_name = "requestUpdate"]
        fn request_update(self: Pin<&mut Pane>);
        #[inherit]
        #[cxx_name = "devicePixelRatio"]
        fn device_pixel_ratio(self: &Pane) -> f64;
        #[inherit]
        fn width(self: &Pane) -> i32;
        #[inherit]
        fn height(self: &Pane) -> i32;
        #[inherit]
        fn screen(self: &Pane) -> *mut QScreen;
        #[inherit]
        #[cxx_name = "isExposed"]
        fn is_exposed(self: &Pane) -> bool;
    }

    // Receives the loader's results on the GUI thread. It lives as long as the event loop, unlike a
    // pane, which closes with its dock.
    extern "RustQt" {
        #[qobject]
        type Waker = super::WakerRust;
    }

    impl cxx_qt::Threading for Waker {}

    /// What the shell reads from a Qt event, positions in logical pixels.
    #[derive(Debug)]
    struct InputFields {
        x: f64,
        y: f64,
        button: i32,
        shift: bool,
        ctrl: bool,
        wheel_y: i32,
        key: i32,
        /// The key's position on the keyboard: an XKB keycode on Wayland and X11.
        native_scan_code: u32,
        auto_repeat: bool,
        surface_about_to_be_destroyed: bool,
    }

    #[namespace = "shim"]
    unsafe extern "C++" {
        include!("shim.h");

        fn new_application();
        fn exec() -> i32;
        fn exit(code: i32);
        fn platform_name() -> QString;
        /// The Qt the shim was compiled against.
        fn qt_compiled_version() -> QString;
        /// The Qt loaded at run time.
        fn qt_runtime_version() -> QString;
        fn single_shot(ms: i32, callback: fn(usize), argument: usize);

        fn new_main_window() -> UniquePtr<QMainWindow>;
        /// A menu at the end of the window's menu bar, which owns it.
        unsafe fn new_menu(window: *mut QMainWindow, title: &QString) -> *mut QMenu;
        /// A menu item that calls `triggered(command)`, also from its shortcut, whichever pane has
        /// the keyboard.
        unsafe fn new_action(
            menu: *mut QMenu,
            text: &QString,
            shortcut: &QString,
            triggered: fn(usize),
            command: usize,
        );
        /// A dock in a `Qt::DockWidgetArea`, listed in `menu`.
        unsafe fn new_dock(
            window: *mut QMainWindow,
            menu: *mut QMenu,
            title: &QString,
            area: i32,
        ) -> *mut QDockWidget;
        /// Puts the pane in the dock, which owns it from then on.
        unsafe fn container(pane: *mut Pane, dock: *mut QDockWidget) -> *mut QWidget;
        /// Puts the pane in the dock above a one-line caption, which the dock owns too.
        unsafe fn container_with_caption(pane: *mut Pane, dock: *mut QDockWidget) -> *mut QLabel;
        /// Sets the caption's text, and its tooltip for when the text is clipped.
        unsafe fn set_caption(label: *mut QLabel, text: &QString);
        unsafe fn caption(label: *mut QLabel) -> QString;
        unsafe fn tabify(
            window: *mut QMainWindow,
            first: *mut QDockWidget,
            second: *mut QDockWidget,
        );
        unsafe fn show_status(window: *mut QMainWindow, text: &QString);
        /// Runs a modal file dialog. Its nested event loop delivers paints, resizes, timers and
        /// the loader's results, but no input to the panes.
        unsafe fn choose_files(
            window: *mut QMainWindow,
            title: &QString,
            several: bool,
        ) -> QStringList;
        /// A path as the file system names it.
        fn encode_name(path: &QString) -> QByteArray;
        fn close_modal_dialog();

        /// Calls `handle(event type, fields)` for every key press and release in the application,
        /// before any window or widget receives it, and for the application losing the
        /// keyboard. The event goes no further if `handle` returns true.
        fn filter_keys(handle: fn(i32, InputFields) -> bool);
        /// Whether a text field, a dialog or a menu has the keyboard, rather than the window.
        fn keyboard_taken() -> bool;

        fn new_pane(index: usize) -> *mut Pane;
        fn new_waker() -> *mut Waker;
        unsafe fn delete_waker(waker: *mut Waker);

        unsafe fn event_type(event: *const QEvent) -> i32;
        unsafe fn input_fields(event: *const QEvent) -> InputFields;
        fn wayland_display() -> usize;
        fn wayland_window(pane: Pin<&mut Pane>) -> *mut QWaylandWindow;
        fn wayland_surface(window: Pin<&mut QWaylandWindow>) -> usize;
        fn x11_display() -> usize;
        fn x11_window(pane: &Pane) -> u64;
    }
}

use std::pin::Pin;

use cxx_qt::CxxQtType;

pub use ffi::{Pane, QWaylandWindow};

/// A pane holds only its index into [`opencompositor_core::Pane::ALL`]; the state is the app's.
#[derive(Debug)]
pub struct PaneRust {
    index: usize,
}

#[derive(Debug, Default)]
pub struct WakerRust;

impl cxx_qt::Constructor<(usize,)> for ffi::Pane {
    type BaseArguments = ();
    type InitializeArguments = ();
    type NewArguments = (usize,);

    fn route_arguments(arguments: (usize,)) -> ((usize,), (), ()) {
        (arguments, (), ())
    }

    fn new((index,): (usize,)) -> PaneRust {
        PaneRust { index }
    }
}

impl ffi::Pane {
    /// # Safety
    ///
    /// Qt passes a live event for the duration of the call.
    unsafe fn event(self: Pin<&mut Self>, event: *mut ffi::QEvent) -> bool {
        let pane = opencompositor_core::Pane::ALL[self.rust().index];
        // SAFETY: as above.
        unsafe { crate::pane::event(self, pane, event) }
    }
}
