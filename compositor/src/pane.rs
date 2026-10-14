//! Qt events on a pane, translated into the app's events, and the pane's native window.

use std::pin::Pin;
use std::ptr::NonNull;

use open_ng::{Button, Key, Modifiers, PhysicalPoint};
use opencompositor_core::{Event, Input, NativeWindow, Pane, PaneSize};
use wgpu::rwh::{
    RawDisplayHandle, RawWindowHandle, WaylandDisplayHandle, WaylandWindowHandle,
    XlibDisplayHandle, XlibWindowHandle,
};

use crate::qt::{self, ffi};

/// `QEvent::Type` values.
mod kind {
    pub const MOUSE_BUTTON_PRESS: i32 = 2;
    pub const MOUSE_BUTTON_RELEASE: i32 = 3;
    pub const MOUSE_MOVE: i32 = 5;
    pub const KEY_PRESS: i32 = 6;
    pub const KEY_RELEASE: i32 = 7;
    pub const FOCUS_OUT: i32 = 9;
    pub const ENTER: i32 = 10;
    pub const LEAVE: i32 = 11;
    pub const RESIZE: i32 = 14;
    pub const WHEEL: i32 = 31;
    pub const UPDATE_REQUEST: i32 = 77;
    pub const EXPOSE: i32 = 206;
    pub const PLATFORM_SURFACE: i32 = 217;
    pub const DEVICE_PIXEL_RATIO_CHANGE: i32 = 222;
}

/// `Qt::Key` values.
mod key {
    pub const SPACE: i32 = 0x20;
    pub const G: i32 = 0x47;
    pub const H: i32 = 0x48;
    pub const Y: i32 = 0x59;
    pub const ESCAPE: i32 = 0x0100_0000;
}

/// `Qt::MouseButton` values.
mod button {
    pub const LEFT: i32 = 1;
    pub const MIDDLE: i32 = 4;
}

/// Handles a Qt event on a pane, passing it on to `QWindow` unless the app used it.
///
/// # Safety
///
/// `event` is live for the duration of the call.
pub unsafe fn event(mut window: Pin<&mut qt::Pane>, pane: Pane, event: *mut ffi::QEvent) -> bool {
    // SAFETY: as above.
    let kind = unsafe { ffi::event_type(event) };
    // SAFETY: as above.
    let fields = || unsafe { ffi::input_fields(event) };
    let used = match kind {
        kind::EXPOSE => {
            if window.is_exposed() {
                crate::dispatch(Event::Exposed(pane));
            }
            false
        }
        kind::PLATFORM_SURFACE => {
            if fields().surface_about_to_be_destroyed {
                crate::dispatch(Event::SurfaceDestroyed(pane));
            }
            false
        }
        kind::RESIZE | kind::DEVICE_PIXEL_RATIO_CHANGE => {
            let size = size(&window);
            crate::dispatch(Event::Resized { pane, size });
            false
        }
        kind::UPDATE_REQUEST => {
            crate::dispatch(Event::Draw(pane));
            true
        }
        kind::FOCUS_OUT => {
            crate::dispatch(Event::FocusLost(pane));
            false
        }
        _ => match input(kind, &fields, window.device_pixel_ratio()) {
            Some(input) => {
                crate::dispatch(Event::Input { pane, input });
                true
            }
            None => false,
        },
    };
    // SAFETY: as above.
    used || unsafe { window.as_mut().base_event(event) }
}

pub fn size(window: &qt::Pane) -> PaneSize {
    let scale = window.device_pixel_ratio();
    let physical = |logical: i32| (f64::from(logical) * scale).round() as u32;
    PaneSize {
        width: physical(window.width()),
        height: physical(window.height()),
        scale_factor: scale as f32,
    }
}

/// The app's input for a Qt input event, if the app uses it.
fn input(kind: i32, fields: &impl Fn() -> ffi::InputFields, scale: f64) -> Option<Input> {
    let position = |f: &ffi::InputFields| PhysicalPoint {
        x: (f.x * scale) as f32,
        y: (f.y * scale) as f32,
    };
    let button = |f: &ffi::InputFields| match f.button {
        button::LEFT => Some(Button::Left),
        button::MIDDLE => Some(Button::Middle),
        _ => None,
    };
    Some(match kind {
        kind::ENTER => Input::PointerEntered,
        kind::LEAVE => Input::PointerLeft,
        kind::MOUSE_MOVE => Input::PointerMoved {
            position: position(&fields()),
        },
        kind::MOUSE_BUTTON_PRESS => {
            let f = fields();
            Input::PointerPressed {
                position: position(&f),
                button: button(&f)?,
                modifiers: Modifiers {
                    shift: f.shift,
                    ctrl: f.ctrl,
                },
            }
        }
        kind::MOUSE_BUTTON_RELEASE => {
            let f = fields();
            Input::PointerReleased {
                position: position(&f),
                button: button(&f)?,
            }
        }
        kind::WHEEL => {
            let f = fields();
            Input::Scrolled {
                position: position(&f),
                lines: f.wheel_y as f32 / 120.0,
            }
        }
        kind::KEY_PRESS | kind::KEY_RELEASE => {
            let f = fields();
            let key = match f.key {
                key::SPACE => Key::Space,
                key::ESCAPE => Key::Escape,
                key::G => Key::G,
                key::H => Key::H,
                key::Y => Key::Y,
                _ => return None,
            };
            if f.auto_repeat {
                return None;
            }
            if kind == kind::KEY_PRESS {
                Input::KeyPressed(key)
            } else {
                Input::KeyReleased(key)
            }
        }
        _ => return None,
    })
}

/// The pane's native window, or `None` while it has none or the platform has no windows. On
/// Wayland it also watches for Qt destroying the window's surface, which floating a dock does
/// without a platform surface event.
pub fn native_window(mut window: Pin<&mut qt::Pane>, pane: Pane) -> Option<NativeWindow> {
    let (display, handle) = match String::from(&ffi::platform_name()).as_str() {
        "wayland" => {
            let wayland = ffi::wayland_window(window.as_mut());
            // SAFETY: Qt Wayland's window lives as long as the pane's platform window, which is
            // live while the pane is exposed.
            let mut wayland = unsafe { Pin::new_unchecked(wayland.as_mut()?) };
            crate::watch_surface(pane, wayland.as_mut());
            let display = NonNull::new(ffi::wayland_display() as *mut _)?;
            let surface = NonNull::new(ffi::wayland_surface(wayland) as *mut _)?;
            (
                RawDisplayHandle::Wayland(WaylandDisplayHandle::new(display)),
                RawWindowHandle::Wayland(WaylandWindowHandle::new(surface)),
            )
        }
        "xcb" => {
            let display = NonNull::new(ffi::x11_display() as *mut _)?;
            (
                RawDisplayHandle::Xlib(XlibDisplayHandle::new(Some(display), 0)),
                RawWindowHandle::Xlib(XlibWindowHandle::new(ffi::x11_window(&window) as _)),
            )
        }
        // Platforms such as offscreen have no window to draw to.
        _ => return None,
    };
    // SAFETY: the pane dispatches `SurfaceDestroyed` on `SurfaceAboutToBeDestroyed` and on Qt
    // Wayland's `surfaceRoleDestroyed`, before either handle dies.
    Some(unsafe { NativeWindow::new(display, handle) })
}
