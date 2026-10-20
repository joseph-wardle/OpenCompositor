//! The window's keys, taken before any window or widget sees them, so that they work whichever
//! pane, dock or bar has the keyboard, and with none. The app passes pane keys to the pane under
//! the pointer.

use open_ng::Modifiers;
use opencompositor_core::{Event, Key, Pane, Slot};

use crate::qt::ffi;

/// `QEvent::Type` values.
const KEY_PRESS: i32 = 6;
const KEY_RELEASE: i32 = 7;
const APPLICATION_DEACTIVATE: i32 = 122;

/// `Qt::Key` values.
mod qt_key {
    pub const SPACE: i32 = 0x20;
    pub const G: i32 = 0x47;
    pub const H: i32 = 0x48;
    pub const Y: i32 = 0x59;
    pub const ESCAPE: i32 = 0x0100_0000;
    pub const BACKSPACE: i32 = 0x0100_0003;
    pub const DELETE: i32 = 0x0100_0007;
}

/// The XKB keycodes of the digit row, 1 to 9 then 0: positions, whatever the layout types there.
const DIGIT_ROW: std::ops::RangeInclusive<u32> = 10..=19;

pub fn start() {
    ffi::filter_keys(handle);
}

/// Handles an event the application's filter passes on. Returns whether to stop it there.
fn handle(kind: i32, fields: ffi::InputFields) -> bool {
    if kind == APPLICATION_DEACTIVATE {
        // Releases go to the application with the keyboard, and buttons may be released there.
        for pane in Pane::ALL {
            crate::dispatch(Event::FocusLost(pane));
        }
        return false;
    }
    let Some(key) = key(&fields) else {
        return false;
    };
    match kind {
        // A release is never stopped, so that whatever saw the press sees it too.
        KEY_RELEASE => {
            if !fields.auto_repeat {
                crate::dispatch(Event::KeyReleased(key));
            }
            false
        }
        // Alt and Meta belong to menus' mnemonics and the desktop's shortcuts.
        KEY_PRESS if !fields.alt && !fields.meta && !ffi::keyboard_taken() => {
            if !fields.auto_repeat {
                let modifiers = Modifiers {
                    shift: fields.shift,
                    ctrl: fields.ctrl,
                };
                crate::dispatch(Event::KeyPressed { key, modifiers });
            }
            true
        }
        _ => false,
    }
}

fn key(fields: &ffi::InputFields) -> Option<Key> {
    if DIGIT_ROW.contains(&fields.native_scan_code) {
        let digit = (fields.native_scan_code - DIGIT_ROW.start() + 1) % 10;
        return Slot::new(digit as u8).map(Key::Slot);
    }
    Some(match fields.key {
        qt_key::SPACE => Key::Space,
        qt_key::ESCAPE => Key::Escape,
        qt_key::G => Key::G,
        qt_key::H => Key::H,
        qt_key::Y => Key::Y,
        qt_key::BACKSPACE | qt_key::DELETE => Key::Delete,
        _ => return None,
    })
}
