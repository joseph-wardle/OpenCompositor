//! Draws each pane at most once per refresh of its screen.
//!
//! Qt delivers a window's `requestUpdate` at once unless Qt itself presents to the window, and
//! here wgpu does: unpaced, a pane redraws as often as input arrives, and on Wayland the GUI thread
//! then waits in `get_current_texture` for frames the display has not taken.

use std::cell::RefCell;
use std::time::{Duration, Instant};

use opencompositor_core::Pane;

use crate::qt::{self, ffi};

/// Used when the pane's screen does not report its refresh rate.
const FALLBACK_REFRESH_HZ: f64 = 60.0;

thread_local! {
    static PANES: RefCell<[Pacing; Pane::COUNT]> = RefCell::default();
}

#[derive(Clone, Copy, Debug, Default)]
struct Pacing {
    /// When the pane last began to draw for a redraw.
    drew: Option<Instant>,
    /// Whether a timer will request the pane's next draw.
    waiting: bool,
}

/// Has the pane draw once it has not drawn within a refresh of its screen: at once if it has
/// not, otherwise from a timer. Redraws before then coalesce into that draw.
pub fn redraw(pane: Pane) {
    let refresh = crate::with_pane_window(pane, |window| refresh_period(&window));
    let wait = PANES.with(|cell| {
        let pacing = &mut cell.borrow_mut()[pane.index()];
        if pacing.waiting {
            return None;
        }
        let wait = pacing.drew.map_or(Duration::ZERO, |drew| {
            refresh.saturating_sub(drew.elapsed())
        });
        pacing.waiting = !wait.is_zero();
        Some(wait)
    });
    match wait {
        None => {}
        Some(Duration::ZERO) => request_update(pane),
        // To the nearest millisecond, Qt's timer resolution, so that draws keep the refresh
        // rate on average.
        Some(wait) => {
            let milliseconds = (wait.as_micros() + 500) / 1000;
            ffi::single_shot(milliseconds as i32, wait_ended, pane.index());
        }
    }
}

/// Records that the pane is drawing for a redraw.
pub fn drawing(pane: Pane) {
    PANES.with(|cell| cell.borrow_mut()[pane.index()].drew = Some(Instant::now()));
}

fn wait_ended(index: usize) {
    PANES.with(|cell| cell.borrow_mut()[index].waiting = false);
    request_update(Pane::ALL[index]);
}

/// Qt coalesces requests until it delivers the pane's `UpdateRequest`.
fn request_update(pane: Pane) {
    crate::with_pane_window(pane, |window| window.request_update());
}

fn refresh_period(window: &qt::Pane) -> Duration {
    // SAFETY: Qt keeps a window's screen alive while it is the window's screen, moving its windows
    // to another screen before deleting it.
    let rate = unsafe { window.screen().as_ref() }.map_or(0.0, ffi::QScreen::refresh_rate);
    let rate = if rate > 0.0 {
        rate
    } else {
        FALLBACK_REFRESH_HZ
    };
    Duration::from_secs_f64(1.0 / rate)
}
