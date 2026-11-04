//! The timebar under the viewer: the app's [`Timebar`] sent to the shim's, and the gestures it
//! sends back made into events.

use cxx_qt_lib::QString;
use opencompositor_core::{Command, Direction, Event, PlaybackMode, Timebar};

use crate::panel::number;
use crate::qt::ffi::{self, TimebarGestureKind};

pub fn show(timebar: &Timebar) {
    let status = match (timebar.playing, timebar.achieved) {
        (None, _) | (Some(_), None) => String::new(),
        (Some(_), Some(achieved)) if timebar.mode == PlaybackMode::RealTime => {
            format!("{achieved:.1} fps, {} dropped", timebar.drops)
        }
        (Some(_), Some(achieved)) => format!("{achieved:.1} fps"),
    };
    let marks = (timebar.marks.iter())
        .map(|mark| u8::from(mark.cached) | u8::from(mark.missing) << 1 | u8::from(mark.held) << 2)
        .collect();
    let data = ffi::TimebarData {
        first: timebar.first,
        last: timebar.last,
        frame: timebar.frame,
        frame_text: QString::from(&number(timebar.frame)),
        first_text: QString::from(&number(timebar.first)),
        last_text: QString::from(&number(timebar.last)),
        rate_text: QString::from(&number(timebar.rate)),
        playing: match timebar.playing {
            Some(Direction::Forward) => 1,
            Some(Direction::Backward) => -1,
            None => 0,
        },
        real_time: timebar.mode == PlaybackMode::RealTime,
        looping: timebar.looping,
        status: QString::from(&status),
        marks,
    };
    // SAFETY: the strip lives as long as the viewer's dock, until the main window is destroyed
    // after the event loop.
    unsafe { ffi::show_timebar(crate::widgets().strip, &data) };
}

/// Shows the app's timebar again, putting back what a refused or unchanged edit left in a field.
fn refresh() {
    if let Some(timebar) = crate::read_app(|app| app.timebar().cloned()) {
        show(&timebar);
    }
}

pub fn gesture(gesture: ffi::TimebarGesture) {
    // Text that is no number goes in as NaN, which the app refuses as from a script.
    let typed = || -> f64 {
        String::from(&gesture.text)
            .trim()
            .parse()
            .unwrap_or(f64::NAN)
    };
    let range = || crate::read_app(|app| app.session().document().timeline());
    let event = match gesture.kind {
        TimebarGestureKind::Scrubbed => Event::FrameSet(gesture.frame),
        TimebarGestureKind::FrameTyped => Event::FrameSet(typed()),
        TimebarGestureKind::FirstTyped => Event::FrameRangeSet {
            first: typed(),
            last: range().last(),
        },
        TimebarGestureKind::LastTyped => Event::FrameRangeSet {
            first: range().first(),
            last: typed(),
        },
        TimebarGestureKind::RateTyped => Event::FrameRateSet(typed()),
        TimebarGestureKind::Restore => return refresh(),
        TimebarGestureKind::FirstFrame => Event::Command(Command::FirstFrame),
        TimebarGestureKind::StepBackward => Event::Command(Command::Step(Direction::Backward)),
        TimebarGestureKind::PlayBackward => Event::Command(Command::Play(Direction::Backward)),
        TimebarGestureKind::Stop => Event::Command(Command::Stop),
        TimebarGestureKind::PlayForward => Event::Command(Command::Play(Direction::Forward)),
        TimebarGestureKind::StepForward => Event::Command(Command::Step(Direction::Forward)),
        TimebarGestureKind::LastFrame => Event::Command(Command::LastFrame),
        TimebarGestureKind::Looping => Event::LoopingSet(gesture.on),
        TimebarGestureKind::RealTime => Event::PlaybackModeSet(if gesture.on {
            PlaybackMode::RealTime
        } else {
            PlaybackMode::EveryFrame
        }),
        _ => return,
    };
    let typing = matches!(
        gesture.kind,
        TimebarGestureKind::FrameTyped
            | TimebarGestureKind::FirstTyped
            | TimebarGestureKind::LastTyped
            | TimebarGestureKind::RateTyped
    );
    crate::dispatch(event);
    // A field keeps what was typed until the timebar changes; after a refusal it must catch up.
    if typing {
        refresh();
    }
}
