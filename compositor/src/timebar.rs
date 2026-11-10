//! The timebar under the viewer: the app's [`Timebar`] sent to the shim's, and the gestures it
//! sends back made into events.

use cxx_qt_lib::{QList, QString, QStringList};
use opencompositor_core::{
    Command, Direction, Event, PlaybackMode, RangeSource, Timebar, parse_timecode, timecode,
};

use crate::panel::number;
use crate::qt::ffi::{self, TimebarGestureKind};

/// The rates the timebar offers, as it writes them. NTSC's are exact fractions, which a rate
/// typed near one becomes in the app.
const RATES: [(&str, f64); 9] = [
    ("23.976", 24_000.0 / 1001.0),
    ("24", 24.0),
    ("25", 25.0),
    ("29.97", 30_000.0 / 1001.0),
    ("30", 30.0),
    ("48", 48.0),
    ("50", 50.0),
    ("59.94", 60_000.0 / 1001.0),
    ("60", 60.0),
];

/// The rates the timebar's rate menu lists.
pub fn rates() -> QStringList {
    let rates: QList<QString> = RATES.iter().map(|(text, _)| QString::from(*text)).collect();
    QStringList::from(&rates)
}

/// A rate as the timebar writes it.
fn rate_text(rate: f64) -> String {
    match RATES.iter().find(|(_, listed)| *listed == rate) {
        Some((text, _)) => (*text).to_owned(),
        None => number(rate),
    }
}

pub fn frame_label(frame: f64, rate: f64, timecode: bool) -> String {
    if timecode {
        self::timecode(frame, rate)
    } else {
        number(frame)
    }
}

pub fn show(timebar: &Timebar) {
    let status = match (timebar.playing, timebar.achieved) {
        (None, _) | (Some(_), None) => String::new(),
        (Some(_), Some(achieved)) if timebar.mode == PlaybackMode::RealTime => {
            format!("{achieved:.1} fps · {} dropped", timebar.drops)
        }
        (Some(_), Some(achieved)) => format!("{achieved:.1} fps"),
    };
    let marks = (timebar.marks.iter())
        .map(|mark| ffi::FrameMark {
            cached: mark.cached,
            missing: mark.missing,
            held: mark.held,
        })
        .collect();
    let (rate, timecode) = (timebar.rate, timebar.timecode);
    let text = |frame| QString::from(&frame_label(frame, rate, timecode));
    let data = ffi::TimebarData {
        first: timebar.first,
        last: timebar.last,
        frame: timebar.frame,
        rate,
        timecode,
        frame_text: text(timebar.frame),
        first_text: text(timebar.first),
        last_text: text(timebar.last),
        rate_text: QString::from(&rate_text(rate)),
        playing: match timebar.playing {
            Some(Direction::Forward) => ffi::Playing::Forward,
            Some(Direction::Backward) => ffi::Playing::Backward,
            None => ffi::Playing::Stopped,
        },
        real_time: timebar.mode == PlaybackMode::RealTime,
        looping: timebar.looping,
        input: timebar.source == RangeSource::Input,
        input_found: timebar.input_found,
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

/// A typed frame: a number, or timecode at the timeline's rate.
fn typed_frame(text: &str, rate: f64) -> Result<f64, String> {
    if text.contains(':') {
        parse_timecode(text, rate).map_err(|error| error.to_string())
    } else {
        // Text that is no number goes in as NaN, which the app refuses as from a script.
        Ok(text.trim().parse().unwrap_or(f64::NAN))
    }
}

pub fn gesture(gesture: ffi::TimebarGesture) {
    let text = String::from(&gesture.text);
    let (timeline, shown) = crate::read_app(|app| {
        let timeline = app.session().document().timeline();
        let shown = (app.timebar()).map(|shown| (shown.looping, shown.mode, shown.timecode));
        (timeline, shown)
    });
    // The timebar is shown before the window takes gestures.
    let Some((looping, mode, timecode)) = shown else {
        return;
    };
    let typed = || typed_frame(&text, timeline.rate());
    let event = match gesture.kind {
        TimebarGestureKind::Scrubbed => Ok(Event::FrameSet(gesture.frame)),
        TimebarGestureKind::FrameTyped => typed().map(Event::FrameSet),
        TimebarGestureKind::FirstTyped => typed().map(|first| Event::FrameRangeSet {
            first,
            last: timeline.last(),
        }),
        TimebarGestureKind::LastTyped => typed().map(|last| Event::FrameRangeSet {
            first: timeline.first(),
            last,
        }),
        TimebarGestureKind::RateTyped => {
            Ok(Event::FrameRateSet(text.trim().parse().unwrap_or(f64::NAN)))
        }
        TimebarGestureKind::Restore => return refresh(),
        TimebarGestureKind::FirstFrame => Ok(Event::Command(Command::FirstFrame)),
        TimebarGestureKind::StepBackward => Ok(Event::Command(Command::Step(Direction::Backward))),
        TimebarGestureKind::PlayBackward => Ok(Event::Command(Command::Play(Direction::Backward))),
        TimebarGestureKind::PlayForward => Ok(Event::Command(Command::Play(Direction::Forward))),
        TimebarGestureKind::StepForward => Ok(Event::Command(Command::Step(Direction::Forward))),
        TimebarGestureKind::LastFrame => Ok(Event::Command(Command::LastFrame)),
        TimebarGestureKind::ToggleLooping => Ok(Event::LoopingSet(!looping)),
        TimebarGestureKind::ToggleRealTime => Ok(Event::PlaybackModeSet(match mode {
            PlaybackMode::EveryFrame => PlaybackMode::RealTime,
            PlaybackMode::RealTime => PlaybackMode::EveryFrame,
        })),
        TimebarGestureKind::ToggleTimecode => Ok(Event::TimecodeSet(!timecode)),
        TimebarGestureKind::GlobalRange => Ok(Event::RangeSourceChosen(RangeSource::Timeline)),
        TimebarGestureKind::InputRange => Ok(Event::RangeSourceChosen(RangeSource::Input)),
        _ => return,
    };
    let typing = matches!(
        gesture.kind,
        TimebarGestureKind::FrameTyped
            | TimebarGestureKind::FirstTyped
            | TimebarGestureKind::LastTyped
            | TimebarGestureKind::RateTyped
    );
    match event {
        Ok(event) => crate::dispatch(event),
        // SAFETY: the window is live while the timebar sends gestures.
        Err(why) => unsafe {
            ffi::show_status(
                crate::widgets().window,
                &QString::from(&format!("Cannot go to the frame: {why}")),
            )
        },
    }
    // A field keeps what was typed until the timebar changes; after a refusal it must catch up.
    if typing {
        refresh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listed_rate_is_written_as_listed() {
        assert_eq!(rate_text(opencompositor_core::ntsc_rate(29.97)), "29.97");
        assert_eq!(rate_text(12.5), "12.5");
    }

    #[test]
    fn a_typed_frame_is_a_number_or_timecode() {
        assert_eq!(typed_frame(" 1001 ", 24.0), Ok(1001.0));
        assert_eq!(typed_frame("00:00:41:17", 24.0), Ok(1001.0));
        assert!(typed_frame("00:00:00:30", 24.0).is_err());
        assert!(typed_frame("x", 24.0).unwrap().is_nan());
    }
}
