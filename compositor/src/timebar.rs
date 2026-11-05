//! The timebar under the viewer: the app's [`Timebar`] sent to the shim's, and the gestures it
//! sends back made into events.

use cxx_qt_lib::{QList, QString, QStringList};
use opencompositor_core::{
    Command, Direction, Event, PlaybackMode, RangeSource, Timebar, parse_timecode, timecode,
};

use crate::panel::number;
use crate::qt::ffi::{self, TimebarGestureKind};

/// The rates the timebar offers, as it writes them. NTSC's are exact fractions, which a rate
/// typed near one becomes.
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

/// A typed rate: the listed one it reads as, such as 23.976 for 24000/1001, or itself.
fn typed_rate(typed: f64) -> f64 {
    (RATES.iter())
        .find(|(_, listed)| (typed - listed).abs() < 0.0005)
        .map_or(typed, |(_, listed)| *listed)
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
        .map(|mark| u8::from(mark.cached) | u8::from(mark.missing) << 1 | u8::from(mark.held) << 2)
        .collect();
    let rate = timebar.rate;
    let data = ffi::TimebarData {
        first: timebar.first,
        last: timebar.last,
        frame: timebar.frame,
        rate,
        frame_text: QString::from(&number(timebar.frame)),
        frame_timecode: QString::from(&timecode(timebar.frame, rate)),
        first_text: QString::from(&number(timebar.first)),
        first_timecode: QString::from(&timecode(timebar.first, rate)),
        last_text: QString::from(&number(timebar.last)),
        last_timecode: QString::from(&timecode(timebar.last, rate)),
        rate_text: QString::from(&rate_text(rate)),
        playing: match timebar.playing {
            Some(Direction::Forward) => 1,
            Some(Direction::Backward) => -1,
            None => 0,
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
    let timeline = crate::read_app(|app| app.session().document().timeline());
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
        TimebarGestureKind::RateTyped => Ok(Event::FrameRateSet(typed_rate(
            text.trim().parse().unwrap_or(f64::NAN),
        ))),
        TimebarGestureKind::Restore => return refresh(),
        TimebarGestureKind::FirstFrame => Ok(Event::Command(Command::FirstFrame)),
        TimebarGestureKind::StepBackward => Ok(Event::Command(Command::Step(Direction::Backward))),
        TimebarGestureKind::PlayBackward => Ok(Event::Command(Command::Play(Direction::Backward))),
        TimebarGestureKind::PlayForward => Ok(Event::Command(Command::Play(Direction::Forward))),
        TimebarGestureKind::StepForward => Ok(Event::Command(Command::Step(Direction::Forward))),
        TimebarGestureKind::LastFrame => Ok(Event::Command(Command::LastFrame)),
        TimebarGestureKind::Looping => Ok(Event::LoopingSet(gesture.on)),
        TimebarGestureKind::RealTime => Ok(Event::PlaybackModeSet(if gesture.on {
            PlaybackMode::RealTime
        } else {
            PlaybackMode::EveryFrame
        })),
        TimebarGestureKind::InputRange => Ok(Event::RangeSourceChosen(if gesture.on {
            RangeSource::Input
        } else {
            RangeSource::Timeline
        })),
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
    fn a_typed_rate_near_a_listed_one_becomes_it_and_is_written_as_listed() {
        assert_eq!(typed_rate(23.976), 24_000.0 / 1001.0);
        assert_eq!(rate_text(typed_rate(29.97)), "29.97");
        assert_eq!(typed_rate(23.9), 23.9);
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
