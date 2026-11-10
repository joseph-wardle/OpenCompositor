//! Playing the timeline: the clock that moves the session's frame while the viewer plays. The
//! shell drives it with [`App::advance`](crate::App::advance) from a timer, when the app asks.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::app::Request;
use crate::files::input_range;
use crate::session::Session;

/// Which way playback steps through the timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// How playback keeps time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Shows every frame: steps once the frame is shown, and no faster than the timeline's rate.
    #[default]
    EveryFrame,
    /// Keeps to the timeline's rate, showing the newest frame ready and counting those it drops.
    RealTime,
}

/// Where the frames the timebar spans and playback plays through come from. The viewer's choice,
/// not the document's; the shell calls the timeline's Global, as Nuke does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RangeSource {
    /// The timeline's, which the document holds.
    #[default]
    Timeline,
    /// The viewed node's sequences', as [`input_range`](crate::input_range) finds them, following
    /// the viewer; the timeline's while it finds none.
    Input,
}

/// The timebar under the viewer, as the shell shows it: the timeline, the frame, playback, and
/// what is known of each frame of the viewed node.
#[derive(Clone, Debug, PartialEq)]
pub struct Timebar {
    pub first: f64,
    pub last: f64,
    pub source: RangeSource,
    /// With [`RangeSource::Input`], whether the viewed node's range was found; if not, the range
    /// is the timeline's.
    pub input_found: bool,
    /// Frames per second.
    pub rate: f64,
    pub frame: f64,
    pub playing: Option<Direction>,
    pub mode: PlaybackMode,
    pub looping: bool,
    /// Whether frames read as [`timecode`] rather than as numbers.
    pub timecode: bool,
    /// While playing, the frames shown per second over the last second, to a tenth, once
    /// measured.
    pub achieved: Option<f64>,
    /// While playing in real time, the frames passed without being shown.
    pub drops: u64,
    /// Each whole frame's from the first, while a node is viewed and the timeline has at most
    /// [`MAX_MARKED_FRAMES`]; otherwise empty.
    pub marks: Vec<FrameMark>,
}

/// What is known of a frame of the viewed node, without evaluating it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameMark {
    /// The viewer cache holds a copy of it that the viewer can show at its zoom.
    pub cached: bool,
    /// A Reader upstream lacks the frame's file, so the frame fails.
    pub missing: bool,
    /// A Reader upstream is outside the frames of its sequence, and holds its nearest end.
    pub held: bool,
}

/// The longest timeline whose frames the timebar marks: each frame costs a key and an upstream
/// walk whenever the document or the files change.
pub const MAX_MARKED_FRAMES: i64 = 2_000;

/// The frames per second a timecode counts at a rate: the whole number nearest it, so that
/// 23.976 counts as 24, without dropping frame numbers.
fn timecode_base(rate: f64) -> i64 {
    (rate.round() as i64).max(1)
}

/// A frame as non-drop timecode, `HH:MM:SS:FF`, counted from frame 0 at the rate's
/// [nearest whole number](timecode_base), negative before it; a frame between whole frames as the
/// one before it.
pub fn timecode(frame: f64, rate: f64) -> String {
    let base = timecode_base(rate);
    let whole = frame.floor() as i64;
    let sign = if whole < 0 { "-" } else { "" };
    let count = whole.unsigned_abs();
    let base = base as u64;
    let (frames, seconds) = (count % base, count / base);
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    format!("{sign}{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
}

/// A rate within 0.0005 of an NTSC rate, such as 23.976, as that rate's exact fraction, such as
/// 24000/1001; any other rate as it is.
pub fn ntsc_rate(rate: f64) -> f64 {
    ([24_000.0, 30_000.0, 60_000.0].map(|numerator| numerator / 1001.0))
        .into_iter()
        .find(|ntsc| (rate - ntsc).abs() < 0.0005)
        .unwrap_or(rate)
}

/// Why typed timecode is not a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimecodeError {
    /// It has other than two to four parts, or a part is not a number.
    Unreadable,
    /// Minutes or seconds of 60 or more, or frames of the rate's whole number or more.
    OutOfRange,
}

impl std::fmt::Display for TimecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimecodeError::Unreadable => write!(f, "not timecode, HH:MM:SS:FF"),
            TimecodeError::OutOfRange => write!(f, "a part of the timecode is out of range"),
        }
    }
}

impl std::error::Error for TimecodeError {}

/// The frame [`timecode`] writes as the text, which may leave out leading parts: `SS:FF`,
/// `MM:SS:FF` or `HH:MM:SS:FF`, after an optional `-`.
pub fn parse_timecode(text: &str, rate: f64) -> Result<f64, TimecodeError> {
    let text = text.trim();
    let (negative, text) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let parts: Vec<u64> = (text.split(':'))
        .map(|part| part.trim().parse::<u64>())
        .collect::<Result<_, _>>()
        .map_err(|_| TimecodeError::Unreadable)?;
    if !(2..=4).contains(&parts.len()) {
        return Err(TimecodeError::Unreadable);
    }
    let base = timecode_base(rate) as u64;
    let mut fields = [0u64; 4];
    fields[4 - parts.len()..].copy_from_slice(&parts);
    let [hours, minutes, seconds, frames] = fields;
    if minutes >= 60 || seconds >= 60 || frames >= base || hours > 1_000_000 {
        return Err(TimecodeError::OutOfRange);
    }
    let count = ((hours * 60 + minutes) * 60 + seconds) * base + frames;
    Ok(if negative {
        -(count as f64)
    } else {
        count as f64
    })
}

/// How the artist chose playback to play and the timebar to read, and playback in progress.
#[derive(Debug)]
pub(crate) struct Playback {
    clock: Option<Clock>,
    mode: PlaybackMode,
    pub(crate) looping: bool,
    pub(crate) timecode: bool,
    pub(crate) range_source: RangeSource,
    /// Whether a [`Request::Advance`] is waiting to be dispatched.
    advance_requested: bool,
}

/// The first and last frames the timebar spans and playback plays through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Range {
    pub(crate) first: f64,
    pub(crate) last: f64,
    pub(crate) input_found: bool,
}

impl Range {
    pub(crate) fn frames(self) -> Frames {
        Frames::between(self.first, self.last)
    }
}

impl Default for Playback {
    fn default() -> Playback {
        Playback {
            clock: None,
            mode: PlaybackMode::default(),
            looping: true,
            timecode: false,
            range_source: RangeSource::default(),
            advance_requested: false,
        }
    }
}

impl Playback {
    pub(crate) fn playing(&self) -> Option<(Direction, PlaybackMode)> {
        let clock = self.clock.as_ref()?;
        Some((clock.direction(), clock.mode()))
    }

    pub(crate) fn direction(&self) -> Option<Direction> {
        self.clock.as_ref().map(Clock::direction)
    }

    /// Starts playing that way in the mode chosen, from the next advance, asked for at once.
    pub(crate) fn play(&mut self, direction: Direction) -> Option<Request> {
        self.clock = Some(Clock::new(direction, self.mode));
        self.request_advance(Duration::ZERO)
    }

    pub(crate) fn stop(&mut self) {
        self.clock = None;
    }

    /// Plays in the mode from now on, starting again at once if playing in another.
    pub(crate) fn set_mode(&mut self, mode: PlaybackMode) -> Option<Request> {
        self.mode = mode;
        match &self.clock {
            Some(clock) if clock.mode() != mode => self.play(clock.direction()),
            _ => None,
        }
    }

    /// The range chosen as the session stands: the timeline's, unless the viewed node's
    /// sequences are chosen and found.
    pub(crate) fn range(&self, session: &Session) -> Range {
        let timeline = session.document().timeline();
        let input = match (self.range_source, session.viewed()) {
            (RangeSource::Input, Some(node)) => {
                input_range(session.document(), session.files(), node, session.frame())
            }
            _ => None,
        };
        match input {
            Some((first, last)) => Range {
                first,
                last,
                input_found: true,
            },
            None => Range {
                first: timeline.first(),
                last: timeline.last(),
                input_found: false,
            },
        }
    }

    pub(crate) fn frames(&self, session: &Session) -> Frames {
        self.range(session).frames()
    }

    /// The frame one step that way from the session's: round to the range's other end when
    /// looping, and from a frame outside the range or between frames the next whole frame.
    pub(crate) fn step(&self, session: &Session, direction: Direction) -> f64 {
        let frame = session.frame();
        let frames = self.frames(session);
        match frames.index(frame) {
            Some(index) => match frames.step(index, 1, direction, self.looping) {
                Ok(index) | Err(index) => frames.frame(index),
            },
            None => match direction {
                Direction::Forward => frame.floor() + 1.0,
                Direction::Backward => frame.ceil() - 1.0,
            },
        }
    }

    /// Moves the clock to `now`. Returns the frame to go to, if it stepped, and the next advance
    /// when the clock knows when; in every-frame mode, [`Playback::follow_shown`] asks once the
    /// frame is shown.
    pub(crate) fn advance(
        &mut self,
        now: Instant,
        session: &Session,
    ) -> (Option<f64>, Option<Request>) {
        self.advance_requested = false;
        let frames = self.frames(session);
        let Some(clock) = &mut self.clock else {
            return (None, None);
        };
        let rate = session.document().timeline().rate();
        let tick = clock.advance(now, session.frame(), frames, rate, self.looping);
        if tick.stopped {
            self.clock = None;
        }
        let next = (tick.next)
            .and_then(|next| self.request_advance(next.saturating_duration_since(Instant::now())));
        (tick.frame, next)
    }

    /// Tells the clock whether its frame is shown, and in every-frame mode asks for the advance
    /// that steps once it is due.
    pub(crate) fn follow_shown(&mut self, shown: bool) -> Option<Request> {
        let clock = self.clock.as_mut()?;
        if shown {
            clock.frame_shown(Instant::now());
        }
        if clock.mode() == PlaybackMode::EveryFrame && clock.is_shown() {
            self.request_advance(Duration::ZERO)
        } else {
            None
        }
    }

    fn request_advance(&mut self, after: Duration) -> Option<Request> {
        if self.advance_requested {
            return None;
        }
        self.advance_requested = true;
        Some(Request::Advance { after })
    }

    pub(crate) fn timebar(&self, session: &Session, marks: &[FrameMark]) -> Timebar {
        let range = self.range(session);
        let clock = self.clock.as_ref();
        Timebar {
            first: range.first,
            last: range.last,
            source: self.range_source,
            input_found: range.input_found,
            rate: session.document().timeline().rate(),
            frame: session.frame(),
            playing: clock.map(Clock::direction),
            mode: self.mode,
            looping: self.looping,
            timecode: self.timecode,
            achieved: (clock.and_then(Clock::achieved_rate))
                .map(|rate| (rate * 10.0).round() / 10.0),
            drops: clock.map_or(0, Clock::drops),
            marks: marks.to_vec(),
        }
    }
}

/// How long before the latest frame shown the achieved rate counts frames over.
const RATE_WINDOW: Duration = Duration::from_secs(1);

/// The frames playback steps through: a range's first frame and each whole frame after it, up to
/// its last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Frames {
    first: f64,
    count: i64,
}

impl Frames {
    /// The frames from `first` to `last`, which is not before it.
    pub(crate) fn between(first: f64, last: f64) -> Frames {
        Frames {
            first,
            count: (last - first).floor() as i64 + 1,
        }
    }

    pub(crate) fn count(self) -> i64 {
        self.count
    }

    pub(crate) fn frame(self, index: i64) -> f64 {
        self.first + index as f64
    }

    /// The index of a frame of these, if it is one.
    pub(crate) fn index(self, frame: f64) -> Option<i64> {
        let index = frame - self.first;
        (index.fract() == 0.0 && (0.0..self.count as f64).contains(&index)).then_some(index as i64)
    }

    /// The index playback starts from at a frame: the nearest whole frame, or the end it plays
    /// from, the first going forwards, if the frame is outside.
    fn start(self, frame: f64, direction: Direction) -> i64 {
        let index = (frame - self.first).round();
        if (0.0..self.count as f64).contains(&index) {
            index as i64
        } else {
            match direction {
                Direction::Forward => 0,
                Direction::Backward => self.count - 1,
            }
        }
    }

    /// The index `steps` frames on from `index`, round to the other end when looping, otherwise
    /// `Err` with the end it stops at.
    pub(crate) fn step(
        self,
        index: i64,
        steps: i64,
        direction: Direction,
        looping: bool,
    ) -> Result<i64, i64> {
        let moved = match direction {
            Direction::Forward => index + steps,
            Direction::Backward => index - steps,
        };
        if (0..self.count).contains(&moved) {
            Ok(moved)
        } else if looping {
            Ok(moved.rem_euclid(self.count))
        } else {
            Err(moved.clamp(0, self.count - 1))
        }
    }
}

/// Playback in progress.
#[derive(Clone, Debug)]
pub(crate) struct Clock {
    direction: Direction,
    mode: PlaybackMode,
    /// When the clock started, at its first advance, and the index it started from.
    started: Option<(Instant, i64)>,
    steps: i64,
    /// In every-frame mode, when the next step may be taken.
    due: Option<Instant>,
    /// Whether the clock's frame has been shown since the clock stepped to it.
    shown: bool,
    /// When each frame recently shown was first shown, oldest first.
    recent: VecDeque<Instant>,
    drops: u64,
}

/// What an advance of the clock did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tick {
    /// The frame to go to, if the clock stepped.
    pub(crate) frame: Option<f64>,
    /// Whether playback reached the end and stopped, not looping.
    pub(crate) stopped: bool,
    /// When to advance next; `None` until the frame is shown, in every-frame mode.
    pub(crate) next: Option<Instant>,
}

impl Clock {
    pub(crate) fn new(direction: Direction, mode: PlaybackMode) -> Clock {
        Clock {
            direction,
            mode,
            started: None,
            steps: 0,
            due: None,
            shown: false,
            recent: VecDeque::new(),
            drops: 0,
        }
    }

    pub(crate) fn direction(&self) -> Direction {
        self.direction
    }

    pub(crate) fn mode(&self) -> PlaybackMode {
        self.mode
    }

    pub(crate) fn drops(&self) -> u64 {
        self.drops
    }

    /// Frames shown per second over the last second of frames shown, once two have been.
    pub(crate) fn achieved_rate(&self) -> Option<f64> {
        let (first, last) = (self.recent.front()?, self.recent.back()?);
        let span = last.duration_since(*first).as_secs_f64();
        (self.recent.len() >= 2 && span > 0.0).then(|| (self.recent.len() - 1) as f64 / span)
    }

    /// Records that the clock's frame is shown, once per frame.
    pub(crate) fn frame_shown(&mut self, now: Instant) {
        if self.shown {
            return;
        }
        self.shown = true;
        self.recent.push_back(now);
        while (self.recent.front()).is_some_and(|&first| now.duration_since(first) > RATE_WINDOW) {
            self.recent.pop_front();
        }
    }

    pub(crate) fn is_shown(&self) -> bool {
        self.shown
    }

    /// Moves the clock to `now`, from the session's `frame`. The first advance starts it, playing
    /// from the frame, or from the end it plays from if the frame is outside the timeline.
    pub(crate) fn advance(
        &mut self,
        now: Instant,
        frame: f64,
        frames: Frames,
        rate: f64,
        looping: bool,
    ) -> Tick {
        let period = Duration::from_secs_f64(1.0 / rate);
        let Some((started, start)) = self.started else {
            let start = frames.start(frame, self.direction);
            self.started = Some((now, start));
            self.due = Some(now + period);
            let moved = frames.frame(start);
            return Tick {
                frame: (moved != frame).then_some(moved),
                stopped: false,
                next: Some(now + period),
            };
        };
        let steps = match self.mode {
            PlaybackMode::EveryFrame => {
                let due = self.due.unwrap_or(now);
                if !self.shown || now < due {
                    return Tick {
                        frame: None,
                        stopped: false,
                        next: self.shown.then_some(due),
                    };
                }
                // A step later than a period behind its time waited for the frame, so the next
                // is timed from now; one less late was the timer's, and keeps the rate.
                let on_time = if now.duration_since(due) > period {
                    now
                } else {
                    due
                };
                self.due = Some(on_time + period);
                1
            }
            PlaybackMode::RealTime => {
                // A tick timed for a frame can land a nanosecond short of it, rounding.
                let elapsed = now.duration_since(started).as_secs_f64();
                let target = (elapsed * rate + 1e-6).floor() as i64;
                let next = started + Duration::from_secs_f64((target + 1) as f64 / rate);
                let steps = target - self.steps;
                if steps <= 0 {
                    return Tick {
                        frame: None,
                        stopped: false,
                        next: Some(next),
                    };
                }
                self.drops += (steps - 1) as u64 + u64::from(!self.shown);
                steps
            }
        };
        self.steps += steps;
        self.shown = false;
        let next = match self.mode {
            PlaybackMode::EveryFrame => None,
            PlaybackMode::RealTime => {
                Some(started + Duration::from_secs_f64((self.steps + 1) as f64 / rate))
            }
        };
        match frames.step(start, self.steps, self.direction, looping) {
            Ok(index) => Tick {
                frame: Some(frames.frame(index)),
                stopped: false,
                next,
            },
            Err(end) => Tick {
                frame: Some(frames.frame(end)),
                stopped: true,
                next: None,
            },
        }
    }
}

#[cfg(test)]
mod tests;
