//! Playing the timeline: the clock that moves the session's frame while the viewer plays. The
//! shell drives it with [`App::advance`](crate::App::advance) from a timer, when the app asks.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::document::Timeline;

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

/// The timebar under the viewer, as the shell shows it: the timeline, the frame, playback, and
/// what is known of each frame of the viewed node.
#[derive(Clone, Debug, PartialEq)]
pub struct Timebar {
    pub first: f64,
    pub last: f64,
    /// Frames per second.
    pub rate: f64,
    /// The session's frame.
    pub frame: f64,
    /// Which way playback plays, if it does.
    pub playing: Option<Direction>,
    /// The mode playback plays in, or will.
    pub mode: PlaybackMode,
    /// Whether playback wraps round at the timeline's end, rather than stopping there.
    pub looping: bool,
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
/// walk, about 2 µs on the workstation, whenever the document or the files change.
pub const MAX_MARKED_FRAMES: i64 = 2_000;

/// How long before the latest frame shown the achieved rate counts frames over.
const RATE_WINDOW: Duration = Duration::from_secs(1);

/// The frames playback steps through: the timeline's first frame and each whole frame after it,
/// up to its last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Frames {
    first: f64,
    count: i64,
}

impl Frames {
    pub(crate) fn of(timeline: &Timeline) -> Frames {
        Frames {
            first: timeline.first(),
            count: (timeline.last() - timeline.first()).floor() as i64 + 1,
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
    /// Steps taken since it started.
    steps: i64,
    /// In every-frame mode, when the next step may be taken.
    due: Option<Instant>,
    /// Whether the clock's frame has been shown since the clock stepped to it.
    shown: bool,
    /// When each frame recently shown was first shown, oldest first.
    recent: VecDeque<Instant>,
    /// Frames the clock passed without their being shown, in real-time mode.
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

    /// Whether the clock's frame has been shown since it stepped to it.
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
