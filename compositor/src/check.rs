//! What the scripted checks share: what the shell tells them, their statistics, the GUI's frame
//! budget, and how they report.

use std::fmt;
use std::time::{Duration, Instant};

use opencompositor_core::{FrameTiming, Pane, Timing, Window};

use crate::qt::ffi;
use crate::{drag, lifecycle, playback_check};

/// Something the shell did that a check may be timing.
pub enum Seen {
    /// The pane drew a frame.
    Frame(Pane, FrameTiming),
    /// The GUI thread advanced playback, taking so long.
    Advanced(Duration),
    /// A result arrived on the GUI thread at `received`, and was dispatched.
    Delivered {
        job: u64,
        frame: f64,
        timing: Timing,
        window: Option<Window>,
        received: Instant,
    },
    /// Retiring a lost device blocked the GUI thread so long.
    Retired(Duration),
    /// An attempt to make a GPU device blocked the GUI thread so long.
    Attempted(Duration),
}

/// Tells the checks; those not running ignore it.
pub fn observe(seen: Seen) {
    drag::observe(&seen);
    playback_check::observe(&seen);
    lifecycle::observe(&seen);
}

/// Durations, sorted, for their percentiles.
#[derive(Clone, Debug, Default)]
pub struct Durations(Vec<Duration>);

impl Durations {
    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The value at the quantile, or zero of none.
    pub fn p(&self, quantile: f64) -> Duration {
        match self.0.len() {
            0 => Duration::ZERO,
            len => self.0[((len - 1) as f64 * quantile).round() as usize],
        }
    }

    pub fn max(&self) -> Duration {
        self.p(1.0)
    }

    pub fn iter(&self) -> impl Iterator<Item = Duration> + '_ {
        self.0.iter().copied()
    }
}

impl FromIterator<Duration> for Durations {
    fn from_iter<I: IntoIterator<Item = Duration>>(durations: I) -> Durations {
        let mut durations: Vec<Duration> = durations.into_iter().collect();
        durations.sort();
        Durations(durations)
    }
}

/// `p50 …, p95 …, max …`, to the precision given, a tenth by default.
impl fmt::Display for Durations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = f.precision().unwrap_or(1);
        write!(
            f,
            "p50 {:.digits$?}, p95 {:.digits$?}, max {:.digits$?}",
            self.p(0.5),
            self.p(0.95),
            self.max()
        )
    }
}

/// The GUI's frame budget in a GPU session: from acquiring a frame to submitting it, p99 and
/// at most, but for one frame per run: the driver alone occasionally takes milliseconds to
/// submit.
const FRAME_GUI_P99: Duration = Duration::from_millis(1);
const FRAME_GUI_MAX: Duration = Duration::from_millis(4);

/// The frames from acquiring to submitting, described; in a GPU session, failed against the
/// GUI's budget.
pub fn gui_frames(frames: &[FrameTiming], gpu: bool, failures: &mut Vec<String>) -> String {
    let times: Durations = frames
        .iter()
        .map(|frame| frame.submitted - frame.acquired)
        .collect();
    let p99 = times.p(0.99);
    if gpu {
        if p99 > FRAME_GUI_P99 {
            failures.push(format!(
                "frames acquired → submitted p99 {p99:.2?}, over {FRAME_GUI_P99:?}"
            ));
        }
        let over: Vec<Duration> = times.iter().filter(|&time| time > FRAME_GUI_MAX).collect();
        if over.len() > 1 {
            failures.push(format!(
                "{} frames acquired → submitted over {FRAME_GUI_MAX:?}: {over:.2?}",
                over.len()
            ));
        }
    }
    format!(
        "{} frames, acquired → submitted p50 {:.2?}, p99 {p99:.2?}, max {:.2?}",
        times.len(),
        times.p(0.5),
        times.max()
    )
}

/// The process's resident set from /proc, as Linux reports it.
pub fn resident_memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .map_or_else(|| "unknown".to_owned(), |rss| rss.trim().to_owned())
}

/// Prints the failures, or that the check passed, and exits: 1 with any failure, else 0.
pub fn finish(name: &str, failures: &[String]) {
    for failure in failures {
        eprintln!("{name} failed: {failure}");
    }
    if failures.is_empty() {
        eprintln!("{name}: passed");
    }
    ffi::exit(i32::from(!failures.is_empty()));
}
