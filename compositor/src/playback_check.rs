//! `--playback-check IMAGE_PATTERN [exposure | blur | merge SECOND_PATTERN] [BUDGET_MIB] [cold]`:
//! the step 5 workloads on a sequence, over the frames its scan finds, at most the first 150, at
//! the timeline's 24 frames a second. The chain is `exposure`, Reader → Exposure at one stop (W1,
//! the default); `blur`, Reader → Blur at Size 50 (W2); or `merge`, Readers of both patterns →
//! Merge → Blur at Size 50 (W3).
//!
//! 1. Every frame from storage, which look-ahead also copies into the viewer cache: recorded, not
//!    gated.
//! 2. Real time from the viewer cache, which must present every frame, with no drop, while the
//!    viewer's frames meet the GUI's budget in a GPU session: from acquiring a frame to
//!    submitting it p99 1 ms, and at most one over 4 ms. With a budget too small for every frame,
//!    the timeline is first narrowed to the longest run of frames the cache holds, which real time
//!    and the cached scrub then keep to.
//! 3. A scrub over cached frames, a frame each refresh of the viewer's screen: each frame
//!    presented p95 within two refreshes of being asked for.
//! 4. The viewed node edited, so that no frame is cached, then jumps to frames one at a time,
//!    whose p95 time to presentation is the uncached frame time; then scrubs over uncached frames,
//!    each after another edit: their frames presented p95 within that time, a refresh, and the
//!    p95 wait of those frames for the frame in progress to stop, of being asked for. The decoder
//!    stops only between the chunks it decodes, which is the cost of stopping at all. A scrub
//!    faster than evaluation presents a frame about each feedback interval, so there are several,
//!    for enough frames that their p95 is not just the slowest.
//!
//! A scrub's last frame is judged on its own, since it may wait for the frame in progress, which
//! the worker stops only within the feedback interval of starting it: it must be presented
//! within the uncached frame time and the longer of it and the interval of being asked for. The
//! viewer cache must stay within its budget, 4 GiB unless BUDGET_MIB sets another. Each stage reports the process's resident memory
//! and what wgpu's allocator holds.
//!
//! With `cold`, the sequences' files are dropped from this machine's page cache before each
//! stage that reads them, every frame and the uncached jumps and scrub; without, they are read as
//! they come. Whether the NFS server's cache was warm is the runner's to say. Run it in a release
//! build, on the workstation, pinned.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::OsString;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use open_ng::Point;
use opencompositor_core::{
    App, Changes, Command, Direction, EvaluationDevice, Event, FEEDBACK_INTERVAL, FrameTiming,
    NodeId, Pane, Pattern, PlaybackMode, Slot, Value,
};

use crate::drag::{FRAME_GUI_MAX, FRAME_GUI_P99, percentile, resident_memory};
use crate::qt::ffi;

/// How often the check looks for what it waits for, and for how long at most.
const POLL: Duration = Duration::from_millis(50);
const LONGEST_WAIT: Duration = Duration::from_secs(600);

/// The most frames played, as many as the workloads' sequences have.
const MOST_FRAMES: f64 = 150.0;

/// The most frames each scrub passes over, the jumps that time an uncached frame, and the scrubs
/// over uncached frames.
const SCRUB_FRAMES: f64 = 48.0;
const JUMPS: usize = 20;
const UNCACHED_SCRUBS: usize = 10;

/// A chain the check plays, from its Reader to the node viewed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Chain {
    Exposure,
    Blur,
    /// Over a second sequence, the Merge's layer.
    Merge(PathBuf),
}

impl Chain {
    /// The viewed node's parameter, its value, and the value the first edit sets to uncache every
    /// frame; each later edit adds a tenth of the difference again.
    fn parameter(&self) -> (&'static str, f64, f64) {
        match self {
            Chain::Exposure => ("stops", 1.0, 1.5),
            Chain::Blur | Chain::Merge(_) => ("size", 50.0, 51.0),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Chain::Exposure => "Exposure",
            Chain::Blur => "Blur",
            Chain::Merge(_) => "Merge → Blur",
        }
    }
}

/// What the command line asks the check for.
pub struct Run {
    pub pattern: PathBuf,
    pub chain: Chain,
    pub budget: Option<u64>,
    pub cold: bool,
}

impl Run {
    /// From the arguments after the pattern; `None` if they make no sense.
    pub fn parse(pattern: &OsString, rest: &[OsString]) -> Option<Run> {
        let mut run = Run {
            pattern: pattern.into(),
            chain: Chain::Exposure,
            budget: None,
            cold: false,
        };
        let mut rest = rest.iter();
        while let Some(argument) = rest.next() {
            match argument.to_str()? {
                "exposure" => run.chain = Chain::Exposure,
                "blur" => run.chain = Chain::Blur,
                "merge" => run.chain = Chain::Merge(rest.next()?.into()),
                "cold" => run.cold = true,
                mebibytes => run.budget = Some(mebibytes.parse::<u64>().ok()? << 20),
            }
        }
        Some(run)
    }
}

/// What the check is waiting for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The scans, then the first frame shown.
    Setup,
    /// Every-frame playback from storage.
    EveryFrame,
    /// The first frame shown again, of the frames the viewer cache holds.
    Rewind,
    /// Real-time playback from the viewer cache.
    RealTime,
    /// A scrub over cached frames.
    CachedScrub,
    /// The edit's frame shown.
    Edited,
    /// Jumps to uncached frames, one at a time.
    Jumps,
    /// A scrub over uncached frames, one of several.
    UncachedScrub,
}

/// A scrub: the frames asked for, with when, and how long the last took to be presented.
#[derive(Default)]
struct Scrub {
    asked: Vec<(f64, Instant)>,
    /// When the worker took each frame it delivered.
    started: Vec<(f64, Instant)>,
    /// Each frame presented, from being asked for.
    latencies: Vec<Duration>,
    /// Each frame presented, from being asked for to the worker taking it: the wait for the
    /// frame in progress to stop, which the decoder does only between the chunks it decodes.
    waits: Vec<Duration>,
    last: Option<Duration>,
}

impl Scrub {
    /// How often it asked for a frame, which a GUI thread held up does less than each refresh.
    fn pace(&self) -> Duration {
        let (Some((_, first)), Some((_, last))) = (self.asked.first(), self.asked.last()) else {
            return Duration::ZERO;
        };
        (*last - *first) / (self.asked.len().max(2) - 1) as u32
    }
}

struct Check {
    run: Run,
    device: EvaluationDevice,
    refresh: Duration,
    stage: Stage,
    waited: Duration,
    first: f64,
    last: f64,
    /// The range before a small budget narrowed it, which the uncached stages go back to.
    whole: (f64, f64),
    viewed: Option<NodeId>,
    /// The playback or scrub in progress: when it began, and each frame's first presentation.
    began: Option<Instant>,
    presented: Vec<(f64, Instant)>,
    /// The viewer's frames during real-time playback.
    frames: Vec<FrameTiming>,
    last_frame: u64,
    /// The most frames the timebar said real-time playback dropped.
    drops: u64,
    /// How long the GUI thread took for each advance of playback.
    advances: Vec<Duration>,
    /// The scrub in progress, and each once finished.
    scrub: Scrub,
    cached_scrub: Option<Scrub>,
    uncached_scrubs: Vec<Scrub>,
    /// The edits made to uncache every frame.
    edits: u32,
    /// The jump in progress, and how long each before it took.
    jump: Option<(f64, Instant)>,
    jumps: Vec<Duration>,
    lines: Vec<String>,
    failures: Vec<String>,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

fn with<R>(use_check: impl FnOnce(&mut Check) -> R) -> R {
    CHECK.with(|cell| use_check(cell.borrow_mut().as_mut().expect("the check has started")))
}

pub fn start(run: Run) {
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            run,
            device: crate::evaluation_device(),
            refresh: Duration::ZERO,
            stage: Stage::Setup,
            waited: Duration::ZERO,
            first: 0.0,
            last: 0.0,
            whole: (0.0, 0.0),
            viewed: None,
            began: None,
            presented: Vec::new(),
            frames: Vec::new(),
            last_frame: 0,
            drops: 0,
            advances: Vec::new(),
            scrub: Scrub::default(),
            cached_scrub: None,
            uncached_scrubs: Vec::new(),
            edits: 0,
            jump: None,
            jumps: Vec::new(),
            lines: Vec::new(),
            failures: Vec::new(),
        })
    });
    ffi::single_shot(1000, setup, 0);
}

/// Whether the check runs, so that the shell times each advance for it.
pub fn running() -> bool {
    CHECK.with(|cell| cell.borrow().is_some())
}

/// Records how long the GUI thread took for an advance of playback.
pub fn advanced(took: Duration) {
    CHECK.with(|cell| {
        if let Some(check) = cell.borrow_mut().as_mut()
            && check.began.is_some()
        {
            check.advances.push(took);
        }
    });
}

/// Records when the worker took a frame it delivered during a scrub.
pub fn delivered(frame: f64, started: Instant) {
    if !running() {
        return;
    }
    with(|check| {
        if matches!(check.stage, Stage::CachedScrub | Stage::UncachedScrub) {
            check.scrub.started.push((frame, started));
        }
    });
}

/// Records the viewer's frame, and the frame of the sequence it presented.
pub fn frame_drawn(pane: Pane) {
    if pane != Pane::Viewer || !running() {
        return;
    }
    let Some(timing) = crate::frame_timing(pane) else {
        return;
    };
    let (displayed, drops) = crate::read_app(|app| {
        let drops = app.timebar().map_or(0, |timebar| timebar.drops);
        (app.displayed(), drops)
    });
    with(|check| {
        if timing.number <= check.last_frame || check.began.is_none() {
            return;
        }
        check.last_frame = timing.number;
        check.drops = check.drops.max(drops);
        if let Some(displayed) = displayed.filter(|displayed| !displayed.failed)
            && check
                .presented
                .last()
                .is_none_or(|(frame, _)| *frame != displayed.frame)
        {
            check.presented.push((displayed.frame, timing.presented));
        }
        if check.stage == Stage::RealTime {
            check.frames.push(timing);
        }
    });
}

/// Makes the chain on Readers of the patterns and views it.
fn setup(_: usize) {
    let refresh =
        crate::with_pane_window(Pane::Viewer, |window| crate::pace::refresh_period(&window));
    let (pattern, chain, budget) = with(|check| {
        check.refresh = refresh;
        (
            check.run.pattern.clone(),
            check.run.chain.clone(),
            check.run.budget,
        )
    });
    let viewed = crate::edit(|session| {
        if let Some(budget) = budget {
            session.set_viewer_cache_budget(budget);
        }
        let reader = created(session.create_readers(vec![pattern], Point::default(), None));
        let (parameter, value, _) = chain.parameter();
        let viewed = match &chain {
            Chain::Exposure => created(
                (session.create_node("exposure", Point::default(), Some(reader)))
                    .expect("the type exists and the Reader exists"),
            ),
            Chain::Blur => created(
                (session.create_node("blur", Point::default(), Some(reader)))
                    .expect("the type exists and the Reader exists"),
            ),
            Chain::Merge(second) => {
                let layer =
                    created(session.create_readers(vec![second.clone()], Point::default(), None));
                let merge = created(
                    (session.create_node("merge", Point::default(), Some(reader)))
                        .expect("the type exists and the Reader exists"),
                );
                (session.connect(merge, ("layer", 1), layer)).expect("a Merge takes a layer");
                created(
                    (session.create_node("blur", Point::default(), Some(merge)))
                        .expect("the type exists and the Merge exists"),
                )
            }
        };
        (session.set_value(viewed, parameter, Value::Float(value))).expect("the node has it");
        (session.set_mark(Slot::new(1).expect("1 is a slot"), Some(viewed))).expect("it exists");
        viewed
    });
    with(|check| check.viewed = Some(viewed));
    ffi::single_shot(POLL.as_millis() as i32, wait, 0);
}

fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

/// Waits for what the stage needs, then begins the next.
fn wait(_: usize) {
    let stage = with(|check| {
        check.waited += POLL;
        check.stage
    });
    let done = match stage {
        Stage::Setup => set_range(),
        // The last frame's copy lands after it is shown, and would otherwise drop a frame from
        // the cache once real time has been kept to those it holds.
        Stage::EveryFrame => crate::read_app(|app| app.playing().is_none() && shown_is_cached(app)),
        Stage::RealTime => crate::read_app(|app| app.playing().is_none()),
        Stage::Rewind => shows(with(|check| check.first)),
        Stage::CachedScrub | Stage::UncachedScrub => scrubbed(),
        Stage::Edited => crate::read_app(|app| app.presented_job())
            .is_some_and(|shown| shown >= crate::latest_job()),
        Stage::Jumps => jumped(),
    };
    if done {
        with(|check| check.waited = Duration::ZERO);
        match stage {
            Stage::Setup if !shows(with(|check| check.first)) => {}
            Stage::Setup => {
                evict_if_cold();
                return play(Stage::EveryFrame);
            }
            Stage::EveryFrame => {
                finish_playing();
                keep_to_cached();
                with(|check| check.stage = Stage::Rewind);
                crate::dispatch(Event::Command(Command::FirstFrame));
            }
            Stage::Rewind => return play(Stage::RealTime),
            Stage::RealTime => {
                finish_playing();
                return scrub(Stage::CachedScrub);
            }
            Stage::CachedScrub => {
                finish_scrub();
                edit_viewed();
            }
            Stage::Edited => {
                evict_if_cold();
                if with(|check| check.jumps.is_empty()) {
                    with(|check| check.stage = Stage::Jumps);
                    jump();
                } else {
                    return scrub(Stage::UncachedScrub);
                }
            }
            Stage::Jumps if with(|check| check.jumps.len()) < JUMPS => jump(),
            Stage::Jumps => {
                finish_jumps();
                edit_viewed();
            }
            Stage::UncachedScrub => {
                finish_scrub();
                if with(|check| check.uncached_scrubs.len()) < UNCACHED_SCRUBS {
                    edit_viewed();
                } else {
                    finish_uncached_scrubs();
                    return report();
                }
            }
        }
    }
    if with(|check| check.waited) >= LONGEST_WAIT {
        let caption = crate::read_app(|app| format!("{:?}", app.displayed()));
        with(|check| {
            (check.failures).push(format!("{stage:?} took over {LONGEST_WAIT:?}: {caption}"));
        });
        return report();
    }
    ffi::single_shot(POLL.as_millis() as i32, wait, 0);
}

/// Sets the timeline to the frames the scan found, at most the first 150, once it has, and once
/// a second sequence's scan has too. Returns whether they have.
fn set_range() -> bool {
    let (pattern, second) = with(|check| {
        let second = match &check.run.chain {
            Chain::Merge(second) => Some(second.clone()),
            _ => None,
        };
        (check.run.pattern.clone(), second)
    });
    let (range, second_scanned) = crate::read_app(|app| {
        let files = app.session().files();
        let range = (files.scan(&pattern))
            .and_then(|scan| scan.range())
            .map(|(first, last)| {
                (
                    first as f64,
                    (last as f64).min(first as f64 + MOST_FRAMES - 1.0),
                )
            });
        (
            range,
            second.is_none_or(|second| files.scan(&second).is_some()),
        )
    });
    let Some((first, last)) = range.filter(|_| second_scanned) else {
        return false;
    };
    let timeline = crate::read_app(|app| app.session().document().timeline());
    if (timeline.first(), timeline.last()) != (first, last) {
        crate::dispatch(Event::FrameRangeSet { first, last });
        crate::dispatch(Event::Command(Command::FirstFrame));
    }
    with(|check| {
        check.first = first;
        check.last = last;
        check.whole = (first, last);
    });
    true
}

/// Whether the viewer cache holds the frame shown, as the timebar marks it.
fn shown_is_cached(app: &App) -> bool {
    let (Some(shown), Some(timebar)) = (app.displayed(), app.timebar()) else {
        return false;
    };
    let index = (shown.frame - timebar.first) as usize;
    (timebar.marks.get(index)).is_some_and(|mark| mark.cached)
}

/// Whether the viewer shows the frame asked for, evaluated or cached.
fn shows(frame: f64) -> bool {
    crate::read_app(|app| {
        app.session().frame() == frame
            && (app.displayed()).is_some_and(|shown| shown.frame == frame)
    })
}

/// Begins presenting afresh, the frame on screen already its first.
fn begin(stage: Stage) {
    let shown = crate::read_app(|app| app.displayed()).map(|displayed| displayed.frame);
    with(|check| {
        let began = Instant::now();
        check.stage = stage;
        check.began = Some(began);
        check.presented = shown.map(|frame| (frame, began)).into_iter().collect();
        check.drops = 0;
        check.advances.clear();
    });
}

fn play(stage: Stage) {
    let mode = match stage {
        Stage::RealTime => PlaybackMode::RealTime,
        _ => PlaybackMode::EveryFrame,
    };
    begin(stage);
    crate::dispatch(Event::LoopingSet(false));
    crate::dispatch(Event::PlaybackModeSet(mode));
    crate::dispatch(Event::Command(Command::Play(Direction::Forward)));
    ffi::single_shot(POLL.as_millis() as i32, wait, 0);
}

/// Describes the playback just finished, and fails real time for what it missed.
fn finish_playing() {
    let (cached, budget, bytes) = crate::read_app(|app| {
        let cache = app.session().viewer_cache();
        let cached = (app.timebar()).map_or(0, |timebar| {
            timebar.marks.iter().filter(|mark| mark.cached).count()
        });
        (cached, cache.budget(), cache.bytes())
    });
    let memory = memory();
    with(|check| {
        let name = match check.stage {
            Stage::RealTime => "real time from the viewer cache",
            _ => "every frame from storage",
        };
        let count = (check.last - check.first) as usize + 1;
        let presented = &check.presented;
        let distinct = {
            let mut frames: Vec<i64> = presented.iter().map(|(frame, _)| *frame as i64).collect();
            frames.sort_unstable();
            frames.dedup();
            frames.len()
        };
        let span = match (presented.first(), presented.last()) {
            (Some((_, first)), Some((_, last))) => *last - *first,
            _ => Duration::ZERO,
        };
        let rate = (presented.len().saturating_sub(1)) as f64 / span.as_secs_f64().max(1e-9);
        let mut gaps: Vec<Duration> = presented
            .windows(2)
            .map(|pair| pair[1].1 - pair[0].1)
            .collect();
        gaps.sort();
        let mut advances = check.advances.clone();
        advances.sort();
        check.lines.push(format!(
            "{name}: {distinct} of {count} frames presented over {span:.2?}, {rate:.1} fps; between \
             frames p50 {:.1?}, p95 {:.1?}, max {:.1?}; {} dropped by the clock; each advance on \
             the GUI thread p95 {:.2?}, max {:.2?}; afterwards {cached} frames cached, {} MiB of \
             {} MiB; {memory}",
            percentile(&gaps, 0.5),
            percentile(&gaps, 0.95),
            gaps.last().copied().unwrap_or_default(),
            check.drops,
            percentile(&advances, 0.95),
            advances.last().copied().unwrap_or_default(),
            bytes >> 20,
            budget >> 20,
        ));
        if bytes > budget {
            (check.failures).push(format!(
                "the viewer cache holds {bytes} bytes, over {budget}"
            ));
        }
        if check.stage != Stage::RealTime {
            return;
        }
        if distinct < count || check.drops > 0 {
            check.failures.push(format!(
                "real time presented {distinct} of {count} frames, and its clock dropped {}",
                check.drops
            ));
        }
        let mut gui: Vec<Duration> = (check.frames.iter())
            .map(|frame| frame.submitted - frame.acquired)
            .collect();
        gui.sort();
        let p99 = percentile(&gui, 0.99);
        check.lines.push(format!(
            "{name}: {} viewer frames, acquired → submitted p50 {:.2?}, p99 {p99:.2?}, max {:.2?}",
            gui.len(),
            percentile(&gui, 0.5),
            gui.last().copied().unwrap_or_default(),
        ));
        if check.device == EvaluationDevice::Gpu {
            if p99 > FRAME_GUI_P99 {
                (check.failures).push(format!("frames acquired → submitted p99 {p99:.2?}"));
            }
            let over = gui.iter().filter(|&&time| time > FRAME_GUI_MAX).count();
            if over > 1 {
                (check.failures).push(format!(
                    "{over} frames acquired → submitted over {FRAME_GUI_MAX:?}"
                ));
            }
        }
    });
}

/// The process's resident memory and what the GPU's allocator holds.
fn memory() -> String {
    let gpu = crate::read_app(|app| app.gpu_memory()).map_or_else(
        || "GPU allocator unreported".to_owned(),
        |gpu| {
            format!(
                "GPU allocator {} MiB in {} MiB of blocks",
                gpu.allocated >> 20,
                gpu.reserved >> 20
            )
        },
    );
    format!("resident {}; {gpu}", resident_memory())
}

/// Narrows the timeline to the longest run of frames the viewer cache holds, if a small budget
/// keeps it from holding them all, so that real time plays what fits.
fn keep_to_cached() {
    let marks: Vec<bool> = crate::read_app(|app| {
        (app.timebar()).map_or_else(Vec::new, |timebar| {
            timebar.marks.iter().map(|mark| mark.cached).collect()
        })
    });
    if marks.iter().all(|&cached| cached) {
        return;
    }
    // The longest run of cached frames: its first's index, and its length.
    let (mut longest, mut start) = ((0, 0), None);
    for (index, &cached) in marks.iter().chain([&false]).enumerate() {
        match (cached, start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                if index - from > longest.1 {
                    longest = (from, index - from);
                }
                start = None;
            }
            _ => {}
        }
    }
    with(|check| {
        let (from, length) = longest;
        if length == 0 {
            check
                .failures
                .push("the viewer cache holds no frame".to_owned());
            return;
        }
        let first = check.first + from as f64;
        let last = first + (length - 1) as f64;
        check.lines.push(format!(
            "the viewer cache holds {} of {} frames; real time and the cached scrub keep to \
             {first}–{last}",
            marks.iter().filter(|&&cached| cached).count(),
            marks.len()
        ));
        check.first = first;
        check.last = last;
    });
    let (first, last) = with(|check| (check.first, check.last));
    crate::dispatch(Event::FrameRangeSet { first, last });
}

/// Begins a scrub over the range's first frames, a frame each refresh.
fn scrub(stage: Stage) {
    begin(stage);
    with(|check| check.scrub = Scrub::default());
    scrub_step(0);
}

fn scrub_step(index: usize) {
    let (frame, count, refresh) = with(|check| {
        let count = SCRUB_FRAMES.min(check.last - check.first + 1.0) as usize;
        (check.first + index as f64, count, check.refresh)
    });
    with(|check| check.scrub.asked.push((frame, Instant::now())));
    crate::dispatch(Event::FrameSet(frame));
    if index + 1 < count {
        let period = ((refresh.as_micros() + 500) / 1000).max(1) as i32;
        ffi::single_shot(period, scrub_step, index + 1);
    } else {
        ffi::single_shot(POLL.as_millis() as i32, wait, 0);
    }
}

/// Whether the scrub's last frame asked for is presented.
fn scrubbed() -> bool {
    let last = with(|check| check.scrub.asked.last().map(|(frame, _)| *frame));
    last.is_some_and(|last| {
        shows(last) && with(|check| check.presented.iter().any(|(frame, _)| *frame == last))
    })
}

/// Measures the scrub just finished: each frame presented during it from being asked for, and
/// the last, which its own gate judges, since it may wait for a frame the worker lets finish.
fn finish_scrub() {
    let memory = memory();
    with(|check| {
        let asked: HashMap<i64, Instant> = (check.scrub.asked.iter())
            .map(|(frame, at)| (*frame as i64, *at))
            .collect();
        let (last_frame, last_asked) = *check.scrub.asked.last().expect("the scrub asked");
        // The first presentation is what was on screen when the scrub began.
        let during = || (check.presented.iter().skip(1)).filter(|(frame, _)| *frame != last_frame);
        let mut latencies: Vec<Duration> = during()
            .filter_map(|(frame, at)| Some(*at - *asked.get(&(*frame as i64))?))
            .collect();
        let mut waits: Vec<Duration> = during()
            .filter_map(|(frame, _)| {
                let (_, started) = check.scrub.started.iter().find(|(f, _)| f == frame)?;
                Some(started.saturating_duration_since(*asked.get(&(*frame as i64))?))
            })
            .collect();
        waits.sort();
        check.scrub.waits = waits;
        check.scrub.last = (check.presented.iter().rev())
            .find(|(frame, _)| *frame == last_frame)
            .map(|(_, at)| *at - last_asked);
        latencies.sort();
        check.scrub.latencies = latencies;
        let finished = std::mem::take(&mut check.scrub);
        if check.stage == Stage::CachedScrub {
            let line = scrub_line("scrub over cached frames", &[&finished], &memory);
            check.lines.push(line);
            check.cached_scrub = Some(finished);
        } else {
            check.uncached_scrubs.push(finished);
        }
    });
}

fn finish_uncached_scrubs() {
    let memory = memory();
    with(|check| {
        let scrubs: Vec<&Scrub> = check.uncached_scrubs.iter().collect();
        let line = scrub_line("scrubs over uncached frames", &scrubs, &memory);
        check.lines.push(line);
    });
}

/// Describes scrubs together: each frame presented from being asked for, the slowest last.
fn scrub_line(name: &str, scrubs: &[&Scrub], memory: &str) -> String {
    let mut latencies: Vec<Duration> = (scrubs.iter())
        .flat_map(|scrub| scrub.latencies.iter().copied())
        .collect();
    latencies.sort();
    let asked: usize = scrubs.iter().map(|scrub| scrub.asked.len()).sum();
    let pace = scrubs
        .iter()
        .map(|scrub| scrub.pace())
        .max()
        .unwrap_or_default();
    let last = scrubs.iter().filter_map(|scrub| scrub.last).max();
    let waits = stop_waits(scrubs);
    let waits = match waits.is_empty() {
        true => String::new(),
        false => format!(
            "; waiting for the frame in progress to stop p50 {:.1?}, p95 {:.1?}",
            percentile(&waits, 0.5),
            percentile(&waits, 0.95),
        ),
    };
    let lasts = match scrubs.len() {
        1 => "the last".to_string(),
        n => format!("of {n} scrubs, the slowest last"),
    };
    format!(
        "{name}: {} of {asked} frames asked for presented before the last, one asked every \
         {pace:.1?} at most; each from being asked for p50 {:.1?}, p95 {:.1?}, max {:.1?}{waits}; \
         {lasts} {:.1?}; {memory}",
        latencies.len(),
        percentile(&latencies, 0.5),
        percentile(&latencies, 0.95),
        latencies.last().copied().unwrap_or_default(),
        last.unwrap_or_default(),
    )
}

/// The waits of the scrubs' frames for the frame in progress to stop, sorted.
fn stop_waits(scrubs: &[&Scrub]) -> Vec<Duration> {
    let mut waits: Vec<Duration> = (scrubs.iter())
        .flat_map(|scrub| scrub.waits.iter().copied())
        .collect();
    waits.sort();
    waits
}

/// Edits the viewed node so that no frame is cached, over the whole range again, at its last
/// frame, which neither the jumps nor the uncached scrubs reach unless the range is short, then
/// waits for it.
fn edit_viewed() {
    let (viewed, parameter, edited, (first, last)) = with(|check| {
        check.stage = Stage::Edited;
        check.began = None;
        (check.first, check.last) = check.whole;
        let viewed = check.viewed.expect("setup made it");
        let (parameter, value, edited) = check.run.chain.parameter();
        let edited = edited + f64::from(check.edits) * (edited - value) / 10.0;
        check.edits += 1;
        (viewed, parameter, edited, check.whole)
    });
    crate::dispatch(Event::FrameRangeSet { first, last });
    crate::dispatch(Event::FrameSet(last));
    crate::edit(|session| {
        (session.set_value(viewed, parameter, Value::Float(edited))).expect("the node has it");
    });
}

/// Asks for the next jump's frame: spread over the range's second half, which the uncached
/// scrub does not reach unless the range is short.
fn jump() {
    let frame = with(|check| {
        let count = check.last - check.first + 1.0;
        let index = check.jumps.len() as f64;
        let frame = (check.first + (count / 2.0).floor() + index * (count / 2.0 / JUMPS as f64))
            .floor()
            .min(check.last);
        check.jump = Some((frame, Instant::now()));
        check.began = Some(Instant::now());
        check.presented.clear();
        frame
    });
    crate::dispatch(Event::FrameSet(frame));
}

/// Whether the jump's frame is presented, recording how long it took if so.
fn jumped() -> bool {
    let Some((frame, asked)) = with(|check| check.jump) else {
        return false;
    };
    if !shows(frame) {
        return false;
    }
    with(|check| {
        let presented = (check.presented.iter()).find(|(shown, _)| *shown == frame);
        match presented {
            Some((_, at)) => {
                check.jumps.push(*at - asked);
                check.jump = None;
                true
            }
            None => false,
        }
    })
}

fn finish_jumps() {
    with(|check| {
        let mut jumps = check.jumps.clone();
        jumps.sort();
        check.lines.push(format!(
            "an uncached frame, jumped to: p50 {:.1?}, p95 {:.1?}, max {:.1?}, of {}",
            percentile(&jumps, 0.5),
            percentile(&jumps, 0.95),
            jumps.last().copied().unwrap_or_default(),
            jumps.len()
        ));
    });
}

/// Fails the scrubs over the gates, which need the uncached frame time the jumps measured.
fn judge_scrubs(check: &mut Check) {
    let mut jumps = check.jumps.clone();
    jumps.sort();
    let uncached = percentile(&jumps, 0.95);
    let final_gate = uncached + FEEDBACK_INTERVAL.max(uncached);
    let mut uncached_latencies: Vec<Duration> = (check.uncached_scrubs.iter())
        .flat_map(|scrub| scrub.latencies.iter().copied())
        .collect();
    uncached_latencies.sort();
    let uncached_scrubs: Vec<&Scrub> = check.uncached_scrubs.iter().collect();
    let stop = percentile(&stop_waits(&uncached_scrubs), 0.95);
    let cached = check.cached_scrub.as_ref();
    let scrubs = [
        (
            "cached scrub",
            cached.map(|scrub| scrub.latencies.clone()),
            cached
                .map(|scrub| scrub.last)
                .into_iter()
                .collect::<Vec<_>>(),
            check.refresh * 2,
            "two refreshes",
        ),
        (
            "uncached scrubs",
            (!check.uncached_scrubs.is_empty()).then_some(uncached_latencies),
            check
                .uncached_scrubs
                .iter()
                .map(|scrub| scrub.last)
                .collect(),
            uncached + check.refresh + stop,
            "the uncached frame time, a refresh and the wait for the frame in progress to stop",
        ),
    ];
    for (name, latencies, lasts, gate, gate_name) in scrubs {
        let Some(latencies) = latencies else {
            continue;
        };
        let p95 = percentile(&latencies, 0.95);
        if p95 > gate {
            check.failures.push(format!(
                "the {name} presented frames p95 {p95:.1?} from being asked for, over \
                 {gate_name}, {gate:.1?}"
            ));
        }
        for last in lasts {
            match last {
                Some(last) if last <= final_gate => {}
                last => check.failures.push(format!(
                    "a last frame of the {name} took {last:.1?}, over the frame in progress and \
                     the uncached frame time, {final_gate:.1?}"
                )),
            }
        }
    }
}

/// Drops the sequences' files in the range from this machine's page cache, when the run asks.
fn evict_if_cold() {
    let (cold, patterns, first, last) = with(|check| {
        let mut patterns = vec![check.run.pattern.clone()];
        if let Chain::Merge(second) = &check.run.chain {
            patterns.push(second.clone());
        }
        (
            check.run.cold,
            patterns,
            check.first as i64,
            check.last as i64,
        )
    });
    if !cold {
        return;
    }
    let paths: Vec<PathBuf> = crate::read_app(|app| {
        let files = app.session().files();
        (patterns.iter())
            .filter_map(|pattern| {
                let scan = files.scan(pattern)?;
                let parsed = Pattern::parse(&Arc::from(pattern.as_path()))?;
                Some(
                    (scan.frames())
                        .filter(|frame| (first..=last).contains(frame))
                        .map(move |frame| parsed.frame_path(frame))
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .collect()
    });
    for path in paths {
        if let Err(error) = evict(&path) {
            with(|check| {
                (check.failures).push(format!("could not evict {}: {error}", path.display()))
            });
        }
    }
}

fn evict(path: &Path) -> std::io::Result<()> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the descriptor is the open file's, which outlives the call; the advice reads no
    // memory of ours.
    let failed = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    match failed {
        0 => Ok(()),
        error => Err(std::io::Error::from_raw_os_error(error)),
    }
}

/// Prints what the stages showed and exits: 1 with any failure, else 0.
fn report() {
    let mut check = CHECK
        .with(|cell| cell.borrow_mut().take())
        .expect("the check has started");
    judge_scrubs(&mut check);
    let name = format!(
        "playback check ({}, {}{})",
        check.run.chain.name(),
        check.device,
        if check.run.cold { ", cold" } else { "" }
    );
    println!(
        "{name}: frames from {}, a refresh {:.2?}",
        check.run.pattern.display(),
        check.refresh
    );
    for line in &check.lines {
        println!("{name}: {line}");
    }
    for failure in &check.failures {
        println!("{name}: FAILED: {failure}");
    }
    if check.failures.is_empty() {
        println!("{name}: passed");
    }
    ffi::exit(i32::from(!check.failures.is_empty()));
}
