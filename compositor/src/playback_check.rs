//! `--playback-check IMAGE_PATTERN [exposure | blur | merge SECOND_PATTERN] [BUDGET_MIB] [cold]`:
//! the step 5 workloads over at most the first 150 frames the scan finds, at 24 frames a second.
//! `exposure` is Reader → Exposure at one stop (W1, the default); `blur`, Reader → Blur at Size
//! 50 (W2); `merge`, Readers of both patterns → Merge → Blur at Size 50 (W3). BUDGET_MIB sets the
//! viewer cache's budget, 4 GiB by default; one too small for every frame narrows real time and
//! the cached scrub to the longest run of frames the cache holds.
//!
//! `cold` drops the sequences' files from this machine's page cache before each stage that reads
//! them; whether the NFS server's cache was warm is the runner's to say.

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

use crate::check::{Durations, Seen, finish, gui_frames, resident_memory};
use crate::qt::ffi;

const POLL: Duration = Duration::from_millis(50);
const LONGEST_WAIT: Duration = Duration::from_secs(600);

const MOST_FRAMES: f64 = 150.0;

/// A scrub faster than evaluation presents a frame about each feedback interval, so there are
/// several uncached scrubs, for enough frames that their p95 is not just the slowest.
const SCRUB_FRAMES: f64 = 48.0;
const JUMPS: usize = 20;
const UNCACHED_SCRUBS: usize = 10;

/// The longest a frame asked for in a scrub may wait for the frame in progress to stop, which
/// the decoder does only between the chunks it decodes: about twice the slowest p95 seen on the
/// workstation. Fixed, so that slower stopping fails rather than loosening the scrubs' gate.
const STOP_BOUND: Duration = Duration::from_millis(20);

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

pub struct Run {
    pub pattern: PathBuf,
    pub chain: Chain,
    pub budget: Option<u64>,
    pub cold: bool,
}

impl Run {
    /// `None` if the arguments after the pattern make no sense.
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

struct Check {
    run: Run,
    name: String,
    gpu: bool,
    refresh: Duration,
    viewed: Option<NodeId>,
    /// The frames the scan found, which a small budget narrows the timeline from and each edit
    /// restores.
    whole: (f64, f64),
    edits: u32,
    wait: Option<Wait>,
    /// The viewer's latest frame, so that a draw presenting none is not counted again.
    last_frame: u64,
    recording: Option<Recording>,
    /// The frames the scrub or jump in progress asked for, with when, and how many it has yet to.
    asked: Vec<(f64, Instant)>,
    to_ask: usize,
    jumps: Vec<Duration>,
    cached_scrub: Option<Scrub>,
    uncached_scrubs: Vec<Scrub>,
    failures: Vec<String>,
}

struct Wait {
    what: &'static str,
    until: fn() -> bool,
    then: fn(),
    since: Instant,
}

/// What the shell did during a playback, scrub or jump.
struct Recording {
    /// Each frame of the sequence at its first presentation; the first is what was on screen
    /// when it began.
    presented: Vec<(f64, Instant)>,
    frames: Vec<FrameTiming>,
    /// The most frames the timebar said real-time playback dropped.
    drops: u64,
    advances: Vec<Duration>,
    /// When the worker took each frame it delivered.
    started: Vec<(f64, Instant)>,
}

impl Recording {
    fn new(on_screen: Option<f64>) -> Recording {
        let now = Instant::now();
        Recording {
            presented: on_screen.map(|frame| (frame, now)).into_iter().collect(),
            frames: Vec::new(),
            drops: 0,
            advances: Vec::new(),
            started: Vec::new(),
        }
    }
}

/// A scrub's frames presented before the last, each from being asked for; their waits for the
/// frame in progress to stop; and the last's own time.
struct Scrub {
    asked: usize,
    /// How often it asked for a frame, which a GUI thread held up does less than each refresh.
    pace: Duration,
    latencies: Vec<Duration>,
    waits: Vec<Duration>,
    last: Option<Duration>,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

fn with<R>(use_check: impl FnOnce(&mut Check) -> R) -> R {
    CHECK.with(|cell| use_check(cell.borrow_mut().as_mut().expect("the check has started")))
}

fn running() -> bool {
    CHECK.with(|cell| cell.borrow().is_some())
}

fn say(line: &str) {
    with(|check| eprintln!("{}: {line}", check.name));
}

pub fn start(run: Run) {
    let device = crate::evaluation_device();
    let name = format!(
        "playback check ({}, {device}{})",
        run.chain.name(),
        if run.cold { ", cold" } else { "" }
    );
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            run,
            name,
            gpu: device == EvaluationDevice::Gpu,
            refresh: Duration::ZERO,
            viewed: None,
            whole: (0.0, 0.0),
            edits: 0,
            wait: None,
            last_frame: 0,
            recording: None,
            asked: Vec::new(),
            to_ask: 0,
            jumps: Vec::new(),
            cached_scrub: None,
            uncached_scrubs: Vec::new(),
            failures: Vec::new(),
        })
    });
    ffi::single_shot(1000, setup, 0);
}

pub fn observe(seen: &Seen) {
    if !running() {
        return;
    }
    let presented = match seen {
        Seen::Frame(Pane::Viewer, frame) => Some(crate::read_app(|app| {
            let drops = app.timebar().map_or(0, |timebar| timebar.drops);
            (app.displayed(), drops, frame)
        })),
        _ => None,
    };
    with(|check| {
        let Some(recording) = check.recording.as_mut() else {
            return;
        };
        match seen {
            Seen::Advanced(took) => recording.advances.push(*took),
            Seen::Delivered { frame, timing, .. } => {
                recording.started.push((*frame, timing.started));
            }
            _ => {}
        }
        let Some((displayed, drops, frame)) = presented else {
            return;
        };
        if frame.number <= check.last_frame {
            return;
        }
        check.last_frame = frame.number;
        recording.drops = recording.drops.max(drops);
        if let Some(displayed) = displayed.filter(|displayed| !displayed.failed)
            && (recording.presented.last()).is_none_or(|(shown, _)| *shown != displayed.frame)
        {
            recording.presented.push((displayed.frame, frame.presented));
        }
        recording.frames.push(frame.clone());
    });
}

/// Waits until `until`, then does `then`, failing the check if that takes too long.
fn wait_for(what: &'static str, until: fn() -> bool, then: fn()) {
    with(|check| {
        check.wait = Some(Wait {
            what,
            until,
            then,
            since: Instant::now(),
        })
    });
}

fn poll(_: usize) {
    if !running() {
        return;
    }
    if let Some(wait) = with(|check| check.wait.take()) {
        if (wait.until)() {
            (wait.then)();
        } else if wait.since.elapsed() >= LONGEST_WAIT {
            let displayed = crate::read_app(|app| format!("{:?}", app.displayed()));
            with(|check| {
                (check.failures).push(format!(
                    "{} took over {LONGEST_WAIT:?}: {displayed}",
                    wait.what
                ))
            });
            return report();
        } else {
            with(|check| check.wait = Some(wait));
        }
    }
    if running() {
        ffi::single_shot(POLL.as_millis() as i32, poll, 0);
    }
}

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
    say(&format!(
        "frames from {}, a refresh {refresh:.2?}",
        pattern.display()
    ));
    if let Some(budget) = budget {
        crate::set_viewer_cache_budget(budget);
    }
    let viewed = crate::edit(|session| {
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
    wait_for("the scans", || scanned().is_some(), set_range);
    poll(0);
}

fn created(changes: Changes) -> NodeId {
    *changes.nodes.iter().next().expect("one node was created")
}

/// The frames the scan found, at most `MOST_FRAMES`; `None` until every pattern's scan is in.
fn scanned() -> Option<(f64, f64)> {
    let (pattern, second) = with(|check| {
        let second = match &check.run.chain {
            Chain::Merge(second) => Some(second.clone()),
            _ => None,
        };
        (check.run.pattern.clone(), second)
    });
    crate::read_app(|app| {
        let files = app.session().files();
        if second.is_some_and(|second| files.scan(&second).is_none()) {
            return None;
        }
        let (first, last) = files.scan(&pattern)?.range()?;
        let (first, last) = (first as f64, last as f64);
        Some((first, last.min(first + MOST_FRAMES - 1.0)))
    })
}

fn set_range() {
    let (first, last) = scanned().expect("the scans are in");
    with(|check| check.whole = (first, last));
    if range() != (first, last) {
        crate::dispatch(Event::FrameRangeSet { first, last });
        crate::dispatch(Event::Command(Command::FirstFrame));
    }
    wait_for("the first frame", shows_first, play_every_frame);
}

fn play_every_frame() {
    evict_if_cold();
    play(PlaybackMode::EveryFrame);
    // The last frame's copy lands after it is shown, and would otherwise drop a frame from the
    // cache once real time has been kept to those it holds.
    wait_for(
        "every frame from storage",
        || crate::read_app(|app| app.playing().is_none() && shown_is_cached(app)),
        after_every_frame,
    );
}

fn after_every_frame() {
    describe_playback("every frame from storage", false);
    keep_to_cached();
    crate::dispatch(Event::Command(Command::FirstFrame));
    wait_for("the first frame again", shows_first, play_real_time);
}

fn play_real_time() {
    play(PlaybackMode::RealTime);
    wait_for(
        "real time from the viewer cache",
        || crate::read_app(|app| app.playing().is_none()),
        after_real_time,
    );
}

fn after_real_time() {
    describe_playback("real time from the viewer cache", true);
    scrub();
    wait_for(
        "the scrub over cached frames",
        asked_presented,
        after_cached_scrub,
    );
}

fn after_cached_scrub() {
    let scrub = finish_scrub();
    say(&scrub_line("scrub over cached frames", &[&scrub]));
    with(|check| check.cached_scrub = Some(scrub));
    edit_viewed(start_jumps);
}

fn start_jumps() {
    evict_if_cold();
    jump();
}

/// Asks for the next jump's frame: spread over the range's second half, which the uncached
/// scrubs do not reach unless the range is short.
fn jump() {
    let (first, last) = range();
    let count = last - first + 1.0;
    let frame = with(|check| {
        let index = check.jumps.len() as f64;
        let frame = (first + (count / 2.0).floor() + index * (count / 2.0 / JUMPS as f64))
            .floor()
            .min(last);
        check.recording = Some(Recording::new(None));
        check.asked = vec![(frame, Instant::now())];
        frame
    });
    crate::dispatch(Event::FrameSet(frame));
    wait_for("a jump to an uncached frame", asked_presented, after_jump);
}

fn after_jump() {
    let jumps = with(|check| {
        let (frame, asked) = check.asked[0];
        let recording = check.recording.as_ref().expect("the jump records");
        let (_, at) = (recording.presented.iter())
            .find(|(shown, _)| *shown == frame)
            .expect("the frame was presented");
        check.jumps.push(*at - asked);
        check.jumps.len()
    });
    if jumps < JUMPS {
        return jump();
    }
    let jumps: Durations = with(|check| check.jumps.iter().copied().collect());
    say(&format!(
        "an uncached frame, jumped to: {jumps}, of {}",
        jumps.len()
    ));
    edit_viewed(uncached_scrub);
}

fn uncached_scrub() {
    evict_if_cold();
    scrub();
    wait_for(
        "a scrub over uncached frames",
        asked_presented,
        after_uncached_scrub,
    );
}

fn after_uncached_scrub() {
    let scrub = finish_scrub();
    let scrubs = with(|check| {
        check.uncached_scrubs.push(scrub);
        check.uncached_scrubs.len()
    });
    if scrubs < UNCACHED_SCRUBS {
        return edit_viewed(uncached_scrub);
    }
    let line = with(|check| {
        let scrubs: Vec<&Scrub> = check.uncached_scrubs.iter().collect();
        scrub_line("scrubs over uncached frames", &scrubs)
    });
    say(&line);
    report();
}

fn range() -> (f64, f64) {
    let timeline = crate::read_app(|app| app.session().document().timeline());
    (timeline.first(), timeline.last())
}

/// Whether the viewer cache holds the frame shown, as the timebar marks it.
fn shown_is_cached(app: &App) -> bool {
    let (Some(shown), Some(timebar)) = (app.displayed(), app.timebar()) else {
        return false;
    };
    let index = (shown.frame - timebar.first) as usize;
    (timebar.marks.get(index)).is_some_and(|mark| mark.cached)
}

fn shows_first() -> bool {
    shows(range().0)
}

/// Whether the viewer shows the frame, evaluated or cached.
fn shows(frame: f64) -> bool {
    crate::read_app(|app| {
        app.session().frame() == frame
            && (app.displayed()).is_some_and(|shown| shown.frame == frame)
    })
}

/// Records afresh, the frame on screen already the first presented.
fn record() {
    let on_screen = crate::read_app(|app| app.displayed()).map(|displayed| displayed.frame);
    with(|check| check.recording = Some(Recording::new(on_screen)));
}

fn play(mode: PlaybackMode) {
    record();
    crate::dispatch(Event::LoopingSet(false));
    crate::dispatch(Event::PlaybackModeSet(mode));
    crate::dispatch(Event::Command(Command::Play(Direction::Forward)));
}

/// Describes the playback just finished; real time fails for any frame it missed, and for the
/// GUI's frames over their budget in a GPU session.
fn describe_playback(name: &str, real_time: bool) {
    let (cached, budget, bytes) = crate::read_app(|app| {
        let cache = app.viewer_cache();
        let cached = (app.timebar()).map_or(0, |timebar| {
            timebar.marks.iter().filter(|mark| mark.cached).count()
        });
        (cached, cache.budget(), cache.bytes())
    });
    let (first, last) = range();
    let count = (last - first) as usize + 1;
    let memory = memory();
    let mut lines = Vec::new();
    with(|check| {
        let recording = check.recording.take().expect("playback records");
        let presented = &recording.presented;
        let mut distinct: Vec<i64> = presented.iter().map(|(frame, _)| *frame as i64).collect();
        distinct.sort_unstable();
        distinct.dedup();
        let distinct = distinct.len();
        let span = match (presented.first(), presented.last()) {
            (Some((_, first)), Some((_, last))) => *last - *first,
            _ => Duration::ZERO,
        };
        let rate = (presented.len().saturating_sub(1)) as f64 / span.as_secs_f64().max(1e-9);
        let gaps: Durations = presented
            .windows(2)
            .map(|pair| pair[1].1 - pair[0].1)
            .collect();
        let advances: Durations = recording.advances.iter().copied().collect();
        lines.push(format!(
            "{name}: {distinct} of {count} frames presented over {span:.2?}, {rate:.1} fps; \
             between frames {gaps}; {} dropped by the clock; each advance on the GUI thread \
             p95 {:.2?}, max {:.2?}; afterwards {cached} frames cached, {} MiB of {} MiB; \
             {memory}",
            recording.drops,
            advances.p(0.95),
            advances.max(),
            bytes >> 20,
            budget >> 20,
        ));
        if bytes > budget {
            (check.failures).push(format!(
                "the viewer cache holds {bytes} bytes, over {budget}"
            ));
        }
        if !real_time {
            return;
        }
        if distinct < count || recording.drops > 0 {
            check.failures.push(format!(
                "real time presented {distinct} of {count} frames, and its clock dropped {}",
                recording.drops
            ));
        }
        let gui = gui_frames(&recording.frames, check.gpu, &mut check.failures);
        lines.push(format!("{name}: the viewer's {gui}"));
    });
    for line in lines {
        say(&line);
    }
}

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
    let (from, length) = longest;
    if length == 0 {
        return with(|check| (check.failures).push("the viewer cache holds no frame".to_owned()));
    }
    let first = range().0 + from as f64;
    let last = first + (length - 1) as f64;
    say(&format!(
        "the viewer cache holds {} of {} frames; real time and the cached scrub keep to \
         {first}–{last}",
        marks.iter().filter(|&&cached| cached).count(),
        marks.len()
    ));
    crate::dispatch(Event::FrameRangeSet { first, last });
}

/// Begins a scrub over the range's first frames, a frame each refresh.
fn scrub() {
    let (first, last) = range();
    record();
    with(|check| {
        check.asked.clear();
        check.to_ask = SCRUB_FRAMES.min(last - first + 1.0) as usize;
    });
    scrub_step(0);
}

fn scrub_step(index: usize) {
    let frame = range().0 + index as f64;
    let (more, refresh) = with(|check| {
        check.asked.push((frame, Instant::now()));
        check.to_ask -= 1;
        (check.to_ask > 0, check.refresh)
    });
    crate::dispatch(Event::FrameSet(frame));
    if more {
        let period = ((refresh.as_micros() + 500) / 1000).max(1) as i32;
        ffi::single_shot(period, scrub_step, index + 1);
    }
}

/// Whether the scrub or jump has asked for every frame, and the last is presented.
fn asked_presented() -> bool {
    let last = with(|check| {
        let recording = check.recording.as_ref()?;
        let (last, _) = *check.asked.last().filter(|_| check.to_ask == 0)?;
        recording
            .presented
            .iter()
            .any(|(frame, _)| *frame == last)
            .then_some(last)
    });
    last.is_some_and(shows)
}

/// The last frame is measured apart from the others, since it may wait for a frame the worker
/// lets finish, and has its own gate.
fn finish_scrub() -> Scrub {
    with(|check| {
        let recording = check.recording.take().expect("the scrub records");
        let asked: HashMap<i64, Instant> = (check.asked.iter())
            .map(|(frame, at)| (*frame as i64, *at))
            .collect();
        let (last_frame, last_asked) = *check.asked.last().expect("the scrub asked");
        let during =
            || (recording.presented.iter().skip(1)).filter(|(frame, _)| *frame != last_frame);
        let latencies = during()
            .filter_map(|(frame, at)| Some(*at - *asked.get(&(*frame as i64))?))
            .collect();
        let waits = during()
            .filter_map(|(frame, _)| {
                let (_, started) = recording.started.iter().find(|(f, _)| f == frame)?;
                Some(started.saturating_duration_since(*asked.get(&(*frame as i64))?))
            })
            .collect();
        let last = (recording.presented.iter().rev())
            .find(|(frame, _)| *frame == last_frame)
            .map(|(_, at)| *at - last_asked);
        let pace = match (check.asked.first(), check.asked.last()) {
            (Some((_, first)), Some((_, last))) => {
                (*last - *first) / (check.asked.len().max(2) - 1) as u32
            }
            _ => Duration::ZERO,
        };
        Scrub {
            asked: check.asked.len(),
            pace,
            latencies,
            waits,
            last,
        }
    })
}

fn scrub_line(name: &str, scrubs: &[&Scrub]) -> String {
    let latencies: Durations = scrubs
        .iter()
        .flat_map(|s| s.latencies.iter().copied())
        .collect();
    let waits: Durations = scrubs
        .iter()
        .flat_map(|s| s.waits.iter().copied())
        .collect();
    let asked: usize = scrubs.iter().map(|scrub| scrub.asked).sum();
    let pace = scrubs
        .iter()
        .map(|scrub| scrub.pace)
        .max()
        .unwrap_or_default();
    let last = scrubs.iter().filter_map(|scrub| scrub.last).max();
    let waits = match waits.is_empty() {
        true => String::new(),
        false => format!(
            "; waiting for the frame in progress to stop p50 {:.1?}, p95 {:.1?}",
            waits.p(0.5),
            waits.p(0.95),
        ),
    };
    let lasts = match scrubs.len() {
        1 => "the last".to_string(),
        n => format!("of {n} scrubs, the slowest last"),
    };
    format!(
        "{name}: {} of {asked} frames asked for presented before the last, one asked every \
         {pace:.1?} at most; each from being asked for {latencies}{waits}; {lasts} {:.1?}; {}",
        latencies.len(),
        last.unwrap_or_default(),
        memory(),
    )
}

/// Edits the viewed node so that no frame is cached, then `then` once it is shown. It is shown
/// at the whole range's last frame, which neither the jumps nor the uncached scrubs reach unless
/// the range is short.
fn edit_viewed(then: fn()) {
    let (viewed, parameter, edited, (first, last)) = with(|check| {
        check.recording = None;
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
    wait_for(
        "the edit's frame",
        || {
            crate::read_app(|app| app.presented_job())
                .is_some_and(|shown| shown >= crate::latest_job())
        },
        then,
    );
}

fn evict_if_cold() {
    let (cold, patterns) = with(|check| {
        let mut patterns = vec![check.run.pattern.clone()];
        if let Chain::Merge(second) = &check.run.chain {
            patterns.push(second.clone());
        }
        (check.run.cold, patterns)
    });
    if !cold {
        return;
    }
    let (first, last) = range();
    let (first, last) = (first as i64, last as i64);
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

/// Judges the scrubs, which need the uncached frame time the jumps measured, and reports.
fn report() {
    let check = CHECK.with(|cell| cell.borrow_mut().take());
    let mut check = check.expect("the check has started");
    let jumps: Durations = check.jumps.iter().copied().collect();
    let uncached = jumps.p(0.95);
    // A scrub's last frame may wait for the frame in progress, which the worker stops only
    // within the feedback interval of starting it.
    let frame_in_progress = FEEDBACK_INTERVAL.max(uncached);
    let last_frame_gate = uncached + frame_in_progress;
    let cached: Vec<&Scrub> = check.cached_scrub.iter().collect();
    let uncached_scrubs: Vec<&Scrub> = check.uncached_scrubs.iter().collect();
    let failures = &mut check.failures;
    judge_scrubs(
        "cached scrub",
        &cached,
        check.refresh * 2,
        last_frame_gate,
        failures,
    );
    let uncached_gate = uncached + check.refresh + STOP_BOUND;
    judge_scrubs(
        "uncached scrubs",
        &uncached_scrubs,
        uncached_gate,
        last_frame_gate,
        failures,
    );
    let waits: Durations = (uncached_scrubs.iter())
        .flat_map(|scrub| scrub.waits.iter().copied())
        .collect();
    if waits.p(0.95) > STOP_BOUND {
        failures.push(format!(
            "frames of the uncached scrubs waited p95 {:.1?} for the frame in progress to stop, \
             over {STOP_BOUND:?}",
            waits.p(0.95)
        ));
    }
    finish(&check.name, &check.failures);
}

/// Fails scrubs whose frames before the last were presented p95 over `gate` from being asked
/// for, or whose last took over `last_gate`.
fn judge_scrubs(
    name: &str,
    scrubs: &[&Scrub],
    gate: Duration,
    last_gate: Duration,
    failures: &mut Vec<String>,
) {
    if scrubs.is_empty() {
        return;
    }
    let latencies: Durations = (scrubs.iter())
        .flat_map(|scrub| scrub.latencies.iter().copied())
        .collect();
    let p95 = latencies.p(0.95);
    if p95 > gate {
        failures.push(format!(
            "the {name} presented frames p95 {p95:.1?} from being asked for, over {gate:.1?}"
        ));
    }
    for scrub in scrubs {
        match scrub.last {
            Some(last) if last <= last_gate => {}
            last => failures.push(format!(
                "a last frame of the {name} took {last:.1?}, over the frame in progress and the \
                 uncached frame time, {last_gate:.1?}"
            )),
        }
    }
}
