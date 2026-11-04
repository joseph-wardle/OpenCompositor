//! `--playback-check IMAGE_PATTERN [exposure | blur] [BUDGET_MIB]`: plays a sequence on a chain,
//! `exposure`, Reader → Exposure at one stop (the default), or `blur`, Reader → Blur at Size 50,
//! over the frames its scan finds, at most the first 150, at the timeline's 24 frames a second.
//!
//! First every frame from storage, which look-ahead also copies into the viewer cache: the
//! frames presented, and the rate they were presented at, are recorded, not gated. Then in real
//! time from the viewer cache, which must present every frame, with no drop, while the viewer's
//! frames meet the GUI's budget in a GPU session: from acquiring a frame to submitting it p99
//! 1 ms, and at most one over 4 ms. The viewer cache must stay within its budget, 4 GiB unless
//! BUDGET_MIB sets another; with a budget too small for every frame, real time plays what it
//! holds and drops the rest, which the check reports and fails. It reports the process's resident
//! memory, and how long the GUI thread took for each advance of playback.
//!
//! Whether the files were in the page cache, or the NFS server's, is the runner's to say: the
//! check reads them as they come. Run it in a release build, on the workstation, pinned.

use std::cell::RefCell;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use open_ng::Point;
use opencompositor_core::{
    Changes, Command, Direction, EvaluationDevice, Event, FrameTiming, NodeId, Pane, PlaybackMode,
    Slot, Value,
};

use crate::drag::{FRAME_GUI_MAX, FRAME_GUI_P99, percentile, resident_memory};
use crate::qt::ffi;

/// How often the check looks for what it waits for, and for how long at most.
const POLL: Duration = Duration::from_millis(50);
const LONGEST_WAIT: Duration = Duration::from_secs(600);

/// The most frames played, as many as the workloads' sequences have.
const MOST_FRAMES: f64 = 150.0;

/// A chain the check plays, from its Reader to the node viewed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Exposure,
    Blur,
}

impl Chain {
    pub fn from_name(name: &str) -> Option<Chain> {
        match name {
            "exposure" => Some(Chain::Exposure),
            "blur" => Some(Chain::Blur),
            _ => None,
        }
    }
}

/// What the check is waiting for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The scan, then the first frame shown.
    Setup,
    /// Every-frame playback from storage.
    EveryFrame,
    /// The first frame shown again.
    Rewind,
    /// Real-time playback from the viewer cache.
    RealTime,
}

struct Check {
    pattern: PathBuf,
    chain: Chain,
    budget: Option<u64>,
    device: EvaluationDevice,
    stage: Stage,
    waited: Duration,
    first: f64,
    last: f64,
    /// The playback in progress: when it began, and each frame's first presentation.
    began: Option<Instant>,
    presented: Vec<(f64, Instant)>,
    /// The viewer's frames during real-time playback.
    frames: Vec<FrameTiming>,
    last_frame: u64,
    /// The most frames the timebar said real-time playback dropped.
    drops: u64,
    /// How long the GUI thread took for each advance of playback.
    advances: Vec<Duration>,
    lines: Vec<String>,
    failures: Vec<String>,
}

thread_local! {
    static CHECK: RefCell<Option<Check>> = const { RefCell::new(None) };
}

fn with<R>(use_check: impl FnOnce(&mut Check) -> R) -> R {
    CHECK.with(|cell| use_check(cell.borrow_mut().as_mut().expect("the check has started")))
}

pub fn start(pattern: PathBuf, chain: Chain, budget: Option<u64>) {
    CHECK.with(|cell| {
        *cell.borrow_mut() = Some(Check {
            pattern,
            chain,
            budget,
            device: crate::evaluation_device(),
            stage: Stage::Setup,
            waited: Duration::ZERO,
            first: 0.0,
            last: 0.0,
            began: None,
            presented: Vec::new(),
            frames: Vec::new(),
            last_frame: 0,
            drops: 0,
            advances: Vec::new(),
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

/// Makes the chain on a Reader of the pattern and views it.
fn setup(_: usize) {
    let (pattern, chain, budget) = with(|check| (check.pattern.clone(), check.chain, check.budget));
    crate::edit(|session| {
        if let Some(budget) = budget {
            session.set_viewer_cache_budget(budget);
        }
        let reader = created(session.create_readers(vec![pattern], Point::default(), None));
        let (type_name, parameter, value) = match chain {
            Chain::Exposure => ("exposure", "stops", 1.0),
            Chain::Blur => ("blur", "size", 50.0),
        };
        let node = created(
            (session.create_node(type_name, Point::default(), Some(reader)))
                .expect("the type exists and the Reader exists"),
        );
        (session.set_value(node, parameter, Value::Float(value))).expect("the node has it");
        (session.set_mark(Slot::new(1).expect("1 is a slot"), Some(node))).expect("it exists");
    });
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
        Stage::EveryFrame | Stage::RealTime => crate::read_app(|app| app.playing().is_none()),
        Stage::Rewind => shows_first(),
    };
    if done {
        with(|check| check.waited = Duration::ZERO);
        match stage {
            Stage::Setup | Stage::Rewind if !shows_first() => {}
            Stage::Setup => return play(Stage::EveryFrame),
            Stage::EveryFrame => {
                finish_playing();
                with(|check| check.stage = Stage::Rewind);
                crate::dispatch(Event::Command(Command::FirstFrame));
            }
            Stage::Rewind => return play(Stage::RealTime),
            Stage::RealTime => {
                finish_playing();
                return report();
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

/// Sets the timeline to the frames the scan found, at most the first 150, once it has. Returns
/// whether it has.
fn set_range() -> bool {
    let pattern = with(|check| check.pattern.clone());
    let range = crate::read_app(|app| {
        (app.session().files().scan(&pattern))
            .and_then(|scan| scan.range())
            .map(|(first, last)| {
                (
                    first as f64,
                    (last as f64).min(first as f64 + MOST_FRAMES - 1.0),
                )
            })
    });
    let Some((first, last)) = range else {
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
    });
    true
}

/// Whether the viewer shows the timeline's first frame, evaluated or cached.
fn shows_first() -> bool {
    let first = with(|check| check.first);
    crate::read_app(|app| {
        (app.displayed()).is_some_and(|shown| shown.frame == first && shown.requested == first)
    })
}

fn play(stage: Stage) {
    let mode = match stage {
        Stage::RealTime => PlaybackMode::RealTime,
        _ => PlaybackMode::EveryFrame,
    };
    // The frame playback starts on is on screen already.
    let shown = crate::read_app(|app| app.displayed()).map(|displayed| displayed.frame);
    with(|check| {
        let began = Instant::now();
        check.stage = stage;
        check.began = Some(began);
        check.presented = shown.map(|frame| (frame, began)).into_iter().collect();
        check.drops = 0;
        check.advances.clear();
    });
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
             {} MiB; resident {}",
            percentile(&gaps, 0.5),
            percentile(&gaps, 0.95),
            gaps.last().copied().unwrap_or_default(),
            check.drops,
            percentile(&advances, 0.95),
            advances.last().copied().unwrap_or_default(),
            bytes >> 20,
            budget >> 20,
            resident_memory(),
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

/// Prints what playback showed and exits: 1 with any failure, else 0.
fn report() {
    let check = CHECK
        .with(|cell| cell.borrow_mut().take())
        .expect("the check has started");
    let name = format!("playback check ({:?}, {})", check.chain, check.device);
    println!(
        "{name}: {} frames from {}",
        check.last - check.first + 1.0,
        check.pattern.display()
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
