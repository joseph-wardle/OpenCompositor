//! Evaluating the viewed image off the GUI thread, on the session's device, and readying its
//! planes for the viewer: staged for the GPU in a CPU session, evaluated on it in a GPU session.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::document::NodeId;
use crate::evaluate::GpuImage;
use crate::evaluate::{
    DEEP_STACK, EvalError, Evaluator, GpuEvaluation, Pool, ResultKey, poll_device,
};
use crate::image::{Image, ImageInfo, Plane, Window};
use crate::viewer::Zoom;
use crate::viewer_cache::{GpuCopy, ViewerCopy};

/// The staging buffer's first size, a DCI 4K image of four planes, and its least.
const FIRST_STAGING_SIZE: (u32, u32) = (4096, 2160);

/// Images in a row that fit in half the staging buffer before it shrinks to fit them.
const IMAGES_BEFORE_SHRINKING: u32 = 3;

/// How often the worker asks whether the GPU has completed the work it waits for: a GPU
/// session's submission, or a CPU session's copy out of the staging buffer. It never blocks on the
/// GPU: a blocking wait would hold the device's lock against the GUI thread's present for as long.
const COMPLETION_POLL: Duration = Duration::from_millis(1);

/// How long the viewer may go without a new result before a newer request for another frame of
/// the node no longer stops the one in progress: a scrub faster than evaluation still shows a
/// result at least this often, give or take one evaluation.
pub const FEEDBACK_INTERVAL: Duration = Duration::from_millis(100);

/// Where a session evaluates. Every request runs on one device, and results never mix devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationDevice {
    /// The CPU, whose images are staged for the viewer.
    Cpu,
    /// The GPU the panes draw on.
    Gpu,
}

impl fmt::Display for EvaluationDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EvaluationDevice::Cpu => "CPU",
            EvaluationDevice::Gpu => "GPU",
        })
    }
}

/// What the worker evaluates: the image of the node the viewer's slot marks at a frame, as that
/// node's key stood when asked, either for the viewer or to prepare ahead. The app numbers its
/// jobs in the order it makes them.
#[derive(Clone, Debug)]
pub(crate) struct Job {
    pub(crate) node: NodeId,
    pub(crate) key: ResultKey,
    /// Not the key's: the worker evaluates the key, and the frame says what its image is of.
    pub(crate) frame: f64,
    pub(crate) number: u64,
    /// How large the viewer draws images, for the copy of the result the viewer cache is given;
    /// `None` for no copy.
    pub(crate) copy: Option<Zoom>,
}

/// A finished evaluation of what the worker was asked for, which may have been superseded since.
#[derive(Debug)]
pub struct Evaluated {
    pub(crate) job: Job,
    pub(crate) result: Result<ViewerImage, ViewerError>,
    pub timing: Timing,
}

impl Evaluated {
    /// The number of the job this answers.
    pub fn job(&self) -> u64 {
        self.job.number
    }

    /// The frame of the job this answers.
    pub fn frame(&self) -> f64 {
        self.job.frame
    }

    /// The image's data window, or `None` for a failure.
    pub fn data_window(&self) -> Option<Window> {
        Some(self.result.as_ref().ok()?.info.data_window)
    }
}

/// When a result passed each stage of the worker, for the shell's drag check.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// The worker took the request.
    pub started: Instant,
    /// In a GPU session, the files of the Readers it uploads were read.
    pub read: Option<Instant>,
    /// The evaluator returned the image or its failure; in a GPU session, once its request was
    /// submitted.
    pub evaluated: Instant,
    pub staging: StagingTiming,
    /// The result left for the GUI thread.
    pub delivered: Instant,
}

/// When a result passed each stage of staging; all `None` with no pixels to stage.
#[derive(Clone, Copy, Debug, Default)]
pub struct StagingTiming {
    /// The staging buffer was taken, mapped for writing.
    pub mapped: Option<Instant>,
    /// The planes were written into it.
    pub written: Option<Instant>,
    /// It was unmapped and the textures made.
    pub staged: Option<Instant>,
}

#[cfg(test)]
impl Timing {
    /// Every stage passed now, for results tests make without a worker.
    pub(crate) fn now() -> Timing {
        let now = Instant::now();
        Timing {
            started: now,
            read: None,
            evaluated: now,
            staging: StagingTiming::default(),
            delivered: now,
        }
    }
}

/// An image as the viewer receives it: its description, and its planes on the GPU.
#[derive(Debug)]
pub struct ViewerImage {
    pub info: ImageInfo,
    /// `None` before the GPU exists, as headless, or if the data window is empty.
    pub(crate) pixels: Option<Pixels>,
}

/// An image's working planes on the device of one generation, for the viewer to draw and read
/// from. Holding them keeps the worker from writing into them again.
#[derive(Debug)]
pub(crate) struct Pixels {
    /// R32Float textures over the data window, bottom row first, in `Channel::ALL`'s order,
    /// which a GPU session's may exceed. Channels that share values share a texture.
    pub(crate) planes: [Arc<wgpu::Texture>; 4],
    pub(crate) device_generation: u64,
    /// A CPU session's planes, still to be copied from the worker's staging buffer; a GPU
    /// session's request wrote them in its own submission, which queue order completes before
    /// any frame the GUI thread submits after receiving it.
    pub(crate) staged: Option<Staged>,
}

/// Where a CPU session's planes are in the worker's staging buffer.
///
/// The GUI thread copies them into their textures with its own queue. Dropping this returns the
/// buffer to the worker, which waits for the copy before writing to it again.
#[derive(Debug)]
pub(crate) struct Staged {
    /// For each distinct plane, its first channel's index in `Pixels::planes` and where its
    /// rows start.
    pub(crate) copies: Vec<(usize, u64)>,
    pub(crate) bytes_per_row: u32,
    staging: Staging,
}

impl Staged {
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.staging.buffer.as_ref().expect("held until dropped")
    }
}

/// The worker's staging buffer, returned unmapped to its connection's slot when dropped, by
/// whichever thread holds it, so that neither a skipped result nor a panic loses it. Without a
/// buffer, as after a failure, it leaves the slot empty for the worker to fill. A buffer for a
/// device no longer connected drops here instead, so that it can never occupy the slot of the
/// device that replaced it, and so does one returned once the session is on the GPU, which
/// stages nothing.
#[derive(Debug)]
struct Staging {
    buffer: Option<wgpu::Buffer>,
    device_generation: u64,
    mailbox: Arc<Mailbox>,
}

impl Drop for Staging {
    fn drop(&mut self) {
        let buffer = self.buffer.take();
        let mut state = self.mailbox.lock();
        let on_cpu = state.device == EvaluationDevice::Cpu;
        let Some(connection) = &mut state.connection else {
            return;
        };
        if connection.generation != self.device_generation {
            return;
        }
        connection.staging = match buffer {
            Some(buffer) if on_cpu => StagingSlot::Waiting(buffer),
            _ => StagingSlot::Empty,
        };
        self.mailbox.changed.notify_all();
    }
}

/// Why the viewer has no image for a key.
#[derive(Debug)]
pub enum ViewerError {
    Eval(EvalError),
    /// Evaluated, but the GPU cannot hold it.
    CannotShow(String),
}

impl fmt::Display for ViewerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViewerError::Eval(error) => error.fmt(f),
            ViewerError::CannotShow(reason) => write!(f, "cannot show it: {reason}"),
        }
    }
}

impl std::error::Error for ViewerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ViewerError::Eval(error) => Some(error),
            ViewerError::CannotShow(_) => None,
        }
    }
}

/// One worker evaluating one key at a time with its own [`Evaluator`], on the session's device. A
/// request replaces any request it has not started, so after a burst of requests it evaluates at
/// most the one in progress and the last.
///
/// Once it has delivered a result, unless a newer request is waiting, it makes the result's copy
/// for the viewer cache if asked, on the device that evaluated it, and passes it on.
///
/// A newer request for another node, or for nothing, stops the evaluation in progress at its
/// next stopping point: between nodes, between chunks of an operation long enough to ask or of an
/// EXR file's decoding, before staging or submitting, and while waiting for the staging buffer;
/// and skips its delivery after. A read blocked on the file or work submitted to the GPU is not
/// interrupted. So does a newer request for another frame of the same node, so that a scrub's
/// final frame comes first, unless the viewer has gone the [`FEEDBACK_INTERVAL`] without a
/// result. A newer request for the same frame, an edit, lets the evaluation in progress finish
/// and be delivered: an edit's work is mostly done when the next arrives, and stopping it showed
/// fewer edits, later.
///
/// With no request waiting, it prepares the frames the app expects to want next, in order: it
/// evaluates each and makes its copy for the viewer cache, delivering nothing. Look-ahead yields
/// to every request but one for the frame it is preparing, whose copy the viewer then shows,
/// and to a pause, which only stops requests. It prepares each key once, until a device change.
///
/// The worker never writes to a texture with `Queue::write_texture`, which copies under a
/// queue-wide lock that the GUI thread's submit then waits for. In a CPU session it never
/// submits; in a GPU session it submits each request once, and submits the next only once the GPU
/// has completed it, so that superseded work on the GPU is at most one request.
#[derive(Debug)]
pub struct Worker {
    mailbox: Arc<Mailbox>,
}

#[derive(Debug)]
struct Mailbox {
    state: Mutex<MailboxState>,
    changed: Condvar,
}

/// The device, the staging buffer and evaluation's state on the device belong here rather than
/// to the worker, which may still be running when the process exits: the last reference to a GPU
/// object must not drop on its thread after the driver has begun tearing itself down. The worker
/// holds them only in a GPU phase, which the worker's drop waits for.
#[derive(Debug)]
struct MailboxState {
    /// A request the worker has not started.
    request: Option<Ask>,
    /// The request the worker started last.
    started: Option<Job>,
    /// What to prepare with no request waiting, in order: the frames the app expects next.
    ahead: Vec<Job>,
    /// The keys of `ahead` already prepared, or that failed.
    prepared: Vec<ResultKey>,
    /// The key the worker is preparing, if it is.
    preparing: Option<ResultKey>,
    /// When the viewer last caught up: the worker's last delivery, or a request reaching it
    /// idle. Within the feedback interval of this, a newer request for another frame stops the
    /// started one.
    caught_up: Instant,
    /// Whether the started request, or the key being prepared, was found superseded. It stays so
    /// although the feedback interval may pass meanwhile: work stopped in part has been dropped,
    /// as by decoder threads that stopped while others went on.
    found_superseded: bool,
    shutting_down: bool,
    /// The device the worker evaluates or stages for, from `connect` until the worker drops.
    connection: Option<Connection>,
    /// Whether the worker is in a GPU phase, and so may hold GPU objects.
    in_gpu_phase: bool,
    /// Where requests are evaluated from the next one started.
    device: EvaluationDevice,
    /// Evaluation's last submission in a GPU session, for telling whether a device was lost
    /// while it had work on the GPU.
    last_submission: Option<Submission>,
}

/// What the app asks of the worker.
#[derive(Debug)]
enum Ask {
    Show(Job),
    /// Stop, keeping what is retained, as while the viewer shows a cached copy.
    Pause,
    /// Release everything retained, as when the viewer wants nothing.
    Release,
}

/// What the worker does next.
enum Work {
    Evaluate(Job, Purpose),
    Release,
}

/// Why the worker evaluates a job.
#[derive(Clone, Copy)]
enum Purpose {
    /// The viewer asked for it: deliver it.
    Show,
    /// Look-ahead, which the worker takes itself from what the app expects next: only make its
    /// copy for the viewer cache.
    Prepare,
}

#[derive(Debug)]
struct Submission {
    generation: u64,
    at: Instant,
    /// When the worker saw the GPU complete it.
    completed: Option<Instant>,
}

#[derive(Debug)]
struct Connection {
    /// Tells this device's images apart from those of the device it replaced.
    generation: u64,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Set by the device's lost callback.
    lost: Arc<AtomicBool>,
    /// A CPU session's.
    staging: StagingSlot,
    /// A GPU session's, which the worker makes in its first GPU phase on the device, off the GUI
    /// thread, and holds during each. Dropping the connection drops the generation's GPU state.
    evaluation: Option<GpuEvaluation>,
}

/// Where a connection's staging buffer is.
#[derive(Debug)]
enum StagingSlot {
    /// None yet, or the last staging failed: the worker makes one when it next stages.
    Empty,
    /// Waiting for the worker, unmapped.
    Waiting(wgpu::Buffer),
    /// With the worker, or with an image staged in it.
    Out,
}

impl Worker {
    /// Starts the worker, evaluating on `device`. It passes each result to `deliver`, and each
    /// copy for the viewer cache to `copied`, on its own thread.
    pub fn start(
        device: EvaluationDevice,
        deliver: impl Fn(Evaluated) + Send + 'static,
        copied: impl Fn(ViewerCopy) + Send + 'static,
    ) -> Worker {
        let mailbox = Arc::new(Mailbox {
            state: Mutex::new(MailboxState {
                request: None,
                started: None,
                ahead: Vec::new(),
                prepared: Vec::new(),
                preparing: None,
                caught_up: Instant::now(),
                found_superseded: false,
                shutting_down: false,
                connection: None,
                in_gpu_phase: false,
                device,
                last_submission: None,
            }),
            changed: Condvar::new(),
        });
        let worker = Arc::clone(&mailbox);
        std::thread::Builder::new()
            .name("evaluator".to_owned())
            .stack_size(DEEP_STACK)
            .spawn(move || {
                let mut evaluator = Evaluator::new();
                let mut sizing = StagingSizing::default();
                let sinks = Sinks { deliver, copied };
                while let Some((work, device)) = worker.next() {
                    match (work, device) {
                        (Work::Evaluate(job, purpose), EvaluationDevice::Cpu) => {
                            evaluate(&worker, &mut evaluator, job, purpose, &sinks, &mut sizing);
                        }
                        (Work::Evaluate(job, purpose), EvaluationDevice::Gpu) => {
                            evaluate_on_gpu(&worker, &mut evaluator, job, purpose, &sinks);
                        }
                        (Work::Release, EvaluationDevice::Cpu) => evaluator.release_all(),
                        (Work::Release, EvaluationDevice::Gpu) => {
                            evaluator.release_all();
                            release_on_gpu(&worker);
                        }
                    }
                }
            })
            .expect("the evaluator thread starts");
        Worker { mailbox }
    }

    /// Asks for a node's image, or with `None` for nothing, releasing everything retained.
    pub(crate) fn request(&self, job: Option<Job>) {
        let ask = match job {
            Some(job) => Ask::Show(job),
            None => Ask::Release,
        };
        self.mailbox.lock().request = Some(ask);
        self.mailbox.changed.notify_all();
    }

    /// Stops the request in progress as a request for another node would, keeping what it
    /// retained, and waits for the next.
    pub(crate) fn pause(&self) {
        self.mailbox.lock().request = Some(Ask::Pause);
        self.mailbox.changed.notify_all();
    }

    /// Has the worker prepare these frames' copies for the viewer cache, in order, whenever no
    /// request waits: the frames playback shows next, each with the zoom to copy it at. Nothing
    /// stops preparing.
    pub(crate) fn look_ahead(&self, ahead: Vec<Job>) {
        let mut state = self.mailbox.lock();
        state
            .prepared
            .retain(|key| ahead.iter().any(|job| job.key == *key));
        state.ahead = ahead;
        drop(state);
        self.mailbox.changed.notify_all();
    }

    /// Lets the worker evaluate or stage pixels on `device`, replacing any device connected
    /// before, and delivers the last request again, under its number, now with its pixels. Until
    /// then a CPU session evaluates images without them, and a GPU session evaluates nothing. The
    /// worker makes its staging buffer, or its state on the device, when it first needs them,
    /// off the caller's thread. `lost` is for the device's lost callback to set.
    pub(crate) fn connect(
        &self,
        device: wgpu::Device,
        queue: wgpu::Queue,
        lost: Arc<AtomicBool>,
        device_generation: u64,
    ) {
        self.replace_connection(Some(Connection {
            generation: device_generation,
            device,
            queue,
            lost,
            staging: StagingSlot::Empty,
            evaluation: None,
        }));
    }

    /// Stops using the connected device, which was lost at `lost_at`, and delivers the last
    /// request again, without pixels in a CPU session, since a result finished for that device
    /// is dropped. Returns whether evaluation had work on that device's GPU when it was lost.
    pub(crate) fn disconnect(&self, lost_at: Instant) -> bool {
        let state = self.mailbox.lock();
        let lost = (state.connection.as_ref()).map(|connection| connection.generation);
        let had_work = (state.last_submission.as_ref()).is_some_and(|submission| {
            Some(submission.generation) == lost
                && submission.at <= lost_at
                && submission
                    .completed
                    .is_none_or(|completed| completed >= lost_at)
        });
        drop(state);
        self.replace_connection(None);
        had_work
    }

    /// Evaluates on `device` from the next request, and delivers the last request again from
    /// there. Leaving the GPU drops evaluation's state on it; leaving the CPU drops the staging
    /// buffer, unless the viewer holds it.
    pub(crate) fn evaluate_on(&self, device: EvaluationDevice) {
        let mut state = self.mailbox.lock();
        if state.device == device {
            return;
        }
        state.device = device;
        // The app drops the copies made on the device left.
        state.prepared.clear();
        let left = state.connection.as_mut().map(|connection| {
            let evaluation = connection.evaluation.take();
            let staging = match connection.staging {
                StagingSlot::Waiting(_) => {
                    std::mem::replace(&mut connection.staging, StagingSlot::Empty)
                }
                StagingSlot::Empty | StagingSlot::Out => StagingSlot::Empty,
            };
            (evaluation, staging)
        });
        if state.request.is_none()
            && let Some(started) = &state.started
        {
            state.request = Some(Ask::Show(started.clone()));
        }
        drop(state);
        self.mailbox.changed.notify_all();
        drop(left);
    }

    fn replace_connection(&self, connection: Option<Connection>) {
        let mut state = self.mailbox.lock();
        let replaced = std::mem::replace(&mut state.connection, connection);
        // A GPU session prepares nothing without a device.
        state.prepared.clear();
        if state.request.is_none()
            && let Some(started) = &state.started
        {
            state.request = Some(Ask::Show(started.clone()));
        }
        drop(state);
        self.mailbox.changed.notify_all();
        drop(replaced);
    }
}

#[cfg(test)]
impl Worker {
    /// Records a submission on the generation that the GPU has not completed, as a request in
    /// flight would.
    pub(crate) fn submitting_on(&self, generation: u64) {
        self.mailbox.lock().last_submission = Some(Submission {
            generation,
            at: Instant::now(),
            completed: None,
        });
    }
}

impl Drop for Worker {
    /// Stops the worker, waiting while it is in a GPU phase but not while it evaluates on the
    /// CPU, and drops the device and everything the worker made on it on this thread. Nothing is
    /// delivered after this returns.
    // The worker's thread can outlive its handle: a wgpu object it dropped after the GUI began
    // tearing down the device, or after the driver's exit handlers ran, would crash the quit.
    fn drop(&mut self) {
        let mut state = self.mailbox.lock();
        state.shutting_down = true;
        self.mailbox.changed.notify_all();
        while state.in_gpu_phase {
            state = self
                .mailbox
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        let connection = state.connection.take();
        drop(state);
        drop(connection);
    }
}

impl Mailbox {
    fn lock(&self) -> std::sync::MutexGuard<'_, MailboxState> {
        // Nothing panics while holding the lock, so a poisoned state is still whole.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Waits for a request, or with none for a frame to prepare, and where to evaluate it, or
    /// `None` once the worker is shutting down. A pause only stops the started request.
    fn next(&self) -> Option<(Work, EvaluationDevice)> {
        // Preparing counts as idle: a request stops it.
        let mut idle = self.lock().preparing.take().is_some();
        let next = self.wait_for(|state| {
            state.found_superseded = false;
            if let Some(ask) = state.request.take() {
                if idle {
                    state.caught_up = Instant::now();
                }
                idle = false;
                state.started = None;
                match ask {
                    Ask::Show(job) => {
                        state.started = Some(job.clone());
                        return Take::Taken((Work::Evaluate(job, Purpose::Show), state.device));
                    }
                    Ask::Release => return Take::Taken((Work::Release, state.device)),
                    Ask::Pause => {}
                }
            }
            // A GPU session prepares nothing without a device: connecting one asks again.
            let can_prepare = state.device == EvaluationDevice::Cpu || state.connection.is_some();
            let unprepared = (state.ahead.iter())
                .find(|job| can_prepare && !state.prepared.contains(&job.key))
                .cloned();
            match unprepared {
                Some(job) => {
                    state.preparing = Some(job.key.clone());
                    Take::Taken((Work::Evaluate(job, Purpose::Prepare), state.device))
                }
                None => {
                    idle = true;
                    Take::Wait
                }
            }
        });
        // For waits on the request being started, which only tests make.
        self.changed.notify_all();
        next
    }

    /// Waits for the device's staging buffer to come back and takes it, or takes its empty slot
    /// to fill. `None` once the started request is superseded,
    /// the device is no longer connected or the worker is shutting down.
    fn take_staging(self: &Arc<Mailbox>, device_generation: u64) -> Option<Staging> {
        let buffer = self.wait_for(|state| {
            let superseded = state.superseded();
            let connection = match &mut state.connection {
                Some(connection) if connection.generation == device_generation && !superseded => {
                    connection
                }
                _ => return Take::GiveUp,
            };
            match std::mem::replace(&mut connection.staging, StagingSlot::Out) {
                StagingSlot::Empty => Take::Taken(None),
                StagingSlot::Waiting(buffer) => Take::Taken(Some(buffer)),
                StagingSlot::Out => Take::Wait,
            }
        })?;
        Some(Staging {
            buffer,
            device_generation,
            mailbox: Arc::clone(self),
        })
    }

    /// Calls `take` each time the state changes until it takes something or gives up. `None`
    /// once it gives up or the worker is shutting down.
    fn wait_for<T>(&self, mut take: impl FnMut(&mut MailboxState) -> Take<T>) -> Option<T> {
        let mut state = self.lock();
        loop {
            if state.shutting_down {
                return None;
            }
            match take(&mut state) {
                Take::Taken(taken) => return Some(taken),
                Take::GiveUp => return None,
                Take::Wait => {}
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn superseded(&self) -> bool {
        self.lock().superseded()
    }

    fn has_newer_request(&self) -> bool {
        self.lock().request.is_some()
    }

    /// Records that the key being prepared is done, copied or failed, so that it is not prepared
    /// again.
    fn prepared(&self, key: ResultKey) {
        self.lock().prepared.push(key);
    }

    /// Begins a GPU phase, in which the worker may hold GPU objects, unless the started request
    /// is superseded.
    fn start_gpu_phase(self: &Arc<Mailbox>) -> Option<GpuPhase<'_>> {
        let mut state = self.lock();
        if state.superseded() {
            return None;
        }
        state.in_gpu_phase = true;
        let device = (state.connection.as_ref()).map(|connection| PhaseDevice {
            device: connection.device.clone(),
            queue: connection.queue.clone(),
            lost: Arc::clone(&connection.lost),
            generation: connection.generation,
        });
        Some(GpuPhase {
            mailbox: self,
            device,
        })
    }

    /// The keys of the results retained on the connected device, for planning a request before
    /// its GPU phase.
    fn retained_on_gpu(&self) -> Vec<ResultKey> {
        let state = self.lock();
        (state.connection.as_ref())
            .and_then(|connection| connection.evaluation.as_ref())
            .map_or_else(Vec::new, GpuEvaluation::retained_keys)
    }

    /// Waits until the state changes or for at most `timeout`.
    fn wait_timeout(&self, timeout: Duration) {
        let state = self.lock();
        if !state.shutting_down {
            drop(self.changed.wait_timeout(state, timeout));
        }
    }
}

/// What [`Mailbox::wait_for`]'s `take` found in the state.
enum Take<T> {
    Taken(T),
    /// Not yet: wait for the state to change.
    Wait,
    GiveUp,
}

impl MailboxState {
    /// Whether the started request should stop: the worker is shutting down, or a newer request is
    /// waiting for another node, for nothing, to pause, or for another frame within the feedback
    /// interval of the viewer catching up. A frame being prepared stops for any request but one
    /// for its key or a pause. Once true, it stays true until the next request starts.
    fn superseded(&mut self) -> bool {
        self.found_superseded =
            self.found_superseded || self.shutting_down || self.newer_supersedes();
        self.found_superseded
    }

    fn newer_supersedes(&self) -> bool {
        let Some(newer) = &self.request else {
            return false;
        };
        if let Some(preparing) = &self.preparing {
            return match newer {
                Ask::Show(newer) => newer.key != *preparing,
                Ask::Pause => false,
                Ask::Release => true,
            };
        }
        match (newer, &self.started) {
            (Ask::Show(newer), Some(started)) if newer.node == started.node => {
                newer.frame != started.frame && self.caught_up.elapsed() < FEEDBACK_INTERVAL
            }
            _ => true,
        }
    }
}

/// The part of a request in which the worker may hold GPU objects: staging and delivering its
/// result in a CPU session; recording, submitting and delivering it, then waiting for the GPU to
/// complete it, in a GPU session. Never decoding or the CPU's kernels. The worker's drop waits for
/// it to end.
struct GpuPhase<'a> {
    mailbox: &'a Arc<Mailbox>,
    /// The device connected when it began.
    device: Option<PhaseDevice>,
}

struct PhaseDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    lost: Arc<AtomicBool>,
    generation: u64,
}

impl GpuPhase<'_> {
    /// Whether another device, or none, has been connected since this began.
    fn device_replaced(&self) -> bool {
        let state = self.mailbox.lock();
        let current = state
            .connection
            .as_ref()
            .map(|connection| connection.generation);
        self.device
            .as_ref()
            .is_some_and(|began| current != Some(began.generation))
    }

    /// Calls `use_evaluation` with evaluation's state on the phase's device, taken out of the
    /// connection, or made if the connection has none yet, then returns it there, or drops it
    /// if another device has been connected since or evaluation has left the GPU. `None` without
    /// a device, or once replaced.
    fn with_evaluation<R>(
        &self,
        use_evaluation: impl FnOnce(&mut GpuEvaluation, &PhaseDevice) -> R,
    ) -> Option<R> {
        let device = self.device.as_ref()?;
        let taken = {
            let mut state = self.mailbox.lock();
            match &mut state.connection {
                Some(connection) if connection.generation == device.generation => {
                    connection.evaluation.take()
                }
                _ => return None,
            }
        };
        let mut evaluation = taken.unwrap_or_else(|| {
            GpuEvaluation::new(&device.device, &device.queue, Arc::clone(&device.lost))
        });
        let result = use_evaluation(&mut evaluation, device);
        let mut state = self.mailbox.lock();
        let on_gpu = state.device == EvaluationDevice::Gpu;
        match &mut state.connection {
            Some(connection) if connection.generation == device.generation && on_gpu => {
                connection.evaluation = Some(evaluation);
            }
            _ => {
                drop(state);
                drop(evaluation);
            }
        }
        Some(result)
    }

    /// Whether the worker is shutting down or another device, or none, has been connected since this
    /// began, so that waiting on this one's GPU is pointless.
    fn abandoned(&self) -> bool {
        let shutting_down = self.mailbox.lock().shutting_down;
        shutting_down || self.device_replaced()
    }
}

impl Drop for GpuPhase<'_> {
    fn drop(&mut self) {
        drop(self.device.take());
        self.mailbox.lock().in_gpu_phase = false;
        self.mailbox.changed.notify_all();
    }
}

/// Where the worker passes what it made, on its own thread.
struct Sinks<D, C> {
    deliver: D,
    copied: C,
}

/// Evaluates the job's key on the worker's thread. To show it, stages and delivers its image,
/// unless it is superseded first, then makes its copy unless a newer request waits. To prepare
/// it, makes only its copy, which is still the frame's although a request may have come since.
fn evaluate(
    mailbox: &Arc<Mailbox>,
    evaluator: &mut Evaluator,
    job: Job,
    purpose: Purpose,
    sinks: &Sinks<impl Fn(Evaluated), impl Fn(ViewerCopy)>,
    sizing: &mut StagingSizing,
) {
    let started = Instant::now();
    let evaluated = restart_on_panic(evaluator, |evaluator| {
        evaluator.evaluate(&job.key, || mailbox.superseded())
    });
    let Some(result) = evaluated.unwrap_or_else(|| Some(Err(panicked(&job.key)))) else {
        return;
    };
    let evaluated_at = Instant::now();
    let key = job.key.clone();
    let result = result.map(Image::into_parts);
    // What the copy is made from, once the result is delivered.
    let to_copy = result.as_ref().ok().and_then(|(info, planes)| {
        let halvings = job.copy?.halvings(info.display_window)?;
        Some((halvings, info.clone(), planes.clone()))
    });
    let copy_now = match purpose {
        Purpose::Show => {
            let Some(phase) = mailbox.start_gpu_phase() else {
                return;
            };
            let mut staging = StagingTiming::default();
            let result = match result {
                Ok((info, planes)) => {
                    let pool = evaluator.pool();
                    match stage(&phase, &info, planes, pool, sizing, &mut staging) {
                        Ok(pixels) => Ok(ViewerImage { info, pixels }),
                        Err(reason) => Err(ViewerError::CannotShow(reason)),
                    }
                }
                Err(error) => Err(ViewerError::Eval(error)),
            };
            let timing = Timing {
                started,
                read: None,
                evaluated: evaluated_at,
                staging,
                delivered: evaluated_at,
            };
            let delivered = deliver_unless_superseded(&phase, job, result, timing, &sinks.deliver);
            // The CPU's kernels run outside a GPU phase.
            drop(phase);
            delivered && !mailbox.has_newer_request()
        }
        // A read a request cut short fails too, and is prepared again later.
        Purpose::Prepare if result.is_err() && mailbox.superseded() => return,
        // The app learns of a failure when the frame is requested.
        Purpose::Prepare => {
            mailbox.prepared(key.clone());
            true
        }
    };
    if copy_now && let Some((halvings, info, planes)) = to_copy {
        let planes = planes.each_ref().map(|plane| &plane[..]);
        let pool = evaluator.pool();
        (sinks.copied)(ViewerCopy::of_planes(key, info, planes, halvings, pool));
    }
}

/// Calls into the evaluator. An operation's panic is already its node's failure, so one that
/// reaches here is the evaluator's own, and its bookkeeping cannot be trusted: it starts again
/// with nothing retained, rather than the worker ending and every later evaluation with it, and
/// this returns `None`. The panic hook has already reported it.
fn restart_on_panic<T>(
    evaluator: &mut Evaluator,
    call: impl FnOnce(&mut Evaluator) -> T,
) -> Option<T> {
    let called = std::panic::catch_unwind(AssertUnwindSafe(|| call(evaluator)));
    if called.is_err() {
        *evaluator = Evaluator::new();
    }
    called.ok()
}

/// The failure of a key whose evaluation the evaluator's own panic ended.
fn panicked(key: &ResultKey) -> EvalError {
    EvalError::Panicked {
        origin: key.clone(),
    }
}

/// Passes on a finished evaluation: delivers it to show, unless it is superseded, or records
/// its key as prepared.
fn finish(
    phase: &GpuPhase,
    purpose: Purpose,
    job: Job,
    result: Result<ViewerImage, ViewerError>,
    timing: Timing,
    deliver: &impl Fn(Evaluated),
) {
    match purpose {
        Purpose::Show => {
            deliver_unless_superseded(phase, job, result, timing, deliver);
        }
        Purpose::Prepare => phase.mailbox.prepared(job.key),
    }
}

/// Delivers a result, unless it is superseded or was finished for a replaced device, which drops
/// it, even a failure, which may be that device's: connecting the new one asks for the request
/// again. Its GPU objects, if any, go before the phase ends. Returns whether it delivered it.
fn deliver_unless_superseded(
    phase: &GpuPhase,
    job: Job,
    result: Result<ViewerImage, ViewerError>,
    mut timing: Timing,
    deliver: &impl Fn(Evaluated),
) -> bool {
    if phase.mailbox.superseded() || phase.device_replaced() {
        return false;
    }
    timing.delivered = Instant::now();
    deliver(Evaluated {
        job,
        result,
        timing,
    });
    phase.mailbox.lock().caught_up = Instant::now();
    true
}

/// Evaluates the job's key in a GPU session: reads the files of the Readers it uploads on the
/// worker's thread, then in a GPU phase records the key's request, submits it as one, finishes
/// the result, and waits without blocking for the GPU to complete it, unless it is superseded
/// first, then makes its copy there. Without a device it evaluates nothing: connecting one asks
/// again.
fn evaluate_on_gpu(
    mailbox: &Arc<Mailbox>,
    evaluator: &mut Evaluator,
    job: Job,
    purpose: Purpose,
    sinks: &Sinks<impl Fn(Evaluated), impl Fn(ViewerCopy)>,
) {
    let started = Instant::now();
    let on_gpu = mailbox.retained_on_gpu();
    let read = restart_on_panic(evaluator, |evaluator| {
        evaluator.read_for_gpu(&job.key, &on_gpu, || mailbox.superseded())
    });
    drop(on_gpu);
    let Some(read) = read.unwrap_or_else(|| Some(Err(panicked(&job.key)))) else {
        return;
    };
    let read_at = Instant::now();
    // A read a request cut short fails too, and is tried again.
    let Some(phase) = mailbox.start_gpu_phase() else {
        return;
    };
    let mut timing = Timing {
        started,
        read: Some(read_at),
        evaluated: read_at,
        staging: StagingTiming::default(),
        delivered: read_at,
    };
    if let Err(error) = read {
        let result = Err(ViewerError::Eval(error));
        finish(&phase, purpose, job, result, timing, &sinks.deliver);
        return;
    }
    phase.with_evaluation(|evaluation, device| {
        let stop = || mailbox.superseded() || phase.abandoned();
        let key = job.key.clone();
        let Some(result) =
            record_and_submit(mailbox, evaluator, evaluation, device, &key, job.copy, stop)
        else {
            return;
        };
        timing.evaluated = Instant::now();
        let (result, to_copy) = match result {
            Ok((image, copy)) => {
                let to_copy = copy.map(|copy| (copy, image.info().clone()));
                (Ok(image), to_copy)
            }
            Err(error) => (Err(error), None),
        };
        let result = result.map(|image| ViewerImage {
            info: image.info().clone(),
            pixels: image.textures().map(|planes| Pixels {
                planes,
                device_generation: device.generation,
                staged: None,
            }),
        });
        finish(&phase, purpose, job, result, timing, &sinks.deliver);
        wait_for_completion(&phase, evaluation);
        // Kept although a newer request waits: the GPU has made it, and reading it back is
        // little beside evaluating the frame again for the cache.
        if let Some((copy, info)) = to_copy
            && let Some(copy) = read_copy(&phase, evaluation, device, copy, key, info)
        {
            (sinks.copied)(copy);
        }
    });
}

/// Waits without blocking for the GPU to complete evaluation's work, or for the phase to be
/// abandoned. The next request is recorded only once this one is complete, so that superseded
/// work on the GPU is at most one request, and so is its memory. A lost device's work never
/// completes; its lost callback has the app replace it.
fn wait_for_completion(phase: &GpuPhase, evaluation: &GpuEvaluation) {
    while evaluation.idle() == Ok(false) && !phase.abandoned() {
        phase.mailbox.wait_timeout(COMPLETION_POLL);
    }
    if evaluation.idle() == Ok(true)
        && let Some(submission) = &mut phase.mailbox.lock().last_submission
    {
        submission.completed.get_or_insert_with(Instant::now);
    }
}

/// An image's copy for the viewer cache, submitted with its request.
struct SubmittedCopy {
    halvings: u32,
    /// None for an empty data window, whose copy has no texels.
    copy: Option<GpuCopy>,
}

/// Reads back a copy for the viewer cache once the GPU has completed its request. `None` if the
/// GPU did not complete it.
fn read_copy(
    phase: &GpuPhase,
    evaluation: &GpuEvaluation,
    device: &PhaseDevice,
    submitted: SubmittedCopy,
    key: ResultKey,
    info: ImageInfo,
) -> Option<ViewerCopy> {
    let Some(copy) = submitted.copy else {
        return Some(ViewerCopy::empty(key, info, submitted.halvings));
    };
    if evaluation.idle() != Ok(true) {
        return None;
    }
    map(phase, &device.device, &copy.readback, wgpu::MapMode::Read).ok()?;
    copy.read(key, info).ok()
}

/// Records the key's request on the device and submits it, unless `stop` returns true first,
/// which records or submits nothing more. With a zoom, the submission also makes the image's
/// copy for the viewer cache at it, if the GPU can.
fn record_and_submit(
    mailbox: &Mailbox,
    evaluator: &mut Evaluator,
    evaluation: &mut GpuEvaluation,
    device: &PhaseDevice,
    key: &ResultKey,
    copy: Option<Zoom>,
    stop: impl Fn() -> bool,
) -> Option<Result<(GpuImage, Option<SubmittedCopy>), ViewerError>> {
    let recorded = restart_on_panic(evaluator, |evaluator| {
        evaluator.record_on(evaluation, key, &stop)
    });
    let mut recorded = match recorded {
        Some(None) => return None,
        Some(Some(Ok(recorded))) => recorded,
        Some(Some(Err(error))) => return Some(Err(ViewerError::Eval(error))),
        // The panic may have left the GPU table half released: it starts again too.
        None => {
            evaluation.release_all();
            return Some(Err(ViewerError::Eval(panicked(key))));
        }
    };
    if stop() {
        return None;
    }
    mailbox.lock().last_submission = Some(Submission {
        generation: device.generation,
        at: Instant::now(),
        completed: None,
    });
    let halvings = copy.and_then(|zoom| zoom.halvings(recorded.image().info().display_window));
    let copy = halvings.and_then(|halvings| {
        let copy = evaluation
            .record_copy_for_viewer(&mut recorded, halvings)
            .ok()?;
        Some(SubmittedCopy { halvings, copy })
    });
    let image = evaluation.submit(recorded).map_err(ViewerError::Eval);
    Some(image.map(|image| (image, copy)))
}

/// Releases everything retained on the connected device, as when the viewer wants nothing.
fn release_on_gpu(mailbox: &Arc<Mailbox>) {
    let Some(phase) = mailbox.start_gpu_phase() else {
        return;
    };
    phase.with_evaluation(|evaluation, _| evaluation.release_all());
}

/// Copies the image's distinct planes into the staging buffer and makes their textures, for the
/// GUI thread to copy them into. `Ok(None)` without a GPU or pixels, or if the request was
/// superseded or the worker stopped while waiting for the buffer, which the check for a
/// superseded result after this catches.
fn stage(
    phase: &GpuPhase,
    info: &ImageInfo,
    planes: [Plane; 4],
    pool: &Pool,
    sizing: &mut StagingSizing,
    timing: &mut StagingTiming,
) -> Result<Option<Pixels>, String> {
    let Some(PhaseDevice {
        device,
        generation: device_generation,
        ..
    }) = &phase.device
    else {
        return Ok(None);
    };
    let window = info.data_window;
    if is_empty(info) {
        return Ok(None);
    }
    // Each distinct plane once, by the first channel that has it, so that channels sharing a
    // plane share its texture.
    let mut firsts: Vec<usize> = Vec::new();
    let mut distinct_of = [0; 4];
    for (channel, plane) in planes.iter().enumerate() {
        distinct_of[channel] =
            match (firsts.iter()).position(|&first| Plane::ptr_eq(&planes[first], plane)) {
                Some(index) => index,
                None => {
                    firsts.push(channel);
                    firsts.len() - 1
                }
            };
    }
    let limits = device.limits();
    let size = staging_size(window.width, window.height, firsts.len());
    let largest = window.width.max(window.height);
    if largest > limits.max_texture_dimension_2d {
        return Err(format!(
            "its data window is {} × {} pixels, and this GPU's images are at most {} pixels on a \
             side",
            window.width, window.height, limits.max_texture_dimension_2d
        ));
    }
    if size > limits.max_buffer_size {
        return Err(format!(
            "its planes take {size} bytes, and this GPU's buffers take at most {}",
            limits.max_buffer_size
        ));
    }
    let Some(mut staging) = phase.mailbox.take_staging(*device_generation) else {
        return Ok(None);
    };
    // Until the buffer is back in `staging`, a failure drops it, leaving the slot empty.
    let buffer = match staging.buffer.take() {
        Some(buffer) => match sizing.replacement(buffer.size(), size, limits.max_buffer_size) {
            None => {
                map(phase, device, &buffer, wgpu::MapMode::Write).map_err(|error| {
                    format!("the GPU could not map its staging buffer: {error}")
                })?;
                buffer
            }
            Some(replacement) => {
                // Destroyed first, so that the two buffers hold memory together only until the
                // GPU finishes any copy out of the old one.
                buffer.destroy();
                staging_buffer(device, replacement)?
            }
        },
        None => staging_buffer(device, sizing.first(size, limits.max_buffer_size))?,
    };
    timing.mapped = Some(Instant::now());
    let bytes_per_row = row_bytes(window.width);
    {
        let mut view = buffer
            .get_mapped_range_mut(..size)
            .map_err(|error| format!("the GPU could not write its staging buffer: {error}"))?;
        let mut bytes = view.slice(..);
        let floats = bytes.len() / size_of::<f32>();
        let start = bytes.as_raw_element_ptr().cast::<f32>();
        debug_assert!(start.as_ptr().is_aligned());
        // SAFETY: the mapping is `size` bytes, initialised, and only this view accesses it until
        // the view drops. wgpu does not promise the host pointer's alignment, but its allocators
        // align mapped memory well beyond f32's, as the assertion checks. Mapped memory may be
        // write-combining, so the copy only writes to it.
        let out = unsafe { std::slice::from_raw_parts_mut(start.as_ptr(), floats) };
        let distinct: Vec<&[f32]> = firsts.iter().map(|&first| &planes[first][..]).collect();
        copy_rows(
            pool,
            &distinct,
            window.width as usize,
            out,
            bytes_per_row as usize / size_of::<f32>(),
        );
    }
    timing.written = Some(Instant::now());
    drop(planes);
    buffer.unmap();
    staging.buffer = Some(buffer);
    let textures = (firsts.iter())
        .map(|_| {
            allocate(device, "the image", || {
                Arc::new(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("staged plane"),
                    size: wgpu::Extent3d {
                        width: window.width,
                        height: window.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    // Copied from by the viewer's readout.
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                }))
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    timing.staged = Some(Instant::now());
    let plane_bytes = u64::from(bytes_per_row) * u64::from(window.height);
    Ok(Some(Pixels {
        planes: distinct_of.map(|index| Arc::clone(&textures[index])),
        device_generation: *device_generation,
        staged: Some(Staged {
            copies: (firsts.iter().zip(0..))
                .map(|(&first, index)| (first, index * plane_bytes))
                .collect(),
            bytes_per_row,
            staging,
        }),
    }))
}

/// When the worker replaces its staging buffer. It grows to fit any larger image with a
/// quarter to spare, and shrinks only after several images in a row would fit in half of it, so
/// that one large image does not hold its memory for the rest of the session, and alternating
/// large and small images does not make a buffer for each. No buffer is larger than `largest`,
/// the GPU's limit, which the image to stage is within.
#[derive(Debug, Default)]
struct StagingSizing {
    images_fitting_half: u32,
}

impl StagingSizing {
    /// The size of buffer to make, where there is none, to stage `needed` bytes.
    fn first(&mut self, needed: u64, largest: u64) -> u64 {
        self.images_fitting_half = 0;
        fitted(needed, largest)
    }

    /// The size of buffer to replace a `current`-byte one with, to stage `needed` bytes, or
    /// `None` to keep it.
    fn replacement(&mut self, current: u64, needed: u64, largest: u64) -> Option<u64> {
        let fitted = fitted(needed, largest);
        if current / 2 > fitted {
            self.images_fitting_half += 1;
        } else {
            self.images_fitting_half = 0;
        }
        if current < needed {
            self.images_fitting_half = 0;
            // Dragging Blur's Size up grows the data window a little each edit; growing to fit
            // exactly would replace the buffer, and wait on the GPU for it, every time.
            Some((fitted + fitted / 4).min(largest))
        } else if self.images_fitting_half == IMAGES_BEFORE_SHRINKING {
            self.images_fitting_half = 0;
            Some(fitted)
        } else {
            None
        }
    }
}

/// The size of buffer that stages `needed` bytes, and at least a first-size image where the GPU
/// allows one.
fn fitted(needed: u64, largest: u64) -> u64 {
    let (first_width, first_height) = FIRST_STAGING_SIZE;
    needed
        .max(staging_size(first_width, first_height, 4))
        .min(largest)
}

pub(crate) fn is_empty(info: &ImageInfo) -> bool {
    info.data_window.width == 0 || info.data_window.height == 0
}

/// Bytes per staged row of a plane, padded as buffer-to-texture copies require.
fn row_bytes(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

fn staging_size(width: u32, height: u32, planes: usize) -> u64 {
    u64::from(row_bytes(width)) * u64::from(height) * planes as u64
}

/// A staging buffer, mapped for writing.
fn staging_buffer(device: &wgpu::Device, size: u64) -> Result<wgpu::Buffer, String> {
    allocate(device, "its staging buffer", || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image staging"),
            size,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        })
    })
}

/// Makes a buffer or texture, failing if the GPU has no memory for it. wgpu reports that for
/// these without losing the device, so it fails one image rather than the session.
pub(crate) fn allocate<T>(
    device: &wgpu::Device,
    what: &str,
    make: impl FnOnce() -> T,
) -> Result<T, String> {
    // Error scopes belong to the thread that pushes them, so the GUI thread's errors stay out.
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let made = make();
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(format!("the GPU has no memory for {what}: {error}")),
        None => Ok(made),
    }
}

/// Maps the buffer once the GPU has finished with it, unless the phase is abandoned first.
fn map(
    phase: &GpuPhase,
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    mode: wgpu::MapMode,
) -> Result<(), String> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(mode, .., move |result| {
        let _ = sender.send(result);
    });
    loop {
        poll_device(device).map_err(|lost| lost.to_string())?;
        if let Ok(result) = receiver.try_recv() {
            return result.map_err(|error| error.to_string());
        }
        if phase.abandoned() {
            return Err("the worker stopped waiting for the GPU".to_owned());
        }
        phase.mailbox.wait_timeout(COMPLETION_POLL);
    }
}

/// Copies planes of rows `width` pixels long into `out`, one after another, their rows
/// `floats_per_row` apart, in the planes' row order.
fn copy_rows(pool: &Pool, planes: &[&[f32]], width: usize, out: &mut [f32], floats_per_row: usize) {
    let height = planes.first().map_or(0, |plane| plane.len() / width);
    pool.fill_rows([out], floats_per_row, |_, first_row, [rows]| {
        for (row, target) in (first_row..).zip(rows.chunks_mut(floats_per_row)) {
            let start = row % height * width;
            target[..width].copy_from_slice(&planes[row / height][start..start + width]);
        }
    });
}

#[cfg(test)]
mod tests;
