//! Evaluating the viewed image off the GUI thread, on the session's device, and readying its
//! planes for the viewer: staged for the GPU in a CPU session, evaluated on it in a GPU session.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use crate::document::NodeId;
use crate::evaluate::{DEEP_STACK, EvalError, Evaluator, GpuEvaluation, Pool, ResultKey};
use crate::gpu::wait_for_device;
use crate::image::{ImageInfo, Plane, Window};

/// The staging buffer's first size, a DCI 4K image of four planes, and its least.
const FIRST_STAGING_SIZE: (u32, u32) = (4096, 2160);

/// Images in a row that fit in half the staging buffer before it shrinks to fit them.
const IMAGES_BEFORE_SHRINKING: u32 = 3;

/// The longest the worker waits on the GPU before checking that it still wants the device, so
/// that a wait that never ends cannot hold the worker's drop, and with it the GUI thread.
const LONGEST_GPU_WAIT: Duration = Duration::from_millis(100);

/// How often a GPU session's worker asks whether the GPU has completed its submission. It never
/// blocks on the GPU: a blocking wait would hold the device's lock against the GUI thread's
/// present for as long.
const COMPLETION_POLL: Duration = Duration::from_millis(1);

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

/// What the viewer asks the worker for: the image of the node its slot marks, as that node's key
/// stood when asked. The app numbers its requests in the order it sends them.
#[derive(Clone, Debug)]
pub(crate) struct Wanted {
    pub(crate) node: NodeId,
    pub(crate) key: ResultKey,
    pub(crate) number: u64,
}

/// A finished evaluation of what the worker was asked for, which may have been superseded since.
#[derive(Debug)]
pub struct Evaluated {
    pub(crate) wanted: Wanted,
    pub(crate) result: Result<ViewerImage, ViewerError>,
    pub timing: Timing,
}

impl Evaluated {
    /// The number of the request this answers.
    pub fn request(&self) -> u64 {
        self.wanted.number
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
    /// R32Float textures over the data window, bottom row first, in `Channel::ALL`'s order.
    /// Channels that share values share a texture.
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
/// device that replaced it.
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
        let Some(connection) = &mut state.connection else {
            return;
        };
        if connection.generation != self.device_generation {
            return;
        }
        connection.staging = match buffer {
            Some(buffer) => StagingSlot::Waiting(buffer),
            None => StagingSlot::Empty,
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
/// A newer request for the same node lets the evaluation in progress finish and be delivered, so
/// that an edit faster than evaluation still shows its results as they finish. A newer request
/// for another node, or for nothing, stops it between nodes, before staging or submitting and
/// while waiting for the staging buffer, and skips its delivery after.
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
    /// A request the worker has not started: what the viewer wants, or `None` for nothing,
    /// which releases everything retained.
    request: Option<Option<Wanted>>,
    /// The request the worker started last.
    started: Option<Wanted>,
    stopping: bool,
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
    /// Starts the worker, evaluating on `device`. It passes each result to `deliver` on its own
    /// thread.
    pub fn start(device: EvaluationDevice, deliver: impl Fn(Evaluated) + Send + 'static) -> Worker {
        let mailbox = Arc::new(Mailbox {
            state: Mutex::new(MailboxState {
                request: None,
                started: None,
                stopping: false,
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
                while let Some((request, device)) = worker.next() {
                    match (request, device) {
                        (Some(wanted), EvaluationDevice::Cpu) => {
                            evaluate(&worker, &mut evaluator, wanted, &deliver, &mut sizing);
                        }
                        (Some(wanted), EvaluationDevice::Gpu) => {
                            evaluate_on_gpu(&worker, &mut evaluator, wanted, &deliver);
                        }
                        (None, EvaluationDevice::Cpu) => evaluator.release_all(),
                        (None, EvaluationDevice::Gpu) => {
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
    pub(crate) fn request(&self, wanted: Option<Wanted>) {
        self.mailbox.lock().request = Some(wanted);
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
        if state.request.is_none() && state.started.is_some() {
            state.request = Some(state.started.clone());
        }
        drop(state);
        self.mailbox.changed.notify_all();
        drop(left);
    }

    fn replace_connection(&self, connection: Option<Connection>) {
        let mut state = self.mailbox.lock();
        let replaced = std::mem::replace(&mut state.connection, connection);
        if state.request.is_none() && state.started.is_some() {
            state.request = Some(state.started.clone());
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
    fn drop(&mut self) {
        let mut state = self.mailbox.lock();
        state.stopping = true;
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

    /// Waits for a request, and where to evaluate it, or `None` once the worker is stopping.
    fn next(&self) -> Option<(Option<Wanted>, EvaluationDevice)> {
        let next = self.wait_for(|state| {
            let Some(request) = state.request.take() else {
                return Take::Wait;
            };
            state.started.clone_from(&request);
            Take::Taken((request, state.device))
        });
        // For waits on the request being started, which only tests make.
        self.changed.notify_all();
        next
    }

    /// Waits for the device's staging buffer to come back and takes it, or takes its empty slot
    /// to fill. `None` once the started request is superseded,
    /// the device is no longer connected or the worker is stopping.
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
    /// once it gives up or the worker is stopping.
    fn wait_for<T>(&self, mut take: impl FnMut(&mut MailboxState) -> Take<T>) -> Option<T> {
        let mut state = self.lock();
        loop {
            if state.stopping {
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
    fn pause(&self, timeout: Duration) {
        let state = self.lock();
        if !state.stopping {
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
    /// Whether a newer request for another node, or for nothing, is waiting, or the worker is
    /// stopping. A newer request for the same node waits for the started one to be delivered.
    fn superseded(&self) -> bool {
        let node = |wanted: &Option<Wanted>| wanted.as_ref().map(|wanted| wanted.node);
        self.stopping
            || (self.request.as_ref()).is_some_and(|newer| node(newer) != node(&self.started))
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

    /// Whether the worker is stopping or another device, or none, has been connected since this
    /// began, so that waiting on this one's GPU is pointless.
    fn abandoned(&self) -> bool {
        let stopping = self.mailbox.lock().stopping;
        stopping || self.device_replaced()
    }
}

impl Drop for GpuPhase<'_> {
    fn drop(&mut self) {
        drop(self.device.take());
        self.mailbox.lock().in_gpu_phase = false;
        self.mailbox.changed.notify_all();
    }
}

/// Evaluates the wanted key on the worker's thread, then stages and delivers its image, unless it
/// is superseded first.
fn evaluate(
    mailbox: &Arc<Mailbox>,
    evaluator: &mut Evaluator,
    wanted: Wanted,
    deliver: &impl Fn(Evaluated),
    sizing: &mut StagingSizing,
) {
    let started = Instant::now();
    let evaluated = std::panic::catch_unwind(AssertUnwindSafe(|| {
        evaluator.evaluate(&wanted.key, || mailbox.superseded())
    }));
    let evaluated_at = Instant::now();
    let result = match evaluated {
        Ok(None) => return,
        Ok(Some(result)) => result,
        // An operation's panic is already its node's failure, so this one is the evaluator's
        // own, and its bookkeeping cannot be trusted: it starts again with nothing retained,
        // rather than the worker ending and every later evaluation with it. The panic hook has
        // already reported it.
        Err(_) => {
            *evaluator = Evaluator::new();
            Err(EvalError::Panicked {
                origin: wanted.key.clone(),
            })
        }
    };
    let Some(phase) = mailbox.start_gpu_phase() else {
        return;
    };
    let mut staging = StagingTiming::default();
    let result = match result {
        Ok(image) => {
            let (info, planes) = image.into_parts();
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
    deliver_unless_superseded(&phase, wanted, result, timing, deliver);
}

/// Delivers a result, unless it is superseded or was finished for a replaced device, which drops
/// it, even a failure, which may be that device's: connecting the new one asks for the request
/// again. Its GPU objects, if any, go before the phase ends.
fn deliver_unless_superseded(
    phase: &GpuPhase,
    wanted: Wanted,
    result: Result<ViewerImage, ViewerError>,
    mut timing: Timing,
    deliver: &impl Fn(Evaluated),
) {
    if phase.mailbox.superseded() || phase.device_replaced() {
        return;
    }
    timing.delivered = Instant::now();
    deliver(Evaluated {
        wanted,
        result,
        timing,
    });
}

/// Evaluates the wanted key in a GPU session: reads the files of the Readers it uploads on the
/// worker's thread, then in a GPU phase records the request, submits it as one, delivers the
/// result, and waits without blocking for the GPU to complete it, unless
/// it is superseded first. Without a device it evaluates nothing: connecting one asks again.
fn evaluate_on_gpu(
    mailbox: &Arc<Mailbox>,
    evaluator: &mut Evaluator,
    wanted: Wanted,
    deliver: &impl Fn(Evaluated),
) {
    let started = Instant::now();
    let on_gpu = mailbox.retained_on_gpu();
    let read = std::panic::catch_unwind(AssertUnwindSafe(|| {
        evaluator.read_for_gpu(&wanted.key, &on_gpu, || mailbox.superseded())
    }));
    drop(on_gpu);
    let read_at = Instant::now();
    let failure = match read {
        Ok(None) => return,
        Ok(Some(Ok(()))) => None,
        Ok(Some(Err(error))) => Some(error),
        // As in a CPU session: the evaluator's own panic, after which it starts again.
        Err(_) => {
            *evaluator = Evaluator::new();
            Some(EvalError::Panicked {
                origin: wanted.key.clone(),
            })
        }
    };
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
    if let Some(error) = failure {
        let result = Err(ViewerError::Eval(error));
        deliver_unless_superseded(&phase, wanted, result, timing, deliver);
        return;
    }
    phase.with_evaluation(|evaluation, device| {
        let stop = || mailbox.superseded() || phase.abandoned();
        let Some(result) =
            record_and_submit(mailbox, evaluator, evaluation, device, &wanted.key, stop)
        else {
            return;
        };
        timing.evaluated = Instant::now();
        deliver_unless_superseded(&phase, wanted, result, timing, deliver);
        // The next request is recorded only once this one is complete, so that superseded work
        // on the GPU is at most one request, and so is its memory. A lost device's work never
        // completes; its lost callback has the app replace it.
        while evaluation.idle() == Ok(false) && !phase.abandoned() {
            mailbox.pause(COMPLETION_POLL);
        }
        if evaluation.idle() == Ok(true)
            && let Some(submission) = &mut mailbox.lock().last_submission
        {
            submission.completed.get_or_insert_with(Instant::now);
        }
    });
}

/// Records the key's request on the device and submits it, unless `stop` returns true first,
/// which records or submits nothing more.
fn record_and_submit(
    mailbox: &Mailbox,
    evaluator: &mut Evaluator,
    evaluation: &mut GpuEvaluation,
    device: &PhaseDevice,
    key: &ResultKey,
    stop: impl Fn() -> bool,
) -> Option<Result<ViewerImage, ViewerError>> {
    let recorded = std::panic::catch_unwind(AssertUnwindSafe(|| {
        evaluator.record_on(evaluation, key, &stop)
    }));
    let recorded = match recorded {
        Ok(None) => return None,
        Ok(Some(Ok(recorded))) => recorded,
        Ok(Some(Err(error))) => return Some(Err(ViewerError::Eval(error))),
        // The evaluator's own panic, which may have left the GPU table half released: both
        // start again with nothing retained.
        Err(_) => {
            *evaluator = Evaluator::new();
            evaluation.release_all();
            return Some(Err(ViewerError::Eval(EvalError::Panicked {
                origin: key.clone(),
            })));
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
    let image = match evaluation.submit(recorded) {
        Ok(image) => image,
        Err(error) => return Some(Err(ViewerError::Eval(error))),
    };
    Some(Ok(ViewerImage {
        info: image.info().clone(),
        pixels: image.textures().map(|planes| Pixels {
            planes,
            device_generation: device.generation,
            staged: None,
        }),
    }))
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
                map(phase, device, &buffer).map_err(|error| {
                    format!("the GPU could not map its staging buffer: {error}")
                })?;
                buffer
            }
            Some(replacement) => {
                // Destroyed first, so that the two buffers never hold memory together.
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
        // SAFETY: the mapping is `size` bytes, aligned to `wgpu::MAP_ALIGNMENT`, which is at least
        // f32's alignment, and only this view accesses it until the view drops. Mapped memory may
        // be write-combining, so the copy only writes to it.
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

/// Maps the buffer for writing once the GPU has finished copying out of it, unless the staging is
/// abandoned first.
fn map(phase: &GpuPhase, device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), String> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Write, .., move |result| {
        let _ = sender.send(result);
    });
    loop {
        wait_for_device(device, Some(LONGEST_GPU_WAIT))?;
        if let Ok(result) = receiver.try_recv() {
            return result.map_err(|error| error.to_string());
        }
        if phase.abandoned() {
            return Err("the worker stopped waiting for the GPU".to_owned());
        }
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
