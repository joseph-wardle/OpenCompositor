//! Evaluating the viewed image off the GUI thread, and staging its pixels for the GPU.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Condvar, Mutex, mpsc};

use half::f16;

use crate::document::NodeId;
use crate::evaluate::{DEEP_STACK, EvalError, Evaluator, ResultKey};
use crate::gpu::wait_for_device;
use crate::image::ImageInfo;

/// The staging buffer's first size, a DCI 4K image, and its least.
const FIRST_STAGING_SIZE: (u32, u32) = (4096, 2160);

/// Images in a row that fit in half the staging buffer before it shrinks to fit them.
const IMAGES_BEFORE_SHRINKING: u32 = 3;

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
}

/// An image as the viewer receives it: its description, and its pixels staged for the GPU. Its
/// planes stay with the evaluator.
#[derive(Debug)]
pub struct ViewerImage {
    pub info: ImageInfo,
    /// `None` before the GPU exists, as headless, or if the data window is empty.
    pub(crate) pixels: Option<Staged>,
}

/// An image's pixels, RGBA f16 rows bottom first in the worker's staging buffer, and the texture
/// for them.
///
/// The GUI thread copies the rows into the texture with its own queue. Dropping this returns the
/// buffer to the worker, which waits for the copy before writing to it again.
#[derive(Debug)]
pub(crate) struct Staged {
    pub(crate) texture: wgpu::Texture,
    pub(crate) bytes_per_row: u32,
    staging: Staging,
}

impl Staged {
    /// The generation of the device the image was staged for.
    pub(crate) fn device_generation(&self) -> u64 {
        self.staging.device_generation
    }

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
            Some(buffer) => StagingSlot::Waiting {
                buffer,
                mapped: false,
            },
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

/// One worker evaluating one key at a time with its own [`Evaluator`]. A request replaces any
/// request it has not started, so after a burst of requests it evaluates at most the one in
/// progress and the last.
///
/// A newer request for the same node lets the evaluation in progress finish and be delivered, so
/// that an edit faster than evaluation still shows its results as they finish. A newer request
/// for another node, or for nothing, stops it between nodes, before staging and while waiting for
/// the staging buffer, and skips its delivery after staging.
///
/// The worker never writes to a texture or submits to the queue: `Queue::write_texture` copies
/// under a queue-wide lock that the GUI thread's submit then waits for.
#[derive(Debug)]
pub struct Worker {
    mailbox: Arc<Mailbox>,
}

#[derive(Debug, Default)]
struct Mailbox {
    state: Mutex<MailboxState>,
    changed: Condvar,
}

/// The device and the staging buffer belong here rather than to the worker, which may still be
/// running when the process exits: the last reference to a GPU object must not drop on its
/// thread after the driver has begun tearing itself down.
#[derive(Debug, Default)]
struct MailboxState {
    /// A request the worker has not started: what the viewer wants, or `None` for nothing,
    /// which releases everything retained.
    request: Option<Option<Wanted>>,
    /// The request the worker started last.
    started: Option<Wanted>,
    stopping: bool,
    /// The device the worker stages for, from `connect` until the worker drops.
    connection: Option<Connection>,
    /// Whether the worker is staging or delivering a result, and so may hold GPU objects.
    finishing: bool,
}

#[derive(Debug)]
struct Connection {
    /// Tells this device's staged images apart from those of the device it replaced.
    generation: u64,
    device: wgpu::Device,
    staging: StagingSlot,
}

/// Where a connection's staging buffer is.
#[derive(Debug)]
enum StagingSlot {
    /// None yet, or the last staging failed: the worker makes one when it next stages.
    Empty,
    /// Waiting for the worker, mapped for writing or not.
    Waiting { buffer: wgpu::Buffer, mapped: bool },
    /// With the worker, or with an image staged in it.
    Out,
}

impl Worker {
    /// Starts the worker. It passes each result to `deliver` on its own thread.
    pub fn start(deliver: impl Fn(Evaluated) + Send + 'static) -> Worker {
        let mailbox = Arc::new(Mailbox::default());
        let worker = Arc::clone(&mailbox);
        std::thread::Builder::new()
            .name("evaluator".to_owned())
            .stack_size(DEEP_STACK)
            .spawn(move || {
                let mut evaluator = Evaluator::new();
                let mut sizing = StagingSizing::default();
                while let Some(request) = worker.next() {
                    match request {
                        Some(wanted) => {
                            evaluate(&worker, &mut evaluator, wanted, &deliver, &mut sizing);
                        }
                        None => evaluator.release_all(),
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

    /// Lets the worker stage pixels for `device`, replacing any device connected before, and
    /// delivers the last request again, under its number, now with its pixels. Until then it
    /// evaluates images without them. The worker makes its staging buffer when it first stages,
    /// off the caller's thread.
    pub(crate) fn connect(&self, device: wgpu::Device, device_generation: u64) {
        let mut state = self.mailbox.lock();
        let replaced = state.connection.replace(Connection {
            generation: device_generation,
            device,
            staging: StagingSlot::Empty,
        });
        if state.request.is_none() && state.started.is_some() {
            state.request = Some(state.started.clone());
        }
        drop(state);
        self.mailbox.changed.notify_all();
        drop(replaced);
    }
}

impl Drop for Worker {
    /// Stops the worker, waiting while it stages or delivers a result but not while it evaluates
    /// one, and drops the device and the staging buffer on this thread. Nothing is delivered
    /// after this returns.
    fn drop(&mut self) {
        let mut state = self.mailbox.lock();
        state.stopping = true;
        self.mailbox.changed.notify_all();
        while state.finishing {
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

    /// Waits for a request, or `None` once the worker is stopping.
    fn next(&self) -> Option<Option<Wanted>> {
        self.wait_for(|state| {
            let request = state.request.take()?;
            state.started.clone_from(&request);
            Some(request)
        })
    }

    /// Waits for the device's staging buffer to come back and takes it, with whether it is
    /// mapped, or takes its empty slot to fill. `None` once the started request is superseded,
    /// the device is no longer connected or the worker is stopping.
    fn take_staging(self: &Arc<Mailbox>, device_generation: u64) -> Option<(Staging, bool)> {
        let taken = self.wait_for(|state| {
            let superseded = state.superseded();
            let connection = match &mut state.connection {
                Some(connection) if connection.generation == device_generation && !superseded => {
                    connection
                }
                _ => return Some(None),
            };
            match std::mem::replace(&mut connection.staging, StagingSlot::Out) {
                StagingSlot::Empty => Some(Some((None, false))),
                StagingSlot::Waiting { buffer, mapped } => Some(Some((Some(buffer), mapped))),
                StagingSlot::Out => None,
            }
        });
        let (buffer, mapped) = taken.flatten()?;
        let staging = Staging {
            buffer,
            device_generation,
            mailbox: Arc::clone(self),
        };
        Some((staging, mapped))
    }

    fn wait_for<T>(&self, mut take: impl FnMut(&mut MailboxState) -> Option<T>) -> Option<T> {
        let mut state = self.lock();
        loop {
            if state.stopping {
                return None;
            }
            if let Some(taken) = take(&mut state) {
                return Some(taken);
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

    /// Begins staging and delivering a result, unless it is superseded.
    fn start_finishing(self: &Arc<Mailbox>) -> Option<Finishing<'_>> {
        let mut state = self.lock();
        if state.superseded() {
            return None;
        }
        state.finishing = true;
        let device = (state.connection.as_ref())
            .map(|connection| (connection.device.clone(), connection.generation));
        Some(Finishing {
            mailbox: self,
            device,
        })
    }
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

/// The worker's staging and delivery of a result. The worker's drop waits for it to end.
struct Finishing<'a> {
    mailbox: &'a Arc<Mailbox>,
    /// The device connected when it began, and its generation.
    device: Option<(wgpu::Device, u64)>,
}

impl Finishing<'_> {
    /// Whether another device, or none, has been connected since this began.
    fn device_replaced(&self) -> bool {
        let state = self.mailbox.lock();
        let current = state
            .connection
            .as_ref()
            .map(|connection| connection.generation);
        self.device
            .as_ref()
            .is_some_and(|&(_, began)| current != Some(began))
    }
}

impl Drop for Finishing<'_> {
    fn drop(&mut self) {
        drop(self.device.take());
        self.mailbox.lock().finishing = false;
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
    let evaluated = std::panic::catch_unwind(AssertUnwindSafe(|| {
        evaluator.evaluate(&wanted.key, || mailbox.superseded())
    }));
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
    let Some(finishing) = mailbox.start_finishing() else {
        return;
    };
    let result = match result {
        Ok(image) => {
            let (info, planes) = image.into_parts();
            match stage(&finishing, &info, planes, sizing) {
                Ok(pixels) => Ok(ViewerImage { info, pixels }),
                Err(reason) => Err(ViewerError::CannotShow(reason)),
            }
        }
        Err(error) => Err(ViewerError::Eval(error)),
    };
    // The result's GPU objects, if any, go before the finishing ends. A result finished for a
    // replaced device is dropped, even a failure, which may be that device's: connecting the
    // new one asks for the request again.
    if mailbox.superseded() || finishing.device_replaced() {
        drop(result);
    } else {
        deliver(Evaluated { wanted, result });
    }
    drop(finishing);
}

/// Converts the planes into the staging buffer and makes their texture. `Ok(None)` without a GPU
/// or pixels, or if the request was superseded or the worker stopped while waiting for the buffer,
/// which the check for a superseded result after this catches.
fn stage(
    finishing: &Finishing,
    info: &ImageInfo,
    planes: [Arc<[f32]>; 4],
    sizing: &mut StagingSizing,
) -> Result<Option<Staged>, String> {
    let Some((device, device_generation)) = &finishing.device else {
        return Ok(None);
    };
    let window = info.data_window;
    if is_empty(info) {
        return Ok(None);
    }
    let limits = device.limits();
    let size = staging_size(window.width, window.height);
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
            "its pixels take {size} bytes at half precision, and this GPU's buffers take at most \
             {}",
            limits.max_buffer_size
        ));
    }
    let Some((mut staging, mapped)) = finishing.mailbox.take_staging(*device_generation) else {
        return Ok(None);
    };
    // Until the buffer is back in `staging`, a failure drops it, leaving the slot empty.
    let buffer = match staging.buffer.take() {
        Some(buffer) => match sizing.replacement(buffer.size(), size) {
            None if mapped => buffer,
            None => {
                map(device, &buffer).map_err(|error| {
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
        None => staging_buffer(device, sizing.first(size))?,
    };
    let bytes_per_row = row_bytes(window.width);
    {
        let mut view = buffer
            .get_mapped_range_mut(..size)
            .map_err(|error| format!("the GPU could not write its staging buffer: {error}"))?;
        let mut bytes = view.slice(..);
        let halves = bytes.len() / size_of::<f16>();
        let start = bytes.as_raw_element_ptr().cast::<f16>();
        // SAFETY: the mapping is `size` bytes, aligned to `wgpu::MAP_ALIGNMENT`, which is at least
        // f16's alignment, and only this view accesses it until the view drops. Mapped memory may
        // be write-combining, so the conversion only writes to it.
        let out = unsafe { std::slice::from_raw_parts_mut(start.as_ptr(), halves) };
        convert_rows(
            planes.each_ref().map(|plane| &plane[..]),
            window.width as usize,
            out,
            bytes_per_row as usize / 2,
        );
    }
    drop(planes);
    buffer.unmap();
    staging.buffer = Some(buffer);
    let texture = allocate(device, "the image", || {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d {
                width: window.width,
                height: window.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    })?;
    Ok(Some(Staged {
        texture,
        bytes_per_row,
        staging,
    }))
}

/// When the worker replaces its staging buffer. It grows to fit any larger image, and shrinks
/// only after several images in a row would fit in half of it, so that one large image does not
/// hold its memory for the rest of the session, and alternating large and small images does not
/// make a buffer for each.
#[derive(Debug, Default)]
struct StagingSizing {
    images_fitting_half: u32,
}

impl StagingSizing {
    /// The size of buffer to make, where there is none, to stage `needed` bytes.
    fn first(&mut self, needed: u64) -> u64 {
        self.images_fitting_half = 0;
        fitted(needed)
    }

    /// The size of buffer to replace a `current`-byte one with, to stage `needed` bytes, or
    /// `None` to keep it.
    fn replacement(&mut self, current: u64, needed: u64) -> Option<u64> {
        let fitted = fitted(needed);
        if current / 2 > fitted {
            self.images_fitting_half += 1;
        } else {
            self.images_fitting_half = 0;
        }
        if current < needed || self.images_fitting_half == IMAGES_BEFORE_SHRINKING {
            self.images_fitting_half = 0;
            Some(fitted)
        } else {
            None
        }
    }
}

/// The size of buffer that stages `needed` bytes, and at least a first-size image.
fn fitted(needed: u64) -> u64 {
    let (first_width, first_height) = FIRST_STAGING_SIZE;
    needed.max(staging_size(first_width, first_height))
}

pub(crate) fn is_empty(info: &ImageInfo) -> bool {
    info.data_window.width == 0 || info.data_window.height == 0
}

/// Bytes per staged row: four f16 channels a pixel, padded as buffer-to-texture copies require.
fn row_bytes(width: u32) -> u32 {
    (width * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

fn staging_size(width: u32, height: u32) -> u64 {
    u64::from(row_bytes(width)) * u64::from(height)
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
fn allocate<T>(device: &wgpu::Device, what: &str, make: impl FnOnce() -> T) -> Result<T, String> {
    // Error scopes belong to the thread that pushes them, so the GUI thread's errors stay out.
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let made = make();
    match pollster::block_on(scope.pop()) {
        Some(error) => Err(format!("the GPU has no memory for {what}: {error}")),
        None => Ok(made),
    }
}

/// Maps the buffer for writing once the GPU has finished copying out of it.
fn map(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), String> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Write, .., move |result| {
        let _ = sender.send(result);
    });
    loop {
        wait_for_device(device)?;
        if let Ok(result) = receiver.try_recv() {
            return result.map_err(|error| error.to_string());
        }
    }
}

/// Interleaves planes of rows `width` pixels long, in `Channel::ALL`'s order, into RGBA f16 rows
/// `halves_per_row` apart in `out`, in the planes' row order.
fn convert_rows(planes: [&[f32]; 4], width: usize, out: &mut [f16], halves_per_row: usize) {
    // A row is interleaved where it stays in cache, then converted in one call, which only writes
    // to `out` and uses the CPU's f16 instructions where it has them.
    let mut interleaved = vec![[0.0; 4]; width];
    let rows = out.chunks_mut(halves_per_row);
    for (start, target) in (0..planes[0].len()).step_by(width).zip(rows) {
        let [r, g, b, a] = planes.map(|plane| &plane[start..start + width]);
        for (x, pixel) in interleaved.iter_mut().enumerate() {
            *pixel = [r[x], g[x], b[x], a[x]];
        }
        half::slice::HalfFloatSliceExt::convert_from_f32_slice(
            &mut target[..width * 4],
            interleaved.as_flattened(),
        );
    }
}

#[cfg(test)]
mod tests;
