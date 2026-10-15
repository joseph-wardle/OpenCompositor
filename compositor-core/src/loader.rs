//! Reading image files off the GUI thread, and staging their pixels for the GPU.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, mpsc};

use half::f16;

use crate::document::NodeId;
use crate::image::ImageInfo;
use crate::read::{ReadError, read_image};

/// The staging buffer's first size, a DCI 4K image, and its least.
const FIRST_STAGING_SIZE: (u32, u32) = (4096, 2160);

/// Loads in a row that fit in half the staging buffer before it shrinks to fit them.
const LOADS_BEFORE_SHRINKING: u32 = 3;

/// A node's file to read, as of its last reload. The worker reads the path, never the document.
///
/// The request is also the result's key: the viewer shows a result only while its request is the
/// one it wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadRequest {
    pub node: NodeId,
    pub path: PathBuf,
    /// How many times the node's file had been reloaded, so that a reload reads it again.
    pub reload: u64,
}

#[derive(Debug)]
pub struct Loaded {
    pub request: LoadRequest,
    pub result: Result<Decoded, LoadError>,
}

/// An image the worker read: its description, and its pixels staged for the GPU.
#[derive(Debug)]
pub struct Decoded {
    pub info: ImageInfo,
    /// `None` before the GPU exists, as headless, or if the data window is empty.
    pub(crate) pixels: Option<Staged>,
}

/// An image's pixels, RGBA f16 rows in the loader's staging buffer, and the texture for them.
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
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        self.staging.buffer.as_ref().expect("held until dropped")
    }
}

/// The loader's staging buffer, returned to its mailbox when dropped, by whichever thread holds
/// it, so that neither a skipped result nor a panic loses it.
#[derive(Debug)]
struct Staging {
    buffer: Option<wgpu::Buffer>,
    mailbox: Arc<Mailbox>,
}

impl Drop for Staging {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let mut state = self.mailbox.lock();
        // A buffer returned after the loader stopped would be dropped with the mailbox, which
        // may be on the worker's thread.
        if !state.stopping {
            state.staging = Some((buffer, false));
            self.mailbox.changed.notify_all();
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Read(ReadError),
    /// Read, but the GPU cannot hold it.
    CannotShow(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Read(error) => error.fmt(f),
            LoadError::CannotShow(reason) => write!(f, "cannot show it: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Read(error) => Some(error),
            LoadError::CannotShow(_) => None,
        }
    }
}

/// One worker reading one file at a time. A request replaces any request it has not started, so
/// after a burst of requests it reads at most the one in progress and the last. It skips the rest
/// of a read that a newer request supersedes, once after decoding and again after staging.
///
/// The worker never writes to a texture or submits to the queue: `Queue::write_texture` copies
/// under a queue-wide lock that the GUI thread's submit then waits for.
#[derive(Debug)]
pub struct Loader {
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
    request: Option<LoadRequest>,
    stopping: bool,
    /// The GPU's device, from `connect` until the loader drops.
    device: Option<wgpu::Device>,
    /// The staging buffer while it waits for the worker, and whether it is mapped.
    staging: Option<(wgpu::Buffer, bool)>,
    /// Whether the worker is staging or delivering a read, and so may hold GPU objects.
    finishing: bool,
}

impl Loader {
    /// Starts the worker. It passes each result to `deliver` on its own thread.
    pub fn start(deliver: impl Fn(Loaded) + Send + 'static) -> Loader {
        let mailbox = Arc::new(Mailbox::default());
        let worker = Arc::clone(&mailbox);
        std::thread::Builder::new()
            .name("loader".to_owned())
            .spawn(move || {
                let mut sizing = StagingSizing::default();
                while let Some(request) = worker.next() {
                    read(&worker, request, &deliver, &mut sizing);
                }
            })
            .expect("the loader thread starts");
        Loader { mailbox }
    }

    pub fn request(&self, request: LoadRequest) {
        self.mailbox.lock().request = Some(request);
        self.mailbox.changed.notify_all();
    }

    /// Lets the worker stage pixels for `device`. Until then it reads images without them.
    pub(crate) fn connect(&self, device: wgpu::Device) {
        // Mapped at creation, the buffer is zeroed here rather than during the first read.
        let (width, height) = FIRST_STAGING_SIZE;
        let staging = staging_buffer(&device, staging_size(width, height));
        let mut state = self.mailbox.lock();
        state.staging = Some((staging, true));
        state.device = Some(device);
    }
}

impl Drop for Loader {
    /// Stops the worker, waiting while it stages or delivers a read but not while it decodes
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
        let gpu = (state.staging.take(), state.device.take());
        drop(state);
        drop(gpu);
    }
}

impl Mailbox {
    fn lock(&self) -> std::sync::MutexGuard<'_, MailboxState> {
        // Nothing panics while holding the lock, so a poisoned state is still whole.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Waits for a request, or `None` once the loader is stopping.
    fn next(&self) -> Option<LoadRequest> {
        self.wait_for(|state| state.request.take())
    }

    /// Waits for the staging buffer to come back, or `None` once the loader is stopping.
    fn take_staging(&self) -> Option<(wgpu::Buffer, bool)> {
        self.wait_for(|state| state.staging.take())
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

    /// Whether a newer request is waiting, or the loader is stopping.
    fn superseded(&self) -> bool {
        let state = self.lock();
        state.stopping || state.request.is_some()
    }

    /// Begins staging and delivering a read, unless it is superseded.
    fn start_finishing(self: &Arc<Mailbox>) -> Option<Finishing<'_>> {
        let mut state = self.lock();
        if state.stopping || state.request.is_some() {
            return None;
        }
        state.finishing = true;
        Some(Finishing {
            mailbox: self,
            device: state.device.clone(),
        })
    }
}

/// The worker's staging and delivery of a read. The loader's drop waits for it to end.
struct Finishing<'a> {
    mailbox: &'a Arc<Mailbox>,
    device: Option<wgpu::Device>,
}

impl Drop for Finishing<'_> {
    fn drop(&mut self) {
        drop(self.device.take());
        self.mailbox.lock().finishing = false;
        self.mailbox.changed.notify_all();
    }
}

/// Reads the request's file on the worker's thread, then stages and delivers it.
fn read(
    mailbox: &Arc<Mailbox>,
    request: LoadRequest,
    deliver: &impl Fn(Loaded),
    sizing: &mut StagingSizing,
) {
    // A panic on this thread would end the worker and every later read with it; the panic hook
    // has already reported it. The EXR reader contains its decoding threads' panics itself. A
    // failed allocation aborts the process regardless.
    let read = std::panic::catch_unwind(AssertUnwindSafe(|| read_image(&request.path)))
        .unwrap_or_else(|_| Err(ReadError::Decode("the reader panicked".to_owned())));
    let Some(finishing) = mailbox.start_finishing() else {
        return;
    };
    let result = match read {
        Ok(image) => {
            let (info, pixels) = image.into_parts();
            match stage(&finishing, &info, pixels, sizing) {
                Ok(pixels) => Ok(Decoded { info, pixels }),
                Err(reason) => Err(LoadError::CannotShow(reason)),
            }
        }
        Err(error) => Err(LoadError::Read(error)),
    };
    // The result's GPU objects, if any, go before the finishing ends.
    if mailbox.superseded() {
        drop(result);
    } else {
        deliver(Loaded { request, result });
    }
    drop(finishing);
}

/// Converts the pixels into the staging buffer and makes their texture. `Ok(None)` without a GPU
/// or pixels, or if the loader stopped while waiting for the buffer, which the check for a
/// superseded read after this catches.
fn stage(
    finishing: &Finishing,
    info: &ImageInfo,
    pixels: Vec<f32>,
    sizing: &mut StagingSizing,
) -> Result<Option<Staged>, String> {
    let Some(device) = &finishing.device else {
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
    let Some((buffer, mapped)) = finishing.mailbox.take_staging() else {
        return Ok(None);
    };
    let buffer = if let Some(replacement) = sizing.replacement(buffer.size(), size) {
        buffer.destroy();
        staging_buffer(device, replacement)
    } else if mapped {
        buffer
    } else {
        if let Err(error) = map(device, &buffer) {
            finishing.mailbox.lock().staging = Some((buffer, false));
            return Err(format!("the GPU could not map its staging buffer: {error}"));
        }
        buffer
    };
    let bytes_per_row = row_bytes(window.width);
    {
        let mut view = buffer
            .get_mapped_range_mut(..size)
            .expect("the buffer is mapped and large enough");
        let mut bytes = view.slice(..);
        let halves = bytes.len() / size_of::<f16>();
        let start = bytes.as_raw_element_ptr().cast::<f16>();
        // SAFETY: the mapping is `size` bytes, aligned to `wgpu::MAP_ALIGNMENT`, which is at least
        // f16's alignment, and only this view accesses it until the view drops. Mapped memory may
        // be write-combining, so the conversion only writes to it.
        let out = unsafe { std::slice::from_raw_parts_mut(start.as_ptr(), halves) };
        convert_rows(
            &pixels,
            window.width as usize,
            out,
            bytes_per_row as usize / 2,
        );
    }
    drop(pixels);
    buffer.unmap();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
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
    });
    Ok(Some(Staged {
        texture,
        bytes_per_row,
        staging: Staging {
            buffer: Some(buffer),
            mailbox: Arc::clone(finishing.mailbox),
        },
    }))
}

/// When the worker replaces its staging buffer. It grows to fit any larger image, and shrinks
/// only after several loads in a row would fit in half of it, so that one large image does not
/// hold its memory for the rest of the session, and alternating large and small images does not
/// make a buffer for each.
#[derive(Debug, Default)]
struct StagingSizing {
    loads_fitting_half: u32,
}

impl StagingSizing {
    /// The size of buffer to replace a `current`-byte one with, to stage `needed` bytes, or
    /// `None` to keep it.
    fn replacement(&mut self, current: u64, needed: u64) -> Option<u64> {
        let (first_width, first_height) = FIRST_STAGING_SIZE;
        let fitted = needed.max(staging_size(first_width, first_height));
        if current / 2 > fitted {
            self.loads_fitting_half += 1;
        } else {
            self.loads_fitting_half = 0;
        }
        if current < needed || self.loads_fitting_half == LOADS_BEFORE_SHRINKING {
            self.loads_fitting_half = 0;
            Some(fitted)
        } else {
            None
        }
    }
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
fn staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("image staging"),
        size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    })
}

/// Maps the buffer for writing once the GPU has finished copying out of it.
fn map(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), String> {
    let (sender, receiver) = mpsc::channel();
    buffer.map_async(wgpu::MapMode::Write, .., move |result| {
        let _ = sender.send(result);
    });
    loop {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| error.to_string())?;
        if let Ok(result) = receiver.try_recv() {
            return result.map_err(|error| error.to_string());
        }
    }
}

/// Converts interleaved RGBA rows of `width` pixels to f16, `halves_per_row` apart in `out`.
fn convert_rows(pixels: &[f32], width: usize, out: &mut [f16], halves_per_row: usize) {
    let row = width * 4;
    for (source, target) in pixels.chunks_exact(row).zip(out.chunks_mut(halves_per_row)) {
        half::slice::HalfFloatSliceExt::convert_from_f32_slice(&mut target[..row], source);
    }
}

#[cfg(test)]
mod tests;
