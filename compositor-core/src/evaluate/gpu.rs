//! The evaluator's GPU half: evaluation's state on one device generation, and a request recorded
//! into one submission on it.
//!
//! The GPU table holds results as per-plane textures under the CPU table's keys. Decoded Readers
//! are the one result the two tables share: the CPU table keeps the decoded image and the GPU
//! table its upload, so a new device generation uploads again without decoding. Nothing is read
//! back.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use opencompositor_kernels::{Axis, BoxPass, BoxSum, Exposure, LineReduce, Merge};

use super::merge::{is_the_base, merged_warnings, merged_window};
use super::{
    EvalError, Evaluator, Recipe, ResultKey, Retained, check_node_bytes, distinct_bytes, needs,
};
use crate::image::{Channel, Image, ImageInfo, Plane, Window};
use crate::kernels::{Kernels, Planes, plane_bytes};
use crate::read::read_image;

/// The most bytes of textures evaluation retains on one device. A default, sized for a GPU of
/// 6 GB beside the viewer's textures and the GUI's surfaces.
const RETAINED_TEXTURE_BYTES_CAP: u64 = 2 << 30;

/// How long waiting for the GPU to free memory sleeps between polls.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// Evaluation's state on one device generation: the kernels' pipelines, the GPU table and the
/// count of submissions the GPU has completed. Dropping it drops the generation's GPU state, all
/// in this one place; nothing else holds the textures but the images it returned.
///
/// [`Evaluator::evaluate_on`] takes it by `&mut`, so retained GPU state cannot live anywhere
/// else.
#[derive(Debug)]
pub struct GpuEvaluation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    kernels: Kernels,
    retained: Vec<GpuRetained>,
    /// Planes released while something else held them, as the viewer holds the result it
    /// shows: reused once it lets go, rather than each edit making new ones.
    lent: Vec<GpuPlane>,
    cap: u64,
    /// Submissions made, and the number of the last the GPU has completed, which a callback on
    /// the queue advances.
    submitted: u64,
    completed: Arc<AtomicU64>,
    /// Set by the device's lost callback, or by a poll that found the device lost.
    lost: Arc<AtomicBool>,
    uploaded_bytes: u64,
    #[cfg(test)]
    shortage: tests::Shortage,
}

#[derive(Debug)]
struct GpuRetained {
    key: ResultKey,
    image: GpuImage,
}

impl GpuEvaluation {
    /// Evaluation's state on the device. `lost` is for the device's lost callback to set:
    /// wgpu keeps one callback a device, so its owner installs it. Without one, a loss is
    /// noticed only when a poll finds it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, lost: Arc<AtomicBool>) -> GpuEvaluation {
        GpuEvaluation {
            device: device.clone(),
            queue: queue.clone(),
            kernels: Kernels::new(device),
            retained: Vec::new(),
            lent: Vec::new(),
            cap: RETAINED_TEXTURE_BYTES_CAP,
            submitted: 0,
            completed: Arc::new(AtomicU64::new(0)),
            lost,
            uploaded_bytes: 0,
            #[cfg(test)]
            shortage: tests::Shortage::default(),
        }
    }

    /// Whether the GPU has completed every submission evaluation made, checked without
    /// blocking: waiting would hold the device's lock against the GUI thread's present for the
    /// whole wait. Fails once the device is lost, whose work never completes.
    pub fn idle(&self) -> Result<bool, DeviceLost> {
        self.poll()?;
        Ok(self.completed.load(Ordering::Acquire) >= self.submitted)
    }

    /// Waits without blocking until the GPU has completed evaluation's submissions, so that the
    /// textures they used and nothing holds any longer are freed. False if `stop` returned true
    /// first.
    pub fn wait_until_idle(&self, stop: impl Fn() -> bool) -> Result<bool, DeviceLost> {
        while !self.idle()? {
            if stop() {
                return Ok(false);
            }
            std::thread::sleep(IDLE_POLL);
        }
        Ok(true)
    }

    /// Runs the callbacks of the device's completed work, its lost callback among them, without
    /// waiting for any. Fails if the device is lost.
    fn poll(&self) -> Result<(), DeviceLost> {
        if poll_device(&self.device).is_err() {
            self.lost.store(true, Ordering::Relaxed);
        }
        self.check_lost()
    }

    fn check_lost(&self) -> Result<(), DeviceLost> {
        if self.lost.load(Ordering::Relaxed) {
            return Err(DeviceLost);
        }
        Ok(())
    }

    /// The bytes of planes uploaded since this generation began.
    pub fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// Releases every retained result, as when the viewer wants nothing.
    pub fn release_all(&mut self) {
        self.retained.clear();
        self.lent.clear();
    }

    fn find(&self, key: &ResultKey) -> Option<GpuImage> {
        (self.retained.iter())
            .find(|retained| Arc::ptr_eq(&retained.key.0, &key.0))
            .map(|retained| retained.image.clone())
    }

    /// Releases every retained result the request does not contain, and returns the released
    /// planes nothing else holds, for the request's outputs and uploads. Planes still held
    /// elsewhere are lent until a later request finds them free.
    fn release_except(&mut self, keys: &[&ResultKey]) -> Planes {
        let mut freed = std::mem::take(&mut self.lent);
        for retained in std::mem::take(&mut self.retained) {
            if keys.iter().any(|key| Arc::ptr_eq(&key.0, &retained.key.0)) {
                self.retained.push(retained);
            } else {
                freed.extend(retained.image.planes.into_iter().flatten());
            }
        }
        let mut released = Planes::default();
        for plane in distinct(freed) {
            if Arc::strong_count(&plane.0) > 1 {
                self.lent.push(plane);
            } else {
                released.keep(plane.0);
            }
        }
        released
    }

    fn retained_bytes(&self) -> u64 {
        distinct_texture_bytes(
            (self.retained.iter()).flat_map(|retained| retained.image.planes.iter().flatten()),
        )
    }

    /// A plane over `window`'s size whose contents the caller overwrites.
    fn plane(&self, released: &mut Planes, window: Window) -> GpuPlane {
        GpuPlane(released.take(&self.device, window))
    }

    /// Records the upload of a decoded image's planes, each distinct plane once, so that planes
    /// its channels share stay shared.
    fn upload(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        origin: &ResultKey,
        image: &Image,
        released: &mut Planes,
    ) -> Result<GpuImage, EvalError> {
        let (info, _) = image.clone().into_parts();
        let window = info.data_window;
        if window.is_empty() {
            return Ok(GpuImage { info, planes: None });
        }
        self.check_side(origin, window)?;
        let mut uploaded: Vec<(&Plane, GpuPlane)> = Vec::new();
        for plane in image.planes() {
            if !uploaded
                .iter()
                .any(|(other, _)| Plane::ptr_eq(other, plane))
            {
                let texture = self.plane(released, window);
                self.copy_in(encoder, plane, &texture.0, window);
                uploaded.push((plane, texture));
            }
        }
        let planes = image.planes().each_ref().map(|plane| {
            let (_, texture) = (uploaded.iter())
                .find(|(other, _)| Plane::ptr_eq(other, plane))
                .expect("every plane uploaded");
            texture.clone()
        });
        Ok(GpuImage {
            info,
            planes: Some(planes),
        })
    }

    /// Fails unless the device's textures can be as large as `window`, which `origin` would
    /// write.
    fn check_side(&self, origin: &ResultKey, window: Window) -> Result<(), EvalError> {
        let side = self.device.limits().max_texture_dimension_2d;
        if window.width.max(window.height) > side {
            return Err(EvalError::BeyondDevice {
                origin: origin.clone(),
                window,
                side,
            });
        }
        Ok(())
    }

    /// Records copying `plane`'s values into `texture` through buffers created mapped, never
    /// `Queue::write_texture`: that holds the queue's lock for pending writes while it allocates
    /// and copies, and every submit takes the lock, so the GUI thread's submits would wait
    /// behind an upload. Rows are padded to the copy alignment, and the plane goes in bands of
    /// rows small enough for the device's buffers.
    fn copy_in(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        plane: &Plane,
        texture: &wgpu::Texture,
        window: Window,
    ) {
        let width = window.width as usize;
        let row_bytes = width * size_of::<f32>();
        let padded = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
        let band_rows = (self.device.limits().max_buffer_size / padded as u64)
            .min(u64::from(window.height)) as usize;
        for (band, rows) in plane.chunks(band_rows * width).enumerate() {
            let height = rows.len() / width;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("upload"),
                size: (padded * height) as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            {
                // Mapping fails only when creating the buffer did, which the request's error
                // scope reports.
                let Ok(mut view) = buffer.slice(..).get_mapped_range_mut() else {
                    continue;
                };
                // Mapped memory may be write-combining, so it is only written, a row at a time
                // unless the rows need no padding.
                if padded == row_bytes {
                    view.copy_from_slice(bytes_of(rows));
                } else {
                    for (row, values) in rows.chunks_exact(width).enumerate() {
                        let start = row * padded;
                        view.slice(start..start + row_bytes)
                            .copy_from_slice(bytes_of(values));
                    }
                }
            }
            buffer.unmap();
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded as u32),
                        rows_per_image: None,
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: (band * band_rows) as u32,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: window.width,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.uploaded_bytes += (plane.len() * size_of::<f32>()) as u64;
    }

    /// Records Exposure, as the CPU's `exposure` computes it: R, G and B each through the map,
    /// planes the channels share staying shared, and A the input's.
    fn exposure(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        image: &GpuImage,
        stops: f64,
        released: &mut Planes,
    ) -> GpuImage {
        let Some([r, g, b, a]) = &image.planes else {
            return image.clone();
        };
        // The host's gain, as the CPU computes it.
        let op = Exposure {
            gain: stops.exp2() as f32,
        };
        let window = image.info.data_window;
        let mut scaled = |plane: &GpuPlane| -> GpuPlane {
            let output = self.plane(released, window);
            self.kernels
                .map(encoder, &op, [(&plane.0, window)], window, [&output.0])
                .expect("planes over the data window");
            output
        };
        let red = scaled(r);
        let green = if GpuPlane::ptr_eq(g, r) {
            red.clone()
        } else {
            scaled(g)
        };
        let blue = if GpuPlane::ptr_eq(b, r) {
            red.clone()
        } else if GpuPlane::ptr_eq(b, g) {
            green.clone()
        } else {
            scaled(b)
        };
        GpuImage {
            info: image.info.clone(),
            planes: Some([red, green, blue, a.clone()]),
        }
    }

    /// Records Merge, as the CPU's `merge` computes it: the fold over the connected inputs, the
    /// base first, into four planes over their union, or the base itself where the CPU returns
    /// it unchanged.
    fn merge(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        key: &ResultKey,
        inputs: &[&GpuImage],
        mix: f32,
        released: &mut Planes,
    ) -> Result<GpuImage, EvalError> {
        let Some((base, layers)) = inputs.split_first() else {
            return Ok(GpuImage::empty());
        };
        if is_the_base(layers.len(), mix) {
            return Ok((*base).clone());
        }
        let window = merged_window(key, inputs.iter().map(|input| input.info.data_window))?;
        let info = ImageInfo {
            display_window: base.info.display_window,
            data_window: window,
            warnings: merged_warnings(inputs.iter().map(|input| &input.info.warnings[..])),
        };
        if window.is_empty() {
            return Ok(GpuImage { info, planes: None });
        }
        // The outputs, and the stack between layers: the planes the last layer's dispatch reads
        // and those it writes, each four over the union.
        let stacks = (layers.len() - 1).min(2) as u64;
        check_node_bytes(
            key,
            window,
            plane_bytes(window).saturating_mul(4 * (1 + stacks)),
        )?;
        self.check_side(key, window)?;
        // An empty input has no textures, and covers nothing, so its planes are never read; the
        // fold still binds four, another input's.
        let stand_in = (inputs.iter())
            .find_map(|input| input.planes.as_ref())
            .expect("an input over the union");
        fn planes_of<'a>(
            image: &'a GpuImage,
            stand_in: &'a [GpuPlane; 4],
        ) -> [&'a wgpu::Texture; 4] {
            (image.planes.as_ref().unwrap_or(stand_in))
                .each_ref()
                .map(|plane| &*plane.0)
        }
        let outputs: [GpuPlane; 4] = [(); 4].map(|()| self.plane(released, window));
        let stack: Vec<([&wgpu::Texture; 4], Window)> = (layers.iter())
            .map(|layer| (planes_of(layer, stand_in), layer.info.data_window))
            .collect();
        self.kernels
            .fold(
                encoder,
                &Merge { mix },
                (planes_of(base, stand_in), base.info.data_window),
                &stack,
                window,
                outputs.each_ref().map(|plane| &*plane.0),
                released,
            )
            .expect("planes over their windows");
        Ok(GpuImage {
            info,
            planes: Some(outputs),
        })
    }

    /// Records Blur, as the CPU's `blur` computes it: each distinct plane's three box passes
    /// along its rows into a scratch plane of the grown width, then along the scratch plane's
    /// columns into an output over the data window grown by the reach. The empty image, and a
    /// size whose reach is zero, give the source itself.
    ///
    /// The scratch plane and the passes' intermediate planes go back to `released` once
    /// recorded, for later nodes of the request.
    fn blur(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        key: &ResultKey,
        image: &GpuImage,
        size: f64,
        released: &mut Planes,
    ) -> Result<GpuImage, EvalError> {
        let pass = BoxPass::new(size);
        let reach = pass.reach();
        let input = image.info.data_window;
        let Some(planes) = image.planes.as_ref().filter(|_| reach > 0) else {
            return Ok(image.clone());
        };
        let window = input.grow(reach).map_err(|oversize| EvalError::Oversize {
            origin: key.clone(),
            oversize,
        })?;
        let scratch_window = Window {
            width: window.width,
            ..input
        };
        let op = BoxSum::new(pass);
        let mut sources: Vec<&GpuPlane> = Vec::new();
        for plane in planes {
            if !sources.iter().any(|other| GpuPlane::ptr_eq(other, plane)) {
                sources.push(plane);
            }
        }
        let bytes = plane_bytes(window)
            .saturating_mul(sources.len() as u64)
            .saturating_add(plane_bytes(scratch_window))
            .saturating_add(intermediate_bytes(&op, input, window));
        check_node_bytes(key, window, bytes)?;
        self.check_side(key, window)?;
        // The outputs before the scratch plane, as on the CPU.
        let outputs: Vec<GpuPlane> = (sources.iter())
            .map(|_| self.plane(released, window))
            .collect();
        let scratch = self.plane(released, scratch_window);
        for (source, output) in sources.iter().zip(&outputs) {
            self.kernels
                .line(
                    encoder,
                    &op,
                    Axis::Rows,
                    BoxPass::PASSES,
                    &source.0,
                    (input.width, input.height),
                    &scratch.0,
                    released,
                )
                .expect("a scratch plane of the grown width");
            self.kernels
                .line(
                    encoder,
                    &op,
                    Axis::Columns,
                    BoxPass::PASSES,
                    &scratch.0,
                    (window.width, input.height),
                    &output.0,
                    released,
                )
                .expect("an output over the grown window");
        }
        released.keep(scratch.0);
        let blurred = planes.each_ref().map(|plane| {
            let index = (sources.iter())
                .position(|source| GpuPlane::ptr_eq(source, plane))
                .expect("every plane blurred");
            outputs[index].clone()
        });
        Ok(GpuImage {
            info: ImageInfo {
                data_window: window,
                ..image.info.clone()
            },
            planes: Some(blurred),
        })
    }

    /// Submits a request recorded on this generation, retains what its recording chose, and
    /// returns its image. Its completion is counted when the GPU reports it. Fails if wgpu
    /// rejected anything in the submission, which then runs none of it, or the device is lost,
    /// so that nothing is retained.
    pub fn submit(&mut self, recorded: Recorded) -> Result<GpuImage, EvalError> {
        let Recorded {
            origin,
            encoder,
            image,
            to_retain,
            validation,
        } = recorded;
        self.queue.submit([encoder.finish()]);
        self.submitted += 1;
        let (completed, number) = (Arc::clone(&self.completed), self.submitted);
        // Every submit polls the device, so this may run on the GUI thread inside its submit:
        // it must stay this small.
        self.queue.on_submitted_work_done(move || {
            completed.fetch_max(number, Ordering::Release);
        });
        let rejected = pollster::block_on(validation.pop());
        // A lost device drops the submission unrun, or fails its validation without a bug.
        if self.poll().is_err() {
            return Err(EvalError::GpuLost { origin });
        }
        if let Some(error) = rejected {
            return Err(EvalError::GpuValidation {
                origin,
                message: error.to_string(),
            });
        }
        self.retained.extend(to_retain);
        Ok(image)
    }

    /// The keys of the retained results, for planning a request before this is at hand.
    pub fn retained_keys(&self) -> Vec<ResultKey> {
        (self.retained.iter())
            .map(|retained| retained.key.clone())
            .collect()
    }
}

/// A plane's values as bytes, for copying into a mapped buffer.
fn bytes_of(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and every byte of it is a valid u8, whose alignment is 1, and
    // the bytes are borrowed for as long as the values are.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
}

/// The bytes of Blur's planes between its first pass and its last along each axis, from
/// `input` to the grown `window`: those of the rows' passes are as tall as the input, and those
/// of the columns' as wide as the window.
fn intermediate_bytes(op: &BoxSum, input: Window, window: Window) -> u64 {
    (1..BoxPass::PASSES)
        .map(|passes| {
            let growth = op.growth() * passes;
            let rows = Window {
                width: input.width + growth,
                ..input
            };
            let columns = Window {
                height: input.height + growth,
                ..window
            };
            plane_bytes(rows).saturating_add(plane_bytes(columns))
        })
        .fold(0, u64::saturating_add)
}

/// The planes, each once however many channels and images share it.
fn distinct(planes: impl IntoIterator<Item = GpuPlane>) -> Vec<GpuPlane> {
    let mut distinct: Vec<GpuPlane> = Vec::new();
    for plane in planes {
        if !distinct.iter().any(|other| GpuPlane::ptr_eq(other, &plane)) {
            distinct.push(plane);
        }
    }
    distinct
}

/// The bytes of the textures, each counted once however often it appears.
fn distinct_texture_bytes<'a>(planes: impl Iterator<Item = &'a GpuPlane>) -> u64 {
    let mut counted: Vec<&GpuPlane> = Vec::new();
    let mut bytes = 0;
    for plane in planes {
        if !counted.iter().any(|other| GpuPlane::ptr_eq(other, plane)) {
            counted.push(plane);
            let size = plane.0.size();
            bytes += u64::from(size.width) * u64::from(size.height) * size_of::<f32>() as u64;
        }
    }
    bytes
}

/// An image whose planes are textures on one device, from [`Evaluator::evaluate_on`]. Holding
/// one keeps its textures from being reused for later results.
#[derive(Clone, Debug)]
pub struct GpuImage {
    info: ImageInfo,
    /// In `Channel::ALL`'s order. `None` exactly when the data window is empty, which no texture
    /// can be.
    planes: Option<[GpuPlane; 4]>,
}

impl GpuImage {
    /// The image of an unconnected input, with no planes.
    fn empty() -> GpuImage {
        GpuImage {
            info: Image::empty().into_parts().0,
            planes: None,
        }
    }

    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// The channel's R32Float texture over the data window, bottom row first, or `None` if the
    /// data window is empty. Channels that share values share a texture. The texture may be
    /// larger than the data window, which then covers its texels from the origin; the rest are
    /// undefined.
    pub fn texture(&self, channel: Channel) -> Option<&wgpu::Texture> {
        (self.planes.as_ref()).map(|planes| &*planes[channel as usize].0)
    }

    /// The planes' textures in `Channel::ALL`'s order, for the viewer to hold while it shows
    /// them; `None` if the data window is empty.
    pub(crate) fn textures(&self) -> Option<[Arc<wgpu::Texture>; 4]> {
        (self.planes.as_ref()).map(|planes| planes.each_ref().map(|plane| Arc::clone(&plane.0)))
    }
}

/// One texture of a result, shared by the channels, images and tables that hold it: one the
/// tables release is reused only once nothing else holds it.
#[derive(Clone, Debug)]
struct GpuPlane(Arc<wgpu::Texture>);

impl GpuPlane {
    fn ptr_eq(a: &GpuPlane, b: &GpuPlane) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }
}

/// The device was lost: nothing more runs on it, and evaluation needs a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLost;

impl fmt::Display for DeviceLost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the GPU device was lost")
    }
}

impl std::error::Error for DeviceLost {}

/// Runs the callbacks of the device's completed work without waiting for any. Fails if the
/// device is lost.
pub(crate) fn poll_device(device: &wgpu::Device) -> Result<(), DeviceLost> {
    // wgpu 30 panics rather than returning an error when the driver reports the device lost
    // during a poll, because `WaitIdleError::Device` has no `PollError`. The panic is raised
    // after wgpu has released its locks and run the device's lost callback, so catching it
    // leaves wgpu usable and the loss already reported.
    match std::panic::catch_unwind(AssertUnwindSafe(|| device.poll(wgpu::PollType::Poll))) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(_)) | Err(_) => Err(DeviceLost),
    }
}

/// A request recorded on one generation and not yet submitted: [`GpuEvaluation::submit`]
/// submits it and retains its results, and dropping it submits and retains nothing. It is
/// submitted on the thread that recorded it: the validation scope that decides whether its
/// results are kept belongs to the thread.
#[derive(Debug)]
pub struct Recorded {
    origin: ResultKey,
    encoder: wgpu::CommandEncoder,
    image: GpuImage,
    to_retain: Vec<GpuRetained>,
    validation: wgpu::ErrorScopeGuard,
}

impl Recorded {
    pub fn image(&self) -> &GpuImage {
        &self.image
    }

    /// The request's encoder, for a test to record what wgpu rejects into its submission.
    #[cfg(test)]
    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }
}

impl Evaluator {
    /// The image a key names, computed on the GPU of `gpu`, its failure, or `None` if `stop`
    /// returned true before it was done.
    ///
    /// The request is one submission: uploads of decoded Readers and every node's dispatches,
    /// submitted when the requested image has been recorded and returned without waiting for
    /// the GPU: whatever the queue runs after it sees its results. A request that stops or fails
    /// submits nothing and retains nothing on the GPU. Decoded Readers are retained in this
    /// evaluator's table, so that a new generation uploads them again without decoding.
    pub fn evaluate_on(
        &mut self,
        gpu: &mut GpuEvaluation,
        key: &ResultKey,
        stop: impl Fn() -> bool,
    ) -> Option<Result<GpuImage, EvalError>> {
        let recorded = self.record_on(gpu, key, stop)?;
        Some(recorded.and_then(|recorded| gpu.submit(recorded)))
    }

    /// Reads the files of the Readers a request on a GPU retaining `on_gpu` would upload, unless
    /// this evaluator retains them already, so that recording it decodes nothing. `None` if
    /// `stop` returned true first. Fails as the first file that cannot be read.
    ///
    /// Like a request, it releases every retained result the key does not contain.
    pub fn read_for_gpu(
        &mut self,
        key: &ResultKey,
        on_gpu: &[ResultKey],
        stop: impl Fn() -> bool,
    ) -> Option<Result<(), EvalError>> {
        let steps = self.plan(key, on_gpu);
        let uploaded = |index: usize| {
            let key = &steps[index].key;
            on_gpu.iter().any(|held| Arc::ptr_eq(&held.0, &key.0))
        };
        let (needed, _) = needs(&steps, uploaded);
        drop(self.release_except(&steps));
        for (index, step) in steps.iter().enumerate() {
            let unread = step.retained.is_none() && !uploaded(index);
            if !needed[index] || !unread || !matches!(*step.key.0, Recipe::Reader { .. }) {
                continue;
            }
            if stop() {
                return None;
            }
            if let Err(error) = self.read_and_retain(&step.key) {
                return Some(Err(error));
            }
        }
        Some(Ok(()))
    }

    /// Records the image a key names on the GPU of `gpu`, as [`Evaluator::evaluate_on`] computes
    /// it, without submitting it.
    ///
    /// If the GPU has no memory for the request, this releases every retained GPU result but
    /// those the request reads and what they were computed from, Reader uploads excepted, waits
    /// without blocking for the GPU to free them, and records the request again, once. This is
    /// the one place that decides a request's evaluation does not fit.
    pub fn record_on(
        &mut self,
        gpu: &mut GpuEvaluation,
        key: &ResultKey,
        stop: impl Fn() -> bool,
    ) -> Option<Result<Recorded, EvalError>> {
        let lost = || EvalError::GpuLost {
            origin: key.clone(),
        };
        if gpu.check_lost().is_err() {
            return Some(Err(lost()));
        }
        match self.record_attempt(gpu, key, &stop)? {
            Err(EvalError::OutOfGpuMemory { .. }) => {}
            recorded => return Some(recorded),
        }
        match gpu.wait_until_idle(&stop) {
            Ok(true) => self.record_attempt(gpu, key, &stop),
            Ok(false) => None,
            Err(DeviceLost) => Some(Err(lost())),
        }
    }

    /// Records the request once, or fails, out of GPU memory, having released every retained
    /// GPU result but those the request reads and their ancestors other than Readers.
    ///
    /// An image the table does not retain is freed once its last reader is recorded, so that
    /// later nodes of the request write its planes rather than new ones.
    fn record_attempt(
        &mut self,
        gpu: &mut GpuEvaluation,
        key: &ResultKey,
        stop: impl Fn() -> bool,
    ) -> Option<Result<Recorded, EvalError>> {
        // Popped in reverse on every return. A creation that fails out of memory makes an
        // invalid object, and recording with it a validation error, which then means nothing.
        let validation = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let out_of_memory = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let gpu_keys = gpu.retained_keys();
        let steps = self.plan(key, &gpu_keys);
        let held: Vec<Option<GpuImage>> = steps.iter().map(|step| gpu.find(&step.key)).collect();
        let (needed, mut uses) = needs(&steps, |index| held[index].is_some());
        let keys: Vec<&ResultKey> = steps.iter().map(|step| &step.key).collect();
        // The CPU table keeps what the request contains, its decoded Readers; nothing it
        // releases serves a GPU request.
        drop(self.release_except(&steps));
        let mut released = gpu.release_except(&keys);
        let mut retained_bytes = gpu.retained_bytes();
        let read: Vec<usize> = (0..steps.len())
            .filter(|&index| needed[index] && held[index].is_some())
            .collect();
        let mut kept: Vec<bool> = held.iter().map(Option::is_some).collect();
        let mut to_retain = Vec::new();
        let mut images: Vec<Option<GpuImage>> = held;
        let mut encoder = (gpu.device).create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("evaluation"),
        });
        for (index, step) in steps.iter().enumerate() {
            if !needed[index] || images[index].is_some() {
                continue;
            }
            if stop() {
                return None;
            }
            let inputs: Vec<&GpuImage> = (step.inputs.iter())
                .map(|&input| images[input].as_ref().expect("inputs come first"))
                .collect();
            let operation = || match &*step.key.0 {
                Recipe::Reader { .. } => {
                    let decoded = match &step.retained {
                        Some(image) => image.clone(),
                        None => self.read_and_retain(&step.key)?,
                    };
                    gpu.upload(&mut encoder, &step.key, &decoded, &mut released)
                }
                Recipe::Exposure { stops, .. } => {
                    let empty = GpuImage::empty();
                    let source = inputs.first().copied().unwrap_or(&empty);
                    let stops = f64::from_bits(*stops);
                    Ok(gpu.exposure(&mut encoder, source, stops, &mut released))
                }
                Recipe::Merge { mix, .. } => {
                    let mix = f64::from_bits(*mix) as f32;
                    gpu.merge(&mut encoder, &step.key, &inputs, mix, &mut released)
                }
                Recipe::Blur { size, .. } => {
                    let empty = GpuImage::empty();
                    let source = inputs.first().copied().unwrap_or(&empty);
                    let size = f64::from_bits(*size);
                    gpu.blur(&mut encoder, &step.key, source, size, &mut released)
                }
            };
            // As on the CPU, a panic fails the node instead of ending the worker; the request
            // submits nothing.
            let image = match std::panic::catch_unwind(AssertUnwindSafe(operation)) {
                Ok(Ok(image)) => image,
                Ok(Err(error)) => return Some(Err(error)),
                Err(_) => {
                    return Some(Err(EvalError::Panicked {
                        origin: step.key.clone(),
                    }));
                }
            };
            // The CPU table's rule: a result only with its inputs, under the cap.
            if step.inputs.iter().all(|&input| kept[input]) {
                let held: Vec<&GpuPlane> = (inputs.iter())
                    .flat_map(|input| input.planes.iter().flatten())
                    .collect();
                let own = (image.planes.iter().flatten())
                    .filter(|plane| !held.iter().any(|other| GpuPlane::ptr_eq(other, plane)));
                let added = distinct_texture_bytes(own);
                if retained_bytes + added <= gpu.cap {
                    retained_bytes += added;
                    to_retain.push(GpuRetained {
                        key: step.key.clone(),
                        image: image.clone(),
                    });
                    kept[index] = true;
                }
            }
            for &input in &step.inputs {
                uses[input] -= 1;
                if uses[input] == 0
                    && let Some(input) = images[input].take()
                {
                    // Planes the table or the node's own image holds stay held.
                    for plane in distinct(input.planes.into_iter().flatten()) {
                        if Arc::strong_count(&plane.0) == 1 {
                            released.keep(plane.0);
                        }
                    }
                }
            }
            images[index] = Some(image);
        }
        let image = images.pop().flatten().expect("the requested image");
        let short = pollster::block_on(out_of_memory.pop()).is_some();
        #[cfg(test)]
        let short = short | gpu.shortage.refuses(&released.made);
        if short {
            drop((encoder, image, images, to_retain, released));
            // Everything else, not just enough: the allocator returns memory to the driver only
            // when a whole block of it empties, so releasing a few results may free nothing.
            // What a kept result was computed from stays, or the retry's plan, finding it gone,
            // would compute the chain anew; a Reader's key stays in the CPU table, with its
            // decoded image.
            let mut keep: Vec<bool> = (0..steps.len())
                .map(|index| read.contains(&index))
                .collect();
            for index in (0..steps.len()).rev() {
                if keep[index] {
                    for &input in &steps[index].inputs {
                        keep[input] |= !matches!(*steps[input].key.0, Recipe::Reader { .. });
                    }
                }
            }
            let keep: Vec<&ResultKey> = (steps.iter().zip(&keep))
                .filter_map(|(step, &keep)| keep.then_some(&step.key))
                .collect();
            drop(gpu.release_except(&keep));
            return Some(Err(EvalError::OutOfGpuMemory {
                origin: key.clone(),
            }));
        }
        Some(Ok(Recorded {
            origin: key.clone(),
            encoder,
            image,
            to_retain,
            validation,
        }))
    }

    /// Reads a Reader's file and retains the decoded image in this evaluator's table, under its
    /// cap, for the GPU table's uploads to come from.
    fn read_and_retain(&mut self, key: &ResultKey) -> Result<Image, EvalError> {
        let Recipe::Reader { path, .. } = &*key.0 else {
            unreachable!("a Reader's key");
        };
        let Some(path) = path else {
            return Ok(Image::empty());
        };
        let image = read_image(path).map_err(|error| EvalError::Read {
            origin: key.clone(),
            error,
        })?;
        let added = distinct_bytes(image.planes().iter());
        if self.retained_bytes() + added <= self.cap {
            self.retained.push(Retained {
                key: key.clone(),
                image: image.clone(),
            });
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests;
