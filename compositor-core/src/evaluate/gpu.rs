//! The evaluator's GPU half: evaluation's state on one device generation, and a request recorded
//! into one submission on it.
//!
//! The GPU table holds results as per-plane textures under the CPU table's keys. Decoded Readers
//! are the one result the two tables share: the CPU table keeps the decoded image and the GPU
//! table its upload, so a new device generation uploads again without decoding. Nothing is read
//! back.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use opencompositor_kernels::{Exposure, Merge};

use super::merge::{is_the_base, merged_warnings, merged_window};
use super::{EvalError, Evaluator, Recipe, ResultKey, Retained, distinct_bytes, needs};
use crate::image::{Channel, Image, ImageInfo, Plane, Window};
use crate::kernels::{Kernels, PLANE_FORMAT, Planes, extent};
use crate::read::read_image;

/// The most bytes of textures evaluation retains on one device. A default, sized for a GPU of
/// 6 GB beside the viewer's textures and the GUI's surfaces.
const RETAINED_TEXTURE_BYTES_CAP: u64 = 2 << 30;

/// How long waiting for the GPU to free memory sleeps between polls.
const IDLE_POLL: Duration = Duration::from_millis(1);

/// What every plane texture may be used for: read and written by the kernels, uploaded to, and
/// copied from by the viewer, so that any released plane can serve any role.
const PLANE_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_DST)
    .union(wgpu::TextureUsages::COPY_SRC);

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
    cap: u64,
    /// Submissions made, and the number of the last the GPU has completed, which a callback on
    /// the queue advances.
    submitted: u64,
    completed: Arc<AtomicU64>,
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
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuEvaluation {
        GpuEvaluation {
            device: device.clone(),
            queue: queue.clone(),
            kernels: Kernels::new(device),
            retained: Vec::new(),
            cap: RETAINED_TEXTURE_BYTES_CAP,
            submitted: 0,
            completed: Arc::new(AtomicU64::new(0)),
            uploaded_bytes: 0,
            #[cfg(test)]
            shortage: tests::Shortage::default(),
        }
    }

    /// Whether the GPU has completed every submission evaluation made, checked without
    /// blocking: waiting would hold the device's lock against the GUI thread's present for the
    /// whole wait.
    pub fn idle(&self) -> bool {
        // A lost device fails the poll; the loss is noticed where the device is.
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.completed.load(Ordering::Acquire) >= self.submitted
    }

    /// Waits without blocking until the GPU has completed evaluation's submissions, so that the
    /// textures they used and nothing holds any longer are freed. False if `stop` returned true
    /// first.
    pub fn wait_until_idle(&self, stop: impl Fn() -> bool) -> bool {
        while !self.idle() {
            if stop() {
                return false;
            }
            std::thread::sleep(IDLE_POLL);
        }
        true
    }

    /// The bytes of planes uploaded since this generation began.
    pub fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// Releases every retained result, as when the viewer wants nothing.
    pub fn release_all(&mut self) {
        self.retained.clear();
    }

    fn find(&self, key: &ResultKey) -> Option<GpuImage> {
        (self.retained.iter())
            .find(|retained| Arc::ptr_eq(&retained.key.0, &key.0))
            .map(|retained| retained.image.clone())
    }

    /// Releases every retained result the request does not contain, and returns the released
    /// planes nothing else holds, for the request's outputs and uploads.
    fn release_except(&mut self, keys: &[&ResultKey]) -> Vec<GpuPlane> {
        let mut released = Vec::new();
        for retained in std::mem::take(&mut self.retained) {
            if keys.iter().any(|key| Arc::ptr_eq(&key.0, &retained.key.0)) {
                self.retained.push(retained);
                continue;
            }
            let unheld = (retained.image.planes.into_iter().flatten())
                .filter(|plane| Arc::strong_count(&plane.0) == 1);
            for plane in unheld {
                // A plane shared by channels appears once per channel.
                if !released.iter().any(|other| GpuPlane::ptr_eq(other, &plane)) {
                    released.push(plane);
                }
            }
        }
        released
    }

    fn retained_bytes(&self) -> u64 {
        distinct_texture_bytes(
            (self.retained.iter()).flat_map(|retained| retained.image.planes.iter().flatten()),
        )
    }

    /// A plane over `window`'s size whose contents the caller overwrites: one of `released` of
    /// that size, else a new one. wgpu clears every new texture, which costs several times
    /// Exposure's own dispatch at 4K, so a request takes released planes first.
    fn plane(&mut self, released: &mut Vec<GpuPlane>, window: Window) -> GpuPlane {
        let size = extent(window);
        if let Some(index) = released.iter().position(|plane| plane.0.size() == size) {
            return released.swap_remove(index);
        }
        let texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plane"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PLANE_FORMAT,
            usage: PLANE_USAGE,
            view_formats: &[],
        }));
        #[cfg(test)]
        self.shortage.made(&texture);
        GpuPlane(texture)
    }

    /// Records the upload of a decoded image's planes, each distinct plane once, so that planes
    /// its channels share stay shared.
    fn upload(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        origin: &ResultKey,
        image: &Image,
        released: &mut Vec<GpuPlane>,
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
        released: &mut Vec<GpuPlane>,
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
        released: &mut Vec<GpuPlane>,
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
        // The stack between layers, dropped once recorded: wgpu keeps its textures until the
        // submission completes.
        let mut scratch = Planes::default();
        self.kernels
            .fold(
                encoder,
                &Merge { mix },
                (planes_of(base, stand_in), base.info.data_window),
                &stack,
                window,
                outputs.each_ref().map(|plane| &*plane.0),
                &mut scratch,
            )
            .expect("planes over their windows");
        Ok(GpuImage {
            info,
            planes: Some(outputs),
        })
    }

    /// Submits a request recorded on this generation, retains what its recording chose, and
    /// returns its image. Its completion is counted when the GPU reports it. Fails if wgpu
    /// rejected anything in the submission, which then runs none of it, so that nothing is
    /// retained.
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
        self.queue.on_submitted_work_done(move || {
            completed.fetch_max(number, Ordering::Release);
        });
        if let Some(error) = pollster::block_on(validation.pop()) {
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
    /// data window is empty. Channels that share values share a texture.
    pub fn texture(&self, channel: Channel) -> Option<&wgpu::Texture> {
        (self.planes.as_ref()).map(|planes| &*planes[channel as usize].0)
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

/// A request recorded on one generation and not yet submitted: [`GpuEvaluation::submit`]
/// submits it and retains its results, and dropping it submits and retains nothing. Its caller
/// may record more into the same submission, reading the image, on the same thread: the
/// validation scope that decides whether its results are kept belongs to the thread.
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

    /// The encoder, to record more into the request's submission, beside the image to read.
    pub fn encoder_and_image(&mut self) -> (&mut wgpu::CommandEncoder, &GpuImage) {
        (&mut self.encoder, &self.image)
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
    ///
    /// A request for an operation that runs only on the CPU fails before anything is recorded.
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
    /// If the GPU has no memory for the request, this releases every retained GPU result the
    /// request does not read, Reader uploads included, waits without blocking for the GPU to
    /// free them, and records the request again, once. This is the one place that decides a
    /// request's evaluation does not fit.
    pub fn record_on(
        &mut self,
        gpu: &mut GpuEvaluation,
        key: &ResultKey,
        stop: impl Fn() -> bool,
    ) -> Option<Result<Recorded, EvalError>> {
        match self.record_attempt(gpu, key, &stop)? {
            Err(EvalError::OutOfGpuMemory { .. }) => {}
            recorded => return Some(recorded),
        }
        if !gpu.wait_until_idle(&stop) {
            return None;
        }
        self.record_attempt(gpu, key, &stop)
    }

    /// Records the request once, or fails, out of GPU memory, having released every retained
    /// GPU result the request does not read.
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
        let (needed, _) = needs(&steps, |index| held[index].is_some());
        let keys: Vec<&ResultKey> = steps.iter().map(|step| &step.key).collect();
        // The CPU table keeps what the request contains, its decoded Readers; nothing it
        // releases serves a GPU request.
        drop(self.release_except(&steps));
        let mut released = gpu.release_except(&keys);
        for (index, step) in steps.iter().enumerate() {
            let computed = needed[index] && held[index].is_none();
            if computed && matches!(*step.key.0, Recipe::Blur { .. }) {
                return Some(Err(EvalError::NoGpuPath {
                    origin: step.key.clone(),
                }));
            }
        }
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
                Recipe::Blur { .. } => unreachable!("refused above"),
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
            images[index] = Some(image);
        }
        let image = images.pop().flatten().expect("the requested image");
        let short = pollster::block_on(out_of_memory.pop()).is_some();
        #[cfg(test)]
        let short = short | gpu.shortage.take_refusal();
        if short {
            drop((encoder, image, images, to_retain, released));
            // Everything else, not just enough: the allocator returns memory to the driver only
            // when a whole block of it empties, so releasing a few results may free nothing.
            let read: Vec<&ResultKey> = read.iter().map(|&index| &steps[index].key).collect();
            drop(gpu.release_except(&read));
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
