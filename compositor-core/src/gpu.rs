//! The GPU: one device for every pane, and a surface for each pane while it has a native window.

use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use open_ng::{Graph, Renderer, View, ViewBuffers, Viewport};
use wgpu::rwh::{RawDisplayHandle, RawWindowHandle};

use crate::app::{Notice, Pane, Waker};
use crate::evaluate::poll_device;
use crate::image::Window;
use crate::viewer::Placement;
use crate::worker::Pixels;

/// Behind everything a pane draws.
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// A pane's native window, from which its surface on each device is made.
#[derive(Clone, Copy, Debug)]
pub struct NativeWindow {
    display: RawDisplayHandle,
    window: RawWindowHandle,
}

impl NativeWindow {
    /// # Safety
    ///
    /// The handles must stay valid until the shell dispatches
    /// [`Event::SurfaceDestroyed`](crate::Event::SurfaceDestroyed) for the pane, or the app drops:
    /// until then the pane's surface may be made from them again, for a new device.
    pub unsafe fn new(display: RawDisplayHandle, window: RawWindowHandle) -> NativeWindow {
        NativeWindow { display, window }
    }
}

/// A pane's size in physical pixels, and its physical pixels per logical pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaneSize {
    pub width: u32,
    pub height: u32,
    pub scale_factor: f32,
}

impl PaneSize {
    pub(crate) fn viewport(self) -> Viewport {
        Viewport {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
            scale_factor: self.scale_factor,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Gpu {
    // Fields drop in order: the device and the surfaces it holds go first.
    device: Option<Device>,
    /// Kept while the pane has a native window, so that every device made can draw to it.
    windows: [Option<PaneWindow>; Pane::COUNT],
    /// Made with the first surface, so that headless use needs no GPU backend.
    instance: Option<wgpu::Instance>,
    /// Only in the lifecycle check.
    checks: Option<FrameChecks>,
    /// Each pane's last frame, only in the drag check.
    timings: Option<[Option<FrameTiming>; Pane::COUNT]>,
    /// Numbers the frames timed.
    frames_timed: u64,
    /// Errors on every device of the session that are bugs: those no error scope captured, and
    /// evaluation's rejected commands.
    errors: Arc<AtomicU64>,
    /// Counts the attempts to make a device, telling each device's notices and staged images
    /// apart.
    device_generation: u64,
    /// Tells the GUI thread that a device was lost.
    waker: Waker,
}

#[derive(Debug)]
struct PaneWindow {
    window: NativeWindow,
    size: PaneSize,
}

/// A device made for the panes, for the worker to evaluate or stage images on.
#[derive(Debug)]
pub(crate) struct NewDevice {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Set by the device's lost callback.
    pub(crate) lost: Arc<AtomicBool>,
    pub(crate) generation: u64,
    /// Why panes with a native window cannot draw on it.
    pub(crate) pane_errors: Vec<String>,
}

/// What the frames a pane drew on the current device prove, for the shell's lifecycle check:
/// that the GPU finished them, and what the last one put at the pane's centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameCheck {
    /// The device's generation, which advances with each attempt to make a device.
    pub generation: u64,
    /// How many of the pane's frames the GPU has finished.
    pub finished: u64,
    /// The centre pixel of the pane's last finished frame, as sRGB-encoded RGBA, if its surface
    /// can be read back.
    pub centre: Option<[u8; 4]>,
}

#[derive(Debug, Default)]
struct FrameChecks {
    /// The device the frames were drawn on. Every new device starts the counts again.
    generation: u64,
    /// Counted from the queue, which can report finished work on the evaluator's thread.
    finished: [Arc<AtomicU64>; Pane::COUNT],
    centres: [Option<[u8; 4]>; Pane::COUNT],
}

/// When a pane's frame passed each stage, for the shell's drag check.
#[derive(Clone, Debug)]
pub struct FrameTiming {
    /// The frame's number among the frames timed, of every pane, from 1.
    pub number: u64,
    /// The surface gave the frame's texture.
    pub acquired: Instant,
    /// The frame was submitted.
    pub submitted: Instant,
    /// The frame was handed to the display, which shows it at a later refresh.
    pub presented: Instant,
    /// When the GPU reported the frame complete, set by whichever thread's poll or submit ran
    /// the report. Nothing polls an idle device, so the check polls while it waits.
    pub completed: Arc<OnceLock<Instant>>,
}

/// What the viewer pane draws: the shown image's windows, and where they appear.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewerFrame {
    pub(crate) placement: Placement,
    pub(crate) display_window: Window,
    pub(crate) data_window: Window,
    /// The pane's physical pixels per logical pixel, which sets the outline's width.
    pub(crate) scale_factor: f32,
    /// The viewer's exposure as a factor on the working values, before they are clamped.
    pub(crate) gain: f32,
}

/// A pixel of a shown image to read its working values at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Probe {
    /// The image, numbered as [`Gpu::show_image`] was given it.
    pub(crate) image: u64,
    /// The pixel's texel in the image's planes, from the data window's bottom-left corner.
    pub(crate) texel: [u32; 2],
}

/// A probe's pixel, read back once the GPU had copied it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PixelRead {
    pub(crate) probe: Probe,
    /// In `Channel::ALL`'s order.
    pub(crate) values: [f32; 4],
}

/// Everything made once there is a surface to choose an adapter for, which goes with the device.
#[derive(Debug)]
struct Device {
    // Fields drop in order: the surfaces go before the device that draws to them.
    /// A pane's surface, while it has a native window and its surface is configured. Surfaces
    /// are never carried to another device: wgpu 30 reconfiguring a surface releases the old
    /// swapchain's fences and semaphores with the new device, which Vulkan does not allow.
    surfaces: [Option<PaneSurface>; Pane::COUNT],
    graph_buffers: ViewBuffers,
    renderer: Renderer,
    viewer: ViewerRenderer,
    format: wgpu::TextureFormat,
    queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    /// Set by the device's lost callback.
    lost: Arc<AtomicBool>,
}

#[derive(Debug)]
struct PaneSurface {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    pub(crate) fn new(waker: Waker) -> Gpu {
        Gpu {
            device: None,
            windows: Default::default(),
            instance: None,
            checks: None,
            timings: None,
            frames_timed: 0,
            errors: Arc::default(),
            device_generation: 0,
            waker,
        }
    }

    /// How many GPU errors there were this session: each is a bug.
    pub(crate) fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Counts an error an error scope captured that is a bug all the same, as wgpu's rejecting
    /// evaluation's commands.
    pub(crate) fn count_error(&self, error: &dyn fmt::Display) {
        count_error(&self.errors, error);
    }

    /// Records what every frame from now on proves, waiting for the GPU after each. Surfaces
    /// made from now on are made readable where they allow it.
    pub(crate) fn check_frames(&mut self) {
        self.checks.get_or_insert_default();
    }

    /// Records when each frame from now on passes each stage.
    pub(crate) fn time_frames(&mut self) {
        self.timings.get_or_insert_default();
    }

    /// The pane's last timed frame; `None` before one or without [`Gpu::time_frames`].
    pub(crate) fn frame_timing(&self, pane: Pane) -> Option<FrameTiming> {
        self.timings.as_ref()?[pane.index()].clone()
    }

    /// Runs the callbacks of the device's completed work, without waiting for any.
    pub(crate) fn poll(&self) {
        if let Some(device) = &self.device {
            // A loss runs the lost callback, which notices it.
            let _ = poll_device(&device.device);
        }
    }

    pub(crate) fn frame_check(&self, pane: Pane) -> Option<FrameCheck> {
        let checks = self.checks.as_ref()?;
        Some(FrameCheck {
            generation: checks.generation,
            finished: checks.finished[pane.index()].load(Ordering::Relaxed),
            centre: checks.centres[pane.index()],
        })
    }

    pub(crate) fn has_device(&self) -> bool {
        self.device.is_some()
    }

    /// Whether the pane has a configured surface to draw to.
    pub(crate) fn has_surface(&self, pane: Pane) -> bool {
        (self.device.as_ref()).is_some_and(|device| device.surfaces[pane.index()].is_some())
    }

    pub(crate) fn has_window(&self, pane: Pane) -> bool {
        self.windows[pane.index()].is_some()
    }

    /// Keeps the pane's native window, for every device made from now on, and makes its surface
    /// on the device if there is one. A pane whose surface fails gives up its window, as in
    /// [`Gpu::make_device`], so that it attaches again when next exposed; the error says why.
    pub(crate) fn attach(
        &mut self,
        pane: Pane,
        window: NativeWindow,
        size: PaneSize,
    ) -> Result<(), String> {
        self.windows[pane.index()] = Some(PaneWindow { window, size });
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let instance = self.instance.as_ref().expect("made before the device");
        let checks = self.checks.is_some();
        let configured = make_surface(instance, pane, window)
            .and_then(|surface| device.configure(pane, surface, size, checks));
        match configured {
            Ok(configured) => {
                device.surfaces[pane.index()] = Some(configured);
                Ok(())
            }
            Err(error) => {
                self.windows[pane.index()] = None;
                Err(error)
            }
        }
    }

    /// Makes a device with a surface on it for each pane with a native window: the one way a
    /// device is made, first and after a loss. A pane whose surface fails does not stop the
    /// others, and gives up its window, so that it attaches again when next exposed; with no
    /// surface the attempt fails and no device is kept. Each attempt is a new generation.
    pub(crate) fn make_device(&mut self) -> Result<NewDevice, String> {
        self.device_generation += 1;
        let instance = self.instance.get_or_insert_with(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            // The shell is tested on Vulkan only; WGPU_BACKEND chooses another.
            if std::env::var_os("WGPU_BACKEND").is_none() {
                descriptor.backends = wgpu::Backends::VULKAN;
            }
            wgpu::Instance::new(descriptor)
        });
        let mut pane_errors = Vec::new();
        let surfaces = Pane::ALL.map(|pane| {
            let window = self.windows[pane.index()].as_ref()?;
            make_surface(instance, pane, window.window)
                .map_err(|error| pane_errors.push(error))
                .ok()
        });
        let Some(first) = surfaces.iter().flatten().next() else {
            return Err(if pane_errors.is_empty() {
                "no pane has a native window".to_owned()
            } else {
                pane_errors.join("; ")
            });
        };
        let mut device = Device::new(
            instance,
            first,
            &self.errors,
            self.device_generation,
            &self.waker,
        )?;
        for (pane, surface) in Pane::ALL.into_iter().zip(surfaces) {
            let (Some(surface), Some(window)) = (surface, &self.windows[pane.index()]) else {
                continue;
            };
            match device.configure(pane, surface, window.size, self.checks.is_some()) {
                Ok(configured) => device.surfaces[pane.index()] = Some(configured),
                Err(error) => pane_errors.push(error),
            }
        }
        if device.surfaces.iter().all(Option::is_none) {
            return Err(pane_errors.join("; "));
        }
        for (window, surface) in self.windows.iter_mut().zip(&device.surfaces) {
            if surface.is_none() {
                *window = None;
            }
        }
        let new = NewDevice {
            device: device.device.clone(),
            queue: device.queue.clone(),
            lost: Arc::clone(&device.lost),
            generation: self.device_generation,
            pane_errors,
        };
        self.device = Some(device);
        if let Some(checks) = &mut self.checks {
            // New counters, so that the old device's late callbacks count nothing.
            *checks = FrameChecks {
                generation: new.generation,
                ..Default::default()
            };
        }
        Ok(new)
    }

    /// Drops the device of the generation, with its surfaces and everything made on it, unless
    /// a notice of its loss came after it was replaced. Returns whether it was dropped.
    pub(crate) fn retire_lost_device(&mut self, generation: u64) -> bool {
        if generation != self.device_generation || self.device.is_none() {
            return false;
        }
        self.device = None;
        true
    }

    /// Destroys the device for the shell's lifecycle check, which then notices its loss as it
    /// would a driver's.
    pub(crate) fn destroy_device(&mut self) {
        let Some(device) = &self.device else {
            return;
        };
        device.device.destroy();
        // `destroy` loses the device only at a later poll, and work on it until then would be
        // counted as errors.
        let _ = wait_for_device(&device.device);
    }

    /// Shows an image's planes in the viewer from the next frame, numbered for the probes that
    /// read it. Staged planes are copied into their textures first. Planes for an earlier device
    /// are dropped.
    ///
    /// The copy is submitted at once rather than with the viewer's next frame, which may never
    /// come while the viewer is closed, so that the staging buffer goes back to the worker.
    pub(crate) fn show_image(&mut self, pixels: Pixels, window: Window, number: u64) {
        let Some(device) = &mut self.device else {
            return;
        };
        if pixels.device_generation != self.device_generation {
            return;
        }
        if let Some(staged) = &pixels.staged {
            let mut encoder = device.device.create_command_encoder(&Default::default());
            for &(channel, offset) in &staged.copies {
                let texture = &pixels.planes[channel];
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: staged.buffer(),
                        layout: wgpu::TexelCopyBufferLayout {
                            offset,
                            bytes_per_row: Some(staged.bytes_per_row),
                            rows_per_image: None,
                        },
                    },
                    texture.as_image_copy(),
                    texture.size(),
                );
            }
            device.queue.submit([encoder.finish()]);
        }
        let image = ShownImage::new(
            &device.device,
            &device.viewer,
            pixels.planes,
            window,
            number,
        );
        device.viewer.image = Some(image);
    }

    /// Stops drawing the viewer's image, which a failed read or an unmarked slot replaces.
    pub(crate) fn clear_image(&mut self) {
        if let Some(device) = &mut self.device {
            device.viewer.image = None;
        }
    }

    /// Reads the working values at the probe's pixel of the image shown, if it is the probe's,
    /// once the reads already asked for are done: one at a time, the latest asked for next.
    pub(crate) fn read_pixel(&mut self, probe: Probe) {
        let Some(device) = &mut self.device else {
            return;
        };
        let reader = &mut device.viewer.reader;
        if reader.in_flight.is_some() {
            reader.next = Some(probe);
        } else {
            reader.start(
                &device.device,
                &device.queue,
                device.viewer.image.as_ref(),
                probe,
            );
        }
    }

    /// Whether a pixel read is waiting for the GPU.
    pub(crate) fn reading_pixel(&self) -> bool {
        (self.device.as_ref()).is_some_and(|device| device.viewer.reader.in_flight.is_some())
    }

    /// The pixel read the GPU has finished, without waiting for it, if its image is still
    /// shown, and starts the next one asked for.
    pub(crate) fn take_pixel_read(&mut self) -> Option<PixelRead> {
        let device = self.device.as_mut()?;
        // Nothing more is read from a lost device; its lost callback noticed the loss.
        poll_device(&device.device).ok()?;
        let reader = &mut device.viewer.reader;
        let image = device.viewer.image.as_ref();
        let read = reader.finish()?;
        if let Some(next) = reader.next.take() {
            reader.start(&device.device, &device.queue, image, next);
        }
        let shown = image.map(|image| image.number);
        read.filter(|read| Some(read.probe.image) == shown)
    }

    /// Drops the pane's surface, if it has one, and its native window, before the window goes.
    pub(crate) fn detach(&mut self, pane: Pane) {
        self.windows[pane.index()] = None;
        if let Some(device) = &mut self.device {
            device.surfaces[pane.index()] = None;
        }
    }

    pub(crate) fn resize(&mut self, pane: Pane, size: PaneSize) {
        if let Some(window) = &mut self.windows[pane.index()] {
            window.size = size;
        }
        let Some(device) = &mut self.device else {
            return;
        };
        let Some(target) = &mut device.surfaces[pane.index()] else {
            return;
        };
        target.config.width = size.width.max(1);
        target.config.height = size.height.max(1);
        target.surface.configure(&device.device, &target.config);
    }

    /// Draws one frame of the pane: the graph, or the viewer's frame if it shows an image.
    /// This can wait for the display while the pane has frames it has not shown.
    pub(crate) fn draw(
        &mut self,
        pane: Pane,
        graph: &Graph,
        view: &View,
        viewer_frame: Option<ViewerFrame>,
    ) -> Drawn {
        let Some(device) = &mut self.device else {
            return Drawn::Skipped;
        };
        let Some(PaneSurface { surface, config }) = &device.surfaces[pane.index()] else {
            return Drawn::Skipped;
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&device.device, config);
                return Drawn::Again;
            }
            wgpu::CurrentSurfaceTexture::Timeout => return Drawn::Again,
            // The pane is hidden; it is drawn again when exposed.
            wgpu::CurrentSurfaceTexture::Occluded => return Drawn::Skipped,
            // Only under an error scope or handler, which this code does not install.
            wgpu::CurrentSurfaceTexture::Validation => return Drawn::Skipped,
        };
        let acquired = Instant::now();
        match pane {
            Pane::Graph => device.renderer.prepare(
                &device.device,
                &device.queue,
                graph,
                view,
                &mut device.graph_buffers,
            ),
            Pane::Viewer => {
                if let Some(frame) = viewer_frame {
                    device.viewer.prepare(&device.queue, frame, config);
                }
            }
        }
        let target = frame.texture.create_view(&Default::default());
        let mut encoder = device.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pane"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            match pane {
                Pane::Graph => device.renderer.draw(&mut pass, &device.graph_buffers),
                Pane::Viewer => {
                    if viewer_frame.is_some() {
                        device.viewer.draw(&mut pass);
                    }
                }
            }
        }
        let readback = match &self.checks {
            Some(_) if config.usage.contains(wgpu::TextureUsages::COPY_SRC) => {
                Some(copy_centre(&device.device, &mut encoder, &frame.texture))
            }
            _ => None,
        };
        device.queue.submit([encoder.finish()]);
        let submitted = Instant::now();
        let completed = self.timings.is_some().then(|| {
            let completed = Arc::new(OnceLock::new());
            let report = Arc::clone(&completed);
            device.queue.on_submitted_work_done(move || {
                let _ = report.set(Instant::now());
            });
            completed
        });
        if let Some(checks) = &mut self.checks {
            checks.record(pane, device, readback);
        }
        device.queue.present(frame);
        if let (Some(timings), Some(completed)) = (&mut self.timings, completed) {
            self.frames_timed += 1;
            timings[pane.index()] = Some(FrameTiming {
                number: self.frames_timed,
                acquired,
                submitted,
                presented: Instant::now(),
                completed,
            });
        }
        Drawn::Presented
    }
}

/// What [`Gpu::draw`] did with the pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Drawn {
    /// A frame was submitted and presented.
    Presented,
    /// Nothing was drawn and nothing needs to be until the pane changes: there is no device or
    /// surface, or the pane is hidden.
    Skipped,
    /// Nothing was drawn, but it should be again: the surface was out of date and has been
    /// reconfigured, or the display took too long to free a frame.
    Again,
}

/// Copies the texture's centre pixel into a new buffer that can be mapped.
fn copy_centre(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("centre pixel"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: texture.width() / 2,
                y: texture.height() / 2,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            // One row needs no row pitch.
            layout: wgpu::TexelCopyBufferLayout::default(),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    buffer
}

impl FrameChecks {
    /// Waits for the frame just submitted, counting it once the GPU has finished it, and reads
    /// back its centre pixel if it was copied.
    fn record(&mut self, pane: Pane, device: &Device, readback: Option<wgpu::Buffer>) {
        let finished = Arc::clone(&self.finished[pane.index()]);
        device.queue.on_submitted_work_done(move || {
            finished.fetch_add(1, Ordering::Relaxed);
        });
        if let Some(buffer) = &readback {
            buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        }
        if wait_for_device(&device.device).is_err() {
            return;
        }
        let Some(view) = readback
            .as_ref()
            .and_then(|buffer| buffer.get_mapped_range(..).ok())
        else {
            return;
        };
        let mut pixel: [u8; 4] = view[..4].try_into().expect("four bytes were copied");
        if device.format == wgpu::TextureFormat::Bgra8UnormSrgb {
            pixel.swap(0, 2);
        }
        self.centres[pane.index()] = Some(pixel);
    }
}

/// Waits until the device has finished its submitted work, and runs the callbacks it made ready.
/// Only the GUI thread's frame checks and teardown wait so; the worker polls instead.
pub(crate) fn wait_for_device(device: &wgpu::Device) -> Result<(), String> {
    let wait = wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    };
    // A loss panics as `poll_device` says. Hypothesis, untested since it needs a real reset:
    // Vulkan's waits return `DEVICE_LOST` promptly after one, rather than holding the frame
    // checks here.
    match std::panic::catch_unwind(AssertUnwindSafe(|| device.poll(wait))) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(error.to_string()),
        Err(_) => Err("the GPU device was lost".to_owned()),
    }
}

/// Makes a surface for the pane's native window.
fn make_surface(
    instance: &wgpu::Instance,
    pane: Pane,
    window: NativeWindow,
) -> Result<wgpu::Surface<'static>, String> {
    let target = wgpu::SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: Some(window.display),
        raw_window_handle: window.window,
    };
    // SAFETY: `NativeWindow::new`'s contract keeps the handles valid until `detach`, which drops
    // the window and any surface made from it.
    unsafe { instance.create_surface_unsafe(target) }
        .map_err(|error| format!("no surface for the {pane:?} pane: {error}"))
}

/// The adapter that WGPU_ADAPTER_NAME names, or else wgpu's choice for the surface, which
/// WGPU_POWER_PREF steers. A CPU renderer is refused unless named: lavapipe can be installed
/// beside a GPU's driver, and recovering onto it would leave the session slow for good.
fn choose_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Result<wgpu::Adapter, String> {
    // Not `wgpu::util::initialize_adapter_from_env`, which panics when no adapter has the name,
    // as while a reset GPU has not come back.
    if let Ok(name) = std::env::var("WGPU_ADAPTER_NAME") {
        let name = name.to_lowercase();
        return pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .find(|adapter| {
                adapter.is_surface_supported(surface)
                    && adapter.get_info().name.to_lowercase().contains(&name)
            })
            .ok_or_else(|| format!("no GPU adapter named {name:?} can draw to the panes"));
    }
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
        compatible_surface: Some(surface),
        ..Default::default()
    };
    let adapter = pollster::block_on(instance.request_adapter(&options))
        .map_err(|error| format!("no GPU adapter: {error}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(format!(
            "the only GPU adapter, {}, renders on the CPU; WGPU_ADAPTER_NAME can select it",
            info.name
        ));
    }
    Ok(adapter)
}

/// Errors logged in full; later ones are only counted, so a bug repeated every frame does not
/// flood the log.
const LOGGED_ERRORS: u64 = 20;

/// Counts the errors on one device that no error scope captured, until the device is lost: work
/// on a lost device fails without being a bug. Its loss is noticed with its generation.
#[derive(Debug)]
struct ErrorCounter {
    counted: Arc<AtomicU64>,
    lost: Arc<AtomicBool>,
}

impl ErrorCounter {
    /// Replaces wgpu's default handler, which panics, so that a bug in one frame does not end the
    /// session. The handler and the lost callback run inside wgpu's calls, on any thread, so they
    /// only count, log and wake the GUI thread.
    ///
    /// A real loss calls the lost callback at once, in the call that failed; `destroy` calls it
    /// only at a later poll. Nothing polls while the panes are idle or hidden: a loss then is
    /// noticed at the next use of the GPU, and until then nothing visible depends on it.
    ///
    /// Returns the flag the lost callback sets.
    fn install(
        device: &wgpu::Device,
        counted: &Arc<AtomicU64>,
        generation: u64,
        waker: &Waker,
    ) -> Arc<AtomicBool> {
        let lost = Arc::new(AtomicBool::new(false));
        let counter = Arc::new(ErrorCounter {
            counted: Arc::clone(counted),
            lost: Arc::clone(&lost),
        });
        let handler = Arc::clone(&counter);
        device.on_uncaptured_error(Arc::new(move |error| handler.count(&error)));
        let waker = waker.clone();
        device.set_device_lost_callback(move |_, _| {
            counter.lost.store(true, Ordering::Relaxed);
            waker.wake(Notice::DeviceLost {
                generation,
                at: Instant::now(),
            });
        });
        lost
    }

    fn count(&self, error: &wgpu::Error) {
        if !self.lost.load(Ordering::Relaxed) {
            count_error(&self.counted, error);
        }
    }
}

/// Counts a GPU error, a bug, logging it unless too many were.
fn count_error(counted: &AtomicU64, error: &dyn fmt::Display) {
    let number = counted.fetch_add(1, Ordering::Relaxed) + 1;
    if number <= LOGGED_ERRORS {
        eprintln!("GPU error {number}: {error}");
    }
    if number == LOGGED_ERRORS {
        eprintln!("later GPU errors are counted, not logged");
    }
}

impl Device {
    fn new(
        instance: &wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        errors: &Arc<AtomicU64>,
        generation: u64,
        waker: &Waker,
    ) -> Result<Device, String> {
        let adapter = choose_adapter(instance, surface)?;
        // Images take the largest textures and staging buffers the adapter allows.
        let supported = adapter.limits();
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: supported.max_texture_dimension_2d,
                max_buffer_size: supported.max_buffer_size,
                ..Default::default()
            },
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
            .map_err(|error| format!("no GPU device: {error}"))?;
        // Before any other call on the device: wgpu notices a loss only when a call reaches the
        // driver, so with nothing in between, none goes unnoticed.
        let lost = ErrorCounter::install(&device, errors, generation, waker);
        // Panes draw clamped linear values, which an sRGB format encodes. Encoding in a shader too
        // would encode twice.
        let formats = surface.get_capabilities(&adapter).formats;
        let format = formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .ok_or("the surface offers no sRGB format")?;
        Ok(Device::for_format(adapter, device, queue, lost, format))
    }

    /// Configures the pane's surface at its size. Its errors are the pane's, not bugs, so they
    /// are captured rather than counted.
    fn configure(
        &self,
        pane: Pane,
        surface: wgpu::Surface<'static>,
        size: PaneSize,
        readable: bool,
    ) -> Result<PaneSurface, String> {
        if !self.adapter.is_surface_supported(&surface) {
            return Err(format!(
                "the adapter cannot draw to the {pane:?} pane's surface"
            ));
        }
        let capabilities = surface.get_capabilities(&self.adapter);
        if !capabilities.formats.contains(&self.format) {
            return Err(format!(
                "the {pane:?} pane's surface cannot show {:?}",
                self.format
            ));
        }
        let mut config = surface
            .get_default_config(&self.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| format!("the {pane:?} pane's surface has no configuration"))?;
        config.format = self.format;
        if readable && capabilities.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        // Mailbox replaces a waiting frame instead of blocking for the display, so a draw waits
        // only when the pane redraws faster than its screen refreshes, which the shell prevents.
        // Fifo, which every surface supports, is the fallback. The default is the driver's first
        // mode, which may be Immediate, which tears.
        let modes = capabilities.present_modes;
        config.present_mode = if modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        surface.configure(&self.device, &config);
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(format!(
                "the {pane:?} pane's surface cannot be configured: {error}"
            ));
        }
        Ok(PaneSurface { surface, config })
    }

    /// The renderers for panes of the format.
    fn for_format(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        lost: Arc<AtomicBool>,
        format: wgpu::TextureFormat,
    ) -> Device {
        let renderer = Renderer::new(&device, format, 1);
        let graph_buffers = ViewBuffers::new(&device, &renderer);
        let viewer = ViewerRenderer::new(&device, format);
        Device {
            graph_buffers,
            renderer,
            viewer,
            surfaces: Default::default(),
            format,
            queue,
            device,
            adapter,
            lost,
        }
    }
}

/// Draws the viewer's image and the outline of its display window, and reads its pixels.
#[derive(Debug)]
struct ViewerRenderer {
    image_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    image_layout: wgpu::BindGroupLayout,
    /// The shown image's planes, while the viewer has one.
    image: Option<ShownImage>,
    reader: PixelReader,
}

#[derive(Debug)]
struct ShownImage {
    group: wgpu::BindGroup,
    /// Held so that the worker writes no later result into them while they are shown, and for
    /// reading their pixels.
    planes: [Arc<wgpu::Texture>; 4],
    /// The data window's width and height, which the planes may exceed.
    size: [u32; 2],
    number: u64,
}

impl ShownImage {
    fn new(
        device: &wgpu::Device,
        viewer: &ViewerRenderer,
        planes: [Arc<wgpu::Texture>; 4],
        window: Window,
        number: u64,
    ) -> ShownImage {
        let views = planes
            .each_ref()
            .map(|texture| texture.create_view(&Default::default()));
        let entries: Vec<wgpu::BindGroupEntry> = (0..)
            .zip(&views)
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer image"),
            layout: &viewer.image_layout,
            entries: &entries,
        });
        ShownImage {
            group,
            planes,
            size: [window.width, window.height],
            number,
        }
    }
}

/// Reads one pixel of each plane of the shown image at a time, into a small buffer the GUI
/// thread maps without waiting.
#[derive(Debug)]
struct PixelReader {
    buffer: wgpu::Buffer,
    /// The probe being read, and once the GPU is done, whether its buffer mapped.
    in_flight: Option<(Probe, Arc<OnceLock<bool>>)>,
    /// The latest probe asked for while one was in flight.
    next: Option<Probe>,
}

/// Bytes of one pixel of a plane.
const PLANE_PIXEL_BYTES: u64 = 4;

impl PixelReader {
    fn new(device: &wgpu::Device) -> PixelReader {
        PixelReader {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pixel read"),
                size: 4 * PLANE_PIXEL_BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            in_flight: None,
            next: None,
        }
    }

    /// Copies the probe's pixel of each plane and maps the copies, unless the image is not the
    /// probe's or the pixel is outside it.
    fn start(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: Option<&ShownImage>,
        probe: Probe,
    ) {
        let Some(image) = image.filter(|image| image.number == probe.image) else {
            return;
        };
        let [x, y] = probe.texel;
        let [width, height] = image.size;
        if x >= width || y >= height {
            return;
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        for (plane, offset) in image.planes.iter().zip((0..).step_by(4)) {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: plane,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.buffer,
                    // One row needs no row pitch.
                    layout: wgpu::TexelCopyBufferLayout {
                        offset,
                        ..Default::default()
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit([encoder.finish()]);
        let mapped = Arc::new(OnceLock::new());
        let report = Arc::clone(&mapped);
        (self.buffer).map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = report.set(result.is_ok());
        });
        self.in_flight = Some((probe, mapped));
    }

    /// The read in flight, once its mapping is done: `Some(None)` if it failed.
    fn finish(&mut self) -> Option<Option<PixelRead>> {
        let (probe, mapped) = self.in_flight.as_ref()?;
        let (probe, mapped) = (*probe, *mapped.get()?);
        self.in_flight = None;
        if !mapped {
            return Some(None);
        }
        let values = self.buffer.get_mapped_range(..).ok().map(|view| {
            let bytes: [[u8; 4]; 4] = *view
                .as_chunks::<4>()
                .0
                .first_chunk()
                .expect("four planes were copied");
            bytes.map(f32::from_ne_bytes)
        });
        self.buffer.unmap();
        Some(values.map(|values| PixelRead { probe, values }))
    }
}

/// The size of the shader's `Frame`.
const FRAME_BYTES: u64 = 64;

impl ViewerRenderer {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> ViewerRenderer {
        let module = device.create_shader_module(wgpu::include_wgsl!("viewer.wgsl"));
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(FRAME_BYTES),
                },
                count: None,
            }],
        });
        let plane = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer image"),
            entries: &[plane(0), plane(1), plane(2), plane(3)],
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewer frame"),
            size: FRAME_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer frame"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });
        let pipeline = |name: &str, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("viewer {name}")),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("viewer {name}")),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(&format!("{name}_vertex")),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(&format!("{name}_fragment")),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        ViewerRenderer {
            image_pipeline: pipeline("image", &[Some(&frame_layout), Some(&image_layout)]),
            outline_pipeline: pipeline("outline", &[Some(&frame_layout)]),
            frame_buffer,
            frame_group,
            image_layout,
            image: None,
            reader: PixelReader::new(device),
        }
    }

    fn prepare(
        &self,
        queue: &wgpu::Queue,
        frame: ViewerFrame,
        config: &wgpu::SurfaceConfiguration,
    ) {
        let rect = |window: Window| {
            [
                window.x as f32,
                window.y as f32,
                window.width as f32,
                window.height as f32,
            ]
        };
        let Placement { scale, offset } = frame.placement;
        // Laid out as the shader's `Frame`, whose last two members align to 16 bytes.
        let [data, display] = [frame.data_window, frame.display_window].map(rect);
        let values: [f32; 16] = [
            offset[0],
            offset[1],
            scale,
            frame.scale_factor.round().max(1.0),
            config.width as f32,
            config.height as f32,
            frame.gain,
            0.0,
            data[0],
            data[1],
            data[2],
            data[3],
            display[0],
            display[1],
            display[2],
            display[3],
        ];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        queue.write_buffer(&self.frame_buffer, 0, &bytes);
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_bind_group(0, &self.frame_group, &[]);
        if let Some(image) = &self.image {
            pass.set_pipeline(&self.image_pipeline);
            pass.set_bind_group(1, &image.group, &[]);
            pass.draw(0..4, 0..1);
        }
        pass.set_pipeline(&self.outline_pipeline);
        pass.draw(0..4, 0..4);
    }
}

/// Production counts uncaptured errors instead of panicking, so a test that left one would
/// otherwise pass.
#[cfg(test)]
impl Drop for Gpu {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert_eq!(self.errors(), 0, "the GPU reported uncaptured errors");
        }
    }
}

/// A device that keeps buffers in memory and draws nothing, with the limits given or the most
/// permissive, and its queue, for tests that need wgpu objects but no GPU.
#[cfg(test)]
pub(crate) fn noop_device(limits: Option<wgpu::Limits>) -> (wgpu::Device, wgpu::Queue) {
    let (_, device, queue) = noop_parts(limits);
    (device, queue)
}

/// Two noop devices from one adapter, with their queues. Objects of devices from separate
/// instances can compare equal, since each instance numbers its objects from the same start.
#[cfg(test)]
pub(crate) fn noop_device_pair() -> [(wgpu::Device, wgpu::Queue); 2] {
    let (adapter, first, queue) = noop_parts(None);
    let descriptor = wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    };
    let second = pollster::block_on(adapter.request_device(&descriptor))
        .expect("the noop adapter has a second device");
    [(first, queue), second]
}

#[cfg(test)]
impl Gpu {
    /// A GPU whose instance has no backends, so that every surface fails to attach.
    pub(crate) fn without_backends() -> Gpu {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::empty();
        let mut gpu = Gpu::new(Waker::channel().0);
        gpu.instance = Some(wgpu::Instance::new(descriptor));
        gpu
    }

    /// A GPU with a noop device of generation 1 that draws no pane, and the viewer's native
    /// window, kept for the next device. Every later attempt fails: the noop backend makes
    /// surfaces it cannot draw to, so no adapter suits them.
    pub(crate) fn with_noop_device(waker: Waker) -> Gpu {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::NOOP;
        descriptor.backend_options.noop.enable = true;
        let mut gpu = Gpu::new(waker);
        gpu.instance = Some(wgpu::Instance::new(descriptor));
        gpu.device_generation = 1;
        let window = PaneWindow {
            window: NativeWindow::none(),
            size: PaneSize {
                width: 400,
                height: 300,
                scale_factor: 1.0,
            },
        };
        gpu.windows[Pane::Viewer.index()] = Some(window);
        let (adapter, device, queue) = noop_parts(None);
        let lost = ErrorCounter::install(&device, &gpu.errors, 1, &gpu.waker);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        gpu.device = Some(Device::for_format(adapter, device, queue, lost, format));
        gpu
    }

    /// The uncaptured errors counted so far, which a test that makes them on purpose takes so
    /// that dropping the GPU does not fail it.
    pub(crate) fn take_errors(&self) -> u64 {
        self.errors.swap(0, Ordering::Relaxed)
    }

    /// Makes an invalid buffer on the device, an uncaptured error.
    pub(crate) fn make_invalid_buffer(&self) {
        let device = &self.device.as_ref().expect("a device").device;
        let _buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("no usage"),
            size: 4,
            usage: wgpu::BufferUsages::empty(),
            mapped_at_creation: false,
        });
    }
}

#[cfg(test)]
impl NativeWindow {
    /// Handles to no window, for a GPU that never opens them.
    pub(crate) fn none() -> NativeWindow {
        use wgpu::rwh::{XlibDisplayHandle, XlibWindowHandle};
        NativeWindow {
            display: RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0)),
            window: RawWindowHandle::Xlib(XlibWindowHandle::new(0)),
        }
    }
}

/// The noop adapter, a device and its queue. A device whose queue has dropped cannot make a
/// command encoder.
#[cfg(test)]
pub(crate) fn noop_parts(
    limits: Option<wgpu::Limits>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::NOOP;
    descriptor.backend_options.noop = wgpu::NoopBackendOptions {
        enable: true,
        limits,
        ..Default::default()
    };
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("the noop backend has an adapter");
    let descriptor = wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
        .expect("the noop adapter has a device");
    (adapter, device, queue)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_stale_loss_notice_keeps_the_device_and_its_own_notice_retires_it() {
        let (waker, notices) = Waker::channel();
        let mut gpu = Gpu::with_noop_device(waker);

        assert!(!gpu.retire_lost_device(0));
        gpu.destroy_device();
        let Ok(Notice::DeviceLost { generation, .. }) = notices.try_recv() else {
            panic!("the lost callback sent no notice");
        };

        assert!(gpu.retire_lost_device(generation));
        assert!(gpu.device.is_none());
        assert!(!gpu.retire_lost_device(generation));
        assert_eq!(gpu.errors(), 0);
    }

    #[test]
    fn a_failed_attempt_keeps_no_device_but_the_window_and_the_next_is_a_new_generation() {
        let mut gpu = Gpu::with_noop_device(Waker::channel().0);
        assert!(gpu.retire_lost_device(1));

        for generation in [2, 3] {
            let error = gpu
                .make_device()
                .expect_err("no adapter can draw to the surface");
            assert!(error.contains("adapter"), "{error}");
            assert_eq!(gpu.device_generation, generation);
        }
        assert!(!gpu.has_device());
        assert!(gpu.has_window(Pane::Viewer));
        gpu.detach(Pane::Viewer);
        assert!(!gpu.has_window(Pane::Viewer));
        assert!(gpu.make_device().is_err());
    }

    #[test]
    fn an_image_staged_for_an_earlier_device_is_not_shown() {
        use std::sync::mpsc;

        use crate::evaluate::result_key;
        use crate::fixtures;
        use crate::session::Session;
        use crate::worker::{EvaluationDevice, Wanted, Worker};

        let mut gpu = Gpu::with_noop_device(Waker::channel().0);
        let device = gpu.device.as_ref().expect("a device").device.clone();
        let queue = gpu.device.as_ref().expect("a device").queue.clone();
        let path = fixtures::directory("staged").join("white.png");
        fixtures::write_png(&path, 1, 1, png::ColorType::Rgba, &[255; 4]);
        let mut session = Session::new();
        let changes = session.create_readers(vec![path], open_ng::Point::default(), None);
        let node = *changes.nodes.first().expect("one Reader");
        let key = result_key(session.document(), session.reloads(), node).expect("a key");
        let (delivered, deliveries) = mpsc::channel();
        let worker = Worker::start(EvaluationDevice::Cpu, move |evaluated| {
            delivered.send(evaluated).unwrap()
        });
        let staged = || {
            let evaluated = deliveries.recv_timeout(Duration::from_secs(10)).unwrap();
            let image = evaluated.result.unwrap();
            (image.pixels.expect("pixels staged"), image.info.data_window)
        };

        // Staged for the generation before the GPU's own, as after a loss.
        worker.connect(device.clone(), queue.clone(), Arc::default(), 0);
        worker.request(Some(Wanted {
            node,
            key,
            number: 1,
        }));
        let (pixels, window) = staged();
        gpu.show_image(pixels, window, 1);
        assert!(gpu.device.as_ref().unwrap().viewer.image.is_none());

        // Connecting delivers the last key again, now for the GPU's generation.
        worker.connect(device, queue, Arc::default(), 1);
        let (pixels, window) = staged();
        gpu.show_image(pixels, window, 1);
        assert!(gpu.device.as_ref().unwrap().viewer.image.is_some());
    }

    #[test]
    fn a_pixel_read_returns_each_planes_working_value_at_the_probe_of_its_image() {
        use crate::kernels::headless;

        let (device, queue) = headless::device();
        let viewer = ViewerRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
        // A 3 × 2 window on 4 × 3 planes, as the evaluator's may be larger than their windows,
        // with values a display copy could not hold at the probed texel, (2, 1).
        let special = [
            f32::from_bits(0x7fc0_1234),
            f32::NEG_INFINITY,
            3e38,
            f32::from_bits(1),
        ];
        let plane = |value: f32| {
            let texture = headless::plane(
                &device,
                4,
                3,
                wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let mut texels = [0.5f32; 12];
            texels[4 + 2] = value;
            let bytes: Vec<u8> = texels.iter().flat_map(|v| v.to_ne_bytes()).collect();
            queue.write_texture(
                texture.as_image_copy(),
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: None,
                },
                texture.size(),
            );
            Arc::new(texture)
        };
        let window = Window {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        };
        let image = ShownImage::new(&device, &viewer, special.map(plane), window, 7);
        let mut reader = PixelReader::new(&device);

        // Another image's probe, and texels outside this one's window though inside its planes,
        // read nothing.
        for probe in [[2, 1], [3, 0]].map(|texel| Probe { image: 8, texel }) {
            reader.start(&device, &queue, Some(&image), probe);
            assert!(reader.in_flight.is_none());
        }
        for outside in [[0, 2], [3, 1]].map(|texel| Probe { image: 7, texel }) {
            reader.start(&device, &queue, Some(&image), outside);
            assert!(reader.in_flight.is_none());
        }

        let probe = Probe {
            image: 7,
            texel: [2, 1],
        };
        reader.start(&device, &queue, Some(&image), probe);
        let start = Instant::now();
        let read = loop {
            let _ = device.poll(wgpu::PollType::Poll);
            if let Some(read) = reader.finish() {
                break read.expect("the buffer maps");
            }
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "the read finishes"
            );
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(read.probe, probe);
        assert_eq!(read.values.map(f32::to_bits), special.map(f32::to_bits));
        assert!(reader.in_flight.is_none());
    }

    #[test]
    fn the_viewers_shader_and_pipelines_validate() {
        let (device, _) = noop_device(None);
        ViewerRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}
