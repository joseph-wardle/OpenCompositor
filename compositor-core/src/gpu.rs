//! The GPU: one device for every pane, and a surface for each pane while it has a native window.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use open_ng::{Graph, Renderer, View, ViewBuffers, Viewport};
use wgpu::rwh::{RawDisplayHandle, RawWindowHandle};

use crate::app::Pane;
use crate::image::Window;
use crate::loader::Staged;
use crate::viewer::Placement;

/// Behind everything a pane draws.
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};

/// A pane's native window, from which its surface is made.
#[derive(Debug)]
pub struct NativeWindow {
    display: RawDisplayHandle,
    window: RawWindowHandle,
}

impl NativeWindow {
    /// # Safety
    ///
    /// The handles must stay valid until the pane's surface is dropped: when the shell dispatches
    /// [`Event::SurfaceDestroyed`](crate::Event::SurfaceDestroyed) for the pane, or the app drops.
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
    // Fields drop in order: the surfaces go before the device that draws to them.
    surfaces: [Option<wgpu::Surface<'static>>; Pane::COUNT],
    device: Option<Device>,
    /// Made with the first surface, so that headless use needs no GPU backend.
    instance: Option<wgpu::Instance>,
    /// Only in the lifecycle check.
    checks: Option<FrameChecks>,
}

/// What the frames a pane drew prove, for the shell's lifecycle check: that the GPU finished
/// them, and what the last one put at the pane's centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameCheck {
    /// How many of the pane's frames the GPU has finished.
    pub finished: u64,
    /// The centre pixel of the pane's last finished frame, as sRGB-encoded RGBA, if its surface
    /// can be read back.
    pub centre: Option<[u8; 4]>,
}

#[derive(Debug, Default)]
struct FrameChecks {
    /// Counted from the queue, which can report finished work on the loader's thread.
    finished: [Arc<AtomicU64>; Pane::COUNT],
    centres: [Option<[u8; 4]>; Pane::COUNT],
}

/// What the viewer pane draws: the shown image's windows, and where they appear.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewerFrame {
    pub(crate) placement: Placement,
    pub(crate) display_window: Window,
    pub(crate) data_window: Window,
    /// The pane's physical pixels per logical pixel, which sets the outline's width.
    pub(crate) scale_factor: f32,
}

/// Everything made once there is a surface to choose an adapter for.
#[derive(Debug)]
struct Device {
    graph_buffers: ViewBuffers,
    renderer: Renderer,
    viewer: ViewerRenderer,
    configs: [Option<wgpu::SurfaceConfiguration>; Pane::COUNT],
    format: wgpu::TextureFormat,
    queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
}

impl Gpu {
    pub(crate) fn new() -> Gpu {
        Gpu {
            surfaces: Default::default(),
            device: None,
            instance: None,
            checks: None,
        }
    }

    /// Records what every frame from now on proves, waiting for the GPU after each. Surfaces
    /// attached from now on are made readable where they allow it.
    pub(crate) fn check_frames(&mut self) {
        self.checks.get_or_insert_default();
    }

    pub(crate) fn frame_check(&self, pane: Pane) -> Option<FrameCheck> {
        let checks = self.checks.as_ref()?;
        Some(FrameCheck {
            finished: checks.finished[pane.index()].load(Ordering::Relaxed),
            centre: checks.centres[pane.index()],
        })
    }

    pub(crate) fn has_surface(&self, pane: Pane) -> bool {
        self.surfaces[pane.index()].is_some()
    }

    pub(crate) fn has_device(&self) -> bool {
        self.device.is_some()
    }

    /// Makes the pane's surface, and the device if there is none yet, which it returns so that
    /// the loader can stage images for it. On failure the pane draws nothing and no new device is
    /// kept; the error says why.
    pub(crate) fn attach(
        &mut self,
        pane: Pane,
        window: NativeWindow,
        size: PaneSize,
    ) -> Result<Option<wgpu::Device>, String> {
        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            raw_display_handle: Some(window.display),
            raw_window_handle: window.window,
        };
        let instance = self.instance.get_or_insert_with(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
            // The shell is tested on Vulkan only; WGPU_BACKEND chooses another.
            if std::env::var_os("WGPU_BACKEND").is_none() {
                descriptor.backends = wgpu::Backends::VULKAN;
            }
            wgpu::Instance::new(descriptor)
        });
        // SAFETY: `NativeWindow::new`'s contract keeps the handles valid until `detach`.
        let surface = unsafe { instance.create_surface_unsafe(target) }
            .map_err(|error| format!("no surface for the {pane:?} pane: {error}"))?;
        let new_device = match &self.device {
            Some(_) => None,
            None => Some(Device::new(instance, &surface)?),
        };
        let device = new_device
            .as_ref()
            .or(self.device.as_ref())
            .expect("one exists");
        let config = configure(device, pane, &surface, size, self.checks.is_some())?;
        let created = new_device.map(|new| self.device.insert(new).device.clone());
        let device = self.device.as_mut().expect("kept above if new");
        device.configs[pane.index()] = Some(config);
        self.surfaces[pane.index()] = Some(surface);
        Ok(created)
    }

    /// Copies a staged image into its texture and shows it in the viewer from the next frame.
    ///
    /// The copy is submitted at once rather than with the viewer's next frame, which may never
    /// come while the viewer is closed, so that the staging buffer goes back to the loader.
    pub(crate) fn show_image(&mut self, staged: Staged) {
        let Some(device) = &mut self.device else {
            return;
        };
        let mut encoder = device.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: staged.buffer(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(staged.bytes_per_row),
                    rows_per_image: None,
                },
            },
            staged.texture.as_image_copy(),
            staged.texture.size(),
        );
        device.queue.submit([encoder.finish()]);
        let view = staged.texture.create_view(&Default::default());
        device.viewer.image = Some(device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer image"),
            layout: &device.viewer.image_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        }));
    }

    /// Stops drawing the viewer's image, which a failed read or an unmarked slot replaces.
    pub(crate) fn clear_image(&mut self) {
        if let Some(device) = &mut self.device {
            device.viewer.image = None;
        }
    }

    /// Drops the pane's surface, if it has one, before its native window goes.
    pub(crate) fn detach(&mut self, pane: Pane) {
        self.surfaces[pane.index()] = None;
        if let Some(device) = &mut self.device {
            device.configs[pane.index()] = None;
        }
    }

    pub(crate) fn resize(&mut self, pane: Pane, size: PaneSize) {
        let (Some(surface), Some(device)) = (&self.surfaces[pane.index()], &mut self.device) else {
            return;
        };
        let Some(config) = &mut device.configs[pane.index()] else {
            return;
        };
        config.width = size.width.max(1);
        config.height = size.height.max(1);
        surface.configure(&device.device, config);
    }

    /// Draws one frame of the pane: the graph, or the viewer's frame if it shows an image.
    /// This can wait for the display while the pane has frames it has not shown.
    ///
    /// Returns whether to draw again: the surface was out of date and has been reconfigured, or
    /// the display took too long to free a frame, which would otherwise lose this draw.
    pub(crate) fn draw(
        &mut self,
        pane: Pane,
        graph: &Graph,
        view: &View,
        viewer_frame: Option<ViewerFrame>,
    ) -> bool {
        let (Some(surface), Some(device)) = (&self.surfaces[pane.index()], &mut self.device) else {
            return false;
        };
        let config = device.configs[pane.index()]
            .as_ref()
            .expect("a surface has a configuration");
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                surface.configure(&device.device, config);
                return true;
            }
            wgpu::CurrentSurfaceTexture::Timeout => return true,
            // The pane is hidden; it is drawn again when exposed.
            wgpu::CurrentSurfaceTexture::Occluded => return false,
            // Only under an error scope or handler, which this code does not install.
            wgpu::CurrentSurfaceTexture::Validation => return false,
        };
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
        if let Some(checks) = &mut self.checks {
            checks.record(pane, device, readback);
        }
        device.queue.present(frame);
        false
    }
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
        let wait = wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        };
        if device.device.poll(wait).is_err() {
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

/// Configures a pane's surface for the device.
fn configure(
    device: &Device,
    pane: Pane,
    surface: &wgpu::Surface,
    size: PaneSize,
    readable: bool,
) -> Result<wgpu::SurfaceConfiguration, String> {
    if !device.adapter.is_surface_supported(surface) {
        return Err(format!(
            "the adapter cannot draw to the {pane:?} pane's surface"
        ));
    }
    let mut config = surface
        .get_default_config(&device.adapter, size.width.max(1), size.height.max(1))
        .ok_or_else(|| format!("the {pane:?} pane's surface has no configuration"))?;
    config.format = device.format;
    // Mailbox replaces a waiting frame instead of blocking for the display, so a draw waits
    // only when the pane redraws faster than its screen refreshes, which the shell prevents.
    // Fifo, which every surface supports, is the fallback. The default is the driver's first
    // mode, which may be Immediate, which tears.
    let capabilities = surface.get_capabilities(&device.adapter);
    if readable && capabilities.usages.contains(wgpu::TextureUsages::COPY_SRC) {
        config.usage |= wgpu::TextureUsages::COPY_SRC;
    }
    let modes = capabilities.present_modes;
    config.present_mode = if modes.contains(&wgpu::PresentMode::Mailbox) {
        wgpu::PresentMode::Mailbox
    } else {
        wgpu::PresentMode::Fifo
    };
    surface.configure(&device.device, &config);
    Ok(config)
}

impl Device {
    fn new(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Result<Device, String> {
        // WGPU_ADAPTER_NAME and WGPU_POWER_PREF choose the adapter.
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            instance,
            Some(surface),
        ))
        .map_err(|error| format!("no GPU adapter: {error}"))?;
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
        // Panes draw clamped linear values, which an sRGB format encodes. Encoding in a shader too
        // would encode twice.
        let formats = surface.get_capabilities(&adapter).formats;
        let format = formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .ok_or("the surface offers no sRGB format")?;
        Ok(Device::for_format(adapter, device, queue, format))
    }

    /// The renderers for panes of the format.
    fn for_format(
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Device {
        let renderer = Renderer::new(&device, format, 1);
        let graph_buffers = ViewBuffers::new(&device, &renderer);
        let viewer = ViewerRenderer::new(&device, format);
        Device {
            graph_buffers,
            renderer,
            viewer,
            configs: Default::default(),
            format,
            queue,
            device,
            adapter,
        }
    }
}

/// Draws the viewer's image and the outline of its display window.
#[derive(Debug)]
struct ViewerRenderer {
    image_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    frame_buffer: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    image_layout: wgpu::BindGroupLayout,
    /// The shown image's texture, while the viewer has one.
    image: Option<wgpu::BindGroup>,
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
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(FRAME_BYTES),
                },
                count: None,
            }],
        });
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewer image"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
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
            0.0,
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
            pass.set_bind_group(1, image, &[]);
            pass.draw(0..4, 0..1);
        }
        pass.set_pipeline(&self.outline_pipeline);
        pass.draw(0..4, 0..4);
    }
}

/// A device that keeps buffers in memory and draws nothing, with the limits given or the most
/// permissive, for tests that need wgpu objects but no GPU.
#[cfg(test)]
pub(crate) fn noop_device(limits: Option<wgpu::Limits>) -> wgpu::Device {
    noop_parts(limits).1
}

#[cfg(test)]
impl Gpu {
    /// A GPU with a [`noop_device`] and no surfaces, and the device for the loader.
    pub(crate) fn noop() -> (Gpu, wgpu::Device) {
        let (adapter, device, queue) = noop_parts(None);
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let connected = device.clone();
        let gpu = Gpu {
            surfaces: Default::default(),
            device: Some(Device::for_format(adapter, device, queue, format)),
            instance: None,
            checks: None,
        };
        (gpu, connected)
    }

    /// A GPU whose instance has no backends, so that every surface fails to attach.
    pub(crate) fn without_backends() -> Gpu {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::empty();
        Gpu {
            instance: Some(wgpu::Instance::new(descriptor)),
            ..Gpu::new()
        }
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

#[cfg(test)]
fn noop_parts(limits: Option<wgpu::Limits>) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
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
    use super::*;

    #[test]
    fn the_viewers_shader_and_pipelines_validate() {
        let device = noop_device(None);
        ViewerRenderer::new(&device, wgpu::TextureFormat::Bgra8UnormSrgb);
    }
}
