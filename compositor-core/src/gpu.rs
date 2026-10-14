//! The GPU: one device for every pane, and a surface for each pane while it has a native window.

use open_ng::{Graph, Renderer, View, ViewBuffers, Viewport};
use wgpu::rwh::{RawDisplayHandle, RawWindowHandle};

use crate::app::Pane;

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
    /// The handles must stay valid until the pane's surface is dropped, which happens when the
    /// shell dispatches [`Event::SurfaceDestroyed`](crate::Event::SurfaceDestroyed) for the pane.
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
}

/// Everything made once there is a surface to choose an adapter for.
#[derive(Debug)]
struct Device {
    graph_buffers: ViewBuffers,
    renderer: Renderer,
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
        }
    }

    pub(crate) fn has_surface(&self, pane: Pane) -> bool {
        self.surfaces[pane.index()].is_some()
    }

    /// Makes the pane's surface, and the device if this is the first. On failure the pane draws
    /// nothing; the error says why.
    pub(crate) fn attach(
        &mut self,
        pane: Pane,
        window: NativeWindow,
        size: PaneSize,
    ) -> Result<(), String> {
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
        let device = match &mut self.device {
            Some(device) => device,
            None => self.device.insert(Device::new(instance, &surface)?),
        };
        if !device.adapter.is_surface_supported(&surface) {
            return Err(format!(
                "the adapter cannot draw to the {pane:?} pane's surface"
            ));
        }
        let mut config = surface
            .get_default_config(&device.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| format!("the {pane:?} pane's surface has no configuration"))?;
        config.format = device.format;
        surface.configure(&device.device, &config);
        device.configs[pane.index()] = Some(config);
        self.surfaces[pane.index()] = Some(surface);
        Ok(())
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

    /// Draws one frame of the pane. With Fifo presentation this can wait for the display.
    ///
    /// Returns whether to draw again: the surface was out of date and has been reconfigured.
    pub(crate) fn draw(&mut self, pane: Pane, graph: &Graph, view: &View) -> bool {
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
            // The pane is hidden or the display is busy; it is drawn again when exposed.
            _ => return false,
        };
        if pane == Pane::Graph {
            device.renderer.prepare(
                &device.device,
                &device.queue,
                graph,
                view,
                &mut device.graph_buffers,
            );
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
            if pane == Pane::Graph {
                device.renderer.draw(&mut pass, &device.graph_buffers);
            }
        }
        device.queue.submit([encoder.finish()]);
        device.queue.present(frame);
        false
    }
}

impl Device {
    fn new(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Result<Device, String> {
        // WGPU_ADAPTER_NAME and WGPU_POWER_PREF choose the adapter.
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            instance,
            Some(surface),
        ))
        .map_err(|error| format!("no GPU adapter: {error}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .map_err(|error| format!("no GPU device: {error}"))?;
        // Panes draw clamped linear values, which an sRGB format encodes. Encoding in a shader too
        // would encode twice.
        let formats = surface.get_capabilities(&adapter).formats;
        let format = formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .ok_or("the surface offers no sRGB format")?;
        let renderer = Renderer::new(&device, format, 1);
        let graph_buffers = ViewBuffers::new(&device, &renderer);
        Ok(Device {
            graph_buffers,
            renderer,
            configs: Default::default(),
            format,
            queue,
            device,
            adapter,
        })
    }
}
