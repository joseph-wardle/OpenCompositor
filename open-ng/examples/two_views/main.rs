//! Two views of one graph, side by side in one window.
//!
//! The example is the host: it owns the document, decides what each response from a view does
//! and keeps the undo history. Wires can be dragged between ports, lifted off inputs and cut by
//! dragging across them with Y held; Ctrl+Z and Ctrl+Shift+Z undo and redo. The rules the host
//! applies are in the document module; L toggles its lock on the Write node, which shows the
//! views judging an edit again when the host's rules change in the middle of it.
//!
//! This file is what every platform shares; each platform's module starts the GPU, translates its
//! events and schedules redraws.
//!
//! Run it natively with `cargo run -p open-ng --example two_views`. For the browser, install
//! [Trunk](https://trunkrs.dev) and run `trunk serve --example two_views
//! examples/two_views/index.html` from the `open-ng` directory, then open the address it prints.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

mod document;

#[cfg(not(target_arch = "wasm32"))]
use native::report;
use open_ng::{InputEvent, Renderer, View, ViewBuffers, Viewport};
#[cfg(target_arch = "wasm32")]
use web::report;

use crate::document::Document;

/// The physical pixels between the two views.
const GAP: u32 = 8;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
    #[cfg(target_arch = "wasm32")]
    web::main();
}

/// The document and the views onto it.
#[derive(Debug)]
struct Editor {
    document: Document,
    views: [View; 2],
}

impl Editor {
    fn new(width: u32, height: u32, scale_factor: f32) -> Editor {
        let [left, right] = viewports(width, height, scale_factor);
        Editor {
            document: Document::new(),
            views: [View::new(left), View::new(right)],
        }
    }

    fn resize(&mut self, width: u32, height: u32, scale_factor: f32) {
        for (view, viewport) in self
            .views
            .iter_mut()
            .zip(viewports(width, height, scale_factor))
        {
            view.set_viewport(viewport);
        }
    }

    /// Sends the event to every view, applying each response before the next view sees the
    /// event, so the views never handle it against a graph the host is about to change. Returns
    /// whether a view used it: the host should redraw, and not also treat the event as its own.
    fn handle(&mut self, input: InputEvent) -> bool {
        let mut handled = false;
        for index in 0..self.views.len() {
            let document = &self.document;
            let judge = |edit: &_| document.judge(edit);
            let response = self.views[index].handle_input(document.graph(), input, judge);
            handled |= response.handled;
            match self.document.apply(response) {
                Ok(true) => self.refresh(),
                Ok(false) => {}
                // The views were refreshed after every change, so this means a bug in the host.
                Err(reason) => report(&format!("Refused a connection edit: {reason}")),
            }
        }
        handled
    }

    /// Call it after every change to the graph or to the host's rules, so that no view keeps
    /// showing what it found under a pointer that has not moved since, or a verdict the host
    /// would no longer give.
    fn refresh(&mut self) {
        for view in &mut self.views {
            let document = &self.document;
            view.refresh(document.graph(), |edit| document.judge(edit));
        }
    }

    fn toggle_write_lock(&mut self) {
        self.document.toggle_write_lock();
        self.refresh();
    }

    /// Call it when the window loses focus.
    fn cancel_gestures(&mut self) {
        for view in &mut self.views {
            view.cancel_gesture();
        }
    }

    /// Undo and redo wait for a gesture to finish, as the views' own keys do. Each returns
    /// whether the graph changed.
    fn undo(&mut self) -> bool {
        let changed = !self.gesture_active() && self.document.undo();
        if changed {
            self.refresh();
        }
        changed
    }

    fn redo(&mut self) -> bool {
        let changed = !self.gesture_active() && self.document.redo();
        if changed {
            self.refresh();
        }
        changed
    }

    fn gesture_active(&self) -> bool {
        self.views.iter().any(View::has_active_gesture)
    }
}

/// The surface the views draw into, and everything on the GPU that draws them.
#[derive(Debug)]
struct Gpu {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    buffers: [ViewBuffers; 2],
}

impl Gpu {
    async fn new(
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Gpu, String> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|error| format!("No GPU adapter: {error}"))?;
        // The renderer needs no more than WebGL2 offers, and asking for no more than that keeps
        // the native example honest about it.
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(&descriptor)
            .await
            .map_err(|error| format!("No GPU device: {error}"))?;
        let mut config = surface
            .get_default_config(&adapter, width.max(1), height.max(1))
            .ok_or("The GPU adapter cannot draw to this surface.")?;
        let max = device.limits().max_texture_dimension_2d;
        config.width = config.width.min(max);
        config.height = config.height.min(max);
        surface.configure(&device, &config);

        let renderer = Renderer::new(&device, config.format, 1);
        let buffers = [
            ViewBuffers::new(&device, &renderer),
            ViewBuffers::new(&device, &renderer),
        ];
        Ok(Gpu {
            surface,
            device,
            queue,
            config,
            renderer,
            buffers,
        })
    }

    /// Fits the surface to a new size, clamped to the largest texture the device supports, and
    /// returns the size the views should use. A zero size leaves the surface as it was, because
    /// a surface cannot have one.
    fn resize(&mut self, width: u32, height: u32) -> (u32, u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        let (width, height) = (width.min(max), height.min(max));
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
        (width, height)
    }

    /// Draws the views into the surface's next texture and returns it for the host to present,
    /// or `None` if the surface has none to give this frame.
    fn draw(&mut self, editor: &Editor) -> Option<wgpu::SurfaceTexture> {
        let frame = self.next_texture()?;
        for (view, buffers) in editor.views.iter().zip(&mut self.buffers) {
            self.renderer.prepare(
                &self.device,
                &self.queue,
                editor.document.graph(),
                view,
                buffers,
            );
        }

        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("two views"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            for buffers in &self.buffers {
                self.renderer.draw(&mut pass, buffers);
            }
        }
        self.queue.submit([encoder.finish()]);
        Some(frame)
    }

    /// Reconfigures the surface once if it is out of date.
    fn next_texture(&mut self) -> Option<wgpu::SurfaceTexture> {
        for _ in 0..2 {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(frame)
                | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => return Some(frame),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(&self.device, &self.config);
                }
                _ => return None,
            }
        }
        None
    }
}

/// The left and right halves of the window, separated by the gap. The right view is offset from
/// the window's corner, which tests that input and drawing both account for the offset.
fn viewports(width: u32, height: u32, scale_factor: f32) -> [Viewport; 2] {
    let left_width = width.saturating_sub(GAP) / 2;
    let right_x = (left_width + GAP).min(width);
    [
        Viewport {
            x: 0,
            y: 0,
            width: left_width,
            height,
            scale_factor,
        },
        Viewport {
            x: right_x,
            y: 0,
            width: width - right_x,
            height,
            scale_factor,
        },
    ]
}
