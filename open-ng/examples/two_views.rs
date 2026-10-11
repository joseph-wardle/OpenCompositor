//! Two views of one graph, side by side in one window.
//!
//! The example is the host: it owns the graph and a small history, and decides what each
//! response from a view does. Moves snap to a 10-unit grid, the bottom node is locked in place,
//! and Ctrl+Z and Ctrl+Shift+Z undo and redo moves and selection changes.

use std::sync::Arc;

use open_ng::{
    Button, Color, Connection, Graph, InputEvent, InputRef, Key, Modifiers, NodeId, NodeRecord,
    OutputRef, PhysicalPoint, Point, Renderer, Response, View, ViewBuffers, Viewport,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::{Window, WindowId};

/// The physical pixels between the two views.
const GAP: u32 = 8;
const GRID: f32 = 10.0;
/// The logical pixels of a pixel scroll delta that count as one wheel notch.
const PIXELS_PER_LINE: f64 = 20.0;
const LOCKED: NodeId = NodeId(5);

fn main() {
    let event_loop = EventLoop::new().expect("an event loop");
    event_loop
        .run_app(&mut App::default())
        .expect("the event loop to run");
}

#[derive(Debug, Default)]
struct App {
    editor: Option<Editor>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.editor.is_none() {
            let attributes = Window::default_attributes().with_title("OpenNG: two views");
            let window = event_loop
                .create_window(attributes)
                .expect("a window to open");
            self.editor = Some(Editor::new(Arc::new(window)));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(editor) = &mut self.editor else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => editor.draw(),
            event => editor.handle(event),
        }
    }
}

#[derive(Debug)]
struct Editor {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    graph: Graph,
    views: [View; 2],
    buffers: [ViewBuffers; 2],
    history: History,
    pointer: PhysicalPoint,
    modifiers: Modifiers,
}

impl Editor {
    fn new(window: Arc<Window>) -> Editor {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("a surface for the window");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("a GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("a GPU device");
        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("a surface the adapter supports");
        surface.configure(&device, &config);

        let renderer = Renderer::new(&device, config.format, 1);
        let buffers = [
            ViewBuffers::new(&device, &renderer),
            ViewBuffers::new(&device, &renderer),
        ];
        let [left, right] = viewports(size.width, size.height, window.scale_factor() as f32);
        Editor {
            window,
            surface,
            device,
            queue,
            config,
            renderer,
            graph: network(),
            views: [View::new(left), View::new(right)],
            buffers,
            history: History::default(),
            pointer: PhysicalPoint::default(),
            modifiers: Modifiers::default(),
        }
    }

    fn handle(&mut self, event: WindowEvent) {
        let input = match event {
            WindowEvent::Resized(size) => {
                self.resize(size.width, size.height);
                return;
            }
            WindowEvent::Focused(false) => {
                for view in &mut self.views {
                    view.cancel_gesture();
                }
                self.window.request_redraw();
                return;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = Modifiers {
                    shift: modifiers.state().shift_key(),
                    ctrl: modifiers.state().control_key(),
                };
                return;
            }
            WindowEvent::KeyboardInput { event, .. } => match self.key(event) {
                Some(input) => input,
                None => return,
            },
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer = PhysicalPoint {
                    x: position.x as f32,
                    y: position.y as f32,
                };
                InputEvent::PointerMoved {
                    position: self.pointer,
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => Button::Left,
                    MouseButton::Middle => Button::Middle,
                    _ => return,
                };
                match state {
                    ElementState::Pressed => InputEvent::PointerPressed {
                        position: self.pointer,
                        button,
                        modifiers: self.modifiers,
                    },
                    ElementState::Released => InputEvent::PointerReleased {
                        position: self.pointer,
                        button,
                    },
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, lines) => lines,
                    MouseScrollDelta::PixelDelta(pixels) => {
                        (pixels.y / (PIXELS_PER_LINE * self.window.scale_factor())) as f32
                    }
                };
                InputEvent::Scrolled {
                    position: self.pointer,
                    lines,
                }
            }
            _ => return,
        };

        // Every view sees the event, and each response is applied before the next view sees
        // it, so the views never handle an event against a graph the host is about to change.
        let mut redraw = false;
        for view in &mut self.views {
            let response = view.handle_input(&self.graph, input);
            redraw |= response.handled;
            redraw |= self.history.record(&mut self.graph, response);
        }
        if redraw {
            self.window.request_redraw();
        }
    }

    /// Handles undo and redo, and translates the keys the views use.
    fn key(&mut self, event: KeyEvent) -> Option<InputEvent> {
        let pressed = event.state == ElementState::Pressed;
        let key = match event.logical_key {
            WinitKey::Named(NamedKey::Space) => Key::Space,
            WinitKey::Named(NamedKey::Escape) => Key::Escape,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("h") => Key::H,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("g") => Key::G,
            WinitKey::Character(c) if c.eq_ignore_ascii_case("z") => {
                if pressed && self.modifiers.ctrl {
                    let changed = if self.modifiers.shift {
                        self.history.redo(&mut self.graph)
                    } else {
                        self.history.undo(&mut self.graph)
                    };
                    if changed {
                        self.window.request_redraw();
                    }
                }
                return None;
            }
            _ => return None,
        };
        Some(if pressed {
            InputEvent::KeyPressed(key)
        } else {
            InputEvent::KeyReleased(key)
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        let scale_factor = self.window.scale_factor() as f32;
        for (view, viewport) in self
            .views
            .iter_mut()
            .zip(viewports(width, height, scale_factor))
        {
            view.set_viewport(viewport);
        }
        // A minimised window has no size, and a surface cannot.
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.window.request_redraw();
        }
    }

    fn draw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return;
            }
            _ => return,
        };
        for (view, buffers) in self.views.iter().zip(&mut self.buffers) {
            self.renderer
                .prepare(&self.device, &self.queue, &self.graph, view, buffers);
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
        self.window.pre_present_notify();
        self.queue.present(frame);
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

/// Two sources merged, blurred and written, laid out around the graph origin where new views
/// are centred.
fn network() -> Graph {
    let node = |x, y, (r, g, b), inputs, outputs| {
        NodeRecord::vertical(Point { x, y }, Color { r, g, b }, inputs, outputs)
    };
    let source = (70, 120, 170);
    let mut graph = Graph::default();
    let nodes = [
        node(-80.0, -160.0, source, 0, 1),
        node(80.0, -160.0, source, 0, 1),
        node(0.0, -60.0, (150, 110, 60), 2, 1),
        node(0.0, 40.0, (90, 140, 90), 1, 1),
        node(0.0, 140.0, (150, 70, 70), 1, 0),
    ];
    for (id, record) in (1..).map(NodeId).zip(nodes) {
        graph.add_node(id, record).expect("a new node ID");
    }
    let wires = [(1, 3, 0), (2, 3, 1), (3, 4, 0), (4, 5, 0)];
    for (from, to, input) in wires {
        let connection = Connection {
            from: OutputRef {
                node: NodeId(from),
                index: 0,
            },
            to: InputRef {
                node: NodeId(to),
                index: input,
            },
        };
        graph.add_connection(connection).expect("ports that exist");
    }
    graph
}

/// The host's undo history: each entry is one response as applied.
#[derive(Debug, Default)]
struct History {
    undo: Vec<Edit>,
    redo: Vec<Edit>,
}

#[derive(Debug)]
struct Edit {
    before: State,
    after: State,
}

/// The positions and selection an edit changes.
#[derive(Debug, Default)]
struct State {
    positions: Vec<(NodeId, Point)>,
    selection: Option<Vec<NodeId>>,
}

impl State {
    fn restore(&self, graph: &mut Graph) {
        graph
            .set_node_positions(&self.positions)
            .expect("nodes are never removed");
        if let Some(selection) = &self.selection {
            graph
                .set_selection(selection)
                .expect("nodes are never removed");
        }
    }
}

impl History {
    /// Applies what the host accepts of a response as one undo step. Returns whether the graph
    /// changed.
    fn record(&mut self, graph: &mut Graph, response: Response) -> bool {
        let mut edit = Edit {
            before: State::default(),
            after: State::default(),
        };
        for (id, position) in response.move_nodes.unwrap_or_default() {
            let current = graph
                .node(id)
                .expect("views request only nodes that exist")
                .position;
            let snapped = Point {
                x: (position.x / GRID).round() * GRID,
                y: (position.y / GRID).round() * GRID,
            };
            if id != LOCKED && snapped != current {
                edit.before.positions.push((id, current));
                edit.after.positions.push((id, snapped));
            }
        }
        if let Some(change) = response.selection {
            edit.before.selection = Some(change.before);
            edit.after.selection = Some(change.after);
        }
        if edit.after.positions.is_empty() && edit.after.selection.is_none() {
            return false;
        }
        edit.after.restore(graph);
        self.undo.push(edit);
        self.redo.clear();
        true
    }

    fn undo(&mut self, graph: &mut Graph) -> bool {
        let Some(edit) = self.undo.pop() else {
            return false;
        };
        edit.before.restore(graph);
        self.redo.push(edit);
        true
    }

    fn redo(&mut self, graph: &mut Graph) -> bool {
        let Some(edit) = self.redo.pop() else {
            return false;
        };
        edit.after.restore(graph);
        self.undo.push(edit);
        true
    }
}
