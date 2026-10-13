//! Two views of one graph, side by side in one window.
//!
//! The example is the host: it owns the graph and a small history, and decides what each
//! response from a view does. Moves snap to a 10-unit grid, the Write node is locked in place,
//! and Ctrl+Z and Ctrl+Shift+Z undo and redo moves and selection changes. Folding a group stays
//! out of the history, as Blender keeps panel toggles out of undo. Wires can be dragged between
//! ports, lifted off inputs and cut by dragging across them with Y held, but the host refuses
//! every connection edit for now.
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

use open_ng::{
    Color, Connection, ConnectionEdit, Direction, Fold, Graph, Group, GroupId, InputEvent, NodeId,
    NodeRecord, Pin, PinId, PinRef, Point, Renderer, Response, Row, Verdict, View, ViewBuffers,
    Viewport,
};

/// The physical pixels between the two views.
const GAP: u32 = 8;
const GRID: f32 = 10.0;
const LOCKED: NodeId = NodeId(5);

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
    #[cfg(target_arch = "wasm32")]
    web::main();
}

/// The document and the views onto it.
#[derive(Debug)]
struct Editor {
    graph: Graph,
    views: [View; 2],
    history: History,
}

impl Editor {
    fn new(width: u32, height: u32, scale_factor: f32) -> Editor {
        let [left, right] = viewports(width, height, scale_factor);
        Editor {
            graph: network(),
            views: [View::new(left), View::new(right)],
            history: History::default(),
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
            let response = self.views[index].handle_input(&self.graph, input, judge);
            handled |= response.handled;
            let mut changed = false;
            if let Some(fold) = response.fold {
                apply_fold(&mut self.graph, fold);
                changed = true;
            }
            changed |= self.history.apply(&mut self.graph, response);
            if changed {
                self.refresh();
            }
        }
        handled
    }

    /// Call it after every change to the graph, so that no view keeps showing what it found
    /// under a pointer that has not moved since.
    fn refresh(&mut self) {
        for view in &mut self.views {
            view.refresh(&self.graph, judge);
        }
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
        let changed = !self.gesture_active() && self.history.undo(&mut self.graph);
        if changed {
            self.refresh();
        }
        changed
    }

    fn redo(&mut self) -> bool {
        let changed = !self.gesture_active() && self.history.redo(&mut self.graph);
        if changed {
            self.refresh();
        }
        changed
    }

    fn gesture_active(&self) -> bool {
        self.views.iter().any(View::has_active_gesture)
    }
}

/// Whether the host would apply a connection edit. The history cannot record connection edits
/// yet, so it refuses them all.
fn judge(_: &ConnectionEdit) -> Verdict {
    Verdict::Refuse {
        reason: Some("Connection edits cannot be undone yet".to_owned()),
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
            self.renderer
                .prepare(&self.device, &self.queue, &editor.graph, view, buffers);
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

/// Two sources merged, blurred and written, from left to right around the graph origin where
/// new views are centred. Every node's input is pin 1 and its output pin 2, except the merge's
/// second input, pin 3. The blur has four more pins in two groups, one folded; its size both
/// receives and supplies a value.
fn network() -> Graph {
    let pin = |id, label: &str, direction| {
        Row::Pin(Pin {
            id: PinId(id),
            label: label.to_owned(),
            direction,
        })
    };
    let input = |label| pin(1, label, Direction::Input);
    let output = pin(2, "Image", Direction::Output);
    let node = |title: &str, x, y, (r, g, b), rows| NodeRecord {
        position: Point { x, y },
        title: title.to_owned(),
        color: Color { r, g, b },
        rows,
    };
    let source = (70, 120, 170);
    let mut graph = Graph::default();
    let nodes = [
        node("Read", -340.0, -90.0, source, vec![output.clone()]),
        node("Read", -340.0, 20.0, source, vec![output.clone()]),
        node(
            "Merge",
            -180.0,
            -40.0,
            (150, 110, 60),
            vec![
                output.clone(),
                input("Foreground"),
                pin(3, "Background", Direction::Input),
            ],
        ),
        node(
            "Directional Blur",
            -20.0,
            -40.0,
            (90, 140, 90),
            vec![
                input("Image"),
                output,
                Row::Group(Group {
                    id: GroupId(1),
                    label: "Options".to_owned(),
                    folded: false,
                    pins: vec![
                        Pin {
                            id: PinId(3),
                            label: "Size".to_owned(),
                            direction: Direction::Both,
                        },
                        Pin {
                            id: PinId(4),
                            label: "Mask".to_owned(),
                            direction: Direction::Input,
                        },
                    ],
                }),
                Row::Group(Group {
                    id: GroupId(2),
                    label: "Advanced".to_owned(),
                    folded: true,
                    pins: vec![
                        Pin {
                            id: PinId(5),
                            label: "Edge handling".to_owned(),
                            direction: Direction::Input,
                        },
                        Pin {
                            id: PinId(6),
                            label: "Premultiplied alpha".to_owned(),
                            direction: Direction::Input,
                        },
                    ],
                }),
            ],
        ),
        node("Write", 140.0, -40.0, (150, 70, 70), vec![input("Image")]),
    ];
    for (id, record) in (1..).map(NodeId).zip(nodes) {
        graph.add_node(id, record).expect("a new node ID");
    }
    let wires = [(1, 3, 1), (2, 3, 3), (3, 4, 1), (4, 5, 1)];
    for (from, to, input) in wires {
        let connection = Connection {
            from: PinRef {
                node: NodeId(from),
                pin: PinId(2),
            },
            to: PinRef {
                node: NodeId(to),
                pin: PinId(input),
            },
        };
        graph
            .add_connection(connection)
            .expect("an output and an input that exist");
    }
    graph
}

fn apply_fold(graph: &mut Graph, fold: Fold) {
    let mut record = graph
        .node(fold.node)
        .expect("views request only nodes that exist")
        .clone();
    for row in &mut record.rows {
        if let Row::Group(group) = row
            && group.id == fold.group
        {
            group.folded = fold.folded;
        }
    }
    graph
        .update_node(fold.node, record)
        .expect("the same pins and groups");
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
    fn apply(&mut self, graph: &mut Graph, response: Response) -> bool {
        let mut edit = Edit {
            before: State::default(),
            after: State::default(),
        };
        if let Some(selection) = response.selection {
            edit.before.selection = Some(graph.selection().to_vec());
            edit.after.selection = Some(selection);
        }
        for (id, position) in response.positions {
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
