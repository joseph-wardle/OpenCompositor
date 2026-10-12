//! Drawing views with wgpu into a render pass the host owns.

use std::collections::HashSet;
use std::mem::size_of;

use bytemuck::{Pod, Zeroable};

use crate::geometry::{self, NodeLayout, Rect};
use crate::graph::{Color, Graph, Point};
use crate::view::{View, Viewport};

const WIRE_SEGMENTS: u32 = 32;
const NODE_CORNER_RADIUS: f32 = 4.0;
const NODE_BORDER: [u8; 4] = [20, 20, 20, 255];
const NODE_BORDER_WIDTH: f32 = 1.0;
const SELECTED_BORDER: [u8; 4] = [255, 200, 40, 255];
const SELECTED_BORDER_WIDTH: f32 = 2.5;
const PORT_RADIUS: f32 = 4.0;
const PORT_FILL: [u8; 4] = [210, 210, 210, 255];
const BOX_FILL: [u8; 4] = [255, 255, 255, 12];
const BOX_BORDER: [u8; 4] = [220, 220, 220, 255];

/// Draws views into render passes on one device whose single color attachment has the format
/// and sample count given at creation, with no depth-stencil attachment.
#[derive(Debug)]
pub struct Renderer {
    transform_layout: wgpu::BindGroupLayout,
    background: wgpu::RenderPipeline,
    wires: wgpu::RenderPipeline,
    shapes: wgpu::RenderPipeline,
}

/// A view's data on the GPU. The host keeps one beside each [`View`].
#[derive(Debug)]
pub struct ViewBuffers {
    viewport: Viewport,
    transform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    wires: InstanceBuffer,
    shapes: InstanceBuffer,
}

#[derive(Debug)]
struct InstanceBuffer {
    label: &'static str,
    buffer: wgpu::Buffer,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ViewTransform {
    centre: [f32; 2],
    clip_per_unit: [f32; 2],
    pixels_per_unit: f32,
    padding: [f32; 3],
}

/// A cubic Bézier wire's control points, in graph units.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Wire {
    points: [[f32; 2]; 4],
}

/// A rounded rectangle in graph units, with sRGB-encoded colors.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Shape {
    centre: [f32; 2],
    half_size: [f32; 2],
    radius: f32,
    border_width: f32,
    fill: [u8; 4],
    border: [u8; 4],
}

impl Renderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Renderer {
        let module = device.create_shader_module(wgpu::include_wgsl!("render.wgsl"));
        let transform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("open-ng view transform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("open-ng"),
            bind_group_layouts: &[Some(&transform_layout)],
            immediate_size: 0,
        });
        let constants = [
            ("srgb_target", f64::from(u8::from(format.is_srgb()))),
            ("wire_segments", f64::from(WIRE_SEGMENTS)),
        ];
        let pipeline = |name: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        topology: wgpu::PrimitiveTopology,
                        blend: Option<wgpu::BlendState>| {
            let compilation_options = wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("open-ng {name}")),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(&format!("{name}_vertex")),
                    compilation_options: compilation_options.clone(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(&format!("{name}_fragment")),
                    compilation_options,
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let wire_attributes = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x2,
        ];
        let shape_attributes = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32, 4 => Unorm8x4, 5 => Unorm8x4,
        ];
        let wire_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Wire>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wire_attributes,
        };
        let shape_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Shape>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &shape_attributes,
        };
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let strip = wgpu::PrimitiveTopology::TriangleStrip;
        Renderer {
            background: pipeline(
                "background",
                &[],
                wgpu::PrimitiveTopology::TriangleList,
                None,
            ),
            wires: pipeline("wire", &[Some(wire_layout)], strip, blend),
            shapes: pipeline("shape", &[Some(shape_layout)], strip, blend),
            transform_layout,
        }
    }

    /// Uploads the view's current state. Call it for every view before encoding the pass that
    /// draws them, because the uploads take effect when the queue is next submitted.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        graph: &Graph,
        view: &View,
        buffers: &mut ViewBuffers,
    ) {
        let viewport = view.viewport();
        buffers.viewport = viewport;
        if viewport.width == 0 || viewport.height == 0 {
            return;
        }
        let pixels_per_unit = view.physical_per_unit();
        let centre = view.centre();
        let transform = ViewTransform {
            centre: [centre.x, centre.y],
            clip_per_unit: [
                2.0 * pixels_per_unit / viewport.width as f32,
                2.0 * pixels_per_unit / viewport.height as f32,
            ],
            pixels_per_unit,
            padding: [0.0; 3],
        };
        queue.write_buffer(&buffers.transform, 0, bytemuck::bytes_of(&transform));
        buffers.wires.write(device, queue, &wires(graph, view));
        buffers.shapes.write(device, queue, &shapes(graph, view));
    }

    /// Draws the view as last prepared, filling its viewport. Leaves the pass's pipeline, bind
    /// groups, vertex buffers, viewport and scissor changed.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, buffers: &ViewBuffers) {
        let Viewport {
            x,
            y,
            width,
            height,
            ..
        } = buffers.viewport;
        if width == 0 || height == 0 {
            return;
        }
        pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, width, height);
        pass.set_bind_group(0, &buffers.bind_group, &[]);
        pass.set_pipeline(&self.background);
        pass.draw(0..3, 0..1);
        if buffers.wires.count > 0 {
            pass.set_pipeline(&self.wires);
            pass.set_vertex_buffer(0, buffers.wires.buffer.slice(..));
            pass.draw(0..2 * (WIRE_SEGMENTS + 1), 0..buffers.wires.count);
        }
        if buffers.shapes.count > 0 {
            pass.set_pipeline(&self.shapes);
            pass.set_vertex_buffer(0, buffers.shapes.buffer.slice(..));
            pass.draw(0..4, 0..buffers.shapes.count);
        }
    }
}

impl ViewBuffers {
    /// Buffers that draw nothing until the view is first prepared.
    pub fn new(device: &wgpu::Device, renderer: &Renderer) -> ViewBuffers {
        let transform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("open-ng view transform"),
            size: size_of::<ViewTransform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("open-ng view transform"),
            layout: &renderer.transform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: transform.as_entire_binding(),
            }],
        });
        ViewBuffers {
            viewport: Viewport {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
                scale_factor: 1.0,
            },
            transform,
            bind_group,
            wires: InstanceBuffer::new(device, "open-ng wires", 0),
            shapes: InstanceBuffer::new(device, "open-ng shapes", 0),
        }
    }
}

impl InstanceBuffer {
    fn new(device: &wgpu::Device, label: &'static str, size: u64) -> InstanceBuffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        InstanceBuffer {
            label,
            buffer,
            count: 0,
        }
    }

    /// Replaces the instances, growing the buffer to the next power of two if they do not fit.
    fn write<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(instances);
        if bytes.len() as u64 > self.buffer.size() {
            let size = (bytes.len() as u64).next_power_of_two();
            *self = InstanceBuffer::new(device, self.label, size);
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        self.count = instances.len() as u32;
    }
}

/// The wires as drawn, following any nodes being dragged.
fn wires(graph: &Graph, view: &View) -> Vec<Wire> {
    let drawn = |id| {
        view.drawn_position(graph, id)
            .expect("connections join nodes in the graph")
    };
    graph
        .connections()
        .iter()
        .map(|connection| {
            let (from, to) = (connection.from, connection.to);
            let output = graph
                .node(from.node)
                .expect("connections join nodes in the graph");
            let input = graph
                .node(to.node)
                .expect("connections join nodes in the graph");
            let points = geometry::wire(
                NodeLayout::new(output, drawn(from.node)).output(from.index),
                output.outputs[from.index].side,
                NodeLayout::new(input, drawn(to.node)).input(to.index),
                input.inputs[to.index].side,
            );
            Wire {
                points: points.map(|point| [point.x, point.y]),
            }
        })
        .collect()
}

/// Each node followed by its ports, in draw order, then any selection box.
fn shapes(graph: &Graph, view: &View) -> Vec<Shape> {
    let selected: HashSet<_> = view.drawn_selection(graph).into_iter().collect();
    let mut shapes = Vec::new();
    for (id, record) in graph.nodes() {
        let centre = view
            .drawn_position(graph, id)
            .expect("the node is in the graph");
        let (border, border_width) = if selected.contains(&id) {
            (SELECTED_BORDER, SELECTED_BORDER_WIDTH)
        } else {
            (NODE_BORDER, NODE_BORDER_WIDTH)
        };
        let layout = NodeLayout::new(record, centre);
        shapes.push(Shape {
            radius: NODE_CORNER_RADIUS,
            border_width,
            fill: opaque(record.color),
            border,
            ..rectangle(layout.rect)
        });
        shapes.extend(layout.ports().map(|port| Shape {
            centre: [port.x, port.y],
            half_size: [PORT_RADIUS; 2],
            radius: PORT_RADIUS,
            border_width: NODE_BORDER_WIDTH,
            fill: PORT_FILL,
            border: NODE_BORDER,
        }));
    }
    if let Some(rect) = view.drawn_box() {
        shapes.push(Shape {
            border_width: 1.0 / view.physical_per_unit(),
            fill: BOX_FILL,
            border: BOX_BORDER,
            ..rectangle(rect)
        });
    }
    shapes
}

fn rectangle(rect: Rect) -> Shape {
    let half_size = Point {
        x: (rect.max.x - rect.min.x) / 2.0,
        y: (rect.max.y - rect.min.y) / 2.0,
    };
    Shape {
        centre: [rect.min.x + half_size.x, rect.min.y + half_size.y],
        half_size: [half_size.x, half_size.y],
        radius: 0.0,
        border_width: 0.0,
        fill: [0; 4],
        border: [0; 4],
    }
}

fn opaque(color: Color) -> [u8; 4] {
    [color.r, color.g, color.b, 255]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Connection, InputRef, NodeId, NodeRecord, OutputRef};
    use crate::input::{Button, InputEvent, Modifiers, PhysicalPoint};

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    fn at(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    #[test]
    fn the_shaders_are_valid() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("render.wgsl")).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn a_wire_end_follows_its_port_during_a_drag() {
        let mut graph = Graph::default();
        for (id, y) in [(1, 0.0), (2, 100.0)] {
            let record = NodeRecord::vertical(at(0.0, y), GREY, 1, 1);
            graph.add_node(NodeId(id), record).unwrap();
        }
        let output = OutputRef {
            node: NodeId(1),
            index: 0,
        };
        let input = InputRef {
            node: NodeId(2),
            index: 0,
        };
        graph
            .add_connection(Connection {
                from: output,
                to: input,
            })
            .unwrap();
        let mut view = View::new(Viewport {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
            scale_factor: 1.0,
        });

        for event in [
            InputEvent::PointerPressed {
                position: PhysicalPoint { x: 400.0, y: 400.0 },
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            InputEvent::PointerMoved {
                position: PhysicalPoint { x: 430.0, y: 380.0 },
            },
        ] {
            let _ = view.handle_input(&graph, event);
        }

        let [wire] = wires(&graph, &view)[..] else {
            panic!("one connection draws one wire");
        };
        assert_eq!(wire.points[0], [0.0, 15.0]);
        assert_eq!(wire.points[3], [30.0, 65.0]);
        let shapes = shapes(&graph, &view);
        let [_, _, _, node, input_port, output_port] = &shapes[..] else {
            panic!("each node draws with its two ports");
        };
        assert_eq!(node.centre, [30.0, 80.0]);
        assert_eq!(input_port.centre, wire.points[3]);
        assert_eq!(output_port.centre, [30.0, 95.0]);
    }
}
