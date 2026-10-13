//! Drawing views with wgpu into a render pass the host owns.

use std::collections::HashSet;
use std::mem::size_of;

use ab_glyph::{Font, FontRef, GlyphId};
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashSet;

use crate::atlas::{ATLAS_SIZE, GlyphAtlas, Ink};
use crate::geometry::{
    self, LABEL_SIZE, LabelText, PORT_RADIUS, Rect, TITLE_HEIGHT, WIRE_SEGMENTS,
};
use crate::graph::{Color, Connection, Graph, NodeId, PinRef, Point, Side};
use crate::input::PhysicalPoint;
use crate::text::{advance, bundled_font, cap_height, text_width};
use crate::view::{CutPreview, View, Viewport, WirePreview};

/// In graph units; never drawn thinner than a pixel.
const WIRE_WIDTH: f32 = 2.0;
const WIRE_COLOR: [u8; 4] = [184, 184, 184, 255];
/// A dragged wire the host has not accepted, which release would not connect.
const LOOSE_WIRE_COLOR: [u8; 4] = [184, 184, 184, 110];
/// A wire the cut in progress would remove.
const CUT_WIRE_COLOR: [u8; 4] = [214, 96, 84, 255];
const CUT_STROKE_COLOR: [u8; 4] = [240, 240, 240, 200];
const NODE_CORNER_RADIUS: f32 = 4.0;
const NODE_BODY: [u8; 4] = [52, 52, 56, 255];
const HEADER_FILL: [u8; 4] = [64, 64, 69, 255];
const NODE_BORDER: [u8; 4] = [20, 20, 20, 255];
const NODE_BORDER_WIDTH: f32 = 1.0;
const SELECTED_BORDER: [u8; 4] = [255, 200, 40, 255];
const SELECTED_BORDER_WIDTH: f32 = 2.5;
const PORT_FILL: [u8; 4] = [210, 210, 210, 255];
const PORT_RING: [u8; 4] = [150, 150, 150, 255];
const PORT_RING_WIDTH: f32 = 1.5;
/// The ring of the port a press would grab.
const HOVERED_RING: [u8; 4] = [255, 255, 255, 255];
const BOX_FILL: [u8; 4] = [255, 255, 255, 12];
const BOX_BORDER: [u8; 4] = [220, 220, 220, 255];
const TITLE_TEXT: [u8; 4] = [240, 240, 240, 255];
const LABEL_TEXT: [u8; 4] = [200, 200, 204, 255];
/// The note beside the pointer giving the reason the host refused a dragged wire. Sizes are in
/// logical pixels, whatever the zoom.
const REFUSAL_SIZE: f32 = 12.0;
const REFUSAL_OFFSET: Point = Point { x: 14.0, y: 18.0 };
const REFUSAL_PADDING: f32 = 6.0;
const REFUSAL_FILL: [u8; 4] = [28, 28, 30, 240];
const REFUSAL_BORDER: [u8; 4] = [214, 96, 84, 255];
const REFUSAL_TEXT: [u8; 4] = [240, 240, 240, 255];
/// Text fades out as its size falls from the first to the second of these, in logical pixels
/// per em, so that the cutoff looks the same on every display.
const TEXT_FADE: [f32; 2] = [7.0, 5.0];

/// Draws views into render passes on one device whose single color attachment has the format
/// and sample count given at creation, with no depth-stencil attachment.
#[derive(Debug)]
pub struct Renderer {
    font: FontRef<'static>,
    view_layout: wgpu::BindGroupLayout,
    background: wgpu::RenderPipeline,
    wires: wgpu::RenderPipeline,
    shapes: wgpu::RenderPipeline,
}

/// A view's data on the GPU. The host keeps one beside each [`View`].
#[derive(Debug)]
pub struct ViewBuffers {
    viewport: Viewport,
    // Per view rather than in the renderer: uploads take effect at the next submit, so one shared
    // transform would draw every view in the pass with the last one prepared.
    transform: wgpu::Buffer,
    // Per view rather than in the renderer: preparing one view could otherwise overwrite glyphs
    // that another view's prepared shapes still refer to.
    atlas: GlyphAtlas,
    atlas_texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    wires: InstanceBuffer,
    shapes: InstanceBuffer,
}

/// Per-instance data in a vertex buffer. Not a storage buffer, because WebGL2 has none.
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

/// A cubic Bézier wire's control points, in graph units, and its sRGB-encoded color.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Wire {
    points: [[f32; 2]; 4],
    color: [u8; 4],
}

/// A rounded rectangle or a bitmap from the view's glyph atlas, in graph units, with
/// sRGB-encoded colors. Glyphs are shapes so that each node's text draws straight after the node
/// in one instance stream: above the nodes below it and below those above.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct Shape {
    centre: [f32; 2],
    half_size: [f32; 2],
    radius: f32,
    border_width: f32,
    fill: [u8; 4],
    border: [u8; 4],
    /// The bitmap's x, y, width and height in the atlas, in texels. It covers the shape one
    /// texel to a physical pixel, as coverage of the fill color. Without a width the shape is a
    /// rounded rectangle.
    atlas: [u16; 4],
}

impl Renderer {
    /// Creates the pipelines. The device must provide at least
    /// [`wgpu::Limits::downlevel_webgl2_defaults`] and WebGL2's downlevel flags, `format` must be
    /// renderable and blendable, and `sample_count` must be supported for it; otherwise this
    /// panics, as wgpu does on invalid use.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Renderer {
        let module = device.create_shader_module(wgpu::include_wgsl!("render.wgsl"));
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("open-ng view"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("open-ng"),
            bind_group_layouts: &[Some(&view_layout)],
            immediate_size: 0,
        });
        let constants = [
            ("srgb_target", f64::from(u8::from(format.is_srgb()))),
            ("wire_segments", f64::from(WIRE_SEGMENTS)),
            ("wire_width", f64::from(WIRE_WIDTH)),
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
            0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x2, 4 => Unorm8x4,
        ];
        let shape_attributes = wgpu::vertex_attr_array![
            0 => Float32x2, 1 => Float32x2, 2 => Float32, 3 => Float32, 4 => Unorm8x4, 5 => Unorm8x4,
            6 => Uint16x4,
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
            font: bundled_font(),
            view_layout,
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
        let mut instances = shapes(graph, view, &self.font, &mut buffers.atlas);
        if buffers.atlas.is_full() {
            // Make room for this frame's glyphs alone. Any that still do not fit go undrawn.
            buffers.atlas.clear();
            instances = shapes(graph, view, &self.font, &mut buffers.atlas);
        }
        for upload in buffers.atlas.take_uploads() {
            let [x, y, width, height] = upload.texels.map(u32::from);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &buffers.atlas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &upload.coverage,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        buffers.shapes.write(device, queue, &instances);
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
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("open-ng glyph atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("open-ng view"),
            layout: &renderer.view_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &atlas_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
            ],
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
            atlas: GlyphAtlas::default(),
            atlas_texture,
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

/// The connections a view draws: the committed ones, less those its accepted wire drag removes,
/// plus those it adds.
struct DrawnConnections<'a> {
    hidden: FxHashSet<Connection>,
    added: &'a [Connection],
}

impl DrawnConnections<'_> {
    fn of(view: &View) -> DrawnConnections<'_> {
        match view.wire_preview() {
            Some(WirePreview::Accepted { edit, displaced }) => DrawnConnections {
                hidden: edit.remove.iter().chain(displaced).copied().collect(),
                added: &edit.add,
            },
            _ => DrawnConnections {
                hidden: FxHashSet::default(),
                added: &[],
            },
        }
    }

    fn iter<'a>(&'a self, graph: &'a Graph) -> impl Iterator<Item = &'a Connection> {
        let committed = graph.connections().iter();
        committed
            .filter(|connection| !self.hidden.contains(connection))
            .chain(self.added)
    }
}

/// The wires in view as drawn, following any nodes being dragged.
fn wires(graph: &Graph, view: &View) -> Vec<Wire> {
    // The drawn width reaches a pixel at least, with another pixel each side for the edge
    // antialiasing.
    let pixel = 1.0 / view.physical_per_unit();
    let margin = WIRE_WIDTH.max(pixel) / 2.0 + pixel;
    let visible = view.visible_rect();
    // A preview's pins may have gone since the view last handled an event.
    let end = |pin: PinRef, side| {
        let top_left = view.drawn_position(graph, pin.node)?;
        graph.node_layout(pin.node)?.port(top_left, pin.pin, side)
    };
    let in_view = |points: [Point; 4], color| {
        let bounds = geometry::wire_bounds(points);
        let bounds = Rect {
            min: Point {
                x: bounds.min.x - margin,
                y: bounds.min.y - margin,
            },
            max: Point {
                x: bounds.max.x + margin,
                y: bounds.max.y + margin,
            },
        };
        bounds.intersects(&visible).then(|| Wire {
            points: points.map(|point| [point.x, point.y]),
            color,
        })
    };
    let drawn = DrawnConnections::of(view);
    let cut = view.cut_preview();
    let cut_wires: FxHashSet<Connection> = cut
        .iter()
        .flat_map(|cut| cut.cut.iter().chain(cut.displaced))
        .copied()
        .collect();
    let mut wires: Vec<Wire> = drawn
        .iter(graph)
        .filter_map(|connection| {
            let output = end(connection.from, Side::Output)?;
            let input = end(connection.to, Side::Input)?;
            let color = if cut_wires.contains(connection) {
                CUT_WIRE_COLOR
            } else {
                WIRE_COLOR
            };
            in_view(geometry::wire(output, input), color)
        })
        .collect();
    if let Some(CutPreview { stroke, .. }) = cut {
        // Each piece of the stroke is a wire with its control points on a straight line.
        wires.extend(stroke.windows(2).filter_map(|piece| {
            let [from, to] = [piece[0], piece[1]];
            let third = |thirds: f32| Point {
                x: from.x + (to.x - from.x) * thirds / 3.0,
                y: from.y + (to.y - from.y) * thirds / 3.0,
            };
            in_view([from, third(1.0), third(2.0), to], CUT_STROKE_COLOR)
        }));
    }
    if let Some(WirePreview::Loose {
        from,
        side,
        pointer,
        ..
    }) = view.wire_preview()
    {
        let pointer = view.to_graph(pointer);
        wires.extend(from.iter().filter_map(|&pin| {
            let port = end(pin, side)?;
            let points = match side {
                Side::Output => geometry::wire(port, pointer),
                Side::Input => geometry::wire(pointer, port),
            };
            in_view(points, LOOSE_WIRE_COLOR)
        }));
    }
    wires
}

/// Each node in view followed by its ports and text, from bottom to top, then any selection box.
/// A port is filled if any pin whose wires end there is connected and hollow otherwise. Glyphs
/// missing from the atlas are rasterised into it.
fn shapes(graph: &Graph, view: &View, font: &FontRef<'_>, atlas: &mut GlyphAtlas) -> Vec<Shape> {
    let selected: HashSet<_> = view.drawn_selection(graph).into_iter().collect();
    let drawn = DrawnConnections::of(view);
    let connected: FxHashSet<(NodeId, usize)> = drawn
        .iter(graph)
        .flat_map(|connection| {
            [
                (connection.from, Side::Output),
                (connection.to, Side::Input),
            ]
        })
        .filter_map(|(pin, side)| {
            let index = graph.node_layout(pin.node)?.port_index(pin.pin, side)?;
            Some((pin.node, index))
        })
        .collect();
    // Ports and the antialiased edge reach past a node's outline.
    let margin = PORT_RADIUS + 1.0 / view.physical_per_unit();
    let visible = view.visible_rect();
    let visible = Rect {
        min: Point {
            x: visible.min.x - margin,
            y: visible.min.y - margin,
        },
        max: Point {
            x: visible.max.x + margin,
            y: visible.max.y + margin,
        },
    };
    let label_opacity = text_opacity(LABEL_SIZE * view.logical_per_unit());
    let mut text = TextPainter { font, atlas, view };
    let mut shapes = Vec::new();
    for (id, record, layout) in graph.nodes_in_draw_order(&selected) {
        let top_left = view
            .drawn_position(graph, id)
            .expect("the node is in the graph");
        let rect = layout.rect(top_left);
        if !rect.intersects(&visible) {
            continue;
        }
        let (border, border_width) = if selected.contains(&id) {
            (SELECTED_BORDER, SELECTED_BORDER_WIDTH)
        } else {
            (NODE_BORDER, NODE_BORDER_WIDTH)
        };
        let title = Rect {
            min: rect.min,
            max: Point {
                x: rect.max.x,
                y: rect.min.y + TITLE_HEIGHT,
            },
        };
        let title_corners = Rect {
            min: Point {
                x: title.min.x,
                y: title.max.y - NODE_CORNER_RADIUS,
            },
            max: title.max,
        };
        shapes.extend([
            Shape {
                radius: NODE_CORNER_RADIUS,
                fill: NODE_BODY,
                ..rectangle(rect)
            },
            Shape {
                radius: NODE_CORNER_RADIUS,
                fill: opaque(record.color),
                ..rectangle(title)
            },
            // Squares the title bar's lower corners.
            Shape {
                fill: opaque(record.color),
                ..rectangle(title_corners)
            },
        ]);
        shapes.extend(layout.headers(top_left).map(|header| Shape {
            fill: HEADER_FILL,
            ..rectangle(header)
        }));
        // The outline goes over the fills so that the title bar does not cover it.
        shapes.push(Shape {
            radius: NODE_CORNER_RADIUS,
            border_width,
            border,
            ..rectangle(rect)
        });
        let hovered = view
            .hovered_side()
            .filter(|(pin, _)| pin.node == id)
            .and_then(|(pin, side)| layout.port_index(pin.pin, side));
        shapes.extend(layout.ports(top_left).enumerate().map(|(index, port)| {
            let connected = connected.contains(&(id, index));
            let fill = if connected { PORT_FILL } else { [0; 4] };
            let (border, border_width) = if hovered == Some(index) {
                (HOVERED_RING, PORT_RING_WIDTH)
            } else if connected {
                (NODE_BORDER, NODE_BORDER_WIDTH)
            } else {
                (PORT_RING, PORT_RING_WIDTH)
            };
            Shape {
                centre: [port.x, port.y],
                half_size: [PORT_RADIUS; 2],
                radius: PORT_RADIUS,
                border_width,
                fill,
                border,
                atlas: [0; 4],
            }
        }));
        for label in layout.labels() {
            let opacity = text_opacity(label.size * view.logical_per_unit());
            if opacity == 0.0 {
                continue;
            }
            let color = match label.text {
                LabelText::Title => TITLE_TEXT,
                LabelText::Pin(_) | LabelText::Group(_) => LABEL_TEXT,
            };
            let origin = geometry::offset(top_left, label.origin);
            let color = faded(color, opacity);
            text.line(
                &mut shapes,
                label.text.of(record),
                origin,
                label.size,
                color,
            );
        }
        if label_opacity > 0.0 {
            for (pen, folded) in layout.carets(top_left) {
                let color = faded(LABEL_TEXT, label_opacity);
                text.ink(&mut shapes, Ink::Caret { folded }, pen, LABEL_SIZE, color);
            }
        }
    }
    if let Some(rect) = view.drawn_box() {
        shapes.push(Shape {
            border_width: 1.0 / view.physical_per_unit(),
            fill: BOX_FILL,
            border: BOX_BORDER,
            ..rectangle(rect)
        });
    }
    if let Some(WirePreview::Loose {
        pointer,
        refusal: Some(reason),
        ..
    }) = view.wire_preview()
    {
        text.refusal(&mut shapes, reason, pointer);
    }
    if let Some(CutPreview {
        pointer,
        refusal: Some(reason),
        ..
    }) = view.cut_preview()
    {
        text.refusal(&mut shapes, reason, pointer);
    }
    shapes
}

/// How opaque text is at a size in logical pixels per em.
fn text_opacity(logical_pixels_per_em: f32) -> f32 {
    let [opaque, transparent] = TEXT_FADE;
    ((logical_pixels_per_em - transparent) / (opaque - transparent)).clamp(0.0, 1.0)
}

fn faded(color: [u8; 4], opacity: f32) -> [u8; 4] {
    let [r, g, b, a] = color;
    [r, g, b, (f32::from(a) * opacity).round() as u8]
}

/// Sets text as glyphs from the view's atlas, on its physical pixel grid.
struct TextPainter<'a> {
    font: &'a FontRef<'a>,
    atlas: &'a mut GlyphAtlas,
    view: &'a View,
}

impl TextPainter<'_> {
    /// One line with its baseline starting at `origin`, at `size` graph units per em. Each glyph
    /// starts at the sum of the advances before it, as `text_width` measures.
    fn line(
        &mut self,
        shapes: &mut Vec<Shape>,
        text: &str,
        origin: Point,
        size: f32,
        color: [u8; 4],
    ) {
        let mut pen = origin;
        for c in text.chars() {
            let glyph = self.font.glyph_id(c);
            if glyph == GlyphId(0) {
                shapes.push(self.missing(pen, size, color));
            } else {
                self.ink(shapes, Ink::Glyph(glyph), pen, size, color);
            }
            pen.x += advance(self.font, glyph, size);
        }
    }

    fn ink(&mut self, shapes: &mut Vec<Shape>, ink: Ink, pen: Point, size: f32, color: [u8; 4]) {
        let per_unit = self.view.physical_per_unit();
        let pen = self.view.to_physical(pen);
        let Some(sprite) = self
            .atlas
            .sprite(self.font, ink, size * per_unit, [pen.x, pen.y])
        else {
            return;
        };
        let [.., width, height] = sprite.texels.map(f32::from);
        let centre = self.view.to_graph(PhysicalPoint {
            x: sprite.min[0] + width / 2.0,
            y: sprite.min[1] + height / 2.0,
        });
        shapes.push(Shape {
            centre: [centre.x, centre.y],
            half_size: [width / 2.0 / per_unit, height / 2.0 / per_unit],
            radius: 0.0,
            border_width: 0.0,
            fill: color,
            border: [0; 4],
            atlas: sprite.texels,
        });
    }

    /// `reason` on a box beside the pointer, on the side with room for it, at a fixed logical
    /// size and on whole physical pixels.
    fn refusal(&mut self, shapes: &mut Vec<Shape>, reason: &str, pointer: PhysicalPoint) {
        let viewport = self.view.viewport();
        let scale = viewport.scale_factor;
        let size = REFUSAL_SIZE * scale;
        let padding = REFUSAL_PADDING * scale;
        let width = (text_width(self.font, reason, size) + 2.0 * padding).ceil();
        let height = (cap_height(size) + 2.0 * padding).ceil();
        // Beside the pointer, or flipped to its other side where that runs past the viewport.
        let beside = |pointer: f32, offset: f32, length: f32, start: u32, extent: u32| {
            let after = pointer + offset * scale;
            let before = pointer - offset * scale - length;
            let place = if after + length > (start + extent) as f32 {
                before.max(start as f32)
            } else {
                after
            };
            place.round()
        };
        let x = beside(
            pointer.x,
            REFUSAL_OFFSET.x,
            width,
            viewport.x,
            viewport.width,
        );
        let y = beside(
            pointer.y,
            REFUSAL_OFFSET.y,
            height,
            viewport.y,
            viewport.height,
        );
        let corner = |x: f32, y: f32| self.view.to_graph(PhysicalPoint { x, y });
        let per_unit = self.view.physical_per_unit();
        shapes.push(Shape {
            border_width: 1.0 / per_unit,
            fill: REFUSAL_FILL,
            border: REFUSAL_BORDER,
            ..rectangle(Rect::from_corners(
                corner(x, y),
                corner(x + width, y + height),
            ))
        });
        let baseline = corner(x + padding.round(), y + height - padding.round());
        self.line(shapes, reason, baseline, size / per_unit, REFUSAL_TEXT);
    }

    /// A box for a character the font lacks, within the `.notdef` advance that measurement gives
    /// it. Geist's own `.notdef` glyph is blank.
    fn missing(&self, pen: Point, size: f32, color: [u8; 4]) -> Shape {
        let advance = advance(self.font, GlyphId(0), size);
        let rect = Rect {
            min: Point {
                x: pen.x + 0.1 * advance,
                y: pen.y - cap_height(size),
            },
            max: Point {
                x: pen.x + 0.9 * advance,
                y: pen.y,
            },
        };
        Shape {
            border_width: 1.0 / self.view.physical_per_unit(),
            border: color,
            ..rectangle(rect)
        }
    }
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
        atlas: [0; 4],
    }
}

fn opaque(color: Color) -> [u8; 4] {
    [color.r, color.g, color.b, 255]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection_edit::{ConnectionEdit, Verdict, accept_all};
    use crate::geometry::{MIN_WIDTH, ROW_HEIGHT, TITLE_HEIGHT};
    use crate::graph::{Connection, Direction, Group, GroupId, NodeRecord, Pin, PinId, Row};
    use crate::input::{Button, InputEvent, Key, Modifiers, PhysicalPoint};

    const GREY: Color = Color {
        r: 128,
        g: 128,
        b: 128,
    };

    fn at(x: f32, y: f32) -> Point {
        Point { x, y }
    }

    fn drawn_shapes(graph: &Graph, view: &View) -> Vec<Shape> {
        shapes(graph, view, &bundled_font(), &mut GlyphAtlas::default())
    }

    fn shapes_without_text(graph: &Graph, view: &View) -> Vec<Shape> {
        let mut shapes = drawn_shapes(graph, view);
        shapes.retain(|shape| shape.atlas == [0; 4]);
        shapes
    }

    fn is_glyph(shape: &Shape, color: [u8; 4]) -> bool {
        shape.atlas != [0; 4] && shape.fill[..3] == color[..3]
    }

    fn view(scale_factor: f32) -> View {
        View::new(Viewport {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
            scale_factor,
        })
    }

    /// A node titled `title` at `x` with an input labelled "Size" and a folded group.
    fn labelled(title: &str, x: f32) -> NodeRecord {
        NodeRecord {
            position: at(x, 0.0),
            title: title.to_owned(),
            color: GREY,
            rows: vec![
                Row::Pin(Pin {
                    id: PinId(1),
                    label: "Size".to_owned(),
                    direction: Direction::Input,
                }),
                Row::Group(Group {
                    id: GroupId(1),
                    label: String::new(),
                    folded: true,
                    pins: Vec::new(),
                }),
            ],
        }
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
    fn the_view_transform_matches_its_shader_uniform() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("render.wgsl")).unwrap();
        let mut layouter = wgpu::naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (_, view) = module
            .global_variables
            .iter()
            .find(|(_, global)| global.name.as_deref() == Some("view"))
            .unwrap();
        let size = layouter[view.ty].size as usize;
        assert_eq!(size, size_of::<ViewTransform>());
        assert_eq!(size % 16, 0, "WebGL2 needs a multiple of 16 bytes");
    }

    #[test]
    fn a_wire_end_follows_its_port_during_a_drag() {
        let mut graph = Graph::default();
        let pin = |id, direction| {
            Row::Pin(Pin {
                id: PinId(id),
                label: "Image".to_owned(),
                direction,
            })
        };
        for (id, y) in [(1, 0.0), (2, 100.0)] {
            let record = NodeRecord {
                position: at(0.0, y),
                title: String::new(),
                color: GREY,
                rows: vec![pin(1, Direction::Input), pin(2, Direction::Output)],
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        let connection = Connection {
            from: PinRef {
                node: NodeId(1),
                pin: PinId(2),
            },
            to: PinRef {
                node: NodeId(2),
                pin: PinId(1),
            },
        };
        graph.add_connection(connection).unwrap();
        let mut view = View::new(Viewport {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
            scale_factor: 1.0,
        });

        // Press node 2, which spans graph (0, 100) to (100, 170), and drag it by (30, -20).
        for event in [
            InputEvent::PointerPressed {
                position: PhysicalPoint { x: 450.0, y: 410.0 },
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            InputEvent::PointerMoved {
                position: PhysicalPoint { x: 480.0, y: 390.0 },
            },
        ] {
            let _ = view.handle_input(&graph, event, accept_all);
        }

        let [wire] = wires(&graph, &view)[..] else {
            panic!("one connection draws one wire");
        };
        let row_middle = |index: f32| TITLE_HEIGHT + (index + 0.5) * ROW_HEIGHT;
        assert_eq!(wire.points[0], [MIN_WIDTH, row_middle(1.0)]);
        assert_eq!(wire.points[3], [30.0, 80.0 + row_middle(0.0)]);
        let shapes = shapes_without_text(&graph, &view);
        let [
            _,
            _,
            _,
            _,
            unconnected,
            _,
            node,
            ..,
            input_port,
            output_port,
        ] = &shapes[..]
        else {
            panic!("each node draws as four shapes and its two ports");
        };
        assert_eq!(unconnected.fill, [0; 4]);
        assert_eq!(input_port.fill, PORT_FILL);
        assert_eq!(node.centre[0] - node.half_size[0], 30.0);
        assert_eq!(node.centre[1] - node.half_size[1], 80.0);
        assert_eq!(input_port.centre, wire.points[3]);
        assert_eq!(
            output_port.centre,
            [30.0 + MIN_WIDTH, 80.0 + row_middle(1.0)]
        );
    }

    #[test]
    fn a_folded_header_port_is_filled_if_any_hidden_member_on_its_side_is_connected() {
        let pin = |id, direction| Pin {
            id: PinId(id),
            label: String::new(),
            direction,
        };
        let group = |id, pins| {
            Row::Group(Group {
                id: GroupId(id),
                label: String::new(),
                folded: true,
                pins,
            })
        };
        let record = NodeRecord {
            position: at(0.0, 0.0),
            title: String::new(),
            color: GREY,
            rows: vec![
                group(
                    1,
                    vec![
                        pin(1, Direction::Input),
                        pin(2, Direction::Input),
                        pin(3, Direction::Output),
                    ],
                ),
                group(2, vec![pin(4, Direction::Input)]),
            ],
        };
        let mut graph = Graph::default();
        graph.add_node(NodeId(1), record.clone()).unwrap();
        graph.add_node(NodeId(2), record).unwrap();
        let connection = Connection {
            from: PinRef {
                node: NodeId(2),
                pin: PinId(3),
            },
            to: PinRef {
                node: NodeId(1),
                pin: PinId(2),
            },
        };
        graph.add_connection(connection).unwrap();
        let view = View::new(Viewport {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
            scale_factor: 1.0,
        });

        let filled: Vec<_> = shapes_without_text(&graph, &view)
            .iter()
            .filter(|shape| shape.half_size == [PORT_RADIUS; 2])
            .map(|shape| shape.fill == PORT_FILL)
            .collect();
        // For each node, its first header's input and output ports, then its second's input.
        assert_eq!(filled, [true, false, false, false, true, false]);
    }

    #[test]
    fn a_wire_between_pins_with_both_sides_uses_and_fills_only_its_own_ports() {
        let mut graph = Graph::default();
        for (id, x) in [(1, 0.0), (2, 200.0)] {
            let record = NodeRecord {
                position: at(x, 0.0),
                title: String::new(),
                color: GREY,
                rows: vec![Row::Pin(Pin {
                    id: PinId(1),
                    label: String::new(),
                    direction: Direction::Both,
                })],
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        let pin = |id| PinRef {
            node: NodeId(id),
            pin: PinId(1),
        };
        graph
            .add_connection(Connection {
                from: pin(1),
                to: pin(2),
            })
            .unwrap();
        let view = view(1.0);

        let [wire] = wires(&graph, &view)[..] else {
            panic!("one connection draws one wire");
        };
        let middle = TITLE_HEIGHT + ROW_HEIGHT / 2.0;
        assert_eq!(wire.points[0], [MIN_WIDTH, middle]);
        assert_eq!(wire.points[3], [200.0, middle]);
        let filled: Vec<_> = shapes_without_text(&graph, &view)
            .iter()
            .filter(|shape| shape.half_size == [PORT_RADIUS; 2])
            .map(|shape| shape.fill == PORT_FILL)
            .collect();
        // Each node's input port, then its output port.
        assert_eq!(filled, [false, true, true, false]);
    }

    /// Node 1 at the origin and node 3 below it, each with output 1, and node 2 at x = 200 with
    /// input 1, which node 3 feeds. The view, at `scale_factor`, has node 1's output dragged to
    /// node 2's input and judged by `verdict`.
    fn dragged_to_an_occupied_input(scale_factor: f32, verdict: Verdict) -> (Graph, View) {
        let mut graph = Graph::default();
        for (id, position, direction) in [
            (1, at(0.0, 0.0), Direction::Output),
            (2, at(200.0, 0.0), Direction::Input),
            (3, at(0.0, 100.0), Direction::Output),
        ] {
            let record = NodeRecord {
                position,
                title: String::new(),
                color: GREY,
                rows: vec![Row::Pin(Pin {
                    id: PinId(1),
                    label: String::new(),
                    direction,
                })],
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        graph.add_connection(occupant()).unwrap();
        let mut view = view(scale_factor);
        let port = |x: f32| PhysicalPoint {
            x: 400.0 + x * scale_factor,
            y: 300.0 + (TITLE_HEIGHT + ROW_HEIGHT / 2.0) * scale_factor,
        };
        let drag = [
            InputEvent::PointerPressed {
                position: port(MIN_WIDTH),
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            InputEvent::PointerMoved {
                position: port(200.0),
            },
        ];
        for event in drag {
            let _ = view.handle_input(&graph, event, |_: &ConnectionEdit| verdict.clone());
        }
        (graph, view)
    }

    fn occupant() -> Connection {
        Connection {
            from: PinRef {
                node: NodeId(3),
                pin: PinId(1),
            },
            to: PinRef {
                node: NodeId(2),
                pin: PinId(1),
            },
        }
    }

    #[test]
    fn an_accepted_wire_drag_draws_the_graph_as_the_host_would_leave_it() {
        let verdict = Verdict::Accept {
            displaced: vec![occupant()],
        };
        let (graph, view) = dragged_to_an_occupied_input(1.0, verdict);

        let [wire] = wires(&graph, &view)[..] else {
            panic!("the new wire alone");
        };
        let middle = TITLE_HEIGHT + ROW_HEIGHT / 2.0;
        assert_eq!(wire.points[0], [MIN_WIDTH, middle]);
        assert_eq!(wire.points[3], [200.0, middle]);
        assert_eq!(wire.color, WIRE_COLOR);
        let filled: Vec<_> = shapes_without_text(&graph, &view)
            .iter()
            .filter(|shape| shape.half_size == [PORT_RADIUS; 2])
            .map(|shape| shape.fill == PORT_FILL)
            .collect();
        assert_eq!(filled, [true, true, false], "nodes 1, 2 and 3");
    }

    #[test]
    fn a_refused_wire_drag_draws_a_loose_wire_and_the_reason_on_whole_pixels_with_room() {
        let scale_factor = 1.5;
        let verdict = Verdict::Refuse {
            reason: Some("Inputs on this node are locked".to_owned()),
        };
        let (graph, view) = dragged_to_an_occupied_input(scale_factor, verdict);

        let colors: Vec<_> = wires(&graph, &view).iter().map(|w| w.color).collect();
        assert_eq!(colors, [WIRE_COLOR, LOOSE_WIRE_COLOR]);
        let shapes = drawn_shapes(&graph, &view);
        let [note] = shapes
            .iter()
            .filter(|shape| shape.border == REFUSAL_BORDER)
            .collect::<Vec<_>>()[..]
        else {
            panic!("one note");
        };
        let corner = |sign: f32| {
            view.to_physical(Point {
                x: note.centre[0] + sign * note.half_size[0],
                y: note.centre[1] + sign * note.half_size[1],
            })
        };
        let (min, max) = (corner(-1.0), corner(1.0));
        for edge in [min.x, min.y, max.x, max.y] {
            assert!((edge - edge.round()).abs() < 1e-3, "{edge} is not whole");
        }
        // The pointer is at physical x 700 of 800, too near the right edge for the note.
        assert!(min.x >= 0.0 && max.x < 700.0);
        assert!(min.y > 300.0);
        assert!(shapes.iter().any(|shape| is_glyph(shape, REFUSAL_TEXT)));
    }

    #[test]
    fn lifted_wires_hang_loose_from_their_outputs_until_dropped_where_the_host_accepts() {
        let (mut graph, _) = dragged_to_an_occupied_input(1.0, Verdict::Refuse { reason: None });
        let wire_1 = Connection {
            from: PinRef {
                node: NodeId(1),
                pin: PinId(1),
            },
            ..occupant()
        };
        graph.add_connection(wire_1).unwrap();
        let mut view = view(1.0);
        let middle = TITLE_HEIGHT + ROW_HEIGHT / 2.0;
        let lift = [
            InputEvent::PointerPressed {
                position: PhysicalPoint {
                    x: 600.0,
                    y: 300.0 + middle,
                },
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            // Over node 1's body.
            InputEvent::PointerMoved {
                position: PhysicalPoint { x: 450.0, y: 310.0 },
            },
        ];
        for event in lift {
            let _ = view.handle_input(&graph, event, accept_all);
        }

        let drawn: Vec<_> = wires(&graph, &view)
            .iter()
            .map(|wire| (wire.points[0], wire.points[3], wire.color))
            .collect();
        let pointer = [50.0, 10.0];
        assert_eq!(
            drawn,
            [
                ([MIN_WIDTH, 100.0 + middle], [200.0, middle], WIRE_COLOR),
                ([MIN_WIDTH, middle], [200.0, middle], WIRE_COLOR),
                ([MIN_WIDTH, 100.0 + middle], pointer, LOOSE_WIRE_COLOR),
                ([MIN_WIDTH, middle], pointer, LOOSE_WIRE_COLOR),
            ],
            "the committed graph, then each lifted wire loose"
        );

        let over_space = InputEvent::PointerMoved {
            position: PhysicalPoint { x: 700.0, y: 500.0 },
        };
        let _ = view.handle_input(&graph, over_space, accept_all);
        assert_eq!(wires(&graph, &view), []);
        let filled: Vec<_> = shapes_without_text(&graph, &view)
            .iter()
            .filter(|shape| shape.half_size == [PORT_RADIUS; 2])
            .map(|shape| shape.fill == PORT_FILL)
            .collect();
        assert_eq!(filled, [false, false, false], "nodes 1, 2 and 3");
    }

    #[test]
    fn a_cut_marks_the_wires_it_would_remove_and_draws_its_stroke() {
        let (mut graph, _) = dragged_to_an_occupied_input(1.0, Verdict::Refuse { reason: None });
        let wire_1 = Connection {
            from: PinRef {
                node: NodeId(1),
                pin: PinId(1),
            },
            ..occupant()
        };
        graph.add_connection(wire_1).unwrap();
        let mut view = view(1.0);
        let keep_occupant = |edit: &ConnectionEdit| {
            if edit.remove.contains(&occupant()) {
                Verdict::Refuse {
                    reason: Some("Kept".to_owned()),
                }
            } else {
                accept_all(edit)
            }
        };
        let at_x_550 = |y| PhysicalPoint { x: 550.0, y };
        // Down between the nodes, across wire 1 and short of the occupant.
        let cut = [
            InputEvent::KeyPressed(Key::Y),
            InputEvent::PointerPressed {
                position: at_x_550(300.0),
                button: Button::Left,
                modifiers: Modifiers::default(),
            },
            InputEvent::PointerMoved {
                position: at_x_550(360.0),
            },
        ];
        for event in cut {
            let _ = view.handle_input(&graph, event, keep_occupant);
        }

        let drawn: Vec<_> = wires(&graph, &view)
            .iter()
            .map(|wire| (wire.points[0], wire.points[3], wire.color))
            .collect();
        let middle = TITLE_HEIGHT + ROW_HEIGHT / 2.0;
        assert_eq!(
            drawn,
            [
                ([MIN_WIDTH, 100.0 + middle], [200.0, middle], WIRE_COLOR),
                ([MIN_WIDTH, middle], [200.0, middle], CUT_WIRE_COLOR),
                ([150.0, 0.0], [150.0, 60.0], CUT_STROKE_COLOR),
            ]
        );
        let note = |view: &View| {
            drawn_shapes(&graph, view)
                .iter()
                .any(|shape| shape.border == REFUSAL_BORDER)
        };
        assert!(!note(&view));

        let across_occupant = InputEvent::PointerMoved {
            position: at_x_550(420.0),
        };
        let _ = view.handle_input(&graph, across_occupant, keep_occupant);
        let colors: Vec<_> = wires(&graph, &view).iter().map(|w| w.color).collect();
        assert_eq!(
            colors,
            [
                WIRE_COLOR,
                CUT_WIRE_COLOR,
                CUT_STROKE_COLOR,
                CUT_STROKE_COLOR
            ]
        );
        assert!(note(&view));
    }

    #[test]
    fn the_port_a_press_would_grab_is_ringed() {
        let mut graph = Graph::default();
        let record = NodeRecord {
            position: at(0.0, 0.0),
            title: String::new(),
            color: GREY,
            rows: vec![Row::Pin(Pin {
                id: PinId(1),
                label: String::new(),
                direction: Direction::Both,
            })],
        };
        graph.add_node(NodeId(1), record).unwrap();
        let mut view = view(1.0);
        let ringed = |view: &View| -> Vec<bool> {
            shapes_without_text(&graph, view)
                .iter()
                .filter(|shape| shape.half_size == [PORT_RADIUS; 2])
                .map(|shape| shape.border == HOVERED_RING)
                .collect()
        };
        let hover = |view: &mut View, x| {
            let position = PhysicalPoint {
                x: 400.0 + x,
                y: 300.0 + TITLE_HEIGHT + ROW_HEIGHT / 2.0,
            };
            let _ = view.handle_input(&graph, InputEvent::PointerMoved { position }, accept_all);
        };

        assert_eq!(ringed(&view), [false, false]);
        hover(&mut view, MIN_WIDTH / 2.0);
        assert_eq!(ringed(&view), [false, true]);
        hover(&mut view, 1.0);
        assert_eq!(ringed(&view), [true, false]);
    }

    /// Nodes 1, 2 and 3, untitled with no rows, side by side from x = -200, -50 and 100.
    fn three_nodes() -> Graph {
        let mut graph = Graph::default();
        for id in 1..=3 {
            let record = NodeRecord {
                position: at(-350.0 + 150.0 * id as f32, 0.0),
                title: String::new(),
                color: GREY,
                rows: Vec::new(),
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        graph
    }

    /// The left edge of each node body drawn, from bottom to top.
    fn bodies(graph: &Graph, view: &View) -> Vec<f32> {
        shapes_without_text(graph, view)
            .iter()
            .filter(|shape| shape.fill == NODE_BODY)
            .map(|shape| shape.centre[0] - shape.half_size[0])
            .collect()
    }

    #[test]
    fn selected_nodes_draw_above_the_rest_with_the_selection_shown() {
        let mut graph = three_nodes();
        graph.set_selection(&[NodeId(1), NodeId(3)]).unwrap();
        let mut view = View::new(Viewport {
            x: 0,
            y: 0,
            width: 800,
            height: 600,
            scale_factor: 1.0,
        });
        assert_eq!(bodies(&graph, &view), [-50.0, -200.0, 100.0]);

        // Pressing node 2 shows it as the only selection before release.
        let press = InputEvent::PointerPressed {
            position: PhysicalPoint { x: 400.0, y: 310.0 },
            button: Button::Left,
            modifiers: Modifiers::default(),
        };
        let _ = view.handle_input(&graph, press, accept_all);
        assert_eq!(bodies(&graph, &view), [-200.0, 100.0, -50.0]);
    }

    #[test]
    fn nodes_out_of_view_are_not_drawn_but_wires_crossing_it_are() {
        let mut graph = three_nodes();
        let pin = |direction| {
            Row::Pin(Pin {
                id: PinId(1),
                label: String::new(),
                direction,
            })
        };
        let mut record = graph.node(NodeId(1)).unwrap().clone();
        record.rows = vec![pin(Direction::Output)];
        graph.update_node(NodeId(1), record).unwrap();
        let mut record = graph.node(NodeId(3)).unwrap().clone();
        record.rows = vec![pin(Direction::Input)];
        graph.update_node(NodeId(3), record).unwrap();
        let connection = Connection {
            from: PinRef {
                node: NodeId(1),
                pin: PinId(1),
            },
            to: PinRef {
                node: NodeId(3),
                pin: PinId(1),
            },
        };
        graph.add_connection(connection).unwrap();
        // 140 units wide from x = -70, so only node 2 is in view.
        let view = View::new(Viewport {
            x: 0,
            y: 0,
            width: 140,
            height: 600,
            scale_factor: 1.0,
        });

        assert_eq!(bodies(&graph, &view), [-50.0]);
        assert_eq!(wires(&graph, &view).len(), 1);

        // Node 1's output port reaches 4 units past its right edge, now at x = -73.
        let mut record = graph.node(NodeId(1)).unwrap().clone();
        record.position.x += 27.0;
        graph.update_node(NodeId(1), record).unwrap();
        assert_eq!(bodies(&graph, &view), [-173.0, -50.0]);

        // With both ends far below the view, so is the whole wire.
        for id in [1, 3] {
            let mut record = graph.node(NodeId(id)).unwrap().clone();
            record.position.y += 1000.0;
            graph.update_node(NodeId(id), record).unwrap();
        }
        assert!(wires(&graph, &view).is_empty());
    }

    #[test]
    fn a_wire_is_drawn_while_any_of_its_stroke_is_in_view() {
        let mut graph = Graph::default();
        for (id, direction) in [(1, Direction::Output), (2, Direction::Input)] {
            let record = NodeRecord {
                position: at(0.0, 0.0),
                title: String::new(),
                color: GREY,
                rows: vec![Row::Pin(Pin {
                    id: PinId(1),
                    label: String::new(),
                    direction,
                })],
            };
            graph.add_node(NodeId(id), record).unwrap();
        }
        let pin = |id| PinRef {
            node: NodeId(id),
            pin: PinId(1),
        };
        graph
            .add_connection(Connection {
                from: pin(1),
                to: pin(2),
            })
            .unwrap();
        // Moves the nodes so that the wire runs between these points.
        let place = |graph: &mut Graph, output: Point, input: Point| {
            for (id, end, side) in [
                (NodeId(1), output, Side::Output),
                (NodeId(2), input, Side::Input),
            ] {
                let mut record = graph.node(id).unwrap().clone();
                let layout = graph.node_layout(id).unwrap();
                let port = layout.port(record.position, PinId(1), side).unwrap();
                record.position.x += end.x - port.x;
                record.position.y += end.y - port.y;
                graph.update_node(id, record).unwrap();
            }
        };
        let view = view(1.0);
        let visible = view.visible_rect();

        // At zoom 1 the stroke is a unit either side of the centre line, and its antialiased edge
        // half a unit more, so only the edge is in view.
        place(
            &mut graph,
            at(-100.0, visible.min.y - 1.25),
            at(100.0, visible.min.y - 1.25),
        );
        assert_eq!(wires(&graph, &view).len(), 1);
        place(
            &mut graph,
            at(-100.0, visible.min.y - 3.0),
            at(100.0, visible.min.y - 3.0),
        );
        assert!(wires(&graph, &view).is_empty());

        // A wire running back to the left first bulges about 8 units to the right of its output,
        // into the view, though both ends are outside it.
        place(
            &mut graph,
            at(visible.min.x - 4.0, 0.0),
            at(visible.min.x - 104.0, 0.0),
        );
        assert_eq!(wires(&graph, &view).len(), 1);
    }

    #[test]
    fn each_nodes_text_draws_after_it_and_before_the_next_node() {
        let mut graph = Graph::default();
        graph.add_node(NodeId(1), labelled("Blur", 0.0)).unwrap();
        graph.add_node(NodeId(2), labelled("Grade", 50.0)).unwrap();
        let shapes = drawn_shapes(&graph, &view(1.0));

        let bodies: Vec<_> = (0..shapes.len())
            .filter(|&i| shapes[i].fill == NODE_BODY)
            .collect();
        let [first, second] = bodies[..] else {
            panic!("two nodes");
        };
        let text = |range: std::ops::Range<usize>| {
            let shapes = &shapes[range];
            let titles = shapes.iter().filter(|s| is_glyph(s, TITLE_TEXT)).count();
            let labels = shapes.iter().filter(|s| is_glyph(s, LABEL_TEXT)).count();
            (titles, labels)
        };
        // "Size" and the fold caret.
        assert_eq!(text(first..second), ("Blur".len(), 5));
        assert_eq!(text(second..shapes.len()), ("Grade".len(), 5));
    }

    #[test]
    fn text_fades_by_logical_size_with_labels_first() {
        let mut graph = Graph::default();
        graph.add_node(NodeId(1), labelled("Blur", 0.0)).unwrap();
        let opacities = |view: &View| {
            let shapes = drawn_shapes(&graph, view);
            let alpha = |color| {
                let alphas: Vec<_> = shapes
                    .iter()
                    .filter(|s| is_glyph(s, color))
                    .map(|s| s.fill[3])
                    .collect();
                assert!(alphas.windows(2).all(|pair| pair[0] == pair[1]));
                alphas.first().copied()
            };
            (alpha(TITLE_TEXT), alpha(LABEL_TEXT))
        };
        let zoom_out = |view: &mut View, notches: f32| {
            let scroll = InputEvent::Scrolled {
                position: PhysicalPoint { x: 0.0, y: 0.0 },
                lines: -notches,
            };
            let _ = view.handle_input(&graph, scroll, accept_all);
        };

        assert_eq!(opacities(&view(1.0)), (Some(255), Some(255)));
        assert_eq!(
            opacities(&view(2.0)),
            (Some(255), Some(255)),
            "logical size"
        );

        // Zoom 0.48: labels are 5.3 logical pixels per em and titles 6.3.
        let mut view = view(2.0);
        zoom_out(&mut view, 4.0);
        let (Some(title), Some(label)) = opacities(&view) else {
            panic!("both are partly faded");
        };
        assert!(label < title && title < 255);

        // Zoom 0.40: labels are 4.4 and titles 5.2.
        zoom_out(&mut view, 1.0);
        let (Some(title), None) = opacities(&view) else {
            panic!("only titles show");
        };
        assert!(title > 0);
        zoom_out(&mut view, 1.0);
        assert_eq!(opacities(&view), (None, None));
    }

    #[test]
    fn a_missing_character_draws_as_a_box_within_its_measured_advance() {
        let font = bundled_font();
        let mut graph = Graph::default();
        graph.add_node(NodeId(1), labelled("A字B", 0.0)).unwrap();
        let shapes = drawn_shapes(&graph, &view(1.0));

        let [a, b] = shapes
            .iter()
            .filter(|s| is_glyph(s, TITLE_TEXT))
            .collect::<Vec<_>>()[..]
        else {
            panic!("two glyphs");
        };
        let boxes: Vec<_> = shapes.iter().filter(|s| s.border == TITLE_TEXT).collect();
        let [missing] = boxes[..] else {
            panic!("one box");
        };
        let notdef = advance(&font, GlyphId(0), geometry::TITLE_SIZE);
        let left = |s: &Shape| s.centre[0] - s.half_size[0];
        let right = |s: &Shape| s.centre[0] + s.half_size[0];
        assert!((right(missing) - left(missing) - 0.8 * notdef).abs() < 1e-3);
        assert!(right(a) <= left(missing) && right(missing) <= left(b));
        assert_eq!(missing.fill, [0; 4]);
    }
}
