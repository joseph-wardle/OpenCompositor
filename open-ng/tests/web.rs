//! Draws two views with WebGL2 in a browser and checks the pixels. WebGL2 is the most limited
//! backend the renderer supports, so this is where a shader or layout it cannot run shows up.
//!
//! Run with `wasm-bindgen-test-runner` as the wasm32 target's runner; `webdriver.json` sets the
//! browser's options.

#![cfg(target_arch = "wasm32")]

use std::cell::Cell;
use std::rc::Rc;

use ab_glyph::{Font, FontRef};
use open_ng::{
    Color, Connection, ConnectionEdit, Direction, Graph, InputEvent, NodeId, NodeRecord,
    PhysicalPoint, Pin, PinId, PinRef, Point, Renderer, Row, Verdict, View, ViewBuffers, Viewport,
};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

const WIDTH: u32 = 400;
const HEIGHT: u32 = 100;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const BACKGROUND: [u8; 3] = [41, 41, 43];
const FILL: Color = Color {
    r: 70,
    g: 120,
    b: 170,
};
const SELECTED_BORDER: [u8; 3] = [255, 200, 40];
const BODY: [u8; 3] = [52, 52, 56];
const WIRE: [u8; 3] = [184, 184, 184];

#[wasm_bindgen_test]
async fn webgl2_draws_each_view_inside_its_viewport() {
    let (device, queue) = webgl2_device().await;

    // Each view is 200 pixels wide with the graph origin at its centre. Node 1 is centred there,
    // and node 2 straddles the right view's left edge: the right view is drawn last, so anything
    // it draws outside itself stays on top. An untitled node with no rows is 100 × 30.
    let mut graph = Graph::default();
    for (id, x) in [(1, -50.0), (2, -150.0)] {
        let record = NodeRecord {
            position: Point { x, y: -15.0 },
            title: String::new(),
            color: FILL,
            rows: Vec::new(),
        };
        graph.add_node(NodeId(id), record).unwrap();
    }
    graph.set_selection(&[NodeId(1)]).unwrap();
    let views = [0, WIDTH / 2].map(|x| {
        View::new(Viewport {
            x,
            y: 0,
            width: WIDTH / 2,
            height: HEIGHT,
            scale_factor: 1.0,
        })
    });
    let pixels = draw(&device, &queue, &graph, &views).await;

    let fill = [FILL.r, FILL.g, FILL.b];
    for left in [0, WIDTH / 2] {
        assert_eq!(pixel(&pixels, left + 5, 5), BACKGROUND, "background");
        assert_eq!(pixel(&pixels, left + 80, 50), fill, "fill");
        assert_eq!(pixel(&pixels, left + 51, 50), SELECTED_BORDER, "border");
    }
    // The viewport alone clips here: SwiftShader rasterizes nothing outside it, so this cannot
    // show that the scissor is set, which GPUs with guard-band clipping need.
    assert_eq!(
        pixel(&pixels, WIDTH / 2 - 5, 50),
        BACKGROUND,
        "the right view drew outside itself"
    );
}

#[wasm_bindgen_test]
async fn webgl2_draws_text_at_each_views_zoom_under_the_nodes_above() {
    const TITLE: &str = "Blur radius";
    const COVER: Color = Color {
        r: 150,
        g: 60,
        b: 60,
    };
    let (device, queue) = webgl2_device().await;

    // Node 1's title bar spans y = -22 to 2 and its one row 2 to 22. Node 2, added later, covers
    // the middle of that row's label, which is long enough to show on either side.
    let mut graph = Graph::default();
    let titled = NodeRecord {
        position: Point { x: -90.0, y: -22.0 },
        title: TITLE.to_owned(),
        color: FILL,
        rows: vec![Row::Pin(Pin {
            id: PinId(1),
            label: "M".repeat(16),
            direction: Direction::Input,
        })],
    };
    graph.add_node(NodeId(1), titled).unwrap();
    let cover = NodeRecord {
        position: Point { x: -80.0, y: 4.0 },
        title: String::new(),
        color: COVER,
        rows: Vec::new(),
    };
    graph.add_node(NodeId(2), cover).unwrap();
    let zooms = [1.0, 1.2];
    let views = [0, 1].map(|index| {
        let mut view = View::new(Viewport {
            x: index * WIDTH / 2,
            y: 0,
            width: WIDTH / 2,
            height: HEIGHT,
            scale_factor: 1.0,
        });
        if index == 1 {
            // One notch in at the view's centre keeps the graph origin there.
            let scroll = InputEvent::Scrolled {
                position: PhysicalPoint { x: 300.0, y: 50.0 },
                lines: 1.0,
            };
            let _ = view.handle_input(&graph, scroll, accept_all);
        }
        view
    });
    let pixels = draw(&device, &queue, &graph, &views).await;

    let font = FontRef::try_from_slice(include_bytes!("../fonts/Geist-Regular.ttf")).unwrap();
    let title_ink = title_ink_width(&font, TITLE, 13.0);
    let fill = [FILL.r, FILL.g, FILL.b];
    for (index, zoom) in zooms.into_iter().enumerate() {
        let left = index as u32 * WIDTH / 2;
        // Physical pixel columns or rows between two graph coordinates, a few pixels inside them
        // and the view.
        let x = |from: f32, to: f32| {
            let centre = left as f32 + WIDTH as f32 / 4.0;
            let from = ((centre + from * zoom).ceil() as u32).max(left) + 2;
            let to = ((centre + to * zoom).floor() as u32).min(left + WIDTH / 2) - 2;
            from..to
        };
        let y = |from: f32, to: f32| {
            let centre = HEIGHT as f32 / 2.0;
            (centre + from * zoom).ceil() as u32 + 2..(centre + to * zoom).floor() as u32 - 2
        };
        // The columns with a pixel clearly unlike `under`.
        let inked = |columns: std::ops::Range<u32>, rows: std::ops::Range<u32>, under: [u8; 3]| {
            columns
                .filter(|&column| {
                    rows.clone().any(|row| {
                        let pixel = pixel(&pixels, column, row);
                        (0..3).any(|i| pixel[i].abs_diff(under[i]) > 24)
                    })
                })
                .collect::<Vec<_>>()
        };

        let title = inked(x(-90.0, 50.0), y(-22.0, 2.0), fill);
        let (first, last) = (title[0], title[title.len() - 1]);
        let width = (last + 1 - first) as f32;
        assert!(
            (width - title_ink * zoom).abs() <= 1.0,
            "zoom {zoom}: title ink is {width} pixels wide, measured {}",
            title_ink * zoom
        );

        let cover = [COVER.r, COVER.g, COVER.b];
        let covered = inked(x(-80.0, 20.0), y(4.0, 28.0), cover).len();
        assert_eq!(covered, 0, "zoom {zoom}: the label shows through node 2");
        let beyond = inked(x(22.0, 60.0), y(4.0, 20.0), BODY).len();
        assert!(beyond > 0, "zoom {zoom}: the label is drawn beside node 2");
    }
}

#[wasm_bindgen_test]
async fn webgl2_draws_a_wire_in_its_color_and_fades_it_below_a_pixel_wide() {
    let (device, queue) = webgl2_device().await;

    // An untitled node's one row has its middle 34 units below the top, so the wire runs straight
    // along y = 0 from x = -100 to 100, past both views' centres.
    let mut graph = Graph::default();
    for (id, x, direction) in [(1, -200.0, Direction::Output), (2, 100.0, Direction::Input)] {
        let record = NodeRecord {
            position: Point { x, y: -34.0 },
            title: String::new(),
            color: FILL,
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
    // Eight notches out zoom the right view to 1.2⁻⁸, where the wire is under half a pixel wide.
    let zoom_out = 1.2_f32.powi(-8);
    let views = [0, 1].map(|index| {
        let mut view = View::new(Viewport {
            x: index * WIDTH / 2,
            y: 0,
            width: WIDTH / 2,
            height: HEIGHT,
            scale_factor: 1.0,
        });
        if index == 1 {
            let scroll = InputEvent::Scrolled {
                position: PhysicalPoint { x: 300.0, y: 50.0 },
                lines: -8.0,
            };
            let _ = view.handle_input(&graph, scroll, accept_all);
        }
        view
    });
    let pixels = draw(&device, &queue, &graph, &views).await;

    // The wire's centre line is the boundary between rows 49 and 50, both fully covered at zoom
    // 1, where the wire is two pixels wide.
    assert_eq!(pixel(&pixels, WIDTH / 4, 49), WIRE);
    assert_eq!(pixel(&pixels, WIDTH / 4, 50), WIRE);
    // Below a pixel the wire keeps its ink per length: each column holds as much ink as a wire
    // its width at full strength.
    for (column, width) in [(WIDTH / 4, 2.0), (3 * WIDTH / 4, 2.0 * zoom_out)] {
        let ink: f32 = (0..HEIGHT)
            .map(|row| {
                let pixel = pixel(&pixels, column, row);
                f32::from(pixel[0] - BACKGROUND[0]) / f32::from(WIRE[0] - BACKGROUND[0])
            })
            .sum();
        assert!(
            (ink - width).abs() < 0.05,
            "column {column}: {ink} pixels of ink, expected {width}"
        );
    }
}

/// The width from the first glyph's ink to the last's, with each glyph at the sum of the
/// advances before it, in graph units at `size` units per em.
fn title_ink_width(font: &FontRef<'_>, text: &str, size: f32) -> f32 {
    let per_unit = size / font.units_per_em().unwrap();
    let glyphs: Vec<_> = text.chars().map(|c| font.glyph_id(c)).collect();
    let pens: f32 = glyphs[..glyphs.len() - 1]
        .iter()
        .map(|&glyph| font.h_advance_unscaled(glyph))
        .sum();
    let first = font.outline(glyphs[0]).unwrap().bounds.min.x;
    let last = font.outline(glyphs[glyphs.len() - 1]).unwrap().bounds.max.x;
    (pens + last - first) * per_unit
}

/// A device on the WebGL2 backend. Fails rather than skips without one, so that a browser that
/// cannot run the test does not pass it.
async fn webgl2_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    // wgpu's WebGL backend gets its context from a canvas, so it offers no adapter without one.
    let canvas = web_sys::OffscreenCanvas::new(WIDTH, HEIGHT).unwrap();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas))
        .expect("a WebGL2 context");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .expect("a WebGL2 adapter");
    assert_eq!(adapter.get_info().backend, wgpu::Backend::Gl);
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            ..Default::default()
        })
        .await
        .unwrap()
}

/// Draws the views into a texture and reads it back as tightly packed RGBA rows.
async fn draw(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &Graph,
    views: &[View],
) -> Vec<u8> {
    let renderer = Renderer::new(device, FORMAT, 1);
    let buffers: Vec<_> = views
        .iter()
        .map(|view| {
            let mut buffers = ViewBuffers::new(device, &renderer);
            renderer.prepare(device, queue, graph, view, &mut buffers);
            buffers
        })
        .collect();

    let size = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let row_bytes = WIDTH * 4;
    let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded_row_bytes * HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let target = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        for buffers in &buffers {
            renderer.draw(&mut pass, buffers);
        }
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);

    let mapped = Rc::new(Cell::new(None));
    let result = mapped.clone();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |outcome| {
            result.set(Some(outcome))
        });
    // The WebGL backend finishes a mapping only when polled after the GPU is done, which the
    // browser reports only between tasks; so poll, then yield a whole task.
    let outcome = loop {
        device.poll(wgpu::PollType::Poll).unwrap();
        if let Some(outcome) = mapped.take() {
            break outcome;
        }
        let next_task = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window()
                .unwrap()
                .set_timeout_with_callback(&resolve)
                .unwrap();
        });
        JsFuture::from(next_task).await.unwrap();
    };
    outcome.unwrap();

    let data = readback.slice(..).get_mapped_range().unwrap();
    data.chunks(padded_row_bytes as usize)
        .flat_map(|row| &row[..row_bytes as usize])
        .copied()
        .collect()
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * WIDTH + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

/// The tests only zoom, which proposes no connection edits.
fn accept_all(_: &ConnectionEdit) -> Verdict {
    Verdict::Accept {
        displaced: Vec::new(),
    }
}
