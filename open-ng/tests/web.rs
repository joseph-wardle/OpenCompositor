//! Draws two views with WebGL2 in a browser and checks the pixels. WebGL2 is the most limited
//! backend the renderer supports, so this is where a shader or layout it cannot run shows up.
//!
//! Run with `wasm-bindgen-test-runner` as the wasm32 target's runner; `webdriver.json` sets the
//! browser's options.

#![cfg(target_arch = "wasm32")]

use std::cell::Cell;
use std::rc::Rc;

use open_ng::{Color, Graph, NodeId, NodeRecord, Point, Renderer, View, ViewBuffers, Viewport};
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

#[wasm_bindgen_test]
async fn webgl2_draws_each_view_inside_its_viewport() {
    let (device, queue) = webgl2_device().await;

    // Each view is 200 pixels wide with the graph origin at its centre. Node 1 sits there, and
    // node 2 straddles the right view's left edge: the right view is drawn last, so anything it
    // draws outside itself stays on top.
    let mut graph = Graph::default();
    for (id, x) in [(1, 0.0), (2, -100.0)] {
        let record = NodeRecord::vertical(Point { x, y: 0.0 }, FILL, 1, 1);
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
        assert_eq!(pixel(&pixels, left + 61, 50), SELECTED_BORDER, "border");
    }
    // The viewport alone clips here: SwiftShader rasterizes nothing outside it, so this cannot
    // show that the scissor is set, which GPUs with guard-band clipping need.
    assert_eq!(
        pixel(&pixels, WIDTH / 2 - 5, 50),
        BACKGROUND,
        "the right view drew outside itself"
    );
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
