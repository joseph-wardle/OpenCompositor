// The viewer: an image's data window, and an outline around its display window.
//
// Positions are in image pixels, y up, until `placed` turns them into the pane's physical pixels,
// y down. The target is sRGB, so fragments are linear values that the target encodes.

struct Frame {
    // An image point lands at `(x * scale + offset.x, offset.y - y * scale)` in the pane's
    // physical pixels.
    offset: vec2<f32>,
    scale: f32,
    // The outline's width in physical pixels.
    line: f32,
    // The pane's size in physical pixels.
    target_size: vec2<f32>,
    // x, y, width, height in image pixels; (x, y) is the bottom-left corner.
    data_window: vec4<f32>,
    display_window: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var image: texture_2d<f32>;

fn placed(point: vec2<f32>) -> vec2<f32> {
    return vec2(point.x * frame.scale + frame.offset.x, frame.offset.y - point.y * frame.scale);
}

fn to_clip(physical: vec2<f32>) -> vec4<f32> {
    let ndc = physical / frame.target_size * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    return vec4(ndc, 0.0, 1.0);
}

// A corner of the unit square, for a four-vertex triangle strip.
fn corner(vertex: u32) -> vec2<f32> {
    return vec2(f32(vertex & 1u), f32(vertex >> 1u));
}

struct ImageVertex {
    @builtin(position) position: vec4<f32>,
    // From the data window's top-left corner, in image pixels down, as the texture's rows are
    // stored.
    @location(0) pixel: vec2<f32>,
}

@vertex
fn image_vertex(@builtin(vertex_index) vertex: u32) -> ImageVertex {
    let from_bottom_left = corner(vertex) * frame.data_window.zw;
    let point = frame.data_window.xy + from_bottom_left;
    let pixel = vec2(from_bottom_left.x, frame.data_window.w - from_bottom_left.y);
    return ImageVertex(to_clip(placed(point)), pixel);
}

// Clamped to what the display shows. A NaN's comparisons may be optimised away, so its bits are
// tested instead; it shows black.
fn shown(value: f32) -> f32 {
    if (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u {
        return 0.0;
    }
    return clamp(value, 0.0, 1.0);
}

@fragment
fn image_fragment(vertex: ImageVertex) -> @location(0) vec4<f32> {
    let last = vec2<i32>(textureDimensions(image)) - 1;
    let texel = clamp(vec2<i32>(floor(vertex.pixel)), vec2(0), last);
    // Premultiplied over black is the colour itself.
    let colour = textureLoad(image, texel, 0);
    return vec4(shown(colour.r), shown(colour.g), shown(colour.b), 1.0);
}

// One of the outline's four sides, drawn just outside the display window.
@vertex
fn outline_vertex(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) side: u32,
) -> @builtin(position) vec4<f32> {
    let window = frame.display_window;
    // The window's top-left and bottom-right in physical pixels, y down.
    let low = placed(window.xy + vec2(0.0, window.w));
    let high = placed(window.xy + vec2(window.z, 0.0));
    let line = frame.line;
    var top_left: vec2<f32>;
    var bottom_right: vec2<f32>;
    switch side {
        case 0u: { top_left = low - line; bottom_right = vec2(high.x + line, low.y); }
        case 1u: { top_left = vec2(low.x - line, high.y); bottom_right = high + line; }
        case 2u: { top_left = vec2(low.x - line, low.y); bottom_right = vec2(low.x, high.y); }
        default: { top_left = vec2(high.x, low.y); bottom_right = vec2(high.x + line, high.y); }
    }
    return to_clip(mix(top_left, bottom_right, corner(vertex)));
}

@fragment
fn outline_fragment() -> @location(0) vec4<f32> {
    return vec4(0.25, 0.25, 0.25, 1.0);
}
