struct ViewTransform {
    // The graph point at the viewport's centre.
    centre: vec2<f32>,
    clip_per_unit: vec2<f32>,
    // WebGL2 needs uniform blocks in multiples of 16 bytes.
    @size(16) pixels_per_unit: f32,
}

@group(0) @binding(0) var<uniform> view: ViewTransform;
// Glyph coverage, read texel for texel: bitmaps are placed on whole physical pixels.
@group(0) @binding(1) var atlas: texture_2d<f32>;

// An sRGB target encodes on write, so it needs linear colors.
override srgb_target: bool;
// Set by the renderer, which also sizes each wire draw from it.
override wire_segments: u32;

const BACKGROUND = vec3<f32>(0.16, 0.16, 0.17);
const WIRE_COLOR = vec3<f32>(0.72, 0.72, 0.72);
const WIRE_WIDTH = 2.0;

fn to_clip(point: vec2<f32>) -> vec4<f32> {
    let clip = (point - view.centre) * view.clip_per_unit;
    return vec4<f32>(clip.x, -clip.y, 0.0, 1.0);
}

fn target_color(srgb: vec3<f32>, alpha: f32) -> vec4<f32> {
    if srgb_target {
        let linear = select(
            pow((srgb + 0.055) / 1.055, vec3<f32>(2.4)),
            srgb / 12.92,
            srgb <= vec3<f32>(0.04045),
        );
        return vec4<f32>(linear, alpha);
    }
    return vec4<f32>(srgb, alpha);
}

// One triangle covering the viewport; the scissor keeps it inside.
@vertex
fn background_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn background_fragment() -> @location(0) vec4<f32> {
    return target_color(BACKGROUND, 1.0);
}

struct WireVertex {
    @builtin(position) position: vec4<f32>,
    // Signed distance from the centre line, in physical pixels.
    @location(0) across: f32,
    @location(1) half_width: f32,
}

// A triangle strip along the cubic Bézier, two vertices per step.
@vertex
fn wire_vertex(
    @builtin(vertex_index) index: u32,
    @location(0) p0: vec2<f32>,
    @location(1) p1: vec2<f32>,
    @location(2) p2: vec2<f32>,
    @location(3) p3: vec2<f32>,
) -> WireVertex {
    let t = f32(index / 2u) / f32(wire_segments);
    let s = 1.0 - t;
    let point = s * s * s * p0 + 3.0 * s * s * t * p1 + 3.0 * s * t * t * p2 + t * t * t * p3;
    var tangent = s * s * (p1 - p0) + 2.0 * s * t * (p2 - p1) + t * t * (p3 - p2);
    // Coincident control points give no tangent at the ends.
    if dot(tangent, tangent) < 1e-12 {
        tangent = p3 - p0;
    }
    if dot(tangent, tangent) < 1e-12 {
        tangent = vec2<f32>(1.0, 0.0);
    }
    let normal = normalize(vec2<f32>(-tangent.y, tangent.x));

    // Thin wires stay one pixel wide when zoomed out; one more pixel each side holds the edge
    // antialiasing.
    let half_width = max(WIRE_WIDTH * view.pixels_per_unit, 1.0) / 2.0;
    let side = select(-1.0, 1.0, (index & 1u) == 1u);
    let across = side * (half_width + 1.0);

    var out: WireVertex;
    out.position = to_clip(point + normal * across / view.pixels_per_unit);
    out.across = across;
    out.half_width = half_width;
    return out;
}

@fragment
fn wire_fragment(in: WireVertex) -> @location(0) vec4<f32> {
    let coverage = clamp(in.half_width + 0.5 - abs(in.across), 0.0, 1.0);
    return target_color(WIRE_COLOR, coverage);
}

struct ShapeVertex {
    @builtin(position) position: vec4<f32>,
    // Offset from the shape's centre, in graph units.
    @location(0) local: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) radius: f32,
    @location(3) border_width: f32,
    @location(4) fill: vec4<f32>,
    @location(5) border: vec4<f32>,
    // A bitmap's x, y, width and height in the atlas, in texels; no width for a rounded rectangle.
    @location(6) @interpolate(flat) atlas: vec4<u32>,
}

// A rounded rectangle, one pixel larger than the shape for the edge antialiasing, or a bitmap
// exactly covering its pixels.
@vertex
fn shape_vertex(
    @builtin(vertex_index) index: u32,
    @location(0) centre: vec2<f32>,
    @location(1) half_size: vec2<f32>,
    @location(2) radius: f32,
    @location(3) border_width: f32,
    @location(4) fill: vec4<f32>,
    @location(5) border: vec4<f32>,
    @location(6) atlas: vec4<u32>,
) -> ShapeVertex {
    let corner = vec2<f32>(f32(index & 1u), f32(index >> 1u)) * 2.0 - 1.0;
    let antialiasing = select(1.0 / view.pixels_per_unit, 0.0, atlas.z > 0u);
    let local = corner * (half_size + antialiasing);

    var out: ShapeVertex;
    out.position = to_clip(centre + local);
    out.local = local;
    out.half_size = half_size;
    out.radius = radius;
    out.border_width = border_width;
    out.fill = fill;
    out.border = border;
    out.atlas = atlas;
    return out;
}

@fragment
fn shape_fragment(in: ShapeVertex) -> @location(0) vec4<f32> {
    if in.atlas.z > 0u {
        // Pixel centres are half a texel in, so truncating finds the pixel's own texel.
        let offset = vec2<u32>(max((in.local + in.half_size) * view.pixels_per_unit, vec2<f32>(0.0)));
        let texel = in.atlas.xy + min(offset, in.atlas.zw - 1u);
        let coverage = textureLoad(atlas, texel, 0).r;
        return target_color(in.fill.rgb, in.fill.a * coverage);
    }
    let q = abs(in.local) - in.half_size + in.radius;
    let distance = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - in.radius;
    let edge = distance * view.pixels_per_unit;
    // The fill covers what lies further inside than the border width, and the border the band
    // between. Weighting each color by its alpha keeps a borderless edge the fill's color rather
    // than darkening it toward a transparent border, which showed as seams between shapes.
    let outer = clamp(0.5 - edge, 0.0, 1.0);
    let inner = clamp(0.5 - edge - in.border_width * view.pixels_per_unit, 0.0, 1.0);
    let fill = in.fill.a * inner;
    let border = in.border.a * (outer - inner);
    let alpha = fill + border;
    if alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    return target_color((in.fill.rgb * fill + in.border.rgb * border) / alpha, alpha);
}
