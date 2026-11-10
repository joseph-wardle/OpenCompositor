// A copy of an image for the viewer cache: its planes averaged over square blocks of pixels,
// `1 << halvings` on a side and aligned to the image's pixel grid, as RGBA f16, bottom row first.
// Pixels outside the data window count as zero, as they are, and NaN as zero, as the viewer shows
// it black. The sums run in the CPU's copy's order, rows up and pixels along them.

struct Copy {
    // The first pixel of the copy's first texel, from the data window's bottom-left corner.
    offset: vec2<i32>,
    // The data window's width and height, which the planes may exceed.
    data_size: vec2<u32>,
    // The copy's width and height in texels.
    size: vec2<u32>,
    halvings: u32,
}

@group(0) @binding(0) var<uniform> parameters: Copy;
@group(0) @binding(1) var red: texture_2d<f32>;
@group(0) @binding(2) var green: texture_2d<f32>;
@group(0) @binding(3) var blue: texture_2d<f32>;
@group(0) @binding(4) var alpha: texture_2d<f32>;
// Each texel's red and green, then its blue and alpha, packed as f16 pairs.
@group(0) @binding(5) var<storage, read_write> texels: array<vec2<u32>>;

// The largest finite f16: WGSL leaves packing anything beyond it undefined.
const F16_MAX: f32 = 65504.0;

// A NaN's comparisons may be optimised away, so its bits are tested instead.
fn counted(value: f32) -> f32 {
    if (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u {
        return 0.0;
    }
    return value;
}

@compute @workgroup_size(8, 8)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= parameters.size.x || id.y >= parameters.size.y {
        return;
    }
    let side = 1 << parameters.halvings;
    let first = parameters.offset + vec2<i32>(id.xy) * side;
    let low = max(first, vec2(0));
    let high = min(first + side, vec2<i32>(parameters.data_size));
    var sum = vec4(0.0);
    for (var y = low.y; y < high.y; y++) {
        for (var x = low.x; x < high.x; x++) {
            let pixel = vec2(x, y);
            sum += vec4(
                counted(textureLoad(red, pixel, 0).r),
                counted(textureLoad(green, pixel, 0).r),
                counted(textureLoad(blue, pixel, 0).r),
                counted(textureLoad(alpha, pixel, 0).r),
            );
        }
    }
    // A NaN from infinities of both signs counts as zero too.
    let averaged = sum / f32(side * side);
    let mean = clamp(
        vec4(counted(averaged.r), counted(averaged.g), counted(averaged.b), counted(averaged.a)),
        vec4(-F16_MAX),
        vec4(F16_MAX),
    );
    texels[id.y * parameters.size.x + id.x] = vec2(pack2x16float(mean.rg), pack2x16float(mean.ba));
}
