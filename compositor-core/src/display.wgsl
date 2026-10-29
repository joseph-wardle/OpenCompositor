// The viewer's display copy of a result: its four planes into one half-float RGBA texture, rows in
// the planes' order, bottom first.

@group(0) @binding(0) var red: texture_2d<f32>;
@group(0) @binding(1) var green: texture_2d<f32>;
@group(0) @binding(2) var blue: texture_2d<f32>;
@group(0) @binding(3) var alpha: texture_2d<f32>;
@group(0) @binding(4) var display: texture_storage_2d<rgba16float, write>;

// The value rounded toward zero to one a half float holds, its magnitude at most the largest
// finite half, 65504, and a NaN kept. The store then converts it exactly, so its own rounding,
// which Vulkan leaves to the implementation, never applies. Only integer operations touch the
// value, so a device that flushes subnormal floats cannot change a sign.
fn to_half(value: f32) -> f32 {
    let bits = bitcast<u32>(value);
    let magnitude = bits & 0x7fffffffu;
    let sign = bits & 0x80000000u;
    if magnitude > 0x7f800000u {
        return value;
    }
    if magnitude >= 0x47800000u {
        return bitcast<f32>(sign | 0x477fe000u);
    }
    let exponent = i32(magnitude >> 23u) - 127;
    if exponent < -24 {
        return bitcast<f32>(sign);
    }
    // A normal half keeps ten of the 23 fraction bits; a subnormal one is a multiple of 2^-24.
    let dropped = u32(max(13, -1 - exponent));
    return bitcast<f32>(bits & ~((1u << dropped) - 1u));
}

@compute @workgroup_size(16, 16)
fn convert(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(display);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let texel = vec2<i32>(id.xy);
    let colour = vec4(
        textureLoad(red, texel, 0).r,
        textureLoad(green, texel, 0).r,
        textureLoad(blue, texel, 0).r,
        textureLoad(alpha, texel, 0).r,
    );
    textureStore(
        display,
        texel,
        vec4(to_half(colour.r), to_half(colour.g), to_half(colour.b), to_half(colour.a)),
    );
}
