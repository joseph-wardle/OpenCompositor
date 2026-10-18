use std::sync::Arc;

use super::*;

fn image(planes: [Arc<[f32]>; 4]) -> Image {
    let window = Window {
        x: 0,
        y: 0,
        width: 2,
        height: 1,
    };
    let source = Source::Png(PngSource {
        bit_depth: 8,
        color_type: PngColorType::Rgba,
        srgb: false,
        gamma: None,
        chromaticities: None,
        icc_profile: false,
        cicp: None,
    });
    Image::new(window, window, planes, 1.0, source, Vec::new())
}

fn plane(values: [f32; 2]) -> [Arc<[f32]>; 4] {
    let plane: Arc<[f32]> = Arc::from(values);
    [(); 4].map(|()| plane.clone())
}

// Debug builds check what every producer keeps by flushing as it writes.
#[test]
#[should_panic(expected = "no plane holds a subnormal")]
fn a_new_image_refuses_a_plane_holding_a_subnormal() {
    image(plane([0.0, f32::from_bits(1)]));
}

#[test]
#[should_panic(expected = "no plane holds a subnormal")]
fn with_planes_refuses_a_plane_holding_a_subnormal() {
    image(plane([-0.0, f32::MIN_POSITIVE])).with_planes(plane([0.0, -f32::MIN_POSITIVE / 2.0]));
}
