//! Multiplies R, G and B by two to the power of its Stops and leaves A. On premultiplied pixels
//! that equals unpremultiplying, scaling and premultiplying again.

use open_ng::Color;
use opencompositor_kernels::Exposure;

use crate::evaluate::{GpuEvaluation, GpuImage, GpuPlane, Pool, output_plane};
use crate::image::{Image, Plane};
use crate::kernels::{Planes, cpu};
use crate::nodes::{NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "exposure",
    label: "Exposure",
    type_name: "OcExposure",
    step: "Create Exposure",
    color: Color {
        r: 90,
        g: 130,
        b: 80,
    },
    parameters: &[Parameter {
        name: "stops",
        label: "Stops",
        kind: ParameterKind::Float(FloatParameter {
            default: 0.0,
            slider: -6.0..=6.0,
            // Keeps the gain, two to these powers, many orders of magnitude inside f32's range.
            limits: -64.0..=64.0,
        }),
    }],
    inputs: &[NodeInput {
        name: "source",
        label: "Source",
        repeating: false,
        required: true,
    }],
};

/// Each colour plane through the map, and A shared. Inputs are never written: an output is a
/// plane from `released` or a new one, and A is the input's.
pub(crate) fn compute(pool: &Pool, image: &Image, stops: f64, released: &mut Vec<Plane>) -> Image {
    // Exact for whole stops, and far inside f32's range at the limits.
    let op = Exposure {
        gain: stops.exp2() as f32,
    };
    let window = image.data_window();
    let [r, g, b, a] = image.planes();
    let mut scaled = |plane: &Plane| -> Plane {
        // As roomy as its input, so that below a Blur whose Size is dragged it grows as rarely.
        let mut output = output_plane(released, plane.len(), plane.capacity());
        let dst = output
            .get_mut()
            .expect("nothing else holds an output plane");
        cpu::map(pool, &op, [(plane, window)], window, [dst]);
        output
    };
    // A grey image's colour planes are one plane, and stay one.
    let red = scaled(r);
    let green = if Plane::ptr_eq(g, r) {
        red.clone()
    } else {
        scaled(g)
    };
    let blue = if Plane::ptr_eq(b, r) {
        red.clone()
    } else if Plane::ptr_eq(b, g) {
        green.clone()
    } else {
        scaled(b)
    };
    image.with_planes([red, green, blue, a.clone()])
}

/// Records what [`compute`] computes: R, G and B each through the map, planes the channels share
/// staying shared, and A the input's.
pub(crate) fn record(
    gpu: &mut GpuEvaluation,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    stops: f64,
    released: &mut Planes,
) -> GpuImage {
    let Some([r, g, b, a]) = image.planes() else {
        return image.clone();
    };
    // The host's gain, as the CPU computes it.
    let op = Exposure {
        gain: stops.exp2() as f32,
    };
    let window = image.info().data_window;
    let mut scaled = |plane: &GpuPlane| -> GpuPlane {
        let output = gpu.plane(released, window);
        gpu.kernels()
            .map(
                encoder,
                &op,
                [(plane.texture(), window)],
                window,
                [output.texture()],
            )
            .expect("planes over the data window");
        output
    };
    let red = scaled(r);
    let green = if GpuPlane::ptr_eq(g, r) {
        red.clone()
    } else {
        scaled(g)
    };
    let blue = if GpuPlane::ptr_eq(b, r) {
        red.clone()
    } else if GpuPlane::ptr_eq(b, g) {
        green.clone()
    } else {
        scaled(b)
    };
    image.with_planes([red, green, blue, a.clone()])
}
