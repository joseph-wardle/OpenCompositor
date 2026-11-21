//! Composes its layers over its background by premultiplied over, each layer in slot order over
//! the result so far, then dissolves that stack by its Mix against the base, the lowest connected
//! input.

use open_ng::Color;
use opencompositor_kernels::Merge;

use crate::evaluate::{
    EvalError, GpuEvaluation, GpuImage, GpuPlane, Pool, ResultKey, check_node_bytes, output_plane,
};
use crate::image::{Image, ImageInfo, Plane, Warning, Window};
use crate::kernels::{Planes, cpu, plane_bytes};
use crate::nodes::{NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "merge",
    label: "Merge",
    type_name: "OcMerge",
    step: "Create Merge",
    color: Color {
        r: 150,
        g: 105,
        b: 60,
    },
    parameters: &[Parameter {
        name: "mix",
        label: "Mix",
        kind: ParameterKind::Float(FloatParameter {
            default: 1.0,
            slider: 0.0..=1.0,
            limits: 0.0..=1.0,
        }),
    }],
    // Neither is required: the base is the lowest connected input, whichever that is, and nothing
    // connected gives the empty image.
    inputs: &[
        NodeInput {
            name: "background",
            label: "Background",
            repeating: false,
            required: false,
        },
        NodeInput {
            name: "layer",
            label: "Layer",
            repeating: true,
            required: false,
        },
    ],
};

/// Merge's image from its connected inputs: the background if connected, then the layers in
/// slot order, so that the base, the lowest connected input, comes first.
///
/// The display window is the base's and the data window the union of the inputs', with each
/// input transparent black outside its own.
pub(crate) fn compute(
    pool: &Pool,
    key: &ResultKey,
    inputs: &[&Image],
    mix: f32,
    released: &mut Vec<Plane>,
) -> Result<Image, EvalError> {
    let Some((base, layers)) = inputs.split_first() else {
        return Ok(Image::empty());
    };
    if is_the_base(layers.len(), mix) {
        return Ok((*base).clone());
    }
    let window = merged_window(key, inputs.iter().map(|input| input.data_window()))?;
    let len = window.area() as usize;
    let mut outputs: [Plane; 4] = [(); 4].map(|()| output_plane(released, len, len));
    let stack: Vec<([&[f32]; 4], Window)> = (layers.iter())
        .map(|layer| (planes_of(layer), layer.data_window()))
        .collect();
    cpu::fold(
        pool,
        &Merge { mix },
        (planes_of(base), base.data_window()),
        &stack,
        window,
        outputs
            .each_mut()
            .map(|plane| plane.get_mut().expect("nothing else holds an output plane")),
    );
    let warnings = merged_warnings(inputs.iter().map(|input| input.warnings()));
    Ok(Image::new(base.display_window(), window, outputs, warnings))
}

/// Records what [`compute`] computes: the fold over the connected inputs, the base first, into
/// four planes over their union, or the base itself where the CPU returns it unchanged.
pub(crate) fn record(
    gpu: &mut GpuEvaluation,
    encoder: &mut wgpu::CommandEncoder,
    key: &ResultKey,
    inputs: &[&GpuImage],
    mix: f32,
    released: &mut Planes,
) -> Result<GpuImage, EvalError> {
    let Some((base, layers)) = inputs.split_first() else {
        return Ok(GpuImage::empty());
    };
    if is_the_base(layers.len(), mix) {
        return Ok((*base).clone());
    }
    let window = merged_window(key, inputs.iter().map(|input| input.info().data_window))?;
    let info = ImageInfo {
        display_window: base.info().display_window,
        data_window: window,
        warnings: merged_warnings(inputs.iter().map(|input| &input.info().warnings[..])),
    };
    if window.is_empty() {
        return Ok(GpuImage::new(info, None));
    }
    // The outputs, and the stack between layers: the planes the last layer's dispatch reads
    // and those it writes, each four over the union.
    let stacks = (layers.len() - 1).min(2) as u64;
    check_node_bytes(
        key,
        window,
        plane_bytes(window).saturating_mul(4 * (1 + stacks)),
    )?;
    gpu.check_side(key, window)?;
    // An empty input has no textures, and covers nothing, so its planes are never read; the
    // fold still binds four, another input's.
    let stand_in = (inputs.iter())
        .find_map(|input| input.planes())
        .expect("an input over the union");
    fn textures_of<'a>(image: &'a GpuImage, stand_in: &'a [GpuPlane; 4]) -> [&'a wgpu::Texture; 4] {
        (image.planes().unwrap_or(stand_in))
            .each_ref()
            .map(GpuPlane::texture)
    }
    let outputs: [GpuPlane; 4] = [(); 4].map(|()| gpu.plane(released, window));
    let stack: Vec<([&wgpu::Texture; 4], Window)> = (layers.iter())
        .map(|layer| (textures_of(layer, stand_in), layer.info().data_window))
        .collect();
    gpu.kernels()
        .fold(
            encoder,
            &Merge { mix },
            (textures_of(base, stand_in), base.info().data_window),
            &stack,
            window,
            outputs.each_ref().map(GpuPlane::texture),
            released,
        )
        .expect("planes over their windows");
    Ok(GpuImage::new(info, Some(outputs)))
}

/// Whether Merge of a base and `layers` layers at `mix` is the base unchanged: the base alone,
/// or Mix 0. Special cases, since the kernel's products by zero would turn an infinite operand
/// into NaN where the artist asked for the base unchanged. Mix is compared as the kernel
/// receives it, so a Mix within f32 rounding of 0 or 1 is the case.
fn is_the_base(layers: usize, mix: f32) -> bool {
    layers == 0 || mix == 0.0
}

/// The data window Merge writes, the union of its inputs', checked against what an image can
/// hold and the cap on one node's allocation before anything is allocated.
fn merged_window(
    key: &ResultKey,
    mut windows: impl Iterator<Item = Window>,
) -> Result<Window, EvalError> {
    let base = windows.next().expect("Merge has a base");
    let window =
        (windows.try_fold(base, Window::union)).map_err(|oversize| EvalError::Oversize {
            origin: key.clone(),
            oversize,
        })?;
    check_node_bytes(key, window, window.plane_bytes(4))?;
    Ok(window)
}

/// The inputs' warnings, each once: a diamond brings one file's warnings by every branch.
fn merged_warnings<'a>(inputs: impl Iterator<Item = &'a [Warning]>) -> Vec<Warning> {
    let mut warnings: Vec<Warning> = Vec::new();
    for warning in inputs.flatten() {
        if !warnings.contains(warning) {
            warnings.push(warning.clone());
        }
    }
    warnings
}

/// An image's four planes as the fold reads them.
fn planes_of(image: &Image) -> [&[f32]; 4] {
    image.planes().each_ref().map(|plane| &plane[..])
}
