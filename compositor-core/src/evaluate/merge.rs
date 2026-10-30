//! Merge: the layers composed over the background by premultiplied over, in slot order, and
//! the stack dissolved against the base by Mix, through the fold.

use opencompositor_kernels::Merge;

use crate::evaluate::{EvalError, Pool, ResultKey, check_node_bytes, output_plane};
use crate::image::{Image, Plane, Warning, Window};
use crate::kernels::cpu;

/// Merge's image from its connected inputs: the background if connected, then the layers in
/// slot order, so that the base, the lowest connected input, comes first.
///
/// The display window is the base's and the data window the union of the inputs', with each
/// input transparent black outside its own.
pub(super) fn merge(
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

/// Whether Merge of a base and `layers` layers at `mix` is the base unchanged: the base alone,
/// or Mix 0. Special cases, since the kernel's products by zero would turn an infinite operand
/// into NaN where the artist asked for the base unchanged. Mix is compared as the kernel
/// receives it, so a Mix within f32 rounding of 0 or 1 is the case.
pub(super) fn is_the_base(layers: usize, mix: f32) -> bool {
    layers == 0 || mix == 0.0
}

/// The data window Merge writes, the union of its inputs', checked against what an image can
/// hold and the cap on one node's allocation before anything is allocated.
pub(super) fn merged_window(
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
pub(super) fn merged_warnings<'a>(inputs: impl Iterator<Item = &'a [Warning]>) -> Vec<Warning> {
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
