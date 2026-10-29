//! Merge: the layers composed over the background by premultiplied over, in slot order, and
//! the stack dissolved against the base by Mix, through the fold.

use opencompositor_kernels::Merge;

use crate::evaluate::{EvalError, NODE_BYTES_CAP, Pool, ResultKey, output_plane};
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
    // The base alone, or Mix 0, is the base: special cases, since the kernel's products by zero
    // would turn an infinite operand into NaN where the artist asked for the base unchanged. Mix
    // is compared as the kernel receives it, so a Mix within f32 rounding of 0 or 1 is the case.
    if layers.is_empty() || mix == 0.0 {
        return Ok((*base).clone());
    }
    let mut window = base.data_window();
    for layer in layers {
        window = (window.union(layer.data_window())).map_err(|oversize| EvalError::Oversize {
            origin: key.clone(),
            oversize,
        })?;
    }
    let bytes = window.plane_bytes(4);
    if bytes > NODE_BYTES_CAP {
        return Err(EvalError::OverCap {
            origin: key.clone(),
            window,
            bytes,
        });
    }
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
    // Each once: a diamond brings one file's warnings by every branch.
    let mut warnings: Vec<Warning> = Vec::new();
    for warning in inputs.iter().flat_map(|input| input.warnings()) {
        if !warnings.contains(warning) {
            warnings.push(warning.clone());
        }
    }
    Ok(Image::new(base.display_window(), window, outputs, warnings))
}

/// An image's four planes as the fold reads them.
fn planes_of(image: &Image) -> [&[f32]; 4] {
    image.planes().each_ref().map(|plane| &plane[..])
}
