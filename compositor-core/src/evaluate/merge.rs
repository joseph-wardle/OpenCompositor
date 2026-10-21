//! Merge on the CPU: the layers composed over the background by premultiplied over, in slot
//! order, and the stack dissolved against the base by Mix.

use std::ops::Range;
use std::sync::Arc;

use crate::evaluate::{EvalError, NODE_BYTES_CAP, Pool, ResultKey, output_plane};
use crate::image::{Image, Window};
use crate::kernels::cpu::{self, Level};

/// Merge's image from its connected inputs: the background if connected, then the layers in
/// slot order, so that the base, the lowest connected input, comes first.
///
/// The display window is the base's and the data window the union of the inputs', with each
/// input transparent black outside its own. Every row of the output is made at once: the base
/// copied in, each layer composed over it in turn, then the dissolve, so the passes between
/// meet in the cache rather than in memory.
pub(super) fn merge(
    pool: &Pool,
    key: &ResultKey,
    inputs: &[&Image],
    mix: f64,
    released: &mut Vec<Arc<[f32]>>,
) -> Result<Image, EvalError> {
    let Some((base, layers)) = inputs.split_first() else {
        return Ok(Image::empty());
    };
    // The base alone, or Mix 0, is the base: special cases, since the kernel's products by zero
    // would turn an infinite operand into NaN where the artist asked for the base unchanged.
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
    let bytes = window.area() * (4 * size_of::<f32>()) as u64;
    if bytes > NODE_BYTES_CAP {
        return Err(EvalError::OverCap {
            origin: key.clone(),
            window,
            bytes,
        });
    }
    // Mix 1 is the stack as composed: the kernel's product by zero would make NaN of an
    // infinite base.
    let mix = (mix != 1.0).then_some(mix as f32);
    let width = window.width as usize;
    let mut outputs: [Arc<[f32]>; 4] =
        [(); 4].map(|()| output_plane(released, window.area() as usize));
    let base_placed = Placed::new(base, window);
    let layers_placed: Vec<Placed> = layers
        .iter()
        .map(|layer| Placed::new(layer, window))
        .collect();
    // The base outside its own window, for the dissolve.
    let zeros = vec![0.0f32; width];
    let planes = outputs
        .each_mut()
        .map(|plane| Arc::get_mut(plane).expect("nothing else holds an output plane"));
    pool.fill_planes(planes, width, |level, first_row, mut chunk| {
        let rows = chunk[0].len() / width.max(1);
        for (index, y) in (first_row..first_row + rows).enumerate() {
            let columns = index * width..(index + 1) * width;
            let mut out: [&mut [f32]; 4] =
                chunk.each_mut().map(|plane| &mut plane[columns.clone()]);
            fill_base(base, base_placed.row(y), &mut out);
            for (layer, placed) in layers.iter().zip(&layers_placed) {
                if let Some((input, output)) = placed.row(y) {
                    let layer = layer.planes().each_ref().map(|plane| &plane[input.clone()]);
                    let dst = out.each_mut().map(|plane| &mut plane[output.clone()]);
                    cpu::over(level, layer, dst);
                }
            }
            if let Some(mix) = mix {
                dissolve(level, base, base_placed.row(y), &zeros, &mut out, mix);
            }
        }
    });
    let warnings = inputs
        .iter()
        .flat_map(|input| input.warnings().iter().cloned())
        .collect();
    Ok(Image::new(base.display_window(), window, outputs, warnings))
}

/// Writes the base's row into an output row, and transparent black where the base is not.
fn fill_base(base: &Image, span: Option<(Range<usize>, Range<usize>)>, out: &mut [&mut [f32]; 4]) {
    match span {
        Some((input, output)) => {
            for (plane, dst) in base.planes().iter().zip(out.iter_mut()) {
                dst[..output.start].fill(0.0);
                dst[output.clone()].copy_from_slice(&plane[input.clone()]);
                dst[output.end..].fill(0.0);
            }
        }
        None => {
            for dst in out.iter_mut() {
                dst.fill(0.0);
            }
        }
    }
}

/// Dissolves an output row, the stack as composed, against the base's row, with `zeros` for
/// the base outside its window.
fn dissolve(
    level: Level,
    base: &Image,
    span: Option<(Range<usize>, Range<usize>)>,
    zeros: &[f32],
    out: &mut [&mut [f32]; 4],
    mix: f32,
) {
    let width = out[0].len();
    let (input, output) = span.unwrap_or((0..0, 0..0));
    for (columns, rows) in [
        (0..output.start, None),
        (output.clone(), Some(input)),
        (output.end..width, None),
    ] {
        let base = match &rows {
            Some(rows) => base.planes().each_ref().map(|plane| &plane[rows.clone()]),
            None => [&zeros[..columns.len()]; 4],
        };
        let dst = out.each_mut().map(|plane| &mut plane[columns.clone()]);
        cpu::mix(level, base, dst, mix);
    }
}

/// Where an input lies in the output: its data window in the output's storage, from the
/// output's bottom-left. An empty input lies nowhere.
#[derive(Clone, Copy, Debug)]
struct Placed {
    column: usize,
    row: usize,
    width: usize,
    height: usize,
}

impl Placed {
    fn new(input: &Image, output: Window) -> Placed {
        let window = input.data_window();
        let (column, row) = window.offset_in(output);
        Placed {
            column,
            row,
            width: window.width as usize,
            height: window.height as usize,
        }
    }

    /// At output row `y`, the input's samples of that row and the output columns they cover, or
    /// `None` where the input has no row there.
    fn row(&self, y: usize) -> Option<(Range<usize>, Range<usize>)> {
        let row = y.checked_sub(self.row)?;
        if row >= self.height {
            return None;
        }
        Some((
            row * self.width..(row + 1) * self.width,
            self.column..self.column + self.width,
        ))
    }
}
