//! Blur on the CPU: each plane's three box passes along its rows into a scratch plane, then
//! along the scratch plane's columns into the output, which the reach grows on every side,
//! each axis one line reduction.

use opencompositor_kernels::{Axis, BoxPass, BoxSum};

use crate::evaluate::{EvalError, Pool, ResultKey, check_node_bytes, output_plane};
use crate::image::{Image, Plane, Window};
use crate::kernels::cpu::{self, Stopped};

/// Blur's image: its source blurred by `size`, every channel through the same passes, over the
/// source's data window grown by the reach, with the display window and warnings the source's.
/// The empty image, and a size whose reach is zero, give the source itself, planes shared.
///
/// The passes along rows go through a scratch plane of the grown width and the source's height,
/// taken from `released` or made, reused for every plane and returned to `released` after. Each
/// chunk of rows and each band of columns asks `stop` first; `None` once any chunk was skipped.
pub(super) fn blur(
    pool: &Pool,
    key: &ResultKey,
    source: &Image,
    size: f64,
    released: &mut Vec<Plane>,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Option<Image>, EvalError> {
    let pass = BoxPass::new(size);
    let reach = pass.reach();
    let input = source.data_window();
    if reach == 0 || input.is_empty() {
        return Ok(Some(source.clone()));
    }
    let window = input.grow(reach).map_err(|oversize| EvalError::Oversize {
        origin: key.clone(),
        oversize,
    })?;
    let scratch_window = Window {
        width: window.width,
        ..input
    };
    let len = window.area() as usize;
    // A quarter to spare, as dragging Size grows the window a little every few units.
    let capacity = len + len / 4;
    // A plane two channels share, as a grey image's colour planes are, is blurred once.
    let holders = (source.planes().each_ref())
        .map(|plane| (source.planes().iter()).position(|other| Plane::ptr_eq(other, plane)));
    let holders = holders.map(|holder| holder.expect("one of the planes"));
    let distinct = (holders.iter().enumerate())
        .filter(|&(index, &holder)| holder == index)
        .count();
    let bytes = (capacity as u64 * 4)
        .saturating_mul(distinct as u64)
        .saturating_add(scratch_window.plane_bytes(1))
        .saturating_add((pool.threads() as u64).saturating_mul(thread_scratch_bytes(pass, input)));
    check_node_bytes(key, window, bytes)?;
    // The outputs before the scratch plane, which would otherwise take a released output's
    // storage and leave one output to be made anew every edit.
    let mut blurred: [Option<Plane>; 4] = std::array::from_fn(|index| {
        (holders[index] == index).then(|| output_plane(released, len, capacity))
    });
    let scratch_len = scratch_window.area() as usize;
    let mut scratch = output_plane(released, scratch_len, scratch_len);
    for (plane, output) in source.planes().iter().zip(&mut blurred) {
        let Some(output) = output else { continue };
        let planes = Planes {
            src: plane,
            scratch: scratch.get_mut().expect("nothing else holds the scratch"),
            dst: output
                .get_mut()
                .expect("nothing else holds an output plane"),
        };
        if blur_plane(pool, pass, input, planes, stop).is_err() {
            released.push(scratch);
            return Ok(None);
        }
    }
    released.push(scratch);
    let outputs = holders.map(|holder| blurred[holder].clone().expect("a blurred plane"));
    Ok(Some(Image::new(
        source.display_window(),
        window,
        outputs,
        source.warnings().to_vec(),
    )))
}

/// The most bytes one thread's temporaries hold over `input`: the rows pass's or the columns
/// pass's.
pub(super) fn thread_scratch_bytes(pass: BoxPass, input: Window) -> u64 {
    let op = BoxSum::new(pass);
    let along = |len: u32| cpu::line_scratch_bytes(&op, BoxPass::PASSES, len as usize);
    along(input.width).max(along(input.height))
}

/// One plane's source over the input window, the scratch plane of the grown width and the
/// input's height, and the output over the grown window.
struct Planes<'a> {
    src: &'a [f32],
    scratch: &'a mut [f32],
    dst: &'a mut [f32],
}

/// The passes over one plane: the rows of `src` into `scratch`, then the columns of `scratch`
/// into `dst`, each over the pool, leaving the rest undone once `stop` says to.
fn blur_plane(
    pool: &Pool,
    pass: BoxPass,
    input: Window,
    planes: Planes<'_>,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<(), Stopped> {
    let Planes { src, scratch, dst } = planes;
    let op = BoxSum::new(pass);
    let (width, height) = (input.width as usize, input.height as usize);
    let grown = width + 2 * pass.reach() as usize;
    let passes = BoxPass::PASSES;
    cpu::line(
        pool,
        &op,
        Axis::Rows,
        passes,
        src,
        (width, height),
        scratch,
        stop,
    )?;
    cpu::line(
        pool,
        &op,
        Axis::Columns,
        passes,
        scratch,
        (grown, height),
        dst,
        stop,
    )
}
