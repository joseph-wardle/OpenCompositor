//! Blur on the CPU: each plane's three box passes along its rows into a scratch plane, then
//! along the scratch plane's columns into the output, which the reach grows on every side.

use std::sync::atomic::{AtomicBool, Ordering};

use opencompositor_kernels::BoxPass;

use crate::evaluate::{Columns, EvalError, NODE_BYTES_CAP, Pool, ResultKey, output_plane};
use crate::image::{Image, Plane, Window};
use crate::kernels::cpu::{self, BLUR_BAND, BLUR_ROWS, BoxScratch, Level};

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
    if bytes > NODE_BYTES_CAP {
        return Err(EvalError::OverCap {
            origin: key.clone(),
            window,
            bytes,
        });
    }
    // The outputs before the scratch plane, which would otherwise take a released output's
    // storage and leave one output to be made anew every edit.
    let mut blurred: [Option<Plane>; 4] = std::array::from_fn(|index| {
        (holders[index] == index).then(|| output_plane(released, len, capacity))
    });
    let scratch_len = scratch_window.area() as usize;
    let mut scratch = output_plane(released, scratch_len, scratch_len);
    let skipped = AtomicBool::new(false);
    for (plane, output) in source.planes().iter().zip(&mut blurred) {
        let Some(output) = output else { continue };
        let planes = Planes {
            src: plane,
            scratch: scratch.get_mut().expect("nothing else holds the scratch"),
            dst: output
                .get_mut()
                .expect("nothing else holds an output plane"),
        };
        blur_plane(pool, pass, input, planes, stop, &skipped);
        if skipped.load(Ordering::Relaxed) {
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

/// The most bytes one thread's temporaries hold over `input`: the rows pass's, or the columns
/// pass's with the band gathered and its results.
pub(super) fn thread_scratch_bytes(pass: BoxPass, input: Window) -> u64 {
    let (width, height) = (input.width as usize, input.height as usize);
    let grown = height as u64 + 2 * u64::from(pass.reach());
    let band = (height as u64 + grown) * BLUR_BAND as u64 * 4;
    cpu::box_scratch_bytes(pass, width).max(cpu::box_scratch_bytes(pass, height) + band)
}

/// One plane's source over the input window, the scratch plane of the grown width and the
/// input's height, and the output over the grown window.
struct Planes<'a> {
    src: &'a [f32],
    scratch: &'a mut [f32],
    dst: &'a mut [f32],
}

/// The passes over one plane: the rows of `src` into `scratch`, then the columns of `scratch`
/// into `dst`, each over the pool. Sets `skipped` and leaves the rest undone once `stop` says
/// to.
fn blur_plane(
    pool: &Pool,
    pass: BoxPass,
    input: Window,
    planes: Planes<'_>,
    stop: &(dyn Fn() -> bool + Sync),
    skipped: &AtomicBool,
) {
    let Planes { src, scratch, dst } = planes;
    let width = input.width as usize;
    let height = input.height as usize;
    let grown = width + 2 * pass.reach() as usize;
    pool.fill_rows_with(
        [scratch],
        grown,
        BLUR_ROWS,
        BoxScratch::default,
        |scratch, level, first_row, [dst]| {
            if stop() {
                skipped.store(true, Ordering::Relaxed);
                return;
            }
            let rows = dst.len() / grown;
            let src = &src[first_row * width..(first_row + rows) * width];
            cpu::blur_rows(level, pass, src, width, dst, scratch);
        },
    );
    if skipped.load(Ordering::Relaxed) {
        return;
    }
    let scratch: &[f32] = scratch;
    pool.fill_columns(
        dst,
        grown,
        BLUR_BAND,
        BandScratch::default,
        |state, level, band| {
            if stop() {
                skipped.store(true, Ordering::Relaxed);
                return;
            }
            band_passes(level, pass, scratch, grown, height, band, state);
        },
    );
}

/// A thread's temporaries for the columns: the kernel's, the band gathered from the scratch
/// plane's rows and the band's results.
#[derive(Debug, Default)]
pub struct BandScratch {
    kernel: BoxScratch,
    gathered: Vec<f32>,
    results: Vec<f32>,
}

/// One band's passes: its columns gathered from the `height` rows of `scratch`, `stride` samples
/// long, the kernel over the band, and its results written to the output's rows.
pub fn band_passes(
    level: Level,
    pass: BoxPass,
    scratch: &[f32],
    stride: usize,
    height: usize,
    mut band: Columns<'_>,
    state: &mut BandScratch,
) {
    let columns = band.columns();
    let lanes = columns.len();
    state.gathered.resize(height * BLUR_BAND, 0.0);
    let (gathered, _) = state.gathered.as_chunks_mut::<BLUR_BAND>();
    for (y, row) in gathered.iter_mut().enumerate() {
        row[..lanes]
            .copy_from_slice(&scratch[y * stride + columns.start..y * stride + columns.end]);
        // The last band's missing lanes, which may hold an earlier band's samples.
        row[lanes..].fill(0.0);
    }
    let grown = height + 2 * pass.reach() as usize;
    state.results.resize(grown * BLUR_BAND, 0.0);
    cpu::blur_band(
        level,
        pass,
        &state.gathered,
        &mut state.results,
        &mut state.kernel,
    );
    let (results, _) = state.results.as_chunks::<BLUR_BAND>();
    for (y, row) in results.iter().enumerate() {
        band.write_row(y, &row[..lanes]);
    }
}
