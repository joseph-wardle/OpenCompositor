//! Blurs its source by its Size, a radius in pixels, every channel as it is, so that
//! premultiplied edges fade without an unpremultiply; the data window grows by the reach. Each
//! plane goes through three box passes along its rows into a scratch plane, then along the scratch
//! plane's columns into the output, each axis one line reduction.

use open_ng::Color;
use opencompositor_kernels::{Axis, BoxPass, BoxSum, LineReduce};

use crate::evaluate::{
    EvalError, GpuEvaluation, GpuImage, GpuPlane, Pool, ResultKey, check_node_bytes, output_plane,
};
use crate::image::{Image, ImageInfo, Plane, Window};
use crate::kernels::cpu::{self, Stopped};
use crate::kernels::{Planes, plane_bytes};
use crate::nodes::{Expects, Makes, NodeInput, NodeType};
use crate::parameter::{FloatParameter, Parameter, ParameterKind, Resolved};

pub(super) const NODE_TYPE: NodeType = NodeType {
    name: "blur",
    label: "Blur",
    type_name: "OcBlur",
    step: "Create Blur",
    color: Color {
        r: 100,
        g: 90,
        b: 140,
    },
    parameters: &[Parameter {
        name: "size",
        label: "Size",
        kind: ParameterKind::Float(FloatParameter {
            // A new Blur changes nothing until its size is set.
            default: 0.0,
            slider: 0.0..=100.0,
            limits: 0.0..=4096.0,
        }),
    }],
    inputs: &[NodeInput {
        name: "source",
        label: "Source",
        repeating: false,
        required: true,
        expects: Expects::Premultiplied,
    }],
    makes: Makes::AsInputs,
};

/// Blur's image: its source blurred by `size`, every channel through the same passes, over the
/// source's data window grown by the reach, with the display window and warnings the source's.
/// The empty image, and a size whose reach is zero, give the source itself, planes shared.
///
/// The passes along rows go through a scratch plane of the grown width and the source's height,
/// taken from `released` or made, reused for every plane and returned to `released` after. Each
/// chunk of rows and each band of columns asks `stop` first; `None` once any chunk was skipped.
pub(crate) fn compute(
    pool: &Pool,
    key: &ResultKey,
    source: &Image,
    values: &Resolved,
    released: &mut Vec<Plane>,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<Option<Image>, EvalError> {
    let pass = pass(values);
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
        let planes = PassPlanes {
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

/// The box pass of the node's Size.
fn pass(values: &Resolved) -> BoxPass {
    BoxPass::new(values.float("size").expect("Size has a default"))
}

/// The most bytes one thread's temporaries hold over `input`: the rows pass's or the columns
/// pass's.
pub(crate) fn thread_scratch_bytes(pass: BoxPass, input: Window) -> u64 {
    let op = BoxSum::new(pass);
    let along = |len: u32| cpu::line_scratch_bytes(&op, BoxPass::PASSES, len as usize);
    along(input.width).max(along(input.height))
}

/// One plane's source over the input window, the scratch plane of the grown width and the
/// input's height, and the output over the grown window.
struct PassPlanes<'a> {
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
    planes: PassPlanes<'_>,
    stop: &(dyn Fn() -> bool + Sync),
) -> Result<(), Stopped> {
    let PassPlanes { src, scratch, dst } = planes;
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

/// Records Blur, as [`compute`] computes it: each distinct plane's three box passes
/// along its rows into a scratch plane of the grown width, then along the scratch plane's
/// columns into an output over the data window grown by the reach. The empty image, and a
/// size whose reach is zero, give the source itself.
///
/// The scratch plane and the passes' intermediate planes go back to `released` once
/// recorded, for later nodes of the request.
pub(crate) fn record(
    gpu: &mut GpuEvaluation,
    encoder: &mut wgpu::CommandEncoder,
    key: &ResultKey,
    image: &GpuImage,
    values: &Resolved,
    released: &mut Planes,
) -> Result<GpuImage, EvalError> {
    let pass = pass(values);
    let reach = pass.reach();
    let input = image.info().data_window;
    let Some(planes) = image.planes().filter(|_| reach > 0) else {
        return Ok(image.clone());
    };
    let window = input.grow(reach).map_err(|oversize| EvalError::Oversize {
        origin: key.clone(),
        oversize,
    })?;
    let scratch_window = Window {
        width: window.width,
        ..input
    };
    let op = BoxSum::new(pass);
    let mut sources: Vec<&GpuPlane> = Vec::new();
    for plane in planes {
        if !sources.iter().any(|other| GpuPlane::ptr_eq(other, plane)) {
            sources.push(plane);
        }
    }
    let bytes = plane_bytes(window)
        .saturating_mul(sources.len() as u64)
        .saturating_add(plane_bytes(scratch_window))
        .saturating_add(intermediate_bytes(&op, input, window));
    check_node_bytes(key, window, bytes)?;
    gpu.check_side(key, window)?;
    // The outputs before the scratch plane, as on the CPU.
    let outputs: Vec<GpuPlane> = (sources.iter())
        .map(|_| gpu.plane(released, window))
        .collect();
    let scratch = gpu.plane(released, scratch_window);
    for (source, output) in sources.iter().zip(&outputs) {
        gpu.kernels()
            .line(
                encoder,
                &op,
                Axis::Rows,
                BoxPass::PASSES,
                source.texture(),
                (input.width, input.height),
                scratch.texture(),
                released,
            )
            .expect("a scratch plane of the grown width");
        gpu.kernels()
            .line(
                encoder,
                &op,
                Axis::Columns,
                BoxPass::PASSES,
                scratch.texture(),
                (window.width, input.height),
                output.texture(),
                released,
            )
            .expect("an output over the grown window");
    }
    released.keep(scratch.into_texture());
    let blurred = planes.each_ref().map(|plane| {
        let index = (sources.iter())
            .position(|source| GpuPlane::ptr_eq(source, plane))
            .expect("every plane blurred");
        outputs[index].clone()
    });
    let info = ImageInfo {
        data_window: window,
        ..image.info().clone()
    };
    Ok(GpuImage::new(info, Some(blurred)))
}

/// The bytes of Blur's planes between its first pass and its last along each axis, from
/// `input` to the grown `window`: those of the rows' passes are as tall as the input, and those
/// of the columns' as wide as the window.
fn intermediate_bytes(op: &BoxSum, input: Window, window: Window) -> u64 {
    (1..BoxPass::PASSES)
        .map(|passes| {
            let growth = op.growth() * passes;
            let rows = Window {
                width: input.width + growth,
                ..input
            };
            let columns = Window {
                height: input.height + growth,
                ..window
            };
            plane_bytes(rows).saturating_add(plane_bytes(columns))
        })
        .fold(0, u64::saturating_add)
}
