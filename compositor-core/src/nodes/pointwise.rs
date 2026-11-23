//! The host of every pointwise node, on both devices: a node says which of the source's
//! channels each of its maps reads and which it writes, and this runs them.

use std::iter::zip;

use opencompositor_kernels::Map;

use crate::evaluate::{GpuEvaluation, GpuImage, GpuPlane, Pool, output_plane};
use crate::image::{Channel, Image, Plane};
use crate::kernels::{Planes, cpu};

/// One map of a pointwise node: `op` over the source's `reads`, its outputs the image's
/// `writes`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChannelMap<M, const I: usize, const O: usize> {
    pub(crate) op: M,
    pub(crate) reads: [Channel; I],
    pub(crate) writes: [Channel; O],
}

/// The source through `maps`, over its data window: each channel a map writes is that map's
/// output, in a plane from `released` or a new one, and each other channel the source's plane.
/// The source is never written.
pub(crate) fn compute<M: Map<I, O>, const I: usize, const O: usize>(
    pool: &Pool,
    image: &Image,
    maps: &[ChannelMap<M, I, O>],
    released: &mut Vec<Plane>,
) -> Image {
    let window = image.data_window();
    let planes = planes(image.planes(), maps, Plane::ptr_eq, |op, reads| {
        // As roomy as the roomiest input, so that below a Blur whose Size is dragged it grows as
        // rarely.
        let capacity = reads.iter().map(|plane| plane.capacity()).max();
        let len = window.area() as usize;
        let mut outputs: [Plane; O] =
            std::array::from_fn(|_| output_plane(released, len, capacity.unwrap_or(len)));
        let dsts = (outputs.each_mut())
            .map(|plane| plane.get_mut().expect("nothing else holds an output plane"));
        cpu::map(
            pool,
            op,
            reads.map(|plane| (&plane[..], window)),
            window,
            dsts,
        );
        outputs
    });
    image.with_planes(planes)
}

/// Records what [`compute`] computes. An image without planes, the empty one, is its own result.
pub(crate) fn record<M: Map<I, O>, const I: usize, const O: usize>(
    gpu: &mut GpuEvaluation,
    encoder: &mut wgpu::CommandEncoder,
    image: &GpuImage,
    maps: &[ChannelMap<M, I, O>],
    released: &mut Planes,
) -> GpuImage {
    let Some(source) = image.planes() else {
        return image.clone();
    };
    let window = image.info().data_window;
    let planes = planes(source, maps, GpuPlane::ptr_eq, |op, reads| {
        let outputs: [GpuPlane; O] = std::array::from_fn(|_| gpu.plane(released, window));
        let inputs = reads.map(|plane| (plane.texture(), window));
        (gpu.kernels())
            .map(
                encoder,
                op,
                inputs,
                window,
                outputs.each_ref().map(GpuPlane::texture),
            )
            .expect("planes over the data window");
        outputs
    });
    image.with_planes(planes)
}

/// The image's planes, in `Channel::ALL`'s order, after `maps`, `run` computing one map's
/// outputs from the planes it reads. A map that is the same work as an earlier one, its op's
/// bytes equal and the planes it reads the same storage, takes the earlier one's outputs: a grey
/// image's colour planes stay one plane. Its bytes, not its numbers, since a gain of -0 is equal
/// to a gain of 0 and gives the other sign of zero.
///
/// Panics, failing the node, if two maps write one channel.
fn planes<P: Clone, M: Map<I, O>, const I: usize, const O: usize>(
    source: &[P; 4],
    maps: &[ChannelMap<M, I, O>],
    same_storage: fn(&P, &P) -> bool,
    mut run: impl FnMut(&M, [&P; I]) -> [P; O],
) -> [P; 4] {
    let read = |channel: Channel| &source[channel as usize];
    let mut done: Vec<(&ChannelMap<M, I, O>, [P; O])> = Vec::with_capacity(maps.len());
    let mut planes = source.clone();
    let mut written = [false; 4];
    for map in maps {
        let earlier = done.iter().find(|(other, _)| {
            other.op.bytes() == map.op.bytes()
                && zip(other.reads, map.reads).all(|(a, b)| same_storage(read(a), read(b)))
        });
        let outputs = match earlier {
            Some((_, outputs)) => outputs.clone(),
            None => {
                let outputs = run(&map.op, map.reads.map(read));
                done.push((map, outputs.clone()));
                outputs
            }
        };
        for (channel, plane) in zip(map.writes, outputs) {
            assert!(
                !written[channel as usize],
                "two maps write {}",
                channel.name()
            );
            written[channel as usize] = true;
            planes[channel as usize] = plane;
        }
    }
    planes
}
