use half::f16;

use super::*;
use crate::evaluate::Evaluator;
use crate::evaluate::tests::{chain, key, reader};
use crate::evaluate::{GpuEvaluation, GpuImage};
use crate::fixtures::{directory, write_exr};
use crate::kernels::headless;

/// The half float a value's display copy holds: rounded toward zero, clamped at the largest
/// finite half, a NaN any NaN. Found from the nearest half, stepped toward zero where that
/// rounded away from it, rather than as the shader truncates bits.
pub(crate) fn expected(value: f32) -> f16 {
    if value.is_nan() {
        return f16::NAN;
    }
    if value.abs() >= 65504.0 {
        return f16::MAX.copysign(f16::from_f32(value));
    }
    let nearest = f16::from_f32(value);
    if f64::from(nearest.to_f32()).abs() <= f64::from(value).abs() {
        return nearest;
    }
    let toward_zero = f16::from_bits(nearest.to_bits() - 1);
    // Stepping the magnitude down from the smallest subnormal reaches zero of the same sign.
    debug_assert!(toward_zero.to_f32().abs() < value.abs());
    toward_zero
}

/// Values at the edges of what a half holds, and between its neighbours.
fn values(count: usize) -> Vec<f32> {
    let edges = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.0 + 2.0f32.powi(-11),
        1.0 + 3.0 * 2.0f32.powi(-12),
        1.0 / 3.0,
        -1.0 / 3.0,
        65504.0,
        65519.0,
        65520.0,
        -65536.0,
        1e10,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        2.0f32.powi(-14),
        2.0f32.powi(-14) * (1.0 - 2.0f32.powi(-11)),
        2.0f32.powi(-24),
        1.5 * 2.0f32.powi(-24),
        -1.5 * 2.0f32.powi(-24),
        2.0f32.powi(-25),
        -2.0f32.powi(-25),
        f32::MIN_POSITIVE,
        -1e-40,
    ];
    (0..count)
        .map(|i| match edges.get(i) {
            Some(&edge) => edge,
            None => ((i % 600) as f32 * 0.05 - 26.0).exp2() * if i % 2 == 0 { 1.0 } else { -1.0 },
        })
        .collect()
}

/// The display texture's texels as halves, rows in the texture's order.
pub(crate) fn read_halves(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Vec<f16> {
    let (width, height) = (texture.width(), texture.height());
    let bytes_per_row = (width * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |result| result.expect("mapped"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finishes");
    let view = buffer.get_mapped_range(..).expect("mapped");
    view.chunks_exact(bytes_per_row as usize)
        .flat_map(|row| {
            let (halves, _) = row[..(width * 8) as usize].as_chunks::<2>();
            halves.iter().map(|half| f16::from_ne_bytes(*half))
        })
        .collect()
}

/// Converts the image in a submission of its own, as the worker does after its request.
fn convert(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    conversion: &mut Conversion,
    image: &GpuImage,
) -> Option<Arc<wgpu::Texture>> {
    let mut encoder = device.create_command_encoder(&Default::default());
    let texture = conversion.record(&mut encoder, image);
    queue.submit([encoder.finish()]);
    texture
}

/// On the RTX and lavapipe the store itself rounds toward zero, so there only the clamp and NaN
/// tell the shader's rounding from none; a device that rounds to nearest tests the rest.
#[test]
fn the_display_copy_rounds_toward_zero_to_halves_and_clamps() {
    let path = directory("display").join("edges.exr");
    let (width, height) = (9, 7);
    let channels = ["R", "G", "B", "A"]
        .into_iter()
        .enumerate()
        .map(|(rotation, name)| {
            let mut values = values(width * height);
            values.rotate_left(rotation * 5);
            (name, values)
        })
        .collect();
    write_exr(&path, ((0, 0), (9, 7)), (2, -1), (width, height), channels);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let key = key(&chain([reader(&path)]), 1);
    let planes = Evaluator::new().evaluate(&key, || false).unwrap().unwrap();
    let image = Evaluator::new()
        .evaluate_on(&mut gpu, &key, || false)
        .unwrap()
        .unwrap();
    let mut conversion = Conversion::new(&device);

    let texture = convert(&device, &queue, &mut conversion, &image).expect("a data window");

    let halves = read_halves(&device, &queue, &texture);
    assert_eq!(halves.len(), width * height * 4);
    for (i, pixel) in halves.as_chunks::<4>().0.iter().enumerate() {
        for (channel, &actual) in Channel::ALL.into_iter().zip(pixel) {
            let value = planes.plane(channel)[i];
            let expected = expected(value);
            let matches = if expected.is_nan() {
                actual.is_nan()
            } else {
                actual.to_bits() == expected.to_bits()
            };
            assert!(
                matches,
                "{channel:?} of pixel {i}, {value:e}: {actual:e} ({:#06x}), not {expected:e} \
                 ({:#06x})",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }
}

#[test]
fn a_display_texture_is_reused_once_nothing_holds_it() {
    let path = directory("display").join("reuse.exr");
    let channels = ["R", "G", "B", "A"]
        .into_iter()
        .map(|name| (name, vec![0.5; 6]))
        .collect();
    write_exr(&path, ((0, 0), (3, 2)), (0, 0), (3, 2), channels);
    let (device, queue) = headless::device();
    let mut gpu = GpuEvaluation::new(&device, &queue);
    let key = key(&chain([reader(&path)]), 1);
    let image = Evaluator::new()
        .evaluate_on(&mut gpu, &key, || false)
        .unwrap()
        .unwrap();
    let mut conversion = Conversion::new(&device);

    let shown = convert(&device, &queue, &mut conversion, &image).unwrap();
    let arriving = convert(&device, &queue, &mut conversion, &image).unwrap();
    assert!(
        !Arc::ptr_eq(&shown, &arriving),
        "a held texture is not reused"
    );
    let shown_pointer = Arc::as_ptr(&shown);
    drop(shown);
    let next = convert(&device, &queue, &mut conversion, &image).unwrap();

    assert_eq!(Arc::as_ptr(&next), shown_pointer);
    assert_eq!(conversion.textures.len(), 2);
}
