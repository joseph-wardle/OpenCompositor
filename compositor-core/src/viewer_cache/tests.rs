use std::path::PathBuf;

use open_ng::Point;

use super::*;
use crate::fixtures::fitted;
use crate::kernels::headless;
use crate::session::Session;

/// The key of a Reader of the path, which need not exist.
fn key(name: &str) -> ResultKey {
    let mut session = Session::new();
    let path = PathBuf::from(format!("/nonexistent/{name}.exr"));
    let changes = session.create_readers(vec![path], Point::default(), None);
    let id = *changes.nodes.first().unwrap();
    session.result_key(id).unwrap()
}

fn window(x: i32, y: i32, width: u32, height: u32) -> Window {
    Window {
        x,
        y,
        width,
        height,
    }
}

fn info(display: Window, data: Window) -> ImageInfo {
    ImageInfo {
        display_window: display,
        data_window: data,
        warnings: Vec::new(),
    }
}

/// A copy halved that many times of a `width` × `height` image, its data window its display
/// window, with every texel zero.
fn copy_of(key: ResultKey, width: u32, height: u32, halvings: u32) -> ViewerCopy {
    let whole = window(0, 0, width, height);
    let planes = vec![0.0; (width * height) as usize];
    let pool = Pool::new();
    ViewerCopy::of_planes(key, info(whole, whole), [&planes[..]; 4], halvings, &pool)
}

/// The copy's texels as computed from the definition: each texel the mean over its block of
/// every pixel, those outside the data window and NaN as zero, then within f16's range.
fn by_definition(data: Window, planes: [&[f32]; 4], halvings: u32) -> Vec<f16> {
    let copy = copy_window(data, halvings);
    let side = 1i64 << halvings;
    let mut texels = Vec::new();
    for texel_y in 0..i64::from(copy.height) {
        for texel_x in 0..i64::from(copy.width) {
            for plane in planes {
                let mut sum = 0.0f32;
                for y in 0..side {
                    for x in 0..side {
                        let pixel_x = (i64::from(copy.x) + texel_x) * side + x - i64::from(data.x);
                        let pixel_y = (i64::from(copy.y) + texel_y) * side + y - i64::from(data.y);
                        let inside = (0..i64::from(data.width)).contains(&pixel_x)
                            && (0..i64::from(data.height)).contains(&pixel_y);
                        if inside {
                            let value = plane[(pixel_y * i64::from(data.width) + pixel_x) as usize];
                            if !value.is_nan() {
                                sum += value;
                            }
                        }
                    }
                }
                texels.push(f16::from_f32(shown(sum / (side * side) as f32)));
            }
        }
    }
    texels
}

/// Every channel's values, varied, with the values a copy treats specially.
fn values(count: usize, seed: usize) -> Vec<f32> {
    const SPECIAL: [f32; 7] = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e6,
        -1.0e6,
        65504.0,
        -0.0,
    ];
    (0..count)
        .map(|i| {
            let i = i * 7 + seed;
            if i.is_multiple_of(5) {
                SPECIAL[(i / 5) % SPECIAL.len()]
            } else {
                (i % 97) as f32 * 0.37 - 11.0
            }
        })
        .collect()
}

#[test]
fn a_copy_window_covers_the_data_window_with_blocks_aligned_to_the_pixel_grid() {
    assert_eq!(copy_window(window(-3, 1, 5, 2), 1), window(-2, 0, 3, 2));
    assert_eq!(
        copy_window(window(0, 0, 3240, 2160), 1),
        window(0, 0, 1620, 1080)
    );
    assert_eq!(
        copy_window(window(0, 0, 3240, 2160), 3),
        window(0, 0, 405, 270)
    );
    assert_eq!(copy_window(window(-5, -1, 3, 2), 2), window(-2, -1, 2, 2));
    assert_eq!(copy_window(window(7, 9, 4, 4), 0), window(7, 9, 4, 4));
    assert_eq!(copy_window(window(7, 9, 0, 4), 2).width, 0);
}

#[test]
fn a_copy_averages_each_block_with_pixels_outside_the_data_window_and_nan_as_zero() {
    let display = window(0, 0, 8, 8);
    let data = window(-3, 1, 7, 5);
    let planes: Vec<Vec<f32>> = (0..4).map(|seed| values(35, seed)).collect();
    let planes = [0, 1, 2, 3].map(|channel| &planes[channel][..]);
    let pool = Pool::new();
    for halvings in 0..4 {
        let copy = ViewerCopy::of_planes(key("a"), info(display, data), planes, halvings, &pool);
        assert_eq!(copy.window, copy_window(data, halvings));
        let expected = by_definition(data, planes, halvings);
        assert_eq!(copy.texels.len(), expected.len());
        for (i, (actual, expected)) in copy.texels.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{halvings} halvings, value {i}: {actual}, not {expected}"
            );
        }
    }
}

#[test]
fn a_block_of_infinities_of_both_signs_counts_as_zero_and_large_values_clip_to_f16() {
    let data = window(0, 0, 2, 1);
    let red = [f32::INFINITY, f32::NEG_INFINITY];
    let green = [1.0e6, 1.0e6];
    let blue = [-1.0e9, 0.0];
    let alpha = [f32::NAN, 1.0];
    let pool = Pool::new();
    let copy = ViewerCopy::of_planes(
        key("a"),
        info(data, data),
        [&red, &green, &blue, &alpha],
        1,
        &pool,
    );
    let texels: Vec<f32> = copy.texels.iter().map(|value| value.to_f32()).collect();
    // One halving's block is two pixels on a side, its top row outside the data window.
    assert_eq!(texels, [0.0, 65504.0, -65504.0, 0.25]);
}

#[test]
fn a_copy_on_the_gpu_is_the_cpu_copy() {
    let (device, queue) = headless::device();
    let copier = GpuCopier::new(&device);
    let display = window(0, 0, 40, 24);
    let data = window(-3, -2, 37, 19);
    let count = (data.width * data.height) as usize;
    let planes: Vec<Vec<f32>> = (0..4).map(|seed| values(count, seed)).collect();
    // The planes are larger than the data window, as the GPU's may be, beyond it with values
    // that would show if read.
    let (texture_width, texture_height) = (data.width + 5, data.height + 3);
    let textures = planes.iter().map(|plane| {
        let mut texels = vec![1.0e4; (texture_width * texture_height) as usize];
        for (row, values) in plane.chunks(data.width as usize).enumerate() {
            let start = row * texture_width as usize;
            texels[start..start + values.len()].copy_from_slice(values);
        }
        headless::upload(&device, &queue, texture_width, texture_height, &texels)
    });
    let textures: Vec<wgpu::Texture> = textures.collect();
    let textures = [0, 1, 2, 3].map(|channel| &textures[channel]);
    let planes = [0, 1, 2, 3].map(|channel| &planes[channel][..]);
    let pool = Pool::new();
    for halvings in 0..4 {
        let mut encoder = device.create_command_encoder(&Default::default());
        let recorded = copier.record(&device, &mut encoder, textures, data, halvings);
        let recorded = recorded.unwrap().expect("texels to copy");
        queue.submit([encoder.finish()]);
        recorded
            .readback
            .map_async(wgpu::MapMode::Read, .., |result| result.expect("mapped"));
        (device.poll(wgpu::PollType::wait_indefinitely())).expect("the GPU finishes");
        let on_gpu = recorded.read(key("a"), info(display, data)).unwrap();
        let on_cpu = ViewerCopy::of_planes(key("a"), info(display, data), planes, halvings, &pool);
        assert_eq!(on_gpu.window, on_cpu.window);
        assert_eq!(on_gpu.texels.len(), on_cpu.texels.len());
        for (i, (gpu, cpu)) in on_gpu.texels.iter().zip(&on_cpu.texels).enumerate() {
            // WGSL leaves the rounding of a conversion to f16 to the implementation.
            let ulps = (i32::from(gpu.to_bits() as i16) - i32::from(cpu.to_bits() as i16)).abs();
            assert!(
                ulps <= 1 || gpu.to_f32() == cpu.to_f32(),
                "{halvings} halvings, value {i}: {gpu} on the GPU, {cpu} on the CPU"
            );
        }
    }
}

#[test]
fn the_cache_serves_the_most_reduced_copy_that_the_viewer_draws_no_larger_than_its_texels() {
    let mut cache = ViewerCache::default();
    let zoom = fitted(200, 200);
    let a = key("a");
    cache.insert(copy_of(a.clone(), 400, 400, 2));
    assert!(cache.get(&a, zoom).is_none(), "a quarter is drawn larger");
    cache.insert(copy_of(a.clone(), 400, 400, 0));
    assert_eq!(cache.get(&a, zoom).unwrap().halvings(), 0);
    cache.insert(copy_of(a.clone(), 400, 400, 1));
    assert_eq!(cache.get(&a, zoom).unwrap().halvings(), 1);
    assert!(cache.serves(&a, zoom));
    assert!(cache.get(&key("b"), zoom).is_none());
    assert!(!cache.serves(&key("b"), zoom));
    assert!(
        cache.get(&a, fitted(0, 0)).is_none(),
        "without a pane, nothing is drawn"
    );

    // A copy of the same key and halvings replaces the one held.
    let held = cache.bytes();
    cache.insert(copy_of(a.clone(), 400, 400, 1));
    assert_eq!((cache.len(), cache.bytes()), (3, held));
}

#[test]
fn a_full_cache_drops_the_copies_least_recently_requested() {
    let mut cache = ViewerCache::default();
    let zoom = fitted(400, 400);
    let bytes = copy_of(key("a"), 100, 100, 0).bytes();
    cache.set_budget(3 * bytes);
    let [a, b, c, d] = ["a", "b", "c", "d"].map(key);
    for key in [&a, &b, &c] {
        cache.insert(copy_of(key.clone(), 100, 100, 0));
    }
    assert!(cache.get(&a, zoom).is_some());
    cache.insert(copy_of(d.clone(), 100, 100, 0));
    assert_eq!(cache.bytes(), 3 * bytes);
    assert!(!cache.contains(&b), "b was requested least recently");
    for key in [&a, &c, &d] {
        assert!(cache.contains(key));
    }

    // A smaller budget drops what no longer fits, and a copy larger than it is not held.
    cache.set_budget(bytes);
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&d));
    cache.insert(copy_of(a.clone(), 200, 100, 0));
    assert!(!cache.contains(&a));
    assert!(cache.contains(&d));
}
