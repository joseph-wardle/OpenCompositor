//! The kernels as compositor-core runs them: the CPU shells in [`cpu`], and here the GPU
//! pipelines, one per operation over the entry points the kernel crate compiled to SPIR-V,
//! dispatched over the evaluator's per-plane R32Float textures.
//!
//! A dispatch records into the caller's encoder and never submits, so one submission can carry
//! an evaluation's dispatches and copies together. Nothing here uploads or reads back: the tests
//! do, as test support until the worker stages the GPU's inputs. Over has no production dispatch
//! yet: its entry point establishes that naga accepts the module and that the numbers land
//! within its bound, ahead of the GPU step.

use std::fmt;

use opencompositor_kernels::{
    EXPOSURE_ENTRY, ExposureParams, OVER_ENTRY, OverParams, SPIRV, WORKGROUP,
};

pub(crate) mod cpu;

/// The format of every plane texture.
pub(crate) const PLANE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// A pointwise entry point's pipeline on one device, made once per device: a uniform of
/// parameters, its input planes read as textures, and one plane written as a storage texture,
/// bound in that order as the kernel crate declares them.
#[derive(Debug)]
struct Pointwise {
    device: wgpu::Device,
    label: &'static str,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params_size: u64,
}

impl Pointwise {
    /// Makes the pipeline, which naga validates, for an entry point reading `inputs` planes.
    fn new(
        device: &wgpu::Device,
        label: &'static str,
        entry: &str,
        inputs: u32,
        params_size: u64,
    ) -> Pointwise {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernels"),
            source: wgpu::util::make_spirv(SPIRV),
        });
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(params_size),
            },
            count: None,
        }];
        entries.extend((1..=inputs).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }));
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: inputs + 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                // rust-gpu declares a storage image readable and writable, and naga holds the
                // layout to the same; R32Float may be bound read-write without any feature.
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: PLANE_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Pointwise {
            device: device.clone(),
            label,
            layout,
            pipeline,
            params_size,
        }
    }

    /// Records the entry point with `params` over `inputs` into `output`, planes of one size,
    /// into the encoder. Workgroups cover the planes rounded up to whole ones; the entry point
    /// skips the invocations past the edge.
    ///
    /// The planes are checked here rather than left to wgpu, whose validation error would count
    /// as a bug on the device.
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &[u8],
        inputs: &[&wgpu::Texture],
        output: &wgpu::Texture,
    ) -> Result<(), PlaneError> {
        assert_eq!(params.len() as u64, self.params_size, "the params' bytes");
        for &input in inputs {
            check_plane(Role::Input, input, wgpu::TextureUsages::TEXTURE_BINDING)?;
            if input.size() != output.size() {
                return Err(PlaneError::Size {
                    input: input.size(),
                    output: output.size(),
                });
            }
        }
        check_plane(Role::Output, output, wgpu::TextureUsages::STORAGE_BINDING)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: self.params_size,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("a buffer mapped at creation")
            .slice(..)
            .copy_from_slice(params);
        buffer.unmap();
        let views: Vec<wgpu::TextureView> = (inputs.iter().chain([&output]))
            .map(|texture| texture.create_view(&Default::default()))
            .collect();
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        entries.extend(
            (views.iter().enumerate()).map(|(index, view)| wgpu::BindGroupEntry {
                binding: index as u32 + 1,
                resource: wgpu::BindingResource::TextureView(view),
            }),
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(self.label),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            output.width().div_ceil(WORKGROUP),
            output.height().div_ceil(WORKGROUP),
            1,
        );
        Ok(())
    }
}

/// Exposure's pipeline on one device.
#[derive(Debug)]
pub(crate) struct Exposure(Pointwise);

impl Exposure {
    pub(crate) fn new(device: &wgpu::Device) -> Exposure {
        let params_size = size_of::<ExposureParams>() as u64;
        Exposure(Pointwise::new(
            device,
            "exposure",
            EXPOSURE_ENTRY,
            1,
            params_size,
        ))
    }

    /// Records Exposure by `gain` over `input` into `output`, two planes of one size, into the
    /// encoder.
    pub(crate) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Texture,
        output: &wgpu::Texture,
        gain: f32,
    ) -> Result<(), PlaneError> {
        let params = ExposureParams {
            width: output.width(),
            height: output.height(),
            gain,
        };
        self.0.dispatch(encoder, &params.bytes(), &[input], output)
    }
}

/// Over's pipeline on one device.
#[derive(Debug)]
pub(crate) struct Over(Pointwise);

impl Over {
    pub(crate) fn new(device: &wgpu::Device) -> Over {
        let params_size = size_of::<OverParams>() as u64;
        Over(Pointwise::new(device, "over", OVER_ENTRY, 3, params_size))
    }

    /// Records one plane of `layer` over the same plane of `background`, with the layer's
    /// `alpha` plane, into `output`, four planes of one size, into the encoder. For the alpha
    /// plane itself, `layer` is `alpha`.
    pub(crate) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        background: &wgpu::Texture,
        layer: &wgpu::Texture,
        alpha: &wgpu::Texture,
        output: &wgpu::Texture,
    ) -> Result<(), PlaneError> {
        let params = OverParams {
            width: output.width(),
            height: output.height(),
        };
        let inputs = [background, layer, alpha];
        self.0.dispatch(encoder, &params.bytes(), &inputs, output)
    }
}

fn check_plane(
    role: Role,
    texture: &wgpu::Texture,
    needed: wgpu::TextureUsages,
) -> Result<(), PlaneError> {
    if texture.format() != PLANE_FORMAT {
        return Err(PlaneError::Format {
            role,
            format: texture.format(),
        });
    }
    let missing = needed - texture.usage();
    if !missing.is_empty() {
        return Err(PlaneError::Usage { role, missing });
    }
    Ok(())
}

/// Which of a dispatch's planes was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Input,
    Output,
}

/// Why a dispatch refused its planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneError {
    /// The plane is not [`PLANE_FORMAT`].
    Format {
        role: Role,
        format: wgpu::TextureFormat,
    },
    /// The plane was made without the usage its binding needs.
    Usage {
        role: Role,
        missing: wgpu::TextureUsages,
    },
    /// The output is not an input's size.
    Size {
        input: wgpu::Extent3d,
        output: wgpu::Extent3d,
    },
}

impl fmt::Display for PlaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlaneError::Format { role, format } => {
                write!(f, "the {role:?} plane is {format:?}, not {PLANE_FORMAT:?}")
            }
            PlaneError::Usage { role, missing } => {
                write!(f, "the {role:?} plane was made without {missing:?}")
            }
            PlaneError::Size { input, output } => write!(
                f,
                "the output plane is {} by {}, the input {} by {}",
                output.width, output.height, input.width, input.height
            ),
        }
    }
}

impl std::error::Error for PlaneError {}

/// A device on a real adapter, and the upload and readback the kernel tests need: the worker
/// stages differently, through mapped buffers, so these stay test support.
#[cfg(test)]
mod headless {
    use super::PLANE_FORMAT;

    /// A device on the Vulkan adapter `WGPU_ADAPTER_NAME` names, or else the first one, a CPU
    /// renderer included: the tests prove the kernels on whatever Mesa or driver is installed,
    /// and print which, so every log says what ran. No adapter is a failure, never a skip.
    pub(super) fn device() -> (wgpu::Device, wgpu::Queue) {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(name) => {
                let name = name.to_lowercase();
                adapters
                    .into_iter()
                    .find(|adapter| adapter.get_info().name.to_lowercase().contains(&name))
                    .unwrap_or_else(|| panic!("no Vulkan adapter named {name:?}"))
            }
            Err(_) => adapters
                .into_iter()
                .next()
                .expect("a Vulkan adapter; the kernel tests need one"),
        };
        let info = adapter.get_info();
        println!(
            "kernel tests on {} ({:?}), {} {}",
            info.name, info.device_type, info.driver, info.driver_info
        );
        pollster::block_on(adapter.request_device(&Default::default()))
            .expect("a device on the adapter")
    }

    /// A plane the kernels can read, holding `texels` in row-major order.
    pub(super) fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        texels: &[f32],
    ) -> wgpu::Texture {
        assert_eq!(texels.len(), (width * height) as usize);
        let texture = plane(
            device,
            width,
            height,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let bytes: Vec<u8> = texels.iter().flat_map(|v| v.to_ne_bytes()).collect();
        queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: None,
            },
            texture.size(),
        );
        texture
    }

    /// A plane the kernels can write and the test read back.
    pub(super) fn output(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        plane(
            device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )
    }

    pub(super) fn plane(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PLANE_FORMAT,
            usage,
            view_formats: &[],
        })
    }

    /// The plane's texels in row-major order, once the GPU has finished with it. Rows are
    /// copied out padded to the copy alignment and the padding dropped.
    pub(super) fn readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<f32> {
        let (width, height) = (texture.width(), texture.height());
        let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
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
                let (texels, _) = row[..(width * 4) as usize].as_chunks::<4>();
                texels.iter().map(|texel| f32::from_ne_bytes(*texel))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use opencompositor_kernels::ulp_distance;

    use super::*;
    use crate::gpu::{noop_device, noop_parts};

    #[test]
    fn the_module_and_pipelines_validate_without_a_gpu() {
        let device = noop_device(None);
        Exposure::new(&device);
        Over::new(&device);
    }

    #[test]
    fn a_wrong_plane_is_refused_before_anything_is_recorded() {
        let (_, device, _queue) = noop_parts(None);
        let exposure = Exposure::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let readable = wgpu::TextureUsages::TEXTURE_BINDING;
        let writable = wgpu::TextureUsages::STORAGE_BINDING;
        let input = headless::plane(&device, 4, 4, readable);
        let output = headless::plane(&device, 4, 4, writable);

        let rgba = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: input.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: readable,
            view_formats: &[],
        });
        assert_eq!(
            exposure.dispatch(&mut encoder, &rgba, &output, 1.0),
            Err(PlaneError::Format {
                role: Role::Input,
                format: wgpu::TextureFormat::Rgba32Float,
            })
        );

        let unwritable = headless::plane(&device, 4, 4, readable);
        assert_eq!(
            exposure.dispatch(&mut encoder, &input, &unwritable, 1.0),
            Err(PlaneError::Usage {
                role: Role::Output,
                missing: writable,
            })
        );

        let smaller = headless::plane(&device, 4, 3, writable);
        assert_eq!(
            exposure.dispatch(&mut encoder, &input, &smaller, 1.0),
            Err(PlaneError::Size {
                input: input.size(),
                output: smaller.size(),
            })
        );
    }

    /// Every texel distinct in position, with the special values on even texels and normal
    /// magnitudes across the whole range on odd ones.
    fn texels(count: usize) -> Vec<f32> {
        const SPECIAL: [f32; 12] = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            65504.0,
            1.0,
            -1.0,
        ];
        (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    SPECIAL[(i / 2) % SPECIAL.len()]
                } else {
                    let magnitude = (i as f32 * 0.0731 - 30.0).exp2();
                    if i % 4 == 1 { magnitude } else { -magnitude }
                }
            })
            .collect()
    }

    /// Plane sizes: one workgroup's texel, exactly one workgroup, a tail on both axes, and
    /// rows whose padded readback differs from their width.
    const SIZES: [(u32, u32); 4] = [(1, 1), (16, 16), (33, 17), (257, 3)];

    #[test]
    fn exposure_on_the_gpu_is_the_cpu_definition() {
        let (device, queue) = headless::device();
        let exposure = Exposure::new(&device);
        // A quarter-stop gain, rounded like the host's, and one that flushes MIN_POSITIVE.
        let gains = [0.75f64.exp2() as f32, (-10f64).exp2() as f32];
        for (width, height) in SIZES {
            let texels = texels((width * height) as usize);
            let input = headless::upload(&device, &queue, width, height, &texels);
            for gain in gains {
                let output = headless::output(&device, width, height);
                let mut encoder = device.create_command_encoder(&Default::default());
                exposure
                    .dispatch(&mut encoder, &input, &output, gain)
                    .expect("two planes of one size");
                queue.submit([encoder.finish()]);
                let result = headless::readback(&device, &queue, &output);
                assert_eq!(result.len(), texels.len());
                for (i, (&actual, &v)) in result.iter().zip(&texels).enumerate() {
                    let expected = opencompositor_kernels::exposure(v, gain);
                    assert_eq!(
                        ulp_distance(actual, expected),
                        0,
                        "{width}x{height} texel {i}: {v:?} * {gain:?} gave {actual:?}, not {expected:?}"
                    );
                    if expected == 0.0 {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{width}x{height} texel {i}: zero sign"
                        );
                    }
                }
            }
        }
    }

    /// Alphas for the GPU over test: the usual range, beyond it, and the non-finite classes.
    fn alphas(count: usize) -> Vec<f32> {
        const ALPHAS: [f32; 9] = [
            0.0,
            0.5,
            1.0,
            0.25,
            -0.75,
            1.5,
            f32::NAN,
            f32::INFINITY,
            2.0e-20,
        ];
        (0..count)
            .map(|i| ALPHAS[(i * 7 + i / 9) % ALPHAS.len()])
            .collect()
    }

    /// Over on the GPU against the CPU definition, within 1 ULP where the result is not a
    /// cancellation, since a driver that splits the fused multiply-add rounds twice; where the
    /// product and the layer cancel, within the ULP of what was summed. Non-finite results agree
    /// by class.
    #[test]
    fn over_on_the_gpu_is_within_a_ulp_of_the_cpu_definition() {
        let (device, queue) = headless::device();
        let over = Over::new(&device);
        let mut worst = 0;
        for (width, height) in SIZES {
            let count = (width * height) as usize;
            let background = texels(count);
            let mut layer = texels(count);
            layer.rotate_left(1);
            let alpha = alphas(count);
            let planes = [&background, &layer, &alpha]
                .map(|texels| headless::upload(&device, &queue, width, height, texels));
            let [background_plane, layer_plane, alpha_plane] = &planes;
            let output = headless::output(&device, width, height);
            let mut encoder = device.create_command_encoder(&Default::default());
            over.dispatch(
                &mut encoder,
                background_plane,
                layer_plane,
                alpha_plane,
                &output,
            )
            .expect("four planes of one size");
            queue.submit([encoder.finish()]);
            let result = headless::readback(&device, &queue, &output);
            assert_eq!(result.len(), count);
            for (i, &actual) in result.iter().enumerate() {
                let (b, l, a) = (background[i], layer[i], alpha[i]);
                let expected = opencompositor_kernels::over(b, l, a);
                let what = format!("{width}x{height} texel {i}: {l:?} at {a:?} over {b:?}");
                if !expected.is_finite() {
                    assert_eq!(
                        ulp_distance(actual, expected),
                        0,
                        "{what} gave {actual:?}, not {expected:?}"
                    );
                    continue;
                }
                let product = f64::from(b) * f64::from(1.0 - a);
                let summed = product.abs() + f64::from(l).abs();
                if summed <= 4.0 * f64::from(expected).abs() {
                    let distance = ulp_distance(actual, expected);
                    assert!(
                        distance <= 1,
                        "{what} gave {actual:?}, {distance} ULP from {expected:?}"
                    );
                    worst = worst.max(distance);
                } else {
                    let summed_ulp = f64::from(summed as f32).abs() * f64::from(f32::EPSILON);
                    let difference = (f64::from(actual) - f64::from(expected)).abs();
                    assert!(
                        difference <= summed_ulp,
                        "{what} gave {actual:?}, not {expected:?}, with {summed} summed"
                    );
                }
            }
        }
        println!("over on the GPU: at worst {worst} ULP from the CPU");
    }
}
