//! The primitives as compositor-core runs them: their CPU halves in [`cpu`], and here their GPU
//! halves, recording an operation's entry points from the module the kernel crate compiled to
//! SPIR-V over the evaluator's per-plane R32Float textures.
//!
//! Recording goes into the caller's encoder and never submits, so one submission can carry an
//! evaluation's dispatches and copies together. Nothing here uploads or reads back: the
//! evaluator's GPU half uploads, and the tests read back.

use std::collections::HashMap;
use std::fmt;

use opencompositor_kernels::{
    Axis, DIRECT_T, Fold, FoldParams, LineParams, LineReduce, Map, MapParams, Params, Placement,
    SCAN_BLOCK, SPIRV, WORKGROUP,
};

use crate::image::Window;

pub(crate) mod cpu;

/// The format of every plane texture.
pub(crate) const PLANE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The primitives on one device: the module, and each entry point's pipeline, built the first
/// time an operation is recorded.
#[derive(Debug)]
pub(crate) struct Kernels {
    device: wgpu::Device,
    module: wgpu::ShaderModule,
    pipelines: HashMap<&'static str, Pipeline>,
}

impl Kernels {
    /// Loads the module, which naga validates as each pipeline is built.
    pub(crate) fn new(device: &wgpu::Device) -> Kernels {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("kernels"),
            source: wgpu::util::make_spirv(SPIRV),
        });
        Kernels {
            device: device.clone(),
            module,
            pipelines: HashMap::new(),
        }
    }

    /// Records `op` over `inputs`, each a plane over its window, into `outputs`, planes over
    /// `output`, one invocation per texel. An input is zero where its window does not reach.
    pub(crate) fn map<M: Map<I, O>, const I: usize, const O: usize>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        op: &M,
        inputs: [(&wgpu::Texture, Window); I],
        output: Window,
        outputs: [&wgpu::Texture; O],
    ) -> Result<(), PlaneError> {
        for (plane, window) in inputs {
            check_input(plane, window)?;
        }
        for plane in outputs {
            check_output(plane, output)?;
        }
        let params = MapParams {
            places: inputs.map(|(_, window)| place(window, output)),
            width: output.width,
            height: output.height,
            op: *op,
        };
        let textures = inputs.map(|(plane, _)| plane);
        let groups = per_texel(output);
        self.dispatch(encoder, M::ENTRY, &params, &textures, &outputs, groups);
        Ok(())
    }

    /// Records `op` over a stack of `layers` on `base`, each planes over a window, into
    /// `outputs`, planes over `output`: a dispatch per layer, each onto the last's planes, the
    /// last with the finish. Every layer but the last writes planes taken from `planes`, which
    /// get them back once recorded. There is at least one layer.
    #[expect(
        clippy::too_many_arguments,
        reason = "a stack, its output and its scratch"
    )]
    pub(crate) fn fold<F: Fold<G, O>, const G: usize, const O: usize>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        op: &F,
        base: ([&wgpu::Texture; O], Window),
        layers: &[([&wgpu::Texture; G], Window)],
        output: Window,
        outputs: [&wgpu::Texture; O],
        planes: &mut Planes,
    ) -> Result<(), PlaneError> {
        assert!(!layers.is_empty(), "a fold has a layer");
        for (textures, window) in std::iter::once((&base.0[..], base.1)).chain(
            layers
                .iter()
                .map(|(textures, window)| (&textures[..], *window)),
        ) {
            for plane in textures {
                check_input(plane, window)?;
            }
        }
        for plane in outputs {
            check_output(plane, output)?;
        }
        // The stack below the next layer, the base's planes until a layer has been recorded.
        let mut below: Option<([wgpu::Texture; O], Window)> = None;
        for (index, (layer, window)) in layers.iter().enumerate() {
            let (below_planes, below_window) = match &below {
                Some((planes, window)) => (planes.each_ref(), *window),
                None => base,
            };
            let params = FoldParams {
                base: place(base.1, output),
                below: place(below_window, output),
                layer: place(*window, output),
                width: output.width,
                height: output.height,
                op: *op,
            };
            let groups = per_texel(output);
            if index + 1 == layers.len() {
                let textures: Vec<&wgpu::Texture> =
                    (base.0.iter().chain(&below_planes).chain(layer))
                        .copied()
                        .collect();
                self.dispatch(encoder, F::LAST, &params, &textures, &outputs, groups);
            } else {
                let textures: Vec<&wgpu::Texture> =
                    (below_planes.iter().chain(layer)).copied().collect();
                let targets: [wgpu::Texture; O] =
                    std::array::from_fn(|_| planes.take(&self.device, output));
                self.dispatch(
                    encoder,
                    F::STEP,
                    &params,
                    &textures,
                    &targets.each_ref(),
                    groups,
                );
                if let Some((previous, _)) = below.replace((targets, output)) {
                    previous.into_iter().for_each(|plane| planes.keep(plane));
                }
            }
        }
        if let Some((previous, _)) = below {
            previous.into_iter().for_each(|plane| planes.keep(plane));
        }
        Ok(())
    }

    /// Records `passes` passes of `op` along `axis` of `input`, a plane of `width` by
    /// `height`, into `output`, which each pass lengthens along the axis by the operation's
    /// growth: a dispatch per pass, directly for narrow windows and by blocks for wide ones,
    /// every pass but the last into a plane taken from `planes`, which get them back once
    /// recorded.
    #[expect(
        clippy::too_many_arguments,
        reason = "a plane, its size, its output and scratch"
    )]
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Blur records it from GPU slice 5")
    )]
    pub(crate) fn line<L: LineReduce>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        op: &L,
        axis: Axis,
        passes: u32,
        input: &wgpu::Texture,
        (width, height): (u32, u32),
        output: &wgpu::Texture,
        planes: &mut Planes,
    ) -> Result<(), PlaneError> {
        assert!(2 * op.t() <= SCAN_BLOCK, "a window the scan's block holds");
        let mut size = Window {
            x: 0,
            y: 0,
            width,
            height,
        };
        check_input(input, size)?;
        let growth = op.growth() * passes;
        let grown = match axis {
            Axis::Rows => Window {
                width: width + growth,
                ..size
            },
            Axis::Columns => Window {
                height: height + growth,
                ..size
            },
        };
        check_output(output, grown)?;
        let mut current: Option<wgpu::Texture> = None;
        for pass in 0..passes {
            let len = match axis {
                Axis::Rows => size.width,
                Axis::Columns => size.height,
            };
            match axis {
                Axis::Rows => size.width += op.growth(),
                Axis::Columns => size.height += op.growth(),
            }
            let target = (pass + 1 < passes).then(|| planes.take(&self.device, size));
            let params = LineParams {
                width: size.width,
                height: size.height,
                axis: axis as u32,
                len,
                op: *op,
            };
            let source = current.as_ref().unwrap_or(input);
            let written = target.as_ref().unwrap_or(output);
            if op.t() <= DIRECT_T {
                let groups = per_texel(size);
                self.dispatch(encoder, L::DIRECT, &params, &[source], &[written], groups);
            } else {
                let (extent, lines) = match axis {
                    Axis::Rows => (size.width, size.height),
                    Axis::Columns => (size.height, size.width),
                };
                let blocks = extent / (2 * op.t() - 1) + 1;
                let groups = [blocks, lines];
                self.dispatch(encoder, L::SCAN, &params, &[source], &[written], groups);
            }
            if let Some(previous) = std::mem::replace(&mut current, target) {
                planes.keep(previous);
            }
        }
        Ok(())
    }

    /// Records `entry` with `params` over `inputs` into `outputs` in `groups` workgroups,
    /// building its pipeline first if nothing has recorded it yet.
    fn dispatch<P: Params>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        entry: &'static str,
        params: &P,
        inputs: &[&wgpu::Texture],
        outputs: &[&wgpu::Texture],
        groups: [u32; 2],
    ) {
        let (device, module) = (&self.device, &self.module);
        let pipeline = self
            .pipelines
            .entry(entry)
            .or_insert_with(|| Pipeline::new(device, module, entry, inputs.len(), outputs.len()));
        pipeline.record(device, encoder, params, inputs, outputs, groups);
    }
}

/// An entry point's pipeline: a storage buffer of parameters, its input planes read as
/// textures, and its output planes written as storage textures, bound in that order as the
/// kernel crate declares them.
#[derive(Debug)]
struct Pipeline {
    label: &'static str,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl Pipeline {
    fn new(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        entry: &'static str,
        inputs: usize,
        outputs: usize,
    ) -> Pipeline {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        entries.extend(
            (1..=inputs as u32).map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }),
        );
        entries.extend(
            (1..=outputs as u32).map(|output| wgpu::BindGroupLayoutEntry {
                binding: inputs as u32 + output,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    // rust-gpu declares a storage image readable and writable, and naga holds the
                    // layout to the same; R32Float may be bound read-write without any feature.
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: PLANE_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }),
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(entry),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        Pipeline {
            label: entry,
            layout,
            pipeline,
        }
    }

    fn record<P: Params>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        params: &P,
        inputs: &[&wgpu::Texture],
        outputs: &[&wgpu::Texture],
        groups: [u32; 2],
    ) {
        // SAFETY: `Params` is implemented only for `#[repr(C)]` structs of four-byte values,
        // which have no padding, so all `size_of::<P>()` bytes are initialised.
        let bytes = unsafe {
            std::slice::from_raw_parts((params as *const P).cast::<u8>(), size_of::<P>())
        };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("a buffer mapped at creation")
            .slice(..)
            .copy_from_slice(bytes);
        buffer.unmap();
        let views: Vec<wgpu::TextureView> = (inputs.iter().chain(outputs))
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        pass.dispatch_workgroups(groups[0], groups[1], 1);
    }
}

/// Workgroups covering a plane over `window`, one invocation per texel.
fn per_texel(window: Window) -> [u32; 2] {
    [
        window.width.div_ceil(WORKGROUP),
        window.height.div_ceil(WORKGROUP),
    ]
}

/// Where `input` lies in `output`, in the output's texels.
fn place(input: Window, output: Window) -> Placement {
    Placement {
        x: input.x - output.x,
        y: input.y - output.y,
        width: input.width,
        height: input.height,
    }
}

/// Planes the primitives' intermediate results are written to, kept for reuse once recorded:
/// a later dispatch in the same submission may write one again, since wgpu orders the two.
#[derive(Debug, Default)]
pub(crate) struct Planes(Vec<wgpu::Texture>);

impl Planes {
    /// A plane over `window`'s size, readable and writable by the kernels, whose contents the
    /// caller overwrites.
    fn take(&mut self, device: &wgpu::Device, window: Window) -> wgpu::Texture {
        let size = extent(window);
        match self.0.iter().position(|plane| plane.size() == size) {
            Some(index) => self.0.swap_remove(index),
            None => device.create_texture(&wgpu::TextureDescriptor {
                label: Some("intermediate"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PLANE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            }),
        }
    }

    fn keep(&mut self, plane: wgpu::Texture) {
        self.0.push(plane);
    }
}

/// The texture size of a plane over `window`.
pub(crate) fn extent(window: Window) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: window.width,
        height: window.height,
        depth_or_array_layers: 1,
    }
}

/// Checks an input plane before anything is recorded, rather than leaving it to wgpu, whose
/// validation error would count as a bug on the device. No texture is empty, so a plane over an
/// empty window may be any plane: its placement covers nothing and it is never read.
fn check_input(texture: &wgpu::Texture, window: Window) -> Result<(), PlaneError> {
    check_plane(Role::Input, texture, wgpu::TextureUsages::TEXTURE_BINDING)?;
    if !window.is_empty() && texture.size() != extent(window) {
        return Err(PlaneError::Size {
            role: Role::Input,
            plane: texture.size(),
            needed: extent(window),
        });
    }
    Ok(())
}

fn check_output(texture: &wgpu::Texture, window: Window) -> Result<(), PlaneError> {
    check_plane(Role::Output, texture, wgpu::TextureUsages::STORAGE_BINDING)?;
    if texture.size() != extent(window) {
        return Err(PlaneError::Size {
            role: Role::Output,
            plane: texture.size(),
            needed: extent(window),
        });
    }
    Ok(())
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
    /// The plane is not the size of its window.
    Size {
        role: Role,
        plane: wgpu::Extent3d,
        needed: wgpu::Extent3d,
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
            PlaneError::Size {
                role,
                plane,
                needed,
            } => write!(
                f,
                "the {role:?} plane is {} by {}, not the {} by {} of its window",
                plane.width, plane.height, needed.width, needed.height
            ),
        }
    }
}

impl std::error::Error for PlaneError {}

/// A device on a real adapter, and the upload and readback the GPU tests need: evaluation
/// uploads through mapped buffers instead, and reads nothing back, so these stay test support.
#[cfg(test)]
pub(crate) mod headless {
    use super::PLANE_FORMAT;

    use std::sync::{Mutex, OnceLock};

    /// A device on the Vulkan adapter `WGPU_ADAPTER_NAME` names, or else the first one, a CPU
    /// renderer included: the tests prove the kernels on whatever Mesa or driver is installed,
    /// and print which, so every log says what ran. No adapter is a failure, never a skip.
    ///
    /// Every device comes from one instance and adapter, made once, and one at a time: the
    /// Vulkan loader's lists of instances and devices are not safe to change from several
    /// threads at once, and with an instance per test, a test enumerating adapters crashed
    /// another inside the loader as it created its device.
    pub(crate) fn device() -> (wgpu::Device, wgpu::Queue) {
        static ADAPTER: OnceLock<(wgpu::Instance, wgpu::Adapter)> = OnceLock::new();
        static CREATING: Mutex<()> = Mutex::new(());
        let _creating = CREATING
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (_, adapter) = ADAPTER.get_or_init(|| {
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
            (instance, adapter)
        });
        let info = adapter.get_info();
        println!(
            "GPU tests on {} ({:?}), {} {}",
            info.name, info.device_type, info.driver, info.driver_info
        );
        pollster::block_on(adapter.request_device(&Default::default()))
            .expect("a device on the adapter")
    }

    /// A plane the kernels can read, holding `texels` in row-major order.
    pub(crate) fn upload(
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
    pub(crate) fn output(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        plane(
            device,
            width,
            height,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        )
    }

    pub(crate) fn plane(
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
    pub(crate) fn readback(
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
    use opencompositor_kernels::{BoxPass, BoxSum, Exposure, Merge, ulp_distance};

    use super::*;
    use crate::gpu::noop_parts;

    fn window(x: i32, y: i32, width: u32, height: u32) -> Window {
        Window {
            x,
            y,
            width,
            height,
        }
    }

    fn at_origin(width: u32, height: u32) -> Window {
        window(0, 0, width, height)
    }

    /// Records every entry point once, which builds its pipeline: naga validates each against
    /// its layout without a GPU.
    #[test]
    fn the_module_and_pipelines_validate_without_a_gpu() {
        // The queue kept: a noop device whose queue dropped refuses to make an encoder.
        let (_, device, _queue) = noop_parts(None);
        let mut kernels = Kernels::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut planes = Planes::default();
        let plane = |width, height| {
            let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
            headless::plane(&device, width, height, usage)
        };
        let size = at_origin(4, 4);
        let (input, output) = (plane(4, 4), plane(4, 4));
        let op = Exposure { gain: 1.0 };
        kernels
            .map(&mut encoder, &op, [(&input, size)], size, [&output])
            .expect("planes of one size");
        let four = [(); 4].map(|()| plane(4, 4));
        let four = four.each_ref();
        let outputs = [(); 4].map(|()| plane(4, 4));
        let layers = [(four, size), (four, size)];
        let merge = Merge { mix: 0.5 };
        kernels
            .fold(
                &mut encoder,
                &merge,
                (four, size),
                &layers,
                size,
                outputs.each_ref(),
                &mut planes,
            )
            .expect("planes of one size");
        for size in [3.0, 1000.0] {
            let pass = BoxPass::new(size);
            let grown = plane(4 + 2 * pass.tmax(), 4);
            let op = BoxSum::new(pass);
            kernels
                .line(
                    &mut encoder,
                    &op,
                    Axis::Rows,
                    1,
                    &input,
                    (4, 4),
                    &grown,
                    &mut planes,
                )
                .expect("an output grown by the pass");
        }
        assert_eq!(kernels.pipelines.len(), 5, "every entry point");
    }

    #[test]
    fn a_wrong_plane_is_refused_before_anything_is_recorded() {
        let (_, device, _queue) = noop_parts(None);
        let mut kernels = Kernels::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        let readable = wgpu::TextureUsages::TEXTURE_BINDING;
        let writable = wgpu::TextureUsages::STORAGE_BINDING;
        let input = headless::plane(&device, 4, 4, readable);
        let output = headless::plane(&device, 4, 4, writable);
        let size = at_origin(4, 4);
        let op = Exposure { gain: 1.0 };

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
            kernels.map(&mut encoder, &op, [(&rgba, size)], size, [&output]),
            Err(PlaneError::Format {
                role: Role::Input,
                format: wgpu::TextureFormat::Rgba32Float,
            })
        );

        let unwritable = headless::plane(&device, 4, 4, readable);
        assert_eq!(
            kernels.map(&mut encoder, &op, [(&input, size)], size, [&unwritable]),
            Err(PlaneError::Usage {
                role: Role::Output,
                missing: writable,
            })
        );

        // An input placed anywhere must be its window's size, and an output the output's.
        let wider = window(-1, 0, 5, 4);
        assert_eq!(
            kernels.map(&mut encoder, &op, [(&input, wider)], size, [&output]),
            Err(PlaneError::Size {
                role: Role::Input,
                plane: input.size(),
                needed: extent(wider),
            })
        );
        let smaller = headless::plane(&device, 4, 3, writable);
        assert_eq!(
            kernels.map(&mut encoder, &op, [(&input, size)], size, [&smaller]),
            Err(PlaneError::Size {
                role: Role::Output,
                plane: smaller.size(),
                needed: input.size(),
            })
        );

        // Three passes of one tap each side along rows grow the plane by six texels.
        let op = BoxSum::new(BoxPass::new(3.0));
        let mut planes = Planes::default();
        assert_eq!(
            kernels.line(
                &mut encoder,
                &op,
                Axis::Rows,
                3,
                &input,
                (4, 4),
                &output,
                &mut planes
            ),
            Err(PlaneError::Size {
                role: Role::Output,
                plane: output.size(),
                needed: extent(at_origin(10, 4)),
            })
        );
        let grown = headless::plane(&device, 4, 10, writable);
        assert_eq!(
            kernels.line(
                &mut encoder,
                &op,
                Axis::Columns,
                3,
                &input,
                (4, 4),
                &grown,
                &mut planes
            ),
            Ok(())
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
        let mut kernels = Kernels::new(&device);
        // A quarter-stop gain, rounded like the host's, and one that flushes MIN_POSITIVE.
        let gains = [0.75f64.exp2() as f32, (-10f64).exp2() as f32];
        for (width, height) in SIZES {
            let size = at_origin(width, height);
            let texels = texels((width * height) as usize);
            let input = headless::upload(&device, &queue, width, height, &texels);
            for gain in gains {
                let output = headless::output(&device, width, height);
                let mut encoder = device.create_command_encoder(&Default::default());
                kernels
                    .map(
                        &mut encoder,
                        &Exposure { gain },
                        [(&input, size)],
                        size,
                        [&output],
                    )
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

    /// Merge of one layer at Mix 1, over alone, on the GPU against the CPU definition: within
    /// 1 ULP where the result is not a cancellation, since a driver that splits the fused
    /// multiply-add rounds twice; where the product and the layer cancel, within the ULP of
    /// what was summed. Non-finite results agree by class. Every plane is one texel pattern
    /// rotated, alpha the alphas.
    #[test]
    fn over_on_the_gpu_is_within_a_ulp_of_the_cpu_definition() {
        let (device, queue) = headless::device();
        let mut kernels = Kernels::new(&device);
        let mut worst = 0;
        for (width, height) in SIZES {
            let size = at_origin(width, height);
            let count = (width * height) as usize;
            let rotated = |by: usize| {
                let mut plane = texels(count);
                plane.rotate_left(by.min(count));
                plane
            };
            let background = [rotated(0), rotated(1), rotated(2), rotated(3)];
            let layer = [rotated(4), rotated(5), rotated(6), alphas(count)];
            let upload =
                |texels: &Vec<f32>| headless::upload(&device, &queue, width, height, texels);
            let background_planes = background.each_ref().map(upload);
            let layer_planes = layer.each_ref().map(upload);
            let outputs = [(); 4].map(|()| headless::output(&device, width, height));
            let mut encoder = device.create_command_encoder(&Default::default());
            kernels
                .fold(
                    &mut encoder,
                    &Merge { mix: 1.0 },
                    (background_planes.each_ref(), size),
                    &[(layer_planes.each_ref(), size)],
                    size,
                    outputs.each_ref(),
                    &mut Planes::default(),
                )
                .expect("planes of one size");
            queue.submit([encoder.finish()]);
            for (channel, output) in outputs.iter().enumerate() {
                let result = headless::readback(&device, &queue, output);
                assert_eq!(result.len(), count);
                for (i, &actual) in result.iter().enumerate() {
                    let (b, l, a) = (background[channel][i], layer[channel][i], layer[3][i]);
                    let expected = opencompositor_kernels::over(b, l, a);
                    let what = format!(
                        "{width}x{height} plane {channel} texel {i}: {l:?} at {a:?} over {b:?}"
                    );
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
        }
        println!("over on the GPU: at worst {worst} ULP from the CPU");
    }

    /// Merge of two layers placed in different parts of the output, with the base smaller
    /// than it, through the step and the last dispatch, against the CPU's fold bit for bit:
    /// the values make every product and sum exact, so a split multiply-add cannot differ.
    #[test]
    fn the_fold_on_the_gpu_places_its_layers_as_the_cpu_does() {
        let (device, queue) = headless::device();
        let mut kernels = Kernels::new(&device);
        let pool = crate::evaluate::Pool::new();
        let base_window = window(0, 0, 20, 9);
        let first_window = window(13, 4, 19, 17);
        let second_window = window(-5, 2, 9, 6);
        let output = window(-5, 0, 37, 21);
        let plane = |window: Window, values: [f32; 2]| -> Vec<f32> {
            (0..window.area())
                .map(|i| values[(i % 2) as usize])
                .collect()
        };
        let base =
            [[0.5, 0.25], [0.125, 1.0], [2.0, 0.0], [1.0, 0.5]].map(|v| plane(base_window, v));
        let first =
            [[0.25, 0.5], [0.5, 0.0], [1.0, 0.75], [0.5, 1.0]].map(|v| plane(first_window, v));
        let second =
            [[0.0625, 1.0], [0.75, 0.5], [1.0, 0.25], [1.0, 0.5]].map(|v| plane(second_window, v));
        for mix in [1.0, 0.5] {
            let op = Merge { mix };
            let mut cpu_planes: [Vec<f32>; 4] = [(); 4].map(|()| vec![0.0; output.area() as usize]);
            cpu::fold(
                &pool,
                &op,
                (base.each_ref().map(Vec::as_slice), base_window),
                &[
                    (first.each_ref().map(Vec::as_slice), first_window),
                    (second.each_ref().map(Vec::as_slice), second_window),
                ],
                output,
                cpu_planes.each_mut().map(Vec::as_mut_slice),
            );
            let upload = |window: Window, planes: &[Vec<f32>; 4]| {
                planes.each_ref().map(|texels| {
                    headless::upload(&device, &queue, window.width, window.height, texels)
                })
            };
            let (base_planes, first_planes, second_planes) = (
                upload(base_window, &base),
                upload(first_window, &first),
                upload(second_window, &second),
            );
            let outputs = [(); 4].map(|()| headless::output(&device, output.width, output.height));
            let mut encoder = device.create_command_encoder(&Default::default());
            kernels
                .fold(
                    &mut encoder,
                    &op,
                    (base_planes.each_ref(), base_window),
                    &[
                        (first_planes.each_ref(), first_window),
                        (second_planes.each_ref(), second_window),
                    ],
                    output,
                    outputs.each_ref(),
                    &mut Planes::default(),
                )
                .expect("planes over their windows");
            queue.submit([encoder.finish()]);
            for (channel, output) in outputs.iter().enumerate() {
                let result = headless::readback(&device, &queue, output);
                let bits = |plane: &[f32]| plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert!(
                    bits(&result) == bits(&cpu_planes[channel]),
                    "Mix {mix}, plane {channel}"
                );
            }
        }
    }

    /// A plane for the box pass: values of both signs over sixty stops on every texel but five
    /// of the first row, which hold the non-finite classes and two spikes near the largest
    /// magnitude, so that the widest pass still leaves most lines finite, and bounded by their
    /// own magnitudes rather than a spike's.
    fn blur_texels(count: usize) -> Vec<f32> {
        const FINITE: [f32; 10] = [
            0.0, -0.0, 1.0, -1.0, 1.0e-30, 65504.0, 2.0e-20, -7.5, 0.25, 1.0e6,
        ];
        (0..count)
            .map(|i| match i {
                1 => f32::NAN,
                5 => f32::INFINITY,
                9 => f32::NEG_INFINITY,
                13 => 3.0e38,
                17 => -3.0e38,
                _ if i % 2 == 0 => FINITE[(i / 2) % FINITE.len()],
                _ => {
                    let magnitude = ((i % 800) as f32 * 0.0731 - 30.0).exp2();
                    if i % 4 == 1 { magnitude } else { -magnitude }
                }
            })
            .collect()
    }

    /// The direct sum of one pass summed in f64 at output `position` along a line of `length`
    /// input texels read through `sample`, and the same sum over the magnitudes.
    fn box_reference(
        pass: BoxPass,
        weights: &[f32],
        position: usize,
        length: usize,
        sample: impl Fn(usize) -> f32,
    ) -> (f64, f64) {
        let t = pass.tmax() as usize;
        let (mut sum, mut magnitude) = (0.0, 0.0);
        // Input texel `i` is `|i + t - position|` taps from the window's centre.
        for i in position.saturating_sub(2 * t)..(position + 1).min(length) {
            let weight = f64::from(weights[(i + t).abs_diff(position)]);
            let v = f64::from(sample(i));
            sum += weight * v;
            magnitude += weight * v.abs();
        }
        (sum, magnitude)
    }

    /// The GPU's K for a box pass: an f32 sum's worst case grows with the tap count, and 256
    /// keeps the error within 2⁻¹⁶ of the magnitude sum. The test prints the factor the
    /// adapter needed.
    const BOX_PASS_K: f64 = 256.0;

    /// One box pass on the GPU, along each axis, against the definition's f64 sum, within
    /// `BOX_PASS_K` times 2⁻²⁴ of the sum over the magnitudes plus one `MIN_POSITIVE`, at sizes
    /// from one tap each side to the limit, summed directly and by blocks, over the usual
    /// planes and a line longer than the widest window; non-finite results agree by class.
    #[test]
    fn a_box_pass_on_the_gpu_is_within_its_bound_of_the_direct_sum() {
        let (device, queue) = headless::device();
        let mut kernels = Kernels::new(&device);
        let mut planes = Planes::default();
        let mut worst: f64 = 0.0;
        for size in [2.0, 3.0, 10.0, 31.4, 100.0, 180.0, 1000.0, 4096.0] {
            let pass = BoxPass::new(size);
            let t = pass.tmax();
            let op = BoxSum::new(pass);
            let weights: Vec<f32> = pass.weights().collect();
            for axis in [Axis::Rows, Axis::Columns] {
                let long_line = match axis {
                    Axis::Rows => (2800, 2),
                    Axis::Columns => (2, 2800),
                };
                for (width, height) in SIZES.into_iter().chain([long_line]) {
                    let texels = blur_texels((width * height) as usize);
                    let input = headless::upload(&device, &queue, width, height, &texels);
                    let (out_width, out_height) = match axis {
                        Axis::Rows => (width + 2 * t, height),
                        Axis::Columns => (width, height + 2 * t),
                    };
                    let output = headless::output(&device, out_width, out_height);
                    let mut encoder = device.create_command_encoder(&Default::default());
                    kernels
                        .line(
                            &mut encoder,
                            &op,
                            axis,
                            1,
                            &input,
                            (width, height),
                            &output,
                            &mut planes,
                        )
                        .expect("an output grown by the pass");
                    queue.submit([encoder.finish()]);
                    let result = headless::readback(&device, &queue, &output);
                    assert_eq!(result.len(), (out_width * out_height) as usize);
                    for (i, &actual) in result.iter().enumerate() {
                        let (x, y) = (i % out_width as usize, i / out_width as usize);
                        let (expected, magnitude) = match axis {
                            Axis::Rows => box_reference(pass, &weights, x, width as usize, |i| {
                                texels[y * width as usize + i]
                            }),
                            Axis::Columns => {
                                box_reference(pass, &weights, y, height as usize, |i| {
                                    texels[i * width as usize + x]
                                })
                            }
                        };
                        let what = || {
                            format!(
                                "size {size} along {axis:?} of {width}x{height} at ({x}, {y}) gave {actual:?}, not {expected:?}"
                            )
                        };
                        if !expected.is_finite() {
                            assert_eq!(ulp_distance(actual, expected as f32), 0, "{}", what());
                            continue;
                        }
                        let difference = (f64::from(actual) - expected).abs();
                        let unit = magnitude * f64::from(f32::EPSILON) / 2.0;
                        assert!(
                            difference <= BOX_PASS_K * unit + f64::from(f32::MIN_POSITIVE),
                            "{}, with {magnitude} summed",
                            what()
                        );
                        if unit > 0.0 {
                            let needed = (difference - f64::from(f32::MIN_POSITIVE)) / unit;
                            worst = worst.max(needed);
                        }
                    }
                }
            }
        }
        println!(
            "box pass on the GPU: K {} at most, under {BOX_PASS_K}",
            worst.ceil()
        );
    }

    /// Three passes in one call, through the intermediate planes, are the three passes one at
    /// a time, bit for bit, along each axis and by both reductions.
    #[test]
    fn passes_in_one_call_are_the_passes_one_at_a_time() {
        let (device, queue) = headless::device();
        let mut kernels = Kernels::new(&device);
        let mut planes = Planes::default();
        let (width, height) = (61, 7);
        let texels = blur_texels((width * height) as usize);
        let input = headless::upload(&device, &queue, width, height, &texels);
        for size in [10.0, 400.0] {
            let op = BoxSum::new(BoxPass::new(size));
            let growth = op.growth();
            for axis in [Axis::Rows, Axis::Columns] {
                let grown = |passes: u32| match axis {
                    Axis::Rows => (width + passes * growth, height),
                    Axis::Columns => (width, height + passes * growth),
                };
                let (w, h) = grown(3);
                let together = headless::output(&device, w, h);
                let mut encoder = device.create_command_encoder(&Default::default());
                kernels
                    .line(
                        &mut encoder,
                        &op,
                        axis,
                        3,
                        &input,
                        (width, height),
                        &together,
                        &mut planes,
                    )
                    .expect("an output grown by three passes");
                let usage = wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC;
                let mut current = (input.clone(), (width, height));
                for pass in 1..=3 {
                    let (w, h) = grown(pass);
                    let next = headless::plane(&device, w, h, usage);
                    kernels
                        .line(
                            &mut encoder,
                            &op,
                            axis,
                            1,
                            &current.0,
                            current.1,
                            &next,
                            &mut planes,
                        )
                        .expect("an output grown by one pass");
                    current = (next, (w, h));
                }
                queue.submit([encoder.finish()]);
                let bits = |plane: Vec<f32>| plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert!(
                    bits(headless::readback(&device, &queue, &together))
                        == bits(headless::readback(&device, &queue, &current.0)),
                    "size {size} along {axis:?}"
                );
            }
        }
    }
}
