//! The viewer's display copy of a result evaluated on the GPU: its planes converted into one
//! Rgba16Float texture by a compute pass in the request's own submission, so that nothing is read
//! back. The copy is display precision only: values round toward zero to half floats and clamp
//! at 65504.

use std::sync::Arc;

use crate::evaluate::GpuImage;
use crate::image::Channel;

/// The format the viewer samples, as the CPU session stages it.
pub(crate) const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Pixels on each side of the conversion's workgroups, as in `display.wgsl`.
const WORKGROUP: u32 = 16;

/// The conversion's pipeline on one device, and the display textures it made.
#[derive(Debug)]
pub(crate) struct Conversion {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Reused once nothing else holds them: wgpu clears every new storage texture, which would
    /// cost a 4K edit more than the conversion.
    textures: Vec<Arc<wgpu::Texture>>,
}

impl Conversion {
    pub(crate) fn new(device: &wgpu::Device) -> Conversion {
        let plane = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display conversion"),
            entries: &[
                plane(0),
                plane(1),
                plane(2),
                plane(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: DISPLAY_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::include_wgsl!("display.wgsl"));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display conversion"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("display conversion"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("convert"),
            compilation_options: Default::default(),
            cache: None,
        });
        Conversion {
            device: device.clone(),
            pipeline,
            layout,
            textures: Vec::new(),
        }
    }

    /// Records converting the image into a display texture, one nothing else holds or else a new
    /// one, and returns it; `None` for an empty data window.
    pub(crate) fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        image: &GpuImage,
    ) -> Option<Arc<wgpu::Texture>> {
        let planes = Channel::ALL.map(|channel| image.texture(channel));
        let [Some(red), Some(green), Some(blue), Some(alpha)] = planes else {
            return None;
        };
        let size = red.size();
        let display = self.texture(size);
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let views = [red, green, blue, alpha, &display].map(view);
        let entries: Vec<wgpu::BindGroupEntry> = (0..)
            .zip(&views)
            .map(|(binding, view)| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display conversion"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("display conversion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            size.width.div_ceil(WORKGROUP),
            size.height.div_ceil(WORKGROUP),
            1,
        );
        drop(pass);
        Some(display)
    }

    /// A display texture of the size, which the caller overwrites whole. Unheld textures of other
    /// sizes are dropped.
    fn texture(&mut self, size: wgpu::Extent3d) -> Arc<wgpu::Texture> {
        self.textures
            .retain(|texture| texture.size() == size || Arc::strong_count(texture) > 1);
        let unheld = (self.textures.iter())
            .find(|texture| texture.size() == size && Arc::strong_count(texture) == 1);
        if let Some(texture) = unheld {
            return Arc::clone(texture);
        }
        let texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DISPLAY_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        self.textures.push(Arc::clone(&texture));
        texture
    }

    /// Drops every display texture nothing else holds, as when the viewer wants nothing.
    pub(crate) fn release_unheld(&mut self) {
        self.textures
            .retain(|texture| Arc::strong_count(texture) > 1);
    }
}

#[cfg(test)]
pub(crate) mod tests;
