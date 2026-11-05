//! The viewer cache: copies of evaluated images as the viewer shows them, reduced to the pane's
//! resolution, which show a frame again without the worker.
//!
//! Finals are cached, not sources: a Merge multiplies its inputs, and a heavy comp plays in real
//! time only from its finals.

use std::sync::Arc;

use half::f16;
use half::slice::HalfFloatSliceExt;

use crate::evaluate::{Pool, ResultKey};
use crate::image::{ImageInfo, Window};
use crate::viewer::Zoom;

/// The bytes of copies the cache holds unless told otherwise.
pub const DEFAULT_BUDGET: u64 = 4 << 30;

/// The largest finite f16, which larger values are clamped to: f16's infinity would show the
/// same, but WGSL leaves converting it undefined.
const F16_MAX: f32 = 65504.0;

/// An image as the viewer shows it, for the viewer cache: its planes averaged over square blocks
/// of pixels, `2^halvings` on a side and aligned to the image's pixel grid, as RGBA f16. Pixels
/// outside the data window count as zero, as they are, and NaN as zero, as the viewer shows it
/// black; values beyond f16's range are clamped to it.
#[derive(Debug)]
pub struct ViewerCopy {
    pub(crate) key: ResultKey,
    pub(crate) info: ImageInfo,
    pub(crate) halvings: u32,
    /// In texels, each covering its block of image pixels: texel `(x, y)` averages the pixels
    /// from `(x << halvings, y << halvings)`.
    pub(crate) window: Window,
    /// RGBA, bottom row first, as the planes are.
    pub(crate) texels: Vec<f16>,
}

impl ViewerCopy {
    /// The copy of an image's planes, in `Channel::ALL`'s order over its data window, made on
    /// the CPU.
    pub(crate) fn of_planes(
        key: ResultKey,
        info: ImageInfo,
        planes: [&[f32]; 4],
        halvings: u32,
        pool: &Pool,
    ) -> ViewerCopy {
        let data = info.data_window;
        let window = copy_window(data, halvings);
        let width = window.width as usize;
        let mut texels = vec![f16::ZERO; width * window.height as usize * 4];
        let side = 1i64 << halvings;
        let area = (side * side) as f32;
        // The first pixel of the copy's first texel, from the data window's bottom-left corner.
        let offset = [
            (i64::from(window.x) << halvings) - i64::from(data.x),
            (i64::from(window.y) << halvings) - i64::from(data.y),
        ];
        let (data_width, data_height) = (i64::from(data.width), i64::from(data.height));
        pool.fill_rows([&mut texels[..]], width * 4, |_, first_row, [rows]| {
            let mut sums = vec![0.0f32; width * 4];
            for (row, out) in (first_row..).zip(rows.chunks_mut(width * 4)) {
                sums.fill(0.0);
                let bottom = offset[1] + row as i64 * side;
                for y in bottom.max(0)..(bottom + side).min(data_height) {
                    let start = (y * data_width) as usize;
                    // Each pixel's texel, along the row: the same order of sums as the GPU's.
                    for x in 0..data_width {
                        let texel = ((x - offset[0]) >> halvings) as usize;
                        for (channel, plane) in planes.iter().enumerate() {
                            let value = plane[start + x as usize];
                            if !value.is_nan() {
                                sums[texel * 4 + channel] += value;
                            }
                        }
                    }
                }
                for sum in &mut sums {
                    *sum = shown(*sum / area);
                }
                out.convert_from_f32_slice(&sums);
            }
        });
        ViewerCopy {
            key,
            info,
            halvings,
            window,
            texels,
        }
    }

    /// The copy of an image whose data window is empty.
    pub(crate) fn empty(key: ResultKey, info: ImageInfo, halvings: u32) -> ViewerCopy {
        ViewerCopy {
            window: copy_window(info.data_window, halvings),
            key,
            info,
            halvings,
            texels: Vec::new(),
        }
    }

    /// The bytes its texels take.
    pub fn bytes(&self) -> u64 {
        (self.texels.len() * size_of::<f16>()) as u64
    }

    /// How many times the image was halved.
    pub fn halvings(&self) -> u32 {
        self.halvings
    }

    /// The texels as bytes, for uploading.
    pub(crate) fn texel_bytes(&self) -> &[u8] {
        // SAFETY: f16 is a u16 with no padding, every byte of which is a valid u8, whose
        // alignment is 1, and the bytes are borrowed for as long as the texels are.
        unsafe {
            std::slice::from_raw_parts(
                self.texels.as_ptr().cast::<u8>(),
                size_of_val(&self.texels[..]),
            )
        }
    }
}

/// A block's mean as a copy holds it: a NaN, from infinities of both signs, as zero, and within
/// f16's range.
fn shown(mean: f32) -> f32 {
    if mean.is_nan() {
        return 0.0;
    }
    mean.clamp(-F16_MAX, F16_MAX)
}

/// The texels of a copy halved `halvings` times that cover the data window.
pub(crate) fn copy_window(data: Window, halvings: u32) -> Window {
    if data.width == 0 || data.height == 0 {
        return Window {
            x: data.x >> halvings,
            y: data.y >> halvings,
            width: 0,
            height: 0,
        };
    }
    let side = 1i64 << halvings;
    let [left, bottom] = [data.x, data.y].map(|start| i64::from(start) >> halvings);
    let right = (i64::from(data.x) + i64::from(data.width) + side - 1) >> halvings;
    let top = (i64::from(data.y) + i64::from(data.height) + side - 1) >> halvings;
    Window {
        x: left as i32,
        y: bottom as i32,
        width: (right - left) as u32,
        height: (top - bottom) as u32,
    }
}

/// Copies of images the viewer showed, by result key and halvings, within a budget of bytes.
/// The session owns it, the worker fills it, and the app shows from it without the worker.
///
/// When full, it drops the copies least recently requested. Keys are compared one by one, each by
/// its fingerprint first: a cache within its budget holds hundreds of copies at most.
#[derive(Debug)]
pub struct ViewerCache {
    budget: u64,
    bytes: u64,
    entries: Vec<Entry>,
    /// Counts insertions and hits, which order the entries by when they were last requested.
    requests: u64,
    /// Counts changes to which copies are held.
    changes: u64,
}

#[derive(Debug)]
struct Entry {
    copy: Arc<ViewerCopy>,
    last_requested: u64,
}

impl Default for ViewerCache {
    fn default() -> ViewerCache {
        ViewerCache {
            budget: DEFAULT_BUDGET,
            bytes: 0,
            entries: Vec::new(),
            requests: 0,
            changes: 0,
        }
    }
}

impl ViewerCache {
    pub fn budget(&self) -> u64 {
        self.budget
    }

    /// The bytes of the copies held.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Changes whenever a copy is added or dropped, so that what was learned of the copies held is
    /// current while this is.
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// Whether a copy as large as the largest held fits without dropping any.
    pub(crate) fn has_room(&self) -> bool {
        let largest = (self.entries.iter().map(|entry| entry.copy.bytes())).max();
        self.bytes + largest.unwrap_or(0) <= self.budget
    }

    /// Whether it holds a copy of the key, at any reduction.
    pub fn contains(&self, key: &ResultKey) -> bool {
        self.entries.iter().any(|entry| entry.copy.key == *key)
    }

    /// Holds at most `budget` bytes from now on, dropping what no longer fits.
    pub(crate) fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
        self.make_room(0);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.changes += 1;
    }

    /// The copy of the key that the viewer at this zoom draws no larger than its texels: the
    /// most reduced such, if any.
    pub(crate) fn get(&mut self, key: &ResultKey, zoom: Zoom) -> Option<Arc<ViewerCopy>> {
        let entry = (self.entries.iter_mut())
            .filter(|entry| serves(&entry.copy, key, zoom))
            .max_by_key(|entry| entry.copy.halvings)?;
        self.requests += 1;
        entry.last_requested = self.requests;
        Some(Arc::clone(&entry.copy))
    }

    /// Whether it holds a copy of the key that the viewer at this zoom draws no larger than its
    /// texels.
    pub(crate) fn serves(&self, key: &ResultKey, zoom: Zoom) -> bool {
        (self.entries.iter()).any(|entry| serves(&entry.copy, key, zoom))
    }

    /// Holds the copy, in place of any of its key and halvings, dropping the copies least
    /// recently requested to make room. A copy larger than the budget is not held.
    pub(crate) fn insert(&mut self, copy: ViewerCopy) {
        if let Some(index) = (self.entries.iter())
            .position(|entry| entry.copy.halvings == copy.halvings && entry.copy.key == copy.key)
        {
            self.remove(index);
        }
        let bytes = copy.bytes();
        if bytes > self.budget {
            return;
        }
        self.make_room(bytes);
        self.requests += 1;
        self.changes += 1;
        self.bytes += bytes;
        self.entries.push(Entry {
            copy: Arc::new(copy),
            last_requested: self.requests,
        });
    }

    /// Drops the copies least recently requested until `bytes` more fit.
    fn make_room(&mut self, bytes: u64) {
        while self.bytes + bytes > self.budget {
            let Some(oldest) = (self.entries.iter().enumerate())
                .min_by_key(|(_, entry)| entry.last_requested)
                .map(|(index, _)| index)
            else {
                return;
            };
            self.remove(oldest);
        }
    }

    fn remove(&mut self, index: usize) {
        let entry = self.entries.swap_remove(index);
        self.bytes -= entry.copy.bytes();
        self.changes += 1;
    }
}

fn serves(copy: &ViewerCopy, key: &ResultKey, zoom: Zoom) -> bool {
    let fine_enough =
        (zoom.halvings(copy.info.display_window)).is_some_and(|halvings| copy.halvings <= halvings);
    fine_enough && copy.key == *key
}

/// Makes copies on the GPU from an image's planes there, for reading back.
#[derive(Debug)]
pub(crate) struct GpuCopier {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// The size of the shader's `Copy`.
const COPY_PARAMETERS_BYTES: u64 = 32;

/// A copy recorded on the GPU, in a buffer to map once the GPU has made it.
#[derive(Debug)]
pub(crate) struct GpuCopy {
    pub(crate) readback: wgpu::Buffer,
    halvings: u32,
    window: Window,
}

impl GpuCopier {
    pub(crate) fn new(device: &wgpu::Device) -> GpuCopier {
        let module = device.create_shader_module(wgpu::include_wgsl!("viewer_copy.wgsl"));
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
            label: Some("viewer copy"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(COPY_PARAMETERS_BYTES),
                    },
                    count: None,
                },
                plane(1),
                plane(2),
                plane(3),
                plane(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewer copy"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("viewer copy"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("copy"),
            compilation_options: Default::default(),
            cache: None,
        });
        GpuCopier { pipeline, layout }
    }

    /// Records the copy of planes over the data window, R32Float textures in `Channel::ALL`'s
    /// order, into a buffer to map. Fails if the GPU cannot hold the copy, and `Ok(None)` for an
    /// empty data window, whose copy has no texels.
    pub(crate) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        planes: [&wgpu::Texture; 4],
        data: Window,
        halvings: u32,
    ) -> Result<Option<GpuCopy>, String> {
        let window = copy_window(data, halvings);
        if window.width == 0 || window.height == 0 {
            return Ok(None);
        }
        let bytes = u64::from(window.width) * u64::from(window.height) * 8;
        let limits = device.limits();
        if bytes > limits.max_buffer_size || bytes > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "its copy takes {bytes} bytes, more than this GPU's buffers"
            ));
        }
        let groups = [window.width, window.height].map(|side| side.div_ceil(8));
        if groups
            .iter()
            .any(|&count| count > limits.max_compute_workgroups_per_dimension)
        {
            return Err("its copy is too large for this GPU".to_owned());
        }
        let offset = [
            (i64::from(window.x) << halvings) - i64::from(data.x),
            (i64::from(window.y) << halvings) - i64::from(data.y),
        ];
        // Laid out as the shader's `Copy`.
        let words: [u32; 8] = [
            offset[0] as i32 as u32,
            offset[1] as i32 as u32,
            data.width,
            data.height,
            window.width,
            window.height,
            halvings,
            0,
        ];
        let parameters = crate::worker::allocate(device, "its copy", || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("viewer copy parameters"),
                size: COPY_PARAMETERS_BYTES,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            })
        })?;
        {
            let mut view = (parameters.get_mapped_range_mut(..)).map_err(|error| {
                format!("the GPU could not write its copy's parameters: {error}")
            })?;
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
            view.slice(..).copy_from_slice(&bytes);
        }
        parameters.unmap();
        let buffer = |label, usage| {
            crate::worker::allocate(device, "its copy", || {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes,
                    usage,
                    mapped_at_creation: false,
                })
            })
        };
        let texels = buffer(
            "viewer copy",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        let readback = buffer(
            "viewer copy readback",
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        )?;
        let views = planes.map(|plane| plane.create_view(&Default::default()));
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: parameters.as_entire_binding(),
        }];
        entries.extend(
            (1..)
                .zip(&views)
                .map(|(binding, view)| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                }),
        );
        entries.push(wgpu::BindGroupEntry {
            binding: 5,
            resource: texels.as_entire_binding(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer copy"),
            layout: &self.layout,
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        encoder.copy_buffer_to_buffer(&texels, 0, &readback, 0, bytes);
        Ok(Some(GpuCopy {
            readback,
            halvings,
            window,
        }))
    }
}

impl GpuCopy {
    /// The copy, from its buffer once mapped for reading.
    pub(crate) fn read(self, key: ResultKey, info: ImageInfo) -> Result<ViewerCopy, String> {
        let texels = {
            let view = (self.readback.get_mapped_range(..))
                .map_err(|error| format!("the GPU could not read back its copy: {error}"))?;
            let (pairs, _) = view.as_chunks::<2>();
            pairs
                .iter()
                .map(|&bytes| f16::from_bits(u16::from_ne_bytes(bytes)))
                .collect()
        };
        self.readback.unmap();
        Ok(ViewerCopy {
            key,
            info,
            halvings: self.halvings,
            window: self.window,
            texels,
        })
    }
}

#[cfg(test)]
mod tests;
