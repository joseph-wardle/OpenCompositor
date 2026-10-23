//! OpenEXR through the `exr` crate.

use std::io::{BufRead, Read, Seek};
use std::num::NonZero;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, mpsc};

use ::exr::block::chunk::{Chunk, CompressedBlock};
use ::exr::block::reader::{ChunksReader, FilteredChunksReader};
use ::exr::block::{BlockIndex, UncompressedBlock};
use ::exr::compression::Compression;
use ::exr::error::Error;
use ::exr::math::Vec2;
use ::exr::meta::MetaData;
use ::exr::meta::attribute::{self, Text};
use ::exr::meta::header::Header;
use half::f16;
use half::slice::HalfFloatSliceExt;
use opencompositor_kernels::flush;

use super::{ReadError, Refusal, check_size, window};
use crate::image::{Channel, Chromaticities, Image, Plane, Warning, Window};

pub(super) fn read(mut file: impl BufRead + Seek + Send) -> Result<Image, ReadError> {
    // Only the headers, nothing allocated for pixels yet. Unvalidated, because validation
    // rejects deep and subsampled parts as unsupported before they can be refused for what they
    // contain.
    let meta = match MetaData::read_from_buffered(&mut file, false) {
        Ok(meta) => meta,
        // `exr` calls a compression newer than it knows invalid, so name it instead.
        Err(error) => {
            return Err(match newer_compression(&mut file) {
                Some(name) => ReadError::Refused(Refusal::Compression(name.to_owned())),
                None => decode_error(error),
            });
        }
    };
    for header in &meta.headers {
        check_size(header.layer_size.0 as u64, header.layer_size.1 as u64)?;
    }
    let parts: Vec<Part> = meta.headers.iter().map(describe_part).collect();
    // `exr` reads no file with a deep part, not even its flat parts.
    if parts.iter().any(|part| part.deep) {
        return Err(ReadError::Refused(Refusal::Deep));
    }
    let shown_part = choose_part(&parts)?;
    let header = &meta.headers[shown_part];
    let unsupported = match header.compression {
        Compression::HTJ2K32 => Some("HTJ2K32"),
        Compression::HTJ2K256 => Some("HTJ2K256"),
        _ => None,
    };
    if let Some(name) = unsupported {
        return Err(ReadError::Refused(Refusal::Compression(name.into())));
    }
    let chromaticities = header
        .shared_attributes
        .chromaticities
        .map(|c| Chromaticities {
            red: [c.red.0, c.red.1],
            green: [c.green.0, c.green.1],
            blue: [c.blue.0, c.blue.1],
            white: [c.white.0, c.white.1],
        });
    if chromaticities.is_some_and(|c| c.is_xyz()) {
        return Err(ReadError::Refused(Refusal::XyzPrimaries));
    }
    let display = header.shared_attributes.display_window;
    let display_window = window(display.position.0, 0, display.size.0, display.size.1)?;
    let views: Vec<String> = (header.own_attributes.multi_view_names.iter())
        .flatten()
        .map(Text::to_string)
        .collect();
    let warnings = warnings(
        chromaticities,
        header.shared_attributes.pixel_aspect,
        &parts,
        shown_part,
        &views,
    );

    file.rewind()?;
    let planes = read_planes(file, shown_part)?;
    let data_window = flipped(
        header,
        header.own_attributes.layer_position,
        header.layer_size,
    )?;
    Ok(Image::new(display_window, data_window, planes, warnings))
}

/// The part's R, G, B and A at its largest resolution, in `Channel::ALL`'s order, bottom row
/// first. A missing A reads as 1. EXR stores premultiplied colour.
fn read_planes(file: impl BufRead + Seek + Send, part: usize) -> Result<[Plane; 4], ReadError> {
    let chunks = ::exr::block::read(file, false).map_err(decode_error)?;
    let header = &chunks.headers()[part];
    let (width, height) = (header.layer_size.width(), header.layer_size.height());
    let bytes_per_pixel = header.channels.bytes_per_pixel;
    // Each of R, G, B and A in a block's line: its plane, the bytes per pixel of the channels
    // before it, and its sample type.
    let mut sources = Vec::new();
    let mut before = 0;
    for channel in &header.channels.list {
        if channel.sampling != Vec2(1, 1) {
            return Err(ReadError::Decode(format!(
                "channel {} is subsampled",
                channel.name
            )));
        }
        let name = channel.name.to_string();
        if let Some(plane) = Channel::ALL.iter().position(|c| c.name() == name) {
            sources.push((plane, before, channel.sample_type));
        }
        before += channel.sample_type.bytes_per_sample();
    }
    let area = width * height;
    let mut planes: [Plane; 4] = std::array::from_fn(|_| Plane::zeroed(area, area));
    if !sources
        .iter()
        .any(|&(plane, ..)| plane == Channel::A as usize)
    {
        planes[Channel::A as usize] = std::iter::repeat_n(1.0, area).collect();
    }
    let mut out = planes
        .each_mut()
        .map(|plane| plane.get_mut().expect("not yet shared"));
    let chunks = chunks
        .filter_chunks(false, |_, tile, block| {
            block.layer == part && tile.is_largest_resolution_level()
        })
        .map_err(decode_error)?;
    let mut halves = Vec::new();
    decode(chunks, |block| {
        let BlockIndex {
            pixel_position: Vec2(left, top),
            pixel_size: Vec2(block_width, block_height),
            ..
        } = block.index;
        if left + block_width > width
            || top + block_height > height
            || block.data.len() != block_width * block_height * bytes_per_pixel
        {
            return Err(ReadError::Decode(
                "a block does not fit its part's data window".to_owned(),
            ));
        }
        let lines = block.data.chunks_exact(block_width * bytes_per_pixel);
        for (line, bytes) in (top..).zip(lines) {
            // EXR counts rows down; planes start at the bottom.
            let start = (height - 1 - line) * width + left;
            // A line holds each channel's samples in turn, in native byte order.
            for &(plane, before, sample_type) in &sources {
                let samples = &bytes[before * block_width..];
                let target = &mut out[plane][start..start + block_width];
                // Only an F32 sample can be subnormal: every f16 is a normal f32 or zero, and a
                // U32 is a whole number.
                match sample_type {
                    attribute::SampleType::F16 => {
                        let (samples, _) = samples.as_chunks();
                        halves.clear();
                        halves.extend(
                            samples[..block_width]
                                .iter()
                                .map(|&b| f16::from_ne_bytes(b)),
                        );
                        halves.convert_to_f32_slice(target);
                    }
                    attribute::SampleType::F32 => {
                        let (samples, _) = samples.as_chunks();
                        for (value, &sample) in target.iter_mut().zip(samples) {
                            *value = flush(f32::from_ne_bytes(sample));
                        }
                    }
                    attribute::SampleType::U32 => {
                        let (samples, _) = samples.as_chunks();
                        for (value, &sample) in target.iter_mut().zip(samples) {
                            *value = u32::from_ne_bytes(sample) as f32;
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(planes)
}

/// Decompresses the chunks across the machine's cores, handing each block to `insert` on this
/// thread in whatever order they finish.
///
/// `exr` decompresses on a rayon pool of its own, where a panic aborts the process. Here a chunk
/// whose decoder panics fails the read instead.
fn decode(
    chunks: FilteredChunksReader<impl BufRead + Seek + Send>,
    mut insert: impl FnMut(UncompressedBlock) -> Result<(), ReadError>,
) -> Result<(), ReadError> {
    let meta = chunks.meta_data().clone();
    let threads = std::thread::available_parallelism()
        .map_or(1, NonZero::get)
        .min(chunks.len());
    let chunks = Mutex::new(chunks);
    let (sender, blocks) = mpsc::sync_channel(threads);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (chunks, meta, sender) = (&chunks, &meta, sender.clone());
            scope.spawn(move || {
                loop {
                    // A panic while reading poisons the lock, which stops the other threads.
                    let block = catch_unwind(AssertUnwindSafe(|| {
                        let chunk = chunks.lock().ok()?.next()?;
                        Some(
                            chunk
                                .map_err(decode_error)
                                .and_then(|chunk| decompress(chunk, meta)),
                        )
                    }))
                    .unwrap_or_else(|_| {
                        Some(Err(ReadError::Decode("the decoder panicked".to_owned())))
                    });
                    let Some(block) = block else { break };
                    let failed = block.is_err();
                    // The receiver is gone once the read has failed.
                    if sender.send(block).is_err() || failed {
                        break;
                    }
                }
            });
        }
        drop(sender);
        for block in blocks {
            insert(block?)?;
        }
        Ok(())
    })
}

fn decompress(chunk: Chunk, meta: &MetaData) -> Result<UncompressedBlock, ReadError> {
    // `exr` has checked that the part exists.
    let header = &meta.headers[chunk.layer_index];
    if is_dwa(header.compression) {
        check_dwa_leader(header, &chunk)?;
    }
    UncompressedBlock::decompress_chunk(chunk, meta, false).map_err(decode_error)
}

/// Refuses a DWA chunk whose leader declares more data than its block can hold.
///
/// `exr` allocates what these counters declare before checking them, and a failed allocation
/// aborts the process, which no panic handler can catch; OpenEXR refuses such chunks.
fn check_dwa_leader(header: &Header, chunk: &Chunk) -> Result<(), ReadError> {
    let pixels = match &chunk.compressed_block {
        CompressedBlock::ScanLine(block) => &block.compressed_pixels_le,
        CompressedBlock::Tile(block) => &block.compressed_pixels_le,
        // Files with deep parts are refused before this.
        CompressedBlock::DeepScanLine(_) | CompressedBlock::DeepTile(_) => return Ok(()),
    };
    let block = header
        .get_block_data_indices(&chunk.compressed_block)
        .and_then(|indices| header.get_absolute_block_pixel_coordinates(indices))
        .map_err(decode_error)?;
    let (width, height) = (block.size.width(), block.size.height());
    let bytes = (width * height * header.channels.bytes_per_pixel) as u64;
    // `exr` reads no leader from an empty chunk or one stored uncompressed, and refuses one too
    // short to hold a leader.
    if pixels.is_empty() || pixels.len() as u64 == bytes {
        return Ok(());
    }
    let Some(leader) = pixels.first_chunk::<88>() else {
        return Ok(());
    };
    // Eleven little-endian counters: the version; the unknown section's inflated and compressed
    // sizes; the AC, DC and RLE sections' compressed sizes; the RLE section's inflated and raw
    // sizes; the AC and DC counts; the AC coder.
    let counter = |index: usize| {
        u64::from_le_bytes(
            *leader[index * 8..]
                .first_chunk::<8>()
                .expect("in the leader"),
        )
    };
    // Each 8 × 8 block of each channel, the edges padded, has one DC coefficient and at most 63
    // AC ones.
    let samples = (width.next_multiple_of(8) * height.next_multiple_of(8)) as u64
        * header.channels.list.len() as u64;
    let fits = counter(1) <= bytes
        // An RLE token never takes more than twice the bytes it stands for.
        && counter(6) <= 2 * bytes
        && counter(7) <= bytes
        && counter(8) <= samples
        && counter(9) <= samples / 64;
    if fits {
        Ok(())
    } else {
        Err(ReadError::Decode(format!(
            "a DWA chunk declares more data than its {width} × {height} block holds"
        )))
    }
}

fn is_dwa(compression: Compression) -> bool {
    matches!(compression, Compression::DWAA(_) | Compression::DWAB(_))
}

/// The first compression in the headers that OpenEXR added after `exr`'s last, found by walking
/// each header's attributes. None if there is none, or the headers end early or are malformed.
fn newer_compression(file: &mut (impl BufRead + Seek)) -> Option<&'static str> {
    const MULTIPART: u32 = 0x1000;
    file.rewind().ok()?;
    let mut start = [0; 8];
    file.read_exact(&mut start).ok()?;
    let version = u32::from_le_bytes(start[4..].try_into().expect("four bytes"));
    let (mut name, mut kind) = (Vec::new(), Vec::new());
    let mut header_is_empty = true;
    loop {
        read_text(file, &mut name)?;
        // An empty name ends a header, and in a multi-part file an empty header ends them all.
        if name.is_empty() {
            if version & MULTIPART == 0 || header_is_empty {
                return None;
            }
            header_is_empty = true;
            continue;
        }
        header_is_empty = false;
        read_text(file, &mut kind)?;
        let mut size = [0; 4];
        file.read_exact(&mut size).ok()?;
        let size = u32::try_from(i32::from_le_bytes(size)).ok()?;
        if kind == b"compression" && size == 1 {
            let mut code = [0];
            file.read_exact(&mut code).ok()?;
            // OpenEXR's codes, after HTJ2K32's 11.
            match code[0] {
                12 => return Some("LJ2K"),
                13 => return Some("ZSTD"),
                _ => {}
            }
        } else {
            file.seek_relative(size.into()).ok()?;
        }
    }
}

/// Reads a header's null-terminated name or type, at most 255 bytes, without the null.
fn read_text(file: &mut impl BufRead, text: &mut Vec<u8>) -> Option<()> {
    text.clear();
    file.take(256).read_until(0, text).ok()?;
    (text.pop() == Some(0)).then_some(())
}

fn decode_error(error: Error) -> ReadError {
    match error {
        Error::Io(error) => ReadError::Io(error),
        other => ReadError::Decode(other.to_string()),
    }
}

/// What choosing a part and warning about the others needs of each.
struct Part {
    name: Option<String>,
    /// The part's `view` attribute, which names its eye in a stereo file.
    view: Option<String>,
    channels: Vec<String>,
    deep: bool,
}

fn describe_part(header: &Header) -> Part {
    let attributes = &header.own_attributes;
    Part {
        name: attributes.layer_name.as_ref().map(Text::to_string),
        view: attributes.view_name.as_ref().map(Text::to_string),
        channels: (header.channels.list.iter())
            .map(|channel| channel.name.to_string())
            .collect(),
        deep: header.deep,
    }
}

/// A window from EXR's y-down position and size, in y-up coordinates whose row 0 is the display
/// window's bottom row.
fn flipped(header: &Header, position: Vec2<i32>, size: Vec2<usize>) -> Result<Window, ReadError> {
    let display = header.shared_attributes.display_window;
    // The y-down edge below the display window, less the one below this window.
    let bottom = i64::from(display.position.1) + display.size.1 as i64
        - (i64::from(position.1) + size.1 as i64);
    let y = i32::try_from(bottom).map_err(|_| {
        ReadError::Decode(format!("a window's bottom row, {bottom}, is out of range"))
    })?;
    window(position.0, y, size.0, size.1)
}

/// The first part with R, G and B; otherwise, why there is none.
fn choose_part(parts: &[Part]) -> Result<usize, ReadError> {
    let has = |part: &Part, name: &str| part.channels.iter().any(|channel| channel == name);
    if let Some(index) = parts
        .iter()
        .position(|part| ["R", "G", "B"].iter().all(|name| has(part, name)))
    {
        return Ok(index);
    }
    let refusal = if parts
        .iter()
        .any(|part| has(part, "Y") && (has(part, "RY") || has(part, "BY")))
    {
        Refusal::LuminanceChroma
    } else {
        Refusal::NoRgb {
            channels: parts
                .iter()
                .flat_map(|part| part.channels.iter().cloned())
                .collect(),
        }
    };
    Err(ReadError::Refused(refusal))
}

/// `views` are the shown part's, from its `multiView` attribute; the first is the one shown.
fn warnings(
    chromaticities: Option<Chromaticities>,
    pixel_aspect: f32,
    parts: &[Part],
    shown_part: usize,
    views: &[String],
) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if let Some(chromaticities) = chromaticities
        && !chromaticities.matches(&Chromaticities::REC_709)
    {
        warnings.push(Warning::Primaries(chromaticities));
    }
    // Unvalidated headers can hold any float here; the warning names it.
    if pixel_aspect.is_nan() || (pixel_aspect - 1.0).abs() > 1e-3 {
        warnings.push(Warning::PixelAspect(pixel_aspect));
    }
    let shown = &parts[shown_part];
    if parts.len() > 1 {
        warnings.push(Warning::SeveralParts {
            count: parts.len(),
            shown: match &shown.name {
                Some(name) => format!("part {shown_part} ({name})"),
                None => format!("part {shown_part}"),
            },
        });
    }
    // A stereo file is either one part with a `multiView` list, or a part per view.
    if views.len() > 1 {
        warnings.push(Warning::SeveralViews {
            views: views.to_vec(),
            shown: views[0].clone(),
        });
    } else if let Some(view) = &shown.view {
        let mut views: Vec<String> = Vec::new();
        for part in parts {
            if let Some(view) = &part.view
                && !views.contains(view)
            {
                views.push(view.clone());
            }
        }
        if views.len() > 1 {
            warnings.push(Warning::SeveralViews {
                views,
                shown: view.clone(),
            });
        }
    }
    warnings
}

#[cfg(test)]
mod tests;
