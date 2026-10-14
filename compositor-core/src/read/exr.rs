//! OpenEXR through the `exr` crate.

use std::io::{BufRead, Seek};
use std::num::NonZero;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, mpsc};

use ::exr::block::UncompressedBlock;
use ::exr::block::chunk::{Chunk, CompressedBlock};
use ::exr::block::reader::{ChunksReader, FilteredChunksReader};
use ::exr::compression::Compression;
use ::exr::error::{Error, UnitResult};
use ::exr::image::read::image::{LayersReader, ReadLayers};
use ::exr::image::read::layers::ReadChannels;
use ::exr::math::Vec2;
use ::exr::meta::attribute::{self, Text};
use ::exr::meta::header::Header;
use ::exr::meta::{BlockDescription, MetaData};

use super::{ReadError, Refusal, check_size, window};
use crate::image::{
    Chromaticities, ExrChannel, ExrCompression, ExrPart, ExrSource, Image, SampleType, Source,
    Warning,
};

/// The data window's pixels as `exr` delivers them, collected into the image's layout.
struct Pixels {
    width: usize,
    rgba: Vec<f32>,
}

pub(super) fn read(mut file: impl BufRead + Seek + Send) -> Result<Image, ReadError> {
    // Only the headers, nothing allocated for pixels yet. Unvalidated, because validation
    // rejects deep and subsampled parts as unsupported before they can be refused for what they
    // contain.
    let meta = MetaData::read_from_buffered(&mut file, false).map_err(decode_error)?;
    for header in &meta.headers {
        check_size(header.layer_size.0 as u64, header.layer_size.1 as u64)?;
    }
    let parts = meta
        .headers
        .iter()
        .map(describe_part)
        .collect::<Result<Vec<_>, _>>()?;
    // `exr` reads no file with a deep part, not even its flat parts.
    if parts.iter().any(|part| part.deep) {
        return Err(ReadError::Refused(Refusal::Deep));
    }
    let shown_part = choose_part(&parts)?;
    let header = &meta.headers[shown_part];
    let compression = parts[shown_part].compression;
    if matches!(
        compression,
        ExrCompression::Htj2k32 | ExrCompression::Htj2k256
    ) {
        return Err(ReadError::Refused(Refusal::Compression(
            compression.to_string(),
        )));
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
    let display_window = window(
        display.position.0,
        display.position.1,
        display.size.0,
        display.size.1,
    )?;
    let pixel_aspect = header.shared_attributes.pixel_aspect;
    let source = ExrSource {
        views: header
            .own_attributes
            .multi_view_names
            .iter()
            .flatten()
            .map(Text::to_string)
            .collect(),
        parts,
        shown_part,
        chromaticities,
    };
    let warnings = warnings(&source, pixel_aspect);
    // Unvalidated headers can hold any float here; the warning names it.
    let pixel_aspect = if pixel_aspect.is_finite() && pixel_aspect > 0.0 {
        pixel_aspect
    } else {
        1.0
    };

    file.rewind()?;
    let read_layer = ::exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |size, _| Pixels {
                width: size.width(),
                rgba: vec![0.0; size.area() * 4],
            },
            // A missing alpha channel reads as 1. EXR stores premultiplied colour.
            |pixels: &mut Pixels, Vec2(x, y), (r, g, b, a): (f32, f32, f32, f32)| {
                let start = (y * pixels.width + x) * 4;
                pixels.rgba[start..start + 4].copy_from_slice(&[r, g, b, a]);
            },
        )
        .first_valid_layer();
    let chunks = ::exr::block::read(file, false).map_err(decode_error)?;
    let mut layer_reader = read_layer
        .create_layers_reader(chunks.headers())
        .map_err(decode_error)?;
    let chunks = chunks
        .filter_chunks(false, |meta, tile, block| {
            layer_reader.filter_block(meta, tile, block)
        })
        .map_err(decode_error)?;
    decode(chunks, |headers, block| {
        layer_reader.read_block(headers, block)
    })?;
    let layer = layer_reader.into_layers();
    // `first_valid_layer` takes the first part with R, G and B, as `choose_part` does, so the
    // description matches the pixels. The pixels' own window is used regardless.
    debug_assert_eq!(
        layer.attributes.layer_name.as_ref().map(Text::to_string),
        source.parts[shown_part].name
    );
    let data_window = window(
        layer.attributes.layer_position.0,
        layer.attributes.layer_position.1,
        layer.size.0,
        layer.size.1,
    )?;
    Ok(Image::new(
        display_window,
        data_window,
        layer.channel_data.pixels.rgba,
        pixel_aspect,
        Source::Exr(source),
        warnings,
    ))
}

/// Decompresses the chunks across the machine's cores, handing each block to `insert` on this
/// thread in whatever order they finish.
///
/// `exr` decompresses on a rayon pool of its own, where a panic aborts the process. Here a chunk
/// whose decoder panics fails the read instead.
fn decode(
    chunks: FilteredChunksReader<impl BufRead + Seek + Send>,
    mut insert: impl FnMut(&[Header], UncompressedBlock) -> UnitResult,
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
            insert(&meta.headers, block?).map_err(decode_error)?;
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

fn decode_error(error: Error) -> ReadError {
    match error {
        Error::Io(error) => ReadError::Io(error),
        other => ReadError::Decode(other.to_string()),
    }
}

fn describe_part(header: &Header) -> Result<ExrPart, ReadError> {
    let attributes = &header.own_attributes;
    Ok(ExrPart {
        name: attributes.layer_name.as_ref().map(Text::to_string),
        view: attributes.view_name.as_ref().map(Text::to_string),
        channels: header
            .channels
            .list
            .iter()
            .map(|channel| ExrChannel {
                name: channel.name.to_string(),
                sample_type: match channel.sample_type {
                    attribute::SampleType::U32 => SampleType::U32,
                    attribute::SampleType::F16 => SampleType::F16,
                    attribute::SampleType::F32 => SampleType::F32,
                },
            })
            .collect(),
        compression: match header.compression {
            Compression::Uncompressed => ExrCompression::None,
            Compression::RLE => ExrCompression::Rle,
            Compression::ZIP1 => ExrCompression::Zips,
            Compression::ZIP16 => ExrCompression::Zip,
            Compression::PIZ => ExrCompression::Piz,
            Compression::PXR24 => ExrCompression::Pxr24,
            Compression::B44 => ExrCompression::B44,
            Compression::B44A => ExrCompression::B44a,
            Compression::DWAA(_) => ExrCompression::Dwaa,
            Compression::DWAB(_) => ExrCompression::Dwab,
            Compression::HTJ2K32 => ExrCompression::Htj2k32,
            Compression::HTJ2K256 => ExrCompression::Htj2k256,
        },
        tiled: matches!(header.blocks, BlockDescription::Tiles(_)),
        deep: header.deep,
        data_window: window(
            attributes.layer_position.0,
            attributes.layer_position.1,
            header.layer_size.0,
            header.layer_size.1,
        )?,
    })
}

/// The first part with R, G and B; otherwise, why there is none.
fn choose_part(parts: &[ExrPart]) -> Result<usize, ReadError> {
    let has = |part: &ExrPart, name: &str| part.channels.iter().any(|c| c.name == name);
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
                .flat_map(|part| &part.channels)
                .map(|channel| channel.name.clone())
                .collect(),
        }
    };
    Err(ReadError::Refused(refusal))
}

fn warnings(source: &ExrSource, pixel_aspect: f32) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if let Some(chromaticities) = source.chromaticities
        && !chromaticities.matches(&Chromaticities::REC_709)
    {
        warnings.push(Warning::Primaries(chromaticities));
    }
    if pixel_aspect.is_nan() || (pixel_aspect - 1.0).abs() > 1e-3 {
        warnings.push(Warning::PixelAspect(pixel_aspect));
    }
    let shown = &source.parts[source.shown_part];
    if source.parts.len() > 1 {
        warnings.push(Warning::SeveralParts {
            count: source.parts.len(),
            shown: match &shown.name {
                Some(name) => format!("part {} ({name})", source.shown_part),
                None => format!("part {}", source.shown_part),
            },
        });
    }
    // A stereo file is either one part with a `multiView` list, or a part per view.
    if source.views.len() > 1 {
        warnings.push(Warning::SeveralViews {
            views: source.views.clone(),
            shown: source.views[0].clone(),
        });
    } else if let Some(view) = &shown.view {
        let mut views: Vec<String> = Vec::new();
        for part in &source.parts {
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
