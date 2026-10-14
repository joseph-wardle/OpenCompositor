//! OpenEXR through the `exr` crate.

use std::io::{BufRead, Seek};

use ::exr::compression::Compression;
use ::exr::error::Error;
use ::exr::image::read::image::ReadLayers;
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

pub(super) fn read(mut file: impl BufRead + Seek) -> Result<Image, ReadError> {
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

    file.rewind()?;
    let image = ::exr::prelude::read()
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
        .first_valid_layer()
        .all_attributes()
        .from_buffered(file)
        .map_err(decode_error)?;
    let layer = image.layer_data;
    // `first_valid_layer` takes the first part with R, G and B, as `choose_part` does.
    debug_assert_eq!(
        layer.attributes.layer_name.as_ref().map(Text::to_string),
        source.parts[shown_part].name
    );
    Ok(Image::new(
        display_window,
        source.parts[shown_part].data_window,
        layer.channel_data.pixels.rgba,
        pixel_aspect,
        Source::Exr(source),
        warnings,
    ))
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

/// The first part with R, G and B that is not deep; otherwise, why there is none.
fn choose_part(parts: &[ExrPart]) -> Result<usize, ReadError> {
    let has = |part: &ExrPart, name: &str| part.channels.iter().any(|c| c.name == name);
    if let Some(index) = parts
        .iter()
        .position(|part| !part.deep && ["R", "G", "B"].iter().all(|name| has(part, name)))
    {
        return Ok(index);
    }
    let refusal = if parts.iter().all(|part| part.deep) {
        Refusal::Deep
    } else if parts
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
    if (pixel_aspect - 1.0).abs() > 1e-3 {
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
