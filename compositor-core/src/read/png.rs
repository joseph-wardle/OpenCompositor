//! PNG through the `png` crate.

use std::io::{BufRead, Seek};

use ::png::{
    BitDepth, ColorType, Decoder, DecodingError, ScaledFloat, SourceChromaticities, Transformations,
};

use super::{ReadError, Refusal, check_size};
use crate::image::{Chromaticities, Cicp, Image, PngColorType, PngSource, Source, Warning, Window};

/// `gAMA`'s value for sRGB, which approximates sRGB's curve with a pure power.
const SRGB_GAMMA: f32 = 1.0 / 2.2;

pub(super) fn read(file: impl BufRead + Seek) -> Result<Image, ReadError> {
    // `png`'s default limit bounds its row buffers and decompressed chunks, such as an ICC
    // profile, not the frame, which `check_size` bounds. Raising it lets a small file with a
    // highly compressed chunk allocate gigabytes.
    let mut decoder = Decoder::new(file);
    // Palettes and transparency become colour and alpha, low bit depths 8 bits; 16 bits stay.
    decoder.set_transformations(Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(decode_error)?;
    let info = reader.info();
    let (width, height) = (info.width, info.height);
    check_size(width.into(), height.into())?;
    let source = PngSource {
        bit_depth: info.bit_depth as u8,
        color_type: match info.color_type {
            ColorType::Grayscale => PngColorType::Grayscale,
            ColorType::GrayscaleAlpha => PngColorType::GrayscaleAlpha,
            ColorType::Rgb => PngColorType::Rgb,
            ColorType::Rgba => PngColorType::Rgba,
            ColorType::Indexed => PngColorType::Indexed,
        },
        srgb: info.srgb.is_some(),
        // `Info::source_gamma` and `source_chromaticities` are never filled when decoding.
        gamma: info.gama_chunk.map(ScaledFloat::into_value),
        chromaticities: info.chrm_chunk.map(chromaticities),
        icc_profile: info.icc_profile.is_some(),
        cicp: info.coding_independent_code_points.map(|cicp| Cicp {
            primaries: cicp.color_primaries,
            transfer: cicp.transfer_function,
        }),
    };
    let mut warnings = color_warnings(&source)?;
    // pHYs gives pixels per unit along each axis; a pixel's width over its height is their
    // inverse ratio.
    let pixel_aspect = match info.pixel_dims {
        Some(dims) if dims.xppu > 0 && dims.yppu > 0 => dims.yppu as f32 / dims.xppu as f32,
        _ => 1.0,
    };
    if (pixel_aspect - 1.0).abs() > 1e-3 {
        warnings.push(Warning::PixelAspect(pixel_aspect));
    }

    let size = reader.output_buffer_size().expect("within MAX_PIXELS");
    let mut buffer = vec![0; size];
    let frame = reader.next_frame(&mut buffer).map_err(decode_error)?;
    let (color_type, bit_depth) = reader.output_color_type();
    let pixels = premultiplied_linear(&buffer[..frame.buffer_size()], color_type, bit_depth);
    let window = Window {
        x: 0,
        y: 0,
        width,
        height,
    };
    Ok(Image::new(
        window,
        window,
        pixels,
        pixel_aspect,
        Source::Png(source),
        warnings,
    ))
}

fn decode_error(error: DecodingError) -> ReadError {
    match error {
        DecodingError::IoError(error) => ReadError::Io(error),
        other => ReadError::Decode(other.to_string()),
    }
}

fn chromaticities(chunk: SourceChromaticities) -> Chromaticities {
    let xy = |(x, y): (ScaledFloat, ScaledFloat)| [x.into_value(), y.into_value()];
    Chromaticities {
        red: xy(chunk.red),
        green: xy(chunk.green),
        blue: xy(chunk.blue),
        white: xy(chunk.white),
    }
}

/// Every PNG is decoded as sRGB. A chunk saying otherwise is refused when the result would be
/// wrong beyond recognition, and warned about otherwise.
///
/// The chunks are read in the specification's order of precedence: `cICP`, then `iCCP`, then
/// `sRGB`, then `gAMA` and `cHRM`. A chunk overridden by an earlier one changes nothing.
fn color_warnings(source: &PngSource) -> Result<Vec<Warning>, ReadError> {
    if let Some(cicp) = source.cicp {
        return match cicp.transfer {
            Cicp::PQ_TRANSFER => Err(ReadError::Refused(Refusal::PqTransfer)),
            Cicp::HLG_TRANSFER => Err(ReadError::Refused(Refusal::HlgTransfer)),
            Cicp::SRGB_TRANSFER if cicp.primaries == Cicp::BT709_PRIMARIES => Ok(Vec::new()),
            _ => Ok(vec![Warning::Cicp(cicp)]),
        };
    }
    if source.icc_profile {
        return Ok(vec![Warning::IccProfile]);
    }
    if source.srgb {
        return Ok(Vec::new());
    }
    let mut warnings = Vec::new();
    if let Some(gamma) = source.gamma
        && (gamma - SRGB_GAMMA).abs() > 1e-3
    {
        warnings.push(Warning::Gamma(gamma));
    }
    if let Some(chromaticities) = source.chromaticities
        && !chromaticities.matches(&Chromaticities::REC_709)
    {
        warnings.push(Warning::Primaries(chromaticities));
    }
    Ok(warnings)
}

/// Converts expanded samples to linear premultiplied RGBA.
///
/// PNG stores straight alpha over sRGB-encoded colour. Colour is decoded to linear before it is
/// multiplied by alpha: the other order darkens every soft edge. Fully transparent pixels become
/// zero, whatever colour they stored, so filtering at an edge never brings in a fringe.
fn premultiplied_linear(samples: &[u8], color_type: ColorType, bit_depth: BitDepth) -> Vec<f32> {
    let sixteen = bit_depth == BitDepth::Sixteen;
    let max = if sixteen { u16::MAX } else { u8::MAX.into() };
    // One decode per code value rather than per sample.
    let linear: Vec<f32> = (0..=max)
        .map(|code| srgb_to_linear(f32::from(code) / f32::from(max)))
        .collect();
    let channels = match color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => unreachable!("EXPAND turns palettes into colour"),
    };
    let sample_size = if sixteen { 2 } else { 1 };
    let mut pixels = Vec::with_capacity(samples.len() / (channels * sample_size) * 4);
    for pixel in samples.chunks_exact(channels * sample_size) {
        let code = |index: usize| match sixteen {
            true => u16::from_be_bytes([pixel[index * 2], pixel[index * 2 + 1]]),
            false => pixel[index].into(),
        };
        let (r, g, b, a) = match channels {
            1 => (code(0), code(0), code(0), max),
            2 => (code(0), code(0), code(0), code(1)),
            3 => (code(0), code(1), code(2), max),
            _ => (code(0), code(1), code(2), code(3)),
        };
        let a = f32::from(a) / f32::from(max);
        let [r, g, b] = [r, g, b].map(|code| linear[usize::from(code)] * a);
        pixels.extend_from_slice(&[r, g, b, a]);
    }
    pixels
}

/// The sRGB transfer function's decoding half.
fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests;
