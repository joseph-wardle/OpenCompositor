//! PNG through the `png` crate.

use std::io::{self, BufRead, Read, Seek, SeekFrom};

use ::png::{
    BitDepth, ColorType, Decoder, DecodingError, ScaledFloat, SourceChromaticities, Transformations,
};

use super::{ReadError, Refusal, check_size};
use crate::image::{Chromaticities, Cicp, Image, PngColorType, PngSource, Source, Warning, Window};

/// `gAMA`'s value for sRGB, which approximates sRGB's curve with a pure power.
const SRGB_GAMMA: f32 = 1.0 / 2.2;

pub(super) fn read(mut file: impl BufRead + Seek) -> Result<Image, ReadError> {
    let icc_profile = has_icc_chunk(&mut file)?;
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
        icc_profile,
        cicp: info.coding_independent_code_points.map(|cicp| Cicp {
            primaries: cicp.color_primaries,
            transfer: cicp.transfer_function,
            full_range: cicp.is_video_full_range_image,
        }),
    };
    let (decoding, mut warnings) = decoding(&source)?;
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
    let pixels = premultiplied_linear(
        &buffer[..frame.buffer_size()],
        color_type,
        bit_depth,
        decoding,
    );
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
        // `png` reports some invalid chunks, such as a `cICP` matrix other than RGB, this way.
        DecodingError::IoError(error) if error.kind() == io::ErrorKind::InvalidData => {
            ReadError::Decode(error.to_string())
        }
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

/// Whether the file has an `iCCP` chunk, which comes before the image data. `png` drops one it
/// cannot decompress without saying so.
fn has_icc_chunk(file: &mut (impl Read + Seek)) -> io::Result<bool> {
    file.seek(SeekFrom::Start(8))?;
    let mut found = false;
    let mut header = [0; 8];
    // A truncated file is `png`'s to report.
    while file.read_exact(&mut header).is_ok() {
        let length = u32::from_be_bytes(header[..4].try_into().expect("four bytes"));
        match &header[4..] {
            b"iCCP" => {
                found = true;
                break;
            }
            b"IDAT" => break,
            // The chunk's data and CRC.
            _ => file.seek(SeekFrom::Current(i64::from(length) + 4))?,
        };
    }
    file.rewind()?;
    Ok(found)
}

/// How a PNG's colour codes become linear values.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Decoding {
    transfer: Transfer,
    /// Codes from 16 to 235 at 8 bits, or 4096 to 60160 at 16, span black to white. Alpha is
    /// always full range.
    narrow_range: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Transfer {
    Srgb,
    Linear,
    /// A pure power: the decoding exponent.
    Power(f32),
    /// The inverse of BT.709's camera curve, which BT.601 and BT.2020 share.
    Bt709,
}

/// How to decode the PNG's colour, from its chunks in the specification's order of precedence:
/// `cICP`, then `iCCP`, then `sRGB`, then `gAMA` and `cHRM`. A chunk overridden by an earlier one
/// changes nothing.
///
/// A curve or a range is honoured. Primaries are warned about, as is an ICC profile, which only
/// colour management could read. A curve the reader has no formula for is refused.
fn decoding(source: &PngSource) -> Result<(Decoding, Vec<Warning>), ReadError> {
    let srgb = Decoding {
        transfer: Transfer::Srgb,
        narrow_range: false,
    };
    if let Some(cicp) = source.cicp {
        let transfer = match cicp.transfer {
            Cicp::SRGB_TRANSFER => Transfer::Srgb,
            Cicp::LINEAR_TRANSFER => Transfer::Linear,
            Cicp::GAMMA_22_TRANSFER => Transfer::Power(2.2),
            Cicp::GAMMA_28_TRANSFER => Transfer::Power(2.8),
            Cicp::BT709_TRANSFER
            | Cicp::BT601_TRANSFER
            | Cicp::BT2020_10_BIT_TRANSFER
            | Cicp::BT2020_12_BIT_TRANSFER => Transfer::Bt709,
            Cicp::PQ_TRANSFER => return Err(ReadError::Refused(Refusal::PqTransfer)),
            Cicp::HLG_TRANSFER => return Err(ReadError::Refused(Refusal::HlgTransfer)),
            other => return Err(ReadError::Refused(Refusal::Transfer(other))),
        };
        let mut warnings = Vec::new();
        if cicp.primaries != Cicp::BT709_PRIMARIES {
            warnings.push(Warning::CicpPrimaries(cicp.primaries));
        }
        let decoding = Decoding {
            transfer,
            narrow_range: !cicp.full_range,
        };
        return Ok((decoding, warnings));
    }
    if source.icc_profile {
        return Ok((srgb, vec![Warning::IccProfile]));
    }
    if source.srgb {
        return Ok((srgb, Vec::new()));
    }
    let transfer = match source.gamma {
        // `png` refuses a gamma of 0.
        Some(gamma) if (gamma - SRGB_GAMMA).abs() > 1e-3 => Transfer::Power(1.0 / gamma),
        // sRGB's own gAMA, or none: PNG's default is sRGB.
        _ => Transfer::Srgb,
    };
    let mut warnings = Vec::new();
    if let Some(chromaticities) = source.chromaticities
        && !chromaticities.matches(&Chromaticities::REC_709)
    {
        warnings.push(Warning::Primaries(chromaticities));
    }
    Ok((
        Decoding {
            transfer,
            narrow_range: false,
        },
        warnings,
    ))
}

impl Decoding {
    /// The linear value of a colour code out of `max`.
    fn decode(self, code: u16, max: u16) -> f32 {
        let encoded = if self.narrow_range {
            // 16 and 219 scaled to the bit depth, as H.273 scales them.
            let scale = (f32::from(max) + 1.0) / 256.0;
            (f32::from(code) - 16.0 * scale) / (219.0 * scale)
        } else {
            f32::from(code) / f32::from(max)
        };
        // Narrow range keeps codes outside black and white, which the curve mirrors below 0.
        let magnitude = encoded.abs();
        let linear = match self.transfer {
            Transfer::Srgb => srgb_to_linear(magnitude),
            Transfer::Linear => magnitude,
            Transfer::Power(exponent) => magnitude.powf(exponent),
            Transfer::Bt709 => bt709_to_linear(magnitude),
        };
        linear.copysign(encoded)
    }
}

/// Converts expanded samples to linear premultiplied RGBA.
///
/// PNG stores straight alpha over encoded colour. Colour is decoded to linear before it is
/// multiplied by alpha: the other order darkens every soft edge. Fully transparent pixels become
/// zero, whatever colour they stored, so filtering at an edge never brings in a fringe.
fn premultiplied_linear(
    samples: &[u8],
    color_type: ColorType,
    bit_depth: BitDepth,
    decoding: Decoding,
) -> Vec<f32> {
    let sixteen = bit_depth == BitDepth::Sixteen;
    let max = if sixteen { u16::MAX } else { u8::MAX.into() };
    // One decode per code value rather than per sample.
    let linear: Vec<f32> = (0..=max).map(|code| decoding.decode(code, max)).collect();
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

/// The inverse of BT.709's camera curve.
fn bt709_to_linear(encoded: f32) -> f32 {
    if encoded < 0.081 {
        encoded / 4.5
    } else {
        ((encoded + 0.099) / 1.099).powf(1.0 / 0.45)
    }
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
