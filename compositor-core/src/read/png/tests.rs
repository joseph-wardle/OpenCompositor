use std::borrow::Cow;
use std::io::Cursor;

use ::png::chunk::ChunkType;
use ::png::{Encoder, Info, PixelDimensions, Unit};

use super::*;
use crate::read::read_from;

/// A one-row PNG; `configure` sets its chunks, and `extra` adds chunks as they are given.
fn png(
    color_type: ColorType,
    bit_depth: BitDepth,
    width: u32,
    data: &[u8],
    configure: impl FnOnce(&mut Info<'static>),
    extra: &[([u8; 4], &[u8])],
) -> Vec<u8> {
    let mut info = Info::with_size(width, 1);
    info.color_type = color_type;
    info.bit_depth = bit_depth;
    configure(&mut info);
    let mut bytes = Vec::new();
    let mut writer = Encoder::with_info(&mut bytes, info)
        .unwrap()
        .write_header()
        .unwrap();
    for &(kind, chunk) in extra {
        writer.write_chunk(ChunkType(kind), chunk).unwrap();
    }
    writer.write_image_data(data).unwrap();
    writer.finish().unwrap();
    bytes
}

/// One opaque grey 8-bit RGB pixel with the given chunks.
fn tagged(configure: impl FnOnce(&mut Info<'static>), extra: &[([u8; 4], &[u8])]) -> Vec<u8> {
    png(
        ColorType::Rgb,
        BitDepth::Eight,
        1,
        &[128, 128, 128],
        configure,
        extra,
    )
}

fn read_bytes(bytes: &[u8]) -> Result<Image, ReadError> {
    read_from(Cursor::new(bytes))
}

fn warnings(bytes: &[u8]) -> Vec<Warning> {
    read_bytes(bytes)
        .expect("the image reads")
        .warnings()
        .to_vec()
}

fn linear(code: u8) -> f32 {
    srgb_to_linear(f32::from(code) / 255.0)
}

fn p3() -> SourceChromaticities {
    SourceChromaticities::new(
        (0.3127, 0.3290),
        (0.680, 0.320),
        (0.265, 0.690),
        (0.150, 0.060),
    )
}

#[test]
fn the_first_row_is_the_top_and_the_windows_are_the_frame() {
    let mut info = Info::with_size(2, 2);
    info.color_type = ColorType::Rgb;
    let mut bytes = Vec::new();
    let mut writer = Encoder::with_info(&mut bytes, info)
        .unwrap()
        .write_header()
        .unwrap();
    // Red, green on the top row; blue, white below.
    writer
        .write_image_data(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .unwrap();
    writer.finish().unwrap();

    let image = read_bytes(&bytes).unwrap();

    let frame = Window {
        x: 0,
        y: 0,
        width: 2,
        height: 2,
    };
    assert_eq!(
        (image.display_window(), image.data_window()),
        (frame, frame)
    );
    assert_eq!(image.pixel(0, 0), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(image.pixel(1, 0), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(image.pixel(0, 1), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(image.pixel(1, 1), [1.0; 4]);
}

#[test]
fn colour_is_linearised_before_it_is_premultiplied() {
    let bytes = png(
        ColorType::Rgba,
        BitDepth::Eight,
        4,
        &[
            255, 255, 255, 0, // transparent, storing white
            188, 188, 188, 128, // sRGB mid grey, half covered
            255, 0, 0, 255, // opaque red
            255, 0, 0, 1, // red, barely covered
        ],
        |_| {},
        &[],
    );

    let image = read_bytes(&bytes).unwrap();

    let half = 128.0 / 255.0;
    assert_eq!(
        image.pixel(0, 0),
        [0.0; 4],
        "a transparent pixel stores no colour"
    );
    let grey = linear(188) * half;
    assert_eq!(image.pixel(1, 0), [grey, grey, grey, half]);
    assert_eq!(image.pixel(2, 0), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(image.pixel(3, 0), [1.0 / 255.0, 0.0, 0.0, 1.0 / 255.0]);
    // Averaging across the edge, as any filter does, brings in no white.
    let [left, right] = [image.pixel(0, 0), image.pixel(2, 0)];
    let edge: Vec<f32> = (0..4).map(|i| (left[i] + right[i]) / 2.0).collect();
    assert_eq!(edge, [0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn every_colour_type_expands_to_rgba() {
    let grey = png(
        ColorType::Grayscale,
        BitDepth::Eight,
        2,
        &[0, 188],
        |_| {},
        &[],
    );
    let image = read_bytes(&grey).unwrap();
    assert_eq!(
        image.pixel(1, 0),
        [linear(188), linear(188), linear(188), 1.0]
    );

    let grey_alpha = png(
        ColorType::GrayscaleAlpha,
        BitDepth::Eight,
        1,
        &[255, 51],
        |_| {},
        &[],
    );
    assert_eq!(
        read_bytes(&grey_alpha).unwrap().pixel(0, 0),
        [0.2, 0.2, 0.2, 0.2]
    );

    let palette = png(
        ColorType::Indexed,
        BitDepth::Eight,
        2,
        &[0, 1],
        |info| {
            info.palette = Some(Cow::Owned(vec![255, 0, 0, 0, 0, 255]));
            info.trns = Some(Cow::Owned(vec![51]));
        },
        &[],
    );
    let image = read_bytes(&palette).unwrap();
    assert_eq!(image.pixel(0, 0), [0.2, 0.0, 0.0, 0.2]);
    assert_eq!(
        image.pixel(1, 0),
        [0.0, 0.0, 1.0, 1.0],
        "entries past tRNS are opaque"
    );

    let sixteen: Vec<u8> = [65535u16, 0, 0, 32768]
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let rgba16 = png(ColorType::Rgba, BitDepth::Sixteen, 1, &sixteen, |_| {}, &[]);
    let image = read_bytes(&rgba16).unwrap();
    let alpha = 32768.0 / 65535.0;
    assert_eq!(image.pixel(0, 0), [alpha, 0.0, 0.0, alpha]);
    let Source::Png(source) = image.source() else {
        panic!("a PNG source")
    };
    assert_eq!(
        (source.color_type, source.bit_depth),
        (PngColorType::Rgba, 16)
    );
}

#[test]
fn hdr_transfers_are_refused() {
    // BT.2020 primaries, then the transfer, then matrix and full range as cICP stores them.
    let pq = tagged(|_| {}, &[(*b"cICP", &[9, 16, 0, 1])]);
    let hlg = tagged(|_| {}, &[(*b"cICP", &[9, 18, 0, 1])]);

    assert!(matches!(
        read_bytes(&pq),
        Err(ReadError::Refused(Refusal::PqTransfer))
    ));
    assert!(matches!(
        read_bytes(&hlg),
        Err(ReadError::Refused(Refusal::HlgTransfer))
    ));
}

#[test]
fn colour_chunks_other_than_srgb_load_with_warnings() {
    let gamma =
        |value| move |info: &mut Info<'static>| info.source_gamma = Some(ScaledFloat::new(value));
    assert_eq!(warnings(&tagged(gamma(1.0), &[])), [Warning::Gamma(1.0)]);
    assert_eq!(warnings(&tagged(gamma(SRGB_GAMMA), &[])), []);

    let p3 = |info: &mut Info<'static>| info.source_chromaticities = Some(p3());
    assert!(matches!(
        warnings(&tagged(p3, &[])).as_slice(),
        [Warning::Primaries(c)] if c.matches(&Chromaticities {
            red: [0.68, 0.32],
            green: [0.265, 0.69],
            blue: [0.15, 0.06],
            white: [0.3127, 0.329],
        })
    ));

    let profile = |info: &mut Info<'static>| info.icc_profile = Some(Cow::Owned(vec![0; 128]));
    assert_eq!(warnings(&tagged(profile, &[])), [Warning::IccProfile]);

    let bt2020: &[([u8; 4], &[u8])] = &[(*b"cICP", &[9, 13, 0, 1])];
    assert_eq!(
        warnings(&tagged(|_| {}, bt2020)),
        [Warning::Cicp(Cicp {
            primaries: 9,
            transfer: 13
        })]
    );
    assert_eq!(warnings(&tagged(|_| {}, &[(*b"cICP", &[1, 13, 0, 1])])), []);
}

#[test]
fn an_srgb_chunk_overrides_gamma_and_primaries() {
    // Written by hand: given an sRGB intent, the encoder writes sRGB's own gAMA and cHRM.
    let gamma_1 = 100_000u32.to_be_bytes();
    let p3: Vec<u8> = [31270u32, 32900, 68000, 32000, 26500, 69000, 15000, 6000]
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect();
    let bytes = tagged(
        |_| {},
        &[(*b"sRGB", &[0]), (*b"gAMA", &gamma_1), (*b"cHRM", &p3)],
    );

    let image = read_bytes(&bytes).unwrap();

    assert_eq!(image.warnings(), []);
    let Source::Png(source) = image.source() else {
        panic!("a PNG source")
    };
    assert!(source.srgb && source.gamma == Some(1.0) && source.chromaticities.is_some());
}

#[test]
fn non_square_pixels_load_with_a_warning() {
    let wide = |info: &mut Info<'static>| {
        info.pixel_dims = Some(PixelDimensions {
            xppu: 1000,
            yppu: 2000,
            unit: Unit::Meter,
        });
    };

    let image = read_bytes(&tagged(wide, &[])).unwrap();

    assert_eq!(image.pixel_aspect(), 2.0);
    assert_eq!(image.warnings(), [Warning::PixelAspect(2.0)]);
}

#[test]
fn a_header_declaring_too_many_pixels_is_refused_before_decoding() {
    let mut bytes = Vec::new();
    let mut writer = Encoder::new(&mut bytes, 1 << 20, 1 << 20)
        .write_header()
        .unwrap();
    // Reading the header stops at the first image data.
    writer.write_chunk(ChunkType(*b"IDAT"), &[]).unwrap();
    drop(writer);

    match read_bytes(&bytes) {
        Err(ReadError::TooLarge { width, height }) => {
            assert_eq!((width, height), (1 << 20, 1 << 20))
        }
        other => panic!("expected TooLarge, got {other:?}"),
    }
}
