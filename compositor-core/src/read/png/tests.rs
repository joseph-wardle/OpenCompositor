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

/// sRGB codes 188 and 128 decoded by the IEC 61966-2-1 curve, worked out in f64 apart from the
/// reader, so that the curve is checked against something other than itself.
const SRGB_188: f32 = 0.502_886_5;
const SRGB_128: f32 = 0.215_860_5;

/// Compares within a few ulps, as `powf` is not exact.
#[track_caller]
fn assert_near<const N: usize>(actual: [f32; N], expected: [f32; N]) {
    let near = actual
        .iter()
        .zip(&expected)
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-6);
    assert!(near, "{actual:?} is not {expected:?}");
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
    let grey = SRGB_188 * half;
    assert_near(image.pixel(1, 0), [grey, grey, grey, half]);
    assert_eq!(image.pixel(2, 0), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(image.pixel(3, 0), [1.0 / 255.0, 0.0, 0.0, 1.0 / 255.0]);
    // Averaging across the edge, as any filter does, brings in no white.
    let [left, right] = [image.pixel(0, 0), image.pixel(2, 0)];
    let edge: Vec<f32> = (0..4).map(|i| (left[i] + right[i]) / 2.0).collect();
    assert_eq!(edge, [0.5, 0.0, 0.0, 0.5]);
}

#[test]
fn a_truncated_file_fails_to_read() {
    let bytes = png(
        ColorType::Rgb,
        BitDepth::Eight,
        2,
        &[0, 64, 128, 188, 255, 32],
        |_| {},
        &[],
    );
    // Shorter than the eight bytes that recognise a format, it is no format at all. Without
    // only its empty IEND chunk, the image is whole and reads.
    for length in 8..bytes.len() - 12 {
        let result = read_bytes(&bytes[..length]);
        assert!(
            matches!(result, Err(ReadError::Decode(_) | ReadError::Io(_))),
            "{length} bytes: {result:?}"
        );
    }
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
    assert_near(image.pixel(1, 0), [SRGB_188, SRGB_188, SRGB_188, 1.0]);

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

/// The red of the one pixel `tagged` writes, as read.
fn red(bytes: &[u8]) -> f32 {
    read_bytes(bytes).expect("the image reads").pixel(0, 0)[0]
}

/// A `cICP` chunk: primaries, transfer, matrix and whether full range, as the chunk stores them.
fn cicp(primaries: u8, transfer: u8, full_range: bool) -> [u8; 4] {
    [primaries, transfer, 0, full_range.into()]
}

#[test]
fn transfer_curves_and_narrow_range_are_honoured() {
    let grey = 128.0 / 255.0;
    let gamma =
        |value| move |info: &mut Info<'static>| info.source_gamma = Some(ScaledFloat::new(value));
    assert_eq!(red(&tagged(gamma(1.0), &[])), grey);
    assert_near([red(&tagged(gamma(SRGB_GAMMA), &[]))], [SRGB_128]);
    assert_eq!(red(&tagged(gamma(0.5), &[])), grey * grey);

    let with_cicp = |transfer| red(&tagged(|_| {}, &[(*b"cICP", &cicp(1, transfer, true))]));
    assert_near([with_cicp(Cicp::SRGB_TRANSFER)], [SRGB_128]);
    assert_eq!(with_cicp(Cicp::LINEAR_TRANSFER), grey);
    assert_eq!(with_cicp(Cicp::GAMMA_22_TRANSFER), grey.powf(2.2));
    assert_eq!(
        with_cicp(Cicp::BT709_TRANSFER),
        ((grey + 0.099) / 1.099).powf(1.0 / 0.45)
    );

    // Black, white and beyond white in narrow range, opaque, then below black and partly
    // covered: alpha's codes are full range.
    let narrow = png(
        ColorType::Rgba,
        BitDepth::Eight,
        2,
        &[16, 235, 255, 255, 0, 0, 0, 16],
        |_| {},
        &[(*b"cICP", &cicp(1, Cicp::LINEAR_TRANSFER, false))],
    );
    let image = read_bytes(&narrow).unwrap();
    assert_eq!(image.pixel(0, 0), [0.0, 1.0, 20.0 / 219.0 + 1.0, 1.0]);
    let alpha = 16.0 / 255.0;
    let below_black = -16.0 / 219.0 * alpha;
    assert_eq!(
        image.pixel(1, 0),
        [below_black, below_black, below_black, alpha]
    );
}

#[test]
fn primaries_and_icc_profiles_load_with_warnings() {
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

    let bt2020 = cicp(9, Cicp::SRGB_TRANSFER, true);
    assert_eq!(
        warnings(&tagged(|_| {}, &[(*b"cICP", &bt2020)])),
        [Warning::CicpPrimaries(9)]
    );
    let srgb = cicp(1, Cicp::SRGB_TRANSFER, true);
    assert_eq!(warnings(&tagged(|_| {}, &[(*b"cICP", &srgb)])), []);

    let profile = |info: &mut Info<'static>| info.icc_profile = Some(Cow::Owned(vec![0; 128]));
    assert_eq!(warnings(&tagged(profile, &[])), [Warning::IccProfile]);
    // A profile that does not decompress, which `png` drops without a word.
    let broken: &[u8] = b"profile\0\0not zlib";
    assert_eq!(
        warnings(&tagged(|_| {}, &[(*b"iCCP", broken)])),
        [Warning::IccProfile]
    );
}

#[test]
fn transfers_without_a_curve_and_matrices_other_than_rgb_are_refused() {
    let read = |chunk: [u8; 4]| read_bytes(&tagged(|_| {}, &[(*b"cICP", &chunk)]));

    assert!(matches!(
        read(cicp(9, Cicp::PQ_TRANSFER, true)),
        Err(ReadError::Refused(Refusal::PqTransfer))
    ));
    assert!(matches!(
        read(cicp(9, Cicp::HLG_TRANSFER, true)),
        Err(ReadError::Refused(Refusal::HlgTransfer))
    ));
    // SMPTE ST 428's.
    assert!(matches!(
        read(cicp(1, 17, true)),
        Err(ReadError::Refused(Refusal::Transfer(17)))
    ));
    // BT.709's YCbCr matrix.
    assert!(matches!(read([1, 13, 1, 1]), Err(ReadError::Decode(_))));
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
