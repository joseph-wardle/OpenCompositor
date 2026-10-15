use std::io::Cursor;

use ::exr::image::Image as ExrImage;
use ::exr::meta::attribute::{self, Text};
use ::exr::prelude::{
    AnyChannel, AnyChannels, Blocks, Encoding, FlatSamples, ImageAttributes, IntegerBounds, Layer,
    LayerAttributes, LineOrder, WritableImage,
};

use super::*;
use crate::image::Window;
use crate::read::read_from;

type Part = Layer<AnyChannels<FlatSamples>>;

const SIZE: (usize, usize) = (4, 3);

/// A part with each channel set to one value everywhere.
fn part(
    name: Option<&str>,
    origin: (i32, i32),
    size: (usize, usize),
    channels: &[(&str, f32)],
) -> Part {
    let attributes = LayerAttributes {
        layer_name: name.map(Text::from),
        layer_position: Vec2(origin.0, origin.1),
        ..LayerAttributes::default()
    };
    let channels = channels
        .iter()
        .map(|&(name, value)| AnyChannel::new(name, FlatSamples::F32(vec![value; size.0 * size.1])))
        .collect();
    Layer::new(
        size,
        attributes,
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(channels),
    )
}

fn display(x: i32, y: i32, width: usize, height: usize) -> ImageAttributes {
    ImageAttributes::new(IntegerBounds::new((x, y), (width, height)))
}

fn write(attributes: ImageAttributes, parts: Vec<Part>) -> Vec<u8> {
    let mut bytes = Vec::new();
    ExrImage::from_layers(attributes, parts)
        .write()
        .to_buffered(Cursor::new(&mut bytes))
        .expect("an EXR writes");
    bytes
}

fn rgb(value: f32) -> [(&'static str, f32); 3] {
    [("R", value), ("G", value), ("B", value)]
}

/// One part of RGB 0.18 in a 4 × 3 display window.
fn grey(attributes: ImageAttributes) -> Vec<u8> {
    write(attributes, vec![part(None, (0, 0), SIZE, &rgb(0.18))])
}

fn read_bytes(bytes: &[u8]) -> Result<Image, ReadError> {
    read_from(Cursor::new(bytes))
}

fn refusal(bytes: &[u8]) -> Refusal {
    match read_bytes(bytes) {
        Err(ReadError::Refused(refusal)) => refusal,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

fn warnings(bytes: &[u8]) -> Vec<Warning> {
    read_bytes(bytes)
        .expect("the image reads")
        .warnings()
        .to_vec()
}

/// Overwrites the value of a header attribute, found by its name and type.
fn patch_attribute(bytes: &mut [u8], name_and_type: &[u8], value: &[u8]) {
    let at = bytes
        .windows(name_and_type.len())
        .position(|window| window == name_and_type)
        .expect("the attribute is in the header");
    // The name and type are followed by the value's size, then the value.
    let start = at + name_and_type.len() + 4;
    bytes[start..start + value.len()].copy_from_slice(value);
}

/// Checks every pixel of the data window: red is the pixel's x and green its row counted down,
/// as EXR counts them, with `top` the y-up row of EXR's row 0.
fn assert_rows_counted_down(image: &Image, top: i32) {
    let window = image.data_window();
    for y in window.y..window.y + window.height as i32 {
        for x in window.x..window.x + window.width as i32 {
            // A missing alpha channel reads as opaque.
            assert_eq!(
                image.pixel(x, y),
                [x as f32, (top - y) as f32, 0.0, 1.0],
                "({x}, {y})"
            );
        }
    }
}

#[test]
fn windows_turn_y_up_about_the_display_window_and_keep_negative_origins() {
    // Red is each pixel's x and green its y, counted down, so every pixel says where it belongs.
    let (origin, size) = ((-7, -5), (16usize, 8usize));
    let mut red = Vec::new();
    let mut green = Vec::new();
    for row in 0..size.1 as i32 {
        for column in 0..size.0 as i32 {
            red.push((origin.0 + column) as f32);
            green.push((origin.1 + row) as f32);
        }
    }
    let attributes = LayerAttributes {
        layer_position: Vec2(origin.0, origin.1),
        ..LayerAttributes::default()
    };
    let channels = vec![
        AnyChannel::new("R", FlatSamples::F32(red)),
        AnyChannel::new("G", FlatSamples::F32(green)),
        AnyChannel::new("B", FlatSamples::F32(vec![0.0; size.0 * size.1])),
    ];
    let layer = Layer::new(
        size,
        attributes,
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(channels.into_iter().collect()),
    );
    // Rows 3 to 12 counted down, so row 12 becomes row 0 and row 3 becomes row 9.
    let bytes = write(display(2, 3, 10, 6), vec![layer]);

    let image = read_bytes(&bytes).unwrap();

    assert_eq!(
        image.display_window(),
        Window {
            x: 2,
            y: 0,
            width: 10,
            height: 6
        }
    );
    // Rows -5 to 2 counted down become rows 10 down to 3: below the display window's top, and
    // overhanging it.
    assert_eq!(
        image.data_window(),
        Window {
            x: -7,
            y: 6,
            width: 16,
            height: 8
        }
    );
    assert_rows_counted_down(&image, 8);
    assert_eq!(
        &image.pixels()[..4],
        [-7.0, -5.0, 0.0, 1.0],
        "the first pixel stored is the top left"
    );
    for (x, y) in [(-8, 6), (9, 6), (-7, 5), (-7, 14)] {
        assert_eq!(
            image.pixel(x, y),
            [0.0; 4],
            "outside the data window is transparent black"
        );
    }
    assert!(image.warnings().is_empty());
}

#[test]
fn a_tiled_zip_file_from_openexr_reads_exactly() {
    // A 16 × 8 data window at (-7, -5) counted down in a 10 × 10 display window, red each pixel's
    // x and green its row, written by OpenEXR 3.5 as one 64 × 64 ZIP tile, so the expected
    // pixels do not come from the `exr` crate that reads them.
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/tiled_zip.exr"
    ));

    let image = read_bytes(bytes).unwrap();

    assert_eq!(
        image.data_window(),
        Window {
            x: -7,
            y: 7,
            width: 16,
            height: 8
        }
    );
    assert_eq!(
        image.display_window(),
        Window {
            x: 0,
            y: 0,
            width: 10,
            height: 10
        }
    );
    assert_rows_counted_down(&image, 9);
    assert!(image.warnings().is_empty());
}

#[test]
fn a_truncated_file_fails_to_read() {
    let bytes = grey(display(0, 0, 4, 3));
    // Shorter than the eight bytes that recognise a format, it is no format at all.
    for length in 8..bytes.len() {
        let result = read_bytes(&bytes[..length]);
        assert!(
            matches!(result, Err(ReadError::Decode(_) | ReadError::Io(_))),
            "{length} bytes: {result:?}"
        );
    }
}

#[test]
fn files_without_rgb_are_refused_by_what_they_contain() {
    let only = |channels: &[(&str, f32)]| {
        write(
            display(0, 0, 4, 3),
            vec![part(None, (0, 0), SIZE, channels)],
        )
    };

    assert_eq!(
        refusal(&only(&[("Y", 0.5)])),
        Refusal::NoRgb {
            channels: vec!["Y".into()]
        }
    );
    assert_eq!(
        refusal(&only(&[("A", 0.5)])),
        Refusal::NoRgb {
            channels: vec!["A".into()]
        }
    );
    let prefixed = only(&[("beauty.R", 0.1), ("beauty.G", 0.1), ("beauty.B", 0.1)]);
    assert_eq!(
        refusal(&prefixed),
        Refusal::NoRgb {
            channels: ["beauty.B", "beauty.G", "beauty.R"]
                .map(String::from)
                .to_vec()
        }
    );
    assert_eq!(
        refusal(&only(&[("Y", 0.5), ("RY", 0.0), ("BY", 0.0)])),
        Refusal::LuminanceChroma
    );
}

#[test]
fn a_file_with_any_deep_part_is_refused() {
    // Written by OpenEXR 3.5, since `exr` cannot write deep data: a 1 × 1 deep part with Z,
    // then a flat part with R, G and B.
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/deep_then_flat.exr"
    ));
    assert_eq!(refusal(bytes), Refusal::Deep);
}

#[test]
fn xyz_and_unsupported_compressions_are_refused() {
    let mut xyz = display(0, 0, 4, 3);
    xyz.chromaticities = Some(attribute::Chromaticities {
        red: Vec2(1.0, 0.0),
        green: Vec2(0.0, 1.0),
        blue: Vec2(0.0, 0.0),
        white: Vec2(1.0 / 3.0, 1.0 / 3.0),
    });
    assert_eq!(refusal(&grey(xyz)), Refusal::XyzPrimaries);

    let mut htj2k = grey(display(0, 0, 4, 3));
    // HTJ2K256's code.
    patch_attribute(&mut htj2k, b"compression\0compression\0", &[10]);
    assert_eq!(refusal(&htj2k), Refusal::Compression("HTJ2K256".into()));
}

#[test]
fn compressions_newer_than_exr_are_refused_by_name() {
    // Written by OpenEXR 3.5: a 1 × 1 RGB half image with ZSTD compression.
    let mut bytes =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/zstd.exr")).to_vec();
    assert_eq!(refusal(&bytes), Refusal::Compression("ZSTD".into()));

    // LJ2K's code.
    patch_attribute(&mut bytes, b"compression\0compression\0", &[12]);
    assert_eq!(refusal(&bytes), Refusal::Compression("LJ2K".into()));

    // A code OpenEXR has not given out stays an error from `exr`.
    patch_attribute(&mut bytes, b"compression\0compression\0", &[14]);
    assert!(matches!(read_bytes(&bytes), Err(ReadError::Decode(_))));

    // In a multi-part file, the second part's.
    let parts = vec![
        part(Some("beauty"), (0, 0), SIZE, &rgb(0.18)),
        part(Some("depth"), (0, 0), SIZE, &[("Z", 1.0)]),
    ];
    let mut bytes = write(display(0, 0, 4, 3), parts);
    let name_and_type = b"compression\0compression\0";
    let second = bytes
        .windows(name_and_type.len())
        .rposition(|window| window == name_and_type)
        .expect("the second part has a compression");
    // After the name, the type and the value's size.
    bytes[second + name_and_type.len() + 4] = 13;
    assert_eq!(refusal(&bytes), Refusal::Compression("ZSTD".into()));
}

#[test]
fn a_header_declaring_too_many_pixels_is_refused_before_decoding() {
    let declaring = |side: i32| {
        let mut bytes = grey(display(0, 0, 4, 3));
        // xMin and yMin stay 0; xMax and yMax follow them.
        let mut window = [0u8; 16];
        window[8..12].copy_from_slice(&(side - 1).to_le_bytes());
        window[12..16].copy_from_slice(&(side - 1).to_le_bytes());
        patch_attribute(&mut bytes, b"dataWindow\0box2i\0", &window);
        read_bytes(&bytes)
    };

    match declaring(16384) {
        Err(ReadError::TooLarge { width, height }) => assert_eq!((width, height), (16384, 16384)),
        other => panic!("expected TooLarge, got {other:?}"),
    }
    // The largest square within the limit gets past the size check, and fails only on its
    // missing chunks.
    let at_limit = declaring(8192);
    assert!(
        matches!(at_limit, Err(ReadError::Decode(_))),
        "expected Decode, got {at_limit:?}"
    );
}

#[test]
fn colour_and_shape_the_reader_cannot_honour_load_with_warnings() {
    let mut p3 = display(0, 0, 4, 3);
    let p3_chromaticities = attribute::Chromaticities {
        red: Vec2(0.680, 0.320),
        green: Vec2(0.265, 0.690),
        blue: Vec2(0.150, 0.060),
        white: Vec2(0.3127, 0.3290),
    };
    p3.chromaticities = Some(p3_chromaticities);
    assert!(
        matches!(warnings(&grey(p3)).as_slice(), [Warning::Primaries(c)] if c.red == [0.680, 0.320])
    );

    let mut rec709 = display(0, 0, 4, 3);
    rec709.chromaticities = Some(attribute::Chromaticities {
        red: Vec2(0.64, 0.33),
        green: Vec2(0.30, 0.60),
        blue: Vec2(0.15, 0.06),
        white: Vec2(0.3127, 0.3290),
    });
    assert_eq!(warnings(&grey(rec709)), []);

    let mut anamorphic = display(0, 0, 4, 3);
    anamorphic.pixel_aspect = 2.0;
    let image = read_bytes(&grey(anamorphic)).unwrap();
    assert_eq!(image.pixel_aspect(), 2.0);
    assert_eq!(image.warnings(), [Warning::PixelAspect(2.0)]);

    for invalid in [f32::NAN, 0.0, -2.0, f32::INFINITY] {
        let mut bytes = grey(display(0, 0, 4, 3));
        patch_attribute(
            &mut bytes,
            b"pixelAspectRatio\0float\0",
            &invalid.to_le_bytes(),
        );
        let image = read_bytes(&bytes).unwrap();
        assert_eq!(image.pixel_aspect(), 1.0, "for {invalid}");
        assert!(
            matches!(image.warnings(), [Warning::PixelAspect(aspect)] if aspect.to_bits() == invalid.to_bits()),
            "for {invalid}: {:?}",
            image.warnings()
        );
    }
}

#[test]
fn the_first_part_with_rgb_is_shown_and_the_others_named() {
    let bytes = write(
        display(0, 0, 4, 3),
        vec![
            part(Some("depth"), (0, 0), SIZE, &[("Z", 5.0)]),
            part(Some("beauty"), (0, 0), SIZE, &rgb(0.18)),
            part(Some("normals"), (0, 0), SIZE, &rgb(-0.5)),
        ],
    );

    let image = read_bytes(&bytes).unwrap();

    assert_eq!(image.pixel(0, 0), [0.18, 0.18, 0.18, 1.0]);
    let Source::Exr(source) = image.source() else {
        panic!("an EXR source")
    };
    assert_eq!(source.shown_part, 1);
    let names: Vec<_> = source
        .parts
        .iter()
        .map(|part| part.name.as_deref())
        .collect();
    assert_eq!(names, [Some("depth"), Some("beauty"), Some("normals")]);
    assert_eq!(
        image.warnings(),
        [Warning::SeveralParts {
            count: 3,
            shown: "part 1 (beauty)".into()
        }]
    );
}

#[test]
fn stereo_files_show_one_view_and_name_it() {
    // One part, the default view's channels unprefixed.
    let channels = [
        ("R", 0.1),
        ("G", 0.1),
        ("B", 0.1),
        ("right.R", 0.9),
        ("right.G", 0.9),
        ("right.B", 0.9),
    ];
    let mut both = part(None, (0, 0), SIZE, &channels);
    both.attributes.multi_view_names = Some(vec![Text::from("left"), Text::from("right")]);
    let image = read_bytes(&write(display(0, 0, 4, 3), vec![both])).unwrap();
    assert_eq!(image.pixel(0, 0), [0.1, 0.1, 0.1, 1.0]);
    assert_eq!(
        image.warnings(),
        [Warning::SeveralViews {
            views: vec!["left".into(), "right".into()],
            shown: "left".into()
        }]
    );

    // A part per view.
    let eye = |name: &str, value: f32| {
        let mut part = part(Some(name), (0, 0), SIZE, &rgb(value));
        part.attributes.view_name = Some(Text::from(name));
        part
    };
    let image = read_bytes(&write(
        display(0, 0, 4, 3),
        vec![eye("left", 0.1), eye("right", 0.9)],
    ))
    .unwrap();
    assert_eq!(image.pixel(0, 0), [0.1, 0.1, 0.1, 1.0]);
    assert_eq!(
        image.warnings(),
        [
            Warning::SeveralParts {
                count: 2,
                shown: "part 0 (left)".into()
            },
            Warning::SeveralViews {
                views: vec!["left".into(), "right".into()],
                shown: "left".into()
            },
        ]
    );
}

/// 32 × 32 pixels in one chunk. In DWA it uses all four sections: half-float RGB for the lossy
/// one, half alpha for RLE, and float Z for the lossless one.
fn one_chunk(compression: Compression, blocks: Blocks) -> Vec<u8> {
    let (width, height) = (32, 32);
    let value = |value: fn(usize, usize) -> f32| {
        (0..width * height).map(move |i| value(i % width, i / width))
    };
    let half = |name: &str, f: fn(usize, usize) -> f32| {
        AnyChannel::new(
            name,
            FlatSamples::F16(value(f).map(half::f16::from_f32).collect()),
        )
    };
    let layer = Layer::new(
        (width, height),
        LayerAttributes::default(),
        Encoding {
            compression,
            blocks,
            line_order: LineOrder::Increasing,
        },
        AnyChannels::sort(
            [
                half("R", |x, _| x as f32 / 32.0),
                half("G", |_, y| y as f32 / 32.0),
                half("B", |x, y| ((x * y) % 7) as f32 / 7.0),
                half("A", |x, _| if x < 16 { 1.0 } else { 0.5 }),
                AnyChannel::new(
                    "Z",
                    FlatSamples::F32(value(|x, y| (x + y) as f32).collect()),
                ),
            ]
            .into_iter()
            .collect(),
        ),
    );
    write(display(0, 0, width, height), vec![layer])
}

/// The byte where the file's only chunk starts its compressed pixels.
fn chunk_pixels(bytes: &[u8], blocks: Blocks) -> usize {
    // The offset table's one entry points just past itself.
    let table = (0..bytes.len() - 8)
        .find(|&at| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap()) == at as u64 + 8)
        .expect("a one-chunk offset table");
    // A chunk starts with its first line's y, or its tile's four coordinates, then its byte
    // count, each an `i32`.
    let coordinates = match blocks {
        Blocks::ScanLines => 1,
        Blocks::Tiles(_) => 4,
    };
    table + 8 + (coordinates + 1) * 4
}

#[test]
fn dwa_chunks_declaring_more_than_their_block_holds_are_refused() {
    let tiles = Blocks::Tiles(Vec2(64, 64));
    for (compression, blocks) in [
        (Compression::DWAA(None), Blocks::ScanLines),
        (Compression::DWAB(None), Blocks::ScanLines),
        (Compression::DWAA(None), tiles),
    ] {
        let bytes = one_chunk(compression, blocks);
        // Eleven `u64` counters, the first the version.
        let leader = chunk_pixels(&bytes, blocks);
        assert_eq!(
            bytes[leader..leader + 8],
            2u64.to_le_bytes(),
            "a version 2 leader"
        );
        read_bytes(&bytes).expect("the unpatched file reads");
        // Each counter that sizes an allocation: unknown and RLE sizes, AC and DC counts. Without
        // the check, `exr` 1.74.2 aborts the process on these.
        for counter in [1, 6, 7, 8, 9] {
            let mut patched = bytes.clone();
            let at = leader + counter * 8;
            patched[at..at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
            assert!(
                matches!(read_bytes(&patched), Err(ReadError::Decode(reason)) if reason.contains("declares more data")),
                "{compression:?} counter {counter} passed the check"
            );
        }
    }
}

#[test]
fn a_chunk_whose_decoder_panics_fails_the_read() {
    let mut bytes = one_chunk(Compression::B44, Blocks::ScanLines);
    let pixels = chunk_pixels(&bytes, Blocks::ScanLines);
    // An all-zero B44 block subtracts its bias from zero. `exr` 1.74.2 does this unchecked, which
    // panics where overflow is checked, as in tests, and wraps in release builds.
    bytes[pixels..pixels + 14].fill(0);

    let read = read_bytes(&bytes);

    if cfg!(debug_assertions) {
        assert!(
            matches!(&read, Err(ReadError::Decode(reason)) if reason == "the decoder panicked"),
            "{read:?}"
        );
    }
}
