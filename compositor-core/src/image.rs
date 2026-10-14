//! Images as evaluation produces them, with a description of the file they came from.

use std::fmt;

/// A rectangle of whole pixels in image coordinates: y down, so `y` is the top row. Either
/// corner may be negative.
///
/// Y down is EXR's convention; OpenFX's y-up CPU images take it through negative row bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Window {
    pub fn contains(self, x: i32, y: i32) -> bool {
        let (x, y) = (i64::from(x), i64::from(y));
        let (left, top) = (i64::from(self.x), i64::from(self.y));
        (left..left + i64::from(self.width)).contains(&x)
            && (top..top + i64::from(self.height)).contains(&y)
    }
}

/// Scene-linear premultiplied RGBA in 32-bit float, in a working space assumed to be linear
/// Rec.709 until colour management arrives.
///
/// The display window is the frame the artist sees; the data window is where pixels are stored.
/// They are independent: the data window may be smaller, larger (overscan) or elsewhere.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    display_window: Window,
    data_window: Window,
    /// The data window's pixels, interleaved RGBA, top row first.
    pixels: Vec<f32>,
    pixel_aspect: f32,
    source: Source,
    warnings: Vec<Warning>,
}

impl Image {
    pub(crate) fn new(
        display_window: Window,
        data_window: Window,
        pixels: Vec<f32>,
        pixel_aspect: f32,
        source: Source,
        warnings: Vec<Warning>,
    ) -> Image {
        let area = data_window.width as usize * data_window.height as usize;
        assert_eq!(
            pixels.len(),
            area * 4,
            "one RGBA pixel per data window pixel"
        );
        Image {
            display_window,
            data_window,
            pixels,
            pixel_aspect,
            source,
            warnings,
        }
    }

    pub fn display_window(&self) -> Window {
        self.display_window
    }

    pub fn data_window(&self) -> Window {
        self.data_window
    }

    /// The data window's pixels, interleaved RGBA, top row first.
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    /// The pixel at a position in image coordinates. Outside the data window, an image is
    /// transparent black.
    pub fn pixel(&self, x: i32, y: i32) -> [f32; 4] {
        let window = self.data_window;
        if !window.contains(x, y) {
            return [0.0; 4];
        }
        let column = (i64::from(x) - i64::from(window.x)) as usize;
        let row = (i64::from(y) - i64::from(window.y)) as usize;
        let start = (row * window.width as usize + column) * 4;
        let mut pixel = [0.0; 4];
        pixel.copy_from_slice(&self.pixels[start..start + 4]);
        pixel
    }

    /// A pixel's width over its height. Pixels are shown square for now.
    pub fn pixel_aspect(&self) -> f32 {
        self.pixel_aspect
    }

    pub fn source(&self) -> &Source {
        &self.source
    }

    /// What the artist should know about how the file was interpreted.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }
}

/// The file an image was read from, described in our own terms so that a decoder can be replaced
/// without changing what callers see.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    Exr(ExrSource),
    Png(PngSource),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExrSource {
    /// Every part in the file, including those not shown.
    pub parts: Vec<ExrPart>,
    /// The index of the part shown.
    pub shown_part: usize,
    /// The shown part's views, from its `multiView` attribute; the first is the default view,
    /// the one shown.
    pub views: Vec<String>,
    pub chromaticities: Option<Chromaticities>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExrPart {
    pub name: Option<String>,
    /// The part's `view` attribute, which names its eye in a stereo file.
    pub view: Option<String>,
    pub channels: Vec<ExrChannel>,
    pub compression: ExrCompression,
    pub tiled: bool,
    pub deep: bool,
    pub data_window: Window,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExrChannel {
    pub name: String,
    pub sample_type: SampleType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleType {
    U32,
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExrCompression {
    None,
    Rle,
    Zips,
    Zip,
    Piz,
    Pxr24,
    B44,
    B44a,
    Dwaa,
    Dwab,
    Htj2k32,
    Htj2k256,
}

impl fmt::Display for ExrCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ExrCompression::None => "none",
            ExrCompression::Rle => "RLE",
            ExrCompression::Zips => "ZIPS",
            ExrCompression::Zip => "ZIP",
            ExrCompression::Piz => "PIZ",
            ExrCompression::Pxr24 => "PXR24",
            ExrCompression::B44 => "B44",
            ExrCompression::B44a => "B44A",
            ExrCompression::Dwaa => "DWAA",
            ExrCompression::Dwab => "DWAB",
            ExrCompression::Htj2k32 => "HTJ2K32",
            ExrCompression::Htj2k256 => "HTJ2K256",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PngSource {
    /// As stored, before expansion to 8 or 16 bits.
    pub bit_depth: u8,
    pub color_type: PngColorType,
    /// The colour chunks the file carried, whether or not they were used.
    pub srgb: bool,
    pub gamma: Option<f32>,
    pub chromaticities: Option<Chromaticities>,
    pub icc_profile: bool,
    pub cicp: Option<Cicp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PngColorType {
    Grayscale,
    GrayscaleAlpha,
    Rgb,
    Rgba,
    Indexed,
}

/// A PNG `cICP` chunk's code points, as ITU-T H.273 numbers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub primaries: u8,
    pub transfer: u8,
}

impl Cicp {
    pub const BT709_PRIMARIES: u8 = 1;
    pub const SRGB_TRANSFER: u8 = 13;
    pub const PQ_TRANSFER: u8 = 16;
    pub const HLG_TRANSFER: u8 = 18;
}

/// CIE xy coordinates of a colour space's primaries and white point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chromaticities {
    pub red: [f32; 2],
    pub green: [f32; 2],
    pub blue: [f32; 2],
    pub white: [f32; 2],
}

impl Chromaticities {
    /// Also sRGB's.
    pub const REC_709: Chromaticities = Chromaticities {
        red: [0.64, 0.33],
        green: [0.30, 0.60],
        blue: [0.15, 0.06],
        white: [0.3127, 0.3290],
    };

    /// Equal to `other` within what files store: PNG keeps five decimal digits.
    pub fn matches(&self, other: &Chromaticities) -> bool {
        close(self.red, other.red)
            && close(self.green, other.green)
            && close(self.blue, other.blue)
            && close(self.white, other.white)
    }

    /// Whether the primaries are CIE XYZ's, which OpenEXR writes for XYZ images.
    pub fn is_xyz(&self) -> bool {
        close(self.red, [1.0, 0.0]) && close(self.green, [0.0, 1.0]) && close(self.blue, [0.0, 0.0])
    }
}

fn close(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3
}

impl fmt::Display for Chromaticities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [r, g, b, w] = [self.red, self.green, self.blue, self.white];
        write!(
            f,
            "red {:.4} {:.4}, green {:.4} {:.4}, blue {:.4} {:.4}, white {:.4} {:.4}",
            r[0], r[1], g[0], g[1], b[0], b[1], w[0], w[1]
        )
    }
}

/// A file that loaded, but not entirely as it asked to be shown.
#[derive(Clone, Debug, PartialEq)]
pub enum Warning {
    /// Shown as if they were Rec.709's.
    Primaries(Chromaticities),
    /// Shown square.
    PixelAspect(f32),
    SeveralParts {
        count: usize,
        shown: String,
    },
    SeveralViews {
        views: Vec<String>,
        shown: String,
    },
    /// A PNG `gAMA` other than sRGB's, decoded as sRGB.
    Gamma(f32),
    /// An ICC profile, ignored and decoded as sRGB.
    IccProfile,
    /// `cICP` code points other than sRGB's, ignored and decoded as sRGB.
    Cicp(Cicp),
}

impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Warning::Primaries(chromaticities) => write!(
                f,
                "primaries are not Rec.709 ({chromaticities}); shown as Rec.709"
            ),
            Warning::PixelAspect(aspect) => {
                write!(f, "pixel aspect is {aspect}; pixels shown square")
            }
            Warning::SeveralParts { count, shown } => {
                write!(f, "{count} parts; showing {shown}")
            }
            Warning::SeveralViews { views, shown } => {
                write!(f, "views {}; showing {shown}", views.join(", "))
            }
            Warning::Gamma(gamma) => write!(f, "gamma is {gamma}; decoded as sRGB"),
            Warning::IccProfile => f.write_str("ICC profile ignored; decoded as sRGB"),
            Warning::Cicp(cicp) => write!(
                f,
                "cICP primaries {} and transfer {} ignored; decoded as sRGB",
                cicp.primaries, cicp.transfer
            ),
        }
    }
}
