//! Images as evaluation produces them, with a description of the file they came from.

use std::fmt;
use std::sync::Arc;

/// A rectangle of whole pixels in image coordinates: y up, so `(x, y)` is the bottom-left pixel.
/// Either corner may be negative.
///
/// Y up is Nuke's and OpenFX's convention; readers convert files that count rows down, such as
/// EXR, keeping the display window's bottom row at 0 so that images of differing formats align
/// at their bottom-left.
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
        let (left, bottom) = (i64::from(self.x), i64::from(self.y));
        (left..left + i64::from(self.width)).contains(&x)
            && (bottom..bottom + i64::from(self.height)).contains(&y)
    }
}

/// One of an image's channels. Only R, G, B and A exist until a workflow needs more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
    A,
}

impl Channel {
    pub const ALL: [Channel; 4] = [Channel::R, Channel::G, Channel::B, Channel::A];

    /// As files and artists name it.
    pub fn name(self) -> &'static str {
        match self {
            Channel::R => "R",
            Channel::G => "G",
            Channel::B => "B",
            Channel::A => "A",
        }
    }
}

/// Scene-linear colour in 32-bit float, in a working space assumed to be linear Rec.709 until
/// colour management arrives.
///
/// The display window is the frame the artist sees; the data window is where pixels are stored.
/// They are independent: the data window may be smaller, larger (overscan) or elsewhere.
///
/// Every channel is a plane: the data window's values, bottom row first, shared between images
/// that leave it unchanged. Every image has R, G, B and A; a file without alpha reads as opaque.
/// Colour is always premultiplied by A. No plane holds a subnormal: a reader or an operation
/// flushes one to signed zero as it writes.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    display_window: Window,
    data_window: Window,
    /// In `Channel::ALL`'s order.
    planes: [Arc<[f32]>; 4],
    pixel_aspect: f32,
    /// `None` for an image no file was read for: the empty image, and what operations make of it.
    source: Option<Source>,
    warnings: Vec<Warning>,
}

impl Image {
    /// `planes` are in `Channel::ALL`'s order.
    pub(crate) fn new(
        display_window: Window,
        data_window: Window,
        planes: [Arc<[f32]>; 4],
        pixel_aspect: f32,
        source: Source,
        warnings: Vec<Warning>,
    ) -> Image {
        check_planes(&planes, data_window);
        Image {
            display_window,
            data_window,
            planes,
            pixel_aspect,
            source: Some(source),
            warnings,
        }
    }

    /// No pixels and no frame: what an unconnected input takes.
    pub(crate) fn empty() -> Image {
        let window = Window {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        };
        let plane: Arc<[f32]> = Arc::new([]);
        Image {
            display_window: window,
            data_window: window,
            planes: [(); 4].map(|()| plane.clone()),
            pixel_aspect: 1.0,
            source: None,
            warnings: Vec::new(),
        }
    }

    /// This image with other values over the same data window, in `Channel::ALL`'s order.
    pub(crate) fn with_planes(&self, planes: [Arc<[f32]>; 4]) -> Image {
        check_planes(&planes, self.data_window);
        Image {
            display_window: self.display_window,
            data_window: self.data_window,
            planes,
            pixel_aspect: self.pixel_aspect,
            source: self.source.clone(),
            warnings: self.warnings.clone(),
        }
    }

    /// Every channel's plane, in `Channel::ALL`'s order.
    pub(crate) fn planes(&self) -> &[Arc<[f32]>; 4] {
        &self.planes
    }

    pub fn display_window(&self) -> Window {
        self.display_window
    }

    pub fn data_window(&self) -> Window {
        self.data_window
    }

    /// The channel's values over the data window, bottom row first.
    pub fn plane(&self, channel: Channel) -> &Arc<[f32]> {
        &self.planes[channel as usize]
    }

    /// The channel's value at a position in image coordinates. Outside the data window, an image
    /// is transparent black.
    pub fn value(&self, channel: Channel, x: i32, y: i32) -> f32 {
        let window = self.data_window;
        if !window.contains(x, y) {
            return 0.0;
        }
        let column = (i64::from(x) - i64::from(window.x)) as usize;
        let row = (i64::from(y) - i64::from(window.y)) as usize;
        self.plane(channel)[row * window.width as usize + column]
    }

    /// A pixel's width over its height. Pixels are shown square for now.
    pub fn pixel_aspect(&self) -> f32 {
        self.pixel_aspect
    }

    /// The file the image's pixels came from, through any operations.
    pub fn source(&self) -> Option<&Source> {
        self.source.as_ref()
    }

    /// What the artist should know about how the file was interpreted.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// The image without its planes, and the planes in `Channel::ALL`'s order.
    pub(crate) fn into_parts(self) -> (ImageInfo, [Arc<[f32]>; 4]) {
        let info = ImageInfo {
            display_window: self.display_window,
            data_window: self.data_window,
            warnings: self.warnings,
        };
        (info, self.planes)
    }
}

/// What every plane of an image meets: one value per data window pixel, and no subnormal. The
/// second is checked only in debug builds, as it reads every plane. A subnormal input is what a
/// GPU in flush-to-zero mode zeroes and the CPU would scale to a normal result, so whatever fills
/// a plane flushes as it writes rather than leaving it to the consumer.
fn check_planes(planes: &[Arc<[f32]>; 4], data_window: Window) {
    let area = data_window.width as usize * data_window.height as usize;
    assert!(
        planes.iter().all(|plane| plane.len() == area),
        "one value per data window pixel"
    );
    debug_assert!(
        planes
            .iter()
            .all(|plane| plane.iter().all(|value| !value.is_subnormal())),
        "no plane holds a subnormal"
    );
}

/// A plane of `len` zeros for a reader or an operation to fill before anything else shares it.
pub(crate) fn zeroed_plane(len: usize) -> Arc<[f32]> {
    // SAFETY: all zero bits are the f32 0.0. Zeroed memory comes from the allocator already
    // cleared, where a safe fill would write every value once more before the reader does.
    unsafe { Arc::new_zeroed_slice(len).assume_init() }
}

/// What the viewer keeps of an image once its pixels are on the GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageInfo {
    pub display_window: Window,
    pub data_window: Window,
    pub warnings: Vec<Warning>,
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

/// A PNG `cICP` chunk's code points, as ITU-T H.273 numbers them. Its matrix is always 0, RGB:
/// `png` refuses a file with any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub primaries: u8,
    pub transfer: u8,
    /// Whether codes span the whole range, rather than 16 to 235 at 8 bits, as video stores them.
    pub full_range: bool,
}

impl Cicp {
    pub const BT709_PRIMARIES: u8 = 1;
    pub const BT709_TRANSFER: u8 = 1;
    pub const GAMMA_22_TRANSFER: u8 = 4;
    pub const GAMMA_28_TRANSFER: u8 = 5;
    pub const BT601_TRANSFER: u8 = 6;
    pub const LINEAR_TRANSFER: u8 = 8;
    pub const SRGB_TRANSFER: u8 = 13;
    pub const BT2020_10_BIT_TRANSFER: u8 = 14;
    pub const BT2020_12_BIT_TRANSFER: u8 = 15;
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
    /// An ICC profile, ignored and decoded as sRGB.
    IccProfile,
    /// `cICP` primaries other than Rec.709's, shown as if they were Rec.709's.
    CicpPrimaries(u8),
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
            Warning::IccProfile => f.write_str("ICC profile ignored; decoded as sRGB"),
            Warning::CicpPrimaries(primaries) => write!(
                f,
                "cICP primaries {primaries} are not Rec.709; shown as Rec.709"
            ),
        }
    }
}

#[cfg(test)]
mod tests;
