//! Images as evaluation produces them, with what the artist should know about their files.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// A rectangle of whole pixels in image coordinates: y up, so `(x, y)` is the bottom-left pixel.
/// Either corner may be negative.
///
/// Y up is Nuke's and OpenFX's convention. Readers convert files that count rows down, such as
/// EXR, placing the display window's bottom row at 0 so that images of differing formats align
/// at their bottom. A file's own row origin is not kept; its column origin is.
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

    /// Whether the window has no pixels.
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// How many pixels the window has.
    pub fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// The bytes `planes` planes of `f32` over the window take. Saturates rather than wraps
    /// where that is beyond a `u64`, which no cap on an allocation admits.
    pub fn plane_bytes(self, planes: u64) -> u64 {
        (self.area().saturating_mul(planes)).saturating_mul(size_of::<f32>() as u64)
    }

    /// The smallest window holding both, or the other alone where one is empty, whatever its
    /// corner. Fails where a side of the result would not fit, as when two windows lie far apart.
    pub fn union(self, other: Window) -> Result<Window, Oversize> {
        if self.is_empty() {
            return Ok(other);
        }
        if other.is_empty() {
            return Ok(self);
        }
        let (x, y) = self.sides();
        let (other_x, other_y) = other.sides();
        Window::from_sides(
            x.start.min(other_x.start)..x.end.max(other_x.end),
            y.start.min(other_y.start)..y.end.max(other_y.end),
        )
    }

    /// The pixels in both. Where they share none, an empty window at the greater of their
    /// bottom-left corners.
    pub fn intersection(self, other: Window) -> Window {
        let (x, y) = self.sides();
        let (other_x, other_y) = other.sides();
        let (left, bottom) = (x.start.max(other_x.start), y.start.max(other_y.start));
        let (right, top) = (x.end.min(other_x.end), y.end.min(other_y.end));
        Window {
            // The greater of two corners is one of them.
            x: left as i32,
            y: bottom as i32,
            width: (right - left).max(0) as u32,
            height: (top - bottom).max(0) as u32,
        }
    }

    /// This window with `reach` more pixels on every side; an empty window stays empty. Fails
    /// where the result would not fit.
    pub fn grow(self, reach: u32) -> Result<Window, Oversize> {
        if self.is_empty() {
            return Ok(self);
        }
        let reach = i64::from(reach);
        let (x, y) = self.sides();
        Window::from_sides(
            x.start - reach..x.end + reach,
            y.start - reach..y.end + reach,
        )
    }

    /// The columns and rows the window spans, in i64 so that no arithmetic on them wraps: a
    /// corner is within `i32` and a side within `u32`, so a side's end is below 2³³.
    fn sides(self) -> (Range<i64>, Range<i64>) {
        let span = |x: i32, width: u32| i64::from(x)..i64::from(x) + i64::from(width);
        (span(self.x, self.width), span(self.y, self.height))
    }

    /// The window with these sides, or why it does not fit.
    fn from_sides(x: Range<i64>, y: Range<i64>) -> Result<Window, Oversize> {
        let oversize = Oversize {
            x: x.start,
            y: y.start,
            width: x.end - x.start,
            height: y.end - y.start,
        };
        match (
            i32::try_from(x.start),
            i32::try_from(y.start),
            u32::try_from(oversize.width),
            u32::try_from(oversize.height),
        ) {
            (Ok(x), Ok(y), Ok(width), Ok(height)) => Ok(Window {
                x,
                y,
                width,
                height,
            }),
            _ => Err(oversize),
        }
    }
}

/// A window that image coordinates cannot hold: a side beyond what an image can hold, as a union
/// of windows far apart would have, or a corner beyond `i32`, as growing a window at the edge of
/// the coordinates would. Its corner and sides, computed without wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Oversize {
    pub x: i64,
    pub y: i64,
    pub width: i64,
    pub height: i64,
}

impl fmt::Display for Oversize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if u32::try_from(self.width).is_err() || u32::try_from(self.height).is_err() {
            write!(
                f,
                "the data window would be {} × {} pixels, more than an image can hold",
                self.width, self.height
            )
        } else {
            write!(
                f,
                "the data window would start at ({}, {}), beyond image coordinates",
                self.x, self.y
            )
        }
    }
}

/// Where a plane's window overlaps another window's storage, row by row: for each row of the
/// outer window, which of the plane's samples lie in it and at which of its columns. Rows and
/// columns outside the plane's window are where an image is transparent black.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Overlap {
    /// The plane's window.
    inner: Window,
    /// The pixels in both windows.
    shared: Window,
    /// The shared pixels' bottom-left from each window's bottom-left.
    inner_offset: (usize, usize),
    outer_offset: (usize, usize),
}

impl Overlap {
    /// Of a plane over `inner` with the storage of `outer`.
    pub(crate) fn of(inner: Window, outer: Window) -> Overlap {
        let shared = inner.intersection(outer);
        if shared.is_empty() {
            // Sharing columns but no rows, or rows but no columns, is sharing no pixels.
            return Overlap {
                inner,
                shared: Window {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                },
                inner_offset: (0, 0),
                outer_offset: (0, 0),
            };
        }
        // The shared window lies in both, so its corner is at or above theirs.
        let offset = |window: Window| {
            (
                (i64::from(shared.x) - i64::from(window.x)) as usize,
                (i64::from(shared.y) - i64::from(window.y)) as usize,
            )
        };
        Overlap {
            inner,
            shared,
            inner_offset: offset(inner),
            outer_offset: offset(outer),
        }
    }

    /// The columns of an outer row that the plane's samples cover, in every row that has any.
    pub(crate) fn columns(&self) -> Range<usize> {
        self.outer_offset.0..self.outer_offset.0 + self.shared.width as usize
    }

    /// At the outer window's row `y`, counted from its bottom: the range of the plane holding
    /// the samples in that row and [`Overlap::columns`], or `None` where the plane has none.
    pub(crate) fn row(&self, y: usize) -> Option<(Range<usize>, Range<usize>)> {
        let row = y.checked_sub(self.outer_offset.1)?;
        if row >= self.shared.height as usize {
            return None;
        }
        let start = (row + self.inner_offset.1) * self.inner.width as usize + self.inner_offset.0;
        Some((start..start + self.shared.width as usize, self.columns()))
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
    warnings: Vec<Warning>,
}

impl Image {
    /// `planes` are in `Channel::ALL`'s order.
    pub(crate) fn new(
        display_window: Window,
        data_window: Window,
        planes: [Arc<[f32]>; 4],
        warnings: Vec<Warning>,
    ) -> Image {
        check_planes(&planes, data_window);
        Image {
            display_window,
            data_window,
            planes,
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
