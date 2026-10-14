//! Reading image files into [`Image`]s.

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek};
use std::path::Path;

use crate::image::{Image, Window};

mod exr;
mod png;

/// The most pixels a file may declare. Allocating for a corrupt or hostile header's size would
/// abort the process, which no panic handler can catch, so readers check this before allocating.
pub const MAX_PIXELS: u64 = 1 << 28;

/// Reads an EXR or PNG file, recognised by its signature rather than its name.
pub fn read_image(path: &Path) -> Result<Image, ReadError> {
    read_from(BufReader::new(File::open(path)?))
}

fn read_from(mut file: impl BufRead + Seek) -> Result<Image, ReadError> {
    let mut signature = [0; 8];
    if let Err(error) = file.read_exact(&mut signature) {
        return Err(match error.kind() {
            io::ErrorKind::UnexpectedEof => ReadError::UnknownFormat,
            _ => ReadError::Io(error),
        });
    }
    file.rewind()?;
    if signature.starts_with(&[0x76, 0x2f, 0x31, 0x01]) {
        exr::read(file)
    } else if signature == *b"\x89PNG\r\n\x1a\n" {
        png::read(file)
    } else {
        Err(ReadError::UnknownFormat)
    }
}

/// Refuses pixel storage over [`MAX_PIXELS`].
fn check_size(width: u64, height: u64) -> Result<(), ReadError> {
    match width.checked_mul(height) {
        Some(pixels) if pixels <= MAX_PIXELS => Ok(()),
        _ => Err(ReadError::TooLarge { width, height }),
    }
}

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    UnknownFormat,
    TooLarge {
        width: u64,
        height: u64,
    },
    Refused(Refusal),
    /// The decoder's own message: the file is damaged or uses something it does not support.
    Decode(String),
}

impl From<io::Error> for ReadError {
    fn from(error: io::Error) -> ReadError {
        ReadError::Io(error)
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Io(error) => error.fmt(f),
            ReadError::UnknownFormat => f.write_str("not an EXR or PNG file"),
            ReadError::TooLarge { width, height } => write!(
                f,
                "declares {width} × {height} pixels, more than the limit of {MAX_PIXELS}"
            ),
            ReadError::Refused(refusal) => refusal.fmt(f),
            ReadError::Decode(message) => write!(f, "could not decode: {message}"),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ReadError::Io(error) => Some(error),
            _ => None,
        }
    }
}

/// A readable file whose contents would be shown wrongly as linear Rec.709 RGB, refused rather
/// than shown with a warning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    XyzPrimaries,
    PqTransfer,
    HlgTransfer,
    LuminanceChroma,
    Deep,
    /// No part has channels named R, G and B. Every channel in the file, by part.
    NoRgb {
        channels: Vec<String>,
    },
    /// The part with R, G and B uses a compression we cannot decode.
    Compression(String),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::XyzPrimaries => f.write_str("contains CIE XYZ, not RGB"),
            Refusal::PqTransfer => f.write_str("contains HDR video encoded with PQ"),
            Refusal::HlgTransfer => f.write_str("contains HDR video encoded with HLG"),
            Refusal::LuminanceChroma => f.write_str("contains luminance and chroma, not RGB"),
            Refusal::Deep => f.write_str("contains only deep data"),
            Refusal::NoRgb { channels } => {
                write!(
                    f,
                    "has no R, G and B channels; contains {}",
                    channels.join(" ")
                )
            }
            Refusal::Compression(name) => write!(f, "uses {name} compression, not supported"),
        }
    }
}

/// A window from a decoder's position and size. A display window's size is not limited, as
/// nothing is stored for it, but its sides must fit a [`Window`].
fn window(x: i32, y: i32, width: usize, height: usize) -> Result<Window, ReadError> {
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(width), Ok(height)) => Ok(Window {
            x,
            y,
            width,
            height,
        }),
        _ => Err(ReadError::TooLarge {
            width: width as u64,
            height: height as u64,
        }),
    }
}
