//! Glyphs rasterised at the size a view shows them, packed for its atlas texture.

use std::collections::HashMap;

use ab_glyph::{
    Font, FontRef, Glyph, GlyphId, Outline, OutlineCurve, OutlinedGlyph, PxScale, PxScaleFactor,
    Rect, point,
};

use crate::text::units_per_em;

/// The atlas texture's width and height: the largest WebGL2 guarantees.
pub(crate) const ATLAS_SIZE: u32 = 2048;
/// Sizes are rasterised to the nearest of this many steps per pixel per em, so that glyphs are
/// shared between nearby zooms but drawn one texel to one pixel.
const SIZE_STEPS_PER_PIXEL: f32 = 4.0;
/// How many horizontal positions within a pixel each glyph is rasterised at.
const SUBPIXEL_BINS: f32 = 4.0;

/// Something drawn from the atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Ink {
    Glyph(GlyphId),
    /// A group header's fold caret: pointing right when folded and down when open.
    Caret {
        folded: bool,
    },
}

/// Ink placed on the physical pixel grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sprite {
    /// The top-left of its bitmap, in whole physical pixels.
    pub(crate) min: [f32; 2],
    /// x, y, width and height in the atlas, in texels.
    pub(crate) texels: [u16; 4],
}

/// A bitmap to copy into the atlas texture: coverage from 0 to 255, row by row.
#[derive(Debug)]
pub(crate) struct Upload {
    pub(crate) texels: [u16; 4],
    pub(crate) coverage: Vec<u8>,
}

/// One view's rasterised glyphs. They are kept until the atlas is full and then all cleared:
/// the next frame rasterises again only what it shows. Missing glyphs are rasterised as they are
/// met, with no per-frame budget, because views draw on demand: nothing would ask for another
/// frame to finish glyphs left pending.
#[derive(Debug, Default)]
pub(crate) struct GlyphAtlas {
    /// `None` for ink with nothing to draw, such as a space.
    entries: HashMap<Key, Option<Entry>>,
    // Bitmaps fill rows from left to right; a row is as tall as its tallest bitmap.
    row_top: u32,
    row_height: u32,
    row_end: u32,
    uploads: Vec<Upload>,
    full: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    ink: Ink,
    size_steps: u32,
    bin: u8,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    texels: [u16; 4],
    /// The bitmap's top-left relative to the pen's whole pixel.
    offset: [i32; 2],
}

impl GlyphAtlas {
    /// The ink at `pixels_per_em` with its pen at `pen` in physical pixels, rasterising it if
    /// needed. The baseline is rounded to a whole pixel. `None` if there is nothing to draw, or
    /// if the atlas has no room left, which [`GlyphAtlas::is_full`] then reports.
    pub(crate) fn sprite(
        &mut self,
        font: &FontRef<'_>,
        ink: Ink,
        pixels_per_em: f32,
        pen: [f32; 2],
    ) -> Option<Sprite> {
        let whole = [pen[0].floor(), pen[1].round()];
        let bin = ((pen[0] - whole[0]) * SUBPIXEL_BINS)
            .floor()
            .min(SUBPIXEL_BINS - 1.0);
        let key = Key {
            ink,
            size_steps: (pixels_per_em * SIZE_STEPS_PER_PIXEL).round() as u32,
            bin: bin as u8,
        };
        let entry = match self.entries.get(&key) {
            Some(&entry) => entry?,
            None => {
                let entry = match outline(font, key) {
                    Some(outlined) => Some(self.add(&outlined)?),
                    None => None,
                };
                self.entries.insert(key, entry);
                entry?
            }
        };
        Some(Sprite {
            min: [
                whole[0] + entry.offset[0] as f32,
                whole[1] + entry.offset[1] as f32,
            ],
            texels: entry.texels,
        })
    }

    /// Whether some ink did not fit since the atlas was last cleared.
    pub(crate) fn is_full(&self) -> bool {
        self.full
    }

    /// Forgets every glyph, so the texture's contents are free to overwrite.
    pub(crate) fn clear(&mut self) {
        *self = GlyphAtlas::default();
    }

    /// The bitmaps rasterised since this was last called, to copy into the texture.
    pub(crate) fn take_uploads(&mut self) -> Vec<Upload> {
        std::mem::take(&mut self.uploads)
    }

    /// Rasterises the glyph into free space, if there is any.
    fn add(&mut self, outlined: &OutlinedGlyph) -> Option<Entry> {
        let bounds = outlined.px_bounds();
        let (width, height) = (bounds.width() as u32, bounds.height() as u32);
        if self.row_end + width > ATLAS_SIZE {
            self.row_top += self.row_height;
            self.row_end = 0;
            self.row_height = 0;
        }
        if width > ATLAS_SIZE || self.row_top + height > ATLAS_SIZE {
            self.full = true;
            return None;
        }
        let texels = [self.row_end, self.row_top, width, height].map(|n| n as u16);
        self.row_end += width;
        self.row_height = self.row_height.max(height);

        let mut coverage = vec![0; (width * height) as usize];
        outlined.draw(|x, y, amount| {
            coverage[(y * width + x) as usize] = (amount.min(1.0) * 255.0).round() as u8;
        });
        self.uploads.push(Upload { texels, coverage });
        Some(Entry {
            texels,
            offset: [bounds.min.x as i32, bounds.min.y as i32],
        })
    }
}

/// The key's ink scaled and offset by its subpixel bin, or `None` if it has no outline.
fn outline(font: &FontRef<'_>, key: Key) -> Option<OutlinedGlyph> {
    let units_per_em = units_per_em(font);
    let (id, outline) = match key.ink {
        Ink::Glyph(id) => (id, font.outline(id)?),
        Ink::Caret { folded } => (GlyphId(0), caret(folded, units_per_em)),
    };
    let pixels_per_em = key.size_steps as f32 / SIZE_STEPS_PER_PIXEL;
    let per_unit = pixels_per_em / units_per_em;
    let glyph = Glyph {
        id,
        scale: PxScale::from(pixels_per_em),
        position: point(f32::from(key.bin) / SUBPIXEL_BINS, 0.0),
    };
    let scale = PxScaleFactor {
        horizontal: per_unit,
        vertical: per_unit,
    };
    Some(OutlinedGlyph::new(glyph, outline, scale))
}

/// A fold caret as a triangle in font units, y up, sitting on the baseline like a capital.
fn caret(folded: bool, units_per_em: f32) -> Outline {
    let em = |x: f32, y: f32| point(x * units_per_em, y * units_per_em);
    let corners = if folded {
        [em(0.12, 0.1), em(0.12, 0.6), em(0.42, 0.35)]
    } else {
        [em(0.02, 0.5), em(0.52, 0.5), em(0.27, 0.2)]
    };
    // ab_glyph's outline bounds put the top in `min`.
    let x = corners.map(|corner| corner.x);
    let y = corners.map(|corner| corner.y);
    let bounds = Rect {
        min: point(
            x.into_iter().fold(f32::MAX, f32::min),
            y.into_iter().fold(f32::MIN, f32::max),
        ),
        max: point(
            x.into_iter().fold(f32::MIN, f32::max),
            y.into_iter().fold(f32::MAX, f32::min),
        ),
    };
    let [a, b, c] = corners;
    Outline {
        bounds,
        curves: vec![
            OutlineCurve::Line(a, b),
            OutlineCurve::Line(b, c),
            OutlineCurve::Line(c, a),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::bundled_font;

    fn glyph(font: &FontRef<'_>, c: char) -> Ink {
        Ink::Glyph(font.glyph_id(c))
    }

    #[test]
    fn ink_is_rasterised_once_per_size_and_subpixel_position() {
        let font = bundled_font();
        let mut atlas = GlyphAtlas::default();
        let a = glyph(&font, 'a');

        let first = atlas.sprite(&font, a, 11.0, [10.0, 20.0]).unwrap();
        assert_eq!(atlas.take_uploads().len(), 1);
        let moved = atlas.sprite(&font, a, 11.05, [30.0, 39.6]).unwrap();
        assert_eq!(moved.texels, first.texels, "the same quarter pixel size");
        assert_eq!(moved.min, [first.min[0] + 20.0, first.min[1] + 20.0]);
        assert!(atlas.take_uploads().is_empty());

        atlas.sprite(&font, a, 11.0, [10.5, 20.0]).unwrap();
        atlas.sprite(&font, a, 12.0, [10.0, 20.0]).unwrap();
        atlas
            .sprite(&font, Ink::Caret { folded: true }, 11.0, [10.0, 20.0])
            .unwrap();
        assert_eq!(atlas.take_uploads().len(), 3);
    }

    #[test]
    fn a_bitmap_covers_its_ink_above_the_baseline() {
        let font = bundled_font();
        let mut atlas = GlyphAtlas::default();
        let sprite = atlas
            .sprite(&font, glyph(&font, 'H'), 20.0, [100.0, 50.0])
            .unwrap();
        let [.., width, height] = sprite.texels.map(f32::from);

        assert!(sprite.min[0] >= 100.0 && sprite.min[0] < 103.0);
        assert!(sprite.min[1] < 40.0, "a capital is most of the em tall");
        assert_eq!(sprite.min[1] + height, 50.0, "H sits on the baseline");
        let [upload] = &atlas.take_uploads()[..] else {
            panic!("one bitmap");
        };
        assert_eq!(upload.coverage.len() as f32, width * height);
        assert!(upload.coverage.contains(&255));
    }

    #[test]
    fn a_space_has_no_sprite_and_takes_no_room() {
        let font = bundled_font();
        let mut atlas = GlyphAtlas::default();
        assert_eq!(
            atlas.sprite(&font, glyph(&font, ' '), 11.0, [0.0, 0.0]),
            None
        );
        assert!(atlas.take_uploads().is_empty());
        assert!(!atlas.is_full());
    }

    #[test]
    fn a_full_atlas_reports_it_until_cleared() {
        let font = bundled_font();
        let mut atlas = GlyphAtlas::default();
        // An M this size is about 1600 texels square, so the second does not fit beside or
        // below the first.
        let huge = 2000.0;
        assert!(
            atlas
                .sprite(&font, glyph(&font, 'M'), huge, [0.0, 0.0])
                .is_some()
        );
        assert_eq!(
            atlas.sprite(&font, glyph(&font, 'W'), huge, [0.0, 0.0]),
            None
        );
        assert!(atlas.is_full());

        atlas.clear();
        assert!(!atlas.is_full());
        assert!(atlas.take_uploads().is_empty());
        assert!(
            atlas
                .sprite(&font, glyph(&font, 'W'), huge, [0.0, 0.0])
                .is_some()
        );
    }
}
