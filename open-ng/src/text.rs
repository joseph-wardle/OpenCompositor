//! The bundled font and the text measurement that node layout and glyph drawing share.
//!
//! Text is set from advances alone: no kerning, shaping or fallback fonts. A character the
//! font lacks takes the `.notdef` advance, so a measured width is the width drawn.

use ab_glyph::{Font, FontRef};

/// Geist Regular from the Geist font release (vercel/geist-font), under the SIL Open Font
/// License in `fonts/OFL.txt`. The upstream static TTF is subset with
///
/// ```text
/// pyftsubset Geist-Regular.ttf --output-file=Geist-Regular.ttf --layout-features='' \
///   --no-hinting --desubroutinize \
///   --unicodes="U+0020-007E,U+00A0-017F,U+2010-2027,U+2030-203A,U+2190-2193"
/// ```
///
/// Dropping the layout features removes kerning, which measurement ignores anyway, and hinting,
/// which `ab_glyph` ignores.
const GEIST_REGULAR: &[u8] = include_bytes!("../fonts/Geist-Regular.ttf");

/// The bundled font. Parsing it only reads the table directory, so each owner parses its own.
pub(crate) fn bundled_font() -> FontRef<'static> {
    FontRef::try_from_slice(GEIST_REGULAR).expect("the bundled font parses")
}

/// The width of `text` in one line at a font size of `size` units per em.
pub(crate) fn text_width(font: &FontRef<'_>, text: &str, size: f32) -> f32 {
    let advance: f32 = text
        .chars()
        .map(|c| font.h_advance_unscaled(font.glyph_id(c)))
        .sum();
    advance * size / units_per_em(font)
}

fn units_per_em(font: &FontRef<'_>) -> f32 {
    font.units_per_em()
        .expect("the bundled font has a head table")
}

#[cfg(test)]
mod tests {
    use super::*;

    use ab_glyph::GlyphId;

    #[test]
    fn the_bundled_font_covers_printable_ascii_and_latin_1() {
        let font = bundled_font();
        // The soft hyphen is the one gap in these ranges; it is invisible when not breaking.
        let missing: Vec<char> = (' '..='~')
            .chain('\u{a0}'..='\u{ff}')
            .filter(|&c| c != '\u{ad}' && font.glyph_id(c) == GlyphId(0))
            .collect();
        assert_eq!(missing, []);
    }

    #[test]
    fn width_is_the_sum_of_advances_without_kerning() {
        let font = bundled_font();
        // A pair that kerns in most fonts.
        let pair = text_width(&font, "AV", 11.0);
        let apart = text_width(&font, "A", 11.0) + text_width(&font, "V", 11.0);
        assert!((pair - apart).abs() < 1e-4);
        assert!(pair > 0.0);
        assert_eq!(text_width(&font, "", 11.0), 0.0);
    }

    #[test]
    fn width_scales_with_font_size() {
        let font = bundled_font();
        let small = text_width(&font, "Blur radius", 11.0);
        let large = text_width(&font, "Blur radius", 22.0);
        assert!((large - 2.0 * small).abs() < 1e-4);
    }

    #[test]
    fn a_missing_character_takes_the_notdef_advance() {
        let font = bundled_font();
        let notdef = font.h_advance_unscaled(GlyphId(0)) * 10.0 / units_per_em(&font);
        assert!(notdef > 0.0);
        // A hyphen variant outside the subset and a character outside every range.
        assert_eq!(text_width(&font, "\u{2010}", 10.0), notdef);
        assert_eq!(text_width(&font, "字", 10.0), notdef);
    }
}
