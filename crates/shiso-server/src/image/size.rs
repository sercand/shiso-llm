//! The size grammar: what a `size` or `aspect_ratio` string means in pixels.
//!
//! Two readers share this one grammar, and they differ in one way on purpose.
//! The wire field is forgiving: a `WxH` off the 16-pixel wire grid is rounded
//! to it (ties up) and a side outside the loaded model's bounds is clamped to
//! them, because a caller that asks for 1000 pixels wants the nearest thing
//! this server can actually make, not a refusal. A size the model's own grid
//! is coarser than 16 still refuses afterwards, at
//! [`ImageGenerationJob::validate`](shiso_vision::generator::ImageGenerationJob::validate),
//! naming that grid. The operator's `--image-default-size` is exact: a default
//! that quietly moved is not the default that was configured, so it is read
//! without rounding and checked against the model's limits at boot.
//!
//! The grammar is `WxH`, the named aspect ratios, and the named resolutions
//! (`2K`, `4K`): a named resolution is the square of N*1024. The names are
//! ours to make servable, so they are rounded to the model's own
//! `size_multiple` and clamped to its bounds. `auto` is not a size here - the
//! wire handler reads it as an omitted size ([`is_auto`]) and falls to the
//! server default, which this module has no business knowing.

use shiso_vision::generator::ImageLimits;
use thiserror::Error;

/// The grid a wire `WxH` is rounded to whatever the model: the finest
/// multiple any shipped model accepts, so a caller's odd size lands near what
/// it asked for. A model with a coarser grid refuses the rounded size by name.
pub const WIRE_GRID: u32 = 16;

/// A size string the server cannot turn into pixels.
#[derive(Debug, Error)]
pub enum SizeError {
    /// The string is neither `WxH`, a named aspect ratio, nor a named
    /// resolution.
    #[error(
        "{value:?} is not a size: expected WxH (e.g. 1024x1024), a named \
         aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 2:3, 3:2, 21:9), or a \
         named resolution (e.g. 2K)"
    )]
    Malformed {
        /// The value as it arrived.
        value: String,
    },
}

/// The aspect ratios the server names, and the concrete pixels each one means.
///
/// Every pair is a multiple of 32 inside 256..=2048 (so the table is servable
/// by every shipped model as written), and every pair keeps a roughly one-megapixel area so that switching ratio
/// does not silently change what a generation costs. The pairs are exact for
/// the simple ratios (`1:1`, `4:3`, `2:3` and their transposes); `16:9` is
/// 1344x768 - 1.75 to 1, the closest the 16-grid keeps beside the megapixel -
/// and `21:9` is 1344x576, which is exact.
const ASPECT_RATIOS: [(&str, u32, u32); 8] = [
    ("1:1", 1024, 1024),
    ("16:9", 1344, 768),
    ("9:16", 768, 1344),
    ("4:3", 1152, 864),
    ("3:4", 864, 1152),
    ("2:3", 832, 1248),
    ("3:2", 1248, 832),
    ("21:9", 1344, 576),
];

/// The wire's forgiving read of a size or ratio string.
///
/// `WxH` has each side rounded to the nearest multiple of [`WIRE_GRID`] (ties
/// up) and clamped into the model's bounds; a named aspect ratio becomes its
/// table entry and a named resolution (`2K`, `4K`) the square of N*1024, both
/// rounded to the model's `size_multiple` and clamped, so `2K` is 2048x2048
/// where the model allows it and `4K` lands on the ceiling rather than being
/// refused. A ratio not in the table is refused rather than approximated:
/// the table is the published mapping, and a second way to read a ratio
/// would be a second answer to the same request.
///
/// # Errors
///
/// [`SizeError::Malformed`] when the string is none of these shapes.
pub fn normalize_size(value: &str, limits: &ImageLimits) -> Result<(u32, u32), SizeError> {
    if value.contains(':') {
        let (width, height) = aspect_ratio_size(value).ok_or_else(|| malformed(value))?;
        return Ok((
            round_side(width, limits.size_multiple, limits),
            round_side(height, limits.size_multiple, limits),
        ));
    }
    if let Some(side) = named_resolution(value) {
        let side = round_side(side, limits.size_multiple, limits);
        return Ok((side, side));
    }
    let (width, height) = split_size(value).ok_or_else(|| malformed(value))?;
    Ok((
        round_side(width, WIRE_GRID, limits),
        round_side(height, WIRE_GRID, limits),
    ))
}

/// Whether a wire `size` string names `auto`: the caller asking for the
/// server's default rather than pixels of its own.
///
/// This is not a size, so [`normalize_size`] does not read it - the wire
/// handler treats it as an omitted size and falls to the server default,
/// which is the one answer `auto` can honestly get.
#[must_use]
pub fn is_auto(value: &str) -> bool {
    value.trim().eq_ignore_ascii_case("auto")
}

/// The operator's exact read of a `WxH` string: no rounding, no clamping,
/// and no names - `auto` and `2K` are refused here, because a default that
/// quietly moved or arrived as a resolution class is not the default that
/// was configured. Whether the model can serve the pixels is checked at boot,
/// when the model is known ([`ImageOverrides::resolve`](super::ImageOverrides::resolve)).
///
/// # Errors
///
/// [`SizeError::Malformed`] for a string that is not `WxH` of two non-zero
/// numbers.
pub fn parse_exact_size(value: &str) -> Result<(u32, u32), SizeError> {
    let (width, height) = split_size(value).ok_or_else(|| malformed(value))?;
    if width == 0 || height == 0 {
        return Err(malformed(value));
    }
    Ok((width, height))
}

/// The concrete pixels a named aspect ratio means, or `None` when the server
/// does not name that ratio.
#[must_use]
fn aspect_ratio_size(value: &str) -> Option<(u32, u32)> {
    let value = value.trim();
    ASPECT_RATIOS
        .iter()
        .find(|(name, _, _)| *name == value)
        .map(|&(_, width, height)| (width, height))
}

/// The square side a named resolution means, or `None` when the string does
/// not name one.
///
/// The number is the multiplier of 1024 the name has always meant on this
/// surface (`2K` is two kilopixels on a side). The result is a raw wish, not
/// a servable size: `4K` names 4096, and it is the caller's rounding and
/// clamping that lands it on the ceiling.
fn named_resolution(value: &str) -> Option<u32> {
    let value = value.trim();
    let cut = value.rfind(['k', 'K'])?;
    let multiplier: u32 = value[..cut].trim().parse().ok()?;
    Some(multiplier.saturating_mul(1024))
}

/// The two numbers of a `WxH` string: either letter's case, spaces allowed.
fn split_size(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once(['x', 'X'])?;
    let width = width.trim().parse().ok()?;
    let height = height.trim().parse().ok()?;
    Some((width, height))
}

/// The nearest multiple of `grid`, ties up, then the model's bounds.
fn round_side(value: u32, grid: u32, limits: &ImageLimits) -> u32 {
    let rounded = (value.saturating_add(grid / 2) / grid) * grid;
    rounded.clamp(limits.min_side, limits.max_side)
}

/// The refusal for a string neither shape.
fn malformed(value: &str) -> SizeError {
    SizeError::Malformed {
        value: value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use shiso_vision::generator::ImageLimits;

    use super::{ASPECT_RATIOS, is_auto, normalize_size, parse_exact_size};

    const LIMITS: ImageLimits = ImageLimits {
        size_multiple: 16,
        min_side: 256,
        max_side: 2048,
        default_size: (1024, 1024),
        default_steps: 4,
        fixed_steps: None,
    };

    fn norm(value: &str) -> (u32, u32) {
        normalize_size(value, &LIMITS).expect("servable")
    }

    /// The table is the published mapping, so its entries are servable by
    /// construction (on a 32 grid too) and a test fails the day one is not.
    #[test]
    fn every_table_entry_is_a_servable_size() {
        let coarse = ImageLimits {
            size_multiple: 32,
            ..LIMITS
        };
        for (name, width, height) in ASPECT_RATIOS {
            for side in [width, height] {
                assert!(
                    side.is_multiple_of(32) && (256..=2048).contains(&side),
                    "{name}"
                );
            }
            assert_eq!(norm(name), (width, height));
            assert_eq!(
                normalize_size(name, &coarse).expect("a listed ratio"),
                (width, height)
            );
        }
    }

    /// Rounding is to the nearest grid multiple with ties up, and the clamp
    /// runs after the round so an extreme value lands on a grid multiple of
    /// a bound rather than beside one.
    #[test]
    fn rounding_is_nearest_with_ties_up() {
        assert_eq!(norm("992x992"), (992, 992));
        assert_eq!(norm("996x996"), (992, 992));
        assert_eq!(norm("1000x1000"), (1008, 1008));
        assert_eq!(norm("1004x1004"), (1008, 1008));
        assert_eq!(norm("10x10"), (256, 256));
        assert_eq!(norm("99999x99999"), (2048, 2048));
    }

    /// A model's own bounds are the clamp, not a fixed pair.
    #[test]
    fn the_clamp_is_the_models_bounds() {
        let narrow = ImageLimits {
            min_side: 512,
            max_side: 1536,
            ..LIMITS
        };
        assert_eq!(
            normalize_size("100x4000", &narrow).expect("clamps"),
            (512, 1536)
        );
    }

    /// A named resolution is the square of N*1024 read through the same
    /// rounding and clamping as any other side, in either letter's case:
    /// `2K` is exactly the ceiling, and `4K` names 4096 and lands on it.
    #[test]
    fn a_named_resolution_is_a_square_through_the_same_round_and_clamp() {
        assert_eq!(norm("2K"), (2048, 2048));
        assert_eq!(norm("2k"), (2048, 2048));
        assert_eq!(norm("1K"), (1024, 1024));
        assert_eq!(norm("4K"), (2048, 2048));
    }

    /// `auto` is not a size, so the grammar refuses it; the wire handler
    /// reads it as an omitted size before this module is ever asked.
    #[test]
    fn auto_is_recognized_but_is_not_a_size() {
        assert!(is_auto("auto"));
        assert!(is_auto(" Auto "));
        assert!(!is_auto("2K"));
        assert!(
            normalize_size("auto", &LIMITS).is_err(),
            "auto names the default, not pixels"
        );
    }

    /// The operator's knob takes neither name, and no zero side.
    #[test]
    fn the_exact_read_takes_no_names() {
        assert!(parse_exact_size("2K").is_err());
        assert!(parse_exact_size("auto").is_err());
        assert!(parse_exact_size("0x1024").is_err());
        assert_eq!(parse_exact_size("1000x1000").expect("exact"), (1000, 1000));
    }
}
