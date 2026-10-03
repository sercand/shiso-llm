//! The size grammar: what a `size` or `aspect_ratio` string means in pixels.
//!
//! Two readers share this one grammar, and they differ in one way on purpose.
//! The wire field is forgiving: a size off the generator's 16-grid is rounded
//! to it (ties up) and a size outside the servable bounds is clamped to them,
//! because a caller that asks for 1000 pixels wants the nearest thing this
//! server can actually make, not a refusal. The operator's
//! `--image-default-size` is exact: a default that quietly moved is not the
//! default that was configured, so anything off the grid or outside the
//! bounds is refused at boot by the same parse.
//!
//! The bounds and the grid are the ones
//! [`ImageGenerationJob::validate`](shiso_vision::generator::ImageGenerationJob::validate)
//! enforces, so a size that survives this module always survives that check.

use thiserror::Error;

/// The shortest side a generated image may have, in pixels.
pub const MIN_SIDE: u32 = 256;
/// The longest side a generated image may have, in pixels.
pub const MAX_SIDE: u32 = 2048;
/// Every side is a multiple of this: the latent grid the generator denoises
/// on, so an off-grid request cannot be honored and is rounded instead.
pub const SIDE_GRID: u32 = 16;

/// A size string the server cannot turn into pixels.
#[derive(Debug, Error)]
pub enum SizeError {
    /// The string is neither `WxH` nor one of the named aspect ratios.
    #[error(
        "{value:?} is not a size: expected WxH (e.g. 1024x1024) or a named \
         aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 2:3, 3:2, 21:9)"
    )]
    Malformed {
        /// The value as it arrived.
        value: String,
    },
    /// The string parsed as `WxH` but names pixels this server will not make,
    /// and the caller asked for exactness rather than a clamp - which only
    /// the operator's setting path does.
    #[error(
        "{width}x{height} is not a servable size: both sides must be a \
         multiple of {grid} within {min}..={max}"
    )]
    Unservable {
        /// The width as parsed.
        width: u32,
        /// The height as parsed.
        height: u32,
        /// The shortest servable side, for the message.
        min: u32,
        /// The longest servable side, for the message.
        max: u32,
        /// The grid every side must sit on, for the message.
        grid: u32,
    },
}

/// The aspect ratios the server names, and the concrete pixels each one means.
///
/// Every pair is a multiple of [`SIDE_GRID`] inside [`MIN_SIDE`]..=[`MAX_SIDE`],
/// and every pair keeps a roughly one-megapixel area so that switching ratio
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
/// `WxH` has each side rounded to the nearest multiple of [`SIDE_GRID`] (ties
/// up) and clamped into [`MIN_SIDE`]..=[`MAX_SIDE`]; a named aspect ratio
/// becomes its table entry. A ratio not in the table is refused rather than
/// approximated: the table is the published mapping, and a second way to read
/// a ratio would be a second answer to the same request. Named resolutions
/// (`2K`, `4K`) are deliberately out of contract - refused, not normalized -
/// because the grammar is `WxH` plus the ratio table and nothing else.
///
/// # Errors
///
/// [`SizeError::Malformed`] when the string is neither shape.
pub fn normalize_size(value: &str) -> Result<(u32, u32), SizeError> {
    if value.contains(':') {
        return aspect_ratio_size(value).ok_or_else(|| malformed(value));
    }
    let (width, height) = split_size(value).ok_or_else(|| malformed(value))?;
    Ok((round_side(width), round_side(height)))
}

/// The operator's exact read of a `WxH` string: no rounding, no clamping.
///
/// # Errors
///
/// [`SizeError::Malformed`] for a string that is not `WxH` of two numbers,
/// and [`SizeError::Unservable`] for numbers off the grid or outside the
/// bounds.
pub fn parse_exact_size(value: &str) -> Result<(u32, u32), SizeError> {
    let (width, height) = split_size(value).ok_or_else(|| malformed(value))?;
    if !(MIN_SIDE..=MAX_SIDE).contains(&width)
        || !(MIN_SIDE..=MAX_SIDE).contains(&height)
        || !width.is_multiple_of(SIDE_GRID)
        || !height.is_multiple_of(SIDE_GRID)
    {
        return Err(SizeError::Unservable {
            width,
            height,
            min: MIN_SIDE,
            max: MAX_SIDE,
            grid: SIDE_GRID,
        });
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

/// The two numbers of a `WxH` string: either letter's case, spaces allowed.
fn split_size(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once(['x', 'X'])?;
    let width = width.trim().parse().ok()?;
    let height = height.trim().parse().ok()?;
    Some((width, height))
}

/// The nearest grid multiple, ties up, then the servable bounds.
fn round_side(value: u32) -> u32 {
    let rounded = (value.saturating_add(SIDE_GRID / 2) / SIDE_GRID) * SIDE_GRID;
    rounded.clamp(MIN_SIDE, MAX_SIDE)
}

/// The refusal for a string neither shape.
fn malformed(value: &str) -> SizeError {
    SizeError::Malformed {
        value: value.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ASPECT_RATIOS, MAX_SIDE, MIN_SIDE, SIDE_GRID, normalize_size};

    /// The table is the published mapping, so its entries are servable by
    /// construction and a test fails the day one of them is not.
    #[test]
    fn every_table_entry_is_a_servable_size() {
        for (name, width, height) in ASPECT_RATIOS {
            assert!(
                width.is_multiple_of(SIDE_GRID) && (MIN_SIDE..=MAX_SIDE).contains(&width),
                "{name}"
            );
            assert!(
                height.is_multiple_of(SIDE_GRID) && (MIN_SIDE..=MAX_SIDE).contains(&height),
                "{name}"
            );
            assert_eq!(
                normalize_size(name).expect("a listed ratio"),
                (width, height)
            );
        }
    }

    /// Rounding is to the nearest grid multiple with ties up, and the clamp
    /// runs after the round so an extreme value lands on a grid multiple of
    /// a bound rather than beside one.
    #[test]
    fn rounding_is_nearest_with_ties_up() {
        assert_eq!(normalize_size("992x992").expect("on the grid"), (992, 992));
        assert_eq!(normalize_size("996x996").expect("nearer down"), (992, 992));
        assert_eq!(
            normalize_size("1000x1000").expect("tie goes up"),
            (1008, 1008)
        );
        assert_eq!(
            normalize_size("1004x1004").expect("nearer up"),
            (1008, 1008)
        );
        assert_eq!(normalize_size("10x10").expect("clamped up"), (256, 256));
        assert_eq!(
            normalize_size("99999x99999").expect("clamped down"),
            (2048, 2048)
        );
    }
}
