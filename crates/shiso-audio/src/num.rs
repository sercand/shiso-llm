//! The numeric conversions the pipeline needs, audited once.
//!
//! Sample counts, frame indices and rates all stay far below 2^52, so a
//! conversion to `f64` is exact. Features are stored as `f32` because that is
//! what the encoders consume. Keeping the casts here means each `expect` below
//! is argued once, not at forty call sites.

/// `n` as `f64`; exact for every count this crate handles.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts stay below 2^52, where f64 is exact"
)]
pub(crate) fn f64_of(n: usize) -> f64 {
    n as f64
}

/// `x` narrowed to the `f32` the encoders consume.
#[expect(
    clippy::cast_possible_truncation,
    reason = "features are f32 by contract"
)]
pub(crate) fn f32_of(x: f64) -> f32 {
    x as f32
}

/// `floor(x)` for a finite, non-negative `x`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "callers pass finite non-negative positions inside a buffer"
)]
pub(crate) fn usize_floor(x: f64) -> usize {
    x.floor() as usize
}

/// `ceil(x)` as a `u32` for a finite, non-negative `x` below 2^32, saturating.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a clip limit in seconds is small and non-negative"
)]
pub(crate) fn u32_ceil(x: f64) -> u32 {
    x.ceil().clamp(0.0, f64::from(u32::MAX)) as u32
}
