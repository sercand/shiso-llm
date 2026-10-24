//! The `audio` module's kernels: the Clef decision head's projections, norm,
//! attention and elementwise steps.
//!
//! Each file mirrors one `kernels/metal/audio/*.metal` source and keeps its
//! ownership arithmetic. Only the entries the head launches are ported; an
//! entry without a row is `Unsupported`, never a silent no-op.

use shiso_core::{CoreError, CoreResult, Extent3};

pub(super) mod attn;
pub(super) mod elementwise;
pub(super) mod gemm;
pub(super) mod norm;

/// Threads of one elementwise group (`AUDIO_LINEAR_BLOCK`): the group owns
/// this many consecutive elements, so a different block would own others.
const LINEAR_BLOCK: u32 = 256;

/// Whether `b` is a one-dimensional block of exactly `x` threads.
fn is_linear(b: Extent3, x: u32) -> bool {
    b == Extent3::new(x, 1, 1)
}

/// A field that must be a usable extent, refused when it does not fit.
fn extent(value: u32, what: &'static str) -> CoreResult<usize> {
    usize::try_from(value).map_err(|_| CoreError::InvalidValue {
        key: what,
        value: value.to_string(),
    })
}

/// `len` floats of `buf` from element `at`, or an error naming `what`.
fn floats<'a>(buf: &'a [f32], at: usize, len: usize, what: &'static str) -> CoreResult<&'a [f32]> {
    at.checked_add(len)
        .and_then(|end| buf.get(at..end))
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: format!(
                "{len} floats at {at} lie outside the {} in the buffer",
                buf.len()
            ),
        })
}
