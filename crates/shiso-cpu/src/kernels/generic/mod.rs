//! The `generic` module's kernels: the backbone steps every family shares.
//!
//! Each file mirrors one `kernels/metal/generic/*.metal` source and keeps its
//! ownership arithmetic, so a group here owns exactly what the Metal
//! threadgroup of the same coordinates owns.

use shiso_core::{CoreError, CoreResult};

use crate::q8::{BLOCK_Q8_0_BYTES, QK8_0};

pub(super) mod get_rows;
pub(super) mod glu;
pub(super) mod mul_mv_ext;
pub(super) mod mul_mv_q8_0;
pub(super) mod norm;

/// A byte offset or stride as an `f32` element index.
///
/// Refused rather than rounded when it is not a multiple of four: the Metal
/// kernels cast the pointer and would read straddling bytes, and a stride that
/// is not a whole number of elements is a caller bug that rounding would hide.
pub(in crate::kernels) fn f32_index(bytes: u64, what: &'static str) -> CoreResult<usize> {
    if !bytes.is_multiple_of(4) {
        return Err(CoreError::InvalidValue {
            key: what,
            value: format!("{bytes} bytes is not a multiple of 4"),
        });
    }
    usize::try_from(bytes / 4).map_err(|_| CoreError::InvalidValue {
        key: what,
        value: format!("{bytes} bytes does not fit an index"),
    })
}

/// A non-negative `i32` argument field as a `usize`.
pub(in crate::kernels) fn count(value: i32, what: &'static str) -> CoreResult<usize> {
    usize::try_from(value).map_err(|_| CoreError::InvalidValue {
        key: what,
        value: value.to_string(),
    })
}

/// Bytes of one `Q8_0` row of `ne00` elements, refusing a width that is not
/// whole blocks (the shared dot would silently drop the tail in release).
pub(in crate::kernels) fn q8_row_bytes(ne00: i32, what: &'static str) -> CoreResult<usize> {
    let ne00 = count(ne00, what)?;
    if ne00 == 0 || !ne00.is_multiple_of(QK8_0) {
        return Err(CoreError::InvalidValue {
            key: what,
            value: format!("{ne00} is not a positive multiple of {QK8_0}"),
        });
    }
    Ok(ne00 / QK8_0 * BLOCK_Q8_0_BYTES)
}

/// `len` bytes of `buf` at the byte offset `start`, or an error naming `what`.
pub(in crate::kernels) fn bytes_at<'a>(
    buf: &'a [u8],
    start: u64,
    len: usize,
    what: &'static str,
) -> CoreResult<&'a [u8]> {
    usize::try_from(start)
        .ok()
        .and_then(|s| buf.get(s..s.checked_add(len)?))
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: format!("{len} bytes at {start} lie outside the buffer"),
        })
}

/// `len` floats of `buf` at the byte offset `start`, or an error naming `what`.
pub(in crate::kernels) fn floats_at<'a>(
    buf: &'a [f32],
    start: u64,
    len: usize,
    what: &'static str,
) -> CoreResult<&'a [f32]> {
    let at = f32_index(start, what)?;
    buf.get(at..at.saturating_add(len))
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: format!("{len} floats at byte {start} lie outside the buffer"),
        })
}

/// A broadcast ratio, which must be positive: it divides a plane index.
pub(in crate::kernels) fn ratio(value: i16, what: &'static str) -> CoreResult<u64> {
    u64::try_from(value)
        .ok()
        .filter(|v| *v > 0)
        .ok_or_else(|| CoreError::InvalidValue {
            key: what,
            value: value.to_string(),
        })
}
