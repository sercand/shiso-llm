//! The `generic` module's kernels: the backbone steps every family shares.
//!
//! Each file mirrors one `kernels/metal/generic/*.metal` source and keeps its
//! ownership arithmetic, so a group here owns exactly what the Metal
//! threadgroup of the same coordinates owns.

use shiso_core::{CoreError, CoreResult};

pub(super) mod get_rows;
pub(super) mod glu;
pub(super) mod norm;

/// A byte offset or stride as an `f32` element index.
///
/// Refused rather than rounded when it is not a multiple of four: the Metal
/// kernels cast the pointer and would read straddling bytes, and a stride that
/// is not a whole number of elements is a caller bug that rounding would hide.
pub(super) fn f32_index(bytes: u64, what: &'static str) -> CoreResult<usize> {
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
pub(super) fn count(value: i32, what: &'static str) -> CoreResult<usize> {
    usize::try_from(value).map_err(|_| CoreError::InvalidValue {
        key: what,
        value: value.to_string(),
    })
}
