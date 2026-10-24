//! The `dense` module's kernels: rotary embedding, the paged f16 KV cache and
//! attention over it, for the families that use the dense attention set.
//!
//! Each file mirrors one `kernels/metal/dense/*.metal` source. The page-table
//! translation lives here because the cache store and both softmax entries
//! must agree on it: a store that put a row where the attention did not look
//! would be wrong with every individual kernel still self-consistent.

use shiso_core::{CoreError, CoreResult};

pub(super) mod attn;
pub(super) mod norm_rope;

/// A non-negative `i32` position or bound as an unsigned value.
///
/// The Metal kernels cast to `uint32_t`, which turns a negative into a position
/// near 2^32 and reads far outside the cache; here it is refused instead.
pub(in crate::kernels) fn position(value: i32, what: &'static str) -> CoreResult<u32> {
    u32::try_from(value).map_err(|_| CoreError::InvalidValue {
        key: what,
        value: value.to_string(),
    })
}

/// Physical cache row of logical position `p` for token `t`.
///
/// `pages[t * page_stride ..]` is the token's own table (`page_stride` 0 is the
/// shared case) and a position becomes
/// `table[p / page_tokens] * page_tokens + p % page_tokens`. An empty `pages`
/// is the "no paging" convention: the position is the row.
///
/// Unlike the device, which would read past the table, a position beyond the
/// table or a negative page is an error: that is a caller bug the device turns
/// into a read of another session's rows.
pub(in crate::kernels) fn cache_row(
    pages: &[i32],
    page_tokens: u32,
    page_stride: u32,
    t: u32,
    p: u32,
) -> CoreResult<usize> {
    if pages.is_empty() {
        return Ok(p as usize);
    }
    if page_tokens == 0 {
        return Err(CoreError::InvalidValue {
            key: "dense page_tokens",
            value: "0".into(),
        });
    }
    let bad = |what: String| CoreError::InvalidValue {
        key: "dense page table",
        value: what,
    };
    let start = t as usize * page_stride as usize;
    let index = start + (p / page_tokens) as usize;
    let page = pages.get(index).ok_or_else(|| {
        bad(format!(
            "entry {index} lies outside the {} entries",
            pages.len()
        ))
    })?;
    let page = usize::try_from(*page).map_err(|_| bad(format!("negative page {page}")))?;
    Ok(page * page_tokens as usize + (p % page_tokens) as usize)
}
