//! Paged key/value geometry.
//!
//! A page is [`PAGE_TOKENS`] token positions of *every* key/value row a
//! position owns: the attention K and V for each full-attention layer, the
//! indexer's raw and pooled keys, and the draft head's K and V. That is 30 KiB
//! per position, hence 7.5 MiB per page across all layers.
//!
//! The page size is 256 because it is a multiple of the sparse-attention
//! indexer's 4-position
//! pooling block, so a pooled block never straddles a page and a page's pooled
//! rows are exactly its own quarter of the pooled table; a multiple of the
//! attention kernels' 32-key tile; and a multiple of the indexer score
//! kernel's 64-block tile, which then spans exactly one page. Every tiled
//! reader therefore translates one position per tile and walks the rest of the
//! tile contiguously.
//!
//! Pages are immutable once the sequence's frontier has moved past them, so the
//! rows a kernel is reading never move underneath it.

/// Token positions in one page.
pub const PAGE_TOKENS: u32 = 256;

/// `PAGE_TOKENS` as a shift, so a page lookup costs one shift instead of a
/// division.
pub const PAGE_SHIFT: u32 = 8;

/// In-page offset of a position.
pub const PAGE_MASK: u32 = PAGE_TOKENS - 1;

/// Key/value bytes for a single token position across every layer.
pub const BYTES_PER_POSITION: u64 = 30 * 1024;

/// Key/value bytes for a whole page: 7.5 MiB.
pub const BYTES_PER_PAGE: u64 = BYTES_PER_POSITION * PAGE_TOKENS as u64;

/// Rows the position table carries in front of a chunk's first token.
///
/// Almost every rope site reads its position directly, but the pooled-key
/// kernel derives a pooled key's position by rounding down to the compression
/// boundary, which can name a position up to `r - 1` slots before the chunk
/// starts. Those rows must exist in the table, hence the prefix. With a
/// compression ratio of 4 this covers the worst case with one row to spare.
pub const MROPE_BACK: u32 = 4;

const _: () = assert!(PAGE_TOKENS == 1 << PAGE_SHIFT);
const _: () = assert!(BYTES_PER_PAGE == 7_864_320);

/// Physical key/value row for logical position `p` through a session's page
/// table.
///
/// `pages[p / PAGE_TOKENS] * PAGE_TOKENS + p % PAGE_TOKENS`. Every kernel that
/// touches the cache addresses rows this way, so the host mirror has to agree
/// exactly or a checkpointed span tree and a live kernel would disagree about
/// which bytes belong to which token.
///
/// Returns `None` when `p` is past the end of the table: the caller has not
/// allocated the page yet, which is a normal condition rather than an error.
#[must_use]
pub fn kv_row(pages: &[u32], p: u32) -> Option<u32> {
    pages
        .get((p >> PAGE_SHIFT) as usize)
        .map(|&page| page * PAGE_TOKENS + (p & PAGE_MASK))
}

/// Number of pages needed to hold `positions` token positions.
#[must_use]
pub const fn pages_for(positions: u32) -> u32 {
    positions.div_ceil(PAGE_TOKENS)
}

/// The element a paged cache stores its attention keys and values as.
///
/// Family-neutral on purpose: it names a storage codec, not a model. Which
/// planes of a page take it is the stack's business - a stack that keeps a
/// plane at full width (an indexer's keys, say) keeps it there whatever this
/// says - and which kernels honour it is the backend's.
///
/// Two caches built at different elements hold different bytes for the same
/// tokens, so anything that persists a page has to carry this in its identity
/// ([`KvElem::codec`]) or it will read one build's pages as the other's.
///
/// Deliberately NOT `#[non_exhaustive]`: every stack and kernel table matches
/// on it, and a new element has to be a compile error in each of them rather
/// than a wildcard arm that quietly treats it as f16.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvElem {
    /// IEEE half precision, two bytes an element. Every stack's default.
    #[default]
    F16,
    /// FP8 E4M3 at a static scale of one, one byte an element.
    ///
    /// The store rounds to nearest even and saturates at +-448 rather than
    /// writing the not-a-number code; a reader recovers the stored value
    /// exactly. Values below 2^-10 in magnitude are stored as zero.
    Fp8E4m3,
}

impl KvElem {
    /// Bytes one stored element occupies.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::F16 => 2,
            Self::Fp8E4m3 => 1,
        }
    }

    /// A short name for the codec, stable across releases because a disk tier
    /// files pages under it: `f16`, or `e4m3s1` (E4M3, static scale 1).
    ///
    /// The scale law is part of the name because it is part of the bytes: a
    /// future per-row-scaled E4M3 cache stores the same width and must not
    /// read these pages.
    #[must_use]
    pub const fn codec(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Fp8E4m3 => "e4m3s1",
        }
    }
}

impl std::fmt::Display for KvElem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.codec())
    }
}
