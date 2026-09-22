//! The serialized path: the payload a spilled entry carries behind its envelope.
//!
//! [`crate::spill`] owns the 48-byte file envelope - the name, the budget, the
//! timestamps. This module owns what sits behind it: one tree path, written so that
//! a reader can size every section from the header alone and refuse a file it would
//! otherwise mis-parse into confident garbage.
//!
//! # The byte table
//!
//! Everything is little-endian. The header is [`PathWords`], a `#[repr(C)]` record of
//! nineteen `u32`s read and written through `bytemuck` rather than by offset
//! arithmetic, and the four sections follow it in this order:
//!
//! | offset | bytes | what |
//! |---|---|---|
//! | `0` | `76` | [`PathWords`]: magic, version, then the seventeen geometry words |
//! | `76` | `4 * tokens` | the path's token ids, root to frontier, one `u32` each |
//! | `76 + 4*tokens` | `4 * vocab` | the path-final logits row, `f32` |
//! | then | `state_bytes` | the recurrent state, in checkpoint-slot order |
//! | then | `pages * page_bytes` | the page section, one contiguous run per page |
//!
//! `state_bytes` is `n_gdn_layers * (gdn_conv_elements + gdn_state_elements) * 4`,
//! plus `ple_window_elements * 4`, plus `hc_width * 4` when a draft head is present.
//! It is derived from the header rather than stored, so a file cannot claim a
//! section length its own geometry does not imply.
//!
//! # Why the state precedes the pages
//!
//! The recurrent state is fixed-size and the page section is not, so a reader that
//! wants only the state - to price an entry, or to refuse one - reads a bounded
//! prefix of the file. Putting the pages first would make that read the whole file.
//!
//! # What this module never interprets
//!
//! The state and page sections are opaque bytes: the state section is exactly what
//! one checkpoint slot holds and the page section is exactly what the engine's page
//! copy produces. That is what keeps every model's layer widths out of this crate -
//! they arrive as element counts in the header, are compared, and are never used to
//! index anything.

use bytemuck::{Pod, Zeroable};

use crate::ids::to_u32_wide;

/// The serialized path: one tree path, self-describing.
///
/// The layout is the C payload's, word for word: 19 little-endian `u32` words, then
/// the path's token ids, then the path-final logits row, then the recurrent state,
/// then one contiguous run of bytes per logical page. Every number a reader needs to
/// size its read is in the header beside the version, which is what lets a file
/// written by a differently shaped build be refused by name instead of being
/// mis-parsed into garbage rows.
///
/// The state and page sections are opaque here on purpose: the state section is
/// exactly what one checkpoint slot holds and the page section is exactly what the
/// engine's page copy produces, so the cache moves those bytes without interpreting
/// them and no model's layer counts appear in this crate.
///
/// The field set is the on-disk word set, so it is deliberately *not*
/// `#[non_exhaustive]`: a field cannot be added here without a version bump that
/// every reader in the directory would fail, and a constructor would hide that
/// a caller left one of the 19 words unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathHeader {
    /// Bytes per pool page, which is the stride of the page section.
    pub page_bytes: u32,
    /// Tokens the path covers. Zero is never valid.
    pub tokens: u32,
    /// Positions per page.
    pub page_tokens: u32,
    /// Logical pages, which the reader checks against `tokens`.
    pub pages: u32,
    /// Bytes of one row across every KV buffer.
    pub kv_width: u32,
    /// Layers the page section interleaves.
    pub n_layers: u32,
    /// Layers that contribute KV.
    pub n_attn_layers: u32,
    /// Layers that contribute recurrent state.
    pub n_gdn_layers: u32,
    /// Whether the page section carries indexer caches. An operator toggle.
    pub indexer: bool,
    /// Whether the page section carries a draft head. An operator toggle.
    pub draft_head: bool,
    /// Vocabulary size, which is the length of the logits row in `f32`s.
    pub vocab: u32,
    /// Elements of one recurrent layer's conv window.
    pub gdn_conv_elements: u32,
    /// Elements of one recurrent layer's state matrix.
    pub gdn_state_elements: u32,
    /// Elements of the PLE block's dilated conv window; zero when there is none.
    pub ple_window_elements: u32,
    /// Width of the hyper-connection plane; zero when there is none.
    pub hc_width: u32,
    /// Whether a pending draft row follows the state section.
    pub pending_draft_row: bool,
    /// The first position that pending row describes.
    pub pending_pos0: u32,
}

/// The payload magic, a four-byte word so it needs no alignment.
const PATH_MAGIC: u32 = u32::from_le_bytes(*b"Q4EP");
/// The payload layout version this build writes and accepts.
const PATH_VERSION: u32 = 1;
/// Bytes in one `f32`, which is what every element count in the header counts.
const F32_BYTES: u64 = 4;

/// The header exactly as it sits in the file: nineteen little-endian `u32`s.
///
/// A `#[repr(C)]` record rather than a `[u32; 19]` indexed by hand, because the
/// field order *is* the byte order and a named field cannot be read out of the wrong
/// slot. Every word is a `u32` even where the value is a flag: the on-disk width is
/// four bytes, and a `bool` field would not be [`Pod`] anyway. [`PathHeader`] is the
/// typed view over this, and the two convert without touching an offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct PathWords {
    /// `"Q4EP"`, read as a little-endian word.
    pub magic: u32,
    /// The payload layout version.
    pub version: u32,
    /// Bytes per pool page.
    pub page_bytes: u32,
    /// Tokens the path covers.
    pub tokens: u32,
    /// Positions per page.
    pub page_tokens: u32,
    /// Logical pages.
    pub pages: u32,
    /// Bytes of one row across every key/value buffer.
    pub kv_width: u32,
    /// Layers the page section interleaves.
    pub n_layers: u32,
    /// Layers that contribute key/value rows.
    pub n_attn_layers: u32,
    /// Layers that contribute recurrent state.
    pub n_gdn_layers: u32,
    /// Non-zero when the page section carries indexer caches.
    pub indexer: u32,
    /// Non-zero when the page section carries a draft head.
    pub draft_head: u32,
    /// Vocabulary size.
    pub vocab: u32,
    /// Elements of one recurrent layer's conv window.
    pub gdn_conv_elements: u32,
    /// Elements of one recurrent layer's state matrix.
    pub gdn_state_elements: u32,
    /// Elements of the dilated side-window block.
    pub ple_window_elements: u32,
    /// Width of the hyper-connection plane.
    pub hc_width: u32,
    /// Non-zero when a pending draft row follows the state section.
    pub pending_draft_row: u32,
    /// The first position that pending row describes.
    pub pending_pos0: u32,
}

/// Bytes of the payload header, which is also the offset of the token section.
pub(crate) const PATH_HEADER_BYTES: usize = size_of::<PathWords>();

// Nineteen four-byte words and no padding: a `#[repr(C)]` record of `u32`s cannot
// grow one, and the assert is what turns a field added mid-struct into a build
// failure rather than a directory of files the next build silently mis-parses.
const _: () = assert!(PATH_HEADER_BYTES == 19 * 4);

// The record is moved to and from the file as raw bytes, which is the format's
// little-endian order only on a little-endian host. Both targets this ships on are
// one; a big-endian port has to byte-swap here rather than discover a directory of
// headers it wrote backwards.
const _: () = assert!(cfg!(target_endian = "little"));

impl PathHeader {
    /// The on-disk record this header writes.
    #[must_use]
    pub fn to_words(self) -> PathWords {
        PathWords {
            magic: PATH_MAGIC,
            version: PATH_VERSION,
            page_bytes: self.page_bytes,
            tokens: self.tokens,
            page_tokens: self.page_tokens,
            pages: self.pages,
            kv_width: self.kv_width,
            n_layers: self.n_layers,
            n_attn_layers: self.n_attn_layers,
            n_gdn_layers: self.n_gdn_layers,
            indexer: u32::from(self.indexer),
            draft_head: u32::from(self.draft_head),
            vocab: self.vocab,
            gdn_conv_elements: self.gdn_conv_elements,
            gdn_state_elements: self.gdn_state_elements,
            ple_window_elements: self.ple_window_elements,
            hc_width: self.hc_width,
            pending_draft_row: u32::from(self.pending_draft_row),
            pending_pos0: self.pending_pos0,
        }
    }

    /// Reads a record back without judging it, so the caller can name the field that
    /// disagrees rather than only that something does.
    #[must_use]
    pub const fn from_words(w: &PathWords) -> Self {
        Self {
            page_bytes: w.page_bytes,
            tokens: w.tokens,
            page_tokens: w.page_tokens,
            pages: w.pages,
            kv_width: w.kv_width,
            n_layers: w.n_layers,
            n_attn_layers: w.n_attn_layers,
            n_gdn_layers: w.n_gdn_layers,
            indexer: w.indexer != 0,
            draft_head: w.draft_head != 0,
            vocab: w.vocab,
            gdn_conv_elements: w.gdn_conv_elements,
            gdn_state_elements: w.gdn_state_elements,
            ple_window_elements: w.ple_window_elements,
            hc_width: w.hc_width,
            pending_draft_row: w.pending_draft_row != 0,
            pending_pos0: w.pending_pos0,
        }
    }

    /// Whether a record starts with this build's magic and version.
    #[must_use]
    pub const fn magic_matches(w: &PathWords) -> bool {
        w.magic == PATH_MAGIC && w.version == PATH_VERSION
    }

    /// The header for a path of `tokens` positions held in `pages` pool pages.
    ///
    /// The layout supplies everything the geometry does not: the layer counts, the
    /// element counts the state section is sized from, and the two operator toggles
    /// that decide whether a reader can use the file at all. `page_bytes` and
    /// `page_tokens` come from the geometry, so a payload can never describe a page
    /// size the cache was not built with.
    #[must_use]
    pub fn from_layout(
        layout: &crate::geom::PathLayout,
        geometry: &crate::geom::CacheGeometry,
        tokens: u32,
        pages: u32,
    ) -> Self {
        Self {
            page_bytes: to_u32_wide(geometry.page_bytes),
            tokens,
            page_tokens: geometry.page_tokens,
            pages,
            kv_width: layout.kv_width,
            n_layers: layout.n_layers,
            n_attn_layers: layout.n_attn_layers,
            n_gdn_layers: layout.n_gdn_layers,
            indexer: layout.indexer,
            draft_head: layout.draft_head,
            vocab: layout.vocab,
            gdn_conv_elements: layout.gdn_conv_elements,
            gdn_state_elements: layout.gdn_state_elements,
            ple_window_elements: layout.ple_window_elements,
            hc_width: layout.hc_width,
            pending_draft_row: false,
            pending_pos0: 0,
        }
    }

    /// Bytes of the state section: per recurrent layer a conv window and a state
    /// matrix, then the PLE window, then the draft head's pending row if it has one.
    #[must_use]
    pub fn state_bytes(&self) -> u64 {
        let per_layer = u64::from(self.gdn_conv_elements) + u64::from(self.gdn_state_elements);
        let mut bytes = u64::from(self.n_gdn_layers) * per_layer * F32_BYTES;
        bytes += u64::from(self.ple_window_elements) * F32_BYTES;
        if self.draft_head {
            bytes += u64::from(self.hc_width) * F32_BYTES;
        }
        bytes
    }

    /// Bytes of the whole payload, header included, or `None` if the header's own
    /// numbers overflow.
    #[must_use]
    pub fn total_bytes(&self) -> Option<u64> {
        let header = u64::try_from(PATH_HEADER_BYTES).ok()?;
        let tokens = u64::from(self.tokens).checked_mul(F32_BYTES)?;
        let logits = u64::from(self.vocab).checked_mul(F32_BYTES)?;
        let pages = u64::from(self.pages).checked_mul(u64::from(self.page_bytes))?;
        header
            .checked_add(tokens)?
            .checked_add(logits)?
            .checked_add(self.state_bytes())?
            .checked_add(pages)
    }
}

/// The four sections a path is made of, as the engine produced them.
#[derive(Debug, Clone, Copy)]
pub struct PathSections<'a> {
    /// The path's token ids, root to frontier.
    pub tokens: &'a [shiso_core::TokenId],
    /// The path-final logits row, `vocab` little-endian `f32`s.
    pub logits: &'a [u8],
    /// The recurrent state section, in checkpoint-slot order.
    pub state: &'a [u8],
    /// The page section: `pages` runs of `page_bytes`, concatenated.
    pub pages: &'a [u8],
}

/// A path read back, split into the sections the engine consumes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedPath {
    /// The header the sections were sized from.
    pub header: PathHeader,
    /// The path's token ids.
    pub tokens: Vec<shiso_core::TokenId>,
    /// The logits row, so a re-send of the same prompt can answer without a row.
    pub logits: Vec<u8>,
    /// The recurrent state, ready for a checkpoint slot.
    pub state: Vec<u8>,
    /// The pages, one buffer each, ready to be copied into fresh pool pages.
    pub pages: Vec<Vec<u8>>,
}

/// Why a payload was refused, and what should happen to the file.
///
/// The two refusals are deliberately distinct because acting on the wrong one is
/// destructive in different directions. A file that *this build* can never read is
/// worthless wherever it sits and is deleted; a file that this build cannot read
/// **as configured** is readable by the same build started with the other toggle
/// value, and the key is the rendered text rather than the toggles, so deleting it
/// would wipe a shared directory every time an operator flips a flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathMismatch {
    /// This build can never restore the file, so the caller may discard it.
    Foreign {
        /// The field that disagrees.
        field: &'static str,
        /// What the file says.
        found: u64,
        /// What this build is.
        want: u64,
    },
    /// This build cannot restore it as configured; another configuration can, so the
    /// file is kept.
    Incompatible {
        /// The field that disagrees.
        field: &'static str,
        /// What the file says.
        found: u64,
        /// What this build is configured for.
        want: u64,
    },
    /// The bytes are shorter than the header says they are.
    Truncated {
        /// Bytes the header asks for.
        needed: u64,
        /// Bytes present.
        have: u64,
    },
}

/// Serializes one path.
#[must_use]
pub fn encode_path(header: &PathHeader, sections: &PathSections<'_>) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(usize::try_from(header.total_bytes().unwrap_or(0)).unwrap_or(0));
    out.extend_from_slice(bytemuck::bytes_of(&header.to_words()));
    for token in sections.tokens {
        out.extend_from_slice(&token.get().to_le_bytes());
    }
    out.extend_from_slice(sections.logits);
    out.extend_from_slice(sections.state);
    out.extend_from_slice(sections.pages);
    out
}

/// Validates a payload against this build and splits it into its sections.
///
/// Every geometry number the reader is about to trust is compared against the
/// expectation before a single byte is copied, and the two failure modes are kept
/// apart: see [`PathMismatch`]. The length check is last rather than first, so a file
/// from another model is reported as that rather than as a short read.
///
/// # Errors
///
/// [`PathMismatch::Foreign`] if the payload describes a build this one can never
/// become, [`PathMismatch::Incompatible`] if it only disagrees on an operator
/// toggle, and [`PathMismatch::Truncated`] if it is shorter than its own header.
/// The first two differ in what the caller should do with the file, which is the
/// entire reason they are not one variant.
pub fn decode_path(bytes: &[u8], expect: &PathHeader) -> Result<DecodedPath, PathMismatch> {
    let header_len = PATH_HEADER_BYTES;
    let Some(record) = bytes.get(..header_len) else {
        return Err(PathMismatch::Truncated {
            needed: u64::try_from(header_len).unwrap_or(0),
            have: u64::try_from(bytes.len()).unwrap_or(0),
        });
    };
    // `pod_read_unaligned` rather than a cast: the payload arrives as a `Vec<u8>` read
    // from a file, which carries no alignment promise at all.
    let words: PathWords = bytemuck::pod_read_unaligned(record);
    if !PathHeader::magic_matches(&words) {
        return Err(PathMismatch::Foreign {
            field: "payload magic or version",
            found: u64::from(words.version),
            want: u64::from(PATH_VERSION),
        });
    }
    let header = PathHeader::from_words(&words);
    check(header, expect)?;

    let state_bytes = usize::try_from(header.state_bytes()).unwrap_or(usize::MAX);
    let tokens_len = usize::try_from(u64::from(header.tokens) * F32_BYTES).unwrap_or(usize::MAX);
    let logits_len = usize::try_from(u64::from(header.vocab) * F32_BYTES).unwrap_or(usize::MAX);
    let one_page_len = usize::try_from(u64::from(header.page_bytes)).unwrap_or(usize::MAX);
    let pages_len = usize::try_from(u64::from(header.pages) * u64::from(header.page_bytes))
        .unwrap_or(usize::MAX);
    let needed = header_len
        .saturating_add(tokens_len)
        .saturating_add(logits_len)
        .saturating_add(state_bytes)
        .saturating_add(pages_len);
    if bytes.len() < needed {
        return Err(PathMismatch::Truncated {
            needed: u64::try_from(needed).unwrap_or(u64::MAX),
            have: u64::try_from(bytes.len()).unwrap_or(0),
        });
    }

    let mut at = header_len;
    let token_bytes = &bytes[at..at + tokens_len];
    at += tokens_len;
    let mut tokens = Vec::with_capacity(usize::try_from(header.tokens).unwrap_or(0));
    for chunk in token_bytes.as_chunks::<4>().0 {
        tokens.push(shiso_core::TokenId::new(u32::from_le_bytes(*chunk)));
    }
    let logits = bytes[at..at + logits_len].to_vec();
    at += logits_len;
    let state = bytes[at..at + state_bytes].to_vec();
    at += state_bytes;
    let page_section = &bytes[at..at + pages_len];
    let pages = if one_page_len == 0 {
        Vec::new()
    } else {
        page_section
            .chunks_exact(one_page_len)
            .map(<[u8]>::to_vec)
            .collect()
    };
    Ok(DecodedPath {
        header,
        tokens,
        logits,
        state,
        pages,
    })
}

/// Compares a file's header against this build, field by field, in the C reader's
/// order, and reports the first disagreement with its failure mode attached.
fn check(found: PathHeader, want: &PathHeader) -> Result<(), PathMismatch> {
    if found.tokens == 0 {
        return Err(PathMismatch::Foreign {
            field: "tokens",
            found: 0,
            want: 1,
        });
    }
    if found.pages != pages_for(found.tokens, found.page_tokens) {
        return Err(PathMismatch::Foreign {
            field: "page count",
            found: u64::from(found.pages),
            want: u64::from(pages_for(found.tokens, found.page_tokens)),
        });
    }
    if found.pending_draft_row && found.pending_pos0 >= found.tokens {
        return Err(PathMismatch::Foreign {
            field: "pending draft row position",
            found: u64::from(found.pending_pos0),
            want: u64::from(found.tokens),
        });
    }
    if found.page_bytes != want.page_bytes {
        return Err(PathMismatch::Incompatible {
            field: "page bytes",
            found: u64::from(found.page_bytes),
            want: u64::from(want.page_bytes),
        });
    }
    if found.indexer != want.indexer {
        return Err(PathMismatch::Incompatible {
            field: "indexer caches",
            found: u64::from(u8::from(found.indexer)),
            want: u64::from(u8::from(want.indexer)),
        });
    }
    if found.draft_head != want.draft_head {
        return Err(PathMismatch::Incompatible {
            field: "draft head",
            found: u64::from(u8::from(found.draft_head)),
            want: u64::from(u8::from(want.draft_head)),
        });
    }
    if found.page_tokens != want.page_tokens {
        return Err(PathMismatch::Foreign {
            field: "page size",
            found: u64::from(found.page_tokens),
            want: u64::from(want.page_tokens),
        });
    }
    for (field, got, expected) in [
        ("KV width", found.kv_width, want.kv_width),
        ("layer count", found.n_layers, want.n_layers),
        ("attention layers", found.n_attn_layers, want.n_attn_layers),
        ("recurrent layers", found.n_gdn_layers, want.n_gdn_layers),
        ("vocabulary", found.vocab, want.vocab),
        (
            "conv window",
            found.gdn_conv_elements,
            want.gdn_conv_elements,
        ),
        (
            "state matrix",
            found.gdn_state_elements,
            want.gdn_state_elements,
        ),
        (
            "PLE window",
            found.ple_window_elements,
            want.ple_window_elements,
        ),
        ("hyper-connection width", found.hc_width, want.hc_width),
    ] {
        if got != expected {
            return Err(PathMismatch::Foreign {
                field,
                found: u64::from(got),
                want: u64::from(expected),
            });
        }
    }
    Ok(())
}

/// Pages needed for `tokens` positions, rounding up. A zero page size yields zero so
/// a malformed header is reported by the field comparison rather than by a divide.
fn pages_for(tokens: u32, page_tokens: u32) -> u32 {
    if page_tokens == 0 {
        return 0;
    }
    tokens.div_ceil(page_tokens)
}
