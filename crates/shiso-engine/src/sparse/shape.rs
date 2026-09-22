//! The geometry of a sparse mixture-of-experts hybrid stack.
//!
//! The dense hybrid in [`crate::shape`] is one residual stream, one
//! feed-forward per slot and a rotary attention block every few slots. This is
//! the same skeleton with three additions, each of which is an architecture
//! fact rather than a product fact:
//!
//! - the residual is not one stream but [`SparseShape::streams`] of them, mixed
//!   into and out of every block by a low-rank pair (the hyper-connections);
//! - the feed-forward is a router over hundreds of experts plus a shared expert
//!   every token pays for;
//! - the attention block carries a second, much narrower tower - the indexer -
//!   which scores mean-pooled key blocks so the attention itself can read a
//!   selected subset of the context instead of all of it.
//!
//! A fourth is optional: a per-layer n-gram embedding block folded into exactly
//! one slot ([`SparseShape::ple`]).
//!
//! # Why this is not [`crate::shape::HybridShape`] with more fields
//!
//! Every field of [`crate::shape::HybridShape`] is read by the dense forward
//! pass, and a dense checkpoint supplies all of them. Widening it with a router
//! width and an indexer head count would make every dense model carry numbers
//! its pass never reads and its loader has to invent. The two shapes share the
//! vocabulary that is genuinely common - [`LayerKind`], [`GdnShape`] - and
//! nothing else. [`GqaShape`] is this stack's own: the dense shape states its
//! attention widths per slot, because a dense family exists whose slots differ,
//! and this one's do not.
//!
//! # Nothing here names a checkpoint family
//!
//! There is no table of constants below, and no `Default`. Which numbers a
//! particular checkpoint fills these in with is the business of the crate that
//! can read that checkpoint's config; a second family with the same blocks and
//! different widths is a second crate up there and no edit here.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;

use crate::shape::{GdnShape, LayerKind};

/// Attention geometry: one set of widths for every attention slot of this stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GqaShape {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads. `q_heads / kv_heads` queries share each one.
    pub kv_heads: u32,
    /// Width of one head, which in these models is not `hidden / q_heads`.
    pub head_dim: u32,
    /// Leading dimensions of a head the rotary embedding touches. The rest of
    /// the head passes through untouched.
    pub n_rot: u32,
}

impl GqaShape {
    /// The four widths of an attention block.
    #[must_use]
    pub const fn new(q_heads: u32, kv_heads: u32, head_dim: u32, n_rot: u32) -> Self {
        Self {
            q_heads,
            kv_heads,
            head_dim,
            n_rot,
        }
    }
}

/// The router geometry this stack shares with every other routed family.
///
/// It lives in [`crate::shape`] rather than here because a routed feed-forward
/// is not a fact about *this* stack: the dense hybrid carries one too, as
/// [`crate::shape::HybridShape::moe`]. Re-exported rather than moved out of
/// sight, so a reader of this file still meets the type that
/// [`SparseShape::moe`] holds.
pub use crate::shape::MoeShape;

/// Indexer geometry: the narrow tower that decides which of the context an
/// attention block actually reads.
///
/// Deliberately NOT `#[non_exhaustive]`, for the reason given on
/// [`MoeShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerShape {
    /// Query heads the scorer runs.
    pub heads: u32,
    /// Width of one indexer head, which is also the width of a stored key row.
    pub head_dim: u32,
    /// Token positions the selection is allowed to reach, as a count of
    /// positions rather than of blocks.
    pub top_k: u32,
    /// Positions one pooled key block averages over.
    ///
    /// The page size is a multiple of this, so a pooled block never straddles
    /// a page and a page's pooled rows are exactly its own share of the pooled
    /// table.
    pub compress: u32,
}

impl IndexerShape {
    /// The four widths of one indexer.
    #[must_use]
    pub const fn new(heads: u32, head_dim: u32, top_k: u32, compress: u32) -> Self {
        Self {
            heads,
            head_dim,
            top_k,
            compress,
        }
    }

    /// Key blocks the selection budget covers.
    ///
    /// The budget is stated in positions because that is what a context length
    /// is comparable against; the scorer and the top-k stage both work in
    /// blocks, so this is the conversion and it lives in one place.
    #[must_use]
    pub const fn budget_blocks(&self) -> u32 {
        if self.compress == 0 {
            return 0;
        }
        self.top_k / self.compress
    }

    /// Blocks one row of the score buffer spans, for a context of `ctx`
    /// positions.
    ///
    /// **Not** [`IndexerShape::budget_blocks`], and the difference is the whole
    /// point of having two: the budget is how many blocks the selection *keeps*,
    /// and this is how many it *ranks*. Every block the context can hold has to
    /// be scored, or the ones past the budget are never candidates - which is
    /// invisible below the budget, where every visible block fits, and past it
    /// turns the selection into "the oldest `top_k` positions" with no error
    /// anywhere and a fluent answer at the end.
    ///
    /// The `+ 1` is the partial trailing block: the scorer marks it invisible
    /// rather than reading past the frontier, but the row still has to have a
    /// slot for it.
    #[must_use]
    pub const fn score_blocks(&self, ctx: u32) -> u32 {
        if self.compress == 0 {
            return 0;
        }
        ctx / self.compress + 1
    }

    /// Whether a context holding `positions` tokens has outgrown the budget.
    ///
    /// Below this point the selection would be every visible token, so the
    /// dense attention kernel is both exact and faster and the sparse path is
    /// not taken at all. The comparison is between *block* counts, and both
    /// sides truncate: a partial trailing block is not a block the scorer can
    /// rank.
    #[must_use]
    pub const fn sparse_at(&self, positions: u32) -> bool {
        if self.compress == 0 {
            return false;
        }
        positions / self.compress > self.budget_blocks()
    }
}

/// The n-gram embedding block, and which slot folds it in.
///
/// Deliberately NOT `#[non_exhaustive]`, for the reason given on
/// [`MoeShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleShape {
    /// The one slot this block runs inside, before that slot's first mix.
    pub layer: LayerIdx,
    /// Gram widths the hash table is partitioned into.
    pub ngram: u32,
    /// Heads each gram width contributes.
    pub heads_per_ngram: u32,
    /// Width of one head's embedding row.
    pub head_dim: u32,
    /// Width of the dilated causal convolution over the gated value.
    pub conv: u32,
}

impl PleShape {
    /// The five facts about one n-gram block.
    #[must_use]
    pub const fn new(
        layer: LayerIdx,
        ngram: u32,
        heads_per_ngram: u32,
        head_dim: u32,
        conv: u32,
    ) -> Self {
        Self {
            layer,
            ngram,
            heads_per_ngram,
            head_dim,
            conv,
        }
    }

    /// Heads the gathered rows decode into.
    ///
    /// The shortest gram width is the token itself, which the ordinary
    /// embedding table already carries, so it contributes no head here - hence
    /// the `- 1`.
    #[must_use]
    pub const fn heads(&self) -> u32 {
        self.ngram
            .saturating_sub(1)
            .saturating_mul(self.heads_per_ngram)
    }

    /// Rows of the convolution window one session carries.
    ///
    /// The convolution is dilated by the gram count, so its window reaches
    /// `(conv - 1) * ngram` positions back rather than `conv - 1`.
    #[must_use]
    pub const fn history(&self) -> u32 {
        self.conv.saturating_sub(1).saturating_mul(self.ngram)
    }

    /// Tokens the host has to read behind a position to hash its grams.
    #[must_use]
    pub const fn lookback(&self) -> u32 {
        self.ngram.saturating_sub(1)
    }
}

/// Interleaved rotary section widths over the frequency pairs, in axis order.
///
/// Three sections rather than one angle because a position in these models is
/// three-dimensional for image tokens: a pair belongs to the time axis, the
/// height axis or the width axis, and the three counts say how the pairs are
/// dealt out. An all-zero triple means the model has no multi-axis rope and
/// every pair falls back to the time axis, which is plain scalar rope.
///
/// Deliberately NOT `#[non_exhaustive]`, for the reason given on
/// [`MoeShape`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MropeSections {
    /// Frequency pairs driven by the time axis.
    pub time: u32,
    /// Frequency pairs driven by the height axis.
    pub height: u32,
    /// Frequency pairs driven by the width axis.
    pub width: u32,
}

impl MropeSections {
    /// The three section widths.
    #[must_use]
    pub const fn new(time: u32, height: u32, width: u32) -> Self {
        Self {
            time,
            height,
            width,
        }
    }

    /// Whether the three sections tile `n_rot / 2` frequency pairs exactly.
    ///
    /// A triple that does not is not a rounding problem: the rope kernels deal
    /// pairs out by walking the sections in order, so a short triple leaves the
    /// tail pairs unrotated and a long one runs off the end of the table.
    #[must_use]
    pub const fn covers(&self, n_rot: u32) -> bool {
        let pairs = n_rot / 2;
        self.time + self.height + self.width == pairs
    }

    /// Whether this is the degenerate triple that means "no multi-axis rope".
    #[must_use]
    pub const fn is_scalar(&self) -> bool {
        self.time == 0 && self.height == 0 && self.width == 0
    }
}

/// The geometry of one sparse mixture-of-experts hybrid stack.
///
/// The layer list is the only place the slot count lives, exactly as in
/// [`crate::shape::HybridShape`]: a separate `n_layers` beside it would be a
/// second copy of one fact and the two could disagree.
///
/// Deliberately NOT `#[non_exhaustive]`. Every field is a geometry fact the
/// forward pass reads, so a new one means a number no model crate is supplying
/// and a compile error at each of them is the correct outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseShape {
    /// Block kind of every slot, index zero first.
    pub layers: Vec<LayerKind>,
    /// Width of one residual stream.
    pub hidden: u32,
    /// Vocabulary size, which is also the logit row width.
    pub vocab: u32,
    /// Residual streams the hyper-connections carry.
    pub streams: u32,
    /// Inner width of the low-rank pair that mixes the streams.
    pub stream_rank: u32,
    /// Attention block geometry.
    pub gqa: GqaShape,
    /// Recurrent block geometry.
    pub gdn: GdnShape,
    /// Router geometry.
    pub moe: MoeShape,
    /// Indexer geometry.
    pub indexer: IndexerShape,
    /// The n-gram block, on the one slot that has it.
    pub ple: Option<PleShape>,
    /// Rotary section widths.
    pub mrope: MropeSections,
    /// Epsilon every normalization in the stack adds.
    pub rms_eps: f32,
    /// Rotary frequency base.
    pub rope_theta: f32,
}

impl SparseShape {
    /// Total layer slots.
    #[must_use]
    pub fn n_layers(&self) -> u32 {
        u32::try_from(self.layers.len()).unwrap_or(u32::MAX)
    }

    /// Block kind of one slot, or `None` past the last one.
    #[must_use]
    pub fn layer_kind(&self, il: LayerIdx) -> Option<LayerKind> {
        self.layers.get(il.get() as usize).copied()
    }

    /// How many slots hold `kind`.
    #[must_use]
    pub fn count(&self, kind: LayerKind) -> u32 {
        let n = self.layers.iter().filter(|slot| **slot == kind).count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    /// Whether the n-gram block runs inside this slot.
    #[must_use]
    pub fn has_ple(&self, il: LayerIdx) -> bool {
        self.ple.is_some_and(|ple| ple.layer == il)
    }

    /// Width of one token's whole residual: every stream, laid out stream-outer.
    #[must_use]
    pub const fn stream_dim(&self) -> u32 {
        self.hidden.saturating_mul(self.streams)
    }

    /// Width of one token's query rows across every attention head.
    #[must_use]
    pub const fn q_dim(&self) -> u32 {
        self.gqa.q_heads.saturating_mul(self.gqa.head_dim)
    }

    /// Width of one token's key rows, which is also its value rows.
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.gqa.kv_heads.saturating_mul(self.gqa.head_dim)
    }

    /// Width of the recurrent block's query rows, which is also its key rows.
    #[must_use]
    pub const fn gdn_qk_dim(&self) -> u32 {
        self.gdn.qk_heads.saturating_mul(self.gdn.head_dim)
    }

    /// Width of the recurrent block's value rows.
    #[must_use]
    pub const fn gdn_v_dim(&self) -> u32 {
        self.gdn.v_heads.saturating_mul(self.gdn.head_dim)
    }

    /// Channels the recurrent projection carries: a query row, a key row and a
    /// value row, in that order.
    ///
    /// The order is what the convolution and the recurrence index by, so the
    /// two offsets a launch passes them ([`SparseShape::gdn_qk_dim`] and twice
    /// it) are derived from this and never restated.
    #[must_use]
    pub const fn gdn_in_dim(&self) -> u32 {
        self.gdn_qk_dim()
            .saturating_mul(2)
            .saturating_add(self.gdn_v_dim())
    }

    /// Width of one indexer query row across every indexer head.
    #[must_use]
    pub const fn indexer_q_dim(&self) -> u32 {
        self.indexer.heads.saturating_mul(self.indexer.head_dim)
    }

    /// Checks every width the block families were written against.
    ///
    /// The dimensions a checkpoint declares and the dimensions the kernels
    /// assume have to agree, and the failure when they do not is silent: rows
    /// read from the wrong offset still decode to finite floats. So the
    /// disagreements that are cheap to detect are detected here, once, by name.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] naming the first width that cannot work,
    /// and [`CoreError::InvalidValue`] when the rotary sections do not tile the
    /// frequency pairs.
    pub fn validate(&self) -> CoreResult<()> {
        if self.layers.is_empty() {
            return Err(CoreError::ShapeMismatch {
                field: "layer count",
                expected: 1,
                got: 0,
            });
        }
        if self.streams == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "residual streams",
                expected: 1,
                got: 0,
            });
        }
        if self.moe.used == 0 || self.moe.used > self.moe.experts {
            return Err(CoreError::ShapeMismatch {
                field: "experts per token",
                expected: u64::from(self.moe.experts),
                got: u64::from(self.moe.used),
            });
        }
        if self.gqa.q_heads == 0 || self.gqa.kv_heads == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "attention heads",
                expected: 1,
                got: u64::from(self.gqa.q_heads.min(self.gqa.kv_heads)),
            });
        }
        if !self.gqa.q_heads.is_multiple_of(self.gqa.kv_heads) {
            return Err(CoreError::ShapeMismatch {
                field: "query heads per key head",
                expected: u64::from(self.gqa.kv_heads),
                got: u64::from(self.gqa.q_heads),
            });
        }
        // The recurrent block reads the query/key head at its own index modulo
        // the query/key head count, so a value-head count that is not a whole
        // number of them would give some query heads more readers than others.
        if self.gdn.qk_heads == 0 || !self.gdn.v_heads.is_multiple_of(self.gdn.qk_heads) {
            return Err(CoreError::ShapeMismatch {
                field: "recurrent value heads per key head",
                expected: u64::from(self.gdn.qk_heads),
                got: u64::from(self.gdn.v_heads),
            });
        }
        if self.gdn.conv_kernel < 2 {
            return Err(CoreError::ShapeMismatch {
                field: "recurrent convolution width",
                expected: 2,
                got: u64::from(self.gdn.conv_kernel),
            });
        }
        if let Some(ple) = self.ple {
            let rows = ple.heads().saturating_mul(ple.head_dim);
            if rows != self.hidden {
                return Err(CoreError::ShapeMismatch {
                    field: "n-gram embedding width",
                    expected: u64::from(self.hidden),
                    got: u64::from(rows),
                });
            }
            if ple.layer.get() >= self.n_layers() {
                return Err(CoreError::ShapeMismatch {
                    field: "n-gram block slot",
                    expected: u64::from(self.n_layers()),
                    got: u64::from(ple.layer.get()),
                });
            }
        }
        if !self.mrope.is_scalar() && !self.mrope.covers(self.gqa.n_rot) {
            return Err(CoreError::InvalidValue {
                key: "rope sections",
                value: format!(
                    "{}+{}+{} pairs do not tile the {} rotated dimensions",
                    self.mrope.time, self.mrope.height, self.mrope.width, self.gqa.n_rot
                ),
            });
        }
        Ok(())
    }
}
