//! What an attention block is made of: its widths, its weights, the positions it
//! is handed and the scratch rows one chunk needs.
//!
//! Not the launches. Which kernels a block dispatches over all of this is a fact
//! about the checkpoint family, so they live behind
//! [`crate::blocks::BlockKernels`] and qwen35's are in `shiso-models`. What is
//! left here is what every family of this block structure shares, and it is the
//! whole of what the stack allocates and hands over.
//!
//! # Why the query projection is twice as wide
//!
//! [`Attn::q`] projects `2 * n_head * head_dim`, because the second half of every
//! head's row is the output gate. That is why [`AttnScratch`] carries a `q` row
//! and a `gate` row rather than one wider row: a family's normalization kernel
//! splits the projection into the two, and both are then indexed flat. Carrying
//! the gate inside the query row would have made every later launch stride by
//! `2 * head_dim` for no gain. A checkpoint whose projection were half this wide
//! would be one without an output gate, and the loader is what notices.
//!
//! # Nothing here computes a cache row
//!
//! [`AttnCache`] is a layer's two runs plus the page table each row of the pass
//! reads, and the positions beside it are absolute. Translating a position into
//! a row is the kernel's, so the engine never computes a cache offset and
//! therefore cannot get one wrong. What this module is responsible for is
//! narrower and still worth stating: the table handed over is the one that
//! session uploaded, and the rows a launch may write are rows that session had
//! claimed. [`Pages`] is what makes that statement hold for a pass carrying
//! several sessions' rows at once.

use std::sync::Arc;

use crate::shape::AttnLayer;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};

use crate::launch::F32_BYTES;
use crate::projection::Weight;

/// One attention block's widths, with the rotary and cache parameters beside them.
///
/// The rotary base and the normalization epsilon are per-checkpoint rather than
/// per-kernel, and the page width is a cache constant, so all three travel with the
/// geometry instead of being threaded through every launch.
///
/// One of these per attention SLOT, because a family exists whose slots differ:
/// [`crate::layers::Geometry`] holds the list and a launch is handed the slot's
/// own. What the shared scratch is sized by is a separate statement,
/// [`AttnScratchSizes`], because no single slot bounds every row.
#[derive(Debug, Clone, Copy)]
pub struct AttnGeometry {
    /// Query heads.
    pub n_head: u32,
    /// Key/value heads.
    pub n_head_kv: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Leading dimensions rotary rotates.
    pub n_rot: u32,
    /// Width the frequency ladder is computed over, which is `n_rot` for a
    /// family that rotates the whole ladder it derived.
    pub freq_dims: u32,
    /// Rotary frequency base.
    pub rope_base: f32,
    /// Epsilon inside the per-head mean square.
    pub eps: f32,
    /// Positions one cache page covers.
    pub page_tokens: u32,
    /// Positions a query may look back over; `0` is full causal attention.
    pub window: u32,
    /// The number the scores are multiplied by.
    pub scale: f32,
    /// Whether the value projection is normalized before it is cached.
    pub v_norm: bool,
    /// Whether the block's output is gated by the second half of the query
    /// projection's rows.
    pub output_gate: bool,
    /// Whether this slot writes the cache runs it reads. `false` for a slot that
    /// reads an earlier slot's.
    pub owns_kv: bool,
}

impl AttnGeometry {
    /// Derives one slot's widths from its own descriptor.
    ///
    /// [`AttnGeometry::v_norm`] and [`AttnGeometry::output_gate`] are the
    /// stack's facts rather than the slot's, so they are not arguments here and
    /// the caller that holds the shape fills them in; this leaves both `false`.
    #[must_use]
    pub fn new(layer: &AttnLayer, eps: f32, page_tokens: u32) -> Self {
        Self {
            n_head: layer.q_heads,
            n_head_kv: layer.kv_heads,
            head_dim: layer.head_dim,
            n_rot: layer.n_rot,
            freq_dims: layer.freq_dims,
            rope_base: layer.rope_theta,
            eps,
            page_tokens,
            window: layer.window.map_or(0, std::num::NonZeroU32::get),
            scale: layer.scale.value(layer.head_dim),
            v_norm: false,
            output_gate: false,
            owns_kv: layer.owns_kv(),
        }
    }

    /// Bytes of one token's fused query row, gate half included.
    #[must_use]
    pub(crate) fn q_row_bytes(&self) -> u64 {
        u64::from(self.n_head)
            .saturating_mul(u64::from(self.head_dim))
            .saturating_mul(2)
            .saturating_mul(F32_BYTES)
    }

    /// Bytes of one token's attention output, before the output projection.
    #[must_use]
    pub(crate) fn out_row_bytes(&self) -> u64 {
        u64::from(self.n_head)
            .saturating_mul(u64::from(self.head_dim))
            .saturating_mul(F32_BYTES)
    }

    /// Bytes of one token's key row, which is also its value row.
    #[must_use]
    pub(crate) fn kv_row_bytes(&self) -> u64 {
        u64::from(self.n_head_kv)
            .saturating_mul(u64::from(self.head_dim))
            .saturating_mul(F32_BYTES)
    }
}

/// One attention block's weights, resident on the device.
///
/// The query projection is twice as wide as the query: the second half of every
/// head's row is the output gate, and the family's normalization launch is what
/// splits the row into a normalized, rotated query and an untouched copy of the
/// gate. A checkpoint whose projection was half this wide would be a checkpoint
/// without an output gate, and the loader is what notices the difference.
///
/// The key half is three `Option`s that are present or absent together: a slot
/// that reads an earlier slot's cache projects no keys and no values and carries
/// no key norm, because the rows it would have produced are rows another slot
/// already wrote. [`Attn::kv`] is how a launcher asks for all three at once.
#[derive(Debug)]
pub struct Attn {
    /// Fused query and output-gate projection.
    pub q: Weight,
    /// Key projection, absent on a slot that reads another's cache.
    pub k: Option<Weight>,
    /// Value projection, absent on a slot that reads another's cache.
    pub v: Option<Weight>,
    /// Projection back to the residual width.
    pub output: Weight,
    /// Per-head query normalization weight.
    pub q_norm: DeviceBuffer,
    /// Per-head key normalization weight, absent on a slot that reads another's
    /// cache.
    pub k_norm: Option<DeviceBuffer>,
}

impl Attn {
    /// The key projection, the value projection and the key norm together.
    ///
    /// One accessor for the three, because they are one decision: a slot either
    /// fills the cache or reads one another slot filled, and a launcher that
    /// took them one at a time could half-launch a block that has none of them.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when any of the three is absent, which for a
    /// slot that owns its cache means the loader bound it as if it did not.
    pub fn kv(&self) -> CoreResult<(&Weight, &Weight, &DeviceBuffer)> {
        match (self.k.as_ref(), self.v.as_ref(), self.k_norm.as_ref()) {
            (Some(k), Some(v), Some(k_norm)) => Ok((k, v, k_norm)),
            _ => Err(CoreError::InvalidValue {
                key: "attention weights",
                value: "a slot that owns its cache was bound without key/value projections"
                    .to_owned(),
            }),
        }
    }
}

/// The positions one chunk's rotation reads.
///
/// Two shapes, because there are two questions. The cache always addresses rows
/// by the LINEAR position — where the token sits in the sequence — and that is
/// the only thing [`Positions::linear`] is for. The rotary angle asks a
/// different question, and for a prompt carrying media the answer is three grid
/// coordinates rather than one count.
///
/// An enum rather than three optional views beside the linear one: a chunk is
/// either multimodal or it is not, and a shape that allowed two of the three
/// streams would be a shape whose meaning depends on which of them happened to
/// be filled.
#[derive(Debug, Clone, Copy)]
pub enum Positions<'a> {
    /// One position per token. This is the whole of a text-only request, and it
    /// dispatches exactly the kernels it dispatched before the multimodal path
    /// existed.
    Linear(DeviceView<'a>),
    /// A linear position per token for the cache, and three streams for the
    /// rotation.
    Mrope {
        /// Sequence position, which is what the page table translates.
        linear: DeviceView<'a>,
        /// Temporal stream.
        t: DeviceView<'a>,
        /// Row stream.
        h: DeviceView<'a>,
        /// Column stream.
        w: DeviceView<'a>,
    },
}

impl<'a> Positions<'a> {
    /// The sequence positions, which every cache-addressing kernel reads.
    #[must_use]
    pub const fn linear(&self) -> DeviceView<'a> {
        match *self {
            Self::Linear(view) | Self::Mrope { linear: view, .. } => view,
        }
    }
}

/// Where each row of a pass finds the page table it addresses the cache
/// through.
///
/// One pass may carry rows belonging to **different sessions** - that is what a
/// batched decode step is - and two sessions share no page. A launch handed one
/// table for the whole batch would answer every row past the first out of row
/// zero's conversation: fluent, at the right shape, through launches that all
/// succeed and a test suite that passes, because each row on its own is a
/// perfectly good answer to somebody's question.
///
/// So the table is a table *and a stride*, and the kernels select row `t`'s own
/// at `table + t * stride`. A stride of zero is the shared case and not a
/// sentinel: `table + t * 0` is `table` for every row, which is exactly what a
/// prefill chunk and a single-session step want.
///
/// It is a type rather than a second field on [`AttnCache`] so that a family's
/// launcher cannot bind the buffer and forget the stride - the buffer is only
/// reachable through [`Pages::table`], and the stride is the other half of the
/// same value.
#[derive(Debug, Clone, Copy)]
pub struct Pages<'a> {
    table: DeviceView<'a>,
    stride: u32,
}

impl<'a> Pages<'a> {
    /// One table every row of the pass reads: a prefill chunk, or a step of a
    /// single session.
    #[must_use]
    pub const fn shared(table: DeviceView<'a>) -> Self {
        Self { table, stride: 0 }
    }

    /// A packed block of `stride`-entry tables, row `t`'s at `t * stride`.
    ///
    /// The stride is in table entries, not bytes, because that is the unit the
    /// kernels index in.
    #[must_use]
    pub const fn per_row(table: DeviceView<'a>, stride: u32) -> Self {
        Self { table, stride }
    }

    /// The buffer a KV-touching launch binds.
    #[must_use]
    pub const fn table(&self) -> DeviceView<'a> {
        self.table
    }

    /// Entries between one row's table and the next; `0` when they share one.
    #[must_use]
    pub const fn stride(&self) -> u32 {
        self.stride
    }

    /// Whether every row of this pass reads the same table.
    ///
    /// Asked by the entry points that tile several rows into one threadgroup
    /// and therefore cannot select a table per row. They refuse rather than
    /// read row `t0`'s for the whole tile.
    #[must_use]
    pub const fn is_shared(&self) -> bool {
        self.stride == 0
    }
}

/// Where one block's cache lives for one launch.
///
/// The two runs are one layer's whole allocations and `pages` says which table
/// each row reads. A launch handed another session's table would read that
/// session's prefix, which is the one failure mode this type cannot prevent by
/// construction, so the caller is a session - or, for a batch, the packer that
/// wrote one row per member - and not a caller that picked two things up
/// separately.
#[derive(Debug, Clone, Copy)]
pub struct AttnCache<'a> {
    /// The layer's key cache.
    pub k: DeviceView<'a>,
    /// The layer's value cache.
    pub v: DeviceView<'a>,
    /// The page table each row of the pass reads.
    pub pages: Pages<'a>,
}

/// Bytes one token occupies in each of the attention scratch rows.
///
/// One set of rows serves every attention slot, so each row has to hold the
/// largest that row ever gets - and "largest" is a question per row, not per
/// slot. A slot with many narrow heads has the widest query row; a slot with few
/// wide key/value heads has the widest key row; neither is ordered by
/// `head_dim`, which bounds none of the three on its own. Sizing from one
/// "widest slot" would therefore under-allocate every row that slot does not
/// happen to win, and a launch would write past the end of a buffer that looked
/// correctly sized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttnScratchSizes {
    /// Bytes of one token's fused query row, gate half included.
    pub q_row_bytes: u64,
    /// Bytes of one token's key row, which is also its value row.
    pub kv_row_bytes: u64,
    /// Bytes of one token's attention output, before the output projection.
    pub out_row_bytes: u64,
}

impl AttnScratchSizes {
    /// Each row's maximum over `slots`, which is what a scratch they share needs.
    ///
    /// Zero for every row when `slots` is empty, which is a stack with no
    /// attention block and therefore no attention row to write.
    #[must_use]
    pub fn over(slots: &[AttnGeometry]) -> Self {
        let mut sizes = Self::default();
        for slot in slots {
            sizes.q_row_bytes = sizes.q_row_bytes.max(slot.q_row_bytes());
            sizes.kv_row_bytes = sizes.kv_row_bytes.max(slot.kv_row_bytes());
            sizes.out_row_bytes = sizes.out_row_bytes.max(slot.out_row_bytes());
        }
        sizes
    }
}

/// The scratch one attention block needs for a chunk of tokens.
#[derive(Debug)]
pub struct AttnScratch {
    /// The query projection: query and gate, interleaved per head.
    pub qkv: DeviceBuffer,
    /// The key projection's output.
    pub k: DeviceBuffer,
    /// The value projection's output.
    pub v: DeviceBuffer,
    /// The normalized, rotated query.
    pub q: DeviceBuffer,
    /// The raw output gate, untouched until the family's gate launch.
    pub gate: DeviceBuffer,
    /// The softmax's output, which the gate scales in place.
    pub attn: DeviceBuffer,
}

impl AttnScratch {
    /// Allocates every row a chunk of `n_tok` tokens needs.
    ///
    /// `sizes` is the per-row maximum over every attention slot rather than one
    /// slot's own widths, because these rows are reused by all of them.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when an allocation fails.
    pub(crate) fn new(
        backend: &Arc<dyn Backend>,
        sizes: &AttnScratchSizes,
        n_tok: u32,
    ) -> CoreResult<Self> {
        let tokens = u64::from(n_tok);
        let qkv = sizes.q_row_bytes.saturating_mul(tokens);
        let kv = sizes.kv_row_bytes.saturating_mul(tokens);
        let out = sizes.out_row_bytes.saturating_mul(tokens);
        Ok(Self {
            qkv: alloc(backend, qkv)?,
            k: alloc(backend, kv)?,
            v: alloc(backend, kv)?,
            q: alloc(backend, out)?,
            gate: alloc(backend, out)?,
            attn: alloc(backend, out)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{AttnGeometry, AttnScratchSizes};
    use crate::kv::PAGE_TOKENS;
    use crate::shape::AttnLayer;

    /// One slot's geometry from its widths, with the rest of the descriptor at
    /// whatever the constructor derives: only the widths reach a row size.
    fn slot(q_heads: u32, kv_heads: u32, head_dim: u32) -> AttnGeometry {
        AttnGeometry::new(
            &AttnLayer::new(q_heads, kv_heads, head_dim, head_dim, head_dim, 1e4),
            1e-6,
            PAGE_TOKENS,
        )
    }

    #[test]
    fn every_scratch_row_is_sized_by_the_slot_that_needs_it_most() {
        // Two slots that win different rows: the first has many narrow heads and
        // the widest query and output rows; the second has few wide heads and,
        // with four times the key/value heads, the widest key row. Neither is
        // "the widest slot", which is the whole point - `head_dim` alone orders
        // none of the three.
        let narrow = slot(32, 8, 256);
        let wide = slot(4, 16, 512);
        let sizes = AttnScratchSizes::over(&[narrow, wide]);

        // 32 heads x 256 channels x 2 (gate) x 4 bytes, against the other slot's
        // 4 x 512 x 2 x 4 = 16384.
        assert_eq!(sizes.q_row_bytes, 65_536);
        assert_eq!(narrow.q_row_bytes(), 65_536);
        assert_eq!(wide.q_row_bytes(), 16_384);
        // 16 key/value heads x 512 channels x 4 bytes, against 8 x 256 x 4 = 8192:
        // this row is the wide slot's, and the widest head is what would have
        // been picked either way - the next assertion is the one that separates
        // them.
        assert_eq!(sizes.kv_row_bytes, 32_768);
        // 32 x 256 x 4, against 4 x 512 x 4 = 8192. A scratch sized from the
        // slot with the widest head would have allocated 8192 here and the first
        // slot's softmax would have written four times its length.
        assert_eq!(sizes.out_row_bytes, 32_768);
        assert_eq!(wide.out_row_bytes(), 8_192);
    }

    #[test]
    fn a_stack_with_no_attention_slot_needs_no_attention_row() {
        assert_eq!(AttnScratchSizes::over(&[]), AttnScratchSizes::default());
    }
}
