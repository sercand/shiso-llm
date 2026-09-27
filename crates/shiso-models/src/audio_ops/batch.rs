//! A batch of independent causal-decoder sequences, each step one launch for
//! all of them: the segmented prefill (stacked prompts, each attending over
//! its own cache) and the batched decode step (one row per sequence, each at
//! its own state's position and length).
//!
//! Every batched launch runs the body of the single-sequence launch it
//! batches, over that sequence's rows, plane and state, so it forms every
//! value exactly as that launch does on the sequence alone: a decoder that
//! batches its sequences keeps their ids.
//!
//! The sequences' caches share one key and one value buffer per layer, and a
//! [`SeqTable`] on the device says where each sequence's plane starts and how
//! many rows it holds. The alternative - a buffer per sequence - would bind
//! two buffers a sequence, and Metal allows 31 bindings a kernel, so sixteen
//! sequences would not fit; a stride shared by every plane would make the
//! whole batch grow when one sequence does. With the table a launch binds
//! three buffers for the cache whatever the batch, and each plane keeps its
//! own capacity.

use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3, KernelArg};

use super::args::{
    BatchAttnArgs, CacheAttnArgs, DecodeStepBatchArgs, EmbedTokenArgs, QkRopeSeqArgs,
};
use super::{
    ATTN_QUERY_BLOCK, DECODE_GROUP_TILE, DECODE_MAX_SPLITS, DECODE_MERGE_THREADS,
    DECODE_SPLITS_PER_TG, DECODE_STATE_WORDS, EmbedTable, LINEAR_BLOCK, MAX_HEAD_DIM, Ops,
    aligned16, attention_smem, decode_attention_partials, dim, limit, need, need_bytes,
};

/// `u32` words of one sequence's entry in a [`SeqTable`]: the plane's first
/// row, its capacity, the first stacked row, the row count, the first row's
/// position, and padding.
pub const SEQ_WORDS: usize = 8;
/// Threads of one batched-step threadgroup, a thread per sequence.
const STEP_BLOCK: u32 = 32;

/// One sequence of a batch: where its cache plane sits in the shared
/// buffers, and for a segmented prefill which stacked rows are its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SeqSpan {
    /// The first row of the sequence's `[kv_heads][cap][head_dim]` plane, in
    /// rows of `head_dim` floats: row `r` of kv head `h` is at
    /// `(plane + h * cap + r) * head_dim`.
    pub plane: usize,
    /// Rows each kv head's plane holds.
    pub cap: usize,
    /// The sequence's first stacked row in a segmented prefill.
    pub row0: usize,
    /// The sequence's stacked rows in a segmented prefill; 0 in decode.
    pub rows: usize,
    /// The position of the first stacked row.
    pub pos0: usize,
}

impl SeqSpan {
    /// Rows of `head_dim` floats the plane covers, from row 0 of the
    /// buffers.
    fn plane_end(&self, kv_heads: usize) -> usize {
        self.plane + kv_heads * self.cap
    }
}

/// A batch's sequence table: the spans, and the same spans as the `u32`
/// words the batched kernels read, [`SEQ_WORDS`] a sequence. Build a new
/// one when a plane moves or grows; queued launches keep the words they were
/// given.
///
/// A table is sound by construction: no two planes share a cache row and no
/// two sequences share a stacked row, because the kernels write every row a
/// span names without asking whose else it is - an overlap would be one
/// sequence's append landing in another's history.
#[derive(Debug)]
pub struct SeqTable {
    spans: Vec<SeqSpan>,
    kv_heads: usize,
    words: DeviceBuffer,
}

/// Refuses spans whose planes of `kv_heads` heads overlap, or whose stacked
/// prefill rows do: [`CoreError::InvalidValue`] naming the two sequences.
fn disjoint(spans: &[SeqSpan], kv_heads: usize) -> CoreResult<()> {
    limit(kv_heads > 0, "seq_table.kv_heads", kv_heads)?;
    let overlap = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
    for (i, a) in spans.iter().enumerate() {
        for (j, b) in spans.iter().enumerate().skip(i + 1) {
            let planes = (
                (a.plane, a.plane_end(kv_heads)),
                (b.plane, b.plane_end(kv_heads)),
            );
            if overlap(planes.0, planes.1) {
                return Err(CoreError::InvalidValue {
                    key: "seq_table.planes_overlap",
                    value: format!(
                        "sequence {i} rows {:?} and sequence {j} rows {:?}",
                        planes.0, planes.1
                    ),
                });
            }
            let stacked = ((a.row0, a.row0 + a.rows), (b.row0, b.row0 + b.rows));
            if a.rows > 0 && b.rows > 0 && overlap(stacked.0, stacked.1) {
                return Err(CoreError::InvalidValue {
                    key: "seq_table.stacked_rows_overlap",
                    value: format!(
                        "sequence {i} rows {:?} and sequence {j} rows {:?}",
                        stacked.0, stacked.1
                    ),
                });
            }
        }
    }
    Ok(())
}

impl SeqTable {
    /// Uploads `spans`, each plane `kv_heads` heads of its capacity.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for no spans, no kv heads, a value past
    /// `u32`, two planes that share a cache row, or two sequences that share
    /// a stacked row; an allocation or copy failure.
    pub fn new(ops: &Ops, spans: &[SeqSpan], kv_heads: usize) -> CoreResult<Self> {
        limit(!spans.is_empty(), "seq_table.spans", 0)?;
        disjoint(spans, kv_heads)?;
        let mut raw = Vec::with_capacity(spans.len() * SEQ_WORDS * 4);
        for s in spans {
            let entry = [
                dim(s.plane, "seq_table.plane")?,
                dim(s.cap, "seq_table.cap")?,
                dim(s.row0, "seq_table.row0")?,
                dim(s.rows, "seq_table.rows")?,
                dim(s.pos0, "seq_table.pos0")?,
                0,
                0,
                0,
            ];
            raw.extend(entry.iter().flat_map(|w| w.to_le_bytes()));
        }
        let words = ops.scratch_uninit(spans.len() * SEQ_WORDS)?;
        ops.backend().copy_h2d(words.view(), &raw)?;
        Ok(Self {
            spans: spans.to_vec(),
            kv_heads,
            words,
        })
    }

    /// The spans, in sequence order.
    #[must_use]
    pub fn spans(&self) -> &[SeqSpan] {
        &self.spans
    }

    /// Sequences in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the batch has no sequence; a table never does.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

/// One layer's cache for a whole batch: the shared key and value buffers,
/// each sequence's plane where the table puts it.
#[derive(Debug, Clone, Copy)]
pub struct BatchPlanes<'a> {
    /// Keys, every sequence's plane.
    pub k: DeviceView<'a>,
    /// Values, laid out as the keys.
    pub v: DeviceView<'a>,
    /// Where each sequence's plane is.
    pub table: &'a SeqTable,
    /// Key and value heads.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
}

/// Causal grouped-query attention over a segmented prefill: sequence `s`'s
/// stacked query rows `[row0, row0 + rows)` sit at positions from its `pos0`
/// and see its own cache rows up to their own positions, as
/// [`super::CausalAttnDesc`] over that sequence alone.
#[derive(Debug, Clone, Copy)]
pub struct CausalAttnSegDesc<'a> {
    /// Destination, `[rows][heads * head_dim]`; a row no sequence owns is not
    /// written.
    pub out: DeviceView<'a>,
    /// The stacked queries, as the destination.
    pub q: DeviceView<'a>,
    /// The layer's cache, each plane already holding every row its queries
    /// see.
    pub cache: BatchPlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// Scale on the logit.
    pub scale: f32,
}

/// The query and key norms, their rotation and the append over a segmented
/// prefill's stacked rows: each sequence's rows at positions from its
/// `pos0`, into its own plane, as [`super::QkRopeAppendDesc`] over that
/// sequence alone.
#[derive(Debug, Clone, Copy)]
pub struct QkRopeAppendSegDesc<'a> {
    /// Stacked queries, `[rows][heads * head_dim]`, normalised and rotated in
    /// place; a row no sequence owns is left alone.
    pub q: DeviceView<'a>,
    /// Stacked keys, `[rows][kv_heads * head_dim]`; read only.
    pub k: DeviceView<'a>,
    /// Stacked values, as the keys.
    pub v: DeviceView<'a>,
    /// The query norm's weight, `[head_dim]`.
    pub q_norm: DeviceView<'a>,
    /// The key norm's weight, `[head_dim]`.
    pub k_norm: DeviceView<'a>,
    /// The layer's cache; a row at or past its plane's capacity is dropped.
    pub cache: BatchPlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// Stacked rows.
    pub rows: usize,
    /// Added to each head's mean square.
    pub eps: f32,
    /// Frequency base.
    pub theta: f32,
}

/// The query and key norms, their rotation and the append of one decode
/// step over a batch: row `s` is sequence `s`'s, at its state's position, as
/// [`super::QkRopeAppendDesc`] of one row over that sequence alone. A
/// finished sequence appends nothing.
#[derive(Debug, Clone, Copy)]
pub struct QkRopeAppendBatchDesc<'a> {
    /// Queries, `[seqs][heads * head_dim]`, normalised and rotated in place.
    pub q: DeviceView<'a>,
    /// Keys, `[seqs][kv_heads * head_dim]`; read only.
    pub k: DeviceView<'a>,
    /// Values, as the keys.
    pub v: DeviceView<'a>,
    /// The query norm's weight, `[head_dim]`.
    pub q_norm: DeviceView<'a>,
    /// The key norm's weight, `[head_dim]`.
    pub k_norm: DeviceView<'a>,
    /// The layer's cache.
    pub cache: BatchPlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// Added to each head's mean square.
    pub eps: f32,
    /// Frequency base.
    pub theta: f32,
    /// The decoder states, `[seqs][DECODE_STATE_WORDS]`.
    pub states: DeviceView<'a>,
}

/// The one query of each sequence of a decode step over its own cache, at
/// its own state's length, as [`super::DecodeAttnDesc`] over that sequence
/// alone; a finished sequence's row is zeros.
#[derive(Debug, Clone, Copy)]
pub struct DecodeAttnBatchDesc<'a> {
    /// Destination, `[seqs][heads * head_dim]`.
    pub out: DeviceView<'a>,
    /// Scratch, at least [`decode_attention_partials`]`(seqs * heads,
    /// head_dim, splits)` floats.
    pub partials: DeviceView<'a>,
    /// Runs each sequence's cached rows are cut into.
    pub splits: usize,
    /// The queries, as the destination.
    pub q: DeviceView<'a>,
    /// The layer's cache.
    pub cache: BatchPlanes<'a>,
    /// Query heads, a multiple of the cache's kv heads.
    pub heads: usize,
    /// The decoder states, `[seqs][DECODE_STATE_WORDS]`.
    pub states: DeviceView<'a>,
    /// Scale on the logit.
    pub scale: f32,
}

/// One decision of each sequence of a batch, as [`super::DecodeStepDesc`]
/// on each sequence alone.
#[derive(Debug, Clone, Copy)]
pub struct DecodeStepBatchDesc<'a> {
    /// The decoder states, `[seqs][DECODE_STATE_WORDS]`, read and advanced.
    pub states: DeviceView<'a>,
    /// `u32`, `[slots][seqs]`: step `slot` of sequence `s` at
    /// `slot * seqs + s`.
    pub out: DeviceView<'a>,
    /// The head's argmax of each sequence's row, `[seqs]` `u32`.
    pub ids: DeviceView<'a>,
    /// The ids that end the text.
    pub eos: [u32; 2],
    /// Tokens each sequence may emit in all.
    pub max_tokens: usize,
    /// Cache rows the pass before this step appended to each live sequence:
    /// 0 or 1.
    pub advance: usize,
    /// This step's slot.
    pub slot: usize,
    /// Sequences.
    pub seqs: usize,
}

/// The f32 embedding row of each sequence's last token, as
/// [`super::EmbedTokenDesc`] on each state.
#[derive(Debug, Clone, Copy)]
pub struct EmbedTokenBatchDesc<'a> {
    /// `[seqs][width]`.
    pub out: DeviceView<'a>,
    /// The table.
    pub table: EmbedTable<'a>,
    /// The decoder states, `[seqs][DECODE_STATE_WORDS]`.
    pub states: DeviceView<'a>,
    /// Row width.
    pub width: usize,
    /// Rows in the table.
    pub vocab: usize,
    /// Sequences.
    pub seqs: usize,
}

/// Refuses shared planes shorter than any sequence's plane, planes of another
/// head count than the table was checked at, or a geometry no cache kernel
/// is written for.
fn need_batch(c: &BatchPlanes<'_>, heads: usize, what: &'static str) -> CoreResult<()> {
    // The table's planes were checked disjoint at this many heads.
    limit(
        c.table.kv_heads == c.kv_heads,
        "batch_cache.kv_heads",
        c.kv_heads,
    )?;
    limit(
        (1..=MAX_HEAD_DIM).contains(&c.head_dim),
        "batch_cache.head_dim",
        c.head_dim,
    )?;
    limit(
        c.kv_heads > 0 && heads.is_multiple_of(c.kv_heads),
        what,
        heads,
    )?;
    let end = c
        .table
        .spans()
        .iter()
        .map(|s| s.plane_end(c.kv_heads))
        .max()
        .unwrap_or(0);
    need(c.k, end * c.head_dim, "batch_cache.k")?;
    need(c.v, end * c.head_dim, "batch_cache.v")
}

/// Stacked rows the table's sequences reach.
fn stacked_rows(table: &SeqTable) -> usize {
    table
        .spans()
        .iter()
        .map(|s| s.row0 + s.rows)
        .max()
        .unwrap_or(0)
}

impl Ops {
    /// [`CausalAttnSegDesc`]: the segmented prefill's attention in one
    /// launch.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head wider than 128, query heads that
    /// are not a multiple of the kv heads, or a sequence's queries past its
    /// plane; [`CoreError::ShapeMismatch`] for a short view; or the launch's
    /// error.
    pub fn causal_attention_seg(&self, d: &CausalAttnSegDesc<'_>) -> CoreResult<()> {
        let c = &d.cache;
        need_batch(c, d.heads, "causal_attention_seg.heads")?;
        for s in c.table.spans() {
            limit(
                s.pos0 + s.rows <= s.cap,
                "causal_attention_seg.pos0",
                s.pos0 + s.rows,
            )?;
        }
        let width = d.heads * c.head_dim;
        let rows = stacked_rows(c.table);
        need(d.out, rows * width, "causal_attention_seg.out")?;
        need(d.q, rows * width, "causal_attention_seg.q")?;
        let most = c.table.spans().iter().map(|s| s.rows).max().unwrap_or(0);
        let args = BatchAttnArgs {
            seqs: dim(c.table.len(), "causal_attention_seg.seqs")?,
            heads: dim(d.heads, "causal_attention_seg.heads")?,
            kv_heads: dim(c.kv_heads, "causal_attention_seg.kv_heads")?,
            head_dim: dim(c.head_dim, "causal_attention_seg.head_dim")?,
            scale: d.scale,
            splits: 0,
        };
        self.launch(
            "kernel_audio_causal_attn_seg_f32",
            Extent3::new(
                dim(d.heads.max(1), "causal_attention_seg.heads")?,
                dim(
                    most.div_ceil(ATTN_QUERY_BLOCK).max(1),
                    "causal_attention_seg.rows",
                )?,
                args.seqs,
            ),
            32,
            dim(attention_smem(c.head_dim), "causal_attention_seg.smem")?,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(c.k),
                KernelArg::Buffer(c.v),
                KernelArg::Buffer(c.table.words.view()),
            ],
        )
    }

    /// [`QkRopeAppendSegDesc`]: the segmented prefill's norms, rotation and
    /// append in one launch.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd or over-wide head, heads that
    /// are not a multiple of the kv heads, or a sequence past the stacked
    /// rows; [`CoreError::ShapeMismatch`] for a short view; or the launch's
    /// error.
    pub fn qk_norm_rope_append_seg(&self, d: &QkRopeAppendSegDesc<'_>) -> CoreResult<()> {
        let c = &d.cache;
        need_batch(c, d.heads, "qk_norm_rope_append_seg.heads")?;
        limit(
            c.head_dim.is_multiple_of(2),
            "qk_norm_rope_append_seg.head_dim",
            c.head_dim,
        )?;
        let reach = stacked_rows(c.table);
        limit(reach <= d.rows, "qk_norm_rope_append_seg.rows", reach)?;
        let (qw, kw) = (d.heads * c.head_dim, c.kv_heads * c.head_dim);
        need(d.q, d.rows * qw, "qk_norm_rope_append_seg.q")?;
        need(d.k, d.rows * kw, "qk_norm_rope_append_seg.k")?;
        need(d.v, d.rows * kw, "qk_norm_rope_append_seg.v")?;
        need(d.q_norm, c.head_dim, "qk_norm_rope_append_seg.q_norm")?;
        need(d.k_norm, c.head_dim, "qk_norm_rope_append_seg.k_norm")?;
        let args = QkRopeSeqArgs {
            rows: dim(d.rows, "qk_norm_rope_append_seg.rows")?,
            seqs: dim(c.table.len(), "qk_norm_rope_append_seg.seqs")?,
            heads: dim(d.heads, "qk_norm_rope_append_seg.heads")?,
            kv_heads: dim(c.kv_heads, "qk_norm_rope_append_seg.kv_heads")?,
            head_dim: dim(c.head_dim, "qk_norm_rope_append_seg.head_dim")?,
            eps: d.eps,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_qk_norm_rope_append_seg_f32",
            Extent3::new(
                dim(d.heads + c.kv_heads, "qk_norm_rope_append_seg.grid")?,
                dim(d.rows.max(1), "qk_norm_rope_append_seg.grid")?,
                1,
            ),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                KernelArg::Buffer(d.q_norm),
                KernelArg::Buffer(d.k_norm),
                KernelArg::Buffer(c.k),
                KernelArg::Buffer(c.v),
                KernelArg::Buffer(c.table.words.view()),
            ],
        )
    }

    /// [`QkRopeAppendBatchDesc`]: a batched decode step's norms, rotation and
    /// append in one launch.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for an odd or over-wide head, or heads
    /// that are not a multiple of the kv heads;
    /// [`CoreError::ShapeMismatch`] for a short view; or the launch's error.
    pub fn qk_norm_rope_append_batch(&self, d: &QkRopeAppendBatchDesc<'_>) -> CoreResult<()> {
        let c = &d.cache;
        need_batch(c, d.heads, "qk_norm_rope_append_batch.heads")?;
        limit(
            c.head_dim.is_multiple_of(2),
            "qk_norm_rope_append_batch.head_dim",
            c.head_dim,
        )?;
        let seqs = c.table.len();
        let (qw, kw) = (d.heads * c.head_dim, c.kv_heads * c.head_dim);
        need(d.q, seqs * qw, "qk_norm_rope_append_batch.q")?;
        need(d.k, seqs * kw, "qk_norm_rope_append_batch.k")?;
        need(d.v, seqs * kw, "qk_norm_rope_append_batch.v")?;
        need(d.q_norm, c.head_dim, "qk_norm_rope_append_batch.q_norm")?;
        need(d.k_norm, c.head_dim, "qk_norm_rope_append_batch.k_norm")?;
        need(
            d.states,
            seqs * DECODE_STATE_WORDS,
            "qk_norm_rope_append_batch.states",
        )?;
        let args = QkRopeSeqArgs {
            rows: dim(seqs, "qk_norm_rope_append_batch.seqs")?,
            seqs: dim(seqs, "qk_norm_rope_append_batch.seqs")?,
            heads: dim(d.heads, "qk_norm_rope_append_batch.heads")?,
            kv_heads: dim(c.kv_heads, "qk_norm_rope_append_batch.kv_heads")?,
            head_dim: dim(c.head_dim, "qk_norm_rope_append_batch.head_dim")?,
            eps: d.eps,
            theta: d.theta,
        };
        self.launch(
            "kernel_audio_qk_norm_rope_append_batch_f32",
            Extent3::new(
                dim(d.heads + c.kv_heads, "qk_norm_rope_append_batch.grid")?,
                args.seqs,
                1,
            ),
            LINEAR_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(d.k),
                KernelArg::Buffer(d.v),
                KernelArg::Buffer(d.q_norm),
                KernelArg::Buffer(d.k_norm),
                KernelArg::Buffer(c.k),
                KernelArg::Buffer(c.v),
                KernelArg::Buffer(d.states),
                KernelArg::Buffer(c.table.words.view()),
            ],
        )
    }

    /// [`DecodeAttnBatchDesc`]: each sequence's decode attention, in the
    /// single path's two launches: the splits of every sequence side by
    /// side, then the single merge over the batch's `seqs * heads` heads.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a head that is not a multiple of four
    /// up to 128, query heads that are not a multiple of the kv heads, splits
    /// outside `1..=256`, or a view not 16-byte aligned (a plane's first row
    /// always is, being whole rows of a multiple of four floats);
    /// [`CoreError::ShapeMismatch`] for a short view; or the launch's error.
    pub fn decode_attention_batch(&self, d: &DecodeAttnBatchDesc<'_>) -> CoreResult<()> {
        let c = &d.cache;
        need_batch(c, d.heads, "decode_attention_batch.heads")?;
        let hd = c.head_dim;
        limit(hd.is_multiple_of(4), "decode_attention_batch.head_dim", hd)?;
        limit(
            (1..=DECODE_MAX_SPLITS).contains(&d.splits),
            "decode_attention_batch.splits",
            d.splits,
        )?;
        let seqs = c.table.len();
        let width = d.heads * hd;
        need(d.out, seqs * width, "decode_attention_batch.out")?;
        need(d.q, seqs * width, "decode_attention_batch.q")?;
        need(
            d.partials,
            decode_attention_partials(seqs * d.heads, hd, d.splits),
            "decode_attention_batch.partials",
        )?;
        need(
            d.states,
            seqs * DECODE_STATE_WORDS,
            "decode_attention_batch.states",
        )?;
        for (view, field) in [
            (d.out, "decode_attention_batch.out"),
            (d.partials, "decode_attention_batch.partials"),
            (d.q, "decode_attention_batch.q"),
            (c.k, "decode_attention_batch.cache.k"),
            (c.v, "decode_attention_batch.cache.v"),
        ] {
            aligned16(view, field)?;
        }
        let args = BatchAttnArgs {
            seqs: dim(seqs, "decode_attention_batch.seqs")?,
            heads: dim(d.heads, "decode_attention_batch.heads")?,
            kv_heads: dim(c.kv_heads, "decode_attention_batch.kv_heads")?,
            head_dim: dim(hd, "decode_attention_batch.head_dim")?,
            scale: d.scale,
            splits: dim(d.splits, "decode_attention_batch.splits")?,
        };
        let tiles = (d.heads / c.kv_heads).div_ceil(DECODE_GROUP_TILE);
        self.launch(
            "kernel_audio_decode_attn_batch_f32",
            Extent3::new(
                args.kv_heads,
                dim(
                    d.splits.div_ceil(DECODE_SPLITS_PER_TG),
                    "decode_attention_batch.splits",
                )?,
                dim(tiles * seqs, "decode_attention_batch.group")?,
            ),
            dim(32 * DECODE_SPLITS_PER_TG, "decode_attention_batch.block")?,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.partials),
                KernelArg::Buffer(d.q),
                KernelArg::Buffer(c.k),
                KernelArg::Buffer(c.v),
                KernelArg::Buffer(d.states),
                KernelArg::Buffer(c.table.words.view()),
            ],
        )?;
        // The partials are the single launch's for `seqs * heads` heads, so
        // its merge closes them into `[seqs][heads * head_dim]`.
        let merge = CacheAttnArgs {
            n_q: 1,
            heads: dim(seqs * d.heads, "decode_attention_batch.heads")?,
            kv_heads: args.kv_heads,
            head_dim: args.head_dim,
            pos0: 0,
            cap: 0,
            scale: d.scale,
            splits: args.splits,
        };
        self.launch(
            "kernel_audio_decode_attn_merge_f32",
            Extent3::linear(merge.heads),
            DECODE_MERGE_THREADS,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&merge)),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.partials),
            ],
        )
    }

    /// [`DecodeStepBatchDesc`]: each sequence's decision in one launch, a
    /// thread per sequence.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for no sequences, a zero token cap or an
    /// advance past one row, [`CoreError::ShapeMismatch`] for a short view,
    /// or the launch's error.
    pub fn decode_step_batch(&self, d: &DecodeStepBatchDesc<'_>) -> CoreResult<()> {
        limit(d.seqs > 0, "decode_step_batch.seqs", d.seqs)?;
        limit(
            d.max_tokens > 0,
            "decode_step_batch.max_tokens",
            d.max_tokens,
        )?;
        limit(d.advance <= 1, "decode_step_batch.advance", d.advance)?;
        need(
            d.states,
            d.seqs * DECODE_STATE_WORDS,
            "decode_step_batch.states",
        )?;
        need(d.out, (d.slot + 1) * d.seqs, "decode_step_batch.out")?;
        need(d.ids, d.seqs, "decode_step_batch.ids")?;
        let args = DecodeStepBatchArgs {
            eos0: d.eos[0],
            eos1: d.eos[1],
            max_tokens: dim(d.max_tokens, "decode_step_batch.max_tokens")?,
            advance: dim(d.advance, "decode_step_batch.advance")?,
            slot: dim(d.slot, "decode_step_batch.slot")?,
            seqs: dim(d.seqs, "decode_step_batch.seqs")?,
        };
        self.launch(
            "kernel_audio_decode_step_batch_u32",
            Extent3::linear(args.seqs.div_ceil(STEP_BLOCK)),
            STEP_BLOCK,
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d.states),
                KernelArg::Buffer(d.out),
                KernelArg::Buffer(d.ids),
            ],
        )
    }

    /// [`EmbedTokenBatchDesc`]: each sequence's embedding row in one launch.
    ///
    /// # Errors
    ///
    /// As [`Self::embed_token`], and [`CoreError::InvalidValue`] for no
    /// sequences.
    pub fn embed_token_batch(&self, d: &EmbedTokenBatchDesc<'_>) -> CoreResult<()> {
        limit(d.seqs > 0, "embed_token_batch.seqs", d.seqs)?;
        need(d.out, d.seqs * d.width, "embed_token_batch.out")?;
        need(
            d.states,
            d.seqs * DECODE_STATE_WORDS,
            "embed_token_batch.states",
        )?;
        self.embed_rows(d.out, d.table, d.states, (d.width, d.vocab, d.seqs), true)
    }

    /// The embedding fetch of `seqs` rows, one per state: the single entry
    /// for one row of one state, or the batched one. The caller has checked
    /// `out` and the states.
    pub(super) fn embed_rows(
        &self,
        out: DeviceView<'_>,
        table: EmbedTable<'_>,
        states: DeviceView<'_>,
        (width, vocab, seqs): (usize, usize, usize),
        batch: bool,
    ) -> CoreResult<()> {
        let mut args = EmbedTokenArgs {
            width: dim(width, "embed_token.width")?,
            vocab: dim(vocab, "embed_token.vocab")?,
            group: 0,
        };
        let grid = Extent3::new(
            dim(
                width.div_ceil(LINEAR_BLOCK as usize).max(1),
                "embed_token.grid",
            )?,
            dim(seqs, "embed_token.seqs")?,
            1,
        );
        match table {
            EmbedTable::Dense { dtype, rows } => {
                let row = dtype.row_bytes(width).ok_or(CoreError::InvalidValue {
                    key: "embed_token.width",
                    value: width.to_string(),
                })?;
                need_bytes(rows, vocab * row, "embed_token.table")?;
                let entry = if batch {
                    dtype.embed_batch_entry()
                } else {
                    dtype.embed_entry()
                };
                self.launch(
                    entry,
                    grid,
                    LINEAR_BLOCK,
                    0,
                    &[
                        KernelArg::Bytes(bytemuck::bytes_of(&args)),
                        KernelArg::Buffer(out),
                        KernelArg::Buffer(rows),
                        KernelArg::Buffer(states),
                    ],
                )
            }
            EmbedTable::Affine8 {
                words,
                scales,
                biases,
                group,
            } => {
                limit(
                    group > 0 && width.is_multiple_of(group) && width.is_multiple_of(4),
                    "embed_token.group",
                    group,
                )?;
                need_bytes(words, vocab * width, "embed_token.words")?;
                let groups = vocab * (width / group);
                need_bytes(scales, groups * 2, "embed_token.scales")?;
                need_bytes(biases, groups * 2, "embed_token.biases")?;
                args.group = dim(group, "embed_token.group")?;
                let entry = if batch {
                    "kernel_audio_embed_token_batch_affine8w"
                } else {
                    "kernel_audio_embed_token_affine8w"
                };
                self.launch(
                    entry,
                    grid,
                    LINEAR_BLOCK,
                    0,
                    &[
                        KernelArg::Bytes(bytemuck::bytes_of(&args)),
                        KernelArg::Buffer(out),
                        KernelArg::Buffer(words),
                        KernelArg::Buffer(scales),
                        KernelArg::Buffer(biases),
                        KernelArg::Buffer(states),
                    ],
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SeqSpan, disjoint};

    fn span(plane: usize, cap: usize, first: usize, count: usize) -> SeqSpan {
        SeqSpan {
            plane,
            cap,
            row0: first,
            rows: count,
            pos0: 0,
        }
    }

    #[test]
    fn a_table_refuses_shared_cache_rows_and_shared_stacked_rows() {
        // Two heads: plane 0 covers rows [0, 20), plane 1 from 20 does not
        // touch it, and a gap between planes is fine.
        disjoint(
            &[span(0, 10, 0, 5), span(20, 4, 5, 3), span(30, 1, 8, 0)],
            2,
        )
        .expect("disjoint");
        // Decode tables carry no stacked rows, so row0 may repeat.
        disjoint(&[span(0, 10, 0, 0), span(20, 10, 0, 0)], 2).expect("no stacked rows");
        let err = disjoint(&[span(0, 10, 0, 5), span(19, 4, 5, 3)], 2).expect_err("row 19");
        assert!(err.to_string().contains("planes_overlap"), "{err}");
        let err = disjoint(&[span(0, 10, 0, 5), span(20, 4, 4, 3)], 2).expect_err("row 4");
        assert!(err.to_string().contains("stacked_rows_overlap"), "{err}");
        assert!(disjoint(&[span(0, 1, 0, 1)], 0).is_err(), "no kv heads");
    }
}
