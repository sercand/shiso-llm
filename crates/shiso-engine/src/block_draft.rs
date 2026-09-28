//! A block drafter: several tokens proposed in one pass from the target's own
//! hidden states.
//!
//! A draft head proposes one token a pass and chains passes; a block drafter
//! proposes a whole block at once. It reads the target's residual at a few of
//! its layers - its *taps* - for every committed position, and runs a short
//! stack of its own over `[pending, mask, mask, ...]`, attending to those
//! positions and to the block itself. Every position of the block comes out of
//! the same pass, so a block of eight costs one small forward instead of seven
//! chained ones.
//!
//! # What the drafter keeps, and what it recomputes
//!
//! Its context is keys and values, one attention cache per drafter layer, in a
//! pool of its own that shares the target session's page table - the same
//! arrangement [`crate::DraftHead`] uses, for the same reason: the drafter
//! writes the positions the target writes, so page `p` of its pool is its own
//! storage for whatever position the target keeps at page `p`. A committed
//! position's keys and values are projected once, from that position's taps
//! ([`BlockDrafter::absorb`]); the block's own keys and values are written at
//! the positions it proposes and overwritten by the next absorb, which is
//! exactly when those positions become context.
//!
//! # Correctness does not depend on it
//!
//! A drafter that is wrong proposes tokens the target rejects, and the answer
//! is the one plain decoding gives. Only the acceptance rate is at stake - which
//! is why its projections run 4-bit, why its context is as long as its window
//! and no longer, and why its last selection step runs on the host.
//!
//! # The layer
//!
//! Pre-norm attention and a gated feed-forward, each wrapped in a grouped
//! dynamic causal convolution: a projection of the block's normed input
//! predicts, per row, a short kernel per channel group; the first half is
//! convolved into the input before the block, the second into its output after.
//! The attention is non-causal inside the block - every drafted position sees
//! every other - and windowed over the context.
//!
//! # The selector
//!
//! The drafter's logits give each position a short list of candidates. A
//! low-rank bigram term scores each candidate against the one chosen before it,
//! `predecessor[prev] . projection(h) . successor[candidate]`, and the greedy
//! path through the lists is the block. Sixteen candidates at seven positions
//! with a rank of 256 is a few hundred thousand multiplies, so it runs on the
//! host over the logits the device wrote.

use std::sync::Arc;

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DraftHint, Speculated};

use crate::attn::{Attn, AttnCache, AttnGeometry, Pages, Positions};
use crate::blocks::AttnIo;
use crate::ffn::Ffn;
use crate::kv::{DeviceKv, KvLayout, PAGE_TOKENS};
use crate::launch::{self, BLOCK_LINEAR, DENSE, F32_BYTES};
use crate::layers::{self, Geometry, ResidualIndex, Scratch, Weights};
use crate::projection::{self, Weight};
use crate::shape::Activation;

/// A block drafter's widths and conventions, as its checkpoint states them.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockDraftShape {
    /// Residual width, which must be the target's.
    pub hidden: u32,
    /// Query heads.
    pub n_head: u32,
    /// Key and value heads.
    pub n_head_kv: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Feed-forward width.
    pub intermediate: u32,
    /// Rows of one block: the pending token and the positions drafted behind it.
    pub block: u32,
    /// Context positions a query looks back over.
    pub window: u32,
    /// Taps of the dynamic convolution.
    pub conv_taps: u32,
    /// Channels that share one dynamic kernel.
    pub conv_group: u32,
    /// Rotary frequency base.
    pub rope_theta: f32,
    /// Epsilon of every norm.
    pub eps: f32,
    /// Target layers whose output residual the drafter reads, in the order its
    /// input projection concatenates them.
    pub taps: Vec<u32>,
    /// The id every drafted position enters the block as.
    pub mask_token: TokenId,
    /// Width of the selector's low-rank bigram term.
    pub selector_rank: u32,
    /// Candidates the selector weighs at each position.
    pub selector_top_k: u32,
}

impl BlockDraftShape {
    /// Checks the shape against itself and against the target it drafts for.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] naming the first disagreement.
    pub fn check(&self, target_hidden: u32, target_layers: u32) -> CoreResult<()> {
        let mismatch = |field: &'static str, expected: u32, got: u32| {
            Err(CoreError::ShapeMismatch {
                field,
                expected: u64::from(expected),
                got: u64::from(got),
            })
        };
        if self.hidden != target_hidden {
            return mismatch("block drafter hidden size", target_hidden, self.hidden);
        }
        if let Some(&late) = self.taps.iter().find(|&&t| t >= target_layers) {
            return mismatch("block drafter tap layer", target_layers - 1, late);
        }
        if self.taps.is_empty() {
            return mismatch("block drafter taps", 1, 0);
        }
        if self.block < 2 {
            return mismatch("block drafter block", 2, self.block);
        }
        if self.conv_group == 0 || !self.hidden.is_multiple_of(self.conv_group) {
            return mismatch("block drafter conv group", 1, self.conv_group);
        }
        if self.n_head_kv == 0 || !self.n_head.is_multiple_of(self.n_head_kv) {
            return mismatch("block drafter heads", self.n_head_kv, self.n_head);
        }
        Ok(())
    }

    /// Width of the dynamic kernel projection: both halves, every tap, every
    /// channel group.
    #[must_use]
    pub const fn dyn_width(&self) -> u32 {
        2 * self.conv_taps * (self.hidden / self.conv_group)
    }
}

/// One grouped dynamic causal convolution: its static kernel and the projection
/// that predicts the dynamic one.
#[derive(Debug)]
pub struct DynConv {
    /// `[2][taps][hidden]` `f32`: the static kernel of each half.
    pub base: DeviceBuffer,
    /// `hidden -> 2 * taps * groups`: the per-row dynamic kernel.
    pub projection: Weight,
}

/// One drafter layer.
#[derive(Debug)]
pub struct BlockDraftLayer {
    /// Norm the attention half reads, `[hidden]`.
    pub input_norm: DeviceBuffer,
    /// Norm the feed-forward half reads, `[hidden]`.
    pub post_norm: DeviceBuffer,
    /// The attention block, with its key and value projections.
    pub attn: Attn,
    /// The gated feed-forward.
    pub ffn: Ffn,
    /// The convolution around the attention.
    pub attn_conv: DynConv,
    /// The convolution around the feed-forward.
    pub mlp_conv: DynConv,
}

/// A `[vocab][rank]` table held on the host as `bfloat16`.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Values per row.
    pub rank: u32,
    /// Rows, back to back.
    pub values: Vec<u16>,
}

impl Codebook {
    /// Row `id`, widened.
    fn row(&self, id: u32) -> impl Iterator<Item = f32> + '_ {
        let rank = self.rank as usize;
        let at = id as usize * rank;
        self.values
            .get(at..at + rank)
            .unwrap_or(&[])
            .iter()
            .map(|&bits| half_to_f32(bits))
    }
}

/// A `bfloat16` bit pattern, widened.
fn half_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Everything a block drafter carries, bound by the crate that knows its
/// checkpoint.
#[derive(Debug)]
pub struct BlockDraftParts {
    /// Its widths.
    pub shape: BlockDraftShape,
    /// `taps * hidden -> hidden`: the concatenated taps to one context row.
    pub fc: Weight,
    /// Norm over that row, `[hidden]`.
    pub hidden_norm: DeviceBuffer,
    /// The layers.
    pub layers: Vec<BlockDraftLayer>,
    /// Norm in front of the vocabulary projection, `[hidden]`.
    pub norm: DeviceBuffer,
    /// `hidden -> rank`: the selector's view of a drafted position.
    pub hidden_projection: Weight,
    /// The selector's table for the token before.
    pub predecessor: Codebook,
    /// The selector's table for the candidate.
    pub successor: Codebook,
}

/// Where a target pass writes its tapped residual rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TapSink<'a> {
    /// The tapped layers, in concatenation order.
    pub(crate) layers: &'a [u32],
    /// `[row][tap][hidden]` `f32`.
    pub(crate) dst: DeviceView<'a>,
    /// Rows the sink holds.
    pub(crate) rows: u32,
}

/// A block drafter resident on the device.
#[derive(Debug)]
pub struct BlockDrafter {
    parts: BlockDraftParts,
    /// Key and value runs, one attention slot per drafter layer.
    kv: DeviceKv,
    /// Activation rows at the target's widths, which are wide enough for the
    /// drafter's own.
    scratch: Scratch,
    /// The tapped residual rows of the last target pass.
    taps: DeviceBuffer,
    /// Convolution output rows.
    conv: DeviceBuffer,
    /// Dynamic kernel rows.
    dynk: DeviceBuffer,
    /// Drafted positions' logits over the whole vocabulary.
    logits: DeviceBuffer,
    /// Drafted positions' selector projections.
    projected: DeviceBuffer,
    /// The widths the attention launches at.
    geometry: AttnGeometry,
    /// Rows every buffer holds.
    rows: u32,
    /// Staging for the int32 rows.
    staging: Vec<u8>,
    /// Each speculating session's copy-lookup and acceptance state.
    copies: std::collections::HashMap<SessionId, CopyState>,
    /// What rounds cost on this device, smoothed.
    costs: RoundCosts,
}

/// Where a session's generation began, the copies it has seen fail, and how
/// well drafting has been paying for it.
#[derive(Debug)]
struct CopyState {
    generated_from: u32,
    fails: crate::mtp::FailRing,
    /// Drafts the target kept per drafted round, smoothed.
    accepted: f64,
    /// Plain rounds since the last drafted one.
    idle: u32,
}

impl CopyState {
    fn new(generated_from: u32) -> Self {
        Self {
            generated_from,
            fails: crate::mtp::FailRing::new(),
            // A prior, not a measurement: about what the drafter keeps on
            // prose, so a session starts drafting and learns from there.
            accepted: 2.0,
            idle: 0,
        }
    }
}

/// Weight of the newest round in the smoothed acceptance.
const ACCEPT_EMA: f64 = 0.3;

/// Weight of the newest round in the smoothed round costs.
const COST_EMA: f64 = 0.1;

/// How far below a plain round's rate a drafted round's expected rate has to
/// fall before a session stops drafting.
const LOSING: f64 = 0.8;

/// Plain rounds after which a session that stopped drafting drafts one round
/// again, so its acceptance estimate follows the text.
const PROBE_EVERY: u32 = 8;

/// What a round costs, smoothed, by whether it verified a block or one row.
#[derive(Debug, Clone, Copy)]
struct RoundCosts {
    /// Milliseconds of a drafted round: the draft and a block-wide verify.
    drafted_ms: f64,
    /// Milliseconds of a one-row round.
    plain_ms: f64,
}

impl Default for RoundCosts {
    /// Priors from an M1 Max on the 27B: a drafted round about 1.9 plain ones.
    /// Replaced by measurement within a few rounds.
    fn default() -> Self {
        Self {
            drafted_ms: 175.0,
            plain_ms: 92.0,
        }
    }
}

impl BlockDrafter {
    /// Makes `parts` resident beside a target of `target`'s geometry and layer
    /// count.
    ///
    /// `rows` is the widest target pass whose taps it absorbs at once - a
    /// prompt chunk - and at least a block.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a drafter the target cannot feed, and
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        parts: BlockDraftParts,
        target: (&Geometry, u32),
        pool_pages: u32,
        rows: u32,
    ) -> CoreResult<Self> {
        let (geometry, target_layers) = target;
        let shape = &parts.shape;
        shape.check(geometry.hidden, target_layers)?;
        let rows = rows.max(shape.block);
        let n_taps = u32::try_from(shape.taps.len()).unwrap_or(u32::MAX);
        if parts.fc.in_dim != n_taps * shape.hidden || parts.fc.out_dim != shape.hidden {
            return Err(CoreError::ShapeMismatch {
                field: "block drafter input projection",
                expected: u64::from(n_taps * shape.hidden),
                got: u64::from(parts.fc.in_dim),
            });
        }
        let layers = u32::try_from(parts.layers.len()).unwrap_or(u32::MAX);
        let kv = DeviceKv::open(
            backend,
            &KvLayout::for_attn_slots(layers, shape.n_head_kv, shape.head_dim),
            pool_pages,
        )?;
        let scratch = Scratch::new(backend, geometry, rows, 1)?;
        let row = u64::from(shape.hidden) * F32_BYTES;
        let taps = alloc(backend, u64::from(rows) * u64::from(n_taps) * row)?;
        let conv = alloc(backend, u64::from(rows) * row)?;
        let dynk = alloc(
            backend,
            u64::from(rows) * u64::from(shape.dyn_width()) * F32_BYTES,
        )?;
        let drafted = u64::from(shape.block - 1);
        let logits = alloc(backend, drafted * u64::from(geometry.vocab) * F32_BYTES)?;
        let projected = alloc(
            backend,
            drafted * u64::from(shape.selector_rank) * F32_BYTES,
        )?;
        let geometry = AttnGeometry {
            n_head: shape.n_head,
            n_head_kv: shape.n_head_kv,
            head_dim: shape.head_dim,
            n_rot: shape.head_dim,
            freq_dims: shape.head_dim,
            rope_base: shape.rope_theta,
            eps: shape.eps,
            page_tokens: PAGE_TOKENS,
            window: shape.window,
            #[expect(clippy::cast_precision_loss, reason = "a head width")]
            scale: 1.0 / (shape.head_dim as f32).sqrt(),
            v_norm: false,
            output_gate: false,
            owns_kv: true,
            qk_norm: true,
        };
        tracing::info!(
            layers,
            block = shape.block,
            taps = ?shape.taps,
            pool_pages,
            rows,
            "block drafter resident"
        );
        Ok(Self {
            parts,
            kv,
            scratch,
            taps,
            conv,
            dynk,
            logits,
            projected,
            geometry,
            rows,
            staging: Vec::new(),
            copies: std::collections::HashMap::new(),
            costs: RoundCosts::default(),
        })
    }

    /// Folds one round's outcome into the session's acceptance and the
    /// device's round costs, and remembers a copy the target rejected.
    fn learn(
        &mut self,
        session: SessionId,
        outcome: &crate::mtp::StepOutcome,
        drafted: usize,
        took: std::time::Duration,
    ) {
        // A sample is held to three times the running estimate: a round that
        // paid for compiling its pipelines, or waited on the host, is not what
        // the next round will cost, and one of them would otherwise keep a
        // session from drafting for dozens of rounds while the average decays.
        let ms = took.as_secs_f64() * 1e3;
        if drafted == 0 {
            let est = self.costs.plain_ms;
            self.costs.plain_ms += COST_EMA * (ms.min(3.0 * est) - est);
        } else {
            let est = self.costs.drafted_ms;
            self.costs.drafted_ms += COST_EMA * (ms.min(3.0 * est) - est);
        }
        let Some(state) = self.copies.get_mut(&session) else {
            return;
        };
        if let Some(id) = outcome.failed_lookup() {
            state.fails.remember(id);
        }
        // A scaffold round says how well the caller knew its structure, not
        // how well the drafter drafts, so it leaves the estimate alone.
        if drafted > 0 && !matches!(outcome.source, Some(crate::mtp::DraftSource::Scaffold)) {
            let kept = f64::from(outcome.logits_row);
            state.accepted += ACCEPT_EMA * (kept - state.accepted);
        }
    }

    /// Rows of one block.
    #[must_use]
    pub const fn block(&self) -> u32 {
        self.parts.shape.block
    }

    /// Where the next target pass writes its taps.
    pub(crate) fn sink(&self) -> TapSink<'_> {
        TapSink {
            layers: &self.parts.shape.taps,
            dst: self.taps.view(),
            rows: self.rows,
        }
    }

    /// Projects rows `from .. from + n` of the last target pass's taps into the
    /// drafter's context at positions `pos ..`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for more rows than the sink holds, and
    /// whatever a launch reports.
    pub(crate) fn absorb(
        &mut self,
        backend: &Arc<dyn Backend>,
        pages: DeviceView<'_>,
        from: u32,
        n: u32,
        pos: u32,
    ) -> CoreResult<()> {
        if n == 0 {
            return Ok(());
        }
        if from + n > self.rows {
            return Err(CoreError::ShapeMismatch {
                field: "absorbed tap rows",
                expected: u64::from(self.rows),
                got: u64::from(from + n),
            });
        }
        let shape = &self.parts.shape;
        let n_taps = u64::try_from(shape.taps.len()).unwrap_or(u64::MAX);
        let row = u64::from(shape.hidden) * F32_BYTES;
        let taps = self
            .taps
            .view()
            .slice(u64::from(from) * n_taps * row, u64::from(n) * n_taps * row)?;
        crate::stack::upload_row(
            backend.as_ref(),
            &mut self.staging,
            self.scratch.positions(),
            (0..n).map(|t| i32::try_from(pos + t).unwrap_or(i32::MAX)),
        )?;
        let ctx = self.scratch.stream_at(ResidualIndex::SECOND)?;
        projection::projection(backend.as_ref(), &self.parts.fc, n, taps, ctx)?;
        layers::rms_norm_mul(
            backend.as_ref(),
            ctx,
            self.parts.hidden_norm.view(),
            self.scratch.normed(),
            shape.hidden,
            n,
            shape.eps,
        )?;
        for (rank, layer) in self.parts.layers.iter().enumerate() {
            let (k, v) = self
                .kv
                .layer_runs(u32::try_from(rank).unwrap_or(u32::MAX))?;
            crate::dense::attn::store(
                backend.as_ref(),
                &layer.attn,
                &self.geometry,
                self.scratch.attn(),
                &AttnCache {
                    k,
                    v,
                    pages: Pages::shared(pages),
                },
                crate::dense::attn::StoreRows {
                    pos: self.scratch.positions(),
                    input: self.scratch.normed(),
                    n_tok: n,
                },
            )?;
        }
        Ok(())
    }

    /// The block behind `first`, which sits at position `pos`: `block - 1`
    /// drafted tokens.
    ///
    /// # Errors
    ///
    /// Whatever a launch or the read-back reports.
    pub(crate) fn propose(
        &mut self,
        backend: &Arc<dyn Backend>,
        target: (&Weights, &Geometry),
        pages: DeviceView<'_>,
        at: (TokenId, u32),
        hint: DraftHint,
    ) -> CoreResult<Vec<TokenId>> {
        let (weights, geometry) = target;
        let (first, pos) = at;
        let shape = &self.parts.shape;
        let b = shape.block;
        let hidden = shape.hidden;
        let row = u64::from(hidden) * F32_BYTES;
        let be = backend.as_ref();
        crate::stack::upload_row(
            be,
            &mut self.staging,
            self.scratch.ids(),
            std::iter::once(first)
                .chain(std::iter::repeat_n(shape.mask_token, (b - 1) as usize))
                .map(|t| i32::try_from(t.get()).unwrap_or(0)),
        )?;
        crate::stack::upload_row(
            be,
            &mut self.staging,
            self.scratch.positions(),
            (0..b).map(|t| i32::try_from(pos + t).unwrap_or(i32::MAX)),
        )?;
        // Every drafted position sees the whole block: one bound for all rows.
        crate::stack::upload_row(
            be,
            &mut self.staging,
            self.scratch.bound(),
            (0..b).map(|_| i32::try_from(pos + b - 1).unwrap_or(i32::MAX)),
        )?;
        let stream = self.scratch.stream_at(ResidualIndex::FIRST)?;
        layers::embed_rows(be, weights, geometry, &self.scratch, stream, b)?;
        for (rank, layer) in self.parts.layers.iter().enumerate() {
            self.layer(
                be,
                layer,
                u32::try_from(rank).unwrap_or(u32::MAX),
                pages,
                stream,
            )?;
        }
        // The drafted positions: every row but the pending token's.
        let drafted = b - 1;
        let tail = stream.slice(row, u64::from(drafted) * row)?;
        let normed = self.scratch.normed().slice(0, u64::from(drafted) * row)?;
        layers::rms_norm_mul(
            be,
            tail,
            self.parts.norm.view(),
            normed,
            hidden,
            drafted,
            shape.eps,
        )?;
        projection::projection(
            be,
            weights.output_matrix(),
            drafted,
            normed,
            self.logits.view(),
        )?;
        projection::projection(
            be,
            &self.parts.hidden_projection,
            drafted,
            normed,
            self.projected.view(),
        )?;
        let vocab = geometry.vocab as usize;
        let mut bytes = vec![0u8; drafted as usize * vocab * 4];
        be.copy_d2h(self.logits.view(), &mut bytes)?;
        let logits = floats(&bytes);
        let rank = shape.selector_rank as usize;
        let mut pbytes = vec![0u8; drafted as usize * rank * 4];
        be.copy_d2h(self.projected.view(), &mut pbytes)?;
        let projected = floats(&pbytes);
        Ok(self.select(first, (&logits, &projected, vocab), hint, pos + 1))
    }

    /// One drafter layer over the block's rows, in place on `stream`.
    fn layer(
        &self,
        be: &dyn Backend,
        layer: &BlockDraftLayer,
        rank: u32,
        pages: DeviceView<'_>,
        stream: DeviceView<'_>,
    ) -> CoreResult<()> {
        self.attention_half(be, layer, rank, pages, stream)?;
        self.feed_forward_half(be, layer, stream)
    }

    /// The block's rows and the convolution buffers every half launches over.
    fn conv_io<'a>(
        &'a self,
        x: DeviceView<'a>,
        base: DeviceView<'a>,
        part: u32,
    ) -> CoreResult<DynConvIo<'a>> {
        let shape = &self.parts.shape;
        Ok(DynConvIo {
            x,
            dynk: self.dynk.view(),
            base,
            out: self.conv.view().slice(
                0,
                u64::from(shape.block) * u64::from(shape.hidden) * F32_BYTES,
            )?,
            part,
            rows: shape.block,
        })
    }

    /// Norm, convolution, attention, convolution, residual add.
    fn attention_half(
        &self,
        be: &dyn Backend,
        layer: &BlockDraftLayer,
        rank: u32,
        pages: DeviceView<'_>,
        stream: DeviceView<'_>,
    ) -> CoreResult<()> {
        let shape = &self.parts.shape;
        let (b, hidden) = (shape.block, shape.hidden);
        let normed = self.scratch.normed();
        let base = layer.attn_conv.base.view();
        layers::rms_norm_mul(
            be,
            stream,
            layer.input_norm.view(),
            normed,
            hidden,
            b,
            shape.eps,
        )?;
        projection::projection(be, &layer.attn_conv.projection, b, normed, self.dynk.view())?;
        let before = self.conv_io(normed, base, 0)?;
        dyn_conv(be, shape, before)?;
        let (k, v) = self.kv.layer_runs(rank)?;
        crate::dense::attn::run(
            be,
            &layer.attn,
            &self.geometry,
            self.scratch.attn(),
            AttnIo {
                input: before.out,
                pos: Positions::Linear(self.scratch.positions()),
                bound: self.scratch.bound(),
                block_out: self.scratch.block_out(),
            },
            &AttnCache {
                k,
                v,
                pages: Pages::shared(pages),
            },
            b,
        )?;
        let after = self.conv_io(self.scratch.block_out(), base, 1)?;
        dyn_conv(be, shape, after)?;
        layers::add2(be, stream, after.out, stream, b * hidden)
    }

    /// Norm, convolution, gated feed-forward, convolution, residual add.
    fn feed_forward_half(
        &self,
        be: &dyn Backend,
        layer: &BlockDraftLayer,
        stream: DeviceView<'_>,
    ) -> CoreResult<()> {
        let shape = &self.parts.shape;
        let (b, hidden) = (shape.block, shape.hidden);
        let normed = self.scratch.normed();
        let base = layer.mlp_conv.base.view();
        layers::rms_norm_mul(
            be,
            stream,
            layer.post_norm.view(),
            normed,
            hidden,
            b,
            shape.eps,
        )?;
        projection::projection(be, &layer.mlp_conv.projection, b, normed, self.dynk.view())?;
        let before = self.conv_io(normed, base, 0)?;
        dyn_conv(be, shape, before)?;
        let [gate, up, mid] = self.scratch.ffn_rows();
        crate::ffn::run(
            be,
            &layer.ffn,
            Activation::SwiGlu,
            before.out,
            gate,
            up,
            mid,
            self.scratch.block_out(),
            b,
        )?;
        let after = self.conv_io(self.scratch.block_out(), base, 1)?;
        dyn_conv(be, shape, after)?;
        layers::add2(be, stream, after.out, stream, b * hidden)
    }

    /// The greedy path through each position's candidates.
    ///
    /// For a sampling caller each candidate's score is divided by its
    /// temperature and carries a share of the Gumbel noise the caller's keyed
    /// draw will add at the position the candidate would occupy - from `at`
    /// for the first drafted row - so the path follows the draw the target is
    /// about to make instead of its most likely token.
    fn select(
        &self,
        first: TokenId,
        rows: (&[f32], &[f32], usize),
        hint: DraftHint,
        at: u32,
    ) -> Vec<TokenId> {
        let (logits, projected, vocab) = rows;
        let shape = &self.parts.shape;
        let rank = shape.selector_rank as usize;
        let k = shape.selector_top_k as usize;
        let sampled = hint.temperature > 0.0;
        let mut prev = first.get();
        let mut out = Vec::with_capacity(shape.block as usize - 1);
        for (i, (row, hidden_row)) in logits
            .chunks_exact(vocab)
            .zip(projected.chunks_exact(rank))
            .enumerate()
        {
            let position = at + u32::try_from(i).unwrap_or(u32::MAX);
            let candidates = top_k(row, k);
            let weighted: Vec<f32> = self
                .parts
                .predecessor
                .row(prev)
                .zip(hidden_row)
                .map(|(p, h)| p * h)
                .collect();
            let mut best = (f32::NEG_INFINITY, candidates.first().copied().unwrap_or(0));
            for &cand in &candidates {
                let edge: f32 = self
                    .parts
                    .successor
                    .row(cand)
                    .zip(&weighted)
                    .map(|(s, p)| s * p)
                    .sum();
                let mut score = row[cand as usize] + edge;
                if sampled {
                    let u = crate::sampling::keyed_uniform(hint.key, position, cand);
                    score = score / hint.temperature + DRAFT_NOISE * -(-u.ln()).ln();
                }
                if score > best.0 {
                    best = (score, cand);
                }
            }
            prev = best.1;
            out.push(TokenId::new(prev));
        }
        out
    }
}

/// The share of the caller's keyed noise a sampled draft's scores carry: the
/// weight a best-first search over this drafter's candidates was measured best
/// at, applied here to a chain.
const DRAFT_NOISE: f32 = 0.7;

/// The `k` highest entries of `row`, highest first, ties to the lower id.
fn top_k(row: &[f32], k: usize) -> Vec<u32> {
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    for (id, &v) in row.iter().enumerate() {
        if best.len() == k && best.last().is_some_and(|&(low, _)| v <= low) {
            continue;
        }
        let at = best.partition_point(|&(x, _)| x >= v);
        best.insert(at, (v, u32::try_from(id).unwrap_or(u32::MAX)));
        best.truncate(k);
    }
    best.into_iter().map(|(_, id)| id).collect()
}

/// Little-endian `f32`s.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// The buffers one convolution launch reads and writes.
#[derive(Debug, Clone, Copy)]
struct DynConvIo<'a> {
    x: DeviceView<'a>,
    dynk: DeviceView<'a>,
    base: DeviceView<'a>,
    out: DeviceView<'a>,
    part: u32,
    rows: u32,
}

/// Argument block of `kernel_dense_dyn_conv_f32`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DynConvArgs {
    n_rows: u32,
    hidden: u32,
    group: u32,
    taps: u32,
    part: u32,
    pad: [u32; 3],
}

const _: () = assert!(size_of::<DynConvArgs>() == 32);

/// One grouped dynamic causal convolution over `io.rows` rows.
fn dyn_conv(backend: &dyn Backend, shape: &BlockDraftShape, io: DynConvIo<'_>) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_dyn_conv_f32";
    let args = DynConvArgs {
        n_rows: io.rows,
        hidden: shape.hidden,
        group: shape.conv_group,
        taps: shape.conv_taps,
        part: io.part,
        pad: [0; 3],
    };
    let n = io.rows * shape.hidden;
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::linear(n.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(io.x),
            KernelArg::Buffer(io.dynk),
            KernelArg::Buffer(io.base),
            KernelArg::Buffer(io.out),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Argument block of `kernel_dense_copy_rows_f32`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyRowsArgs {
    n_rows: u32,
    width: u32,
    dst_stride: u32,
    dst_offset: u32,
}

/// Copies one tapped layer's `n_tok` residual rows into its slot of the sink.
pub(crate) fn tap_rows(
    backend: &dyn Backend,
    sink: TapSink<'_>,
    slot: u32,
    src: DeviceView<'_>,
    hidden: u32,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_copy_rows_f32";
    if n_tok > sink.rows {
        return Err(CoreError::ShapeMismatch {
            field: "tapped rows",
            expected: u64::from(sink.rows),
            got: u64::from(n_tok),
        });
    }
    let n_taps = u32::try_from(sink.layers.len()).unwrap_or(u32::MAX);
    let args = CopyRowsArgs {
        n_rows: n_tok,
        width: hidden,
        dst_stride: n_taps * hidden,
        dst_offset: slot * hidden,
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::linear((n_tok * hidden).div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(src),
            KernelArg::Buffer(sink.dst),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

impl crate::stack::HybridStack {
    /// Makes `drafter` this stack's block drafter.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a stack built without a speculative depth,
    /// whose sessions keep no rollback records, and on a stack that already
    /// carries a draft head: a stack drafts one way.
    pub fn bind_block_drafter(&mut self, drafter: BlockDrafter) -> CoreResult<()> {
        if !self.opts.spec.is_speculative() {
            return Err(CoreError::Unsupported {
                what: "a block drafter on a stack whose sessions were opened with no rollback \
                       records, which is what a speculative depth of zero asks for",
                backend: "engine",
            });
        }
        if self.draft.is_some() {
            return Err(CoreError::Unsupported {
                what: "a block drafter on a stack that already carries a draft head",
                backend: "engine",
            });
        }
        self.block = Some(drafter);
        Ok(())
    }

    /// A verbatim continuation of what the session is generating, when the
    /// text it is writing matches something earlier in the conversation on the
    /// longest key the lookup tries.
    ///
    /// Text that copies - an edited file, a quoted passage - is followed more
    /// reliably by the history than by any drafter, and a copy costs no draft
    /// pass. So a match on the longest key replaces the drafter's block for the
    /// round, and a copy the target rejects is remembered and not proposed from
    /// the same place again.
    // None: no earlier text continues what the session is writing, so the
    // drafter drafts this round.
    fn copy(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
        budget: u32,
    ) -> CoreResult<Option<(Vec<TokenId>, crate::mtp::ProposalId)>> {
        let Self {
            sessions, block, ..
        } = self;
        let Some(drafter) = block.as_mut() else {
            return Ok(None);
        };
        let live = sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let history = live.history();
        let len = u32::try_from(history.len()).unwrap_or(u32::MAX);
        let reach = budget.min(drafter.block().saturating_sub(1));
        let state = drafter
            .copies
            .entry(session)
            .or_insert_with(|| CopyState::new(len));
        // A handle whose history is shorter than where its generation began
        // is a new conversation under a reused handle.
        if state.generated_from > len {
            *state = CopyState::new(len);
        }
        let policy = crate::mtp::NgramPolicy::new(reach, crate::mtp::NGRAM_MAX_MATCH, eos);
        Ok(
            crate::mtp::propose(history, first, state.generated_from, &state.fails, policy)
                .map(|p| (p.tokens().to_vec(), p.id())),
        )
    }

    /// Whether a drafted round is expected to commit more tokens a
    /// millisecond than a plain one, for this session now.
    ///
    /// A drafted round commits its kept drafts and one more; a plain round
    /// commits one. On a device where every width up to a block verifies at
    /// about the same cost, those are the only two choices worth weighing. A
    /// session that stopped drafting drafts one round in [`PROBE_EVERY`], so
    /// text that turns predictable is noticed.
    fn drafting_pays(&mut self, session: SessionId) -> bool {
        let Some(drafter) = self.block.as_mut() else {
            return false;
        };
        let costs = drafter.costs;
        let Some(state) = drafter.copies.get_mut(&session) else {
            return true;
        };
        // Stopped only when drafting is clearly losing: near the break-even
        // the two cost the same, and a session that stopped on noise gives up
        // every round the text turns predictable again until the next probe.
        let pays = (state.accepted + 1.0) / costs.drafted_ms > LOSING / costs.plain_ms;
        if pays || state.idle >= PROBE_EVERY {
            state.idle = 0;
            true
        } else {
            state.idle += 1;
            false
        }
    }

    /// The drafter's block behind `first`, which sits at `frontier`.
    fn draft_block(
        &mut self,
        session: SessionId,
        first: TokenId,
        frontier: u32,
        hint: DraftHint,
    ) -> CoreResult<Vec<TokenId>> {
        let Self {
            backend,
            geometry,
            resident,
            sessions,
            block,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let pages = sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .kv()
            .view();
        let drafter = block.as_mut().ok_or(CoreError::Unsupported {
            what: "a block draft on a stack that carries no block drafter",
            backend: "engine",
        })?;
        drafter.propose(
            backend,
            (&resident.weights, geometry),
            pages,
            (first, frontier),
            hint,
        )
    }

    /// Hands the drafter the first `n` rows of the verify pass that just ran,
    /// at positions from `frontier`.
    fn absorb_kept(&mut self, session: SessionId, n: u32, frontier: u32) -> CoreResult<()> {
        let Self {
            backend,
            sessions,
            block,
            ..
        } = self;
        let pages = sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .kv()
            .view();
        match block.as_mut() {
            Some(drafter) => drafter.absorb(backend, pages, 0, n, frontier),
            None => Ok(()),
        }
    }

    /// Drafts a block behind `first` and verifies it, committing nothing: the
    /// drafts and the target's logits over them.
    ///
    /// The round is left pending on the stack until
    /// [`crate::stack::HybridStack::settle_verified`] says how many drafts the
    /// caller kept; a second verify before that replaces it.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], and whatever the draft and the verify
    /// pass report.
    pub fn draft_and_verify(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
        hint: DraftHint,
        proposal: &[TokenId],
    ) -> CoreResult<shiso_core::model::VerifiedBlock> {
        let live = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let frontier = live.written();
        let depth = live.spec_depth();
        let block = self.block.as_ref().map_or(1, BlockDrafter::block);
        let budget = crate::mtp::draft_budget(
            depth,
            frontier,
            self.opts.ctx,
            crate::stack::MAX_DECODE_BATCH.min(block),
        );
        // The drafter writes its block's keys at the positions it proposes, in
        // pages the target session's table names: claimed before anything
        // writes, for the reason `speculate` gives.
        let cache = Arc::clone(&self.resident()?.cache);
        let room = crate::kv::Room {
            cache: &cache,
            now: self.now(),
        };
        self.sessions
            .get_mut(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .reserve_to(frontier.saturating_add(block), room)?;
        let started = std::time::Instant::now();
        let (drafts, source) = match self.copy(session, first, eos, budget)? {
            Some((tokens, id)) => (tokens, crate::mtp::DraftSource::Lookup(id)),
            None if self.drafting_pays(session) => (
                self.draft_block(session, first, frontier, hint)?,
                crate::mtp::DraftSource::Head,
            ),
            None => (Vec::new(), crate::mtp::DraftSource::Head),
        };
        // The caller's proposal where the drafts disagree with it: it is
        // structure the caller knows comes next, so a block that does not
        // begin with it is wrong from its first mismatch. A block that does is
        // the proposal and more, and is kept.
        let (mut drafts, source) = if proposal.is_empty() || drafts.starts_with(proposal) {
            (drafts, source)
        } else {
            (proposal.to_vec(), crate::mtp::DraftSource::Scaffold)
        };
        drafts.truncate(budget as usize);
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(first);
        rows.extend_from_slice(&drafts);
        let verify_rows = u32::try_from(rows.len()).unwrap_or(1);
        self.run_verify_rows(session, &rows)?;
        self.write_logits_rows(verify_rows)?;
        let vocab = self.geometry.vocab as usize;
        let mut bytes = vec![0u8; rows.len() * vocab * 4];
        self.backend.copy_d2h(
            self.resident()?
                .scratch
                .logits()
                .view()
                .slice(0, bytes.len() as u64)?,
            &mut bytes,
        )?;
        self.pending = Some(crate::stack::PendingBlock {
            session,
            first,
            frontier,
            drafts: drafts.clone(),
            source,
            started,
        });
        Ok(shiso_core::model::VerifiedBlock {
            drafts,
            logits: floats(&bytes),
            vocab,
        })
    }

    /// Commits `first` and the first `accepted` drafts of the round
    /// [`crate::stack::HybridStack::draft_and_verify`] left pending for
    /// `session`, winding the session back past the rest.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] with no round pending for `session`,
    /// [`CoreError::ShapeMismatch`] for more kept drafts than were drafted, and
    /// whatever the rewind and the commit report.
    pub fn settle_verified(&mut self, session: SessionId, accepted: u32) -> CoreResult<()> {
        let pending =
            self.pending
                .take()
                .filter(|p| p.session == session)
                .ok_or(CoreError::Unsupported {
                    what: "a settle with no verified block pending for this session",
                    backend: "engine",
                })?;
        let drafted = u32::try_from(pending.drafts.len()).unwrap_or(u32::MAX);
        if accepted > drafted {
            return Err(CoreError::ShapeMismatch {
                field: "kept drafts",
                expected: u64::from(drafted),
                got: u64::from(accepted),
            });
        }
        let taken = accepted as usize;
        let mut committed = Vec::with_capacity(taken + 1);
        committed.push(pending.first);
        committed.extend_from_slice(&pending.drafts[..taken]);
        let outcome = crate::mtp::StepOutcome {
            committed,
            rollback: crate::mtp::Rollback::new(accepted, drafted),
            logits_row: accepted,
            source: (drafted > 0).then_some(pending.source),
        };
        if let Some(drafter) = self.block.as_mut() {
            drafter.learn(
                session,
                &outcome,
                pending.drafts.len(),
                pending.started.elapsed(),
            );
        }
        if let Some(rewind) = outcome.rollback {
            let live = self
                .sessions
                .get(&session)
                .ok_or(CoreError::UnknownSession(session))?;
            live.rewind_states(&self.backend, rewind)?;
            let keep = pending.frontier.saturating_add(accepted).saturating_add(1);
            self.sessions
                .get_mut(&session)
                .ok_or(CoreError::UnknownSession(session))?
                .rewind_to(keep);
        }
        self.commit_step(session)?;
        // The kept rows are the drafter's context now: `first` and the drafts
        // the caller kept, at the positions the verify pass gave them.
        self.absorb_kept(session, accepted + 1, pending.frontier)?;
        tracing::debug!(
            round_ms = pending.started.elapsed().as_secs_f64() * 1e3,
            drafted,
            accepted,
            "block speculative step"
        );
        Ok(())
    }

    /// One greedy speculative step through the block drafter: the two phases,
    /// with each row's token the target's argmax.
    ///
    /// # Errors
    ///
    /// As [`crate::stack::HybridStack::draft_and_verify`] and
    /// [`crate::stack::HybridStack::settle_verified`].
    pub(crate) fn speculate_block(
        &mut self,
        session: SessionId,
        first: TokenId,
        eos: TokenId,
    ) -> CoreResult<Speculated> {
        let block = self.draft_and_verify(session, first, eos, DraftHint::greedy(), &[])?;
        let picks: Vec<TokenId> = (0..=block.drafts.len())
            .map(|i| crate::sampling::argmax(block.row(i)))
            .collect();
        let accepted = crate::mtp::accepted_prefix(
            &block.drafts,
            picks.get(..block.drafts.len()).unwrap_or(&[]),
            eos,
        )?;
        self.settle_verified(session, accepted)?;
        let mut produced = block.drafts[..accepted as usize].to_vec();
        produced.push(picks[accepted as usize]);
        Ok(Speculated {
            produced,
            accepted,
            drafted: u32::try_from(block.drafts.len()).unwrap_or(u32::MAX),
        })
    }
}
