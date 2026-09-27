//! Running the vision tower.
//!
//! The encoder turns one media item's patch rows into the vectors the language
//! model consumes: patch projection, learned position embedding, a stack of
//! transformer blocks, then the merger that folds a square block of patches into
//! one token. It is the device half of what [`shiso_vision::preprocess_image`] starts.
//!
//! # Nothing here knows how wide the tower is
//!
//! Every width, depth, head count and patch geometry is read from
//! [`VisionDims`], which the loader derived from whichever container was opened.
//! That is not tidiness. The two towers this crate loads differ in all four, and
//! a literal here would be an encoder that runs the wrong arithmetic on the
//! other checkpoint while reporting success - every buffer the right size, every
//! kernel launching, and the answer about a different picture. The tests run the
//! same code at two dimension sets for exactly that reason.
//!
//! # The stream is f32, the weights are not
//!
//! Activations are f32 from the patch rows to the merger's output, and so is
//! every accumulator. The weights stay in the width the checkpoint stored them,
//! which is brain-float, and the kernels widen on read. That is a deliberate
//! departure from the reference this tower was described by, whose hidden stream
//! is brain-float throughout: the residual stream these vectors are scattered
//! into is f32, so narrowing here would only mean rounding twice.
//!
//! # Where the position embedding is resampled
//!
//! On the host, in [`VisionEncoder::resample_positions`]. The learned table is a
//! fixed square grid and an image's patch grid is whatever its aspect ratio
//! produced, so the table is bilinearly resampled onto that grid before it is
//! added. It is a few hundred kilobytes of arithmetic against a tower that is
//! about to run two dozen blocks over the same rows, and doing it on the host
//! keeps the interpolation - the part that is easy to get subtly wrong and
//! invisible afterwards - in code that a plain unit test can read.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use shiso_core::{
    Backend, DType, DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef, LaunchArgs, alloc,
};

use crate::qwen35::vision::dims::VisionDims;
use crate::qwen35::vision::loader::{TowerTensor, VisionWeights};
use crate::qwen35::vision::manifest::{Affine, TowerBlock, TowerMerger};
use shiso_vision::{VisionError, VisionItem, VisionStages};

/// Registry module the tower's kernels live in.
const MODULE: &str = "vision";

/// Square tile the blocked GEMM walks; `VIT_GEMM_TILE` in the kernel set.
const GEMM_TILE: u32 = 16;
/// Threads the elementwise kernels are launched with. They stride by this value
/// internally, so it is the one block size they may be launched with.
const LINEAR_BLOCK: u32 = 256;
/// Threads the normalization is launched with.
const NORM_BLOCK: u32 = 256;
/// Threadgroup bytes the normalization's block reduction needs.
const NORM_SMEM: u32 = 132;
/// Threads the rotation and the attention are launched with.
const ATTN_BLOCK: u32 = 128;
/// Widest head the attention kernel can hold: `VIT_ATTN_SLOTS` accumulator slots
/// of 32 values each. A wider head would be silently truncated, which is the one
/// failure nothing downstream could detect.
const MAX_HEAD_DIM: usize = 256;

/// Default epsilon inside the normalization's square root.
///
/// Both descriptions of this tower state it, neither checkpoint container
/// carries it as a field, and it is not a number a tower would vary without
/// saying so - so it is a named default here rather than a guess made at each
/// call site. [`VisionEncoder::with_layer_norm_eps`] exists for a tower that
/// does say otherwise.
const DEFAULT_LAYER_NORM_EPS: f32 = 1e-6;

/// Default base of the rotation's frequency ladder.
const DEFAULT_ROPE_BASE: f32 = 10_000.0;

/// Runs a loaded vision tower over one media item.
///
/// Holds only the two numbers no container states - the normalization's
/// epsilon and the rotation's base - and whether the device's faster entries
/// may be used. Everything else comes from the weights it is handed, so one
/// encoder value runs any tower the loader accepts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VisionEncoder {
    eps: f32,
    rope_base: f32,
    scalar_only: bool,
}

impl Default for VisionEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Sink for recording intermediate stages during encoding.
pub(crate) trait StageSink {
    fn on_pixel_values(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_patch_embed(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_pos_added(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_block(&mut self, _index: usize, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_pooled(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_normed(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
    fn on_projected(&mut self, _buf: &DeviceBuffer) -> Result<(), VisionError> {
        Ok(())
    }
}

/// A sink that drops all stages.
struct NoopSink;

impl StageSink for NoopSink {}

/// A sink that captures all stages as cloned device buffers.
struct CollectorSink<'a> {
    backend: &'a Arc<dyn Backend>,
    pixel_values: Option<DeviceBuffer>,
    patch_embed: Option<DeviceBuffer>,
    pos_added: Option<DeviceBuffer>,
    blocks: Vec<DeviceBuffer>,
    pooled: Option<DeviceBuffer>,
    normed: Option<DeviceBuffer>,
    projected: Option<DeviceBuffer>,
}

impl<'a> CollectorSink<'a> {
    fn new(backend: &'a Arc<dyn Backend>, depth: usize) -> Self {
        Self {
            backend,
            pixel_values: None,
            patch_embed: None,
            pos_added: None,
            blocks: Vec::with_capacity(depth),
            pooled: None,
            normed: None,
            projected: None,
        }
    }

    fn clone_buf(
        &self,
        step: &'static str,
        buf: &DeviceBuffer,
    ) -> Result<DeviceBuffer, VisionError> {
        let dst = alloc(self.backend, buf.len())
            .map_err(|source| VisionError::Encode { step, source })?;
        self.backend
            .copy_d2d(dst.view(), buf.view())
            .map_err(|source| VisionError::Encode { step, source })?;
        Ok(dst)
    }

    fn finish(self) -> Result<VisionStages, VisionError> {
        let err = |stage: &'static str| VisionError::Unsupported { what: stage };
        Ok(VisionStages {
            pixel_values: self.pixel_values.ok_or_else(|| err("pixel_values"))?,
            patch_embed: self.patch_embed.ok_or_else(|| err("patch_embed"))?,
            pos_added: self.pos_added.ok_or_else(|| err("pos_added"))?,
            blocks: self.blocks,
            pooled: self.pooled.ok_or_else(|| err("pooled"))?,
            normed: self.normed.ok_or_else(|| err("normed"))?,
            projected: self.projected.ok_or_else(|| err("projected"))?,
        })
    }
}

impl StageSink for CollectorSink<'_> {
    fn on_pixel_values(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.pixel_values = Some(self.clone_buf("cloning pixel_values", buf)?);
        Ok(())
    }

    fn on_patch_embed(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.patch_embed = Some(self.clone_buf("cloning patch_embed", buf)?);
        Ok(())
    }

    fn on_pos_added(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.pos_added = Some(self.clone_buf("cloning pos_added", buf)?);
        Ok(())
    }

    fn on_block(&mut self, _index: usize, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.blocks
            .push(self.clone_buf("cloning block output", buf)?);
        Ok(())
    }

    fn on_pooled(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.pooled = Some(self.clone_buf("cloning pooled", buf)?);
        Ok(())
    }

    fn on_normed(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.normed = Some(self.clone_buf("cloning normed", buf)?);
        Ok(())
    }

    fn on_projected(&mut self, buf: &DeviceBuffer) -> Result<(), VisionError> {
        self.projected = Some(self.clone_buf("cloning projected", buf)?);
        Ok(())
    }
}

impl VisionEncoder {
    /// An encoder with the defaults both descriptions of this tower state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            eps: DEFAULT_LAYER_NORM_EPS,
            rope_base: DEFAULT_ROPE_BASE,
            scalar_only: false,
        }
    }

    /// The same encoder, held to the portable single-precision kernels even
    /// where the device has tensor-core ones.
    ///
    /// The faster path rounds a projection's operand and the attention's
    /// operands to brain-float; this is the door a parity case uses to measure
    /// what that rounding moves.
    #[must_use]
    pub const fn with_scalar_kernels(mut self) -> Self {
        self.scalar_only = true;
        self
    }

    /// Overrides the epsilon inside the normalization's square root.
    #[must_use]
    pub const fn with_layer_norm_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Overrides the base of the rotation's frequency ladder.
    #[must_use]
    pub const fn with_rope_base(mut self, base: f32) -> Self {
        self.rope_base = base;
        self
    }

    /// The epsilon the normalization uses.
    #[must_use]
    pub const fn layer_norm_eps(&self) -> f32 {
        self.eps
    }

    /// The base of the rotation's frequency ladder.
    #[must_use]
    pub const fn rope_base(&self) -> f32 {
        self.rope_base
    }

    /// Encodes one media item into `[merged rows, out_hidden]` f32 on the device.
    ///
    /// The returned buffer has exactly [`VisionItem::pad_count`] rows, which is
    /// the same number of placeholder tokens the item occupies in the prompt.
    /// That equality is the contract the prefill scatter leans on, and it holds
    /// by construction here: the merger emits one row per merge block and the
    /// pad count is the merge-block count of the same grid.
    ///
    /// # Errors
    ///
    /// [`VisionError::ItemGridNotMergeable`] when the patch grid is not a whole
    /// number of merge blocks, [`VisionError::ItemRowWidth`] when the item's
    /// rows are not the width this tower's patch projection consumes,
    /// [`VisionError::HeadTooWide`] for a head the attention kernel cannot hold,
    /// [`VisionError::WeightDType`] for a weight stored in a width the kernels
    /// do not read, and [`VisionError::Encode`] carrying the device's own error
    /// for anything that failed on the device, naming the step it failed in.
    pub fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        weights: &VisionWeights,
        item: &VisionItem,
    ) -> Result<DeviceBuffer, VisionError> {
        let dims = weights.dims();
        let plan = Plan::new(dims, item)?;
        let run = Run::new(backend, *self, &plan)?;
        let positions = Self::resample_positions(backend, weights, plan.grid_h, plan.grid_w)?;
        Self::forward(&run, &plan, weights, item, &positions, &mut NoopSink)
    }

    /// Encodes one media item and captures its intermediate stages.
    ///
    /// # Errors
    ///
    /// [`VisionError`] on validation or device failure.
    pub fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        weights: &VisionWeights,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        let dims = weights.dims();
        let plan = Plan::new(dims, item)?;
        let run = Run::new(backend, *self, &plan)?;
        let positions = Self::resample_positions(backend, weights, plan.grid_h, plan.grid_w)?;
        let mut sink = CollectorSink::new(backend, weights.dims().depth());
        let _out = Self::forward(&run, &plan, weights, item, &positions, &mut sink)?;
        sink.finish()
    }

    /// Issues the whole tower, from patch rows to merged output.
    ///
    /// Split out from [`VisionEncoder::encode`] so the argument checks and the
    /// host-side resample stay separable from the launch sequence; nothing here
    /// validates, and nothing above here launches.
    fn forward<S: StageSink>(
        run: &Run<'_>,
        plan: &Plan,
        weights: &VisionWeights,
        item: &VisionItem,
        positions: &[f32],
        sink: &mut S,
    ) -> Result<DeviceBuffer, VisionError> {
        let tree = weights.tensors();
        let mut buffers = Buffers::new(run, plan)?;
        let pixels = run.upload("patch rows", item.as_flat())?;
        sink.on_pixel_values(&pixels)?;
        let position_rows = run.upload("resampled positions", positions)?;

        // Patch projection, then the resampled position embedding. The period is
        // one temporal group's worth of rows, so a clip's frames each receive the
        // same grid without the host replicating it.
        run.gemm(
            "patch projection",
            &buffers.state,
            &pixels,
            &tree.patch_embed,
            Shape::new(plan.rows, plan.hidden, plan.patch_dim),
        )?;
        sink.on_patch_embed(&buffers.state)?;
        run.add(
            "position embedding",
            &buffers.state,
            &position_rows,
            plan.rows * plan.hidden,
            plan.plane * plan.hidden,
        )?;
        sink.on_pos_added(&buffers.state)?;

        for (i, block) in tree.blocks.iter().enumerate() {
            Self::block(run, plan, block, &mut buffers)?;
            sink.on_block(i, &buffers.state)?;
        }
        let out = Self::merge(run, plan, &tree.merger, &buffers, sink)?;
        run.sync("encode")?;
        Ok(out)
    }

    /// One transformer block, in place on the running state.
    ///
    /// Both halves have the same shape: normalize, do the work, add what came in
    /// back on. The state and the scratch buffer swap after each half rather than
    /// the residual being copied, so a block costs no copy of the hidden stream.
    fn block(
        run: &Run<'_>,
        plan: &Plan,
        block: &TowerBlock<TowerTensor>,
        buffers: &mut Buffers,
    ) -> Result<(), VisionError> {
        if run.staged_operand().is_some() {
            return Self::block_fused(run, plan, block, buffers);
        }
        let hidden_rows = plan.rows * plan.hidden;
        let rows = plan.rows;

        run.layer_norm(
            "norm1",
            &buffers.scratch,
            &buffers.state,
            &block.norm1,
            rows,
            plan.hidden,
        )?;
        run.gemm(
            "fused projection",
            &buffers.qkv,
            &buffers.scratch,
            &block.qkv,
            Shape::new(rows, 3 * plan.hidden, plan.hidden),
        )?;
        run.rope(plan, &buffers.q, &buffers.k, &buffers.v, &buffers.qkv)?;
        run.attention(plan, &buffers.attn, &buffers.q, &buffers.k, &buffers.v)?;
        run.gemm(
            "attention output",
            &buffers.scratch,
            &buffers.attn,
            &block.proj,
            Shape::new(rows, plan.hidden, plan.hidden),
        )?;
        run.add(
            "attention residual",
            &buffers.scratch,
            &buffers.state,
            hidden_rows,
            hidden_rows,
        )?;
        buffers.advance();

        run.layer_norm(
            "norm2",
            &buffers.scratch,
            &buffers.state,
            &block.norm2,
            rows,
            plan.hidden,
        )?;
        run.gemm(
            "feed-forward in",
            &buffers.feed_forward,
            &buffers.scratch,
            &block.fc1,
            Shape::new(rows, plan.intermediate, plan.hidden),
        )?;
        run.unary(
            "block activation",
            "kernel_vit_gelu_tanh",
            &buffers.feed_forward,
            rows * plan.intermediate,
        )?;
        run.gemm(
            "feed-forward out",
            &buffers.scratch,
            &buffers.feed_forward,
            &block.fc2,
            Shape::new(rows, plan.hidden, plan.intermediate),
        )?;
        run.add(
            "feed-forward residual",
            &buffers.scratch,
            &buffers.state,
            hidden_rows,
            hidden_rows,
        )?;
        buffers.advance();
        Ok(())
    }

    /// [`Self::block`] with the projections in the device's BLAS and every
    /// elementwise pass that sits between two of them folded into its
    /// neighbour: each normalization writes the next projection's operand
    /// directly, the first feed-forward layer's bias and activation write the
    /// second's, and each residual add carries the projection's bias.
    ///
    /// The arithmetic is the unfused sequence's, value for value; what is gone
    /// is six passes a block over the hidden stream or the feed-forward rows.
    fn block_fused(
        run: &Run<'_>,
        plan: &Plan,
        block: &TowerBlock<TowerTensor>,
        buffers: &mut Buffers,
    ) -> Result<(), VisionError> {
        let rows = plan.rows;
        run.norm_split("norm1", &buffers.state, &block.norm1, rows, plan.hidden)?;
        let fused = Shape::new(rows, 3 * plan.hidden, plan.hidden);
        run.multiply("fused projection", &buffers.qkv, &block.qkv, fused)?;
        if run.fast.rope_split {
            run.rope_split(plan, &buffers.q, &buffers.qkv, &block.qkv)?;
            run.attention_staged(plan, &buffers.attn, &buffers.q)?;
        } else {
            run.add_bias("fused projection", &buffers.qkv, &block.qkv, fused)?;
            run.rope(plan, &buffers.q, &buffers.k, &buffers.v, &buffers.qkv)?;
            run.attention(plan, &buffers.attn, &buffers.q, &buffers.k, &buffers.v)?;
        }
        let out = Shape::new(rows, plan.hidden, plan.hidden);
        run.stage("attention output", &buffers.attn, rows * plan.hidden)?;
        run.multiply("attention output", &buffers.scratch, &block.proj, out)?;
        run.bias_residual(
            "attention residual",
            &buffers.scratch,
            &block.proj,
            &buffers.state,
            out,
        )?;
        buffers.advance();

        run.norm_split("norm2", &buffers.state, &block.norm2, rows, plan.hidden)?;
        let inner = Shape::new(rows, plan.intermediate, plan.hidden);
        run.multiply("feed-forward in", &buffers.feed_forward, &block.fc1, inner)?;
        run.bias_gelu_split("block activation", &buffers.feed_forward, &block.fc1, inner)?;
        let back = Shape::new(rows, plan.hidden, plan.intermediate);
        run.multiply("feed-forward out", &buffers.scratch, &block.fc2, back)?;
        run.bias_residual(
            "feed-forward residual",
            &buffers.scratch,
            &block.fc2,
            &buffers.state,
            back,
        )?;
        buffers.advance();
        Ok(())
    }

    /// The merger: normalize each patch row, fold the square block, then two
    /// layers with the exact activation between them.
    ///
    /// The normalization runs BEFORE the fold and on one hidden width, not on the
    /// folded width. That order is not a choice made here: the merger's scale and
    /// offset are one hidden width long, which leaves no other reading.
    fn merge<S: StageSink>(
        run: &Run<'_>,
        plan: &Plan,
        merger: &TowerMerger<TowerTensor>,
        buffers: &Buffers,
        sink: &mut S,
    ) -> Result<DeviceBuffer, VisionError> {
        run.layer_norm(
            "merger norm",
            &buffers.scratch,
            &buffers.state,
            &merger.norm,
            plan.rows,
            plan.hidden,
        )?;
        let folded = run.scratch("merged rows", plan.merged_rows * plan.merge_input)?;
        run.spatial_merge(plan, &folded, &buffers.scratch)?;
        sink.on_normed(&folded)?;
        let projected = run.scratch("merger inner", plan.merged_rows * plan.merge_input)?;
        run.gemm(
            "merger in",
            &projected,
            &folded,
            &merger.fc1,
            Shape::new(plan.merged_rows, plan.merge_input, plan.merge_input),
        )?;
        // The merger's activation is the exact one, not the approximation the
        // blocks use. The two differ by about 1e-4 per value and the layers
        // genuinely declare different activations; this is not a place to reuse.
        run.unary(
            "merger activation",
            "kernel_vit_gelu_erf",
            &projected,
            plan.merged_rows * plan.merge_input,
        )?;
        sink.on_pooled(&projected)?;
        let out = run.scratch("encoder output", plan.merged_rows * plan.out_hidden)?;
        run.gemm(
            "merger out",
            &out,
            &projected,
            &merger.fc2,
            Shape::new(plan.merged_rows, plan.out_hidden, plan.merge_input),
        )?;
        sink.on_projected(&out)?;
        Ok(out)
    }

    /// Bilinearly resamples the learned position table onto this item's grid.
    ///
    /// The table is a fixed square of learned vectors; an image's patch grid is
    /// whatever its aspect ratio produced. The sample points are spread so the
    /// first and last patch land exactly on the first and last row of the table,
    /// a grid of one patch samples the corner rather than dividing by zero, and
    /// the four surrounding table rows are mixed by the usual bilinear weights.
    ///
    /// Returns one temporal group's worth of rows, `grid_h * grid_w` of them, in
    /// the same grid-major order the patch rows arrive in. A clip's later frames
    /// reuse those rows rather than getting their own.
    ///
    /// # Errors
    ///
    /// [`VisionError::WeightDType`] when the table is not stored in the width
    /// the widening below reads, and [`VisionError::Encode`] when reading it
    /// back from the device fails.
    fn resample_positions(
        backend: &Arc<dyn Backend>,
        weights: &VisionWeights,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Vec<f32>, VisionError> {
        let dims = weights.dims();
        let hidden = dims.hidden();
        let side = dims.position_grid();
        let table = read_bf16(backend, &weights.tensors().pos_embed)?;

        let mut out = vec![0.0f32; grid_h * grid_w * hidden];
        let span = widen(side - 1);
        for gh in 0..grid_h {
            let (y_low, y_high, dy) = sample(gh, grid_h, span, side);
            for gw in 0..grid_w {
                let (x_low, x_high, dx) = sample(gw, grid_w, span, side);
                let w00 = (1.0 - dy) * (1.0 - dx);
                let w01 = (1.0 - dy) * dx;
                let w10 = dy * (1.0 - dx);
                let w11 = dy * dx;
                let i00 = (y_low * side + x_low) * hidden;
                let i01 = (y_low * side + x_high) * hidden;
                let i10 = (y_high * side + x_low) * hidden;
                let i11 = (y_high * side + x_high) * hidden;
                let base = (gh * grid_w + gw) * hidden;
                for d in 0..hidden {
                    out[base + d] = w00 * table[i00 + d]
                        + w01 * table[i01 + d]
                        + w10 * table[i10 + d]
                        + w11 * table[i11 + d];
                }
            }
        }
        Ok(out)
    }
}

/// The scratch the tower runs in, allocated once per item.
///
/// Every buffer here is written in full before it is read, so none is zeroed;
/// see [`Run::scratch`]. The two hidden-width buffers swap after each residual
/// add rather than one being copied into the other, which is why they are a pair
/// with a name for the swap instead of two independent locals.
struct Buffers {
    state: DeviceBuffer,
    scratch: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn: DeviceBuffer,
    feed_forward: DeviceBuffer,
}

impl Buffers {
    fn new(run: &Run<'_>, plan: &Plan) -> Result<Self, VisionError> {
        let hidden_rows = plan.rows * plan.hidden;
        Ok(Self {
            state: run.scratch("hidden state", hidden_rows)?,
            scratch: run.scratch("hidden scratch", hidden_rows)?,
            qkv: run.scratch("fused projection", plan.rows * 3 * plan.hidden)?,
            q: run.scratch("query plane", hidden_rows)?,
            k: run.scratch("key plane", hidden_rows)?,
            v: run.scratch("value plane", hidden_rows)?,
            attn: run.scratch("attention output", hidden_rows)?,
            feed_forward: run.scratch("feed-forward rows", plan.rows * plan.intermediate)?,
        })
    }

    /// Makes the buffer a residual was just written into the running state.
    fn advance(&mut self) {
        std::mem::swap(&mut self.state, &mut self.scratch);
    }
}

/// The two table rows a grid position falls between, and how far between them.
///
/// `extent` of one is the case that has no interval at all: it samples the first
/// row, which is what keeps a one-patch grid from dividing by zero.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the floor of a sample position is non-negative and bounded by the table side, which the min below clamps to regardless"
)]
fn sample(index: usize, extent: usize, span: f32, side: usize) -> (usize, usize, f32) {
    let position = if extent <= 1 {
        0.0
    } else {
        widen(index) * span / widen(extent - 1)
    };
    let low = position.floor();
    let low_index = (low as usize).min(side - 1);
    let high_index = (low_index + 1).min(side - 1);
    (low_index, high_index, position - low)
}

/// The tensor-core attention's argument block.
#[expect(clippy::cast_precision_loss, reason = "head_dim is a small dimension")]
fn mma_args(plan: &Plan, split: &SplitPlanes) -> AttentionMmaArgs {
    AttentionMmaArgs {
        rows: narrow(plan.rows),
        heads: narrow(plan.heads),
        head_dim: narrow(plan.head_dim),
        plane: narrow(plan.plane),
        plane_pad: narrow(split.plane_pad),
        scale: 1.0 / (plan.head_dim as f32).sqrt(),
    }
}

/// Widens a brain-float tensor off the device into host floats.
fn read_bf16(backend: &Arc<dyn Backend>, tensor: &TowerTensor) -> Result<Vec<f32>, VisionError> {
    if tensor.dtype() != DType::BF16 {
        return Err(VisionError::WeightDType {
            name: tensor.name().to_owned(),
            dtype: tensor.dtype(),
        });
    }
    let len = usize::try_from(tensor.buffer().len()).unwrap_or(usize::MAX);
    let mut bytes = vec![0u8; len];
    backend
        .copy_d2h(tensor.view(), &mut bytes)
        .map_err(|source| VisionError::Encode {
            step: "position table read-back",
            source,
        })?;
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect())
}

/// Every shape one encode needs, checked once.
///
/// Built before any allocation, so an item whose grid does not match the tower
/// fails before a single byte reaches the device rather than halfway through the
/// block loop with a buffer already in flight.
#[derive(Debug, Clone, Copy)]
struct Plan {
    rows: usize,
    plane: usize,
    grid_h: usize,
    grid_w: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
    out_hidden: usize,
    patch_dim: usize,
    merge: usize,
    merge_input: usize,
    merged_rows: usize,
}

impl Plan {
    fn new(dims: VisionDims, item: &VisionItem) -> Result<Self, VisionError> {
        let (grid_t, grid_h, grid_w) = item.grid();
        let merge = dims.geometry().spatial_merge_size();
        if !grid_h.is_multiple_of(merge) || !grid_w.is_multiple_of(merge) {
            return Err(VisionError::ItemGridNotMergeable {
                grid_h,
                grid_w,
                merge,
            });
        }
        if item.row_len() != dims.patch_dim() {
            return Err(VisionError::ItemRowWidth {
                expected: dims.patch_dim(),
                got: item.row_len(),
            });
        }
        let head_dim = dims.head_dim();
        // Four channels per rotation pair per axis: the head splits in half for
        // the pairing and each half splits again between the row and the column
        // axis, so an odd quarter would leave a channel with no partner.
        if head_dim > MAX_HEAD_DIM || !head_dim.is_multiple_of(4) {
            return Err(VisionError::HeadTooWide {
                head_dim,
                max: MAX_HEAD_DIM,
            });
        }
        let plane = grid_h * grid_w;
        Ok(Self {
            rows: grid_t * plane,
            plane,
            grid_h,
            grid_w,
            hidden: dims.hidden(),
            heads: dims.heads(),
            head_dim,
            intermediate: dims.intermediate(),
            out_hidden: dims.out_hidden(),
            patch_dim: dims.patch_dim(),
            merge,
            merge_input: dims.merge_input(),
            merged_rows: grid_t * plane / (merge * merge),
        })
    }
}

/// The three widths one projection is issued with.
///
/// A named triple rather than three positional arguments: `rows`, the output
/// width and the input width are all plain counts, and a call site that swapped
/// the last two would produce a launch that is the right size and the wrong
/// multiply.
#[derive(Debug, Clone, Copy)]
struct Shape {
    rows: usize,
    out_width: usize,
    in_width: usize,
}

impl Shape {
    const fn new(rows: usize, out_width: usize, in_width: usize) -> Self {
        Self {
            rows,
            out_width,
            in_width,
        }
    }
}

/// Which of the device's faster entries an encode may use.
///
/// Probed from the module's own entry list rather than from the backend's
/// name: a build that compiled the tensor-core entries has them, one that did
/// not (Metal, or a build with no kernels) runs the scalar set, and nothing
/// above the backend has to know which device it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fast {
    /// `kernel_vit_attention_mma`, for a head no wider than it holds.
    attention: bool,
    /// The device's BLAS for the projections, with the two kernels that
    /// round its operand and add its bias.
    blas: bool,
    /// `kernel_vit_qkv_rope_split`, which folds the fused projection's bias,
    /// the rotation and the key/value split into one pass.
    rope_split: bool,
}

impl Fast {
    /// The scalar set everywhere.
    pub(crate) const SCALAR: Self = Self {
        attention: false,
        blas: false,
        rope_split: false,
    };

    /// What `backend` offers, for a tower whose head is `head_dim` wide.
    fn probe(backend: &Arc<dyn Backend>, head_dim: usize) -> Self {
        let entries = backend.module_entries(MODULE);
        let has = |name: &str| entries.iter().any(|entry| entry == name);
        let attention = has(ATTENTION_MMA)
            && has(SPLIT_KV)
            && head_dim <= MMA_HEAD_DIM
            && head_dim.is_multiple_of(2);
        Self {
            attention,
            rope_split: attention && has(ROPE_SPLIT),
            blas: backend.as_blas().is_some()
                && [
                    TO_BF16,
                    ADD_BIAS_BF16,
                    NORM_SPLIT,
                    BIAS_GELU_SPLIT,
                    BIAS_RESIDUAL,
                ]
                .iter()
                .all(|name| has(name)),
        }
    }
}

/// The tensor-core attention's entry; see `kernels/cuda/vision/vit_fast.cuh`.
const ATTENTION_MMA: &str = "kernel_vit_attention_mma";
/// The operand rounding a BLAS projection reads.
const TO_BF16: &str = "kernel_vit_to_bf16";
/// A BLAS projection's bias epilogue.
const ADD_BIAS_BF16: &str = "kernel_vit_add_bias_bf16";
/// The normalization written as the next projection's split operand.
const NORM_SPLIT: &str = "kernel_vit_layer_norm_split";
/// The first feed-forward layer's bias and activation, as the second's operand.
const BIAS_GELU_SPLIT: &str = "kernel_vit_bias_gelu_split";
/// A projection's bias and the block's residual in one pass.
const BIAS_RESIDUAL: &str = "kernel_vit_add_bias_residual";
/// The fused projection's bias, the rotation and the key/value split.
const ROPE_SPLIT: &str = "kernel_vit_qkv_rope_split";
/// Widest head the tensor-core attention holds: `VIT_FA_HD`.
const MMA_HEAD_DIM: usize = 80;
/// Queries one tensor-core attention block owns: `VIT_FA_BQ`.
const MMA_QUERIES: usize = 128;
/// Keys one tensor-core attention tile stages: `VIT_FA_BK`.
const MMA_KEYS: usize = 64;
/// The split into brain-float pairs the tensor-core attention reads.
const SPLIT_KV: &str = "kernel_vit_split_kv";
/// Threads of one tensor-core attention block: eight warps.
const MMA_THREADS: u32 = 256;
/// Dynamic shared memory of one tensor-core attention block: `VIT_FA_SMEM`,
/// two staged tiles of brain-float key and value pairs. Past the 48 KiB
/// default, which the backend raises for the entry on its first launch.
const MMA_SMEM: u32 = 2 * 2 * 2 * (64 * (80 + 8) + 80 * (64 + 8));

/// One encode's device handle, the two constants no container states, and
/// which faster path it runs.
struct Run<'a> {
    backend: &'a Arc<dyn Backend>,
    eps: f32,
    rope_base: f32,
    fast: Fast,
    /// A projection's operand as two brain-float planes, the nearest value and
    /// what it left over, sized for the widest one of this encode; `None`
    /// unless the projections run in the BLAS.
    operand: Option<DeviceBuffer>,
    /// The key and value planes split into brain-float pairs, in the layout
    /// the tensor-core attention stages them from; `None` unless it runs.
    split: Option<SplitPlanes>,
}

/// `kernel_vit_split_kv`'s two outputs.
struct SplitPlanes {
    keys: DeviceBuffer,
    values: DeviceBuffer,
    /// A temporal group's length rounded up to whole key tiles.
    plane_pad: usize,
}

impl<'a> Run<'a> {
    /// The run for one encode of `plan` on `backend`.
    fn new(
        backend: &'a Arc<dyn Backend>,
        encoder: VisionEncoder,
        plan: &Plan,
    ) -> Result<Self, VisionError> {
        let fast = if encoder.scalar_only {
            Fast::SCALAR
        } else {
            Fast::probe(backend, plan.head_dim)
        };
        // The widest operand any projection reads: the patch rows, the
        // hidden stream, the feed-forward rows, or the merger's folded rows,
        // which are the hidden stream regrouped.
        let widest = plan.rows * plan.patch_dim.max(plan.hidden).max(plan.intermediate);
        let operand = if fast.blas {
            Some(
                alloc(backend, u64::try_from(widest).unwrap_or(u64::MAX) * 4).map_err(
                    |source| VisionError::Encode {
                        step: "brain-float operand",
                        source,
                    },
                )?,
            )
        } else {
            None
        };
        let split = if fast.attention {
            let plane_pad = plan.plane.div_ceil(MMA_KEYS) * MMA_KEYS;
            let groups = plan.rows / plan.plane.max(1);
            // Two halves of every head's padded keys, each MMA_HEAD_DIM wide,
            // two bytes an element.
            let bytes = u64::try_from(2 * plan.heads * groups * plane_pad * MMA_HEAD_DIM * 2)
                .unwrap_or(u64::MAX);
            // Cleared once: `kernel_vit_qkv_rope_split` writes only real key
            // rows, and a padding row must hold zeros for the attention's
            // masked keys to stay finite.
            let plane = |step| {
                let buffer =
                    alloc(backend, bytes).map_err(|source| VisionError::Encode { step, source })?;
                backend
                    .fill_zero(buffer.view())
                    .map_err(|source| VisionError::Encode { step, source })?;
                Ok::<_, VisionError>(buffer)
            };
            Some(SplitPlanes {
                keys: plane("split key plane")?,
                values: plane("split value plane")?,
                plane_pad,
            })
        } else {
            None
        };
        Ok(Self {
            backend,
            eps: encoder.eps,
            rope_base: encoder.rope_base,
            fast,
            operand,
            split,
        })
    }
}

impl Run<'_> {
    /// Allocates an f32 buffer of `len` values.
    ///
    /// Deliberately not zeroed. Every buffer this hands out is written in full
    /// before it is read: the GEMM's grid covers its whole output and only
    /// guards the threads that fall outside it, and the normalization, the
    /// rotation, the attention and the merge each write every element of their
    /// own output. A zeroing pass would be a copy of a few hundred megabytes per
    /// image to overwrite values nothing reads - but it is also the thing that
    /// would hide a kernel whose grid did not cover its output, so if one is ever
    /// added here it should be added with a test that says which kernel needed it.
    fn scratch(&self, step: &'static str, len: usize) -> Result<DeviceBuffer, VisionError> {
        alloc(self.backend, u64::try_from(len).unwrap_or(u64::MAX) * 4)
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// Allocates and fills a buffer with host floats.
    fn upload(&self, step: &'static str, values: &[f32]) -> Result<DeviceBuffer, VisionError> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = alloc(self.backend, bytes.len() as u64)
            .map_err(|source| VisionError::Encode { step, source })?;
        self.backend
            .copy_h2d(buffer.view(), &bytes)
            .map_err(|source| VisionError::Encode { step, source })?;
        Ok(buffer)
    }

    /// Launches one kernel, attributing a failure to the step it happened in.
    fn launch(
        &self,
        step: &'static str,
        entry: &'static str,
        grid: Extent3,
        block: Extent3,
        smem_bytes: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), VisionError> {
        self.backend
            .run(
                KernelRef::new(MODULE, entry),
                &LaunchArgs {
                    grid,
                    block,
                    smem_bytes,
                    args,
                },
            )
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// Drains the queue, which is what makes the output buffer meaningful.
    fn sync(&self, step: &'static str) -> Result<(), VisionError> {
        self.backend
            .sync()
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// `out[m, n] = in[m, k] . weight[n, k] + bias[n]`.
    fn gemm(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        shape: Shape,
    ) -> Result<(), VisionError> {
        if self.staged_operand().is_some() {
            self.stage(step, input, shape.rows * shape.in_width)?;
            self.multiply(step, out, layer, shape)?;
            return self.add_bias(step, out, layer, shape);
        }
        let weight = bf16_view(&layer.weight)?;
        let bias = bf16_view(&layer.bias)?;
        let args = GemmArgs {
            m: narrow(shape.rows),
            n: narrow(shape.out_width),
            k: narrow(shape.in_width),
        };
        self.launch(
            step,
            "kernel_vit_gemm_bias",
            Extent3::new(
                grid_for(shape.out_width, GEMM_TILE),
                grid_for(shape.rows, GEMM_TILE),
                1,
            ),
            Extent3::new(GEMM_TILE, GEMM_TILE, 1),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
                KernelArg::Buffer(weight),
                KernelArg::Buffer(bias),
            ],
        )
    }

    /// The split-operand staging, when the projections run in the device's
    /// BLAS; `None` runs them on the scalar GEMM.
    fn staged_operand(&self) -> Option<&DeviceBuffer> {
        self.operand
            .as_ref()
            .filter(|_| self.fast.blas && self.backend.as_blas().is_some())
    }

    /// The operand planes as a view of `reads` elements each, hi then lo.
    fn operand_planes(
        &self,
        step: &'static str,
        reads: usize,
    ) -> Result<DeviceView<'_>, VisionError> {
        let operand = self.staged_operand().ok_or(VisionError::Encode {
            step,
            source: shiso_core::CoreError::Unsupported {
                what: "a split operand on a run with no BLAS projections",
                backend: "vision",
            },
        })?;
        let bytes = u64::try_from(reads).unwrap_or(u64::MAX).saturating_mul(4);
        operand
            .slice(0, bytes)
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// Splits `input` into the operand planes the next [`Self::multiply`] reads.
    fn stage(
        &self,
        step: &'static str,
        input: &DeviceBuffer,
        reads: usize,
    ) -> Result<(), VisionError> {
        let planes = self.operand_planes(step, reads)?;
        self.launch(
            step,
            TO_BF16,
            Extent3::linear(grid_for(reads, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&UnaryArgs {
                    count: narrow(reads),
                })),
                KernelArg::Buffer(planes),
                KernelArg::Buffer(input.view()),
            ],
        )
    }

    /// `out[m, n] = operand[m, k] . weight[n, k]`, from the staged planes: one
    /// brain-float multiply per plane, accumulated in single precision.
    ///
    /// Two multiplies rather than one because a single rounding of the operand
    /// moved this tower's output by a percent of its largest value; the weights
    /// are brain-float as stored, so the pair loses nothing on either side.
    fn multiply(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        shape: Shape,
    ) -> Result<(), VisionError> {
        let fail = |source| VisionError::Encode { step, source };
        let blas = self.backend.as_blas().ok_or(VisionError::Encode {
            step,
            source: shiso_core::CoreError::Unsupported {
                what: "a BLAS projection on a device with no BLAS",
                backend: "vision",
            },
        })?;
        let reads = shape.rows * shape.in_width;
        let planes = self.operand_planes(step, reads)?;
        let plane = u64::try_from(reads).unwrap_or(u64::MAX) * 2;
        let weight = bf16_view(&layer.weight)?;
        let (m, n, k) = (
            narrow(shape.out_width),
            narrow(shape.rows),
            narrow(shape.in_width),
        );
        for (half, beta) in [(0u64, 0.0f32), (1, 1.0)] {
            blas.gemm(&shiso_core::GemmDesc {
                op_a: shiso_core::GemmOp::Transposed,
                op_b: shiso_core::GemmOp::AsStored,
                m,
                n,
                k,
                lda: k,
                ldb: k,
                ldc: m,
                a: weight,
                b: planes.slice(half * plane, plane).map_err(fail)?,
                c: out.view(),
                alpha: 1.0,
                // The second plane adds onto the first's product.
                beta,
                dtype: shiso_core::GemmDType::BF16,
                reduction: shiso_core::GemmReduction::Fastest,
            })
            .map_err(fail)?;
        }
        Ok(())
    }

    /// A BLAS projection's bias, in place.
    fn add_bias(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        shape: Shape,
    ) -> Result<(), VisionError> {
        let writes = shape.rows * shape.out_width;
        self.launch(
            step,
            ADD_BIAS_BF16,
            Extent3::linear(grid_for(writes, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&AddArgs {
                    count: narrow(writes),
                    period: narrow(shape.out_width),
                })),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(bf16_view(&layer.bias)?),
            ],
        )
    }

    /// [`Self::layer_norm`] written straight into the operand planes of the
    /// projection that reads it.
    fn norm_split(
        &self,
        step: &'static str,
        input: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        rows: usize,
        dim: usize,
    ) -> Result<(), VisionError> {
        let planes = self.operand_planes(step, rows * dim)?;
        let args = LayerNormArgs {
            rows: narrow(rows),
            dim: narrow(dim),
            eps: self.eps,
        };
        self.launch(
            step,
            NORM_SPLIT,
            Extent3::linear(narrow(rows)),
            Extent3::linear(NORM_BLOCK),
            NORM_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(planes),
                KernelArg::Buffer(input.view()),
                KernelArg::Buffer(bf16_view(&layer.weight)?),
                KernelArg::Buffer(bf16_view(&layer.bias)?),
            ],
        )
    }

    /// The first feed-forward layer's bias and activation, written as the
    /// second layer's operand planes.
    fn bias_gelu_split(
        &self,
        step: &'static str,
        x: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        shape: Shape,
    ) -> Result<(), VisionError> {
        let count = shape.rows * shape.out_width;
        let planes = self.operand_planes(step, count)?;
        self.launch(
            step,
            BIAS_GELU_SPLIT,
            Extent3::linear(grid_for(count, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&AddArgs {
                    count: narrow(count),
                    period: narrow(shape.out_width),
                })),
                KernelArg::Buffer(planes),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(bf16_view(&layer.bias)?),
            ],
        )
    }

    /// A BLAS projection's bias and the block's residual, in place on `x`.
    fn bias_residual(
        &self,
        step: &'static str,
        x: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        residual: &DeviceBuffer,
        shape: Shape,
    ) -> Result<(), VisionError> {
        let count = shape.rows * shape.out_width;
        self.launch(
            step,
            BIAS_RESIDUAL,
            Extent3::linear(grid_for(count, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&AddArgs {
                    count: narrow(count),
                    period: narrow(shape.out_width),
                })),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(bf16_view(&layer.bias)?),
                KernelArg::Buffer(residual.view()),
            ],
        )
    }

    /// Mean/variance normalization of every row, with a scale and an offset.
    fn layer_norm(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
        rows: usize,
        dim: usize,
    ) -> Result<(), VisionError> {
        let weight = bf16_view(&layer.weight)?;
        let bias = bf16_view(&layer.bias)?;
        let args = LayerNormArgs {
            rows: narrow(rows),
            dim: narrow(dim),
            eps: self.eps,
        };
        self.launch(
            step,
            "kernel_vit_layer_norm",
            Extent3::linear(narrow(rows)),
            Extent3::linear(NORM_BLOCK),
            NORM_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
                KernelArg::Buffer(weight),
                KernelArg::Buffer(bias),
            ],
        )
    }

    /// One elementwise activation, in place.
    fn unary(
        &self,
        step: &'static str,
        entry: &'static str,
        x: &DeviceBuffer,
        count: usize,
    ) -> Result<(), VisionError> {
        let args = UnaryArgs {
            count: narrow(count),
        };
        self.launch(
            step,
            entry,
            Extent3::linear(grid_for(count, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
            ],
        )
    }

    /// `x[i] += add[i % period]`, in place.
    fn add(
        &self,
        step: &'static str,
        x: &DeviceBuffer,
        addend: &DeviceBuffer,
        count: usize,
        period: usize,
    ) -> Result<(), VisionError> {
        let args = AddArgs {
            count: narrow(count),
            period: narrow(period),
        };
        self.launch(
            step,
            "kernel_vit_add_inplace",
            Extent3::linear(grid_for(count, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(addend.view()),
            ],
        )
    }

    /// Splits the fused projection into three planes and rotates two of them.
    fn rope(
        &self,
        plan: &Plan,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
        qkv: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        let args = RopeArgs {
            rows: narrow(plan.rows),
            heads: narrow(plan.heads),
            head_dim: narrow(plan.head_dim),
            grid_w: narrow(plan.grid_w),
            plane: narrow(plan.plane),
            rope_base: self.rope_base,
        };
        self.launch(
            "rotation",
            "kernel_vit_qkv_rope",
            Extent3::new(narrow(plan.heads), narrow(plan.rows), 1),
            Extent3::linear(ATTN_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(q.view()),
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
                KernelArg::Buffer(qkv.view()),
            ],
        )
    }

    /// Bidirectional attention within each temporal group.
    fn attention(
        &self,
        plan: &Plan,
        out: &DeviceBuffer,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        #[expect(clippy::cast_precision_loss, reason = "head_dim is a small dimension")]
        let args = AttentionArgs {
            rows: narrow(plan.rows),
            heads: narrow(plan.heads),
            head_dim: narrow(plan.head_dim),
            plane: narrow(plan.plane),
            // The head's inverse root width, which is the value the kernel used
            // to derive itself. This tower scales by it; the kernel now takes it
            // as an argument so a tower that does not (gemma 4) can pass one.
            scale: 1.0 / (plan.head_dim as f32).sqrt(),
        };
        let planes = [
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
        ];
        if let Some(split) = self.split.as_ref() {
            let groups = plan.rows / plan.plane.max(1);
            let mma = mma_args(plan, split);
            self.launch(
                "attention operand split",
                SPLIT_KV,
                Extent3::linear(narrow(groups * split.plane_pad)),
                Extent3::linear(LINEAR_BLOCK),
                0,
                &[
                    KernelArg::Bytes(bytemuck::bytes_of(&mma)),
                    KernelArg::Buffer(split.keys.view()),
                    KernelArg::Buffer(split.values.view()),
                    KernelArg::Buffer(k.view()),
                    KernelArg::Buffer(v.view()),
                ],
            )?;
            return self.attention_staged(plan, out, q);
        }
        // The query row, one accumulator per simdgroup, then the per-simdgroup
        // softmax maximum and denominator. Same formula as `KERNELS.md`.
        let warps = ATTN_BLOCK / 32;
        let smem = 4 * (narrow(plan.head_dim) * (warps + 1) + 2 * warps);
        self.launch(
            "attention",
            "kernel_vit_attention",
            Extent3::new(narrow(plan.heads), narrow(plan.rows), 1),
            Extent3::linear(ATTN_BLOCK),
            smem,
            &planes,
        )
    }

    /// The split planes, or the refusal a run without them owes `step`.
    fn split_planes(&self, step: &'static str) -> Result<&SplitPlanes, VisionError> {
        self.split.as_ref().ok_or(VisionError::Encode {
            step,
            source: shiso_core::CoreError::Unsupported {
                what: "a tensor-core attention step on a run with no split planes",
                backend: "vision",
            },
        })
    }

    /// The tensor-core attention over queries `q` and the split planes as they
    /// stand, however they were written.
    fn attention_staged(
        &self,
        plan: &Plan,
        out: &DeviceBuffer,
        q: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        let split = self.split_planes("attention")?;
        let groups = plan.rows / plan.plane.max(1);
        let tiles = plan.plane.div_ceil(MMA_QUERIES);
        let mma = mma_args(plan, split);
        self.launch(
            "attention",
            ATTENTION_MMA,
            Extent3::new(narrow(plan.heads), narrow(groups * tiles), 1),
            Extent3::linear(MMA_THREADS),
            MMA_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&mma)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(q.view()),
                KernelArg::Buffer(split.keys.view()),
                KernelArg::Buffer(split.values.view()),
            ],
        )
    }

    /// The fused projection's bias, the rotation, and the key/value planes
    /// written straight into the split planes; `q` comes out rotated.
    fn rope_split(
        &self,
        plan: &Plan,
        q: &DeviceBuffer,
        qkv: &DeviceBuffer,
        layer: &Affine<TowerTensor>,
    ) -> Result<(), VisionError> {
        let split = self.split_planes("rotation")?;
        let args = RopeSplitArgs {
            rows: narrow(plan.rows),
            heads: narrow(plan.heads),
            head_dim: narrow(plan.head_dim),
            grid_w: narrow(plan.grid_w),
            plane: narrow(plan.plane),
            plane_pad: narrow(split.plane_pad),
            rope_base: self.rope_base,
        };
        self.launch(
            "rotation",
            ROPE_SPLIT,
            Extent3::new(narrow(plan.heads), narrow(plan.rows), 1),
            Extent3::linear(ATTN_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(q.view()),
                KernelArg::Buffer(split.keys.view()),
                KernelArg::Buffer(split.values.view()),
                KernelArg::Buffer(qkv.view()),
                KernelArg::Buffer(bf16_view(&layer.bias)?),
            ],
        )
    }

    /// Folds each square block of patch rows into one merged row.
    fn spatial_merge(
        &self,
        plan: &Plan,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        let args = MergeArgs {
            merged_rows: narrow(plan.merged_rows),
            hidden: narrow(plan.hidden),
            grid_h: narrow(plan.grid_h),
            grid_w: narrow(plan.grid_w),
            merge: narrow(plan.merge),
        };
        self.launch(
            "spatial merge",
            "kernel_vit_spatial_merge",
            Extent3::linear(narrow(plan.merged_rows)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
            ],
        )
    }
}

/// A tower weight as the kernels read it, or a named refusal.
///
/// The kernels widen a brain-float by shifting its bits, which is exact and
/// cheap but is not a conversion: handed a tensor stored some other way they
/// would read the wrong numbers and say nothing. The loader accepts three
/// storage widths, so this is where the encoder's narrower requirement is
/// stated, per tensor, with the tensor's own name on it.
fn bf16_view(tensor: &TowerTensor) -> Result<DeviceView<'_>, VisionError> {
    if tensor.dtype() != DType::BF16 {
        return Err(VisionError::WeightDType {
            name: tensor.name().to_owned(),
            dtype: tensor.dtype(),
        });
    }
    Ok(tensor.view())
}

/// A grid extent in the float domain the sample arithmetic works in.
///
/// Exact rather than lossy: grids and position tables are bounded by the
/// image-size cap long before this runs, which keeps every value far inside the
/// range a float represents integers exactly in.
#[expect(
    clippy::cast_precision_loss,
    reason = "every value that reaches this is a grid extent bounded by the image-size cap"
)]
fn widen(value: usize) -> f32 {
    value as f32
}

/// Threadgroups needed to cover `count` units at `per_block` each.
fn grid_for(count: usize, per_block: u32) -> u32 {
    let per = per_block as usize;
    narrow(count.div_ceil(per))
}

/// Narrows a plan dimension for a kernel argument.
///
/// Saturating rather than fallible: every value that reaches it is bounded by
/// the image-size cap and the tower's own dimensions long before this point, so
/// the saturating arm is unreachable, and a `Result` here would be a second
/// error path nothing can produce.
fn narrow(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

// -- argument blocks, mirroring `metal/vision/vit_common.h` ------------------

/// `vit_args_gemm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GemmArgs {
    m: u32,
    n: u32,
    k: u32,
}

/// `vit_args_layer_norm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LayerNormArgs {
    rows: u32,
    dim: u32,
    eps: f32,
}

/// `vit_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

/// `vit_args_add`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    count: u32,
    period: u32,
}

/// `vit_args_qkv_rope`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    grid_w: u32,
    plane: u32,
    rope_base: f32,
}

/// `vit_args_qkv_rope_split`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RopeSplitArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    grid_w: u32,
    plane: u32,
    plane_pad: u32,
    rope_base: f32,
}

/// `vit_args_attention`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    scale: f32,
}

/// `vit_args_attention_mma`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionMmaArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    plane_pad: u32,
    scale: f32,
}

/// `vit_args_merge`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    merged_rows: u32,
    hidden: u32,
    grid_h: u32,
    grid_w: u32,
    merge: u32,
}

/// The layouts the kernels read. A field added on one side and not the other
/// would bind silently, so the widths are asserted rather than trusted.
const _: () = {
    assert!(size_of::<GemmArgs>() == 12);
    assert!(size_of::<LayerNormArgs>() == 12);
    assert!(size_of::<UnaryArgs>() == 4);
    assert!(size_of::<AddArgs>() == 8);
    assert!(size_of::<RopeArgs>() == 24);
    assert!(size_of::<AttentionArgs>() == 20);
    assert!(size_of::<MergeArgs>() == 20);
};

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
