//! Running the Gemma 4 vision tower.
//!
//! One media item's patch rows go in; the projected rows the language model
//! consumes come out. Between them: the patch GEMM, a learned position add,
//! sixteen transformer blocks, a pooling fold, and the projection into the
//! language model's width. Every width comes from [`Gemma4Dims`], which the
//! loader derived from the container — nothing in this file knows how wide
//! the shipped tower is, for the reason the Qwen encoder states: a literal
//! here is an encoder that runs the right arithmetic on the wrong checkpoint
//! and reports success.
//!
//! # The kernels are borrowed from three sets
//!
//! The GEMM, the attention and the residual add are the vision set's, which
//! the Qwen tower also drives; the score scale rides in the attention's
//! argument block now, so this tower passes the `1.0` its reference uses where
//! Qwen passes the inverse root of its head width. The normalization is the
//! language model's — the generic `rms_norm_mul`, with the fused template's
//! unread second-operand slot bound the same way the engine binds it. The
//! rotation, the pooling fold and the clamp are the dense set's new tower
//! kernels, and the gated GELU is its gate unit. A tower that owned its
//! kernels would be a tower whose arithmetic drifts from the reference's;
//! sharing the reductions is the point.
//!
//! # Where the position embedding is built
//!
//! On the host, in [`position_rows`]. The table is two flat grids of rows —
//! one indexed by patch column, one by patch row — and an item's grid is
//! whatever its aspect ratio produced. Summing `tbl_x[col] + tbl_y[row]` for
//! a few hundred thousand values is nothing against the sixteen blocks that
//! follow, and keeping the gather in plain Rust keeps the one easy-to-get-
//! silently-wrong indexing decision in code a unit test can read.
//!
//! # The clamps are data, not graph
//!
//! The reference wraps every linear in two clamps read from the container,
//! defaulting to the float extremes. The shipped checkpoint stores the
//! extremes everywhere — the clamp is the identity on this file — but the
//! kernel exists because a future quantization pass may store bounds that
//! bite, and an identity that is *checked* is a fact; one that is skipped
//! because it is assumed is a hope. So each projection asks its [`Clamps`]
//! which of its two bounds can move a value, and launches the kernel only for
//! the ones that can.

#![expect(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "vision encoder executes complex multi-step tensor graphs with dimensional math"
)]

use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use shiso_core::{
    Backend, DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs, VisionGrid, alloc,
};
use shiso_vision::{
    EncodedRows, MediaPlacement, PreprocessSpec, VisionError, VisionItem, VisionStages, VisionTower,
};

use super::loader::{Clamps, Gemma4Block, Gemma4Linear, Gemma4VisionWeights, load_gemma4_weights};

/// Registry module the shared vision kernels live in.
const MODULE_VISION: &str = "vision";
/// Registry module the tower's own dense kernels live in.
const MODULE_DENSE: &str = "dense";
/// Registry module the language model's normalization lives in.
const MODULE_GENERIC: &str = "generic";

/// Threads the elementwise kernels are launched with. They stride by this
/// value internally, so it is the one block size they may be launched with.
const LINEAR_BLOCK: u32 = 256;
/// Threads the rotation and the attention are launched with.
const ATTN_BLOCK: u32 = 128;
/// Threads the generic normalization is launched with, and the threadgroup
/// bytes its block reduction needs — the engine's `WARP` and
/// `NORM_SMEM_BYTES`, restated because both are crate-private there.
const NORM_BLOCK: u32 = 32;
/// Threadgroup bytes the normalization's block reduction needs.
const NORM_SMEM: u32 = NORM_BLOCK * 4;
/// Floats the dense set's block reduction keeps its scratch in; the fused
/// rotation's threadgroup request is this plus one float per plane element,
/// `4 * (BLOCK_SUM_SLOTS + 3 * head_dim)` bytes, as the kernel documents.
const BLOCK_SUM_SLOTS: u32 = 33;
/// Widest head the attention kernel can hold: `VIT_ATTN_SLOTS` accumulator
/// slots of 32 values each, the same bound the Qwen tower checks.
const MAX_HEAD_DIM: usize = 256;

/// Side of the patch block the pooling fold consumes.
const POOL: usize = 3;

/// Narrows a plan dimension for a kernel argument.
///
/// Saturating rather than fallible: every value that reaches it is bounded by
/// the image-size cap and the tower's own dimensions long before this point,
/// so the saturating arm is unreachable, and a `Result` here would be a
/// second error path nothing can produce.
fn narrow(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn grid_for(total: usize, block: u32) -> u32 {
    let block = usize::try_from(block).unwrap_or(usize::MAX);
    narrow(total.div_ceil(block))
}

// -- argument blocks, mirroring the kernel headers ---------------------------

/// `vit_args_add`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    count: u32,
    period: u32,
}

/// `vit_args_attention`. The scale is the field this tower's family added.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttentionArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    plane: u32,
    scale: f32,
}

/// `dense_args_vit_rope2d`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Rope2dArgs {
    rows: u32,
    heads: u32,
    head_dim: u32,
    grid_w: u32,
    eps: f32,
    rope_base: f32,
}

/// `dense_args_vit_pool`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PoolArgs {
    grid_h: u32,
    grid_w: u32,
    hidden: u32,
    scale: f32,
}

/// `dense_args_clamp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ClampArgs {
    count: u32,
    min: f32,
    max: f32,
}

/// The gate unit's argument block, mirroring `shiso_engine::launch::GluArgs`.
///
/// The padding is spelled out because `bytemuck::Pod` refuses a struct with
/// implicit padding, and the declared layout leaves four bytes before each
/// 64-bit stride.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GluArgs {
    ne00: i32,
    pad0: [u8; 4],
    nb01: u64,
    ne10: i32,
    pad1: [u8; 4],
    nb11: u64,
    ne0: i32,
    pad2: [u8; 4],
    nb1: u64,
    i00: i32,
    i10: i32,
    alpha: f32,
    limit: f32,
}

/// The generic normalization's argument block, mirroring
/// `shiso_engine::launch::NormArgs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}

/// The layouts the kernels read. A field added on one side and not the other
/// would bind silently, so the widths are asserted rather than trusted.
const _: () = {
    assert!(size_of::<AddArgs>() == 8);
    assert!(size_of::<AttentionArgs>() == 20);
    assert!(size_of::<Rope2dArgs>() == 24);
    assert!(size_of::<PoolArgs>() == 16);
    assert!(size_of::<ClampArgs>() == 12);
    assert!(size_of::<GluArgs>() == 64);
    assert!(size_of::<NormArgs>() == 144);
};

/// A Gemma 4 vision tower: the loaded weights, the preprocessing contract
/// this family publishes, and the widths both sides read.
#[derive(Debug)]
pub struct Gemma4Tower {
    weights: Gemma4VisionWeights,
    spec: PreprocessSpec,
    /// Side of the patch block the fold merges; the pad grid is divided by it.
    merge: usize,
    /// Width of the rows the language model consumes.
    out_hidden: u32,
}

impl Gemma4Tower {
    /// How this family wants a picture marked in the prompt.
    pub const PLACEMENT: MediaPlacement = MediaPlacement::GEMMA4;

    /// Opens the mmproj at `path` as this family's tower.
    /// # Errors
    ///
    /// Returns [`VisionError`] if the tower weights or metadata cannot be loaded.
    pub fn open(backend: &Arc<dyn Backend>, path: &Path) -> Result<Self, VisionError> {
        let weights = load_gemma4_weights(backend, path)?;
        let dims = weights.dims();
        let spec = PreprocessSpec::gemma4();
        // The container's patch geometry and the family's preprocessing
        // contract are two readings of one fact; if they disagree the tower
        // would preprocess for one width and project for another, so the
        // disagreement ends the open rather than starting a confused encode.
        if spec.geometry.patch_dim() != dims.patch_dim {
            return Err(VisionError::ItemRowWidth {
                expected: dims.patch_dim,
                got: spec.geometry.patch_dim(),
            });
        }
        Ok(Self {
            merge: spec.geometry.spatial_merge_size(),
            out_hidden: narrow(dims.out_hidden),
            weights,
            spec,
        })
    }

    /// Runs the whole graph over one item and returns its intermediate stages.
    ///
    /// # Errors
    ///
    /// [`VisionError`] if dimensions mismatch or device operations fail.
    pub fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        encode_item_staged(backend, &self.weights, item)
    }
}

impl VisionTower for Gemma4Tower {
    fn preprocess(&self) -> &PreprocessSpec {
        &self.spec
    }

    fn placement(&self) -> MediaPlacement {
        Self::PLACEMENT
    }

    fn out_hidden(&self) -> u32 {
        self.out_hidden
    }

    fn device_bytes(&self) -> u64 {
        self.weights.device_bytes()
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError> {
        let row_bytes = u64::from(self.out_hidden) * 4;
        let mut encoded: Vec<DeviceBuffer> = Vec::with_capacity(items.len());
        let mut grids = Vec::with_capacity(items.len());
        for item in items {
            let buffer = encode_item(backend, &self.weights, item)?;
            let (t_len, gh, gw) = item.grid();
            grids.push(VisionGrid::new(
                narrow(t_len),
                narrow(gh / self.merge),
                narrow(gw / self.merge),
            ));
            encoded.push(buffer);
        }
        EncodedRows::gather(backend, &encoded, grids, row_bytes)
    }

    fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        self.encode_stages(backend, item)
    }
}

/// The per-item widths every launch counts in.
#[derive(Debug, Clone, Copy)]
struct Plan {
    /// Patch rows: `grid_h * grid_w`, one temporal group.
    rows: usize,
    grid_h: usize,
    grid_w: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
    out_hidden: usize,
    eps: f32,
    rope_base: f32,
}

/// One item's activation stream, named for its role in the graph.
///
/// The buffers are reused where the graph makes the earlier contents dead —
/// the query plane becomes the attention output projection's destination, the
/// up plane the down projection's — and nowhere else, so the aliasing a
/// reader has to check is exactly the two the block's order proves.
struct Streams {
    x: DeviceBuffer,
    h: DeviceBuffer,
    c1: DeviceBuffer,
    c2: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn: DeviceBuffer,
    post: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mid: DeviceBuffer,
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
            pixel_values: self
                .pixel_values
                .ok_or_else(|| err("missing pixel_values stage"))?,
            patch_embed: self
                .patch_embed
                .ok_or_else(|| err("missing patch_embed stage"))?,
            pos_added: self
                .pos_added
                .ok_or_else(|| err("missing pos_added stage"))?,
            blocks: self.blocks,
            pooled: self.pooled.ok_or_else(|| err("missing pooled stage"))?,
            normed: self.normed.ok_or_else(|| err("missing normed stage"))?,
            projected: self
                .projected
                .ok_or_else(|| err("missing projected stage"))?,
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

/// Runs the whole graph over one item and returns its projected rows.
fn encode_item(
    backend: &Arc<dyn Backend>,
    weights: &Gemma4VisionWeights,
    item: &VisionItem,
) -> Result<DeviceBuffer, VisionError> {
    encode_item_with_sink(backend, weights, item, &mut NoopSink)
}

/// Runs the whole graph over one item and captures intermediate stages.
fn encode_item_staged(
    backend: &Arc<dyn Backend>,
    weights: &Gemma4VisionWeights,
    item: &VisionItem,
) -> Result<VisionStages, VisionError> {
    let mut sink = CollectorSink::new(backend, weights.blocks.len());
    let _out = encode_item_with_sink(backend, weights, item, &mut sink)?;
    sink.finish()
}

/// The single graph implementation, parametrized by a stage sink.
fn encode_item_with_sink<S: StageSink>(
    backend: &Arc<dyn Backend>,
    weights: &Gemma4VisionWeights,
    item: &VisionItem,
    sink: &mut S,
) -> Result<DeviceBuffer, VisionError> {
    let dims = weights.dims();
    let (t_len, grid_h, grid_w) = item.grid();
    if t_len != 1 {
        return Err(VisionError::Unsupported {
            what: "the gemma 4 vision tower reads stills; a temporal group is not one",
        });
    }
    if grid_h % POOL != 0 || grid_w % POOL != 0 {
        return Err(VisionError::ItemGridNotMergeable {
            grid_h,
            grid_w,
            merge: POOL,
        });
    }
    if item.row_len() != dims.patch_dim {
        return Err(VisionError::ItemRowWidth {
            expected: dims.patch_dim,
            got: item.row_len(),
        });
    }
    if dims.head_dim > MAX_HEAD_DIM || !dims.head_dim.is_multiple_of(4) {
        return Err(VisionError::HeadTooWide {
            head_dim: dims.head_dim,
            max: MAX_HEAD_DIM,
        });
    }
    let rows = grid_h * grid_w;
    let plan = Plan {
        rows,
        grid_h,
        grid_w,
        hidden: dims.hidden,
        heads: dims.heads,
        head_dim: dims.head_dim,
        intermediate: dims.intermediate,
        out_hidden: dims.out_hidden,
        eps: dims.eps,
        rope_base: dims.rope_base,
    };
    let n = rows * plan.hidden;
    let ni = rows * plan.intermediate;
    let streams = Streams {
        x: zeros(backend, n)?,
        h: zeros(backend, n)?,
        c1: zeros(backend, n)?,
        c2: zeros(backend, ni)?,
        q: zeros(backend, n)?,
        k: zeros(backend, n)?,
        v: zeros(backend, n)?,
        attn: zeros(backend, n)?,
        post: zeros(backend, n)?,
        gate: zeros(backend, ni)?,
        up: zeros(backend, ni)?,
        mid: zeros(backend, ni)?,
    };

    // -- the patch projection and the position add --------------------------
    let patch_in = upload(backend, "patch rows", item.as_flat())?;
    sink.on_pixel_values(&patch_in)?;
    gemm(
        backend,
        "patch projection",
        &streams.x,
        &patch_in,
        &weights.patch,
        crate::audio_ops::GemmType::Bf16,
        rows,
        plan.hidden,
        dims.patch_dim,
    )?;
    sink.on_patch_embed(&streams.x)?;
    let pos = upload(
        backend,
        "position rows",
        &position_rows(weights, grid_h, grid_w),
    )?;
    add_inplace(backend, "position add", &streams.x, &pos, n, n)?;
    sink.on_pos_added(&streams.x)?;
    for (i, block) in weights.blocks.iter().enumerate() {
        encode_block(backend, &plan, block, &streams)?;
        sink.on_block(i, &streams.x)?;
    }

    // -- the fold, the unweighted norm, the projection ----------------------
    let tokens = rows / (POOL * POOL);
    let pooled = zeros(backend, tokens * plan.hidden)?;
    let normed = zeros(backend, tokens * plan.hidden)?;
    let out = zeros(backend, tokens * plan.out_hidden)?;
    avgpool(
        backend,
        &pooled,
        &streams.x,
        &plan,
        (plan.hidden as f32).sqrt(),
    )?;
    sink.on_pooled(&pooled)?;
    rms_norm_mul(
        backend,
        "pool norm",
        &normed,
        &pooled,
        &weights.ones,
        plan.hidden,
        tokens,
        plan.eps,
    )?;
    sink.on_normed(&normed)?;
    gemm(
        backend,
        "projection",
        &out,
        &normed,
        &weights.projection.weight,
        weights.projection.dtype,
        tokens,
        plan.out_hidden,
        plan.hidden,
    )?;
    clamp_out(backend, &out, &weights.projection.clamps)?;
    sink.on_projected(&out)?;

    backend.sync().map_err(|source| VisionError::Encode {
        step: "draining the encode",
        source,
    })?;
    Ok(out)
}

/// An f32 device buffer of `count` values.
fn zeros(backend: &Arc<dyn Backend>, count: usize) -> Result<DeviceBuffer, VisionError> {
    let step = "allocating the encode buffers";
    alloc(
        backend,
        u64::try_from(count).unwrap_or(u64::MAX).saturating_mul(4),
    )
    .map_err(|source| VisionError::Encode { step, source })
}

/// One transformer block, in the order the reference builds it.
///
/// The normalized input `h` survives the whole attention arm, which is why
/// the query, key, value and output projections can share one clamped copy
/// `c1`; after the first residual add it is rebuilt for the feed-forward from
/// the same buffer. The query plane dies in the attention and the up plane in
/// the gate unit, and each hosts the next projection's destination.
fn encode_block(
    backend: &Arc<dyn Backend>,
    plan: &Plan,
    block: &Gemma4Block,
    s: &Streams,
) -> Result<(), VisionError> {
    let rows = plan.rows;
    let n = rows * plan.hidden;
    let h = plan.hidden;

    rms_norm_mul(
        backend,
        "attn norm",
        &s.h,
        &s.x,
        &block.ln1,
        h,
        rows,
        plan.eps,
    )?;
    projected(backend, plan, &block.attn_q, &s.h, &s.c1, &s.q, h, h)?;
    projected(backend, plan, &block.attn_k, &s.h, &s.c1, &s.k, h, h)?;
    projected(backend, plan, &block.attn_v, &s.h, &s.c1, &s.v, h, h)?;
    rope2d(
        backend,
        &s.q,
        &s.k,
        &s.v,
        &block.q_norm,
        &block.k_norm,
        plan,
    )?;
    attention(backend, &s.attn, &s.q, &s.k, &s.v, plan)?;
    projected(backend, plan, &block.attn_out, &s.attn, &s.c1, &s.q, h, h)?;
    rms_norm_mul(
        backend,
        "attn post norm",
        &s.post,
        &s.q,
        &block.attn_post_norm,
        h,
        rows,
        plan.eps,
    )?;
    add_inplace(backend, "attn residual", &s.x, &s.post, n, n)?;

    rms_norm_mul(
        backend, "ffn norm", &s.h, &s.x, &block.ln2, h, rows, plan.eps,
    )?;
    projected(
        backend,
        plan,
        &block.ffn_gate,
        &s.h,
        &s.c1,
        &s.gate,
        plan.intermediate,
        h,
    )?;
    projected(
        backend,
        plan,
        &block.ffn_up,
        &s.h,
        &s.c1,
        &s.up,
        plan.intermediate,
        h,
    )?;
    geglu(backend, &s.mid, &s.gate, &s.up, rows * plan.intermediate)?;
    projected(
        backend,
        plan,
        &block.ffn_down,
        &s.mid,
        &s.c2,
        &s.up,
        h,
        plan.intermediate,
    )?;
    rms_norm_mul(
        backend,
        "ffn post norm",
        &s.post,
        &s.up,
        &block.ffn_post_norm,
        h,
        rows,
        plan.eps,
    )?;
    add_inplace(backend, "ffn residual", &s.x, &s.post, n, n)?;
    Ok(())
}

/// One clamped linear projection: input clamp over the source, the GEMM, and
/// the output clamp in place.
///
/// The input clamp writes a copy because the normalized source is read again
/// by the next projection; the output clamp runs in place because the kernel
/// documents each element as read and written once by the same thread.
fn projected(
    backend: &Arc<dyn Backend>,
    plan: &Plan,
    layer: &Gemma4Linear,
    input: &DeviceBuffer,
    copy: &DeviceBuffer,
    out: &DeviceBuffer,
    width: usize,
    in_width: usize,
) -> Result<(), VisionError> {
    let src = if layer.clamps.input_is_identity() {
        input
    } else {
        clamp(
            backend,
            "projection input clamp",
            copy,
            input,
            layer.clamps.input_min,
            layer.clamps.input_max,
        )?;
        copy
    };
    gemm(
        backend,
        "projection",
        out,
        src,
        &layer.weight,
        layer.dtype,
        plan.rows,
        width,
        in_width,
    )?;
    if layer.clamps.output_is_identity() {
        return Ok(());
    }
    clamp(
        backend,
        "projection output clamp",
        out,
        out,
        layer.clamps.output_min,
        layer.clamps.output_max,
    )
}

/// `out[m, n] = in[m, k] . weight[n, k]`: f32 activations against the
/// tower's weights - brain-float, or `Q8_0` blocks as a quantized projector
/// stores them - on the shared encoder set's tiled GEMM.
///
/// That GEMM tiles on the matrix units and the vision set's own blocked one
/// does not; on a 640x480 picture (2394 patch rows through sixteen blocks)
/// the projections are most of the tower's time. This tower has no biases,
/// so the product alone is the result.
fn gemm(
    backend: &Arc<dyn Backend>,
    step: &'static str,
    out: &DeviceBuffer,
    input: &DeviceBuffer,
    weight: &DeviceBuffer,
    dtype: crate::audio_ops::GemmType,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), VisionError> {
    let ops = crate::audio_ops::Ops::new(Arc::clone(backend));
    let bytes = |floats: usize| u64::try_from(floats * 4).unwrap_or(u64::MAX);
    let encode = |source| VisionError::Encode { step, source };
    ops.gemm(
        dtype,
        out.slice(0, bytes(m * n)).map_err(encode)?,
        input.slice(0, bytes(m * k)).map_err(encode)?,
        weight.view(),
        None,
        (m, n, k),
    )
    .map_err(encode)
}

/// `x[i] += addend[i % period]`, in place.
fn add_inplace(
    backend: &Arc<dyn Backend>,
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
    launch(
        backend,
        step,
        MODULE_VISION,
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

/// Bidirectional attention over the item's single temporal group, at the
/// scale this family's reference uses: one. The score is the bare dot.
///
/// On the shared encoder set's tiled span attention whenever the head is a
/// whole number of 32-lane slices, as every shipped tower's 64-wide head is:
/// over a 640x480 picture's 2394 rows the vision set's one-row-per-group
/// kernel was most of the tower's time. Any other width keeps that kernel.
fn attention(
    backend: &Arc<dyn Backend>,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    plan: &Plan,
) -> Result<(), VisionError> {
    if !plan.head_dim.is_multiple_of(32) || plan.head_dim > 512 {
        return attention_scalar(backend, out, q, k, v, plan);
    }
    if plan.head_dim == FLASH_HEAD_DIM
        && backend
            .max_block_threads(KernelRef::new(FLASH_MODULE, FLASH_ENTRY))
            .is_some()
    {
        return attention_flash(backend, out, q, k, v, plan);
    }
    let ops = crate::audio_ops::Ops::new(Arc::clone(backend));
    ops.span_attention(&crate::audio_ops::SpanAttnDesc {
        out: out.view(),
        q: q.view(),
        k: k.view(),
        v: v.view(),
        segments: None,
        n: plan.rows,
        heads: plan.heads,
        kv_heads: plan.heads,
        head_dim: plan.head_dim,
        left: plan.rows,
        right: plan.rows,
        scale: 1.0,
    })
    .map_err(|source| VisionError::Encode {
        step: "attention",
        source,
    })
}

/// The 64-row flash kernel's module, entry and head width.
const FLASH_MODULE: &str = "diffusion";
const FLASH_ENTRY: &str = "kernel_dit_flash_sdpa_unit_d64_f32";
const FLASH_HEAD_DIM: usize = 64;
/// Query rows and threads of one flash threadgroup.
const FLASH_QROWS: u32 = 64;
const FLASH_THREADS: u32 = 256;

/// The flash kernel's arguments: a one-section sequence, no cache.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FlashArgs {
    txt_tokens: u32,
    img_tokens: u32,
    cache_tokens: u32,
    heads: u32,
    head_dim: u32,
    txt_stride: u32,
    img_stride: u32,
    cache_stride: u32,
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
}
const _: () = assert!(size_of::<FlashArgs>() == 44);

/// [`attention`] on the image transformer's flash kernel where the device
/// has it: 64 query rows a threadgroup share every key and value block,
/// staged as half with f32 sums, where the span kernel's tile of 32 rows
/// read them straight from memory. Measured on an M4 over a 640x480
/// picture's 2394 rows, 12 heads of 64: see the commit that added it.
fn attention_flash(
    backend: &Arc<dyn Backend>,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    plan: &Plan,
) -> Result<(), VisionError> {
    let width = narrow(plan.heads * plan.head_dim);
    let args = FlashArgs {
        txt_tokens: 0,
        img_tokens: narrow(plan.rows),
        cache_tokens: 0,
        heads: narrow(plan.heads),
        head_dim: narrow(plan.head_dim),
        txt_stride: width,
        img_stride: width,
        cache_stride: width,
        q_offset: 0,
        k_offset: 0,
        v_offset: 0,
    };
    // The text-section and cache slots are unread with no rows in them;
    // any live buffer binds them.
    launch(
        backend,
        "attention",
        FLASH_MODULE,
        FLASH_ENTRY,
        Extent3::new(grid_for(plan.rows, FLASH_QROWS), narrow(plan.heads), 1),
        Extent3::linear(FLASH_THREADS),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
        ],
    )
}

/// [`attention`] on the vision set's own kernel: one row per threadgroup,
/// any head width. The fallback for a head the tiled kernel does not take.
fn attention_scalar(
    backend: &Arc<dyn Backend>,
    out: &DeviceBuffer,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    plan: &Plan,
) -> Result<(), VisionError> {
    let args = AttentionArgs {
        rows: narrow(plan.rows),
        heads: narrow(plan.heads),
        head_dim: narrow(plan.head_dim),
        plane: narrow(plan.rows),
        scale: 1.0,
    };
    let warps = ATTN_BLOCK / 32;
    let smem = 4 * (narrow(plan.head_dim) * (warps + 1) + 2 * warps);
    launch(
        backend,
        "attention",
        MODULE_VISION,
        "kernel_vit_attention",
        Extent3::new(narrow(plan.heads), narrow(plan.rows), 1),
        Extent3::linear(ATTN_BLOCK),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
        ],
    )
}

/// The fused per-head normalization and two-axis rotation, in place over the
/// three planes. The value plane is normalized too, and not rotated.
fn rope2d(
    backend: &Arc<dyn Backend>,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    q_norm: &DeviceBuffer,
    k_norm: &DeviceBuffer,
    plan: &Plan,
) -> Result<(), VisionError> {
    let args = Rope2dArgs {
        rows: narrow(plan.rows),
        heads: narrow(plan.heads),
        head_dim: narrow(plan.head_dim),
        grid_w: narrow(plan.grid_w),
        eps: plan.eps,
        rope_base: plan.rope_base,
    };
    let smem = 4 * (BLOCK_SUM_SLOTS + 3 * narrow(plan.head_dim));
    launch(
        backend,
        "rotation",
        MODULE_DENSE,
        "kernel_dense_vit_qkv_norm_rope2d",
        Extent3::new(narrow(plan.heads), narrow(plan.rows), 1),
        Extent3::linear(ATTN_BLOCK),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(q.view()),
            KernelArg::Buffer(k.view()),
            KernelArg::Buffer(v.view()),
            KernelArg::Buffer(q_norm.view()),
            KernelArg::Buffer(k_norm.view()),
        ],
    )
}

/// Folds every `3 x 3` block of patch rows into one row and scales by
/// `sqrt(hidden)`, the fold the reference's pooling does with its scale.
fn avgpool(
    backend: &Arc<dyn Backend>,
    out: &DeviceBuffer,
    input: &DeviceBuffer,
    plan: &Plan,
    scale: f32,
) -> Result<(), VisionError> {
    let args = PoolArgs {
        grid_h: narrow(plan.grid_h),
        grid_w: narrow(plan.grid_w),
        hidden: narrow(plan.hidden),
        scale,
    };
    let out_len = usize::try_from(out.len() / 4).unwrap_or(usize::MAX);
    launch(
        backend,
        "pool",
        MODULE_DENSE,
        "kernel_dense_vit_avgpool3_scale",
        Extent3::linear(grid_for(out_len, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(input.view()),
        ],
    )
}

/// The generic weighted RMS normalization: `dst[row] = rmsnorm(src[row]) * w`.
///
/// The fourth binding is the fused template's unread second operand, bound to
/// the weight exactly as the engine's own wrapper binds it — the slot exists
/// whatever depth the template was built at, and leaving it empty would send
/// the destination where the unread operand belongs.
fn rms_norm_mul(
    backend: &Arc<dyn Backend>,
    step: &'static str,
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    weight: &DeviceBuffer,
    hidden: usize,
    rows: usize,
    eps: f32,
) -> Result<(), VisionError> {
    let row = u64::try_from(hidden).unwrap_or(u64::MAX) * 4;
    let planes = row * u64::try_from(rows).unwrap_or(u64::MAX);
    let args = NormArgs {
        ne00: narrow(hidden) as i32,
        ne00_t: (narrow(hidden) / 4) as i32,
        nb1: row,
        nb2: planes,
        nb3: planes,
        eps,
        nef1: [1, 1, 1],
        nef2: [1, 1, 1],
        nef3: [1, 1, 1],
        nbf1: [row, 0, 0],
        nbf2: [planes, 0, 0],
        nbf3: [planes, 0, 0],
    };
    launch(
        backend,
        step,
        MODULE_GENERIC,
        "kernel_rms_norm_mul_f32_4",
        Extent3::new(narrow(rows), 1, 1),
        Extent3::linear(NORM_BLOCK),
        NORM_SMEM,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(src.view()),
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(weight.view()),
            KernelArg::Buffer(dst.view()),
        ],
    )
}

/// `dst[i] = gelu_tanh(gate[i]) * up[i]`, the dense gate unit at full width,
/// unclamped and unscaled: this tower's bounds are the clamps' business, and
/// the reference's gate applies no tanh limit of its own.
fn geglu(
    backend: &Arc<dyn Backend>,
    dst: &DeviceBuffer,
    gate: &DeviceBuffer,
    up: &DeviceBuffer,
    count: usize,
) -> Result<(), VisionError> {
    let row = u64::try_from(count).unwrap_or(u64::MAX) * 4;
    let args = GluArgs {
        ne00: narrow(count) as i32,
        pad0: [0; 4],
        nb01: row,
        ne10: narrow(count) as i32,
        pad1: [0; 4],
        nb11: row,
        ne0: narrow(count) as i32,
        pad2: [0; 4],
        nb1: row,
        i00: 0,
        i10: 0,
        alpha: 1.0,
        limit: 0.0,
    };
    launch(
        backend,
        "gate unit",
        MODULE_DENSE,
        "kernel_dense_geglu_flat_f32",
        Extent3::linear(grid_for(count, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(gate.view()),
            KernelArg::Buffer(up.view()),
            KernelArg::Buffer(dst.view()),
        ],
    )
}

/// One clamp launch, out-of-place or aliased; the kernel documents both as
/// the same single read and single write per element.
fn clamp(
    backend: &Arc<dyn Backend>,
    step: &'static str,
    dst: &DeviceBuffer,
    src: &DeviceBuffer,
    min: f32,
    max: f32,
) -> Result<(), VisionError> {
    let count = usize::try_from(dst.len() / 4).unwrap_or(usize::MAX);
    let args = ClampArgs {
        count: narrow(count),
        min,
        max,
    };
    launch(
        backend,
        step,
        MODULE_DENSE,
        "kernel_dense_clamp_f32",
        Extent3::linear(grid_for(count, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(dst.view()),
            KernelArg::Buffer(src.view()),
        ],
    )
}

/// Clamps a finished projection at its output bounds, in place, when the
/// bounds can bite at all.
fn clamp_out(
    backend: &Arc<dyn Backend>,
    x: &DeviceBuffer,
    clamps: &Clamps,
) -> Result<(), VisionError> {
    if clamps.output_is_identity() {
        return Ok(());
    }
    clamp(
        backend,
        "projection output clamp",
        x,
        x,
        clamps.output_min,
        clamps.output_max,
    )
}

/// Launches one kernel against one named registry module.
fn launch(
    backend: &Arc<dyn Backend>,
    step: &'static str,
    module: &'static str,
    entry: &'static str,
    grid: Extent3,
    block: Extent3,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) -> Result<(), VisionError> {
    backend
        .run(
            KernelRef::new(module, entry),
            &LaunchArgs {
                grid,
                block,
                smem_bytes,
                args,
            },
        )
        .map_err(|source| VisionError::Encode { step, source })
}

/// Uploads one host buffer, attributing a failure to its step.
fn upload(
    backend: &Arc<dyn Backend>,
    step: &'static str,
    values: &[f32],
) -> Result<DeviceBuffer, VisionError> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = alloc(backend, u64::try_from(bytes.len()).unwrap_or(u64::MAX))
        .map_err(|source| VisionError::Encode { step, source })?;
    backend
        .copy_h2d(buffer.view(), &bytes)
        .map_err(|source| VisionError::Encode { step, source })?;
    Ok(buffer)
}

/// The position rows for one item: `tbl_x[col] + tbl_y[row]` per patch, in
/// the raster order the preprocessor packed the rows.
fn position_rows(weights: &Gemma4VisionWeights, grid_h: usize, grid_w: usize) -> Vec<f32> {
    let hidden = weights.dims().hidden;
    let (tbl_x, tbl_y) = (&weights.position_x, &weights.position_y);
    let mut rows = vec![0.0f32; grid_h * grid_w * hidden];
    for r in 0..grid_h {
        for c in 0..grid_w {
            let at = (r * grid_w + c) * hidden;
            for d in 0..hidden {
                rows[at + d] = tbl_x[c * hidden + d] + tbl_y[r * hidden + d];
            }
        }
    }
    rows
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
