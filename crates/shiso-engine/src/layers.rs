//! The residual stream and the norm that guards every block.
//!
//! One residual stream runs the whole model and every block reads and writes it,
//! so the residual update and the normalization that precedes a block are the two
//! launches every layer slot performs. They live together here because they are the
//! only two launches that are the same for a recurrent slot and an attention slot.
//!
//! # The fused pair, and why it is never split
//!
//! `kernel_add_rms_norm_mul_f32_4` writes the summed residual to one destination
//! and the normalized row to another in a single pass. Running an add and then a
//! norm instead would round the residual twice: once when it is stored and again
//! when it is read back. The fused kernel keeps the sum in registers between the
//! two writes, so the residual stream a layer sees is the one the previous layer
//! intended.

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::{CoreError, CoreResult, DType, DeviceView, Extent3};
use shiso_ple::PleStream;

use crate::attn::{Attn, AttnCache, AttnGeometry, AttnScratch, AttnScratchSizes, Positions};
use crate::blocks::{AttnIo, BlockKernels, GdnIo, GdnState, GdnStates, Trail};
use crate::ffn::{self, Ffn};
use crate::gdn::{Gdn, GdnGeometry, GdnScratch};
use crate::kv::{DeviceKv, PAGE_TOKENS};
use crate::launch::{
    self, AddArgs, DENSE_PLE_DIM, DENSE_PLE_SMEM_BYTES, DensePleArgs, DensePleGateArgs,
    GetRowsArgs, NORM_SMEM_BYTES, NormArgs, ScaleArgs, SoftcapArgs,
};
use crate::moe::{self, Moe, MoeIo, MoeScratch};
use crate::moe_biased::{self, BiasedMoe};
use crate::projection::{self, Weight, WeightAux};
use crate::sessions::Session;
use crate::shape::{
    Activation, HybridShape, LayerKind, MediaPositions, MoeShape, NormScheme, PerLayerShape,
};
use crate::short_conv::{ShortConv, ShortConvGeometry, ShortConvScratch};
use crate::ssm::{Ssm, SsmGeometry, SsmScratch};
use crate::vision::{self, MediaChunk};

/// Elements of one activation row, as the kernel counts them: `float4`.
///
/// The norm strides its work loop in `float4` and takes the row length from the
/// argument block, so a width that is not a multiple of four would leave a tail
/// no lane ever visits.
const NORM_VEC: u64 = 4;

/// The norm kernel's argument block for one activation shape.
///
/// One weight row serves every activation row, which is what the `1` extents and
/// the zero weight strides say: the kernel reaches the weight at
/// `(z % nef3[1]) * nbf3[1]`, and a model with a single plane never leaves
/// offset `0`. `nef3[1]` is still `1` rather than `0` because the kernel takes
/// that modulo unconditionally.
fn norm_args(hidden: u32, n_tok: u32, eps: f32) -> CoreResult<NormArgs> {
    let row = u64::from(hidden).saturating_mul(launch::F32_BYTES);
    let planes = row.saturating_mul(u64::from(n_tok));
    let width = u64::from(hidden);
    if width % NORM_VEC != 0 {
        return Err(CoreError::ShapeMismatch {
            field: "hidden",
            expected: width - width % NORM_VEC,
            got: width,
        });
    }
    let hidden_t = i32::try_from(width / NORM_VEC).map_err(|_| CoreError::ShapeMismatch {
        field: "hidden",
        expected: u64::from(u32::try_from(i32::MAX).unwrap_or(u32::MAX)),
        got: width,
    })?;
    Ok(NormArgs {
        ne00: i32::try_from(width).unwrap_or(i32::MAX),
        ne00_t: hidden_t,
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
    })
}

/// Adds the residual and normalizes, writing both results.
///
/// `sum` receives `residual + x`; `norm` receives that sum scaled by
/// `rsqrt(mean(sum^2) + eps)` and multiplied by `weight`. `sum` is what the next
/// block adds into, so a layer that dropped this write would silently train
/// against a residual stream that never accumulates.
///
/// One threadgroup per row, `(32, 1, 1)` threads: the kernel's work loop advances
/// by `threads_per_threadgroup.x` alone and never reads its simdgroup index, so a
/// taller threadgroup would have every extra simdgroup recompute and rewrite the
/// same row.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the hidden width is not a whole number of
/// `float4` the kernel can stride over, and [`CoreError::Backend`] when the launch
/// fails.
#[expect(
    clippy::too_many_arguments,
    reason = "the fused kernel's two destinations are part of its contract"
)]
pub(crate) fn add_rms_norm_mul(
    backend: &dyn Backend,
    residual: DeviceView<'_>,
    x: DeviceView<'_>,
    weight: DeviceView<'_>,
    sum: DeviceView<'_>,
    norm: DeviceView<'_>,
    hidden: u32,
    n_tok: u32,
    eps: f32,
) -> CoreResult<()> {
    let args = norm_args(hidden, n_tok, eps)?;
    let kernel = KernelRef::new(launch::GENERIC, "kernel_add_rms_norm_mul_f32_4");
    launch::run(
        backend,
        kernel,
        Extent3::new(n_tok, 1, 1),
        Extent3::new(launch::WARP, 1, 1),
        NORM_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(residual),
            KernelArg::Buffer(x),
            KernelArg::Buffer(weight),
            KernelArg::Buffer(sum),
            KernelArg::Buffer(norm),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_add_rms_norm_mul_f32_4", err))
}

/// Adds the residual, scales the sum, and normalizes what that leaves.
///
/// [`add_rms_norm_mul`] with one difference and no others: `sum` receives
/// `(residual + x) * scale` and the mean square is taken over that same scaled
/// row, so the scale reaches both the stream the next block adds into and the
/// magnitudes the normalization divides by. A family whose block output carries
/// a per-layer scalar folds it in here rather than in a launch of its own.
///
/// The scale rides in its own argument rather than in the norm's argument block:
/// that block is the generic norm's, shared with every other norm entry in the
/// library, so the one field this variant adds is bound beside it.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the hidden width is not a whole number of
/// `float4` the kernel can stride over, and [`CoreError::Backend`] when the launch
/// fails.
#[expect(
    clippy::too_many_arguments,
    reason = "the fused kernel's two destinations and its scale are part of its contract"
)]
fn add_scale_rms_norm_mul(
    backend: &dyn Backend,
    residual: DeviceView<'_>,
    x: DeviceView<'_>,
    weight: DeviceView<'_>,
    sum: DeviceView<'_>,
    norm: DeviceView<'_>,
    scale: f32,
    hidden: u32,
    n_tok: u32,
    eps: f32,
) -> CoreResult<()> {
    let args = norm_args(hidden, n_tok, eps)?;
    let kernel = KernelRef::new(launch::DENSE, "kernel_dense_add_scale_rms_norm_mul_f32_4");
    launch::run(
        backend,
        kernel,
        Extent3::new(n_tok, 1, 1),
        Extent3::new(launch::WARP, 1, 1),
        NORM_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(residual),
            KernelArg::Buffer(x),
            KernelArg::Buffer(weight),
            KernelArg::Buffer(sum),
            KernelArg::Buffer(norm),
            KernelArg::Bytes(bytemuck::bytes_of(&scale)),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_dense_add_scale_rms_norm_mul_f32_4", err))
}

/// `RMSNorm` over `x` multiplied by `weight`, with no residual to fold in.
///
/// This is the one norm in the model that has nothing to add: the first slot of
/// a chunk normalizes the embedding it was handed, and there is no earlier
/// partial sum behind it. Every other norm in the pass is the fused pair, which
/// keeps the residual's rounding in exactly one place.
pub(crate) fn rms_norm_mul(
    backend: &dyn Backend,
    x: DeviceView<'_>,
    weight: DeviceView<'_>,
    dst: DeviceView<'_>,
    hidden: u32,
    n_tok: u32,
    eps: f32,
) -> CoreResult<()> {
    let args = norm_args(hidden, n_tok, eps)?;
    let kernel = KernelRef::new(launch::GENERIC, "kernel_rms_norm_mul_f32_4");
    launch::run(
        backend,
        kernel,
        Extent3::new(n_tok, 1, 1),
        Extent3::new(launch::WARP, 1, 1),
        NORM_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(x),
            KernelArg::Buffer(weight),
            // The second fuse operand, which this instantiation never reads.
            //
            // The kernel is one template over a fuse depth, and it declares both
            // operand buffers whatever depth it was built at: `src0`, `src1_0`,
            // `src1_1`, then the destination. At depth two the body never
            // dereferences `src1_1`, but the *binding* slot is still there, so
            // binding four arguments puts the destination where the unread operand
            // belongs and leaves the real destination slot empty. The store then
            // goes nowhere, and the block that reads the normalized rows reads
            // whatever was in the buffer before - zeros on the first pass, and the
            // previous pass's output on every one after.
            KernelArg::Buffer(weight),
            KernelArg::Buffer(dst),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_rms_norm_mul_f32_4", err))
}

// ---------------------------------------------------------------------------------------
// The layer stack
// ---------------------------------------------------------------------------------------

/// Which block a layer slot holds, together with that block's weights.
///
/// An enum rather than two `Option` fields: a slot holds exactly one block, and a
/// pair of options would let a slot hold both or neither. The variant is also what
/// the ordinal beside it means, which is the second reason the two cannot drift
/// apart — [`LayerWeights::ordinal`] indexes a recurrent stack for one variant and
/// a cache layer for the other.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Block is stored per-layer in small static layer vectors, not in dynamic sequences"
)]
pub enum Block {
    /// A recurrent slot, whose ordinal indexes the session's recurrent stack.
    Gdn(Gdn),
    /// An attention slot, whose ordinal indexes the cache's per-layer runs.
    Attn(Attn),
    /// A state-space slot, whose ordinal indexes the session's recurrent stack
    /// as a [`Block::Gdn`] slot's does.
    Ssm(Ssm),
    /// A gated short-convolution slot, whose ordinal indexes the session's
    /// recurrent stack as a [`Block::Gdn`] slot's does.
    ShortConv(ShortConv),
}

/// The routed half of a slot, in one of the two router and expert shapes.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per slot in a small static layer vector, as `Block` is"
)]
pub enum Routed {
    /// A softmax router over gated experts in the storage-tiled layout, with a
    /// sigmoid-gated shared expert.
    Glu(Moe),
    /// A biased sigmoid router over ungated squared-rectifier experts in the
    /// container's row-major layout, with an ungated shared expert.
    Biased(BiasedMoe),
}

/// The per-layer embedding table a stack reads a row of per token.
///
/// Two ways to hold the same thing, and the pass tells them apart by one field
/// of an argument block: a streamed table has been narrowed to the chunk before
/// the kernel sees it, so row `t` is token `t`'s, and a resident one is indexed
/// by the token id. Everything else about the launch is identical, which is why
/// this is one enum rather than two code paths.
///
/// Gemma 4 E4B's table is `42 * 256` values a token id in `q6_K`, which is some
/// 2.3 GiB: streaming it is the default and holding it whole is what an operator
/// with the memory to spare can ask for.
#[derive(Debug)]
pub enum PleTable {
    /// Rows come off the disk per chunk; `row_dtype` says how a row is stored.
    Streamed {
        /// The reader, shared because two stacks may run one table.
        stream: Arc<PleStream>,
        /// How one slot's span of a row is packed. Only [`DType::Q6_K`] has a
        /// prepare kernel here.
        row_dtype: DType,
    },
    /// The whole table is resident; the kernel indexes it by token id.
    Resident {
        /// One row per token id, `n_layers` blocks wide.
        table: Weight,
    },
}

/// One slot's per-layer-input weights.
///
/// Three tensors that only mean anything together: the gate reads the residual
/// stream at the point the feed-forward has been folded into it, the per-layer
/// row multiplies what the gate's activation produced, and the projection takes
/// that back to the residual width for the norm below it.
#[derive(Debug)]
pub struct PerLayerWeights {
    /// `hidden -> dim`: what the per-layer row gates.
    pub gate: Weight,
    /// `dim -> hidden`: what the gated row comes back through.
    pub proj: Weight,
    /// The norm over that projection's output, `[hidden]`.
    pub post_norm: DeviceBuffer,
}

/// One layer slot: its norms, the feed-forward, and the block between them.
#[derive(Debug)]
pub struct LayerWeights {
    /// Norm the block reads.
    pub input_norm: DeviceBuffer,
    /// Norm applied to the block's OUTPUT, before it is folded into the stream.
    ///
    /// `None` for a family that normalizes only what a block reads, which is
    /// every slot of a [`crate::shape::NormScheme::PreOnly`] stack.
    pub post_attn_norm: Option<DeviceBuffer>,
    /// Norm the feed-forward reads, which is what GGUF stores as
    /// `blk.N.ffn_norm` (qwen35's files spell the same role
    /// `blk.N.post_attention_norm`).
    ///
    /// `None` exactly when [`LayerWeights::ffn`] is: a slot with no
    /// feed-forward half has nothing to read it.
    pub ffn_norm: Option<DeviceBuffer>,
    /// Norm applied to the feed-forward's OUTPUT, before it is folded in.
    ///
    /// `None` for the same stacks [`LayerWeights::post_attn_norm`] is absent
    /// from: the two arrive together or not at all.
    pub post_ffn_norm: Option<DeviceBuffer>,
    /// The feed-forward half of the slot, `None` for a slot the shape lists in
    /// [`HybridShape::no_ffn`].
    ///
    /// For a routed stack this is the SHARED expert - the always-on one - and
    /// the routed half is [`LayerWeights::moe`] beside it. The two are not
    /// alternatives with one field: a routed slot has both, the shared expert
    /// IS a dense feed-forward, and giving it the same field every dense slot
    /// uses is what lets `crate::ffn::run` serve it unchanged.
    pub ffn: Option<Ffn>,
    /// The routed half of the slot, for a stack whose shape carries a router.
    ///
    /// `None` for every slot of a dense stack. A stack whose shape states a
    /// router and whose slots do not carry one is a binding that went wrong
    /// rather than a slot that quietly runs dense, and the pass refuses it by
    /// name.
    pub moe: Option<Routed>,
    /// The block half of the slot.
    pub block: Block,
    /// Rank of this slot among the slots of its own kind, counted from zero.
    ///
    /// A recurrent slot's rank indexes the session's recurrent stack; an attention
    /// slot's rank indexes the per-slot attention geometry. Both were built in
    /// slot order, so the rank is the index in either case.
    pub ordinal: u32,
    /// Rank of the cache runs an attention slot reads, or `None` for a recurrent
    /// slot.
    ///
    /// Not the same number as [`LayerWeights::ordinal`]: a slot that reads an
    /// earlier slot's cache has an ordinal of its own and the owner's run rank,
    /// and once a stack has one such slot every later slot's two numbers differ.
    /// The loader fills it from [`crate::shape::HybridShape::kv_owner_rank`].
    pub kv_rank: Option<u32>,
    /// What this slot's residual sum is multiplied by as it is folded in.
    ///
    /// `1.0` for a family that carries no such scalar, which is also the value
    /// that would leave the sum untouched — but the two are not the same launch:
    /// a stack whose shape says it has no layer scalar closes a block out
    /// through the unscaled kernel, and only a stack that says it has one
    /// reaches the scaled twin. [`Geometry::layer_scalar`] is what chooses; this
    /// is the number it then carries.
    pub layer_scalar: f32,
    /// This slot's per-layer-input weights, for a family that has a per-layer
    /// embedding table.
    ///
    /// `None` for every stack whose shape states no table, which is every slot
    /// or none: the branch is chosen by [`Geometry::ple`], so a slot missing
    /// these in a stack that says it has them is a binding that went wrong
    /// rather than a slot that quietly skips its own branch.
    pub ple: Option<PerLayerWeights>,
}

/// Every resident weight, in the order a forward pass reads them.
#[derive(Debug)]
pub struct Weights {
    /// The vocabulary table the pass gathers rows out of.
    pub token_embd: Weight,
    /// Norm before the output projection.
    pub output_norm: DeviceBuffer,
    /// The output projection when the file stores one of its own.
    ///
    /// `None` is the common case and means the projection is the embedding table
    /// read the other way round, which is why it is an `Option` here rather than a
    /// second copy of a vocabulary-wide matrix.
    pub lm_head: Option<Weight>,
    /// `hidden -> n_layers * dim`: the model's own contribution to every slot's
    /// per-layer input, projected once per chunk.
    ///
    /// `None` for a stack with no per-layer embedding table, which is the only
    /// thing that reads it.
    pub ple_model_proj: Option<Weight>,
    /// The norm applied to each slot's span of that projection, `[dim]`.
    ///
    /// `None` alongside [`Weights::ple_model_proj`]: the two arrive together or
    /// not at all, because the prepare kernel takes both.
    pub ple_proj_norm: Option<DeviceBuffer>,
    /// One entry per layer slot, index zero first.
    pub layers: Vec<LayerWeights>,
}

impl Weights {
    /// The matrix the logits come out of: the file's own projection when it has
    /// one, and the embedding table otherwise.
    #[must_use]
    pub fn output_matrix(&self) -> &Weight {
        self.lm_head.as_ref().unwrap_or(&self.token_embd)
    }
}

/// Every width the pass derives a launch from, computed once at load.
#[derive(Debug, Clone)]
pub struct Geometry {
    /// Residual-stream width.
    pub hidden: u32,
    /// Logit row width.
    pub vocab: u32,
    /// Feed-forward inner width, which is what the two widest scratch rows are.
    ///
    /// For a routed stack this is the shared expert's width; one routed
    /// expert's is [`MoeShape::intermediate`] on [`Geometry::moe`].
    pub intermediate: u32,
    /// The router, for a stack whose feed-forward is a mixture of experts.
    ///
    /// `None` is a dense feed-forward. It decides which of two blocks the
    /// layer loop runs at the feed-forward site and whether the routed scratch
    /// is allocated at all.
    pub moe: Option<MoeShape>,
    /// Epsilon every norm in the pass adds.
    pub eps: f32,
    /// The gate curve the feed-forward's activation uses, which is a fact about
    /// the family rather than about a slot: every feed-forward in a stack shares
    /// it.
    pub activation: Activation,
    /// Which norms a slot carries, which is what decides whether a block's
    /// output is normalized before it is folded into the stream.
    pub norms: NormScheme,
    /// How a media prompt is positioned, which decides whether a chunk carrying
    /// media reaches the rotation's three-stream entry points at all.
    pub media_positions: MediaPositions,
    /// Multiplier on the gathered embedding rows; `None` is no launch at all
    /// rather than a launch by one.
    pub embed_scale: Option<f32>,
    /// Cap the logits row is squashed through; `None` is no launch at all.
    pub final_softcap: Option<f32>,
    /// Whether each slot's residual sum carries the scalar beside it.
    ///
    /// The switch and the number are apart on purpose: the switch is the
    /// family's and picks the kernel, the number is
    /// [`LayerWeights::layer_scalar`] and differs per slot.
    pub layer_scalar: bool,
    /// The gather entry point the embedding table's format is read by.
    pub embed: EmbedKernel,
    /// Per-layer embedding widths, or `None` for a family with no such table.
    ///
    /// This is the switch the loop's per-layer branch is chosen by, exactly as
    /// the four beside it are: a stack that states no table dispatches the
    /// sequence it dispatched before the branch existed.
    pub ple: Option<PerLayerGeometry>,
    /// Recurrent-block widths, or `None` for a stack with no recurrent slots.
    pub gdn: Option<GdnGeometry>,
    /// State-space block widths, or `None` for a stack with no such slots.
    pub ssm: Option<SsmGeometry>,
    /// Short-convolution block widths, or `None` for a stack with no such slots.
    pub short_conv: Option<ShortConvGeometry>,
    /// Whether each slot has a feed-forward half, index zero first.
    pub has_ffn: Vec<bool>,
    /// Attention-block widths, one per attention slot in slot order.
    pub attn: Vec<AttnGeometry>,
    /// What the shared attention scratch is sized by: each row's maximum over
    /// every slot.
    ///
    /// Scratch is one set of rows every slot writes into, so it has to hold the
    /// largest each row ever gets; a per-slot allocation would be tens of
    /// megabytes multiplied by the slot count for buffers that never outlive a
    /// launch. Per ROW rather than per slot, because no one slot is the widest
    /// at all three - see [`AttnScratchSizes`].
    pub attn_scratch: AttnScratchSizes,
}

/// The per-layer embedding table's widths, with the layer count its rows carry.
///
/// [`crate::shape::PerLayerShape`] states the two widths a checkpoint declares;
/// this adds the two numbers every launch derives from them and the slot count -
/// the row stride and the element count of one prepared block - so neither is
/// recomputed at a launch site where it could be recomputed differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerLayerGeometry {
    /// Weights in one slot's row, which is [`launch::DENSE_PLE_DIM`] exactly:
    /// one `q6_K` block, which is what the prepare kernel stages.
    pub dim: u32,
    /// Rows the table holds, a vocabulary of its own.
    pub vocab: u32,
    /// Layer slots, which is how many `dim`-wide spans one row carries.
    pub layers: u32,
}

impl PerLayerGeometry {
    /// Bytes one token's whole row occupies: one `q6_K` block per layer slot.
    ///
    /// The number a row identifier becomes an offset through, in the kernel and
    /// in the streaming reader alike. A stride one block out reads every row
    /// after the first from the middle of its predecessor, and every value that
    /// comes back is finite.
    /// Bytes one token's whole row occupies for the default `Q6_K` codec.
    #[must_use]
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes_for(DType::Q6_K)
    }

    /// Bytes one token's whole row occupies for a specific element type.
    #[must_use]
    pub fn row_bytes_for(&self, dtype: DType) -> u64 {
        u64::from(self.layers)
            .saturating_mul(u64::from(dtype.block_bytes().get()))
            .saturating_mul(u64::from(self.dim / dtype.block_elems().get()))
    }

    /// Elements of one prepared block: one `dim`-wide span per layer slot.
    #[must_use]
    pub fn block_elems(&self) -> u64 {
        u64::from(self.layers).saturating_mul(u64::from(self.dim))
    }
}

impl Geometry {
    /// Derives every launch width from a stack shape and the embedding's format.
    ///
    /// The format is an argument rather than a field of the shape because it is a
    /// property of the *file*, not of the architecture: two checkpoints of the
    /// same family can store their vocabulary table differently, and only one of
    /// them is in front of you.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when no row-gather kernel reads that format,
    /// and whatever [`GdnGeometry::new`] reports for a recurrent width it cannot
    /// carry.
    pub fn new(shape: &HybridShape, embed: DType) -> CoreResult<Self> {
        let embed = EmbedKernel::for_dtype(embed).ok_or(CoreError::Unsupported {
            what: "an embedding table in a format no row-gather kernel reads",
            backend: "engine",
        })?;
        // The two flags are the stack's rather than the slot's, so they are
        // filled here instead of being arguments every family would pass the
        // same way at every slot.
        let attn: Vec<AttnGeometry> = shape
            .attn
            .iter()
            .map(|layer| AttnGeometry {
                v_norm: shape.v_norm,
                output_gate: shape.attn_output_gate,
                ..AttnGeometry::new(layer, shape.rms_eps, PAGE_TOKENS)
            })
            .collect();
        let attn_scratch = AttnScratchSizes::over(&attn);
        // A family with no recurrent slot allocates no recurrent geometry and no
        // recurrent scratch: zero of something is not a width every backend is
        // willing to allocate, and a stack that has none should not be asking.
        let gdn = if shape.count(LayerKind::Gdn) == 0 {
            None
        } else {
            Some(GdnGeometry::new(&shape.gdn)?)
        };
        let ssm = match (shape.count(LayerKind::Ssm), shape.ssm) {
            (0, _) => None,
            (_, Some(ssm)) => Some(SsmGeometry::new(ssm, shape.rms_eps)?),
            (_, None) => {
                return Err(CoreError::ShapeMismatch {
                    field: "state-space geometry",
                    expected: 1,
                    got: 0,
                });
            }
        };
        let short_conv = match (shape.count(LayerKind::ShortConv), shape.short_conv) {
            (0, _) => None,
            (_, Some(short)) => Some(ShortConvGeometry::new(short, shape.hidden)?),
            (_, None) => {
                return Err(CoreError::ShapeMismatch {
                    field: "short-convolution geometry",
                    expected: 1,
                    got: 0,
                });
            }
        };
        let has_ffn = (0..shape.n_layers())
            .map(|il| shape.has_ffn(shiso_core::ids::LayerIdx::new(il)))
            .collect();
        let ple = shape
            .ple
            .map(|ple| per_layer(ple, shape.n_layers()))
            .transpose()?;
        Ok(Self {
            hidden: shape.hidden,
            vocab: shape.vocab,
            intermediate: shape.intermediate,
            moe: shape.moe,
            eps: shape.rms_eps,
            activation: shape.activation,
            norms: shape.norms,
            media_positions: shape.media_positions,
            embed_scale: shape.embed_scale,
            final_softcap: shape.final_softcap,
            layer_scalar: shape.layer_scalar,
            embed,
            ple,
            gdn,
            ssm,
            short_conv,
            has_ffn,
            attn,
            attn_scratch,
        })
    }
}

/// One slot's feed-forward: dense, or routed when the shape states a router.
///
/// The whole of the difference between a dense hybrid and this family's stack
/// is these eight lines. That is the argument
/// [`crate::shape::HybridShape::moe`] makes for being an `Option` rather than
/// a shape of its own: a second copy of this file and `crate::stack` would
/// have differed from the first here and nowhere else.
fn feed_forward(
    backend: &dyn Backend,
    geometry: &Geometry,
    scratch: &Scratch,
    layer: &LayerWeights,
    ffn: &Ffn,
    n_tok: u32,
) -> CoreResult<()> {
    let Some(shape) = geometry.moe else {
        return ffn::run(
            backend,
            ffn,
            geometry.activation,
            scratch.normed.view(),
            scratch.ffn_gate.view(),
            scratch.ffn_up.view(),
            scratch.ffn_mid.view(),
            scratch.block_out.view(),
            n_tok,
        );
    };
    // A shape that states a router and a slot that carries none is a binding
    // that went wrong. Running the shared expert alone would be a plausible
    // answer from a ninth of the feed-forward.
    let (routed, moe_scratch) =
        layer
            .moe
            .as_ref()
            .zip(scratch.moe.as_ref())
            .ok_or(CoreError::ShapeMismatch {
                field: "routed feed-forward weights",
                expected: 1,
                got: 0,
            })?;
    let io = MoeIo {
        input: scratch.normed.view(),
        block_out: scratch.block_out.view(),
    };
    // The slot's own feed-forward IS the shared expert, so it is handed in
    // rather than held twice.
    match routed {
        Routed::Glu(moe) => moe::run(backend, moe, ffn, shape, moe_scratch, io, n_tok),
        Routed::Biased(moe) => moe_biased::run(backend, moe, ffn, shape, moe_scratch, io, n_tok),
    }
}

/// The per-layer widths a stack of `layers` slots runs, or the refusal.
///
/// The one thing checked here is the row width, and it is checked because it is
/// not a parameter: the prepare kernel's row IS a `q6_K` block and its staging
/// is sized for one, so a table of another width has no kernel in this build. A
/// kernel cannot refuse, and a launch at the wrong width would read past the
/// staging into the block sum's own slots.
fn per_layer(ple: PerLayerShape, layers: u32) -> CoreResult<PerLayerGeometry> {
    if ple.dim != DENSE_PLE_DIM {
        return Err(CoreError::Unsupported {
            what: "a per-layer embedding row that is not one q6_K block wide",
            backend: "engine",
        });
    }
    Ok(PerLayerGeometry {
        dim: ple.dim,
        vocab: ple.vocab,
        layers,
    })
}

/// Elements of one `q8_0` or `q4_0` block, which is what their gathers
/// index a threadgroup by.
const GATHER_BLOCK_32: u32 = 32;

/// The row-gather entry point one embedding format is read by.
///
/// A typed reference rather than a formatted name: the three formats that have a
/// gather share one argument block but not one module, and picking the pair at
/// load is what keeps the launch below from having to know either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedKernel {
    /// Kernel set the entry point was compiled into.
    module: &'static str,
    /// The entry point itself.
    entry: &'static str,
    /// Elements of a row one threadgroup writes: the format's block.
    ///
    /// The gathers index their threadgroups by block, so the grid has to be
    /// counted in the same unit. A 256-element span for a format whose block
    /// is 32 launches an eighth of the threadgroups, and every row comes back
    /// with its first 256 values right and the rest left as they were.
    span: u32,
}

impl EmbedKernel {
    /// The gather for `dtype`, or `None` when no kernel reads that format.
    ///
    /// The six that share `GetRowsArgs` exactly are what makes one launch site
    /// enough; a seventh format with its own argument block would need its own
    /// launch, so this returns `None` rather than guessing.
    ///
    /// The `f16` entry is the odd one, because it is the only unpacked table
    /// here. It is what a directory checkpoint arrives as: its vocabulary table
    /// is packed in a scheme no kernel reads, so the loader decodes it, and half
    /// precision is what the decode produces. There is a second `f16` gather in
    /// the same module taking the general strided descriptor; it is not this one,
    /// and naming it here would fill in the wrong struct.
    #[must_use]
    pub const fn for_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::Q6_K => Some(Self {
                module: launch::Q35,
                entry: "kernel_get_rows_q6_K_f32",
                span: launch::QK_K,
            }),
            DType::F16 => Some(Self {
                module: launch::GENERIC,
                entry: "kernel_get_rows_f16_f32",
                span: launch::QK_K,
            }),
            DType::Q4_K => Some(Self {
                module: launch::GENERIC,
                entry: "kernel_get_rows_q4_K_f32",
                span: launch::QK_K,
            }),
            DType::Q8_0 => Some(Self {
                module: launch::GENERIC,
                entry: "kernel_get_rows_q8_0_f32",
                span: GATHER_BLOCK_32,
            }),
            DType::Q4_0 => Some(Self {
                module: launch::GENERIC,
                entry: "kernel_get_rows_q4_0_f32",
                span: GATHER_BLOCK_32,
            }),
            // The Saluki export stores its vocabulary table in `iq1_m`; the
            // gather takes the same argument block as the other five, so it
            // needs the pair and no new launch.
            DType::IQ1_M => Some(Self {
                module: launch::GENERIC,
                entry: "kernel_get_rows_iq1_m_f32",
                span: launch::QK_K,
            }),
            DType::MlxAffine4Gs64 => Some(Self {
                module: launch::DENSE,
                entry: "kernel_dense_embed_affine4_f32",
                span: launch::QK_K,
            }),
            // The ternary set's gathers take an argument block of their own;
            // [`embed_rows`] launches them on a branch of their own, and this
            // entry is what lets the geometry accept the table at load.
            DType::Pq2_0 | DType::Ptq1_0 | DType::MlxAffine2Gs128 => {
                match launch::ternary_rows_name(dtype) {
                    Some(entry) => Some(Self {
                        module: launch::TERNARY,
                        entry,
                        span: launch::QK_K,
                    }),
                    None => None,
                }
            }
            _ => None,
        }
    }
}

/// Every activation buffer the pass writes, sized once for the widest chunk.
///
/// Scratch is allocated for [`crate::MAX_PREFILL_CHUNK`] rows and reused by
/// every layer: the alternative is per-layer allocation, which for the
/// feed-forward's inner width alone would be tens of mebibytes multiplied by the
/// layer count, for buffers that never outlive a launch.
#[derive(Debug)]
pub struct Scratch {
    /// The residual stream, in two buffers the pass alternates between.
    ///
    /// The fused norm reads a residual and writes the summed one, and a kernel
    /// whose source and destination are the same range would have lanes reading
    /// rows other lanes have already rewritten. Two buffers and an index is what
    /// removes the question.
    stream: [DeviceBuffer; 2],
    /// The normalized rows a block reads. Also where the pass leaves its answer:
    /// the final norm writes here.
    normed: DeviceBuffer,
    /// A block's contribution to the stream, before it is added in.
    block_out: DeviceBuffer,
    /// The normalized copy of that contribution, for a family that normalizes a
    /// block's output.
    ///
    /// A second buffer rather than a norm in place: the norm reads a whole row
    /// before it writes one, so a destination that is its own source would have
    /// lanes reading what other lanes have already rewritten. The two then swap
    /// roles - the close reads whichever of them the block's output ended up in
    /// - so the extra buffer costs no copy.
    ///
    /// `None` for a [`NormScheme::PreOnly`] stack, which never writes it: a
    /// residual-width buffer at the widest chunk is megabytes, and a stack that
    /// has no post-norm should not be paying for its destination.
    block_out2: Option<DeviceBuffer>,
    /// The feed-forward's gating projection.
    ffn_gate: DeviceBuffer,
    /// The feed-forward's up projection.
    ffn_up: DeviceBuffer,
    /// The activated inner rows the down projection reads.
    ffn_mid: DeviceBuffer,
    /// The chunk's token ids, as the `int32` row the gather indexes with.
    ids: DeviceBuffer,
    /// The chunk's absolute positions, as the `int32` row rope and the cache read.
    pos: DeviceBuffer,
    /// The last key position each token may read, one `int32` per token.
    ///
    /// The positions again for a causal chunk, and wider than the positions for
    /// a family whose media spans are attended both ways. Allocated for every
    /// stack: one row of the widest chunk is four bytes a token, against
    /// activation buffers measured in megabytes, and a second allocation path
    /// that only some requests take is the more expensive of the two.
    bound: DeviceBuffer,
    /// The chunk's multimodal positions: temporal, row and column, one `int32`
    /// row each.
    ///
    /// Allocated for every stack, text-only ones included, and left untouched by
    /// a chunk with no media. Three rows of the widest chunk is twelve bytes a
    /// token against activation buffers measured in megabytes, which is a
    /// smaller price than a second allocation path that only some requests
    /// take.
    mrope: [DeviceBuffer; 3],
    /// Encoder row per token, negative for a token that keeps its embedding.
    splice_map: DeviceBuffer,
    /// The logits block a step's rows are written into.
    logits: DeviceBuffer,
    /// The model's own per-layer projection, `[n_tok, n_layers * dim]`.
    ///
    /// The four rows below it are the per-layer embedding branch's, and all
    /// four are `None` for a stack whose shape states no table - like the
    /// recurrent scratch, and for the same reason: at the widest chunk they are
    /// megabytes apiece, and a stack that never dispatches the branch should
    /// not be paying for its destinations.
    ple_proj: Option<DeviceBuffer>,
    /// The prepared per-layer block, `[n_tok, n_layers, dim]`: the normalized
    /// projection mixed with the dequantized row.
    ple_rows: Option<DeviceBuffer>,
    /// One slot's gate, `[n_tok, dim]`, activated and multiplied in place.
    ple_gate: Option<DeviceBuffer>,
    /// The quantized rows a streamed table stages, `[n_tok, row_bytes]`.
    ///
    /// Allocated for a resident table too, and left untouched by it: a second
    /// allocation path that only some stacks take costs more in places to get
    /// it wrong than it saves in bytes, and the row is 210 bytes a slot against
    /// activation buffers measured in megabytes.
    ple_stream: Option<DeviceBuffer>,
    /// Routed feed-forward scratch, absent when the stack has no router.
    ///
    /// Eleven buffers, and at the widest chunk the routed columns alone are
    /// four thousand rows of the expert width - megabytes a dense stack has no
    /// reason to be paying for.
    moe: Option<MoeScratch>,
    /// Recurrent-block scratch, absent when the stack has no recurrent slots.
    gdn: Option<GdnScratch>,
    /// State-space block scratch, absent when the stack has no such slots.
    ssm: Option<SsmScratch>,
    /// Short-convolution block scratch, absent when the stack has no such slots.
    short_conv: Option<ShortConvScratch>,
    /// Attention-block scratch, sized for the widest attention slot.
    attn: AttnScratch,
}

impl Scratch {
    /// Allocates every row a chunk of `rows` tokens needs.
    ///
    /// `logit_rows` is sized separately because the logits block is indexed by
    /// batch member rather than by token: a prefill writes one row of it and a
    /// decode step writes one per session.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        geometry: &Geometry,
        rows: u32,
        logit_rows: u32,
    ) -> CoreResult<Self> {
        let tokens = u64::from(rows);
        let hidden = u64::from(geometry.hidden)
            .saturating_mul(launch::F32_BYTES)
            .saturating_mul(tokens);
        let inner = u64::from(geometry.intermediate)
            .saturating_mul(launch::F32_BYTES)
            .saturating_mul(tokens);
        let ids = tokens.saturating_mul(ID_BYTES);
        let logits = u64::from(geometry.vocab)
            .saturating_mul(launch::F32_BYTES)
            .saturating_mul(u64::from(logit_rows));
        Ok(Self {
            stream: [alloc(backend, hidden)?, alloc(backend, hidden)?],
            normed: alloc(backend, hidden)?,
            block_out: alloc(backend, hidden)?,
            block_out2: match geometry.norms {
                NormScheme::PreOnly => None,
                NormScheme::PrePost => Some(alloc(backend, hidden)?),
            },
            ffn_gate: alloc(backend, inner)?,
            ffn_up: alloc(backend, inner)?,
            ffn_mid: alloc(backend, inner)?,
            ids: alloc(backend, ids)?,
            pos: alloc(backend, ids)?,
            bound: alloc(backend, ids)?,
            mrope: [
                alloc(backend, ids)?,
                alloc(backend, ids)?,
                alloc(backend, ids)?,
            ],
            splice_map: alloc(backend, ids)?,
            logits: alloc(backend, logits)?,
            ple_proj: ple_row(backend, geometry, tokens, PleRow::Block)?,
            ple_rows: ple_row(backend, geometry, tokens, PleRow::Block)?,
            ple_gate: ple_row(backend, geometry, tokens, PleRow::Gate)?,
            ple_stream: ple_row(backend, geometry, tokens, PleRow::Quantized)?,
            moe: geometry
                .moe
                .map(|moe| MoeScratch::new(backend, moe, geometry.hidden, rows))
                .transpose()?,
            gdn: geometry
                .gdn
                .as_ref()
                .map(|gdn| GdnScratch::new(backend, gdn, rows))
                .transpose()?,
            ssm: geometry
                .ssm
                .as_ref()
                .map(|ssm| SsmScratch::new(backend, ssm, rows))
                .transpose()?,
            short_conv: geometry
                .short_conv
                .as_ref()
                .map(|short| ShortConvScratch::new(backend, short, rows))
                .transpose()?,
            attn: AttnScratch::new(backend, &geometry.attn_scratch, rows)?,
        })
    }

    /// The token-id row the gather reads.
    #[must_use]
    pub fn ids(&self) -> DeviceView<'_> {
        self.ids.view()
    }

    /// The position row rope and the cache read.
    #[must_use]
    pub fn positions(&self) -> DeviceView<'_> {
        self.pos.view()
    }

    /// The row saying how far along the sequence each token may look.
    #[must_use]
    pub fn bound(&self) -> DeviceView<'_> {
        self.bound.view()
    }

    /// The logits block, whole.
    #[must_use]
    pub fn logits(&self) -> &DeviceBuffer {
        &self.logits
    }

    /// The temporal, row and column position rows, in that order.
    ///
    /// One accessor for the three, because the rotation reads them together and
    /// a stream handed over on its own has no meaning: the sections are assigned
    /// by index, so two of these swapped is a rotation that is wrong in a way
    /// nothing downstream can see.
    #[must_use]
    pub fn mrope(&self) -> [DeviceView<'_>; 3] {
        [
            self.mrope[0].view(),
            self.mrope[1].view(),
            self.mrope[2].view(),
        ]
    }

    /// The row the splice indexes the encoder block with.
    #[must_use]
    pub fn splice_map(&self) -> DeviceView<'_> {
        self.splice_map.view()
    }

    /// The residual stream buffer at `index`, which a pass alternates between.
    ///
    /// `index` is what [`forward`] hands back, and the type is what stops
    /// anything else computing it: [`ResidualIndex`] has no public
    /// constructor, so the only values in circulation are one a pass returned
    /// and the two named constants the draft head's fixed-parity pass uses.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when `index` is neither of the two, which
    /// the type makes unreachable and which is checked anyway rather than
    /// indexed blindly.
    pub(crate) fn stream_at(&self, index: ResidualIndex) -> CoreResult<DeviceView<'_>> {
        self.stream
            .get(index.0)
            .map(DeviceBuffer::view)
            .ok_or(CoreError::OutOfBounds {
                offset: u64::try_from(index.0).unwrap_or(u64::MAX),
                len: 1,
                buffer_bytes: 2,
            })
    }

    /// The normalized rows a block reads, which is also where a pass leaves its
    /// answer.
    pub(crate) fn normed(&self) -> DeviceView<'_> {
        self.normed.view()
    }

    /// A block's contribution to the stream, before it is added in.
    pub(crate) fn block_out(&self) -> DeviceView<'_> {
        self.block_out.view()
    }

    /// The three feed-forward rows, in the order [`crate::ffn::run`] takes them.
    pub(crate) fn ffn_rows(&self) -> [DeviceView<'_>; 3] {
        [
            self.ffn_gate.view(),
            self.ffn_up.view(),
            self.ffn_mid.view(),
        ]
    }

    /// The attention scratch, which every attention slot shares.
    pub(crate) fn attn(&self) -> &AttnScratch {
        &self.attn
    }

    /// Where a streamed per-layer table's quantized rows are staged.
    ///
    /// `None` for a stack whose shape states no per-layer table, which is the
    /// only stack that has no such row at all - a resident table has one and
    /// never writes it. A caller that reached this with `None` in hand is
    /// holding a table the shape does not describe.
    #[must_use]
    pub fn ple_stream(&self) -> Option<DeviceView<'_>> {
        self.ple_stream.as_ref().map(DeviceBuffer::view)
    }
}

/// Which of the per-layer branch's four rows is being sized.
#[derive(Debug, Clone, Copy)]
enum PleRow {
    /// A whole prepared or projected block: `n_layers * dim` floats a token.
    Block,
    /// One slot's gate: `dim` floats a token.
    Gate,
    /// One token's quantized row, as it comes off the file.
    Quantized,
}

/// Allocates one of the per-layer branch's rows, for a stack that has the
/// branch at all.
///
/// `None` rather than a zero-byte allocation when the shape states no table:
/// zero bytes is a width some backends refuse and the rest hand back a buffer no
/// launch may touch, and the branch that would read it never runs.
fn ple_row(
    backend: &Arc<dyn Backend>,
    geometry: &Geometry,
    tokens: u64,
    row: PleRow,
) -> CoreResult<Option<DeviceBuffer>> {
    let Some(ple) = geometry.ple.as_ref() else {
        return Ok(None);
    };
    let width = match row {
        PleRow::Block => ple.block_elems().saturating_mul(launch::F32_BYTES),
        PleRow::Gate => u64::from(ple.dim).saturating_mul(launch::F32_BYTES),
        PleRow::Quantized => ple.row_bytes(),
    };
    alloc(backend, width.saturating_mul(tokens)).map(Some)
}

/// Bytes of one token id and of one position, both of which are `int32`.
const ID_BYTES: u64 = 4;

/// Bytes one cached key or value element occupies: the cache is f16.
const F16_BYTES: u64 = 2;

/// Gathers one embedding row per token id.
///
/// The three formats that have a gather share this argument block exactly, so the
/// format only chooses the entry point; everything else about the launch is the
/// row geometry, which is the same either way.
///
/// A table stored in a rotated basis is taken back to the residual stream's
/// basis here, right after the gather, so no caller - the chunk's opening, a
/// drafter reading the target's table - can see a rotated row.
pub(crate) fn embed_rows(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    dst: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    gather_rows(backend, weights, geometry, scratch, dst, n_tok)?;
    match weights.token_embd.rotation() {
        Some(rotation) => rotation.inverse_in_place(backend, n_tok, dst),
        None => Ok(()),
    }
}

/// The gather [`embed_rows`] runs before any rotation.
fn gather_rows(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    dst: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    if let Some(entry) = launch::ternary_rows_name(weights.token_embd.dtype) {
        let table = &weights.token_embd;
        let (scales, biases) = match table.aux().and_then(WeightAux::affine) {
            Some(aux) => (aux.scales.view(), aux.biases.view()),
            None => (table.view(), table.view()),
        };
        let args = launch::TernaryRowsArgs {
            in_dim: geometry.hidden,
            n_tok,
        };
        return launch::run(
            backend,
            KernelRef::new(launch::TERNARY, entry),
            Extent3::new(geometry.hidden / TERNARY_GROUP, n_tok, 1),
            Extent3::linear(TERNARY_GROUP),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(table.view()),
                KernelArg::Buffer(scales),
                KernelArg::Buffer(biases),
                KernelArg::Buffer(scratch.ids()),
                KernelArg::Buffer(dst),
            ],
        )
        .map_err(|err| launch::launch_failure(entry, err));
    }
    if weights.token_embd.dtype.is_affine() {
        let aux = weights
            .token_embd
            .aux()
            .and_then(WeightAux::affine)
            .ok_or_else(|| CoreError::InvalidValue {
                key: "token_embd aux",
                value: "affine embedding is missing scales/biases".into(),
            })?;
        let args = launch::DenseEmbedAffineArgs {
            hidden: geometry.hidden,
            n_tok,
        };
        // One thread per column, so the grid counts threadgroups of
        // `EMBED_AFFINE_BLOCK` columns rather than columns: `Extent3` here is a
        // threadgroup count, and passing the column count directly would launch
        // `EMBED_AFFINE_BLOCK` times the threadgroups and let all but the first
        // few return on the bounds check.
        return launch::run(
            backend,
            KernelRef::new(launch::DENSE, "kernel_dense_embed_affine4_f32"),
            Extent3::new(geometry.hidden.div_ceil(EMBED_AFFINE_BLOCK), n_tok, 1),
            Extent3::linear(EMBED_AFFINE_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(dst),
                KernelArg::Buffer(weights.token_embd.view()),
                KernelArg::Buffer(aux.scales.view()),
                KernelArg::Buffer(aux.biases.view()),
                KernelArg::Buffer(scratch.ids()),
            ],
        )
        .map_err(|err| launch::launch_failure("kernel_dense_embed_affine4_f32", err));
    }
    let args = GetRowsArgs {
        n_embd: i32::try_from(geometry.hidden).unwrap_or(i32::MAX),
        n_vocab: i32::try_from(geometry.vocab).unwrap_or(i32::MAX),
        n_tokens: i32::try_from(n_tok).unwrap_or(i32::MAX),
        pad0: 0,
        src_row_bytes: weights.token_embd.row_bytes()?,
        dst_row_bytes: u64::from(geometry.hidden).saturating_mul(launch::F32_BYTES),
        token_stride: ID_BYTES,
    };
    launch::run(
        backend,
        KernelRef::new(geometry.embed.module, geometry.embed.entry),
        Extent3::new(geometry.hidden.div_ceil(geometry.embed.span), n_tok, 1),
        Extent3::new(geometry.embed.span.min(launch::GET_ROWS_BLOCK), 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(weights.token_embd.view()),
            KernelArg::Buffer(scratch.ids()),
            KernelArg::Buffer(dst),
        ],
    )
    .map_err(|err| launch::launch_failure(geometry.embed.entry, err))
}

/// Elements one threadgroup of the ternary gather covers: one scale group.
const TERNARY_GROUP: u32 = 128;

/// Columns one threadgroup of the affine embedding gather covers.
const EMBED_AFFINE_BLOCK: u32 = 64;

/// Which position streams this chunk's rotation reads.
///
/// A chunk with no media takes the one-dimensional rotary entry points, the same
/// ones it took before the multimodal path existed. Three agreeing streams would
/// give the same angles, but the same angles are not the same instructions, and
/// the text path is pinned to a recorded vector.
///
/// So does a chunk WITH media, for a family that positions a picture the way it
/// positions text: there the three streams are the position row repeated, and
/// the rotation has no three-stream entry point to reach. What makes a media
/// span a span for that family is [`Scratch::bound`], not the angle.
fn positions<'a>(geometry: &Geometry, scratch: &'a Scratch, multimodal: bool) -> Positions<'a> {
    if !multimodal || geometry.media_positions == MediaPositions::Linear {
        return Positions::Linear(scratch.positions());
    }
    let [t, h, w] = scratch.mrope();
    Positions::Mrope {
        linear: scratch.positions(),
        t,
        h,
        w,
    }
}

/// Fills the residual stream a chunk starts from.
///
/// One embedding row per token, scaled where the family scales them, and then —
/// for a chunk carrying media — the rows at the placeholder positions replaced
/// by the vision tower's. The placeholders are gathered like every other token
/// first and overwritten after, rather than being skipped, because that keeps
/// the gather one launch over the whole chunk; and the replacement lands before
/// the first norm reads a row.
///
/// The scale runs between the two: it belongs to the vocabulary table the gather
/// read, and the encoder rows that land on top of it arrive at the scale the
/// tower produced them at.
fn open(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    media: Option<MediaChunk<'_>>,
    cur: usize,
    n_tok: u32,
) -> CoreResult<()> {
    embed_rows(
        backend,
        weights,
        geometry,
        scratch,
        scratch.stream[cur].view(),
        n_tok,
    )?;
    if let Some(scale) = geometry.embed_scale {
        scale_rows(
            backend,
            scratch.stream[cur].view(),
            geometry.hidden.saturating_mul(n_tok),
            scale,
        )?;
    }
    if let Some(media) = media {
        vision::splice(
            backend,
            scratch.stream[cur].view(),
            scratch.splice_map(),
            media,
            geometry.hidden,
            n_tok,
        )?;
    }
    Ok(())
}

/// Every row the per-layer branch reads or writes, resolved once per pass.
///
/// A stack whose shape states a table and whose scratch or weights are missing
/// one of these is a binding that went wrong, and running it as if it were the
/// other family would be a plausible answer from the wrong architecture. So the
/// resolution happens once, before the loop, and every launch below is total.
struct PerLayerIo<'a> {
    geometry: &'a PerLayerGeometry,
    /// The table's quantized rows: the whole table when it is resident, and the
    /// chunk's staging when it is streamed.
    rows: DeviceView<'a>,
    /// Whether `rows` is indexed by the token id rather than by the token's own
    /// position.
    resident: bool,
    proj: DeviceView<'a>,
    block: DeviceView<'a>,
    row_dtype: DType,
    gate: DeviceView<'a>,
}

/// Binds the per-layer rows, or says which half of the pair is missing.
///
/// `None` is not a refusal: it is a stack whose shape states no per-layer
/// embedding table, which is every stack of every family that has none. The
/// branch is then never dispatched, and there is nothing to bind. A shape that
/// DOES state a table and is missing a row is the error case, and each missing
/// row is named.
fn per_layer_io<'a>(
    geometry: &'a Geometry,
    scratch: &'a Scratch,
    table: Option<&'a PleTable>,
) -> CoreResult<Option<PerLayerIo<'a>>> {
    let Some(ple) = geometry.ple.as_ref() else {
        return Ok(None);
    };
    let missing = |field: &'static str| CoreError::ShapeMismatch {
        field,
        expected: 1,
        got: 0,
    };
    let (rows, resident, row_dtype) = match table.ok_or(missing("per-layer table"))? {
        PleTable::Resident { table } => (table.view(), true, table.dtype),
        PleTable::Streamed { row_dtype, .. } => (
            scratch.ple_stream().ok_or(missing("per-layer staging"))?,
            false,
            *row_dtype,
        ),
    };
    Ok(Some(PerLayerIo {
        geometry: ple,
        rows,
        resident,
        row_dtype,
        proj: scratch
            .ple_proj
            .as_ref()
            .ok_or(missing("per-layer projection scratch"))?
            .view(),
        block: scratch
            .ple_rows
            .as_ref()
            .ok_or(missing("per-layer block scratch"))?
            .view(),
        gate: scratch
            .ple_gate
            .as_ref()
            .ok_or(missing("per-layer gate scratch"))?
            .view(),
    }))
}

/// Projects the model's own per-layer input and mixes it with the table's row.
///
/// Once per chunk, right after the embedding is in the stream: every slot reads
/// a span of what this writes, so doing it per slot would be the same
/// projection dispatched `n_layers` times over.
///
/// The three scales are the family's: the projection is divided by
/// `sqrt(hidden)` before it is normalized, the embedding multiplied by
/// `sqrt(dim)`, and the sum halved in quadrature.
fn per_layer_prepare(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    io: &PerLayerIo<'_>,
    cur: usize,
    n_tok: u32,
) -> CoreResult<()> {
    let model = weights
        .ple_model_proj
        .as_ref()
        .ok_or(CoreError::ShapeMismatch {
            field: "per-layer model projection",
            expected: 1,
            got: 0,
        })?;
    let norm = weights
        .ple_proj_norm
        .as_ref()
        .ok_or(CoreError::ShapeMismatch {
            field: "per-layer projection norm",
            expected: 1,
            got: 0,
        })?;
    projection::projection(backend, model, n_tok, scratch.stream[cur].view(), io.proj)?;
    let (entry, row_bytes) = match io.row_dtype {
        DType::Q6_K => (
            "kernel_dense_ple_prepare_q6_K",
            io.geometry.row_bytes_for(DType::Q6_K),
        ),
        DType::MlxAffine4Gs64 => (
            "kernel_dense_ple_prepare_affine4",
            io.geometry.row_bytes_for(DType::MlxAffine4Gs64),
        ),
        _ => {
            return Err(CoreError::Unsupported {
                what: "unsupported per-layer embedding row dtype",
                backend: "metal",
            });
        }
    };
    let args = DensePleArgs {
        n_tok,
        n_layers: io.geometry.layers,
        dim: io.geometry.dim,
        row_bytes: u32::try_from(row_bytes).unwrap_or(u32::MAX),
        resident: u32::from(io.resident),
        eps: geometry.eps,
        proj_scale: inv_sqrt(geometry.hidden),
        embed_scale: sqrt_of(io.geometry.dim),
        mix_scale: std::f32::consts::FRAC_1_SQRT_2,
    };
    launch::run(
        backend,
        KernelRef::new(launch::DENSE, entry),
        Extent3::new(io.geometry.layers, n_tok, 1),
        Extent3::new(launch::BLOCK_128, 1, 1),
        DENSE_PLE_SMEM_BYTES,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(io.block),
            KernelArg::Buffer(io.proj),
            KernelArg::Buffer(norm.view()),
            KernelArg::Buffer(io.rows),
            KernelArg::Buffer(scratch.ids()),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// `1 / sqrt(n)`, the scale the model projection carries.
#[expect(
    clippy::cast_precision_loss,
    reason = "a residual width is a small integer, and the scale it implies is a float"
)]
fn inv_sqrt(n: u32) -> f32 {
    1.0 / (n as f32).sqrt()
}

/// `sqrt(n)`, the scale the dequantized embedding carries.
#[expect(
    clippy::cast_precision_loss,
    reason = "a row width is a small integer, and the scale it implies is a float"
)]
fn sqrt_of(n: u32) -> f32 {
    (n as f32).sqrt()
}

/// The per-layer branch of one slot, between the feed-forward's own norm and the
/// close.
///
/// `block` is the rows the feed-forward produced, normalized where the family
/// normalizes them - the same view the close would otherwise have folded in.
/// Here it is folded in FIRST, into the other stream buffer, because the gate
/// reads the residual at that point rather than before it. What the close then
/// folds in is the per-layer branch's own output, which is why this returns the
/// stream index it left the sum in and leaves its result in
/// [`Scratch::block_out`].
#[expect(
    clippy::too_many_arguments,
    reason = "the branch reads the geometry, the scratch, the chunk's prepared block, this slot's own weights and its index, the stream buffer it folds into and the rows it folds; bundling them would name the same six borrows twice"
)]
fn per_layer_slot(
    backend: &dyn Backend,
    geometry: &Geometry,
    scratch: &Scratch,
    io: &PerLayerIo<'_>,
    slot: (&PerLayerWeights, u32),
    cur: usize,
    block: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<usize> {
    let (weights, layer) = slot;
    let next = 1 - cur;
    let elements = u64::from(geometry.hidden).saturating_mul(u64::from(n_tok));
    add2(
        backend,
        scratch.stream[cur].view(),
        block,
        scratch.stream[next].view(),
        u32::try_from(elements).unwrap_or(u32::MAX),
    )?;
    projection::projection(
        backend,
        &weights.gate,
        n_tok,
        scratch.stream[next].view(),
        io.gate,
    )?;
    per_layer_gate(backend, io, layer, n_tok)?;
    let out = scratch
        .block_out2
        .as_ref()
        .ok_or(CoreError::ShapeMismatch {
            field: "post norm scratch",
            expected: 1,
            got: 0,
        })?
        .view();
    projection::projection(backend, &weights.proj, n_tok, io.gate, out)?;
    rms_norm_mul(
        backend,
        out,
        weights.post_norm.view(),
        scratch.block_out.view(),
        geometry.hidden,
        n_tok,
        geometry.eps,
    )?;
    Ok(next)
}

/// `gate[t, i] = gelu_tanh(gate[t, i]) * ple[t, layer, i]`, in place.
fn per_layer_gate(
    backend: &dyn Backend,
    io: &PerLayerIo<'_>,
    layer: u32,
    n_tok: u32,
) -> CoreResult<()> {
    let args = DensePleGateArgs {
        n_tok,
        n_layers: io.geometry.layers,
        dim: io.geometry.dim,
        layer,
    };
    let n = n_tok.saturating_mul(io.geometry.dim);
    launch::run(
        backend,
        KernelRef::new(launch::DENSE, "kernel_dense_ple_gate_mul"),
        Extent3::linear(n.div_ceil(launch::BLOCK_LINEAR)),
        Extent3::new(launch::BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(io.gate),
            KernelArg::Buffer(io.block),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_dense_ple_gate_mul", err))
}

/// `out[i] = a[i] + b[i]` over `n` elements.
///
/// A destination of its own rather than an add in place: the residual stream is
/// two buffers the pass alternates between, and this writes the one the slot is
/// about to start reading.
pub(crate) fn add2(
    backend: &dyn Backend,
    a: DeviceView<'_>,
    b: DeviceView<'_>,
    out: DeviceView<'_>,
    n: u32,
) -> CoreResult<()> {
    let args = AddArgs { n };
    launch::run(
        backend,
        KernelRef::new(launch::GENERIC, "kernel_add2_f32"),
        Extent3::linear(n.div_ceil(launch::BLOCK_LINEAR)),
        Extent3::new(launch::BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(a),
            KernelArg::Buffer(b),
            KernelArg::Buffer(out),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_add2_f32", err))
}
/// Which of the two residual buffers a pass's sum landed in.
///
/// A newtype and `#[must_use]`, not a bare `usize`, because the caller has to
/// store it and forgetting to is invisible. A draft head starts from the
/// target's residual for the row it drafts from; the buffer that residual
/// lives in is the parity of however many closes the pass ran. A caller that
/// dropped this and left [`crate::HybridStack::residual`] at its previous
/// value would hand the head a residual one layer old, at exactly the right
/// shape, through launches that all succeed.
///
/// Workspace clippy runs pedantic at `-D warnings`, so the `#[must_use]` here
/// is a build failure rather than a lint note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "this is the buffer the pass's residual landed in; a caller that \
              drops it leaves the draft head reading a stale one"]
pub(crate) struct ResidualIndex(usize);

impl ResidualIndex {
    /// The buffer a pass opens in, before any block has closed.
    ///
    /// The ONLY construction of this type outside [`forward`], and it exists
    /// for two callers: [`crate::HybridStack::new`], which has to put
    /// something in the field before a pass has run, and the draft head, whose
    /// own pass runs a fixed two-close sequence and therefore knows its parity
    /// statically where the target's thirty-two-layer one does not.
    pub(crate) const FIRST: Self = Self(0);

    /// The other buffer of the pair.
    ///
    /// For the draft head's own pass, which alternates once. Nothing derives
    /// the target's parity from this.
    pub(crate) const SECOND: Self = Self(1);
}

/// Who the rows of one pass belong to.
///
/// Every launch a pass issues is over `n_tok` rows of the same scratch, and for
/// all but two of them that is the whole story: a projection, a norm, a gate and
/// a feed-forward do the same arithmetic on row `i` whichever rows it was issued
/// with. The two that are not are the ones that reach per-session state - the
/// attention block, through the page table a position is translated by, and the
/// recurrent block, through the state matrix a row folds into - and this is what
/// tells them apart.
///
/// An enum rather than "a slice, of length one for a chunk", because a chunk is
/// not a one-row batch. Its rows are one session's **consecutive positions**, so
/// the recurrent block may walk them inside one launch and the attention block
/// may hand one table to every row; a batch's rows are different conversations
/// and neither is true. Collapsing the two would make the cheap case reachable
/// only by accident.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PassRows<'a> {
    /// Every row is this session's, at consecutive positions: a prefill chunk,
    /// a single-session decode step, or a verify pass.
    Chunk(&'a Session),
    /// Row `i` is `members[i]`'s own single token - a batched decode step.
    ///
    /// `pages` is the packed block [`crate::rows::RowPages`] wrote, one row per
    /// member and in the same order, because the kernels reach a row's table by
    /// its row index and nothing downstream can re-derive the order.
    Batch {
        /// The sessions, row for row with the pass.
        members: &'a [&'a Session],
        /// Their page tables, packed one row per member.
        pages: crate::attn::Pages<'a>,
    },
    /// Several sessions' prompt runs back to back: `runs[0].rows` rows of the
    /// first session at consecutive positions, then the second's, and so on -
    /// a batched prompt pass.
    ///
    /// Everything that reads a row alone - the projections, the norms, the
    /// feed-forward - runs once over every row. The two blocks that read a
    /// session's history run once per run over that run's rows, through
    /// exactly the chunk path a solo read of the run takes: attention through
    /// the session's own page table, a recurrent block through its own state.
    Segments(&'a [SegmentRun<'a>]),
}

/// One session's run inside a [`PassRows::Segments`] pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentRun<'a> {
    /// The session the run's rows belong to.
    pub(crate) session: &'a Session,
    /// The position of the run's first row in its session.
    pub(crate) start: u32,
    /// How many consecutive rows of the pass are this run's.
    pub(crate) rows: u32,
    /// Sessions that read this run's rows as their own: after the run's
    /// attention, its cache rows are copied into each of theirs at the same
    /// positions, and after its recurrent block, its state is copied over
    /// theirs. Empty for a run nothing shares.
    ///
    /// This is how a trunk several prompts open with is read once: the trunk
    /// runs in one session and is forked into the others layer by layer, each
    /// before that layer's runs of theirs read it. What they receive is what
    /// they would have computed, so their answers do not move.
    pub(crate) forks: &'a [&'a Session],
}

/// Copies `run`'s recurrent state for slot `ordinal` over every fork's.
fn fork_state(backend: &dyn Backend, run: &SegmentRun<'_>, ordinal: u32) -> CoreResult<()> {
    let from = run.session.gdn(ordinal)?;
    for fork in run.forks {
        let to = fork.gdn(ordinal)?;
        backend.copy_d2d(to.conv(), from.conv())?;
        backend.copy_d2d(to.state(), from.state())?;
    }
    Ok(())
}

/// The physical page `session` keeps position `at` in.
fn page_of(session: &Session, at: u32) -> CoreResult<u64> {
    let table = session.kv().frontier().table().as_slice();
    table
        .get(usize::try_from(at / PAGE_TOKENS).unwrap_or(usize::MAX))
        .map(|page| u64::from(u32::from(*page)))
        .ok_or(CoreError::ShapeMismatch {
            field: "forked cache page",
            expected: u64::from(at / PAGE_TOKENS) + 1,
            got: u64::try_from(table.len()).unwrap_or(u64::MAX),
        })
}

/// Copies `run`'s cache rows of one attention slot - keys in `k`, values in
/// `v`, `row_bytes` a position - into every fork's pages at the same
/// positions, one copy a page each side.
fn fork_rows(
    backend: &dyn Backend,
    (k, v): (DeviceView<'_>, DeviceView<'_>),
    row_bytes: u64,
    run: &SegmentRun<'_>,
) -> CoreResult<()> {
    let end = run.start.saturating_add(run.rows);
    let mut at = run.start;
    while at < end {
        let within = at % PAGE_TOKENS;
        let take = (PAGE_TOKENS - within).min(end - at);
        let bytes = u64::from(take).saturating_mul(row_bytes);
        let offset = |page: u64| {
            page.saturating_mul(u64::from(PAGE_TOKENS))
                .saturating_add(u64::from(within))
                .saturating_mul(row_bytes)
        };
        let from = offset(page_of(run.session, at)?);
        for fork in run.forks {
            let to = offset(page_of(fork, at)?);
            backend.copy_d2d(k.slice(to, bytes)?, k.slice(from, bytes)?)?;
            backend.copy_d2d(v.slice(to, bytes)?, v.slice(from, bytes)?)?;
        }
        at = at.saturating_add(take);
    }
    Ok(())
}

impl<'a> PassRows<'a> {
    /// The page table each row reads.
    ///
    /// `None` for a segmented pass, whose runs each read their own session's
    /// table and are dispatched one at a time.
    fn pages(self) -> Option<crate::attn::Pages<'a>> {
        match self {
            Self::Chunk(session) => Some(crate::attn::Pages::shared(session.kv().view())),
            Self::Batch { pages, .. } => Some(pages),
            Self::Segments(_) => None,
        }
    }
}

/// The refusal of a whole-pass path handed a segmented pass, which the
/// branches above it dispatch run by run before it is reached.
fn segments_here() -> CoreError {
    CoreError::Unsupported {
        what: "a segmented pass on a path that reads one table or one state for every row",
        backend: "engine",
    }
}

/// Rows `first .. first + rows` of a buffer whose rows are `row_bytes` wide.
fn row_span(
    view: DeviceView<'_>,
    row_bytes: u64,
    first: u32,
    rows: u32,
) -> CoreResult<DeviceView<'_>> {
    view.slice(
        u64::from(first).saturating_mul(row_bytes),
        u64::from(rows).saturating_mul(row_bytes),
    )
}

/// One recurrent slot's block over `n_tok` rows of `io`.
fn recurrent_block(
    backend: &dyn Backend,
    kernels: &dyn BlockKernels,
    geometry: &Geometry,
    scratch: &Scratch,
    block: &Block,
    io: GdnIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    match block {
        Block::Gdn(gdn) => {
            let (gdn_geometry, gdn_scratch) = recurrent_slot(geometry, scratch)?;
            kernels.recurrent(backend, gdn, gdn_geometry, gdn_scratch, io, n_tok)
        }
        Block::Ssm(ssm) => {
            let (ssm_geometry, ssm_scratch) = state_space_slot(geometry, scratch)?;
            crate::ssm::run(backend, ssm, ssm_geometry, ssm_scratch, io, n_tok)
        }
        Block::ShortConv(short) => {
            let (short_geometry, short_scratch) = short_conv_slot(geometry, scratch)?;
            crate::short_conv::run(backend, short, short_geometry, short_scratch, io, n_tok)
        }
        Block::Attn(_) => Ok(()),
    }
}

/// Runs the whole stack over `n_tok` rows and leaves the normalized result in
/// [`Scratch::normed`].
///
/// The session is read, not advanced: it supplies the recurrent stack each
/// recurrent slot rewrites in place and the page table each attention slot
/// addresses the cache through. The caller advances it first, because the pages
/// this pass writes have to be claimed before a kernel is handed the table that
/// names them.
///
/// # The loop carries two buffers and no copies
///
/// Every block reads a normalized residual and writes an unnormalized
/// contribution, so one fused launch closes a block out: it adds the contribution
/// into the stream and normalizes the result for whatever runs next. The weight
/// that fused launch uses is therefore the *next* thing's norm — the feed-forward's
/// own norm after a block, the next slot's input norm after a feed-forward, and the
/// output norm after the last one. That is why there is no separate final norm
/// launch and no per-block copy.
///
/// # What the shape moves
///
/// Four family switches reach this loop, and each of them is a launch that
/// either happens or does not: the embedding scale, the pair of norms over a
/// block's output, the scalar the residual sum carries, and the cap on the
/// logits row. A stack that states none of them dispatches exactly the sequence
/// it dispatched before any of them existed — which is not a nicety but the
/// regression guarantee, because the text path is pinned to a recorded vector.
/// So each switch is read from [`Geometry`], never inferred from a weight being
/// present or a number being one.
///
/// `kernels` is the checkpoint family's own pair of block launchers, handed
/// down from the stack rather than reached for here: which kernels a block
/// dispatches is a fact about the family, and [`Geometry`] is a fact about the
/// widths every family of this shape shares.
///
/// # Errors
///
/// Whatever the block families report for a shape or format they cannot carry, and
/// [`CoreError::Backend`] naming the kernel when a launch fails.
#[expect(
    clippy::too_many_arguments,
    reason = "a pass takes the weights, the geometry, the scratch, the family's block kernels, the cache and the session, and a struct holding the six would name them twice"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the body IS the launch order, and the order is the thing this file exists to state; splitting it into helpers would put the sequence a reader has to check in three places"
)]
pub(crate) fn forward(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    kernels: &dyn BlockKernels,
    kv: &DeviceKv,
    rows: PassRows<'_>,
    ple: Option<&PleTable>,
    n_tok: u32,
    media: Option<MediaChunk<'_>>,
    extras: PassExtras<'_>,
) -> CoreResult<ResidualIndex> {
    let trail = extras.trail;
    let Some(first) = weights.layers.first() else {
        return Err(CoreError::ShapeMismatch {
            field: "layer count",
            expected: 1,
            got: 0,
        });
    };
    let positions = positions(geometry, scratch, media.is_some());
    let ple = per_layer_io(geometry, scratch, ple)?;
    let mut cur = 0usize;
    open(backend, weights, geometry, scratch, media, cur, n_tok)?;
    // Once per chunk, and before the first norm: every slot reads a span of the
    // block this writes, so a per-slot prepare would be the same projection
    // dispatched once per layer.
    if let Some(io) = ple.as_ref() {
        per_layer_prepare(backend, weights, geometry, scratch, io, cur, n_tok)?;
    }
    rms_norm_mul(
        backend,
        scratch.stream[cur].view(),
        first.input_norm.view(),
        scratch.normed.view(),
        geometry.hidden,
        n_tok,
        geometry.eps,
    )?;
    crate::trace::residual(
        backend,
        "embed",
        scratch.stream[cur].view(),
        geometry.hidden,
        n_tok,
    );
    // One per recurrent slot of a batched pass, refilled in place. Declared
    // here rather than inside the loop so the pass allocates once; the borrow
    // it hands out ends with each iteration.
    let mut per_row_states: Vec<GdnState<'_>> = Vec::new();
    for (index, layer) in weights.layers.iter().enumerate() {
        match &layer.block {
            Block::Gdn(_) | Block::Ssm(_) | Block::ShortConv(_) => {
                let row_bytes = u64::from(geometry.hidden).saturating_mul(launch::F32_BYTES);
                if let PassRows::Segments(runs) = rows {
                    // Each run folds its rows into its own session's state in
                    // order, exactly as its solo chunk would.
                    let mut first = 0u32;
                    for run in runs {
                        let state = run.session.gdn(layer.ordinal)?;
                        let io = GdnIo {
                            input: row_span(scratch.normed.view(), row_bytes, first, run.rows)?,
                            states: GdnStates::Chunk(GdnState {
                                conv: state.conv(),
                                state: state.state(),
                            }),
                            block_out: row_span(
                                scratch.block_out.view(),
                                row_bytes,
                                first,
                                run.rows,
                            )?,
                            trail: None,
                        };
                        recurrent_block(
                            backend,
                            kernels,
                            geometry,
                            scratch,
                            &layer.block,
                            io,
                            run.rows,
                        )?;
                        fork_state(backend, run, layer.ordinal)?;
                        first = first.saturating_add(run.rows);
                    }
                } else {
                    let states = match rows {
                        PassRows::Chunk(session) => {
                            let state = session.gdn(layer.ordinal)?;
                            GdnStates::Chunk(GdnState {
                                conv: state.conv(),
                                state: state.state(),
                            })
                        }
                        PassRows::Batch { members, .. } => {
                            // Refilled rather than rebuilt: the buffer is hoisted
                            // out of this loop, so a forty-slot pass costs one host
                            // allocation instead of forty.
                            per_row_states.clear();
                            for session in members {
                                let state = session.gdn(layer.ordinal)?;
                                per_row_states.push(GdnState {
                                    conv: state.conv(),
                                    state: state.state(),
                                });
                            }
                            GdnStates::PerRow(&per_row_states)
                        }
                        PassRows::Segments(_) => return Err(segments_here()),
                    };
                    // A trail only on a pass a rejection may undo. A prefill chunk
                    // carries no trail even on a speculative session: it would
                    // record the first few rows of a 512-row chunk, which is tens
                    // of mebibytes copied per slot for records that name positions
                    // no rollback can reach.
                    let trail = match (trail.records(), rows) {
                        (true, PassRows::Chunk(session)) => session.gdn_trail(layer.ordinal)?,
                        // A batch cannot be recorded, and the family refuses one
                        // that claims to be; nothing here builds the claim.
                        _ => None,
                    };
                    let io = GdnIo {
                        input: scratch.normed.view(),
                        states,
                        block_out: scratch.block_out.view(),
                        trail,
                    };
                    recurrent_block(backend, kernels, geometry, scratch, &layer.block, io, n_tok)?;
                }
            }
            Block::Attn(attn) => {
                let (attn_geometry, rank) = attention_slot(geometry, layer)?;
                let (k, v) = kv.layer_runs(rank)?;
                if let PassRows::Segments(runs) = rows {
                    // Each run attends through its own session's table, as
                    // its solo chunk would; a segmented pass carries no media,
                    // so its positions are the linear row.
                    let row_bytes = u64::from(geometry.hidden).saturating_mul(launch::F32_BYTES);
                    let mut first = 0u32;
                    for run in runs {
                        let cache = AttnCache {
                            k,
                            v,
                            pages: crate::attn::Pages::shared(run.session.kv().view()),
                        };
                        kernels.attention(
                            backend,
                            attn,
                            attn_geometry,
                            &scratch.attn,
                            AttnIo {
                                input: row_span(scratch.normed.view(), row_bytes, first, run.rows)?,
                                pos: Positions::Linear(row_span(
                                    scratch.positions(),
                                    ID_BYTES,
                                    first,
                                    run.rows,
                                )?),
                                bound: row_span(scratch.bound(), ID_BYTES, first, run.rows)?,
                                block_out: row_span(
                                    scratch.block_out.view(),
                                    row_bytes,
                                    first,
                                    run.rows,
                                )?,
                            },
                            &cache,
                            run.rows,
                        )?;
                        // A slot that reads another slot's cache has no rows
                        // of its own to hand on: the owner's fork carried them.
                        if attn_geometry.owns_kv && !run.forks.is_empty() {
                            let kv_row = u64::from(attn_geometry.n_head_kv)
                                .saturating_mul(u64::from(attn_geometry.head_dim))
                                .saturating_mul(F16_BYTES);
                            fork_rows(backend, (k, v), kv_row, run)?;
                        }
                        first = first.saturating_add(run.rows);
                    }
                } else {
                    let cache = AttnCache {
                        k,
                        v,
                        pages: rows.pages().ok_or_else(segments_here)?,
                    };
                    kernels.attention(
                        backend,
                        attn,
                        attn_geometry,
                        &scratch.attn,
                        AttnIo {
                            input: scratch.normed.view(),
                            pos: positions,
                            // How far along the sequence each token may look. The
                            // positions again wherever attention is causal, which
                            // is every text chunk and every slot of a family whose
                            // media spans are causal too.
                            bound: scratch.bound(),
                            block_out: scratch.block_out.view(),
                        },
                        &cache,
                        n_tok,
                    )?;
                }
            }
        }
        if crate::trace::inner() {
            crate::trace::residual(
                backend,
                &format!("  slot {index} block"),
                scratch.block_out.view(),
                geometry.hidden,
                n_tok,
            );
        }
        let mut block = norm_block_output(
            backend,
            geometry,
            scratch,
            layer.post_attn_norm.as_ref(),
            n_tok,
        )?;
        // The feed-forward half, for a slot that has one. A slot without one
        // folds its block's output in at the close below, which is the same
        // residual block the pair would have been had its feed-forward been
        // the next slot's.
        let ffn_half = slot_ffn(geometry, layer, index)?;
        if let Some((ffn, ffn_norm)) = ffn_half {
            cur = close(
                backend,
                geometry,
                scratch,
                cur,
                block,
                ffn_norm.view(),
                n_tok,
            )?;
            if crate::trace::inner() {
                crate::trace::residual(
                    backend,
                    &format!("  slot {index} ffn_in"),
                    scratch.normed.view(),
                    geometry.hidden,
                    n_tok,
                );
            }
            feed_forward(backend, geometry, scratch, layer, ffn, n_tok)?;
            if crate::trace::inner() {
                crate::trace::residual(
                    backend,
                    &format!("  slot {index} ffn_out"),
                    scratch.block_out.view(),
                    geometry.hidden,
                    n_tok,
                );
            }
            block = norm_block_output(
                backend,
                geometry,
                scratch,
                layer.post_ffn_norm.as_ref(),
                n_tok,
            )?;
        }
        // A family with a per-layer embedding table folds the feed-forward in
        // here rather than at the close, gates this slot's row against the
        // residual that leaves, and hands the close the branch's own output.
        let block = match ple.as_ref() {
            None => block,
            Some(io) => {
                if ffn_half.is_none() {
                    return Err(CoreError::Unsupported {
                        what: "a per-layer embedding branch on a slot with no feed-forward",
                        backend: "engine",
                    });
                }
                let slot = layer.ple.as_ref().ok_or(CoreError::ShapeMismatch {
                    field: "per-layer slot weights",
                    expected: 1,
                    got: 0,
                })?;
                let index = u32::try_from(index).unwrap_or(u32::MAX);
                cur = per_layer_slot(
                    backend,
                    geometry,
                    scratch,
                    io,
                    (slot, index),
                    cur,
                    block,
                    n_tok,
                )?;
                scratch.block_out.view()
            }
        };
        crate::trace::residual(
            backend,
            &format!(
                "slot {index} {}",
                match &layer.block {
                    Block::Gdn(_) => "gdn",
                    Block::Attn(_) => "attn",
                    Block::Ssm(_) => "ssm",
                    Block::ShortConv(_) => "short_conv",
                }
            ),
            scratch.stream[cur].view(),
            geometry.hidden,
            n_tok,
        );
        let next = weights
            .layers
            .get(index + 1)
            .map_or_else(|| weights.output_norm.view(), |l| l.input_norm.view());
        cur = if geometry.layer_scalar {
            close_scaled(
                backend,
                geometry,
                scratch,
                cur,
                block,
                next,
                layer.layer_scalar,
                n_tok,
            )?
        } else {
            close(backend, geometry, scratch, cur, block, next, n_tok)?
        };
        // The slot's whole output - block and feed-forward folded in - is what
        // the close just wrote, and it is what a block drafter reads.
        if let Some(sink) = extras.taps {
            let index = u32::try_from(index).unwrap_or(u32::MAX);
            if let Some(slot) = sink.layers.iter().position(|&l| l == index) {
                crate::block_draft::tap_rows(
                    backend,
                    sink,
                    u32::try_from(slot).unwrap_or(u32::MAX),
                    scratch.stream[cur].view(),
                    geometry.hidden,
                    n_tok,
                )?;
            }
        }
    }
    // Which of the two stream buffers the last close landed in. A draft head
    // reads the target's residual for the row it drafts from, and the parity of
    // thirty-two closes is not something a caller should be re-deriving: a
    // reader that guessed the other buffer would find a residual one layer old,
    // at exactly the right shape.
    Ok(ResidualIndex(cur))
}

/// What a pass carries besides its rows: whether it records a trail, and where
/// it writes a block drafter's taps.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassExtras<'a> {
    /// Whether the recurrent slots record a per-row trail.
    pub(crate) trail: Trail,
    /// The tapped layers and where their rows go, when a drafter is bound.
    pub(crate) taps: Option<crate::block_draft::TapSink<'a>>,
}

/// Normalizes a block's output, for a family that normalizes it.
///
/// Named for what it normalizes: this is the norm applied to what a block
/// produced, and [`LayerWeights::ffn_norm`] is the one the feed-forward reads.
///
/// Returns the rows the close should fold into the stream: the block's own
/// output for a [`NormScheme::PreOnly`] stack, and the normalized copy of it
/// otherwise. The two buffers swap roles rather than being copied over one
/// another, which is why this hands back a view instead of writing in place.
///
/// The scheme decides, not the presence of the weight: a stack that says it
/// carries post-norms and was loaded without them is a binding that went wrong,
/// and running it as if it were the other family would be a plausible answer
/// from the wrong architecture.
fn norm_block_output<'a>(
    backend: &dyn Backend,
    geometry: &Geometry,
    scratch: &'a Scratch,
    weight: Option<&DeviceBuffer>,
    n_tok: u32,
) -> CoreResult<DeviceView<'a>> {
    match geometry.norms {
        NormScheme::PreOnly => Ok(scratch.block_out.view()),
        NormScheme::PrePost => {
            let weight = weight.ok_or(CoreError::ShapeMismatch {
                field: "post norm weight",
                expected: 1,
                got: 0,
            })?;
            let dst = scratch
                .block_out2
                .as_ref()
                .ok_or(CoreError::ShapeMismatch {
                    field: "post norm scratch",
                    expected: 1,
                    got: 0,
                })?
                .view();
            rms_norm_mul(
                backend,
                scratch.block_out.view(),
                weight.view(),
                dst,
                geometry.hidden,
                n_tok,
                geometry.eps,
            )?;
            Ok(dst)
        }
    }
}

/// The recurrent widths and scratch a recurrent slot runs over.
///
/// A recurrent slot in a stack whose geometry has none is a shape and a weight
/// list that disagree, which is a refusal rather than an allocation made here on
/// the spot.
fn recurrent_slot<'a>(
    geometry: &'a Geometry,
    scratch: &'a Scratch,
) -> CoreResult<(&'a GdnGeometry, &'a GdnScratch)> {
    match (geometry.gdn.as_ref(), scratch.gdn.as_ref()) {
        (Some(gdn), Some(rows)) => Ok((gdn, rows)),
        _ => Err(CoreError::ShapeMismatch {
            field: "recurrent geometry",
            expected: 1,
            got: 0,
        }),
    }
}

/// The state-space widths and scratch a state-space slot runs over, or the
/// refusal for a stack whose geometry has none.
fn state_space_slot<'a>(
    geometry: &'a Geometry,
    scratch: &'a Scratch,
) -> CoreResult<(&'a SsmGeometry, &'a SsmScratch)> {
    match (geometry.ssm.as_ref(), scratch.ssm.as_ref()) {
        (Some(ssm), Some(rows)) => Ok((ssm, rows)),
        _ => Err(CoreError::ShapeMismatch {
            field: "state-space geometry",
            expected: 1,
            got: 0,
        }),
    }
}

/// The short-convolution widths and scratch a short-convolution slot runs
/// over, or the refusal for a stack whose geometry has none.
fn short_conv_slot<'a>(
    geometry: &'a Geometry,
    scratch: &'a Scratch,
) -> CoreResult<(&'a ShortConvGeometry, &'a ShortConvScratch)> {
    match (geometry.short_conv.as_ref(), scratch.short_conv.as_ref()) {
        (Some(short), Some(rows)) => Ok((short, rows)),
        _ => Err(CoreError::ShapeMismatch {
            field: "short-convolution geometry",
            expected: 1,
            got: 0,
        }),
    }
}

/// A slot's feed-forward and the norm it reads, or `None` for a slot the
/// shape says has none.
///
/// The shape decides, not the weights: a slot the shape gives a feed-forward
/// and the loader bound without one is refused rather than run as a bare
/// block, and the other way round is refused too.
fn slot_ffn<'a>(
    geometry: &Geometry,
    layer: &'a LayerWeights,
    index: usize,
) -> CoreResult<Option<(&'a Ffn, &'a DeviceBuffer)>> {
    let wants = geometry.has_ffn.get(index).copied().unwrap_or(true);
    match (wants, layer.ffn.as_ref(), layer.ffn_norm.as_ref()) {
        (true, Some(ffn), Some(norm)) => Ok(Some((ffn, norm))),
        (false, None, None) => Ok(None),
        (_, ffn, _) => Err(CoreError::ShapeMismatch {
            field: "slot feed-forward weights",
            expected: u64::from(wants),
            got: u64::from(ffn.is_some()),
        }),
    }
}

/// One attention slot's own widths, and the rank of the cache runs it reads.
///
/// Two different numbers: the ordinal picks the slot's own widths, and the run
/// rank picks the cache allocation it reads, which for a slot that shares a
/// cache is an earlier slot's.
fn attention_slot<'a>(
    geometry: &'a Geometry,
    layer: &LayerWeights,
) -> CoreResult<(&'a AttnGeometry, u32)> {
    let widths = geometry
        .attn
        .get(layer.ordinal as usize)
        .ok_or(CoreError::ShapeMismatch {
            field: "attention geometry",
            expected: u64::from(layer.ordinal) + 1,
            got: u64::try_from(geometry.attn.len()).unwrap_or(u64::MAX),
        })?;
    let rank = layer.kv_rank.ok_or(CoreError::ShapeMismatch {
        field: "kv run rank",
        expected: 1,
        got: 0,
    })?;
    Ok((widths, rank))
}

/// Folds `block` into the stream and normalizes the result with `weight`.
///
/// `block` is the rows a block produced, which for a family that normalizes its
/// blocks' outputs is the normalized copy rather than the block's own buffer.
///
/// Returns the index of the stream buffer the sum landed in, which is the one the
/// next block adds into.
fn close(
    backend: &dyn Backend,
    geometry: &Geometry,
    scratch: &Scratch,
    cur: usize,
    block: DeviceView<'_>,
    weight: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<usize> {
    let next = 1 - cur;
    add_rms_norm_mul(
        backend,
        scratch.stream[cur].view(),
        block,
        weight,
        scratch.stream[next].view(),
        scratch.normed.view(),
        geometry.hidden,
        n_tok,
        geometry.eps,
    )?;
    Ok(next)
}

/// [`close`] with the block's contribution scaled before it is folded in.
///
/// Returns the index of the stream buffer the sum landed in, which is the one the
/// next block adds into — the same contract [`close`] keeps, so a family that
/// scales and one that does not differ in the call and not in what follows it.
///
/// A family whose block output carries a per-layer scalar closes a block out with
/// this: the scalar multiplies the residual sum, which is both what the stream
/// carries forward and what the normalization measures.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the hidden width is not a whole number of
/// `float4`, and [`CoreError::Backend`] when the launch fails.
#[expect(
    clippy::too_many_arguments,
    reason = "the rows a block produced and the norm the next one reads are two \
              different owners' buffers, and the scale is a third thing again"
)]
pub(crate) fn close_scaled(
    backend: &dyn Backend,
    geometry: &Geometry,
    scratch: &Scratch,
    cur: usize,
    block: DeviceView<'_>,
    weight: DeviceView<'_>,
    scale: f32,
    n_tok: u32,
) -> CoreResult<usize> {
    let next = 1 - cur;
    add_scale_rms_norm_mul(
        backend,
        scratch.stream[cur].view(),
        block,
        weight,
        scratch.stream[next].view(),
        scratch.normed.view(),
        scale,
        geometry.hidden,
        n_tok,
        geometry.eps,
    )?;
    Ok(next)
}

/// `row[i] = cap * tanh(row[i] / cap)` over `n` elements, in place.
///
/// Gemma's logit softcap: a smooth clamp that leaves the middle of the range
/// nearly untouched and bounds either tail, applied to a logits row before it is
/// sampled from. In place because the row it caps is the row the sampler reads,
/// and a copy would be a second buffer the size of the vocabulary.
///
/// # Errors
///
/// [`CoreError::Backend`] naming the kernel when the launch fails.
pub(crate) fn softcap_row(
    backend: &dyn Backend,
    row: DeviceView<'_>,
    n: u32,
    cap: f32,
) -> CoreResult<()> {
    let args = SoftcapArgs { n, cap };
    launch::run(
        backend,
        KernelRef::new(launch::DENSE, "kernel_dense_softcap_f32"),
        Extent3::linear(n.div_ceil(launch::BLOCK_LINEAR)),
        Extent3::new(launch::BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(row),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_dense_softcap_f32", err))
}

/// `rows[i] *= s` over `n` elements, in place.
///
/// Flat over the whole run rather than per row: the embedding scale multiplies
/// every gathered row of a chunk by the same number, so the launch counts
/// elements and never needs to know where one row ends.
///
/// # Errors
///
/// [`CoreError::Backend`] naming the kernel when the launch fails.
pub(crate) fn scale_rows(
    backend: &dyn Backend,
    rows: DeviceView<'_>,
    n: u32,
    s: f32,
) -> CoreResult<()> {
    let args = ScaleArgs { n, s };
    launch::run(
        backend,
        KernelRef::new(launch::DENSE, "kernel_dense_scale_f32"),
        Extent3::linear(n.div_ceil(launch::BLOCK_LINEAR)),
        Extent3::new(launch::BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(rows),
        ],
    )
    .map_err(|err| launch::launch_failure("kernel_dense_scale_f32", err))
}

/// Projects normalized rows `0 .. n` into logits rows `0 .. n` in one call.
///
/// A verify pass's rows are contiguous in both blocks, so its logits are one
/// projection over `n` rows - which reads the vocabulary matrix once - rather
/// than `n` projections of one row that read it `n` times. The rows' bits are
/// the one-row call's: the projection is row-exact.
///
/// # Errors
///
/// As [`logits_row`].
pub(crate) fn logits_rows(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    n: u32,
) -> CoreResult<()> {
    let hidden = u64::from(geometry.hidden).saturating_mul(launch::F32_BYTES);
    let vocab = u64::from(geometry.vocab).saturating_mul(launch::F32_BYTES);
    let src = scratch
        .normed
        .slice(0, u64::from(n).saturating_mul(hidden))?;
    let dst = scratch
        .logits
        .slice(0, u64::from(n).saturating_mul(vocab))?;
    projection::projection(backend, weights.output_matrix(), n, src, dst)?;
    if let Some(cap) = geometry.final_softcap {
        for row in 0..n {
            softcap_row(
                backend,
                dst.slice(u64::from(row).saturating_mul(vocab), vocab)?,
                geometry.vocab,
                cap,
            )?;
        }
    }
    Ok(())
}

/// Projects one normalized row into one logits row, capped where the family caps
/// it.
///
/// One row at a time on purpose: a prefill wants the last row of its chunk and a
/// decode step wants one row per session, and neither is a contiguous run of the
/// normalized block. The projection family covers a single row, so the caller
/// slices and this launches.
///
/// The cap belongs here rather than at the sampler: it is a fact about the
/// model's output layer, and a row that reached a caller uncapped would be a row
/// every sampler downstream had to remember to cap.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] when either slice is shorter than a row, and whatever
/// the projection family reports.
pub(crate) fn logits_row(
    backend: &dyn Backend,
    weights: &Weights,
    geometry: &Geometry,
    scratch: &Scratch,
    token: u32,
    row: u32,
) -> CoreResult<()> {
    let hidden = u64::from(geometry.hidden).saturating_mul(launch::F32_BYTES);
    let vocab = u64::from(geometry.vocab).saturating_mul(launch::F32_BYTES);
    let src = scratch
        .normed
        .slice(u64::from(token).saturating_mul(hidden), hidden)?;
    let dst = scratch
        .logits
        .slice(u64::from(row).saturating_mul(vocab), vocab)?;
    projection::projection(backend, weights.output_matrix(), 1, src, dst)?;
    if let Some(cap) = geometry.final_softcap {
        softcap_row(backend, dst, geometry.vocab, cap)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::EmbedKernel;
    use crate::launch;
    use shiso_core::DType;

    #[test]
    fn the_iq1_m_embedding_reads_the_generic_gather() {
        // The Saluki export stores its vocabulary table in `iq1_m`, so the
        // gather is the one entry point that format has; the pair is what the
        // load records and the launch replays, and a misspelling here is a
        // driver failure at the first embed, not a compile error.
        let kernel = EmbedKernel::for_dtype(DType::IQ1_M).expect("the iq1_m gather exists");
        assert_eq!(kernel.entry, "kernel_get_rows_iq1_m_f32");
        assert_eq!(kernel.module, launch::GENERIC);
    }
}
