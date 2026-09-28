//! The routed feed-forward, for a dense-hybrid stack whose shape carries a
//! router.
//!
//! The dense feed-forward in [`crate::ffn`] is two projections up, a gated
//! activation and one projection back, at the same width in every slot. This is
//! the same block with the three matrices chosen per token: a router scores
//! every expert, the best handful become *columns*, the columns run through
//! their own experts, and the results are recombined into one contribution per
//! token with an always-on shared expert added on top.
//!
//! # Where this sits, and what it is not
//!
//! It is **not** `crate::sparse::moe`. That one is the same block for a stack
//! with several residual streams and an indexer, expressed in the sparse
//! module's plan/IR and executed by `sparse::exec`; this is the same block for
//! the dense layer loop, issued as direct launches like everything else here.
//! The two share every kernel that does not touch a weight - see [`Q4E_MOE`] -
//! and share no host code, because the two execution paths have nothing in
//! common below the launch.
//!
//! # The eight launches, and the four conventions that tie them together
//!
//! Every seam below is a convention neither kernel can check:
//!
//! 1. a routed **column** is `token * used + slot`, so `col / used` is the
//!    token whose activation row the gate and up projections read;
//! 2. the down projection reads its own column's row of the activated
//!    intermediate, not its token's;
//! 3. the down projection writes `[column, hidden]` row-major, which is
//!    column-major over (token, slot), because that is what the combine reads;
//! 4. the per-expert scale is an **array** indexed by expert id, not one
//!    number for the layer.
//!
//! `crates/shiso-kerneltest/src/qwen35moe/experts.rs` grades all four on a
//! device, stage by stage.

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3};

use crate::ffn::Ffn;
use crate::launch::{self, F32_BYTES, Q35MOE};
use crate::projection::{self, Weight, WeightAux};
use crate::shape::MoeShape;

/// The sparse family's module, whose six general entries this block reaches
/// unchanged.
///
/// A module name is the key of a compiled image and says nothing about which
/// family may name it; `crate::launch` has a table of exactly these decisions.
/// What makes these six safe to share is not the name but the fidelity gate:
/// they are extracted verbatim from the reference engine and
/// `scripts/q4e_ptx_oracle.sh --mode=fidelity` refuses a tree in which any of
/// them has changed. Re-implementing them would have put the renormalization
/// clamp, the argmax tie-break and the shared expert's sigmoid gate back in
/// play, which is where a re-implementation goes quietly wrong.
pub const Q4E_MOE: &str = "q4e_moe";

/// Threads per block of the linear entries, matching `sparse::kernels`.
const BLOCK_LINEAR: u32 = 256;

/// Threads per block of the two expert entries, `SHISO_Q35MOE_EXPERT_BLOCK`.
const EXPERT_BLOCK: u32 = 128;

/// Output rows one expert block covers, `SHISO_Q35MOE_EXPERT_WARPS`.
const EXPERT_ROWS: u32 = 4;

/// Lanes the serial scan runs in, which is one warp.
const SCAN_BLOCK: u32 = 32;

/// The four widths a routed expert stage is launched over.
///
/// Sixteen bytes, one struct by value, which is what the entries declare -
/// `.param .align 4 .b8 ..._param_0[16]` in the generated PTX.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExpertArgs {
    in_dim: i32,
    out_dim: i32,
    n_expert: i32,
    n_used: i32,
}
const _: () = assert!(size_of::<ExpertArgs>() == 16);

/// One layer's routed feed-forward, resident on the device.
///
/// The three expert stacks are **slabs**: expert `e`'s matrix begins at
/// `e * out_dim * in_dim / 2` bytes into the weight slab and at
/// `e * out_dim * in_dim / 16` into the scale slab, and its per-tensor scale
/// is `globals[e]`. The checkpoint stores 256 separate tensors per projection;
/// stacking them is the loader's job, and the entries index them by stride.
///
/// Those strides are the NVFP4 codec's; [`ExpertSlab`] is an enum over the two
/// codecs, and [`Moe::codec`] is the one all three stacks are in.
///
/// # Why every field is private
///
/// The codec is not a free-standing label: [`run`] picks the entry from it
/// while the slab-to-views helper hands over three views whatever variant it
/// finds, so a block whose codec and whose slabs disagree launches an NVFP4
/// entry against affine buffers. That entry reads per-group `bfloat` biases
/// as the `globals[expert]`
/// `f32` array, finishes, and writes plausible numbers - which is the exact
/// failure the [`ExpertSlab`] enum split exists to prevent, one level up.
///
/// A public `codec` field, or public slab fields beside a private one, would
/// both allow it. So the invariant is held by construction: [`Moe::new`] is the
/// only door, it DERIVES the codec from the stacks rather than taking it, and
/// it refuses a block whose stacks disagree.
#[derive(Debug)]
pub struct Moe {
    /// `hidden -> experts`, which scores every expert for every token.
    router: Weight,
    /// The routed experts' gate projection, as one slab.
    gate: ExpertSlab,
    /// The routed experts' up projection.
    up: ExpertSlab,
    /// The routed experts' down projection.
    down: ExpertSlab,
    /// `hidden -> 1`: the scalar the shared expert's output is gated by.
    ///
    /// The shared expert itself is NOT here. It is an ordinary dense
    /// feed-forward, so it is the slot's own `LayerWeights::ffn` and [`run`]
    /// takes it as an argument: a copy in this struct would be the same three
    /// matrices resident twice, which at forty slots is about seventy
    /// megabytes of the device for nothing.
    shared_gate: Weight,
    /// The stored format of the gate and up stacks, derived from them.
    ///
    /// **Per STAGE rather than per block**, and that is not a generalization
    /// for its own sake: `unsloth`'s `Qwen3.6-35B-A3B-UD-Q4_K_XL` stores every
    /// expert's gate and up in `Q4_K` and every down in `Q5_K`, so a block of
    /// that container has two codecs and the old single field refused it -
    /// "down is q5k where gate is q4k" - before it could be loaded.
    ///
    /// Gate and up still have to agree with each other, because they are ONE
    /// launch and the codec is a template argument of it rather than a value it
    /// reads.
    gate_up_codec: ExpertCodec,
    /// The stored format of the down stack.
    down_codec: ExpertCodec,
    /// The float the routed activations are held in.
    dtype: ActDtype,
}

impl Moe {
    /// Builds one layer's routed feed-forward from stacks that agree.
    ///
    /// The codec is **derived** from `gate` rather than taken as an argument,
    /// so there is no way to state one that the stacks do not have.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the stage and both codecs when `up`
    /// or `down` is stored differently from `gate`, naming the pair when no
    /// entry was built for this codec in this activation dtype, and saying why
    /// when `dtype` is [`ActDtype::Bf16`]: those entries exist and are graded,
    /// but [`MoeScratch`] is f32-only, so nothing can drive them end to end
    /// yet and a block that tried would finish and write nonsense rather than
    /// fault. The second is checked here rather than only at the launch so
    /// that a stack nothing can run is refused while the checkpoint is being
    /// read, rather than after five launches of a decode have already been
    /// issued.
    pub fn new(
        router: Weight,
        gate: ExpertSlab,
        up: ExpertSlab,
        down: ExpertSlab,
        shared_gate: Weight,
        dtype: ActDtype,
    ) -> CoreResult<Self> {
        let gate_up_codec = gate.codec();
        let down_codec = down.codec();
        // Gate and up alone, because they are one launch: the codec is that
        // entry's template argument, so a pair that disagreed would need two
        // entries rather than a wider one. Down is its own launch and is free
        // to differ, which is what the `UD-Q4_K_XL` container needs.
        if up.codec() != gate_up_codec {
            return Err(CoreError::InvalidValue {
                key: "routed expert stack codec",
                value: format!(
                    "up is {} where gate is {}, and the two share one launch",
                    up.codec().name(),
                    gate_up_codec.name()
                ),
            });
        }
        for (stage, codec) in [
            (ExpertStageKind::GateUp, gate_up_codec),
            (ExpertStageKind::Down, down_codec),
        ] {
            if expert_entry(stage, codec, dtype).is_none() {
                return Err(unbuilt(codec, dtype));
            }
        }
        if matches!(dtype, ActDtype::Bf16) {
            return Err(undriveable(gate_up_codec));
        }
        Ok(Self {
            router,
            gate,
            up,
            down,
            shared_gate,
            gate_up_codec,
            down_codec,
            dtype,
        })
    }

    /// The codec the gate and up stacks are stored in.
    #[must_use]
    pub const fn codec(&self) -> ExpertCodec {
        self.gate_up_codec
    }

    /// The codec the down stack is stored in, which need not be the same.
    #[must_use]
    pub const fn down_codec(&self) -> ExpertCodec {
        self.down_codec
    }

    /// The float the routed activations are held in.
    #[must_use]
    pub const fn dtype(&self) -> ActDtype {
        self.dtype
    }
}

/// Which of the two expert stages an entry runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertStageKind {
    /// Gate and up together, into two outputs.
    GateUp,
    /// Down, back to the residual width.
    Down,
}

/// The stored format of an expert stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCodec {
    /// Four-bit NVFP4 with an `e4m3` scale per sixteen and an `f32` per expert.
    Nvfp4,
    /// Four-bit affine with a scale and a bias per group of sixty-four.
    Q4Affine,
    /// `ggml` `Q4_K`: 256-element super-blocks, self-describing.
    Q4K,
    /// `ggml` `Q5_K`, which is `Q4_K` plus a fifth-bit plane.
    Q5K,
    /// `ggml` `Q6_K`: 256-element super-blocks, six bits a weight.
    Q6K,
    /// `ggml` `Q8_0`: 32-element blocks, one `f16` scale each.
    Q8_0,
}

impl ExpertCodec {
    /// The codec's name as the entry points spell it.
    ///
    /// The same token the kernel names are built from, so a refusal that
    /// interpolates it names something an operator can grep the module for.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4",
            Self::Q4Affine => "q4affine",
            Self::Q4K => "q4k",
            Self::Q5K => "q5k",
            Self::Q6K => "q6k",
            Self::Q8_0 => "q8_0",
        }
    }

    /// Whether this codec's slab is self-describing: one buffer, no sidecars.
    ///
    /// The `ggml` super-block formats carry their scales inside the block. The
    /// other two carry them in buffers of their own, which is what makes the
    /// count vary and why [`ExpertSlab::buffer_count`] is a method rather than
    /// the constant it used to be.
    #[must_use]
    pub const fn is_block(self) -> bool {
        matches!(self, Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8_0)
    }
}

/// The float the activations crossing a launch boundary are held in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActDtype {
    /// Four-byte float, which is what the CUDA lane runs.
    F32,
    /// Two-byte brain float, which is what the Metal lane runs.
    Bf16,
}

impl ActDtype {
    /// The dtype's name as the entry points spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

/// The entry point for one (stage, codec, dtype), or `None` when that
/// combination was never built.
///
/// Three of the four combinations exist. `Nvfp4` in `Bf16` is not one of them:
/// nothing asks for it, and a template instantiation nothing asks for is a
/// kernel nobody grades.
#[must_use]
pub const fn expert_entry(
    stage: ExpertStageKind,
    codec: ExpertCodec,
    dtype: ActDtype,
) -> Option<&'static str> {
    match (stage, codec, dtype) {
        (ExpertStageKind::GateUp, ExpertCodec::Nvfp4, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_nvfp4_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Nvfp4, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_nvfp4_f32")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q4Affine, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_q4affine_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Q4Affine, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_q4affine_f32")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q4Affine, ActDtype::Bf16) => {
            Some("kernel_q35moe_experts_gate_up_q4affine_bf16")
        }
        (ExpertStageKind::Down, ExpertCodec::Q4Affine, ActDtype::Bf16) => {
            Some("kernel_q35moe_experts_down_q4affine_bf16")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q4K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_q4k_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Q4K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_q4k_f32")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q5K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_q5k_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Q5K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_q5k_f32")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q6K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_q6k_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Q6K, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_q6k_f32")
        }
        (ExpertStageKind::GateUp, ExpertCodec::Q8_0, ActDtype::F32) => {
            Some("kernel_q35moe_experts_gate_up_q8_0_f32")
        }
        (ExpertStageKind::Down, ExpertCodec::Q8_0, ActDtype::F32) => {
            Some("kernel_q35moe_experts_down_q8_0_f32")
        }
        // NVFP4 has no `bf16` entry and the three super-block codecs have none
        // either, for two different reasons that land in one arm. NVFP4's is
        // historical - nothing asked. The super-block entries exist for one
        // container whose activations are `f32`, and a `bf16` arm nothing
        // drives would be an entry nothing grades.
        (
            _,
            ExpertCodec::Nvfp4
            | ExpertCodec::Q4K
            | ExpertCodec::Q5K
            | ExpertCodec::Q6K
            | ExpertCodec::Q8_0,
            ActDtype::Bf16,
        ) => None,
    }
}

/// One projection of every expert in a layer, stacked, in one of two codecs.
///
/// The third buffer is not the same quantity in the two: NVFP4 carries one
/// `f32` per expert, affine carries one bias per group. A struct with both
/// fields would let a slab built through the wrong constructor present a
/// buffer of the right size holding something else.
///
/// The variant is what [`Moe::new`] reads to derive a block's codec, so the
/// enum is not only documentation: it is the only thing that knows which
/// quantity the third buffer holds, and [`codec`](Self::codec) is how that
/// knowledge leaves the type.
#[derive(Debug)]
pub enum ExpertSlab {
    /// NVFP4: codes, `e4m3` scales per sixteen, one `f32` per expert.
    Nvfp4 {
        /// Packed four-bit codes, `[experts, out_dim, in_dim / 2]`.
        weights: DeviceBuffer,
        /// One `e4m3` byte per group of sixteen,
        /// `[experts, out_dim, in_dim / 16]`.
        scales: DeviceBuffer,
        /// One `f32` per expert. **An array, not a scalar**: every expert of a
        /// layer has its own `weight_scale_2` and they differ, so a kernel
        /// reading expert zero's for all of them would be wrong by whatever
        /// the spread is.
        globals: DeviceBuffer,
    },
    /// A `ggml` super-block slab: ONE buffer, because the format carries its
    /// scales inside each block.
    ///
    /// The codec is a field rather than three variants because nothing else
    /// about the three differs here - the loader hands over bytes and the
    /// kernel name is what the codec selects. It is the one variant whose
    /// buffer count is not three.
    Block {
        /// Packed super-blocks, `[experts, out_dim, in_dim / elements_per_block]`.
        weights: DeviceBuffer,
        /// Which super-block format those bytes are.
        codec: ExpertCodec,
    },
    /// Affine group-64: codes, a scale per group, a bias per group.
    Q4Affine {
        /// Packed four-bit codes, `[experts, out_dim, in_dim / 2]`.
        weights: DeviceBuffer,
        /// One `bfloat` per group of sixty-four.
        scales: DeviceBuffer,
        /// One `bfloat` per group of sixty-four.
        biases: DeviceBuffer,
    },
}

impl ExpertSlab {
    /// Buffers this slab contributes to a launch, in kernel order.
    ///
    /// **Not a constant.** It was `3` for every slab, which was true while
    /// every codec kept its scales in buffers of their own. A `ggml`
    /// super-block carries them inside the block, so its slab is one buffer and
    /// its entries take one pointer per stack; a constant three would hand
    /// those entries two pointers they do not declare.
    #[must_use]
    pub const fn buffer_count(&self) -> usize {
        match self {
            Self::Block { .. } => 1,
            Self::Nvfp4 { .. } | Self::Q4Affine { .. } => 3,
        }
    }

    /// This slab's codec.
    #[must_use]
    pub const fn codec(&self) -> ExpertCodec {
        match self {
            Self::Nvfp4 { .. } => ExpertCodec::Nvfp4,
            Self::Q4Affine { .. } => ExpertCodec::Q4Affine,
            Self::Block { codec, .. } => *codec,
        }
    }
}

/// The rows a routed block reads and writes for one chunk.
#[derive(Debug, Clone, Copy)]
pub struct MoeIo<'a> {
    /// Normalized residual rows, `[n_tok, hidden]` f32.
    pub input: DeviceView<'a>,
    /// The block's contribution to the stream, `[n_tok, hidden]` f32.
    pub block_out: DeviceView<'a>,
}

/// Scratch a routed block needs, sized once per stack.
///
/// Eleven buffers, and every one of them is per chunk rather than per layer:
/// at the widest chunk the routed columns alone are 4,096 rows of the
/// intermediate width, which is megabytes that would otherwise be allocated
/// forty times over for buffers that never outlive a launch.
#[derive(Debug)]
pub struct MoeScratch {
    /// The router's logit block, `[n_tok, experts]`.
    pub(crate) logits: DeviceBuffer,
    /// One expert id per routed column, `int32`.
    pub(crate) ids: DeviceBuffer,
    /// One routing weight per routed column.
    pub(crate) weights: DeviceBuffer,
    /// Per-expert column counts; also the scatter's cursor.
    pub(crate) counts: DeviceBuffer,
    /// The scatter's cursor, kept apart from the histogram.
    pub(crate) cursor: DeviceBuffer,
    /// Exclusive prefix sums over the counts, `experts + 1` of them.
    pub(crate) bounds: DeviceBuffer,
    /// The experts that received a column, with the count at `[experts]`.
    pub(crate) active: DeviceBuffer,
    /// The routed columns in expert order.
    pub(crate) sorted: DeviceBuffer,
    /// The routed gate projection, `[columns, intermediate]`.
    pub(crate) gate: DeviceBuffer,
    /// The routed up projection.
    pub(crate) up: DeviceBuffer,
    /// `silu(gate) * up`.
    pub(crate) mid: DeviceBuffer,
    /// The routed down projection, `[columns, hidden]`.
    pub(crate) down: DeviceBuffer,
    /// The shared expert's scalar logit, one per token.
    pub(crate) shared_logit: DeviceBuffer,
    /// The shared expert's output, `[n_tok, hidden]`.
    pub(crate) shared_out: DeviceBuffer,
}

impl MoeScratch {
    /// Allocates every row a chunk of `rows` tokens needs.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &std::sync::Arc<dyn Backend>,
        moe: MoeShape,
        hidden: u32,
        rows: u32,
    ) -> CoreResult<Self> {
        let tokens = u64::from(rows);
        let columns = moe.slot_rows(rows);
        let ids = u64::from(shiso_core::DType::I32.block_bytes().get());
        let floats = |n: u64| n.saturating_mul(F32_BYTES);
        Ok(Self {
            logits: alloc(backend, floats(tokens * u64::from(moe.experts)))?,
            ids: alloc(backend, columns.saturating_mul(ids))?,
            weights: alloc(backend, floats(columns))?,
            counts: alloc(backend, u64::from(moe.experts).saturating_mul(ids))?,
            cursor: alloc(backend, u64::from(moe.experts).saturating_mul(ids))?,
            bounds: alloc(backend, u64::from(moe.experts + 1).saturating_mul(ids))?,
            active: alloc(backend, u64::from(moe.experts + 1).saturating_mul(ids))?,
            sorted: alloc(backend, columns.saturating_mul(ids))?,
            gate: alloc(backend, floats(columns * u64::from(moe.intermediate)))?,
            up: alloc(backend, floats(columns * u64::from(moe.intermediate)))?,
            mid: alloc(backend, floats(columns * u64::from(moe.intermediate)))?,
            down: alloc(backend, floats(columns * u64::from(hidden)))?,
            shared_logit: alloc(backend, floats(tokens))?,
            shared_out: alloc(backend, floats(tokens * u64::from(hidden)))?,
        })
    }
}

/// Allocates a device buffer of `bytes`.
fn alloc(backend: &std::sync::Arc<dyn Backend>, bytes: u64) -> CoreResult<DeviceBuffer> {
    shiso_core::alloc(backend, bytes.max(F32_BYTES))
}

/// Runs one routed feed-forward over `n_tok` rows of `io.input`.
///
/// # Errors
///
/// Whatever [`projection::projection`] reports for a weight it cannot carry,
/// [`CoreError::ShapeMismatch`] when the shape and the weights disagree about
/// the expert count, [`CoreError::InvalidValue`] naming the pair when no expert
/// entry was built for this block's codec and activation dtype, and
/// [`CoreError::Backend`] naming the entry when a launch fails.
pub fn run(
    backend: &dyn Backend,
    moe: &Moe,
    shared: &Ffn,
    shape: MoeShape,
    scratch: &MoeScratch,
    io: MoeIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    if moe.router.out_dim != shape.experts {
        return Err(CoreError::ShapeMismatch {
            field: "router output width",
            expected: u64::from(shape.experts),
            got: u64::from(moe.router.out_dim),
        });
    }
    let hidden = moe.router.in_dim;
    let columns = u32::try_from(shape.slot_rows(n_tok)).unwrap_or(u32::MAX);

    // The router's own stage: one projection, then the selection.
    projection::projection(backend, &moe.router, n_tok, io.input, scratch.logits.view())?;
    route(backend, shape, scratch, n_tok)?;
    sort(backend, shape, scratch, columns)?;

    // The experts' stage: gate and up together, the activation, then down.
    expert_stage(
        backend,
        ExpertStage {
            entry: entry_for(ExpertStageKind::GateUp, moe)?,
            args: ExpertArgs {
                in_dim: extent(hidden)?,
                out_dim: extent(shape.intermediate)?,
                n_expert: extent(shape.experts)?,
                n_used: extent(shape.used)?,
            },
            columns,
            experts: shape.experts,
            out_dim: shape.intermediate,
        },
        &[
            slab(&moe.gate).as_slice(),
            slab(&moe.up).as_slice(),
            [io.input].as_slice(),
            [
                scratch.sorted.view(),
                scratch.bounds.view(),
                scratch.active.view(),
                scratch.gate.view(),
                scratch.up.view(),
            ]
            .as_slice(),
        ]
        .concat(),
    )?;
    activate(
        backend,
        scratch.mid.view(),
        scratch.gate.view(),
        scratch.up.view(),
        u64::from(columns) * u64::from(shape.intermediate),
    )?;
    expert_stage(
        backend,
        ExpertStage {
            entry: entry_for(ExpertStageKind::Down, moe)?,
            args: ExpertArgs {
                in_dim: extent(shape.intermediate)?,
                out_dim: extent(hidden)?,
                n_expert: extent(shape.experts)?,
                n_used: extent(shape.used)?,
            },
            columns,
            experts: shape.experts,
            out_dim: hidden,
        },
        &[
            slab(&moe.down).as_slice(),
            [
                scratch.mid.view(),
                scratch.sorted.view(),
                scratch.bounds.view(),
                scratch.active.view(),
                scratch.down.view(),
            ]
            .as_slice(),
        ]
        .concat(),
    )?;

    // One contribution per token, then the shared expert on top of it.
    combine(backend, shape, scratch, io.block_out, hidden, n_tok)?;
    add_shared(backend, moe, shared, scratch, io, hidden, n_tok)
}

/// The entry one stage of `moe` resolves to, or a refusal naming the pair.
///
/// [`Moe::new`] has already refused an unbuilt pair, so reaching this is either
/// a block assembled inside this module or a combination that gained a caller
/// before it gained a kernel. It stays because the alternative to refusing here
/// is launching an entry name that no image holds, and a `Moe` is cheap to
/// build wrongly from inside the crate.
fn entry_for(stage: ExpertStageKind, moe: &Moe) -> CoreResult<&'static str> {
    let codec = match stage {
        ExpertStageKind::GateUp => moe.gate_up_codec,
        ExpertStageKind::Down => moe.down_codec,
    };
    expert_entry(stage, codec, moe.dtype).ok_or_else(|| unbuilt(codec, moe.dtype))
}

/// The refusal for a (codec, dtype) nothing was built for, naming the pair.
///
/// [`CoreError::InvalidValue`] rather than [`CoreError::Unsupported`] because
/// the latter's `what` is a `&'static str` and so cannot hold the pair: an
/// operator would read "this codec and activation dtype" and have to go to the
/// source to learn which two. The point of the refusal is that it names them.
fn unbuilt(codec: ExpertCodec, dtype: ActDtype) -> CoreError {
    CoreError::InvalidValue {
        key: "routed expert stage",
        value: format!(
            "{}/{} was never built; nothing asks for it",
            codec.name(),
            dtype.name()
        ),
    }
}

/// The refusal for a block whose entries exist but whose scratch cannot feed
/// them, which today is every [`ActDtype::Bf16`] block.
///
/// The two `bf16` expert entries are compiled and graded -
/// `crates/shiso-kerneltest/src/qwen35moe/q4affine.rs` measures both against
/// the same host reference the `f32` pair is measured against - but nothing can
/// drive them end to end yet, and the way they fail is the bad way:
/// [`MoeScratch::new`] sizes `gate`, `up`, `mid` and `down` at four bytes an
/// element, while `q4e_swiglu_kernel` and `q4e_moe_combine_kernel` take
/// `float *`. An `f32` buffer is **twice** the bytes a `bf16` stage needs, so
/// every access stays in bounds: the expert entry would read `f32` bit patterns
/// as `bfloat`, the activation would read `bfloat` output as `f32`, and the
/// block would finish without a fault and write plausible nonsense.
///
/// So this is refused rather than half-wired. Driving it properly needs a
/// dtype-aware [`MoeScratch`], `bf16` variants of the activation and the
/// combine, and a decision about `MoeIo::input`'s own dtype - a task, not a
/// patch, and one that starts from entries that are already graded.
fn undriveable(codec: ExpertCodec) -> CoreError {
    CoreError::InvalidValue {
        key: "routed expert stage",
        value: format!(
            "{}/{} entries exist and are graded, but the routed scratch is f32-only \
             (MoeScratch sizes gate/up/mid/down at 4 bytes an element and the \
             activation and combine entries take float*), so the pair cannot be \
             driven end to end yet",
            codec.name(),
            ActDtype::Bf16.name(),
        ),
    }
}

/// The views one expert slab contributes to a launch, in kernel order.
///
/// A `Vec` rather than a fixed array: a super-block slab contributes one view
/// and the other two contribute three. See [`ExpertSlab::buffer_count`].
fn slab(slab: &ExpertSlab) -> Vec<DeviceView<'_>> {
    match slab {
        ExpertSlab::Block { weights, .. } => vec![weights.view()],
        ExpertSlab::Nvfp4 {
            weights,
            scales,
            globals,
        } => vec![weights.view(), scales.view(), globals.view()],
        ExpertSlab::Q4Affine {
            weights,
            scales,
            biases,
        } => vec![weights.view(), scales.view(), biases.view()],
    }
}

/// One dimension, as the entries spell it.
fn extent(value: u32) -> CoreResult<i32> {
    i32::try_from(value).map_err(|_| CoreError::ShapeMismatch {
        field: "routed feed-forward width",
        expected: u64::try_from(i32::MAX).unwrap_or(u64::MAX),
        got: u64::from(value),
    })
}

/// What one expert stage is launched as.
#[derive(Debug, Clone, Copy)]
struct ExpertStage {
    entry: &'static str,
    args: ExpertArgs,
    columns: u32,
    experts: u32,
    out_dim: u32,
}

/// Launches one expert stage over the expert-sorted column map.
///
/// The second grid extent is `min(columns, experts)` rather than `experts`:
/// the entry indexes the ACTIVE experts through the list the scan leaves
/// behind, and a chunk cannot activate more experts than it has columns. At
/// one token that is eight blocks rather than two hundred and fifty-six.
fn expert_stage(
    backend: &dyn Backend,
    stage: ExpertStage,
    buffers: &[DeviceView<'_>],
) -> CoreResult<()> {
    let mut args = Vec::with_capacity(buffers.len() + 1);
    args.push(KernelArg::Bytes(bytemuck::bytes_of(&stage.args)));
    args.extend(buffers.iter().copied().map(KernelArg::Buffer));
    launch::run(
        backend,
        KernelRef::new(Q35MOE, stage.entry),
        Extent3::new(
            stage.out_dim.div_ceil(EXPERT_ROWS),
            stage.columns.min(stage.experts),
            1,
        ),
        Extent3::linear(EXPERT_BLOCK),
        0,
        &args,
    )
    .map_err(|err| launch::launch_failure(stage.entry, err))
}

/// The selection: softmax over every expert, the best `used`, renormalized.
fn route(
    backend: &dyn Backend,
    shape: MoeShape,
    scratch: &MoeScratch,
    n_tok: u32,
) -> CoreResult<()> {
    let entry = "q4e_moe_route_kernel";
    launch::run(
        backend,
        KernelRef::new(Q4E_MOE, entry),
        Extent3::linear(n_tok),
        Extent3::linear(BLOCK_LINEAR),
        // The entry stages the whole logit row in dynamic shared memory, one
        // float per expert.
        shape.experts.saturating_mul(4),
        &[
            KernelArg::Buffer(scratch.ids.view()),
            KernelArg::Buffer(scratch.weights.view()),
            KernelArg::Buffer(scratch.logits.view()),
            KernelArg::U32(shape.experts),
            KernelArg::U32(shape.used),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// The counting sort that groups the routed columns by expert.
///
/// Three passes over the columns. The histogram row is cleared first: it is
/// accumulated with `atomicAdd` and outlives the layer, so without this every
/// earlier layer's counts are still in it and an expert's span grows past what
/// the sorted map holds.
pub(crate) fn sort(
    backend: &dyn Backend,
    shape: MoeShape,
    scratch: &MoeScratch,
    columns: u32,
) -> CoreResult<()> {
    backend.fill_zero(scratch.counts.view())?;
    backend.fill_zero(scratch.cursor.view())?;
    let linear = |n: u32| Extent3::linear(n.div_ceil(BLOCK_LINEAR).max(1));
    let passes: [(&'static str, Extent3, Extent3, Vec<KernelArg<'_>>); 3] = [
        (
            "q4e_moe_sort_count_kernel",
            linear(columns),
            Extent3::linear(BLOCK_LINEAR),
            vec![
                KernelArg::Buffer(scratch.counts.view()),
                KernelArg::Buffer(scratch.ids.view()),
                KernelArg::U32(columns),
                KernelArg::U32(shape.experts),
            ],
        ),
        (
            "q4e_moe_sort_scan_kernel",
            Extent3::linear(1),
            Extent3::linear(SCAN_BLOCK),
            vec![
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.active.view()),
                KernelArg::Buffer(scratch.counts.view()),
                KernelArg::U32(shape.experts),
            ],
        ),
        (
            "q4e_moe_sort_scatter_kernel",
            linear(columns),
            Extent3::linear(BLOCK_LINEAR),
            vec![
                KernelArg::Buffer(scratch.sorted.view()),
                KernelArg::Buffer(scratch.cursor.view()),
                KernelArg::Buffer(scratch.bounds.view()),
                KernelArg::Buffer(scratch.ids.view()),
                KernelArg::U32(columns),
                KernelArg::U32(shape.experts),
            ],
        ),
    ];
    for (entry, grid, block, args) in passes {
        launch::run(
            backend,
            KernelRef::new(Q4E_MOE, entry),
            grid,
            block,
            0,
            &args,
        )
        .map_err(|err| launch::launch_failure(entry, err))?;
    }
    Ok(())
}

/// `out = silu(gate) * up` over `n` elements.
fn activate(
    backend: &dyn Backend,
    out: DeviceView<'_>,
    gate: DeviceView<'_>,
    up: DeviceView<'_>,
    n: u64,
) -> CoreResult<()> {
    let entry = "q4e_swiglu_kernel";
    let blocks = u32::try_from(n.div_ceil(u64::from(BLOCK_LINEAR))).unwrap_or(u32::MAX);
    launch::run(
        backend,
        KernelRef::new(Q4E_MOE, entry),
        Extent3::linear(blocks.max(1)),
        Extent3::linear(BLOCK_LINEAR),
        0,
        &[
            KernelArg::Buffer(out),
            KernelArg::Buffer(gate),
            KernelArg::Buffer(up),
            KernelArg::U64(n),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// The weighted sum of every column into one contribution per token.
pub(crate) fn combine(
    backend: &dyn Backend,
    shape: MoeShape,
    scratch: &MoeScratch,
    out: DeviceView<'_>,
    hidden: u32,
    n_tok: u32,
) -> CoreResult<()> {
    let entry = "q4e_moe_combine_kernel";
    launch::run(
        backend,
        KernelRef::new(Q4E_MOE, entry),
        Extent3::new(hidden.div_ceil(BLOCK_LINEAR).max(1), n_tok, 1),
        Extent3::linear(BLOCK_LINEAR),
        0,
        &[
            KernelArg::Buffer(out),
            KernelArg::Buffer(scratch.down.view()),
            KernelArg::Buffer(scratch.weights.view()),
            KernelArg::U32(hidden),
            KernelArg::U32(shape.used),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// The always-on expert: a dense feed-forward, gated by a scalar per token and
/// added onto what the routed experts already wrote.
///
/// It reads the same normalized rows the router did, not the routed
/// contribution, and its add is an add rather than a write for that reason.
fn add_shared(
    backend: &dyn Backend,
    moe: &Moe,
    shared: &Ffn,
    scratch: &MoeScratch,
    io: MoeIo<'_>,
    hidden: u32,
    n_tok: u32,
) -> CoreResult<()> {
    crate::ffn::run(
        backend,
        shared,
        crate::shape::Activation::SwiGlu,
        io.input,
        scratch.gate.view(),
        scratch.up.view(),
        scratch.mid.view(),
        scratch.shared_out.view(),
        n_tok,
    )?;
    projection::projection(
        backend,
        &moe.shared_gate,
        n_tok,
        io.input,
        scratch.shared_logit.view(),
    )?;
    let entry = "q4e_shared_add_kernel";
    launch::run(
        backend,
        KernelRef::new(Q4E_MOE, entry),
        Extent3::new(hidden.div_ceil(BLOCK_LINEAR).max(1), n_tok, 1),
        Extent3::linear(BLOCK_LINEAR),
        0,
        &[
            KernelArg::Buffer(io.block_out),
            KernelArg::Buffer(scratch.shared_out.view()),
            KernelArg::Buffer(scratch.shared_logit.view()),
            KernelArg::U32(hidden),
        ],
    )
    .map_err(|err| launch::launch_failure(entry, err))
}

/// The codec a weight's auxiliary data says it is stored in.
///
/// Public for a loader to classify a tensor by before it strides one into a
/// slab, which is the check that stops an affine weight reaching an NVFP4
/// stack. **No loader calls it yet**: [`Moe::new`] enforces the same agreement
/// one level up, from the slabs it is handed, so this is the per-tensor half
/// of the check and is currently unused. The claim that a loader already calls
/// it was inherited from `is_nvfp4` and was not true of that either.
#[must_use]
pub fn codec_of(weight: &Weight) -> Option<ExpertCodec> {
    match weight.aux() {
        Some(WeightAux::Nvfp4 { .. }) => Some(ExpertCodec::Nvfp4),
        Some(WeightAux::Affine(_)) => Some(ExpertCodec::Q4Affine),
        Some(WeightAux::Fp8 { .. }) | None => None,
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
