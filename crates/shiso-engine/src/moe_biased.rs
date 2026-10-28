//! The routed feed-forward with a biased sigmoid router and ungated
//! squared-rectifier experts, over row-major affine stacks.
//!
//! The same block as [`crate::moe`] in outline: a router chooses a handful of
//! experts per token, the chosen columns are grouped by expert, each group runs
//! through its expert, and the columns are recombined with the always-on shared
//! expert added on top. Three things differ, and each is a different kernel
//! rather than a flag on one of that module's:
//!
//! - the router scores every expert with a sigmoid and chooses by that score
//!   plus a learned per-expert bias, while the weight a chosen expert gets is
//!   the unbiased score, renormalized and scaled;
//! - an expert is two projections, `down(max(up(x), 0)^2)`, with no gate, and
//!   the square is fused into the first projection's store;
//! - the stacks are the container's own row-major layout, so an expert width
//!   that is not a multiple of a storage tile needs no padding.
//!
//! The sort between the router and the experts and the combine after them are
//! [`crate::moe`]'s own launches: the column conventions are the same.

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3};

use crate::ffn::{self, Ffn};
use crate::launch::{self, DENSE, F32_BYTES};
use crate::moe::{MoeIo, MoeScratch};
use crate::projection::{self, Weight};
use crate::shape::{Activation, MoeShape};
use crate::sparse::ExpertStream;
use crate::sparse::weights::WeightPart;

/// Output rows one expert-stage threadgroup covers: four simdgroups of four.
const EXPERT_ROWS_PER_GROUP: u32 = 16;

/// Output rows one tiled `IQ4_NL` threadgroup covers.
const IQ4NL_TILE_ROWS: u32 = 64;

/// Threads an expert-stage threadgroup runs.
const EXPERT_THREADS: u32 = 128;

/// Threads a combine threadgroup runs.
const COMBINE_THREADS: u32 = 256;

/// Threads the router's threadgroup runs.
const ROUTE_THREADS: u32 = 128;

/// Routed columns up to which one threadgroup routes and sorts the whole
/// chunk in one launch. Past it the sort is three passes over the grid: the
/// fused launch's serial count and scatter grow with the columns, and a
/// decode step - the case the fusion exists for - is at most a few dozen.
const FUSED_ROUTE_COLUMNS: u32 = 256;

/// One projection of every expert in a layer, in the container's row-major
/// 4-bit affine layout: expert `e`'s row `r` is `in_dim / 8` packed words at
/// `(e * out_dim + r) * in_dim / 8`, with `in_dim / 64` bf16 scales and biases
/// at `(e * out_dim + r) * in_dim / 64`.
#[derive(Debug)]
pub struct RowAffine4Stack {
    /// Packed codes, `[experts, out_dim, in_dim / 8]` words.
    pub weights: DeviceBuffer,
    /// bf16 scales, `[experts, out_dim, in_dim / 64]`.
    pub scales: DeviceBuffer,
    /// bf16 biases, `[experts, out_dim, in_dim / 64]`.
    pub biases: DeviceBuffer,
    /// Inputs a row reads.
    pub in_dim: u32,
    /// Rows one expert holds.
    pub out_dim: u32,
}

/// One projection of every expert in a layer, as row-major `IQ4_NL` blocks:
/// expert `e`'s row `r` is `in_dim / 32` blocks of 18 bytes (an f16 scale and
/// sixteen bytes of 4-bit indices) starting at block
/// `(e * out_dim + r) * in_dim / 32`.
#[derive(Debug)]
pub struct RowIq4NlStack {
    /// Where the blocks are.
    pub store: StackStore,
    /// Inputs a row reads.
    pub in_dim: u32,
    /// Rows one expert holds.
    pub out_dim: u32,
}

/// Bytes one `IQ4_NL` block occupies: an f16 scale and sixteen packed bytes.
const IQ4NL_BLOCK_BYTES: u64 = 18;

impl RowIq4NlStack {
    /// Bytes one expert's projection occupies.
    #[must_use]
    pub const fn expert_bytes(&self) -> u64 {
        self.out_dim as u64 * (self.in_dim as u64 / 32) * IQ4NL_BLOCK_BYTES
    }
}

/// Where a routed stack's bytes live.
#[derive(Debug)]
pub enum StackStore {
    /// Every expert, resident: expert `e` at position `e`.
    Resident(DeviceBuffer),
    /// A pool of streamed slots: expert `e` of `layer` is wherever the
    /// stream's map for that layer says, and only after the stream fetched it.
    Streamed {
        /// The stream the slots belong to.
        stream: Arc<ExpertStream>,
        /// Which of the source's projections this stack is.
        matrix: usize,
        /// The stream's layer index for this block.
        layer: u32,
    },
}

/// One projection of every routed expert, in one of the stored codecs.
#[derive(Debug)]
pub enum ExpertStack {
    /// 4-bit affine in groups of 64, resident.
    Affine4(RowAffine4Stack),
    /// `IQ4_NL` blocks of 32, resident or streamed.
    Iq4Nl(RowIq4NlStack),
}

impl ExpertStack {
    /// Inputs a row reads.
    #[must_use]
    pub const fn in_dim(&self) -> u32 {
        match self {
            Self::Affine4(s) => s.in_dim,
            Self::Iq4Nl(s) => s.in_dim,
        }
    }

    /// Rows one expert holds.
    #[must_use]
    pub const fn out_dim(&self) -> u32 {
        match self {
            Self::Affine4(s) => s.out_dim,
            Self::Iq4Nl(s) => s.out_dim,
        }
    }
}

/// One layer's biased-sigmoid routed feed-forward, resident on the device.
///
/// The shared expert is the slot's own feed-forward, as it is for
/// [`crate::moe::Moe`], so it is not held here.
#[derive(Debug)]
pub struct BiasedMoe {
    /// `hidden -> experts`: the router's logits.
    pub router: Weight,
    /// The per-expert bias the choice adds to the sigmoid score, `[experts]`
    /// f32.
    pub bias: DeviceBuffer,
    /// What the chosen weights are multiplied by after renormalization.
    pub scale: f32,
    /// Whether the chosen weights are renormalized to sum to one.
    pub renormalize: bool,
    /// `hidden -> expert width`, squared-rectified at its store.
    pub up: ExpertStack,
    /// `expert width -> hidden`.
    pub down: ExpertStack,
    /// The next routed layer's router, on the host, when the banks are
    /// streamed and the experts that layer will want are guessed ahead.
    pub guess: Option<RouterGuess>,
}

/// The next routed layer's router, held on the host so the experts it is
/// likely to choose can be read while the device computes.
///
/// The guess scores this layer's normalized input rows with the next layer's
/// router: the residual stream moves little across one mixing block, so the
/// next layer's choice is mostly the one it would make on these rows. A wrong
/// guess costs a read nobody uses; it never changes a result, because the next
/// layer's own fetch reads whatever the guess missed.
#[derive(Debug, Clone)]
pub struct RouterGuess {
    /// The stream layer the guesses are for.
    pub layer: u32,
    /// `[experts][hidden]` router weights.
    pub weights: Arc<[f32]>,
    /// `[experts]` selection bias.
    pub bias: Arc<[f32]>,
}

impl RouterGuess {
    /// The experts the router would choose, `used` a row, for any of `rows`
    /// (`n` rows of `hidden` values).
    #[must_use]
    pub fn choose(&self, rows: &[f32], hidden: usize, used: usize) -> Vec<u32> {
        let experts = self.bias.len();
        let mut chosen: Vec<u32> = Vec::new();
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(experts);
        for row in rows.chunks_exact(hidden.max(1)) {
            scored.clear();
            for (e, w) in self.weights.chunks_exact(hidden.max(1)).enumerate() {
                let logit: f32 = w.iter().zip(row).map(|(a, b)| a * b).sum();
                let score = 1.0 / (1.0 + (-logit).exp()) + self.bias[e];
                scored.push((score, u32::try_from(e).unwrap_or(u32::MAX)));
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for &(_, e) in scored.iter().take(used) {
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
        }
        chosen
    }
}

/// Rows a pass may carry and still guess the next layer's experts: a decode
/// step or a verify pass. A prompt pass wants nearly every expert anyway.
const GUESS_MAX_ROWS: u32 = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RouteArgs {
    experts: u32,
    used: u32,
    scale: f32,
    norm: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExpertArgs {
    in_dim: u32,
    out_dim: u32,
    n_expert: u32,
    n_used: u32,
    by_token: u32,
    act: u32,
    /// Non-zero reads each expert through the stream's slot map.
    mapped: u32,
    /// Where this layer's entries start in that map.
    layer_base: u32,
}

const _: () = assert!(size_of::<ExpertArgs>() == 32);

/// Runs one biased routed feed-forward over `n_tok` rows of `io.input`,
/// shared expert included.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the router, the stacks and the shape
/// disagree about a width, whatever [`projection::projection`] reports for the
/// router or the shared expert, and [`CoreError::Backend`] naming the entry
/// when a launch fails.
pub fn run(
    backend: &dyn Backend,
    moe: &BiasedMoe,
    shared: &Ffn,
    shape: MoeShape,
    scratch: &MoeScratch,
    io: MoeIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let hidden = moe.router.in_dim;
    for (field, expected, got) in [
        ("router output width", shape.experts, moe.router.out_dim),
        ("routed up width", shape.intermediate, moe.up.out_dim()),
        ("routed up input", hidden, moe.up.in_dim()),
        ("routed down input", shape.intermediate, moe.down.in_dim()),
        ("routed down width", hidden, moe.down.out_dim()),
    ] {
        if expected != got {
            return Err(CoreError::ShapeMismatch {
                field,
                expected: u64::from(expected),
                got: u64::from(got),
            });
        }
    }
    let columns = u32::try_from(shape.slot_rows(n_tok)).unwrap_or(u32::MAX);
    // The shared expert first, into its own rows: it reads the same normalized
    // rows the router does, and its scratch is free again before the routed
    // stages write theirs.
    ffn::run(
        backend,
        shared,
        Activation::Relu2,
        io.input,
        scratch.gate.view(),
        scratch.up.view(),
        scratch.mid.view(),
        scratch.shared_out.view(),
        n_tok,
    )?;
    projection::projection(backend, &moe.router, n_tok, io.input, scratch.logits.view())?;
    if columns <= FUSED_ROUTE_COLUMNS {
        route_sort(backend, moe, shape, scratch, n_tok)?;
    } else {
        route(backend, moe, shape, scratch, n_tok)?;
        crate::moe::sort(backend, shape, scratch, columns)?;
    }
    // A streamed bank's experts are read in now, after the router chose them
    // and before a stage reads them: the fetch drains the device, reads the
    // chosen ids back and fills the slots the layer's map then points at. The
    // up and down banks of a layer are one stream layer, so one fetch serves
    // both.
    fetch_streamed(backend, moe, shape, scratch, io, n_tok, columns)?;
    let active = columns.min(shape.experts);
    let tiled = n_tok > crate::MAX_DECODE_BATCH;
    let stage = |stack: &ExpertStack, first: bool, input: DeviceView<'_>, out: DeviceView<'_>| {
        let args = ExpertArgs {
            in_dim: stack.in_dim(),
            out_dim: stack.out_dim(),
            n_expert: shape.experts,
            n_used: shape.used,
            by_token: u32::from(first),
            act: u32::from(first),
            mapped: 0,
            layer_base: 0,
        };
        let routing = [
            input,
            scratch.sorted.view(),
            scratch.bounds.view(),
            scratch.active.view(),
            out,
        ];
        match stack {
            ExpertStack::Affine4(stack) => expert_stage(backend, stack, args, active, routing),
            ExpertStack::Iq4Nl(stack) => match &stack.store {
                StackStore::Resident(blocks) => iq4nl_stage(
                    backend,
                    stack,
                    args,
                    active,
                    tiled,
                    [blocks.view(), scratch.bounds.view()],
                    routing,
                ),
                StackStore::Streamed {
                    stream,
                    matrix,
                    layer,
                } => iq4nl_stage(
                    backend,
                    stack,
                    ExpertArgs { mapped: 1, ..args },
                    active,
                    tiled,
                    [
                        stream.pool(*matrix, WeightPart::Whole)?,
                        stream.map(*layer)?,
                    ],
                    routing,
                ),
            },
        }
    };
    stage(&moe.up, true, io.input, scratch.up.view())?;
    stage(&moe.down, false, scratch.up.view(), scratch.down.view())?;
    combine_add(backend, moe, shape, scratch, io.block_out, hidden, n_tok)
}

/// Brings a streamed bank's chosen experts in, after the router chose them and
/// before a stage reads them, and guesses the next layer's.
///
/// The fetch drains the device, reads the chosen ids back and fills the slots
/// the layer's map then points at. The up and down banks of a layer are one
/// stream layer, so one fetch serves both. With the device drained this
/// pass's normalized rows are final, so the next layer's likely experts can be
/// read while it computes.
fn fetch_streamed(
    backend: &dyn Backend,
    moe: &BiasedMoe,
    shape: MoeShape,
    scratch: &MoeScratch,
    io: MoeIo<'_>,
    n_tok: u32,
    columns: u32,
) -> CoreResult<()> {
    let Some((stream, layer)) = streamed(&moe.up).or_else(|| streamed(&moe.down)) else {
        return Ok(());
    };
    stream.fetch(backend, layer, scratch.ids.view(), columns)?;
    if let Some(guess) = moe.guess.as_ref()
        && n_tok <= GUESS_MAX_ROWS
    {
        let width = usize::try_from(moe.router.in_dim).unwrap_or(usize::MAX);
        let mut rows = vec![0f32; width * n_tok as usize];
        backend.copy_d2h(
            io.input.slice(0, rows.len() as u64 * F32_BYTES)?,
            bytemuck::cast_slice_mut(&mut rows),
        )?;
        stream.prefetch(
            guess.layer,
            &guess.choose(&rows, width, shape.used as usize),
        );
    }
    Ok(())
}

/// The stream and stream layer a stack reads through, when it is streamed.
fn streamed(stack: &ExpertStack) -> Option<(&ExpertStream, u32)> {
    match stack {
        ExpertStack::Iq4Nl(RowIq4NlStack {
            store: StackStore::Streamed { stream, layer, .. },
            ..
        }) => Some((stream.as_ref(), *layer)),
        _ => None,
    }
}

fn route_args(moe: &BiasedMoe, shape: MoeShape) -> RouteArgs {
    RouteArgs {
        experts: shape.experts,
        used: shape.used,
        scale: moe.scale,
        norm: u32::from(moe.renormalize),
    }
}

/// The router and the counting sort in one threadgroup, for a small chunk.
fn route_sort(
    backend: &dyn Backend,
    moe: &BiasedMoe,
    shape: MoeShape,
    scratch: &MoeScratch,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_moe_route_sort_sigmoid_f32";
    let args = route_args(moe, shape);
    let smem = u32::try_from(u64::from(shape.experts) * 3 * F32_BYTES).unwrap_or(u32::MAX);
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::linear(1),
        Extent3::linear(ROUTE_THREADS),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.ids.view()),
            KernelArg::Buffer(scratch.weights.view()),
            KernelArg::Buffer(scratch.logits.view()),
            KernelArg::Buffer(moe.bias.view()),
            KernelArg::Buffer(scratch.sorted.view()),
            KernelArg::Buffer(scratch.bounds.view()),
            KernelArg::Buffer(scratch.active.view()),
            KernelArg::U32(n_tok),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// The routed columns recombined per token, with the shared expert added.
fn combine_add(
    backend: &dyn Backend,
    moe: &BiasedMoe,
    shape: MoeShape,
    scratch: &MoeScratch,
    out: DeviceView<'_>,
    hidden: u32,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_moe_combine_add_f32";
    let args = route_args(moe, shape);
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(hidden.div_ceil(COMBINE_THREADS), n_tok, 1),
        Extent3::linear(COMBINE_THREADS),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out),
            KernelArg::Buffer(scratch.shared_out.view()),
            KernelArg::Buffer(scratch.down.view()),
            KernelArg::Buffer(scratch.weights.view()),
            KernelArg::U32(hidden),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// The biased sigmoid choice, one threadgroup a token.
fn route(
    backend: &dyn Backend,
    moe: &BiasedMoe,
    shape: MoeShape,
    scratch: &MoeScratch,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_moe_route_sigmoid_f32";
    let args = route_args(moe, shape);
    let smem = u32::try_from(u64::from(shape.experts) * 2 * F32_BYTES).unwrap_or(u32::MAX);
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::linear(n_tok),
        Extent3::linear(ROUTE_THREADS),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.ids.view()),
            KernelArg::Buffer(scratch.weights.view()),
            KernelArg::Buffer(scratch.logits.view()),
            KernelArg::Buffer(moe.bias.view()),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// One expert stage over the expert-sorted column map.
///
/// `active` threadgroups on the second extent: the entry indexes the active
/// experts through the list the sort left behind, and a chunk cannot activate
/// more experts than it has columns.
fn expert_stage(
    backend: &dyn Backend,
    stack: &RowAffine4Stack,
    args: ExpertArgs,
    active: u32,
    views: [DeviceView<'_>; 5],
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_experts_affine4_f32";
    let [input, sorted, bounds, active_list, out] = views;
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(stack.out_dim.div_ceil(EXPERT_ROWS_PER_GROUP), active, 1),
        Extent3::linear(EXPERT_THREADS),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(stack.weights.view()),
            KernelArg::Buffer(stack.scales.view()),
            KernelArg::Buffer(stack.biases.view()),
            KernelArg::Buffer(input),
            KernelArg::Buffer(sorted),
            KernelArg::Buffer(bounds),
            KernelArg::Buffer(active_list),
            KernelArg::Buffer(out),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// One `IQ4_NL` routed stage: the affine stage's grid and column walk, over
/// `weights[0]` - the whole stack, or a slot pool - with `weights[1]` the slot
/// map the entry reads when `args.mapped` is set (any buffer otherwise; it is
/// not read).
///
/// A pass wider than a decode batch takes the tiled entry, which dequantizes
/// a band of an expert's weights once for sixteen of its columns: measured on
/// an M4, the 802-row prompt of a one-minute recording spent 10.0 s of its
/// 11.6 s of kernel time in the one-column entry. A decode step and a verify
/// pass stay on the one-column entry, so a verified row is summed in the
/// order the serial step that it checks would sum it.
fn iq4nl_stage(
    backend: &dyn Backend,
    stack: &RowIq4NlStack,
    args: ExpertArgs,
    active: u32,
    tiled: bool,
    weights: [DeviceView<'_>; 2],
    views: [DeviceView<'_>; 5],
) -> CoreResult<()> {
    let (name, rows_per_group) = if tiled {
        ("kernel_dense_experts_iq4nl_gemm_f32", IQ4NL_TILE_ROWS)
    } else {
        ("kernel_dense_experts_iq4nl_f32", EXPERT_ROWS_PER_GROUP)
    };
    let [blocks, map] = weights;
    let [input, sorted, bounds, active_list, out] = views;
    launch::run(
        backend,
        KernelRef::new(DENSE, name),
        Extent3::new(stack.out_dim.div_ceil(rows_per_group), active, 1),
        Extent3::linear(EXPERT_THREADS),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(blocks),
            KernelArg::Buffer(map),
            KernelArg::Buffer(input),
            KernelArg::Buffer(sorted),
            KernelArg::Buffer(bounds),
            KernelArg::Buffer(active_list),
            KernelArg::Buffer(out),
        ],
    )
    .map_err(|err| launch::launch_failure(name, err))
}

/// Refuses stacks whose widths the stage kernel cannot walk.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for an input width that is not a whole number
/// of the codec's groups (64 values affine, 32 values `IQ4_NL`): a packed row's
/// last group would be read past. [`CoreError::ShapeMismatch`] for a streamed
/// stack whose slots are not one expert's projection wide.
pub fn check_stack(stack: &ExpertStack) -> CoreResult<()> {
    match stack {
        ExpertStack::Affine4(stack) => {
            if stack.in_dim == 0 || !stack.in_dim.is_multiple_of(64) {
                return Err(CoreError::InvalidValue {
                    key: "routed expert input width",
                    value: format!("{} is not a whole number of 64-value groups", stack.in_dim),
                });
            }
        }
        ExpertStack::Iq4Nl(stack) => {
            if stack.in_dim == 0 || !stack.in_dim.is_multiple_of(32) {
                return Err(CoreError::InvalidValue {
                    key: "routed expert input width",
                    value: format!(
                        "{} is not a whole number of 32-value IQ4_NL blocks",
                        stack.in_dim
                    ),
                });
            }
            if let StackStore::Streamed { stream, matrix, .. } = &stack.store {
                let have = stream.slot_layout().get(*matrix).copied();
                if have != Some(stack.expert_bytes()) {
                    return Err(CoreError::ShapeMismatch {
                        field: "streamed IQ4_NL expert bytes",
                        expected: stack.expert_bytes(),
                        got: have.unwrap_or(0),
                    });
                }
            }
        }
    }
    Ok(())
}
