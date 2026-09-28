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

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::{CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3};

use crate::ffn::{self, Ffn};
use crate::launch::{self, DENSE, F32_BYTES};
use crate::moe::{MoeIo, MoeScratch};
use crate::projection::{self, Weight};
use crate::shape::{Activation, MoeShape};

/// Output rows one expert-stage threadgroup covers: four simdgroups of four.
const EXPERT_ROWS_PER_GROUP: u32 = 16;

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
    pub up: RowAffine4Stack,
    /// `expert width -> hidden`.
    pub down: RowAffine4Stack,
}

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
    pad: [u32; 2],
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
        ("routed up width", shape.intermediate, moe.up.out_dim),
        ("routed up input", hidden, moe.up.in_dim),
        ("routed down input", shape.intermediate, moe.down.in_dim),
        ("routed down width", hidden, moe.down.out_dim),
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
    let stage =
        |stack: &RowAffine4Stack, first: bool, input: DeviceView<'_>, out: DeviceView<'_>| {
            expert_stage(
                backend,
                stack,
                ExpertArgs {
                    in_dim: stack.in_dim,
                    out_dim: stack.out_dim,
                    n_expert: shape.experts,
                    n_used: shape.used,
                    by_token: u32::from(first),
                    act: u32::from(first),
                    pad: [0; 2],
                },
                columns.min(shape.experts),
                [
                    input,
                    scratch.sorted.view(),
                    scratch.bounds.view(),
                    scratch.active.view(),
                    out,
                ],
            )
        };
    stage(&moe.up, true, io.input, scratch.up.view())?;
    stage(&moe.down, false, scratch.up.view(), scratch.down.view())?;
    combine_add(backend, moe, shape, scratch, io.block_out, hidden, n_tok)
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

/// Refuses stacks whose widths the stage kernel cannot walk.
///
/// # Errors
///
/// [`CoreError::InvalidValue`] for an input width that is not a whole number
/// of 64-value groups: a packed row's last group would be read past.
pub fn check_stack(stack: &RowAffine4Stack) -> CoreResult<()> {
    if stack.in_dim == 0 || !stack.in_dim.is_multiple_of(64) {
        return Err(CoreError::InvalidValue {
            key: "routed expert input width",
            value: format!("{} is not a whole number of 64-value groups", stack.in_dim),
        });
    }
    Ok(())
}
