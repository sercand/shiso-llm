//! The feed-forward block: two projections up, a gated activation, one
//! projection back.
//!
//! This is the only block whose shape is the same at every layer slot, so it is
//! also the only one that can be written without consulting the layer kind. It
//! takes its input already normalized and hands back a row the caller adds into
//! the residual stream, which is the same contract the recurrent and attention
//! blocks keep, so [`crate::layers`] drives all three the same way.
//!
//! # Why the activation is its own launch
//!
//! Both projections write a full inner row and the activation reads both rows to
//! produce one, so fusing it into either projection would make one of them own an
//! output the other still has to write. Keeping it separate also leaves the
//! clamp toggles in the argument block readable at the call site instead of
//! buried in a fused kernel.

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::{CoreError, CoreResult, DeviceView, Extent3};

use crate::launch::{self, BLOCK_LINEAR, GluArgs};
use crate::projection::{self, Weight};
use crate::shape::Activation;

/// The three matrices of one feed-forward block, resident on the device.
///
/// Not `Debug`: a [`Weight`] owns a device allocation, and printing one would
/// name a buffer no host read can inspect.
#[derive(Debug)]
pub struct Ffn {
    /// Gating projection, whose row the activation pushes through its curve.
    ///
    /// `None` for an ungated activation ([`Activation::Relu2`]), which reads
    /// the up projection alone.
    pub gate: Option<Weight>,
    /// Up projection, multiplied into the gated half.
    pub up: Weight,
    /// Projection back to the residual width.
    pub down: Weight,
}

impl Ffn {
    /// The inner width.
    ///
    /// Both halves agree on it because the activation multiplies their rows
    /// element for element, and the binding step already rejected a file where
    /// the two projections were different widths.
    #[must_use]
    pub const fn intermediate(&self) -> u32 {
        self.up.out_dim
    }
}

/// The gate unit's argument block for one token's inner row.
///
/// `alpha` 1.0 and `limit` 0.0 are the only pair this model uses: the kernel
/// scales by `alpha` unconditionally and skips the clamp pair when `limit` is at
/// or below its epsilon, so the row comes out as `silu(gate) * up` with no extra
/// factor. The two row offsets are zero because the caller hands each token a
/// view already sliced to its own row.
fn glu_args(intermediate: u32) -> GluArgs {
    let width = i32::try_from(u64::from(intermediate)).unwrap_or(i32::MAX);
    let row = u64::from(intermediate).saturating_mul(launch::F32_BYTES);
    GluArgs {
        ne00: width,
        pad0: [0; 4],
        nb01: row,
        ne10: width,
        pad1: [0; 4],
        nb11: row,
        ne0: width,
        pad2: [0; 4],
        nb1: row,
        i00: 0,
        i10: 0,
        alpha: 1.0,
        limit: 0.0,
    }
}

/// The entry point one activation reaches.
///
/// Two kernels rather than one with a flag: the gated `SiLU` is what a qwen35
/// decode dispatches and the gated GELU is what a Gemma-shaped one does, the
/// choice belongs to the checkpoint rather than to the launch, and a branch
/// inside the kernel would be a branch that could be got wrong at one call site
/// without anything failing loudly. Both read the same [`GluArgs`] and both are
/// launched over the same grid, so the pair differ in nothing but the curve.
const fn activation_kernel(activation: Activation) -> Option<KernelRef> {
    match activation {
        Activation::SwiGlu => Some(KernelRef::new(launch::GENERIC, "kernel_swiglu_flat_f32")),
        Activation::GeGluTanh => Some(KernelRef::new(launch::DENSE, "kernel_dense_geglu_flat_f32")),
        Activation::Relu2 => None,
    }
}

/// Argument block of `kernel_dense_relu2_f32`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlatArgs {
    n: u32,
    pad: [u32; 3],
}

/// `dst = max(src, 0)^2` over `n` floats, in one launch for the whole chunk.
pub(crate) fn relu2(
    backend: &dyn Backend,
    src: DeviceView<'_>,
    dst: DeviceView<'_>,
    n: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_relu2_f32";
    let args = FlatArgs { n, pad: [0; 3] };
    launch::run(
        backend,
        KernelRef::new(launch::DENSE, NAME),
        Extent3::linear(n.div_ceil(BLOCK_LINEAR).max(1)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(src),
            KernelArg::Buffer(dst),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}

/// Runs the activation over one token's inner row.
///
/// The kernel is flat: it takes its element index straight from the grid and
/// reaches the row through the offsets in the argument block, so a chunk costs
/// one launch per token rather than a taller grid. At decode that is one launch,
/// and a prefill chunk trades an extra launch per token for a kernel with no row
/// arithmetic in it.
fn activation_row(
    backend: &dyn Backend,
    activation: Activation,
    gate: DeviceView<'_>,
    up: DeviceView<'_>,
    dst: DeviceView<'_>,
    intermediate: u32,
) -> CoreResult<()> {
    let args = glu_args(intermediate);
    let kernel = activation_kernel(activation).ok_or(CoreError::Unsupported {
        what: "a gated activation row for an ungated activation",
        backend: "engine",
    })?;
    launch::run(
        backend,
        kernel,
        Extent3::linear(intermediate.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(gate),
            KernelArg::Buffer(up),
            KernelArg::Buffer(dst),
        ],
    )
    .map_err(|err| launch::launch_failure(kernel.entry, err))
}

/// Runs the block over `n_tok` rows of `input`.
///
/// `activation` is the checkpoint's own gate curve, carried down from the shape:
/// [`Activation::SwiGlu`] reaches the generic `kernel_swiglu_flat_f32` with the
/// same arguments and the same grid it always did, and [`Activation::GeGluTanh`]
/// reaches the dense twin of it.
///
/// `gate_out`, `up_out` and `mid` are scratch the caller keeps across layers: the
/// two projection outputs are `intermediate` wide, wider than the residual
/// stream, so a model that allocated them per layer would carry that width once
/// per slot for buffers that never outlive a call.
///
/// # Errors
///
/// Whatever [`projection::projection`] reports for a shape or format it cannot
/// carry, and [`shiso_core::CoreError::Backend`] when a launch fails.
#[expect(
    clippy::too_many_arguments,
    reason = "the scratch rows belong to the caller, so they arrive as arguments"
)]
pub fn run(
    backend: &dyn Backend,
    ffn: &Ffn,
    activation: Activation,
    input: DeviceView<'_>,
    gate_out: DeviceView<'_>,
    up_out: DeviceView<'_>,
    mid: DeviceView<'_>,
    block_out: DeviceView<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    let inner = u64::from(ffn.intermediate());
    let row_bytes = inner.saturating_mul(launch::F32_BYTES);
    // The activation decides whether a gate exists, and a weight list that
    // disagrees with it is a binding that went wrong: an ungated curve run over
    // a checkpoint that has a gate would drop a third of the block.
    let gate = match (activation, ffn.gate.as_ref()) {
        (Activation::Relu2, None) => None,
        (Activation::SwiGlu | Activation::GeGluTanh, Some(gate)) => Some(gate),
        (_, gate) => {
            return Err(CoreError::ShapeMismatch {
                field: "feed-forward gate projection",
                expected: u64::from(!matches!(activation, Activation::Relu2)),
                got: u64::from(gate.is_some()),
            });
        }
    };
    let Some(gate) = gate else {
        projection::projection(backend, &ffn.up, n_tok, input, up_out)?;
        let n = u32::try_from(inner.saturating_mul(u64::from(n_tok))).unwrap_or(u32::MAX);
        relu2(backend, up_out, mid, n)?;
        return projection::projection(backend, &ffn.down, n_tok, mid, block_out);
    };
    projection::projections(
        backend,
        n_tok,
        input,
        &[(&ffn.up, up_out), (gate, gate_out)],
    )?;
    for itok in 0..n_tok {
        let offset = u64::from(itok).saturating_mul(row_bytes);
        activation_row(
            backend,
            activation,
            gate_out.slice(offset, row_bytes)?,
            up_out.slice(offset, row_bytes)?,
            mid.slice(offset, row_bytes)?,
            ffn.intermediate(),
        )?;
    }
    projection::projection(backend, &ffn.down, n_tok, mid, block_out)
}
