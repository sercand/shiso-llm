//! The selective state-space block: its weights, the scratch one chunk uses,
//! and the five launches that run it.
//!
//! One input projection produces three things side by side in every row: the
//! gate, the channels the convolution mixes (the heads' inputs and both of the
//! groups' state projections), and one step size per head. The convolution
//! reads the middle part in place, the scan reads its output and the step
//! sizes, and the gated norm reads the scan's output and the gate, so the
//! projection's rows are written once and never copied.
//!
//! The launches are the engine's own (`metal/dense/dense_ssm.metal`): nothing
//! in them is a fact about one checkpoint, every width is an argument.
//!
//! # The two states are not scratch
//!
//! As for [`crate::gdn`], the convolution window and the state matrices belong
//! to one session and arrive as views the caller owns; nothing here allocates
//! them. A batched decode step folds each row into its own session's state, so
//! the two sequential launches run a row at a time there while the three
//! projections and the norm stay batched.

use std::sync::Arc;

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};

use crate::blocks::{GdnIo, GdnState, GdnStates};
use crate::launch::{self, BLOCK_LINEAR, DENSE, F32_BYTES};
use crate::projection::{self, Weight};
use crate::shape::SsmShape;

/// Widest convolution the kernel's window registers hold.
const MAX_TAPS: u32 = 8;

/// Widest per-lane share of a state the scan's registers hold: a state of up
/// to `32 * 8` values.
const MAX_STATE: u32 = 32 * 8;

/// Channels one scan threadgroup covers: eight simdgroups, one channel each.
const SCAN_CHANNELS: u32 = 8;

/// Widest norm group the threadgroup stages.
const MAX_NORM_GROUP: u32 = 4096;

/// Floats the block-sum helper reserves ahead of the staged group.
const BLOCK_SUM_SLOTS: u32 = 33;

/// One state-space block's widths and the norm epsilon, checked against what
/// the kernels were written for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmGeometry {
    /// The block's widths.
    pub shape: SsmShape,
    /// Epsilon inside the gated norm.
    pub eps: f32,
}

impl SsmGeometry {
    /// Derives the block's geometry, refusing widths the kernels cannot carry.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the width: a convolution wider than
    /// the window registers, a state that is not a whole number of simdgroup
    /// lanes or is wider than their registers, heads that do not split into
    /// the groups evenly, or a norm group wider than the staging.
    pub fn new(shape: SsmShape, eps: f32) -> CoreResult<Self> {
        let refuse = |key: &'static str, value: String| Err(CoreError::InvalidValue { key, value });
        if shape.conv_kernel == 0 || shape.conv_kernel > MAX_TAPS {
            return refuse(
                "state-space convolution width",
                format!(
                    "{} taps, and the window holds 1 to {MAX_TAPS}",
                    shape.conv_kernel
                ),
            );
        }
        if shape.state == 0 || !shape.state.is_multiple_of(32) || shape.state > MAX_STATE {
            return refuse(
                "state-space state width",
                format!(
                    "{} values, and a lane carries a 32nd of at most {MAX_STATE}",
                    shape.state
                ),
            );
        }
        if shape.groups == 0 || !shape.heads.is_multiple_of(shape.groups) || shape.heads == 0 {
            return refuse(
                "state-space groups",
                format!(
                    "{} heads do not split into {} groups",
                    shape.heads, shape.groups
                ),
            );
        }
        let group = shape.inner() / shape.groups;
        if group > MAX_NORM_GROUP {
            return refuse(
                "state-space norm group",
                format!("{group} channels, and the norm stages at most {MAX_NORM_GROUP}"),
            );
        }
        Ok(Self { shape, eps })
    }

    /// Channels one gated-norm group covers.
    #[must_use]
    pub const fn norm_group(&self) -> u32 {
        self.shape.inner() / self.shape.groups
    }
}

/// One state-space block's weights, resident on the device.
///
/// The two projections are quantized matrices; the vectors beside them are f32.
#[derive(Debug)]
pub struct Ssm {
    /// `hidden -> [gate | convolved channels | step sizes]`.
    pub in_proj: Weight,
    /// The depthwise convolution, `[channels, taps]`.
    pub conv_weight: DeviceBuffer,
    /// The convolution's bias, `[channels]`.
    pub conv_bias: DeviceBuffer,
    /// Added to each head's raw step size before the softplus, `[heads]`.
    pub dt_bias: DeviceBuffer,
    /// The log of each head's decay rate, `[heads]`; the decay is
    /// `exp(-exp(a_log) * dt)`.
    pub a_log: DeviceBuffer,
    /// Each head's skip connection, `[heads]`.
    pub d: DeviceBuffer,
    /// The gated norm's weight, `[heads * head_dim]`.
    pub norm: DeviceBuffer,
    /// `heads * head_dim -> hidden`.
    pub out_proj: Weight,
}

/// Scratch one chunk's state-space block uses, sized once per stack.
#[derive(Debug)]
pub struct SsmScratch {
    /// The input projection's rows, `[rows, in_width]`.
    proj: DeviceBuffer,
    /// The convolution's output, `[rows, channels]`.
    xbc: DeviceBuffer,
    /// The scan's output, normalized in place, `[rows, inner]`.
    y: DeviceBuffer,
}

impl SsmScratch {
    /// Allocates the rows a chunk of `rows` tokens needs.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(backend: &Arc<dyn Backend>, geometry: &SsmGeometry, rows: u32) -> CoreResult<Self> {
        let floats = |width: u32| {
            u64::from(width)
                .saturating_mul(u64::from(rows))
                .saturating_mul(F32_BYTES)
        };
        Ok(Self {
            proj: alloc(backend, floats(geometry.shape.in_width()))?,
            xbc: alloc(backend, floats(geometry.shape.conv_channels()))?,
            y: alloc(backend, floats(geometry.shape.inner()))?,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvArgs {
    n_tok: u32,
    channels: u32,
    taps: u32,
    in_stride: u32,
    in_offset: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScanArgs {
    n_tok: u32,
    heads: u32,
    head_dim: u32,
    state: u32,
    groups: u32,
    channels: u32,
    in_stride: u32,
    dt_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormArgs {
    n_tok: u32,
    inner: u32,
    group_size: u32,
    in_stride: u32,
    eps: f32,
    pad: [u32; 3],
}

const _: () = assert!(size_of::<ConvArgs>() == 32);
const _: () = assert!(size_of::<ScanArgs>() == 32);
const _: () = assert!(size_of::<NormArgs>() == 32);

/// The rows of one sequential stage: where they start in the chunk, how many,
/// and whose state they fold into.
#[derive(Debug, Clone, Copy)]
struct Span<'a> {
    first: u32,
    rows: u32,
    state: GdnState<'a>,
}

/// Runs one state-space block over `n_tok` rows of `io.input`.
///
/// # Errors
///
/// [`CoreError::OutOfBounds`] for a recording pass wider than its trail;
/// [`CoreError::ShapeMismatch`] when a batch's
/// state list is not as long as the pass; whatever [`projection::projection`]
/// reports for either projection; and [`CoreError::Backend`] naming the entry
/// when a launch fails.
pub fn run(
    backend: &dyn Backend,
    ssm: &Ssm,
    geometry: &SsmGeometry,
    scratch: &SsmScratch,
    io: GdnIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    projection::projection(backend, &ssm.in_proj, n_tok, io.input, scratch.proj.view())?;
    match io.states {
        // A verify pass: the two sequential stages a row at a time, the state
        // copied out after each, as `StateTrail` describes. Both kernels carry
        // their state across rows in registers and write it back at the end,
        // so splitting the loop is the same arithmetic.
        GdnStates::Chunk(state) if io.trail.is_some() => {
            let trail = io.trail.ok_or(CoreError::ShapeMismatch {
                field: "state-space trail",
                expected: 1,
                got: 0,
            })?;
            // One record per drafted row: the pass's last row is either
            // rejected or the state the session keeps, so it is never one a
            // rollback names, and a trail is `n_tok - 1` records deep.
            if n_tok > trail.rows.saturating_add(1) {
                return Err(CoreError::OutOfBounds {
                    offset: u64::from(n_tok),
                    len: 1,
                    buffer_bytes: u64::from(trail.rows),
                });
            }
            let (conv_bytes, state_bytes) = (state.conv.len(), state.state.len());
            for row in 0..n_tok {
                sequential(
                    backend,
                    ssm,
                    geometry,
                    scratch,
                    Span {
                        first: row,
                        rows: 1,
                        state,
                    },
                )?;
                if row >= trail.rows {
                    continue;
                }
                backend.copy_d2d(
                    trail.conv.slice(u64::from(row) * conv_bytes, conv_bytes)?,
                    state.conv,
                )?;
                backend.copy_d2d(
                    trail
                        .state
                        .slice(u64::from(row) * state_bytes, state_bytes)?,
                    state.state,
                )?;
            }
        }
        GdnStates::Chunk(state) => sequential(
            backend,
            ssm,
            geometry,
            scratch,
            Span {
                first: 0,
                rows: n_tok,
                state,
            },
        )?,
        GdnStates::PerRow(states) => {
            if states.len() != n_tok as usize {
                return Err(CoreError::ShapeMismatch {
                    field: "state-space batch states",
                    expected: u64::from(n_tok),
                    got: u64::try_from(states.len()).unwrap_or(u64::MAX),
                });
            }
            for (row, state) in (0..n_tok).zip(states) {
                sequential(
                    backend,
                    ssm,
                    geometry,
                    scratch,
                    Span {
                        first: row,
                        rows: 1,
                        state: *state,
                    },
                )?;
            }
        }
    }
    gate_norm(backend, ssm, geometry, scratch, n_tok)?;
    projection::projection(
        backend,
        &ssm.out_proj,
        n_tok,
        scratch.y.view(),
        io.block_out,
    )
}

/// A float offset into a buffer of rows `width` floats wide, as a view.
fn rows_of(buffer: &DeviceBuffer, width: u32, first: u32, rows: u32) -> CoreResult<DeviceView<'_>> {
    let row = u64::from(width).saturating_mul(F32_BYTES);
    buffer.view().slice(
        u64::from(first).saturating_mul(row),
        u64::from(rows).saturating_mul(row),
    )
}

/// The convolution and the scan over one span, which fold its rows into one
/// session's state in order.
fn sequential(
    backend: &dyn Backend,
    ssm: &Ssm,
    geometry: &SsmGeometry,
    scratch: &SsmScratch,
    span: Span<'_>,
) -> CoreResult<()> {
    const CONV: &str = "kernel_dense_ssm_conv_f32";
    const SCAN: &str = "kernel_dense_ssm_scan_f32";
    let s = geometry.shape;
    let proj = rows_of(&scratch.proj, s.in_width(), span.first, span.rows)?;
    let xbc = rows_of(&scratch.xbc, s.conv_channels(), span.first, span.rows)?;
    let y = rows_of(&scratch.y, s.inner(), span.first, span.rows)?;
    let conv_args = ConvArgs {
        n_tok: span.rows,
        channels: s.conv_channels(),
        taps: s.conv_kernel,
        in_stride: s.in_width(),
        in_offset: s.inner(),
        pad: [0; 3],
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, CONV),
        Extent3::linear(s.conv_channels().div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&conv_args)),
            KernelArg::Buffer(proj),
            KernelArg::Buffer(span.state.conv),
            KernelArg::Buffer(ssm.conv_weight.view()),
            KernelArg::Buffer(ssm.conv_bias.view()),
            KernelArg::Buffer(xbc),
        ],
    )
    .map_err(|err| launch::launch_failure(CONV, err))?;
    let scan_args = ScanArgs {
        n_tok: span.rows,
        heads: s.heads,
        head_dim: s.head_dim,
        state: s.state,
        groups: s.groups,
        channels: s.conv_channels(),
        in_stride: s.in_width(),
        dt_offset: s.inner() + s.conv_channels(),
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, SCAN),
        Extent3::new(s.head_dim.div_ceil(SCAN_CHANNELS), s.heads, 1),
        Extent3::new(SCAN_CHANNELS * 32, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&scan_args)),
            KernelArg::Buffer(xbc),
            KernelArg::Buffer(proj),
            KernelArg::Buffer(ssm.dt_bias.view()),
            KernelArg::Buffer(ssm.a_log.view()),
            KernelArg::Buffer(ssm.d.view()),
            KernelArg::Buffer(span.state.state),
            KernelArg::Buffer(y),
        ],
    )
    .map_err(|err| launch::launch_failure(SCAN, err))
}

/// `y = w * rms_norm(silu(gate) * y)` per group, in place, over the whole chunk.
fn gate_norm(
    backend: &dyn Backend,
    ssm: &Ssm,
    geometry: &SsmGeometry,
    scratch: &SsmScratch,
    n_tok: u32,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_ssm_gate_norm_f32";
    let s = geometry.shape;
    let group = geometry.norm_group();
    let args = NormArgs {
        n_tok,
        inner: s.inner(),
        group_size: group,
        in_stride: s.in_width(),
        eps: geometry.eps,
        pad: [0; 3],
    };
    let smem = u32::try_from((u64::from(BLOCK_SUM_SLOTS) + u64::from(group)) * F32_BYTES)
        .unwrap_or(u32::MAX);
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::new(s.groups, n_tok, 1),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        smem,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(scratch.y.view()),
            KernelArg::Buffer(scratch.proj.view()),
            KernelArg::Buffer(ssm.norm.view()),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}
