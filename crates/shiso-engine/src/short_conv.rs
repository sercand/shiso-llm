//! The gated short-convolution block: its weights, the scratch one chunk uses,
//! and the three launches that run it.
//!
//! `out_proj(c * conv(b * x))`, where one input projection writes `b`, `c` and
//! `x` side by side in every row and the convolution is depthwise and causal
//! over a few taps of the residual width. There is no bias, no activation and
//! no state matrix: the block's whole memory is the last `taps - 1` gated
//! inputs `b * x`.
//!
//! The sequential launch is the engine's own (`metal/dense/dense_short_conv.metal`)
//! and reads the projection's rows in place, both gates included, so the
//! block is two projections and one launch between them.
//!
//! # The window is not scratch
//!
//! As for [`crate::ssm`], the window belongs to one session and arrives as a
//! view the caller owns. A batched decode step folds each row into its own
//! session's window, so the convolution runs a row at a time there while both
//! projections stay batched.

use std::sync::Arc;

use shiso_core::Extent3;
use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};

use crate::blocks::{GdnIo, GdnState, GdnStates};
use crate::launch::{self, BLOCK_LINEAR, DENSE, F32_BYTES};
use crate::projection::{self, Weight};
use crate::shape::ShortConvShape;

/// Widest convolution the kernel's window registers hold.
const MAX_TAPS: u32 = 8;

/// The input projection's width in residual widths: `b`, `c` and `x`.
const GATED_WIDTHS: u32 = 3;

/// One short-convolution block's widths, checked against what the kernel was
/// written for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvGeometry {
    /// Channels the convolution runs over: the residual width.
    pub channels: u32,
    /// Width of the convolution, the current input included.
    pub taps: u32,
}

impl ShortConvGeometry {
    /// Derives the block's geometry, refusing widths the kernel cannot carry.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a convolution wider than the window
    /// registers or narrower than two taps (one tap has no window to carry),
    /// and for a zero residual width.
    pub fn new(shape: ShortConvShape, hidden: u32) -> CoreResult<Self> {
        if shape.taps < 2 || shape.taps > MAX_TAPS {
            return Err(CoreError::InvalidValue {
                key: "short-convolution width",
                value: format!("{} taps, and the window holds 2 to {MAX_TAPS}", shape.taps),
            });
        }
        if hidden == 0 {
            return Err(CoreError::InvalidValue {
                key: "short-convolution channels",
                value: "0".to_owned(),
            });
        }
        Ok(Self {
            channels: hidden,
            taps: shape.taps,
        })
    }

    /// Width of the block's input projection: `[b | c | x]`.
    #[must_use]
    pub const fn in_width(&self) -> u32 {
        GATED_WIDTHS * self.channels
    }
}

/// One short-convolution block's weights, resident on the device.
#[derive(Debug)]
pub struct ShortConv {
    /// `hidden -> [b | c | x]`, three residual widths.
    pub in_proj: Weight,
    /// The depthwise convolution, `[channels, taps]` f32, oldest tap first.
    pub conv_weight: DeviceBuffer,
    /// `hidden -> hidden`.
    pub out_proj: Weight,
}

/// Scratch one chunk's short-convolution block uses, sized once per stack.
#[derive(Debug)]
pub struct ShortConvScratch {
    /// The input projection's rows, `[rows, 3 * channels]`.
    proj: DeviceBuffer,
    /// The gated convolution's output, `[rows, channels]`.
    y: DeviceBuffer,
}

impl ShortConvScratch {
    /// Allocates the rows a chunk of `rows` tokens needs.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation fails.
    pub fn new(
        backend: &Arc<dyn Backend>,
        geometry: &ShortConvGeometry,
        rows: u32,
    ) -> CoreResult<Self> {
        let floats = |width: u32| {
            u64::from(width)
                .saturating_mul(u64::from(rows))
                .saturating_mul(F32_BYTES)
        };
        Ok(Self {
            proj: alloc(backend, floats(geometry.in_width()))?,
            y: alloc(backend, floats(geometry.channels))?,
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvArgs {
    n_tok: u32,
    channels: u32,
    taps: u32,
    pad: u32,
}

const _: () = assert!(size_of::<ConvArgs>() == 16);

/// Runs one short-convolution block over `n_tok` rows of `io.input`.
///
/// # Errors
///
/// [`CoreError::Unsupported`] for a pass that asks for a per-row trail, which
/// this block does not record; [`CoreError::ShapeMismatch`] when a batch's
/// state list is not as long as the pass; whatever [`projection::projection`]
/// reports for either projection; and [`CoreError::Backend`] naming the entry
/// when a launch fails.
pub fn run(
    backend: &dyn Backend,
    block: &ShortConv,
    geometry: &ShortConvGeometry,
    scratch: &ShortConvScratch,
    io: GdnIo<'_>,
    n_tok: u32,
) -> CoreResult<()> {
    if io.trail.is_some() {
        // Accepting a trail and leaving it unwritten would let a rollback
        // restore whatever the allocator last left there.
        return Err(CoreError::Unsupported {
            what: "a per-row state trail on a short-convolution block",
            backend: "engine",
        });
    }
    projection::projection(
        backend,
        &block.in_proj,
        n_tok,
        io.input,
        scratch.proj.view(),
    )?;
    match io.states {
        GdnStates::Chunk(state) => conv(backend, block, *geometry, scratch, 0, n_tok, state)?,
        GdnStates::PerRow(states) => {
            if states.len() != n_tok as usize {
                return Err(CoreError::ShapeMismatch {
                    field: "short-convolution batch states",
                    expected: u64::from(n_tok),
                    got: u64::try_from(states.len()).unwrap_or(u64::MAX),
                });
            }
            for (row, state) in (0..n_tok).zip(states) {
                conv(backend, block, *geometry, scratch, row, 1, *state)?;
            }
        }
    }
    projection::projection(
        backend,
        &block.out_proj,
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

/// The gated convolution over `rows` rows from `first`, folded into one
/// session's window in order.
fn conv(
    backend: &dyn Backend,
    block: &ShortConv,
    geometry: ShortConvGeometry,
    scratch: &ShortConvScratch,
    first: u32,
    rows: u32,
    state: GdnState<'_>,
) -> CoreResult<()> {
    const NAME: &str = "kernel_dense_short_conv_f32";
    let proj = rows_of(&scratch.proj, geometry.in_width(), first, rows)?;
    let y = rows_of(&scratch.y, geometry.channels, first, rows)?;
    let args = ConvArgs {
        n_tok: rows,
        channels: geometry.channels,
        taps: geometry.taps,
        pad: 0,
    };
    launch::run(
        backend,
        KernelRef::new(DENSE, NAME),
        Extent3::linear(geometry.channels.div_ceil(BLOCK_LINEAR)),
        Extent3::new(BLOCK_LINEAR, 1, 1),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(proj),
            KernelArg::Buffer(state.conv),
            KernelArg::Buffer(block.conv_weight.view()),
            KernelArg::Buffer(y),
        ],
    )
    .map_err(|err| launch::launch_failure(NAME, err))
}
