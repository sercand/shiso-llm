//! What a recurrent block is made of: its widths, the two offsets every launch
//! after the convolution needs, its weights, and the scratch rows one chunk uses.
//!
//! Not the launches. Which kernels a block dispatches over all of this is a fact
//! about the checkpoint family, so they live behind
//! [`crate::blocks::BlockKernels`] and qwen35's are in `shiso-models`. The block
//! runs over a whole chunk of tokens, so nothing here is sized per token except
//! by `n_tok`, and a prefill and a decode differ by that number alone.
//!
//! # The two states are not scratch
//!
//! The convolution window and the delta-rule state are deliberately absent from
//! [`GdnScratch`]. They are per-session state: a launch rewrites them in place,
//! a checkpoint is a copy of exactly those bytes, and they outlive every call in
//! this file. They therefore arrive as views the caller owns, and nothing here
//! allocates them or gets to decide how long they live.

use std::sync::Arc;

use crate::shape::GdnShape;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, alloc};
use shiso_core::error::{CoreError, CoreResult};

use crate::launch::{self, F32_BYTES};
use crate::projection::Weight;

/// Checks a derived width against the `int32_t` the argument blocks carry.
fn wide(value: u64, field: &'static str) -> CoreResult<u32> {
    u32::try_from(value).map_err(|_| CoreError::ShapeMismatch {
        field,
        expected: value,
        got: u64::from(u32::MAX),
    })
}

/// One recurrent block's widths, with the two derived strides spelled out once.
///
/// The mix row is the block's widest buffer and the two offsets into it are what
/// every kernel after the convolution needs, so they are derived here rather than
/// recomputed at each launch: a mix row is `[q heads | k heads | v heads]`, one
/// head width per head, and the value heads decide both the output width and the
/// number of recurrent states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnGeometry {
    /// Query/key heads.
    pub qk_heads: u32,
    /// Value heads, which is also the number of recurrent states.
    pub v_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// Window of the short causal convolution.
    pub conv_kernel: u32,
    /// Floats in one mix row, which is also the convolution's channel count.
    pub channels: u32,
    /// Width of the block's output before its output projection.
    pub width: u32,
}

impl GdnGeometry {
    /// Derives the block's widths, rejecting any width an argument block cannot
    /// carry.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::ShapeMismatch`] when the head widths do not make a
    /// mix row the kernels can stride over.
    pub fn new(gdn: &GdnShape) -> CoreResult<Self> {
        let qk = u64::from(gdn.qk_heads);
        let values = u64::from(gdn.v_heads);
        let head = u64::from(gdn.head_dim);
        let channels = qk
            .saturating_mul(2)
            .saturating_mul(head)
            .saturating_add(values.saturating_mul(head));
        Ok(Self {
            qk_heads: gdn.qk_heads,
            v_heads: gdn.v_heads,
            head_dim: gdn.head_dim,
            conv_kernel: gdn.conv_kernel,
            channels: wide(channels, "gdn mix width")?,
            width: wide(values.saturating_mul(head), "gdn output width")?,
        })
    }

    /// Floats from the start of a mix row to its key half.
    #[must_use]
    pub const fn k_offset(&self) -> u32 {
        self.qk_heads.saturating_mul(self.head_dim)
    }

    /// Floats from the start of a mix row to its value half.
    #[must_use]
    pub const fn v_offset(&self) -> u32 {
        self.k_offset().saturating_mul(2)
    }

    /// Bytes of one token's mix row.
    #[must_use]
    pub fn mix_row_bytes(&self) -> u64 {
        u64::from(self.channels).saturating_mul(F32_BYTES)
    }

    /// Bytes of one token's recurrent output, before the output projection.
    #[must_use]
    pub fn out_row_bytes(&self) -> u64 {
        u64::from(self.width).saturating_mul(F32_BYTES)
    }

    /// Bytes of one token's per-head rows: decay and the update strength.
    #[must_use]
    pub fn head_row_bytes(&self) -> u64 {
        u64::from(self.v_heads).saturating_mul(F32_BYTES)
    }
}

/// One recurrent block's weights, resident on the device.
///
/// The four projections are quantized matrices and go through
/// [`Weight`]; the four vectors beside them are f32 in every checkpoint and are
/// held as plain buffers, because a [`Weight`] would insist on a block layout
/// they do not have.
#[derive(Debug)]
pub struct Gdn {
    /// Combined query, key and value projection.
    pub qkv: Weight,
    /// Output gate projection, whose row the block normalizes around.
    pub gate: Weight,
    /// Decay driver, one row per value head.
    pub alpha: Weight,
    /// Write-strength driver, one row per value head.
    pub beta: Weight,
    /// Projection back to the residual width.
    pub out: Weight,
    /// Convolution weights, `channels * conv_kernel` f32.
    pub conv1d: DeviceBuffer,
    /// Per-head recurrence coefficient, already negated.
    pub a: DeviceBuffer,
    /// Per-head bias added before the softplus.
    pub dt_bias: DeviceBuffer,
    /// Per-head normalization weight, applied raw.
    pub norm: DeviceBuffer,
}

/// The scratch a recurrent block needs for `n_tok` tokens, reused across layers.
///
/// The two state buffers are deliberately absent: the convolution window and the
/// recurrent state belong to a session and survive a whole generation, while
/// everything here is scratch that outlives no call.
#[derive(Debug)]
pub struct GdnScratch {
    /// The projection's output, which the convolution reads.
    pub x: DeviceBuffer,
    /// The convolution's output, which normalization then rewrites in place.
    pub mix: DeviceBuffer,
    /// The decay driver's projection.
    pub alpha_proj: DeviceBuffer,
    /// The write strength's projection.
    pub beta_proj: DeviceBuffer,
    /// The decay the recurrence consumes, one f32 per head and token.
    pub decay: DeviceBuffer,
    /// The write strengths, as the gate kernel writes them.
    pub beta_out: DeviceBuffer,
    /// The gate projection's output.
    pub z: DeviceBuffer,
    /// The recurrence's output, before the gate.
    pub attn_out: DeviceBuffer,
    /// The gated output, which is what the output projection reads.
    pub gated: DeviceBuffer,
    /// Where the chunk hybrid spills its deltas: one run per value head.
    ///
    /// Allocated for every stack rather than only for the machine the tuner
    /// hands the hybrid to. Its size is a geometry fact - a run of
    /// [`launch::GDN_CHUNK_RUN`] tokens for each of a head's columns, about a
    /// megabyte at this model's widths, beside weights measured in gigabytes -
    /// and a buffer whose existence depended on a benchmark result would be a
    /// buffer that is missing on exactly one machine and present on every one
    /// the developer has.
    pub spill: DeviceBuffer,
}

impl GdnScratch {
    /// Allocates every row a chunk of `n_tok` tokens needs.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when an allocation fails.
    pub(crate) fn new(
        backend: &Arc<dyn Backend>,
        geometry: &GdnGeometry,
        n_tok: u32,
    ) -> CoreResult<Self> {
        let tokens = u64::from(n_tok);
        let mix = geometry.mix_row_bytes().saturating_mul(tokens);
        let out = geometry.out_row_bytes().saturating_mul(tokens);
        let heads = geometry.head_row_bytes().saturating_mul(tokens);
        // A whole run per value head, whatever the chunk this scratch is sized
        // for: the chunk kernel walks `GDN_CHUNK_RUN` tokens at a time and
        // reuses these rows between runs, so the spill is as wide as one run
        // and never as wide as `n_tok`.
        let spill = u64::from(geometry.v_heads)
            .saturating_mul(u64::from(launch::GDN_CHUNK_RUN))
            .saturating_mul(u64::from(geometry.head_dim))
            .saturating_mul(F32_BYTES);
        Ok(Self {
            x: alloc(backend, mix)?,
            mix: alloc(backend, mix)?,
            alpha_proj: alloc(backend, heads)?,
            beta_proj: alloc(backend, heads)?,
            decay: alloc(backend, heads)?,
            beta_out: alloc(backend, heads)?,
            z: alloc(backend, out)?,
            attn_out: alloc(backend, out)?,
            gated: alloc(backend, out)?,
            spill: alloc(backend, spill)?,
        })
    }
}
