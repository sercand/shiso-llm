//! The rotary transformer all three of the encoder's branches run.
//!
//! Pre-norm blocks between an input and an output layer norm (eps 1e-5,
//! weight and bias): attention whose queries and keys are, for the speech
//! branch and the timestamp head, layer-normed per head before the first
//! `rot_dims` of each head turn, bidirectional over every row of the call;
//! then an exact-GELU feed-forward. Positions restart at zero every call,
//! which is how the reference encodes each window on its own.

#![expect(
    clippy::cast_precision_loss,
    reason = "a head width of at most 128 is exact in f32"
)]

use shiso_audio::AudioError;
use shiso_core::{DeviceBuffer, DeviceView};

use super::weights::{Norm, Transformer};
use crate::audio_ops::{
    AttnWindow, AxpyDesc, NormDesc, Ops, RopePartialDesc, UnaryDesc, UnaryOp, WindowAttnDesc,
};

/// Epsilon of every layer norm.
const EPS: f32 = 1e-5;

/// The rotary base.
const THETA: f32 = 10_000.0;

/// `out = layer_norm(input)` over `rows` rows of `dim`.
pub(crate) fn layer_norm(
    ops: &Ops,
    out: DeviceView<'_>,
    input: DeviceView<'_>,
    norm: &Norm,
    rows: usize,
    dim: usize,
) -> Result<(), AudioError> {
    Ok(ops.layer_norm(&NormDesc {
        out,
        input,
        weight: Some(norm.weight.view()),
        bias: Some(norm.bias.view()),
        rows,
        dim,
        eps: EPS,
    })?)
}

/// `y += x` over `count` floats.
pub(crate) fn add(
    ops: &Ops,
    y: DeviceView<'_>,
    x: DeviceView<'_>,
    count: usize,
) -> Result<(), AudioError> {
    Ok(ops.axpy(&AxpyDesc {
        y,
        x,
        alpha: 1.0,
        count,
    })?)
}

impl Transformer {
    /// The transformer over `n` rows of `x` (`[n][d_model]`, on the device):
    /// its final-normed output.
    ///
    /// # Errors
    ///
    /// [`AudioError::Empty`] for no rows, or a device error.
    #[expect(
        clippy::many_single_char_names,
        reason = "the buffers keep the names the model math uses: h, a, q, k, v, t, f"
    )]
    pub fn forward(
        &self,
        ops: &Ops,
        x: &DeviceBuffer,
        n: usize,
    ) -> Result<DeviceBuffer, AudioError> {
        if n == 0 {
            return Err(AudioError::Empty);
        }
        let c = self.dims;
        let d = c.d_model;
        let h = ops.scratch(n * d)?;
        let a = ops.scratch(n * d)?;
        let q = ops.scratch(n * d)?;
        let k = ops.scratch(n * d)?;
        let v = ops.scratch(n * d)?;
        let t = ops.scratch(n * d)?;
        let f = ops.scratch(n * c.ff)?;
        layer_norm(ops, h.view(), x.view(), &self.embed_norm, n, d)?;
        let window = AttnWindow {
            q_abs0: 0,
            k_abs0: 0,
            k_end: n,
            chunk: n,
            left_chunks: 0,
        };
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        for layer in &self.layers {
            layer_norm(ops, a.view(), h.view(), &layer.norm1, n, d)?;
            layer.q.forward(ops, a.view(), q.view(), n)?;
            layer.k.forward(ops, a.view(), k.view(), n)?;
            layer.v.forward(ops, a.view(), v.view(), n)?;
            if let Some((q_norm, k_norm)) = &layer.qk_norm {
                // `[n][heads][head_dim]` is `[n·heads][head_dim]` in the same
                // bytes, so a per-head norm is a norm over shorter rows.
                layer_norm(ops, q.view(), q.view(), q_norm, n * c.heads, c.head_dim)?;
                layer_norm(ops, k.view(), k.view(), k_norm, n * c.heads, c.head_dim)?;
            }
            for buf in [&q, &k] {
                ops.rope_partial(&RopePartialDesc {
                    x: buf.view(),
                    rows: n,
                    heads: c.heads,
                    head_dim: c.head_dim,
                    rot_dims: c.rot_dims,
                    pos0: 0,
                    theta: THETA,
                })?;
            }
            ops.window_attention(&WindowAttnDesc {
                out: a.view(),
                q: q.view(),
                k: k.view(),
                v: v.view(),
                n_q: n,
                n_k: n,
                heads: c.heads,
                head_dim: c.head_dim,
                window,
                bias_uv: None,
                q_dim_scale: None,
                rel: None,
                scale,
                ac_scale: 1.0,
                softcap: None,
            })?;
            layer.o.forward(ops, a.view(), t.view(), n)?;
            add(ops, h.view(), t.view(), n * d)?;
            layer_norm(ops, a.view(), h.view(), &layer.norm2, n, d)?;
            layer.fc1.forward(ops, a.view(), f.view(), n)?;
            ops.unary(&UnaryDesc {
                op: UnaryOp::GeluErf,
                out: f.view(),
                input: f.view(),
                count: n * c.ff,
            })?;
            layer.fc2.forward(ops, f.view(), t.view(), n)?;
            add(ops, h.view(), t.view(), n * d)?;
        }
        layer_norm(ops, a.view(), h.view(), &self.final_norm, n, d)?;
        Ok(a)
    }
}
