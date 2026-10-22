//! The encoder pass on the device: token rows in, the first position's final
//! vector out.
//!
//! One pass over one unpadded sequence, so attention needs no mask: every
//! position sees every other. Scratch is held between calls in a
//! [`Workspace`] and grows to the longest input seen.
//!
//! The order of a layer (post-norm):
//!
//! ```text
//! h = norm_a(h + out(attn(q(h), k(h), v(h))))
//! h = norm_b(h + lin2(gelu(lin1(h))))
//! ```

use shiso_core::DeviceView;

use super::ClassifyError;
use super::config::ClassifierConfig;
use super::weights::{ClassifierWeights, NormWeights};
use crate::audio_ops::load::download_f32;
use crate::audio_ops::{AxpyDesc, NormDesc, Ops, SpanAttnDesc, UnaryDesc, UnaryOp, Workspace};

/// Workspace slots, by what each holds.
mod slot {
    /// The summed embedding rows, before their norm.
    pub(super) const INPUT: usize = 0;
    /// The residual stream.
    pub(super) const STREAM: usize = 1;
    pub(super) const Q: usize = 2;
    pub(super) const K: usize = 3;
    pub(super) const V: usize = 4;
    /// Attention's output, before the output projection.
    pub(super) const ATTN: usize = 5;
    /// A block's output, before it joins the stream.
    pub(super) const BLOCK: usize = 6;
    /// The feed-forward's wide intermediate.
    pub(super) const WIDE: usize = 7;
    pub(super) const COUNT: usize = 8;
}

/// The resident encoder.
#[derive(Debug)]
pub(super) struct Encoder {
    ops: Ops,
    cfg: ClassifierConfig,
    weights: ClassifierWeights,
    work: Workspace,
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "used as the function a `map_err` calls"
)]
fn device(error: shiso_core::CoreError) -> ClassifyError {
    ClassifyError::Device(error.to_string())
}

fn view(work: &Workspace, slot: usize, floats: usize) -> Result<DeviceView<'_>, ClassifyError> {
    work.get(slot)
        .slice(0, u64::try_from(floats * 4).unwrap_or(u64::MAX))
        .map_err(device)
}

fn norm<'a>(
    out: DeviceView<'a>,
    input: DeviceView<'a>,
    n: &'a NormWeights,
    (rows, dim): (usize, usize),
    eps: f32,
) -> NormDesc<'a> {
    NormDesc {
        out,
        input,
        weight: Some(n.weight.view()),
        bias: Some(n.bias.view()),
        rows,
        dim,
        eps,
    }
}

impl Encoder {
    /// An encoder over weights already resident.
    #[must_use]
    pub(super) fn new(ops: Ops, cfg: ClassifierConfig, weights: ClassifierWeights) -> Self {
        Self {
            ops,
            cfg,
            weights,
            work: Workspace::new(),
        }
    }

    /// The geometry the encoder runs.
    #[must_use]
    pub(super) const fn config(&self) -> &ClassifierConfig {
        &self.cfg
    }

    /// The head's weights, which are on the host.
    #[must_use]
    pub(super) const fn weights(&self) -> &ClassifierWeights {
        &self.weights
    }

    /// The final vector of the first position of `ids`, `[dim]`.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::BadTensor`] for an id or a length the tables do not
    /// have, [`ClassifyError::Device`] when a launch or copy fails.
    #[expect(
        clippy::many_single_char_names,
        reason = "n rows, d width and q k v are the names of the pass's own quantities"
    )]
    pub(super) fn cls_vector(&mut self, ids: &[u32]) -> Result<Vec<f32>, ClassifyError> {
        let n = ids.len();
        let cfg = &self.cfg;
        let d = cfg.dim;
        if n == 0 || n > cfg.max_positions {
            return Err(ClassifyError::BadConfig(format!(
                "a sequence of {n} ids does not fit {} positions",
                cfg.max_positions
            )));
        }
        let mut rows = vec![0f32; n * d];
        let mut pos = vec![0f32; d];
        for (i, (id, row)) in ids.iter().zip(rows.chunks_exact_mut(d)).enumerate() {
            self.weights.word.row(*id as usize, row)?;
            self.weights.position.row(i, &mut pos)?;
            for (r, p) in row.iter_mut().zip(&pos) {
                *r += p;
            }
        }
        let mut sizes = [0usize; slot::COUNT];
        sizes[slot::INPUT] = n * d;
        sizes[slot::STREAM] = n * d;
        sizes[slot::Q] = n * d;
        sizes[slot::K] = n * d;
        sizes[slot::V] = n * d;
        sizes[slot::ATTN] = n * d;
        sizes[slot::BLOCK] = n * d;
        sizes[slot::WIDE] = n * cfg.hidden_dim;
        self.work.reserve(&self.ops, &sizes).map_err(device)?;

        let ops = &self.ops;
        let work = &self.work;
        let input = view(work, slot::INPUT, n * d)?;
        let stream = view(work, slot::STREAM, n * d)?;
        let block = view(work, slot::BLOCK, n * d)?;
        let (q, k, v) = (
            view(work, slot::Q, n * d)?,
            view(work, slot::K, n * d)?,
            view(work, slot::V, n * d)?,
        );
        let attn = view(work, slot::ATTN, n * d)?;
        let wide = view(work, slot::WIDE, n * cfg.hidden_dim)?;
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        ops.backend().copy_h2d(input, &bytes).map_err(device)?;
        ops.layer_norm(&norm(
            stream,
            input,
            &self.weights.embed_norm,
            (n, d),
            cfg.eps,
        ))
        .map_err(device)?;
        let add = |y, x| {
            ops.axpy(&AxpyDesc {
                y,
                x,
                alpha: 1.0,
                count: n * d,
            })
            .map_err(device)
        };
        for lw in &self.weights.layers {
            lw.q.forward_few(ops, stream, q, n).map_err(device)?;
            lw.k.forward_few(ops, stream, k, n).map_err(device)?;
            lw.v.forward_few(ops, stream, v, n).map_err(device)?;
            ops.span_attention(&SpanAttnDesc {
                out: attn,
                q,
                k,
                v,
                segments: None,
                n,
                heads: cfg.heads,
                kv_heads: cfg.heads,
                head_dim: cfg.head_dim(),
                left: n,
                right: n,
                scale: 1.0 / super::f32_of(cfg.head_dim()).sqrt(),
            })
            .map_err(device)?;
            lw.out.forward_few(ops, attn, block, n).map_err(device)?;
            add(stream, block)?;
            ops.layer_norm(&norm(stream, stream, &lw.attn_norm, (n, d), cfg.eps))
                .map_err(device)?;
            lw.ffn_in
                .forward_few(ops, stream, wide, n)
                .map_err(device)?;
            ops.unary(&UnaryDesc {
                op: UnaryOp::GeluErf,
                out: wide,
                input: wide,
                count: n * cfg.hidden_dim,
            })
            .map_err(device)?;
            lw.ffn_out
                .forward_few(ops, wide, block, n)
                .map_err(device)?;
            add(stream, block)?;
            ops.layer_norm(&norm(stream, stream, &lw.ffn_norm, (n, d), cfg.eps))
                .map_err(device)?;
        }
        // Only the first position feeds the head.
        download_f32(ops, stream, d).map_err(|e| ClassifyError::Device(e.to_string()))
    }
}
