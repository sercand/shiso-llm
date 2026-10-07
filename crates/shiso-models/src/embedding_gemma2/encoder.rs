//! The bidirectional pass: input rows in, the mean of the final rows out.
//!
//! One pass over the whole input, every layer at once. Nothing is chunked and
//! nothing is cached: attention reads keys on both sides of a query, so a
//! position cannot be finished before every position it can see has been
//! written, and an input is at most a few thousand rows. The scratch is held
//! between inputs in a [`Workspace`] and grows to the longest one seen.
//!
//! The order of a layer, as the reference writes it:
//!
//! ```text
//! h += post_attn_norm(attn(attn_norm(h)))
//! h += post_ffn_norm(down(gelu_tanh(gate(t)) * up(t)))     t = ffn_norm(h)
//! h += ple_norm(proj(gelu_tanh(ple_gate(h)) * ple_l))
//! h *= scalar
//! ```
//!
//! where `ple_l = norm(x0 W_l / sqrt(hidden))` is layer `l`'s slice of the
//! per-layer signal, made from the input rows `x0` alone.

use shiso_core::{DeviceBuffer, DeviceView};

use super::config::EncoderConfig;
use super::weights::EncoderWeights;
use crate::audio_ops::load::download_f32;
use crate::audio_ops::{
    AddRmsNormDesc, AxpyDesc, GeluMulDesc, NormDesc, Ops, RopeDesc, RopePosDesc, SpanAttnDesc,
    UnaryDesc, UnaryOp, Workspace,
};
use crate::embedder::EmbedError;

/// Workspace slots, by what each holds.
mod slot {
    /// The input rows, kept for every layer's per-layer slice.
    pub(super) const INPUT: usize = 0;
    /// The residual stream.
    pub(super) const RESIDUAL: usize = 1;
    /// A normalized copy of the stream.
    pub(super) const NORMED: usize = 2;
    pub(super) const Q: usize = 3;
    pub(super) const K: usize = 4;
    pub(super) const V: usize = 5;
    /// Attention's output, before the output projection.
    pub(super) const ATTN: usize = 6;
    /// A block's output, before it is added to the stream.
    pub(super) const BLOCK: usize = 7;
    /// The feed-forward gate, and the per-layer gate.
    pub(super) const GATE: usize = 8;
    pub(super) const UP: usize = 9;
    /// One layer's slice of the per-layer signal.
    pub(super) const PLE: usize = 10;
    /// Each row's sequence bounds, u32 `[rows][2]`, when inputs are packed.
    pub(super) const SEGMENTS: usize = 11;
    /// Each row's position in its sequence, u32 `[rows]`, when packed.
    pub(super) const POSITIONS: usize = 12;
    pub(super) const COUNT: usize = 13;
}

/// The resident encoder.
#[derive(Debug)]
pub struct Encoder {
    ops: Ops,
    config: EncoderConfig,
    weights: EncoderWeights,
    work: Workspace,
}

/// `n` floats of a slot.
fn view(buffer: &DeviceBuffer, floats: usize) -> Result<DeviceView<'_>, EmbedError> {
    Ok(buffer.slice(0, u64::try_from(floats * 4).unwrap_or(u64::MAX))?)
}

impl Encoder {
    /// An encoder over weights already resident.
    #[must_use]
    pub fn new(ops: Ops, config: EncoderConfig, weights: EncoderWeights) -> Self {
        Self {
            ops,
            config,
            weights,
            work: Workspace::new(),
        }
    }

    /// The geometry the encoder runs.
    #[must_use]
    pub const fn config(&self) -> &EncoderConfig {
        &self.config
    }

    /// The token table, read on the host.
    #[must_use]
    pub const fn weights(&self) -> &EncoderWeights {
        &self.weights
    }

    /// The device the encoder runs on.
    #[must_use]
    pub const fn ops(&self) -> &Ops {
        &self.ops
    }

    /// Runs several inputs' rows packed back to back (`[rows][hidden]`,
    /// already scaled where the reference scales them; `lengths[i]` rows for
    /// input `i`) and returns each input's mean of its final normalized rows,
    /// `[hidden]` each.
    ///
    /// Packing is invisible in the result: every row-wise step is the same
    /// whatever sits beside it, attention keeps each input to its own rows,
    /// and the rotation restarts at each input's first row. What it changes
    /// is the cost - one pass's six hundred launches serve every input in it,
    /// where a short input alone is all launch overhead.
    ///
    /// # Errors
    ///
    /// A device error, or [`EmbedError::Input`] for an empty input.
    #[expect(
        clippy::too_many_lines,
        reason = "one layer's launches in the reference's order read best as one sequence"
    )]
    pub fn pooled(
        &mut self,
        input: &[f32],
        lengths: &[usize],
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let cfg = &self.config;
        let width = cfg.hidden;
        let rows: usize = lengths.iter().sum();
        if lengths.contains(&0) || rows == 0 || input.len() != rows * width {
            return Err(EmbedError::Input {
                what: format!("an input of {rows} rows over {} values", input.len()),
            });
        }
        let widest_head = cfg.layers.iter().map(|l| l.head_dim).max().unwrap_or(0);
        let widest_kv = cfg
            .layers
            .iter()
            .map(|l| l.kv_heads * l.head_dim)
            .max()
            .unwrap_or(0);
        let q_width = cfg.q_heads * widest_head;
        let mut sizes = [0usize; slot::COUNT];
        sizes[slot::INPUT] = rows * width;
        sizes[slot::RESIDUAL] = rows * width;
        sizes[slot::NORMED] = rows * width;
        sizes[slot::Q] = rows * q_width;
        sizes[slot::K] = rows * widest_kv;
        sizes[slot::V] = rows * widest_kv;
        sizes[slot::ATTN] = rows * q_width;
        sizes[slot::BLOCK] = rows * width;
        sizes[slot::GATE] = rows * cfg.intermediate.max(cfg.ple_dim);
        sizes[slot::UP] = rows * cfg.intermediate;
        sizes[slot::PLE] = rows * cfg.ple_dim;
        sizes[slot::SEGMENTS] = rows * 2;
        sizes[slot::POSITIONS] = rows;
        self.work.reserve(&self.ops, &sizes)?;

        let ops = &self.ops;
        let wts = &self.weights;
        let work = &self.work;
        let x0 = view(work.get(slot::INPUT), rows * width)?;
        let stream = view(work.get(slot::RESIDUAL), rows * width)?;
        let normed = view(work.get(slot::NORMED), rows * width)?;
        let block = view(work.get(slot::BLOCK), rows * width)?;
        let bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        ops.backend().copy_h2d(x0, &bytes)?;
        ops.backend().copy_d2d(stream, x0)?;
        // One input runs exactly as it always has; several carry each row's
        // sequence bounds and position, which only then are read.
        let packed = lengths.len() > 1;
        let segments = view(work.get(slot::SEGMENTS), rows * 2)?;
        let positions = view(work.get(slot::POSITIONS), rows)?;
        if packed {
            let mut bounds = Vec::with_capacity(rows * 8);
            let mut at = Vec::with_capacity(rows * 4);
            let mut start = 0u32;
            for &len in lengths {
                let len = u32::try_from(len).unwrap_or(u32::MAX);
                let end = start.saturating_add(len);
                for p in 0..len {
                    bounds.extend(start.to_le_bytes());
                    bounds.extend(end.to_le_bytes());
                    at.extend(p.to_le_bytes());
                }
                start = end;
            }
            ops.backend().copy_h2d(segments, &bounds)?;
            ops.backend().copy_h2d(positions, &at)?;
        }

        let ple_scale = 1.0 / f32_of(width).sqrt();
        for (l, (dims, lw)) in cfg.layers.iter().zip(&wts.layers).enumerate() {
            let hd = dims.head_dim;
            let q_w = cfg.q_heads * hd;
            let kv_w = dims.kv_heads * hd;
            let q = view(work.get(slot::Q), rows * q_w)?;
            let k = view(work.get(slot::K), rows * kv_w)?;
            let v = view(work.get(slot::V), rows * kv_w)?;
            let attn = view(work.get(slot::ATTN), rows * q_w)?;

            // Attention.
            ops.rms_norm(&norm(
                normed,
                stream,
                Some(&lw.attn_norm),
                rows,
                width,
                cfg.eps,
            ))?;
            lw.q.forward(ops, normed, q, rows)?;
            lw.k.forward(ops, normed, k, rows)?;
            lw.v.forward(ops, normed, v, rows)?;
            // Per head: a query or key row of `heads * hd` is `heads` rows
            // of `hd`, so the head norms are one launch over all of them.
            ops.rms_norm(&norm(
                q,
                q,
                Some(&lw.q_norm),
                rows * cfg.q_heads,
                hd,
                cfg.eps,
            ))?;
            ops.rms_norm(&norm(
                k,
                k,
                Some(&lw.k_norm),
                rows * dims.kv_heads,
                hd,
                cfg.eps,
            ))?;
            // The value norm has no weight in this family.
            ops.rms_norm(&norm(v, v, None, rows * dims.kv_heads, hd, cfg.eps))?;
            for (x, heads) in [(q, cfg.q_heads), (k, dims.kv_heads)] {
                if packed {
                    ops.rope_half_rows(&RopePosDesc {
                        x,
                        positions,
                        rows,
                        heads,
                        head_dim: hd,
                        theta: dims.theta,
                    })?;
                } else {
                    ops.rope_half(&RopeDesc {
                        x,
                        rows,
                        heads,
                        head_dim: hd,
                        pos0: 0,
                        theta: dims.theta,
                    })?;
                }
            }
            // A global layer sees the whole input; `rows` on both sides is
            // every row there is.
            let radius = dims.radius.unwrap_or(rows);
            ops.span_attention(&SpanAttnDesc {
                out: attn,
                q,
                k,
                v,
                segments: packed.then_some(segments),
                n: rows,
                heads: cfg.q_heads,
                kv_heads: dims.kv_heads,
                head_dim: hd,
                left: radius,
                right: radius,
                // The query and key norms stand in for the usual
                // `1/sqrt(hd)`: the reference multiplies by exactly 1.
                scale: 1.0,
            })?;
            lw.o.forward(ops, attn, block, rows)?;
            ops.rms_norm(&norm(
                block,
                block,
                Some(&lw.post_attn_norm),
                rows,
                width,
                cfg.eps,
            ))?;

            // Feed-forward, with the residual add fused into its pre-norm.
            ops.add_rms_norm(&AddRmsNormDesc {
                out: normed,
                x: stream,
                add: block,
                weight: Some(lw.ffn_norm.view()),
                rows,
                dim: width,
                eps: cfg.eps,
            })?;
            let inter = cfg.intermediate;
            let gate = view(work.get(slot::GATE), rows * inter)?;
            let up = view(work.get(slot::UP), rows * inter)?;
            lw.gate.forward(ops, normed, gate, rows)?;
            lw.up.forward(ops, normed, up, rows)?;
            ops.gelu_tanh_mul(&GeluMulDesc {
                out: gate,
                gate,
                up,
                rows,
                cols: inter,
                up_stride: inter,
                up_offset: 0,
            })?;
            lw.down.forward(ops, gate, block, rows)?;
            ops.rms_norm(&norm(
                block,
                block,
                Some(&lw.post_ffn_norm),
                rows,
                width,
                cfg.eps,
            ))?;
            ops.axpy(&AxpyDesc {
                y: stream,
                x: block,
                alpha: 1.0,
                count: rows * width,
            })?;

            // The per-layer block: this layer's slice of the input's signal
            // gates a bottleneck over the stream.
            let pd = cfg.ple_dim;
            let ple = view(work.get(slot::PLE), rows * pd)?;
            wts.ple_model_proj[l].forward(ops, x0, ple, rows)?;
            ops.unary(&UnaryDesc {
                op: UnaryOp::Scale(ple_scale),
                out: ple,
                input: ple,
                count: rows * pd,
            })?;
            ops.rms_norm(&norm(ple, ple, Some(&wts.ple_proj_norm), rows, pd, cfg.eps))?;
            let ple_gate = view(work.get(slot::GATE), rows * pd)?;
            lw.ple_gate.forward(ops, stream, ple_gate, rows)?;
            ops.gelu_tanh_mul(&GeluMulDesc {
                out: ple_gate,
                gate: ple_gate,
                up: ple,
                rows,
                cols: pd,
                up_stride: pd,
                up_offset: 0,
            })?;
            lw.ple_proj.forward(ops, ple_gate, block, rows)?;
            ops.rms_norm(&norm(
                block,
                block,
                Some(&lw.ple_norm),
                rows,
                width,
                cfg.eps,
            ))?;
            ops.axpy(&AxpyDesc {
                y: stream,
                x: block,
                alpha: 1.0,
                count: rows * width,
            })?;
            ops.unary(&UnaryDesc {
                op: UnaryOp::Scale(lw.scalar),
                out: stream,
                input: stream,
                count: rows * width,
            })?;
        }
        ops.rms_norm(&norm(
            normed,
            stream,
            Some(&wts.output_norm),
            rows,
            width,
            cfg.eps,
        ))?;
        let last = download_f32(ops, normed, rows * width).map_err(EmbedError::Audio)?;
        // The mean in f64: an input of thousands of rows summed in f32 loses
        // the low bits of every row past the first few hundred.
        let mut rows_left = last.chunks_exact(width);
        Ok(lengths
            .iter()
            .map(|&len| {
                let mut mean = vec![0f64; width];
                for row in rows_left.by_ref().take(len) {
                    for (m, v) in mean.iter_mut().zip(row) {
                        *m += f64::from(*v);
                    }
                }
                let inv = 1.0 / f64::from(f32_of(len));
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the mean of f32 rows is inside f32's range"
                )]
                mean.into_iter().map(|m| (m * inv) as f32).collect()
            })
            .collect())
    }

    /// The output projection of a pooled row: `[out_dim]`.
    #[must_use]
    pub fn project(&self, pooled: &[f32]) -> Vec<f32> {
        let h = self.config.hidden;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a projected component is inside f32's range"
        )]
        self.weights
            .output
            .chunks_exact(h)
            .map(|row| {
                row.iter()
                    .zip(pooled)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

/// A norm descriptor over `rows` rows of `dim`.
fn norm<'a>(
    out: DeviceView<'a>,
    input: DeviceView<'a>,
    weight: Option<&'a DeviceBuffer>,
    rows: usize,
    dim: usize,
    eps: f32,
) -> NormDesc<'a> {
    NormDesc {
        out,
        input,
        weight: weight.map(DeviceBuffer::view),
        bias: None,
        rows,
        dim,
        eps,
    }
}

/// A count as `f32`: exact for every width and row count an input reaches.
#[expect(
    clippy::cast_precision_loss,
    reason = "widths and row counts stay far below 2^24, where f32 is exact"
)]
pub(super) fn f32_of(n: usize) -> f32 {
    n as f32
}
