//! The layers the family's four graphs share, composed from the diffusion
//! driver's kernels.
//!
//! Every graph here is a transformer over token rows, and every one of them
//! needs a few things the driver does not carry as one launch: a layer norm
//! with an affine, a rotary turn from an arbitrary per-token table, a
//! per-head scaled RMS norm, and attention whose queries and keys are
//! different rows. Each is built from launches the driver already has, so the
//! family adds no kernel:
//!
//! - **Affine layer norm**: the weightless norm, then `x + (w - 1) * x + b`
//!   through [`Ops::modulate`] (`x * (1 + s) + t` with `s = w - 1`).
//! - **Rotary turn**: `rope(q) = q * cos + turn(q) * sin`, where `turn` is the
//!   fixed signed permutation of a head's dimensions the rotation pairs them
//!   with. `turn` is linear, so `turn(q) = x @ turn(W)^T + turn(b)`: a second
//!   projection whose weight rows are the query weight's, permuted and
//!   negated at load. The two products then meet through two gated adds with
//!   full-size gates (`q + (cos - 1) * q`, then `+ sin * turn(q)`).
//! - **Per-head RMS norm** (`normalize(x) * gamma * sqrt(d)`): the driver's
//!   row RMS norm over `(tokens * heads, d)` rows with a unit weight, which is
//!   `normalize(x) * sqrt(d)`, then the per-head gamma as a gated add.
//! - **Attention**: a block of query rows at a time, per head, as two GEMMs
//!   around a row softmax - the VAE mid block's scheme. It serves cross
//!   attention, which the joint kernel cannot (it attends a row set to
//!   itself), and self attention too: the joint kernel's fused path stages
//!   queries and keys in half precision, which over the image tower's 32
//!   blocks cost 7.5e-3 of the reference's scale where this path measures
//!   4.7e-5 (and runs faster at these widths).

use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{
    AddGatedCall, GemmCall, LayerNormCall, LinWeight, ModulateCall, NormCall, Ops, Scratch,
    SliceExt, WeightKind,
};
use crate::diffusion::weights::StStore;

/// Scores the attention forms at once, in f32 elements (256 MiB).
const SCORE_BUDGET: usize = 1 << 26;

/// Bytes of `n` f32 values.
pub(crate) const fn bytes(n: usize) -> u64 {
    (n * 4) as u64
}

/// A linear built from host values (a split, permuted or widened weight).
///
/// # Errors
///
/// [`ImageGenError::Backend`] on an upload failure or a weight whose length
/// is not `out_dim * in_dim`.
pub(crate) fn host_lin(
    ops: &Ops,
    w: &[f32],
    bias: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Lin, ImageGenError> {
    if w.len() != out_dim * in_dim || bias.is_some_and(|b| b.len() != out_dim) {
        return Err(ImageGenError::Backend(format!(
            "a {out_dim}x{in_dim} linear was handed {} weights",
            w.len()
        )));
    }
    Ok(Lin {
        w: ops.upload_f32("host linear", w)?,
        kind: WeightKind::F32,
        scale: 1.0,
        row_scale: None,
        rotate: None,
        bias: bias
            .map(|b| ops.upload_f32("host linear bias", b))
            .transpose()?,
        out_dim,
        in_dim,
    })
}

/// Reads `{name}.weight` (and `.bias` when present) as host f32 values.
///
/// # Errors
///
/// [`ImageGenError::Checkpoint`] for a missing or non-float tensor.
pub(crate) fn read_linear(
    store: &StStore,
    name: &str,
) -> Result<(Vec<f32>, Option<Vec<f32>>), ImageGenError> {
    let w = store.f32_vec(&format!("{name}.weight"))?;
    let bias_name = format!("{name}.bias");
    let bias = if store.contains(&bias_name) {
        Some(store.f32_vec(&bias_name)?)
    } else {
        None
    };
    Ok((w, bias))
}

/// `out = x @ W^T + b` over `rows` rows, into a fresh scratch buffer.
///
/// # Errors
///
/// A device failure.
pub(crate) fn linear<'o>(
    ops: &'o Ops,
    step: &'static str,
    lin: &Lin,
    x: shiso_core::DeviceView<'_>,
    rows: usize,
) -> Result<Scratch<'o>, ImageGenError> {
    let out = ops.alloc_f32(step, rows * lin.out_dim)?;
    ops.gemm(&GemmCall {
        step,
        out: out.view(),
        input: x,
        weight: lin.weight(),
        m: rows,
        n: lin.out_dim,
        k: lin.in_dim,
    })?;
    Ok(out)
}

/// The denoiser's sinusoidal embedding of 3D positions: per axis, 170
/// frequencies `2^0 .. 2^15` then a ramp of `2^(16 i / 154)`, as
/// `[sin(2 pi x f)..., cos(2 pi x f)...]`, the three axes side by side and
/// zero-padded to `channels`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "frequency indices below 200 are exact in f32"
)]
pub(crate) fn point_embed_v1(points: &[f32], channels: usize) -> Vec<f32> {
    const MAX_RES: usize = 16;
    let freq_dim = channels / 3 / 2;
    let extra = freq_dim.saturating_sub(MAX_RES);
    let freqs: Vec<f32> = (0..MAX_RES)
        .map(|i| i as f32)
        .chain((0..extra).map(|i| i as f32 / extra as f32 * MAX_RES as f32))
        .take(freq_dim)
        .map(|e| exact_pow2(f64::from(e)))
        .collect();
    embed_points(points, channels, &freqs, 2.0 * std::f32::consts::PI)
}

/// The decoders' sinusoidal embedding of 3D points: per axis, 170
/// frequencies `2^linspace(0, 10)`, as `[sin(pi x f)..., cos(pi x f)...]`,
/// the three axes side by side and zero-padded to `channels`.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "frequency indices below 200 are exact in f32"
)]
pub(crate) fn point_embed_v2(points: &[f32], channels: usize) -> Vec<f32> {
    const MAX_RES: f64 = 10.0;
    let freq_dim = channels / 3 / 2;
    let last = freq_dim.saturating_sub(1).max(1) as f64;
    // The exponents `linspace(0, 10, 170)`, each rounded once from f64 (see
    // `exact_pow2` for why the port does not imitate the reference's own rounding).
    let freqs: Vec<f32> = (0..freq_dim)
        .map(|i| exact_pow2(MAX_RES * i as f64 / last))
        .collect();
    embed_points(points, channels, &freqs, std::f32::consts::PI)
}

/// `2^e`, rounded once to f32.
///
/// The reference's `2 ** e` in f32 is one ulp low on 4 of the denoiser's 170
/// frequencies and 6 of the decoders' (the reference's vectorised pow is not
/// correctly rounded). At the denoiser's top frequency, 2^16 over angles up to
/// 4e5 rad, one ulp moves a sinusoid by 0.03, which the velocity carries as
/// ~3.7e-4 of its scale (measured against the reference with its own values).
/// The port rounds correctly instead of imitating one library's pow.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the power is computed in f64 and rounded once to the f32 the tables hold"
)]
fn exact_pow2(e: f64) -> f32 {
    2.0f64.powf(e) as f32
}

/// `[sin(scale x f)..., cos(scale x f)...]` per axis, axes side by side, padded.
fn embed_points(points: &[f32], channels: usize, freqs: &[f32], scale: f32) -> Vec<f32> {
    let n = points.len() / 3;
    let per_axis = 2 * freqs.len();
    let mut out = vec![0.0f32; n * channels];
    for p in 0..n {
        for axis in 0..3 {
            let x = points[p * 3 + axis];
            let base = p * channels + axis * per_axis;
            for (i, f) in freqs.iter().enumerate() {
                let angle = x * f * scale;
                out[base + i] = angle.sin();
                out[base + freqs.len() + i] = angle.cos();
            }
        }
    }
    out
}

/// A layer norm's affine, stored as the modulation's `(w - 1, b)`.
#[derive(Debug)]
pub(crate) struct Affine {
    scale: DeviceBuffer,
    shift: DeviceBuffer,
    dim: usize,
}

impl Affine {
    /// Reads `{name}.weight` and `{name}.bias`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a missing tensor or mismatched pair.
    pub(crate) fn load(store: &StStore, ops: &Ops, name: &str) -> Result<Self, ImageGenError> {
        let w = store.f32_vec(&format!("{name}.weight"))?;
        let b = store.f32_vec(&format!("{name}.bias"))?;
        if w.len() != b.len() {
            return Err(ImageGenError::Checkpoint(format!(
                "{name}: weight and bias disagree on width"
            )));
        }
        let scale: Vec<f32> = w.iter().map(|v| v - 1.0).collect();
        Ok(Self {
            scale: ops.upload_f32("norm affine", &scale)?,
            shift: ops.upload_f32("norm affine", &b)?,
            dim: w.len(),
        })
    }
}

/// `LayerNorm` over `rows` rows of `dim`, with the affine when given.
///
/// # Errors
///
/// A device failure, or an affine of another width.
pub(crate) fn layer_norm<'o>(
    ops: &'o Ops,
    step: &'static str,
    x: shiso_core::DeviceView<'_>,
    rows: usize,
    dim: usize,
    eps: f32,
    affine: Option<&Affine>,
) -> Result<Scratch<'o>, ImageGenError> {
    let out = ops.alloc_f32(step, rows * dim)?;
    ops.layer_norm(&LayerNormCall {
        step,
        out: out.view(),
        input: x,
        rows,
        dim,
        eps,
    })?;
    if let Some(affine) = affine {
        if affine.dim != dim {
            return Err(ImageGenError::Backend(format!(
                "{step}: a {}-wide affine on {dim}-wide rows",
                affine.dim
            )));
        }
        ops.modulate(&ModulateCall {
            step,
            out: out.view(),
            x: out.view(),
            scale: affine.scale.view(),
            shift: affine.shift.view(),
            count: rows * dim,
            period: dim,
        })?;
    }
    Ok(out)
}

/// `x += gate * y` in place, `gate` repeating every `period` values.
///
/// # Errors
///
/// A device failure.
pub(crate) fn add_gated_in_place(
    ops: &Ops,
    step: &'static str,
    x: shiso_core::DeviceView<'_>,
    gate: shiso_core::DeviceView<'_>,
    y: shiso_core::DeviceView<'_>,
    count: usize,
    period: usize,
) -> Result<(), ImageGenError> {
    ops.add_gated(&AddGatedCall {
        step,
        out: x,
        a: x,
        gate,
        b: y,
        count,
        period,
    })
}

/// How a head's dimensions pair up for a rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pairing {
    /// `(i, i + d/2)`: the half-split rotation (`rotate_half`).
    Halves,
    /// `(2i, 2i + 1)`: the interleaved, complex-number rotation.
    Adjacent,
}

/// The rotation's partner map: `turn(q)[j] = sign[j] * q[source[j]]`, within
/// one head of `head_dim`, such that `rope(q) = q * cos + turn(q) * sin`.
#[must_use]
pub(crate) fn turn_map(pairing: Pairing, head_dim: usize) -> Vec<(usize, f32)> {
    let half = head_dim / 2;
    (0..head_dim)
        .map(|j| match pairing {
            Pairing::Halves if j < half => (j + half, -1.0),
            Pairing::Halves => (j - half, 1.0),
            Pairing::Adjacent if j.is_multiple_of(2) => (j + 1, -1.0),
            Pairing::Adjacent => (j - 1, 1.0),
        })
        .collect()
}

/// The rows of `turn(W)` and `turn(b)` for a `(heads * head_dim, in_dim)`
/// projection: row `h * head_dim + j` is `sign * W[h * head_dim + source]`.
#[must_use]
pub(crate) fn turned(
    w: &[f32],
    bias: Option<&[f32]>,
    heads: usize,
    head_dim: usize,
    pairing: Pairing,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let in_dim = w.len() / (heads * head_dim);
    let map = turn_map(pairing, head_dim);
    let mut tw = vec![0.0f32; w.len()];
    let mut tb = bias.map(|b| vec![0.0f32; b.len()]);
    for h in 0..heads {
        for (j, (source, sign)) in map.iter().enumerate() {
            let (dst, src) = (h * head_dim + j, h * head_dim + source);
            for (d, s) in tw[dst * in_dim..(dst + 1) * in_dim]
                .iter_mut()
                .zip(&w[src * in_dim..(src + 1) * in_dim])
            {
                *d = sign * s;
            }
            if let (Some(tb), Some(b)) = (tb.as_mut(), bias) {
                tb[dst] = sign * b[src];
            }
        }
    }
    (tw, tb)
}

/// `q = q * (1 + cos_m1) + turned * sin`, in place, with full-size tables:
/// the rotary turn of the module doc.
///
/// # Errors
///
/// A device failure.
pub(crate) fn rotate_in_place(
    ops: &Ops,
    step: &'static str,
    q: shiso_core::DeviceView<'_>,
    turned: shiso_core::DeviceView<'_>,
    cos_m1: shiso_core::DeviceView<'_>,
    sin: shiso_core::DeviceView<'_>,
    count: usize,
) -> Result<(), ImageGenError> {
    add_gated_in_place(ops, step, q, cos_m1, q, count, count)?;
    add_gated_in_place(ops, step, q, sin, turned, count, count)
}

/// A per-head RMS norm's gamma, `(heads, head_dim)`, stored as `gamma - 1`.
#[derive(Debug)]
pub(crate) struct HeadNorm {
    gamma_m1: DeviceBuffer,
    ones: DeviceBuffer,
    heads: usize,
    head_dim: usize,
}

/// The norm's floor on `mean(x^2)`: the reference divides by
/// `max(|x|, 1e-12)`, so only an all-zero head can tell the two apart.
const HEAD_NORM_EPS: f32 = 1e-24;

impl HeadNorm {
    /// Reads `{name}.gamma`, `(heads, head_dim)`.
    ///
    /// # Errors
    ///
    /// [`ImageGenError::Checkpoint`] for a missing or misshapen tensor.
    pub(crate) fn load(
        store: &StStore,
        ops: &Ops,
        name: &str,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self, ImageGenError> {
        let gamma = store.f32_vec(&format!("{name}.gamma"))?;
        if gamma.len() != heads * head_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "{name}.gamma is {} values, expected {heads}x{head_dim}",
                gamma.len()
            )));
        }
        let gamma_m1: Vec<f32> = gamma.iter().map(|g| g - 1.0).collect();
        Ok(Self {
            gamma_m1: ops.upload_f32("head norm", &gamma_m1)?,
            ones: ops.upload_f32("head norm", &vec![1.0; head_dim])?,
            heads,
            head_dim,
        })
    }

    /// Normalises every head of `tokens` packed `(tokens, heads * head_dim)`
    /// rows in place.
    ///
    /// # Errors
    ///
    /// A device failure.
    pub(crate) fn apply(
        &self,
        ops: &Ops,
        step: &'static str,
        x: shiso_core::DeviceView<'_>,
        tokens: usize,
    ) -> Result<(), ImageGenError> {
        ops.rmsnorm(&NormCall {
            step,
            out: x,
            input: x,
            weight: self.ones.view(),
            rows: tokens * self.heads,
            dim: self.head_dim,
            eps: HEAD_NORM_EPS,
        })?;
        let width = self.heads * self.head_dim;
        add_gated_in_place(ops, step, x, self.gamma_m1.view(), x, tokens * width, width)
    }
}

/// Three packed projections of the same rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Qkv<'a> {
    /// Queries.
    pub q: shiso_core::DeviceView<'a>,
    /// Keys.
    pub k: shiso_core::DeviceView<'a>,
    /// Values.
    pub v: shiso_core::DeviceView<'a>,
}

/// The shape of one attention.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnShape {
    /// Query rows.
    pub queries: usize,
    /// Key and value rows.
    pub keys: usize,
    /// Heads.
    pub heads: usize,
    /// Width of one head.
    pub head_dim: usize,
}

/// Attention, in f32 throughout: packed `q (queries, heads * head_dim)`
/// against packed
/// `k`, `v (keys, heads * head_dim)`, scale `1 / sqrt(head_dim)`.
///
/// Per head, the head's columns are gathered by a transpose of the whole
/// buffer - `(rows, width)` to `(width, rows)` makes every head's block of
/// `head_dim` rows contiguous - and a second transpose of that block, then
/// `softmax(q k^T) v` runs as two GEMMs a block of queries at a time. The
/// head's answer is written back transposed and the whole result turned once
/// more at the end.
///
/// # Errors
///
/// A device failure.
#[expect(
    clippy::cast_precision_loss,
    reason = "a head width (64) is exact in f32"
)]
pub(crate) fn attention<'o>(
    ops: &'o Ops,
    step: &'static str,
    qkv: Qkv<'_>,
    shape: AttnShape,
) -> Result<Scratch<'o>, ImageGenError> {
    let Qkv { q, k, v } = qkv;
    let AttnShape {
        queries,
        keys,
        heads,
        head_dim,
    } = shape;
    let width = heads * head_dim;
    let q_t = ops.alloc_f32(step, queries * width)?;
    let k_t = ops.alloc_f32(step, keys * width)?;
    let v_t = ops.alloc_f32(step, keys * width)?;
    ops.transpose(step, q_t.view(), q, queries, width)?;
    ops.transpose(step, k_t.view(), k, keys, width)?;
    ops.transpose(step, v_t.view(), v, keys, width)?;
    let out_t = ops.alloc_f32(step, queries * width)?;
    let q_h = ops.alloc_f32(step, queries * head_dim)?;
    let k_h = ops.alloc_f32(step, keys * head_dim)?;
    let o_h = ops.alloc_f32(step, queries * head_dim)?;
    let block = (SCORE_BUDGET / keys).clamp(1, queries);
    let scores = ops.alloc_f32(step, block * keys)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    for h in 0..heads {
        let q_rows = q_t
            .view()
            .cut(bytes(h * head_dim * queries), bytes(head_dim * queries))?;
        let k_rows = k_t
            .view()
            .cut(bytes(h * head_dim * keys), bytes(head_dim * keys))?;
        let v_rows = v_t
            .view()
            .cut(bytes(h * head_dim * keys), bytes(head_dim * keys))?;
        ops.transpose(step, q_h.view(), q_rows, head_dim, queries)?;
        ops.transpose(step, k_h.view(), k_rows, head_dim, keys)?;
        for r0 in (0..queries).step_by(block) {
            let rows = block.min(queries - r0);
            let slab = scores.view().cut(0, bytes(rows * keys))?;
            ops.gemm(&GemmCall {
                step,
                out: slab,
                input: q_h
                    .view()
                    .cut(bytes(r0 * head_dim), bytes(rows * head_dim))?,
                weight: LinWeight::f32(k_h.view()),
                m: rows,
                n: keys,
                k: head_dim,
            })?;
            ops.softmax_rows(step, slab, rows, keys, scale)?;
            ops.gemm(&GemmCall {
                step,
                out: o_h
                    .view()
                    .cut(bytes(r0 * head_dim), bytes(rows * head_dim))?,
                input: slab,
                weight: LinWeight::f32(v_rows),
                m: rows,
                n: head_dim,
                k: keys,
            })?;
        }
        let o_rows = out_t
            .view()
            .cut(bytes(h * head_dim * queries), bytes(head_dim * queries))?;
        ops.transpose(step, o_rows, o_h.view(), queries, head_dim)?;
    }
    let out = ops.alloc_f32(step, queries * width)?;
    ops.transpose(step, out.view(), out_t.view(), width, queries)?;
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::cast_precision_loss,
    reason = "the fixtures are small integers"
)]
mod tests {
    use super::*;

    /// Applies a turn map to one head on the host.
    fn turn(q: &[f32], map: &[(usize, f32)]) -> Vec<f32> {
        map.iter().map(|(s, sign)| sign * q[*s]).collect()
    }

    #[test]
    fn the_half_split_turn_is_rotate_half() {
        // rotate_half([a, b, c, d]) = [-c, -d, a, b].
        let q = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            turn(&q, &turn_map(Pairing::Halves, 4)),
            vec![-3.0, -4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn the_adjacent_turn_multiplies_each_pair_by_i() {
        // (a + ib) * i = -b + ia, pair by pair.
        let q = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            turn(&q, &turn_map(Pairing::Adjacent, 4)),
            vec![-2.0, 1.0, -4.0, 3.0]
        );
    }

    #[test]
    fn a_turned_projection_is_the_turn_of_the_projection() {
        // Two heads of four, two inputs: projecting with the turned weight
        // equals turning the projected head, which is what lets the rotation
        // ride a second GEMM.
        let (heads, head_dim, in_dim) = (2, 4, 2);
        let w: Vec<f32> = (0..heads * head_dim * in_dim).map(|v| v as f32).collect();
        let b: Vec<f32> = (0..heads * head_dim).map(|v| 0.5 * v as f32).collect();
        let x = [0.25f32, -1.5];
        let project = |w: &[f32], b: &[f32]| -> Vec<f32> {
            (0..heads * head_dim)
                .map(|r| b[r] + w[r * in_dim] * x[0] + w[r * in_dim + 1] * x[1])
                .collect()
        };
        for pairing in [Pairing::Halves, Pairing::Adjacent] {
            let (tw, tb) = turned(&w, Some(&b), heads, head_dim, pairing);
            let q = project(&w, &b);
            let tq = project(&tw, tb.as_deref().expect("a bias"));
            let map = turn_map(pairing, head_dim);
            for h in 0..heads {
                let want = turn(&q[h * head_dim..(h + 1) * head_dim], &map);
                assert_eq!(&tq[h * head_dim..(h + 1) * head_dim], want.as_slice());
            }
        }
    }
}
