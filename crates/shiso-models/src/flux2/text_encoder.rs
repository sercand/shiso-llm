//! The FLUX.2 text encoder: a Qwen3-style decoder run as a prefill-only
//! feature extractor.
//!
//! The prompt arrives as token ids; the encoder embeds them and runs the
//! transformer layers causally - grouped-query attention, per-head QK `RMSNorm`,
//! rotate-half 1D `RoPE`, `SwiGLU` MLP - and taps the residual stream after
//! layers 8, 17 and 26. Those are the reference's `hidden_states[9]`,
//! `[18]` and `[27]` counting `hidden_states[0]` as the embedding output,
//! i.e. the stream after 0-indexed layers 8, 17, 26. The three hidden states
//! are concatenated into the (`seq_len`, 3 * hidden) conditioning the
//! diffusion transformer reads. Layers past the last tap are neither loaded
//! nor evaluated: their output reaches no consumer.
//!
//! Every dimension is read off the checkpoint's own tensors - hidden width
//! from the layer norms, head counts from the projection widths, head
//! dimension from the QK norms, depth by counting `model.layers.<n>.` names -
//! and only `rope_theta` and the norm epsilon, which no tensor carries, come
//! from the config document beside the weights. That document is required:
//! a missing or unparsable one is refused, not defaulted.

use std::path::Path;

use serde::Deserialize;
use shiso_core::DeviceBuffer;
use shiso_vision::generator::ImageGenError;

use super::rope::rope_half_freqs;
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{GemmCall, GqaCall, NormCall, Ops};
use crate::diffusion::weights::StStore;

/// The residual-stream layers whose hidden states condition the transformer:
/// the reference taps `hidden_states[9]`, `[18]` and `[27]` with
/// `hidden_states[0]` being the embedding output, so the 0-indexed layers
/// after which those states exist are 8, 17 and 26.
const TAP_LAYERS: [usize; 3] = [8, 17, 26];

/// Architecture numbers, all derived from tensor shapes.
#[derive(Debug, Clone)]
pub(crate) struct TextEncoderParams {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub theta: f32,
    pub rms_eps: f32,
}

#[derive(Debug)]
struct TextLayer {
    input_norm: DeviceBuffer,
    q: Lin,
    k: Lin,
    v: Lin,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    o: Lin,
    post_norm: DeviceBuffer,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// The loaded text encoder: embedding table, layers, and `RoPE` table.
#[derive(Debug)]
pub(crate) struct TextEncoder {
    params: TextEncoderParams,
    embed: DeviceBuffer,
    layers: Vec<TextLayer>,
    freqs: DeviceBuffer,
}

#[derive(Debug, Deserialize)]
struct TextEncoderConfig {
    rope_theta: Option<f32>,
    rms_norm_eps: Option<f32>,
}

impl TextEncoder {
    /// Opens the component at `path` and uploads every weight in its stored
    /// format (norms and biases widened to f32).
    #[expect(
        clippy::too_many_lines,
        reason = "one component load: config read, shape-derived geometry with its refusals, then the per-layer weight upload, straight-line by construction"
    )]
    pub(crate) fn load(path: &Path, ops: &Ops) -> Result<Self, ImageGenError> {
        let store = StStore::open(path)?;
        // The config document is required: `rope_theta` and `rms_norm_eps`
        // are the two architecture numbers no tensor carries, and guessing
        // them would rotate every head and renormalize every layer on a
        // base nobody declared.
        let config: TextEncoderConfig =
            super::config::read_component_config(&path.join("config.json"), "text_encoder")?;

        let embed_entry = store.require("model.embed_tokens.weight")?;
        let hidden = usize::try_from(*embed_entry.shape.get(1).ok_or_else(|| {
            ImageGenError::Checkpoint("model.embed_tokens.weight is not a matrix".to_string())
        })?)
        .unwrap_or(usize::MAX);
        let layers = store.count_blocks("model.layers.").ok_or_else(|| {
            ImageGenError::Checkpoint("text encoder has no model.layers.<n> tensors".to_string())
        })?;
        if layers <= *TAP_LAYERS.last().unwrap_or(&0) {
            return Err(ImageGenError::Checkpoint(format!(
                "text encoder has {layers} layers but conditioning taps layers {}",
                TAP_LAYERS
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let head_dim = {
            let e = store.require("model.layers.0.self_attn.q_norm.weight")?;
            usize::try_from(*e.shape.first().ok_or_else(|| {
                ImageGenError::Checkpoint("q_norm.weight has empty shape".to_string())
            })?)
            .unwrap_or(usize::MAX)
        };
        if head_dim == 0 {
            return Err(ImageGenError::Checkpoint(
                "model.layers.0.self_attn.q_norm.weight declares a zero-wide head".to_string(),
            ));
        }
        // Head counts come from the projection widths; a width that is not
        // a whole number of heads, or a missing shape axis, is a malformed
        // checkpoint and is refused rather than divided or indexed into a
        // panic.
        let head_count = |name: &str| -> Result<usize, ImageGenError> {
            let rows =
                *store.require(name)?.shape.first().ok_or_else(|| {
                    ImageGenError::Checkpoint(format!("'{name}' has empty shape"))
                })?;
            if rows % u64::try_from(head_dim).unwrap_or(u64::MAX) != 0 {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{name}' is {rows} rows wide, not a whole number of {head_dim}-wide heads"
                )));
            }
            Ok(
                usize::try_from(rows / u64::try_from(head_dim).unwrap_or(u64::MAX))
                    .unwrap_or(usize::MAX),
            )
        };
        let q_heads = head_count("model.layers.0.self_attn.q_proj.weight")?;
        let kv_heads = head_count("model.layers.0.self_attn.k_proj.weight")?;
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(ImageGenError::Checkpoint(format!(
                "text encoder has {q_heads} query heads and {kv_heads} key/value heads; \
                 the query count must be a positive multiple of the key/value count"
            )));
        }
        let intermediate = usize::try_from(
            *store
                .require("model.layers.0.mlp.gate_proj.weight")?
                .shape
                .first()
                .ok_or_else(|| {
                    ImageGenError::Checkpoint(
                        "model.layers.0.mlp.gate_proj.weight has empty shape".to_string(),
                    )
                })?,
        )
        .unwrap_or(usize::MAX);
        let params = TextEncoderParams {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            intermediate,
            theta: config.rope_theta.ok_or_else(|| {
                ImageGenError::Checkpoint(
                    "text_encoder/config.json carries no rope_theta: the RoPE frequency \
                     table would be built on a guessed base"
                        .to_string(),
                )
            })?,
            rms_eps: config.rms_norm_eps.ok_or_else(|| {
                ImageGenError::Checkpoint(
                    "text_encoder/config.json carries no rms_norm_eps: every norm would \
                     run on a guessed epsilon"
                        .to_string(),
                )
            })?,
        };

        let embed = store.import(ops.backend(), "model.embed_tokens.weight")?;
        let freqs = ops.upload_f32(
            "text rope freqs",
            &rope_half_freqs(params.head_dim, params.theta),
        )?;

        // Load only through the last tap layer: `prefill` runs `0..=last` and
        // the layers past it feed no tap, so importing their weights would
        // hold device memory for rows nothing reads.
        let run_layers = *TAP_LAYERS.last().unwrap_or(&0) + 1;
        let mut loaded = Vec::with_capacity(run_layers);
        for l in 0..run_layers {
            let base = format!("model.layers.{l}.");
            loaded.push(TextLayer {
                input_norm: ops.upload_f32(
                    "text norm",
                    &store.f32_vec(&format!("{base}input_layernorm.weight"))?,
                )?,
                q: Lin::load(&store, ops, &format!("{base}self_attn.q_proj"))?,
                k: Lin::load(&store, ops, &format!("{base}self_attn.k_proj"))?,
                v: Lin::load(&store, ops, &format!("{base}self_attn.v_proj"))?,
                q_norm: ops.upload_f32(
                    "text qk norm",
                    &store.f32_vec(&format!("{base}self_attn.q_norm.weight"))?,
                )?,
                k_norm: ops.upload_f32(
                    "text qk norm",
                    &store.f32_vec(&format!("{base}self_attn.k_norm.weight"))?,
                )?,
                o: Lin::load(&store, ops, &format!("{base}self_attn.o_proj"))?,
                post_norm: ops.upload_f32(
                    "text norm",
                    &store.f32_vec(&format!("{base}post_attention_layernorm.weight"))?,
                )?,
                gate: Lin::load(&store, ops, &format!("{base}mlp.gate_proj"))?,
                up: Lin::load(&store, ops, &format!("{base}mlp.up_proj"))?,
                down: Lin::load(&store, ops, &format!("{base}mlp.down_proj"))?,
            });
        }

        Ok(Self {
            params,
            embed,
            layers: loaded,
            freqs,
        })
    }

    /// The conditioning width the taps produce: three hidden states wide.
    pub(crate) fn context_width(&self) -> usize {
        self.params.hidden * TAP_LAYERS.len()
    }

    /// Runs the prefill over `ids` and returns the concatenated tap hidden
    /// states as host f32, row-major (tokens, 3 * hidden).
    ///
    /// The concat is a host round-trip on purpose: the taps are strided rows
    /// of a (tokens, 3 * hidden) target, and one round-trip per generation
    /// (a few megabytes) is cheaper than a scatter kernel the fallback could
    /// not run anyway.
    #[expect(
        clippy::too_many_lines,
        reason = "one transformer prefill over all layers, straight-line block graph by construction"
    )]
    #[expect(
        clippy::many_single_char_names,
        reason = "attention math reads clearer with t/p/x/q/k/v than with long names"
    )]
    pub(crate) fn prefill(&self, ops: &Ops, ids: &[u32]) -> Result<Vec<f32>, ImageGenError> {
        let t = ids.len();
        let p = &self.params;
        let q_dim = p.q_heads * p.head_dim;
        let kv_dim = p.kv_heads * p.head_dim;

        let ids_buf = ops.upload_u32("text ids", ids)?;
        let x = ops.alloc_f32("text embed", t * p.hidden)?;
        ops.embed(&crate::diffusion::ops::EmbedCall {
            step: "text embed",
            out: x.view(),
            table: self.embed.view(),
            ids: ids_buf.view(),
            tokens: t,
            hidden: p.hidden,
        })?;

        let xn = ops.alloc_f32("text normed", t * p.hidden)?;
        let q = ops.alloc_f32("text q", t * q_dim)?;
        let k = ops.alloc_f32("text k", t * kv_dim)?;
        let v = ops.alloc_f32("text v", t * kv_dim)?;
        let attn = ops.alloc_f32("text attn", t * q_dim)?;
        let tmp = ops.alloc_f32("text tmp", t * p.hidden)?;
        let gate = ops.alloc_f32("text gate", t * p.intermediate)?;
        let up = ops.alloc_f32("text up", t * p.intermediate)?;
        let act = ops.alloc_f32("text act", t * p.intermediate)?;
        let mut taps = Vec::with_capacity(TAP_LAYERS.len());
        for _ in TAP_LAYERS {
            taps.push(ops.alloc_f32("text tap", t * p.hidden)?);
        }

        let last = *TAP_LAYERS.last().unwrap_or(&0);
        for l in 0..=last {
            let layer = &self.layers[l];
            ops.rmsnorm(&NormCall {
                step: "text ln1",
                out: xn.view(),
                input: x.view(),
                weight: layer.input_norm.view(),
                rows: t,
                dim: p.hidden,
                eps: p.rms_eps,
            })?;
            for (lin, out, n) in [
                (&layer.q, q.view(), q_dim),
                (&layer.k, k.view(), kv_dim),
                (&layer.v, v.view(), kv_dim),
            ] {
                ops.gemm(&GemmCall {
                    step: "text qkv",
                    out,
                    input: xn.view(),
                    weight: lin.w.view(),
                    weight_scale: lin.scale,
                    row_scale: lin.row_scale.as_ref().map(DeviceBuffer::view),
                    rotate: lin.rotate,
                    bias: lin.bias.as_ref().map(DeviceBuffer::view),
                    m: t,
                    n,
                    k: p.hidden,
                    kind: lin.kind,
                })?;
            }
            ops.rmsnorm(&NormCall {
                step: "text q norm",
                out: q.view(),
                input: q.view(),
                weight: layer.q_norm.view(),
                rows: t * p.q_heads,
                dim: p.head_dim,
                eps: p.rms_eps,
            })?;
            ops.rmsnorm(&NormCall {
                step: "text k norm",
                out: k.view(),
                input: k.view(),
                weight: layer.k_norm.view(),
                rows: t * p.kv_heads,
                dim: p.head_dim,
                eps: p.rms_eps,
            })?;
            ops.rope_half(
                "text rope",
                q.view(),
                self.freqs.view(),
                t,
                p.q_heads,
                p.head_dim,
            )?;
            ops.rope_half(
                "text rope",
                k.view(),
                self.freqs.view(),
                t,
                p.kv_heads,
                p.head_dim,
            )?;
            ops.gqa(&GqaCall {
                step: "text attention",
                out: attn.view(),
                q: q.view(),
                k: k.view(),
                v: v.view(),
                tokens: t,
                q_heads: p.q_heads,
                kv_heads: p.kv_heads,
                head_dim: p.head_dim,
            })?;
            ops.gemm(&GemmCall {
                step: "text o proj",
                out: tmp.view(),
                input: attn.view(),
                weight: layer.o.w.view(),
                weight_scale: layer.o.scale,
                row_scale: layer.o.row_scale.as_ref().map(DeviceBuffer::view),
                rotate: layer.o.rotate,
                bias: layer.o.bias.as_ref().map(DeviceBuffer::view),
                m: t,
                n: p.hidden,
                k: q_dim,
                kind: layer.o.kind,
            })?;
            ops.add(
                "text residual",
                x.view(),
                x.view(),
                tmp.view(),
                t * p.hidden,
            )?;

            ops.rmsnorm(&NormCall {
                step: "text ln2",
                out: xn.view(),
                input: x.view(),
                weight: layer.post_norm.view(),
                rows: t,
                dim: p.hidden,
                eps: p.rms_eps,
            })?;
            for (lin, out) in [(&layer.gate, gate.view()), (&layer.up, up.view())] {
                ops.gemm(&GemmCall {
                    step: "text mlp",
                    out,
                    input: xn.view(),
                    weight: lin.w.view(),
                    weight_scale: lin.scale,
                    row_scale: lin.row_scale.as_ref().map(DeviceBuffer::view),
                    rotate: lin.rotate,
                    bias: lin.bias.as_ref().map(DeviceBuffer::view),
                    m: t,
                    n: p.intermediate,
                    k: p.hidden,
                    kind: lin.kind,
                })?;
            }
            ops.swiglu2(
                "text swiglu",
                act.view(),
                gate.view(),
                up.view(),
                t * p.intermediate,
            )?;
            ops.gemm(&GemmCall {
                step: "text down",
                out: tmp.view(),
                input: act.view(),
                weight: layer.down.w.view(),
                weight_scale: layer.down.scale,
                row_scale: layer.down.row_scale.as_ref().map(DeviceBuffer::view),
                rotate: layer.down.rotate,
                bias: layer.down.bias.as_ref().map(DeviceBuffer::view),
                m: t,
                n: p.hidden,
                k: p.intermediate,
                kind: layer.down.kind,
            })?;
            ops.add(
                "text residual",
                x.view(),
                x.view(),
                tmp.view(),
                t * p.hidden,
            )?;

            if let Some(slot) = TAP_LAYERS.iter().position(|&tap| tap == l) {
                ops.copy("text tap", taps[slot].view(), x.view())?;
            }
        }

        let mut rows = Vec::with_capacity(t * self.context_width());
        for tap in &taps {
            rows.push(ops.download("text tap readback", tap.view(), t * p.hidden)?);
        }
        let mut out = vec![0f32; t * self.context_width()];
        for (row, token) in out.chunks_mut(self.context_width()).enumerate() {
            for (slot, tap) in rows.iter().enumerate() {
                token[slot * p.hidden..(slot + 1) * p.hidden]
                    .copy_from_slice(&tap[row * p.hidden..(row + 1) * p.hidden]);
            }
        }
        Ok(out)
    }
}
