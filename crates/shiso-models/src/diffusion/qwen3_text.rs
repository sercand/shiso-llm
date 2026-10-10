//! A Qwen3-style decoder run as a prefill-only text feature extractor.
//!
//! Image families condition on the hidden states of a causal language model
//! read over the prompt: the encoder embeds the ids and runs the decoder
//! layers - grouped-query attention, per-head QK `RMSNorm`, rotate-half 1D
//! `RoPE`, `SwiGLU` MLP - and returns the residual stream at the layers a
//! [`TapSet`] names. Layers past the last tap are neither loaded nor
//! evaluated, the final norm and the LM head are never imported, and neither
//! is any vision tower that shares the file: their outputs reach no consumer.
//!
//! Every dimension is read off the checkpoint's own tensors - hidden width
//! from the embedding, head counts from the projection widths, head
//! dimension from the QK norms, depth by counting `<prefix>layers.<n>.`
//! names - and only `rope_theta` and the norm epsilon, which no tensor
//! carries, come from the config document. That document is required: a
//! missing or unparsable one is refused, not defaulted. Both a flat language
//! model config and a vision-language config that nests the language model
//! under `text_config` are read.
//!
//! # Positions
//!
//! A vision-language decoder (`rope_scaling.mrope_section`) rotates each head
//! with three position axes - time, height, width - interleaved across its
//! frequency pairs. For text-only input all three axes carry the token index,
//! so every pair turns by `index * freq` whichever axis it belongs to, and
//! the multi-axis rotation is exactly the 1D rotate-half one with the same
//! base. That is the only input this encoder takes: an id the config names
//! as a vision placeholder would need non-text positions, and is refused.
//!
//! # Weights
//!
//! Linears load through [`Lin::load`] in their stored format, so a BF16, FP8
//! or per-row INT8 (`ConvRot`) export runs through the matching GEMM. A BF16
//! embedding table is imported whole and gathered on the device; a per-row
//! INT8 one is gathered on the host instead, only the prompt's rows read
//! from the mapping, dequantized (`q * s`) and rotated back out of the
//! export's `ConvRot` frame ([`unrotate_row_f32`]) - the table is never
//! dequantized as a whole.

use std::path::{Path, PathBuf};

use shiso_core::DeviceBuffer;
use shiso_modelfile::st_header::StDType;
use shiso_vision::generator::ImageGenError;

use super::lin::Lin;
use super::ops::{EmbedCall, GemmCall, GqaCall, NormCall, Ops};
use super::quant::{RotGroup, read_int8_layout, unrotate_row_f32};
use super::weights::StStore;

/// What one text encoder load reads, and from where.
#[derive(Debug)]
pub(crate) struct TextEncoderDesc<'a> {
    /// Weight file(s) holding the decoder, and the config.json describing
    /// the text model.
    pub weights: &'a [PathBuf],
    /// The config document beside the weights.
    pub config: &'a Path,
    /// Tensor-name prefix of the decoder, e.g. `"model."` or
    /// `"model.language_model."`.
    pub prefix: &'a str,
    /// Which hidden states to return.
    pub taps: TapSet,
}

/// The residual-stream states a prefill returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TapSet {
    /// The stream after each listed 0-indexed layer, concatenated per token
    /// in list order.
    Layers(Vec<usize>),
    /// The last layer's output, with the model's final norm *not* applied.
    LastPreNorm,
}

/// Architecture numbers, all derived from tensor shapes except the two the
/// config carries.
#[derive(Debug, Clone)]
pub(crate) struct TextEncoderParams {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
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

/// Where the embedding rows come from.
#[derive(Debug)]
enum Embedding {
    /// A BF16 table imported whole and gathered by the device.
    Device(DeviceBuffer),
    /// Per-row INT8 codes gathered, dequantized and un-rotated on the host.
    HostInt8 {
        name: String,
        row_scale: Vec<f32>,
        rotate: Option<RotGroup>,
    },
}

/// The language-model facts the config document carries.
#[derive(Debug)]
struct LmConfig {
    theta: f32,
    rms_eps: f32,
    /// `num_hidden_layers`, when declared.
    layers: Option<usize>,
    /// `rope_scaling.mrope_section`, when the decoder is multi-axis.
    mrope_section: Option<Vec<usize>>,
    /// Ids that stand for vision input in a vision-language config.
    vision_ids: Vec<u32>,
}

fn checkpoint(message: String) -> ImageGenError {
    ImageGenError::Checkpoint(message)
}

/// Reads the text model's config: flat, or nested under `text_config`.
fn read_lm_config(path: &Path) -> Result<LmConfig, ImageGenError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        checkpoint(format!(
            "text_encoder config {}: {e}; every text_encoder checkpoint carries this document \
             and no default may stand in for it",
            path.display()
        ))
    })?;
    let doc: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| checkpoint(format!("parsing {}: {e}", path.display())))?;
    let nested = doc.get("text_config").filter(|v| v.is_object());
    let lm = nested.unwrap_or(&doc);
    let number = |key: &str| lm.get(key).and_then(serde_json::Value::as_f64);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rope bases and norm epsilons are f32 quantities in every export"
    )]
    let theta = number("rope_theta")
        .or_else(|| {
            lm.get("rope_parameters")
                .and_then(|p| p.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
        })
        .ok_or_else(|| {
            checkpoint(
                "text_encoder/config.json carries no rope_theta: the RoPE frequency \
                 table would be built on a guessed base"
                    .to_string(),
            )
        })? as f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rope bases and norm epsilons are f32 quantities in every export"
    )]
    let rms_eps = number("rms_norm_eps").ok_or_else(|| {
        checkpoint(
            "text_encoder/config.json carries no rms_norm_eps: every norm would \
             run on a guessed epsilon"
                .to_string(),
        )
    })? as f32;
    let layers = lm
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX));

    let scaling = lm
        .get("rope_scaling")
        .or_else(|| lm.get("rope_parameters"))
        .filter(|v| v.is_object());
    let mut mrope_section = None;
    if let Some(scaling) = scaling {
        // A scaled base (yarn, linear, dynamic) changes every frequency; only
        // the plain table is implemented, so anything else is refused.
        let kind = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        if kind != "default" && kind != "mrope" {
            return Err(checkpoint(format!(
                "{}: rope type '{kind}' is not implemented by the text encoder (only the \
                 unscaled table is)",
                path.display()
            )));
        }
        if let Some(section) = scaling.get("mrope_section") {
            let parts = section
                .as_array()
                .and_then(|a| {
                    a.iter()
                        .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
                        .collect::<Option<Vec<usize>>>()
                })
                .ok_or_else(|| {
                    checkpoint(format!(
                        "{}: mrope_section is not a list of counts",
                        path.display()
                    ))
                })?;
            mrope_section = Some(parts);
        }
    }

    let vision_ids = if nested.is_some() {
        [
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]
        .iter()
        .filter_map(|key| doc.get(*key).and_then(serde_json::Value::as_u64))
        .map(|id| u32::try_from(id).unwrap_or(u32::MAX))
        .collect()
    } else {
        Vec::new()
    };
    Ok(LmConfig {
        theta,
        rms_eps,
        layers,
        mrope_section,
        vision_ids,
    })
}

/// Rotate-half frequency table for the 1-axis `RoPE`:
/// `theta^(-2i / head_dim)` for `i in 0..head_dim/2`.
#[expect(
    clippy::cast_precision_loss,
    reason = "head_dim is a small dimension count, far below f32's exact-integer range"
)]
pub(crate) fn rope_half_freqs(head_dim: usize, theta: f32) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|i| {
            theta.powf(
                -2.0 * f32::from(u16::try_from(i).unwrap_or(u16::MAX)) / head_dim.max(1) as f32,
            )
        })
        .collect()
}

fn axis(name: &str, shape: &[u64], index: usize) -> Result<usize, ImageGenError> {
    let value = shape
        .get(index)
        .ok_or_else(|| checkpoint(format!("'{name}' has no axis {index} (shape {shape:?})")))?;
    usize::try_from(*value)
        .map_err(|_| checkpoint(format!("'{name}' dimension {value} does not fit usize")))
}

/// The loaded text encoder: embedding, the layers through the last tap, and
/// the `RoPE` table.
#[derive(Debug)]
pub(crate) struct TextEncoder {
    params: TextEncoderParams,
    /// Kept for the host embedding gather; every device import already
    /// holds its shard mapping alive on its own.
    store: StStore,
    embed: Embedding,
    layers: Vec<TextLayer>,
    taps: Vec<usize>,
    vision_ids: Vec<u32>,
    freqs: DeviceBuffer,
}

impl TextEncoder {
    /// Opens the weights and config `desc` names and uploads every layer
    /// through the last tap in its stored format (norms widened to f32).
    #[expect(
        clippy::too_many_lines,
        reason = "one component load: config read, shape-derived geometry with its refusals, then the per-layer weight upload, straight-line by construction"
    )]
    pub(crate) fn load(desc: &TextEncoderDesc<'_>, ops: &Ops) -> Result<Self, ImageGenError> {
        let store = StStore::open_files(desc.weights)?;
        let config = read_lm_config(desc.config)?;
        let prefix = desc.prefix;

        let embed_name = format!("{prefix}embed_tokens.weight");
        let embed_entry = store.require(&embed_name)?;
        let vocab = axis(&embed_name, &embed_entry.shape, 0)?;
        let hidden = axis(&embed_name, &embed_entry.shape, 1)?;
        let layers_prefix = format!("{prefix}layers.");
        let layers = store
            .count_blocks(&layers_prefix)
            .ok_or_else(|| checkpoint(format!("text encoder has no {layers_prefix}<n> tensors")))?;
        let taps = match &desc.taps {
            TapSet::Layers(taps) => {
                let last = taps.iter().copied().max().ok_or_else(|| {
                    checkpoint("the text encoder was asked for no layer taps".to_string())
                })?;
                if layers <= last {
                    return Err(checkpoint(format!(
                        "text encoder has {layers} layers but conditioning taps layers {}",
                        taps.iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                taps.clone()
            }
            TapSet::LastPreNorm => {
                // "The last layer" is the config's depth: a file missing its
                // final layers would otherwise hand back an earlier layer's
                // stream as the features, silently.
                let declared = config.layers.ok_or_else(|| {
                    checkpoint(format!(
                        "{} declares no num_hidden_layers; the last layer is not guessed \
                         from the tensor names",
                        desc.config.display()
                    ))
                })?;
                if declared != layers {
                    return Err(checkpoint(format!(
                        "{} declares {declared} layers but the weights carry {layers}",
                        desc.config.display()
                    )));
                }
                vec![layers - 1]
            }
        };

        let head_dim = {
            let name = format!("{prefix}layers.0.self_attn.q_norm.weight");
            axis(&name, &store.require(&name)?.shape, 0)?
        };
        if head_dim == 0 {
            return Err(checkpoint(format!(
                "{prefix}layers.0.self_attn.q_norm.weight declares a zero-wide head"
            )));
        }
        if let Some(section) = &config.mrope_section {
            // The sections split the head's frequency pairs between the
            // three axes; a split that does not cover them is a different
            // rotation than the one the text-only identity is argued for.
            if section.iter().sum::<usize>() != head_dim / 2 {
                return Err(checkpoint(format!(
                    "{}: mrope_section {section:?} does not split the {} frequency pairs of a \
                     {head_dim}-wide head",
                    desc.config.display(),
                    head_dim / 2
                )));
            }
        }
        // Head counts come from the projection widths; a width that is not
        // a whole number of heads, or a missing shape axis, is a malformed
        // checkpoint and is refused rather than divided or indexed into a
        // panic.
        let head_count = |name: &str| -> Result<usize, ImageGenError> {
            let rows = axis(name, &store.require(name)?.shape, 0)?;
            if rows % head_dim != 0 {
                return Err(checkpoint(format!(
                    "'{name}' is {rows} rows wide, not a whole number of {head_dim}-wide heads"
                )));
            }
            Ok(rows / head_dim)
        };
        let q_heads = head_count(&format!("{prefix}layers.0.self_attn.q_proj.weight"))?;
        let kv_heads = head_count(&format!("{prefix}layers.0.self_attn.k_proj.weight"))?;
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(checkpoint(format!(
                "text encoder has {q_heads} query heads and {kv_heads} key/value heads; \
                 the query count must be a positive multiple of the key/value count"
            )));
        }
        let intermediate = {
            let name = format!("{prefix}layers.0.mlp.gate_proj.weight");
            axis(&name, &store.require(&name)?.shape, 0)?
        };
        let params = TextEncoderParams {
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            intermediate,
            vocab,
            theta: config.theta,
            rms_eps: config.rms_eps,
        };

        let embed = match embed_entry.dtype {
            StDType::Bf16 => Embedding::Device(store.import(ops.backend(), &embed_name)?),
            StDType::I8 => {
                let name = format!("{prefix}embed_tokens");
                let layout = read_int8_layout(&store, &name)?;
                Embedding::HostInt8 {
                    name: embed_name.clone(),
                    row_scale: layout.row_scale,
                    rotate: layout.rotate,
                }
            }
            other => {
                return Err(checkpoint(format!(
                    "'{embed_name}' has dtype {other:?}; the text encoder gathers BF16 rows on \
                     the device or per-row INT8 rows on the host"
                )));
            }
        };
        let freqs = ops.upload_f32(
            "text rope freqs",
            &rope_half_freqs(params.head_dim, params.theta),
        )?;

        // Load only through the last tap layer: `prefill` runs `0..=last` and
        // the layers past it feed no tap, so importing their weights would
        // hold device memory for rows nothing reads.
        let run_layers = taps.iter().copied().max().unwrap_or(0) + 1;
        let mut loaded = Vec::with_capacity(run_layers);
        for l in 0..run_layers {
            let base = format!("{prefix}layers.{l}.");
            let norm = |suffix: &str| -> Result<DeviceBuffer, ImageGenError> {
                ops.upload_f32("text norm", &store.f32_vec(&format!("{base}{suffix}"))?)
            };
            let lin = |suffix: &str| Lin::load(&store, ops, &format!("{base}{suffix}"));
            loaded.push(TextLayer {
                input_norm: norm("input_layernorm.weight")?,
                q: lin("self_attn.q_proj")?,
                k: lin("self_attn.k_proj")?,
                v: lin("self_attn.v_proj")?,
                q_norm: norm("self_attn.q_norm.weight")?,
                k_norm: norm("self_attn.k_norm.weight")?,
                o: lin("self_attn.o_proj")?,
                post_norm: norm("post_attention_layernorm.weight")?,
                gate: lin("mlp.gate_proj")?,
                up: lin("mlp.up_proj")?,
                down: lin("mlp.down_proj")?,
            });
        }

        Ok(Self {
            params,
            store,
            embed,
            layers: loaded,
            taps,
            vision_ids: config.vision_ids,
            freqs,
        })
    }

    /// The width of one returned row: the hidden size times the tap count.
    pub(crate) fn context_width(&self) -> usize {
        self.params.hidden * self.taps.len()
    }

    /// The embedding rows of `ids` as f32, gathered on the host from per-row
    /// INT8 codes: `(q * s) . H` per row, the inverse of the export's
    /// rotation (`H` is its own inverse).
    fn host_rows(
        &self,
        name: &str,
        row_scale: &[f32],
        rotate: Option<RotGroup>,
        ids: &[u32],
    ) -> Result<Vec<f32>, ImageGenError> {
        let hidden = self.params.hidden;
        // The table is borrowed from the mapping; only the rows indexed
        // below are ever paged in.
        let codes = self.store.bytes(name)?;
        let mut rows = vec![0f32; ids.len() * hidden];
        for (row, &id) in rows.chunks_exact_mut(hidden).zip(ids) {
            let id = id as usize;
            let src = codes
                .get(id * hidden..(id + 1) * hidden)
                .ok_or_else(|| checkpoint(format!("'{name}' has no row {id}")))?;
            let scale = row_scale[id];
            for (out, &code) in row.iter_mut().zip(src) {
                *out = f32::from(code.cast_signed()) * scale;
            }
            if let Some(group) = rotate {
                unrotate_row_f32(row, group.width())?;
            }
        }
        Ok(rows)
    }

    /// Runs the prefill over `ids` and returns the tapped hidden states as
    /// host f32, row-major (tokens, [`Self::context_width`]).
    ///
    /// The concat is a host round-trip on purpose: the taps are strided rows
    /// of a (tokens, taps * hidden) target, and one round-trip per generation
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
        if t == 0 {
            return Err(ImageGenError::EmptyPrompt);
        }
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= p.vocab) {
            return Err(ImageGenError::Backend(format!(
                "token id {bad} is outside the text encoder's {}-row vocabulary",
                p.vocab
            )));
        }
        if let Some(&vision) = ids.iter().find(|&&id| self.vision_ids.contains(&id)) {
            // Text-only positions are the token index on every axis; a vision
            // placeholder would need the grid positions this encoder does not
            // compute (see the module docs), so it is refused, not run 1D.
            return Err(ImageGenError::Backend(format!(
                "token id {vision} is a vision placeholder; the text encoder computes \
                 text-only positions"
            )));
        }
        let q_dim = p.q_heads * p.head_dim;
        let kv_dim = p.kv_heads * p.head_dim;

        let x = match &self.embed {
            Embedding::Device(table) => {
                let ids_buf = ops.upload_u32("text ids", ids)?;
                let x = ops.alloc_f32("text embed", t * p.hidden)?;
                ops.embed(&EmbedCall {
                    step: "text embed",
                    out: x.view(),
                    table: table.view(),
                    ids: ids_buf.view(),
                    tokens: t,
                    hidden: p.hidden,
                })?;
                x
            }
            Embedding::HostInt8 {
                name,
                row_scale,
                rotate,
            } => {
                let rows = self.host_rows(name, row_scale, *rotate, ids)?;
                ops.upload_scratch("text embed", &rows)?
            }
        };

        let xn = ops.alloc_f32("text normed", t * p.hidden)?;
        let q = ops.alloc_f32("text q", t * q_dim)?;
        let k = ops.alloc_f32("text k", t * kv_dim)?;
        let v = ops.alloc_f32("text v", t * kv_dim)?;
        let attn = ops.alloc_f32("text attn", t * q_dim)?;
        let tmp = ops.alloc_f32("text tmp", t * p.hidden)?;
        let gate = ops.alloc_f32("text gate", t * p.intermediate)?;
        let up = ops.alloc_f32("text up", t * p.intermediate)?;
        let act = ops.alloc_f32("text act", t * p.intermediate)?;
        let mut taps = Vec::with_capacity(self.taps.len());
        for _ in &self.taps {
            taps.push(ops.alloc_f32("text tap", t * p.hidden)?);
        }

        let gemm = |step: &'static str,
                    lin: &Lin,
                    out: &DeviceBuffer,
                    input: &DeviceBuffer,
                    n: usize,
                    k: usize| {
            ops.gemm(&GemmCall {
                step,
                out: out.view(),
                input: input.view(),
                weight: lin.weight(),
                m: t,
                n,
                k,
            })
        };

        for (l, layer) in self.layers.iter().enumerate() {
            ops.rmsnorm(&NormCall {
                step: "text ln1",
                out: xn.view(),
                input: x.view(),
                weight: layer.input_norm.view(),
                rows: t,
                dim: p.hidden,
                eps: p.rms_eps,
            })?;
            gemm("text qkv", &layer.q, &q, &xn, q_dim, p.hidden)?;
            gemm("text qkv", &layer.k, &k, &xn, kv_dim, p.hidden)?;
            gemm("text qkv", &layer.v, &v, &xn, kv_dim, p.hidden)?;
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
            gemm("text o proj", &layer.o, &tmp, &attn, p.hidden, q_dim)?;
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
            gemm(
                "text mlp",
                &layer.gate,
                &gate,
                &xn,
                p.intermediate,
                p.hidden,
            )?;
            gemm("text mlp", &layer.up, &up, &xn, p.intermediate, p.hidden)?;
            ops.swiglu2(
                "text swiglu",
                act.view(),
                gate.view(),
                up.view(),
                t * p.intermediate,
            )?;
            gemm(
                "text down",
                &layer.down,
                &tmp,
                &act,
                p.hidden,
                p.intermediate,
            )?;
            ops.add(
                "text residual",
                x.view(),
                x.view(),
                tmp.view(),
                t * p.hidden,
            )?;

            for (slot, _) in self.taps.iter().enumerate().filter(|&(_, &tap)| tap == l) {
                ops.copy("text tap", taps[slot].view(), x.view())?;
            }
        }

        let mut rows = Vec::with_capacity(taps.len());
        for tap in &taps {
            rows.push(ops.download("text tap readback", tap.view(), t * p.hidden)?);
        }
        if rows.len() == 1 {
            return Ok(rows.pop().unwrap_or_default());
        }
        let width = self.context_width();
        let mut out = vec![0f32; t * width];
        for (row, token) in out.chunks_mut(width).enumerate() {
            for (slot, tap) in rows.iter().enumerate() {
                token[slot * p.hidden..(slot + 1) * p.hidden]
                    .copy_from_slice(&tap[row * p.hidden..(row + 1) * p.hidden]);
            }
        }
        Ok(out)
    }
}
