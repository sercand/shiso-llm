//! Clef: the joint schema head of a decision checkpoint.
//!
//! A Clef GGUF is a language-model backbone under the architecture name
//! `clef`, with a small transformer beside it. [`ClefBackbone`] says which
//! backbone a file carries: the qwen35 hybrid, or a dense llama stack whose
//! logits are tied to its embedding. The head reads the backbone's
//! final normalized row for every prompt token, pools the rows of each
//! question's wording and of each of its options, lets the options read the
//! prompt and the questions read each other, and scores every option of every
//! question in one pass. Nothing is generated.
//!
//! The rows that grow with the prompt - the memory projection and the key and
//! value projections every attention reads - run on the device. Everything that
//! grows only with the schema (a few dozen rows at most) runs on the host,
//! where the pooling, the per-question softmax and the final features are a
//! handful of loops rather than a handful of kernels.
//!
//! The arithmetic follows `JointSchemaHead` in the release's
//! `joint_schema_model.py`, as the GGUF converter lays it out: the routing
//! blocks are `dec.blk.0..routing`, the joint blocks follow them, and the
//! three learned scalars are stored as the values inference uses
//! (`decision.scales` = prior scale, joint scale, residual gate).

#![expect(
    clippy::cast_precision_loss,
    reason = "token counts and head widths are a few thousand at most"
)]

mod host;
mod prompt;
#[cfg(test)]
mod prompt_tests;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use shiso_audio::AudioError;
use shiso_core::{Backend, CoreError, CoreResult, DType, DecisionQuestion, DeviceBuffer, TokenId};
use shiso_modelfile::GgufFile;

use crate::audio_ops::load::{LinearSource, download_f32, linear, upload_f32};
use crate::audio_ops::store::{TensorStore, dequant};
use crate::audio_ops::{
    AttnWindow, AxpyDesc, Linear, NormDesc, Ops, UnaryDesc, UnaryOp, WindowAttnDesc,
};

use host::{HostNorm, l2_normalized, mean_rows};
pub use prompt::ClefLayout;

/// The architecture name a Clef GGUF declares.
pub const ARCH: &str = "clef";

/// The epsilon `torch.nn.functional.normalize` divides by at least.
const L2_EPS: f32 = 1e-12;

/// A layer norm on the device: weight and bias.
#[derive(Debug)]
struct Norm {
    weight: DeviceBuffer,
    bias: DeviceBuffer,
}

/// One multi-head attention's four projections.
#[derive(Debug)]
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

/// The feed-forward every head block ends in.
#[derive(Debug)]
struct Ffn {
    norm: Norm,
    up: Linear,
    down: Linear,
}

/// A routing block: the options read the normalized prompt memory.
#[derive(Debug)]
struct Routing {
    norm: Norm,
    norm_kv: Norm,
    attn: Attn,
    ffn: Ffn,
}

/// A joint block: the questions read each other, then the raw prompt memory.
#[derive(Debug)]
struct Joint {
    self_norm: Norm,
    self_attn: Attn,
    cross_norm: Norm,
    cross_attn: Attn,
    ffn: Ffn,
}

/// The output embedding's tensor, when the file has one of its own.
const OUTPUT: &str = "output.weight";

/// The input embedding, which is also the output embedding of a backbone whose
/// logits are tied to it.
const TOKEN_EMBD: &str = "token_embd.weight";

/// Rows of the output embedding, read on the host by token id.
#[derive(Debug)]
struct Lexicon {
    file: Arc<GgufFile>,
    /// [`OUTPUT`], or [`TOKEN_EMBD`] for a tied backbone.
    tensor: &'static str,
    dtype: DType,
    width: usize,
    vocab: usize,
}

impl Lexicon {
    /// The output embedding row of `token`, as f32.
    fn row(&self, token: TokenId) -> CoreResult<Vec<f32>> {
        let index = token.get() as usize;
        if index >= self.vocab {
            return Err(CoreError::InvalidValue {
                key: "clef.token",
                value: index.to_string(),
            });
        }
        let data = self
            .file
            .get(self.tensor)
            .ok_or_else(|| invalid(format!("{} is missing", self.tensor)))?
            .data;
        host::dequant_row(data, self.dtype, self.width, index)
    }
}

/// The joint schema head, resident.
#[derive(Debug)]
pub struct ClefHead {
    ops: Ops,
    /// The backbone's hidden width.
    hidden: usize,
    /// The head's own width.
    width: usize,
    /// Attention heads in every head block.
    heads: usize,
    /// Layer-norm epsilon of every head norm.
    eps: f32,
    hidden_norm: HostNorm,
    proj_memory: Linear,
    proj_question: Linear,
    proj_option_question: Linear,
    proj_global: Linear,
    proj_option_context: Linear,
    proj_option_lexical: Linear,
    /// `[3][width]`: one row per question kind.
    type_embd: Vec<f32>,
    routing: Vec<Routing>,
    joint: Vec<Joint>,
    option_summary_norm: HostNorm,
    field_norm: HostNorm,
    option_norm: HostNorm,
    scorer: Linear,
    scorer_out: Vec<f32>,
    scorer_out_bias: f32,
    /// Prior scale, joint scale, residual gate.
    scales: [f32; 3],
    lexicon: Lexicon,
}

fn invalid(what: impl Into<String>) -> CoreError {
    CoreError::Backend {
        backend: "clef",
        op: "load",
        source: Box::new(std::io::Error::other(what.into())),
    }
}

fn audio(op: &'static str) -> impl Fn(AudioError) -> CoreError {
    move |error| CoreError::Backend {
        backend: "clef",
        op,
        source: Box::new(error),
    }
}

/// Whether the checkpoint at `path` is a Clef GGUF: a file whose declared
/// architecture is [`ARCH`].
#[must_use]
pub fn is_clef(path: &Path) -> bool {
    path.is_file()
        && GgufFile::open(path)
            .ok()
            .and_then(|file| {
                file.metadata()
                    .get_str("general.architecture")
                    .map(|arch| arch == ARCH)
            })
            .unwrap_or(false)
}

/// The backbone a Clef GGUF carries under the head.
///
/// The architecture name is `clef` for every backbone, so the name cannot say
/// which stack to build. The tensors can: the qwen35 hybrid has recurrent
/// (`ssm_*`) slots and per-head query norms on its attention slots, and the
/// dense llama stack has neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClefBackbone {
    /// The qwen35 hybrid: gated delta-rule slots and normalized attention.
    Qwen35,
    /// A dense llama stack: attention slots that rotate without a norm.
    Llama,
}

impl ClefBackbone {
    /// Which backbone the Clef GGUF `file` carries.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] when the file is not a Clef GGUF, and
    /// [`CoreError::Unsupported`] when its block tensors are neither stack's.
    pub fn of(file: &GgufFile) -> CoreResult<Self> {
        let arch = file.metadata().architecture().unwrap_or_default();
        if arch != ARCH {
            return Err(CoreError::UnknownArch {
                model_type: arch.to_owned(),
            });
        }
        let blocks = || {
            file.tensors()
                .iter()
                .map(|t| t.name.as_ref())
                .filter(|name: &&str| name.starts_with("blk."))
        };
        if blocks().any(|name| name.contains(".ssm_") || name.ends_with(".attn_q_norm.weight")) {
            return Ok(Self::Qwen35);
        }
        if file.find("blk.0.attn_q.weight").is_some() {
            return Ok(Self::Llama);
        }
        Err(CoreError::Unsupported {
            what: "a clef backbone with neither recurrent slots nor a plain attention block 0",
            backend: "clef",
        })
    }

    /// Which backbone the Clef GGUF at `path` carries, or `None` when the path
    /// is not a Clef GGUF at all.
    ///
    /// # Errors
    ///
    /// As [`Self::of`], for a file that declares [`ARCH`]; and
    /// [`CoreError::Backend`] when a file that does cannot be opened.
    // `Ok(None)`: the path is not a GGUF that declares `clef`, which is every
    // other checkpoint, and the caller's answer to it is "not mine".
    pub fn at(path: &Path) -> CoreResult<Option<Self>> {
        if !is_clef(path) {
            return Ok(None);
        }
        let file = GgufFile::open_shared(path).map_err(|e| invalid(e.to_string()))?;
        Self::of(&file).map(Some)
    }
}

impl ClefHead {
    /// Loads the head from the Clef GGUF at `path`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] naming what is missing or misshaped, and a device
    /// error while importing.
    #[expect(
        clippy::too_many_lines,
        reason = "every tensor of the head, named in the order the converter writes them"
    )]
    pub fn load(backend: &Arc<dyn Backend>, path: &Path) -> CoreResult<Self> {
        let ops = Ops::new(Arc::clone(backend));
        let store = TensorStore::open_gguf(path).map_err(audio("open"))?;
        let file = Arc::new(GgufFile::open(path).map_err(|e| invalid(e.to_string()))?);
        let meta = file.metadata();
        let arch = meta.get_str("general.architecture").unwrap_or_default();
        if arch != ARCH {
            return Err(invalid(format!(
                "{} declares {arch}, not {ARCH}",
                path.display()
            )));
        }
        let key = |suffix: &str| {
            meta.arch_u32(suffix)
                .map(|v| v as usize)
                .ok_or_else(|| invalid(format!("{ARCH}.{suffix} is missing")))
        };
        let hidden = key("embedding_length")?;
        let n_routing = key("decision.routing_block_count")?;
        let n_joint = key("decision.block_count")?;
        let heads = key("decision.head_count")?;
        let eps = meta
            .arch_f32("attention.layer_norm_epsilon")
            .unwrap_or(1e-5);

        let memory = store
            .require("decision.proj_memory.weight")
            .map_err(audio("load"))?;
        let width = usize::try_from(memory.shape[0]).unwrap_or(0);
        if width == 0 || heads == 0 || width % heads != 0 {
            return Err(invalid(format!(
                "a head {width} wide cannot be split into {heads} heads"
            )));
        }
        let ff = usize::try_from(
            store
                .require("dec.blk.0.ffn_up.weight")
                .map_err(audio("load"))?
                .shape[0],
        )
        .unwrap_or(0);

        let lin = |name: &str, bias: bool, out: usize, inp: usize| -> CoreResult<Linear> {
            let weight = format!("{name}.weight");
            let bias_name = format!("{name}.bias");
            linear(
                &ops,
                &store,
                &LinearSource {
                    weight: &weight,
                    bias: bias.then_some(bias_name.as_str()),
                    affine_group: None,
                },
                out,
                inp,
            )
            .map_err(audio("load"))
        };
        let host_values = |name: &str, n: usize| -> CoreResult<Vec<f32>> {
            let values =
                dequant(&store.require(name).map_err(audio("load"))?).map_err(audio("load"))?;
            if values.len() == n {
                Ok(values)
            } else {
                Err(invalid(format!(
                    "{name} holds {} values, not {n}",
                    values.len()
                )))
            }
        };
        let host_norm = |name: &str, n: usize| -> CoreResult<HostNorm> {
            Ok(HostNorm {
                weight: host_values(&format!("{name}.weight"), n)?,
                bias: host_values(&format!("{name}.bias"), n)?,
                eps,
            })
        };
        let norm = |name: &str| -> CoreResult<Norm> {
            let n = host_norm(name, width)?;
            Ok(Norm {
                weight: upload_f32(&ops, &n.weight).map_err(audio("load"))?,
                bias: upload_f32(&ops, &n.bias).map_err(audio("load"))?,
            })
        };
        let attn = |prefix: &str| -> CoreResult<Attn> {
            Ok(Attn {
                q: lin(&format!("{prefix}_q"), true, width, width)?,
                k: lin(&format!("{prefix}_k"), true, width, width)?,
                v: lin(&format!("{prefix}_v"), true, width, width)?,
                o: lin(&format!("{prefix}_o"), true, width, width)?,
            })
        };
        let ffn = |blk: &str| -> CoreResult<Ffn> {
            Ok(Ffn {
                norm: norm(&format!("{blk}.ffn_norm"))?,
                up: lin(&format!("{blk}.ffn_up"), true, ff, width)?,
                down: lin(&format!("{blk}.ffn_down"), true, width, ff)?,
            })
        };

        let routing = (0..n_routing)
            .map(|il| {
                let blk = format!("dec.blk.{il}");
                Ok(Routing {
                    norm: norm(&format!("{blk}.cross_attn_norm"))?,
                    norm_kv: norm(&format!("{blk}.cross_attn_norm_kv"))?,
                    attn: attn(&format!("{blk}.cross_attn"))?,
                    ffn: ffn(&blk)?,
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let joint = (n_routing..n_routing + n_joint)
            .map(|il| {
                let blk = format!("dec.blk.{il}");
                Ok(Joint {
                    self_norm: norm(&format!("{blk}.attn_norm"))?,
                    self_attn: attn(&format!("{blk}.attn"))?,
                    cross_norm: norm(&format!("{blk}.cross_attn_norm"))?,
                    cross_attn: attn(&format!("{blk}.cross_attn"))?,
                    ffn: ffn(&blk)?,
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;

        let scales = host_values("decision.scales", 3)?;
        // The option words are read out of the output embedding. A backbone
        // whose logits are tied to its input embedding ships no output tensor,
        // and its input embedding is that matrix.
        let tensor = if file.find(OUTPUT).is_some() {
            OUTPUT
        } else {
            TOKEN_EMBD
        };
        let out = file
            .find(tensor)
            .ok_or_else(|| invalid(format!("neither {OUTPUT} nor {TOKEN_EMBD} is present")))?;
        let dims = out.shape.dims();
        let lexicon = Lexicon {
            file: Arc::clone(&file),
            tensor,
            dtype: out.dtype,
            width: usize::try_from(dims.first().copied().unwrap_or(0)).unwrap_or(0),
            vocab: usize::try_from(dims.get(1).copied().unwrap_or(0)).unwrap_or(0),
        };
        if lexicon.width != hidden {
            return Err(invalid(format!(
                "{tensor} rows are {} wide, not the backbone's {hidden}",
                lexicon.width
            )));
        }

        Ok(Self {
            hidden,
            width,
            heads,
            eps,
            hidden_norm: host_norm("decision.hidden_norm", hidden)?,
            proj_memory: lin("decision.proj_memory", false, width, hidden)?,
            proj_question: lin("decision.proj_question", false, width, hidden)?,
            proj_option_question: lin("decision.proj_option_question", false, width, hidden)?,
            proj_global: lin("decision.proj_global", false, width, hidden)?,
            proj_option_context: lin("decision.proj_option_context", false, width, hidden)?,
            proj_option_lexical: lin("decision.proj_option_lexical", false, width, hidden)?,
            type_embd: host_values("token_types.weight", 3 * width)?,
            routing,
            joint,
            option_summary_norm: host_norm("decision.option_summary_norm", width)?,
            field_norm: host_norm("decision.field_norm", width)?,
            option_norm: host_norm("decision.option_norm", width)?,
            scorer: lin("decision.scorer", true, width, 4 * width)?,
            scorer_out: host_values("decision.scorer_out.weight", width)?,
            scorer_out_bias: host_values("decision.scorer_out.bias", 1)?[0],
            scales: [scales[0], scales[1], scales[2]],
            lexicon,
            ops,
        })
    }

    /// The backbone width the head reads.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    /// Scores every option of every question.
    ///
    /// `rows` is the backbone's final normalized row of every token of
    /// `prompt`, `hidden` floats each.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] when the rows do not cover the prompt, a
    /// span is empty or past its end, or a question has no options; a device
    /// error otherwise.
    #[expect(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "one pass of the head, kept in the order and the names the reference reads"
    )]
    pub fn score(
        &self,
        rows: &[f32],
        prompt: &[TokenId],
        questions: &[DecisionQuestion],
    ) -> CoreResult<Vec<Vec<f32>>> {
        let (d, w) = (self.hidden, self.width);
        let n = prompt.len();
        if n == 0 || rows.len() != n * d {
            return Err(CoreError::ShapeMismatch {
                field: "clef.rows",
                expected: (n * d) as u64,
                got: rows.len() as u64,
            });
        }
        let in_prompt = |span: &std::ops::Range<usize>| span.start < span.end && span.end <= n;
        if questions.is_empty()
            || questions.iter().any(|q| {
                q.kind > 2
                    || !in_prompt(&q.span)
                    || q.options.is_empty()
                    || !q.options.iter().all(in_prompt)
            })
        {
            return Err(CoreError::ShapeMismatch {
                field: "clef.spans",
                expected: n as u64,
                got: 0,
            });
        }

        // The prompt rows, normalized once; every pooled vector reads these.
        let hidden: Vec<f32> = rows
            .chunks_exact(d)
            .flat_map(|row| self.hidden_norm.apply(row))
            .collect();
        let global = &hidden[(n - 1) * d..];
        let question_vectors: Vec<Vec<f32>> = questions
            .iter()
            .map(|q| mean_rows(&hidden, d, q.span.clone()))
            .collect();
        let mut owner = Vec::new();
        let mut contexts = Vec::new();
        let mut lexical = Vec::new();
        let mut own_questions = Vec::new();
        for (qi, q) in questions.iter().enumerate() {
            for span in &q.options {
                owner.push(qi);
                contexts.extend(mean_rows(&hidden, d, span.clone()));
                let mut sum = vec![0.0f32; d];
                for token in &prompt[span.clone()] {
                    for (s, v) in sum.iter_mut().zip(self.lexicon.row(*token)?) {
                        *s += v;
                    }
                }
                let count = span.len() as f32;
                lexical.extend(sum.iter().map(|s| s / count));
                own_questions.extend_from_slice(&question_vectors[qi]);
            }
        }
        let n_opt = owner.len();
        let n_q = questions.len();

        let ops = &self.ops;
        let up = |values: &[f32]| upload_f32(ops, values).map_err(audio("score"));
        let down = |buffer: &DeviceBuffer, floats: usize| {
            download_f32(ops, buffer.view(), floats).map_err(audio("score"))
        };
        let project = |layer: &Linear, x: &DeviceBuffer, rows: usize| -> CoreResult<DeviceBuffer> {
            let y = ops.scratch(rows * layer.out_dim())?;
            layer.forward(ops, x.view(), y.view(), rows)?;
            Ok(y)
        };
        let add = |y: &DeviceBuffer, x: &DeviceBuffer, count: usize| {
            ops.axpy(&AxpyDesc {
                y: y.view(),
                x: x.view(),
                alpha: 1.0,
                count,
            })
        };
        let normed = |x: &DeviceBuffer, norm: &Norm, rows: usize| -> CoreResult<DeviceBuffer> {
            let y = ops.scratch(rows * w)?;
            ops.layer_norm(&NormDesc {
                out: y.view(),
                input: x.view(),
                weight: Some(norm.weight.view()),
                bias: Some(norm.bias.view()),
                rows,
                dim: w,
                eps: self.eps,
            })?;
            Ok(y)
        };
        let attend = |attn: &Attn,
                      q_in: &DeviceBuffer,
                      n_qr: usize,
                      kv_in: &DeviceBuffer,
                      n_kv: usize|
         -> CoreResult<DeviceBuffer> {
            let q = project(&attn.q, q_in, n_qr)?;
            let k = project(&attn.k, kv_in, n_kv)?;
            let v = project(&attn.v, kv_in, n_kv)?;
            let out = ops.scratch(n_qr * w)?;
            let head_dim = w / self.heads;
            ops.window_attention(&WindowAttnDesc {
                out: out.view(),
                q: q.view(),
                k: k.view(),
                v: v.view(),
                n_q: n_qr,
                n_k: n_kv,
                heads: self.heads,
                head_dim,
                window: AttnWindow {
                    q_abs0: 0,
                    k_abs0: 0,
                    k_end: n_kv,
                    chunk: n_qr.max(n_kv) + 1,
                    left_chunks: 0,
                },
                bias_uv: None,
                q_dim_scale: None,
                rel: None,
                scale: 1.0 / (head_dim as f32).sqrt(),
                ac_scale: 1.0,
                softcap: None,
            })?;
            project(&attn.o, &out, n_qr)
        };
        let feed_forward = |ffn: &Ffn, x: &DeviceBuffer, rows: usize| -> CoreResult<()> {
            let normed_x = normed(x, &ffn.norm, rows)?;
            let inner = project(&ffn.up, &normed_x, rows)?;
            ops.unary(&UnaryDesc {
                op: UnaryOp::GeluErf,
                out: inner.view(),
                input: inner.view(),
                count: rows * ffn.up.out_dim(),
            })?;
            let out = project(&ffn.down, &inner, rows)?;
            add(x, &out, rows * w)
        };

        // The prompt memory every attention reads.
        let memory = project(&self.proj_memory, &up(&hidden)?, n)?;

        // One query per option: its context, its words, its question.
        let options = project(&self.proj_option_context, &up(&contexts)?, n_opt)?;
        add(
            &options,
            &project(&self.proj_option_lexical, &up(&lexical)?, n_opt)?,
            n_opt * w,
        )?;
        add(
            &options,
            &project(&self.proj_option_question, &up(&own_questions)?, n_opt)?,
            n_opt * w,
        )?;

        // The options read the prompt.
        for block in &self.routing {
            let queries = normed(&options, &block.norm, n_opt)?;
            let keys = normed(&memory, &block.norm_kv, n)?;
            add(
                &options,
                &attend(&block.attn, &queries, n_opt, &keys, n)?,
                n_opt * w,
            )?;
            feed_forward(&block.ffn, &options, n_opt)?;
        }
        let routed = down(&options, n_opt * w)?;

        // One vector per question: its wording, a summary of its options, the
        // end of the prompt and its kind.
        let question_rows: Vec<f32> = question_vectors.concat();
        let base = down(
            &project(&self.proj_question, &up(&question_rows)?, n_q)?,
            n_q * w,
        )?;
        let global_w = down(&project(&self.proj_global, &up(global)?, 1)?, w)?;
        let scale = 1.0 / (w as f32).sqrt();
        let mut fields = Vec::with_capacity(n_q * w);
        for (qi, q) in questions.iter().enumerate() {
            let field = &base[qi * w..(qi + 1) * w];
            let own: Vec<&[f32]> = owner
                .iter()
                .enumerate()
                .filter(|(_, o)| **o == qi)
                .map(|(oi, _)| &routed[oi * w..(oi + 1) * w])
                .collect();
            let logits: Vec<f32> = own.iter().map(|o| host::dot(o, field) * scale).collect();
            let weights = host::softmax(&logits);
            let mut summary = vec![0.0f32; w];
            for (o, p) in own.iter().zip(&weights) {
                for (s, v) in summary.iter_mut().zip(o.iter()) {
                    *s += p * v;
                }
            }
            let summary = self.option_summary_norm.apply(&summary);
            let kind = q.kind as usize;
            let type_row = &self.type_embd[kind * w..(kind + 1) * w];
            fields.extend((0..w).map(|i| field[i] + summary[i] + global_w[i] + type_row[i]));
        }

        // The questions read each other and the prompt.
        let fields_dev = up(&fields)?;
        for block in &self.joint {
            let own = normed(&fields_dev, &block.self_norm, n_q)?;
            add(
                &fields_dev,
                &attend(&block.self_attn, &own, n_q, &own, n_q)?,
                n_q * w,
            )?;
            let queries = normed(&fields_dev, &block.cross_norm, n_q)?;
            add(
                &fields_dev,
                &attend(&block.cross_attn, &queries, n_q, &memory, n)?,
                n_q * w,
            )?;
            feed_forward(&block.ffn, &fields_dev, n_q)?;
        }
        let fields = down(&fields_dev, n_q * w)?;
        let fields: Vec<Vec<f32>> = fields
            .chunks_exact(w)
            .map(|row| self.field_norm.apply(row))
            .collect();

        // Prior: the option's words against its question and the prompt's end.
        let anchors: Vec<Vec<f32>> = question_vectors
            .iter()
            .map(|qv| {
                let sum: Vec<f32> = qv.iter().zip(global).map(|(a, b)| a + b).collect();
                l2_normalized(&sum, L2_EPS)
            })
            .collect();
        let mut prior = Vec::with_capacity(n_opt);
        let mut features = Vec::with_capacity(n_opt * 4 * w);
        let mut cosine = Vec::with_capacity(n_opt);
        for (oi, qi) in owner.iter().enumerate() {
            let words = l2_normalized(&lexical[oi * d..(oi + 1) * d], L2_EPS);
            prior.push(host::dot(&words, &anchors[*qi]));
            let option = self.option_norm.apply(&routed[oi * w..(oi + 1) * w]);
            let field = &fields[*qi];
            cosine.push(host::dot(
                &l2_normalized(field, L2_EPS),
                &l2_normalized(&option, L2_EPS),
            ));
            features.extend_from_slice(field);
            features.extend_from_slice(&option);
            features.extend(field.iter().zip(&option).map(|(a, b)| a * b));
            features.extend(field.iter().zip(&option).map(|(a, b)| (a - b).abs()));
        }

        // Joint: each option against its question's vector, plus a learned
        // residual over both.
        let residual = project(&self.scorer, &up(&features)?, n_opt)?;
        ops.unary(&UnaryDesc {
            op: UnaryOp::GeluErf,
            out: residual.view(),
            input: residual.view(),
            count: n_opt * w,
        })?;
        let residual = down(&residual, n_opt * w)?;
        let [prior_scale, joint_scale, gate] = self.scales;
        let mut logits: Vec<Vec<f32>> = questions
            .iter()
            .map(|q| Vec::with_capacity(q.options.len()))
            .collect();
        for (oi, qi) in owner.iter().enumerate() {
            let hidden_row = &residual[oi * w..(oi + 1) * w];
            let r = hidden_row
                .iter()
                .zip(&self.scorer_out)
                .map(|(x, k)| x * k)
                .sum::<f32>()
                + self.scorer_out_bias;
            let joint = joint_scale * cosine[oi] + r;
            logits[*qi].push(prior_scale * prior[oi] + gate * joint);
        }
        Ok(logits)
    }
}
