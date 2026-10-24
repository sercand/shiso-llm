//! The model type: a dense llama checkpoint with its Clef head, loaded and
//! answering the shared contract.
//!
//! The stack is [`shiso_engine::HybridStack`] over the engine's dense kernels,
//! every slot an attention slot that rotates its heads without normalizing
//! them. Nothing here names a kernel.
//!
//! Weights are imported out of the container's own mapping. The one tensor
//! with two jobs is `token_embd`: the stack gathers the prompt's rows out of it
//! and, with no `output.weight` in the file, projects the logits through it.
//! The container's `post_attention_norm` is the norm in front of the
//! feed-forward - the reference names it after the attention it follows - so
//! it is bound as the feed-forward norm, not as a post-attention one.
//!
//! # Every tensor is accounted for
//!
//! A tensor the loader does not bind is a weight the forward would silently
//! skip: an attention bias, an untied output projection, a frequency table for
//! a scaled rotation. So the load ends with a walk over the file, and any
//! tensor that is neither one this loader bound nor one the head owns is
//! refused by name.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::model::{
    DecisionModel, DecisionQuestion, DecodeStep, Logits, Model, ModelInfo, PromptRun, StopTokens,
};
use shiso_core::{Backend, DType};
use shiso_engine::attn::Attn;
use shiso_engine::dense::DenseKernels;
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::{EngineOptions, HybridStack};
use shiso_modelfile::{GgufFile, TensorDesc};

use crate::clef::{ClefHead, ClefLayout};
use crate::llama::config::LlamaConfig;

/// Formats a projection or the embedding may be stored in: the two the
/// engine's dense matmuls and its row gather both read. A `bf16` export has no
/// kernel here and is refused by name rather than widened on the host.
const WEIGHT_TYPES: &[DType] = &[DType::Q8_0, DType::F16];

/// The only format a norm is stored in.
const VECTOR_TYPE: DType = DType::F32;

/// The name prefixes of the Clef head's tensors, which [`ClefHead::load`]
/// binds and refuses on its own.
const HEAD_PREFIXES: [&str; 3] = ["dec.", "decision.", "token_types."];

/// The roles of one block, in the order the container writes them.
const BLOCK_ROLES: [&str; 9] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// A loaded llama checkpoint and its decision head.
#[derive(Debug)]
pub struct LlamaModel {
    stack: HybridStack,
    info: ModelInfo,
    head: ClefHead,
    /// Bits a weight of the first feed-forward was stored at, which is what
    /// the checkpoint tier tags stored rows with.
    quant_bits: u8,
}

impl LlamaModel {
    /// Opens a Clef GGUF over a dense llama backbone and puts the backbone and
    /// the head on the device.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming the first tensor the container does
    /// not hold, [`CoreError::Unsupported`] for a tensor stored in a format this
    /// family does not run or a tensor this loader would leave unread,
    /// [`CoreError::ShapeMismatch`] for one whose extent disagrees with the
    /// geometry, and whatever the head, the device or the stack reports.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &LlamaConfig,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        let file = GgufFile::open_shared(path).map_err(|source| CoreError::Backend {
            backend: "llama",
            op: "open the checkpoint",
            source: Box::new(source),
        })?;
        let loader = Loader {
            file: &file,
            backend,
        };
        loader.refuse_unread(config)?;
        let weights = loader.weights(config)?;
        let embed = loader.desc("token_embd.weight")?.dtype;
        let quant_bits = match loader.desc("blk.0.ffn_gate.weight")?.dtype {
            DType::F16 => 16,
            _ => 8,
        };
        let shape = config.shape();
        let head = ClefHead::load(backend, path)?;
        if head.hidden() != shape.hidden as usize {
            return Err(CoreError::ShapeMismatch {
                field: "clef head width",
                expected: u64::from(shape.hidden),
                got: head.hidden() as u64,
            });
        }
        let stack = HybridStack::new(
            backend,
            shape.clone(),
            weights,
            embed,
            Arc::new(DenseKernels),
            None,
            opts,
        )?;
        // Every imported weight holds its own share of the mapping, so the
        // file handle can go: the bytes stay mapped as long as a buffer names
        // them.
        drop(file);
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: shape.vocab,
                hidden: shape.hidden,
                n_layers: shape.n_layers(),
                stop_tokens: StopTokens::new(config.stop_tokens())?,
            },
            stack,
            head,
            quant_bits,
        })
    }

    /// Bits the feed-forward weights were stored at.
    pub(crate) const fn quant_bits(&self) -> u8 {
        self.quant_bits
    }

    /// The stack this model forwards to.
    pub(crate) const fn stack(&self) -> &HybridStack {
        &self.stack
    }

    /// The stack this model forwards to, mutably.
    pub(crate) const fn stack_mut(&mut self) -> &mut HybridStack {
        &mut self.stack
    }
}

/// The container and the device, for the imports below.
struct Loader<'f> {
    file: &'f GgufFile,
    backend: &'f Arc<dyn Backend>,
}

impl Loader<'_> {
    /// Refuses a container holding a tensor that neither this loader nor the
    /// head reads.
    fn refuse_unread(&self, config: &LlamaConfig) -> CoreResult<()> {
        let mut bound: HashSet<String> = ["token_embd.weight", "output_norm.weight"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        for slot in 0..config.shape().n_layers() {
            bound.extend(BLOCK_ROLES.iter().map(|role| format!("blk.{slot}.{role}")));
        }
        let unread = self.file.tensors().iter().find(|desc| {
            let name = desc.name.as_ref();
            !bound.contains(name) && !HEAD_PREFIXES.iter().any(|p| name.starts_with(p))
        });
        match unread {
            None => Ok(()),
            Some(desc) if desc.name.as_ref() == "output.weight" => Err(CoreError::Unsupported {
                what: "an untied output projection in this family, whose logits are tied to \
                       the embedding",
                backend: "llama",
            }),
            Some(desc) => Err(CoreError::InvalidValue {
                key: "llama tensors",
                value: format!(
                    "{} is a tensor this family's forward does not read, so running the file \
                     would silently skip it",
                    desc.name
                ),
            }),
        }
    }

    /// Every resident weight, slot by slot.
    fn weights(&self, config: &LlamaConfig) -> CoreResult<Weights> {
        let shape = config.shape();
        let hidden = u64::from(shape.hidden);
        let mut layers = Vec::with_capacity(shape.layers.len());
        for (index, slot) in (0..shape.n_layers()).enumerate() {
            let at = |role: &str| format!("blk.{slot}.{role}");
            let ordinal = u32::try_from(index).map_err(|_| CoreError::ShapeMismatch {
                field: "layer slot index",
                expected: u64::from(u32::MAX),
                got: index as u64,
            })?;
            layers.push(LayerWeights {
                input_norm: self.vector(&at("attn_norm.weight"), &[hidden])?,
                post_attn_norm: None,
                // The reference's `post_attention_layernorm` is the norm in
                // front of the feed-forward, whatever its name says.
                ffn_norm: Some(self.vector(&at("post_attention_norm.weight"), &[hidden])?),
                post_ffn_norm: None,
                ffn: Some(Ffn {
                    gate: Some(self.weight(&at("ffn_gate.weight"))?),
                    up: self.weight(&at("ffn_up.weight"))?,
                    down: self.weight(&at("ffn_down.weight"))?,
                }),
                moe: None,
                block: Block::Attn(Attn {
                    q: self.weight(&at("attn_q.weight"))?,
                    k: Some(self.weight(&at("attn_k.weight"))?),
                    v: Some(self.weight(&at("attn_v.weight"))?),
                    output: self.weight(&at("attn_output.weight"))?,
                    // The slot normalizes neither head; its shape says so.
                    q_norm: None,
                    k_norm: None,
                }),
                ordinal,
                kv_rank: shape.kv_owner_rank(LayerIdx::new(slot)),
                layer_scalar: 1.0,
                ple: None,
            });
        }
        Ok(Weights {
            token_embd: self.weight("token_embd.weight")?,
            output_norm: self.vector("output_norm.weight", &[hidden])?,
            // Tied: the stack projects the logits through the embedding.
            lm_head: None,
            ple_model_proj: None,
            ple_proj_norm: None,
            layers,
        })
    }

    /// One tensor's directory entry, or the name of the one that is absent.
    fn desc(&self, name: &str) -> CoreResult<&TensorDesc> {
        self.file
            .find(name)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: name.to_owned(),
            })
    }

    /// One projection matrix, imported out of the mapping as it is stored.
    fn weight(&self, name: &str) -> CoreResult<Weight> {
        let desc = self.desc(name)?;
        if !WEIGHT_TYPES.contains(&desc.dtype) {
            return Err(CoreError::Unsupported {
                what: "a weight stored in a format this family's loader does not accept \
                       (q8_0 and f16 only)",
                backend: "llama",
            });
        }
        let dims = desc.shape.dims();
        let in_dim = extent(dims.first().copied().unwrap_or(0))?;
        let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
        Weight::import(
            self.backend,
            desc.dtype,
            in_dim,
            out_dim,
            self.file.mapping(),
            self.file.range_of(desc),
        )
    }

    /// One `f32` vector of exactly `dims`.
    fn vector(&self, name: &str, dims: &[u64]) -> CoreResult<DeviceBuffer> {
        let desc = self.desc(name)?;
        if desc.dtype != VECTOR_TYPE {
            return Err(CoreError::Unsupported {
                what: "a norm stored as anything but single precision",
                backend: "llama",
            });
        }
        if desc.shape.dims() != dims {
            return Err(CoreError::ShapeMismatch {
                field: "vector extent",
                expected: dims.iter().product(),
                got: desc.shape.dims().iter().product(),
            });
        }
        import_host(self.backend, self.file.mapping(), self.file.range_of(desc))
    }
}

/// One stored dimension as the `u32` every launch takes.
fn extent(dim: u64) -> CoreResult<u32> {
    u32::try_from(dim).map_err(|_| CoreError::ShapeMismatch {
        field: "tensor dimension",
        expected: u64::from(u32::MAX),
        got: dim,
    })
}

impl Model for LlamaModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.stack.open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack.close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.stack.prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.stack.decode_batch(steps)
    }

    // A decision request's questions share their opening; one pass over all
    // of them reads the weights once.
    fn prefill_last_many(&mut self, runs: &[PromptRun<'_>]) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last_many(runs)
    }

    // Every checkpoint this family loads carries the head: the container is
    // the Clef one, and the config refuses a file that does not name it.
    fn decision_mut(&mut self) -> Option<&mut dyn DecisionModel> {
        Some(self)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack.teardown()
    }
}

/// The schema decision path: the head reads this stack's final rows.
impl DecisionModel for LlamaModel {
    fn layout(&self) -> Arc<dyn shiso_core::DecisionLayout> {
        Arc::new(ClefLayout)
    }

    fn capture_hidden(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack.capture_hidden(session)
    }

    fn decide(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        questions: &[DecisionQuestion],
    ) -> CoreResult<Vec<Vec<f32>>> {
        let rows = self.stack.take_hidden(session)?;
        self.head.score(&rows, prompt, questions)
    }
}
