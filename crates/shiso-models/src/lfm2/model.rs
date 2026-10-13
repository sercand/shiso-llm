//! The model type: an LFM2 checkpoint, loaded and answering the shared
//! contract.
//!
//! The stack is [`shiso_engine::HybridStack`] over the engine's dense kernels:
//! the attention slots are the dense set's (per-head query and key norms fused
//! with a half-split rotation of the whole head), and the convolution slots are
//! the engine's gated short convolution. Nothing here names a kernel.
//!
//! Weights are imported out of the container's own mapping. The one tensor
//! with two jobs is `token_embd`: the stack gathers the prompt's rows out of it
//! and, with no `output.weight` in the file, projects the logits through it.
//! `token_embd_norm` is the norm applied after the last slot, not to the
//! embeddings - the reference names its final norm after the embedding and
//! applies it at the end of the stack - so it is bound as the output norm.

use std::path::Path;
use std::sync::Arc;

use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::media::{MediaModel, VisionPrefill};
use shiso_core::model::{DecodeStep, Logits, Model, ModelInfo, StopTokens};
use shiso_core::{Backend, DType, DecisionReadout};
use shiso_engine::attn::Attn;
use shiso_engine::dense::DenseKernels;
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::{Block, LayerWeights, Weights};
use shiso_engine::projection::Weight;
use shiso_engine::shape::LayerKind;
use shiso_engine::short_conv::ShortConv;
use shiso_engine::{EngineOptions, HybridStack};
use shiso_modelfile::{GgufFile, TensorDesc};

use crate::lfm2::config::{GGUF_ARCH, Lfm2Config, Lfm2Readout};
use crate::lfm2::decision::D1Readout;

/// Formats a projection of this family may be stored in: the two the
/// published exports use and the `q8_0` of the higher-precision one.
const WEIGHT_TYPES: &[DType] = &[DType::Q4_K, DType::Q6_K, DType::Q8_0];

/// The only format a norm or a convolution kernel is stored in.
const VECTOR_TYPE: DType = DType::F32;

/// A loaded LFM2 checkpoint.
#[derive(Debug)]
pub struct Lfm2Model {
    stack: HybridStack,
    info: ModelInfo,
    readout: Lfm2Readout,
    /// Bits a weight of the first feed-forward was stored at, which is what
    /// the checkpoint tier tags stored rows with.
    quant_bits: u8,
}

impl Lfm2Model {
    /// Opens a GGUF checkpoint and puts it on the device.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming the first tensor the container does
    /// not hold, [`CoreError::Unsupported`] for a tensor stored in a format
    /// this family does not ship, [`CoreError::ShapeMismatch`] for one whose
    /// extent disagrees with the geometry, and whatever the device or the
    /// stack reports.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Lfm2Config,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        let file = GgufFile::open_shared(path).map_err(|source| CoreError::Backend {
            backend: GGUF_ARCH,
            op: "open the checkpoint",
            source: Box::new(source),
        })?;
        let loader = Loader {
            file: &file,
            backend,
        };
        let weights = loader.weights(config)?;
        let embed = loader.desc("token_embd.weight")?.dtype;
        let quant_bits = match loader.desc("blk.0.ffn_gate.weight")?.dtype {
            DType::Q8_0 => 8,
            DType::Q6_K => 6,
            _ => 4,
        };
        let shape = config.shape();
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
            readout: config.readout(),
            quant_bits,
        })
    }

    /// How the checkpoint is meant to be read.
    #[must_use]
    pub const fn readout(&self) -> Lfm2Readout {
        self.readout
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
    /// Every resident weight, slot by slot.
    fn weights(&self, config: &Lfm2Config) -> CoreResult<Weights> {
        let shape = config.shape();
        let hidden = u64::from(shape.hidden);
        let mut layers = Vec::with_capacity(shape.layers.len());
        let (mut conv_rank, mut attn_rank) = (0u32, 0u32);
        for (index, kind) in shape.layers.iter().enumerate() {
            let slot = u32::try_from(index).map_err(|_| CoreError::ShapeMismatch {
                field: "layer slot index",
                expected: u64::from(u32::MAX),
                got: u64::try_from(index).unwrap_or(u64::MAX),
            })?;
            let il = LayerIdx::new(slot);
            let at = |role: &str| format!("blk.{slot}.{role}");
            // The ordinal indexes the slot's own kind: the session's
            // recurrent records for a convolution slot, the attention widths
            // for an attention slot.
            let (block, ordinal) = match kind {
                LayerKind::ShortConv => {
                    conv_rank += 1;
                    (
                        Block::ShortConv(self.short_conv(slot, config)?),
                        conv_rank - 1,
                    )
                }
                LayerKind::Attn => {
                    attn_rank += 1;
                    (Block::Attn(self.attn(slot)?), attn_rank - 1)
                }
                LayerKind::Gdn | LayerKind::Ssm => {
                    return Err(CoreError::Unsupported {
                        what: "a gated delta-rule or state-space slot in this family",
                        backend: "lfm2",
                    });
                }
            };
            layers.push(LayerWeights {
                // The container calls every slot's input norm `attn_norm`,
                // the convolution slots' included.
                input_norm: self.vector(&at("attn_norm.weight"), &[hidden])?,
                post_attn_norm: None,
                ffn_norm: Some(self.vector(&at("ffn_norm.weight"), &[hidden])?),
                post_ffn_norm: None,
                ffn: Some(Ffn {
                    gate: Some(self.weight(&at("ffn_gate.weight"))?),
                    up: self.weight(&at("ffn_up.weight"))?,
                    down: self.weight(&at("ffn_down.weight"))?,
                }),
                moe: None,
                block,
                ordinal,
                kv_rank: shape.kv_owner_rank(il),
                layer_scalar: 1.0,
                ple: None,
            });
        }
        if self.file.find("output.weight").is_some() {
            // This family ties its logits to the embedding. A file that ships
            // a head of its own is not one this loader was written against,
            // and running it tied would silently ignore that head.
            return Err(CoreError::Unsupported {
                what: "an untied output projection in this family",
                backend: "lfm2",
            });
        }
        Ok(Weights {
            token_embd: self.weight("token_embd.weight")?,
            output_norm: self.vector("token_embd_norm.weight", &[hidden])?,
            lm_head: None,
            ple_model_proj: None,
            ple_proj_norm: None,
            layers,
        })
    }

    /// One short-convolution slot.
    fn short_conv(&self, slot: u32, config: &Lfm2Config) -> CoreResult<ShortConv> {
        let shape = config.shape();
        let taps = shape.short_conv.map_or(0, |short| short.taps);
        let at = |role: &str| format!("blk.{slot}.shortconv.{role}");
        Ok(ShortConv {
            in_proj: self.weight(&at("in_proj.weight"))?,
            // `[channels][taps]` in memory, oldest tap first, which is the
            // layout the kernel reads.
            conv_weight: self.vector(
                &at("conv.weight"),
                &[u64::from(taps), u64::from(shape.hidden)],
            )?,
            out_proj: self.weight(&at("out_proj.weight"))?,
        })
    }

    /// One attention slot: four projections and the two per-head norms.
    fn attn(&self, slot: u32) -> CoreResult<Attn> {
        let at = |role: &str| format!("blk.{slot}.{role}");
        let q_norm = self.desc(&at("attn_q_norm.weight"))?;
        let head = q_norm.shape.dims().to_vec();
        Ok(Attn {
            q: self.weight(&at("attn_q.weight"))?,
            k: Some(self.weight(&at("attn_k.weight"))?),
            v: Some(self.weight(&at("attn_v.weight"))?),
            output: self.weight(&at("attn_output.weight"))?,
            q_norm: Some(self.vector(&at("attn_q_norm.weight"), &head)?),
            k_norm: Some(self.vector(&at("attn_k_norm.weight"), &head)?),
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
                what: "a weight stored in a format this family's loader does not accept",
                backend: "lfm2",
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
                what: "a norm or convolution kernel stored as anything but single precision",
                backend: "lfm2",
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

impl Model for Lfm2Model {
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

    // The stack splices a tower's rows wherever the placeholder runs are; the
    // tower itself is the family's and is opened beside the model.
    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    // Only a checkpoint whose container names the d1 readout answers this
    // way; a plain LFM2 language model stays a chat model, because reading
    // its next-token row as a calibrated answer is not what it was made for.
    fn decision_readout(&self) -> Option<Arc<dyn DecisionReadout>> {
        match self.readout {
            Lfm2Readout::Decision => Some(Arc::new(D1Readout)),
            Lfm2Readout::Generative => None,
        }
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack.teardown()
    }
}

/// The media path: one call reads the whole media prompt, as the other
/// families' implementations state.
impl MediaModel for Lfm2Model {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last_media(session, tokens, media)
    }
}
