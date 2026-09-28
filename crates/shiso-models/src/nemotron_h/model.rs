//! The model type: a checkpoint of this family, loaded and answering the shared
//! contract.
//!
//! The container is a directory of MLX safetensors shards, 4-bit affine in
//! groups of 64 for every matrix but the router, which is `bf16` and is widened
//! here. The stack is [`shiso_engine::HybridStack`] over the engine's own
//! parametric kernels: the state-space and routed blocks and the plain
//! attention store are the engine's, and nothing here names a kernel.
//!
//! # What is imported in place and what is copied
//!
//! The routed expert stacks - most of the checkpoint's bytes - are the
//! container's own row-major arrays, which is exactly what the expert stage
//! reads, so they are wrapped on the device straight out of the mapping. The
//! other matrices go through the projection's own import, and the vectors are
//! widened from `bf16` to the `f32` the kernels read.

use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceBuffer;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, SessionId, TokenId};
use shiso_core::model::{DecodeStep, Logits, Model, ModelInfo, StopTokens};
use shiso_engine::attn::Attn;
use shiso_engine::dense::DenseKernels;
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::{Block, LayerWeights, Routed, Weights};
use shiso_engine::moe_biased::{self, BiasedMoe, RowAffine4Stack};
use shiso_engine::projection::Weight;
use shiso_engine::ssm::Ssm;
use shiso_engine::{EngineOptions, HybridStack};

use crate::nemotron_h::config::{Mixer, NemotronHConfig};
use crate::qwen4exp::mlx::{MlxCheckpoint, f32_bytes, upload};

/// The producer tag this container's shard headers carry.
const FORMAT: &str = "mlx";

/// Values one affine group covers.
const GROUP: u32 = 64;

/// A loaded checkpoint of this family.
#[derive(Debug)]
pub struct NemotronHModel {
    stack: HybridStack,
    info: ModelInfo,
}

impl NemotronHModel {
    /// The stack this model forwards to.
    pub(crate) const fn stack(&self) -> &HybridStack {
        &self.stack
    }

    /// The stack this model forwards to, mutably.
    pub(crate) const fn stack_mut(&mut self) -> &mut HybridStack {
        &mut self.stack
    }

    /// Opens a checkpoint directory and puts it on the device.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming the first tensor the checkpoint does
    /// not hold, [`CoreError::InvalidValue`] for one stored at another dtype or
    /// shape than the config implies, and whatever the device or the stack
    /// reports.
    pub fn load(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        config: &NemotronHConfig,
        opts: EngineOptions,
    ) -> CoreResult<Self> {
        let ck = MlxCheckpoint::open_as(dir, Some(FORMAT), |name| {
            name.starts_with("backbone.") || name.starts_with("lm_head.")
        })?;
        let shape = config.shape();
        let weights = Loader { ck: &ck, backend }.weights(config)?;
        let stack = HybridStack::new(
            backend,
            shape.clone(),
            weights,
            DType::MlxAffine4Gs64,
            Arc::new(DenseKernels),
            None,
            opts,
        )?;
        // The expert stacks were wrapped in place, so the mapping stays alive
        // as long as the device buffers do: they hold the shards themselves.
        drop(ck);
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: shape.vocab,
                hidden: shape.hidden,
                n_layers: shape.n_layers(),
                stop_tokens: StopTokens::new(config.stop_tokens())?,
            },
            stack,
        })
    }
}

/// The checkpoint and the device, for the imports below.
struct Loader<'c> {
    ck: &'c MlxCheckpoint,
    backend: &'c Arc<dyn Backend>,
}

impl Loader<'_> {
    /// Every resident weight, slot by slot.
    fn weights(&self, config: &NemotronHConfig) -> CoreResult<Weights> {
        let shape = config.shape();
        let hidden = shape.hidden;
        let mut layers = Vec::with_capacity(config.slots().len());
        let (mut ssm_rank, mut attn_rank) = (0u32, 0u32);
        for (index, slot) in config.slots().iter().enumerate() {
            let il = LayerIdx::new(u32::try_from(index).unwrap_or(u32::MAX));
            let base = format!("backbone.layers.{}", slot.mixer);
            let (block, ordinal) = match config.mixers()[slot.mixer as usize] {
                Mixer::Mamba => {
                    ssm_rank += 1;
                    (Block::Ssm(self.ssm(&base, config)?), ssm_rank - 1)
                }
                Mixer::Attention => {
                    attn_rank += 1;
                    (Block::Attn(self.attn(&base, config)?), attn_rank - 1)
                }
                Mixer::Moe => {
                    return Err(CoreError::InvalidValue {
                        key: "layers_block_type",
                        value: format!("slot {index} opens on routed layer {}", slot.mixer),
                    });
                }
            };
            let (ffn_norm, ffn, moe) = match slot.ffn {
                None => (None, None, None),
                Some(layer) => {
                    let at = format!("backbone.layers.{layer}");
                    let (shared, routed) = self.moe(&at, config)?;
                    (
                        Some(self.vector(&format!("{at}.norm.weight"), &[u64::from(hidden)])?),
                        Some(shared),
                        Some(Routed::Biased(routed)),
                    )
                }
            };
            layers.push(LayerWeights {
                input_norm: self.vector(&format!("{base}.norm.weight"), &[u64::from(hidden)])?,
                post_attn_norm: None,
                ffn_norm,
                post_ffn_norm: None,
                ffn,
                moe,
                block,
                ordinal,
                kv_rank: shape.kv_owner_rank(il),
                layer_scalar: 1.0,
                ple: None,
            });
        }
        Ok(Weights {
            token_embd: self.affine("backbone.embeddings", hidden, shape.vocab)?,
            output_norm: self.vector("backbone.norm_f.weight", &[u64::from(hidden)])?,
            lm_head: Some(self.affine("lm_head", hidden, shape.vocab)?),
            ple_model_proj: None,
            ple_proj_norm: None,
            layers,
        })
    }

    /// One state-space block.
    fn ssm(&self, base: &str, config: &NemotronHConfig) -> CoreResult<Ssm> {
        let shape = config.shape();
        let s = shape.ssm.ok_or(CoreError::ShapeMismatch {
            field: "state-space geometry",
            expected: 1,
            got: 0,
        })?;
        let at = |role: &str| format!("{base}.mixer.{role}");
        let heads = [u64::from(s.heads)];
        let channels = u64::from(s.conv_channels());
        Ok(Ssm {
            in_proj: self.affine(&at("in_proj"), shape.hidden, s.in_width())?,
            // `[channels, taps, 1]`, which is `[channels, taps]` in memory.
            conv_weight: self.vector(
                &at("conv1d.weight"),
                &[channels, u64::from(s.conv_kernel), 1],
            )?,
            conv_bias: self.vector(&at("conv1d.bias"), &[channels])?,
            dt_bias: self.vector(&at("dt_bias"), &heads)?,
            a_log: self.vector(&at("A_log"), &heads)?,
            d: self.vector(&at("D"), &heads)?,
            norm: self.vector(&at("norm.weight"), &[u64::from(s.inner())])?,
            out_proj: self.affine(&at("out_proj"), s.inner(), shape.hidden)?,
        })
    }

    /// One attention block: four projections and no norms.
    fn attn(&self, base: &str, config: &NemotronHConfig) -> CoreResult<Attn> {
        let shape = config.shape();
        let widths = shape.attn.first().ok_or(CoreError::ShapeMismatch {
            field: "attention slots",
            expected: 1,
            got: 0,
        })?;
        let q = widths.q_heads * widths.head_dim;
        let kv = widths.kv_heads * widths.head_dim;
        let at = |role: &str| format!("{base}.mixer.{role}");
        Ok(Attn {
            q: self.affine(&at("q_proj"), shape.hidden, q)?,
            k: Some(self.affine(&at("k_proj"), shape.hidden, kv)?),
            v: Some(self.affine(&at("v_proj"), shape.hidden, kv)?),
            output: self.affine(&at("o_proj"), q, shape.hidden)?,
            q_norm: None,
            k_norm: None,
        })
    }

    /// One routed layer: the shared expert and the routed half.
    fn moe(&self, base: &str, config: &NemotronHConfig) -> CoreResult<(Ffn, BiasedMoe)> {
        let shape = config.shape();
        let moe = shape.moe.ok_or(CoreError::ShapeMismatch {
            field: "router geometry",
            expected: 1,
            got: 0,
        })?;
        let hidden = shape.hidden;
        let at = |role: &str| format!("{base}.mixer.{role}");
        let shared = Ffn {
            gate: None,
            up: self.affine(&at("shared_experts.up_proj"), hidden, shape.intermediate)?,
            down: self.affine(&at("shared_experts.down_proj"), shape.intermediate, hidden)?,
        };
        let router = self.widened(&at("gate.weight"), hidden, moe.experts)?;
        let bias = self.ck.import(
            self.backend,
            &at("gate.e_score_correction_bias"),
            "F32",
            &[u64::from(moe.experts)],
        )?;
        let up = self.stack(&at("switch_mlp.fc1"), moe.experts, hidden, moe.intermediate)?;
        let down = self.stack(&at("switch_mlp.fc2"), moe.experts, moe.intermediate, hidden)?;
        moe_biased::check_stack(&up)?;
        moe_biased::check_stack(&down)?;
        Ok((
            shared,
            BiasedMoe {
                router,
                bias,
                scale: config.routed_scale(),
                renormalize: config.renormalize(),
                up,
                down,
            },
        ))
    }

    /// One projection of every expert, wrapped in place.
    fn stack(
        &self,
        base: &str,
        experts: u32,
        in_dim: u32,
        out_dim: u32,
    ) -> CoreResult<RowAffine4Stack> {
        let lead = [u64::from(experts), u64::from(out_dim)];
        let words = [lead[0], lead[1], u64::from(in_dim / 8)];
        let params = [lead[0], lead[1], u64::from(in_dim / GROUP)];
        Ok(RowAffine4Stack {
            weights: self
                .ck
                .import(self.backend, &format!("{base}.weight"), "U32", &words)?,
            scales: self
                .ck
                .import(self.backend, &format!("{base}.scales"), "BF16", &params)?,
            biases: self
                .ck
                .import(self.backend, &format!("{base}.biases"), "BF16", &params)?,
            in_dim,
            out_dim,
        })
    }

    /// One 4-bit affine matrix `[out_dim, in_dim]`.
    fn affine(&self, base: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        let rows = u64::from(out_dim);
        let words = self.ck.bytes(
            &format!("{base}.weight"),
            "U32",
            &[rows, u64::from(in_dim / 8)],
        )?;
        let params = [rows, u64::from(in_dim / GROUP)];
        let scales = self.ck.bytes(&format!("{base}.scales"), "BF16", &params)?;
        let biases = self.ck.bytes(&format!("{base}.biases"), "BF16", &params)?;
        Weight::import_affine_slices(
            self.backend,
            DType::MlxAffine4Gs64,
            in_dim,
            out_dim,
            words,
            scales,
            biases,
        )
    }

    /// A `bf16` matrix `[out_dim, in_dim]`, widened to `f32`.
    fn widened(&self, name: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        let values = self
            .ck
            .floats(name, &[u64::from(out_dim), u64::from(in_dim)])?;
        let buffer = upload(self.backend, f32_bytes(&values))?;
        Ok(Weight::from_buffer(DType::F32, in_dim, out_dim, buffer))
    }

    /// A `bf16` vector, widened to `f32`.
    fn vector(&self, name: &str, shape: &[u64]) -> CoreResult<DeviceBuffer> {
        let values = self.ck.floats(name, shape)?;
        upload(self.backend, f32_bytes(&values))
    }
}

impl Model for NemotronHModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.stack_mut().open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack_mut().close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.stack_mut().prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.stack_mut().prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.stack_mut().decode_batch(steps)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack_mut().teardown()
    }
}
