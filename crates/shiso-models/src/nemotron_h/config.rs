//! This family's config document, read into the engine's stack shape.
//!
//! # A layer of the checkpoint is not a slot of the stack
//!
//! The checkpoint lists 52 residual blocks, each `x + mixer(norm(x))` with one
//! of three mixers: a state-space block, an attention block, or a routed
//! feed-forward. A slot of the stack is a mixing block followed by a
//! feed-forward, each with its own norm and its own residual add, so a
//! state-space or attention layer followed by a routed one is exactly one slot,
//! and the arithmetic is unchanged: two residual blocks in a row are what a
//! slot computes. A mixing layer followed directly by another becomes a slot
//! with no feed-forward half, listed in [`HybridShape::no_ffn`].
//!
//! A routed layer that does not follow a mixing layer - two in a row, or one
//! first - has no slot to join, and is refused: no checkpoint of this family
//! has one, and the engine has no slot without a mixing block.
//!
//! # What is refused rather than approximated
//!
//! Every switch the reference implementation has and this build does not is
//! checked to be off: grouped expert selection, a latent projection around the
//! experts, projection and convolution biases other than the one the kernels
//! add, a clamp on the step size, activations other than the two the kernels
//! compute. A checkpoint that turns one on is refused by the key's name.

use std::path::Path;

use serde::Deserialize;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{LayerIdx, TokenId};
use shiso_engine::shape::{
    Activation, AttnLayer, GdnShape, HybridShape, LayerKind, MediaPositions, MoeShape, NormScheme,
    SsmShape,
};

/// The `model_type` this family answers to.
pub const MODEL_TYPE: &str = "nemotron_h";

/// The config document's keys this family reads.
#[derive(Debug, Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the document's own switches, each checked against the one value this build runs"
)]
struct Document {
    model_type: String,
    vocab_size: u32,
    hidden_size: u32,
    layers_block_type: Vec<String>,
    layer_norm_epsilon: f32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    attention_bias: bool,
    mamba_num_heads: u32,
    mamba_head_dim: u32,
    ssm_state_size: u32,
    n_groups: u32,
    conv_kernel: u32,
    use_conv_bias: bool,
    mamba_proj_bias: bool,
    mamba_hidden_act: String,
    #[serde(default)]
    time_step_limit: Option<Vec<f64>>,
    mlp_hidden_act: String,
    mlp_bias: bool,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    moe_shared_expert_intermediate_size: u32,
    n_shared_experts: u32,
    #[serde(default)]
    moe_latent_size: Option<u32>,
    n_group: u32,
    topk_group: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    #[serde(default)]
    tie_word_embeddings: bool,
    eos_token_id: EosIds,
}

/// `eos_token_id`, which the document writes as one id or a list.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosIds {
    One(u32),
    Many(Vec<u32>),
}

/// One checkpoint layer's mixer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mixer {
    /// The selective state-space block.
    Mamba,
    /// Attention with no rotation and no query/key norm.
    Attention,
    /// The routed feed-forward.
    Moe,
}

/// Which checkpoint layers one slot of the stack is built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLayers {
    /// The checkpoint layer holding the slot's mixing block.
    pub mixer: u32,
    /// The checkpoint layer holding its feed-forward, if it has one.
    pub ffn: Option<u32>,
}

/// A parsed config of this family.
#[derive(Debug, Clone)]
pub struct NemotronHConfig {
    shape: HybridShape,
    slots: Vec<SlotLayers>,
    mixers: Vec<Mixer>,
    routed_scale: f32,
    renormalize: bool,
    stop_tokens: Vec<TokenId>,
}

impl NemotronHConfig {
    /// Whether `model_type` names this family.
    #[must_use]
    pub fn describes_type(model_type: &str) -> bool {
        model_type == MODEL_TYPE
    }

    /// Reads `config.json` out of a checkpoint directory.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] when the document is absent,
    /// [`CoreError::UnknownArch`] when it names another family,
    /// [`CoreError::Json`] when it does not deserialize, and
    /// [`CoreError::InvalidValue`] naming the key of a switch this build does
    /// not implement.
    pub fn from_dir(dir: &Path) -> CoreResult<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|_| CoreError::ConfigMissing {
            key: path.display().to_string(),
        })?;
        Self::from_json(&text)
    }

    /// Reads a config document already in memory.
    ///
    /// # Errors
    ///
    /// As [`NemotronHConfig::from_dir`], less the missing file.
    pub fn from_json(text: &str) -> CoreResult<Self> {
        let doc: Document = serde_json::from_str(text)?;
        if !Self::describes_type(&doc.model_type) {
            return Err(CoreError::UnknownArch {
                model_type: doc.model_type,
            });
        }
        refuse_switches(&doc)?;
        let mixers = doc
            .layers_block_type
            .iter()
            .map(|kind| match kind.as_str() {
                "mamba" => Ok(Mixer::Mamba),
                "attention" => Ok(Mixer::Attention),
                "moe" => Ok(Mixer::Moe),
                other => Err(CoreError::InvalidValue {
                    key: "layers_block_type",
                    value: format!("{other:?} is not a layer kind this build runs"),
                }),
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let slots = pair(&mixers)?;
        let mut layers = Vec::with_capacity(slots.len());
        let mut no_ffn = Vec::new();
        for (index, slot) in slots.iter().enumerate() {
            layers.push(match mixers[slot.mixer as usize] {
                Mixer::Mamba => LayerKind::Ssm,
                Mixer::Attention => LayerKind::Attn,
                // `pair` opens slots on mixing layers only.
                Mixer::Moe => {
                    return Err(CoreError::InvalidValue {
                        key: "layers_block_type",
                        value: format!("slot {index} opens on routed layer {}", slot.mixer),
                    });
                }
            });
            if slot.ffn.is_none() {
                no_ffn.push(LayerIdx::new(u32::try_from(index).unwrap_or(u32::MAX)));
            }
        }
        let attn_slots = layers
            .iter()
            .filter(|kind| **kind == LayerKind::Attn)
            .count();
        // No rotation at all: the attention layers carry no positional
        // encoding, the state-space layers are what orders the sequence.
        let attn = vec![
            AttnLayer::new(
                doc.num_attention_heads,
                doc.num_key_value_heads,
                doc.head_dim,
                0,
                0,
                0.0
            )
            .without_qk_norm();
            attn_slots
        ];
        let shape = HybridShape {
            layers,
            attn,
            hidden: doc.hidden_size,
            vocab: doc.vocab_size,
            intermediate: doc.moe_shared_expert_intermediate_size,
            // Unused: the recurrent slots are state-space ones.
            gdn: GdnShape::new(0, 0, 0, 0),
            attn_output_gate: false,
            rms_eps: doc.layer_norm_epsilon,
            activation: Activation::Relu2,
            norms: NormScheme::PreOnly,
            embed_scale: None,
            final_softcap: None,
            layer_scalar: false,
            ple: None,
            media_positions: MediaPositions::Linear,
            v_norm: false,
            moe: Some(MoeShape::new(
                doc.n_routed_experts,
                doc.num_experts_per_tok,
                doc.n_shared_experts,
                doc.moe_intermediate_size,
            )),
            ssm: Some(SsmShape {
                heads: doc.mamba_num_heads,
                head_dim: doc.mamba_head_dim,
                state: doc.ssm_state_size,
                groups: doc.n_groups,
                conv_kernel: doc.conv_kernel,
            }),
            short_conv: None,
            no_ffn,
        };
        shape.validate()?;
        let stop_tokens = match doc.eos_token_id {
            EosIds::One(id) => vec![TokenId::new(id)],
            EosIds::Many(ids) => ids.into_iter().map(TokenId::new).collect(),
        };
        Ok(Self {
            shape,
            slots,
            mixers,
            routed_scale: doc.routed_scaling_factor,
            renormalize: doc.norm_topk_prob,
            stop_tokens,
        })
    }

    /// The stack's shape.
    #[must_use]
    pub const fn shape(&self) -> &HybridShape {
        &self.shape
    }

    /// Which checkpoint layers each slot is built from, slot zero first.
    #[must_use]
    pub fn slots(&self) -> &[SlotLayers] {
        &self.slots
    }

    /// Each checkpoint layer's mixer, layer zero first.
    #[must_use]
    pub fn mixers(&self) -> &[Mixer] {
        &self.mixers
    }

    /// What the chosen experts' weights are multiplied by.
    #[must_use]
    pub const fn routed_scale(&self) -> f32 {
        self.routed_scale
    }

    /// Whether the chosen experts' weights are renormalized to sum to one.
    #[must_use]
    pub const fn renormalize(&self) -> bool {
        self.renormalize
    }

    /// The tokens that end a turn.
    #[must_use]
    pub fn stop_tokens(&self) -> &[TokenId] {
        &self.stop_tokens
    }
}

/// Groups the checkpoint's layers into slots: each mixing layer opens one, and
/// a routed layer joins the slot the mixing layer before it opened.
fn pair(mixers: &[Mixer]) -> CoreResult<Vec<SlotLayers>> {
    let mut slots: Vec<SlotLayers> = Vec::new();
    for (layer, mixer) in mixers.iter().enumerate() {
        let layer = u32::try_from(layer).unwrap_or(u32::MAX);
        match mixer {
            Mixer::Mamba | Mixer::Attention => slots.push(SlotLayers {
                mixer: layer,
                ffn: None,
            }),
            Mixer::Moe => match slots.last_mut() {
                Some(slot) if slot.ffn.is_none() && slot.mixer + 1 == layer => {
                    slot.ffn = Some(layer);
                }
                _ => {
                    return Err(CoreError::InvalidValue {
                        key: "layers_block_type",
                        value: format!(
                            "routed layer {layer} does not follow a mixing layer, and a slot \
                             has one mixing block before its feed-forward"
                        ),
                    });
                }
            },
        }
    }
    Ok(slots)
}

/// Refuses every switch of the reference implementation this build does not
/// implement, by its key.
fn refuse_switches(doc: &Document) -> CoreResult<()> {
    let refuse = |key: &'static str, value: String| Err(CoreError::InvalidValue { key, value });
    if doc.n_group != 1 || doc.topk_group != 1 {
        return refuse(
            "n_group",
            format!(
                "grouped expert selection ({} groups, {} kept) is not implemented",
                doc.n_group, doc.topk_group
            ),
        );
    }
    if let Some(latent) = doc.moe_latent_size {
        return refuse(
            "moe_latent_size",
            format!("a {latent}-wide latent projection around the experts is not implemented"),
        );
    }
    if doc.n_shared_experts != 1 {
        return refuse(
            "n_shared_experts",
            format!(
                "{} shared experts; the slot carries exactly one",
                doc.n_shared_experts
            ),
        );
    }
    if doc.attention_bias || doc.mlp_bias || doc.mamba_proj_bias {
        return refuse(
            "attention_bias",
            "a projection bias is not implemented".to_owned(),
        );
    }
    if !doc.use_conv_bias {
        return refuse(
            "use_conv_bias",
            "the convolution kernel always adds its bias".to_owned(),
        );
    }
    if doc.mamba_hidden_act != "silu" {
        return refuse("mamba_hidden_act", doc.mamba_hidden_act.clone());
    }
    if doc.mlp_hidden_act != "relu2" {
        return refuse("mlp_hidden_act", doc.mlp_hidden_act.clone());
    }
    if let Some(limit) = &doc.time_step_limit
        && !(limit.len() == 2 && limit[0] <= 0.0 && limit[1].is_infinite())
    {
        return refuse(
            "time_step_limit",
            format!("{limit:?}: a clamp on the step size is not implemented"),
        );
    }
    if doc.tie_word_embeddings {
        return refuse(
            "tie_word_embeddings",
            "this family's loader reads its own output projection".to_owned(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Mixer, SlotLayers, pair};

    #[test]
    fn a_routed_layer_joins_the_mixing_layer_before_it() {
        use Mixer::{Attention as A, Mamba as M, Moe as E};
        let slots = pair(&[M, E, M, A, E, M]).expect("pairs");
        assert_eq!(
            slots,
            [
                SlotLayers {
                    mixer: 0,
                    ffn: Some(1)
                },
                SlotLayers {
                    mixer: 2,
                    ffn: None
                },
                SlotLayers {
                    mixer: 3,
                    ffn: Some(4)
                },
                SlotLayers {
                    mixer: 5,
                    ffn: None
                },
            ]
        );
    }

    #[test]
    fn a_routed_layer_with_no_mixing_layer_before_it_is_refused() {
        use Mixer::{Mamba as M, Moe as E};
        assert!(pair(&[E, M]).is_err());
        assert!(pair(&[M, E, E]).is_err());
    }
}
