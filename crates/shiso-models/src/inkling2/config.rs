//! Configuration parser and validator for Inkling-2 checkpoints.

#![expect(
    clippy::doc_markdown,
    clippy::items_after_statements,
    reason = "config parser mirrors checkpoint JSON field names"
)]

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;
use shiso_core::{CoreError, CoreResult, StopTokens, TokenId};

/// Top-level `config.json` for an Inkling-2 checkpoint.
#[derive(Debug, Clone)]
pub struct Inkling2Config {
    /// Model type tag (`"inkling2_loop"` or `"inkling2"`).
    pub model_type: String,
    /// Primary EOS token id.
    pub eos_token_id: TokenId,
    /// Stop token set.
    pub stop_tokens: StopTokens,
    /// Text decoder configuration.
    pub text: TextConfig,
    /// Optional vision encoder configuration.
    pub vision: Option<VisionConfig>,
    /// Optional MTP / DSpark drafter configuration.
    pub mtp: Option<MtpConfig>,
}

/// Looped core configuration (`text_config.loop_config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct LoopConfig {
    /// First physical layer in the looped core (inclusive).
    pub core_start: u32,
    /// End of the looped core (exclusive).
    pub core_end: u32,
    /// Number of sequential passes through `core_start..core_end`.
    pub num_passes: u32,
    /// Low-rank bottleneck width of the cross-pass loop controllers.
    pub gate_rank: u32,
    /// Variable-loop mode (unsupported when `true`).
    #[serde(default)]
    pub is_varloop: bool,
}

/// Text decoder configuration (`text_config`).
#[derive(Debug, Clone)]
pub struct TextConfig {
    /// Maximum context length.
    pub model_max_length: u32,
    /// Hidden / residual width.
    pub hidden_size: u32,
    /// Number of physical decoder layers in the checkpoint.
    pub num_hidden_layers: u32,
    /// Padded vocabulary size stored in `embed` and `unembed`.
    pub vocab_size: u32,
    /// Unpadded vocabulary size exposed in output logits.
    pub unpadded_vocab_size: u32,
    /// Number of query heads.
    pub num_attention_heads: u32,
    /// Number of key/value heads.
    pub num_key_value_heads: u32,
    /// Width of one attention head.
    pub head_dim: u32,
    /// Relative position state width per head.
    pub d_rel: u32,
    /// Relative position bias extent (max backward distance).
    pub rel_extent: u32,
    /// Log-length scaling floor for full-attention layers.
    pub log_scaling_n_floor: f32,
    /// Log-length scaling alpha for full-attention layers.
    pub log_scaling_alpha: f32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// L2 clip threshold for Q and K heads.
    pub qk_l2_clip_threshold: f32,
    /// L2 clip threshold for relative states.
    pub rel_l2_clip_threshold: f32,
    /// Physical layer indices that use local sliding-window attention.
    pub local_layer_ids: HashSet<u32>,
    /// Short causal convolution kernel width.
    pub sconv_kernel_size: u32,
    /// Final logit softcapping value.
    pub final_logit_softcapping: f32,
    /// Sliding window size for local layers.
    pub sliding_window_size: u32,
    /// Physical layers per block-attention-residual block.
    pub n_layers_per_attn_res_block: u32,
    /// Number of routed experts per MoE layer.
    pub n_routed_experts: u32,
    /// Active experts routed per token.
    pub num_experts_per_tok: u32,
    /// Per-expert MoE intermediate width.
    pub intermediate_size: u32,
    /// Softcap on SwiGLU gate factor.
    pub mlp_gate_softcap: f32,
    /// Softcap on SwiGLU up factor.
    pub mlp_up_softcap: f32,
    /// Router scaling factor after top-k normalization.
    pub route_scale: f32,
    /// Optional looped middle core configuration.
    pub loop_config: Option<LoopConfig>,
}

/// Vision encoder configuration (`vision_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Text hidden size that the projector outputs.
    pub text_hidden_size: u32,
    /// Temporal patch size (2 in checkpoint).
    pub temporal_patch_size: u32,
    /// Native spatial patch size in pixels (20).
    pub native_patch_size: u32,
    /// Spatial merge factor (2 → 40×40 pixels per LM image token).
    pub adapter_merge_size: u32,
    /// Projector hidden dimension (4096).
    pub projection_hidden_dim: u32,
    /// Whether projector linears carry bias.
    pub projector_bias: bool,
    /// ViT hidden dimension (1024).
    pub hidden_size: u32,
    /// Number of ViT encoder layers (23).
    pub num_hidden_layers: u32,
    /// Number of ViT attention heads (16).
    pub num_attention_heads: u32,
    /// ViT MLP expansion ratio (4.0).
    pub mlp_ratio: f32,
    /// Learned 2D position embedding grid side (32).
    pub posemb_grid_size: u32,
    /// LayerNorm epsilon (1e-5).
    pub layer_norm_eps: f32,
}

/// MTP / DSpark speculative drafter configuration (`mtp_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct MtpConfig {
    /// Draft hidden width (must match text hidden size).
    pub hidden_size: u32,
    /// Number of draft decoder layers (3).
    pub num_hidden_layers: u32,
    /// Draft query heads (24).
    pub num_attention_heads: u32,
    /// Draft KV heads (8).
    pub num_key_value_heads: u32,
    /// Draft head width (128).
    pub head_dim: u32,
    /// Draft dense MLP intermediate width (9216).
    pub intermediate_size: u32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency (`1_000_000.0`).
    pub rope_theta: f32,
    /// Draft layer indices using sliding-window attention (`[0, 2]`).
    pub local_layer_ids: Vec<u32>,
    /// Draft sliding window size (`1024`).
    pub sliding_window_size: u32,
    /// Parallel draft block size (`8` = 1 anchor + 7 drafted).
    pub block_size: u32,
    /// Markov / RNN head bottleneck rank (`256`).
    pub markov_rank: u32,
    /// Markov head type (`"rnn_hidden"`).
    pub markov_head_type: String,
    /// Logical target layer indices captured as draft context taps (`[5, 23, 41, 48]`).
    pub target_layer_ids: Vec<u32>,
}

/// One logical layer visit in the unrolled execution schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalLayer {
    /// Logical index `0..total_logical_layers`.
    pub logical_idx: u32,
    /// Physical weight layer index `0..num_hidden_layers`.
    pub physical_idx: u32,
    /// Loop pass index (`0` for prelude/postlude/first core pass, `1..num_passes` for repeats).
    pub pass_idx: u32,
    /// Whether this logical layer is inside the looped core.
    pub in_loop_core: bool,
    /// Whether this layer uses local sliding-window attention.
    pub is_local: bool,
}

impl TextConfig {
    /// Whether physical layer `layer_idx` uses local sliding-window attention.
    #[must_use]
    pub fn is_local(&self, layer_idx: u32) -> bool {
        self.local_layer_ids.contains(&layer_idx)
    }

    /// Total logical layers executed per token (48 for Inkling-2-Mini).
    #[must_use]
    pub fn num_logical_layers(&self) -> u32 {
        match self.loop_config {
            Some(lc) => {
                let core = lc.core_end - lc.core_start;
                self.num_hidden_layers + core * (lc.num_passes - 1)
            }
            None => self.num_hidden_layers,
        }
    }

    /// Builds the full unrolled execution schedule of logical layers.
    #[must_use]
    pub fn schedule(&self) -> Vec<LogicalLayer> {
        let mut out = Vec::with_capacity(self.num_logical_layers() as usize);
        let mut logical_idx = 0u32;
        if let Some(lc) = self.loop_config {
            for p in 0..lc.core_start {
                out.push(LogicalLayer {
                    logical_idx,
                    physical_idx: p,
                    pass_idx: 0,
                    in_loop_core: false,
                    is_local: self.is_local(p),
                });
                logical_idx += 1;
            }
            for pass in 0..lc.num_passes {
                for p in lc.core_start..lc.core_end {
                    out.push(LogicalLayer {
                        logical_idx,
                        physical_idx: p,
                        pass_idx: pass,
                        in_loop_core: true,
                        is_local: self.is_local(p),
                    });
                    logical_idx += 1;
                }
            }
            for p in lc.core_end..self.num_hidden_layers {
                out.push(LogicalLayer {
                    logical_idx,
                    physical_idx: p,
                    pass_idx: 0,
                    in_loop_core: false,
                    is_local: self.is_local(p),
                });
                logical_idx += 1;
            }
        } else {
            for p in 0..self.num_hidden_layers {
                out.push(LogicalLayer {
                    logical_idx,
                    physical_idx: p,
                    pass_idx: 0,
                    in_loop_core: false,
                    is_local: self.is_local(p),
                });
                logical_idx += 1;
            }
        }
        out
    }
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    #[serde(default = "default_eos")]
    eos_token_id: u32,
    text_config: RawTextConfig,
    vision_config: Option<VisionConfig>,
    mtp_config: Option<MtpConfig>,
}

const fn default_eos() -> u32 {
    200_006
}

#[derive(Deserialize)]
struct RawTextConfig {
    model_max_length: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    vocab_size: u32,
    unpadded_vocab_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    d_rel: u32,
    rel_extent: u32,
    log_scaling_n_floor: f32,
    log_scaling_alpha: f32,
    rms_norm_eps: f32,
    qk_l2_clip_threshold: f32,
    rel_l2_clip_threshold: f32,
    local_layer_ids: Vec<u32>,
    sconv_kernel_size: u32,
    final_logit_softcapping: f32,
    sliding_window_size: u32,
    n_layers_per_attn_res_block: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    intermediate_size: u32,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    loop_config: Option<LoopConfig>,
}

impl Inkling2Config {
    /// Returns `true` when `model_type` names an Inkling-2 architecture.
    #[must_use]
    pub fn describes_type(model_type: &str) -> bool {
        matches!(model_type, "inkling2_loop" | "inkling2")
    }

    /// Returns `true` when `dir/config.json` names an Inkling-2 architecture.
    #[must_use]
    pub fn is_inkling2_dir(dir: &Path) -> bool {
        let Ok(raw) = std::fs::read_to_string(dir.join("config.json")) else {
            return false;
        };
        #[derive(Deserialize)]
        struct Probe {
            model_type: Option<String>,
        }
        let Ok(probe) = serde_json::from_str::<Probe>(&raw) else {
            return false;
        };
        matches!(
            probe.model_type.as_deref(),
            Some("inkling2_loop" | "inkling2")
        )
    }

    /// Reads and validates `config.json` inside `dir`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] or [`CoreError::InvalidValue`] when the document is
    /// missing, malformed, or declares unsupported dimensions.
    pub fn from_dir(dir: &Path) -> CoreResult<Self> {
        let path = dir.join("config.json");
        let raw_str = std::fs::read_to_string(&path)?;
        Self::from_json_str(&raw_str)
    }

    /// Parses and validates a `config.json` string.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when fields are invalid or unsupported.
    pub fn from_json_str(raw_str: &str) -> CoreResult<Self> {
        let raw: RawConfig =
            serde_json::from_str(raw_str).map_err(|err| CoreError::InvalidValue {
                key: "config.json",
                value: err.to_string(),
            })?;
        if !matches!(raw.model_type.as_str(), "inkling2_loop" | "inkling2") {
            return Err(CoreError::InvalidValue {
                key: "model_type",
                value: raw.model_type,
            });
        }
        let t = raw.text_config;
        if t.hidden_size == 0 || t.num_hidden_layers == 0 || t.head_dim == 0 {
            return Err(CoreError::InvalidValue {
                key: "text_config",
                value: "zero dimension in text_config".to_owned(),
            });
        }
        if t.unpadded_vocab_size == 0 || t.unpadded_vocab_size > t.vocab_size {
            return Err(CoreError::InvalidValue {
                key: "unpadded_vocab_size",
                value: format!("{} > {}", t.unpadded_vocab_size, t.vocab_size),
            });
        }
        if t.n_layers_per_attn_res_block == 0 {
            return Err(CoreError::InvalidValue {
                key: "n_layers_per_attn_res_block",
                value: "0".to_owned(),
            });
        }
        if let Some(lc) = t.loop_config {
            if lc.is_varloop {
                return Err(CoreError::InvalidValue {
                    key: "loop_config.is_varloop",
                    value: "true is not supported".to_owned(),
                });
            }
            if lc.core_start >= lc.core_end
                || lc.core_end > t.num_hidden_layers
                || lc.num_passes < 2
                || lc.gate_rank == 0
                || lc.core_start % t.n_layers_per_attn_res_block != 0
                || lc.core_end % t.n_layers_per_attn_res_block != 0
            {
                return Err(CoreError::InvalidValue {
                    key: "loop_config",
                    value: format!("{lc:?}"),
                });
            }
        }
        let eos = TokenId::new(raw.eos_token_id);
        let text = TextConfig {
            model_max_length: t.model_max_length,
            hidden_size: t.hidden_size,
            num_hidden_layers: t.num_hidden_layers,
            vocab_size: t.vocab_size,
            unpadded_vocab_size: t.unpadded_vocab_size,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            d_rel: t.d_rel,
            rel_extent: t.rel_extent,
            log_scaling_n_floor: t.log_scaling_n_floor,
            log_scaling_alpha: t.log_scaling_alpha,
            rms_norm_eps: t.rms_norm_eps,
            qk_l2_clip_threshold: t.qk_l2_clip_threshold,
            rel_l2_clip_threshold: t.rel_l2_clip_threshold,
            local_layer_ids: t.local_layer_ids.into_iter().collect(),
            sconv_kernel_size: t.sconv_kernel_size,
            final_logit_softcapping: t.final_logit_softcapping,
            sliding_window_size: t.sliding_window_size,
            n_layers_per_attn_res_block: t.n_layers_per_attn_res_block,
            n_routed_experts: t.n_routed_experts,
            num_experts_per_tok: t.num_experts_per_tok,
            intermediate_size: t.intermediate_size,
            mlp_gate_softcap: t.mlp_gate_softcap,
            mlp_up_softcap: t.mlp_up_softcap,
            route_scale: t.route_scale,
            loop_config: t.loop_config,
        };
        Ok(Self {
            model_type: raw.model_type,
            eos_token_id: eos,
            stop_tokens: StopTokens::one(eos),
            text,
            vision: raw.vision_config,
            mtp: raw.mtp_config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_unrolls_24_physical_layers_into_48_logical_layers() {
        let json = r#"{
          "model_type": "inkling2_loop",
          "eos_token_id": 200006,
          "text_config": {
            "model_max_length": 1048576,
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "d_rel": 16,
            "rel_extent": 1024,
            "log_scaling_n_floor": 8192,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.627417,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
            "sconv_kernel_size": 4,
            "final_logit_softcapping": 30.0,
            "sliding_window_size": 1024,
            "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "intermediate_size": 1152,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "loop_config": {
              "core_start": 6,
              "core_end": 18,
              "num_passes": 3,
              "gate_rank": 64,
              "is_varloop": false
            }
          }
        }"#;
        let cfg = Inkling2Config::from_json_str(json).expect("valid config");
        assert_eq!(cfg.text.num_logical_layers(), 48);
        let sched = cfg.text.schedule();
        assert_eq!(sched.len(), 48);
        assert_eq!(sched[5].physical_idx, 5);
        assert!(!sched[5].is_local);
        assert_eq!(sched[6].physical_idx, 6);
        assert_eq!(sched[6].pass_idx, 0);
        assert_eq!(sched[18].physical_idx, 6);
        assert_eq!(sched[18].pass_idx, 1);
        assert_eq!(sched[30].physical_idx, 6);
        assert_eq!(sched[30].pass_idx, 2);
        assert_eq!(sched[42].physical_idx, 18);
        assert_eq!(sched[47].physical_idx, 23);
        let full_count = sched.iter().filter(|l| !l.is_local).count();
        assert_eq!(full_count, 8);
    }
}
