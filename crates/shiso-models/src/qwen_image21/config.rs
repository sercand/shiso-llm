//! The Qwen-Image-2.1 transformer's documents and tensor names.
//!
//! Three documents describe a checkpoint beside its weights:
//! `transformer/config.json` (the geometry), `model_index.json` (the
//! pipeline, and for a distilled checkpoint the `sample_sigmas` it was
//! distilled at) and `scheduler/scheduler_config.json` (the time shift).
//! [`QwenImageConfig::read`] reads all three and refuses a missing one: a
//! distilled checkpoint run without its own sigmas is a different, worse
//! model, so `model_index.json` is never optional.
//!
//! The geometry is then checked against the tensors themselves
//! ([`DitNames::check`]): a config that disagrees with the shapes it
//! describes is refused at load, naming the tensor, rather than run with
//! one of the two silently winning.
//!
//! Two exports spell the weights differently. The single-file export puts
//! every tensor at its module path (optionally under `model.diffusion_model.`)
//! and fuses the MLP's gate and value projections into one
//! `img_mlp.gate_up` matrix, gate rows first; the multi-file export keeps
//! them as `img_mlp.gate_layer` and `img_mlp.proj`. [`DitNames`] resolves
//! both. The fused matrix is run as ONE product whose output row is
//! `[gate | value]`, which is exactly the row the fused output linear's
//! `SwiGLU` reader takes (gate first) - one launch and one input rotation
//! instead of two, and no row views of the weight or its scale vector.

use std::path::Path;

use serde::Deserialize;
use shiso_vision::generator::ImageGenError;

use super::scheduler::ShiftConfig;
use crate::diffusion::weights::StStore;

/// The prefix a combined single-file export may put before every tensor.
const COMBINED_PREFIX: &str = "model.diffusion_model.";

/// Width of the sinusoidal timestep features the time embedder reads.
pub(crate) const SINUSOID_DIM: usize = 256;

/// `transformer/config.json`, the fields the forward reads.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DitConfig {
    /// Width of one attention head.
    pub attention_head_dim: usize,
    /// Head widths of the frame, height and width rotary axes.
    pub axes_dims_rope: Vec<usize>,
    /// Whether condition tokens are embedded at `t = 0` and attend causally.
    pub causal_condition: bool,
    /// Width of the text features `txt_in` reads.
    pub context_in_dim: usize,
    /// Epsilon of every norm in the blocks.
    pub eps: f64,
    /// Latent channels per token in.
    pub in_channels: usize,
    /// MLP hidden width as a multiple of the model width.
    pub mlp_ratio: usize,
    /// Attention heads per block.
    pub num_attention_heads: usize,
    /// Transformer blocks.
    pub num_layers: usize,
    /// Latent channels per token out (the velocity).
    pub out_channels: usize,
    /// Latent cells per token side.
    pub patch_size: usize,
}

impl DitConfig {
    /// The model width, `heads * head_dim`.
    pub(crate) fn hidden(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The MLP's hidden width.
    pub(crate) fn mlp_hidden(&self) -> usize {
        self.mlp_ratio * self.hidden()
    }

    /// The three rotary axis widths, checked: even, summing to the head.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when the config's axes cannot tile a head.
    pub(crate) fn rope_axes(&self) -> Result<[usize; 3], ImageGenError> {
        let axes: [usize; 3] = self.axes_dims_rope.as_slice().try_into().map_err(|_| {
            ImageGenError::Checkpoint(format!(
                "axes_dims_rope must name three axes (frame, height, width), got {:?}",
                self.axes_dims_rope
            ))
        })?;
        if axes.iter().any(|a| a % 2 != 0) || axes.iter().sum::<usize>() != self.attention_head_dim
        {
            return Err(ImageGenError::Checkpoint(format!(
                "axes_dims_rope {axes:?} must be even widths summing to the {}-wide head",
                self.attention_head_dim
            )));
        }
        Ok(axes)
    }

    /// Refuses a geometry this implementation does not run.
    fn validate(&self) -> Result<(), ImageGenError> {
        if !self.causal_condition {
            // The prefix cache is exact only because text never sees the
            // target; a bidirectional-condition model has no such prefix.
            return Err(ImageGenError::Checkpoint(
                "causal_condition is false: only the causal-condition transformer is implemented"
                    .to_string(),
            ));
        }
        if self.patch_size != 1 {
            return Err(ImageGenError::Checkpoint(format!(
                "patch_size {} is not supported: the transformer reads one latent cell per token",
                self.patch_size
            )));
        }
        if self.num_layers == 0 || self.num_attention_heads == 0 || self.mlp_ratio == 0 {
            return Err(ImageGenError::Checkpoint(
                "num_layers, num_attention_heads and mlp_ratio must all be positive".to_string(),
            ));
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err(ImageGenError::Checkpoint(format!(
                "eps {} must be positive",
                self.eps
            )));
        }
        self.rope_axes().map(|_| ())
    }
}

#[derive(Debug, Deserialize)]
struct ModelIndex {
    #[serde(default)]
    sample_sigmas: Option<Vec<f32>>,
}

/// Every document of a Qwen-Image-2.1 install that the transformer and
/// its schedule read.
#[derive(Debug, Clone)]
pub(crate) struct QwenImageConfig {
    /// The transformer geometry.
    pub dit: DitConfig,
    /// A distilled checkpoint's own sigma schedule (`model_index.json`),
    /// run verbatim when present.
    pub sample_sigmas: Option<Vec<f32>>,
    /// The scheduler's time-shift parameters.
    pub shift: ShiftConfig,
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ImageGenError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| ImageGenError::Checkpoint(format!("parsing {}: {e}", path.display())))
}

impl QwenImageConfig {
    /// Reads `transformer/config.json`, `model_index.json` and
    /// `scheduler/scheduler_config.json` under `assets`.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] naming the document that is missing or
    /// unparsable, or the geometry this implementation refuses.
    pub(crate) fn read(assets: &Path) -> Result<Self, ImageGenError> {
        let dit: DitConfig = read_json(&assets.join("transformer/config.json"))?;
        dit.validate()?;
        let index: ModelIndex = read_json(&assets.join("model_index.json"))?;
        let shift: ShiftConfig = read_json(&assets.join("scheduler/scheduler_config.json"))?;
        shift.validate()?;
        Ok(Self {
            dit,
            sample_sigmas: index.sample_sigmas,
            shift,
        })
    }
}

/// How the export spells the block MLP's input projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MlpNaming {
    /// One `img_mlp.gate_up` matrix, gate rows first.
    Fused,
    /// Separate `img_mlp.gate_layer` and `img_mlp.proj` matrices.
    Split,
}

/// The tensor names of one export.
#[derive(Debug, Clone)]
pub(crate) struct DitNames {
    prefix: &'static str,
    /// The block MLP's spelling.
    pub mlp: MlpNaming,
}

impl DitNames {
    /// Finds the export's prefix and MLP spelling from its tensor names.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] when the store holds no `img_in`
    /// under either prefix, or block 0 has neither MLP spelling.
    pub(crate) fn detect(store: &StStore) -> Result<Self, ImageGenError> {
        let prefix = if store.contains("img_in.weight") {
            ""
        } else if store.contains(&format!("{COMBINED_PREFIX}img_in.weight")) {
            COMBINED_PREFIX
        } else {
            return Err(ImageGenError::Checkpoint(
                "the transformer weights hold no 'img_in.weight' (bare or under 'model.diffusion_model.')"
                    .to_string(),
            ));
        };
        let has = |name: &str| store.contains(&format!("{prefix}transformer_blocks.0.{name}"));
        let mlp = match (
            has("img_mlp.gate_up.weight"),
            has("img_mlp.gate_layer.weight") && has("img_mlp.proj.weight"),
        ) {
            (true, false) => MlpNaming::Fused,
            (false, true) => MlpNaming::Split,
            (true, true) => {
                return Err(ImageGenError::Checkpoint(
                    "block 0 holds both a fused img_mlp.gate_up and split gate_layer/proj weights"
                        .to_string(),
                ));
            }
            (false, false) => {
                return Err(ImageGenError::Checkpoint(
                    "block 0 holds neither img_mlp.gate_up nor img_mlp.gate_layer + img_mlp.proj"
                        .to_string(),
                ));
            }
        };
        Ok(Self { prefix, mlp })
    }

    /// A non-block module's name (without `.weight`).
    pub(crate) fn top(&self, module: &str) -> String {
        format!("{}{module}", self.prefix)
    }

    /// Module `part` of block `index` (without `.weight`).
    pub(crate) fn block(&self, index: usize, part: &str) -> String {
        format!("{}transformer_blocks.{index}.{part}", self.prefix)
    }

    /// Blocks the export holds.
    pub(crate) fn count_blocks(&self, store: &StStore) -> Option<usize> {
        store.count_blocks(&format!("{}transformer_blocks.", self.prefix))
    }

    /// Checks every tensor the forward reads against the shape `config`
    /// implies, and the block count against `num_layers`.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] naming the first tensor whose shape
    /// disagrees, or the block counts.
    pub(crate) fn check(&self, store: &StStore, config: &DitConfig) -> Result<(), ImageGenError> {
        let found = self.count_blocks(store).unwrap_or(0);
        if found != config.num_layers {
            return Err(ImageGenError::Checkpoint(format!(
                "config.json names {} layers, the weights hold {found} transformer_blocks",
                config.num_layers
            )));
        }
        let (h, ctx, mlp) = (
            config.hidden() as u64,
            config.context_in_dim as u64,
            config.mlp_hidden() as u64,
        );
        let hd = config.attention_head_dim as u64;
        let mut want: Vec<(String, Vec<u64>)> = vec![
            (self.top("img_in"), vec![h, config.in_channels as u64]),
            (self.top("txt_in.text_norm"), vec![ctx]),
            (self.top("txt_in.in_layer"), vec![h, ctx]),
            (self.top("txt_in.out_layer"), vec![h, h]),
            (
                self.top("time_text_embed.timestep_embedder.linear_1"),
                vec![h, SINUSOID_DIM as u64],
            ),
            (
                self.top("time_text_embed.timestep_embedder.linear_2"),
                vec![h, h],
            ),
            (self.top("modulation.1"), vec![4 * h, h]),
            (self.top("norm_out.linear"), vec![h, h]),
            (self.top("proj_out"), vec![config.out_channels as u64, h]),
        ];
        for i in 0..config.num_layers {
            for part in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
                want.push((self.block(i, part), vec![h, h]));
            }
            want.push((self.block(i, "attn.norm_q"), vec![hd]));
            want.push((self.block(i, "attn.norm_k"), vec![hd]));
            match self.mlp {
                MlpNaming::Fused => {
                    want.push((self.block(i, "img_mlp.gate_up"), vec![2 * mlp, h]));
                }
                MlpNaming::Split => {
                    want.push((self.block(i, "img_mlp.gate_layer"), vec![mlp, h]));
                    want.push((self.block(i, "img_mlp.proj"), vec![mlp, h]));
                }
            }
            want.push((self.block(i, "img_mlp.out"), vec![h, mlp]));
        }
        for (module, shape) in want {
            let name = format!("{module}.weight");
            let entry = store.require(&name)?;
            if entry.shape != shape {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{name}' has shape {:?}, the config implies {shape:?}",
                    entry.shape
                )));
            }
        }
        Ok(())
    }
}
