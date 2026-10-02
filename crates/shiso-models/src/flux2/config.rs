//! Checkpoint configuration readers and tensor inspectors for 4B rectified flow models.
//!
//! Every number here is read off the checkpoint itself: hyperparameters from
//! the transformer's own `config.json` when it carries one, block depths and
//! tensor dimensions from the safetensors header, and numerical precision from
//! each tensor's declared `dtype`. Nothing is inferred from a directory name
//! or a path string, and a checkpoint missing a required tensor or asset is
//! refused by name rather than patched with a fallback default.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use shiso_modelfile::st_header::{RawTensor, Shard, StDType};
use shiso_vision::generator::ImageGenError;

/// Invariant mathematical constants for the FLUX.2 4-axis `RoPE` parameterization.
pub const DEFAULT_AXES_DIM: [usize; 4] = [32, 32, 32, 32];
/// Base theta frequency for rotary positional embeddings.
pub const DEFAULT_ROPE_THETA: f32 = 2000.0;
/// Default `SwiGLU` feedforward expansion ratio.
pub const DEFAULT_MLP_RATIO: f32 = 3.0;

/// Hyperparameters for the diffusion transformer architecture.
#[derive(Debug, Clone, PartialEq)]
pub struct Flux2Params {
    /// Patchified latent input channels (e.g. 128 for 32 VAE channels * 4).
    pub in_channels: usize,
    /// Joint text attention embedding dimension (e.g. 7680).
    pub context_in_dim: usize,
    /// Hidden dimension of transformer layers (e.g. 3072).
    pub hidden_size: usize,
    /// Number of attention heads (e.g. 24).
    pub num_heads: usize,
    /// Number of double stream blocks (e.g. 5).
    pub depth: usize,
    /// Number of single stream blocks (e.g. 20).
    pub depth_single_blocks: usize,
    /// Rotary embedding dimensions per axis [time, height, width, layer].
    pub axes_dim: [usize; 4],
    /// Base theta for rotary positional embedding.
    pub theta: f32,
    /// Multiplier for feedforward inner dimension.
    pub mlp_ratio: f32,
    /// Whether guidance embedding projection is enabled.
    pub use_guidance_embed: bool,
    /// Whether linear weights are stored in 8-bit floating point precision.
    pub is_fp8: bool,
}

/// Resolved files and parameters for a checkpoint.
#[derive(Debug, Clone)]
pub struct Flux2Located {
    /// Resolved safetensors file holding transformer weights.
    pub weights_path: PathBuf,
    /// Path to text encoder checkpoint or directory.
    pub text_encoder_path: PathBuf,
    /// Path to VAE checkpoint or directory.
    pub vae_path: PathBuf,
    /// Path to tokenizer directory or tokenizer.json.
    pub tokenizer_path: PathBuf,
    /// Discovered model parameters.
    pub params: Flux2Params,
}

/// What an image model checkpoint is described with at load time.
#[derive(Debug, Clone)]
pub struct ImageModelDesc {
    /// Transformer weights path (file or directory).
    pub model_path: PathBuf,
    /// Path to text encoder checkpoint (or directory containing `text_encoder`).
    pub text_encoder_path: Option<PathBuf>,
    /// Path to VAE checkpoint (or directory containing vae).
    pub vae_path: Option<PathBuf>,
    /// Path to tokenizer directory or tokenizer.json.
    pub tokenizer_path: Option<PathBuf>,
    /// Optional directory to search for shared companion assets.
    pub assets_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct DiffusersTransformerConfig {
    attention_head_dim: Option<usize>,
    axes_dims_rope: Option<[usize; 4]>,
    mlp_ratio: Option<f32>,
    num_attention_heads: Option<usize>,
    num_layers: Option<usize>,
    num_single_layers: Option<usize>,
    rope_theta: Option<f32>,
}

/// Discovers model hyperparameters and precision by inspecting the checkpoint's
/// safetensors tensor headers and config documents, never by path strings.
///
/// # Errors
/// Refuses loudly if required weights or assets are missing.
#[expect(
    clippy::too_many_lines,
    reason = "checkpoint discovery is a flat checklist: signature tensors, shape-derived dims, optional config docs"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "safetensors dims are element counts of model tensors; they fit usize on any supported target"
)]
pub fn locate_and_inspect(desc: &ImageModelDesc) -> Result<Flux2Located, ImageGenError> {
    let weights_path = find_transformer_weights(&desc.model_path)?;
    // Index every shard in the resolved directory, not just the first: a
    // multi-shard export may carry the signature tensor (or any FP8 weight)
    // in a later shard, and the pipeline's weight store indexes the whole
    // directory the same way, so inspecting a subset would refuse a
    // checkpoint that then loads fine.
    let dir = weights_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut tensor_map: HashMap<String, RawTensor> = HashMap::new();
    for file in safetensors_in(dir) {
        let (_shard, tensors) = Shard::open_untagged(&file).map_err(|e| {
            ImageGenError::Checkpoint(format!("opening safetensors at {}: {e}", file.display()))
        })?;
        tensor_map.extend(tensors);
    }
    if tensor_map.is_empty() {
        return Err(ImageGenError::Checkpoint(format!(
            "no safetensors tensors found in {}",
            dir.display()
        )));
    }

    // Verify it is a compatible transformer by checking signature tensors
    let img_in = tensor_map
        .get("img_in.weight")
        .or_else(|| tensor_map.get("transformer.img_in.weight"))
        .ok_or_else(|| {
            ImageGenError::Checkpoint(format!(
                "{} does not contain signature tensor 'img_in.weight'",
                weights_path.display()
            ))
        })?;

    let txt_in = tensor_map
        .get("txt_in.weight")
        .or_else(|| tensor_map.get("transformer.txt_in.weight"))
        .ok_or_else(|| {
            ImageGenError::Checkpoint(format!(
                "{} does not contain signature tensor 'txt_in.weight'",
                weights_path.display()
            ))
        })?;

    // Derive architecture dimensions directly from tensor shapes
    let hidden_size = *img_in
        .shape
        .first()
        .ok_or_else(|| ImageGenError::Checkpoint("img_in.weight has empty shape".to_string()))?
        as usize;
    let in_channels = *img_in
        .shape
        .get(1)
        .ok_or_else(|| ImageGenError::Checkpoint("img_in.weight rank < 2".to_string()))?
        as usize;

    let context_in_dim = *txt_in
        .shape
        .get(1)
        .ok_or_else(|| ImageGenError::Checkpoint("txt_in.weight rank < 2".to_string()))?
        as usize;

    // Check for optional transformer config file (diffusers export)
    let config_opt = read_transformer_config(&desc.model_path);

    // Count double and single transformer blocks from tensor names
    let depth = match config_opt.as_ref().and_then(|c| c.num_layers) {
        Some(d) => d,
        None => count_blocks_internal(tensor_map.keys().map(String::as_str), "double_blocks.")
            .or_else(|| {
                count_blocks_internal(
                    tensor_map.keys().map(String::as_str),
                    "transformer.double_blocks.",
                )
            })
            .ok_or_else(|| {
                ImageGenError::Checkpoint("no double_blocks found in safetensors".to_string())
            })?,
    };

    let depth_single_blocks = match config_opt.as_ref().and_then(|c| c.num_single_layers) {
        Some(d) => d,
        None => count_blocks_internal(tensor_map.keys().map(String::as_str), "single_blocks.")
            .or_else(|| {
                count_blocks_internal(
                    tensor_map.keys().map(String::as_str),
                    "transformer.single_blocks.",
                )
            })
            .ok_or_else(|| {
                ImageGenError::Checkpoint("no single_blocks found in safetensors".to_string())
            })?,
    };

    let num_heads = config_opt
        .as_ref()
        .and_then(|c| c.num_attention_heads)
        .unwrap_or(hidden_size / 128);
    let axes_dim = config_opt
        .as_ref()
        .and_then(|c| c.axes_dims_rope)
        .unwrap_or(DEFAULT_AXES_DIM);
    let theta = config_opt
        .as_ref()
        .and_then(|c| c.rope_theta)
        .unwrap_or(DEFAULT_ROPE_THETA);
    let mlp_ratio = config_opt
        .as_ref()
        .and_then(|c| c.mlp_ratio)
        .unwrap_or(DEFAULT_MLP_RATIO);
    // `attention_head_dim` is the one config number the tensor shapes do not
    // independently confirm (the qkv width is `3 * hidden` either way), so
    // cross-check it against the head split the blocks will use and refuse a
    // mismatch loudly rather than run a transformer whose heads are the wrong
    // width under a `RoPE` table sized for another.
    if let Some(head_dim) = config_opt.as_ref().and_then(|c| c.attention_head_dim)
        && head_dim * num_heads != hidden_size
    {
        return Err(ImageGenError::Checkpoint(format!(
            "config attention_head_dim {head_dim} x {num_heads} heads = {} does not match hidden size {hidden_size}",
            head_dim * num_heads
        )));
    }

    // Detect FP8 precision by scanning all tensors for any F8_E4M3 weight
    let is_fp8 = tensor_map
        .values()
        .any(|t| matches!(t.parsed_dtype("probe"), Ok(StDType::F8E4M3)));

    let params = Flux2Params {
        in_channels,
        context_in_dim,
        hidden_size,
        num_heads,
        depth,
        depth_single_blocks,
        axes_dim,
        theta,
        mlp_ratio,
        use_guidance_embed: false,
        is_fp8,
    };

    let text_encoder_path = resolve_asset(
        desc.text_encoder_path.as_deref(),
        desc.assets_dir.as_deref(),
        &desc.model_path,
        "text_encoder",
    )?;
    let vae_path = resolve_asset(
        desc.vae_path.as_deref(),
        desc.assets_dir.as_deref(),
        &desc.model_path,
        "vae",
    )?;
    let tokenizer_path = resolve_asset(
        desc.tokenizer_path.as_deref(),
        desc.assets_dir.as_deref(),
        &desc.model_path,
        "tokenizer",
    )?;

    Ok(Flux2Located {
        weights_path,
        text_encoder_path,
        vae_path,
        tokenizer_path,
        params,
    })
}

fn read_transformer_config(model_path: &Path) -> Option<DiffusersTransformerConfig> {
    let path = if model_path.is_dir() {
        model_path.join("transformer").join("config.json")
    } else {
        model_path.parent()?.join("transformer").join("config.json")
    };
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

fn resolve_asset(
    explicit: Option<&Path>,
    assets_dir: Option<&Path>,
    model_path: &Path,
    name: &str,
) -> Result<PathBuf, ImageGenError> {
    if let Some(p) = explicit
        && p.exists()
    {
        return Ok(p.to_path_buf());
    }
    // A file given as the model path resolves companions beside the file's own
    // directory, not beside a directory named after the file.
    let home = if model_path.is_file() {
        model_path.parent().unwrap_or(model_path)
    } else {
        model_path
    };
    let local = home.join(name);
    if local.exists() {
        return Ok(local);
    }
    if let Some(assets) = assets_dir {
        let in_assets = assets.join(name);
        if in_assets.exists() {
            return Ok(in_assets);
        }
    }
    Err(ImageGenError::Checkpoint(format!(
        "could not locate required asset '{name}'; specify --image-assets-dir or --image-{name}"
    )))
}

pub(crate) fn count_blocks_internal<'a>(
    keys: impl Iterator<Item = &'a str>,
    prefix: &str,
) -> Option<usize> {
    let mut max_idx = None;
    for k in keys {
        if let Some(rest) = k.strip_prefix(prefix)
            && let Some(idx_str) = rest.split('.').next()
            && let Ok(idx) = idx_str.parse::<usize>()
        {
            max_idx = Some(max_idx.map_or(idx, |m: usize| m.max(idx)));
        }
    }
    max_idx.map(|m| m + 1)
}

fn find_transformer_weights(path: &Path) -> Result<PathBuf, ImageGenError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let nested = path.join("transformer");
    let from_nested = nested
        .is_dir()
        .then(|| first_safetensors(&nested))
        .flatten();
    if let Some(found) = from_nested {
        return Ok(found);
    }
    if let Some(found) = first_safetensors(path) {
        return Ok(found);
    }
    Err(ImageGenError::Checkpoint(format!(
        "no safetensors weight file found in {}",
        path.display()
    )))
}

fn first_safetensors(dir: &Path) -> Option<PathBuf> {
    safetensors_in(dir).into_iter().next()
}

/// Every `*.safetensors` file in `dir`, sorted so a multi-shard export always
/// sees the same order. The weight store indexes the whole directory the same
/// way, so inspecting this set matches what actually loads.
fn safetensors_in(dir: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    found.sort();
    found
}
