//! Checkpoint configuration readers and tensor inspectors for 4B rectified flow models.
//!
//! Every number here is read off the checkpoint itself: hyperparameters from
//! the transformer's own `config.json` when it carries one, block depths and
//! tensor dimensions from the safetensors header, and numerical precision from
//! each tensor's declared `dtype`. Nothing is inferred from a directory name
//! or a path string, and a checkpoint missing a required tensor or asset is
//! refused by name rather than patched with a fallback default.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use shiso_modelfile::st_header::{RawTensor, Shard, StDType};
use shiso_vision::generator::ImageGenError;

use crate::diffusion::weights::count_blocks_internal;
use crate::image_registry::ImageModelDesc;

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
    /// The exact safetensors files the transformer store must open: the
    /// selected file plus its disjoint shards, never a sibling file that
    /// re-declares the same tensor names (a second export of the same
    /// model, e.g. an FP8 copy beside the BF16 one).
    pub weight_files: Vec<PathBuf>,
    /// Path to text encoder checkpoint or directory.
    pub text_encoder_path: PathBuf,
    /// Path to VAE checkpoint or directory.
    pub vae_path: PathBuf,
    /// Path to tokenizer directory or tokenizer.json.
    pub tokenizer_path: PathBuf,
    /// Discovered model parameters.
    pub params: Flux2Params,
}

#[derive(Debug, Deserialize)]
struct TransformerConfigFile {
    attention_head_dim: Option<usize>,
    axes_dims_rope: Option<[usize; 4]>,
    guidance_embeds: Option<bool>,
    mlp_ratio: Option<f32>,
    num_attention_heads: Option<usize>,
    num_layers: Option<usize>,
    num_single_layers: Option<usize>,
    rope_theta: Option<f32>,
}

/// The transformer weight files discovery settled on, with the union of
/// their tensors already read off the headers.
struct WeightSelection {
    /// The file whose header carries the signature tensor.
    primary: PathBuf,
    /// The primary plus every same-directory file whose tensor names are
    /// disjoint from it (its shards). A same-directory file that shares a
    /// name is another export of the same model, not a shard, and is left
    /// out rather than silently overriding the winner's tensors.
    files: Vec<PathBuf>,
    /// Every tensor of `files`, keyed by name. Disjoint by construction:
    /// selection only admits files with no name in common with the set.
    tensors: HashMap<String, RawTensor>,
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
    // Inspect exactly the files the selection admits, and hand that same
    // set to the pipeline: a directory holding two exports of the model (a
    // BF16 file beside an FP8 one) must not mix their tensors into one
    // dtype scan, one depth count, or one weight store. A multi-shard
    // export may carry the signature tensor in any shard, so selection
    // reads every candidate's header and admits every shard whose names
    // are disjoint from the winner's.
    let selection = select_weight_files(&desc.model_path)?;
    let weights_path = selection.primary;
    let weight_files = selection.files;
    let tensor_map = selection.tensors;
    if tensor_map.is_empty() {
        return Err(ImageGenError::Checkpoint(format!(
            "no safetensors tensors found in {}",
            weights_path.display()
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

    // The transformer's own config document: required, because the head
    // split, the RoPE parameterization and the MLP expansion are not
    // derivable from tensor shapes, and read strictly, because a document
    // that exists but cannot be parsed would otherwise fall through to
    // guesses about the architecture it describes.
    let config = read_transformer_config(&desc.model_path, desc.assets_dir.as_deref())?;

    // Count double and single transformer blocks from tensor names, and
    // cross-check the config's counts against them: a config naming fewer
    // blocks than the checkpoint carries would silently truncate the model
    // (the extra blocks' weights never load), and one naming more would
    // fail mid-load on a missing tensor with a less honest message.
    let depth = reconcile_depth("double_blocks", config.num_layers, &tensor_map)?;
    let depth_single_blocks =
        reconcile_depth("single_blocks", config.num_single_layers, &tensor_map)?;

    // The four values below have no tensor-shaped counterpart at all: the
    // qkv width is `3 * hidden` for any head split, the RoPE table is built
    // from the config's axes, and the MLP width check needs the ratio it
    // was built with. A checkpoint whose config omits one is a checkpoint
    // this loader cannot describe, so it is refused by the value's name.
    let num_heads = config.num_attention_heads.ok_or_else(|| {
        ImageGenError::Checkpoint(
            "transformer config carries no num_attention_heads: the head split is not \
             derivable from tensor shapes (the qkv width is 3 * hidden for any split)"
                .to_string(),
        )
    })?;
    let axes_dim = config.axes_dims_rope.ok_or_else(|| {
        ImageGenError::Checkpoint(
            "transformer config carries no axes_dims_rope: the 4-axis RoPE widths cannot \
             be guessed without silently mis-rotating every head"
                .to_string(),
        )
    })?;
    let theta = config.rope_theta.ok_or_else(|| {
        ImageGenError::Checkpoint(
            "transformer config carries no rope_theta: the RoPE frequency table would be \
             built on a guessed base"
                .to_string(),
        )
    })?;
    let mlp_ratio = config.mlp_ratio.ok_or_else(|| {
        ImageGenError::Checkpoint(
            "transformer config carries no mlp_ratio: the feedforward width cannot be \
             checked against the block tensors without it"
                .to_string(),
        )
    })?;
    // `attention_head_dim` is the one config number the tensor shapes do not
    // independently confirm (the qkv width is `3 * hidden` either way), so
    // cross-check it against the head split the blocks will use and refuse a
    // mismatch loudly rather than run a transformer whose heads are the wrong
    // width under a `RoPE` table sized for another.
    if let Some(head_dim) = config.attention_head_dim
        && head_dim * num_heads != hidden_size
    {
        return Err(ImageGenError::Checkpoint(format!(
            "config attention_head_dim {head_dim} x {num_heads} heads = {} does not match hidden size {hidden_size}",
            head_dim * num_heads
        )));
    }

    // Detect FP8 precision by scanning the selected tensors for any
    // F8_E4M3 weight. The scan sees only the selected files, so an FP8
    // export beside the selected BF16 one cannot flip this flag.
    let is_fp8 = tensor_map
        .values()
        .any(|t| matches!(t.parsed_dtype("probe"), Ok(StDType::F8E4M3)));

    // A guidance-conditioned checkpoint projects the guidance scale through
    // `guidance_in.*` weights the transformer never loads. Either signal —
    // the config flag or the tensors themselves — routes to the refusal in
    // `DiT::load`, so such a checkpoint is refused by name instead of
    // running unconditioned.
    let use_guidance_embed = config.guidance_embeds.unwrap_or(false)
        || tensor_map.keys().any(|name| {
            name.strip_prefix("transformer.")
                .unwrap_or(name)
                .starts_with("guidance_in.")
        });

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
        use_guidance_embed,
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
        weight_files,
        text_encoder_path,
        vae_path,
        tokenizer_path,
        params,
    })
}

/// Reads the transformer's `config.json`: beside the checkpoint's own
/// directory (or beside the weights file, when the model path names one),
/// else beside `assets_dir` — the FP8 export ships no config of its own and
/// reads the BF16 tree's through `--image-assets-dir`.
///
/// A document that exists but cannot be read or parsed is a refusal, not a
/// miss to search past: the file describes the architecture, and silently
/// continuing on guesses about a half-read document is how a loader runs a
/// model it cannot describe. With no document reachable at all, the four
/// values no tensor carries (`num_attention_heads`, `axes_dims_rope`,
/// `rope_theta`, `mlp_ratio`) are named in the refusal.
fn read_transformer_config(
    model_path: &Path,
    assets_dir: Option<&Path>,
) -> Result<TransformerConfigFile, ImageGenError> {
    let home = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf()
    };
    let mut tried = vec![home.join("transformer").join("config.json")];
    if let Some(assets) = assets_dir {
        let in_assets = assets.join("transformer").join("config.json");
        if !tried.contains(&in_assets) {
            tried.push(in_assets);
        }
    }
    for path in &tried {
        if !path.exists() {
            continue;
        }
        let content = std::fs::read_to_string(path)
            .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", path.display())))?;
        return serde_json::from_str(&content)
            .map_err(|e| ImageGenError::Checkpoint(format!("parsing {}: {e}", path.display())));
    }
    Err(ImageGenError::Checkpoint(format!(
        "no transformer config.json reachable (searched {}); its num_attention_heads, \
         axes_dims_rope, rope_theta and mlp_ratio are not derivable from tensor shapes, so \
         the architecture cannot be read — point --image-assets-dir at a tree carrying \
         transformer/config.json",
        tried
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

/// Reads a component's own `config.json` document, refusing rather than
/// substituting defaults: every real checkpoint carries these files beside
/// its weights, and a missing or unparsable one means the loader would run
/// the component with a shape nobody declared. Derivable-from-tensors
/// fallbacks are a separate matter (pure inspection stays); a missing
/// *document* is an error.
pub(crate) fn read_component_config<T: serde::de::DeserializeOwned>(
    path: &Path,
    component: &str,
) -> Result<T, ImageGenError> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        ImageGenError::Checkpoint(format!(
            "{component} config {}: {e}; every {} checkpoint carries this document and no \
             default may stand in for it",
            path.display(),
            component
        ))
    })?;
    serde_json::from_str(&text)
        .map_err(|e| ImageGenError::Checkpoint(format!("parsing {}: {e}", path.display())))
}

/// One block family's depth: the config's count cross-checked against the
/// count the tensor names carry, with the pure-inspection count standing
/// when the config names none.
fn reconcile_depth(
    prefix: &str,
    config_depth: Option<usize>,
    tensor_map: &HashMap<String, RawTensor>,
) -> Result<usize, ImageGenError> {
    let counted =
        count_blocks_internal(tensor_map.keys().map(String::as_str), &format!("{prefix}."))
            .or_else(|| {
                count_blocks_internal(
                    tensor_map.keys().map(String::as_str),
                    &format!("transformer.{prefix}."),
                )
            });
    match (config_depth, counted) {
        (Some(config), Some(counted)) if config != counted => {
            Err(ImageGenError::Checkpoint(format!(
                "transformer config names {config} {prefix} blocks but the tensors carry \
                 {counted}: refusing rather than loading a truncated or half-described model"
            )))
        }
        (Some(config), _) => Ok(config),
        (None, Some(counted)) => Ok(counted),
        (None, None) => Err(ImageGenError::Checkpoint(format!(
            "no {prefix} blocks found in safetensors and the config names none"
        ))),
    }
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

/// The tensor that marks a store as a transformer this loader can read: the
/// upstream patchify input projection. A file without it carries a different
/// tensor layout (the alternative export naming), which the loader refuses.
const SIGNATURE_TENSOR: &str = "img_in.weight";

/// Picks the safetensors files holding the transformer weights by inspecting
/// headers, never names-by-convention: root-level files are tried before a
/// `transformer/` subfolder, and the primary is the first whose header
/// carries [`SIGNATURE_TENSOR`] (or its `transformer.`-prefixed spelling).
/// Every same-directory file whose tensor names are disjoint from the
/// primary's is one of its shards and joins the selection; a same-directory
/// file that shares a tensor name is another export of the same model — a
/// BF16 file beside an FP8 one — and is left out, because indexing both
/// would resolve each shared name to whichever file was read last and
/// silently load the wrong weights. A directory whose candidates all lack
/// the signature is refused by file name: a converted-folder export would
/// index every tensor under names the loader never asks for and either fail
/// mid-load or, worse, run a half-populated model, so guessing from the
/// directory shape is not an option.
fn select_weight_files(path: &Path) -> Result<WeightSelection, ImageGenError> {
    if path.is_file() {
        let (_shard, tensors) = Shard::open_untagged(path)
            .map_err(|e| ImageGenError::Checkpoint(format!("opening {}: {e}", path.display())))?;
        return Ok(WeightSelection {
            primary: path.to_path_buf(),
            files: vec![path.to_path_buf()],
            tensors: tensors.into_iter().collect(),
        });
    }
    let mut candidates = safetensors_in(path);
    candidates.extend(safetensors_in(&path.join("transformer")));
    if candidates.is_empty() {
        return Err(ImageGenError::Checkpoint(format!(
            "no safetensors weight file found in {}",
            path.display()
        )));
    }
    let mut opened = Vec::with_capacity(candidates.len());
    for file in &candidates {
        let (_shard, tensors) = Shard::open_untagged(file)
            .map_err(|e| ImageGenError::Checkpoint(format!("opening {}: {e}", file.display())))?;
        opened.push((file.clone(), tensors));
    }
    let winner = opened
        .iter()
        .position(|(_file, tensors)| {
            tensors.iter().any(|(name, _)| {
                name == SIGNATURE_TENSOR || name == &format!("transformer.{SIGNATURE_TENSOR}")
            })
        })
        .ok_or_else(|| {
            ImageGenError::Checkpoint(format!(
                "no safetensors file under {} carries the upstream transformer tensor \
                 '{SIGNATURE_TENSOR}'; found [{}], which is not the upstream tensor layout \
                 (the converted folder layout is not supported) — point the model path at \
                 the upstream-named single-file export instead",
                path.display(),
                candidates
                    .iter()
                    .map(|p| p
                        .file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into_owned()))
                    .collect::<Vec<_>>()
                    .join(", "),
            ))
        })?;
    let (primary, winner_tensors) = opened.swap_remove(winner);
    let primary_dir = primary
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut names: HashSet<String> = winner_tensors.iter().map(|(n, _)| n.clone()).collect();
    let mut tensors: HashMap<String, RawTensor> = winner_tensors.into_iter().collect();
    let mut files = vec![primary.clone()];
    for (file, shard_tensors) in opened {
        // Only same-directory files can be this export's shards; a
        // `transformer/` subfolder is a different layout, not a shard set.
        if file.parent().filter(|p| !p.as_os_str().is_empty()) != Some(primary_dir) {
            continue;
        }
        if shard_tensors
            .iter()
            .any(|(name, _)| names.contains(String::as_str(name)))
        {
            continue;
        }
        names.extend(shard_tensors.iter().map(|(n, _)| n.clone()));
        files.push(file);
        tensors.extend(shard_tensors);
    }
    files.sort();
    Ok(WeightSelection {
        primary,
        files,
        tensors,
    })
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
