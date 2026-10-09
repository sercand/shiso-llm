//! Opens an image generation checkpoint as whichever family claims it.
//!
//! The family is read off the checkpoint's own tensor names, never off a path
//! or a file name: [`detect_image_family`] opens headers only, so deciding
//! costs no weight bytes, and [`open_image_generator`] dispatches on its
//! answer. Adding a family adds a signature here and a directory beside this
//! file, nothing else.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_modelfile::GgufFile;
use shiso_modelfile::st_header::Shard;
use shiso_vision::generator::{ImageGenError, ImageGenerator};

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

/// The image families this build recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFamily {
    /// The rectified-flow double/single stream transformer family.
    Flux2,
    /// The Qwen-Image-2.1 single-stream-per-block `MMDiT` family.
    QwenImage21,
}

/// The tensor prefix a diffusion checkpoint may nest its transformer under.
const DIFFUSION_PREFIX: &str = "model.diffusion_model.";
/// The nesting a combined export puts the transformer under.
const TRANSFORMER_PREFIX: &str = "transformer.";

/// The signature the first family looks for, spelled for the refusal message.
const FLUX2_SIGNATURE: &str =
    "'img_in.weight' plus any 'double_blocks.*' or 'single_blocks.*' tensor";
/// The signature the second family looks for, spelled for the refusal message.
const QWEN_IMAGE21_SIGNATURE: &str = "'transformer_blocks.0.attn.to_q.weight' plus \
     'transformer_blocks.0.*modulation.1.weight' or '*_mod.1.weight' (optionally under 'model.diffusion_model.')";

/// Decides which family `desc.model_path` holds.
///
/// Reads safetensors headers only (a file, or every `*.safetensors` in a
/// directory and its `transformer/` subfolder) or a GGUF tensor directory.
/// Exactly one signature must match: a checkpoint that satisfies both is
/// refused rather than guessed at.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] if the headers cannot be read, no family's
/// signature matches (the message lists both), or more than one does.
pub fn detect_image_family(desc: &ImageModelDesc) -> Result<ImageFamily, ImageGenError> {
    let names = tensor_names(&desc.model_path)?;
    let flux2 = looks_like_flux2(&names);
    let qwen = looks_like_qwen_image21(&names);
    match (flux2, qwen) {
        (true, false) => Ok(ImageFamily::Flux2),
        (false, true) => Ok(ImageFamily::QwenImage21),
        (true, true) => Err(ImageGenError::Checkpoint(format!(
            "{} matches both the flux2 signature ({FLUX2_SIGNATURE}) and the qwen_image21 \
             signature ({QWEN_IMAGE21_SIGNATURE}); refusing to guess",
            desc.model_path.display()
        ))),
        (false, false) => Err(ImageGenError::Checkpoint(format!(
            "{} is not a known image checkpoint: looked for the flux2 signature \
             ({FLUX2_SIGNATURE}) and the qwen_image21 signature ({QWEN_IMAGE21_SIGNATURE})",
            desc.model_path.display()
        ))),
    }
}

/// Opens the image generator from `desc` on `backend`.
///
/// # Errors
/// Returns [`ImageGenError::Checkpoint`] if the checkpoint cannot be resolved or loaded.
pub fn open_image_generator(
    desc: &ImageModelDesc,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn ImageGenerator>, ImageGenError> {
    match detect_image_family(desc)? {
        ImageFamily::Flux2 => {
            let located = crate::flux2::config::locate_and_inspect(desc)?;
            let model = crate::flux2::pipeline::Flux2KleinModel::load(&located, backend)?;
            Ok(Arc::new(model))
        }
        ImageFamily::QwenImage21 => Err(ImageGenError::Checkpoint(
            "qwen_image21 family is not built yet".to_owned(),
        )),
    }
}

/// `name` or the same tensor under either nesting a checkpoint may use.
fn has(names: &HashSet<String>, name: &str) -> bool {
    names.contains(name)
        || names.contains(&format!("{DIFFUSION_PREFIX}{name}"))
        || names.contains(&format!("{TRANSFORMER_PREFIX}{name}"))
}

/// The same test the flux2 locator applies (the `img_in.weight` signature,
/// bare or under `transformer.`), plus a block tensor: the stem tensor alone
/// is not a family, since other transformers carry one.
fn looks_like_flux2(names: &HashSet<String>) -> bool {
    let stem = has(names, "img_in.weight");
    let blocks = names.iter().any(|name| {
        let bare = name.strip_prefix(TRANSFORMER_PREFIX).unwrap_or(name);
        bare.starts_with("double_blocks.") || bare.starts_with("single_blocks.")
    });
    stem && blocks
}

/// Block zero's query projection and a modulation linear, with or without
/// the `model.diffusion_model.` nesting. The brief's `modulation.1` spelling
/// and the diffusers `img_mod.1` / `txt_mod.1` one both count.
fn looks_like_qwen_image21(names: &HashSet<String>) -> bool {
    let bare: Vec<&str> = names
        .iter()
        .map(|n| n.strip_prefix(DIFFUSION_PREFIX).unwrap_or(n))
        .collect();
    bare.contains(&"transformer_blocks.0.attn.to_q.weight")
        && bare.iter().any(|n| {
            n.starts_with("transformer_blocks.0.")
                && (n.ends_with("modulation.1.weight") || n.ends_with("_mod.1.weight"))
        })
}

/// Every tensor name `path` declares, from headers alone.
fn tensor_names(path: &Path) -> Result<HashSet<String>, ImageGenError> {
    if path.is_file() {
        return names_of_file(path);
    }
    let mut files = safetensors_in(path);
    files.extend(safetensors_in(&path.join("transformer")));
    if files.is_empty() {
        return Err(ImageGenError::Checkpoint(format!(
            "no safetensors weight file found in {}",
            path.display()
        )));
    }
    let mut names = HashSet::new();
    for file in &files {
        names.extend(names_of_file(file)?);
    }
    Ok(names)
}

/// The tensor names one weight file declares.
fn names_of_file(path: &Path) -> Result<HashSet<String>, ImageGenError> {
    let open_err = |e: &dyn std::fmt::Display| {
        ImageGenError::Checkpoint(format!("opening {}: {e}", path.display()))
    };
    if path.extension().is_some_and(|ext| ext == "gguf") {
        let file = GgufFile::open(path).map_err(|e| open_err(&e))?;
        return Ok(file.tensors().iter().map(|t| t.name.to_string()).collect());
    }
    let (_shard, tensors) = Shard::open_untagged(path).map_err(|e| open_err(&e))?;
    Ok(tensors.into_iter().map(|(name, _)| name).collect())
}

/// Every `*.safetensors` file directly in `dir`, sorted.
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
