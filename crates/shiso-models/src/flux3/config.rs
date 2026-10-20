//! Locates a FLUX.3 checkpoint and reads its architecture from tensor headers.
//!
//! Nothing here reads a weight byte: the `DiT` file is opened through the
//! shared weight store (a mapping plus an index of names, shapes and stored
//! formats), and every number is taken from a tensor shape. A checkpoint
//! that is the Base (non-distilled) release, holds a quant format no kernel
//! reads, or carries a shape the kernels were not built for is refused,
//! naming the tensor, instead of loading into a model that would run wrong.

use std::path::{Path, PathBuf};

use shiso_modelfile::GgufFile;
use shiso_vision::generator::ImageGenError;

use crate::diffusion::weights::{StStore, safetensors_files};
use crate::image_registry::ImageModelDesc;

/// The per-head width every attention kernel of the family is built for.
const HEAD_DIM: usize = 128;
/// The width of the text conditioning: eight Qwen3-VL taps of 2560.
const TXT_IN: usize = 20_480;
/// The channels of one image token: the VAE's 32 latent channels, patchified 2x2.
const IMG_IN: usize = 128;
/// The GGUF `general.architecture` a FLUX.3 export declares.
const GGUF_ARCHITECTURE: &str = "flux";
/// Present only in the distilled (flow-map) release.
const REF_TIME_IN: &str = "ref_time_in.in_layer.weight";
/// The modes that own a block stack before the shared tail.
const CONTENT_MODES: [&str; 2] = ["image", "image_cond"];

/// Dimensions of the diffusion transformer, read from tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flux3Params {
    /// Hidden width of every block.
    pub hidden: usize,
    /// Attention heads (`hidden / 128`).
    pub heads: usize,
    /// Blocks in each per-mode stack (`image`, `image_cond` and `text` agree).
    pub depth_mode: usize,
    /// Blocks in the shared single-stream tail.
    pub depth_single: usize,
    /// Width of the MLP's hidden layer (half of `mlp_in`'s output).
    pub mlp_hidden: usize,
    /// Width of the text conditioning the `DiT` projects in.
    pub txt_in: usize,
    /// Channels of one image token.
    pub img_in: usize,
}

/// Resolved files and parameters for a FLUX.3 checkpoint.
#[derive(Debug, Clone)]
pub struct Flux3Located {
    /// Dimensions read from the `DiT`'s tensor shapes.
    pub params: Flux3Params,
    /// The `DiT` weight file (a GGUF, or one safetensors file).
    pub weight_file: PathBuf,
    /// The text encoder's weight file(s).
    pub text_encoder_files: Vec<PathBuf>,
    /// The `config.json` describing the text encoder.
    pub text_encoder_config: PathBuf,
    /// The VAE weights (`ae.safetensors`).
    pub vae_path: PathBuf,
    /// The tokenizer directory or `tokenizer.json`.
    pub tokenizer_path: PathBuf,
}

fn refuse(message: String) -> ImageGenError {
    ImageGenError::Checkpoint(message)
}

/// The tensors every FLUX.3 `DiT` carries once, with no block index.
const GLOBAL_TENSORS: [&str; 14] = [
    REF_TIME_IN,
    "ref_time_in.out_layer.weight",
    "time_in.in_layer.weight",
    "time_in.out_layer.weight",
    "txt_in.weight",
    "emb_in.image.weight",
    "emb_in.image_cond.weight",
    "early_stream_modulations.image.lin.weight",
    "early_stream_modulations.image_cond.lin.weight",
    "early_stream_modulations.txt.lin.weight",
    "single_stream_modulations.image.lin.weight",
    "single_stream_modulations.image_cond.lin.weight",
    "single_stream_modulations.txt.lin.weight",
    "final_layer.image.linear.weight",
];

/// What every block (per-mode or single-stream) carries, after its prefix.
const BLOCK_TENSORS: [&str; 8] = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "attn_out.weight",
    "mlp_in.weight",
    "mlp_out.weight",
    "norm.query_norm.scale",
    "norm.key_norm.scale",
];

/// The shape of `name`, or a refusal naming it.
fn shape_of(store: &StStore, name: &str) -> Result<Vec<usize>, ImageGenError> {
    let entry = store.require(name)?;
    entry
        .shape
        .iter()
        .map(|&d| {
            usize::try_from(d)
                .map_err(|_| refuse(format!("'{name}' has a dimension {d} too large")))
        })
        .collect()
}

/// Refuses `name` unless its shape is exactly `want`.
fn expect_shape(store: &StStore, name: &str, want: &[usize]) -> Result<(), ImageGenError> {
    let got = shape_of(store, name)?;
    if got == want {
        Ok(())
    } else {
        Err(refuse(format!(
            "'{name}' has shape {got:?}, but this FLUX.3 build needs {want:?}"
        )))
    }
}

/// Reads the architecture of the `DiT` in `store`, refusing what the kernels
/// were not built for.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] naming the missing or misshapen tensor.
pub(crate) fn inspect(store: &StStore) -> Result<Flux3Params, ImageGenError> {
    if !store.contains(REF_TIME_IN) {
        return Err(refuse(format!(
            "tensor '{REF_TIME_IN}' is missing: this is the Base (guidance, 50-step) release, \
             and only the distilled flow-map release is supported"
        )));
    }
    for name in GLOBAL_TENSORS {
        store.require(name)?;
    }
    let q0 = "content_mode_blocks.image.0.q_proj.weight";
    let q_shape = shape_of(store, q0)?;
    let hidden = match q_shape.as_slice() {
        [out, input] if out == input => *out,
        other => {
            return Err(refuse(format!(
                "'{q0}' has shape {other:?}, but the attention projections must be square"
            )));
        }
    };
    let norm0 = "content_mode_blocks.image.0.norm.query_norm.scale";
    let head_dim = match shape_of(store, norm0)?.as_slice() {
        [width] => *width,
        other => {
            return Err(refuse(format!(
                "'{norm0}' has shape {other:?}, expected [head_dim]"
            )));
        }
    };
    if head_dim != HEAD_DIM || !hidden.is_multiple_of(head_dim) {
        return Err(refuse(format!(
            "'{norm0}' gives head width {head_dim} and '{q0}' hidden {hidden}; the attention \
             kernels need {HEAD_DIM}-wide heads that divide the hidden width"
        )));
    }
    let heads = hidden / head_dim;

    let depth = |prefix: &str| {
        store
            .count_blocks(prefix)
            .ok_or_else(|| refuse(format!("no tensor named '{prefix}<n>.*' in the checkpoint")))
    };
    let depth_mode = depth("content_mode_blocks.image.")?;
    let depth_cond = depth("content_mode_blocks.image_cond.")?;
    let depth_text = depth("txt_mode_blocks.")?;
    if depth_cond != depth_mode || depth_text != depth_mode {
        return Err(refuse(format!(
            "the per-mode stacks disagree: image has {depth_mode} blocks, image_cond \
             {depth_cond}, text {depth_text}"
        )));
    }
    let depth_single = depth("single_blocks.")?;

    let mut stacks: Vec<(String, usize)> = CONTENT_MODES
        .iter()
        .map(|mode| (format!("content_mode_blocks.{mode}."), depth_mode))
        .collect();
    stacks.push(("txt_mode_blocks.".to_string(), depth_mode));
    stacks.push(("single_blocks.".to_string(), depth_single));
    let mlp_in0 = shape_of(store, "single_blocks.0.mlp_in.weight")?;
    let mlp_hidden = match mlp_in0.as_slice() {
        [out, input] if *input == hidden && out.is_multiple_of(2) => out / 2,
        other => {
            return Err(refuse(format!(
                "'single_blocks.0.mlp_in.weight' has shape {other:?}, expected [2 * mlp, {hidden}]"
            )));
        }
    };
    for (prefix, count) in &stacks {
        for index in 0..*count {
            for tensor in BLOCK_TENSORS {
                let name = format!("{prefix}{index}.{tensor}");
                let want: Vec<usize> = match tensor {
                    "mlp_in.weight" => vec![2 * mlp_hidden, hidden],
                    "mlp_out.weight" => vec![hidden, mlp_hidden],
                    t if t.starts_with("norm.") => vec![head_dim],
                    _ => vec![hidden, hidden],
                };
                expect_shape(store, &name, &want)?;
            }
        }
    }

    expect_shape(store, "txt_in.weight", &[hidden, TXT_IN]).map_err(|_| {
        refuse(format!(
            "'txt_in.weight' has shape {:?}, but the text conditioning is {TXT_IN} wide \
             (eight taps of the 2560-wide text encoder) into hidden {hidden}",
            shape_of(store, "txt_in.weight").unwrap_or_default()
        ))
    })?;
    for name in ["emb_in.image.weight", "emb_in.image_cond.weight"] {
        expect_shape(store, name, &[hidden, IMG_IN])?;
    }
    Ok(Flux3Params {
        hidden,
        heads,
        depth_mode,
        depth_single,
        mlp_hidden,
        txt_in: TXT_IN,
        img_in: IMG_IN,
    })
}

/// Refuses the options a FLUX.3 job has no meaning for.
///
/// Swap residency is supported (the pipeline drives it itself); the draft
/// stage and the last-step block reuse are not, and are refused rather than
/// ignored.
pub(crate) fn check_options(desc: &ImageModelDesc) -> Result<(), ImageGenError> {
    if desc.draft_steps != 0 {
        return Err(refuse(
            "--image-draft-steps is not implemented for the flux3 family, which runs every \
             step at the requested size"
                .to_owned(),
        ));
    }
    if desc.final_reuse.is_some() {
        return Err(refuse(
            "--image-final-reuse is not implemented for the flux3 family".to_owned(),
        ));
    }
    Ok(())
}

/// `explicit` if it exists, else the first of `names` beside the model or
/// under the assets directory.
fn resolve(
    explicit: Option<&Path>,
    assets_dir: Option<&Path>,
    home: &Path,
    names: &[&str],
    flag: &str,
) -> Result<PathBuf, ImageGenError> {
    if let Some(path) = explicit
        && path.exists()
    {
        return Ok(path.to_path_buf());
    }
    for base in std::iter::once(home).chain(assets_dir) {
        for name in names {
            let candidate = base.join(name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }
    Err(refuse(format!(
        "could not locate '{}'; pass {flag} or --image-assets-dir",
        names.join("' or '")
    )))
}

/// Resolves the text encoder: one `.safetensors` file with its
/// `config.json` beside it, or a directory of shards with `config.json`.
fn resolve_text_encoder(path: &Path) -> Result<(Vec<PathBuf>, PathBuf), ImageGenError> {
    let (files, config) = if path.is_dir() {
        (safetensors_files(path)?, path.join("config.json"))
    } else {
        (
            vec![path.to_path_buf()],
            path.parent().unwrap_or(Path::new(".")).join("config.json"),
        )
    };
    if files.is_empty() {
        return Err(refuse(format!(
            "no .safetensors text encoder weights in {}",
            path.display()
        )));
    }
    if !config.is_file() {
        return Err(refuse(format!(
            "the text encoder at {} needs its config.json at {}",
            path.display(),
            config.display()
        )));
    }
    Ok((files, config))
}

/// Locates a FLUX.3 checkpoint's components and reads the `DiT`'s dimensions.
///
/// Reads headers only. The model path is the `DiT`: a GGUF (whose
/// `general.architecture` must be `flux`) or one safetensors file.
///
/// # Errors
/// [`ImageGenError::Checkpoint`] naming what is wrong or missing.
pub fn locate_and_inspect(desc: &ImageModelDesc) -> Result<Flux3Located, ImageGenError> {
    check_options(desc)?;
    let weight_file = desc.model_path.clone();
    if !weight_file.is_file() {
        return Err(refuse(format!(
            "{} is not a file: the FLUX.3 transformer is one .gguf or .safetensors file",
            weight_file.display()
        )));
    }
    if weight_file.extension().is_some_and(|ext| ext == "gguf") {
        let arch = GgufFile::read_architecture(&weight_file)
            .map_err(|e| refuse(format!("reading {}: {e}", weight_file.display())))?;
        if arch.as_deref() != Some(GGUF_ARCHITECTURE) {
            return Err(refuse(format!(
                "{} declares general.architecture {:?}, but a FLUX.3 export declares '{GGUF_ARCHITECTURE}'",
                weight_file.display(),
                arch.as_deref().unwrap_or("<none>")
            )));
        }
    }
    let store = StStore::open(&weight_file)?;
    let params = inspect(&store)?;

    let home = weight_file.parent().unwrap_or(Path::new("."));
    let assets = desc.assets_dir.as_deref();
    let text_encoder = resolve(
        desc.text_encoder_path.as_deref(),
        assets,
        home,
        &["text_encoder"],
        "--image-text-encoder",
    )?;
    let (text_encoder_files, text_encoder_config) = resolve_text_encoder(&text_encoder)?;
    let vae_path = resolve(
        desc.vae_path.as_deref(),
        assets,
        home,
        &["ae.safetensors", "vae"],
        "--image-vae",
    )?;
    let tokenizer_path = resolve(
        desc.tokenizer_path.as_deref(),
        assets,
        home,
        &["tokenizer", "tokenizer.json"],
        "--image-tokenizer",
    )?;
    Ok(Flux3Located {
        params,
        weight_file,
        text_encoder_files,
        text_encoder_config,
        vae_path,
        tokenizer_path,
    })
}
