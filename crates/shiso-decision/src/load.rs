//! Opening a model directory: the device, the checkpoint and the vocabulary.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, CoreError, HfConfig, PromptEncoder, TokenId};
use shiso_engine::EngineOptions;
use shiso_models::lfm2::{Lfm2Config, Lfm2Model};

use crate::decider::OpenDesc;
use crate::error::DecideError;
use crate::literal::{Markers, encode_piece};
use crate::worker::{Kind, ModelEngine};

/// Opens the model in `desc` and wraps it in an engine.
///
/// The directory is checked before the device is opened, so a directory that
/// is no model fails the same way on every build.
pub(crate) fn open(desc: &OpenDesc<'_>) -> Result<ModelEngine, DecideError> {
    let files = Files::find(desc.model_dir)?;
    let backend = open_backend()?;
    let config = HfConfig::from_file(&files.config)
        .map_err(|error| load_error("reading config.json", &error))?;
    // The one family this loader opens. A decision checkpoint of another
    // family needs its own arm here.
    if !Lfm2Config::describes(&config) {
        return Err(DecideError::Load(format!(
            "the architecture {:?} is not one a decision model is loaded for",
            config.model_type()
        )));
    }
    let checkpoint = files.checkpoint()?;
    let family = Lfm2Config::from_hf(&config, &checkpoint)
        .map_err(|error| load_error("reading the checkpoint's geometry", &error))?;
    let mut opts = EngineOptions::default();
    opts.ctx = u32::try_from(desc.ctx)
        .map_err(|_| DecideError::Load(format!("a context of {} tokens is too long", desc.ctx)))?;
    // A loader that panics (an allocation or a driver call that cannot be
    // refused with an error) is a failed load, not a dead caller.
    let mut model = caught("loading the checkpoint", || {
        Lfm2Model::load(&backend, &checkpoint, &family, opts)
            .map_err(|error| load_error("loading the checkpoint", &error))
    })?;
    // Every weight is resident now, so whatever the backend kept to make the
    // imports fast can go.
    backend.finish_imports();
    let kind = {
        use shiso_core::Model as _;
        // A head answers when there is one, as on the server's routes.
        match (
            model.decision_mut().map(|head| head.layout()),
            model.decision_readout(),
        ) {
            (Some(layout), _) => Kind::Head(layout),
            (None, Some(readout)) => Kind::Readout(readout),
            (None, None) => {
                return Err(DecideError::Load("not a decision model".to_owned()));
            }
        }
    };
    let tokens = PromptTokens::from_file(&files.tokenizer)?;
    Ok(ModelEngine::new(
        Box::new(model),
        backend,
        tokens,
        kind,
        desc.ctx,
    ))
}

/// Runs `body`, turning a panic into [`DecideError::Load`] naming `what`.
pub(crate) fn caught<T>(
    what: &str,
    body: impl FnOnce() -> Result<T, DecideError>,
) -> Result<T, DecideError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(body))
        .unwrap_or_else(|_| Err(DecideError::Load(format!("{what}: the loader panicked"))))
}

fn load_error(what: &str, error: &CoreError) -> DecideError {
    DecideError::Load(format!("{what}: {}", error.chain()))
}

/// Where a model directory keeps what loading needs.
struct Files {
    config: PathBuf,
    tokenizer: PathBuf,
    weights_dir: PathBuf,
}

impl Files {
    /// Finds `config.json` and `tokenizer.json` in `dir` or its `config/`
    /// subdirectory, and the weights in `dir` or its `base/`.
    fn find(dir: &Path) -> Result<Self, DecideError> {
        let under = |sub: &str| {
            let nested = dir.join(sub);
            if nested.is_dir() {
                nested
            } else {
                dir.to_path_buf()
            }
        };
        let config_dir = under("config");
        let config = config_dir.join("config.json");
        if !config.is_file() {
            return Err(DecideError::Load(format!(
                "no config.json in {}",
                config_dir.display()
            )));
        }
        let tokenizer = config_dir.join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(DecideError::Load(format!(
                "no tokenizer.json in {}",
                config_dir.display()
            )));
        }
        Ok(Self {
            config,
            tokenizer,
            weights_dir: under("base"),
        })
    }

    /// The one `.gguf` checkpoint, leaving out vision sidecars.
    fn checkpoint(&self) -> Result<PathBuf, DecideError> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.weights_dir)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            let gguf = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
            if gguf && !name.starts_with("mmproj") {
                found.push(path);
            }
        }
        match found.as_slice() {
            [only] => Ok(only.clone()),
            [] => Err(DecideError::Load(format!(
                "no .gguf checkpoint in {}",
                self.weights_dir.display()
            ))),
            _ => Err(DecideError::Load(format!(
                "more than one .gguf checkpoint in {}",
                self.weights_dir.display()
            ))),
        }
    }
}

/// Opens the device this build was compiled for.
///
/// A backend that panics while it is created (a driver that is not there) is
/// an error to the caller, not a dead thread.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn open_backend() -> Result<Arc<dyn Backend>, DecideError> {
    guarded(|| {
        shiso_metal::MetalBackend::new()
            .map(|backend| -> Arc<dyn Backend> { backend })
            .map_err(|error| error.chain().to_string())
    })
}

/// Opens the device this build was compiled for, without decode graphs: a
/// decision pass is never replayed.
#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn open_backend() -> Result<Arc<dyn Backend>, DecideError> {
    guarded(|| {
        let options = shiso_cuda::CudaOptions::default().with_decode_graphs(false);
        shiso_cuda::CudaBackend::new(0, options)
            .map(|backend| -> Arc<dyn Backend> { backend })
            .map_err(|error| error.chain().to_string())
    })
}

/// This build has no device: a typed refusal at the one place one is needed.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn open_backend() -> Result<Arc<dyn Backend>, DecideError> {
    Err(DecideError::NoDevice(
        "this build has no compute backend; rebuild with the `metal` or `cuda` feature".to_owned(),
    ))
}

/// Runs `create`, turning its failure and its panic into [`DecideError::NoDevice`].
#[cfg(any(feature = "metal", feature = "cuda"))]
fn guarded(
    create: impl FnOnce() -> Result<Arc<dyn Backend>, String>,
) -> Result<Arc<dyn Backend>, DecideError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(create)) {
        Ok(Ok(backend)) => Ok(backend),
        Ok(Err(message)) => Err(DecideError::NoDevice(message)),
        Err(_) => Err(DecideError::NoDevice(
            "the backend panicked while it was opened".to_owned(),
        )),
    }
}

/// The checkpoint's vocabulary, as a decision layout reaches it.
#[derive(Debug)]
pub(crate) struct PromptTokens {
    tokenizer: tokenizers::Tokenizer,
    markers: Markers,
}

impl PromptTokens {
    /// Reads `tokenizer.json`, and which spellings in it are markers.
    pub(crate) fn from_file(path: &Path) -> Result<Self, DecideError> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| DecideError::Load(format!("reading {}: {error}", path.display())))?;
        Ok(Self {
            markers: Markers::of(&tokenizer),
            tokenizer,
        })
    }
}

impl PromptEncoder for PromptTokens {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        encode_piece(&self.tokenizer, &self.markers, text, literal).map_err(|source| {
            CoreError::Backend {
                backend: "tokenizer",
                op: "encode",
                source,
            }
        })
    }
}
