//! Opening a model directory: the device, the checkpoint and the vocabulary.

use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, CoreError, HfConfig, PromptEncoder, TokenId};
use shiso_engine::EngineOptions;
use shiso_models::lfm2::{Lfm2Config, Lfm2Model};

use crate::decider::OpenDesc;
use crate::error::DecideError;
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
    let mut model = Lfm2Model::load(&backend, &checkpoint, &family, opts)
        .map_err(|error| load_error("loading the checkpoint", &error))?;
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
pub(crate) struct PromptTokens(tokenizers::Tokenizer);

impl PromptTokens {
    /// Reads `tokenizer.json`.
    pub(crate) fn from_file(path: &Path) -> Result<Self, DecideError> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self)
            .map_err(|error| DecideError::Load(format!("reading {}: {error}", path.display())))
    }

    /// Encodes `text` with its added tokens read as markers.
    fn plain(&self, text: &str) -> Result<Vec<TokenId>, CoreError> {
        Ok(self
            .0
            .encode(text, false)
            .map_err(failed)?
            .get_ids()
            .iter()
            .copied()
            .map(TokenId::new)
            .collect())
    }

    /// Encodes `text` with the byte ranges in `literal` read as ordinary text
    /// even where they spell an added token.
    ///
    /// Those ranges are text a caller wrote, and a caller must not be able to
    /// plant a marker in the prompt. The text is encoded once as usual; each
    /// stretch between two added tokens the layout wrote that holds a literal
    /// range is encoded again with added tokens read as text. Added tokens are
    /// split out before any other step of the encoder, so re-encoding such a
    /// stretch alone gives the ids the same characters would get anywhere.
    fn with_literals(
        &self,
        text: &str,
        literal: &[Range<usize>],
    ) -> Result<Vec<TokenId>, CoreError> {
        let encoded = self.0.encode(text, false).map_err(failed)?;
        let added = self.0.get_added_tokens_decoder();
        // Cloned per call rather than kept: only a prompt that spells a marker
        // in caller text comes here, and a second resident copy of a large
        // vocabulary is paid by every prompt that does not.
        let mut as_text = self.0.clone();
        as_text.set_encode_special_tokens(true);
        let overlaps = |(start, end): (usize, usize)| {
            literal
                .iter()
                .any(|range| start < range.end && range.start < end)
        };
        let flush = |tokens: &mut Vec<TokenId>,
                     stretch: &mut Vec<TokenId>,
                     span: Range<usize>,
                     holds_literal: bool|
         -> Result<(), CoreError> {
            if holds_literal {
                let piece = text.get(span).unwrap_or_default();
                let again = as_text.encode(piece, false).map_err(failed)?;
                tokens.extend(again.get_ids().iter().copied().map(TokenId::new));
                stretch.clear();
            } else {
                tokens.append(stretch);
            }
            Ok(())
        };

        let mut tokens = Vec::with_capacity(encoded.get_ids().len());
        let mut stretch: Vec<TokenId> = Vec::new();
        let mut stretch_start = 0usize;
        let mut holds_literal = false;
        for (&id, &offsets) in encoded.get_ids().iter().zip(encoded.get_offsets()) {
            let inside = overlaps(offsets);
            if added.contains_key(&id) && !inside {
                flush(
                    &mut tokens,
                    &mut stretch,
                    stretch_start..offsets.0,
                    holds_literal,
                )?;
                tokens.push(TokenId::new(id));
                stretch_start = offsets.1;
                holds_literal = false;
            } else {
                stretch.push(TokenId::new(id));
                holds_literal |= inside;
            }
        }
        flush(
            &mut tokens,
            &mut stretch,
            stretch_start..text.len(),
            holds_literal,
        )?;
        Ok(tokens)
    }
}

fn failed(error: tokenizers::Error) -> CoreError {
    CoreError::Backend {
        backend: "tokenizer",
        op: "encode",
        source: error,
    }
}

impl PromptEncoder for PromptTokens {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        // Every marker this vocabulary has opens with `<`, so a piece whose
        // caller text holds none takes the plain encode: the literal-aware one
        // clones the tokenizer on every call, and a schema is dozens of pieces.
        let marked = literal
            .iter()
            .any(|range| text.get(range.clone()).is_some_and(|t| t.contains('<')));
        if marked {
            self.with_literals(text, literal)
        } else {
            self.plain(text)
        }
    }
}
