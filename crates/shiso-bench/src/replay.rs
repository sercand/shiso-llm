//! Running a fixture case against a real checkpoint.
//!
//! One type, [`Harness`], owns a device, a loaded model and the host-side
//! readback buffer, and exposes the three things a parity run needs: the logits
//! row at a frontier, a greedy continuation from one, and the same row reached
//! by a different route.
//!
//! # Why the chunk size is a parameter
//!
//! The last thing this harness can do that a golden file cannot is compare the
//! engine against *itself* along two different paths. A prompt fed in one call
//! and the same prompt fed in chunks visit different kernels with different
//! launch geometry and land in the same state, so their last rows have to agree.
//! That invariant needs no reference implementation and no captured fixture,
//! which is why it is the part of parity that a machine with only one backend
//! can still settle.
//!
//! # No timing lives here
//!
//! Not an omission. A parity run records what a build answers, and a throughput
//! number recorded beside it would invite a comparison the run is not
//! controlled for.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Model, ModelInfo};
use shiso_engine::sparse::capture;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::sparse::tap::ResidualTap;
use shiso_engine::{EngineOptions, argmax};
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};
use shiso_models::qwen4exp::{Qwen4ExpConfig, Qwen4ExpModel};
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};

use crate::compare::decode_row;

/// Which backend a run is executed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// The Apple GPU backend.
    Metal,
    /// The NVIDIA backend.
    Cuda,
}

impl Device {
    /// Parses the spelling a command line uses.
    ///
    /// # Errors
    ///
    /// [`ReplayError::UnknownDevice`] for anything else.
    pub fn parse(text: &str) -> Result<Self, ReplayError> {
        match text {
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            other => Err(ReplayError::UnknownDevice {
                found: other.to_owned(),
            }),
        }
    }

    /// The word a record files this device under.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

/// What can go wrong setting up or running a replay.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// The backend name was not one this runner knows.
    #[error("unknown backend {found:?}, expected metal or cuda")]
    UnknownDevice {
        /// What was asked for.
        found: String,
    },
    /// The requested device is not present on this machine.
    ///
    /// A distinct variant rather than a generic backend failure: a run that
    /// could not open a device produced no evidence at all, and a caller that
    /// treated that as a comparison result would be recording a vacuous pass.
    #[error("no {device} device on this machine: {detail}")]
    NoDevice {
        /// The device that was asked for.
        device: &'static str,
        /// What the backend said.
        detail: String,
    },
    /// A file the run needs is missing or unreadable.
    #[error("reading {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: std::io::Error,
    },
    /// The vocabulary could not be loaded or could not encode a prompt.
    #[error("the vocabulary at {path}: {detail}")]
    Vocabulary {
        /// The vocabulary file.
        path: PathBuf,
        /// What it said.
        detail: String,
    },
    /// A case asks for more prompt than it tokenizes to.
    #[error("case {case}: the prompt is {available} tokens, short of the {frontier} it declares")]
    ShortPrompt {
        /// The case.
        case: String,
        /// Tokens the prompt actually produced.
        available: usize,
        /// Tokens the case declares.
        frontier: u32,
    },
    /// A per-layer dump was asked of a checkpoint whose stack has no tap.
    ///
    /// Only the sparse stack has one. The dense families' diagnostic is
    /// `SHISO_LAYER_DUMP`, which prints statistics rather than rows.
    #[error("a per-layer residual dump needs a checkpoint of the sparse family")]
    NoLayerTap,
    /// The engine refused something.
    #[error(transparent)]
    Engine(#[from] shiso_core::CoreError),
}

/// A replay result.
pub type ReplayResult<T> = Result<T, ReplayError>;

/// An open device with one loaded checkpoint.
///
/// The model is a trait object because this runner serves both families now: a
/// recorded number still describes one checkpoint, and which family that
/// checkpoint belongs to is answered the same way the server answers it - by
/// asking each family's reader in turn and taking the one that recognises the
/// document.
#[derive(Debug)]
pub struct Harness {
    backend: Arc<dyn Backend>,
    model: Loaded,
    info: ModelInfo,
    row: Vec<u8>,
}

impl Harness {
    /// Opens `device` and loads the checkpoint at `model_path`.
    ///
    /// `ctx` is the session ceiling every case in this run is replayed at, so a
    /// fixture whose cases disagree about context needs one harness per value
    /// rather than a resize part-way through.
    ///
    /// # Errors
    ///
    /// [`ReplayError::NoDevice`] when the backend will not open,
    /// [`ReplayError::Io`] when the configuration cannot be read, and
    /// [`ReplayError::Engine`] when the checkpoint will not load.
    pub fn open(
        device: Device,
        model_path: &Path,
        config_path: &Path,
        ctx: u32,
    ) -> ReplayResult<Self> {
        let backend = open_backend(device)?;
        // Read here rather than handing the path to the parser: the parser's own
        // error would name the failure but not the file, and a run pointed at
        // the wrong configuration is the likeliest way to get here.
        let text = std::fs::read_to_string(config_path).map_err(|source| ReplayError::Io {
            path: config_path.to_path_buf(),
            source,
        })?;
        let config = HfConfig::from_json_str(&text)?;
        let mut opts = EngineOptions::new();
        opts.ctx = ctx;
        let mut model = load_loaded(&backend, model_path, &config, opts, None)?;
        let info = *model.get().info();
        let row = vec![0u8; usize::try_from(info.logits_row_bytes()).unwrap_or(0)];
        Ok(Self {
            backend,
            model,
            info,
            row,
        })
    }

    /// What the loaded checkpoint reports about itself.
    #[must_use]
    pub const fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// The logits row after `tokens`, fed in chunks of at most `chunk`.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Engine`] when the session or the forward pass fails.
    pub fn frontier_row(&mut self, tokens: &[TokenId], chunk: usize) -> ReplayResult<Vec<f32>> {
        let session = self.model.get().open_session()?;
        let result = self.frontier_row_in(session, tokens, chunk);
        let closed = self.model.get().close_session(session);
        let row = result?;
        closed?;
        Ok(row)
    }

    /// The body of [`Harness::frontier_row`], so the session is closed on either
    /// path.
    fn frontier_row_in(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        chunk: usize,
    ) -> ReplayResult<Vec<f32>> {
        let chunk = chunk.max(1);
        let split = tokens.len().saturating_sub(1);
        let (head, tail) = tokens.split_at(split.min(tokens.len()));
        for piece in head.chunks(chunk) {
            self.model.get().prefill_chunk(session, piece)?;
        }
        let last = self.model.get().prefill_last(session, tail)?;
        self.backend.copy_d2h(last.row(0)?, &mut self.row)?;
        Ok(decode_row(&self.row))
    }

    /// `steps` greedy tokens continuing `tokens`.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Engine`] when the session or the forward pass fails.
    pub fn greedy_after(
        &mut self,
        tokens: &[TokenId],
        chunk: usize,
        steps: usize,
    ) -> ReplayResult<Vec<TokenId>> {
        let session = self.model.get().open_session()?;
        let result = self.greedy_after_in(session, tokens, chunk, steps);
        let closed = self.model.get().close_session(session);
        let produced = result?;
        closed?;
        Ok(produced)
    }

    /// The body of [`Harness::greedy_after`].
    fn greedy_after_in(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        chunk: usize,
        steps: usize,
    ) -> ReplayResult<Vec<TokenId>> {
        let chunk = chunk.max(1);
        let split = tokens.len().saturating_sub(1);
        let (head, tail) = tokens.split_at(split.min(tokens.len()));
        for piece in head.chunks(chunk) {
            self.model.get().prefill_chunk(session, piece)?;
        }
        let last = self.model.get().prefill_last(session, tail)?;
        self.backend.copy_d2h(last.row(0)?, &mut self.row)?;

        let mut produced = Vec::with_capacity(steps);
        let mut token = argmax(&decode_row(&self.row));
        for _ in 0..steps {
            produced.push(token);
            let out = self
                .model
                .get()
                .decode_batch(&[DecodeStep { session, token }])?;
            self.backend.copy_d2h(out.row(0)?, &mut self.row)?;
            token = argmax(&decode_row(&self.row));
        }
        Ok(produced)
    }

    /// Releases the device resources, reporting a failure rather than dropping
    /// it.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Engine`] when a device resource could not be released.
    pub fn close(mut self) -> ReplayResult<()> {
        self.model.get().teardown()?;
        Ok(())
    }

    /// Layers of the loaded stack, when it is the sparse family's: the one
    /// stack a per-layer dump can be taken of.
    #[must_use]
    pub fn sparse_layers(&self) -> Option<u32> {
        match &self.model {
            Loaded::Sparse(model) => Some(model.stack().shape().n_layers()),
            Loaded::Other(_) => None,
        }
    }

    /// The logits row after `tokens`, and the residual tap over the pass that
    /// produced it: one residual row per layer boundary at the frontier.
    ///
    /// # Errors
    ///
    /// [`ReplayError::NoLayerTap`] when the checkpoint is not of the sparse
    /// family, and whatever [`crate::layers::frontier_layers`] reports.
    pub fn frontier_layers(
        &mut self,
        tokens: &[TokenId],
        chunk: usize,
    ) -> ReplayResult<(Vec<f32>, ResidualTap)> {
        let Loaded::Sparse(model) = &mut self.model else {
            return Err(ReplayError::NoLayerTap);
        };
        crate::layers::frontier_layers(self.backend.as_ref(), model, tokens, chunk)
    }
}

/// Loads `path` as whichever family `config` describes.
///
/// The same ordered list of readers the server's factory uses, and for the same
/// reason: `UnknownArch` is the one failure that means "not mine", and every
/// other failure from a reader means the family matched and the document is
/// wrong.
///
/// `ple` is `None` for every family but Gemma 4, which is the only one that has
/// a per-layer table, and `None` for that one means the holding an unconfigured
/// server would use. A caller that wants the other holding - the whole table in
/// memory rather than streamed off the checkpoint - says so here rather than
/// opening the family itself, because a second call to `Gemma4Model::load`
/// outside this function would be a second answer to "which family is this
/// checkpoint", and the two would eventually disagree.
pub(crate) fn load_model_with(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    opts: EngineOptions,
    ple: Option<PleOptions>,
) -> ReplayResult<Box<dyn Model>> {
    Ok(match load_loaded(backend, path, config, opts, ple)? {
        Loaded::Sparse(model) => model,
        Loaded::Other(model) => model,
    })
}

/// A loaded checkpoint, keeping the sparse family's concrete type.
///
/// The residual tap is armed on the sparse stack itself, which the `Model`
/// contract does not expose; keeping the type here is what lets the harness
/// reach it without widening that contract for a diagnostic.
#[derive(Debug)]
enum Loaded {
    /// The sparse family.
    Sparse(Box<Qwen4ExpModel>),
    /// Every other family.
    Other(Box<dyn Model>),
}

impl Loaded {
    /// The model, whichever family it is.
    fn get(&mut self) -> &mut dyn Model {
        match self {
            Self::Sparse(model) => model.as_mut(),
            Self::Other(model) => model.as_mut(),
        }
    }
}

/// The body of [`load_model_with`].
fn load_loaded(
    backend: &Arc<dyn Backend>,
    path: &Path,
    config: &HfConfig,
    opts: EngineOptions,
    ple: Option<PleOptions>,
) -> ReplayResult<Loaded> {
    match Qwen35Config::from_hf(config) {
        Ok(family) => {
            return Ok(Loaded::Other(Box::new(Qwen35Model::load(
                backend, path, &family, opts,
            )?)));
        }
        Err(shiso_core::CoreError::UnknownArch { .. }) => {}
        Err(other) => return Err(other.into()),
    }
    match Qwen4ExpConfig::from_hf(config, path) {
        Ok(family) => {
            // The two device facts this family's stack cannot default: the
            // template entry names a build captured off its own modules, and the
            // ceiling every cooperative grid is clamped to. Both are measured,
            // and both report their own absence rather than a guess.
            let entries = capture::capture(backend.as_ref());
            let opts = SparseOptions {
                engine: opts,
                spec: SpecDepth::NONE,
                entries,
                co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
                ..SparseOptions::default()
            };
            return Ok(Loaded::Sparse(Box::new(Qwen4ExpModel::load(
                backend, path, &family, opts,
            )?)));
        }
        Err(shiso_core::CoreError::UnknownArch { .. }) => {}
        Err(other) => return Err(other.into()),
    }
    // A replay measures the decode path, not the loader's knobs, so an
    // unasked-for per-layer table is held the way an unconfigured server holds
    // it.
    match Gemma4Config::from_hf(config, path) {
        Ok(family) => Ok(Loaded::Other(Box::new(Gemma4Model::load(
            backend,
            path,
            &family,
            opts,
            ple.unwrap_or_default(),
        )?))),
        Err(other) => Err(other.into()),
    }
}

/// Opens the named backend, or says why it could not.
///
/// Public because a parity case that loads a family directly - rather than
/// through [`Harness`], which holds the loader's knobs the way an unconfigured
/// server holds them - still has to open the same device the rest of the lane
/// opened, and naming a backend type at the call site is what made
/// `parity_gemma4.rs` a Metal-only file.
///
/// # Errors
///
/// [`ReplayError::NoDevice`] when the platform has no such device, or when the
/// driver is present and the backend will not open.
pub fn open_backend(device: Device) -> ReplayResult<Arc<dyn Backend>> {
    match device {
        Device::Metal => {
            if !shiso_metal::metal_available() {
                return Err(ReplayError::NoDevice {
                    device: "metal",
                    detail: "the platform reports no Metal device".to_owned(),
                });
            }
            let backend = shiso_metal::MetalBackend::new().map_err(|e| ReplayError::NoDevice {
                device: "metal",
                detail: e.to_string(),
            })?;
            Ok(backend)
        }
        Device::Cuda => {
            if !shiso_cuda::cuda_available() {
                return Err(ReplayError::NoDevice {
                    device: "cuda",
                    detail: "no driver on this machine".to_owned(),
                });
            }
            let backend = shiso_cuda::CudaBackend::new(0, shiso_cuda::CudaOptions::default())
                .map_err(|e| ReplayError::NoDevice {
                    device: "cuda",
                    detail: e.to_string(),
                })?;
            Ok(backend)
        }
    }
}

/// A loaded vocabulary, and the two ways a case asks it to encode a prompt.
#[derive(Debug)]
pub struct Vocabulary {
    inner: tokenizers::Tokenizer,
    path: PathBuf,
}

impl Vocabulary {
    /// Loads `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Vocabulary`] when the file will not load.
    pub fn load(path: &Path) -> ReplayResult<Self> {
        let inner =
            tokenizers::Tokenizer::from_file(path).map_err(|e| ReplayError::Vocabulary {
                path: path.to_path_buf(),
                detail: e.to_string(),
            })?;
        Ok(Self {
            inner,
            path: path.to_path_buf(),
        })
    }

    /// Encodes `text`, adding the vocabulary's control tokens when `special`.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Vocabulary`] when the text will not encode.
    pub fn encode(&self, text: &str, special: bool) -> ReplayResult<Vec<TokenId>> {
        let encoded = self
            .inner
            .encode(text, special)
            .map_err(|e| ReplayError::Vocabulary {
                path: self.path.clone(),
                detail: e.to_string(),
            })?;
        Ok(encoded
            .get_ids()
            .iter()
            .copied()
            .map(TokenId::new)
            .collect())
    }

    /// Renders token ids back to text, for the human check a capture needs.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Vocabulary`] when the ids will not decode.
    pub fn decode(&self, ids: &[TokenId]) -> ReplayResult<String> {
        let ids: Vec<u32> = ids.iter().map(|t| t.get()).collect();
        self.inner
            .decode(&ids, true)
            .map_err(|e| ReplayError::Vocabulary {
                path: self.path.clone(),
                detail: e.to_string(),
            })
    }

    /// A digest of the vocabulary file, recorded in a fixture so a replay can
    /// tell that it is encoding the prompt the same way the capture did.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Io`] when the file cannot be read.
    pub fn digest(&self) -> ReplayResult<String> {
        let bytes = std::fs::read(&self.path).map_err(|source| ReplayError::Io {
            path: self.path.clone(),
            source,
        })?;
        Ok(fnv_digest(&bytes))
    }
}

/// A 64-bit content digest, rendered as hex.
///
/// Deliberately not a cryptographic hash: nothing here is defending against a
/// forged fixture, and the question being asked is only "is this the same file
/// the capture used". A dependency for that would be a dependency for nothing.
#[must_use]
pub fn fnv_digest(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}
