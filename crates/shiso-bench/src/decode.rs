//! The device fixture both timed phases run against, and the decode phase.
//!
//! # One fixture, two phases
//!
//! Loading a checkpoint, opening a device, sizing the readback buffers and
//! deciding what the prompt is are the same work for prefill as for decode, and
//! a second copy of any of them would be a second definition of what was
//! measured. So [`Rig`] owns all of it and both phases borrow it.
//!
//! # What a decode step costs, and what is timed
//!
//! A step is one token in, one logits row out, read back to the host. The
//! readback is inside the timed window on purpose: without it the clock would
//! stop when the work was *submitted* rather than when it was done, and a
//! backend that queued more deeply would look faster for it.
//!
//! The sampler is greedy and the end-of-sequence id is excluded, so the run
//! always takes the full number of steps. A run that stopped early would be
//! timing a different amount of work each time it was repeated, and the
//! comparison between two of them would be meaningless.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::error::CoreError;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{EngineOptions, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::gemma4::PleOptions;
use shiso_models::qwen35::{Qwen35Config, Qwen35Model};
use shiso_models::qwen35moe::{Qwen35MoeConfig, Qwen35MoeModel};

use crate::measure::{Protocol, after_warmup, median};

/// Prompt length, in tokens.
///
/// The same length the first recorded baseline used, so a new number can be
/// read against the old one. A different length would measure a different
/// thing, and the point of re-recording is to be able to say what moved.
pub const PREFILL_TOKENS: u32 = 64;

/// Context a measured session is opened with.
pub const CTX: u32 = 4096;

/// What can stop a measured run before it has a number.
#[derive(Debug, thiserror::Error)]
pub enum RigError {
    /// No checkpoint at the path the run was pointed at.
    #[error("no checkpoint at {path}")]
    NoCheckpoint {
        /// The path that was tried.
        path: PathBuf,
    },
    /// The device this target needs is not on this machine.
    #[error("no {backend} device on this machine")]
    NoDevice {
        /// The backend that was asked for.
        backend: &'static str,
    },
    /// The shape profile beside the test vectors would not load.
    #[error("reading the shape profile at {path}")]
    Profile {
        /// The config that would not load.
        path: PathBuf,
        /// What the loader said.
        #[source]
        source: CoreError,
    },
    /// The device or the model refused.
    #[error(transparent)]
    Device(#[from] CoreError),
    /// The machine has the device, but this build cannot drive it well enough
    /// to produce a number.
    #[error("{target} cannot be recorded from this build: {why}")]
    Unrecordable {
        /// The target that was asked for.
        target: &'static str,
        /// What is missing.
        why: &'static str,
    },
    /// The checkpoint would not load as any family this runner carries.
    #[error("the checkpoint at {path} would not load: {detail}")]
    Load {
        /// The checkpoint that was tried.
        path: PathBuf,
        /// What the loader said.
        detail: String,
    },
    /// A phase produced no timed sample, so it has no statistic.
    #[error("the {phase} phase produced no timed sample")]
    NoSamples {
        /// Which phase came back empty.
        phase: &'static str,
    },
}

/// A loaded checkpoint on an open device, with the buffers a phase needs.
#[derive(Debug)]
pub struct Rig {
    /// The device the numbers describe.
    pub(crate) backend: Arc<dyn Backend>,
    /// The checkpoint under measurement.
    ///
    /// Boxed rather than named, because the two phases below use nothing but
    /// the [`Model`] contract and the second family this workspace carries is
    /// the one the NVIDIA target measures. Which family a checkpoint is comes
    /// from its own config document, as it does everywhere else here.
    pub(crate) model: Box<dyn Model>,
    /// Bytes one logits row reads back into.
    pub(crate) row: Vec<u8>,
    /// The same row as floats, for the sampler.
    pub(crate) scratch: Vec<f32>,
    /// The fixed prompt every phase runs.
    pub(crate) prompt: Vec<TokenId>,
    /// The checkpoint file's own name, for the record.
    name: String,
}

impl Rig {
    /// Loads `path` onto this platform's local device.
    ///
    /// The backend is chosen by `cfg`, not named: a bench lane that binds one
    /// backend does not fail on a machine with the other, it *skips*, and a
    /// skip and a pass print the same colour. This lane was the fourth on this
    /// branch found doing that, after `parity.rs`'s device tier,
    /// `parity_gemma4.rs` and both vision-tower cases.
    ///
    /// The recorded profile stays: [`Rig::open_with_config`] exists for the
    /// checkpoint's own document, and a number taken through one is not
    /// comparable to a number taken through the other.
    ///
    /// # Errors
    ///
    /// [`RigError::NoCheckpoint`] when the file is absent,
    /// [`RigError::NoDevice`] when this platform's backend has no device, and
    /// [`RigError::Device`] when the load itself fails.
    pub fn open(path: &Path) -> Result<Self, RigError> {
        if !path.is_file() {
            return Err(RigError::NoCheckpoint {
                path: path.to_path_buf(),
            });
        }
        let config = profile()?;
        let backend = platform_backend()?;
        let mut opts = EngineOptions::default();
        opts.ctx = CTX;
        let loading = Instant::now();
        let model = Qwen35Model::load(&backend, path, &config, opts)?;
        tracing::info!(
            seconds = loading.elapsed().as_secs_f64(),
            "checkpoint resident"
        );

        let vocab = usize::try_from(model.info().vocab).unwrap_or(0);
        Ok(Self {
            backend,
            row: vec![0u8; vocab.saturating_mul(4)],
            scratch: vec![0f32; vocab],
            prompt: prompt(&config),
            name: file_name(path),
            model: Box::new(model),
        })
    }

    /// Loads `path` onto the local Metal device, as whichever family its own
    /// config document describes, holding a per-layer table the way an
    /// unconfigured server holds it.
    ///
    /// This is [`Rig::open`] generalised: the same device, the checkpoint's own
    /// config instead of the recorded profile, and the family dispatcher instead
    /// of one named reader. [`Rig::open`] is not written in terms of it, because
    /// the numbers it produces are graded against baselines taken through the
    /// recorded profile and a number is comparable only to one taken the same
    /// way.
    ///
    /// # Errors
    ///
    /// [`RigError::NoCheckpoint`] when either file is absent,
    /// [`RigError::NoDevice`] when the machine has no Metal device, and
    /// [`RigError::Load`] when the checkpoint will not load.
    pub fn open_with_config(path: &Path, config: &Path) -> Result<Self, RigError> {
        Self::open_metal(path, config, None)
    }

    /// The same opener, with the per-layer table's holding chosen by the caller.
    ///
    /// The holding is a property of the run rather than of the checkpoint, and
    /// it moves the number: a streamed table pays a read for a row that has aged
    /// out of the cache, and a resident one pays for the whole table at load and
    /// never again. Which one a record describes has to be a decision somebody
    /// made, not a default nobody saw.
    ///
    /// # Errors
    ///
    /// As [`Rig::open_with_config`].
    pub fn open_with_ple(path: &Path, config: &Path, ple: PleOptions) -> Result<Self, RigError> {
        Self::open_metal(path, config, Some(ple))
    }

    /// Loads `path` onto the local NVIDIA device, as whichever family its own
    /// config document describes.
    ///
    /// `config` is that document. The Metal arm above reads a *recorded*
    /// profile instead, because its baselines were taken against that file and
    /// a number is only comparable to one taken the same way; this arm has no
    /// such history and reads the checkpoint's own.
    ///
    /// # Errors
    ///
    /// [`RigError::NoCheckpoint`] when either file is absent,
    /// [`RigError::NoDevice`] when the machine has no NVIDIA device, and
    /// [`RigError::Load`] when the checkpoint will not load.
    pub fn open_cuda(path: &Path, config: &Path) -> Result<Self, RigError> {
        present(path, config)?;
        let backend = crate::replay::open_backend(crate::replay::Device::Cuda)
            .map_err(|_| RigError::NoDevice { backend: "CUDA" })?;
        Self::dispatch(backend, path, config, None)
    }

    /// Opens the Qwen 3.5 `MoE` checkpoint from its snapshot directory.
    ///
    /// A directory rather than a file: this family ships three shards and two
    /// config documents, `config.json` for the geometry and
    /// `hf_quant_config.json` for the codec map, and `Qwen35MoeConfig::from_dir`
    /// is what reads both. The directory is handed to this rig's own dispatcher as the
    /// checkpoint path, which is what that arm reads it as.
    ///
    /// # Errors
    ///
    /// [`RigError::NoCheckpoint`] when the directory or its config document is
    /// absent, [`RigError::NoDevice`] when this platform has no device, and
    /// [`RigError::Load`] when the shards refuse.
    pub fn open_qwen35moe(dir: &Path) -> Result<Self, RigError> {
        let config = dir.join("config.json");
        present(dir, &config)?;
        let backend = platform_backend()?;
        Self::dispatch(backend, dir, &config, None)
    }

    /// The Metal half of the generalised openers.
    fn open_metal(path: &Path, config: &Path, ple: Option<PleOptions>) -> Result<Self, RigError> {
        present(path, config)?;
        if !metal_available() {
            return Err(RigError::NoDevice { backend: "Metal" });
        }
        let backend: Arc<dyn Backend> = MetalBackend::new()?;
        Self::dispatch(backend, path, config, ple)
    }

    /// Loads the checkpoint on an open device, as whichever family its config
    /// document describes, and sizes the buffers the phases read back into.
    ///
    /// One body for both devices: which family a checkpoint is, and what a rig
    /// holds, are the same questions whichever device answered them, and two
    /// copies of this would be two definitions of what a rig is.
    fn dispatch(
        backend: Arc<dyn Backend>,
        path: &Path,
        config: &Path,
        ple: Option<PleOptions>,
    ) -> Result<Self, RigError> {
        let mut opts = EngineOptions::default();
        opts.ctx = CTX;
        let loading = Instant::now();
        // The MoE family is the one whose checkpoint is a directory: three
        // shards and two config documents, neither of which is the single
        // container `HfConfig::from_file` below expects. A directory is never
        // any other family this dispatcher carries, so the two arms cannot
        // both claim the same path.
        let model: Box<dyn Model> = if path.is_dir() {
            let family = Qwen35MoeConfig::from_dir(path).map_err(|source| RigError::Load {
                path: path.to_path_buf(),
                detail: source.to_string(),
            })?;
            Box::new(
                Qwen35MoeModel::load(&backend, path, &family, opts).map_err(|source| {
                    RigError::Load {
                        path: path.to_path_buf(),
                        detail: source.to_string(),
                    }
                })?,
            )
        } else {
            let document = HfConfig::from_file(config).map_err(|source| RigError::Profile {
                path: config.to_path_buf(),
                source,
            })?;
            crate::replay::load_model_with(&backend, path, &document, opts, ple).map_err(|e| {
                RigError::Load {
                    path: path.to_path_buf(),
                    detail: e.to_string(),
                }
            })?
        };
        tracing::info!(
            seconds = loading.elapsed().as_secs_f64(),
            "checkpoint resident"
        );
        let vocab = usize::try_from(model.info().vocab).unwrap_or(0);
        Ok(Self {
            backend,
            row: vec![0u8; vocab.saturating_mul(4)],
            scratch: vec![0f32; vocab],
            prompt: prompt_of(model.info().vocab),
            name: file_name(path),
            model,
        })
    }

    /// The checkpoint file's name, as the record spells it.
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.name
    }

    /// Releases the model and the device.
    ///
    /// # Errors
    ///
    /// [`RigError::Device`] when the release itself fails, which is reported
    /// rather than swallowed: a teardown that failed means the next run on this
    /// machine starts from somewhere unknown.
    pub fn teardown(mut self) -> Result<(), RigError> {
        self.model.teardown()?;
        Ok(())
    }

    /// Opens a session standing at the end of the prompt, and the token the
    /// model would produce next.
    pub(crate) fn stand_at_frontier(&mut self) -> Result<(SessionId, TokenId), RigError> {
        let session = self.model.open_session()?;
        let last = self.model.prefill_last(session, &self.prompt)?;
        self.backend.copy_d2h(last.row(0)?, &mut self.row)?;
        Ok((session, self.greedy()))
    }

    /// The same, over a prompt shifted by `salt` so that two rows of one batch
    /// are two different conversations.
    ///
    /// `salt` moves every id of the prompt, which moves the whole session: a
    /// different prefix, a different set of pages, a different frontier and a
    /// different next token. A batch of rows that all ran the same ids would be
    /// a batch in which every row's answer is every other row's, and a defect
    /// that handed one row's page table to the whole batch would then measure
    /// and grade exactly as a correct one does.
    pub(crate) fn stand_at_frontier_salted(
        &mut self,
        salt: u32,
    ) -> Result<(SessionId, TokenId), RigError> {
        let vocab = self.scratch.len();
        let vocab = u32::try_from(vocab).unwrap_or(u32::MAX).max(1);
        let prompt: Vec<TokenId> = self
            .prompt
            .iter()
            .map(|id| TokenId::new(id.get().saturating_add(salt.saturating_mul(13)) % vocab))
            .collect();
        let session = self.model.open_session()?;
        let last = self.model.prefill_last(session, &prompt)?;
        self.backend.copy_d2h(last.row(0)?, &mut self.row)?;
        Ok((session, self.greedy()))
    }

    /// Rebuilds the fixed prompt at `tokens` ids.
    ///
    /// Prompt length is the independent variable of a prefill sweep, and
    /// [`PREFILL_TOKENS`] is only the length the committed baselines happen to
    /// have been taken at. Rebuilt from the same arithmetic span rather than
    /// truncated, because truncation can only go down from 64 and the one
    /// like-for-like comparison available against `llama-bench` is at 512.
    ///
    /// The session buffers are untouched: a prompt is ids, and the readback
    /// row is sized by the vocabulary rather than by the prompt.
    pub fn set_prompt_len(&mut self, tokens: u32) {
        let vocab = u32::try_from(self.scratch.len()).unwrap_or(u32::MAX);
        self.prompt = span_of(vocab, tokens);
    }

    /// The highest-scoring id of whatever is in the readback buffer.
    pub(crate) fn greedy(&mut self) -> TokenId {
        for (slot, bytes) in self.scratch.iter_mut().zip(self.row.as_chunks::<4>().0) {
            *slot = f32::from_le_bytes(*bytes);
        }
        argmax(&self.scratch)
    }
}

/// One repetition of the decode phase: the milliseconds each step took.
///
/// The session is opened, prefilled and closed inside the repetition, so every
/// repetition decodes from the same position. Reusing one session across
/// repetitions would have each of them start deeper in the cache than the last,
/// and the resulting drift would show up as spread that no amount of quiescing
/// could remove.
///
/// # Errors
///
/// Whatever the device reports.
pub fn decode_rep(rig: &mut Rig, steps: u32) -> Result<Vec<f64>, RigError> {
    let (session, first) = rig.stand_at_frontier()?;
    let mut token = first;
    let mut per_step = Vec::with_capacity(usize::try_from(steps).unwrap_or(0));
    for _ in 0..steps {
        let started = Instant::now();
        {
            let out = rig.model.decode_batch(&[DecodeStep { session, token }])?;
            rig.backend.copy_d2h(out.row(0)?, &mut rig.row)?;
        }
        per_step.push(started.elapsed().as_secs_f64() * 1000.0);
        token = rig.greedy();
    }
    rig.model.close_session(session)?;
    Ok(per_step)
}

/// The median milliseconds per decode step, for one repetition.
///
/// The leading steps the protocol drops are removed before the median is taken:
/// the first step is where the device lays the work out for the first time, and
/// every step after it reuses that.
///
/// # Errors
///
/// [`RigError::NoSamples`] when the protocol drops every step it asked for, and
/// whatever the device reports otherwise.
pub fn decode_ms_per_step(rig: &mut Rig, protocol: Protocol) -> Result<f64, RigError> {
    let per_step = decode_rep(rig, protocol.decode_steps)?;
    let steady = after_warmup(&per_step, protocol.dropped_decode_steps);
    median(steady).ok_or(RigError::NoSamples { phase: "decode" })
}

/// A fixed span of ids, every one inside the vocabulary.
///
/// Arithmetic rather than a captured sentence because the kernels cannot tell
/// the difference and a sentence would tie the record to a vocabulary file.
fn prompt(config: &Qwen35Config) -> Vec<TokenId> {
    prompt_of(config.shape().vocab)
}

/// The same fixed prompt, for a family whose vocabulary is read off the loaded
/// model rather than off a recorded profile.
fn prompt_of(vocab: u32) -> Vec<TokenId> {
    span_of(vocab, PREFILL_TOKENS)
}

/// `tokens` ids of the same arithmetic span, every one inside the vocabulary.
///
/// Split out of [`prompt_of`] so that the length is a parameter: a prefill
/// sweep varies it, and a second span written somewhere else would be a second
/// prompt whose numbers could not be read against these.
fn span_of(vocab: u32, tokens: u32) -> Vec<TokenId> {
    let vocab = vocab.max(1);
    (0..tokens)
        .map(|i| TokenId::new(i.saturating_mul(97).saturating_add(1) % vocab))
        .collect()
}

/// The device this platform has, opened.
///
/// One place where a backend is named, so that a lane added later cannot pick
/// the wrong one by copying the wrong neighbour.
///
/// # Errors
///
/// [`RigError::NoDevice`] naming the backend this platform expected.
fn platform_backend() -> Result<Arc<dyn Backend>, RigError> {
    #[cfg(target_os = "macos")]
    {
        if !metal_available() {
            return Err(RigError::NoDevice { backend: "Metal" });
        }
        Ok(MetalBackend::new()?)
    }
    #[cfg(not(target_os = "macos"))]
    {
        crate::replay::open_backend(crate::replay::Device::Cuda)
            .map_err(|_| RigError::NoDevice { backend: "CUDA" })
    }
}

/// Refuses before a device is opened when either file the run names is absent.
///
/// A missing checkpoint is the likeliest way to get here, and finding out after
/// the device is up turns a one-line refusal into a stack of backend errors.
fn present(path: &Path, config: &Path) -> Result<(), RigError> {
    if !path.exists() {
        return Err(RigError::NoCheckpoint {
            path: path.to_path_buf(),
        });
    }
    if !config.is_file() {
        return Err(RigError::NoCheckpoint {
            path: config.to_path_buf(),
        });
    }
    Ok(())
}

/// The checkpoint file's own name, as the record spells it.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The geometry, read from the recorded config beside the test vectors.
fn profile() -> Result<Qwen35Config, RigError> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json");
    let config = HfConfig::from_file(&path).map_err(|source| RigError::Profile {
        path: path.clone(),
        source,
    })?;
    Qwen35Config::from_hf(&config).map_err(|source| RigError::Profile { path, source })
}

// ---------------------------------------------------------------------------
// the batched phase
// ---------------------------------------------------------------------------

/// Stream counts a batched decode run reports, which is what
/// `scripts/streams.py` asks a server for.
///
/// The same four counts, so a number taken here and a number taken over HTTP
/// describe the same experiment. They are measured here as well as there
/// because **this build's server serves no `qwen3_5_moe` arm**: the `MoE` family
/// is reachable through [`Rig::open_qwen35moe`] and through nothing else, so
/// the phase whose target is aggregate throughput over several streams would
/// have no number at all for that family otherwise.
pub const STREAM_COUNTS: [u32; 4] = [1, 2, 4, 8];

/// One repetition of a batched decode phase: the milliseconds each step took
/// with `rows` sessions advancing together.
///
/// Every row is a session of its own standing at the end of a prompt of its
/// own. The prompts differ by row, which is not decoration: two sessions over
/// the same ids would still hold different pages, but they would stand at the
/// same positions and produce the same token, so a defect that mixed two rows
/// up would leave every number unchanged. The salted prompt is the same
/// precaution `scripts/streams.py` takes with its serial number.
///
/// Each row's logits are read back inside the timed window, as the single-row
/// phase reads its one row back, because the clock has to stop when the work is
/// done rather than when it was queued.
///
/// # Errors
///
/// Whatever the device reports.
pub fn decode_rep_batched(rig: &mut Rig, steps: u32, rows: u32) -> Result<Vec<f64>, RigError> {
    let width = usize::try_from(rows).unwrap_or(0);
    let mut sessions = Vec::with_capacity(width);
    let mut tokens = Vec::with_capacity(width);
    for row in 0..rows {
        let (session, token) = rig.stand_at_frontier_salted(row)?;
        sessions.push(session);
        tokens.push(token);
    }
    // One readback buffer per row, because the sampler cannot run while the
    // logits block is still borrowed: the rows are copied out first and picked
    // over afterwards.
    let mut host = vec![vec![0u8; rig.row.len()]; width];
    let mut per_step = Vec::with_capacity(usize::try_from(steps).unwrap_or(0));
    for _ in 0..steps {
        let batch: Vec<DecodeStep> = sessions
            .iter()
            .zip(&tokens)
            .map(|(session, token)| DecodeStep {
                session: *session,
                token: *token,
            })
            .collect();
        let started = Instant::now();
        {
            let out = rig.model.decode_batch(&batch)?;
            for (row, dst) in host.iter_mut().enumerate() {
                let row = u32::try_from(row).unwrap_or(u32::MAX);
                rig.backend.copy_d2h(out.row(row)?, dst)?;
            }
        }
        per_step.push(started.elapsed().as_secs_f64() * 1000.0);
        for (token, bytes) in tokens.iter_mut().zip(&host) {
            rig.row.copy_from_slice(bytes);
            *token = rig.greedy();
        }
    }
    for session in sessions {
        rig.model.close_session(session)?;
    }
    Ok(per_step)
}

/// The median milliseconds one batched decode step of `rows` rows took.
///
/// # Errors
///
/// [`RigError::NoSamples`] when the protocol drops every step it asked for, and
/// whatever the device reports otherwise.
pub fn decode_batched_ms_per_step(
    rig: &mut Rig,
    protocol: Protocol,
    rows: u32,
) -> Result<f64, RigError> {
    let per_step = decode_rep_batched(rig, protocol.decode_steps, rows)?;
    let steady = after_warmup(&per_step, protocol.dropped_decode_steps);
    median(steady).ok_or(RigError::NoSamples { phase: "decode" })
}
