//! Turning a model reference into an installed, verified model directory, and
//! a registry reference into the paths [`ServeOptions`] serves (spec §6).
//!
//! This is the one pull in the tree: `shiso pull`, `shiso serve <name>` and an
//! embedder all go through it, and only the progress they show differs.
//!
//! * [`resolve`] looks at the model home and never at the network.
//! * [`pull`] makes a model available, downloading what is missing, with a
//!   `(done, total) -> keep_going` progress callback.
//! * [`pull_observed`] is [`pull`] with the full event stream, for a command
//!   line that prints the steps as well as the bytes.
//! * [`resolve_serve_model`] fills a [`ServeOptions`] from an installed model.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[cfg(any(feature = "metal", feature = "cuda"))]
use shiso_hub::Backend;
use shiso_hub::{
    BlobStore, Downloader, HostProfile, HubConfig, HubError, InstalledVariantSummary, ModelName,
    ModelRef, ProgressEvent, Recipe, RegistryClient, Role, ServingDevice, Sha256, Variant,
    VariantKey, select_variant,
};
use tokio::sync::mpsc;

use crate::device::{Decision, DeviceChoice, GpuState, Want, resolve as resolve_device};
use crate::options::{ExplicitlySet, ServeOptions};
use crate::private_runtime::PrivateRuntime;
use crate::recipe_defaults::apply_recipe_defaults_unless_set;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod variant_tests;

/// A model installed in the model home, and the claim that keeps it there.
///
/// Holds the variant's lock shared for as long as it lives, as `shiso serve`
/// does, so `shiso rm` and a re-install in any process wait for it. Drop it to
/// let them go; in particular drop it before pulling the same variant again in
/// this process, which needs the lock exclusively.
#[derive(Debug)]
pub struct InstalledModel {
    /// The installed variant's directory, ready for a
    /// serve path.
    pub dir: PathBuf,
    /// The bytes of the variant's files.
    pub total_bytes: u64,
    /// The shared variant lock; released on drop.
    #[expect(
        dead_code,
        reason = "held for its Drop: the claim is the file being open"
    )]
    lock: File,
}

/// Why a model could not be made available.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PullError {
    /// This build has no compute backend, or no variant of the model runs on
    /// this host.
    #[error("{0}")]
    NoDevice(String),
    /// The registry does not list the model.
    #[error("model '{0}' not found in registry")]
    UnknownModel(String),
    /// The model reference is not a model name, or `name:variant`.
    #[error("invalid model reference '{reference}': {reason}")]
    InvalidReference {
        /// The text given.
        reference: String,
        /// What is wrong with it.
        reason: String,
    },
    /// `serve` was given a model reference and also a flag that names a
    /// component the reference already supplies.
    #[error("cannot pass component flags together with model reference '{0}'")]
    ComponentFlagConflict(String),
    /// The model (or the named variant) is not in the model home.
    #[error("Error: {0}")]
    NotInstalled(String),
    /// The installed files cannot be read as a model.
    #[error("{0}")]
    Inconsistent(String),
    /// The variant's lock could not be taken.
    #[error("locking variant {variant}")]
    Lock {
        /// The variant key.
        variant: String,
        /// The operating system's refusal.
        #[source]
        source: std::io::Error,
    },
    /// The progress callback asked to stop. Partial files stay, so the next
    /// call resumes them.
    #[error("the download was cancelled")]
    Cancelled,
    /// The pull cannot run: the private runtime could not start, or this
    /// crate's version is not semver.
    #[error("{0}")]
    Request(String),
    /// Installing or checking the downloaded files failed.
    #[error("the model could not be installed: {0}")]
    Install(String),
    /// The registry, the file host or the model home failed.
    #[error(transparent)]
    Hub(#[from] HubError),
}

/// Why a build with no backend cannot pull for an embedder.
const NO_BACKEND: &str = "no device can run the model: this build has no compute backend; rebuild with the `cpu`, `metal` or `cuda` feature";

/// Whether [`pull_observed`] asks the registry about a model that is already
/// installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullPolicy {
    /// An installed, complete variant is the answer; the registry is not asked
    /// (what an embedder wants: no network once the model is there).
    ReuseInstalled,
    /// Always ask the registry, so a newer manifest is installed over an old
    /// one (what `shiso pull` means).
    CheckRegistry,
}

/// One step of a pull, for a command line that prints them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullEvent {
    /// The registry's index is being fetched.
    FetchingIndex,
    /// A variant was chosen for this host.
    Selected {
        /// Why this one.
        explanation: String,
        /// A caution about the choice, if any.
        warning: Option<String>,
    },
    /// The chosen variant is installed and matches the registry's manifest;
    /// nothing is downloaded.
    UpToDate {
        /// The model.
        model: String,
        /// The variant.
        variant: String,
    },
    /// The downloader's own event: sizes, files, bytes, and the install.
    Progress(ProgressEvent),
    /// About every 200 ms while nothing else arrives: the cancel poll.
    Tick,
}

/// How often the observer is asked whether to go on while nothing is
/// arriving: a stalled connection must not make a cancel wait for a byte.
const TICK: Duration = Duration::from_millis(200);

/// A hub configuration for `home` with production endpoints and the Hugging
/// Face token `shiso pull` would use, for repositories that need one.
#[must_use]
pub fn config_for_home(home: &Path) -> HubConfig {
    let mut config = HubConfig::with_defaults(home.to_path_buf());
    config.hf_token = BlobStore::new(config.clone())
        .read_auth_token()
        .unwrap_or_else(|error| {
            tracing::debug!(%error, "no usable Hugging Face token; downloading without one");
            None
        });
    config
}

/// Builds a [`HostProfile`] from compile-time backend features, the devices
/// behind them (spec §5.2) and the device `choice` resolves to.
///
/// Only the cheap probes run (is the driver there, does a device exist); no
/// backend is opened, so `shiso pull` costs the same as before. A choice that
/// cannot be honoured (a GPU asked for and absent, or a build with no device)
/// leaves the profile on the GPU, where the pull's own refusal reports it.
#[must_use]
pub fn probe_host_profile(choice: DeviceChoice) -> HostProfile {
    #[cfg(feature = "cuda")]
    let cuda = shiso_cuda::cuda_available().then_some(shiso_hub::CudaInfo {
        // The server's own stand-in until the CUDA backend reports facts.
        sm: 89,
        vram_bytes: 16 * 1024 * 1024 * 1024,
    });
    #[cfg(not(feature = "cuda"))]
    let cuda: Option<shiso_hub::CudaInfo> = None;
    #[cfg(feature = "metal")]
    let metal = shiso_metal::device_facts().map(|facts| shiso_hub::MetalInfo {
        name: facts.name,
        family: facts.apple_family,
        working_set_bytes: facts.working_set_bytes,
    });
    #[cfg(not(feature = "metal"))]
    let metal: Option<shiso_hub::MetalInfo> = None;
    // A backend is listed only when its device is there: a CUDA build on a
    // host without the driver has no CUDA variant to run.
    #[cfg_attr(
        not(any(feature = "metal", feature = "cuda")),
        expect(unused_mut, reason = "a build with no GPU backend lists none")
    )]
    let mut backends = Vec::new();
    #[cfg(feature = "metal")]
    if metal.is_some() {
        backends.push(Backend::Metal);
    }
    #[cfg(feature = "cuda")]
    if cuda.is_some() {
        backends.push(Backend::Cuda);
    }
    let gpu = if cfg!(feature = "cuda") {
        if cuda.is_some() {
            GpuState::Present
        } else {
            GpuState::Absent("no CUDA device found".to_owned())
        }
    } else if cfg!(feature = "metal") {
        if metal.is_some() {
            GpuState::Present
        } else {
            GpuState::Absent("no Metal device found".to_owned())
        }
    } else {
        GpuState::NotCompiled
    };
    let device = match resolve_device(Want::from(choice), cfg!(feature = "cpu"), gpu) {
        Ok(Decision::Cpu { .. }) => ServingDevice::Cpu,
        Ok(Decision::Gpu) | Err(_) => ServingDevice::Gpu,
    };
    HostProfile::probe(backends, cuda, metal).on(device)
}

/// The installed variant `model_ref` names, when it is on disk and complete.
///
/// Reads only the model home, never the network: sizes and presence, since
/// the digests were checked when it was installed. A reference without a
/// variant means the one this host would select. `Ok(None)` is "not
/// installed (completely)".
///
/// # Errors
///
/// [`PullError::InvalidReference`] for text that is not a model reference, and
/// [`PullError::Hub`] when the model home cannot be read or locked.
pub fn resolve(model_ref: &str, config: &HubConfig) -> Result<Option<InstalledModel>, PullError> {
    // The registry's spelling is lowercase; an embedder need not know that.
    resolve_on(
        &probe_host_profile(DeviceChoice::Auto),
        &model_ref.to_lowercase(),
        config,
    )
}

// None is not a refusal: the model is not in the home (or not complete), which the caller answers by pulling.
fn resolve_on(
    host: &HostProfile,
    model_ref: &str,
    config: &HubConfig,
) -> Result<Option<InstalledModel>, PullError> {
    let parsed = parse_ref(model_ref)?;
    let store = BlobStore::new(config.clone());
    let Some((key, bytes)) = already_installed(&store, host, &parsed)? else {
        return Ok(None);
    };
    let lock = lock_shared(&store, &parsed.model, &key)?;
    Ok(Some(InstalledModel {
        dir: store.variant_dir(&parsed.model, &key),
        total_bytes: bytes,
        lock,
    }))
}

/// Makes `model_ref` available under the home in `config` and returns where.
///
/// A variant that is already installed and complete is returned without
/// touching the network. Otherwise the registry is asked, the missing files
/// are downloaded and checked against their pinned digests, and the variant is
/// installed and hashed once more end to end (the installed files are
/// hash-verified again, so a large model is read a second time). A reference may name a variant
/// (`name:variant`); without one this host's best is chosen.
///
/// `progress(done, total)` is called as bytes arrive, with the bytes still
/// missing when the download began as `total`. Returning `false` cancels: the
/// call returns [`PullError::Cancelled`] and partial files stay under the
/// home, so the next call resumes them.
///
/// An installed model is never checked against a newer manifest here; use
/// [`pull_observed`] with [`PullPolicy::CheckRegistry`] for that.
///
/// This blocks the calling thread while a runtime of its own does the work.
/// From an async task that blocks the task's worker, as any blocking I/O
/// does; a plain thread or a `spawn_blocking` one is the right place. `progress`
/// runs on that thread, between the download's steps and about every 200 ms
/// while nothing arrives, so a slow callback stalls the download and a cancel
/// is noticed within one tick even when a connection is stalled. `total` is 0
/// until the registry has answered; that poll is how a cancel reaches a
/// stalled index fetch. Installing and hashing run off that thread, so they do
/// not delay a cancel; the install then finishes in the background and the
/// call returns at once.
///
/// # Errors
///
/// [`PullError::NoDevice`] when this build has no backend feature, or no
/// variant of the model runs on it; [`PullError::UnknownModel`] for a name the
/// registry does not list; [`PullError::Cancelled`]; and the other variants
/// for whatever else fails along the way.
pub fn pull(
    model_ref: &str,
    config: &HubConfig,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, PullError> {
    pull_for(DeviceChoice::Auto, model_ref, config, progress)
}

/// [`pull`] for the device `choice` names, so a process that will serve with
/// `--device cpu` fetches the variant the CPU runs.
///
/// # Errors
///
/// As [`pull`].
pub fn pull_for(
    choice: DeviceChoice,
    model_ref: &str,
    config: &HubConfig,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, PullError> {
    pull_on(&probe_host_profile(choice), model_ref, config, progress)
}

fn pull_on(
    host: &HostProfile,
    model_ref: &str,
    config: &HubConfig,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, PullError> {
    // The device comes first: an embedder whose build cannot run a model must
    // not spend a request, let alone gigabytes of bandwidth, on one.
    if host.backends.is_empty() && host.device == ServingDevice::Gpu {
        return Err(PullError::NoDevice(NO_BACKEND.to_owned()));
    }
    let mut tally = Tally::default();
    let mut asked_registry = false;
    // The registry's spelling is lowercase; an embedder need not know that.
    let model = pull_observed_on(
        host,
        &model_ref.to_lowercase(),
        config,
        PullPolicy::ReuseInstalled,
        &mut |event| match event {
            PullEvent::Progress(event) => tally.report(event, progress),
            PullEvent::Tick => progress(tally.done(), tally.total),
            PullEvent::FetchingIndex => {
                asked_registry = true;
                true
            }
            PullEvent::Selected { .. } | PullEvent::UpToDate { .. } => true,
        },
    )?;
    if !asked_registry {
        // Installed already: nothing was fetched, so there is no progress.
        return Ok(model);
    }
    // The work is done, so a caller's bar ends full; with nothing to download
    // the tally is empty and the model's own size stands for the total.
    let total = if tally.total == 0 {
        model.total_bytes
    } else {
        tally.total
    };
    let _ = progress(total, total);
    Ok(model)
}

/// [`pull`] with every step as a [`PullEvent`], and the choice of whether an
/// installed model is checked against the registry.
///
/// `observer` returns `false` to cancel, as `progress` does in [`pull`]; the
/// same blocking rules apply.
///
/// # Errors
///
/// As [`pull`].
pub fn pull_observed(
    model_ref: &str,
    config: &HubConfig,
    policy: PullPolicy,
    observer: &mut dyn FnMut(&PullEvent) -> bool,
) -> Result<InstalledModel, PullError> {
    pull_observed_on(
        &probe_host_profile(DeviceChoice::Auto),
        model_ref,
        config,
        policy,
        observer,
    )
}

/// [`pull_observed`] for an explicit host profile, as an embedder or a test
/// that already knows its device supplies it. Unlike [`pull`], this asks the
/// registry even when the host has no backend; the variant selection then
/// refuses with the registry's own reasons.
///
/// # Errors
///
/// As [`pull`].
pub fn pull_observed_on(
    host: &HostProfile,
    model_ref: &str,
    config: &HubConfig,
    policy: PullPolicy,
    observer: &mut dyn FnMut(&PullEvent) -> bool,
) -> Result<InstalledModel, PullError> {
    let parsed = parse_ref(model_ref)?;
    let store = BlobStore::new(config.clone());
    if policy == PullPolicy::ReuseInstalled
        && let Some((key, bytes)) = already_installed(&store, host, &parsed)?
    {
        let lock = lock_shared(&store, &parsed.model, &key)?;
        return Ok(InstalledModel {
            dir: store.variant_dir(&parsed.model, &key),
            total_bytes: bytes,
            lock,
        });
    }
    // A runtime of its own with a worker to drive its timers and tasks, entered
    // for the calling thread; the future is polled by a bare park loop, which
    // has no objection to being called from inside any executor (it blocks that
    // executor's thread, as any blocking I/O does).
    // Dropped (shut down without waiting) on every path: downloads still in
    // flight after a cancel, an error or a panic are dropped, not awaited, and
    // their partial files are what the next call resumes.
    let runtime = PrivateRuntime::build(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all(),
    )
    .map_err(|error| PullError::Request(format!("could not start a runtime: {error}")))?;
    let _entered = runtime.handle().enter();
    park_on(drive(config, host, &store, &parsed, observer))
}

/// Wakes a parked thread.
struct Unpark(std::thread::Thread);

impl std::task::Wake for Unpark {
    fn wake(self: std::sync::Arc<Self>) {
        self.0.unpark();
    }
}

/// Polls `future` on this thread, parking between polls. Unlike an executor's
/// `block_on` it keeps no thread-local "already inside an executor" guard, so
/// it may be called from inside any executor, tokio's or not.
fn park_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = std::task::Waker::from(std::sync::Arc::new(Unpark(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    loop {
        if let std::task::Poll::Ready(output) = future.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

fn parse_ref(model_ref: &str) -> Result<ModelRef, PullError> {
    ModelRef::parse(model_ref).map_err(|error| PullError::InvalidReference {
        reference: model_ref.to_owned(),
        reason: error.to_string(),
    })
}

/// What `pull` found out and did, besides the model.
enum Outcome {
    /// Installed and matching the manifest already.
    UpToDate,
    /// Downloaded and installed now.
    Installed { model: String, variant: String },
}

/// Runs [`pull_steps`] while feeding `observer`, from its events and from a
/// timer, and drops it the moment `observer` says stop.
async fn drive(
    config: &HubConfig,
    host: &HostProfile,
    store: &BlobStore,
    parsed: &ModelRef,
    observer: &mut dyn FnMut(&PullEvent) -> bool,
) -> Result<InstalledModel, PullError> {
    // Steps before any bytes (`control`) and the downloader's events
    // (`progress`) are separate channels: the first always happen first, so
    // reading `control` ahead of `progress` keeps the order a reader sees.
    let (control_tx, mut control) = mpsc::unbounded_channel::<PullEvent>();
    let (progress_tx, mut progress) = mpsc::channel::<ProgressEvent>(64);
    let work = pull_steps(config, host, store, parsed, &control_tx, progress_tx);
    tokio::pin!(work);
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let result = loop {
        tokio::select! {
            biased;
            Some(event) = control.recv() => {
                if !observer(&event) {
                    return Err(PullError::Cancelled);
                }
            }
            Some(event) = progress.recv() => {
                if !observer(&PullEvent::Progress(event)) {
                    return Err(PullError::Cancelled);
                }
            }
            result = &mut work => break result,
            _ = tick.tick() => {
                // Also the cancel poll while the registry is slow.
                if !observer(&PullEvent::Tick) {
                    return Err(PullError::Cancelled);
                }
            }
        }
    };
    // What the finished work left in the channels, before its own last event.
    while let Ok(event) = control.try_recv() {
        if !observer(&event) {
            return Err(PullError::Cancelled);
        }
    }
    while let Ok(event) = progress.try_recv() {
        if !observer(&PullEvent::Progress(event)) {
            return Err(PullError::Cancelled);
        }
    }
    let (installed, outcome) = result?;
    if let Outcome::Installed { model, variant } = outcome
        && !observer(&PullEvent::Progress(ProgressEvent::Installed {
            model,
            variant,
        }))
    {
        return Err(PullError::Cancelled);
    }
    Ok(installed)
}

/// What `shiso pull` does: index, manifest, variant, download, install.
async fn pull_steps(
    config: &HubConfig,
    host: &HostProfile,
    store: &BlobStore,
    parsed: &ModelRef,
    control: &mpsc::UnboundedSender<PullEvent>,
    events: mpsc::Sender<ProgressEvent>,
) -> Result<(InstalledModel, Outcome), PullError> {
    // A closed `control` means the observer already stopped; the work is
    // about to be dropped, so a lost event is moot.
    let say = |event: PullEvent| {
        let _ = control.send(event);
    };
    say(PullEvent::FetchingIndex);
    let client = RegistryClient::new(config.clone());
    let index = client.fetch_index().await?;
    let entry = client
        .resolve_model(&index, parsed.model.as_str())
        .ok_or_else(|| PullError::UnknownModel(parsed.model.to_string()))?;
    let version = semver::Version::parse(env!("CARGO_PKG_VERSION")).map_err(|error| {
        PullError::Request(format!("this crate's version is not semver: {error}"))
    })?;
    let recipe = client.fetch_manifest(entry, &version).await?;
    let selected =
        select_variant(&recipe, host, parsed.variant.as_ref()).map_err(|error| match error {
            HubError::NoCompatibleVariant { .. } => PullError::NoDevice(error.to_string()),
            other => PullError::Hub(other),
        })?;
    let variant = selected.variant;
    say(PullEvent::Selected {
        explanation: selected.explanation.clone(),
        warning: selected.warning.clone(),
    });
    let dir = store.variant_dir(&recipe.name, &variant.key);
    let total_bytes = variant_bytes(variant);

    if store.is_up_to_date(&recipe.name, &variant.key, &entry.manifest_sha256, variant) {
        say(PullEvent::UpToDate {
            model: recipe.name.to_string(),
            variant: variant.key.to_string(),
        });
        // Waiting for another process's installer is blocking work.
        let (store, name, key) = (store.clone(), recipe.name.clone(), variant.key.clone());
        let lock = tokio::task::spawn_blocking(move || lock_shared(&store, &name, &key))
            .await
            .map_err(|error| PullError::Install(format!("lock task failed: {error}")))??;
        return Ok((
            InstalledModel {
                dir,
                total_bytes,
                lock,
            },
            Outcome::UpToDate,
        ));
    }

    let downloader = Downloader::new(config.clone());
    downloader
        .download_variant(recipe.name.as_ref(), variant, Some(&events))
        .await?;
    // Installing and hashing are blocking file work, gigabytes of it: off the
    // runtime's thread, so the progress tick (and a cancel) keeps running. A
    // cancel drops the join handle, not the task: the install finishes in the
    // background under its locks.
    let (store, downloader) = (store.clone(), downloader.clone());
    let (recipe, variant) = (recipe.clone(), variant.clone());
    let manifest = entry.manifest_sha256.clone();
    let outcome = Outcome::Installed {
        model: recipe.name.to_string(),
        variant: variant.key.to_string(),
    };
    let lock = tokio::task::spawn_blocking(move || {
        install_and_verify(&store, &downloader, &recipe, &variant, &manifest)
    })
    .await
    .map_err(|error| PullError::Install(format!("install task failed: {error}")))??;
    Ok((
        InstalledModel {
            dir,
            total_bytes,
            lock,
        },
        outcome,
    ))
}

/// Installs `variant`, hashes it once end to end, and on a mismatch removes
/// what must not be trusted again.
fn install_and_verify(
    store: &BlobStore,
    downloader: &Downloader,
    recipe: &Recipe,
    variant: &Variant,
    manifest: &Sha256,
) -> Result<File, PullError> {
    store.install_variant(recipe, variant, manifest)?;
    // The one full hash of what was installed.
    let checked = store.verify_variant(&recipe.name, &variant.key)?;
    let bad: Vec<&str> = checked
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(file, _)| file.as_str())
        .collect();
    if bad.is_empty() {
        return lock_shared(store, &recipe.name, &variant.key);
    }
    let mut failures = discard_bad(store, downloader, recipe, variant, &bad);
    failures.sort();
    let mut message = format!(
        "{} does not match its digest after installing",
        bad.join(", ")
    );
    if !failures.is_empty() {
        message = format!(
            "{message} (and could not be removed: {})",
            failures.join("; ")
        );
    }
    Err(PullError::Install(message))
}

/// Removes the variant directory and the blobs in `bad` (`role/name`), so the
/// next call neither takes the directory for a good install nor reuses the
/// blobs: the downloader trusts a blob's size alone. Holds the variant lock
/// and each blob's download lock while removing, as `shiso pull` and `rm`
/// would. Returns what could not be removed, each as a printable line.
fn discard_bad(
    store: &BlobStore,
    downloader: &Downloader,
    recipe: &Recipe,
    variant: &Variant,
    bad: &[&str],
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut note = |what: String| {
        tracing::warn!("{what}");
        failures.push(what);
    };
    let variant_lock = store
        .open_variant_lock(&recipe.name, &variant.key)
        .and_then(|file| {
            file.lock().map_err(|source| HubError::Io {
                path: store.variant_dir(&recipe.name, &variant.key),
                source,
            })?;
            Ok(file)
        });
    if let Err(error) = &variant_lock {
        note(format!("variant lock: {error}"));
    }
    let dir = store.variant_dir(&recipe.name, &variant.key);
    if let Err(error) = std::fs::remove_dir_all(&dir)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        note(format!("{}: {error}", dir.display()));
    }
    for (role, component) in &variant.components {
        for file in &component.files {
            if !bad.contains(&format!("{}/{}", role.as_str(), file.name).as_str()) {
                continue;
            }
            let lock = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(
                    downloader
                        .tmp_dir()
                        .join(format!("{}.lock", file.sha256.as_str())),
                )
                .and_then(|lock| lock.lock().map(|()| lock));
            if let Err(error) = &lock {
                note(format!("blob lock {}: {error}", file.sha256.as_str()));
            }
            let blob = downloader.blob_path(&file.sha256);
            if let Err(error) = std::fs::remove_file(&blob)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                note(format!("{}: {error}", blob.display()));
            }
            drop(lock);
        }
    }
    drop(variant_lock);
    failures
}

/// The variant `parsed` names here (or the host's pick), as its key and size,
/// when it is on disk and complete. Reads only the model home.
// None is not a refusal: the variant is not installed (completely).
fn already_installed(
    store: &BlobStore,
    host: &HostProfile,
    parsed: &ModelRef,
) -> Result<Option<(VariantKey, u64)>, PullError> {
    Ok(store
        .list_installed()?
        .into_iter()
        .filter(|summary| summary.model == parsed.model)
        .find_map(|summary| {
            let recipe = summary.recipe.as_ref()?;
            let wanted = select_variant(recipe, host, parsed.variant.as_ref())
                .ok()?
                .variant;
            let complete = wanted.key == summary.variant
                && store.is_up_to_date(
                    &parsed.model,
                    &wanted.key,
                    &summary.manifest_sha256,
                    wanted,
                );
            complete.then(|| (wanted.key.clone(), variant_bytes(wanted)))
        }))
}

/// Takes `model:key`'s variant lock shared, waiting for an installer to
/// finish.
fn lock_shared(store: &BlobStore, model: &ModelName, key: &VariantKey) -> Result<File, PullError> {
    let file = store.open_variant_lock(model, key)?;
    file.lock_shared().map_err(|source| PullError::Lock {
        variant: key.to_string(),
        source,
    })?;
    Ok(file)
}

fn variant_bytes(variant: &Variant) -> u64 {
    variant
        .components
        .values()
        .flat_map(|component| &component.files)
        .map(|file| file.size)
        .sum()
}

/// Folds the downloader's per-file events into one `(done, total)` pair.
#[derive(Default)]
struct Tally {
    total: u64,
    /// Bytes so far and size, by file.
    files: HashMap<String, (u64, u64)>,
}

impl Tally {
    fn done(&self) -> u64 {
        let done: u64 = self.files.values().map(|(done, _)| done).sum();
        done.min(self.total)
    }

    /// Applies `event` and asks `progress` whether to go on.
    fn report(
        &mut self,
        event: &ProgressEvent,
        progress: &mut dyn FnMut(u64, u64) -> bool,
    ) -> bool {
        match event {
            ProgressEvent::Resolved { total, .. } => self.total = *total,
            ProgressEvent::FileStarted {
                role, file, total, ..
            } => {
                self.files.insert(format!("{role}/{file}"), (0, *total));
            }
            ProgressEvent::FileProgress {
                role,
                file,
                bytes,
                total,
                ..
            } => {
                self.files
                    .insert(format!("{role}/{file}"), ((*bytes).min(*total), *total));
            }
            ProgressEvent::FileVerified {
                role, file, total, ..
            } => {
                self.files
                    .insert(format!("{role}/{file}"), (*total, *total));
            }
            ProgressEvent::Installed { .. } | ProgressEvent::Error { .. } => return true,
        }
        progress(self.done(), self.total)
    }
}

/// Classification of a positional model argument (spec §6.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArgKind {
    /// Path to a checkpoint file or directory.
    Path(PathBuf),
    /// Model reference from the registry.
    RegistryRef(String),
}

/// Classifies a positional model argument as a path or registry reference
/// (spec §6.6).
#[must_use]
pub fn classify_model_arg(arg: &str) -> ModelArgKind {
    if arg.contains('/') || arg.starts_with('.') || arg.starts_with('~') {
        ModelArgKind::Path(PathBuf::from(arg))
    } else {
        ModelArgKind::RegistryRef(arg.to_string())
    }
}

/// Points `options` at the model `model_ref` names (spec §6.6): a path becomes
/// `options.model`; a registry reference must be installed, and its
/// components fill the path options and its recipe fills the defaults the
/// caller left alone.
///
/// Returns the installed model, whose lock the caller holds while it serves,
/// or `None` for a path (nothing was locked). Never touches the network:
/// `shiso pull` first.
///
/// # Errors
///
/// [`PullError::ComponentFlagConflict`] when a component option is also set,
/// [`PullError::NotInstalled`] when the model or variant is not in the home,
/// [`PullError::InvalidReference`], [`PullError::Inconsistent`] for a damaged
/// install, and [`PullError::Hub`] when the home cannot be read or locked.
pub fn resolve_serve_model(
    model_ref: &str,
    options: &mut ServeOptions,
    explicit: &ExplicitlySet,
    config: &HubConfig,
) -> Result<Option<InstalledModel>, PullError> {
    let host = probe_host_profile(options.device);
    resolve_serve_model_on(&host, model_ref, options, explicit, config)
}

// None is not a refusal: a path was given, so nothing was installed or locked.
fn resolve_serve_model_on(
    host: &HostProfile,
    model_ref: &str,
    options: &mut ServeOptions,
    explicit: &ExplicitlySet,
    config: &HubConfig,
) -> Result<Option<InstalledModel>, PullError> {
    let ref_str = match classify_model_arg(model_ref) {
        ModelArgKind::Path(path) => {
            options.model = Some(path);
            return Ok(None);
        }
        ModelArgKind::RegistryRef(ref_str) => ref_str,
    };
    check_component_flag_conflicts(options, &ref_str)?;
    let parsed = ModelRef::parse(&ref_str).map_err(|error| PullError::InvalidReference {
        reference: ref_str.clone(),
        reason: error.to_string(),
    })?;

    let store = BlobStore::new(config.clone());
    let installed = store.list_installed()?;
    let chosen = find_installed_serve_variant(host, &installed, &parsed, &ref_str)?;

    let key = &chosen.variant;
    let vdir = store.variant_dir(&parsed.model, key);
    let lock = lock_shared(&store, &parsed.model, key)?;

    let recipe = chosen
        .recipe
        .as_ref()
        .ok_or_else(|| PullError::Inconsistent("installed recipe.json missing".to_owned()))?;
    let total_bytes = fill_from_recipe(recipe, key, &vdir, options, explicit)?;
    Ok(Some(InstalledModel {
        dir: vdir,
        total_bytes,
        lock,
    }))
}

/// Points `options` at the model in `dir`, for an embedder that has a
/// directory rather than a registry name.
///
/// A directory with a `recipe.json` is an installed variant (its name is the
/// variant key): its components fill the path options and its recipe fills the
/// defaults `explicit` leaves alone, exactly as [`resolve_serve_model`] does for
/// a name. Any other directory (or file) is a checkpoint and becomes
/// `options.model`, as `shiso serve <path>` takes it. Nothing is locked: a path
/// is the caller's to keep in place.
///
/// # Errors
///
/// [`PullError::ComponentFlagConflict`] when a component option is also set for
/// a variant directory, and [`PullError::Inconsistent`] for a `recipe.json` that
/// cannot be read as the variant its directory names.
pub fn resolve_serve_dir(
    dir: &Path,
    options: &mut ServeOptions,
    explicit: &ExplicitlySet,
) -> Result<(), PullError> {
    let bytes = match std::fs::read(dir.join("recipe.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            options.model = Some(dir.to_owned());
            return Ok(());
        }
        Err(error) => {
            return Err(PullError::Inconsistent(format!(
                "reading {}/recipe.json: {error}",
                dir.display()
            )));
        }
    };
    check_component_flag_conflicts(options, &dir.display().to_string())?;
    let inconsistent = |what: String| PullError::Inconsistent(format!("{}: {what}", dir.display()));
    let recipe: Recipe = serde_json::from_slice(&bytes)
        .map_err(|error| inconsistent(format!("recipe.json does not parse: {error}")))?;
    let key = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| inconsistent("the directory name is not a variant key".to_owned()))
        .and_then(|name| {
            VariantKey::parse(name).map_err(|error| inconsistent(error.to_string()))
        })?;
    fill_from_recipe(&recipe, &key, dir, options, explicit)?;
    Ok(())
}

/// Fills `options` from variant `key` of `recipe` installed at `vdir`, and
/// returns the variant's size in bytes.
fn fill_from_recipe(
    recipe: &Recipe,
    key: &VariantKey,
    vdir: &Path,
    options: &mut ServeOptions,
    explicit: &ExplicitlySet,
) -> Result<u64, PullError> {
    let variant = recipe
        .variants
        .iter()
        .find(|v| &v.key == key)
        .ok_or_else(|| PullError::Inconsistent("variant not in recipe".to_owned()))?;
    fill_options_from_variant(options, vdir, variant, &recipe.name)?;
    if let Some(defaults) = &variant.server_defaults {
        apply_recipe_defaults_unless_set(options, explicit, defaults);
    }
    Ok(variant_bytes(variant))
}

fn check_component_flag_conflicts(options: &ServeOptions, ref_str: &str) -> Result<(), PullError> {
    if options.config.is_some()
        || options.ngram.is_some()
        || options.vision.is_some()
        || options.draft_model.is_some()
        || options.draft_vocab.is_some()
        || options.asr_model.is_some()
        || options.asr_mmproj.is_some()
        || options.diarization_model.is_some()
        || options.image_model.is_some()
        || options.image_assets_dir.is_some()
        || options.image_text_encoder.is_some()
        || options.image_vae.is_some()
        || options.image_tokenizer.is_some()
    {
        return Err(PullError::ComponentFlagConflict(ref_str.to_owned()));
    }
    Ok(())
}

fn find_installed_serve_variant<'a>(
    host: &HostProfile,
    installed: &'a [InstalledVariantSummary],
    parsed_ref: &ModelRef,
    ref_str: &str,
) -> Result<&'a InstalledVariantSummary, PullError> {
    let matching: Vec<_> = installed
        .iter()
        .filter(|iv| iv.model == parsed_ref.model)
        .collect();

    if matching.is_empty() {
        return Err(PullError::NotInstalled(format!(
            "model '{}' is not installed. Run 'shiso pull {ref_str}' first.",
            parsed_ref.model
        )));
    }

    if let Some(target_key) = &parsed_ref.variant {
        matching
            .into_iter()
            .find(|iv| &iv.variant == target_key)
            .ok_or_else(|| {
                PullError::NotInstalled(format!(
                    "variant '{target_key}' for model '{}' is not installed.",
                    parsed_ref.model
                ))
            })
    } else {
        let recipe = matching
            .first()
            .and_then(|iv| iv.recipe.as_ref())
            .ok_or_else(|| {
                PullError::Inconsistent("could not load recipe for installed model".to_owned())
            })?;
        let selected = select_variant(recipe, host, None).map_err(|error| {
            PullError::NoDevice(format!(
                "no compatible installed variant for this host: {error}"
            ))
        })?;
        matching
            .into_iter()
            .find(|iv| iv.variant == selected.variant.key)
            .ok_or_else(|| {
                PullError::NotInstalled(format!(
                    "variant '{}' is not installed.",
                    selected.variant.key
                ))
            })
    }
}

fn fill_options_from_variant(
    options: &mut ServeOptions,
    vdir: &Path,
    variant: &Variant,
    recipe_name: &ModelName,
) -> Result<(), PullError> {
    // A role is supplied by its own component or by a file another component
    // binds to it; `role_paths` folds both into one answer per flag.
    let roles = variant.role_paths().map_err(|error| {
        PullError::Inconsistent(format!("installed recipe is inconsistent: {error}"))
    })?;
    for (role, place) in roles {
        let role_dir = vdir.join(place.dir.as_str());
        let target = match place.file {
            Some(file) => role_dir.join(file),
            None => role_dir,
        };
        match role {
            Role::Base => options.model = Some(target),
            Role::Draft => options.draft_model = Some(target),
            Role::DraftVocab => options.draft_vocab = Some(target),
            Role::Vision => options.vision = Some(target),
            Role::Asr => options.asr_model = Some(target),
            Role::AsrMmproj => options.asr_mmproj = Some(target),
            Role::Diarization => options.diarization_model = Some(target),
            Role::Config => options.config = Some(target),
            Role::Ngram => options.ngram = Some(target),
            Role::Image => options.image_model = Some(target),
            Role::ImageAssets => options.image_assets_dir = Some(target),
            Role::ImageTextEncoder => options.image_text_encoder = Some(target),
            Role::ImageVae => options.image_vae = Some(target),
            Role::ImageTokenizer => options.image_tokenizer = Some(target),
        }
    }
    if options.image_model.is_some() && options.image_model_id.is_none() {
        options.image_model_id = Some(recipe_name.to_string());
    }
    if options.model_name.is_none() {
        options.model_name = Some(recipe_name.to_string());
    }
    Ok(())
}
