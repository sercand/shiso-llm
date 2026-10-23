//! Turning a registry name into an installed, verified model directory.
//!
//! [`ensure_model`] is the one door an embedder needs: it resolves the name
//! against the model registry, picks the variant this build's backends can
//! run, downloads what is missing (resuming partial files), installs it and
//! returns the directory [`crate::Decider::open`] reads. The registry client,
//! downloader and blob store are the ones `shiso pull` drives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use shiso_hub::{
    BlobStore, Downloader, HostProfile, HubConfig, HubError, ModelName, ProgressEvent,
    RegistryClient, Variant, select_variant,
};
use tokio::sync::mpsc;

#[cfg(any(feature = "metal", feature = "cuda"))]
use shiso_hub::Backend;

use crate::error::DecideError;

/// A model installed in the model home.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledModel {
    /// The installed variant's directory, ready for [`crate::Decider::open`].
    pub dir: PathBuf,
    /// The bytes of the variant's files.
    pub total_bytes: u64,
}

/// Makes `name` available under `home` and returns where.
///
/// A variant that is already installed and complete is returned without
/// touching the network. Otherwise the registry is asked, the missing files
/// are downloaded and checked against their pinned digests, and the variant is
/// installed and hashed once more end to end.
///
/// `progress(done, total)` is called as bytes arrive, with the bytes still
/// missing when the download began as `total`. Returning `false` cancels:
/// the call returns [`DecideError::Cancelled`] and partial files stay under
/// `home`, so the next call resumes them.
///
/// An installed model is never checked against a newer manifest: once it is
/// there and complete, the registry is not asked again.
///
/// This blocks the calling thread on its own runtime, so call it from a
/// thread that may block, not from inside an async executor. `progress` runs
/// on that thread, between the download's steps and about every 200 ms while
/// nothing arrives, so a slow callback stalls the download and a cancel is
/// noticed within one tick even when a connection is stalled.
///
/// # Errors
///
/// [`DecideError::NoDevice`] when this build has no backend feature, or no
/// variant of the model runs on it; [`DecideError::UnknownModel`] for a name
/// the registry does not list; [`DecideError::Cancelled`]; and
/// [`DecideError::Download`] for any other failure along the way.
pub fn ensure_model(
    name: &str,
    home: &Path,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, DecideError> {
    let mut config = HubConfig::with_defaults(home.to_path_buf());
    // The same token `shiso pull` would use, for repositories that need one.
    config.hf_token = BlobStore::new(config.clone())
        .read_auth_token()
        .unwrap_or_else(|error| {
            tracing::debug!(%error, "no usable Hugging Face token; downloading without one");
            None
        });
    ensure_with(&config, &probe_host(), name, progress)
}

/// [`ensure_model`] against an explicit registry and host, which the tests
/// point at a local mock.
fn ensure_with(
    config: &HubConfig,
    host: &HostProfile,
    name: &str,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, DecideError> {
    // The device comes first: a build that cannot run a model must not spend
    // gigabytes of bandwidth on one.
    if host.backends.is_empty() {
        return Err(DecideError::NoDevice(
            "this build has no compute backend; rebuild with the `metal` or `cuda` feature"
                .to_owned(),
        ));
    }
    let unknown = || DecideError::UnknownModel(name.to_owned());
    let model = ModelName::parse(&name.to_lowercase()).map_err(|_| unknown())?;
    let store = BlobStore::new(config.clone());
    if let Some(installed) = already_installed(&store, host, &model) {
        return Ok(installed);
    }
    if tokio::runtime::Handle::try_current().is_ok() {
        return Err(DecideError::Request(
            "ensure_model blocks; call it from a thread outside an async executor".to_owned(),
        ));
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(drive(config, host, &store, name, progress));
    // Downloads still in flight after a cancel or an error are dropped here,
    // not awaited: their partial files are what the next call resumes.
    runtime.shutdown_background();
    result
}

/// How often the caller is asked whether to go on while nothing is arriving:
/// a stalled connection must not make a cancel wait for the next byte.
const TICK: Duration = Duration::from_millis(200);

/// Runs [`pull`] while feeding `progress`, from download events and from a
/// timer, and drops it the moment `progress` says stop.
async fn drive(
    config: &HubConfig,
    host: &HostProfile,
    store: &BlobStore,
    name: &str,
    progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<InstalledModel, DecideError> {
    let (tx, mut rx) = mpsc::channel::<ProgressEvent>(64);
    let work = pull(config, host, store, name, tx);
    tokio::pin!(work);
    let mut tally = Tally::default();
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let result = loop {
        tokio::select! {
            result = &mut work => break result,
            Some(event) = rx.recv() => {
                if !tally.report(&event, progress) {
                    return Err(DecideError::Cancelled);
                }
            }
            _ = tick.tick() => {
                if !progress(tally.done(), tally.total) {
                    return Err(DecideError::Cancelled);
                }
            }
        }
    };
    let installed = result?;
    // The work is done, so a caller's bar ends full; with nothing to download
    // the tally is empty and the model's own size stands for the total.
    let total = if tally.total == 0 {
        installed.total_bytes
    } else {
        tally.total
    };
    let _ = progress(total, total);
    Ok(installed)
}

/// What `shiso pull` does: index, manifest, variant, download, install.
async fn pull(
    config: &HubConfig,
    host: &HostProfile,
    store: &BlobStore,
    name: &str,
    events: mpsc::Sender<ProgressEvent>,
) -> Result<InstalledModel, DecideError> {
    let client = RegistryClient::new(config.clone());
    let index = client
        .fetch_index()
        .await
        .map_err(|error| download(&error))?;
    let entry = client
        .resolve_model(&index, name)
        .ok_or_else(|| DecideError::UnknownModel(name.to_owned()))?;
    let version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .map_err(|error| DecideError::Download(format!("this crate's version: {error}")))?;
    let recipe = client
        .fetch_manifest(entry, &version)
        .await
        .map_err(|error| download(&error))?;
    let selected = select_variant(&recipe, host, None).map_err(|error| match error {
        HubError::NoCompatibleVariant { .. } | HubError::VariantNotFound { .. } => {
            DecideError::NoDevice(error.to_string())
        }
        other => download(&other),
    })?;
    let variant = selected.variant;

    if !store.is_up_to_date(&recipe.name, &variant.key, &entry.manifest_sha256, variant) {
        Downloader::new(config.clone())
            .download_variant(recipe.name.as_ref(), variant, Some(&events))
            .await
            .map_err(|error| download(&error))?;
        store
            .install_variant(&recipe, variant, &entry.manifest_sha256)
            .map_err(|error| download(&error))?;
        // The one full hash of what was installed.
        let checked = store
            .verify_variant(&recipe.name, &variant.key)
            .map_err(|error| download(&error))?;
        let bad: Vec<&str> = checked
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(file, _)| file.as_str())
            .collect();
        if !bad.is_empty() {
            // The next call must not take this directory for a good install,
            // nor reuse the blobs that failed: the downloader trusts a blob's
            // size alone.
            let _ = std::fs::remove_dir_all(store.variant_dir(&recipe.name, &variant.key));
            for (role, component) in &variant.components {
                for file in &component.files {
                    if bad.contains(&format!("{}/{}", role.as_str(), file.name).as_str()) {
                        let blob = config
                            .home
                            .join("blobs")
                            .join("sha256")
                            .join(file.sha256.as_str());
                        let _ = std::fs::remove_file(blob);
                    }
                }
            }
            return Err(DecideError::Download(format!(
                "{} does not match its digest after installing",
                bad.join(", ")
            )));
        }
    }
    Ok(InstalledModel {
        dir: store.variant_dir(&recipe.name, &variant.key),
        total_bytes: variant_bytes(variant),
    })
}

/// The variant `model` would be installed as here, when it is on disk and
/// complete. Reads only the model home, never the network: sizes and
/// presence, since the digests were checked when it was installed.
fn already_installed(
    store: &BlobStore,
    host: &HostProfile,
    model: &ModelName,
) -> Option<InstalledModel> {
    store
        .list_installed()
        .ok()?
        .into_iter()
        .filter(|summary| &summary.model == model)
        .find_map(|summary| {
            let recipe = summary.recipe.as_ref()?;
            let wanted = select_variant(recipe, host, None).ok()?.variant;
            let complete = wanted.key == summary.variant
                && store.is_up_to_date(model, &wanted.key, &summary.manifest_sha256, wanted);
            complete.then(|| InstalledModel {
                dir: store.variant_dir(model, &wanted.key),
                total_bytes: variant_bytes(wanted),
            })
        })
}

fn variant_bytes(variant: &Variant) -> u64 {
    variant
        .components
        .values()
        .flat_map(|component| &component.files)
        .map(|file| file.size)
        .sum()
}

fn download(error: &HubError) -> DecideError {
    DecideError::Download(error.to_string())
}

/// Folds the downloader's per-file events into one `(done, total)` pair.
#[derive(Default)]
struct Tally {
    total: u64,
    /// Bytes so far and size, by file.
    files: HashMap<String, (u64, u64)>,
}

impl Tally {
    /// Applies `event` and reports the new totals; `false` when the caller
    /// asked to stop.
    fn done(&self) -> u64 {
        let done: u64 = self.files.values().map(|(done, _)| done).sum();
        done.min(self.total)
    }

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

/// The backends this build was compiled with, and the devices behind them.
fn probe_host() -> HostProfile {
    let backends = vec![
        #[cfg(feature = "metal")]
        Backend::Metal,
        #[cfg(feature = "cuda")]
        Backend::Cuda,
    ];
    #[cfg(feature = "cuda")]
    let cuda = shiso_cuda::cuda_available().then_some(shiso_hub::CudaInfo {
        // The server's own stand-in until the CUDA backend reports facts.
        sm: 89,
        vram_bytes: 16 * 1024 * 1024 * 1024,
    });
    #[cfg(not(feature = "cuda"))]
    let cuda = None;
    #[cfg(feature = "metal")]
    let metal = shiso_metal::device_facts().map(|facts| shiso_hub::MetalInfo {
        name: facts.name,
        family: facts.apple_family,
        working_set_bytes: facts.working_set_bytes,
    });
    #[cfg(not(feature = "metal"))]
    let metal = None;
    HostProfile::probe(backends, cuda, metal)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::num::NonZeroUsize;

    use serde_json::json;
    use sha2::{Digest, Sha256};
    use shiso_hub::{Backend, RegistryEndpoint};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    fn digest(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// A registry and a file host in one mock server, serving one model.
    struct Fixture {
        /// The runtime the mock server lives on; `ensure_model` blocks on its own.
        rt: tokio::runtime::Runtime,
        server: MockServer,
        home: tempfile::TempDir,
        weights: Vec<u8>,
    }

    impl Fixture {
        fn new(extra_models: &[&str]) -> Self {
            Self::slow(extra_models, "", Duration::ZERO)
        }

        /// A fixture whose routes ending in `slow` answer only after `delay`.
        fn slow(extra_models: &[&str], slow: &str, delay: Duration) -> Self {
            let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| panic!("runtime: {e}"));
            let server = rt.block_on(MockServer::start());
            // Large enough that the progress interval (512 KiB) fires.
            let weights: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
            let files: [(&str, &str, Vec<u8>); 3] = [
                (
                    "config",
                    "config.json",
                    br#"{"model_type":"lfm2"}"#.to_vec(),
                ),
                ("config", "tokenizer.json", b"{}".to_vec()),
                ("base", "mini.gguf", weights.clone()),
            ];
            let component = |role: &str| {
                let entries: Vec<_> = files
                    .iter()
                    .filter(|(r, ..)| *r == role)
                    .map(|(_, name, bytes)| {
                        json!({"name": name, "size": bytes.len(), "sha256": digest(bytes)})
                    })
                    .collect();
                json!({
                    "provider": "huggingface",
                    "repo": format!("acme/{role}"),
                    "revision": REVISION,
                    "entry": if role == "base" { "mini.gguf" } else { "config.json" },
                    "files": entries,
                })
            };
            let manifest = json!({
                "schema_version": 1,
                "name": "mini",
                "aliases": [],
                "family": "lfm2",
                "min_shiso_version": "0.1.0",
                "description": "fixture",
                "tags": [],
                "tasks": ["decision"],
                "modalities": {"inputs": ["text"], "outputs": ["text"]},
                "features": {
                    "tools": false, "reasoning": false, "reasoning_efforts": [],
                    "default_reasoning_effort": null, "structured_output": false
                },
                "variants": [{
                    "key": "q8_0",
                    "backends": ["metal", "cuda"],
                    "quant": "Q8_0",
                    "requirements": {"min_ram_bytes": 1, "disk_bytes": 3_000_023},
                    "components": {"base": component("base"), "config": component("config")},
                }],
            })
            .to_string()
            .into_bytes();
            let mut models = vec![json!({
                "name": "mini",
                "family": "lfm2",
                "min_shiso_version": "0.1.0",
                "description": "fixture",
                "tasks": ["decision"],
                "updated_at": "2026-10-05T00:00:00Z",
                "manifest": "models/mini/index.json",
                "manifest_sha256": digest(&manifest),
                "variants": [],
            })];
            models.extend(extra_models.iter().map(|extra| other_entry(extra)));
            let index = json!({
                "schema_version": 1, "generated_at": "2026-10-05T00:00:00Z", "models": models
            })
            .to_string()
            .into_bytes();
            let serve = |route: &str, body: Vec<u8>| {
                let wait = if !slow.is_empty() && route.ends_with(slow) {
                    delay
                } else {
                    Duration::ZERO
                };
                let mock = Mock::given(method("GET"))
                    .and(path(route.to_owned()))
                    .respond_with(
                        ResponseTemplate::new(200)
                            .set_body_bytes(body)
                            .set_delay(wait),
                    );
                rt.block_on(mock.mount(&server));
            };
            serve("/v1/index.json", index);
            serve("/v1/models/mini/index.json", manifest);
            for (role, name, bytes) in files {
                serve(&format!("/acme/{role}/resolve/{REVISION}/{name}"), bytes);
            }
            Self {
                rt,
                server,
                home: tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}")),
                weights,
            }
        }

        fn config(&self) -> HubConfig {
            let uri = self.server.uri();
            HubConfig {
                home: self.home.path().to_path_buf(),
                registry: RegistryEndpoint {
                    url: url::Url::parse(&format!("{uri}/v1")).unwrap_or_else(|e| panic!("{e}")),
                },
                hf_endpoint: url::Url::parse(&uri).unwrap_or_else(|e| panic!("{e}")),
                hf_token: None,
                max_parallel_files: NonZeroUsize::MIN,
            }
        }

        fn requests(&self) -> usize {
            self.rt
                .block_on(self.server.received_requests())
                .map_or(0, |requests| requests.len())
        }

        fn ensure(
            &self,
            name: &str,
            progress: &mut dyn FnMut(u64, u64) -> bool,
        ) -> Result<InstalledModel, DecideError> {
            ensure_with(&self.config(), &metal_host(), name, progress)
        }
    }

    /// An index entry whose task this client has never heard of.
    fn other_entry(name: &str) -> serde_json::Value {
        json!({
            "name": name,
            "family": "lfm2",
            "min_shiso_version": "0.1.0",
            "description": "other",
            "tasks": ["a_task_from_the_future"],
            "updated_at": "2026-10-05T00:00:00Z",
            "manifest": format!("models/{name}/index.json"),
            "manifest_sha256": "0".repeat(64),
            "variants": [],
        })
    }

    fn metal_host() -> HostProfile {
        let metal = shiso_hub::MetalInfo {
            name: "test device".to_owned(),
            family: 9,
            working_set_bytes: 64 << 30,
        };
        HostProfile::probe(vec![Backend::Metal], None, Some(metal))
    }

    #[test]
    fn ensure_model_installs_then_reuses() {
        let fx = Fixture::new(&[]);
        let mut last = (0, 0);
        let installed = fx
            .ensure("mini", &mut |done, total| {
                last = (done, total);
                true
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(installed.total_bytes, 3_000_023);
        assert_eq!(last.0, last.1, "progress ends complete");
        assert_eq!(last.1, 3_000_023);
        assert_eq!(
            installed.dir,
            fx.home.path().join("models").join("mini").join("q8_0")
        );
        assert_eq!(
            fs::read(installed.dir.join("base").join("mini.gguf")).unwrap_or_default(),
            fx.weights
        );
        assert!(
            installed
                .dir
                .join("config")
                .join("tokenizer.json")
                .is_file()
        );

        fx.rt.block_on(fx.server.reset());
        let again = fx
            .ensure("mini", &mut |_, _| {
                panic!("nothing is downloaded the second time")
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(again, installed);
        assert_eq!(fx.requests(), 0, "an installed model needs no network");
    }

    #[test]
    fn a_damaged_install_is_fetched_again() {
        let fx = Fixture::new(&[]);
        let installed = fx
            .ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}"));
        let weights = installed.dir.join("base").join("mini.gguf");
        fs::remove_file(&weights).unwrap_or_else(|e| panic!("{e}"));
        let again = fx
            .ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(fs::read(weights).unwrap_or_default(), fx.weights);
        assert_eq!(again.dir, installed.dir);
    }

    #[test]
    fn ensure_model_cancel_keeps_partials() {
        let fx = Fixture::new(&[]);
        let result = fx.ensure("mini", &mut |done, _| done == 0);
        assert!(matches!(result, Err(DecideError::Cancelled)), "{result:?}");
        let home = fx.home.path();
        assert!(
            !home
                .join("models")
                .join("mini")
                .join("q8_0")
                .join("installed.json")
                .exists(),
            "a cancelled pull installs nothing"
        );
        let kept = |dir: &str| {
            fs::read_dir(home.join(dir)).map_or(0, |entries| entries.filter_map(Result::ok).count())
        };
        assert!(
            kept("tmp") + kept("blobs/sha256") > 0,
            "partial downloads stay for the next attempt"
        );
        // The next call finishes the job.
        let installed = fx
            .ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(installed.total_bytes, 3_000_023);
    }

    #[test]
    fn unknown_name_is_typed_error() {
        let fx = Fixture::new(&[]);
        let result = fx.ensure("nope", &mut |_, _| true);
        assert!(
            matches!(&result, Err(DecideError::UnknownModel(name)) if name == "nope"),
            "{result:?}"
        );
        let result = fx.ensure("Not A Name!", &mut |_, _| true);
        assert!(
            matches!(result, Err(DecideError::UnknownModel(_))),
            "{result:?}"
        );
    }

    #[test]
    fn an_index_entry_with_an_unknown_task_hides_nothing() {
        let fx = Fixture::new(&["future"]);
        let installed = fx.ensure("mini", &mut |_, _| true);
        assert!(installed.is_ok(), "{installed:?}");
    }

    #[test]
    fn no_backend_is_no_device_before_any_request() {
        let fx = Fixture::new(&[]);
        let host = HostProfile::probe(Vec::new(), None, None);
        let result = ensure_with(&fx.config(), &host, "mini", &mut |_, _| true);
        assert!(
            matches!(result, Err(DecideError::NoDevice(_))),
            "{result:?}"
        );
        assert_eq!(fx.requests(), 0);
    }

    #[test]
    fn a_model_no_backend_here_runs_is_no_device() {
        let fx = Fixture::new(&[]);
        let host = HostProfile::probe(vec![Backend::Cuda], None, None);
        // The fixture runs on cuda too, but this host lacks the cuda facts.
        let result = ensure_with(&fx.config(), &host, "mini", &mut |_, _| true);
        assert!(
            matches!(result, Err(DecideError::NoDevice(_))),
            "{result:?}"
        );
    }

    #[test]
    fn a_blob_that_rotted_is_never_installed_as_good() {
        let fx = Fixture::new(&[]);
        let installed = fx
            .ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}"));
        // Same size, wrong bytes: the downloader reuses a blob on size alone.
        let blob = fx
            .home
            .path()
            .join("blobs")
            .join("sha256")
            .join(digest(&fx.weights));
        fs::remove_file(&blob).unwrap_or_else(|e| panic!("{e}"));
        fs::write(&blob, vec![7u8; fx.weights.len()]).unwrap_or_else(|e| panic!("{e}"));
        fs::remove_dir_all(&installed.dir).unwrap_or_else(|e| panic!("{e}"));

        let result = fx.ensure("mini", &mut |_, _| true);
        assert!(
            matches!(result, Err(DecideError::Download(_))),
            "{result:?}"
        );
        assert!(
            !installed.dir.join("installed.json").exists(),
            "a failed verify leaves no install behind"
        );
        // The next call never accepts the bad bytes: it errors or fetches them again.
        match fx.ensure("mini", &mut |_, _| true) {
            Ok(again) => assert_eq!(
                fs::read(again.dir.join("base").join("mini.gguf")).unwrap_or_default(),
                fx.weights
            ),
            Err(error) => assert!(matches!(error, DecideError::Download(_)), "{error:?}"),
        }
    }

    /// Cancels once `after` has passed, and checks the call returns promptly.
    fn cancelled_within(fx: &Fixture, after: Duration) {
        let started = std::time::Instant::now();
        let result = fx.ensure("mini", &mut |_, _| started.elapsed() < after);
        assert!(matches!(result, Err(DecideError::Cancelled)), "{result:?}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "cancel took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn cancel_works_while_the_weights_are_stalled() {
        let fx = Fixture::slow(&[], "mini.gguf", Duration::from_secs(30));
        cancelled_within(&fx, Duration::from_millis(300));
    }

    #[test]
    fn cancel_works_while_the_manifest_is_stalled() {
        let fx = Fixture::slow(&[], "models/mini/index.json", Duration::from_secs(30));
        cancelled_within(&fx, Duration::from_millis(300));
    }

    #[test]
    fn cancel_works_while_the_index_is_stalled() {
        let fx = Fixture::slow(&[], "/v1/index.json", Duration::from_secs(30));
        cancelled_within(&fx, Duration::from_millis(300));
    }

    #[test]
    fn calling_inside_an_async_runtime_is_a_request_error() {
        let fx = Fixture::new(&[]);
        let config = fx.config();
        let result = fx
            .rt
            .block_on(async { ensure_with(&config, &metal_host(), "mini", &mut |_, _| true) });
        assert!(matches!(result, Err(DecideError::Request(_))), "{result:?}");
    }

    #[test]
    fn an_already_present_model_ends_its_progress_full() {
        let fx = Fixture::new(&[]);
        fx.ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}"));
        // Same files in the home, but a manifest that reaches this call as new:
        // remove only the install record so the network path runs with every
        // blob already present.
        let record = fx.home.path().join("models/mini/q8_0/installed.json");
        fs::remove_file(record).unwrap_or_else(|e| panic!("{e}"));
        let mut last = (0, 0);
        fx.ensure("mini", &mut |done, total| {
            last = (done, total);
            true
        })
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(last, (3_000_023, 3_000_023));
    }

    /// Downloads the real `d1-3b` into a fresh home and opens it.
    #[test]
    #[cfg(feature = "metal")]
    #[ignore = "downloads about 3.4 GB from the public registry and needs a Metal device"]
    #[expect(
        clippy::print_stderr,
        reason = "the size and time of the download are the point of the run"
    )]
    fn pull_d1_3b() {
        let home = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let started = std::time::Instant::now();
        let mut ticks = 0u32;
        let installed = ensure_model("d1-3b", home.path(), &mut |_, _| {
            ticks += 1;
            true
        })
        .unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "pull_d1_3b: {} bytes in {:.1}s ({ticks} progress calls) at {}",
            installed.total_bytes,
            started.elapsed().as_secs_f64(),
            installed.dir.display()
        );
        assert!(installed.total_bytes > 3_000_000_000);
        let decider = crate::Decider::open(&crate::OpenDesc::new(&installed.dir));
        assert!(decider.is_ok(), "{:?}", decider.err());
        // A second call is the offline path.
        let again = ensure_model("d1-3b", home.path(), &mut |_, _| panic!("no download"))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(again, installed);
    }
}
