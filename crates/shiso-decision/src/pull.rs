//! Turning a registry name into an installed, verified model directory.
//!
//! [`ensure_model`] is the one door an embedder needs: it resolves the name
//! against the model registry, picks the variant this build's backends can
//! run, downloads what is missing (resuming partial files), installs it and
//! returns the directory [`crate::Decider::open`] reads. The registry client,
//! downloader and blob store are the ones `shiso pull` drives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
/// This blocks the calling thread on its own runtime, so call it from a
/// thread that may block, not from inside an async executor.
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
        .ok()
        .flatten();
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
        return Err(DecideError::Download(
            "ensure_model blocks; call it from a thread outside an async executor".to_owned(),
        ));
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(pull(config, host, &store, name, progress));
    // Downloads still in flight after a cancel or an error are dropped here,
    // not awaited: their partial files are what the next call resumes.
    runtime.shutdown_background();
    result
}

/// What `shiso pull` does: index, manifest, variant, download, install.
async fn pull(
    config: &HubConfig,
    host: &HostProfile,
    store: &BlobStore,
    name: &str,
    progress: &mut dyn FnMut(u64, u64) -> bool,
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
    let total_bytes = variant_bytes(variant);

    if !store.is_up_to_date(&recipe.name, &variant.key, &entry.manifest_sha256, variant) {
        let downloader = Downloader::new(config.clone());
        let (tx, mut rx) = mpsc::channel::<ProgressEvent>(64);
        let transfer = downloader.download_variant(recipe.name.as_ref(), variant, Some(&tx));
        tokio::pin!(transfer);
        let mut tally = Tally::default();
        let outcome = loop {
            tokio::select! {
                result = &mut transfer => break result,
                Some(event) = rx.recv() => {
                    if !tally.report(&event, progress) {
                        return Err(DecideError::Cancelled);
                    }
                }
            }
        };
        outcome.map_err(|error| download(&error))?;
        while let Ok(event) = rx.try_recv() {
            if !tally.report(&event, progress) {
                return Err(DecideError::Cancelled);
            }
        }
        store
            .install_variant(&recipe, variant, &entry.manifest_sha256)
            .map_err(|error| download(&error))?;
        // The one full hash of what was installed.
        let checked = store
            .verify_variant(&recipe.name, &variant.key)
            .map_err(|error| download(&error))?;
        if let Some((file, _)) = checked.iter().find(|(_, ok)| !ok) {
            return Err(DecideError::Download(format!(
                "{file} does not match its digest after installing"
            )));
        }
        // The count of the work done, so a caller's bar ends full.
        let _ = progress(tally.total, tally.total);
    }
    Ok(InstalledModel {
        dir: store.variant_dir(&recipe.name, &variant.key),
        total_bytes,
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
        let done: u64 = self.files.values().map(|(done, _)| done).sum();
        progress(done.min(self.total), self.total)
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
            for extra in extra_models {
                models.push(json!({
                    "name": extra,
                    "family": "lfm2",
                    "min_shiso_version": "0.1.0",
                    "description": "other",
                    "tasks": ["a_task_from_the_future"],
                    "updated_at": "2026-10-05T00:00:00Z",
                    "manifest": format!("models/{extra}/index.json"),
                    "manifest_sha256": "0".repeat(64),
                    "variants": [],
                }));
            }
            let index = json!({
                "schema_version": 1, "generated_at": "2026-10-05T00:00:00Z", "models": models
            })
            .to_string()
            .into_bytes();
            let serve = |route: &str, body: Vec<u8>| {
                let mock = Mock::given(method("GET"))
                    .and(path(route.to_owned()))
                    .respond_with(ResponseTemplate::new(200).set_body_bytes(body));
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
