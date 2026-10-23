//! The pull, against a mock registry and file host.

use std::fs;
use std::num::NonZeroUsize;
use std::time::Duration;

use serde_json::json;
use sha2::{Digest, Sha256};
use shiso_hub::{Backend, RegistryEndpoint, ServingDevice};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::options::{DEFAULT_CTX, ExplicitlySet, ServeOptions};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

/// Runs `work` on a thread of its own and fails if it takes over a minute, so
/// a hang is a failure and not a stuck suite.
fn guarded<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(work());
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("the call neither returned nor panicked within a minute"))
}

fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// A registry and a file host in one mock server, serving one model.
struct Fixture {
    /// The runtime the mock server lives on; `pull` blocks on its own.
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
                "server_defaults": {"ctx": 32768},
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
    ) -> Result<InstalledModel, PullError> {
        pull_on(&metal_host(), name, &self.config(), progress)
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

/// The pair `(dir, total_bytes)`; an [`InstalledModel`] also holds a lock.
fn same(a: &InstalledModel, b: &InstalledModel) -> bool {
    a.dir == b.dir && a.total_bytes == b.total_bytes
}

#[test]
fn pull_installs_then_reuses() {
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
    // Another shared claim on the variant is fine; an installer would wait.
    let again = fx
        .ensure("mini", &mut |_, _| {
            panic!("nothing is downloaded the second time")
        })
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(same(&again, &installed));
    assert_eq!(fx.requests(), 0, "an installed model needs no network");
}

#[test]
fn resolve_is_offline_and_none_when_absent() {
    let fx = Fixture::new(&[]);
    let config = fx.config();
    let nothing = resolve_on(&metal_host(), "mini", &config).unwrap_or_else(|e| panic!("{e}"));
    assert!(nothing.is_none());
    fx.ensure("mini", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    fx.rt.block_on(fx.server.reset());
    let found = resolve_on(&metal_host(), "mini", &config)
        .unwrap_or_else(|e| panic!("{e}"))
        .expect("installed");
    assert_eq!(found.total_bytes, 3_000_023);
    let named = resolve_on(&metal_host(), "mini:q8_0", &config).unwrap_or_else(|e| panic!("{e}"));
    assert!(named.is_some(), "an explicit installed variant resolves");
    let other = resolve_on(&metal_host(), "mini:q4_0", &config).unwrap_or_else(|e| panic!("{e}"));
    assert!(other.is_none(), "an explicit variant that is not there");
    assert_eq!(fx.requests(), 0);
}

/// A pull runs from a blocking-pool thread, and from an async task, where it
/// blocks that task's worker and neither panics nor hangs.
#[test]
fn pull_runs_on_a_blocking_thread_and_in_a_task() {
    guarded(pull_on_a_blocking_thread_and_in_a_task);
}

fn pull_on_a_blocking_thread_and_in_a_task() {
    let fx = Fixture::new(&[]);
    let (config, host) = (fx.config(), metal_host());
    let installed = fx
        .rt
        .block_on(async {
            tokio::task::spawn_blocking(move || pull_on(&host, "mini", &config, &mut |_, _| true))
                .await
        })
        .expect("the blocking task ends")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(installed.total_bytes, 3_000_023);
    drop(installed);

    // Installed models return before any work, so remove it and pull again.
    fs::remove_dir_all(fx.home.path().join("models/mini")).unwrap_or_else(|e| panic!("{e}"));
    let (config, host) = (fx.config(), metal_host());
    let again = fx
        .rt
        .block_on(async {
            tokio::spawn(async move { pull_on(&host, "mini", &config, &mut |_, _| true) }).await
        })
        .expect("the task ends")
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(again.total_bytes, 3_000_023);
}

/// `shiso pull MiNi` was a parse error before the move and still is: only
/// the embedder's `pull` and `resolve` fold case.
#[test]
fn the_command_line_path_does_not_fold_case() {
    let fx = Fixture::new(&[]);
    let result = pull_observed_on(
        &metal_host(),
        "MiNi",
        &fx.config(),
        PullPolicy::CheckRegistry,
        &mut |_| true,
    );
    assert!(
        matches!(result, Err(PullError::InvalidReference { .. })),
        "{:?}",
        result.err()
    );
    assert_eq!(fx.requests(), 0);
}

/// With no backend the command line still announces the index fetch and
/// fails after it, as it always did; only the embedder's `pull` refuses
/// before a request.
#[test]
fn the_command_line_with_no_backend_fetches_the_index_then_fails() {
    let fx = Fixture::new(&[]);
    let host = HostProfile::probe(Vec::new(), None, None);
    let mut events = Vec::new();
    let result = pull_observed_on(
        &host,
        "mini",
        &fx.config(),
        PullPolicy::CheckRegistry,
        &mut |event| {
            events.push(event.clone());
            true
        },
    );
    assert!(
        events.iter().any(|e| matches!(e, PullEvent::FetchingIndex)),
        "{events:?}"
    );
    assert!(fx.requests() > 0);
    let error = result.expect_err("no variant runs without a backend");
    assert!(matches!(error, PullError::NoDevice(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .starts_with("no compatible variant found for model 'mini'"),
        "the registry's own text: {error}"
    );
}

/// A model home that cannot be read is an error, not "nothing installed".
#[test]
#[cfg(unix)]
fn resolve_reports_an_unreadable_home() {
    use std::os::unix::fs::PermissionsExt as _;
    let fx = Fixture::new(&[]);
    install(&fx);
    let models = fx.home.path().join("models");
    fs::set_permissions(&models, fs::Permissions::from_mode(0o000))
        .unwrap_or_else(|e| panic!("{e}"));
    let result = resolve_on(&metal_host(), "mini", &fx.config());
    fs::set_permissions(&models, fs::Permissions::from_mode(0o755))
        .unwrap_or_else(|e| panic!("{e}"));
    // Running as root reads everything; there the premise does not hold.
    if matches!(result, Ok(Some(_))) {
        return;
    }
    assert!(
        matches!(result, Err(PullError::Hub(_))),
        "{:?}",
        result.err()
    );
}

#[test]
fn a_name_in_any_case_installs_and_is_found_again() {
    let fx = Fixture::new(&[]);
    let installed = fx
        .ensure("MiNi", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    let again = fx
        .ensure("mini", &mut |_, _| panic!("already installed"))
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(same(&again, &installed));
}

#[test]
fn a_damaged_install_is_fetched_again() {
    let fx = Fixture::new(&[]);
    let installed = fx
        .ensure("mini", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    let dir = installed.dir.clone();
    // An installer needs the variant lock exclusively.
    drop(installed);
    let weights = dir.join("base").join("mini.gguf");
    fs::remove_file(&weights).unwrap_or_else(|e| panic!("{e}"));
    let again = fx
        .ensure("mini", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fs::read(weights).unwrap_or_default(), fx.weights);
    assert_eq!(again.dir, dir);
}

#[test]
fn cancel_keeps_partials() {
    let fx = Fixture::new(&[]);
    let result = fx.ensure("mini", &mut |done, _| done == 0);
    assert!(matches!(result, Err(PullError::Cancelled)), "{result:?}");
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
        matches!(&result, Err(PullError::UnknownModel(name)) if name == "nope"),
        "{result:?}"
    );
    let result = fx.ensure("Not A Name!", &mut |_, _| true);
    assert!(
        matches!(result, Err(PullError::InvalidReference { .. })),
        "{result:?}"
    );
}

#[test]
fn an_index_entry_with_an_unknown_task_hides_nothing() {
    let fx = Fixture::new(&["future"]);
    let installed = fx.ensure("mini", &mut |_, _| true);
    assert!(installed.is_ok(), "{:?}", installed.err());
}

#[test]
fn no_backend_is_no_device_before_any_request() {
    let fx = Fixture::new(&[]);
    let host = HostProfile::probe(Vec::new(), None, None);
    let result = pull_on(&host, "mini", &fx.config(), &mut |_, _| true);
    assert!(matches!(result, Err(PullError::NoDevice(_))), "{result:?}");
    assert_eq!(fx.requests(), 0);
}

/// The no-backend refusals above are the no-CPU-build case: a host serving on
/// the CPU has no backend and still pulls, because the registry has no CPU
/// variant and every variant is a candidate there.
#[test]
fn a_cpu_device_pulls_a_metal_only_variant() {
    let fx = Fixture::new(&[]);
    let host = HostProfile::probe(Vec::new(), None, None).on(ServingDevice::Cpu);
    let installed = pull_on(&host, "mini", &fx.config(), &mut |_, _| true);
    assert!(installed.is_ok(), "{:?}", installed.err());
}

#[test]
fn a_model_no_backend_here_runs_is_no_device() {
    let fx = Fixture::new(&[]);
    let host = HostProfile::probe(vec![Backend::Cuda], None, None);
    // The fixture runs on cuda too, but this host lacks the cuda facts.
    let result = pull_on(&host, "mini", &fx.config(), &mut |_, _| true);
    assert!(matches!(result, Err(PullError::NoDevice(_))), "{result:?}");
}

#[test]
fn a_blob_that_rotted_is_never_installed_as_good() {
    let fx = Fixture::new(&[]);
    let installed = fx
        .ensure("mini", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    let dir = installed.dir.clone();
    drop(installed);
    // Same size, wrong bytes: the downloader reuses a blob on size alone.
    let blob = fx
        .home
        .path()
        .join("blobs")
        .join("sha256")
        .join(digest(&fx.weights));
    fs::remove_file(&blob).unwrap_or_else(|e| panic!("{e}"));
    fs::write(&blob, vec![7u8; fx.weights.len()]).unwrap_or_else(|e| panic!("{e}"));
    fs::remove_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));

    let result = fx.ensure("mini", &mut |_, _| true);
    assert!(matches!(result, Err(PullError::Install(_))), "{result:?}");
    assert!(
        !dir.join("installed.json").exists(),
        "a failed verify leaves no install behind"
    );
    assert!(!blob.exists(), "the blob that failed its hash is removed");
    // The bad blob and the install are gone, so the next call fetches the
    // good bytes again and succeeds.
    let again = fx
        .ensure("mini", &mut |_, _| true)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        fs::read(again.dir.join("base").join("mini.gguf")).unwrap_or_default(),
        fx.weights
    );
}

/// Cancels once `after` has passed, and checks the call returns promptly.
fn cancelled_within(fx: &Fixture, after: Duration) {
    let started = std::time::Instant::now();
    let result = fx.ensure("mini", &mut |_, _| started.elapsed() < after);
    assert!(matches!(result, Err(PullError::Cancelled)), "{result:?}");
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

/// A pull inside a `futures::executor::block_on` main future, which refuses
/// nested use of itself: the pull must neither panic nor hang.
#[test]
fn calling_inside_a_futures_executor_works() {
    let fx = Fixture::new(&[]);
    let (config, host) = (fx.config(), metal_host());
    let result = guarded(move || {
        futures::executor::block_on(async { pull_on(&host, "mini", &config, &mut |_, _| true) })
    });
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn calling_inside_a_block_on_main_future_works() {
    guarded(|| {
        let fx = Fixture::new(&[]);
        let config = fx.config();
        let result = fx
            .rt
            .block_on(async { pull_on(&metal_host(), "mini", &config, &mut |_, _| true) });
        assert!(result.is_ok(), "{:?}", result.err());
    });
}

#[test]
fn an_already_present_model_ends_its_progress_full() {
    let fx = Fixture::new(&[]);
    drop(
        fx.ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}")),
    );
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

/// What a command line prints: the steps in order, then the download's
/// own events, then the install - and, asked to check the registry, an
/// installed model is reported up to date without a download.
#[test]
fn observed_events_arrive_in_order_and_the_registry_is_asked_on_request() {
    let fx = Fixture::new(&[]);
    let config = fx.config();
    let mut kinds = Vec::new();
    let first = pull_observed_on(
        &metal_host(),
        "mini",
        &config,
        PullPolicy::CheckRegistry,
        &mut |event| {
            kinds.push(match event {
                PullEvent::FetchingIndex => "index",
                PullEvent::Selected { .. } => "selected",
                PullEvent::UpToDate { .. } => "up-to-date",
                PullEvent::Progress(ProgressEvent::Resolved { .. }) => "resolved",
                PullEvent::Progress(ProgressEvent::FileStarted { .. }) => "file-started",
                PullEvent::Progress(ProgressEvent::FileProgress { .. }) => "file-progress",
                PullEvent::Progress(ProgressEvent::FileVerified { .. }) => "file-verified",
                PullEvent::Progress(ProgressEvent::Installed { .. }) => "installed",
                PullEvent::Progress(ProgressEvent::Error { .. }) => "error",
                PullEvent::Tick => return true,
            });
            true
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    drop(first);
    kinds.dedup();
    assert_eq!(kinds.first(), Some(&"index"));
    assert_eq!(kinds.get(1), Some(&"selected"));
    assert_eq!(kinds.last(), Some(&"installed"), "{kinds:?}");
    let started = kinds.iter().position(|k| *k == "file-started");
    let verified = kinds.iter().rposition(|k| *k == "file-verified");
    assert!(started < verified, "{kinds:?}");

    let requests = fx.requests();
    let mut again = Vec::new();
    let second = pull_observed_on(
        &metal_host(),
        "mini",
        &config,
        PullPolicy::CheckRegistry,
        &mut |event| {
            if !matches!(event, PullEvent::Tick) {
                again.push(event.clone());
            }
            true
        },
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(second.total_bytes, 3_000_023);
    assert!(fx.requests() > requests, "the registry was asked");
    assert!(
        matches!(again.first(), Some(PullEvent::FetchingIndex))
            && again
                .iter()
                .any(|e| matches!(e, PullEvent::UpToDate { model, variant }
                        if model == "mini" && variant == "q8_0")),
        "{again:?}"
    );
    assert!(
        !again.iter().any(|e| matches!(e, PullEvent::Progress(_))),
        "nothing is downloaded: {again:?}"
    );
}

fn install(fx: &Fixture) {
    drop(
        fx.ensure("mini", &mut |_, _| true)
            .unwrap_or_else(|e| panic!("{e}")),
    );
}

#[test]
fn serve_resolution_fills_paths_names_and_recipe_defaults() {
    let fx = Fixture::new(&[]);
    install(&fx);
    let mut options = ServeOptions::default();
    let model = resolve_serve_model_on(
        &metal_host(),
        "mini",
        &mut options,
        &ExplicitlySet::default(),
        &fx.config(),
    )
    .unwrap_or_else(|e| panic!("{e}"))
    .expect("a registry model");
    let vdir = fx.home.path().join("models/mini/q8_0");
    assert_eq!(model.dir, vdir);
    assert_eq!(options.model, Some(vdir.join("base").join("mini.gguf")));
    assert_eq!(
        options.config,
        Some(vdir.join("config").join("config.json"))
    );
    assert_eq!(options.model_name.as_deref(), Some("mini"));
    assert_eq!(options.ctx, 32768, "the recipe default applies");
}

/// The command line's `--ctx 4096` is the built-in default's value; the
/// recipe still must not move it.
#[test]
fn a_flag_explicitly_equal_to_the_default_still_beats_the_recipe() {
    let fx = Fixture::new(&[]);
    install(&fx);
    let mut options = ServeOptions {
        ctx: DEFAULT_CTX,
        ..ServeOptions::default()
    };
    let explicit: ExplicitlySet = ["ctx"].into_iter().collect();
    resolve_serve_model_on(&metal_host(), "mini", &mut options, &explicit, &fx.config())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(options.ctx, DEFAULT_CTX);
}

#[test]
fn serve_resolution_refuses_conflicts_and_missing_models() {
    let fx = Fixture::new(&[]);
    let explicit = ExplicitlySet::default();
    let mut options = ServeOptions::default();
    let error =
        resolve_serve_model_on(&metal_host(), "mini", &mut options, &explicit, &fx.config())
            .expect_err("not installed yet");
    assert!(matches!(error, PullError::NotInstalled(_)), "{error:?}");
    assert!(
        error
            .to_string()
            .contains("model 'mini' is not installed. Run 'shiso pull mini' first."),
        "{error}"
    );

    let mut options = ServeOptions {
        vision: Some("/v".into()),
        ..ServeOptions::default()
    };
    let error =
        resolve_serve_model_on(&metal_host(), "mini", &mut options, &explicit, &fx.config())
            .expect_err("conflict");
    assert!(
        matches!(error, PullError::ComponentFlagConflict(_)),
        "{error:?}"
    );
}

/// A directory with a recipe is an installed variant: the same options as
/// the name gives, with the recipe defaults held back by `explicit`.
#[test]
fn a_variant_directory_resolves_as_its_name_does() {
    let fx = Fixture::new(&[]);
    install(&fx);
    let vdir = fx.home.path().join("models/mini/q8_0");
    let mut by_name = ServeOptions::default();
    resolve_serve_model_on(
        &metal_host(),
        "mini",
        &mut by_name,
        &ExplicitlySet::default(),
        &fx.config(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut by_dir = ServeOptions::default();
    resolve_serve_dir(&vdir, &mut by_dir, &ExplicitlySet::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(by_dir, by_name);

    let mut kept = ServeOptions {
        ctx: 2048,
        ..ServeOptions::default()
    };
    let explicit: ExplicitlySet = ["ctx"].into_iter().collect();
    resolve_serve_dir(&vdir, &mut kept, &explicit).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(kept.ctx, 2048, "an explicit setting beats the recipe");
}

/// A directory with no recipe is a checkpoint, taken as `serve <path>` takes
/// it; a recipe that does not parse is refused, not guessed around.
#[test]
fn a_directory_without_a_recipe_is_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
    let mut options = ServeOptions::default();
    resolve_serve_dir(dir.path(), &mut options, &ExplicitlySet::default())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(options.model, Some(dir.path().to_owned()));

    let broken = dir.path().join("q8_0");
    fs::create_dir(&broken).unwrap_or_else(|e| panic!("{e}"));
    fs::write(broken.join("recipe.json"), b"{").unwrap_or_else(|e| panic!("{e}"));
    let error = resolve_serve_dir(&broken, &mut options, &ExplicitlySet::default())
        .expect_err("a recipe that does not parse");
    assert!(matches!(error, PullError::Inconsistent(_)), "{error:?}");
}

#[test]
fn a_path_is_taken_as_the_model_without_a_lock() {
    let fx = Fixture::new(&[]);
    let mut options = ServeOptions::default();
    let none = resolve_serve_model_on(
        &metal_host(),
        "./models/x",
        &mut options,
        &ExplicitlySet::default(),
        &fx.config(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(none.is_none());
    assert_eq!(options.model, Some("./models/x".into()));
}

#[test]
fn classify_splits_paths_from_references() {
    assert_eq!(
        classify_model_arg("a/b"),
        ModelArgKind::Path(PathBuf::from("a/b"))
    );
    assert_eq!(
        classify_model_arg("qwen3.5-9b:q4"),
        ModelArgKind::RegistryRef("qwen3.5-9b:q4".to_owned())
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
    let config = config_for_home(home.path());
    let started = std::time::Instant::now();
    let mut ticks = 0u32;
    let installed = pull("d1-3b", &config, &mut |_, _| {
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
    // A second call is the offline path.
    let again =
        pull("d1-3b", &config, &mut |_, _| panic!("no download")).unwrap_or_else(|e| panic!("{e}"));
    assert!(same(&again, &installed));
}
