//! shiso-server: the command line, the HTTP surface, and the boundary between
//! them and the model.
//!
//! # The shape of the whole thing
//!
//! A request arrives at [`router`], passes the auth gate in [`auth`], and
//! reaches a handler in [`api`]. A handler that needs a generation puts a
//! message on the queue in [`scheduler::queue`] and awaits a stream of
//! [`shiso_ir::StreamEvent`]. One OS thread on the other side of that queue
//! owns the model and is the only thing that ever calls a method on it.
//!
//! Everything a handler can fail with is [`error::AppError`], which renders
//! itself in whichever wire format the surface that was asked for speaks.
//!
//! # What runs behind the queue
//!
//! A batching coordinator: it holds a bounded set of resident sessions, runs one
//! batched decode step over everything that is ready, reads prompt in bounded
//! quanta so a long prompt cannot stall an answer already streaming, cuts what
//! outlived its deadline or lost its client, and stores what is worth storing to
//! the checkpoint tier. It is the only thing in this process that touches the
//! model.
#![forbid(unsafe_code)]

pub use shiso_router;

pub mod anthropic;
pub mod api;
pub mod audio;
pub mod auth;
pub mod cli;
pub mod error;
pub mod ids;
pub mod media;
pub mod media_cache;
pub mod model_factory;
pub mod openai;
pub mod response_store;
pub mod router;
pub mod scheduler;
pub mod shutdown;
pub mod state;
pub mod tokenizer;
pub mod vision;

use std::sync::Arc;

use anyhow::{Context as _, anyhow};

use crate::cli::{Cli, Command, ServeArgs};
use crate::state::AppState;

/// The tracing filter's environment variable.
///
/// Deliberately not the conventional one: every knob this binary reads is
/// prefixed, and a logging level that came from an unprefixed variable would be
/// the one setting an operator could not find by listing the prefix.
const LOG_ENV: &str = "SHISO_LOG";

/// Runs the command line.
///
/// This is the binary's edge, which is why it is the one function in the crate
/// that returns an opaque error: every failure here ends the process, and the
/// only consumer is a human reading the message.
///
/// # Errors
///
/// Anything that stops the server starting: an unservable configuration, a
/// build with no backend, a checkpoint that does not load, an address already
/// in use.
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    install_tracing();
    match cli.command {
        Command::Serve(args) => serve(*args).await,
        Command::Router(args) => router(*args).await,
    }
}

/// Runs the router proxy with the given command-line arguments.
///
/// # Errors
///
/// Returns an error if the router fails to start or serve.
async fn router(args: shiso_router::cli::RouterArgs) -> anyhow::Result<()> {
    shiso_router::run(args).await
}
/// Installs the log subscriber.
///
/// Failure is ignored on purpose: a subscriber is already installed when a test
/// or an embedder set one up first, and refusing to serve because the logging
/// was already configured would be absurd.
fn install_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::builder()
        .with_env_var(LOG_ENV)
        .try_from_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

/// Loads a checkpoint and serves it.
///
/// The order is load, then bind, then serve, and it is load-first on purpose: a
/// checkpoint that cannot be loaded ends the process having never accepted a
/// connection, so there is no readiness state to report and no window in which
/// a client can be told the server is up and then refused.
async fn serve(args: ServeArgs) -> anyhow::Result<()> {
    args.validate()?;

    // The language model is optional: a server started with only a recognizer
    // serves transcription and answers every generation route with the typed
    // "no model is loaded" error.
    // The speech models load first: a KV pool sized to a share of free device
    // memory must see their weights already resident, or it takes the memory
    // they then load into.
    let audio = boot_audio(&args)?;
    let (mut state, engine_thread) = match args.model.as_deref() {
        Some(model) => {
            let (state, join) = boot_language_model(&args, model)?;
            (state, Some(join))
        }
        None => (AppState::without_language_model(), None),
    };
    if let Some(audio) = audio {
        state = state.with_audio(audio);
    }
    if let Some(auth) = auth::AuthConfig::new(args.auth_tokens()) {
        tracing::info!(tokens = auth.token_count(), "bearer authentication is on");
        state = state.with_auth(auth);
    } else {
        tracing::warn!("no bearer tokens configured; every inference route is open");
    }

    shutdown::install();
    let state = Arc::new(state);
    let result = router::serve(state, &args.bind, args.port).await;

    // The engine thread is joined after serving stops so that teardown finishes
    // before the process exits. Dropping the handle is what closes the queue,
    // and the queue closing is what ends the loop.
    match engine_thread.map(std::thread::JoinHandle::join) {
        None | Some(Ok(Ok(()))) => {}
        Some(Ok(Err(error))) => {
            tracing::error!(error = %shiso_core::chain_of(&error), "the engine did not shut down cleanly");
        }
        Some(Err(_panicked)) => tracing::error!("the engine thread panicked"),
    }
    result
        .map_err(|error| anyhow!(error).context(format!("serving on {}:{}", args.bind, args.port)))
}

/// Opens the speech models and starts the audio worker, when `--asr-model`
/// was given.
///
/// On backend instances of their own, one for the recognizer and one for the
/// diarizer: a device that records the language model's launches for replay
/// must not find the recognizer's launches in the recording. The worker thread runs until every handle to the service is
/// dropped, which is when the server stops.
///
/// # Errors
///
/// A recognizer or diarizer that will not load, a lookahead the recognizer
/// was not trained for, or a device that will not open.
// `Ok(None)` is a server without transcription: no `--asr-model` was given.
pub fn boot_audio(args: &ServeArgs) -> anyhow::Result<Option<audio::AudioService>> {
    let Some(asr) = args.asr_model.as_deref() else {
        return Ok(None);
    };
    let backend = open_audio_backend(args)?;
    let started = std::time::Instant::now();
    let recognizer = shiso_models::audio_registry::open_speech_recognizer(
        asr,
        &shiso_models::audio_registry::RecognizerDesc {
            mmproj: args.asr_mmproj.as_deref(),
            offline_batch: args.asr_batch,
        },
        &backend,
    )
    .with_context(|| format!("opening the speech recognizer at {}", asr.display()))?;
    if recognizer.lookaheads().is_empty() {
        // A recognizer without lookahead has nothing to check the flag
        // against; its streams ignore it.
        tracing::info!(
            lookahead = args.asr_lookahead,
            "this recognizer has no lookahead; --asr-lookahead is ignored"
        );
    } else if !recognizer.lookaheads().contains(&args.asr_lookahead) {
        return Err(anyhow!(
            "--asr-lookahead {} is not one this recognizer was trained for; it takes {:?}",
            args.asr_lookahead,
            recognizer.lookaheads()
        ));
    }
    // The diarizer runs on its own thread beside the recognizer, so it gets
    // its own command stream too: on a shared one, every decode round trip
    // of the recognizer would wait behind the diarizer's queued encoder work.
    let diarizer = match args.diarization_model.as_deref() {
        Some(path) => {
            let own = open_audio_backend(args)?;
            let diarizer = shiso_models::audio_registry::open_diarizer(path, &own)
                .with_context(|| format!("opening the diarizer at {}", path.display()))?;
            own.finish_imports();
            Some(diarizer)
        }
        None => None,
    };
    backend.finish_imports();
    let asr_id = args.asr_id().unwrap_or_else(|| "asr".to_owned());
    tracing::info!(
        asr = %asr.display(),
        id = %asr_id,
        lookahead = args.asr_lookahead,
        diarizer = diarizer.is_some(),
        offline_batch = recognizer.batches_offline(),
        offline_min_secs = args.asr_offline_min_secs,
        skip_silence = args.asr_skip_silence,
        seconds = started.elapsed().as_secs_f64(),
        "the speech models are resident"
    );
    let (service, _worker) = audio::AudioService::start(audio::AudioServiceDesc {
        recognizer,
        diarizer,
        asr_id,
        lookahead: args.asr_lookahead,
        max_streams: args.audio_max_streams,
        queue: args.audio_max_streams.saturating_mul(4),
        limits: args.audio_limits(),
        ffmpeg: args.audio_ffmpeg_policy(),
        max_upload_bytes: args.audio_max_upload_bytes(),
        offline_min_secs: args.asr_offline_min_secs,
        skip_silence: args.asr_skip_silence,
    })
    .context("starting the audio worker")?;
    Ok(Some(service))
}

/// The engine thread's handle, joined at shutdown.
type EngineThread = std::thread::JoinHandle<Result<(), scheduler::EngineFailure>>;

/// Loads the language model at `model` and starts its engine thread: the
/// state every generation route reads, before authentication is attached.
fn boot_language_model(
    args: &ServeArgs,
    model_path: &std::path::Path,
) -> anyhow::Result<(AppState, EngineThread)> {
    let backend = open_backend(args)?;
    // The first mark: whatever is in use now belongs to the driver and the
    // rest of the machine, and the ledger's unaccounted figure is measured
    // from here.
    mark_memory(backend.as_ref(), "before_load");
    let config_path = args.config_path();
    let config = shiso_core::HfConfig::from_file(&config_path).with_context(|| {
        format!(
            "reading the architecture config at {}",
            config_path.display()
        )
    })?;
    // Before the weights move: a card that exists and does not parse is an
    // operator mistake, and finding it after a minutes-long load is the most
    // expensive place to find it.
    let sampling_preset = load_sampling_preset(args)?;
    let model_factory::Loaded {
        mut model,
        open_tower,
        open_audio,
        placement,
        defaults,
    } = load_checkpoint(&backend, &config, args, model_path)?;
    // Before anything stands on a session: the tier is a property of the cache
    // the sessions draw their pages from, so it is chosen while there are none.
    let tier = open_checkpoint_tier(model.as_mut(), args);
    // Shared with the request threads: `/cache` reads it and a queued request prefetches.
    let cache = tier.as_ref().and_then(|_| model.kv_cache());

    // A vocabulary is how generated ids become text. Its absence is a warning
    // rather than a failure: the ids still reach the caller, and a checkpoint
    // whose vocabulary lives somewhere this flag was not pointed at is an
    // operator mistake that should be reported, not a reason to refuse to run.
    let vocabulary = load_vocabulary(args);

    // Before the listener binds, and only when the operator named one. A tower
    // that will not load, or that does not pair with this checkpoint, ends the
    // process here rather than failing the first request that carries a picture.
    let vision = open_vision_tower(&backend, args, model.as_mut(), open_tower)?;
    // Taken before the tower is handed to the engine thread: a tower that is
    // open is the authority on how its own pictures are preprocessed, and a
    // server without one falls back to what the checkpoint's config declares.
    let vision_spec = vision.as_ref().map(|tower| tower.preprocess().clone());
    // The same container, asked for the other tower it may carry.
    let audio = open_audio_tower(&backend, args, model.as_mut(), open_audio)?;
    // Every weight this process will read is resident now, the tower's too, so
    // whatever the backend kept to make those imports fast can go.
    backend.finish_imports();
    let memory = report_memory(&backend);
    log_kv_capacity(model.as_ref(), args, tier.is_some());

    let model_id = args.model_id();
    let scheduler = scheduler_config(args)?;
    // The renderer reads the store the scheduler writes.
    let turn_replay = scheduler.turn_replay.clone();
    let reasoning_scheme = args.reasoning_parser.reasoning(defaults.reasoning);
    let tool_call_scheme = args.tool_call_parser.tools(defaults.tools);
    tracing::info!(
        reasoning = ?reasoning_scheme,
        tools = ?tool_call_scheme,
        thinking_default = defaults.thinking_default,
        "output parsers configured"
    );
    // One cache, read by the renderer and filled by the engine thread, so a
    // picture is decoded and encoded once however many turns carry it.
    let media_cache = (vision.is_some() || audio.is_some()).then(|| {
        Arc::new(media_cache::MediaCache::new(
            args.vision_cache_mb.saturating_mul(1 << 20),
        ))
    });
    let (engine, join) = scheduler::spawn_engine_with_checkpoints(
        Arc::clone(&backend),
        model,
        media::MediaTowers {
            vision,
            audio: audio.clone(),
            cache: media_cache.clone(),
        },
        vocabulary.clone(),
        &model_id,
        args.ctx,
        scheduler,
        (reasoning_scheme, tool_call_scheme),
    );

    let mut state = AppState::new(engine, model_id, args.ctx);
    if let Some(memory) = memory {
        state = state.with_memory(memory);
    }
    if let Some(preset) = sampling_preset {
        state = state.with_sampling_preset(preset);
    }
    if let Some(tier) = tier {
        state = state.with_checkpoint_tier(tier);
    }
    state.kv = cache;
    if let Some(vocabulary) = vocabulary {
        if let Some(renderer) = build_renderer(
            args,
            &config,
            Arc::clone(&vocabulary),
            placement,
            vision_spec.as_ref(),
            audio,
            turn_replay,
        ) {
            let mut renderer = defaults.configure(renderer);
            if let Some(cache) = media_cache {
                renderer = renderer.with_media_cache(cache);
            }
            state = state.with_renderer(Arc::new(renderer));
        }
        state = state.with_vocabulary(vocabulary);
    }
    Ok((state, join))
}

/// The scheduler's configuration: its environment, then the flags that
/// override it.
fn scheduler_config(args: &ServeArgs) -> anyhow::Result<scheduler::SchedulerConfig> {
    // The one place the scheduler's environment is read. Once, at boot: a knob
    // that could change between two ticks of the same request would be a policy
    // nobody configured.
    let mut scheduler = scheduler::SchedulerConfig::from_env();
    // A flag rather than one of the scheduler's own variables: the same count
    // is what a `--gpu-memory-utilization` pool set sessions aside for, and the
    // two must not be able to disagree.
    scheduler.max_num_seqs = args.max_num_seqs;
    scheduler.store.pump_bytes = args.kvstore_pump_mb.saturating_mul(1024 * 1024);
    scheduler.store.drain_timeout = std::time::Duration::from_secs(args.kvstore_drain_secs);
    scheduler.stats_interval = (args.log_stats_interval > 0)
        .then(|| std::time::Duration::from_secs(args.log_stats_interval));
    // Validated at boot, so the widths are known to be servable here.
    let widths = args.prefill_widths()?;
    scheduler.solo_chunk = widths.solo_chunk;
    scheduler.mixed_quantum = widths.mixed_quantum;
    tracing::info!(
        solo_chunk = widths.solo_chunk,
        mixed_quantum = widths.mixed_quantum,
        "prefill widths"
    );
    // Written by the scheduler as turns retire, read by the renderer. Zero is
    // the operator turning it off, not a store that keeps nothing.
    scheduler.turn_replay = (args.turn_replay_tokens > 0).then(|| {
        Arc::new(tokenizer::turn_replay::TurnReplay::new(
            args.turn_replay_tokens,
        ))
    });
    Ok(scheduler)
}

/// Loads the checkpoint, timing it.
///
/// Under a ledger scope: a family's loader labels its weights by class, and
/// whatever it leaves unlabelled is still a weight.
fn load_checkpoint(
    backend: &Arc<dyn shiso_core::Backend>,
    config: &shiso_core::HfConfig,
    args: &ServeArgs,
    model: &std::path::Path,
) -> anyhow::Result<model_factory::Loaded> {
    let _tag = shiso_core::ledger::scope(shiso_core::MemTag::WeightsOther);
    let started = std::time::Instant::now();
    let loaded = model_factory::load_model(backend, model, config, args)
        .with_context(|| format!("loading the checkpoint at {}", model.display()))?;
    tracing::info!(
        seconds = started.elapsed().as_secs_f64(),
        "the checkpoint is loaded"
    );
    Ok(loaded)
}

/// The KV cache's size, one info line at boot: the pool in tokens and bytes,
/// and how many full-context requests it holds at once.
///
/// Nothing for a model with no paged cache.
#[expect(
    clippy::cast_precision_loss,
    reason = "sizes and ratios printed for a human to read to two decimal places"
)]
fn log_kv_capacity(
    model: &dyn shiso_engine::ckpt::Checkpointed,
    args: &ServeArgs,
    checkpoint_tier: bool,
) {
    let Some(cache) = model.kv_cache() else {
        return;
    };
    let stats = cache.stats();
    let geometry = cache.geometry();
    let tokens = u64::from(stats.pool_pages) * u64::from(geometry.page_tokens);
    let pool_bytes = u64::from(stats.pool_pages).saturating_mul(geometry.page_bytes);
    let ckpt_bytes = u64::from(stats.ckpt_capacity).saturating_mul(geometry.ckpt_bytes);
    let gib = |bytes: u64| format!("{:.2}", bytes as f64 / f64::from(1u32 << 30));
    let concurrency = tokens as f64 / f64::from(args.ctx.max(1));
    tracing::info!(
        tokens,
        gib = %gib(pool_bytes),
        kib_per_token = format_args!(
            "{:.1}",
            geometry.page_bytes as f64 / f64::from(geometry.page_tokens.max(1)) / 1024.0
        ),
        ctx = args.ctx,
        max_num_seqs = args.max_num_seqs,
        max_concurrency = format_args!("{concurrency:.2}x"),
        checkpoint_slots = stats.ckpt_capacity,
        checkpoint_gib = %gib(ckpt_bytes),
        checkpoint_tier,
        "KV cache"
    );
}

/// Records what the device says now in the backend's memory ledger.
fn mark_memory(backend: &dyn shiso_core::Backend, label: &'static str) {
    if let Some(ledger) = backend.ledger() {
        ledger.mark(label, backend.memory_info());
    }
}

/// Marks the end of the load and logs the memory ledger, one line per
/// consumer, returning the handle `/metrics` reads it through.
fn report_memory(backend: &Arc<dyn shiso_core::Backend>) -> Option<state::MemorySource> {
    mark_memory(backend.as_ref(), "ready");
    let memory = state::MemorySource::new(Arc::clone(backend))?;
    if let Some(report) = memory.report() {
        report.log();
    }
    Some(memory)
}

/// Opens the vision tower, when the operator named one.
///
/// `Ok(None)` is a text-only server: no `--vision`, so no file is read and no
/// device memory is taken. Every other outcome is decided here, before anything
/// is serving - a tower that will not load and a tower that does not pair with
/// this checkpoint are both configuration mistakes, and the request that would
/// otherwise discover them is one whose answer would be fluent and wrong.
fn open_vision_tower(
    backend: &Arc<dyn shiso_core::Backend>,
    args: &ServeArgs,
    model: &mut dyn shiso_engine::ckpt::Checkpointed,
    open: model_factory::TowerOpener,
) -> anyhow::Result<Option<vision::VisionTower>> {
    let Some(path) = args.vision.as_ref() else {
        return Ok(None);
    };
    let _tag = shiso_core::ledger::scope(shiso_core::MemTag::VisionWeights);
    // Which container the tower lives in, and how its dimensions read back out,
    // is the family's answer and not this function's: the opener came out of
    // the same arm of the factory that built the model.
    let tower = open(backend, path).map_err(|source| vision::TowerError::Load {
        path: path.display().to_string(),
        source,
    })?;
    let tower = vision::VisionTower::from_tower(tower);
    let hidden = shiso_core::Model::info(model).hidden;
    tower.check_against(hidden)?;
    if shiso_core::Model::media_mut(model).is_none() {
        return Err(anyhow!(
            "--vision names a tower but this checkpoint has no media path to splice \
             its rows into; serve it without the flag or serve a checkpoint that has one"
        ));
    }
    tracing::info!(
        path = %path.display(),
        out_hidden = tower.out_hidden(),
        merge = tower.spatial_merge_size(),
        marker = tower.placement().image_marker,
        device_bytes = tower.device_bytes(),
        "the vision tower is resident"
    );
    Ok(Some(tower))
}

/// Opens the audio tower from the `--vision` container, when it carries one.
///
/// # Errors
///
/// A container that claims a tower and will not load it, or a tower that does
/// not pair with this checkpoint: both end boot here, before a request carrying
/// a clip is answered from its words alone.
// `Ok(None)` is audio input off: no `--vision`, a family that does not hear,
// or a container exported without the tower. Each is logged.
pub fn open_audio_tower(
    backend: &Arc<dyn shiso_core::Backend>,
    args: &ServeArgs,
    model: &mut dyn shiso_engine::ckpt::Checkpointed,
    open: model_factory::AudioTowerOpener,
) -> anyhow::Result<Option<Arc<dyn shiso_audio::AudioTower>>> {
    let Some(path) = args.vision.as_ref() else {
        return Ok(None);
    };
    let _tag = shiso_core::ledger::scope(shiso_core::MemTag::VisionWeights);
    let model_factory::AudioTowerFound::Tower(tower) = open(backend, path)
        .with_context(|| format!("opening the audio tower in {}", path.display()))?
    else {
        tracing::info!(
            path = %path.display(),
            "the tower container carries no audio tower; audio input is off"
        );
        return Ok(None);
    };
    let hidden = shiso_core::Model::info(model).hidden;
    let width = u32::try_from(tower.out_hidden()).unwrap_or(u32::MAX);
    if width != hidden {
        return Err(anyhow!(
            "the audio tower projects to {width}-wide rows but this checkpoint's residual \
             stream is {hidden} wide; they are not a pair"
        ));
    }
    if shiso_core::Model::media_mut(model).is_none() {
        return Err(anyhow!(
            "the tower container carries an audio tower but this checkpoint has no media \
             path to splice its rows into"
        ));
    }
    tracing::info!(
        path = %path.display(),
        out_hidden = width,
        max_seconds = tower.max_seconds(),
        device_bytes = tower.device_bytes(),
        "the audio tower is resident"
    );
    Ok(Some(tower))
}

/// Turns the checkpoint tier on, when the operator named a directory for it.
///
/// Reports rather than fails: a directory that cannot be opened costs every
/// request a prefill it would not otherwise have paid, and nothing else. A
/// server that refused to start over it would trade a slower service for no
/// service.
fn open_checkpoint_tier(
    model: &mut dyn shiso_engine::ckpt::Checkpointed,
    args: &ServeArgs,
) -> Option<state::CheckpointTier> {
    let dir = args.kvstore_dir.as_ref()?;
    let mut options = shiso_engine::ckpt::CkptOptions::new();
    options.budget_mb = args.kvstore_budget_mb;
    options.min_tokens = args.kvstore_min_tokens;
    options.staging_mb = args.kvstore_staging_mb;
    let config = shiso_engine::ckpt::CkptConfig::new(dir.clone()).with_options(options);
    match model.kv_enable(&config) {
        Ok(()) => {
            let budget_bytes = model
                .kv_cache()
                .and_then(|cache| cache.disk_stats())
                .map_or(0, |stats| stats.budget_bytes);
            tracing::info!(
                dir = %dir.display(),
                budget_bytes,
                min_tokens = options.min_tokens,
                staging_mb = options.staging_mb,
                "the checkpoint tier is open"
            );
            Some(state::CheckpointTier {
                dir: dir.clone(),
                budget_bytes,
                min_tokens: options.min_tokens,
            })
        }
        Err(error) => {
            tracing::error!(
                dir = %dir.display(),
                %error,
                "the checkpoint tier could not be opened; this server will prefill \
                 every prompt and /cache will report the tier as off"
            );
            None
        }
    }
}

/// A build that named two devices, refused at compile time.
///
/// One binary serves one device: the backend is opened once at startup and
/// every model holds it for the process's life, so "both features on" has no
/// meaning that is not a precedence rule invented here. Saying so with the
/// compiler is better than picking one and logging about it.
#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("features `metal` and `cuda` name two devices; build with one");

/// Opens the compute device this build was compiled with.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn open_backend(_args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    let backend = shiso_metal::MetalBackend::new().context("opening the Metal device")?;
    Ok(backend)
}

/// Opens the backend instance the speech models run on: the same device, a
/// command stream of its own, and no launch recording.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
fn open_audio_backend(args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    open_backend(args)
}

/// Opens the backend instance the speech models run on. Decode graphs stay
/// off here: the speech models' launches are never replayed.
#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn open_audio_backend(_args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    let options = shiso_cuda::CudaOptions::default().with_decode_graphs(false);
    let backend = shiso_cuda::CudaBackend::new(0, options)
        .context("opening the CUDA device for the speech models")?;
    Ok(backend)
}

/// As [`open_backend`]: this build has no device for the speech models.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn open_audio_backend(args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    open_backend(args)
}

/// Opens the NVIDIA device.
///
/// Ordinal zero and the default options: which device to run on is a placement
/// decision that belongs outside the process (`CUDA_VISIBLE_DEVICES`), not a
/// flag this server would then have to keep meaningful across a multi-device
/// box it does not otherwise model.
#[cfg(all(feature = "cuda", not(feature = "metal")))]
fn open_backend(args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    let options = shiso_cuda::CudaOptions::default().with_decode_graphs(!args.no_decode_graphs);
    let backend = shiso_cuda::CudaBackend::new(0, options).context("opening the CUDA device")?;
    Ok(backend)
}

/// Refuses to serve, because this build has no compute device.
///
/// A typed failure at the one place a device is needed, rather than a
/// feature-gated router or a build that does not compile: the whole crate type
/// checks without a backend, which is what lets every test above this line run
/// on a machine with no device at all.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
fn open_backend(_args: &ServeArgs) -> anyhow::Result<Arc<dyn shiso_core::Backend>> {
    Err(anyhow!(
        "this build has no compute backend; rebuild with --features metal or --features cuda"
    ))
}

/// How the server writes `tojson` in every chat template it renders.
///
/// Spaced, for every family: each template this server renders was written for
/// the Python engine, whose `tojson` separates with `", "` and `": "` and leaves
/// `<`, `>`, `&` and `'` alone. The template engine's built-in escapes those four
/// as `\u003c`-style sequences and packs the separators, so a tool schema that
/// said `<content> & metadata` reached the model in a spelling it was never
/// trained on - and so did every non-string argument replayed from history.
///
/// One constant, read by the boot path and by the parity tests, so the tests
/// render exactly what production sends: a test that picked its own style would
/// pass while the server rendered something else.
pub const SERVER_TOOL_JSON_STYLE: tokenizer::jinja_helpers::ToolJsonStyle =
    tokenizer::jinja_helpers::ToolJsonStyle::Spaced;

/// Builds the prompt renderer, reporting rather than failing when the checkpoint
/// ships no chat template.
///
/// A server with no template still serves everything that does not generate,
/// and the generation routes answer 503 naming what is missing. The alternative
/// (inventing a prompt format) produces fluent answers to a prompt the model was
/// never trained on, which is the one failure nobody can see in the output.
fn build_renderer(
    args: &ServeArgs,
    config: &shiso_core::HfConfig,
    vocabulary: Arc<tokenizer::Vocabulary>,
    placement: shiso_vision::MediaPlacement,
    spec: Option<&shiso_vision::PreprocessSpec>,
    audio: Option<Arc<dyn shiso_audio::AudioTower>>,
    replay: Option<Arc<tokenizer::turn_replay::TurnReplay>>,
) -> Option<tokenizer::render_prompt::PromptRenderer> {
    let template = load_chat_template(args)?;
    let renderer = match tokenizer::render_prompt::PromptRenderer::new(
        &template,
        SERVER_TOOL_JSON_STYLE,
        vocabulary,
    ) {
        Ok(renderer) => renderer,
        Err(error) => {
            tracing::error!(%error, "the chat template does not compile; no prompt can be rendered");
            return None;
        }
    };
    let renderer = renderer
        .with_remote_media(args.vision_allow_remote_images)
        .with_video(args.ffmpeg_policy(), args.video_fps)
        .with_turn_replay(replay);
    // The clip's frontend and token rule are the tower's, so a renderer that
    // sizes a clip's pad run needs the tower itself; without one an audio part
    // is refused.
    let renderer = match audio {
        Some(tower) => renderer.with_audio_tower(
            tower,
            placement,
            args.audio_limits(),
            args.audio_ffmpeg_policy(),
        ),
        None => renderer,
    };
    // The tower geometry is what turns one media marker into the right number of
    // pad tokens, so a renderer without it must refuse media rather than send a
    // prompt whose pad run disagrees with the embedding rows. An open tower is
    // the authority; without one the checkpoint's own config is, so that a
    // text-only server still renders the prompt its template writes.
    if let Some(spec) = spec {
        return Some(renderer.with_tower(spec, placement, args.image_limits()));
    }
    match config.vision_config() {
        Ok(Some(vision)) => Some(renderer.with_vision(&vision, placement, args.image_limits())),
        Ok(None) => Some(renderer),
        Err(error) => {
            tracing::warn!(
                %error,
                "the vision block of the config did not parse; this server will \
                 refuse media rather than guess its geometry"
            );
            Some(renderer)
        }
    }
}

/// The sampling preset from the checkpoint's generation config.
///
/// `Ok(None)` when there is no file to read, which keeps neutral sampling and
/// says so in the log: plenty of checkpoints ship none. A file that is there
/// and does not parse is an error, because serving on with neutral sampling
/// would silently be a different model from the one the operator pointed at.
fn load_sampling_preset(args: &ServeArgs) -> anyhow::Result<Option<shiso_core::SamplingParams>> {
    let path = args.generation_config_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let preset = api::pipeline::preset_from_generation_config(&text)
                .with_context(|| format!("parsing the generation config at {}", path.display()))?;
            tracing::info!(
                path = %path.display(),
                temperature = preset.temperature,
                top_k = preset.top_k.map(std::num::NonZeroU32::get),
                top_p = preset.top_p,
                min_p = preset.min_p,
                "sampling preset from the generation config"
            );
            Ok(Some(preset))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(
                path = %path.display(),
                "no generation config; a caller who names no sampling knob gets neutral sampling"
            );
            Ok(None)
        }
        Err(error) => Err(anyhow!(error).context(format!(
            "reading the generation config at {}",
            path.display()
        ))),
    }
}

/// The chat template text, from the file the operator named or from the two
/// places a checkpoint conventionally keeps one.
fn load_chat_template(args: &ServeArgs) -> Option<String> {
    let named = args.chat_template_path();
    match std::fs::read_to_string(&named) {
        Ok(template) => {
            tracing::info!(path = %named.display(), "chat template loaded");
            return Some(template);
        }
        Err(error) => tracing::debug!(path = %named.display(), %error, "no chat template file"),
    }

    let config = args.tokenizer_config_path();
    let template = std::fs::read_to_string(&config)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| {
            value
                .get("chat_template")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        });
    if let Some(template) = template {
        tracing::info!(
            path = %config.display(),
            "chat template loaded from the tokenizer config"
        );
        return Some(template);
    }
    tracing::warn!(
        looked_in = %named.display(),
        and = %config.display(),
        "no chat template was found; the generation routes will refuse rather \
         than invent a prompt format"
    );
    None
}

/// Loads the vocabulary, reporting rather than failing when it is absent.
fn load_vocabulary(args: &ServeArgs) -> Option<Arc<tokenizer::Vocabulary>> {
    let path = args.tokenizer_path();
    match tokenizer::Vocabulary::from_file(&path) {
        Ok(vocabulary) => Some(Arc::new(vocabulary)),
        Err(error) => {
            tracing::warn!(
                path = %path.display(),
                %error,
                "no vocabulary was loaded; generated tokens will be reported as ids only"
            );
            None
        }
    }
}
