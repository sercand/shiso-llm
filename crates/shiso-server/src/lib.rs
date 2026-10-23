//! shiso-server: the command line, the HTTP surface, and the boundary between
//! them and the model.
//!
//! # The shape of the whole thing
//! A request arrives at [`router`] and reaches a handler in [`api`]. A
//! handler that needs a generation puts a message on the queue in
//! [`scheduler::queue`] and awaits a stream of [`shiso_ir::StreamEvent`]. One
//! OS thread on the other side of that queue owns the model and is the only
//! thing that ever calls a method on it.
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
pub mod cli;
pub mod decision;
pub mod error;
pub mod hub;
pub mod openai;
pub mod router;
pub mod shutdown;

// The engine moved to `shiso-runtime`; these keep `crate::scheduler::...` and
// `shiso_server::state::...` naming the same items.
pub use shiso_runtime::boot::{
    SERVER_TOOL_JSON_STYLE, boot_audio, boot_embedding, boot_image, boot_language_model,
    log_media_decoders, log_os_decoder, open_audio_tower,
};
pub use shiso_runtime::{
    audio, embed, ids, image, media, media_cache, model_factory, profile, response_store,
    scheduler, state, tokenizer, vision,
};

use std::sync::Arc;

use anyhow::anyhow;

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
    run_with_matches(cli, None).await
}

/// Runs the CLI command, merging matching argument sources when provided (§6.6).
///
/// # Errors
/// Returns an error if the server fails to start, model cannot be resolved,
/// or hub subcommand encounters an error.
pub async fn run_with_matches(cli: Cli, matches: Option<clap::ArgMatches>) -> anyhow::Result<()> {
    install_tracing();
    match cli.command {
        Command::Serve(mut args) => {
            let serve_matches = matches.as_ref().and_then(|m| m.subcommand_matches("serve"));
            let _lock = hub::resolve_serve_model(&mut args, serve_matches)?;
            serve(*args).await
        }
        Command::Router(args) => router(*args).await,
        Command::Pull(args) => hub::run_pull(args).await,
        Command::List(args) => hub::run_list(args).await,
        Command::Info(args) => hub::run_info(args).await,
        Command::Verify(args) => hub::run_verify(args).await,
        Command::Rm(args) => hub::run_rm(&args),
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
    log_media_decoders(&args);
    log_os_decoder(&args);

    // The language model is optional: a server started with only a recognizer
    // serves transcription and answers every generation route with the typed
    // "no model is loaded" error.
    // The speech models load first: a KV pool sized to a share of free device
    // memory must see their weights already resident, or it takes the memory
    // they then load into.
    let audio = boot_audio(&args)?;
    // The image model loads before the language model for the same reason the
    // speech models do: a KV pool sized to a share of free device memory must
    // see its weights already resident.
    let image = boot_image(&args)?;
    let (mut state, engine_thread) = match args.model.as_deref() {
        // An embedding checkpoint is not a language model: it gets its own
        // worker instead of the engine thread, and every generation route
        // answers "no model".
        Some(model) if shiso_models::embedder::is_embedder(model) => (
            AppState::without_language_model().with_embedding(boot_embedding(&args, model)?),
            None,
        ),
        Some(model) => {
            let (state, join) = boot_language_model(&args, model)?;
            (state, Some(join))
        }
        None => (AppState::without_language_model(), None),
    };
    if let Some(audio) = audio {
        state = state.with_audio(audio);
    }
    if let Some(image) = image {
        state = state.with_image(image);
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
