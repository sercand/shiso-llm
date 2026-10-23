//! The command line: the subcommands, and the recipe-default merge.
//!
//! [`ServeArgs`] and the enums behind it live in `shiso_runtime::cli`; they are
//! re-exported here so `crate::cli::ServeArgs` keeps naming them.

use std::path::PathBuf;

pub use shiso_router::cli::RouterArgs;
pub use shiso_runtime::cli::*;

use clap::parser::ValueSource;
use shiso_hub::recipe::{ExpertResidency, ServerDefaults};

/// `shiso`, the inference server.
#[derive(Debug, clap::Parser)]
#[command(name = "shiso", version, about = "shiso inference server")]
pub struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// The subcommands this binary offers.
///
/// One today. It is a subcommand rather than a bare flag set because the second
/// one - a tokenizer probe, a checkpoint dump - should not force every existing
/// invocation to be rewritten.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Serve a checkpoint over HTTP.
    Serve(Box<ServeArgs>),
    /// Run an LLM router proxying requests to third-party or local backends.
    Router(Box<RouterArgs>),
    /// Download a model from the registry.
    Pull(PullArgs),
    /// List installed models, or the registry's catalog with `--remote`.
    List(ListArgs),
    /// Describe a model's recipe and variants.
    Info(InfoArgs),
    /// Re-hash an installed model against its recipe.
    Verify(VerifyArgs),
    /// Remove an installed model.
    Rm(RmArgs),
}

/// Output format for download progress (§6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum ProgressStyle {
    /// Bars on TTY, plain text when redirected.
    #[default]
    Auto,
    /// Interactive multi-bar progress display.
    Bars,
    /// Plain line-oriented progress logging.
    Plain,
    /// NDJSON stream for programmatic integration.
    Json,
}

/// Shared hub arguments for registry and downloading commands (§6).
#[derive(Debug, Clone, clap::Args)]
pub struct HubArgs {
    /// Local storage home directory.
    #[arg(long, env = "SHISO_HOME", default_value = "~/.shiso")]
    pub home: String,

    /// Static HTTPS registry URL.
    #[arg(
        long,
        env = "SHISO_REGISTRY_URL",
        default_value = "https://hub.shiso.run/registry/v1"
    )]
    pub registry_url: String,

    /// Hugging Face endpoint URL.
    #[arg(
        long,
        env = "SHISO_HF_ENDPOINT",
        default_value = "https://huggingface.co"
    )]
    pub hf_endpoint: String,

    /// Hugging Face bearer token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Maximum concurrent file downloads.
    #[arg(long, env = "SHISO_PULL_JOBS", default_value_t = 4)]
    pub jobs: usize,

    /// Progress reporting style (`auto`, `bars`, `plain`, `json`).
    #[arg(long, env = "SHISO_PROGRESS", default_value = "auto")]
    pub progress: ProgressStyle,
}

/// Arguments for `shiso pull <ref>`.
#[derive(Debug, Clone, clap::Args)]
pub struct PullArgs {
    /// Model reference to download (`<name-or-alias>[:<variant-key>]`).
    pub model_ref: String,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso list [--remote] [--task <t>] [--tag <t>]`.
#[derive(Debug, Clone, clap::Args)]
pub struct ListArgs {
    /// Query remote registry index instead of local installations.
    #[arg(long)]
    pub remote: bool,
    /// Filter models by task.
    #[arg(long)]
    pub task: Option<String>,
    /// Filter models by tag.
    #[arg(long)]
    pub tag: Option<String>,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso info <ref>`.
#[derive(Debug, Clone, clap::Args)]
pub struct InfoArgs {
    /// Model reference to describe.
    pub model_ref: String,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso verify <ref> [--repair]`.
#[derive(Debug, Clone, clap::Args)]
pub struct VerifyArgs {
    /// Model reference to verify.
    pub model_ref: String,
    /// Re-download corrupt or missing files.
    #[arg(long)]
    pub repair: bool,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso rm <ref> [--force]`.
#[derive(Debug, Clone, clap::Args)]
pub struct RmArgs {
    /// Model reference to remove (`name` removes all variants, `name:key` removes one).
    pub model_ref: String,
    /// Remove even if model is in use by a running server.
    #[arg(short, long)]
    pub force: bool,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Classification of a positional model argument (§6.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArgKind {
    /// Path to a checkpoint file or directory.
    Path(PathBuf),
    /// Model reference from the registry.
    RegistryRef(String),
}

/// Classifies a positional model argument as a path or registry reference (§6.6).
#[must_use]
pub fn classify_model_arg(arg: &str) -> ModelArgKind {
    if arg.contains('/') || arg.starts_with('.') || arg.starts_with('~') {
        ModelArgKind::Path(PathBuf::from(arg))
    } else {
        ModelArgKind::RegistryRef(arg.to_string())
    }
}

/// Merges recipe `server_defaults` into `args` respecting precedence:
/// `explicit CLI flag > SHISO_* env var > recipe server_defaults > built-in default` (§6.6).
///
/// Overwrites a field only when `matches.value_source(id) == Some(ValueSource::DefaultValue)`
/// (or when `matches` is `None`).
pub fn apply_recipe_defaults(
    args: &mut ServeArgs,
    matches: Option<&clap::ArgMatches>,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| defaulted(matches, id);

    if let Some(ctx) = defaults.ctx
        && check_override("ctx")
    {
        args.ctx = u32::try_from(ctx).unwrap_or(args.ctx);
        tracing::info!(ctx = args.ctx, "applied recipe default for ctx");
    }

    if let Some(max_num_seqs) = defaults.max_num_seqs
        && check_override("max_num_seqs")
    {
        args.max_num_seqs = max_num_seqs;
        tracing::info!(
            max_num_seqs = args.max_num_seqs,
            "applied recipe default for max_num_seqs"
        );
    }

    if let Some(spec_k) = defaults.spec_k
        && check_override("spec_k")
    {
        args.spec_k = u32::try_from(spec_k).unwrap_or(args.spec_k);
        tracing::info!(spec_k = args.spec_k, "applied recipe default for spec_k");
    }

    if let Some(turn_replay_tokens) = defaults.turn_replay_tokens
        && check_override("turn_replay_tokens")
    {
        args.turn_replay_tokens = turn_replay_tokens;
        tracing::info!(
            turn_replay_tokens = args.turn_replay_tokens,
            "applied recipe default for turn_replay_tokens"
        );
    }

    if let Some(vision_max_pixels) = defaults.vision_max_pixels
        && check_override("vision_max_pixels")
    {
        args.vision_max_pixels = vision_max_pixels;
        tracing::info!(
            vision_max_pixels = args.vision_max_pixels,
            "applied recipe default for vision_max_pixels"
        );
    }

    if let Some(video_fps) = defaults.video_fps
        && check_override("video_fps")
    {
        args.video_fps = f64::from(video_fps);
        tracing::info!(
            video_fps = args.video_fps,
            "applied recipe default for video_fps"
        );
    }

    if let Some(video_max_frames) = defaults.video_max_frames
        && check_override("video_max_frames")
    {
        args.video_max_frames = video_max_frames;
        tracing::info!(
            video_max_frames = args.video_max_frames,
            "applied recipe default for video_max_frames"
        );
    }

    if let Some(asr_lookahead) = defaults.asr_lookahead
        && check_override("asr_lookahead")
    {
        args.asr_lookahead = asr_lookahead;
        tracing::info!(
            asr_lookahead = args.asr_lookahead,
            "applied recipe default for asr_lookahead"
        );
    }

    if let Some(asr_batch) = defaults.asr_batch
        && check_override("asr_batch")
    {
        args.asr_batch = asr_batch;
        tracing::info!(
            asr_batch = args.asr_batch,
            "applied recipe default for asr_batch"
        );
    }

    apply_engine_defaults(args, matches, defaults);
}

/// Whether flag `id` still holds its built-in default, which is the only case
/// a recipe default may replace.
fn defaulted(matches: Option<&clap::ArgMatches>, id: &str) -> bool {
    matches.is_none_or(|m| m.value_source(id) == Some(ValueSource::DefaultValue))
}

/// The recipe defaults that tune the engine rather than the request surface:
/// expert residency, speculation and capture, and the memory and prefill
/// budgets.
fn apply_engine_defaults(
    args: &mut ServeArgs,
    matches: Option<&clap::ArgMatches>,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| defaulted(matches, id);

    if let Some(residency) = defaults.expert_residency
        && check_override("expert_residency")
    {
        args.expert_residency = match residency {
            ExpertResidency::Auto => ExpertResidencyChoice::Auto,
            ExpertResidency::Resident => ExpertResidencyChoice::Resident,
            ExpertResidency::Stream => ExpertResidencyChoice::Stream,
        };
        tracing::info!(?residency, "applied recipe default for expert_residency");
    }

    if let Some(kv) = defaults.kv_cache_dtype
        && check_override("kv_cache_dtype")
    {
        args.kv_cache_dtype = match kv {
            shiso_hub::recipe::KvCacheDtype::F16 => KvCacheDtype::F16,
            shiso_hub::recipe::KvCacheDtype::Fp8 => KvCacheDtype::Fp8,
        };
        tracing::info!(?kv, "applied recipe default for kv_cache_dtype");
    }

    if let Some(reserve) = defaults.expert_reserve_gb
        && check_override("expert_reserve_gb")
    {
        args.expert_reserve_gb = reserve;
        tracing::info!(reserve, "applied recipe default for expert_reserve_gb");
    }

    if let Some(slots) = defaults.expert_slots
        && check_override("expert_slots")
    {
        args.expert_slots = slots;
        tracing::info!(slots, "applied recipe default for expert_slots");
    }

    if let Some(spec_lookup) = defaults.spec_lookup
        && check_override("spec_lookup")
    {
        args.spec_lookup = spec_lookup;
        tracing::info!(spec_lookup, "applied recipe default for spec_lookup");
    }

    if let Some(capture_verify) = defaults.capture_verify
        && check_override("capture_verify")
    {
        args.capture_verify = capture_verify;
        tracing::info!(capture_verify, "applied recipe default for capture_verify");
    }

    if let Some(capture_batch) = defaults.capture_batch
        && check_override("capture_batch")
    {
        args.capture_batch = capture_batch;
        tracing::info!(capture_batch, "applied recipe default for capture_batch");
    }

    // An `Option` flag has no default value source, so "not given" is `None`.
    if let Some(fraction) = defaults.gpu_memory_utilization
        && args.gpu_memory_utilization.is_none()
    {
        args.gpu_memory_utilization = Some(fraction);
        tracing::info!(
            fraction,
            "applied recipe default for gpu_memory_utilization"
        );
    }

    if let Some(chunk) = defaults.solo_prefill_chunk
        && check_override("solo_prefill_chunk")
    {
        args.solo_prefill_chunk = chunk;
        tracing::info!(chunk, "applied recipe default for solo_prefill_chunk");
    }
}
