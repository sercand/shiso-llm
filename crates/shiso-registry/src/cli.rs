//! CLI arguments for `shiso-registry` (§7.1).

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Maintainer CLI for editing, inspecting, validating, and building the Shiso model registry.
#[derive(Debug, Parser)]
#[command(
    name = "shiso-registry",
    version,
    about = "maintainer tool for the Shiso model registry"
)]
pub(crate) struct Cli {
    /// Action to execute.
    #[command(subcommand)]
    pub(crate) command: Command,
}

/// Headless subcommands and web server invocation (§7.1).
#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Launch the embedded maintainer web UI on loopback.
    Serve(ServeArgs),

    /// Inspect a Hugging Face repository for commit SHA, file sizes, and LFS digests.
    Inspect(InspectArgs),

    /// Strictly validate all recipes under `recipes/`.
    Validate(ValidateArgs),

    /// Compile the registry index, manifests and the hub site.
    Build(BuildArgs),

    /// Measure a served variant's prefill and decode speed on this device and
    /// write the record under `benchmarks/`.
    Bench(Box<BenchArgs>),
}

/// Arguments for `serve`.
#[derive(Debug, clap::Args)]
pub(crate) struct ServeArgs {
    /// Path to repository root containing `recipes/`.
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,

    /// Local port to bind on 127.0.0.1.
    #[arg(long, default_value_t = 8095)]
    pub port: u16,

    /// Open web browser automatically at startup.
    #[arg(long)]
    pub open: bool,

    /// Require a per-run session token on every `/api` call.
    ///
    /// Off by default: the editor binds loopback only, and the Host and Origin
    /// checks that stop a web page from reaching it stay on either way. On,
    /// a random token is generated at startup and printed in the editor's URL,
    /// which keeps out other local users and processes as well.
    #[arg(long, env = "SHISO_REGISTRY_REQUIRE_TOKEN")]
    pub require_token: bool,

    /// `shiso` home holding installed variants, for the benchmark runner.
    #[arg(long, env = "SHISO_HOME")]
    pub home: Option<PathBuf>,
}

/// Arguments for `inspect <hf-repo>`.
#[derive(Debug, clap::Args)]
pub(crate) struct InspectArgs {
    /// Hugging Face repository (`owner/name`).
    pub repo: String,

    /// Git branch, tag, or revision to inspect.
    #[arg(long, default_value = "main")]
    pub revision: String,
}

/// Arguments for `validate`.
#[derive(Debug, clap::Args)]
pub(crate) struct ValidateArgs {
    /// Path to repository root containing `recipes/`.
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,
}

/// Arguments for `build`.
#[derive(Debug, clap::Args)]
pub(crate) struct BuildArgs {
    /// Path to repository root containing `recipes/`.
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,

    /// Site root to write: `index.html` and `models/` at its top, the
    /// registry under `registry/v1/`. Defaults to `<repo>`, the directory the
    /// host serves; a relative path is relative to `--repo`, not to the
    /// working directory.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Allow building with uncommitted changes in `recipes/`.
    #[arg(long)]
    pub allow_dirty: bool,

    /// Build from the `cards/` cache only; a model whose card is not cached
    /// gets a page without one.
    #[arg(long)]
    pub offline: bool,
}

/// Arguments for `bench`.
#[derive(Debug, clap::Args)]
pub(crate) struct BenchArgs {
    /// Model to measure, `name` or `name:variant`; the variant defaults to the
    /// only one installed.
    pub model: String,

    /// Path to repository root; the record goes under `benchmarks/`.
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,

    /// Base URL of the running `shiso serve` that serves the model.
    #[arg(long, env = "SHISO_BENCH_URL", default_value = "http://127.0.0.1:8090")]
    pub server_url: String,

    /// Device name for the record (`m1-max`, `gb10`); derived from the chip
    /// when absent.
    #[arg(long)]
    pub device: Option<String>,

    /// Task to measure (`chat`, `decision`, `image_generation`,
    /// `transcription`); defaults to the recipe's only measurable task.
    #[arg(long)]
    pub task: Option<String>,

    /// Prompt lengths to measure, in tokens (chat and decision).
    #[arg(long, value_delimiter = ',', default_value = "512,4096")]
    pub prompt_tokens: Vec<u64>,

    /// Tokens generated per run (chat).
    #[arg(long, default_value_t = 128)]
    pub decode_tokens: u64,

    /// Image sizes to measure, `WxH`; the server default when absent.
    #[arg(long, value_delimiter = ',')]
    pub image_size: Vec<String>,

    /// WAV clip to transcribe; synthesized speech on macOS when absent.
    #[arg(long)]
    pub audio: Option<PathBuf>,

    /// Measured runs per prompt length, after one discarded warm-up.
    #[arg(long, default_value_t = 3)]
    pub runs: u32,

    /// `shiso` home holding the installed variant.
    #[arg(long, env = "SHISO_HOME")]
    pub home: Option<PathBuf>,

    /// Version of the serving `shiso`; read from `shiso --version` when absent.
    #[arg(long)]
    pub shiso_version: Option<String>,

    /// A note stored with the record (a non-default flag, thermals).
    #[arg(long)]
    pub notes: Option<String>,
}
