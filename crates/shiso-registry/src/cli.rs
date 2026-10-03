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

    /// Compile and validate the registry index and manifests.
    Build(BuildArgs),
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

    /// Output directory for compiled manifests and index.
    /// Defaults to `<repo>/registry/v1`, the path the registry repo serves.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Allow building with uncommitted changes in `recipes/`.
    #[arg(long)]
    pub allow_dirty: bool,
}
