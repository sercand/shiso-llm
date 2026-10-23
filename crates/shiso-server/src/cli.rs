//! The command line: the subcommands, and the recipe-default merge.
//!
//! [`ServeOptions`] and the enums behind it live in `shiso_runtime::options`;
//! they are re-exported here. [`ServeArgs`] is the options plus the one
//! argument only a command line has, the positional model reference.

use std::ops::{Deref, DerefMut};

pub use shiso_router::cli::RouterArgs;
pub use shiso_runtime::hub::{ModelArgKind, classify_model_arg};
pub use shiso_runtime::options::*;
pub use shiso_runtime::recipe_defaults::apply_recipe_defaults_unless_set;

use clap::parser::ValueSource;
use shiso_hub::recipe::ServerDefaults;

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

/// Everything `shiso serve` accepts: the runtime's [`ServeOptions`] and the
/// positional model reference.
///
/// Dereferences to the options, so a field or a method of either reads the same
/// off this. Never give `ServeArgs` an inherent method or field named like one
/// on [`ServeOptions`]: it would shadow the options' silently for every caller
/// that reaches it through the `Deref`.
#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// The serve options themselves.
    #[command(flatten)]
    pub options: ServeOptions,
    /// Model reference (registry recipe) or path.
    ///
    /// If it contains `/` or starts with `.` or `~`, it is treated as a path (`--model`).
    /// Otherwise, it resolves to an installed registry model variant.
    #[arg(value_name = "MODEL", conflicts_with = "model")]
    pub model_ref: Option<String>,
}

impl Deref for ServeArgs {
    type Target = ServeOptions;

    fn deref(&self) -> &ServeOptions {
        &self.options
    }
}

impl DerefMut for ServeArgs {
    fn deref_mut(&mut self) -> &mut ServeOptions {
        &mut self.options
    }
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
    /// Which device the model will serve on, so the variant pulled is the one
    /// `shiso serve --device` runs: `auto`, or one this build compiled.
    #[arg(long, env = "SHISO_DEVICE", value_enum, default_value_t = shiso_runtime::device::DeviceChoice::Auto)]
    pub device: shiso_runtime::device::DeviceChoice,
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

/// Merges recipe `server_defaults` into `args` respecting precedence:
/// `explicit CLI flag > SHISO_* env var > recipe server_defaults > built-in default` (§6.6).
///
/// Overwrites a field only when the user did not set it: `matches` is what the
/// command line parsed, and `None` means nothing was set explicitly.
///
/// Tests only: the binary and embedders go through
/// [`shiso_runtime::hub::resolve_serve_model`] with an [`ExplicitlySet`].
#[doc(hidden)]
pub fn apply_recipe_defaults(
    args: &mut ServeOptions,
    matches: Option<&clap::ArgMatches>,
    defaults: &ServerDefaults,
) {
    let explicit = matches.map(explicitly_set).unwrap_or_default();
    apply_recipe_defaults_unless_set(args, &explicit, defaults);
}

/// The options the command line set itself, by flag or by its `SHISO_*`
/// variable, as the names [`apply_recipe_defaults_unless_set`] asks about.
#[must_use]
pub fn explicitly_set(matches: &clap::ArgMatches) -> ExplicitlySet {
    matches
        .ids()
        .filter(|id| {
            matches!(
                matches.value_source(id.as_str()),
                Some(ValueSource::CommandLine | ValueSource::EnvVariable)
            )
        })
        .map(clap::Id::as_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe() -> ServerDefaults {
        ServerDefaults {
            ctx: Some(32768),
            max_num_seqs: Some(64),
            ..ServerDefaults::default()
        }
    }

    /// A flag given the very value the built-in default has is still the
    /// operator's choice: the recipe must not move it.
    #[test]
    fn a_flag_equal_to_the_default_still_beats_the_recipe() {
        use clap::{CommandFactory as _, FromArgMatches as _};
        let ctx = DEFAULT_CTX.to_string();
        let matches =
            Cli::command().get_matches_from(["shiso", "serve", "-m", "/m", "--ctx", ctx.as_str()]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let mut args = ServeArgs::from_arg_matches(serve).expect("parses");
        assert_eq!(args.ctx, DEFAULT_CTX);
        apply_recipe_defaults(&mut args, Some(serve), &recipe());
        assert_eq!(
            args.ctx, DEFAULT_CTX,
            "an explicit value equal to the default"
        );
        assert_eq!(args.max_num_seqs, 64, "the untouched one takes the recipe");
    }

    #[test]
    fn explicitly_set_reads_flags_and_skips_defaults() {
        use clap::CommandFactory as _;
        let matches = Cli::command().get_matches_from(["shiso", "serve", "-m", "/m", "--ctx", "8"]);
        let serve = matches.subcommand_matches("serve").expect("serve");
        let set = explicitly_set(serve);
        assert!(set.contains("ctx"));
        assert!(set.contains("model"));
        assert!(!set.contains("max_num_seqs"));
    }
}
