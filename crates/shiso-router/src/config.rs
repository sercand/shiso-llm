use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::catalog::{BackendDef, GroupDef, ModelDef, OnShedWithoutFallback, SchedulingClass};

/// Configuration errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Failed to read config file from disk.
    #[error("failed to read config file {0}: {1}")]
    Read(PathBuf, std::io::Error),
    /// Failed to parse TOML configuration.
    #[error("failed to parse config: {0}")]
    Parse(#[from] toml::de::Error),
    /// Validation error in configuration values.
    #[error("invalid config: {0}")]
    Invalid(String),
    /// Referenced environment variable is unset.
    #[error("environment variable {0} referenced by config is not set")]
    MissingEnv(String),
}

fn invalid(msg: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(msg.into())
}

/// Complete router configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Address to listen on for proxy traffic.
    pub listen: SocketAddr,
    /// When set, the admin API + UI are served ONLY on this listener and
    /// disappear from the main one. When absent, admin shares `listen`.
    pub admin_listen: Option<SocketAddr>,
    /// Path to the SQLite database.
    pub sqlite_path: PathBuf,
    /// Maximum accepted request body size in bytes.
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: usize,
    /// Literal admin API token.
    pub admin_token: Option<String>,
    /// Environment variable containing the admin API token.
    pub admin_token_env: Option<String>,
    /// Seed data only: imported into the SQLite catalog the first time the
    /// router starts against an empty database, then ignored — groups,
    /// backends and models are owned by the admin API afterwards.
    /// Seed exclusive groups imported into an empty database.
    #[serde(default)]
    pub exclusive_groups: Vec<ExclusiveGroupCfg>,
    /// Seed backends imported into an empty database.
    #[serde(default)]
    pub backends: Vec<BackendCfg>,
    /// Seed models imported into an empty database.
    #[serde(default)]
    pub models: Vec<ModelCfg>,
}

fn default_max_body_bytes() -> usize {
    1024 * 1024 * 1024
}

/// API wire format kinds supported by backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// OpenAI Chat Completions, Responses, and Embeddings APIs.
    Openai,
    /// Anthropic Messages API.
    Anthropic,
}

/// A set of backends that share one accelerator and must not generate at
/// the same time. See `catalog::GroupDef` for what the timings mean.
/// Configuration for an exclusive accelerator group.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusiveGroupCfg {
    /// Group identifier.
    pub id: String,
    /// Anti-thrash minimum hold time in milliseconds.
    pub min_hold_ms: i64,
    /// Anti-starvation maximum hold time in milliseconds.
    pub max_hold_ms: i64,
    /// Stuck drain watchdog timeout in milliseconds.
    pub drain_timeout_ms: i64,
}

/// Backend configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendCfg {
    /// Unique backend name.
    pub name: String,
    /// API kinds this base_url serves. Accepts a bare string
    /// (`kind = "openai"`) or an array (`kind = ["openai", "anthropic"]`)
    /// for gateways that expose both wire formats on one URL.
    #[serde(deserialize_with = "kind_one_or_many")]
    pub kind: Vec<BackendKind>,
    /// Base URL of the upstream server.
    pub base_url: String,
    /// Literal API key for outbound authentication.
    pub api_key: Option<String>,
    /// Environment variable containing the API key.
    pub api_key_env: Option<String>,
    /// Anthropic version header value.
    pub anthropic_version: Option<String>,
    /// openai-kind only: inject `stream_options.include_usage` into
    /// streaming chat requests that don't set it, so usage can be logged.
    /// Escape hatch for OpenAI-compatible servers that reject the field.
    #[serde(default = "crate::catalog::default_inject_usage")]
    pub inject_usage: bool,
    /// "unrestricted" (default), "exempt", or "grouped" — see
    /// `catalog::SchedulingClass`.
    #[serde(default)]
    pub scheduling_class: SchedulingClass,
    /// Id of the `[[exclusive_groups]]` entry this backend belongs to.
    #[serde(default)]
    pub exclusive_group: Option<String>,
    /// Maximum in-flight requests permitted on this backend.
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    /// Generated token limit clamp for grouped backends.
    #[serde(default)]
    pub max_tokens_cap: Option<i64>,
    /// Output throughput estimate in tokens per second.
    #[serde(default)]
    pub tok_s_estimate: Option<f64>,
}

fn kind_one_or_many<'de, D>(de: D) -> Result<Vec<BackendKind>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(BackendKind),
        Many(Vec<BackendKind>),
    }
    Ok(match OneOrMany::deserialize(de)? {
        OneOrMany::One(k) => vec![k],
        OneOrMany::Many(v) => v,
    })
}

/// Configuration for a single model route.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCfg {
    /// Canonical model name.
    pub name: String,
    /// Additional inbound names for this same route.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Default pricing for every target of this model; targets may override.
    pub pricing: Option<PricingCfg>,
    /// Deadlines for predictive admission on grouped backends.
    pub qos: Option<QosCfg>,
    /// What to do when the deadline is missed and the route has no
    /// fallback: "wait" (default) or "error".
    #[serde(default)]
    pub on_shed_without_fallback: OnShedWithoutFallback,
    /// Upstream targets for this model route.
    pub targets: Vec<TargetCfg>,
}

/// Per-route QoS deadlines, in milliseconds of predicted queue wait.
/// An unset field means "wait indefinitely" for that class.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QosCfg {
    /// Interactive request deadline in milliseconds.
    pub max_wait_ms: Option<i64>,
    /// Selected by `x-router-qos: background`.
    pub background_max_wait_ms: Option<i64>,
}

/// A backend target for a model route.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetCfg {
    /// Backend name.
    pub backend: String,
    /// Model name rewrite on dispatch.
    pub model: Option<String>,
    /// Weighted selection weight.
    #[serde(default = "crate::catalog::default_weight")]
    pub weight: u32,
    /// Pricing override for this target.
    pub pricing: Option<PricingCfg>,
    /// Reachable only on a QoS shed, never by normal weighted selection.
    #[serde(default)]
    pub fallback_only: bool,
}

/// USD per 1M tokens. Cache rates differ per provider (OpenAI discounts
/// cache reads; Anthropic bills cache reads at ~0.1x and cache writes at
/// ~1.25x input) — when unset they fall back to `input_per_mtok`, which
/// never undercharges.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PricingCfg {
    /// USD per 1M prompt tokens.
    pub input_per_mtok: f64,
    /// USD per 1M completion tokens.
    pub output_per_mtok: f64,
    /// USD per 1M cached prompt tokens.
    #[serde(default)]
    pub cached_input_per_mtok: Option<f64>,
    /// USD per 1M Anthropic cache write tokens.
    #[serde(default)]
    pub cache_write_per_mtok: Option<f64>,
}

/// Settings that can override the config file (e.g. from command-line args).
/// Backends and models are deliberately absent — they live in the database
/// and are edited through the admin API.
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    /// Listen address override.
    pub listen: Option<SocketAddr>,
    /// Admin listen address override.
    pub admin_listen: Option<SocketAddr>,
    /// SQLite database path override.
    pub sqlite_path: Option<PathBuf>,
    /// Maximum body bytes override.
    pub max_body_bytes: Option<usize>,
    /// Literal admin token override.
    pub admin_token: Option<String>,
    /// Admin token environment variable override.
    pub admin_token_env: Option<String>,
}

impl Config {
    /// Loads configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the file cannot be read, parsed, or validated.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        Self::load_with(path, &Overrides::default())
    }

    /// Loads configuration with command-line overrides.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError` if the file cannot be read, parsed, or validated.
    pub fn load_with(path: &Path, overrides: &Overrides) -> Result<Self, ConfigError> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| ConfigError::Read(path.to_path_buf(), e))?;
        let mut cfg: Config = toml::from_str(&raw)?;
        cfg.apply_overrides(overrides);
        cfg.validate()?;
        Ok(cfg)
    }

    /// Applies overrides onto this configuration.
    pub fn apply_overrides(&mut self, ov: &Overrides) {
        if let Some(listen) = ov.listen {
            self.listen = listen;
        }
        if let Some(admin_listen) = ov.admin_listen {
            self.admin_listen = Some(admin_listen);
        }
        if let Some(sqlite_path) = &ov.sqlite_path {
            self.sqlite_path = sqlite_path.clone();
        }
        if let Some(max_body_bytes) = ov.max_body_bytes {
            self.max_body_bytes = max_body_bytes;
        }
        // The admin token override replaces both fields so the
        // exactly-one-of invariant checked by validate() still holds.
        if ov.admin_token.is_some() || ov.admin_token_env.is_some() {
            self.admin_token = ov.admin_token.clone();
            self.admin_token_env = ov.admin_token_env.clone();
        }
    }

    /// Validates configuration invariants and seed catalogs.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::Invalid` if any constraint is violated.
    pub fn validate(&self) -> Result<(), ConfigError> {
        match (&self.admin_token, &self.admin_token_env) {
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "set exactly one of admin_token / admin_token_env, not both",
                ));
            }
            (None, None) => {
                return Err(invalid("one of admin_token / admin_token_env is required"));
            }
            _ => {}
        }

        if self.admin_listen == Some(self.listen) {
            return Err(invalid(
                "admin_listen must differ from listen (omit it to share the main listener)",
            ));
        }

        // The optional seed sections are checked with the same validators
        // the admin API uses, so a config file cannot inject a catalog the
        // API would have rejected.
        let mut group_ids = HashSet::new();
        for g in &self.exclusive_groups {
            if !group_ids.insert(g.id.as_str()) {
                return Err(invalid(format!("duplicate exclusive group id {:?}", g.id)));
            }
            crate::catalog::validate_group(&GroupDef::from(g)).map_err(invalid)?;
        }

        let known_groups: Vec<String> =
            self.exclusive_groups.iter().map(|g| g.id.clone()).collect();
        let mut names = HashSet::new();
        for b in &self.backends {
            if !names.insert(b.name.as_str()) {
                return Err(invalid(format!("duplicate backend name {:?}", b.name)));
            }
            crate::catalog::validate_backend(&BackendDef::from(b), &known_groups)
                .map_err(invalid)?;
        }

        let known: Vec<String> = self.backends.iter().map(|b| b.name.clone()).collect();
        // Names and aliases share one namespace, so each model is checked
        // against the ones already accepted — same rule the admin API applies.
        let mut accepted: Vec<ModelDef> = Vec::with_capacity(self.models.len());
        for m in &self.models {
            let def = ModelDef::from(m);
            crate::catalog::validate_model(&def, &known).map_err(invalid)?;
            crate::catalog::validate_no_name_conflict(&def, &accepted).map_err(invalid)?;
            accepted.push(def);
        }

        Ok(())
    }

    /// Resolves the admin token from literal value or environment variable.
    ///
    /// # Errors
    ///
    /// Returns `ConfigError::MissingEnv` if the environment variable is unset.
    pub fn resolved_admin_token(&self) -> Result<String, ConfigError> {
        resolve_secret(self.admin_token.as_deref(), self.admin_token_env.as_deref())
    }
}

pub(crate) fn resolve_secret(
    literal: Option<&str>,
    env_name: Option<&str>,
) -> Result<String, ConfigError> {
    if let Some(v) = literal {
        return Ok(v.to_string());
    }
    let name = env_name.expect("validate() guarantees one source is set");
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => Ok(v),
        _ => Err(ConfigError::MissingEnv(name.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> String {
        r#"
listen = "127.0.0.1:8080"
sqlite_path = "test.db"
admin_token = "secret"

[[backends]]
name = "openai"
kind = "openai"
base_url = "https://api.openai.com"
api_key = "sk-test"

[[models]]
name = "gpt-4o"
  [[models.targets]]
  backend = "openai"
"#
        .to_string()
    }

    fn parse(s: &str) -> Result<Config, ConfigError> {
        let cfg: Config = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    #[test]
    fn parses_example_config() {
        let raw = include_str!("../config.example.toml");
        let cfg: Config = toml::from_str(raw).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.backends.len(), 7);
        assert!(cfg.backends[0].inject_usage); // default
        assert_eq!(cfg.models.len(), 3);
        assert_eq!(cfg.models[0].aliases, ["gpt-4o-latest", "gpt4o"]);
        assert!(cfg.models[1].aliases.is_empty());

        // The scheduling seed: one device group, two grouped members with
        // deliberately asymmetric token caps, one exempt embedder.
        assert_eq!(cfg.exclusive_groups.len(), 1);
        let dgx = &cfg.exclusive_groups[0];
        assert_eq!(dgx.id, "dgx");
        assert!(dgx.min_hold_ms <= dgx.max_hold_ms);
        assert!(dgx.max_hold_ms <= dgx.drain_timeout_ms);

        let by_name = |n: &str| cfg.backends.iter().find(|b| b.name == n).unwrap();
        let moe = by_name("moe-local");
        assert_eq!(moe.scheduling_class, SchedulingClass::Grouped);
        assert_eq!(moe.exclusive_group.as_deref(), Some("dgx"));
        assert_eq!(moe.max_concurrency, Some(4));
        assert_eq!(moe.max_tokens_cap, Some(1024));
        assert_eq!(moe.tok_s_estimate, Some(90.0));
        // The slower dense model gets the smaller cap, so both drain in a
        // comparable wall-clock time.
        let dense = by_name("dense-local");
        assert!(dense.max_tokens_cap < moe.max_tokens_cap);
        assert!(dense.tok_s_estimate < moe.tok_s_estimate);
        let embed = by_name("embed-local");
        assert_eq!(embed.scheduling_class, SchedulingClass::Exempt);
        assert_eq!(embed.exclusive_group.as_deref(), Some("dgx"));
        assert_eq!(embed.max_tokens_cap, None);
        // Everything off the device stays unrestricted by default.
        assert_eq!(
            by_name("openai").scheduling_class,
            SchedulingClass::Unrestricted
        );
        assert_eq!(by_name("openai").exclusive_group, None);

        let qwen = cfg.models.iter().find(|m| m.name == "qwen-moe").unwrap();
        let qos = qwen.qos.as_ref().unwrap();
        assert_eq!(qos.max_wait_ms, Some(2000));
        assert_eq!(qos.background_max_wait_ms, Some(60000));
        assert_eq!(qwen.on_shed_without_fallback, OnShedWithoutFallback::Wait);
        assert!(!qwen.targets[0].fallback_only);
        assert!(qwen.targets[1].fallback_only);
        // Routes that predate scheduling carry no deadline at all.
        assert!(cfg.models[0].qos.is_none());
        assert!(cfg.models[0].targets.iter().all(|t| !t.fallback_only));
        assert_eq!(
            cfg.models[0].targets[1].model.as_deref(),
            Some("llama-3.3-70b")
        );
        assert_eq!(cfg.models[0].targets[0].weight, 3);
        assert_eq!(cfg.models[1].targets[0].weight, 1); // default

        let gpt = cfg.models[0].pricing.as_ref().unwrap();
        assert_eq!(gpt.input_per_mtok, 2.50);
        assert_eq!(gpt.output_per_mtok, 10.00);
        assert_eq!(gpt.cached_input_per_mtok, Some(1.25));
        assert_eq!(gpt.cache_write_per_mtok, None);
        // vllm target overrides the model default with free pricing.
        let vllm = cfg.models[0].targets[1].pricing.as_ref().unwrap();
        assert_eq!(vllm.input_per_mtok, 0.0);
        assert!(cfg.models[0].targets[0].pricing.is_none());
        let claude = cfg.models[1].pricing.as_ref().unwrap();
        assert_eq!(claude.cache_write_per_mtok, Some(3.75));
    }

    #[test]
    fn kind_accepts_string_or_array() {
        let cfg = parse(&base_config()).unwrap();
        assert_eq!(cfg.backends[0].kind, vec![BackendKind::Openai]);

        let raw = base_config().replace("kind = \"openai\"", "kind = [\"openai\", \"anthropic\"]");
        let cfg = parse(&raw).unwrap();
        assert_eq!(
            cfg.backends[0].kind,
            vec![BackendKind::Openai, BackendKind::Anthropic]
        );
    }

    #[test]
    fn rejects_empty_kind_list() {
        let raw = base_config().replace("kind = \"openai\"", "kind = []");
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_duplicate_kind_entries() {
        let raw = base_config().replace("kind = \"openai\"", "kind = [\"openai\", \"openai\"]");
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn admin_listen_must_differ_from_listen() {
        let raw = base_config().replace(
            "listen = \"127.0.0.1:8080\"",
            "listen = \"127.0.0.1:8080\"\nadmin_listen = \"127.0.0.1:8080\"",
        );
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
        let raw = base_config().replace(
            "listen = \"127.0.0.1:8080\"",
            "listen = \"127.0.0.1:8080\"\nadmin_listen = \"127.0.0.1:9090\"",
        );
        let cfg = parse(&raw).unwrap();
        assert_eq!(cfg.admin_listen, Some("127.0.0.1:9090".parse().unwrap()));
    }

    #[test]
    fn rejects_negative_pricing() {
        let raw = base_config().replace(
            "name = \"gpt-4o\"",
            "name = \"gpt-4o\"\n[models.pricing]\ninput_per_mtok = -1.0\noutput_per_mtok = 1.0",
        );
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn happy_path() {
        let cfg = parse(&base_config()).unwrap();
        assert_eq!(cfg.max_body_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.resolved_admin_token().unwrap(), "secret");
        assert_eq!(
            BackendDef::from(&cfg.backends[0])
                .resolved_api_key()
                .unwrap()
                .as_deref(),
            Some("sk-test")
        );
    }

    #[test]
    fn rejects_duplicate_backend() {
        let raw = base_config().replace(
            "[[models]]",
            r#"[[backends]]
name = "openai"
kind = "openai"
base_url = "https://example.com"
api_key = "x"

[[models]]"#,
        );
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_unknown_target_backend() {
        let raw = base_config().replace("backend = \"openai\"", "backend = \"nope\"");
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_zero_weight() {
        let raw =
            base_config().replace("backend = \"openai\"", "backend = \"openai\"\n  weight = 0");
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_both_api_key_sources() {
        let raw = base_config().replace(
            "api_key = \"sk-test\"",
            "api_key = \"sk-test\"\napi_key_env = \"K\"",
        );
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn allows_missing_api_key_source() {
        let raw = base_config().replace("api_key = \"sk-test\"\n", "");
        let cfg = parse(&raw).unwrap();
        assert_eq!(
            BackendDef::from(&cfg.backends[0])
                .resolved_api_key()
                .unwrap(),
            None
        );
    }

    #[test]
    fn rejects_missing_admin_token() {
        let raw = base_config().replace("admin_token = \"secret\"\n", "");
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn rejects_model_without_targets() {
        let raw = base_config().replace(
            "name = \"gpt-4o\"\n  [[models.targets]]\n  backend = \"openai\"",
            "name = \"gpt-4o\"\ntargets = []",
        );
        assert!(matches!(parse(&raw), Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn overrides_take_priority_over_file_values() {
        let mut cfg = parse(&base_config()).unwrap();
        cfg.apply_overrides(&Overrides {
            listen: Some("0.0.0.0:9999".parse().unwrap()),
            sqlite_path: Some(PathBuf::from("other.db")),
            max_body_bytes: Some(42),
            ..Overrides::default()
        });
        cfg.validate().unwrap();
        assert_eq!(cfg.listen, "0.0.0.0:9999".parse().unwrap());
        assert_eq!(cfg.sqlite_path, PathBuf::from("other.db"));
        assert_eq!(cfg.max_body_bytes, 42);
        // untouched by the override
        assert_eq!(cfg.resolved_admin_token().unwrap(), "secret");
    }

    #[test]
    fn empty_overrides_change_nothing() {
        let mut cfg = parse(&base_config()).unwrap();
        cfg.apply_overrides(&Overrides::default());
        assert_eq!(cfg.listen, "127.0.0.1:8080".parse().unwrap());
        assert_eq!(cfg.sqlite_path, PathBuf::from("test.db"));
        assert_eq!(cfg.max_body_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.resolved_admin_token().unwrap(), "secret");
    }

    #[test]
    fn admin_token_env_override_replaces_file_admin_token() {
        let mut cfg = parse(&base_config()).unwrap();
        cfg.apply_overrides(&Overrides {
            admin_token_env: Some("LLM_ROUTER_TEST_SURELY_UNSET_VAR".into()),
            ..Overrides::default()
        });
        // The file's literal admin_token was cleared, so validate() still
        // sees exactly one source and resolution goes through the env var.
        cfg.validate().unwrap();
        assert_eq!(cfg.admin_token, None);
        assert!(matches!(
            cfg.resolved_admin_token(),
            Err(ConfigError::MissingEnv(_))
        ));
    }

    #[test]
    fn admin_token_override_replaces_file_admin_token_env() {
        let raw = base_config().replace(
            "admin_token = \"secret\"",
            "admin_token_env = \"SOME_UNSET_ADMIN_VAR\"",
        );
        let mut cfg = parse(&raw).unwrap();
        cfg.apply_overrides(&Overrides {
            admin_token: Some("cli-secret".into()),
            ..Overrides::default()
        });
        cfg.validate().unwrap();
        assert_eq!(cfg.admin_token_env, None);
        assert_eq!(cfg.resolved_admin_token().unwrap(), "cli-secret");
    }

    #[test]
    fn missing_env_var_fails_resolution() {
        let raw = base_config().replace(
            "api_key = \"sk-test\"",
            "api_key_env = \"LLM_ROUTER_TEST_SURELY_UNSET_VAR\"",
        );
        let cfg = parse(&raw).unwrap();
        assert!(matches!(
            BackendDef::from(&cfg.backends[0]).resolved_api_key(),
            Err(ConfigError::MissingEnv(_))
        ));
    }
}
