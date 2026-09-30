//! Configuration file format and loading.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

pub use crate::catalog::{BackendKind, PricingCfg};

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
}

fn default_max_body_bytes() -> usize {
    1024 * 1024 * 1024
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

    /// Validates configuration invariants.
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
        assert_eq!(cfg.listen, "127.0.0.1:8080".parse().unwrap());
        assert_eq!(cfg.sqlite_path, PathBuf::from("shiso-router.db"));
        assert_eq!(cfg.max_body_bytes, 1073741824);
        assert_eq!(
            cfg.admin_token_env.as_deref(),
            Some("LLM_ROUTER_ADMIN_TOKEN")
        );
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
    fn happy_path() {
        let cfg = parse(&base_config()).unwrap();
        assert_eq!(cfg.max_body_bytes, 1024 * 1024 * 1024);
        assert_eq!(cfg.resolved_admin_token().unwrap(), "secret");
    }

    #[test]
    fn rejects_missing_admin_token() {
        let raw = base_config().replace("admin_token = \"secret\"\n", "");
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
}
