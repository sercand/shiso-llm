use std::net::SocketAddr;
use std::path::PathBuf;

use clap::Parser;

use crate::config::Overrides;

/// Command-line arguments for the router.
///
/// Options given on the command line take priority over the config file.
/// Backends and models are not settable here: they live in the database and
/// are managed through the admin API (`/admin/backends`, `/admin/models`).
#[derive(Debug, Parser, Clone)]
#[command(name = "shiso-router", version)]
pub struct RouterArgs {
    /// Path to the TOML config file
    #[arg(short, long, default_value = "config.toml", value_name = "PATH")]
    pub config: PathBuf,

    /// Address to listen on, overriding the config file (e.g. 0.0.0.0:8080)
    #[arg(short, long, value_name = "ADDR")]
    pub listen: Option<SocketAddr>,

    /// Port to listen on, overriding the port of the listen address
    #[arg(short, long, value_name = "PORT")]
    pub port: Option<u16>,

    /// Separate address for the admin API + UI, overriding the config file
    #[arg(long, value_name = "ADDR")]
    pub admin_listen: Option<SocketAddr>,

    /// Path to the SQLite token database, overriding the config file
    #[arg(long, value_name = "PATH")]
    pub sqlite_path: Option<PathBuf>,

    /// Maximum request body size in bytes, overriding the config file
    #[arg(long, value_name = "BYTES")]
    pub max_body_bytes: Option<usize>,

    /// Admin API token, overriding the config file (prefer --admin-token-env)
    #[arg(long, value_name = "TOKEN", conflicts_with = "admin_token_env")]
    pub admin_token: Option<String>,

    /// Environment variable to read the admin API token from, overriding the config file
    #[arg(long, value_name = "ENV_VAR")]
    pub admin_token_env: Option<String>,
}

impl RouterArgs {
    /// Extracts the command-line overrides for configuration settings.
    pub fn overrides(&self) -> Overrides {
        Overrides {
            listen: self.listen,
            admin_listen: self.admin_listen,
            sqlite_path: self.sqlite_path.clone(),
            max_body_bytes: self.max_body_bytes,
            admin_token: self.admin_token.clone(),
            admin_token_env: self.admin_token_env.clone(),
        }
    }
}

/// Backward-compatible alias for [`RouterArgs`].
pub type Cli = RouterArgs;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_leave_overrides_empty() {
        let cli = Cli::try_parse_from(["llm-router"]).unwrap();
        assert_eq!(cli.config, PathBuf::from("config.toml"));
        let ov = cli.overrides();
        assert!(ov.listen.is_none());
        assert!(ov.sqlite_path.is_none());
        assert!(ov.max_body_bytes.is_none());
        assert!(ov.admin_token.is_none());
        assert!(ov.admin_token_env.is_none());
    }

    #[test]
    fn parses_all_args() {
        let cli = Cli::try_parse_from([
            "llm-router",
            "--config",
            "other.toml",
            "--listen",
            "0.0.0.0:9000",
            "--port",
            "9001",
            "--sqlite-path",
            "other.db",
            "--max-body-bytes",
            "1024",
            "--admin-token-env",
            "MY_TOKEN",
        ])
        .unwrap();
        assert_eq!(cli.config, PathBuf::from("other.toml"));
        assert_eq!(cli.listen, Some("0.0.0.0:9000".parse().unwrap()));
        assert_eq!(cli.port, Some(9001));
        assert_eq!(cli.sqlite_path, Some(PathBuf::from("other.db")));
        assert_eq!(cli.max_body_bytes, Some(1024));
        assert_eq!(cli.admin_token_env.as_deref(), Some("MY_TOKEN"));
    }

    #[test]
    fn short_flags() {
        let cli = Cli::try_parse_from([
            "llm-router",
            "-c",
            "x.toml",
            "-p",
            "1234",
            "-l",
            "[::]:8080",
        ])
        .unwrap();
        assert_eq!(cli.config, PathBuf::from("x.toml"));
        assert_eq!(cli.port, Some(1234));
        assert_eq!(cli.listen, Some("[::]:8080".parse().unwrap()));
    }

    #[test]
    fn admin_token_sources_conflict() {
        let err = Cli::try_parse_from([
            "llm-router",
            "--admin-token",
            "secret",
            "--admin-token-env",
            "MY_TOKEN",
        ])
        .unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn rejects_invalid_listen() {
        let err = Cli::try_parse_from(["llm-router", "--listen", "not-an-addr"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::ValueValidation);
    }
}
