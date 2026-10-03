//! The `shiso` binary: parse the command line, hand it to the library.
//!
//! Everything this file could contain lives in the library instead, so that the
//! whole server can be driven from a test without a process.

use clap::{CommandFactory as _, FromArgMatches as _};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let matches = shiso_server::cli::Cli::command().get_matches();
    let cli = shiso_server::cli::Cli::from_arg_matches(&matches)?;
    shiso_server::run_with_matches(cli, Some(matches)).await
}
