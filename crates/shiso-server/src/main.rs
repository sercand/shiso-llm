//! The `shiso` binary: parse the command line, hand it to the library.
//!
//! Everything this file could contain lives in the library instead, so that the
//! whole server can be driven from a test without a process.

use clap::Parser as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    shiso_server::run(shiso_server::cli::Cli::parse()).await
}
