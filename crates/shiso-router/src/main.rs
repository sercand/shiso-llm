//! The `shiso-router` binary entry point.

use clap::Parser as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "shiso_router=info".into()),
        )
        .init();

    let args = shiso_router::cli::RouterArgs::parse();
    shiso_router::run(args).await
}
