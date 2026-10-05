//! `shiso-registry` - maintainer tool for editing, inspecting, validating and building the model registry.
#![forbid(unsafe_code)]
#![expect(clippy::print_stdout, reason = "maintainer CLI command output")]

mod bench;
mod card;
mod cli;
mod editor;
mod server;
mod site;

#[cfg(test)]
mod tests;

use clap::Parser as _;
use cli::{Cli, Command};

/// A task name as recipes spell it.
fn parse_task(s: &str) -> anyhow::Result<shiso_hub::Task> {
    s.parse().map_err(|e| anyhow::anyhow!("task '{s}': {e}"))
}

/// `--home`, else `~/.shiso`, with a leading `~` expanded.
fn resolve_home(home: Option<std::path::PathBuf>) -> std::path::PathBuf {
    match home {
        Some(p) => match (p.strip_prefix("~"), std::env::var_os("HOME")) {
            (Ok(rest), Some(h)) => std::path::PathBuf::from(h).join(rest),
            _ => p,
        },
        None => shiso_hub::HubConfig::default_home(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        // Every path the tool writes or runs git in hangs off the repository,
        // so it is made absolute once, here, against the directory the tool was
        // started from - and nothing later depends on that directory again.
        // Canonical rather than only absolute, so a missing repository fails
        // here and the paths the tool prints carry no `..`.
        Command::Serve(mut args) => {
            args.repo = std::fs::canonicalize(&args.repo)?;
            server::run_server(server::ServeDesc {
                repo: args.repo,
                port: args.port,
                open: args.open,
                require_token: args.require_token,
                home: resolve_home(args.home),
            })
            .await?;
        }
        Command::Inspect(args) => {
            let res = editor::inspect_hf(&args.repo, &args.revision, None).await?;
            println!("Commit: {}", res.commit);
            println!("{:<40} {:<10} {:<10}", "FILE", "SIZE", "LFS");
            for f in res.files {
                println!(
                    "{:<40} {:<10} {:<10}",
                    f.name,
                    f.size,
                    if f.lfs { "yes" } else { "no" }
                );
            }
        }
        Command::Validate(mut args) => {
            args.repo = std::fs::canonicalize(&args.repo)?;
            let reports = editor::validate_all(&args.repo)?;
            let mut failed = false;
            for r in reports {
                if r.valid {
                    println!("[OK]     {} ({})", r.file, r.name);
                } else {
                    println!("[FAILED] {} ({}): {}", r.file, r.name, r.errors.join("; "));
                    failed = true;
                }
            }
            if failed {
                anyhow::bail!("one or more recipes failed strict validation");
            }
            println!("All recipes strictly valid.");
        }
        Command::Build(args) => build(args).await?,
        Command::Bench(args) => bench(args).await?,
    }

    Ok(())
}

async fn build(mut args: cli::BuildArgs) -> anyhow::Result<()> {
    args.repo = std::fs::canonicalize(&args.repo)?;
    if !args.offline {
        let recipes: Vec<_> = editor::load_recipes(&args.repo)?
            .into_iter()
            .map(|(_, r)| r)
            .collect();
        let stats = card::refresh_cache(&args.repo, &recipes).await?;
        println!(
            "Cards: {} fetched, {} without a README, {} cached",
            stats.fetched, stats.missing, stats.cached
        );
    }
    let out = editor::resolve_out(&args.repo, args.out.as_deref());
    let res = editor::build_registry(&args.repo, &out, args.allow_dirty)?;
    for p in &res.pruned {
        println!("Removed stale {p}");
    }
    if !res.missing_cards.is_empty() {
        println!("No card for: {}", res.missing_cards.join(", "));
    }
    println!(
        "Build succeeded: {} model(s) and {} site file(s) written into {}",
        res.models_count,
        res.site_files.len(),
        out.display()
    );
    Ok(())
}

async fn bench(mut args: Box<cli::BenchArgs>) -> anyhow::Result<()> {
    args.repo = std::fs::canonicalize(&args.repo)?;
    let model_ref: shiso_hub::ModelRef = args
        .model
        .parse()
        .map_err(|e| anyhow::anyhow!("model '{}': {e}", args.model))?;
    let path = bench::run(bench::BenchDesc {
        repo: args.repo,
        model: model_ref.model.clone(),
        variant: model_ref.variant.clone(),
        server_url: args.server_url,
        device: args
            .device
            .as_deref()
            .map(bench::DeviceSlug::parse)
            .transpose()?,
        task: args.task.as_deref().map(parse_task).transpose()?,
        prompt_tokens: args.prompt_tokens,
        decode_tokens: args.decode_tokens,
        image_sizes: args.image_size,
        audio: args.audio,
        runs: args.runs,
        home: resolve_home(args.home),
        shiso_version: args.shiso_version,
        notes: args.notes,
        log: None,
    })
    .await?;
    println!("Wrote {}", path.display());
    Ok(())
}
