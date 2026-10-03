//! `shiso-registry` - maintainer tool for editing, inspecting, validating and building the model registry.
#![forbid(unsafe_code)]
#![expect(clippy::print_stdout, reason = "maintainer CLI command output")]

mod cli;
mod editor;
mod server;

#[cfg(test)]
mod tests;

use clap::Parser as _;
use cli::{Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        // Every path the tool writes or runs git in hangs off the repository,
        // so it is made absolute once, here, against the directory the tool was
        // started from - and nothing later depends on that directory again.
        Command::Serve(mut args) => {
            args.repo = std::path::absolute(&args.repo)?;
            server::run_server(args.repo, args.port, args.open).await?;
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
            args.repo = std::path::absolute(&args.repo)?;
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
        Command::Build(mut args) => {
            args.repo = std::path::absolute(&args.repo)?;
            let out = editor::resolve_out(&args.repo, args.out.as_deref());
            let res = editor::build_registry(&args.repo, &out, args.allow_dirty)?;
            println!(
                "Build succeeded: {} model(s) compiled into {}",
                res.models_count,
                out.display()
            );
        }
    }

    Ok(())
}
