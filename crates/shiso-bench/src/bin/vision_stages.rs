//! Dumps intermediate vision stages from shiso's vision encoder.
//!
//! Produces `<out>/<stage>.bin` and `<out>/manifest.json` matching the
//! reference dumper layout from `scripts/dump_vision_stages.py`.

#![expect(clippy::print_stderr, reason = "CLI tool status output")]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use shiso_bench::replay::fnv_digest;
use shiso_core::Backend;
use shiso_metal::MetalBackend;
use shiso_models::gemma4::Gemma4Tower;
use shiso_models::qwen35::Qwen35Tower;
use shiso_vision::{ImageLimits, VisionItem, VisionStages, VisionTower, preprocess_image};

/// CLI arguments for dumping vision stages.
#[derive(Debug, Parser)]
#[command(
    name = "vision_stages",
    about = "Dumps intermediate vision stages to disk"
)]
struct Args {
    /// Model directory or mmproj GGUF file.
    #[arg(long)]
    model: PathBuf,

    /// Input image file (PNG).
    #[arg(long)]
    image: PathBuf,

    /// Output directory for stage dumps.
    #[arg(long)]
    out: PathBuf,

    /// Model family: gemma4 or qwen35.
    #[arg(long)]
    family: String,

    /// Device backend: metal.
    #[arg(long, default_value = "metal")]
    backend: String,

    /// Writes the GPU time per kernel of one plain encode of the image here,
    /// after a warm-up encode.
    #[arg(long)]
    profile: Option<PathBuf>,
}

/// Metadata record for one captured stage in manifest.json.
#[derive(Debug, Serialize)]
struct StageMeta {
    shape: Vec<usize>,
    digest: String,
    head: Vec<f32>,
    tail: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out)
        .with_context(|| format!("creating output directory {}", args.out.display()))?;

    let backend: Arc<dyn Backend> = match args.backend.as_str() {
        "metal" => MetalBackend::new().context("opening Metal backend")?,
        other => bail!("unsupported backend: {other}"),
    };

    let tower: Box<dyn VisionTower> = match args.family.as_str() {
        "gemma4" => {
            let t = Gemma4Tower::open(&backend, &args.model)
                .with_context(|| format!("opening gemma4 tower from {}", args.model.display()))?;
            Box::new(t)
        }
        "qwen35" => {
            let t = Qwen35Tower::open(&backend, &args.model)
                .with_context(|| format!("opening qwen35 tower from {}", args.model.display()))?;
            Box::new(t)
        }
        other => bail!("unsupported family: {other}; expected gemma4 or qwen35"),
    };

    let image_bytes = fs::read(&args.image)
        .with_context(|| format!("reading image from {}", args.image.display()))?;
    let item = preprocess_image(&image_bytes, tower.preprocess(), ImageLimits::default())
        .context("preprocessing image")?;
    if let Some(path) = args.profile.as_deref() {
        let items = std::slice::from_ref(&item);
        tower.encode(&backend, items).context("warm-up encode")?;
        let ((), report) = shiso_bench::profile::scoped(backend.as_ref(), || -> Result<()> {
            tower.encode(&backend, items).context("profiled encode")?;
            Ok(())
        })?;
        shiso_bench::profile::write(
            path,
            &serde_json::json!({ "family": args.family, "grid": format!("{:?}", item.grid()) }),
            &report,
        )?;
        tracing::info!(path = %path.display(), "wrote the vision profile");
    }
    eprintln!(
        "Encoding stages for family={}, image={}, grid={:?}",
        args.family,
        args.image.display(),
        item.grid()
    );

    let stages = tower
        .encode_stages(&backend, &item)
        .context("running staged vision encode")?;

    write_stages(&backend, &args.out, &item, &stages, &*tower)?;

    Ok(())
}

fn write_stages(
    backend: &Arc<dyn Backend>,
    out_dir: &std::path::Path,
    item: &VisionItem,
    stages: &VisionStages,
    tower: &dyn VisionTower,
) -> Result<()> {
    let (_, grid_h, grid_w) = item.grid();
    let rows = grid_h * grid_w;
    let patch_dim = item.row_len();

    let block_00 = stages
        .blocks
        .first()
        .context("no transformer blocks captured")?;
    let block_07 = stages
        .blocks
        .get(7)
        .context("fewer than 8 transformer blocks captured")?;
    let block_15 = stages
        .blocks
        .get(15)
        .context("fewer than 16 transformer blocks captured")?;

    let tokens = (stages.projected.len() / 4) as usize / tower.out_hidden() as usize;

    let named_stages = [
        ("pixel_values", &stages.pixel_values, vec![rows, patch_dim]),
        (
            "patch_embed",
            &stages.patch_embed,
            vec![rows, (stages.patch_embed.len() / 4) as usize / rows],
        ),
        (
            "pos_added",
            &stages.pos_added,
            vec![rows, (stages.pos_added.len() / 4) as usize / rows],
        ),
        (
            "block_00",
            block_00,
            vec![rows, (block_00.len() / 4) as usize / rows],
        ),
        (
            "block_07",
            block_07,
            vec![rows, (block_07.len() / 4) as usize / rows],
        ),
        (
            "block_15",
            block_15,
            vec![rows, (block_15.len() / 4) as usize / rows],
        ),
        (
            "pooled",
            &stages.pooled,
            vec![tokens, (stages.pooled.len() / 4) as usize / tokens],
        ),
        (
            "normed",
            &stages.normed,
            vec![tokens, (stages.normed.len() / 4) as usize / tokens],
        ),
        (
            "projected",
            &stages.projected,
            vec![tokens, tower.out_hidden() as usize],
        ),
    ];

    let mut manifest = BTreeMap::new();

    for (name, buf, shape) in named_stages {
        let len = usize::try_from(buf.len()).context("stage buffer length exceeds usize")?;
        let mut bytes = vec![0u8; len];
        backend
            .copy_d2h(buf.view(), &mut bytes)
            .with_context(|| format!("reading back stage {name}"))?;

        let bin_path = out_dir.join(format!("{name}.bin"));
        fs::write(&bin_path, &bytes)
            .with_context(|| format!("writing binary stage to {}", bin_path.display()))?;

        let digest = fnv_digest(&bytes);
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let head: Vec<f32> = floats.iter().take(8).copied().collect();
        let tail: Vec<f32> = if floats.len() >= 8 {
            floats[floats.len() - 8..].to_vec()
        } else {
            floats
        };

        eprintln!("  Stage {name:12}: shape={shape:?}, digest={digest}");

        manifest.insert(
            name.to_string(),
            StageMeta {
                shape,
                digest,
                head,
                tail,
            },
        );
    }

    let manifest_path = out_dir.join("manifest.json");
    let file = File::create(&manifest_path)
        .with_context(|| format!("creating manifest at {}", manifest_path.display()))?;
    serde_json::to_writer_pretty(file, &manifest)
        .with_context(|| format!("writing manifest to {}", manifest_path.display()))?;
    eprintln!("Wrote manifest to {}", manifest_path.display());

    Ok(())
}
