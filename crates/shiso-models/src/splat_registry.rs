//! Opens a 3D splat checkpoint.
//!
//! One family serves the splat route today, so there is nothing to detect:
//! the opener reads the checkpoint's `config.json`, refuses one of another
//! shape, and loads the family. A second family adds a signature here, the
//! way [`crate::image_registry`] dispatches, and nothing above this crate
//! changes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;
use shiso_vision::splat::SplatGenerator;

/// Where a splat model's parts live on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplatModelDesc {
    /// The model id clients see.
    pub id: String,
    /// The splat checkpoint directory: `config.json` and the three networks'
    /// weights, and the background remover's when it ships beside them.
    pub model_dir: PathBuf,
    /// The image encoder's directory: its `config.json` and weights.
    pub encoder_dir: PathBuf,
    /// The latent autoencoder's directory: its `config.json` and weights, or
    /// a repository root holding them under `vae/`.
    pub vae_dir: PathBuf,
    /// The background remover's weights. `None` serves only pictures that
    /// carry their own transparency.
    pub matte: Option<PathBuf>,
}

/// The background remover's file name inside the checkpoint directory.
pub const MATTE_FILE: &str = "birefnet.safetensors";

/// Opens the model `desc` names on `backend`.
///
/// # Errors
///
/// [`ImageGenError::Checkpoint`] for a part of another shape or a missing
/// file, and [`ImageGenError::Backend`] for a device without the kernels.
pub fn open_splat_generator(
    desc: &SplatModelDesc,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn SplatGenerator>, ImageGenError> {
    let desc = SplatModelDesc {
        vae_dir: vae_dir(&desc.vae_dir),
        ..desc.clone()
    };
    Ok(Arc::new(crate::idol::model::IdolGenerator::open(
        &desc, backend,
    )?))
}

/// The autoencoder's own directory: `dir` itself when it holds the config
/// document, else its `vae/` subdirectory when that does. The autoencoder
/// ships as the `vae/` folder of a larger repository, and a recipe's
/// component is the repository's root.
fn vae_dir(dir: &Path) -> PathBuf {
    let nested = dir.join("vae");
    if !dir.join("config.json").is_file() && nested.join("config.json").is_file() {
        nested
    } else {
        dir.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::vae_dir;

    #[test]
    fn the_vae_is_found_in_its_repository_subfolder() {
        let root = tempfile::tempdir().expect("a temp dir");
        std::fs::create_dir(root.path().join("vae")).expect("mkdir");
        std::fs::write(root.path().join("vae/config.json"), "{}").expect("write");
        assert_eq!(vae_dir(root.path()), root.path().join("vae"));
        // A directory that is the component itself stays as given.
        std::fs::write(root.path().join("config.json"), "{}").expect("write");
        assert_eq!(vae_dir(root.path()), root.path());
    }
}
