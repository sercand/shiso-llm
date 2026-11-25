//! Opens a background remover.
//!
//! One remover serves today: the dichotomous-segmentation network whose port
//! lives with the splat family, because it is built on that family's device
//! operations (deformable convolution, shifted-window attention). This opener
//! is the family-neutral door the image route uses, so nothing above this
//! crate names where the network lives.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::Backend;
use shiso_vision::generator::ImageGenError;
use shiso_vision::matte::BackgroundRemover;

use crate::diffusion::ops::Ops;

/// The weight file names a remover directory may hold, in the order they are
/// looked for: the hub repository's layout, then the name it ships under
/// beside a splat checkpoint.
const WEIGHT_FILES: [&str; 2] = ["model.safetensors", crate::splat_registry::MATTE_FILE];

/// Opens the background remover at `path` on `backend`: a `.safetensors`
/// file, or a directory holding one of the known weight file names.
///
/// # Errors
///
/// [`ImageGenError::Checkpoint`] for a path with no weights or weights of
/// another shape, and [`ImageGenError::Backend`] for a device without the
/// kernels the remover runs on.
pub fn open_background_remover(
    path: &Path,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn BackgroundRemover>, ImageGenError> {
    let weights = weights_file(path)?;
    let ops = Arc::new(Ops::new(backend)?);
    Ok(Arc::new(crate::idol::birefnet::BiRefNet::load(
        &weights, ops,
    )?))
}

/// The device bytes the remover at `path` holds once loaded: its weights
/// ship in fp16 and run widened to f32, so twice the file. Known before it
/// loads, so an image model sharing the device can leave the room.
///
/// # Errors
///
/// [`ImageGenError::Checkpoint`] for a path with no weights.
pub fn resident_bytes(path: &Path) -> Result<u64, ImageGenError> {
    let weights = weights_file(path)?;
    let len = std::fs::metadata(&weights)
        .map_err(|e| ImageGenError::Checkpoint(format!("{}: {e}", weights.display())))?
        .len();
    Ok(len.saturating_mul(2))
}

/// The weights file `path` names: itself when it is a file, else the first
/// known weight file inside it.
fn weights_file(path: &Path) -> Result<PathBuf, ImageGenError> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    WEIGHT_FILES
        .iter()
        .map(|name| path.join(name))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            ImageGenError::Checkpoint(format!(
                "no background remover weights at {}: expected a .safetensors file or a \
                 directory holding {}",
                path.display(),
                WEIGHT_FILES.join(" or ")
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::{resident_bytes, weights_file};

    #[test]
    fn the_resident_size_is_the_weights_widened_to_f32() {
        let root = tempfile::tempdir().expect("a temp dir");
        std::fs::write(root.path().join("model.safetensors"), [0u8; 1000]).expect("write");
        assert_eq!(resident_bytes(root.path()).expect("sized"), 2000);
    }

    #[test]
    fn a_directory_resolves_to_its_weights_file() {
        let root = tempfile::tempdir().expect("a temp dir");
        assert!(weights_file(root.path()).is_err(), "an empty directory");
        std::fs::write(root.path().join("birefnet.safetensors"), b"x").expect("write");
        assert_eq!(
            weights_file(root.path()).expect("found"),
            root.path().join("birefnet.safetensors")
        );
        // The hub layout's file name wins when both are present.
        std::fs::write(root.path().join("model.safetensors"), b"x").expect("write");
        assert_eq!(
            weights_file(root.path()).expect("found"),
            root.path().join("model.safetensors")
        );
        // A file is taken as given.
        let file = root.path().join("birefnet.safetensors");
        assert_eq!(weights_file(&file).expect("a file"), file);
        let missing = weights_file(&root.path().join("nope")).expect_err("missing");
        assert!(missing.to_string().contains("nope"), "{missing}");
    }
}
