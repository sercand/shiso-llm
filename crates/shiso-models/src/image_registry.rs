//! Opens an image generation checkpoint.

use shiso_core::Backend;
use shiso_vision::generator::{ImageGenError, ImageGenerator};
use std::sync::Arc;

use crate::flux2::config::ImageModelDesc;

/// Opens the image generator from `desc` on `backend`.
///
/// # Errors
/// Returns [`ImageGenError::Checkpoint`] if the checkpoint cannot be resolved or loaded.
pub fn open_image_generator(
    desc: &ImageModelDesc,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn ImageGenerator>, ImageGenError> {
    let located = crate::flux2::config::locate_and_inspect(desc)?;
    let model = crate::flux2::pipeline::Flux2KleinModel::load(&located, backend)?;
    Ok(Arc::new(model))
}
