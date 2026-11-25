//! Background removal: a picture in, one alpha byte per pixel out.
//!
//! A seam rather than a concrete model, for the same reasons on both sides
//! that use it: the image route's background post-processing and the splat
//! route's preprocessing can each be tested without a device, and a host that
//! never needs a cut-out leaves the remover unloaded.

/// A background remover.
pub trait BackgroundRemover: Send + Sync + std::fmt::Debug {
    /// The alpha channel for `rgb`, one byte per pixel in row-major order.
    ///
    /// # Errors
    ///
    /// A human-readable reason the remover could not run.
    fn alpha(&self, rgb: &image::RgbImage) -> Result<Vec<u8>, String>;
}
