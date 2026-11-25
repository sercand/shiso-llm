//! What counts as a transparent picture.
//!
//! Measured on Qwen-Image-2.1-Turbo at 1024x1024: a prompt that asks for a
//! cut-out comes back with roughly half its pixels below alpha 32, while an
//! ordinary prompt comes back with every alpha in 245..=255 - opaque, with a
//! few levels of noise. So transparency is a threshold, not "any pixel below
//! 255": a picture is transparent when at least one percent of its pixels are
//! below half coverage. An opaque one is composited over white, so a client
//! never sees a 253 it has to explain.

/// Alpha below this is "mostly transparent".
const HALF: u8 = 128;
/// The share of mostly transparent pixels, in hundredths, that makes a
/// picture a cut-out.
const PERCENT: usize = 1;

/// Whether interleaved RGBA8 `rgba` is a cut-out: at least one percent of its
/// pixels have alpha below 128.
#[must_use]
pub fn is_transparent(rgba: &[u8]) -> bool {
    let pixels = rgba.len() / 4;
    let low = rgba.chunks_exact(4).filter(|p| p[3] < HALF).count();
    pixels > 0 && low * 100 >= pixels * PERCENT
}
