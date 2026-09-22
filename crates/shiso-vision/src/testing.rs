//! Fixtures shared by this crate's tests: encoded media and the recorded
//! checkpoint config.
//!
//! Everything a test needs is generated here rather than checked in as a blob,
//! with one exception: the vision geometry comes from the recorded checkpoint
//! config, because a hand-written geometry would let the tests keep passing
//! after the real one changed. The tower's own fixtures - the recorded manifest
//! and the synthetic container - belong to the family that reads them and live
//! beside its loader.

use shiso_core::{HfConfig, VisionConfig};

/// The recorded checkpoint config, the single source of the geometry and the
/// media token ids every test below asserts against.
pub(crate) const RECORDED_CONFIG: &str =
    include_str!("../../../test-vectors/loader/qwen35-config.json");

/// The vision block of the recorded checkpoint config.
pub(crate) fn vision_config() -> VisionConfig {
    HfConfig::from_json_str(RECORDED_CONFIG)
        .expect("the recorded config parses")
        .vision_config()
        .expect("the recorded config's vision block parses")
        .expect("the recorded config carries a vision block")
}

/// A solid-colour image, encoded as PNG.
pub(crate) fn png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("encoding a solid-colour PNG");
    bytes
}

/// An animated GIF of `frames` solid-colour squares, 100 ms apart, so the
/// decoded rate is 10 fps.
///
/// Each frame is a different colour, so a frame-ordering or frame-duplication
/// bug shows up in the pixel data rather than hiding behind a correct count.
pub(crate) fn gif(frames: u16, edge: u16) -> Vec<u8> {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, RgbaImage};

    let mut bytes: Vec<u8> = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut bytes);
        for index in 0..frames {
            let shade = u8::try_from(20 + (u32::from(index) * 37) % 200).unwrap_or(u8::MAX);
            let side = u32::from(edge);
            let image =
                RgbaImage::from_pixel(side, side, image::Rgba([shade, 40, 255 - shade, 255]));
            let frame = Frame::from_parts(image, 0, 0, Delay::from_numer_denom_ms(100, 1));
            encoder.encode_frame(frame).expect("encoding a GIF frame");
        }
    }
    bytes
}

/// The full `{:#}`-style chain of an error, so a test can assert on the
/// decoder's own complaint rather than on this crate's paraphrase of it.
pub(crate) fn chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        text.push_str(": ");
        text.push_str(&inner.to_string());
        source = inner.source();
    }
    text
}
