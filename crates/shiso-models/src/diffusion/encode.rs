//! Encodes a finished image as PNG or JPEG, whatever family produced it.
//!
//! Lives beside the other family-neutral diffusion pieces rather than in
//! `shiso-vision`: that crate owns the contract (formats, errors), while the
//! codecs are an implementation dependency of the generators and `shiso-vision`
//! carries none.

use image::ImageEncoder as _;
use shiso_vision::generator::{ImageGenError, ImageOutputFormat};

/// The channel layout of an interleaved 8-bit pixel buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channels {
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue, straight (non-premultiplied) alpha.
    Rgba,
}

impl Channels {
    /// Bytes one pixel occupies.
    const fn per_pixel(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Composites straight-alpha RGBA over white into RGB, per channel
/// `(c*a + 255*(255-a) + 127) / 255` in integers: JPEG has no alpha, and a
/// transparent pixel must land on white rather than on whatever colour its
/// hidden RGB happens to hold.
pub(crate) fn composite_over_white(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        let a = u32::from(px[3]);
        for &c in &px[..3] {
            let v = (u32::from(c) * a + 255 * (255 - a) + 127) / 255;
            rgb.push(u8::try_from(v).unwrap_or(u8::MAX));
        }
    }
    rgb
}

/// Encodes interleaved 8-bit `pixels` as PNG or JPEG.
///
/// PNG keeps alpha (RGBA8) when `channels` is [`Channels::Rgba`]. JPEG
/// composites RGBA over white first ([`composite_over_white`]). `compression`
/// is the JPEG quality (1..=100, default 90) and is ignored by PNG.
///
/// # Errors
/// [`ImageGenError::Encoding`] if the buffer length does not match the
/// dimensions, or the codec fails.
pub(crate) fn encode_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: Channels,
    format: ImageOutputFormat,
    compression: Option<u32>,
) -> Result<Vec<u8>, ImageGenError> {
    let expected = width as usize * height as usize * channels.per_pixel();
    if pixels.len() != expected {
        return Err(ImageGenError::Encoding(format!(
            "expected {expected} colour bytes for {width}x{height}, got {}",
            pixels.len()
        )));
    }
    let mut out = Vec::new();
    match format {
        ImageOutputFormat::Png => {
            let color = match channels {
                Channels::Rgb => image::ExtendedColorType::Rgb8,
                Channels::Rgba => image::ExtendedColorType::Rgba8,
            };
            image::codecs::png::PngEncoder::new(&mut out)
                .write_image(pixels, width, height, color)
                .map_err(|e| ImageGenError::Encoding(format!("png: {e}")))?;
        }
        ImageOutputFormat::Jpeg => {
            let quality = u8::try_from(compression.unwrap_or(90).clamp(1, 100)).unwrap_or(90);
            let flattened;
            let rgb = match channels {
                Channels::Rgb => pixels,
                Channels::Rgba => {
                    flattened = composite_over_white(pixels);
                    &flattened
                }
            };
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
                .write_image(rgb, width, height, image::ExtendedColorType::Rgb8)
                .map_err(|e| ImageGenError::Encoding(format!("jpeg: {e}")))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGBA: [u8; 16] = [
        255, 0, 0, 0, //
        0, 255, 0, 128, //
        0, 0, 255, 255, //
        10, 20, 30, 255,
    ];

    #[test]
    fn png_keeps_alpha_exactly() {
        let png = encode_image(&RGBA, 2, 2, Channels::Rgba, ImageOutputFormat::Png, None)
            .expect("encodes");
        let decoded = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .expect("decodes")
            .to_rgba8();
        assert_eq!(decoded.as_raw().as_slice(), RGBA.as_slice());
    }

    #[test]
    fn jpeg_composites_over_white_with_exact_integers() {
        // a=0 -> white; a=128 -> (0*128 + 255*127 + 127)/255 = 127 for the
        // zero channels and (255*128 + 255*127 + 127)/255 = 255 for the full
        // one; a=255 -> unchanged.
        assert_eq!(
            composite_over_white(&RGBA),
            [255, 255, 255, 127, 255, 127, 0, 0, 255, 10, 20, 30]
        );
        let jpeg = encode_image(&RGBA, 2, 2, Channels::Rgba, ImageOutputFormat::Jpeg, None)
            .expect("encodes");
        assert!(jpeg.starts_with(&[0xff, 0xd8]));
    }

    #[test]
    fn a_buffer_off_its_dimensions_is_refused() {
        let error = encode_image(&RGBA, 2, 2, Channels::Rgb, ImageOutputFormat::Png, None)
            .expect_err("wrong length");
        assert!(matches!(error, ImageGenError::Encoding(_)));
    }
}
