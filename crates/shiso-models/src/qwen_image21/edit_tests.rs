//! Reference preparation: the condition-size table is the reference's
//! `calculate_dimensions` evaluated by hand at the area of 1024x1024.

use image::{ImageFormat, Rgba, RgbaImage};
use shiso_vision::generator::{ImageGenError, InputReferenceImage};

use super::edit::{
    MAX_REFERENCES, check_count, condition_size, default_size, prepare, references_hash,
};

fn encode(img: &RgbaImage) -> InputReferenceImage {
    let mut bytes = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("encodes");
    InputReferenceImage { bytes }
}

/// An opaque gradient of the given size.
fn png(w: u32, h: u32) -> InputReferenceImage {
    encode(&RgbaImage::from_fn(w, h, |x, y| {
        let v = |c: u32| u8::try_from((7 * x + 13 * y + 29 * c) % 256).expect("below 256");
        Rgba([v(0), v(1), v(2), 255])
    }))
}

fn is_reference_error<T: std::fmt::Debug>(result: Result<T, ImageGenError>) -> String {
    match result {
        Err(ImageGenError::ReferenceImage(message)) => message,
        other => panic!("expected a reference-image refusal, got {other:?}"),
    }
}

#[test]
fn condition_size_matches_reference() {
    let table = [
        ((1024, 1024), (1024, 1024)),
        ((1920, 1080), (1376, 768)),
        ((1080, 1920), (768, 1376)),
        ((100, 40), (1632, 640)),
    ];
    for ((w, h), want) in table {
        assert_eq!(condition_size(w, h).expect("sizes"), want, "{w}x{h}");
    }
}

#[test]
fn extreme_aspect_is_refused_or_on_grid() {
    let (w, h) = condition_size(4000, 100).expect("a 40:1 picture still has a grid");
    assert_eq!((w, h), (6464, 160));
    assert!(w % 32 == 0 && h % 32 == 0 && w >= 32 && h >= 32);
    // 5000:1 rounds its short side to no slot at all.
    is_reference_error(condition_size(80_000, 16));
    is_reference_error(condition_size(1, 1));
    is_reference_error(condition_size(0, 64));
}

#[test]
fn alpha_is_composited_over_white_for_the_tower_only() {
    let clear_red = encode(&RgbaImage::from_pixel(64, 64, Rgba([255, 0, 0, 0])));
    let prepared = prepare(&clear_red, 0).expect("prepares");
    assert_eq!((prepared.width, prepared.height), (1024, 1024));
    assert!(
        prepared.rgb.iter().all(|&v| v == 255),
        "tower copy is white"
    );
    assert!(
        prepared.rgba.chunks_exact(4).all(|p| p[3] == 0),
        "the VAE copy keeps its alpha"
    );
    assert_eq!(prepared.rgba.len(), 1024 * 1024 * 4);
    assert_eq!(prepared.rgb.len(), 1024 * 1024 * 3);
    assert_eq!(prepared.slots(), (32, 32));
    let grid = prepared.latent_grid();
    assert_eq!((grid.h, grid.w), (64, 64));
}

#[test]
fn count_is_bounded() {
    assert!(check_count(1, MAX_REFERENCES).is_ok());
    assert!(check_count(10, MAX_REFERENCES).is_ok());
    is_reference_error(check_count(0, MAX_REFERENCES));
    is_reference_error(check_count(11, MAX_REFERENCES));
    let message = is_reference_error(check_count(3, 2));
    assert!(message.contains('3') && message.contains('2'), "{message}");
}

#[test]
fn default_size_is_the_last_references() {
    assert_eq!(
        default_size(&[png(100, 40), png(1080, 1920)]).expect("two"),
        (768, 1376)
    );
    // A size outside what the model serves is clamped per side.
    assert_eq!(default_size(&[png(100, 40)]).expect("one"), (1632, 640));
    assert_eq!(default_size(&[png(400, 16)]).expect("wide"), (2048, 256));
    is_reference_error(default_size(&[]));
}

#[test]
fn garbage_bytes_are_refused() {
    let garbage = InputReferenceImage {
        bytes: b"not an image".to_vec(),
    };
    let message = is_reference_error(prepare(&garbage, 2));
    assert!(message.contains("reference image 3"), "{message}");
    is_reference_error(default_size(&[garbage]));
}

#[test]
fn hash_changes_with_pixels_and_order() {
    let a = prepare(&png(64, 64), 0).expect("a");
    let b = prepare(&png(64, 96), 1).expect("b");
    let ab = references_hash(&[&a, &b]);
    assert_eq!(ab, references_hash(&[&a, &b]));
    assert_ne!(ab, references_hash(&[&b, &a]));
    assert_ne!(ab, references_hash(&[&a]));
    assert_ne!(references_hash(&[&a]), references_hash(&[&b]));
}
