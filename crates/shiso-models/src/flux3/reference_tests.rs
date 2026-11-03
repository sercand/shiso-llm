//! Reference-image preparation against the reference implementation's own
//! numbers: the canvas table, the padded pictures' position-weighted sums
//! and the multi-reference crop sizes were printed by
//! `scripts/reference/flux3_reference_prep.py` running its `pipeline.py`.

use image::{ImageFormat, Rgb, RgbImage};
use shiso_vision::generator::{ImageGenError, InputReferenceImage};

use super::reference::{
    check_reference_count, default_size_for_references, prepare_multi_reference,
    prepare_single_reference, reference_canvas,
};

/// An input size, its fitted size and its canvas.
type CanvasCase = ((u32, u32), (u32, u32), (u32, u32));

/// The deterministic picture the capture script builds:
/// `v(x, y, c) = (7x + 13y + 29c + xy) mod 256`.
fn picture(w: u32, h: u32) -> RgbImage {
    RgbImage::from_fn(w, h, |x, y| {
        let v = |c: u32| u8::try_from((7 * x + 13 * y + 29 * c + x * y) % 256).expect("below 256");
        Rgb([v(0), v(1), v(2)])
    })
}

fn weighted_sum(img: &RgbImage) -> u64 {
    img.as_raw()
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u64 + 1) * u64::from(v))
        .sum()
}

fn png(w: u32, h: u32) -> InputReferenceImage {
    let mut bytes = Vec::new();
    picture(w, h)
        .write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("encodes");
    InputReferenceImage { bytes }
}

#[test]
fn reference_canvas_math() {
    // (input, fitted size, canvas), from the reference's `reference_canvas`.
    let table: [CanvasCase; 13] = [
        ((3000, 2000), (2496, 1664), (2496, 1664)),
        ((40, 4000), (40, 4000), (64, 4000)),
        ((1000, 750), (1000, 750), (1008, 752)),
        ((1024, 1024), (1024, 1024), (1024, 1024)),
        ((5000, 100), (4096, 82), (4096, 96)),
        ((4096, 4096), (2048, 2048), (2048, 2048)),
        ((17, 17), (17, 17), (64, 64)),
        ((100, 40), (100, 40), (112, 64)),
        ((1, 1), (1, 1), (64, 64)),
        ((2000, 2000), (2000, 2000), (2000, 2000)),
        ((4000, 1000), (4000, 1000), (4000, 1008)),
        ((2048, 2048), (2048, 2048), (2048, 2048)),
        ((2049, 2048), (2048, 2048), (2048, 2048)),
    ];
    for ((w, h), fit, canvas) in table {
        let got = reference_canvas(w, h);
        assert_eq!(got, (fit, canvas), "{w}x{h}");
        assert_eq!(canvas.0 % 16, 0, "{w}x{h}: canvas width on the grid");
        assert_eq!(canvas.1 % 16, 0, "{w}x{h}: canvas height on the grid");
        assert!(
            u64::from(canvas.0) * u64::from(canvas.1) <= 4 << 20,
            "{w}x{h}"
        );
    }
}

#[test]
fn narrow_reference_uses_symmetric_pad() {
    // 4x40: a side of 16 or less cannot reflect, so the pad mirrors with the
    // edge repeated, and folds again because the pad is wider than the side.
    let (canvas, boxed) = prepare_single_reference(&picture(4, 40));
    assert_eq!(canvas.dimensions(), (64, 64));
    assert_eq!(boxed, Some([30, 12, 34, 52]));
    assert_eq!(weighted_sum(&canvas), 9_440_054_784);
}

#[test]
fn wide_references_reflect_without_repeating_the_edge() {
    for ((w, h), want_box, want_sum) in [
        ((20, 17), [22, 23, 42, 40], 9_834_985_344),
        ((33, 20), [15, 22, 48, 42], 9_703_610_880),
    ] {
        let (canvas, boxed) = prepare_single_reference(&picture(w, h));
        assert_eq!(canvas.dimensions(), (64, 64), "{w}x{h}");
        assert_eq!(boxed, Some(want_box), "{w}x{h}");
        assert_eq!(weighted_sum(&canvas), want_sum, "{w}x{h}");
    }
}

#[test]
fn a_reference_on_its_canvas_is_returned_as_it_is() {
    let original = picture(128, 96);
    let (canvas, boxed) = prepare_single_reference(&original);
    assert_eq!(boxed, None);
    assert_eq!(canvas, original);
}

#[test]
fn multi_reference_fits_under_one_megapixel_and_crops_to_the_grid() {
    for ((w, h), want) in [
        ((2000, 1500), (1168, 880)),
        ((1000, 1000), (992, 992)),
        ((1023, 17), (1008, 16)),
        ((31, 47), (16, 32)),
    ] {
        let got = prepare_multi_reference(&picture(w, h));
        assert_eq!(got.dimensions(), want, "{w}x{h}");
    }
}

#[test]
fn multi_reference_crop_is_centred() {
    // 1000x1000 is under 1 MP, so no resize: the 992 crop takes 4 pixels
    // off each side, and pixel (0, 0) of the crop is the source's (4, 4).
    let source = picture(1000, 1000);
    let cropped = prepare_multi_reference(&source);
    assert_eq!(cropped.get_pixel(0, 0), source.get_pixel(4, 4));
    assert_eq!(cropped.get_pixel(991, 991), source.get_pixel(995, 995));
}

#[test]
fn more_than_ten_references_refused() {
    assert!(check_reference_count(10).is_ok());
    match check_reference_count(11) {
        Err(ImageGenError::ReferenceImage(message)) => {
            assert!(
                message.contains("11") && message.contains("10"),
                "{message}"
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn the_default_size_is_the_first_references_canvas() {
    assert_eq!(
        default_size_for_references(&[]).expect("none"),
        (1024, 1024)
    );
    assert_eq!(
        default_size_for_references(&[png(100, 40)]).expect("one"),
        (112, 64)
    );
    // Several references: the first one's prepared (cropped) size.
    assert_eq!(
        default_size_for_references(&[png(100, 40), png(64, 64)]).expect("two"),
        (96, 32)
    );
}

#[test]
fn a_reference_that_does_not_decode_or_is_too_small_is_refused() {
    let garbage = InputReferenceImage {
        bytes: b"not an image".to_vec(),
    };
    assert!(matches!(
        default_size_for_references(&[garbage]),
        Err(ImageGenError::ReferenceImage(_))
    ));
    match default_size_for_references(&[png(8, 64)]) {
        Err(ImageGenError::ReferenceImage(message)) => {
            assert!(
                message.contains("8x64") && message.contains("16"),
                "{message}"
            );
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}
