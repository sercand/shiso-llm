//! The conversion against hand-computed values.

use super::*;

/// A one-pixel-wide helper: converts a 2x2 frame whose pixels share one colour.
fn flat(luma: u8, blue: u8, red: u8, matrix: YuvMatrix, range: YuvRange) -> [u8; 3] {
    let planes = [luma; 4];
    let frame = Yuv420 {
        layout: YuvLayout::I420,
        y: &planes,
        y_stride: 2,
        u: &[blue],
        v: &[red],
        uv_stride: 1,
        width: 2,
        height: 2,
    };
    let rgb = yuv420_to_rgb8(&frame, matrix, range).expect("converts");
    [rgb[0], rgb[1], rgb[2]]
}

fn near(a: [u8; 3], b: [u8; 3], tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

#[test]
fn limited_range_black_and_white_hit_the_ends() {
    assert_eq!(
        flat(16, 128, 128, YuvMatrix::Bt709, YuvRange::Limited),
        [0, 0, 0]
    );
    assert_eq!(
        flat(235, 128, 128, YuvMatrix::Bt709, YuvRange::Limited),
        [255, 255, 255]
    );
}

#[test]
fn full_range_is_not_stretched() {
    assert_eq!(
        flat(0, 128, 128, YuvMatrix::Bt601, YuvRange::Full),
        [0, 0, 0]
    );
    assert_eq!(
        flat(200, 128, 128, YuvMatrix::Bt601, YuvRange::Full),
        [200, 200, 200]
    );
}

#[test]
fn the_matrix_changes_the_colour() {
    // Pure red, limited range: the two matrices disagree on Y and U.
    let bt601 = flat(81, 90, 240, YuvMatrix::Bt601, YuvRange::Limited);
    let bt709 = flat(63, 102, 240, YuvMatrix::Bt709, YuvRange::Limited);
    assert!(near(bt601, [255, 0, 0], 3), "601 red decoded as {bt601:?}");
    assert!(near(bt709, [255, 0, 0], 3), "709 red decoded as {bt709:?}");
    // The same samples under the wrong matrix are visibly off.
    let wrong = flat(81, 90, 240, YuvMatrix::Bt709, YuvRange::Limited);
    assert!(
        !near(wrong, [255, 0, 0], 8),
        "a wrong matrix went unnoticed: {wrong:?}"
    );
}

#[test]
fn nv12_and_i420_agree() {
    let luma: Vec<u8> = (0..16).map(|i| 40 + i * 9).collect();
    let (blue, red) = ([90u8, 110, 130, 150], [200u8, 180, 160, 140]);
    let i420 = Yuv420 {
        layout: YuvLayout::I420,
        y: &luma,
        y_stride: 4,
        u: &blue,
        v: &red,
        uv_stride: 2,
        width: 4,
        height: 4,
    };
    let uv = [90u8, 200, 110, 180, 130, 160, 150, 140];
    let nv12 = Yuv420 {
        layout: YuvLayout::Nv12,
        y: &luma,
        y_stride: 4,
        u: &uv,
        v: &[],
        uv_stride: 4,
        width: 4,
        height: 4,
    };
    let planar = yuv420_to_rgb8(&i420, YuvMatrix::Bt709, YuvRange::Limited).expect("i420");
    let semi = yuv420_to_rgb8(&nv12, YuvMatrix::Bt709, YuvRange::Limited).expect("nv12");
    assert_eq!(planar, semi);
}

#[test]
fn ten_bit_samples_match_their_eight_bit_equivalent() {
    let eight = flat(81, 90, 240, YuvMatrix::Bt601, YuvRange::Limited);
    let (luma, blue, red) = ([81u16 << 2; 4], [90u16 << 2], [240u16 << 2]);
    let frame = Yuv420Deep {
        y: &luma,
        y_stride: 2,
        u: &blue,
        v: &red,
        uv_stride: 1,
        width: 2,
        height: 2,
        bit_depth: 10,
    };
    let rgb = yuv420_deep_to_rgb8(&frame, YuvMatrix::Bt601, YuvRange::Limited).expect("converts");
    assert!(
        near([rgb[0], rgb[1], rgb[2]], eight, 2),
        "{rgb:?} vs {eight:?}"
    );
}

#[test]
fn a_short_plane_is_an_error_not_a_panic() {
    let frame = Yuv420 {
        layout: YuvLayout::I420,
        y: &[0; 3],
        y_stride: 2,
        u: &[128],
        v: &[128],
        uv_stride: 1,
        width: 2,
        height: 2,
    };
    assert!(yuv420_to_rgb8(&frame, YuvMatrix::Bt601, YuvRange::Full).is_err());
}

#[test]
fn an_unspecified_matrix_follows_the_picture_height() {
    assert_eq!(YuvMatrix::from_code(2, 1080), YuvMatrix::Bt709);
    assert_eq!(YuvMatrix::from_code(2, 480), YuvMatrix::Bt601);
    assert_eq!(YuvMatrix::from_code(9, 480), YuvMatrix::Bt2020);
}
