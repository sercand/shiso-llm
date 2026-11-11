//! The vision-language prefill's host arithmetic: which position axis each
//! rotary pair reads, the angle table, and the layout a call must have.
//!
//! The axis table is the reference's `apply_interleaved_mrope` written out:
//! pair `j` reads height when `j % 3 == 1` and `j < 3 * section[1]`, width
//! when `j % 3 == 2` and `j < 3 * section[2]`, time otherwise.

#![expect(
    clippy::cast_precision_loss,
    reason = "pair and token indices are small counts, exact in f64"
)]

use shiso_vision::generator::ImageGenError;

use super::qwen3_text::{VisionLayout, check_vision_layout, mrope_angles, mrope_axis};

/// The Qwen3-VL sections: 24 time pairs, 20 height, 20 width.
const SECTION: [usize; 3] = [24, 20, 20];
const IMAGE: u32 = 151_655;

#[test]
fn mrope_axis_assignment() {
    let axes: Vec<usize> = (0..64).map(|j| mrope_axis(j, SECTION)).collect();
    for (axis, &want) in SECTION.iter().enumerate() {
        assert_eq!(
            axes.iter().filter(|&&a| a == axis).count(),
            want,
            "axis {axis}"
        );
    }
    assert_eq!(&axes[..6], &[0, 1, 2, 0, 1, 2]);
    assert_eq!(&axes[57..], &[0, 1, 2, 0, 0, 0, 0]);
}

#[test]
fn text_positions_reduce_to_one_axis() {
    let (head_dim, theta) = (128usize, 5_000_000f32);
    let positions: Vec<[u32; 3]> = (0..7).map(|i| [i; 3]).collect();
    let (cos, sin) = mrope_angles(&positions, head_dim, theta, SECTION);
    assert_eq!(cos.len(), 7 * 64);
    for (t, row) in cos.chunks_exact(64).enumerate() {
        for (j, &c) in row.iter().enumerate() {
            let freq = f64::from(theta).powf(-2.0 * j as f64 / head_dim as f64);
            let angle = t as f64 * freq;
            assert!((f64::from(c) - angle.cos()).abs() < 1e-6, "cos t{t} j{j}");
            assert!(
                (f64::from(sin[t * 64 + j]) - angle.sin()).abs() < 1e-6,
                "sin t{t} j{j}"
            );
        }
    }
}

#[test]
fn an_image_slot_turns_each_pair_by_its_own_axis() {
    let (head_dim, theta) = (128usize, 5_000_000f32);
    // Time 22, height 25, width 31: one slot of a picture.
    let (cos, _) = mrope_angles(&[[22, 25, 31]], head_dim, theta, SECTION);
    let freq = |j: usize| f64::from(theta).powf(-2.0 * j as f64 / head_dim as f64);
    for (j, position) in [(0usize, 22.0f64), (1, 25.0), (2, 31.0), (61, 22.0)] {
        let want = (position * freq(j)).cos();
        assert!((f64::from(cos[j]) - want).abs() < 1e-6, "pair {j}");
    }
}

fn refusal(result: Result<(), ImageGenError>) -> String {
    match result {
        Err(ImageGenError::Backend(message)) => message,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_vision_layout_must_cover_exactly_the_image_ids() {
    let ids = [1, 2, IMAGE, IMAGE, 3, IMAGE, 4];
    let facts = |runs: &[(std::ops::Range<usize>, usize)], positions: usize| VisionLayout {
        ids: &ids,
        positions,
        runs: runs.to_vec(),
        image_token: IMAGE,
        deep_taps: 3,
    };
    assert!(check_vision_layout(&facts(&[(2..4, 3), (5..6, 3)], 7)).is_ok());
    // A run over a text id.
    let message = refusal(check_vision_layout(&facts(&[(1..4, 3), (5..6, 3)], 7)));
    assert!(message.contains("row 1"), "{message}");
    // An image id no run covers.
    refusal(check_vision_layout(&facts(&[(2..4, 3)], 7)));
    // Runs out of order, and a run past the sequence.
    refusal(check_vision_layout(&facts(&[(5..6, 3), (2..4, 3)], 7)));
    refusal(check_vision_layout(&facts(&[(2..4, 3), (5..9, 3)], 7)));
    // Too few deepstack row sets for the tower's taps.
    let message = refusal(check_vision_layout(&facts(&[(2..4, 2), (5..6, 3)], 7)));
    assert!(message.contains('3'), "{message}");
    // One position per id.
    refusal(check_vision_layout(&facts(&[(2..4, 3), (5..6, 3)], 6)));
}
