//! The CSF decode on hand-built records, where every byte's meaning is known.

use super::*;

/// One 16-row record for a 16 x 64 matrix (four scale columns).
fn record(bases: [u8; 16], codes: [[u8; 2]; 16]) -> Vec<u8> {
    let mut out = bases.to_vec();
    for row in codes {
        out.extend_from_slice(&row);
    }
    out
}

const LAYOUT: CsfLayout = CsfLayout { rows: 16, cols: 64 };

#[test]
fn a_scale_is_its_rows_base_plus_its_code_with_the_even_column_low() {
    let mut bases = [0u8; 16];
    bases[0] = 100;
    bases[1] = 7;
    let mut codes = [[0u8; 2]; 16];
    // Row 0: columns 0..4 carry codes 1, 2, 3, 15.
    codes[0] = [0x21, 0xF3];
    codes[1] = [0x00, 0x10];
    let mut out = vec![0u8; LAYOUT.scale_bytes()];
    decode_csf("t", LAYOUT, &record(bases, codes), &[], &mut out).expect("decodes");
    assert_eq!(&out[0..4], &[101, 102, 103, 115]);
    assert_eq!(&out[4..8], &[7, 7, 7, 8]);
}

#[test]
fn an_exception_replaces_the_whole_byte_at_its_position() {
    let bases = [50u8; 16];
    let codes = [[0x11u8; 2]; 16];
    // Row 2, column 3 -> position 2 * 4 + 3 = 11, value 0x7E (448).
    let word: u32 = (0x7E << 24) | 0x0B;
    let mut out = vec![0u8; LAYOUT.scale_bytes()];
    decode_csf(
        "t",
        LAYOUT,
        &record(bases, codes),
        &word.to_le_bytes(),
        &mut out,
    )
    .expect("decodes");
    assert_eq!(out[11], 0x7E);
    assert_eq!(out[10], 51);
}

#[test]
fn a_base_above_240_is_refused_because_its_sum_could_wrap() {
    let mut bases = [0u8; 16];
    bases[5] = 241;
    let mut out = vec![0u8; LAYOUT.scale_bytes()];
    let err = decode_csf("t", LAYOUT, &record(bases, [[0; 2]; 16]), &[], &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "{err:?}"
    );
}

#[test]
fn exceptions_out_of_order_or_out_of_range_are_refused() {
    let fixed = record([0; 16], [[0; 2]; 16]);
    let mut out = vec![0u8; LAYOUT.scale_bytes()];
    let mut words = Vec::new();
    words.extend_from_slice(&5u32.to_le_bytes());
    words.extend_from_slice(&5u32.to_le_bytes());
    let err = decode_csf("t", LAYOUT, &fixed, &words, &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "duplicate: {err:?}"
    );
    let outside = 64u32.to_le_bytes();
    let err = decode_csf("t", LAYOUT, &fixed, &outside, &mut out);
    assert!(
        matches!(err, Err(GgufError::LayoutMismatch { .. })),
        "outside: {err:?}"
    );
}

#[test]
fn arrays_of_the_wrong_length_are_refused() {
    let fixed = record([0; 16], [[0; 2]; 16]);
    let mut out = vec![0u8; LAYOUT.scale_bytes()];
    let err = decode_csf("t", LAYOUT, &fixed[1..], &[], &mut out);
    assert!(
        matches!(err, Err(GgufError::SizeMismatch { .. })),
        "{err:?}"
    );
    let err = decode_csf("t", LAYOUT, &fixed, &[0, 0, 0], &mut out);
    assert!(
        matches!(err, Err(GgufError::SizeMismatch { .. })),
        "{err:?}"
    );
    let err = decode_csf("t", LAYOUT, &fixed, &[], &mut out[1..]);
    assert!(
        matches!(err, Err(GgufError::SizeMismatch { .. })),
        "{err:?}"
    );
}

#[test]
fn the_familys_expert_shapes_have_the_record_sizes_the_checkpoint_stores() {
    // gate/up: [640, 2560] -> fixed [40, 1296]; down: [2560, 640] -> [160, 336].
    let gate = CsfLayout {
        rows: 640,
        cols: 2560,
    };
    assert_eq!(
        (gate.rows / ROWS_PER_RECORD, gate.record_bytes()),
        (40, 1296)
    );
    let down = CsfLayout {
        rows: 2560,
        cols: 640,
    };
    assert_eq!(
        (down.rows / ROWS_PER_RECORD, down.record_bytes()),
        (160, 336)
    );
}
