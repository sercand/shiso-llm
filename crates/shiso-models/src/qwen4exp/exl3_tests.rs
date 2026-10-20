#![expect(clippy::indexing_slicing, reason = "test fixtures index known shapes")]

//! The trellis moves, checked on payloads whose every word names where it was
//! stored, so a word in the wrong place is visible as a wrong label.

use std::collections::HashMap;

use shiso_modelfile::st_header::RawTensor;

use super::{Move, NGRAM_ROW_WORDS, TableRows, Trellis, assemble, stack, table_rows};

/// Bits per weight of the fixtures: the narrowest the kernels decode.
const BITS: usize = 4;
/// Words in one tile at that width.
const TILE_WORDS: usize = 16 * BITS;

/// A projection `k x n` whose tile `(row, col)` holds `row * 1000 + col` in
/// every word, input scale `i` holds `i` and output scale `j` holds `j`.
fn labelled(k: usize, n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut tiles = Vec::new();
    for row in 0..k / 16 {
        for col in 0..n / 16 {
            let label = u16::try_from(row * 1000 + col).expect("fits");
            for _ in 0..TILE_WORDS {
                tiles.extend_from_slice(&label.to_le_bytes());
            }
        }
    }
    let scales = |len: usize| -> Vec<u8> {
        (0..len)
            .flat_map(|i| u16::try_from(i).expect("fits").to_le_bytes())
            .collect()
    };
    (tiles, scales(k), scales(n))
}

/// The label of tile `(row, col)` in an assembled payload of output width `n`.
fn tile_at(payload: &[u8], n: usize, row: usize, col: usize) -> u16 {
    let at = ((row * (n / 16) + col) * TILE_WORDS) * 2;
    u16::from_le_bytes([payload[at], payload[at + 1]])
}

/// The label of scale `i` in a region starting at byte `base`.
fn scale_at(payload: &[u8], base: usize, i: usize) -> u16 {
    u16::from_le_bytes([payload[base + 2 * i], payload[base + 2 * i + 1]])
}

fn trellis(k: usize, n: usize, parts: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> Trellis<'_> {
    Trellis {
        tiles: &parts.0,
        suh: &parts.1,
        svh: &parts.2,
        k,
        n,
        bits: BITS,
    }
}

#[test]
fn in_place_is_tiles_then_input_then_output_scales() {
    let (k, n) = (256, 256);
    let parts = labelled(k, n);
    let (payload, width) = assemble(&trellis(k, n, &parts), Move::InPlace).expect("moves");
    assert_eq!(width, n);
    let tiles = parts.0.len();
    assert_eq!(payload.len(), tiles + 2 * k + 2 * n);
    assert_eq!(tile_at(&payload, n, 3, 5), 3005);
    assert_eq!(scale_at(&payload, tiles, 200), 200);
    assert_eq!(scale_at(&payload, tiles + 2 * k, 17), 17);
}

/// An output block is eight tile columns and 128 output scales: moving it
/// moves both, past the blocks that stay in front.
#[test]
fn output_blocks_move_tile_columns_and_output_scales() {
    let (k, n) = (32, 512);
    let parts = labelled(k, n);
    // Keep block 0, then old blocks 3, 1, 2.
    let perm = [2, 0, 1];
    let (payload, width) = assemble(
        &trellis(k, n, &parts),
        Move::Outputs {
            lead: 1,
            perm: &perm,
        },
    )
    .expect("moves");
    assert_eq!(width, n);
    for row in 0..k / 16 {
        let label = |block: usize, within: usize| {
            u16::try_from(row * 1000 + block * 8 + within).expect("fits")
        };
        assert_eq!(
            tile_at(&payload, n, row, 3),
            label(0, 3),
            "lead block stays"
        );
        assert_eq!(tile_at(&payload, n, row, 8), label(3, 0));
        assert_eq!(tile_at(&payload, n, row, 16 + 7), label(1, 7));
        assert_eq!(tile_at(&payload, n, row, 24 + 2), label(2, 2));
    }
    let out = parts.0.len() + 2 * k;
    assert_eq!(scale_at(&payload, out, 5), 5);
    assert_eq!(scale_at(&payload, out, 128 + 9), 3 * 128 + 9);
    assert_eq!(scale_at(&payload, out, 256), 128);
    // The input scales are untouched.
    assert_eq!(scale_at(&payload, parts.0.len(), 31), 31);
}

/// An input block is eight tile rows and 128 input scales.
#[test]
fn input_blocks_move_tile_rows_and_input_scales() {
    let (k, n) = (384, 32);
    let parts = labelled(k, n);
    let perm = [2, 0, 1];
    let (payload, _) =
        assemble(&trellis(k, n, &parts), Move::Inputs { perm: &perm }).expect("moves");
    // New tile row 0 is old row 16 (first row of old block 2).
    assert_eq!(tile_at(&payload, n, 0, 1), 16 * 1000 + 1);
    assert_eq!(tile_at(&payload, n, 8 + 3, 0), 3 * 1000);
    assert_eq!(tile_at(&payload, n, 16 + 7, 1), (8 + 7) * 1000 + 1);
    let input = parts.0.len();
    assert_eq!(scale_at(&payload, input, 0), 256);
    assert_eq!(scale_at(&payload, input, 128), 0);
    assert_eq!(scale_at(&payload, input + 2 * k, 20), 20);
}

/// A slice keeps a range of tile columns and its output scales; the input
/// scales are the projection's own.
#[test]
fn slice_keeps_a_range_of_outputs() {
    let (k, n) = (32, 640);
    let parts = labelled(k, n);
    let (payload, width) =
        assemble(&trellis(k, n, &parts), Move::Slice { lo: 512, hi: 640 }).expect("moves");
    assert_eq!(width, 128);
    let tiles = (k / 16) * (128 / 16) * TILE_WORDS * 2;
    assert_eq!(payload.len(), tiles + 2 * k + 2 * 128);
    assert_eq!(tile_at(&payload, 128, 1, 0), 1000 + 32);
    assert_eq!(tile_at(&payload, 128, 0, 7), 39);
    assert_eq!(scale_at(&payload, tiles + 2 * k, 0), 512);
}

#[test]
fn moves_that_are_not_whole_blocks_are_refused() {
    let (k, n) = (32, 256);
    let parts = labelled(k, n);
    let t = trellis(k, n, &parts);
    assert!(assemble(&t, Move::Slice { lo: 64, hi: 256 }).is_err());
    assert!(
        assemble(
            &t,
            Move::Outputs {
                lead: 1,
                perm: &[0, 1]
            }
        )
        .is_err()
    );
    assert!(assemble(&t, Move::Inputs { perm: &[0] }).is_err());
}

/// A bank is every expert's tiles, then every expert's input scales, then
/// every expert's output scales - the three regions the kernel strides.
#[test]
fn a_bank_stacks_each_region_across_experts() {
    let (k, n) = (32, 128);
    let a = labelled(k, n);
    let mut b = labelled(k, n);
    b.0.iter_mut().for_each(|byte| *byte = byte.wrapping_add(1));
    let payload = stack(&[trellis(k, n, &a), trellis(k, n, &b)]).expect("stacks");
    let tiles = a.0.len();
    assert_eq!(&payload[..tiles], &a.0[..]);
    assert_eq!(&payload[tiles..2 * tiles], &b.0[..]);
    assert_eq!(&payload[2 * tiles..2 * tiles + 2 * k], &a.1[..]);
    let svh = 2 * tiles + 4 * k;
    assert_eq!(&payload[svh + 2 * n..], &b.2[..]);

    let narrow = labelled(k, 256);
    assert!(stack(&[trellis(k, n, &a), trellis(k, 256, &narrow)]).is_err());
}

/// The n-gram tensors' prefix in the table file.
const NGRAM: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
/// Bytes in one stored n-gram row.
const ROW_BYTES: u64 = NGRAM_ROW_WORDS * 2;

/// A header entry of `rows` table rows stored from byte `start`.
fn rows_at(start: u64, rows: u64) -> RawTensor {
    RawTensor {
        dtype: "I16".to_string(),
        shape: vec![rows, NGRAM_ROW_WORDS],
        data_offsets: [start, start + rows * ROW_BYTES],
    }
}

/// A table file's header: the given table tensors after a small tensor that
/// stands in for the bias, which the row lookup must ignore.
fn header(table: Vec<(String, RawTensor)>) -> HashMap<String, RawTensor> {
    let mut map: HashMap<String, RawTensor> = table.into_iter().collect();
    map.insert(
        format!("{NGRAM}.head_bias"),
        RawTensor {
            dtype: "F16".to_string(),
            shape: vec![16, 160],
            data_offsets: [0, 5120],
        },
    );
    map
}

/// Shards `0..count` of `rows` rows each, stored back to back from byte
/// 5120, in index order.
fn shards(count: u64, rows: u64) -> Vec<(String, RawTensor)> {
    (0..count)
        .map(|i| {
            (
                format!("{NGRAM}.shard_{i}.trellis"),
                rows_at(5120 + i * rows * ROW_BYTES, rows),
            )
        })
        .collect()
}

/// The single-tensor table is found as before.
#[test]
fn a_single_table_tensor_is_its_own_range() {
    let map = header(vec![(format!("{NGRAM}.trellis"), rows_at(5120, 1000))]);
    assert_eq!(
        table_rows(&map).expect("single table"),
        TableRows {
            start: 5120,
            rows: 1000
        }
    );
}

/// Shards stored back to back are one range from the first shard's start,
/// whatever order the header lists them in, and past shard 9 (a name order
/// would put `shard_10` before `shard_2`).
#[test]
fn contiguous_shards_join_into_one_range() {
    let mut parts = shards(12, 7);
    parts.reverse();
    let map = header(parts);
    assert_eq!(
        table_rows(&map).expect("joined"),
        TableRows {
            start: 5120,
            rows: 84
        }
    );
}

/// A gap between two shards, a missing index, a second layout beside the
/// shards, or a shard of another width is refused: joining any of them would
/// read rows from the wrong bytes.
#[test]
fn shards_that_are_not_one_range_are_refused() {
    let mut gap = shards(3, 7);
    gap[2].1 = rows_at(gap[2].1.data_offsets[0] + 2, 7);
    assert!(table_rows(&header(gap)).is_err());

    let mut missing = shards(3, 7);
    missing.remove(1);
    assert!(table_rows(&header(missing)).is_err());

    let mut both = shards(2, 7);
    both.push((format!("{NGRAM}.trellis"), rows_at(5120, 14)));
    assert!(table_rows(&header(both)).is_err());

    let mut wide = shards(2, 7);
    wide[1].1.shape = vec![7, NGRAM_ROW_WORDS + 1];
    assert!(table_rows(&header(wide)).is_err());

    assert!(table_rows(&header(Vec::new())).is_err());
}
