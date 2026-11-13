#![expect(clippy::indexing_slicing, reason = "test fixtures index known shapes")]

//! The draft head's index parser, the plane-to-block re-layout and the
//! indexer split, on synthetic bytes and on the real `dense.txt`.

use half::f16;
use shiso_core::error::CoreError;

use super::{
    BLOCK_BYTES, BLOCK_CODES, BLOCK_VALUES, DenseIndex, DenseKind, RECORD_BYTES, RecordShape,
    blocks_of_row, record_to_blocks, split_rows,
};

/// The real pack's `mtp/dense.txt`, byte for byte.
const REAL_INDEX: &str = "\
fc_embedding.weight q8_0 2560 2560 0 6963200
fc_hidden.weight q8_0 2560 2560 6963200 6963200
hyper_connection_mixer.hc_norm.weight f32 1 10240 13926400 40960
hyper_connection_mixer.input_mix_weight_down.weight bf16 320 10240 13967360 6553600
hyper_connection_mixer.input_mix_weight_up.weight bf16 10240 320 20520960 6553600
attn_hyper_connection.block_inject_weight.weight bf16 4 10240 27074560 81920
attn_hyper_connection.hc_norm.weight f32 1 10240 27156480 40960
attn_hyper_connection.input_mix_weight_down.weight bf16 320 10240 27197440 6553600
attn_hyper_connection.input_mix_weight_up.weight bf16 10240 320 33751040 6553600
mlp.gate.weight bf16 512 2560 40304640 2621440
mlp.shared_expert.down_proj.weight q8_0 2560 640 42926080 1740800
mlp.shared_expert.gate_proj.weight q8_0 640 2560 44666880 1740800
mlp.shared_expert.up_proj.weight q8_0 640 2560 46407680 1740800
mlp.shared_expert_gate.weight bf16 1 2560 48148480 5120
mlp_hyper_connection.block_inject_weight.weight bf16 4 10240 48153600 81920
mlp_hyper_connection.hc_norm.weight f32 1 10240 48235520 40960
mlp_hyper_connection.input_mix_weight_down.weight bf16 320 10240 48276480 6553600
mlp_hyper_connection.input_mix_weight_up.weight bf16 10240 320 54830080 6553600
self_attn.indexer.index_qk_proj.weight q8_0 640 2560 61383680 1740800
self_attn.indexer.k_layernorm.weight f32 1 128 63124480 512
self_attn.indexer.q_layernorm.weight f32 1 128 63124992 512
self_attn.k_norm.weight f32 1 256 63125504 1024
self_attn.k_proj.weight q8_0 512 2560 63126528 1392640
self_attn.o_proj.weight q8_0 2560 6144 64519168 16711680
self_attn.q_norm.weight f32 1 256 81230848 1024
self_attn.q_proj.weight q8_0 12288 2560 81231872 33423360
self_attn.v_proj.weight q8_0 512 2560 114655232 1392640
pre_fc_norm_embedding.weight f32 1 2560 116047872 10240
pre_fc_norm_hidden.weight f32 1 10240 116058112 40960
";

fn invalid(result: Result<DenseIndex, CoreError>) -> String {
    match result {
        Err(CoreError::InvalidValue { key, value }) => {
            assert_eq!(key, "dense.txt");
            value
        }
        other => panic!("wanted a refusal naming dense.txt, got {other:?}"),
    }
}

#[test]
fn dense_index_parses() {
    let index = DenseIndex::parse(REAL_INDEX).expect("the real index parses");
    assert_eq!(index.entries.len(), 29);
    let q = index
        .entries
        .iter()
        .find(|e| e.name == "self_attn.q_proj.weight")
        .expect("q_proj is listed");
    assert_eq!(q.kind, DenseKind::Q8);
    assert_eq!((q.rows, q.cols), (12_288, 2_560));
    assert_eq!(q.offset, 81_231_872);
    assert_eq!(q.bytes, 33_423_360);
    // The last tensor ends where dense.bin does.
    assert_eq!(index.extent(), 116_099_072);
    // Comments and blank lines are allowed; the tensors are the same.
    let padded = format!("# draft head\n\n{REAL_INDEX}\n");
    assert_eq!(
        DenseIndex::parse(&padded).expect("parses").entries,
        index.entries
    );
}

#[test]
fn dense_index_refuses_overlap_and_unknown_kind() {
    // Two tensors of 8 bytes, the second starting inside the first.
    let overlap = "a.weight f32 1 2 0 8\nb.weight f32 1 2 4 8\n";
    assert!(invalid(DenseIndex::parse(overlap)).contains("overlaps"));
    // Out of file order is fine while they do not overlap.
    assert!(DenseIndex::parse("b.weight f32 1 2 8 8\na.weight f32 1 2 0 8\n").is_ok());
    let kind = invalid(DenseIndex::parse("a.weight q4_k 1 256 0 144\n"));
    assert!(kind.contains("unknown kind"), "{kind}");
    // A byte count the shape does not give.
    assert!(invalid(DenseIndex::parse("a.weight f32 1 2 0 9\n")).contains("is 8 bytes"));
    // A q8_0 row that is not whole blocks.
    assert!(invalid(DenseIndex::parse("a.weight q8_0 1 40 0 34\n")).contains("whole blocks"));
    // Wrong field count, bad number, repeated name.
    assert!(invalid(DenseIndex::parse("a.weight f32 1 2 0\n")).contains("fields"));
    assert!(invalid(DenseIndex::parse("a.weight f32 x 2 0 8\n")).contains("not a number"));
    let twice = "a.weight f32 1 2 0 8\na.weight f32 1 2 8 8\n";
    assert!(invalid(DenseIndex::parse(twice)).contains("twice"));
}

/// A deterministic byte stream.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ((x >> 11) & 0xFF) as u8
        })
        .collect()
}

/// Scale bytes: finite half-precision values, sign and mantissa random.
fn scales(count: usize, seed: u32) -> Vec<u8> {
    noise(count * 2, seed)
        .chunks_exact(2)
        .flat_map(|p| (u16::from_le_bytes([p[0], p[1]]) & 0xFBFF).to_le_bytes())
        .collect()
}

/// A weight of a plane row, by the pack's own formula: value of input `j` is
/// `((codes[j / 4] >> 2 * (j % 4)) & 3) - 1` times the scale of block `j / 64`.
fn from_planes(codes_row: &[u8], scales_row: &[u8], j: usize) -> f32 {
    let d = f16::from_le_bytes([scales_row[2 * (j / 64)], scales_row[2 * (j / 64) + 1]]);
    let c = (codes_row[j / 4] >> (2 * (j % 4))) & 3;
    (f32::from(c) - 1.0) * d.to_f32()
}

/// A weight of a row of shiso's `Q2_0` blocks (18 bytes a 64 values: `f16 d`,
/// then four 2-bit codes a byte, low bits first; the codes 0..3 stand for -1..2
/// steps of the scale), as `kernels/cuda/q4e/ggml.cu` and the kernel test's
/// decoder read it.
fn from_blocks(row: &[u8], j: usize) -> f32 {
    let block = &row[(j / 64) * 18..][..18];
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let e = j % 64;
    let c = (block[2 + e / 4] >> ((e % 4) * 2)) & 3;
    d * (f32::from(c) - 1.0)
}

#[test]
fn expert_planes_to_blocks() {
    assert_eq!(BLOCK_BYTES, 18);
    assert_eq!(BLOCK_CODES * 4, BLOCK_VALUES);
    // The family's widths give the pack's record size.
    assert_eq!(RecordShape::FAMILY.record_bytes(), RECORD_BYTES);

    // One plane pair, 4 rows of 64 inputs: each block is its scale, then its
    // 16 code bytes.
    let (codes, sc) = (noise(4 * 16, 1), scales(4, 2));
    let mut out = [0u8; 4 * 18];
    for r in 0..4 {
        blocks_of_row(
            &codes[r * 16..][..16],
            &sc[r * 2..][..2],
            &mut out[r * 18..][..18],
        );
    }
    for r in 0..4 {
        assert_eq!(&out[r * 18..][..2], &sc[r * 2..][..2]);
        assert_eq!(&out[r * 18 + 2..][..16], &codes[r * 16..][..16]);
    }

    // A whole record at a toy width: 128 inputs, 64 inner, so gate and up rows
    // are two blocks and the down rows one.
    let shape = RecordShape {
        hidden: 128,
        inner: 64,
    };
    let [gu_codes, down_codes, gu_scales, down_scales] = shape.planes();
    let mut record = noise(gu_codes + down_codes, 3);
    record.extend(scales(gu_scales / 2, 4));
    record.extend(scales(down_scales / 2, 5));
    assert_eq!(record.len(), shape.record_bytes());
    let (mut gate, mut up, mut down) = (
        vec![0u8; shape.gate_bytes()],
        vec![0u8; shape.gate_bytes()],
        vec![0u8; shape.down_bytes()],
    );
    record_to_blocks(shape, &record, &mut gate, &mut up, &mut down).expect("re-lays out");

    let (codes, rest) = record.split_at(gu_codes);
    let (down_codes, rest) = rest.split_at(down_codes);
    let (gu_scales, down_scales) = rest.split_at(gu_scales);
    let (gu_row, gu_sc, gu_out) = (128 / 4, 128 / 64 * 2, 128 / 64 * 18);
    for r in 0..shape.inner {
        // Gate row r is interleaved row 2r, up row r is 2r + 1.
        for (plane_row, bank) in [(2 * r, &gate), (2 * r + 1, &up)] {
            let (c, s) = (
                &codes[plane_row * gu_row..][..gu_row],
                &gu_scales[plane_row * gu_sc..][..gu_sc],
            );
            let blocks = &bank[r * gu_out..][..gu_out];
            for j in 0..shape.hidden {
                assert_eq!(
                    from_planes(c, s, j).to_bits(),
                    from_blocks(blocks, j).to_bits(),
                    "row {r} input {j} of plane row {plane_row}"
                );
            }
        }
    }
    let (d_row, d_sc, d_out) = (64 / 4, 2, 18);
    for r in 0..shape.hidden {
        let (c, s) = (
            &down_codes[r * d_row..][..d_row],
            &down_scales[r * d_sc..][..d_sc],
        );
        let blocks = &down[r * d_out..][..d_out];
        for j in 0..shape.inner {
            assert_eq!(
                from_planes(c, s, j).to_bits(),
                from_blocks(blocks, j).to_bits()
            );
        }
    }
    // Gate and up are not the same rows: the interleave is real.
    assert_ne!(gate, up);

    // A slice of the wrong size is refused, naming nothing it cannot know.
    let mut short = vec![0u8; shape.gate_bytes() - 1];
    assert!(matches!(
        record_to_blocks(shape, &record, &mut short, &mut up, &mut down),
        Err(CoreError::ShapeMismatch { .. })
    ));
}

#[test]
fn indexer_proj_splits_512_128() {
    // 640 rows of one 34-byte block, row `r` filled with `r % 251`.
    let bytes: Vec<u8> = (0..640u32).flat_map(|r| [(r % 251) as u8; 34]).collect();
    let (q, k) = split_rows(&bytes, 640, 512).expect("splits");
    assert_eq!((q.len(), k.len()), (512 * 34, 128 * 34));
    assert_eq!(q[0], 0);
    assert_eq!(q[511 * 34], (511 % 251) as u8);
    assert_eq!(k[0], (512 % 251) as u8);
    assert_eq!(k[127 * 34], (639 % 251) as u8);
    // Bytes that are not whole rows, or a cut past the end, are refused.
    assert!(split_rows(&bytes[1..], 640, 512).is_none());
    assert!(split_rows(&bytes, 640, 641).is_none());
}
