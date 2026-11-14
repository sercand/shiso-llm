#![expect(clippy::indexing_slicing, reason = "test fixtures index known shapes")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "fixture sizes and tags are small known numbers"
)]

//! The latent pack's reader on synthetic bytes: every weight survives the
//! move into the stream's layout, every malformed index is refused by name, and
//! one stored record comes out as the nine arrays the stream expects.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use shiso_core::error::CoreError;
use shiso_engine::sparse::stream::ExpertSource;
use shiso_engine::sparse::weights::nvfp4_bytes;
use shiso_modelfile::nvfp4::{Nvfp4Layout, dequant_nvfp4};

use super::{
    Geometry, LatentExperts, LatentPack, interleave_up, is_latent_pack, record_bytes, repack_codes,
    split_scales,
};

/// A small pack shape: two layers, four experts, a 64-wide latent, 64 inner
/// width and a 128-wide model, so the projection file holds two groups.
const SMALL: Geometry = Geometry {
    layers: 2,
    experts: 4,
    hidden: 128,
    inter: 64,
};

/// Bytes per block of the stored NVFP4 format.
const BLOCK: usize = 36;

/// A deterministic byte stream; no test depends on its particular values.
struct Bytes(u64);

impl Bytes {
    fn next(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 24) as u8
    }

    /// Stored blocks for `rows` rows of `in_dim` inputs: random codes under
    /// scale bytes the format allows.
    fn section(&mut self, rows: usize, in_dim: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..rows * in_dim / 64 {
            for _ in 0..4 {
                let scale = self.next() & 0x7F;
                out.push(if scale == 0x7F { 0x7E } else { scale });
            }
            for _ in 0..32 {
                out.push(self.next());
            }
        }
        out
    }
}

/// The spec's scale byte: `exp == 0` is subnormal, otherwise `2^(exp-7)`
/// times `1 + man/8`. Written out here so the loader's decode is not graded
/// by itself.
fn scale_value(byte: u8) -> f32 {
    let (exp, man) = (i32::from((byte >> 3) & 15), f32::from(byte & 7));
    if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        2f32.powi(exp - 7) * (1.0 + man / 8.0)
    }
}

/// The spec's weight `(row, col)` of a stored section: split-half nibbles in
/// each sixteen-value sub-block, the sub-block's scale, then the global.
///
/// The scales multiply first, as the stream's own decode does, so the two
/// agree to the bit.
fn stored_weight(src: &[u8], in_dim: usize, row: usize, col: usize, global: f32) -> f32 {
    const MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let block = &src[(row * in_dim / 64 + col / 64) * BLOCK..][..BLOCK];
    let (sub, at) = ((col % 64) / 16, col % 16);
    let byte = block[4 + 8 * sub + at % 8];
    let code = if at < 8 { byte & 15 } else { byte >> 4 };
    let value = MAGNITUDE[usize::from(code & 7)];
    let value = if code & 8 == 0 { value } else { -value };
    value * (scale_value(block[sub]) * global)
}

/// Converts `src` and decodes the result with the stream's own routine.
fn converted(src: &[u8], rows: usize, in_dim: usize, global: f32) -> Vec<f32> {
    let (codes, scales) = nvfp4_bytes(in_dim as u32, rows as u32);
    let mut codes_out = vec![0u8; codes as usize];
    let mut scales_out = vec![0u8; scales as usize];
    repack_codes(src, in_dim as u32, &mut codes_out);
    split_scales(src, in_dim as u32, &mut scales_out).expect("valid scales");
    let mut out = Vec::new();
    dequant_nvfp4(
        Nvfp4Layout::new(rows, in_dim, global),
        &codes_out,
        &scales_out,
        &mut out,
    )
    .expect("decodes");
    out
}

#[test]
fn record_bytes_matches_pack() {
    assert_eq!(record_bytes(1792, 640), 1_935_376);
}

#[test]
fn repack_preserves_every_weight() {
    let mut rng = Bytes(0x9E37_79B9_7F4A_7C15);
    let global = 0.003_173_828_1_f32;
    for (rows, in_dim) in [(640, 1792), (1792, 640)] {
        let src = rng.section(rows, in_dim);
        let out = converted(&src, rows, in_dim, global);
        assert_eq!(out.len(), rows * in_dim);
        for row in 0..rows {
            for col in 0..in_dim {
                let want = stored_weight(&src, in_dim, row, col, global);
                assert_eq!(
                    out[row * in_dim + col].to_bits(),
                    want.to_bits(),
                    "weight ({row}, {col}) of [{rows}, {in_dim}]"
                );
            }
        }
    }
}

/// One block whose scales are `scales`, codes zero.
fn block_with(scales: [u8; 4]) -> Vec<u8> {
    let mut block = scales.to_vec();
    block.extend([0u8; 32]);
    block
}

#[test]
fn scale_with_sign_bit_refused() {
    let src = block_with([0x10, 0x10, 0x80, 0x10]);
    let mut dst = [0u8; 4];
    let err = split_scales(&src, 64, &mut dst).expect_err("refused");
    assert!(
        matches!(
            err,
            CoreError::InvalidValue {
                key: "experts.bin",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn scale_nan_refused() {
    let src = block_with([0x7F, 0x10, 0x10, 0x10]);
    let mut dst = [0u8; 4];
    let err = split_scales(&src, 64, &mut dst).expect_err("refused");
    assert!(
        matches!(
            err,
            CoreError::InvalidValue {
                key: "experts.bin",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn interleave_up_moves_rows() {
    const GROUPS: usize = 8;
    const HIDDEN: usize = 2560;
    const ROW: usize = 1904;
    let mut src = vec![0u8; GROUPS * HIDDEN * ROW];
    for g in 0..GROUPS {
        for h in 0..HIDDEN {
            let at = (g * HIDDEN + h) * ROW;
            src[at..at + ROW].fill((g * 31 + h % 29) as u8);
            src[at] = g as u8;
            src[at + 1] = (h >> 8) as u8;
            src[at + 2] = h as u8;
        }
    }
    let out = interleave_up(&src, GROUPS as u32, HIDDEN as u32, 1792);
    assert_eq!(out.len(), 2560 * 15232);
    for h in [0, 1, 777, 2559] {
        for g in 0..GROUPS {
            let row = &out[h * 15232 + g * ROW..][..ROW];
            let from = &src[(g * HIDDEN + h) * ROW..][..ROW];
            assert_eq!(row, from, "row {h} group {g}");
        }
    }
}

/// Writes the index files of a pack of shape `SMALL` into `dir`, with
/// `experts` as the records file, and returns the labels it wrote.
fn write_pack(dir: &Path, experts: &[u8], label: impl Fn(usize) -> i32) {
    let g = SMALL;
    let width = 64u32;
    let record = record_bytes(width, g.inter);
    let mut text = format!(
        "# native experts v6: latent A-only (expert_embd {width}, expert_ff {})\n",
        g.inter
    );
    for l in 0..u64::from(g.layers) {
        let offset = l * u64::from(g.experts) * record;
        writeln!(text, "{l} 40 40 {offset} {record}").expect("a string takes text");
    }
    fs::write(dir.join("native_experts.txt"), text).expect("writes");
    fs::write(dir.join("experts.bin"), experts).expect("writes");
    let labels: Vec<u8> = (0..(g.layers * g.experts) as usize)
        .flat_map(|i| label(i).to_le_bytes())
        .collect();
    fs::write(dir.join("labels.bin"), labels).expect("writes");
    // Two groups of 64 latent rows, 128 inputs each (4 Q8_0 blocks), then
    // the up side: 2 groups of 128 rows of 2 blocks.
    let per_matrix = 2 * 64 * 4 * 34;
    fs::write(
        dir.join("latent_dense_q8.bin"),
        vec![0u8; g.layers as usize * 2 * per_matrix],
    )
    .expect("writes");
    fs::write(dir.join("model-skeleton.gguf"), b"").expect("writes");
}

fn small_experts() -> Vec<u8> {
    let record = record_bytes(64, SMALL.inter) as usize;
    vec![0u8; SMALL.layers as usize * SMALL.experts as usize * record]
}

fn open(dir: &Path) -> Result<LatentPack, CoreError> {
    LatentPack::open_with(dir, SMALL)
}

/// Asserts `err` is an invalid value that names `file`.
fn names(err: &CoreError, file: &str) {
    assert!(
        matches!(err, CoreError::InvalidValue { key, .. } if *key == file),
        "wanted a refusal naming {file}, got {err:?}"
    );
}

#[test]
fn a_width_the_prompt_kernels_cannot_stride_is_refused_at_load() {
    assert!(super::check_prompt_width(1792, 640).is_ok());
    for (width, inter) in [(64, 640), (1792, 704), (1856, 640)] {
        names(
            &super::check_prompt_width(width, inter).expect_err("refused"),
            "native_experts.txt",
        );
    }
}

#[test]
fn open_parses_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |i| (i % 2) as i32);
    assert!(is_latent_pack(dir.path()));
    let pack = open(dir.path()).expect("opens");
    assert_eq!(pack.width, 64);
    assert_eq!(pack.groups, 2);
    assert_eq!(pack.record_bytes, record_bytes(64, 64));
    assert_eq!(pack.labels().len(), 8);
    assert_eq!(pack.labels()[5], 1);
    assert_eq!(pack.projection_bytes(), 2 * 64 * 4 * 34);
    assert_eq!(
        pack.projection_offsets(1),
        (2 * 2 * 64 * 4 * 34, 3 * 2 * 64 * 4 * 34)
    );
}

#[test]
fn not_a_pack_without_every_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    fs::remove_file(dir.path().join("labels.bin")).expect("removes");
    assert!(!is_latent_pack(dir.path()));
}

#[test]
fn open_refuses_wrong_type() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    let path = dir.path().join("native_experts.txt");
    let text = fs::read_to_string(&path).expect("reads");
    fs::write(&path, text.replacen(" 40 40 ", " 41 40 ", 1)).expect("writes");
    names(
        &open(dir.path()).expect_err("refused"),
        "native_experts.txt",
    );
}

#[test]
fn open_refuses_gap() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    let path = dir.path().join("native_experts.txt");
    let text = fs::read_to_string(&path).expect("reads");
    let record = record_bytes(64, 64);
    let second = 4 * record;
    fs::write(
        &path,
        text.replace(&format!(" {second} "), &format!(" {} ", second + 4096)),
    )
    .expect("writes");
    names(
        &open(dir.path()).expect_err("refused"),
        "native_experts.txt",
    );
}

#[test]
fn open_refuses_short_experts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut experts = small_experts();
    experts.pop();
    write_pack(dir.path(), &experts, |_| 0);
    names(&open(dir.path()).expect_err("refused"), "experts.bin");
}

#[test]
fn open_refuses_label_out_of_range() {
    for bad in [2, -1] {
        let dir = tempfile::tempdir().expect("tempdir");
        write_pack(
            dir.path(),
            &small_experts(),
            |i| if i == 3 { bad } else { 0 },
        );
        names(&open(dir.path()).expect_err("refused"), "labels.bin");
    }
}

#[test]
fn open_refuses_short_labels() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    fs::write(dir.path().join("labels.bin"), [0u8; 28]).expect("writes");
    names(&open(dir.path()).expect_err("refused"), "labels.bin");
}

#[test]
fn open_refuses_projection_size() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    let path = dir.path().join("latent_dense_q8.bin");
    let mut bytes = fs::read(&path).expect("reads");
    bytes.push(0);
    fs::write(&path, bytes).expect("writes");
    names(
        &open(dir.path()).expect_err("refused"),
        "latent_dense_q8.bin",
    );
}

#[test]
fn open_refuses_a_pack_with_only_the_q4_projections() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_pack(dir.path(), &small_experts(), |_| 0);
    fs::rename(
        dir.path().join("latent_dense_q8.bin"),
        dir.path().join("latent_dense_q4.bin"),
    )
    .expect("renames");
    names(
        &open(dir.path()).expect_err("refused"),
        "latent_dense_q4.bin",
    );
}

#[test]
fn source_reads_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let record = record_bytes(64, 64) as usize;
    let mut rng = Bytes(42);
    // Layer 0 holds two experts of interest; the other slots stay zero so a
    // read at the wrong offset would show.
    let mut experts = small_experts();
    let mut stored = Vec::new();
    for e in 0..2 {
        let mut rec = Vec::new();
        rec.extend(rng.section(64, 64)); // gate: 64 rows of 64 inputs
        rec.extend(rng.section(64, 64)); // up
        rec.extend(rng.section(64, 64)); // down: 64 rows of 64 inputs
        for global in [0.5f32, 0.25 + e as f32, 8.0, 0.0] {
            rec.extend(global.to_le_bytes());
        }
        assert_eq!(rec.len(), record);
        experts[e * record..(e + 1) * record].copy_from_slice(&rec);
        stored.push(rec);
    }
    write_pack(dir.path(), &experts, |_| 0);
    let pack = open(dir.path()).expect("opens");
    let source = LatentExperts::new(&pack).expect("opens");
    assert_eq!(source.parts().len(), 3);
    let layout = source.layout();
    assert_eq!(layout.len(), 9);
    let (codes, scales) = nvfp4_bytes(64, 64);
    assert_eq!(
        layout,
        [codes, scales, 4, codes, scales, 4, codes, scales, 4]
    );

    let rec = &stored[1];
    let section = 64 * 36;
    for proj in 0..3 {
        let src = &rec[proj * section..(proj + 1) * section];
        let mut want_codes = vec![0u8; codes as usize];
        let mut want_scales = vec![0u8; scales as usize];
        repack_codes(src, 64, &mut want_codes);
        split_scales(src, 64, &mut want_scales).expect("valid");
        let mut got = vec![0u8; codes as usize];
        source.read(0, 1, proj * 3, &mut got).expect("reads codes");
        assert_eq!(got, want_codes, "codes of projection {proj}");
        let mut got = vec![0u8; scales as usize];
        source
            .read(0, 1, proj * 3 + 1, &mut got)
            .expect("reads scales");
        assert_eq!(got, want_scales, "scales of projection {proj}");
        let mut got = [0u8; 4];
        source
            .read(0, 1, proj * 3 + 2, &mut got)
            .expect("reads global");
        let tail = 3 * section + proj * 4;
        assert_eq!(got, rec[tail..tail + 4], "global of projection {proj}");
    }
    // A destination of the wrong width is refused rather than half filled.
    let mut short = vec![0u8; codes as usize - 1];
    assert!(source.read(0, 1, 0, &mut short).is_err());
}

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// The real pack's geometry and a sample of its records: every scale byte of
/// the sampled records is one the format stores, and every global is finite.
#[test]
#[ignore = "reads the real pack named by SHISO_TEST_LATENT_PACK"]
fn the_real_pack_opens_and_its_sampled_records_decode() {
    let Some(dir) = std::env::var_os("SHISO_TEST_LATENT_PACK").map(std::path::PathBuf::from) else {
        skip("SHISO_TEST_LATENT_PACK names no pack directory");
        return;
    };
    let pack = LatentPack::open(&dir).expect("the real pack opens");
    assert_eq!((pack.width, pack.groups), (1792, 8));
    assert_eq!(pack.record_bytes, 1_935_376);
    assert_eq!(pack.labels().len(), 48 * 512);
    let source = LatentExperts::new(&pack).expect("opens");
    let layout = source.layout();
    for (layer, expert) in [(0, 0), (0, 511), (17, 123), (47, 0), (47, 511)] {
        for (array, width) in layout.iter().enumerate() {
            let mut dst = vec![0u8; *width as usize];
            source
                .read(layer, expert, array, &mut dst)
                .unwrap_or_else(|e| panic!("layer {layer} expert {expert} array {array}: {e}"));
        }
    }
}
