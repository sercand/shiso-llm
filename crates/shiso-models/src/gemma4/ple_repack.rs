//! One-time repack of MLX per-layer embedding weights into a streamed sidecar.
//!
//! The MLX export stores `embed_tokens_per_layer` as three separate arrays:
//! `weight` (packed u32 words), `scales` (bf16), and `biases` (bf16). A streaming
//! read during prefill and decode fetches one row per token, which would be
//! three disjoint file reads per token.
//!
//! This module repacks the three arrays into a single row-grouped binary file
//! `<dir>/shiso-ple-affine4.bin`:
//! - a [`HEADER_BYTES`]-byte header (`magic: b"SPLA"`, version, rows, row stride,
//!   group size, bit width, slot count, and the source byte count the row data
//!   was built from)
//! - per row: [`WORDS_BYTES`] of packed words, then [`SCALES_BYTES`] of scales,
//!   then [`SCALES_BYTES`] of biases - [`ROW_BYTES`] in all.
//!
//! Every one of those numbers is derived from the element type, the slot count
//! and the slot width rather than written down, and the block below asserts the
//! derivation against the row stride the engine computes for the same element
//! type. `kernel_dense_ple_prepare_affine4` derives the same three bases from
//! the slot count and slot width it is handed, so neither side carries the
//! layout as a literal.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use shiso_core::{CoreError, CoreResult, DType};
use shiso_modelfile::safetensors::AffineTriple;

use crate::gemma4::config::{GEMMA4_LAYERS, GEMMA4_PLE_DIM, GEMMA4_PLE_VOCAB};

/// Sidecar magic bytes ("SPLA" = Shiso Per-Layer Affine).
pub const MAGIC: &[u8; 4] = b"SPLA";
/// Sidecar format version.
pub const VERSION: u32 = 1;
/// Header size in bytes.
pub const HEADER_BYTES: usize = 64;

/// Element type one repacked row is written in.
const ROW_DTYPE: DType = DType::MlxAffine4Gs64;
/// Layer slots spanned by one row.
pub const SLOTS: u32 = GEMMA4_LAYERS;
/// Weights one slot of a row carries.
const SLOT_DIM: u32 = GEMMA4_PLE_DIM;
/// Group size for quantization.
pub const GROUP_SIZE: u32 = ROW_DTYPE.block_elems().get();
/// Quantization bits.
pub const BITS: u32 = 4;
/// Quantization groups one row carries, over every slot.
const GROUPS: u32 = SLOTS * (SLOT_DIM / GROUP_SIZE);
/// Bytes of packed words one row carries.
pub const WORDS_BYTES: u32 = GROUPS * (GROUP_SIZE * BITS / 8);
/// Bytes of scales - and, separately, of biases - one row carries.
pub const SCALES_BYTES: u32 = GROUPS * 2;
/// Row stride in the sidecar: words, then scales, then biases.
pub const ROW_BYTES: u32 = WORDS_BYTES + 2 * SCALES_BYTES;
/// Total rows in the vocabulary.
pub const VOCAB_ROWS: u64 = GEMMA4_PLE_VOCAB;

// `kernel_dense_ple_prepare_affine4` derives the same three bases from the slot
// count and the slot width it is passed, so the one thing the two sides have to
// agree about is the four-bit, group-of-64 layout those derivations assume.
const _: () = assert!(BITS == 4 && GROUP_SIZE == 64);
// And the row stride has to be the one the engine derives for this element
// type: the streamed reader's stride and the kernel's `row_bytes` argument both
// come from there, and a row written at another stride is read from the middle
// of its predecessor.
const _: () = assert!(
    ROW_BYTES as u64
        == SLOTS as u64
            * ROW_DTYPE.block_bytes().get() as u64
            * (SLOT_DIM / ROW_DTYPE.block_elems().get()) as u64
);

/// Sidecar file name.
pub const SIDECAR_FILENAME: &str = "shiso-ple-affine4.bin";

/// Validates an existing sidecar file. Returns `true` if valid.
fn validate_existing(path: &Path, source_len: u64) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let Ok(meta) = file.metadata() else {
        return false;
    };
    let expected_len = (HEADER_BYTES as u64) + VOCAB_ROWS * u64::from(ROW_BYTES);
    if meta.len() != expected_len {
        return false;
    }

    let mut header = [0u8; HEADER_BYTES];
    if file.read_exact(&mut header).is_err() {
        return false;
    }

    if &header[0..4] != MAGIC {
        return false;
    }
    let Some(ver) = header[4..8].try_into().ok().map(u32::from_le_bytes) else {
        return false;
    };
    let Some(rows) = header[8..16].try_into().ok().map(u64::from_le_bytes) else {
        return false;
    };
    let Some(row_bytes) = header[16..20].try_into().ok().map(u32::from_le_bytes) else {
        return false;
    };
    let Some(group_size) = header[20..24].try_into().ok().map(u32::from_le_bytes) else {
        return false;
    };
    let Some(bits) = header[24..28].try_into().ok().map(u32::from_le_bytes) else {
        return false;
    };
    let Some(slots) = header[28..32].try_into().ok().map(u32::from_le_bytes) else {
        return false;
    };
    let Some(src_len) = header[32..40].try_into().ok().map(u64::from_le_bytes) else {
        return false;
    };

    ver == VERSION
        && rows == VOCAB_ROWS
        && row_bytes == ROW_BYTES
        && group_size == GROUP_SIZE
        && bits == BITS
        && slots == SLOTS
        && src_len == source_len
}

/// Ensures the repacked sidecar binary exists beside the checkpoint, creating it if absent.
///
/// # Errors
///
/// I/O errors or validation failures.
pub fn ensure_affine_sidecar(dir: &Path, table: AffineTriple<'_>) -> CoreResult<PathBuf> {
    let sidecar_path = dir.join(SIDECAR_FILENAME);
    let source_len = (table.words.len() + table.scales.len() + table.biases.len()) as u64;

    if validate_existing(&sidecar_path, source_len) {
        tracing::info!(
            path = %sidecar_path.display(),
            "existing MLX affine PLE sidecar verified"
        );
        return Ok(sidecar_path);
    }

    let words_row_bytes = WORDS_BYTES as usize;
    let scales_row_bytes = SCALES_BYTES as usize;
    let biases_row_bytes = SCALES_BYTES as usize;

    let rows = usize::try_from(VOCAB_ROWS).unwrap_or(usize::MAX);
    let total_words = rows * words_row_bytes;
    let total_scales = rows * scales_row_bytes;
    let total_biases = rows * biases_row_bytes;

    if table.words.len() != total_words
        || table.scales.len() != total_scales
        || table.biases.len() != total_biases
    {
        return Err(CoreError::InvalidValue {
            key: "embed_tokens_per_layer shapes",
            value: format!(
                "expected words {}B, scales {}B, biases {}B; got {}B, {}B, {}B",
                total_words,
                total_scales,
                total_biases,
                table.words.len(),
                table.scales.len(),
                table.biases.len(),
            ),
        });
    }

    tracing::info!(
        path = %sidecar_path.display(),
        "repacking MLX affine PLE table into sidecar binary (~1.58 GB)..."
    );
    let start = Instant::now();

    let tmp_path = dir.join(format!("{SIDECAR_FILENAME}.tmp"));
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)
        .map_err(|e| CoreError::Backend {
            backend: "io",
            op: "create_sidecar_tmp",
            source: Box::new(e),
        })?;

    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    // Write 64-byte header
    let mut header = [0u8; HEADER_BYTES];
    header[0..4].copy_from_slice(MAGIC);
    header[4..8].copy_from_slice(&VERSION.to_le_bytes());
    header[8..16].copy_from_slice(&VOCAB_ROWS.to_le_bytes());
    header[16..20].copy_from_slice(&ROW_BYTES.to_le_bytes());
    header[20..24].copy_from_slice(&GROUP_SIZE.to_le_bytes());
    header[24..28].copy_from_slice(&BITS.to_le_bytes());
    header[28..32].copy_from_slice(&SLOTS.to_le_bytes());
    header[32..40].copy_from_slice(&source_len.to_le_bytes());

    writer.write_all(&header).map_err(|e| CoreError::Backend {
        backend: "io",
        op: "write_sidecar_header",
        source: Box::new(e),
    })?;

    // Write per-row interleaved bytes
    for r in 0..rows {
        let w_slice = &table.words[r * words_row_bytes..(r + 1) * words_row_bytes];
        let s_slice = &table.scales[r * scales_row_bytes..(r + 1) * scales_row_bytes];
        let b_slice = &table.biases[r * biases_row_bytes..(r + 1) * biases_row_bytes];

        writer.write_all(w_slice).map_err(|e| CoreError::Backend {
            backend: "io",
            op: "write_row_words",
            source: Box::new(e),
        })?;
        writer.write_all(s_slice).map_err(|e| CoreError::Backend {
            backend: "io",
            op: "write_row_scales",
            source: Box::new(e),
        })?;
        writer.write_all(b_slice).map_err(|e| CoreError::Backend {
            backend: "io",
            op: "write_row_biases",
            source: Box::new(e),
        })?;
    }

    writer.flush().map_err(|e| CoreError::Backend {
        backend: "io",
        op: "flush_sidecar",
        source: Box::new(e),
    })?;
    drop(writer);

    std::fs::rename(&tmp_path, &sidecar_path).map_err(|e| CoreError::Backend {
        backend: "io",
        op: "rename_sidecar",
        source: Box::new(e),
    })?;

    tracing::info!(
        duration_sec = ?start.elapsed().as_secs_f32(),
        "repacked MLX affine PLE sidecar complete"
    );

    Ok(sidecar_path)
}
