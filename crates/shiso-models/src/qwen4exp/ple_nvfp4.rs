//! The NVFP4 n-gram table of the NVIDIA export: 128 shards, each two arrays -
//! the packed `e2m1` codes and one `e4m3` scale byte per sixteen weights - and
//! one `f32` scale for the whole table.
//!
//! The reader assembles a row from a shard's two arrays, codes first, which is
//! the row [`PleRowType::Nvfp4`] decodes; the table scale is a fact of the
//! format, so it rides in the row type for the host codec and goes to the
//! device as the decode kernel's one float.

use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::ple::NgramTable;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_ple::{PleRowType, PleSegment, PleShards, PleStream};

use crate::qwen4exp::config::{PLE_HEAD_DIM, PLE_LAYER, Qwen4ExpConfig};
use crate::qwen4exp::mlx::{MlxCheckpoint, upload};

/// Prefix of every language-model tensor of this export.
const LM: &str = "model.language_model";
/// Shards the n-gram table is split into.
const NGRAM_SHARDS: usize = 128;
/// Weights one group scale covers.
const GROUP: u32 = 16;

/// Opens the NVFP4 n-gram table over its 128 two-array shards.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for a missing shard array or table scale,
/// [`CoreError::InvalidValue`] for an array of the wrong type or width, shards
/// of unequal row counts, or a table scale that is not a positive finite
/// number, and whatever the reader refuses.
pub fn open_ngram_nvfp4(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    config: &Qwen4ExpConfig,
    opts: &SparseOptions,
) -> CoreResult<NgramTable> {
    let ple = config.shape().ple.ok_or(CoreError::ShapeMismatch {
        field: "n-gram slot",
        expected: 1,
        got: 0,
    })?;
    let base = format!("{LM}.layers.{PLE_LAYER}.ple.ple_embedding.ngram_embedding");
    let global_name = format!("{base}.weight_scale_2");
    let raw = ck.bytes(&global_name, "F32", &[1])?;
    let global = raw
        .first_chunk::<4>()
        .map(|word| f32::from_le_bytes(*word))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: global_name.clone(),
        })?;
    // A zero, negative or non-finite scale decodes every row to zeros, sign
    // flips or NaNs - all finite-looking failures far from here.
    if !(global.is_finite() && global > 0.0) {
        return Err(CoreError::InvalidValue {
            key: "n-gram table scale",
            value: format!("{global_name} is {global}"),
        });
    }
    let row_type = PleRowType::Nvfp4 {
        global_bits: global.to_bits(),
    };
    let params = config.ngram().params(ple, row_type, None)?;
    let mut shards = Vec::with_capacity(NGRAM_SHARDS);
    let mut rows_per_shard = 0u64;
    for s in 0..NGRAM_SHARDS {
        let mut segments = Vec::with_capacity(2);
        for (part, dtype, width) in [
            ("weight", "U8", PLE_HEAD_DIM / 2),
            ("weight_scale", "F8_E4M3", PLE_HEAD_DIM / GROUP),
        ] {
            let name = format!("{base}.shard_{s}.{part}");
            let (_, raw) = ck.raw(&name)?;
            let rows = raw.shape.first().copied().unwrap_or(0);
            if raw.dtype != dtype || raw.shape.get(1).copied() != Some(u64::from(width)) {
                return Err(CoreError::InvalidValue {
                    key: "n-gram shard layout",
                    value: format!("{name} is {} {:?}", raw.dtype, raw.shape),
                });
            }
            if rows_per_shard == 0 {
                rows_per_shard = rows;
            } else if rows != rows_per_shard {
                return Err(CoreError::InvalidValue {
                    key: "n-gram shard rows",
                    value: format!("{name} holds {rows} rows, not {rows_per_shard}"),
                });
            }
            let (file, offset) = ck.file_offset(&name)?;
            segments.push(PleSegment {
                file,
                offset,
                row_bytes: width,
            });
        }
        shards.push(segments);
    }
    let stream = PleStream::open_sharded(
        PleShards {
            rows_per_shard,
            shards,
        },
        rows_per_shard * NGRAM_SHARDS as u64,
        params,
        opts.ngram_cache_bytes,
        &opts.ngram_reader,
    )?;
    let global_buffer = upload(backend, global.to_le_bytes().to_vec())?;
    NgramTable::new(DType::Nvfp4Gs16, Arc::new(stream), Some(global_buffer))
}
