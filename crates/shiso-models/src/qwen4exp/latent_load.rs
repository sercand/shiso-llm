//! Putting a latent pack's pieces on the device: the dense latent projections,
//! the label buffers, the stream over the records and the n-gram table.
//!
//! [`super::latent`] reads and checks the files; this module moves what they
//! hold. The model type's own `load_latent` strings these together with the
//! skeleton's dense import, and owns the order the memory is claimed in.
//!
//! # What is copied and what is not
//!
//! Nothing is widened. `D` is the stored bytes of a `Q8_0` tensor of dims
//! `[2560, 14336]`; `U` is the same blocks laid out as one `[14336, 2560]`
//! matrix ([`interleave_up`]), the only byte movement in the load. Both are
//! read with the page cache bypassed (3.7 GB that is then resident on the
//! device would otherwise be cached a second time) and uploaded.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::ple::NgramTable;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::sparse::stream::ExpertStream;
use shiso_engine::sparse::weights::{LatentBank, Weight};
use shiso_modelfile::GgufFile;
use shiso_modelfile::st_header::open_uncached;
use shiso_ple::{PleRowType, PleStream};

use crate::qwen4exp::config::{
    EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_USED, HIDDEN, N_LAYERS, PLE_HEAD_DIM, Qwen4ExpConfig,
};
use crate::qwen4exp::latent::{LatentExperts, LatentPack, interleave_up};
use crate::qwen4exp::mlx::upload;

/// The pack's n-gram container.
const NGRAM_FILE: &str = "ple-fp8.gguf";
/// The one tensor of that container.
const NGRAM_TENSOR: &str = "per_layer_token_embd.weight";
/// Suffix of the metadata key that names the rows' codec.
const FORMAT_SUFFIX: &str = ".ple.format";
/// Suffix of the metadata key that holds the one scale of the table.
const SCALE_SUFFIX: &str = ".ple.scale";
/// The codec name the table must state.
const FORMAT_FP8: &str = "f8_e4m3";

/// Bytes one expert occupies in the pool: its nine arrays.
#[must_use]
pub(super) fn slot_bytes(pack: &LatentPack) -> u64 {
    let (width, inter) = (pack.width, EXPERT_INTERMEDIATE);
    [(width, inter), (width, inter), (inter, width)]
        .iter()
        .map(|&(i, o)| {
            let (codes, scales) = shiso_engine::sparse::weights::nvfp4_bytes(i, o);
            codes + scales + 4
        })
        .sum()
}

/// The pack's dense latent projections, read layer by layer.
#[derive(Debug)]
pub(super) struct LatentProjections {
    file: File,
}

/// The routed half of one layer's [`ExpertBank`].
#[derive(Debug)]
pub(super) struct LatentBanks {
    /// The routed gate, streamed.
    pub(super) gate: Weight,
    /// The routed up, streamed.
    pub(super) up: Weight,
    /// The routed down, streamed.
    pub(super) down: Weight,
    /// The latent projections and labels.
    pub(super) latent: LatentBank,
}

/// Bytes both projections of every layer occupy once resident.
#[must_use]
pub(super) fn projection_resident_bytes(pack: &LatentPack) -> u64 {
    pack.projection_bytes() * 2 * u64::from(N_LAYERS)
}

impl LatentProjections {
    /// Opens the projections file of `pack`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] when the file cannot be opened.
    pub(super) fn open(pack: &LatentPack) -> CoreResult<Self> {
        Ok(Self {
            file: open_uncached(pack.projection_path()).map_err(CoreError::Io)?,
        })
    }

    /// Layer `layer`'s routed banks: the placeholders of the gate, up and down
    /// at the latent width (the stream fills their slots) and the latent half,
    /// the two projections and the label buffer.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] for a short read, and whatever the uploads report.
    pub(super) fn attach(
        &self,
        backend: &Arc<dyn Backend>,
        pack: &LatentPack,
        layer: u32,
    ) -> CoreResult<LatentBanks> {
        let hidden = HIDDEN;
        let (groups, width) = (pack.groups, pack.width);
        let rows = groups * width;
        if layer >= N_LAYERS {
            return Err(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: u64::from(N_LAYERS),
            });
        }
        let (down_at, up_at) = pack.projection_offsets(layer);
        let bytes = pack.projection_bytes();
        let len = usize::try_from(bytes).map_err(|_| CoreError::ShapeMismatch {
            field: "latent projection bytes",
            expected: u64::from(u32::MAX),
            got: bytes,
        })?;
        let mut stored = vec![0u8; len];
        self.file
            .read_exact_at(&mut stored, down_at)
            .map_err(CoreError::Io)?;
        let down = Weight::uploaded(backend, DType::Q8_0, hidden, rows, 1, stored)?;
        let mut stored = vec![0u8; len];
        self.file
            .read_exact_at(&mut stored, up_at)
            .map_err(CoreError::Io)?;
        let up = Weight::uploaded(
            backend,
            DType::Q8_0,
            rows,
            hidden,
            1,
            interleave_up(&stored, groups, hidden, width),
        )?;
        let at = layer as usize * EXPERTS as usize;
        let mine = pack
            .labels()
            .get(at..at + EXPERTS as usize)
            .ok_or(CoreError::OutOfBounds {
                offset: at as u64,
                len: u64::from(EXPERTS),
                buffer_bytes: pack.labels().len() as u64,
            })?;
        // The pack reader refused a label past the group count; asserted again
        // here because the kernels cannot check it.
        if let Some(bad) = mine.iter().find(|l| **l >= groups) {
            return Err(CoreError::InvalidValue {
                key: "labels.bin",
                value: format!("label {bad} with {groups} groups"),
            });
        }
        let labels = upload(backend, mine.iter().flat_map(|l| l.to_le_bytes()).collect())?;
        let streamed = |in_dim: u32, out_dim: u32| -> CoreResult<Weight> {
            Ok(Weight::streamed(
                DType::Nvfp4Gs16,
                in_dim,
                out_dim,
                EXPERTS,
                shiso_core::buffer::alloc(backend, 16)?,
            ))
        };
        Ok(LatentBanks {
            gate: streamed(width, EXPERT_INTERMEDIATE)?,
            up: streamed(width, EXPERT_INTERMEDIATE)?,
            down: streamed(EXPERT_INTERMEDIATE, width)?,
            latent: LatentBank {
                down,
                up,
                labels,
                groups,
                width,
            },
        })
    }
}

/// Opens the stream over the pack's records; a pool that holds every expert
/// is filled at once rather than on first use, as
/// [`super::nvfp4::open_stream`] does.
///
/// # Errors
///
/// Whatever [`LatentExperts::new`], [`ExpertStream::open`] and the fill report.
pub(super) fn open_stream(
    backend: &Arc<dyn Backend>,
    pack: &LatentPack,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let source = Arc::new(LatentExperts::new(pack)?);
    let min_slots = widest_pass.saturating_mul(EXPERTS_USED).min(EXPERTS);
    let stream = ExpertStream::open(backend, source, N_LAYERS, EXPERTS, slots, min_slots)?;
    if slots >= EXPERTS * N_LAYERS {
        stream.preload(backend.as_ref())?;
    }
    Ok(stream)
}

/// Rows the table is stored in are rounded up to a multiple of this past the
/// last head: the pack's table holds 320,001,536 rows behind 320,001,446
/// addressable ones, which is the next multiple of 128. A table longer than
/// that rounding is not this pack's, and rows read past the hash would be
/// another tensor's bytes.
const ROW_ROUNDING: u64 = 128;

/// Refuses a stored row count that is not the one the hashing metadata
/// needs, allowing the writer's rounding ([`ROW_ROUNDING`]) and nothing more.
pub(super) fn check_rows(stored: u64, needed: u64) -> CoreResult<()> {
    if stored >= needed && stored - needed < ROW_ROUNDING {
        return Ok(());
    }
    Err(CoreError::ShapeMismatch {
        field: "n-gram table rows",
        expected: needed,
        got: stored,
    })
}

/// Widens a half-precision vector to `f32` bytes, exactly.
///
/// The widening is lossless (every half value, subnormals included, is an
/// `f32`), so it is not the dequantisation the load otherwise avoids. A byte
/// length that is odd or does not match `elements` is refused rather than
/// truncated.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when `bytes` is not exactly `2 * elements`.
pub(super) fn widen_half(bytes: &[u8], elements: u64) -> CoreResult<Vec<u8>> {
    let want = elements.saturating_mul(2);
    if bytes.len() as u64 != want || !bytes.len().is_multiple_of(2) {
        return Err(CoreError::ShapeMismatch {
            field: "half-precision vector bytes",
            expected: want,
            got: bytes.len() as u64,
        });
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|pair| f32::from(half::f16::from_le_bytes(*pair)).to_le_bytes())
        .collect())
}

/// The n-gram container, checked: where its rows start and how they decode.
pub(super) struct NgramFile {
    file: GgufFile,
    offset: u64,
    rows: u64,
    scale: f32,
}

/// Opens `dir/ple-fp8.gguf` and holds it to what this reader decodes: the
/// codec name, a tensor of raw bytes, rows exactly one n-gram head wide, a
/// positive finite scale, and bytes that cover every row. The two metadata
/// keys are found by their suffix, since the prefix names the container's
/// writer.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for an absent file, key or tensor,
/// [`CoreError::InvalidValue`] for a codec, type or scale this reader does not
/// decode, [`CoreError::ShapeMismatch`] for a row width that disagrees, and
/// [`CoreError::Backend`] for a file whose tensor runs past its end.
pub(super) fn read_table(dir: &Path) -> CoreResult<NgramFile> {
    let path = dir.join(NGRAM_FILE);
    let file = GgufFile::open_shared(&path).map_err(|source| CoreError::Backend {
        backend: "qwen4exp",
        op: "open the n-gram table",
        source: Box::new(source),
    })?;
    let meta = file.metadata();
    let by_suffix = |suffix: &str| {
        let mut hits = meta
            .entries()
            .iter()
            .filter(|(key, _)| key.ends_with(suffix));
        let first = hits.next();
        (first, hits.next().is_some())
    };
    let (format, many) = by_suffix(FORMAT_SUFFIX);
    let format = format
        .filter(|_| !many)
        .and_then(|(key, _)| meta.get_str(key))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{NGRAM_FILE}: one key ending {FORMAT_SUFFIX}"),
        })?;
    if format != FORMAT_FP8 {
        return Err(CoreError::InvalidValue {
            key: NGRAM_FILE,
            value: format!("row format {format:?}, only {FORMAT_FP8:?} is decoded"),
        });
    }
    let (scale, many) = by_suffix(SCALE_SUFFIX);
    let scale = scale
        .filter(|_| !many)
        .and_then(|(key, _)| meta.get_f32(key))
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{NGRAM_FILE}: one f32 key ending {SCALE_SUFFIX}"),
        })?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(CoreError::InvalidValue {
            key: NGRAM_FILE,
            value: format!("table scale {scale}"),
        });
    }
    let desc = file
        .find(NGRAM_TENSOR)
        .ok_or_else(|| CoreError::ConfigMissing {
            key: NGRAM_TENSOR.to_owned(),
        })?;
    if desc.dtype != DType::I8 {
        return Err(CoreError::InvalidValue {
            key: NGRAM_FILE,
            value: format!("{NGRAM_TENSOR} is {}, expected i8", desc.dtype.type_name()),
        });
    }
    let dims = desc.shape.dims();
    let (width, rows) = (
        dims.first().copied().unwrap_or(0),
        dims.get(1).copied().unwrap_or(0),
    );
    if dims.len() != 2 || width != u64::from(PLE_HEAD_DIM) {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram row bytes",
            expected: u64::from(PLE_HEAD_DIM),
            got: width,
        });
    }
    // The container's reader has already refused a tensor that runs past the
    // end of the file; this is the same fact stated for the rows, so a change
    // in how bytes are counted cannot let a short table through.
    let bytes = rows.saturating_mul(width);
    if desc.bytes != bytes {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram table bytes",
            expected: bytes,
            got: desc.bytes,
        });
    }
    let offset = u64::try_from(file.range_of(desc).start).unwrap_or(u64::MAX);
    Ok(NgramFile {
        file,
        offset,
        rows,
        scale,
    })
}

/// Opens the pack's n-gram table, `ple-fp8.gguf`, as one contiguous stream of
/// `f8_e4m3` rows under the one scale its metadata holds.
///
/// # Errors
///
/// As [`read_table`], plus [`CoreError::ShapeMismatch`] for a row count other
/// than the hash needs ([`check_rows`]), and whatever the stream's open
/// reports.
pub(super) fn open_ngram(
    backend: &Arc<dyn Backend>,
    dir: &Path,
    config: &Qwen4ExpConfig,
    opts: &SparseOptions,
) -> CoreResult<NgramTable> {
    let table = read_table(dir)?;
    let ple = config.shape().ple.ok_or(CoreError::ShapeMismatch {
        field: "n-gram slot",
        expected: 1,
        got: 0,
    })?;
    let params = config.ngram().params(
        ple,
        PleRowType::Fp8E4M3 {
            scale_bits: table.scale.to_bits(),
        },
        None,
    )?;
    check_rows(table.rows, params.total_rows())?;
    let desc = table
        .file
        .find(NGRAM_TENSOR)
        .ok_or_else(|| CoreError::ConfigMissing {
            key: NGRAM_TENSOR.to_owned(),
        })?;
    let shard = File::open(table.file.path_of(desc)).map_err(CoreError::Io)?;
    let stream = PleStream::open(
        shard,
        table.offset,
        table.rows,
        params,
        opts.ngram_cache_bytes,
        &opts.ngram_reader,
    )?;
    let scale_buffer = upload(backend, table.scale.to_le_bytes().to_vec())?;
    NgramTable::new(DType::Fp8E4M3, Arc::new(stream), Some(scale_buffer))
}

#[cfg(test)]
#[path = "latent_load_tests.rs"]
mod tests;
