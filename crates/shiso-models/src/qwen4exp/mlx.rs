//! The array-framework checkpoint of this family: MLX 4-bit affine weights in
//! a directory of safetensors shards, loaded onto the sparse stack.
//!
//! # What changes between the two containers, and what does not
//!
//! The stack's kernels read a layout the trellis container was converted into,
//! and that conversion did more than change codecs: it moved rows. Loading this
//! container means doing the same moves here, once, at load, on the stored
//! bytes. Every one of them lands on whole affine rows or whole affine groups,
//! so each is a permutation of the stored arrays and not a requantization:
//!
//! - the recurrent block's value heads go from the reference's grouped order
//!   (every value head of a key head together) to the tiled order the kernels
//!   index (the `r`-th value head of every key head together) - rows of the
//!   fused input projection, of the output gate's projection and of the decay
//!   and write-strength projections, 128-column blocks of the output
//!   projection's contraction (two groups of 64 each), the per-head decay base
//!   and bias, and the convolution's value channels;
//! - the decay base is stored as `-exp(A_log)` (the norms the reference
//!   applies as `1 + w` need nothing: this container's conversion already
//!   stored them with the one added, which is also what the kernels read);
//! - the indexer's fused query/key projection is two row ranges of one tensor.
//!
//! Everything else is imported in place: the file's own pages are what the
//! device reads, and nothing is copied.
//!
//! # The routed banks
//!
//! Most of the checkpoint's bytes are the routed experts. When they fit beside
//! everything else they are imported in place like every other weight; when
//! they do not, they are described but not imported, and an
//! [`ExpertStream`] reads each expert out of these same arrays when a router
//! first picks it. [`ExpertResidency`] is that choice.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::TokenId;
use shiso_core::ledger::{self, MemTag};
use shiso_engine::sparse::ple::NgramTable;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::sparse::stream::{ExpertSource, ExpertStream};
use shiso_engine::sparse::weights::WeightPart;
use shiso_engine::sparse::weights::{
    ExpertBank, Indexer, NgramBlock, Recurrent, SparseAttn, SparseBlock, SparseLayer,
    SparseWeights, StreamMix, Weight,
};
use shiso_modelfile::st_header::{RawTensor, Shard};
use shiso_ple::{PleRowType, PleSegment, PleShards, PleStream};

use crate::qwen4exp::config::{
    self as cfg, EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_USED, GDN_HEAD_DIM, GDN_QK_HEADS,
    GDN_V_HEADS, HEAD_DIM, HIDDEN, INDEXER_HEAD_DIM, INDEXER_HEADS, KV_HEADS, N_LAYERS, NgramHash,
    PLE_EOS_TOKEN, PLE_HEAD_DIM, PLE_HEADS_PER_NGRAM, PLE_LAYER, PLE_NGRAM, Q_HEADS,
    Qwen4ExpConfig, STREAM_RANK, STREAMS, VOCAB, is_attention,
};

/// Producer tag every shard of this container carries.
const FORMAT: &str = "mlx";
/// The shard index.
const INDEX: &str = "model.safetensors.index.json";
/// Prefix of every language-model tensor.
const LM: &str = "language_model.model";
/// Codes per affine group of every projection.
const GROUP: u32 = 64;
/// Shards the n-gram table is split into.
const NGRAM_SHARDS: usize = 128;

/// Where the routed banks live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertResidency {
    /// Imported in place, like every other weight.
    Resident,
    /// Read from disk into a pool of this many expert slots as routers pick
    /// them.
    Streamed {
        /// Experts the pool holds.
        slots: u32,
    },
}

/// The shard index document.
#[derive(Debug, Deserialize)]
struct IndexDoc {
    /// Tensor name to shard file.
    weight_map: HashMap<String, String>,
}

/// An open checkpoint directory: its language-model shards and their tensor
/// tables.
#[derive(Debug)]
pub struct MlxCheckpoint {
    dir: PathBuf,
    shards: Vec<Arc<Shard>>,
    files: Vec<Arc<File>>,
    tensors: HashMap<String, (usize, RawTensor)>,
}

impl MlxCheckpoint {
    /// Maps every shard a language-model tensor lives in.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] for a missing index or shard, and
    /// [`CoreError::Backend`] for a shard whose header does not parse or was
    /// not written by this container's producer.
    pub fn open(dir: &Path) -> CoreResult<Self> {
        Self::open_as(dir, Some(FORMAT), |name| {
            name.starts_with("language_model.")
        })
    }

    /// Maps every shard a language-model tensor of the reference export lives
    /// in: `model.language_model.*` and `lm_head.*`, no producer tag, the
    /// vision tower's and the draft head's shards left unmapped.
    ///
    /// # Errors
    ///
    /// As [`MlxCheckpoint::open`].
    pub fn open_reference(dir: &Path) -> CoreResult<Self> {
        Self::open_as(dir, None, |name| {
            name.starts_with("model.language_model.") || name.starts_with("lm_head.")
        })
    }

    /// The body of both openers: the producer tag the shards must carry, if
    /// any, and which tensor names are the language model's.
    pub(crate) fn open_as(
        dir: &Path,
        tag: Option<&'static str>,
        keep: fn(&str) -> bool,
    ) -> CoreResult<Self> {
        let text = std::fs::read_to_string(dir.join(INDEX)).map_err(CoreError::Io)?;
        let index: IndexDoc = serde_json::from_str(&text).map_err(|err| CoreError::Backend {
            backend: "qwen4exp",
            op: "read the shard index",
            source: Box::new(err),
        })?;
        let mut names: Vec<&String> = index
            .weight_map
            .iter()
            .filter(|(name, _)| keep(name))
            .map(|(_, file)| file)
            .collect();
        names.sort();
        names.dedup();
        let mut shards = Vec::new();
        let mut files = Vec::new();
        let mut tensors = HashMap::new();
        for name in names {
            let path = dir.join(name);
            let opened = match tag {
                Some(tag) => Shard::open(&path, tag),
                None => Shard::open_untagged(&path),
            };
            let (shard, table) = opened.map_err(|err| CoreError::Backend {
                backend: "qwen4exp",
                op: "open a checkpoint shard",
                source: Box::new(err),
            })?;
            let slot = shards.len();
            shards.push(Arc::new(shard));
            files.push(Arc::new(File::open(&path).map_err(CoreError::Io)?));
            for (tensor, raw) in table {
                if keep(&tensor) {
                    tensors.insert(tensor, (slot, raw));
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            shards,
            files,
            tensors,
        })
    }

    /// The directory the checkpoint was opened from.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Bytes of every language-model tensor but the n-gram table's shards,
    /// which is what making the checkpoint resident costs.
    #[must_use]
    pub fn resident_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|(name, _)| !name.contains("ngram_embedding.shard_"))
            .map(|(_, (_, raw))| raw.data_offsets[1].saturating_sub(raw.data_offsets[0]))
            .sum()
    }

    /// One tensor's table entry, or a failure naming it.
    pub(crate) fn raw(&self, name: &str) -> CoreResult<(usize, &RawTensor)> {
        self.tensors
            .get(name)
            .map(|(slot, raw)| (*slot, raw))
            .ok_or_else(|| CoreError::ConfigMissing {
                key: name.to_owned(),
            })
    }

    /// One tensor's shard, its span inside the shard's mapping, its dtype and
    /// its shape - checked against what the caller expects.
    pub(crate) fn span(
        &self,
        name: &str,
        dtype: &str,
        shape: &[u64],
    ) -> CoreResult<(usize, std::ops::Range<usize>)> {
        let (slot, raw) = self.raw(name)?;
        if raw.dtype != dtype || raw.shape != shape {
            return Err(CoreError::InvalidValue {
                key: "tensor layout",
                value: format!(
                    "{name} is {} {:?}, the loader expects {dtype} {shape:?}",
                    raw.dtype, raw.shape
                ),
            });
        }
        let range = self.shards[slot]
            .payload_range(raw.data_offsets[0], raw.data_offsets[1])
            .map_err(|err| CoreError::Backend {
                backend: "qwen4exp",
                op: "locate a tensor payload",
                source: Box::new(err),
            })?;
        Ok((slot, range))
    }

    /// A tensor's stored bytes, checked against the dtype and shape the
    /// caller expects.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] for an absent tensor and
    /// [`CoreError::InvalidValue`] for one of another layout.
    pub fn tensor_bytes(&self, name: &str, dtype: &str, shape: &[u64]) -> CoreResult<&[u8]> {
        self.bytes(name, dtype, shape)
    }

    /// A tensor's bytes wrapped on the device in place, as the loader imports
    /// every weight it does not move.
    ///
    /// # Errors
    ///
    /// As [`MlxCheckpoint::tensor_bytes`], plus whatever the device reports.
    pub fn import_tensor(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        dtype: &str,
        shape: &[u64],
    ) -> CoreResult<DeviceBuffer> {
        self.import(backend, name, dtype, shape)
    }

    /// A tensor's stored bytes.
    pub(crate) fn bytes(&self, name: &str, dtype: &str, shape: &[u64]) -> CoreResult<&[u8]> {
        let (slot, range) = self.span(name, dtype, shape)?;
        Ok(&AsRef::<[u8]>::as_ref(&*self.shards[slot])[range])
    }

    /// A tensor's bytes on the device, wrapped in place.
    pub(crate) fn import(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        dtype: &str,
        shape: &[u64],
    ) -> CoreResult<DeviceBuffer> {
        let (slot, range) = self.span(name, dtype, shape)?;
        let host: Arc<Shard> = Arc::clone(&self.shards[slot]);
        import_host(backend, host, range)
    }

    /// One mapped shard, by slot.
    pub(crate) fn shard(&self, slot: usize) -> Arc<Shard> {
        Arc::clone(&self.shards[slot])
    }

    /// Whether the directory at `dir` is the reference export rather than the
    /// MLX conversion: its shard index names tensors the reference way.
    #[must_use]
    pub fn is_reference_export(dir: &Path) -> bool {
        std::fs::read_to_string(dir.join(INDEX))
            .ok()
            .and_then(|text| serde_json::from_str::<IndexDoc>(&text).ok())
            .is_some_and(|index| {
                index
                    .weight_map
                    .keys()
                    .any(|name| name.starts_with("model.language_model."))
            })
    }

    /// The shard files, in the order their slots number them.
    pub(crate) fn shard_paths(&self) -> Vec<PathBuf> {
        self.shards
            .iter()
            .map(|shard| shard.path().to_path_buf())
            .collect()
    }

    /// Which shard a tensor lives in and where its payload starts in that
    /// shard's file.
    pub(crate) fn shard_offset(&self, name: &str) -> CoreResult<(usize, u64)> {
        let (slot, raw) = self.raw(name)?;
        Ok((slot, self.shards[slot].payload_base() + raw.data_offsets[0]))
    }

    /// Where a tensor's payload starts in its file, for a positional reader,
    /// with the file.
    pub(crate) fn file_offset(&self, name: &str) -> CoreResult<(Arc<File>, u64)> {
        let (slot, raw) = self.raw(name)?;
        Ok((
            Arc::clone(&self.files[slot]),
            self.shards[slot].payload_base() + raw.data_offsets[0],
        ))
    }

    /// A small integer vector.
    pub(crate) fn i64s(&self, name: &str, len: u64) -> CoreResult<Vec<u64>> {
        let bytes = self.bytes(name, "I64", &[len])?;
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| {
                u64::try_from(i64::from_le_bytes(*word)).map_err(|_| CoreError::InvalidValue {
                    key: "n-gram table metadata",
                    value: format!("{name} holds a negative value"),
                })
            })
            .collect()
    }

    /// A brain-float tensor widened to single precision.
    pub(crate) fn floats(&self, name: &str, shape: &[u64]) -> CoreResult<Vec<f32>> {
        Ok(self
            .bytes(name, "BF16", shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect())
    }
}

/// Uploads owned bytes as a device buffer: wrapped where the device reads
/// host memory in place, copied and dropped where it copies - a wrapped copy
/// would keep the bytes alive beside the device's own.
pub(crate) fn upload(backend: &Arc<dyn Backend>, bytes: Vec<u8>) -> CoreResult<DeviceBuffer> {
    if backend.imports_copy() {
        let buffer = shiso_core::buffer::alloc(backend, bytes.len() as u64)?;
        backend.copy_h2d(buffer.view(), &bytes)?;
        return Ok(buffer);
    }
    let len = bytes.len();
    import_host(backend, Arc::new(bytes), 0..len)
}

/// Single-precision values as bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The grouped-to-tiled order of the recurrent value heads: new position
/// `r * qk + kh` takes old head `kh * per + r`.
pub(crate) fn value_heads() -> Vec<usize> {
    let (qk, v) = (GDN_QK_HEADS as usize, GDN_V_HEADS as usize);
    let per = v / qk;
    let mut perm = vec![0usize; v];
    for kh in 0..qk {
        for r in 0..per {
            perm[r * qk + kh] = kh * per + r;
        }
    }
    perm
}

/// `src` as blocks of `block` bytes, reordered so block `i` is old block
/// `perm[i]`; bytes past the permuted blocks are kept where they were
/// (`lead` blocks in front stay in place as well).
pub(crate) fn permute_blocks(src: &[u8], block: usize, lead: usize, perm: &[usize]) -> Vec<u8> {
    let mut out = src.to_vec();
    for (i, &old) in perm.iter().enumerate() {
        let to = (lead + i) * block;
        let from = (lead + old) * block;
        out[to..to + block].copy_from_slice(&src[from..from + block]);
    }
    out
}

/// Per-row column-block reordering of a row-major array of `rows` rows of
/// `row_bytes` each: inside every row, block `i` of `block` bytes becomes old
/// block `perm[i]`.
pub(crate) fn permute_columns(
    src: &[u8],
    row_bytes: usize,
    block: usize,
    perm: &[usize],
) -> Vec<u8> {
    let mut out = src.to_vec();
    for (dst, row) in out.chunks_mut(row_bytes).zip(src.chunks(row_bytes)) {
        for (i, &old) in perm.iter().enumerate() {
            dst[i * block..(i + 1) * block].copy_from_slice(&row[old * block..(old + 1) * block]);
        }
    }
    out
}

/// How one affine matrix is moved on its way to the device.
#[derive(Debug, Clone, Copy)]
enum Move<'p> {
    /// As stored, wrapped in place.
    InPlace,
    /// Output rows in blocks of `rows`, the first `lead` blocks kept, the rest
    /// reordered by `perm`.
    Rows {
        rows: usize,
        lead: usize,
        perm: &'p [usize],
    },
    /// Contraction columns in blocks of `cols`, reordered by `perm`.
    Cols { cols: usize, perm: &'p [usize] },
}

/// The loader's state over one open checkpoint.
struct Loader<'c> {
    ck: &'c MlxCheckpoint,
    backend: &'c Arc<dyn Backend>,
}

impl Loader<'_> {
    /// One affine matrix bank `[slots,] out, in`.
    fn affine(
        &self,
        base: &str,
        in_dim: u32,
        out_dim: u32,
        slots: Option<u32>,
        how: Move<'_>,
    ) -> CoreResult<Weight> {
        let lead: Vec<u64> = slots.map(u64::from).into_iter().collect();
        let dims = |last: u32| -> Vec<u64> {
            lead.iter()
                .copied()
                .chain([u64::from(out_dim), u64::from(last)])
                .collect()
        };
        let words = dims(in_dim / 8);
        let params = dims(in_dim / GROUP);
        let names = [
            (format!("{base}.weight"), "U32", words),
            (format!("{base}.scales"), "BF16", params.clone()),
            (format!("{base}.biases"), "BF16", params),
        ];
        let row_bytes = [in_dim as usize / 2, in_dim as usize / GROUP as usize * 2];
        let arrays = names
            .iter()
            .enumerate()
            .map(|(i, (name, dtype, shape))| {
                let per_row = row_bytes[i.min(1)];
                match how {
                    Move::InPlace => self.ck.import(self.backend, name, dtype, shape),
                    Move::Rows { rows, lead, perm } => {
                        let src = self.ck.bytes(name, dtype, shape)?;
                        upload(
                            self.backend,
                            permute_blocks(src, rows * per_row, lead, perm),
                        )
                    }
                    Move::Cols { cols, perm } => {
                        let src = self.ck.bytes(name, dtype, shape)?;
                        let block = if i == 0 {
                            cols / 2
                        } else {
                            cols / GROUP as usize * 2
                        };
                        upload(self.backend, permute_columns(src, per_row, block, perm))
                    }
                }
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let [codes, scales, biases]: [DeviceBuffer; 3] =
            arrays.try_into().map_err(|_| CoreError::ShapeMismatch {
                field: "affine arrays",
                expected: 3,
                got: 0,
            })?;
        Weight::affine(
            DType::MlxAffine4Gs64,
            in_dim,
            out_dim,
            slots.unwrap_or(1),
            [codes, scales, biases],
        )
    }

    /// A brain-float matrix `out, in`, optionally a row range of a larger
    /// tensor and optionally with its rows reordered.
    fn brain(
        &self,
        name: &str,
        stored_rows: u32,
        in_dim: u32,
        rows: std::ops::Range<u32>,
        perm: Option<&[usize]>,
    ) -> CoreResult<Weight> {
        let shape = [u64::from(stored_rows), u64::from(in_dim)];
        let row_bytes = in_dim as usize * 2;
        let out_dim = rows.end - rows.start;
        let (slot, span) = self.ck.span(name, "BF16", &shape)?;
        let start = span.start + rows.start as usize * row_bytes;
        let end = span.start + rows.end as usize * row_bytes;
        let buffer = match perm {
            None => {
                let host: Arc<Shard> = Arc::clone(&self.ck.shards[slot]);
                import_host(self.backend, host, start..end)?
            }
            Some(perm) => {
                let src = &AsRef::<[u8]>::as_ref(&*self.ck.shards[slot])[start..end];
                upload(self.backend, permute_blocks(src, row_bytes, 0, perm))?
            }
        };
        Ok(Weight::over(DType::BF16, in_dim, out_dim, buffer))
    }

    /// A single-precision vector from a brain-float tensor, through `op`.
    fn vector(
        &self,
        name: &str,
        shape: &[u64],
        op: impl Fn(f32) -> f32,
    ) -> CoreResult<DeviceBuffer> {
        let values: Vec<f32> = self.ck.floats(name, shape)?.into_iter().map(op).collect();
        upload(self.backend, f32_bytes(&values))
    }

    /// A single-precision vector whose blocks of `block` values are
    /// reordered, the first `lead` blocks kept.
    fn vector_perm(
        &self,
        name: &str,
        shape: &[u64],
        block: usize,
        lead: usize,
        perm: &[usize],
        op: impl Fn(f32) -> f32,
    ) -> CoreResult<DeviceBuffer> {
        let values: Vec<f32> = self.ck.floats(name, shape)?.into_iter().map(op).collect();
        upload(
            self.backend,
            permute_blocks(&f32_bytes(&values), block * 4, lead, perm),
        )
    }

    /// One stream mix. `inject` is absent on the collapsing mix.
    fn mix(&self, base: &str, inject: bool) -> CoreResult<StreamMix> {
        let wide = HIDDEN * STREAMS;
        Ok(StreamMix {
            norm: self.vector(
                &format!("{base}.hc_norm.weight"),
                &[u64::from(wide)],
                std::convert::identity,
            )?,
            down: self.affine(
                &format!("{base}.input_mix_weight_down"),
                wide,
                STREAM_RANK,
                None,
                Move::InPlace,
            )?,
            up: self.affine(
                &format!("{base}.input_mix_weight_up"),
                STREAM_RANK,
                wide,
                None,
                Move::InPlace,
            )?,
            inject: inject
                .then(|| {
                    self.brain(
                        &format!("{base}.block_inject_weight.weight"),
                        STREAMS,
                        wide,
                        0..STREAMS,
                        None,
                    )
                })
                .transpose()?,
        })
    }

    /// One attention slot.
    fn attn(&self, base: &str) -> CoreResult<SparseAttn> {
        let a = format!("{base}.self_attn");
        let q_rows = Q_HEADS * HEAD_DIM * 2;
        let kv_rows = KV_HEADS * HEAD_DIM;
        let idx_q = INDEXER_HEADS * INDEXER_HEAD_DIM;
        let idx_rows = idx_q + INDEXER_HEAD_DIM;
        Ok(SparseAttn {
            q: self.affine(&format!("{a}.q_proj"), HIDDEN, q_rows, None, Move::InPlace)?,
            k: self.affine(&format!("{a}.k_proj"), HIDDEN, kv_rows, None, Move::InPlace)?,
            v: self.affine(&format!("{a}.v_proj"), HIDDEN, kv_rows, None, Move::InPlace)?,
            output: self.affine(
                &format!("{a}.o_proj"),
                Q_HEADS * HEAD_DIM,
                HIDDEN,
                None,
                Move::InPlace,
            )?,
            q_norm: self.vector(
                &format!("{a}.q_norm.weight"),
                &[u64::from(HEAD_DIM)],
                std::convert::identity,
            )?,
            k_norm: self.vector(
                &format!("{a}.k_norm.weight"),
                &[u64::from(HEAD_DIM)],
                std::convert::identity,
            )?,
            indexer: Indexer {
                q: self.brain(
                    &format!("{a}.indexer.index_qk_proj.weight"),
                    idx_rows,
                    HIDDEN,
                    0..idx_q,
                    None,
                )?,
                k: self.brain(
                    &format!("{a}.indexer.index_qk_proj.weight"),
                    idx_rows,
                    HIDDEN,
                    idx_q..idx_rows,
                    None,
                )?,
                q_norm: self.vector(
                    &format!("{a}.indexer.q_layernorm.weight"),
                    &[u64::from(INDEXER_HEAD_DIM)],
                    std::convert::identity,
                )?,
                k_norm: self.vector(
                    &format!("{a}.indexer.k_layernorm.weight"),
                    &[u64::from(INDEXER_HEAD_DIM)],
                    std::convert::identity,
                )?,
            },
        })
    }

    /// One recurrent slot, its value heads moved to the tiled order.
    fn recurrent(&self, base: &str) -> CoreResult<Recurrent> {
        let la = format!("{base}.linear_attn");
        let heads = value_heads();
        let hd = GDN_HEAD_DIM as usize;
        let qk_blocks = 2 * GDN_QK_HEADS as usize;
        let v_rows = GDN_V_HEADS * GDN_HEAD_DIM;
        let qkv_rows = 2 * GDN_QK_HEADS * GDN_HEAD_DIM + v_rows;
        let conv_channels = u64::from(qkv_rows);
        let n_v = u64::from(GDN_V_HEADS);
        Ok(Recurrent {
            qkv: self.affine(
                &format!("{la}.in_proj_qkv"),
                HIDDEN,
                qkv_rows,
                None,
                Move::Rows {
                    rows: hd,
                    lead: qk_blocks,
                    perm: &heads,
                },
            )?,
            gate: self.affine(
                &format!("{la}.in_proj_z"),
                HIDDEN,
                v_rows,
                None,
                Move::Rows {
                    rows: hd,
                    lead: 0,
                    perm: &heads,
                },
            )?,
            alpha: self.brain(
                &format!("{la}.in_proj_a.weight"),
                GDN_V_HEADS,
                HIDDEN,
                0..GDN_V_HEADS,
                Some(&heads),
            )?,
            beta: self.brain(
                &format!("{la}.in_proj_b.weight"),
                GDN_V_HEADS,
                HIDDEN,
                0..GDN_V_HEADS,
                Some(&heads),
            )?,
            out: self.affine(
                &format!("{la}.out_proj"),
                v_rows,
                HIDDEN,
                None,
                Move::Cols {
                    cols: hd,
                    perm: &heads,
                },
            )?,
            // `[channels, kernel, 1]`, channel-major: a channel's taps are
            // contiguous, and the value channels move a head (128 channels,
            // 4 taps each) at a time.
            conv1d: self.vector_perm(
                &format!("{la}.conv1d.weight"),
                &[conv_channels, u64::from(cfg::GDN_CONV), 1],
                hd * cfg::GDN_CONV as usize,
                qk_blocks,
                &heads,
                std::convert::identity,
            )?,
            a: self.vector_perm(&format!("{la}.A_log"), &[n_v], 1, 0, &heads, |a| -a.exp())?,
            dt_bias: self.vector_perm(
                &format!("{la}.dt_bias"),
                &[n_v],
                1,
                0,
                &heads,
                std::convert::identity,
            )?,
            norm: self.vector(
                &format!("{la}.norm.weight"),
                &[u64::from(GDN_HEAD_DIM)],
                std::convert::identity,
            )?,
        })
    }

    /// One routed feed-forward. `streamed` banks are described, not imported.
    fn experts(&self, base: &str, streamed: bool) -> CoreResult<ExpertBank> {
        let m = format!("{base}.mlp");
        let bank = |name: &str, in_dim: u32, out_dim: u32| -> CoreResult<Weight> {
            if streamed {
                // Checked all the same: a streamed bank is read out of these
                // arrays later, and a layout the source does not expect is
                // better refused now than read into the wrong slots.
                let lead = [u64::from(EXPERTS), u64::from(out_dim)];
                self.ck.span(
                    &format!("{m}.switch_mlp.{name}.weight"),
                    "U32",
                    &[lead[0], lead[1], u64::from(in_dim / 8)],
                )?;
                let placeholder = shiso_core::buffer::alloc(self.backend, 16)?;
                Ok(Weight::streamed(
                    DType::MlxAffine4Gs64,
                    in_dim,
                    out_dim,
                    EXPERTS,
                    placeholder,
                ))
            } else {
                self.affine(
                    &format!("{m}.switch_mlp.{name}"),
                    in_dim,
                    out_dim,
                    Some(EXPERTS),
                    Move::InPlace,
                )
            }
        };
        Ok(ExpertBank {
            router: self.brain(
                &format!("{m}.gate.weight"),
                EXPERTS,
                HIDDEN,
                0..EXPERTS,
                None,
            )?,
            gate: bank("gate_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            up: bank("up_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            down: bank("down_proj", EXPERT_INTERMEDIATE, HIDDEN)?,
            shared_router: self.brain(
                &format!("{m}.shared_expert_gate.weight"),
                1,
                HIDDEN,
                0..1,
                None,
            )?,
            shared_gate: self.affine(
                &format!("{m}.shared_expert.gate_proj"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
                None,
                Move::InPlace,
            )?,
            shared_up: self.affine(
                &format!("{m}.shared_expert.up_proj"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
                None,
                Move::InPlace,
            )?,
            shared_down: self.affine(
                &format!("{m}.shared_expert.down_proj"),
                EXPERT_INTERMEDIATE,
                HIDDEN,
                None,
                Move::InPlace,
            )?,
        })
    }

    /// The n-gram block.
    fn ngram(&self, base: &str) -> CoreResult<NgramBlock> {
        let p = format!("{base}.ple");
        let wide = u64::from(HIDDEN * STREAMS);
        Ok(NgramBlock {
            key: self.affine(
                &format!("{p}.key_proj"),
                HIDDEN,
                HIDDEN * STREAMS,
                None,
                Move::InPlace,
            )?,
            value: self.affine(
                &format!("{p}.value_proj"),
                HIDDEN,
                HIDDEN,
                None,
                Move::InPlace,
            )?,
            norm_key: self.vector(
                &format!("{p}.norm_key.weight"),
                &[wide],
                std::convert::identity,
            )?,
            norm_query: self.vector(
                &format!("{p}.norm_query.weight"),
                &[wide],
                std::convert::identity,
            )?,
            norm_conv: self.vector(
                &format!("{p}.norm_conv.weight"),
                &[wide],
                std::convert::identity,
            )?,
            conv1d: self.vector(
                &format!("{p}.conv1d.weight"),
                &[wide, u64::from(cfg::PLE_CONV), 1],
                std::convert::identity,
            )?,
        })
    }
}

/// Imports every weight of the checkpoint.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first tensor the directory does not
/// carry, [`CoreError::InvalidValue`] for one whose layout is not the one this
/// family stores, and whatever the device reports.
pub fn import(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    config: &Qwen4ExpConfig,
    residency: ExpertResidency,
) -> CoreResult<SparseWeights> {
    let loader = Loader { ck, backend };
    let streamed = matches!(residency, ExpertResidency::Streamed { .. });
    let interval = config.full_attention_interval();
    let mut layers = Vec::with_capacity(N_LAYERS as usize);
    let (mut gdn_rank, mut attn_rank) = (0u32, 0u32);
    for il in 0..config.shape().n_layers() {
        let base = format!("{LM}.layers.{il}");
        let (block, ordinal) = if is_attention(il, interval) {
            let _tag = ledger::scope(MemTag::WeightsAttention);
            attn_rank += 1;
            (SparseBlock::Attn(loader.attn(&base)?), attn_rank - 1)
        } else {
            let _tag = ledger::scope(MemTag::WeightsRecurrent);
            gdn_rank += 1;
            (
                SparseBlock::Recurrent(loader.recurrent(&base)?),
                gdn_rank - 1,
            )
        };
        let mix = ledger::scope(MemTag::WeightsStreamMix);
        let attn_mix = loader.mix(&format!("{base}.attn_hyper_connection"), true)?;
        let ffn_mix = loader.mix(&format!("{base}.mlp_hyper_connection"), true)?;
        drop(mix);
        let experts = {
            let _tag = ledger::scope(MemTag::WeightsExperts);
            loader.experts(&base, streamed)?
        };
        let ngram = (il == PLE_LAYER)
            .then(|| {
                let _tag = ledger::scope(MemTag::WeightsEmbedding);
                loader.ngram(&base)
            })
            .transpose()?;
        layers.push(SparseLayer {
            attn_mix,
            ffn_mix,
            block,
            experts,
            ngram,
            ordinal,
        });
    }
    let token_embd = {
        let _tag = ledger::scope(MemTag::WeightsEmbedding);
        loader.affine(
            &format!("{LM}.embed_tokens"),
            HIDDEN,
            VOCAB,
            None,
            Move::InPlace,
        )?
    };
    let _tag = ledger::scope(MemTag::WeightsHead);
    let lm_head = loader.affine("language_model.lm_head", HIDDEN, VOCAB, None, Move::InPlace)?;
    let output_mix = loader.mix(&format!("{LM}.hyper_connection_mixer"), false)?;
    let identity: Vec<u8> = (0..EXPERTS).flat_map(u32::to_le_bytes).collect();
    Ok(SparseWeights {
        token_embd,
        lm_head: Some(lm_head),
        output_mix,
        layers,
        draft: None,
        expert_map: Some(upload(backend, identity)?),
    })
}

/// Bytes of the routed banks, every layer, as stored.
#[must_use]
pub fn expert_bytes() -> u64 {
    let one = |in_dim: u32, out_dim: u32| -> u64 {
        u64::from(out_dim) * (u64::from(in_dim) / 2 + u64::from(in_dim / GROUP) * 4)
    };
    let per_expert = 2 * one(HIDDEN, EXPERT_INTERMEDIATE) + one(EXPERT_INTERMEDIATE, HIDDEN);
    per_expert * u64::from(EXPERTS) * u64::from(N_LAYERS)
}

/// Bytes one streamed expert occupies across its nine arrays.
#[must_use]
pub fn slot_bytes() -> u64 {
    expert_bytes() / (u64::from(EXPERTS) * u64::from(N_LAYERS))
}

/// The routed banks of this checkpoint, read expert by expert.
#[derive(Debug)]
pub struct MlxExperts {
    /// Per layer, per array: the file, the array's first byte, and one
    /// expert's bytes in it.
    arrays: Vec<Vec<(Arc<File>, u64, u64)>>,
}

impl MlxExperts {
    /// Locates every layer's nine arrays.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming a missing array.
    pub fn new(ck: &MlxCheckpoint, layers: u32) -> CoreResult<Self> {
        // Handles of their own, opened past the page cache: the pool is the
        // copy these reads are for, and a cached read would be a second one.
        let uncached = ck
            .shard_paths()
            .iter()
            .map(|path| {
                shiso_modelfile::st_header::open_uncached(path)
                    .map(Arc::new)
                    .map_err(CoreError::Io)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let mut arrays = Vec::with_capacity(layers as usize);
        for il in 0..layers {
            let m = format!("{LM}.layers.{il}.mlp.switch_mlp");
            let mut row = Vec::with_capacity(9);
            for (proj, in_dim, out_dim) in [
                ("gate_proj", HIDDEN, EXPERT_INTERMEDIATE),
                ("up_proj", HIDDEN, EXPERT_INTERMEDIATE),
                ("down_proj", EXPERT_INTERMEDIATE, HIDDEN),
            ] {
                let codes = u64::from(out_dim) * u64::from(in_dim) / 2;
                let params = u64::from(out_dim) * u64::from(in_dim / GROUP) * 2;
                for (part, bytes) in [("weight", codes), ("scales", params), ("biases", params)] {
                    let (slot, offset) = ck.shard_offset(&format!("{m}.{proj}.{part}"))?;
                    row.push((Arc::clone(&uncached[slot]), offset, bytes));
                }
            }
            arrays.push(row);
        }
        Ok(Self { arrays })
    }
}

impl ExpertSource for MlxExperts {
    fn parts(&self) -> &'static [WeightPart] {
        &[
            WeightPart::Whole,
            WeightPart::AffineScales,
            WeightPart::AffineBiases,
        ]
    }

    fn layout(&self) -> Vec<u64> {
        self.arrays
            .first()
            .map(|row| row.iter().map(|(_, _, bytes)| *bytes).collect())
            .unwrap_or_default()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        use std::os::unix::fs::FileExt;
        let (file, base, bytes) = self
            .arrays
            .get(layer as usize)
            .and_then(|row| row.get(array))
            .ok_or(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: self.arrays.len() as u64,
            })?;
        file.read_exact_at(dst, base + u64::from(expert) * bytes)
            .map_err(CoreError::Io)
    }
}

/// Opens the stream over this checkpoint's routed banks.
///
/// # Errors
///
/// Whatever [`MlxExperts::new`] and [`ExpertStream::open`] report.
pub fn open_stream(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let source = Arc::new(MlxExperts::new(ck, N_LAYERS)?);
    let min_slots = widest_pass.saturating_mul(EXPERTS_USED).min(EXPERTS);
    ExpertStream::open(backend, source, N_LAYERS, EXPERTS, slots, min_slots)
}

/// Opens the n-gram table over its 128 shards.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for a missing shard array and whatever the
/// reader refuses.
pub fn open_ngram(
    ck: &MlxCheckpoint,
    config: &Qwen4ExpConfig,
    opts: &SparseOptions,
) -> CoreResult<NgramTable> {
    let ple = config.shape().ple.ok_or(CoreError::ShapeMismatch {
        field: "n-gram slot",
        expected: 1,
        got: 0,
    })?;
    let params = config.ngram().params(ple, PleRowType::MlxAffine4, None)?;
    let base = format!("{LM}.layers.{PLE_LAYER}.ple.ple_embedding.ngram_embedding");
    let groups = u64::from(PLE_HEAD_DIM) / 32;
    let mut shards = Vec::with_capacity(NGRAM_SHARDS);
    let mut rows_per_shard = 0u64;
    for s in 0..NGRAM_SHARDS {
        let mut segments = Vec::with_capacity(3);
        for (part, dtype, width, row_bytes) in [
            (
                "weight",
                "U32",
                u64::from(PLE_HEAD_DIM) / 8,
                u64::from(PLE_HEAD_DIM) / 2,
            ),
            ("scales", "BF16", groups, groups * 2),
            ("biases", "BF16", groups, groups * 2),
        ] {
            let name = format!("{base}.shard_{s}.{part}");
            let (_, raw) = ck.raw(&name)?;
            let rows = raw.shape.first().copied().unwrap_or(0);
            if raw.dtype != dtype || raw.shape.get(1).copied() != Some(width) {
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
                row_bytes: u32::try_from(row_bytes).unwrap_or(u32::MAX),
            });
        }
        shards.push(segments);
    }
    let n_rows = rows_per_shard * NGRAM_SHARDS as u64;
    let stream = PleStream::open_sharded(
        PleShards {
            rows_per_shard,
            shards,
        },
        n_rows,
        params,
        opts.ngram_cache_bytes,
        &opts.ngram_reader,
    )?;
    NgramTable::new(DType::MlxAffine4Gs32, Arc::new(stream), None)
}

/// The text half of the checkpoint's `config.json`, the keys this loader
/// checks.
#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    linear_num_value_heads: u32,
    linear_num_key_heads: u32,
    linear_key_head_dim: u32,
    vocab_size: u32,
    hc_count: u32,
    hc_lowrank: u32,
    full_attention_interval: u32,
    ngram_size: u32,
    heads_per_ngram: u32,
    indexer_n_heads: u32,
    indexer_head_dim: u32,
}

/// The generation config's stop list.
#[derive(Debug, Deserialize)]
struct GenerationConfig {
    #[serde(default)]
    eos_token_id: serde_json::Value,
}

impl Qwen4ExpConfig {
    /// Reads the geometry of an MLX checkpoint directory: the dimensions out
    /// of its `config.json`, each checked against the constant the kernels
    /// were written for; the turn marker out of its generation config; and the
    /// n-gram hash out of the three small tensors the table carries.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownArch`] for another family, [`CoreError::ShapeMismatch`]
    /// naming the first dimension that disagrees, and whatever the checkpoint
    /// reports for the n-gram tensors.
    pub fn from_mlx(config: &HfConfig, ck: &MlxCheckpoint) -> CoreResult<Self> {
        config_from(
            config,
            ck,
            &format!("{LM}.layers.{PLE_LAYER}.ple.ple_embedding"),
        )
    }
}

/// The geometry of a checkpoint directory of this family: the dimensions out
/// of its `config.json`, each checked against the constant the kernels were
/// written for; the turn marker out of its generation config; and the n-gram
/// hash out of the three small tensors under `ngram_base`.
///
/// # Errors
///
/// As [`Qwen4ExpConfig::from_mlx`].
pub(crate) fn config_from(
    config: &HfConfig,
    ck: &MlxCheckpoint,
    ngram_base: &str,
) -> CoreResult<Qwen4ExpConfig> {
    let heads = u64::from((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
    let base = ngram_base;
    // Read lazily: a document of another family is refused by name first,
    // before any n-gram tensor is looked for.
    config_checked(config, ck.dir(), || {
        Ok(NgramHash {
            multipliers: ck.i64s(&format!("{base}.layer_multipliers"), u64::from(PLE_NGRAM))?,
            head_offsets: ck.i64s(&format!("{base}.ngram_heads_offsets"), heads)?,
            head_vocab: ck.i64s(&format!("{base}.ngram_heads_vocab_sizes"), heads)?,
            eos_token: TokenId::new(PLE_EOS_TOKEN),
        })
    })
}

/// The geometry of a checkpoint directory at `dir` whose n-gram hash `ngram`
/// reads: the checks [`config_from`] makes, for a container that keeps the
/// hash's tensors somewhere else.
///
/// # Errors
///
/// As [`Qwen4ExpConfig::from_mlx`], plus whatever `ngram` reports.
pub(crate) fn config_checked(
    config: &HfConfig,
    dir: &Path,
    ngram: impl FnOnce() -> CoreResult<NgramHash>,
) -> CoreResult<Qwen4ExpConfig> {
    {
        if !Qwen4ExpConfig::describes(config) {
            return Err(CoreError::UnknownArch {
                model_type: config.model_type().to_owned(),
            });
        }
        let text: TextConfig = serde_json::from_value(config.section("text_config")?.clone())
            .map_err(|err| CoreError::Backend {
                backend: "qwen4exp",
                op: "read text_config",
                source: Box::new(err),
            })?;
        for (field, got, want) in [
            ("hidden_size", text.hidden_size, HIDDEN),
            ("num_hidden_layers", text.num_hidden_layers, N_LAYERS),
            ("num_attention_heads", text.num_attention_heads, Q_HEADS),
            ("num_key_value_heads", text.num_key_value_heads, KV_HEADS),
            ("head_dim", text.head_dim, HEAD_DIM),
            ("num_experts", text.num_experts, EXPERTS),
            (
                "num_experts_per_tok",
                text.num_experts_per_tok,
                EXPERTS_USED,
            ),
            (
                "moe_intermediate_size",
                text.moe_intermediate_size,
                EXPERT_INTERMEDIATE,
            ),
            (
                "linear_num_value_heads",
                text.linear_num_value_heads,
                GDN_V_HEADS,
            ),
            (
                "linear_num_key_heads",
                text.linear_num_key_heads,
                GDN_QK_HEADS,
            ),
            (
                "linear_key_head_dim",
                text.linear_key_head_dim,
                GDN_HEAD_DIM,
            ),
            ("vocab_size", text.vocab_size, VOCAB),
            ("hc_count", text.hc_count, STREAMS),
            ("hc_lowrank", text.hc_lowrank, STREAM_RANK),
            (
                "full_attention_interval",
                text.full_attention_interval,
                cfg::FULL_ATTENTION_INTERVAL,
            ),
            ("ngram_size", text.ngram_size, PLE_NGRAM),
            ("heads_per_ngram", text.heads_per_ngram, PLE_HEADS_PER_NGRAM),
            ("indexer_n_heads", text.indexer_n_heads, INDEXER_HEADS),
            ("indexer_head_dim", text.indexer_head_dim, INDEXER_HEAD_DIM),
        ] {
            if got != want {
                return Err(CoreError::ShapeMismatch {
                    field,
                    expected: u64::from(want),
                    got: u64::from(got),
                });
            }
        }
        let generation =
            std::fs::read_to_string(dir.join("generation_config.json")).map_err(CoreError::Io)?;
        let generation: GenerationConfig =
            serde_json::from_str(&generation).map_err(|err| CoreError::Backend {
                backend: "qwen4exp",
                op: "read generation_config.json",
                source: Box::new(err),
            })?;
        let eos = match &generation.eos_token_id {
            serde_json::Value::Array(ids) => ids.first().and_then(serde_json::Value::as_u64),
            other => other.as_u64(),
        }
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| CoreError::ConfigMissing {
            key: "generation_config.json eos_token_id".to_owned(),
        })?;
        let ngram = ngram()?;
        Qwen4ExpConfig::new(
            cfg::shape(),
            TokenId::new(eos),
            cfg::FULL_ATTENTION_INTERVAL,
            ngram,
        )
    }
}
