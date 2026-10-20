//! The trellis-quantized export of this family as it is published: a
//! directory of safetensors shards in which every quantized projection is
//! three tensors (`trellis` tiles, `suh` input scales, `svh` output scales),
//! every routed expert is a projection of its own, and the n-gram table is a
//! file of its own.
//!
//! # What the load does to the stored bytes
//!
//! The kernels read the payload the single-file container stores: one buffer
//! per projection holding the tiles, then the input scales, then the output
//! scales, with a bank's experts stacked inside each of the three regions. The
//! shards keep the three apart and the experts apart, so every quantized
//! projection is assembled into that buffer here, once, at load. Beyond the
//! assembly, the same moves every container of this family needs are made
//! (see [`super::mlx`]), and each lands on whole tiles:
//!
//! - the recurrent block's value heads go to the tiled order. A value head is
//!   128 outputs, which is eight tile columns and one 128-wide rotation block,
//!   so moving it moves whole tile columns and the matching output scales, and
//!   the decoded weights are the stored rows in the new order, bit for bit;
//!   the output projection moves eight tile rows and the input scales instead;
//! - the indexer's fused query/key projection is split at a 128-output
//!   boundary into two projections, which is a range of tile columns;
//! - the norms the reference applies as `1 + w` are stored as `w` here and
//!   get the one added; the decay base is stored as `-exp(A_log)`.
//!
//! Nothing is requantized and nothing is decoded: the tiles reach the device
//! as they are stored. The half-precision matrices are imported in place.
//!
//! # The codebook
//!
//! A trellis tile is decoded through a codebook the kernels hard-code, and
//! every projection names the one it was quantized against in a scalar
//! beside it. A projection that names another is refused rather than read
//! through the wrong one, which would decode to finite, wrong weights.

use std::collections::HashMap;
use std::path::Path;
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
use shiso_engine::sparse::weights::{
    DraftHead, ExpertBank, Indexer, NgramBlock, Recurrent, SparseAttn, SparseBlock, SparseLayer,
    SparseWeights, StreamMix, Weight,
};
use shiso_modelfile::st_header::{RawTensor, Shard};
use shiso_ple::{PleRowType, PleStream};

use crate::qwen4exp::config::{
    self as cfg, EXPERT_INTERMEDIATE, EXPERTS, GDN_HEAD_DIM, GDN_QK_HEADS, GDN_V_HEADS, HEAD_DIM,
    HIDDEN, INDEXER_HEAD_DIM, INDEXER_HEADS, KV_HEADS, N_LAYERS, NgramHash, PLE_EOS_TOKEN,
    PLE_HEAD_DIM, PLE_HEADS_PER_NGRAM, PLE_LAYER, PLE_NGRAM, Q_HEADS, Qwen4ExpConfig, STREAM_RANK,
    STREAMS, VOCAB, is_attention,
};
use crate::qwen4exp::mlx::{
    MlxCheckpoint, config_checked, f32_bytes, permute_blocks, upload, value_heads,
};

/// Prefix of every language-model tensor.
const LM: &str = "model.language_model";
/// Prefix of every draft-head tensor.
const MTP: &str = "mtp";
/// The shard index.
const INDEX: &str = "model.safetensors.index.json";
/// The n-gram table's tensors, under the n-gram slot's layer.
const NGRAM_BASE: &str = "ple.ple_embedding.ngram_embedding";
/// The codebook multiplier the trellis kernels decode with
/// (`kernels/cuda/exl3/codebook.cuh`). Stored as a signed word.
pub(crate) const CODEBOOK: u32 = 0x83DC_D12D;
/// Outputs or inputs one rotation block covers, which is the unit every move
/// of a trellis projection is made in.
const BLOCK: usize = 128;
/// Weights along one side of a tile.
const TILE: usize = 16;
/// Words one n-gram row is stored in: one scale word, then 160 six-bit codes.
const NGRAM_ROW_WORDS: u64 = 1 + PLE_HEAD_DIM as u64 * 6 / 16;

/// Whether the directory at `dir` is this export: its shard index names the
/// vocabulary projection's trellis.
#[must_use]
pub fn is_exl3_dir(dir: &Path) -> bool {
    #[derive(Deserialize)]
    struct IndexDoc {
        weight_map: HashMap<String, String>,
    }
    std::fs::read_to_string(dir.join(INDEX))
        .ok()
        .and_then(|text| serde_json::from_str::<IndexDoc>(&text).ok())
        .is_some_and(|index| index.weight_map.contains_key("lm_head.trellis"))
}

/// Maps every shard of the export that holds a language-model or draft-head
/// tensor.
///
/// # Errors
///
/// As [`MlxCheckpoint::open_reference`].
pub fn open(dir: &Path) -> CoreResult<MlxCheckpoint> {
    MlxCheckpoint::open_as(dir, None, |name| {
        name.starts_with("model.language_model.")
            || name.starts_with("lm_head.")
            || name.starts_with("mtp.")
    })
}

/// The n-gram table's file: its rows, its per-head bias, and the three small
/// tensors the hash reads.
#[derive(Debug)]
pub struct NgramFile {
    shard: Arc<Shard>,
    tensors: HashMap<String, RawTensor>,
}

/// The table file's header metadata, the keys the trellis rows depend on.
#[derive(Debug, Deserialize)]
struct NgramMeta {
    format: String,
    #[serde(rename = "K")]
    bits: String,
    codebook: String,
    row_dim: String,
}

impl NgramFile {
    /// Maps the table file at `path` and checks it holds six-bit trellis rows
    /// of the slot's width, decoded through the kernels' codebook.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] for a missing file, [`CoreError::Backend`] for a
    /// header that does not parse, and [`CoreError::InvalidValue`] for rows of
    /// another codec, width or codebook.
    pub fn open(path: &Path) -> CoreResult<Self> {
        let meta = read_meta(path)?;
        let want = PLE_HEAD_DIM.to_string();
        if meta.format != "exl3_ngram_trellis"
            || meta.bits != "6"
            || meta.codebook != "mul1"
            || meta.row_dim != want
        {
            return Err(CoreError::InvalidValue {
                key: "n-gram table codec",
                value: format!(
                    "{} holds {} rows of {} {}-bit codes through codebook {}; this build reads \
                     exl3_ngram_trellis rows of {want} 6-bit codes through mul1",
                    path.display(),
                    meta.format,
                    meta.row_dim,
                    meta.bits,
                    meta.codebook
                ),
            });
        }
        let (shard, table) = Shard::open_untagged(path).map_err(|err| CoreError::Backend {
            backend: "qwen4exp",
            op: "open the n-gram table",
            source: Box::new(err),
        })?;
        Ok(Self {
            shard: Arc::new(shard),
            tensors: table.into_iter().collect(),
        })
    }

    /// One tensor of the table file, by its name under the n-gram slot.
    fn raw(&self, part: &str) -> CoreResult<&RawTensor> {
        let name = format!("{LM}.layers.{PLE_LAYER}.{NGRAM_BASE}.{part}");
        self.tensors
            .get(&name)
            .ok_or(CoreError::ConfigMissing { key: name })
    }

    /// One tensor's bytes, checked against the dtype and shape expected.
    fn bytes(&self, part: &str, dtype: &str, shape: &[u64]) -> CoreResult<&[u8]> {
        let raw = self.raw(part)?;
        if raw.dtype != dtype || raw.shape != shape {
            return Err(CoreError::InvalidValue {
                key: "n-gram tensor layout",
                value: format!(
                    "{part} is {} {:?}, the loader expects {dtype} {shape:?}",
                    raw.dtype, raw.shape
                ),
            });
        }
        self.shard
            .payload(raw.data_offsets[0], raw.data_offsets[1])
            .map_err(|err| CoreError::Backend {
                backend: "qwen4exp",
                op: "locate an n-gram tensor",
                source: Box::new(err),
            })
    }

    /// A small integer vector of the hash.
    fn i64s(&self, part: &str, len: u64) -> CoreResult<Vec<u64>> {
        self.bytes(part, "I64", &[len])?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| {
                u64::try_from(i64::from_le_bytes(*word)).map_err(|_| CoreError::InvalidValue {
                    key: "n-gram table metadata",
                    value: format!("{part} holds a negative value"),
                })
            })
            .collect()
    }

    /// The hash the table is addressed by.
    fn hash(&self) -> CoreResult<NgramHash> {
        let heads = u64::from((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
        Ok(NgramHash {
            multipliers: self.i64s("layer_multipliers", u64::from(PLE_NGRAM))?,
            head_offsets: self.i64s("head_offsets", heads)?,
            head_vocab: self.i64s("head_vocab_sizes", heads)?,
            eos_token: TokenId::new(PLE_EOS_TOKEN),
        })
    }
}

/// Reads a safetensors file's `__metadata__` object.
fn read_meta(path: &Path) -> CoreResult<NgramMeta> {
    use std::io::Read as _;
    #[derive(Deserialize)]
    struct Header {
        #[serde(rename = "__metadata__")]
        meta: NgramMeta,
    }
    let mut file = std::fs::File::open(path).map_err(CoreError::Io)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).map_err(CoreError::Io)?;
    // A header is a few kilobytes; a length past this is a file of another
    // kind, and reading it whole would be reading the table into memory.
    let len = u64::from_le_bytes(len);
    if len > 1 << 24 {
        return Err(CoreError::InvalidValue {
            key: "n-gram table header",
            value: format!("{} declares a {len}-byte header", path.display()),
        });
    }
    let mut text = vec![0u8; usize::try_from(len).unwrap_or(0)];
    file.read_exact(&mut text).map_err(CoreError::Io)?;
    let header: Header = serde_json::from_slice(&text).map_err(|err| CoreError::Backend {
        backend: "qwen4exp",
        op: "read the n-gram table's metadata",
        source: Box::new(err),
    })?;
    Ok(header.meta)
}

/// Where the table's rows lie in the file's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TableRows {
    /// Byte offset of the first row, relative to the payload base.
    start: u64,
    /// Rows stored, every one [`NGRAM_ROW_WORDS`] words.
    rows: u64,
}

/// Finds the table's rows among the file's tensors.
///
/// Exporters write the table either as one `trellis` tensor or as
/// `shard_0.trellis`, `shard_1.trellis`, ... over consecutive row ranges. A
/// row's bytes do not depend on which shard holds it (its scale word travels
/// with it), so shards stored back to back in index order are the single
/// tensor under other names, and the reader is handed their joint range; no
/// row is copied. Shards with a gap between them, out of order, or of another
/// width are refused rather than joined, since the reader addresses a row by
/// its index times the row size from the first one.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] when neither layout is present and
/// [`CoreError::InvalidValue`] for both at once, a missing shard index,
/// shards that are not contiguous, or a tensor of another dtype or width.
fn table_rows(tensors: &HashMap<String, RawTensor>) -> CoreResult<TableRows> {
    let base = format!("{LM}.layers.{PLE_LAYER}.{NGRAM_BASE}");
    let checked = |name: &str, raw: &RawTensor| -> CoreResult<u64> {
        if raw.dtype != "I16" || raw.shape.len() != 2 || raw.shape.get(1) != Some(&NGRAM_ROW_WORDS)
        {
            return Err(CoreError::InvalidValue {
                key: "n-gram table layout",
                value: format!(
                    "{name} is {} {:?}, the loader expects I16 [rows, {NGRAM_ROW_WORDS}]",
                    raw.dtype, raw.shape
                ),
            });
        }
        Ok(raw.shape.first().copied().unwrap_or(0))
    };
    let shard_prefix = format!("{base}.shard_");
    let mut shards: Vec<(u64, &RawTensor)> = Vec::new();
    for (name, raw) in tensors {
        let Some(index) = name
            .strip_prefix(&shard_prefix)
            .and_then(|rest| rest.strip_suffix(".trellis"))
        else {
            continue;
        };
        let index = index.parse::<u64>().map_err(|_| CoreError::InvalidValue {
            key: "n-gram table shard",
            value: format!("{name} has no shard index"),
        })?;
        shards.push((index, raw));
    }
    let single_name = format!("{base}.trellis");
    match (tensors.get(&single_name), shards.is_empty()) {
        (Some(raw), true) => Ok(TableRows {
            start: raw.data_offsets[0],
            rows: checked(&single_name, raw)?,
        }),
        (None, true) => Err(CoreError::ConfigMissing { key: single_name }),
        (Some(_), false) => Err(CoreError::InvalidValue {
            key: "n-gram table layout",
            value: format!("the file holds both {single_name} and {shard_prefix}N.trellis"),
        }),
        (None, false) => {
            shards.sort_unstable_by_key(|&(index, _)| index);
            let mut rows = 0u64;
            let mut end = None;
            for (want, &(index, raw)) in (0u64..).zip(&shards) {
                let name = format!("{shard_prefix}{index}.trellis");
                if index != want {
                    return Err(CoreError::InvalidValue {
                        key: "n-gram table shard",
                        value: format!("shard {want} is missing; the next stored is {name}"),
                    });
                }
                if end.is_some_and(|end| end != raw.data_offsets[0]) {
                    return Err(CoreError::InvalidValue {
                        key: "n-gram table shard",
                        value: format!("{name} does not start where the shard before it ends"),
                    });
                }
                rows += checked(&name, raw)?;
                end = Some(raw.data_offsets[1]);
            }
            let start = shards.first().map_or(0, |(_, raw)| raw.data_offsets[0]);
            Ok(TableRows { start, rows })
        }
    }
}

impl Qwen4ExpConfig {
    /// Reads the geometry of the trellis export's directory; see
    /// [`Qwen4ExpConfig::from_mlx`], whose checks this shares. The n-gram hash
    /// comes from the table's own file.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpConfig::from_mlx`].
    pub fn from_exl3(config: &HfConfig, dir: &Path, ngram: &NgramFile) -> CoreResult<Self> {
        config_checked(config, dir, || ngram.hash())
    }
}

/// Opens the n-gram table as a reader over its file, with the per-head bias
/// the trellis codec finishes every decoded row with resident beside it.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when the hash names more rows than the table
/// holds, [`CoreError::InvalidValue`] for a tensor of another layout, and
/// whatever the reader refuses.
pub fn open_ngram(
    backend: &Arc<dyn Backend>,
    file: &NgramFile,
    config: &Qwen4ExpConfig,
    opts: &SparseOptions,
) -> CoreResult<NgramTable> {
    let ple = config.shape().ple.ok_or(CoreError::ShapeMismatch {
        field: "n-gram slot",
        expected: 1,
        got: 0,
    })?;
    let heads = u64::from((PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM);
    let bias = file.bytes("head_bias", "F16", &[heads, u64::from(PLE_HEAD_DIM)])?;
    let bias_words: Vec<u16> = bias
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect();
    let params = config
        .ngram()
        .params(ple, PleRowType::TrellisK6, Some(Arc::new(bias_words)))?;
    let table = table_rows(&file.tensors)?;
    let rows = table.rows;
    // The stored rows may run past the last addressable one (the published
    // table rounds up); fewer than the hash names is a read past the end.
    if params.total_rows() > rows {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram table rows",
            expected: rows,
            got: params.total_rows(),
        });
    }
    let offset = file.shard.payload_base() + table.start;
    let reader = std::fs::File::open(file.shard.path()).map_err(CoreError::Io)?;
    let stream = PleStream::open(
        reader,
        offset,
        rows,
        params,
        opts.ngram_cache_bytes,
        &opts.ngram_reader,
    )?;
    let resident_bias = upload(backend, bias.to_vec())?;
    NgramTable::new(DType::Exl3Ngram6, Arc::new(stream), Some(resident_bias))
}

/// One quantized projection as the shards store it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Trellis<'a> {
    /// `[k / 16, n / 16, 16 * bits]` words: tile rows over the contraction,
    /// tile columns over the outputs.
    pub tiles: &'a [u8],
    /// `k` half-precision input scales.
    pub suh: &'a [u8],
    /// `n` half-precision output scales.
    pub svh: &'a [u8],
    /// Contraction width.
    pub k: usize,
    /// Output width.
    pub n: usize,
    /// Bits per weight.
    pub bits: usize,
}

impl Trellis<'_> {
    /// Bytes one tile occupies.
    const fn tile_bytes(&self) -> usize {
        TILE * self.bits * 2
    }

    /// Bytes one row of tiles occupies.
    const fn tile_row_bytes(&self) -> usize {
        self.n / TILE * self.tile_bytes()
    }
}

/// How one quantized projection is moved on its way to the device. Every
/// move is in whole 128-wide rotation blocks.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Move<'p> {
    /// As stored.
    InPlace,
    /// Outputs in 128-wide blocks, the first `lead` kept, the rest reordered:
    /// new block `lead + i` is old block `lead + perm[i]`.
    Outputs { lead: usize, perm: &'p [usize] },
    /// Contraction in 128-wide blocks, reordered by `perm`.
    Inputs { perm: &'p [usize] },
    /// Only outputs `lo..hi`, both multiples of 128.
    Slice { lo: usize, hi: usize },
}

/// The device payload of one projection: tiles, then input scales, then
/// output scales, after `how`. Returns the payload and its output width.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] for a move that is not whole blocks of this
/// projection.
pub(crate) fn assemble(t: &Trellis<'_>, how: Move<'_>) -> CoreResult<(Vec<u8>, usize)> {
    let tile = t.tile_bytes();
    let row = t.tile_row_bytes();
    let col_block = BLOCK / TILE * tile;
    let scale_block = BLOCK * 2;
    let out = |n: usize, tiles: Vec<u8>, suh: &[u8], svh: Vec<u8>| {
        let mut payload = tiles;
        payload.extend_from_slice(suh);
        payload.extend_from_slice(&svh);
        (payload, n)
    };
    let fits = |blocks: usize, width: usize| blocks * BLOCK <= width;
    match how {
        Move::InPlace => Ok(out(t.n, t.tiles.to_vec(), t.suh, t.svh.to_vec())),
        Move::Outputs { lead, perm } => {
            if !fits(lead + perm.len(), t.n) || perm.iter().any(|&p| p >= perm.len()) {
                return Err(move_mismatch("output blocks", t.n, lead + perm.len()));
            }
            let mut tiles = t.tiles.to_vec();
            for (dst, src) in tiles.chunks_mut(row).zip(t.tiles.chunks(row)) {
                for (i, &old) in perm.iter().enumerate() {
                    let to = (lead + i) * col_block;
                    let from = (lead + old) * col_block;
                    dst[to..to + col_block].copy_from_slice(&src[from..from + col_block]);
                }
            }
            let svh = permute_blocks(t.svh, scale_block, lead, perm);
            Ok(out(t.n, tiles, t.suh, svh))
        }
        Move::Inputs { perm } => {
            if perm.len() * BLOCK != t.k || perm.iter().any(|&p| p >= perm.len()) {
                return Err(move_mismatch("input blocks", t.k, perm.len()));
            }
            let tiles = permute_blocks(t.tiles, BLOCK / TILE * row, 0, perm);
            let suh = permute_blocks(t.suh, scale_block, 0, perm);
            let mut payload = tiles;
            payload.extend_from_slice(&suh);
            payload.extend_from_slice(t.svh);
            Ok((payload, t.n))
        }
        Move::Slice { lo, hi } => {
            if lo % BLOCK != 0 || hi % BLOCK != 0 || lo >= hi || hi > t.n {
                return Err(move_mismatch("output slice", t.n, hi));
            }
            let (from, to) = (lo / TILE * tile, hi / TILE * tile);
            let tiles: Vec<u8> = t
                .tiles
                .chunks(row)
                .flat_map(|r| r[from..to].iter().copied())
                .collect();
            Ok(out(hi - lo, tiles, t.suh, t.svh[lo * 2..hi * 2].to_vec()))
        }
    }
}

/// The refusal for a move that is not whole blocks of the projection.
fn move_mismatch(field: &'static str, width: usize, blocks: usize) -> CoreError {
    CoreError::ShapeMismatch {
        field,
        expected: width as u64,
        got: blocks as u64,
    }
}

/// The device payload of a bank: every expert's tiles, then every expert's
/// input scales, then every expert's output scales.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] for an expert of another shape or width than
/// the first.
pub(crate) fn stack(experts: &[Trellis<'_>]) -> CoreResult<Vec<u8>> {
    let first = experts.first().ok_or(CoreError::ShapeMismatch {
        field: "experts in a bank",
        expected: 1,
        got: 0,
    })?;
    if let Some(odd) = experts
        .iter()
        .find(|e| (e.k, e.n, e.bits) != (first.k, first.n, first.bits))
    {
        return Err(CoreError::ShapeMismatch {
            field: "expert shape in a bank",
            expected: (first.k * first.n * first.bits) as u64,
            got: (odd.k * odd.n * odd.bits) as u64,
        });
    }
    let per = first.tiles.len() + first.suh.len() + first.svh.len();
    let mut payload = Vec::with_capacity(per * experts.len());
    for e in experts {
        payload.extend_from_slice(e.tiles);
    }
    for e in experts {
        payload.extend_from_slice(e.suh);
    }
    for e in experts {
        payload.extend_from_slice(e.svh);
    }
    Ok(payload)
}

/// The kernels' format for a trellis width.
fn trellis_dtype(bits: usize) -> CoreResult<DType> {
    match bits {
        4 => Ok(DType::Exl3K4),
        5 => Ok(DType::Exl3K5),
        6 => Ok(DType::Exl3K6),
        8 => Ok(DType::Exl3K8),
        other => Err(CoreError::InvalidValue {
            key: "trellis width",
            value: format!("{other} bits; the kernels decode 4, 5, 6 and 8"),
        }),
    }
}

/// `1 + w`: the centred norms, which this export stores uncentred.
fn plus_one(w: f32) -> f32 {
    1.0 + w
}

/// The loader's state over one open directory.
struct Loader<'c> {
    ck: &'c MlxCheckpoint,
    backend: &'c Arc<dyn Backend>,
    /// Repack half-precision matrices to twelve bits a weight.
    pack: bool,
}

impl Loader<'_> {
    /// One projection's three tensors, checked against each other and against
    /// the codebook the kernels decode with.
    fn trellis(&self, base: &str) -> CoreResult<Trellis<'_>> {
        let name = format!("{base}.trellis");
        let (_, raw) = self.ck.raw(&name)?;
        let shape = raw.shape.clone();
        let &[rows, cols, words] = shape.as_slice() else {
            return Err(CoreError::InvalidValue {
                key: "trellis layout",
                value: format!("{name} is {:?}, not [k/16, n/16, 16*bits]", raw.shape),
            });
        };
        let bits = usize::try_from(words / 16).unwrap_or(0);
        if words % 16 != 0 {
            return Err(CoreError::InvalidValue {
                key: "trellis layout",
                value: format!("{name} has {words} words a tile"),
            });
        }
        trellis_dtype(bits)?;
        let (k, n) = (rows * 16, cols * 16);
        let tiles = self.ck.bytes(&name, "I16", &shape)?;
        let suh = self.ck.bytes(&format!("{base}.suh"), "F16", &[k])?;
        let svh = self.ck.bytes(&format!("{base}.svh"), "F16", &[n])?;
        let mul = self.ck.bytes(&format!("{base}.mul1"), "I32", &[])?;
        let codebook = mul
            .first_chunk::<4>()
            .map(|word| u32::from_le_bytes(*word))
            .unwrap_or_default();
        if codebook != CODEBOOK {
            return Err(CoreError::InvalidValue {
                key: "trellis codebook",
                value: format!(
                    "{base} was quantized against codebook multiplier {codebook:#010x}; the \
                     kernels decode {CODEBOOK:#010x}"
                ),
            });
        }
        Ok(Trellis {
            tiles,
            suh,
            svh,
            k: usize::try_from(k).unwrap_or(0),
            n: usize::try_from(n).unwrap_or(0),
            bits,
        })
    }

    /// Drops the stored pages of `base`'s three tensors once the device has
    /// its copy.
    fn release_trellis(&self, base: &str) {
        for part in ["trellis", "suh", "svh"] {
            self.release(&format!("{base}.{part}"));
        }
    }

    /// Drops one tensor's stored pages from the page cache.
    fn release(&self, name: &str) {
        if let Ok((slot, raw)) = self.ck.raw(name)
            && let Ok(range) = self
                .ck
                .shard(slot)
                .payload_range(raw.data_offsets[0], raw.data_offsets[1])
        {
            self.ck.shard(slot).release_cached(range);
        }
    }

    /// One quantized projection, checked against the widths the family's
    /// shapes say it has.
    fn projection(
        &self,
        base: &str,
        in_dim: u32,
        out_dim: u32,
        how: Move<'_>,
    ) -> CoreResult<Weight> {
        let t = self.trellis(base)?;
        let (payload, n) = assemble(&t, how)?;
        if t.k != in_dim as usize || n != out_dim as usize {
            return Err(CoreError::InvalidValue {
                key: "projection widths",
                value: format!(
                    "{base} is {}x{n} after the load's move, this family's is {in_dim}x{out_dim}",
                    t.k
                ),
            });
        }
        let dtype = trellis_dtype(t.bits)?;
        let weight = Weight::uploaded(self.backend, dtype, in_dim, out_dim, 1, payload)?;
        // The payload is a copy; the stored pages are now only cache.
        self.release_trellis(base);
        Ok(weight)
    }

    /// One routed bank: `EXPERTS` projections stacked into one payload.
    fn bank(&self, mlp: &str, proj: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        let bases: Vec<String> = (0..EXPERTS)
            .map(|e| format!("{mlp}.experts.{e}.{proj}"))
            .collect();
        let experts = bases
            .iter()
            .map(|base| self.trellis(base))
            .collect::<CoreResult<Vec<_>>>()?;
        let first = experts.first().copied().ok_or(CoreError::ShapeMismatch {
            field: "experts in a bank",
            expected: u64::from(EXPERTS),
            got: 0,
        })?;
        if first.k != in_dim as usize || first.n != out_dim as usize {
            return Err(CoreError::InvalidValue {
                key: "expert widths",
                value: format!(
                    "{mlp} {proj} is {}x{}, this family's is {in_dim}x{out_dim}",
                    first.k, first.n
                ),
            });
        }
        let payload = stack(&experts)?;
        let dtype = trellis_dtype(first.bits)?;
        drop(experts);
        let weight = Weight::uploaded(self.backend, dtype, in_dim, out_dim, EXPERTS, payload)?;
        for base in &bases {
            self.release_trellis(base);
        }
        Ok(weight)
    }

    /// A half-precision matrix `[out, in]`, its rows optionally reordered,
    /// repacked to twelve bits a weight when the loader packs and it has more
    /// than one row.
    fn half(
        &self,
        name: &str,
        in_dim: u32,
        out_dim: u32,
        rows: Option<&[usize]>,
    ) -> CoreResult<Weight> {
        let shape = [u64::from(out_dim), u64::from(in_dim)];
        let (slot, span) = self.ck.span(name, "F16", &shape)?;
        let host: Arc<Shard> = self.ck.shard(slot);
        let stored = &AsRef::<[u8]>::as_ref(&*host)[span.clone()];
        let moved = rows.map(|perm| permute_blocks(stored, in_dim as usize * 2, 0, perm));
        let halves = moved.as_deref().unwrap_or(stored);
        // A one-row matrix is a vector, and the single-file container stores
        // it as one, which its loader never packs. Kept plain here too, so the
        // two containers of the same weights run the same kernels and answer
        // the same tokens: the packing is lossless, but its kernel sums in
        // another order, and that alone moved a greedy answer.
        if self.pack
            && out_dim > 1
            && let Some(packed) = Weight::packed_half(self.backend, in_dim, out_dim, halves)?
        {
            self.release(name);
            return Ok(packed);
        }
        let buffer = match moved {
            Some(bytes) => upload(self.backend, bytes)?,
            None => import_host(self.backend, host, span)?,
        };
        if self.backend.imports_copy() || rows.is_some() {
            self.release(name);
        }
        Ok(Weight::over(DType::F16, in_dim, out_dim, buffer))
    }

    /// A brain-float matrix `[out, in]`, imported in place.
    fn brain(&self, name: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        let shape = [u64::from(out_dim), u64::from(in_dim)];
        let (slot, span) = self.ck.span(name, "BF16", &shape)?;
        let buffer = import_host(self.backend, self.ck.shard(slot), span)?;
        if self.backend.imports_copy() {
            self.release(name);
        }
        Ok(Weight::over(DType::BF16, in_dim, out_dim, buffer))
    }

    /// A single-precision vector from a half- or brain-float tensor, through
    /// `op`, its blocks of `block` values reordered past the first `lead`
    /// when `perm` is given.
    fn vector(
        &self,
        name: &str,
        shape: &[u64],
        op: impl Fn(f32) -> f32,
        perm: Option<(usize, usize, &[usize])>,
    ) -> CoreResult<DeviceBuffer> {
        let (_, raw) = self.ck.raw(name)?;
        let dtype = raw.dtype.clone();
        let widen: fn([u8; 2]) -> f32 = match dtype.as_str() {
            "F16" => |pair| f32::from(half::f16::from_le_bytes(pair)),
            "BF16" => |pair| f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16),
            _ => {
                return Err(CoreError::InvalidValue {
                    key: "vector dtype",
                    value: format!("{name} is {dtype}, not a half- or brain-float vector"),
                });
            }
        };
        let values: Vec<f32> = self
            .ck
            .bytes(name, &dtype, shape)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| op(widen(*pair)))
            .collect();
        let bytes = f32_bytes(&values);
        let bytes = match perm {
            Some((block, lead, perm)) => permute_blocks(&bytes, block * 4, lead, perm),
            None => bytes,
        };
        upload(self.backend, bytes)
    }

    /// One stream mix. `inject` is absent on the collapsing mix.
    fn mix(&self, base: &str, inject: bool) -> CoreResult<StreamMix> {
        let wide = HIDDEN * STREAMS;
        Ok(StreamMix {
            norm: self.vector(
                &format!("{base}.hc_norm.weight"),
                &[u64::from(wide)],
                plus_one,
                None,
            )?,
            down: self.half(
                &format!("{base}.input_mix_weight_down.weight"),
                wide,
                STREAM_RANK,
                None,
            )?,
            up: self.half(
                &format!("{base}.input_mix_weight_up.weight"),
                STREAM_RANK,
                wide,
                None,
            )?,
            inject: inject
                .then(|| {
                    self.half(
                        &format!("{base}.block_inject_weight.weight"),
                        wide,
                        STREAMS,
                        None,
                    )
                })
                .transpose()?,
        })
    }

    /// One attention slot, the target's or the draft head's.
    fn attn(&self, base: &str) -> CoreResult<SparseAttn> {
        let a = format!("{base}.self_attn");
        let kv = KV_HEADS * HEAD_DIM;
        let idx_q = INDEXER_HEADS * INDEXER_HEAD_DIM;
        let idx_rows = idx_q + INDEXER_HEAD_DIM;
        let idx = format!("{a}.indexer.index_qk_proj");
        let norm = |name: &str, width: u32| {
            self.vector(
                &format!("{a}.{name}.weight"),
                &[u64::from(width)],
                plus_one,
                None,
            )
        };
        let indexer = Indexer {
            q: self.projection(
                &idx,
                HIDDEN,
                idx_q,
                Move::Slice {
                    lo: 0,
                    hi: idx_q as usize,
                },
            )?,
            k: self.projection(
                &idx,
                HIDDEN,
                idx_rows - idx_q,
                Move::Slice {
                    lo: idx_q as usize,
                    hi: idx_rows as usize,
                },
            )?,
            q_norm: norm("indexer.q_layernorm", INDEXER_HEAD_DIM)?,
            k_norm: norm("indexer.k_layernorm", INDEXER_HEAD_DIM)?,
        };
        Ok(SparseAttn {
            q: self.projection(
                &format!("{a}.q_proj"),
                HIDDEN,
                Q_HEADS * HEAD_DIM * 2,
                Move::InPlace,
            )?,
            k: self.projection(&format!("{a}.k_proj"), HIDDEN, kv, Move::InPlace)?,
            v: self.projection(&format!("{a}.v_proj"), HIDDEN, kv, Move::InPlace)?,
            output: self.projection(
                &format!("{a}.o_proj"),
                Q_HEADS * HEAD_DIM,
                HIDDEN,
                Move::InPlace,
            )?,
            q_norm: norm("q_norm", HEAD_DIM)?,
            k_norm: norm("k_norm", HEAD_DIM)?,
            indexer,
        })
    }

    /// One recurrent slot, its value heads moved to the tiled order.
    fn recurrent(&self, base: &str) -> CoreResult<Recurrent> {
        let la = format!("{base}.linear_attn");
        let heads = value_heads();
        let hd = GDN_HEAD_DIM as usize;
        // The query and key heads ahead of the value heads, in 128-wide blocks.
        let qk_blocks = 2 * GDN_QK_HEADS as usize * hd / BLOCK;
        let v_rows = GDN_V_HEADS * GDN_HEAD_DIM;
        let qkv_rows = 2 * GDN_QK_HEADS * GDN_HEAD_DIM + v_rows;
        let n_v = u64::from(GDN_V_HEADS);
        Ok(Recurrent {
            qkv: self.projection(
                &format!("{la}.in_proj_qkv"),
                HIDDEN,
                qkv_rows,
                Move::Outputs {
                    lead: qk_blocks,
                    perm: &heads,
                },
            )?,
            gate: self.projection(
                &format!("{la}.in_proj_z"),
                HIDDEN,
                v_rows,
                Move::Outputs {
                    lead: 0,
                    perm: &heads,
                },
            )?,
            alpha: self.half(
                &format!("{la}.in_proj_a.weight"),
                HIDDEN,
                GDN_V_HEADS,
                Some(&heads),
            )?,
            beta: self.half(
                &format!("{la}.in_proj_b.weight"),
                HIDDEN,
                GDN_V_HEADS,
                Some(&heads),
            )?,
            out: self.projection(
                &format!("{la}.out_proj"),
                v_rows,
                HIDDEN,
                Move::Inputs { perm: &heads },
            )?,
            // `[channels, 1, kernel]`: a channel's taps are contiguous, which
            // is the flat order the kernels read.
            conv1d: self.vector(
                &format!("{la}.conv1d.weight"),
                &[u64::from(qkv_rows), 1, u64::from(cfg::GDN_CONV)],
                std::convert::identity,
                Some((
                    hd * cfg::GDN_CONV as usize,
                    2 * GDN_QK_HEADS as usize,
                    &heads,
                )),
            )?,
            a: self.vector(
                &format!("{la}.A_log"),
                &[n_v],
                |a| -a.exp(),
                Some((1, 0, &heads)),
            )?,
            dt_bias: self.vector(
                &format!("{la}.dt_bias"),
                &[n_v],
                std::convert::identity,
                Some((1, 0, &heads)),
            )?,
            norm: self.vector(
                &format!("{la}.norm.weight"),
                &[u64::from(GDN_HEAD_DIM)],
                std::convert::identity,
                None,
            )?,
        })
    }

    /// One routed feed-forward, its banks resident.
    fn experts(&self, base: &str) -> CoreResult<ExpertBank> {
        let m = format!("{base}.mlp");
        Ok(ExpertBank {
            router: self.half(&format!("{m}.gate.weight"), HIDDEN, EXPERTS, None)?,
            gate: self.bank(&m, "gate_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            up: self.bank(&m, "up_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            down: self.bank(&m, "down_proj", EXPERT_INTERMEDIATE, HIDDEN)?,
            shared_router: self.half(&format!("{m}.shared_expert_gate.weight"), HIDDEN, 1, None)?,
            shared_gate: self.projection(
                &format!("{m}.shared_expert.gate_proj"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
                Move::InPlace,
            )?,
            shared_up: self.projection(
                &format!("{m}.shared_expert.up_proj"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
                Move::InPlace,
            )?,
            shared_down: self.projection(
                &format!("{m}.shared_expert.down_proj"),
                EXPERT_INTERMEDIATE,
                HIDDEN,
                Move::InPlace,
            )?,
        })
    }

    /// The n-gram block.
    fn ngram(&self, base: &str) -> CoreResult<NgramBlock> {
        let p = format!("{base}.ple");
        let wide = HIDDEN * STREAMS;
        let norm = |name: &str| {
            self.vector(
                &format!("{p}.{name}.weight"),
                &[u64::from(wide)],
                plus_one,
                None,
            )
        };
        Ok(NgramBlock {
            key: self.half(&format!("{p}.key_proj.weight"), HIDDEN, wide, None)?,
            value: self.half(&format!("{p}.value_proj.weight"), HIDDEN, HIDDEN, None)?,
            norm_key: norm("norm_key")?,
            norm_query: norm("norm_query")?,
            norm_conv: norm("norm_conv")?,
            conv1d: self.vector(
                &format!("{p}.conv1d.weight"),
                &[u64::from(wide), 1, u64::from(cfg::PLE_CONV)],
                std::convert::identity,
                None,
            )?,
        })
    }
}

/// Imports every weight of the export's language model; the routed banks are
/// resident.
///
/// `pack` repacks the half-precision matrices to twelve bits a weight, as the
/// single-file container's loader does.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first tensor the directory does not
/// carry, [`CoreError::InvalidValue`] for one of another layout or codebook,
/// and whatever the device reports.
pub fn import(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    config: &Qwen4ExpConfig,
    pack: bool,
) -> CoreResult<SparseWeights> {
    let loader = Loader { ck, backend, pack };
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
            loader.experts(&base)?
        };
        let ngram = (il == PLE_LAYER)
            .then(|| {
                let _tag = ledger::scope(MemTag::WeightsEmbedding);
                loader.ngram(&base)
            })
            .transpose()?;
        tracing::debug!(layer = il, "imported a layer of the trellis export");
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
        loader.brain(&format!("{LM}.embed_tokens.weight"), HIDDEN, VOCAB)?
    };
    let _tag = ledger::scope(MemTag::WeightsHead);
    let lm_head = loader.projection("lm_head", HIDDEN, VOCAB, Move::InPlace)?;
    let output_mix = loader.mix(&format!("{LM}.hyper_connection_mixer"), false)?;
    Ok(SparseWeights {
        token_embd,
        lm_head: Some(lm_head),
        output_mix,
        layers,
        draft: None,
        expert_map: None,
    })
}

/// Imports the draft head the export carries beside the language model.
///
/// `rank` is the head's rank among attention slots: one past the target's
/// last, so its cache planes are the extra rank the pool is opened with.
///
/// # Errors
///
/// As [`import`].
pub fn import_draft(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    rank: u32,
    pack: bool,
) -> CoreResult<DraftHead> {
    let loader = Loader { ck, backend, pack };
    let _tag = ledger::scope(MemTag::DraftWeights);
    let layer = format!("{MTP}.layers.0");
    let wide = HIDDEN * STREAMS;
    Ok(DraftHead {
        norm_embed: loader.vector(
            &format!("{MTP}.pre_fc_norm_embedding.weight"),
            &[u64::from(HIDDEN)],
            plus_one,
            None,
        )?,
        norm_hidden: loader.vector(
            &format!("{MTP}.pre_fc_norm_hidden.weight"),
            &[u64::from(wide)],
            plus_one,
            None,
        )?,
        fc_embed: loader.projection(
            &format!("{MTP}.fc_embedding"),
            HIDDEN,
            HIDDEN,
            Move::InPlace,
        )?,
        fc_hidden: loader.projection(&format!("{MTP}.fc_hidden"), HIDDEN, HIDDEN, Move::InPlace)?,
        layer: SparseLayer {
            attn_mix: loader.mix(&format!("{layer}.attn_hyper_connection"), true)?,
            ffn_mix: loader.mix(&format!("{layer}.mlp_hyper_connection"), true)?,
            block: SparseBlock::Attn(loader.attn(&layer)?),
            experts: loader.experts(&layer)?,
            ngram: None,
            ordinal: rank,
        },
        output_mix: loader.mix(&format!("{MTP}.hyper_connection_mixer"), false)?,
        head: None,
    })
}

#[cfg(test)]
#[path = "exl3_tests.rs"]
mod tests;
