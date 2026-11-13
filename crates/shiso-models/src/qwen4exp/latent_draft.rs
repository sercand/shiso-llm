//! The latent pack's draft head: the `mtp/` folder's flat files, read into the
//! same [`DraftHead`] the family's other loaders build.
//!
//! The folder holds
//!
//! - `dense.txt` and `dense.bin`: an index and the head's dense tensors, flat
//!   (no container), each stored as `Q8_0`, `BF16` or `F32` and named as the
//!   reference checkpoint names them (`self_attn.q_proj.weight`,
//!   `mlp_hyper_connection.hc_norm.weight`, ...);
//! - `experts.bin`: the head's 512 routed experts, one record of
//!   [`RECORD_BYTES`] each, no header;
//! - `draft_vocab.bin`, which is not read: `--draft-vocab` keeps its own file.
//!
//! The head is not latent. Its experts are ordinary 2560-wide, 640-inner
//! experts, so they run on the same draft path as any other head's and need
//! nothing of the engine's latent stage.
//!
//! # Norms are loaded as stored
//!
//! A norm weight in the checkpoints this family reads from safetensors is
//! stored without its "+1" and the loader adds it. This pack stores the sum
//! already (its own forward multiplies by the stored value), exactly as a
//! GGUF of the family does, so every norm, the stream mixes' and the
//! attention's included, goes to the device unchanged. The hidden-state norm
//! is one vector over all four streams (10240 values), as the engine's draft
//! pass applies it.
//!
//! # Experts: a re-layout, never a decode
//!
//! A record stores each projection as *planes*: all the 2-bit codes, then all
//! the `f16` block scales. Shiso's `Q2_0` stores the two together, one block
//! of 64 values being `f16 d` and 16 code bytes (18 bytes), the value of
//! element `j` being `((codes[j / 4] >> 2 * (j % 4)) & 3) - 1` times `d`. The
//! planes already use exactly that code packing, so gathering each block's
//! scale and 16 code bytes next to each other is lossless; the tests decode
//! the planes and the blocks and compare every value bit for bit.
//!
//! Gate and up share one plane of `2 * 640` rows, interleaved: row `2r` is the
//! gate's row `r` and row `2r + 1` the up's. The engine keeps the two as
//! banks of their own, so the rows are dealt out on the way.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceBuffer;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ledger::{self, MemTag};
use shiso_engine::sparse::weights::{
    DraftHead, ExpertBank, Indexer, SparseAttn, SparseBlock, SparseLayer, StreamMix, Weight,
};
use shiso_modelfile::st_header::open_uncached;

use crate::qwen4exp::config::{
    EXPERT_INTERMEDIATE, EXPERTS, HEAD_DIM, HIDDEN, INDEXER_HEAD_DIM, INDEXER_HEADS, KV_HEADS,
    Q_HEADS, STREAM_RANK, STREAMS,
};
use crate::qwen4exp::mlx::upload;

/// The dense tensors' index.
const INDEX: &str = "dense.txt";
/// The dense tensors.
const DENSE: &str = "dense.bin";
/// The routed experts.
const EXPERTS_FILE: &str = "experts.bin";

/// Values one `Q2_0` block holds.
const BLOCK_VALUES: usize = 64;
/// Code bytes of one block: four 2-bit codes a byte.
const BLOCK_CODES: usize = BLOCK_VALUES / 4;
/// Bytes of one stored `Q2_0` block: `f16 d`, then the codes.
const BLOCK_BYTES: usize = 2 + BLOCK_CODES;

/// Bytes of one stored expert record at the family's widths: both planes of
/// each of the three projections.
pub(super) const RECORD_BYTES: usize = 1_382_400;

/// How a dense tensor is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DenseKind {
    /// `Q8_0` blocks of 32 along the input.
    Q8,
    /// Brain-float.
    Bf16,
    /// Single precision.
    F32,
}

impl DenseKind {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "q8_0" => Some(Self::Q8),
            "bf16" => Some(Self::Bf16),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    const fn dtype(self) -> DType {
        match self {
            Self::Q8 => DType::Q8_0,
            Self::Bf16 => DType::BF16,
            Self::F32 => DType::F32,
        }
    }

    /// Bytes of a `rows x cols` tensor in this storage, `None` when `cols`
    /// is not whole blocks.
    fn bytes(self, rows: u64, cols: u64) -> Option<u64> {
        let row = match self {
            Self::Q8 => cols.is_multiple_of(32).then_some(cols / 32 * 34)?,
            Self::Bf16 => cols * 2,
            Self::F32 => cols * 4,
        };
        rows.checked_mul(row)
    }
}

/// One line of `dense.txt`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DenseEntry {
    /// The tensor's name.
    pub(super) name: String,
    /// How it is stored.
    pub(super) kind: DenseKind,
    /// Output width.
    pub(super) rows: u32,
    /// Input width.
    pub(super) cols: u32,
    /// Where it starts in `dense.bin`.
    pub(super) offset: u64,
    /// How many bytes it occupies.
    pub(super) bytes: u64,
}

/// The parsed, checked index of `dense.bin`.
#[derive(Debug)]
pub(super) struct DenseIndex {
    entries: Vec<DenseEntry>,
}

/// One refusal, naming the file it is about.
fn refuse(file: &'static str, why: impl std::fmt::Display) -> CoreError {
    CoreError::InvalidValue {
        key: file,
        value: why.to_string(),
    }
}

impl DenseIndex {
    /// Parses `dense.txt`: per line `name kind rows cols offset bytes`.
    /// Blank lines and `#` comments are skipped.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming `dense.txt` for a line of the wrong
    /// shape, a number that is not one, a kind that is not `q8_0`, `bf16` or
    /// `f32`, a byte count the kind and shape do not give, a repeated name, or
    /// two tensors that overlap.
    pub(super) fn parse(text: &str) -> CoreResult<Self> {
        let mut entries: Vec<DenseEntry> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let at = n + 1;
            let cols: Vec<&str> = line.split_whitespace().collect();
            let [name, kind, rows, width, offset, bytes] = cols[..] else {
                return Err(refuse(
                    INDEX,
                    format!("line {at}: {} fields, want 6", cols.len()),
                ));
            };
            let number = |what: &str, text: &str| {
                text.parse::<u64>().map_err(|_| {
                    refuse(INDEX, format!("line {at}: {what} {text:?} is not a number"))
                })
            };
            let kind = DenseKind::parse(kind)
                .ok_or_else(|| refuse(INDEX, format!("line {at}: unknown kind {kind:?}")))?;
            let (rows, width) = (number("rows", rows)?, number("cols", width)?);
            let (offset, bytes) = (number("offset", offset)?, number("bytes", bytes)?);
            let want = kind.bytes(rows, width).ok_or_else(|| {
                refuse(
                    INDEX,
                    format!("line {at}: {name}: {width} columns are not whole blocks"),
                )
            })?;
            if rows == 0 || bytes != want {
                return Err(refuse(
                    INDEX,
                    format!(
                        "line {at}: {name}: {rows}x{width} {kind:?} is {want} bytes, not {bytes}"
                    ),
                ));
            }
            if entries.iter().any(|e| e.name == name) {
                return Err(refuse(INDEX, format!("line {at}: {name} appears twice")));
            }
            let narrow = |what: &str, v: u64| {
                u32::try_from(v)
                    .map_err(|_| refuse(INDEX, format!("line {at}: {what} {v} is too large")))
            };
            entries.push(DenseEntry {
                name: name.to_owned(),
                kind,
                rows: narrow("rows", rows)?,
                cols: narrow("cols", width)?,
                offset,
                bytes,
            });
        }
        let mut order: Vec<&DenseEntry> = entries.iter().collect();
        order.sort_by_key(|e| e.offset);
        for pair in order.windows(2) {
            let end = pair[0].offset.saturating_add(pair[0].bytes);
            if end > pair[1].offset {
                return Err(refuse(
                    INDEX,
                    format!("{} overlaps {}", pair[0].name, pair[1].name),
                ));
            }
        }
        Ok(Self { entries })
    }

    /// Bytes `dense.bin` has to hold: the end of the last tensor.
    fn extent(&self) -> u64 {
        self.entries
            .iter()
            .map(|e| e.offset.saturating_add(e.bytes))
            .max()
            .unwrap_or(0)
    }
}

/// `dense.bin` read whole, with the index that names its tensors.
struct Dense {
    index: DenseIndex,
    bytes: Vec<u8>,
}

impl Dense {
    /// A tensor's bytes, after checking it is the kind and shape the head's
    /// role needs.
    fn tensor(&self, name: &str, kind: DenseKind, rows: u32, cols: u32) -> CoreResult<&[u8]> {
        let entry = self
            .index
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: format!("mtp/{name} (listed in {INDEX})"),
            })?;
        if entry.kind != kind || entry.rows != rows || entry.cols != cols {
            return Err(refuse(
                INDEX,
                format!(
                    "{name} is {}x{} {:?}, the draft head needs {rows}x{cols} {kind:?}",
                    entry.rows, entry.cols, entry.kind
                ),
            ));
        }
        let start = usize::try_from(entry.offset).unwrap_or(usize::MAX);
        let end = start.saturating_add(usize::try_from(entry.bytes).unwrap_or(usize::MAX));
        self.bytes.get(start..end).ok_or(CoreError::OutOfBounds {
            offset: entry.offset,
            len: entry.bytes,
            buffer_bytes: self.bytes.len() as u64,
        })
    }

    /// A matrix of `rows` outputs over `cols` inputs on the device.
    fn matrix(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        kind: DenseKind,
        rows: u32,
        cols: u32,
    ) -> CoreResult<Weight> {
        let bytes = self.tensor(name, kind, rows, cols)?;
        Weight::uploaded(backend, kind.dtype(), cols, rows, 1, bytes.to_vec())
    }

    /// A single-precision vector, as stored.
    fn vector(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        width: u32,
    ) -> CoreResult<DeviceBuffer> {
        upload(
            backend,
            self.tensor(name, DenseKind::F32, 1, width)?.to_vec(),
        )
    }

    /// One stream mix: `base` is the tensor-name prefix.
    fn mix(&self, backend: &Arc<dyn Backend>, base: &str, inject: bool) -> CoreResult<StreamMix> {
        let wide = HIDDEN * STREAMS;
        let name = |what: &str| format!("{base}.{what}.weight");
        Ok(StreamMix {
            norm: self.vector(backend, &name("hc_norm"), wide)?,
            down: self.matrix(
                backend,
                &name("input_mix_weight_down"),
                DenseKind::Bf16,
                STREAM_RANK,
                wide,
            )?,
            up: self.matrix(
                backend,
                &name("input_mix_weight_up"),
                DenseKind::Bf16,
                wide,
                STREAM_RANK,
            )?,
            inject: inject
                .then(|| {
                    self.matrix(
                        backend,
                        &name("block_inject_weight"),
                        DenseKind::Bf16,
                        STREAMS,
                        wide,
                    )
                })
                .transpose()?,
        })
    }

    /// The head's attention block.
    fn attn(&self, backend: &Arc<dyn Backend>) -> CoreResult<SparseAttn> {
        let kv = KV_HEADS * HEAD_DIM;
        let idx_q = INDEXER_HEADS * INDEXER_HEAD_DIM;
        let idx_rows = idx_q + INDEXER_HEAD_DIM;
        let q8 = DenseKind::Q8;
        let indexer = self.tensor(
            "self_attn.indexer.index_qk_proj.weight",
            q8,
            idx_rows,
            HIDDEN,
        )?;
        let (q_rows, k_rows) =
            split_rows(indexer, idx_rows, idx_q).ok_or(CoreError::ShapeMismatch {
                field: "the indexer projection's rows",
                expected: u64::from(idx_rows),
                got: indexer.len() as u64,
            })?;
        let sub = |bytes: &[u8], rows: u32| {
            Weight::uploaded(backend, DType::Q8_0, HIDDEN, rows, 1, bytes.to_vec())
        };
        Ok(SparseAttn {
            q: self.matrix(
                backend,
                "self_attn.q_proj.weight",
                q8,
                Q_HEADS * HEAD_DIM * 2,
                HIDDEN,
            )?,
            k: self.matrix(backend, "self_attn.k_proj.weight", q8, kv, HIDDEN)?,
            v: self.matrix(backend, "self_attn.v_proj.weight", q8, kv, HIDDEN)?,
            output: self.matrix(
                backend,
                "self_attn.o_proj.weight",
                q8,
                HIDDEN,
                Q_HEADS * HEAD_DIM,
            )?,
            q_norm: self.vector(backend, "self_attn.q_norm.weight", HEAD_DIM)?,
            k_norm: self.vector(backend, "self_attn.k_norm.weight", HEAD_DIM)?,
            indexer: Indexer {
                q: sub(q_rows, idx_q)?,
                k: sub(k_rows, INDEXER_HEAD_DIM)?,
                q_norm: self.vector(
                    backend,
                    "self_attn.indexer.q_layernorm.weight",
                    INDEXER_HEAD_DIM,
                )?,
                k_norm: self.vector(
                    backend,
                    "self_attn.indexer.k_layernorm.weight",
                    INDEXER_HEAD_DIM,
                )?,
            },
        })
    }
}

/// The first `first` rows of a `rows`-row tensor and the rest, as two slices.
///
/// The indexer's query and key projections are stored as one matrix of
/// 640 rows (512 for the four query heads, then the key head's 128); a row is
/// a whole number of blocks, so the split is a cut in the bytes.
fn split_rows(bytes: &[u8], rows: u32, first: u32) -> Option<(&[u8], &[u8])> {
    let (rows, first) = (rows as usize, first as usize);
    if rows == 0 || first > rows || !bytes.len().is_multiple_of(rows) {
        return None;
    }
    Some(bytes.split_at(bytes.len() / rows * first))
}

/// Where each plane of one record starts, and the widths they hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RecordShape {
    /// The expert's input width, which is also the down projection's output.
    pub(super) hidden: usize,
    /// The expert's inner width.
    pub(super) inner: usize,
}

impl RecordShape {
    /// The family's own: 2560 wide, 640 inner.
    pub(super) const FAMILY: Self = Self {
        hidden: HIDDEN as usize,
        inner: EXPERT_INTERMEDIATE as usize,
    };

    /// Blocks a gate or up row holds, and a down row.
    const fn blocks(self) -> (usize, usize) {
        (self.hidden / BLOCK_VALUES, self.inner / BLOCK_VALUES)
    }

    /// Bytes of the gate/up code plane, of the down code plane and of the
    /// gate/up scale plane: the record's four planes in file order.
    const fn planes(self) -> [usize; 4] {
        let (gu, down) = self.blocks();
        [
            2 * self.inner * gu * BLOCK_CODES,
            self.hidden * down * BLOCK_CODES,
            2 * self.inner * gu * 2,
            self.hidden * down * 2,
        ]
    }

    /// Bytes of one record.
    pub(super) const fn record_bytes(self) -> usize {
        let p = self.planes();
        p[0] + p[1] + p[2] + p[3]
    }

    /// Bytes of one expert's gate (or up) projection as blocks.
    const fn gate_bytes(self) -> usize {
        self.inner * self.blocks().0 * BLOCK_BYTES
    }

    /// Bytes of one expert's down projection as blocks.
    const fn down_bytes(self) -> usize {
        self.hidden * self.blocks().1 * BLOCK_BYTES
    }
}

/// One row of blocks from its two planes: `codes` is `blocks * 16` bytes,
/// `scales` is `blocks` half-precision values, `out` is `blocks * 18` bytes.
fn blocks_of_row(codes: &[u8], scales: &[u8], out: &mut [u8]) {
    for ((block, scale), out) in codes
        .chunks_exact(BLOCK_CODES)
        .zip(scales.chunks_exact(2))
        .zip(out.chunks_exact_mut(BLOCK_BYTES))
    {
        out[..2].copy_from_slice(scale);
        out[2..].copy_from_slice(block);
    }
}

/// One stored record as the three projections' blocks of one expert.
///
/// `gate`, `up` and `down` are the expert's slots in the three banks: `inner`
/// rows of `hidden / 64` blocks for gate and up, `hidden` rows of
/// `inner / 64` blocks for down. Gate row `r` is plane row `2r` and up row
/// `r` plane row `2r + 1`.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] when a slice is not the size the shape gives.
pub(super) fn record_to_blocks(
    shape: RecordShape,
    record: &[u8],
    gate: &mut [u8],
    up: &mut [u8],
    down: &mut [u8],
) -> CoreResult<()> {
    let sizes = [
        (record.len(), shape.record_bytes()),
        (gate.len(), shape.gate_bytes()),
        (up.len(), shape.gate_bytes()),
        (down.len(), shape.down_bytes()),
    ];
    if let Some((got, want)) = sizes.into_iter().find(|(got, want)| got != want) {
        return Err(CoreError::ShapeMismatch {
            field: "draft expert record",
            expected: want as u64,
            got: got as u64,
        });
    }
    let [gu_codes, down_codes, gu_scales, _] = shape.planes();
    let (gu_blocks, down_blocks) = shape.blocks();
    let (codes, rest) = record.split_at(gu_codes);
    let (down_codes_at, rest) = rest.split_at(down_codes);
    let (gu_scales_at, down_scales_at) = rest.split_at(gu_scales);
    let (row_codes, row_scales) = (gu_blocks * BLOCK_CODES, gu_blocks * 2);
    let row_out = gu_blocks * BLOCK_BYTES;
    for r in 0..shape.inner {
        for (plane_row, out) in [(2 * r, &mut *gate), (2 * r + 1, &mut *up)] {
            blocks_of_row(
                &codes[plane_row * row_codes..][..row_codes],
                &gu_scales_at[plane_row * row_scales..][..row_scales],
                &mut out[r * row_out..][..row_out],
            );
        }
    }
    let (row_codes, row_scales) = (down_blocks * BLOCK_CODES, down_blocks * 2);
    let row_out = down_blocks * BLOCK_BYTES;
    for r in 0..shape.hidden {
        blocks_of_row(
            &down_codes_at[r * row_codes..][..row_codes],
            &down_scales_at[r * row_scales..][..row_scales],
            &mut down[r * row_out..][..row_out],
        );
    }
    Ok(())
}

/// Reads `experts.bin` into the three resident `Q2_0` banks.
fn expert_banks(
    backend: &Arc<dyn Backend>,
    file: &File,
    shape: RecordShape,
) -> CoreResult<(Weight, Weight, Weight)> {
    let experts = EXPERTS as usize;
    let (g, d) = (shape.gate_bytes(), shape.down_bytes());
    let mut gate = vec![0u8; g * experts];
    let mut up = vec![0u8; g * experts];
    let mut down = vec![0u8; d * experts];
    let mut record = vec![0u8; shape.record_bytes()];
    for e in 0..experts {
        file.read_exact_at(&mut record, (e * shape.record_bytes()) as u64)
            .map_err(CoreError::Io)?;
        record_to_blocks(
            shape,
            &record,
            &mut gate[e * g..][..g],
            &mut up[e * g..][..g],
            &mut down[e * d..][..d],
        )?;
    }
    let (hidden, inner) = (HIDDEN, EXPERT_INTERMEDIATE);
    let bank = |in_dim, out_dim, bytes| {
        Weight::uploaded(backend, DType::Q2_0, in_dim, out_dim, EXPERTS, bytes)
    };
    Ok((
        bank(hidden, inner, gate)?,
        bank(hidden, inner, up)?,
        bank(inner, hidden, down)?,
    ))
}

/// Reads the pack's `mtp/` folder as a draft head.
///
/// `rank` is the head's rank among attention slots: one past the target's
/// last, so its cache planes are the extra rank the pool is opened with.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for a missing file or a tensor the index does
/// not list, [`CoreError::InvalidValue`] naming the file for an index or a
/// file size that is wrong, [`CoreError::InvalidValue`] for a tensor whose kind
/// or shape the head's role does not take, and whatever the uploads report.
pub(super) fn open(backend: &Arc<dyn Backend>, dir: &Path, rank: u32) -> CoreResult<DraftHead> {
    let _tag = ledger::scope(MemTag::DraftWeights);
    let read = |name: &'static str| match std::fs::read(dir.join(name)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CoreError::ConfigMissing {
            key: format!("mtp/{name}"),
        }),
        other => other.map_err(CoreError::Io),
    };
    let text = String::from_utf8(read(INDEX)?).map_err(|_| refuse(INDEX, "not UTF-8 text"))?;
    let index = DenseIndex::parse(&text)?;
    let bytes = read(DENSE)?;
    if bytes.len() as u64 != index.extent() {
        return Err(refuse(
            DENSE,
            format!("{} bytes, the index covers {}", bytes.len(), index.extent()),
        ));
    }
    let dense = Dense { index, bytes };
    let shape = RecordShape::FAMILY;
    debug_assert_eq!(shape.record_bytes(), RECORD_BYTES);
    let experts = open_uncached(&dir.join(EXPERTS_FILE)).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => CoreError::ConfigMissing {
            key: format!("mtp/{EXPERTS_FILE}"),
        },
        _ => CoreError::Io(e),
    })?;
    let stored = experts.metadata().map_err(CoreError::Io)?.len();
    let want = (RECORD_BYTES * EXPERTS as usize) as u64;
    if stored != want {
        return Err(refuse(
            EXPERTS_FILE,
            format!("{stored} bytes, {EXPERTS} records of {RECORD_BYTES} are {want}"),
        ));
    }

    let (q8, bf16) = (DenseKind::Q8, DenseKind::Bf16);
    let wide = HIDDEN * STREAMS;
    let m = |name: &str, kind, rows, cols| dense.matrix(backend, name, kind, rows, cols);
    let (gate, up, down) = expert_banks(backend, &experts, shape)?;
    let bank = ExpertBank {
        router: m("mlp.gate.weight", bf16, EXPERTS, HIDDEN)?,
        gate,
        up,
        down,
        shared_router: m("mlp.shared_expert_gate.weight", bf16, 1, HIDDEN)?,
        shared_gate: m(
            "mlp.shared_expert.gate_proj.weight",
            q8,
            EXPERT_INTERMEDIATE,
            HIDDEN,
        )?,
        shared_up: m(
            "mlp.shared_expert.up_proj.weight",
            q8,
            EXPERT_INTERMEDIATE,
            HIDDEN,
        )?,
        shared_down: m(
            "mlp.shared_expert.down_proj.weight",
            q8,
            HIDDEN,
            EXPERT_INTERMEDIATE,
        )?,
        latent: None,
    };
    Ok(DraftHead {
        norm_embed: dense.vector(backend, "pre_fc_norm_embedding.weight", HIDDEN)?,
        norm_hidden: dense.vector(backend, "pre_fc_norm_hidden.weight", wide)?,
        fc_embed: m("fc_embedding.weight", q8, HIDDEN, HIDDEN)?,
        fc_hidden: m("fc_hidden.weight", q8, HIDDEN, HIDDEN)?,
        layer: SparseLayer {
            attn_mix: dense.mix(backend, "attn_hyper_connection", true)?,
            ffn_mix: dense.mix(backend, "mlp_hyper_connection", true)?,
            block: SparseBlock::Attn(dense.attn(backend)?),
            experts: bank,
            ngram: None,
            ordinal: rank,
        },
        output_mix: dense.mix(backend, "hyper_connection_mixer", false)?,
        head: None,
    })
}

#[cfg(test)]
#[path = "latent_draft_tests.rs"]
mod tests;
