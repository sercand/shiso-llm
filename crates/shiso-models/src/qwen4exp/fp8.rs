//! The reference export of this family: brain-float dense weights, routed
//! experts in FP8 E4M3 with one scale per 128 x 128 block, and an FP8 n-gram
//! table under one scale - a directory of safetensors shards as the model's
//! own library writes it.
//!
//! The dense weights are imported as the brain floats they are, with the same
//! moves the other containers need made on the way (see [`super::mlx`]): the
//! recurrent block's value heads into the tiled order, the decay base as
//! `-exp(A_log)`, the indexer's fused projection as two row ranges. This export
//! is the unconverted one, so the norms the reference applies as `1 + w` are
//! stored as `w`, and the one is added here.
//!
//! The routed experts are 112 GB, one tensor per expert and projection. They
//! are always read through an [`ExpertStream`]: into a pool that holds every
//! expert on a device that fits them, filled once at load, and into a pool of
//! as many slots as fit otherwise.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::config::HfConfig;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ledger::{self, MemTag};
use shiso_engine::sparse::ple::NgramTable;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_engine::sparse::stream::{ExpertSource, ExpertStream};
use shiso_engine::sparse::weights::{
    DraftHead, ExpertBank, Indexer, NgramBlock, Recurrent, SparseAttn, SparseBlock, SparseLayer,
    SparseWeights, StreamMix, Weight, WeightPart, fp8_block_bytes, nvfp4_bytes,
};
use shiso_modelfile::st_header::Shard;
use shiso_ple::{PleRowType, PleSegment, PleShards, PleStream};

use crate::qwen4exp::config::{
    self as cfg, EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_USED, GDN_HEAD_DIM, GDN_QK_HEADS,
    GDN_V_HEADS, HEAD_DIM, HIDDEN, INDEXER_HEAD_DIM, INDEXER_HEADS, KV_HEADS, N_LAYERS,
    PLE_HEAD_DIM, PLE_LAYER, Q_HEADS, Qwen4ExpConfig, STREAM_RANK, STREAMS, VOCAB, is_attention,
};
use crate::qwen4exp::mlx::{
    MlxCheckpoint, config_from, f32_bytes, permute_blocks, permute_columns, upload, value_heads,
};

/// Prefix of every language-model tensor of this export.
const LM: &str = "model.language_model";
/// Shards the n-gram table is split into.
const NGRAM_SHARDS: usize = 128;

impl Qwen4ExpConfig {
    /// Reads the geometry of the reference export's directory; see
    /// [`Qwen4ExpConfig::from_mlx`], whose checks this shares.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpConfig::from_mlx`].
    pub fn from_reference(config: &HfConfig, ck: &MlxCheckpoint) -> CoreResult<Self> {
        config_from(
            config,
            ck,
            &format!("{LM}.layers.{PLE_LAYER}.ple.ple_embedding"),
        )
    }
}

/// How one brain-float matrix is moved on its way to the device.
#[derive(Debug, Clone, Copy)]
enum Move<'p> {
    /// As stored, wrapped in place.
    InPlace,
    /// Output rows in blocks of `rows`, the first `lead` blocks kept.
    Rows {
        rows: usize,
        lead: usize,
        perm: &'p [usize],
    },
    /// Contraction columns in blocks of `cols`.
    Cols { cols: usize, perm: &'p [usize] },
}

/// `1 + w`: the centred norms, which this export stores uncentred.
fn plus_one(w: f32) -> f32 {
    1.0 + w
}

/// What the dense brain-float matrices become on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFormat {
    /// Kept as brain floats and wrapped in place: the backend reads them as
    /// stored.
    Brain,
    /// Widened to half precision on the way in. The brain-float tier has no
    /// matrix-unit kernel on every backend and plans a wide prompt chunk as
    /// many sixteen-row passes, each re-reading the weight; half precision is
    /// on the BLAS tier wherever there is one, and on the half matvec for a
    /// step. Exact for every weight of this family's magnitude: the two
    /// formats share a mantissa width of at least seven bits, and only a
    /// value below half precision's subnormal floor loses anything.
    Half,
    /// MXFP8 where the export stored a matrix so, kept as stored and served
    /// by the MXFP8 tier; widened to half precision, as [`DenseFormat::Half`],
    /// wherever it stored brain floats (the output head, the routers, the stream mixes and the draft
    /// head's input projections). Decided per tensor by the stored format, so
    /// a checkpoint that quantized one more or one fewer matrix still loads.
    Mxfp8,
}

/// How [`import`] reads one export: its dense format and its routed codec.
#[derive(Debug, Clone, Copy)]
pub struct ImportLayout {
    /// What the dense matrices become.
    pub dense: DenseFormat,
    /// The routed banks' format: [`DType::Fp8E4M3Block128`] for the
    /// reference export, [`DType::Nvfp4Gs16`] for the `modelopt` one.
    pub experts: DType,
    /// Whether half-precision matrices are repacked to twelve bits a weight
    /// ([`Weight::packed_half`]), losslessly: their values are brain floats,
    /// and the stream mixes among them are read whole every decode step.
    pub pack: bool,
}

impl ImportLayout {
    /// The reference export, as it has always loaded.
    pub const REFERENCE: Self = Self {
        dense: DenseFormat::Brain,
        experts: DType::Fp8E4M3Block128,
        pack: false,
    };
}

/// The loader's state over one open directory.
struct Loader<'c> {
    ck: &'c MlxCheckpoint,
    backend: &'c Arc<dyn Backend>,
    layout: ImportLayout,
}

/// Brain-float bytes to half-precision bytes, rounding once through `f32`.
fn brain_to_half(bytes: &[u8]) -> Vec<u8> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|pair| {
            half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

// The bytes of one MXFP8 array after `how`, or `None` when `how` leaves it in
// place and the caller wraps the stored span as it is. `per_value` is 1 for the
// codes (one byte a weight) and the group width for the scales (one byte a
// group); a column move narrower than a group is refused.
fn mxfp8_moved(
    name: &str,
    src: &[u8],
    row_bytes: usize,
    per_value: usize,
    how: Move<'_>,
) -> CoreResult<Option<Vec<u8>>> {
    use shiso_modelfile::mxfp8::GROUP;
    Ok(match how {
        Move::InPlace => None,
        Move::Rows {
            rows: r,
            lead,
            perm,
        } => Some(permute_blocks(src, r * row_bytes, lead, perm)),
        Move::Cols { cols, perm } => {
            if cols % GROUP != 0 {
                return Err(CoreError::InvalidValue {
                    key: "tensor layout",
                    value: format!(
                        "{name}: a column move of {cols} would split MXFP8 scale groups"
                    ),
                });
            }
            let block = if per_value == 1 { cols } else { cols / GROUP };
            Some(permute_columns(src, row_bytes, block, perm))
        }
    })
}

impl Loader<'_> {
    /// A brain-float matrix `[out, in]`, optionally a row range of a larger
    /// one, optionally moved.
    fn dense(
        &self,
        name: &str,
        stored_rows: u32,
        in_dim: u32,
        rows: std::ops::Range<u32>,
        how: Move<'_>,
    ) -> CoreResult<Weight> {
        if self.layout.dense == DenseFormat::Mxfp8 && self.ck.raw(name)?.1.dtype == "F8_E4M3" {
            return self.mxfp8(name, stored_rows, in_dim, rows, how);
        }
        let shape = [u64::from(stored_rows), u64::from(in_dim)];
        let row_bytes = in_dim as usize * 2;
        let (slot, span) = self.ck.span(name, "BF16", &shape)?;
        let start = span.start + rows.start as usize * row_bytes;
        let end = span.start + rows.end as usize * row_bytes;
        let host: Arc<Shard> = self.ck.shard(slot);
        // The mixed export's brain-float matrices (the stream mixes, the routers,
        // the n-gram projections, the output head) take the half-precision
        // path too: the brain-float tier plans a prompt chunk as sixteen-row
        // passes that each re-read the weight, which measured 78% of a
        // 2k-token prefill there (215 tok/s); half precision reaches the
        // tensor-core tiers and the twelve-bit packing.
        if matches!(self.layout.dense, DenseFormat::Half | DenseFormat::Mxfp8) {
            let src = &AsRef::<[u8]>::as_ref(&*host)[start..end];
            let moved = match how {
                Move::InPlace => None,
                Move::Rows {
                    rows: r,
                    lead,
                    perm,
                } => Some(permute_blocks(src, r * row_bytes, lead, perm)),
                Move::Cols { cols, perm } => Some(permute_columns(src, row_bytes, cols * 2, perm)),
            };
            let halves = brain_to_half(moved.as_deref().unwrap_or(src));
            let rows_out = rows.end - rows.start;
            if self.layout.pack
                && let Some(packed) = Weight::packed_half(self.backend, in_dim, rows_out, &halves)?
            {
                return Ok(packed);
            }
            return Ok(Weight::over(
                DType::F16,
                in_dim,
                rows.end - rows.start,
                upload(self.backend, halves)?,
            ));
        }
        let buffer = match how {
            Move::InPlace => import_host(self.backend, host, start..end)?,
            Move::Rows {
                rows: r,
                lead,
                perm,
            } => {
                let src = &AsRef::<[u8]>::as_ref(&*host)[start..end];
                upload(self.backend, permute_blocks(src, r * row_bytes, lead, perm))?
            }
            Move::Cols { cols, perm } => {
                let src = &AsRef::<[u8]>::as_ref(&*host)[start..end];
                upload(
                    self.backend,
                    permute_columns(src, row_bytes, cols * 2, perm),
                )?
            }
        };
        Ok(Weight::over(
            DType::BF16,
            in_dim,
            rows.end - rows.start,
            buffer,
        ))
    }

    /// An MXFP8 matrix `[out, in]` with its `ue8m0` scales `[out, in/32]`,
    /// optionally a row range, optionally moved, kept as stored.
    ///
    /// A move applies to both arrays at once, because a row of scales belongs
    /// to its row of codes: a row move carries `in/32` scale bytes with each
    /// `in` code bytes. A column move is only exact when its blocks are whole
    /// scale groups, which the recurrent output projection's 128-wide head
    /// blocks are; anything narrower is refused rather than regrouped.
    fn mxfp8(
        &self,
        name: &str,
        stored_rows: u32,
        in_dim: u32,
        rows: std::ops::Range<u32>,
        how: Move<'_>,
    ) -> CoreResult<Weight> {
        use shiso_modelfile::mxfp8::GROUP;
        let groups = in_dim as usize / GROUP;
        if !(in_dim as usize).is_multiple_of(GROUP) {
            return Err(CoreError::InvalidValue {
                key: "tensor layout",
                value: format!("{name}: MXFP8 contraction {in_dim} is not whole groups of 32"),
            });
        }
        let scale_name = format!(
            "{}.weight_scale",
            name.strip_suffix(".weight").unwrap_or(name)
        );
        let (code_slot, code_span) = self.ck.span(
            name,
            "F8_E4M3",
            &[u64::from(stored_rows), u64::from(in_dim)],
        )?;
        let (scale_slot, scale_span) =
            self.ck
                .span(&scale_name, "U8", &[u64::from(stored_rows), groups as u64])?;
        let pick = |slot: usize, span: std::ops::Range<usize>, row_bytes: usize| {
            let host = self.ck.shard(slot);
            let start = span.start + rows.start as usize * row_bytes;
            let end = span.start + rows.end as usize * row_bytes;
            (host, start..end)
        };
        let (code_host, code_range) = pick(code_slot, code_span, in_dim as usize);
        let (scale_host, scale_range) = pick(scale_slot, scale_span, groups);
        let moved = |src: &[u8], row_bytes: usize, per_value: usize| {
            mxfp8_moved(name, src, row_bytes, per_value, how)
        };
        let codes_src = &AsRef::<[u8]>::as_ref(&*code_host)[code_range.clone()];
        let scales_src = &AsRef::<[u8]>::as_ref(&*scale_host)[scale_range.clone()];
        // The kernels read codes sixteen bytes at a time, and a safetensors
        // payload sits wherever the header put it: a span that is not on a
        // sixteen-byte boundary is copied rather than wrapped.
        let codes = match moved(codes_src, in_dim as usize, 1)? {
            Some(bytes) => upload(self.backend, bytes)?,
            None if codes_src.as_ptr().align_offset(16) != 0 => {
                upload(self.backend, codes_src.to_vec())?
            }
            None => import_host(self.backend, code_host.clone(), code_range)?,
        };
        let scales = match moved(scales_src, groups, GROUP)? {
            Some(bytes) => upload(self.backend, bytes)?,
            None => import_host(self.backend, scale_host.clone(), scale_range)?,
        };
        Weight::mxfp8(in_dim, rows.end - rows.start, [codes, scales])
    }

    /// A whole brain-float matrix, as stored.
    fn matrix(&self, name: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        self.dense(name, out_dim, in_dim, 0..out_dim, Move::InPlace)
    }

    /// The vocabulary table, which a row gather reads rather than a
    /// projection: kept as brain floats whatever the dense format, since the
    /// gather has a brain-float entry and widening would buy it nothing.
    fn table(&self, name: &str) -> CoreResult<Weight> {
        let shape = [u64::from(VOCAB), u64::from(HIDDEN)];
        let (slot, span) = self.ck.span(name, "BF16", &shape)?;
        let buffer = import_host(self.backend, self.ck.shard(slot), span)?;
        Ok(Weight::over(DType::BF16, HIDDEN, VOCAB, buffer))
    }

    /// A single-precision vector from a brain-float tensor, through `op`,
    /// its blocks of `block` values reordered past the first `lead` when
    /// `perm` is given.
    fn vector(
        &self,
        name: &str,
        shape: &[u64],
        op: impl Fn(f32) -> f32,
        perm: Option<(usize, usize, &[usize])>,
    ) -> CoreResult<DeviceBuffer> {
        let values: Vec<f32> = self.ck.floats(name, shape)?.into_iter().map(op).collect();
        let bytes = f32_bytes(&values);
        let bytes = match perm {
            Some((block, lead, perm)) => permute_blocks(&bytes, block * 4, lead, perm),
            None => bytes,
        };
        upload(self.backend, bytes)
    }

    fn mix(&self, base: &str, inject: bool) -> CoreResult<StreamMix> {
        let wide = HIDDEN * STREAMS;
        Ok(StreamMix {
            norm: self.vector(
                &format!("{base}.hc_norm.weight"),
                &[u64::from(wide)],
                plus_one,
                None,
            )?,
            down: self.matrix(
                &format!("{base}.input_mix_weight_down.weight"),
                wide,
                STREAM_RANK,
            )?,
            up: self.matrix(
                &format!("{base}.input_mix_weight_up.weight"),
                STREAM_RANK,
                wide,
            )?,
            inject: inject
                .then(|| self.matrix(&format!("{base}.block_inject_weight.weight"), wide, STREAMS))
                .transpose()?,
        })
    }

    fn attn(&self, base: &str) -> CoreResult<SparseAttn> {
        let a = format!("{base}.self_attn");
        let kv_rows = KV_HEADS * HEAD_DIM;
        let idx_q = INDEXER_HEADS * INDEXER_HEAD_DIM;
        let idx_rows = idx_q + INDEXER_HEAD_DIM;
        let idx = format!("{a}.indexer.index_qk_proj.weight");
        let norm = |name: &str, width: u32| {
            self.vector(
                &format!("{a}.{name}.weight"),
                &[u64::from(width)],
                plus_one,
                None,
            )
        };
        Ok(SparseAttn {
            q: self.matrix(
                &format!("{a}.q_proj.weight"),
                HIDDEN,
                Q_HEADS * HEAD_DIM * 2,
            )?,
            k: self.matrix(&format!("{a}.k_proj.weight"), HIDDEN, kv_rows)?,
            v: self.matrix(&format!("{a}.v_proj.weight"), HIDDEN, kv_rows)?,
            output: self.matrix(&format!("{a}.o_proj.weight"), Q_HEADS * HEAD_DIM, HIDDEN)?,
            q_norm: norm("q_norm", HEAD_DIM)?,
            k_norm: norm("k_norm", HEAD_DIM)?,
            indexer: Indexer {
                q: self.dense(&idx, idx_rows, HIDDEN, 0..idx_q, Move::InPlace)?,
                k: self.dense(&idx, idx_rows, HIDDEN, idx_q..idx_rows, Move::InPlace)?,
                q_norm: norm("indexer.q_layernorm", INDEXER_HEAD_DIM)?,
                k_norm: norm("indexer.k_layernorm", INDEXER_HEAD_DIM)?,
            },
        })
    }

    fn recurrent(&self, base: &str) -> CoreResult<Recurrent> {
        let la = format!("{base}.linear_attn");
        let heads = value_heads();
        let hd = GDN_HEAD_DIM as usize;
        let qk_blocks = 2 * GDN_QK_HEADS as usize;
        let v_rows = GDN_V_HEADS * GDN_HEAD_DIM;
        let qkv_rows = 2 * GDN_QK_HEADS * GDN_HEAD_DIM + v_rows;
        let n_v = u64::from(GDN_V_HEADS);
        Ok(Recurrent {
            qkv: self.dense(
                &format!("{la}.in_proj_qkv.weight"),
                qkv_rows,
                HIDDEN,
                0..qkv_rows,
                Move::Rows {
                    rows: hd,
                    lead: qk_blocks,
                    perm: &heads,
                },
            )?,
            gate: self.dense(
                &format!("{la}.in_proj_z.weight"),
                v_rows,
                HIDDEN,
                0..v_rows,
                Move::Rows {
                    rows: hd,
                    lead: 0,
                    perm: &heads,
                },
            )?,
            alpha: self.dense(
                &format!("{la}.in_proj_a.weight"),
                GDN_V_HEADS,
                HIDDEN,
                0..GDN_V_HEADS,
                Move::Rows {
                    rows: 1,
                    lead: 0,
                    perm: &heads,
                },
            )?,
            beta: self.dense(
                &format!("{la}.in_proj_b.weight"),
                GDN_V_HEADS,
                HIDDEN,
                0..GDN_V_HEADS,
                Move::Rows {
                    rows: 1,
                    lead: 0,
                    perm: &heads,
                },
            )?,
            out: self.dense(
                &format!("{la}.out_proj.weight"),
                HIDDEN,
                v_rows,
                0..HIDDEN,
                Move::Cols {
                    cols: hd,
                    perm: &heads,
                },
            )?,
            // `[channels, 1, kernel]`: a channel's taps are contiguous, which
            // is the flat order the kernels read.
            conv1d: self.vector(
                &format!("{la}.conv1d.weight"),
                &[u64::from(qkv_rows), 1, u64::from(cfg::GDN_CONV)],
                std::convert::identity,
                Some((hd * cfg::GDN_CONV as usize, qk_blocks, &heads)),
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

    fn experts(&self, base: &str) -> CoreResult<ExpertBank> {
        let m = format!("{base}.mlp");
        let streamed = |in_dim: u32, out_dim: u32| -> CoreResult<Weight> {
            Ok(Weight::streamed(
                self.layout.experts,
                in_dim,
                out_dim,
                EXPERTS,
                shiso_core::buffer::alloc(self.backend, 16)?,
            ))
        };
        Ok(ExpertBank {
            router: self.matrix(&format!("{m}.gate.weight"), HIDDEN, EXPERTS)?,
            gate: streamed(HIDDEN, EXPERT_INTERMEDIATE)?,
            up: streamed(HIDDEN, EXPERT_INTERMEDIATE)?,
            down: streamed(EXPERT_INTERMEDIATE, HIDDEN)?,
            shared_router: self.matrix(&format!("{m}.shared_expert_gate.weight"), HIDDEN, 1)?,
            shared_gate: self.matrix(
                &format!("{m}.shared_expert.gate_proj.weight"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
            )?,
            shared_up: self.matrix(
                &format!("{m}.shared_expert.up_proj.weight"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
            )?,
            shared_down: self.matrix(
                &format!("{m}.shared_expert.down_proj.weight"),
                EXPERT_INTERMEDIATE,
                HIDDEN,
            )?,
        })
    }

    /// Every expert of one projection as one resident block-scaled bank:
    /// expert `e`'s codes and block scales at slot `e`, which is the layout
    /// the identity expert map addresses.
    fn resident_fp8(
        &self,
        base: &str,
        proj: &str,
        in_dim: u32,
        out_dim: u32,
    ) -> CoreResult<Weight> {
        let (one_codes, one_scales) = fp8_block_bytes(in_dim, out_dim);
        let mut codes =
            Vec::with_capacity(usize::try_from(one_codes * u64::from(EXPERTS)).unwrap_or(0));
        let mut scales =
            Vec::with_capacity(usize::try_from(one_scales * u64::from(EXPERTS)).unwrap_or(0));
        let code_shape = [u64::from(out_dim), u64::from(in_dim)];
        let scale_shape = [
            u64::from(out_dim.div_ceil(128)),
            u64::from(in_dim.div_ceil(128)),
        ];
        for e in 0..EXPERTS {
            let at = format!("{base}.mlp.experts.{e}.{proj}");
            for (name, dtype, shape, out) in [
                (format!("{at}.weight"), "F8_E4M3", &code_shape, &mut codes),
                (
                    format!("{at}.weight_scale_inv"),
                    "BF16",
                    &scale_shape,
                    &mut scales,
                ),
            ] {
                let (slot, span) = self.ck.span(&name, dtype, shape)?;
                out.extend_from_slice(&AsRef::<[u8]>::as_ref(&*self.ck.shard(slot))[span]);
            }
        }
        Weight::block_scaled(
            in_dim,
            out_dim,
            EXPERTS,
            [upload(self.backend, codes)?, upload(self.backend, scales)?],
        )
    }

    /// Every expert of one projection as one resident NVFP4 bank: expert
    /// `e`'s codes, group scales and global at slot `e`, the layout the
    /// identity expert map addresses.
    fn resident_nvfp4(
        &self,
        base: &str,
        proj: &str,
        in_dim: u32,
        out_dim: u32,
    ) -> CoreResult<Weight> {
        let (one_codes, one_scales) = nvfp4_bytes(in_dim, out_dim);
        let n = u64::from(EXPERTS);
        let mut codes = Vec::with_capacity(usize::try_from(one_codes * n).unwrap_or(0));
        let mut scales = Vec::with_capacity(usize::try_from(one_scales * n).unwrap_or(0));
        let mut globals = Vec::with_capacity(EXPERTS as usize * 4);
        let code_shape = [u64::from(out_dim), u64::from(in_dim / 2)];
        let scale_shape = [u64::from(out_dim), u64::from(in_dim / 16)];
        for e in 0..EXPERTS {
            let at = format!("{base}.mlp.experts.{e}.{proj}");
            for (name, dtype, shape, out) in [
                (format!("{at}.weight"), "U8", &code_shape[..], &mut codes),
                (
                    format!("{at}.weight_scale"),
                    "F8_E4M3",
                    &scale_shape[..],
                    &mut scales,
                ),
                (format!("{at}.weight_scale_2"), "F32", &[][..], &mut globals),
            ] {
                let (slot, span) = self.ck.span(&name, dtype, shape)?;
                out.extend_from_slice(&AsRef::<[u8]>::as_ref(&*self.ck.shard(slot))[span]);
            }
        }
        Weight::nvfp4(
            in_dim,
            out_dim,
            EXPERTS,
            [
                upload(self.backend, codes)?,
                upload(self.backend, scales)?,
                upload(self.backend, globals)?,
            ],
        )
    }

    /// The draft head's routed feed-forward: block-scaled FP8 banks, resident
    /// because a draft pass never fetches, beside the dense router and shared
    /// expert.
    fn draft_experts(&self, base: &str) -> CoreResult<ExpertBank> {
        let m = format!("{base}.mlp");
        // Which codec the head's banks are in is a fact of the stored bytes:
        // the reference and first NVFP4 exports ship block-scaled FP8 here,
        // the mixed-precision one NVFP4.
        let probe = format!("{base}.mlp.experts.0.gate_proj.weight");
        let resident = |proj: &str, in_dim: u32, out_dim: u32| {
            if self.ck.raw(&probe)?.1.dtype == "U8" {
                self.resident_nvfp4(base, proj, in_dim, out_dim)
            } else {
                self.resident_fp8(base, proj, in_dim, out_dim)
            }
        };
        Ok(ExpertBank {
            router: self.matrix(&format!("{m}.gate.weight"), HIDDEN, EXPERTS)?,
            gate: resident("gate_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            up: resident("up_proj", HIDDEN, EXPERT_INTERMEDIATE)?,
            down: resident("down_proj", EXPERT_INTERMEDIATE, HIDDEN)?,
            shared_router: self.matrix(&format!("{m}.shared_expert_gate.weight"), HIDDEN, 1)?,
            shared_gate: self.matrix(
                &format!("{m}.shared_expert.gate_proj.weight"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
            )?,
            shared_up: self.matrix(
                &format!("{m}.shared_expert.up_proj.weight"),
                HIDDEN,
                EXPERT_INTERMEDIATE,
            )?,
            shared_down: self.matrix(
                &format!("{m}.shared_expert.down_proj.weight"),
                EXPERT_INTERMEDIATE,
                HIDDEN,
            )?,
        })
    }

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
            key: self.matrix(&format!("{p}.key_proj.weight"), HIDDEN, wide)?,
            value: self.matrix(&format!("{p}.value_proj.weight"), HIDDEN, HIDDEN)?,
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

/// Imports every dense weight of the export; the routed banks are described
/// and left to the stream. `layout` says what the dense matrices become and
/// which codec the banks are described as.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first tensor the directory does not
/// carry, [`CoreError::InvalidValue`] for one of another layout, and whatever
/// the device reports.
pub fn import(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    config: &Qwen4ExpConfig,
    layout: ImportLayout,
) -> CoreResult<SparseWeights> {
    let loader = Loader {
        ck,
        backend,
        layout,
    };
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
        loader.table(&format!("{LM}.embed_tokens.weight"))?
    };
    let _tag = ledger::scope(MemTag::WeightsHead);
    // The head stays unpacked: a draft's narrowed head is cut out of it by
    // rows, which the twelve-bit layout's per-row exception lists do not allow.
    let head_loader = Loader {
        layout: ImportLayout {
            pack: false,
            ..layout
        },
        ..loader
    };
    let lm_head = head_loader.matrix("lm_head.weight", HIDDEN, VOCAB)?;
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

/// Prefix of every draft-head tensor.
const MTP: &str = "mtp";

/// Imports the draft head an export carries under `mtp.*`: one attention
/// layer with its own stream mixes, a routed feed-forward over block-scaled
/// FP8 experts held resident, and the two input projections and norms that
/// fold the target's hidden state into it.
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
    layout: ImportLayout,
) -> CoreResult<DraftHead> {
    let loader = Loader {
        ck,
        backend,
        layout,
    };
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
        fc_embed: loader.matrix(&format!("{MTP}.fc_embedding.weight"), HIDDEN, HIDDEN)?,
        fc_hidden: loader.matrix(&format!("{MTP}.fc_hidden.weight"), HIDDEN, HIDDEN)?,
        layer: SparseLayer {
            attn_mix: loader.mix(&format!("{layer}.attn_hyper_connection"), true)?,
            ffn_mix: loader.mix(&format!("{layer}.mlp_hyper_connection"), true)?,
            block: SparseBlock::Attn(loader.attn(&layer)?),
            experts: loader.draft_experts(&layer)?,
            ngram: None,
            ordinal: rank,
        },
        output_mix: loader.mix(&format!("{MTP}.hyper_connection_mixer"), false)?,
        head: None,
    })
}

/// The three projections of one expert, each as codes then block scales.
const PROJECTIONS: [(&str, u32, u32); 3] = [
    ("gate_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("up_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("down_proj", EXPERT_INTERMEDIATE, HIDDEN),
];

/// Bytes one expert occupies across its six arrays.
#[must_use]
pub fn slot_bytes() -> u64 {
    PROJECTIONS
        .iter()
        .map(|(_, i, o)| {
            let (codes, scales) = fp8_block_bytes(*i, *o);
            codes + scales
        })
        .sum()
}

/// Bytes of the routed banks, every layer.
#[must_use]
pub fn expert_bytes() -> u64 {
    slot_bytes() * u64::from(EXPERTS) * u64::from(N_LAYERS)
}

/// The routed banks of the export, one tensor per expert and array.
#[derive(Debug)]
pub struct Fp8Experts {
    /// `(file, offset)` per layer, expert and array, flattened in that order.
    arrays: Vec<(Arc<File>, u64)>,
    layout: Vec<u64>,
}

impl Fp8Experts {
    /// Locates every expert's six arrays, reading through handles that bypass
    /// the page cache.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming a missing tensor, and
    /// [`CoreError::InvalidValue`] for one of another layout.
    pub fn new(ck: &MlxCheckpoint) -> CoreResult<Self> {
        let uncached = ck
            .shard_paths()
            .iter()
            .map(|path| {
                shiso_modelfile::st_header::open_uncached(path)
                    .map(Arc::new)
                    .map_err(CoreError::Io)
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let mut layout = Vec::with_capacity(6);
        for (_, in_dim, out_dim) in PROJECTIONS {
            let (codes, scales) = fp8_block_bytes(in_dim, out_dim);
            layout.push(codes);
            layout.push(scales);
        }
        let mut arrays = Vec::with_capacity((N_LAYERS * EXPERTS) as usize * 6);
        for il in 0..N_LAYERS {
            for e in 0..EXPERTS {
                let base = format!("{LM}.layers.{il}.mlp.experts.{e}");
                for (proj, in_dim, out_dim) in PROJECTIONS {
                    let codes = format!("{base}.{proj}.weight");
                    let scales = format!("{base}.{proj}.weight_scale_inv");
                    ck.span(&codes, "F8_E4M3", &[u64::from(out_dim), u64::from(in_dim)])?;
                    ck.span(
                        &scales,
                        "BF16",
                        &[
                            u64::from(out_dim.div_ceil(128)),
                            u64::from(in_dim.div_ceil(128)),
                        ],
                    )?;
                    for name in [codes, scales] {
                        let (slot, offset) = ck.shard_offset(&name)?;
                        arrays.push((Arc::clone(&uncached[slot]), offset));
                    }
                }
            }
        }
        Ok(Self { arrays, layout })
    }
}

impl ExpertSource for Fp8Experts {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole, WeightPart::BlockScales]
    }

    fn layout(&self) -> Vec<u64> {
        self.layout.clone()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        use std::os::unix::fs::FileExt;
        let at = (layer as usize * EXPERTS as usize + expert as usize) * 6 + array;
        let (file, offset) = self.arrays.get(at).ok_or(CoreError::OutOfBounds {
            offset: at as u64,
            len: 1,
            buffer_bytes: self.arrays.len() as u64,
        })?;
        file.read_exact_at(dst, *offset).map_err(CoreError::Io)
    }
}

/// Opens the stream over the export's routed banks; a pool that holds every
/// expert is filled at once rather than on first use.
///
/// # Errors
///
/// Whatever [`Fp8Experts::new`], [`ExpertStream::open`] and the fill report.
pub fn open_stream(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let source = Arc::new(Fp8Experts::new(ck)?);
    let min_slots = widest_pass.saturating_mul(EXPERTS_USED).min(EXPERTS);
    let stream = ExpertStream::open(backend, source, N_LAYERS, EXPERTS, slots, min_slots)?;
    if slots >= EXPERTS * N_LAYERS {
        stream.preload(backend.as_ref())?;
    }
    Ok(stream)
}

/// Opens the FP8 n-gram table: 128 shards of one array each, one scale for
/// the whole table, which goes to the device as the decode's one float.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for a missing shard or scale, and whatever the
/// reader refuses.
pub fn open_ngram(
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
    let scale = ck
        .floats(&format!("{base}.weight_scale"), &[1])?
        .first()
        .copied()
        .ok_or_else(|| CoreError::ConfigMissing {
            key: format!("{base}.weight_scale"),
        })?;
    let row_type = PleRowType::Fp8E4M3 {
        scale_bits: scale.to_bits(),
    };
    let params = config.ngram().params(ple, row_type, None)?;
    let mut shards = Vec::with_capacity(NGRAM_SHARDS);
    let mut rows_per_shard = 0u64;
    for s in 0..NGRAM_SHARDS {
        let name = format!("{base}.shard_{s}.weight");
        let (_, raw) = ck.raw(&name)?;
        let rows = raw.shape.first().copied().unwrap_or(0);
        if raw.dtype != "F8_E4M3" || raw.shape.get(1).copied() != Some(u64::from(PLE_HEAD_DIM)) {
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
        shards.push(vec![PleSegment {
            file,
            offset,
            row_bytes: PLE_HEAD_DIM,
        }]);
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
    let scale_buffer = upload(backend, scale.to_le_bytes().to_vec())?;
    NgramTable::new(DType::Fp8E4M3, Arc::new(stream), Some(scale_buffer))
}

/// Opens the reference export's directory as a checkpoint.
///
/// # Errors
///
/// As [`MlxCheckpoint::open_reference`].
pub fn open(dir: &Path) -> CoreResult<MlxCheckpoint> {
    MlxCheckpoint::open_reference(dir)
}
