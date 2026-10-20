//! NVIDIA's mixed-precision `modelopt` export of this family: NVFP4 routed
//! experts with CSF-compressed scales, MXFP8 projections, an NVFP4 n-gram
//! table and an NVFP4 draft head.
//!
//! The tensors carry the reference export's names and the forward pass is the
//! one [`super::fp8`] imports; what differs is how each kind of matrix is
//! stored, which the checkpoint's own `config.json` states per layer under
//! `quantization_config.quantized_layers`:
//!
//! | what | stored as | served as |
//! |---|---|---|
//! | routed experts (68 GB) | NVFP4, group scales in CSF, one `f32` per expert | NVFP4: CSF expanded at load into the stream's plain scale slots |
//! | attention, indexer, recurrent and shared-expert projections | MXFP8 (`e4m3`, one `ue8m0` per 32) | MXFP8, as stored |
//! | output head, routers, stream mixes, n-gram projections, draft head's dense | BF16 | BF16 |
//! | stream-mix and n-gram norms | F32 (100 of them) | F32 |
//! | n-gram table | NVFP4 rows, 128 shards, one global | NVFP4 rows from SSD |
//! | draft head's experts | NVFP4 with plain scales | NVFP4, resident |
//!
//! # Why the CSF scales are expanded here and not on the device
//!
//! The compressed form is about half the bytes of the plain one (about 2 GB
//! over the routed banks), and the reference keeps it on the device and
//! expands a layer's scales into scratch on every call. This tree's NVFP4
//! kernels read plain `e4m3` scales out of the expert stream's slots, so the
//! expansion happens once per fetched expert as the stream fills a slot - on
//! a pool that holds every expert, once at load - and no kernel changes. The
//! expansion is lossless ([`shiso_modelfile::csf`]).
//!
//! # Detection
//!
//! There is no `hf_quant_config.json` beside the shards, which is what the
//! earlier NVFP4 export is recognised by; this one says what it is in
//! `config.json`. [`is_export`] reads that, and is asked first, because the
//! directory also passes the reference export's test.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::stream::{ExpertSource, ExpertStream};
use shiso_engine::sparse::weights::{WeightPart, nvfp4_bytes};
use shiso_modelfile::csf::{CsfLayout, decode_csf};

use crate::qwen4exp::config::{EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_USED, HIDDEN, N_LAYERS};
use crate::qwen4exp::mlx::MlxCheckpoint;

/// Prefix of every language-model tensor of this export.
const LM: &str = "model.language_model";

/// The quantization entry every routed bank of this export carries.
const EXPERT_ALGO: &str = "NVFP4";

/// The scale encoding that marks the routed banks as CSF.
const CSF_ENCODING: &str = "csf";

/// Arrays one projection of one expert occupies in a stream slot.
const PARTS: [WeightPart; 3] = [
    WeightPart::Whole,
    WeightPart::GroupScales,
    WeightPart::GlobalScale,
];

/// Bytes of the one `f32` global scale.
const GLOBAL_BYTES: u64 = 4;

/// Bytes of one CSF exception word.
const EXCEPTION_BYTES: u64 = 4;

/// The three projections of one expert.
const PROJECTIONS: [(&str, u32, u32); 3] = [
    ("gate_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("up_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("down_proj", EXPERT_INTERMEDIATE, HIDDEN),
];

/// Whether `dir` is this export: `config.json` names `modelopt` mixed
/// precision, and the first layer's routed bank is NVFP4 with CSF scales.
#[must_use]
pub fn is_export(dir: &Path) -> bool {
    #[derive(Deserialize)]
    struct Target {
        quant_algo: String,
        #[serde(default)]
        weight_scale_encoding: Option<String>,
    }
    #[derive(Deserialize)]
    struct Quantization {
        quant_method: String,
        quant_algo: String,
        #[serde(default)]
        quantized_layers: std::collections::HashMap<String, Target>,
    }
    #[derive(Deserialize)]
    struct Doc {
        quantization_config: Quantization,
    }
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Doc>(&text).ok())
        .is_some_and(|doc| {
            let q = doc.quantization_config;
            q.quant_method == "modelopt"
                && q.quant_algo == "MIXED_PRECISION"
                && q.quantized_layers
                    .get(&format!("{LM}.layers.0.mlp.experts"))
                    .is_some_and(|t| {
                        t.quant_algo == EXPERT_ALGO
                            && t.weight_scale_encoding.as_deref() == Some(CSF_ENCODING)
                    })
        })
}

/// Opens the export's directory: the language model, the output projection
/// and the draft head. The vision tower is opened by its own loader.
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

/// Bytes one expert occupies in a stream slot: codes, plain scales and the
/// global, per projection - the same as the uncompressed NVFP4 export, since
/// the slot holds the expanded scales.
#[must_use]
pub fn slot_bytes() -> u64 {
    crate::qwen4exp::nvfp4::slot_bytes()
}

/// Bytes of the routed banks, every layer, as the pool holds them.
#[must_use]
pub fn expert_bytes() -> u64 {
    slot_bytes() * u64::from(EXPERTS) * u64::from(N_LAYERS)
}

/// Where one projection of one expert lives in the shards.
#[derive(Debug)]
struct ProjectionArrays {
    /// The packed codes.
    codes: (Arc<File>, u64),
    /// The fixed CSF record array.
    fixed: (Arc<File>, u64),
    /// The CSF exception words, and how many bytes of them.
    exceptions: (Arc<File>, u64, u64),
    /// The `f32` global scale.
    global: (Arc<File>, u64),
    /// The matrix's geometry.
    layout: CsfLayout,
}

/// The routed banks of the export, read through handles that bypass the page
/// cache, with each expert's scales expanded from CSF as its slot is filled.
#[derive(Debug)]
pub struct CsfExperts {
    /// Per layer, expert and projection, flattened in that order.
    arrays: Vec<ProjectionArrays>,
    /// Bytes of each of the nine arrays a slot holds.
    layout: Vec<u64>,
}

impl CsfExperts {
    /// Locates every expert's arrays and checks each against the layout the
    /// widths imply.
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
        let mut layout = Vec::with_capacity(9);
        for (_, in_dim, out_dim) in PROJECTIONS {
            let (codes, scales) = nvfp4_bytes(in_dim, out_dim);
            layout.extend([codes, scales, GLOBAL_BYTES]);
        }
        let at = |name: &str| -> CoreResult<(Arc<File>, u64)> {
            let (slot, offset) = ck.shard_offset(name)?;
            Ok((Arc::clone(&uncached[slot]), offset))
        };
        let mut arrays = Vec::with_capacity((N_LAYERS * EXPERTS) as usize * 3);
        for il in 0..N_LAYERS {
            for e in 0..EXPERTS {
                let base = format!("{LM}.layers.{il}.mlp.experts.{e}");
                for (proj, in_dim, out_dim) in PROJECTIONS {
                    let csf = CsfLayout {
                        rows: out_dim as usize,
                        cols: in_dim as usize,
                    };
                    let p = format!("{base}.{proj}");
                    let codes = format!("{p}.weight");
                    let fixed = format!("{p}.weight_scale.nvfp4_csf_fixed");
                    let exceptions = format!("{p}.weight_scale.nvfp4_csf_exceptions");
                    let global = format!("{p}.weight_scale_2");
                    ck.span(&codes, "U8", &[u64::from(out_dim), u64::from(in_dim / 2)])?;
                    ck.span(
                        &fixed,
                        "U8",
                        &[u64::from(out_dim) / 16, csf.record_bytes() as u64],
                    )?;
                    let (_, raw) = ck.raw(&exceptions)?;
                    let words = match raw.shape.as_slice() {
                        [n] if raw.dtype == "U32" => *n,
                        _ => {
                            return Err(CoreError::InvalidValue {
                                key: "tensor layout",
                                value: format!(
                                    "{exceptions} is {} {:?}, the loader expects U32 [n]",
                                    raw.dtype, raw.shape
                                ),
                            });
                        }
                    };
                    ck.span(&global, "F32", &[])?;
                    let (ef, eo) = at(&exceptions)?;
                    arrays.push(ProjectionArrays {
                        codes: at(&codes)?,
                        fixed: at(&fixed)?,
                        exceptions: (ef, eo, words * EXCEPTION_BYTES),
                        global: at(&global)?,
                        layout: csf,
                    });
                }
            }
        }
        Ok(Self { arrays, layout })
    }
}

impl ExpertSource for CsfExperts {
    fn parts(&self) -> &'static [WeightPart] {
        &PARTS
    }

    fn layout(&self) -> Vec<u64> {
        self.layout.clone()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        use std::os::unix::fs::FileExt;
        let at = (layer as usize * EXPERTS as usize + expert as usize) * 3 + array / 3;
        let p = self.arrays.get(at).ok_or(CoreError::OutOfBounds {
            offset: at as u64,
            len: 1,
            buffer_bytes: self.arrays.len() as u64,
        })?;
        let read = |(file, offset): &(Arc<File>, u64), buf: &mut [u8]| {
            file.read_exact_at(buf, *offset).map_err(CoreError::Io)
        };
        match array % 3 {
            0 => read(&p.codes, dst),
            1 => {
                let mut fixed = vec![0u8; p.layout.fixed_bytes()];
                read(&p.fixed, &mut fixed)?;
                let (file, offset, len) = &p.exceptions;
                let mut exceptions = vec![0u8; usize::try_from(*len).unwrap_or(usize::MAX)];
                file.read_exact_at(&mut exceptions, *offset)
                    .map_err(CoreError::Io)?;
                decode_csf("routed expert scales", p.layout, &fixed, &exceptions, dst).map_err(
                    |err| CoreError::Backend {
                        backend: "qwen4exp",
                        op: "expand CSF expert scales",
                        source: Box::new(err),
                    },
                )
            }
            _ => read(&p.global, dst),
        }
    }
}

/// Opens the stream over the export's routed banks; a pool that holds every
/// expert is filled at once rather than on first use.
///
/// # Errors
///
/// Whatever [`CsfExperts::new`], [`ExpertStream::open`] and the fill report.
pub fn open_stream(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let source = Arc::new(CsfExperts::new(ck)?);
    let min_slots = widest_pass.saturating_mul(EXPERTS_USED).min(EXPERTS);
    let stream = ExpertStream::open(backend, source, N_LAYERS, EXPERTS, slots, min_slots)?;
    if slots >= EXPERTS * N_LAYERS {
        stream.preload(backend.as_ref())?;
    }
    Ok(stream)
}
