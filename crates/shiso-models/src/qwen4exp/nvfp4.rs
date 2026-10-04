//! NVIDIA's `modelopt` export of this family: the reference export's layout
//! and names, with the routed experts in NVFP4 rather than block-scaled FP8.
//!
//! Everything but the routed banks is [`super::fp8`]'s: brain-float dense
//! weights under the same names and moves, the norms stored uncentred, and the
//! FP8 n-gram table in 128 shards under one scale (here in a file of its own
//! beside the main shards, which the index names like any other). The dense
//! matrices are widened to half precision on the way in
//! ([`super::fp8::DenseFormat::Half`]), which puts them on the tiers every
//! backend has a fast path for.
//!
//! The routed experts are 71 GB: per expert and projection, packed four-bit
//! codes, one `e4m3` scale per sixteen weights, and one `f32` above both
//! (`weight_scale_2`, which differs per expert). They go through an
//! [`ExpertStream`] exactly as the FP8 banks do, the global scale riding in
//! each slot as a third array, so a fetched expert carries its own.
//!
//! The draft head this export ships (`mtp.*`: brain-float dense weights in the
//! main shards, block-scaled FP8 experts in the side file) is the reference
//! export's head and is read by [`super::fp8::import_draft`], its experts
//! resident.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use shiso_core::backend::Backend;
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::stream::{ExpertSource, ExpertStream};
use shiso_engine::sparse::weights::{WeightPart, nvfp4_bytes};

use crate::qwen4exp::config::{EXPERT_INTERMEDIATE, EXPERTS, EXPERTS_USED, HIDDEN, N_LAYERS};
use crate::qwen4exp::mlx::MlxCheckpoint;

/// Prefix of every language-model tensor of this export.
const LM: &str = "model.language_model";

/// The document `modelopt` writes beside the shards.
const QUANT_DOCUMENT: &str = "hf_quant_config.json";

/// Arrays one projection of one expert is stored as: codes, group scales,
/// global scale.
const PARTS: [WeightPart; 3] = [
    WeightPart::Whole,
    WeightPart::GroupScales,
    WeightPart::GlobalScale,
];

/// Bytes of the one `f32` global scale.
const GLOBAL_BYTES: u64 = 4;

/// The three projections of one expert.
const PROJECTIONS: [(&str, u32, u32); 3] = [
    ("gate_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("up_proj", HIDDEN, EXPERT_INTERMEDIATE),
    ("down_proj", EXPERT_INTERMEDIATE, HIDDEN),
];

/// Whether `dir` is the `modelopt` export: its quantization document says the
/// routed experts are NVFP4.
///
/// Asked before the reference export's own test, which the directory also
/// passes - the two share their names and differ only in the experts' codec.
#[must_use]
pub fn is_nvfp4_export(dir: &Path) -> bool {
    #[derive(Deserialize)]
    struct Target {
        quant_algo: String,
    }
    #[derive(Deserialize)]
    struct Quantization {
        #[serde(default)]
        quantized_layers: std::collections::HashMap<String, Target>,
    }
    #[derive(Deserialize)]
    struct Doc {
        quantization: Quantization,
    }
    std::fs::read_to_string(dir.join(QUANT_DOCUMENT))
        .ok()
        .and_then(|text| serde_json::from_str::<Doc>(&text).ok())
        .is_some_and(|doc| {
            doc.quantization
                .quantized_layers
                .get(&format!("{LM}.layers.0.mlp.experts"))
                .is_some_and(|target| target.quant_algo == "NVFP4")
        })
}

/// Opens the export's directory: the language model, the output projection,
/// and the draft head, which the reference export's opener leaves out.
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

/// Bytes one expert occupies across its nine arrays.
#[must_use]
pub fn slot_bytes() -> u64 {
    PROJECTIONS
        .iter()
        .map(|(_, i, o)| {
            let (codes, scales) = nvfp4_bytes(*i, *o);
            codes + scales + GLOBAL_BYTES
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
pub struct Nvfp4Experts {
    /// `(file, offset)` per layer, expert and array, flattened in that order.
    arrays: Vec<(Arc<File>, u64)>,
    layout: Vec<u64>,
}

impl Nvfp4Experts {
    /// Locates every expert's nine arrays, reading through handles that
    /// bypass the page cache, and checks each against the layout the widths
    /// imply.
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
        let mut arrays = Vec::with_capacity((N_LAYERS * EXPERTS) as usize * 9);
        for il in 0..N_LAYERS {
            for e in 0..EXPERTS {
                let base = format!("{LM}.layers.{il}.mlp.experts.{e}");
                for (proj, in_dim, out_dim) in PROJECTIONS {
                    let (rows, cols) = (u64::from(out_dim), u64::from(in_dim));
                    let codes = format!("{base}.{proj}.weight");
                    let scales = format!("{base}.{proj}.weight_scale");
                    let global = format!("{base}.{proj}.weight_scale_2");
                    ck.span(&codes, "U8", &[rows, cols / 2])?;
                    ck.span(&scales, "F8_E4M3", &[rows, cols / 16])?;
                    ck.span(&global, "F32", &[])?;
                    for name in [codes, scales, global] {
                        let (slot, offset) = ck.shard_offset(&name)?;
                        arrays.push((Arc::clone(&uncached[slot]), offset));
                    }
                }
            }
        }
        Ok(Self { arrays, layout })
    }
}

impl ExpertSource for Nvfp4Experts {
    fn parts(&self) -> &'static [WeightPart] {
        &PARTS
    }

    fn layout(&self) -> Vec<u64> {
        self.layout.clone()
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        use std::os::unix::fs::FileExt;
        let at = (layer as usize * EXPERTS as usize + expert as usize) * 9 + array;
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
/// Whatever [`Nvfp4Experts::new`], [`ExpertStream::open`] and the fill report.
pub fn open_stream(
    backend: &Arc<dyn Backend>,
    ck: &MlxCheckpoint,
    slots: u32,
    widest_pass: u32,
) -> CoreResult<ExpertStream> {
    let source = Arc::new(Nvfp4Experts::new(ck)?);
    let min_slots = widest_pass.saturating_mul(EXPERTS_USED).min(EXPERTS);
    let stream = ExpertStream::open(backend, source, N_LAYERS, EXPERTS, slots, min_slots)?;
    if slots >= EXPERTS * N_LAYERS {
        stream.preload(backend.as_ref())?;
    }
    Ok(stream)
}
