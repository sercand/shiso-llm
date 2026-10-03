//! The safetensors weight store one checkpoint component is read through.
//!
//! A component (transformer, text encoder, VAE) is a directory of shards or a
//! single file. This opens every shard of one component, indexes the union of
//! their tensors, and hands tensors to a device as borrowed imports: the shard
//! mapping is the host memory the device wraps, so a weight travels as
//! (mapping, byte range) and is never re-encoded, dequantized, or copied into
//! an intermediate buffer of ours.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use half::bf16;
use shiso_core::{Backend, DeviceBuffer, import_host};
use shiso_modelfile::st_header::{Shard, StDType};
use shiso_vision::generator::ImageGenError;

use super::ops::Ops;

/// One tensor as its shard header described it, resolved to a shard slot.
#[derive(Debug)]
pub(crate) struct StEntry {
    /// Declared storage format.
    pub dtype: StDType,
    /// Dimensions, slowest-varying axis first.
    pub shape: Vec<u64>,
    shard: usize,
    start: u64,
    end: u64,
}

impl StEntry {
    /// Elements the declared shape holds.
    pub(crate) fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Every tensor of one component's shards, addressable by name.
#[derive(Debug)]
pub(crate) struct StStore {
    shards: Vec<Arc<Shard>>,
    entries: HashMap<String, StEntry>,
}

impl StStore {
    /// Opens `path` (a `.safetensors` file, or a directory holding them) and
    /// indexes every tensor across every shard.
    ///
    /// Opening a directory is right only for a component home that holds
    /// exactly one export: the transformer's store goes through
    /// [`StStore::open_files`] with the set discovery selected, so a second
    /// export in the same directory cannot reach this path for the
    /// transformer at all.
    ///
    /// Every span the index carries is validated by the shard header parser
    /// (`shiso-modelfile::st_header::check_raw`): a declared shape whose
    /// product times the dtype size disagrees with the tensor's byte range,
    /// or a range that runs past the file, is refused at open — before any
    /// weight is imported — so no `Lin::load` below re-checks it.
    pub(crate) fn open(path: &Path) -> Result<Self, ImageGenError> {
        let files = if path.is_dir() {
            let mut found: Vec<_> = std::fs::read_dir(path)
                .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", path.display())))?
                .flatten()
                .map(|entry| entry.path())
                .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
            found.sort();
            found
        } else {
            vec![path.to_path_buf()]
        };
        Self::open_files(&files)
    }

    /// Opens exactly these shard files, refusing a tensor name that two of
    /// them carry.
    ///
    /// A multi-shard export splits tensors across files with disjoint names;
    /// a name in two files means two exports of the same tensors sit in one
    /// directory (a BF16 file beside an FP8 one), and indexing both would
    /// resolve every shared name to whichever file was read last — the wrong
    /// weights, silently. The refusal names the tensor and both files.
    pub(crate) fn open_files(files: &[PathBuf]) -> Result<Self, ImageGenError> {
        if files.is_empty() {
            return Err(ImageGenError::Checkpoint(
                "no safetensors weight file was selected".to_string(),
            ));
        }
        let mut shards = Vec::with_capacity(files.len());
        let mut entries: HashMap<String, StEntry> = HashMap::new();
        for file in files {
            let (shard, tensors) = Shard::open_untagged(file).map_err(|e| {
                ImageGenError::Checkpoint(format!("opening {}: {e}", file.display()))
            })?;
            let slot = shards.len();
            shards.push(Arc::new(shard));
            for (name, raw) in tensors {
                let dtype = raw
                    .parsed_dtype(&name)
                    .map_err(|e| ImageGenError::Checkpoint(format!("{name}: {e}")))?;
                let [start, end] = raw.data_offsets;
                if let Some(clash) = entries.get(&name) {
                    return Err(ImageGenError::Checkpoint(format!(
                        "tensor '{name}' is declared by both {} and {}; a weight set must \
                         hold either one export's shards (disjoint names) or a single \
                         export, not two exports side by side",
                        files[clash.shard].display(),
                        file.display()
                    )));
                }
                entries.insert(
                    name,
                    StEntry {
                        dtype,
                        shape: raw.shape,
                        shard: slot,
                        start,
                        end,
                    },
                );
            }
        }
        Ok(Self { shards, entries })
    }

    /// Looks a tensor up by its canonical name, also accepting the
    /// `transformer.`-prefixed spelling a combined export writes.
    pub(crate) fn get(&self, name: &str) -> Option<&StEntry> {
        self.entries
            .get(name)
            .or_else(|| self.entries.get(&format!("transformer.{name}")))
    }

    /// Requires a tensor by name; the refusal names it, because a missing
    /// weight is a checkpoint we cannot run and no default may stand in.
    pub(crate) fn require(&self, name: &str) -> Result<&StEntry, ImageGenError> {
        self.get(name).ok_or_else(|| {
            ImageGenError::Checkpoint(format!("checkpoint is missing required tensor '{name}'"))
        })
    }

    /// Whether a tensor exists under either accepted spelling.
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Every indexed tensor name, for block counting and dtype scans.
    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Payload bytes of one tensor, borrowed from the shard mapping.
    pub(crate) fn bytes(&self, name: &str) -> Result<&[u8], ImageGenError> {
        let entry = self.require(name)?;
        self.shards[entry.shard]
            .payload(entry.start, entry.end)
            .map_err(|e| ImageGenError::Checkpoint(format!("{name}: {e}")))
    }

    /// Imports one tensor's payload range straight onto `backend`.
    ///
    /// The `Arc<Shard>` is what keeps the mapping alive as long as the device
    /// can see it; the declared dtype travels beside the buffer because device
    /// memory is byte-counted and the format is a fact of the header.
    pub(crate) fn import(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let entry = self.require(name)?;
        let range = self.shards[entry.shard]
            .payload_range(entry.start, entry.end)
            .map_err(|e| ImageGenError::Checkpoint(format!("{name}: {e}")))?;
        let shard: Arc<Shard> = Arc::clone(&self.shards[entry.shard]);
        let host: Arc<dyn shiso_core::HostBytes> = shard;
        import_host(backend, host, range)
            .map_err(|e| ImageGenError::Backend(format!("importing '{name}': {e}")))
    }

    /// Reads a one-element `F32` tensor (a per-tensor weight scale) as a scalar.
    pub(crate) fn f32_scalar(&self, name: &str) -> Result<f32, ImageGenError> {
        let bytes = self.bytes(name)?;
        let raw: [u8; 4] = bytes.try_into().map_err(|_| {
            ImageGenError::Checkpoint(format!("'{name}' is not a 4-byte F32 scalar"))
        })?;
        Ok(f32::from_le_bytes(raw))
    }

    /// Reads a small tensor as `f32` values, widening `BF16` and `F16` exactly.
    ///
    /// This is for the family's short vectors - norms, biases, statistics -
    /// and, through [`StStore::f32_widen`], the f32-run VAE's kernels; the
    /// transformer's linear weights are imported whole in their stored format.
    pub(crate) fn f32_vec(&self, name: &str) -> Result<Vec<f32>, ImageGenError> {
        let entry = self.require(name)?;
        let bytes = self.bytes(name)?;
        let count = usize::try_from(entry.elements()).unwrap_or(usize::MAX);
        let values: Vec<f32> = match entry.dtype {
            StDType::F32 => {
                if bytes.len() != count * 4 {
                    return Err(ImageGenError::Checkpoint(format!(
                        "'{name}' payload does not match its F32 shape"
                    )));
                }
                bytes
                    .chunks_exact(4)
                    .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            }
            StDType::Bf16 => bytes
                .chunks_exact(2)
                .map(|w| bf16::from_le_bytes([w[0], w[1]]).to_f32())
                .collect(),
            StDType::F16 => bytes
                .chunks_exact(2)
                .map(|w| half::f16::from_le_bytes([w[0], w[1]]).to_f32())
                .collect(),
            other => {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{name}' has dtype {other:?}, which is not read as a float vector"
                )));
            }
        };
        if values.len() != count {
            return Err(ImageGenError::Checkpoint(format!(
                "'{name}' decoded {} values for a {count}-element shape",
                values.len()
            )));
        }
        Ok(values)
    }

    /// Reads a tensor as f32 and uploads it as an f32 buffer, widening
    /// `BF16` and `F16` exactly.
    ///
    /// This is the import for a component that runs in f32 out of a
    /// checkpoint that stores it in `BF16` (the VAE decoder): widening is
    /// lossless, so the zero-dequant rule - which protects *quantized*
    /// formats from being flattened into f32 - is not engaged. `F8_E4M3`
    /// still refuses here, because reading it as f32 would be a
    /// dequantization; those weights stay native in the transformer.
    pub(crate) fn f32_widen(
        &self,
        ops: &Ops,
        step: &'static str,
        name: &str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let values = self.f32_vec(name)?;
        ops.upload_f32(step, &values)
    }

    /// The per-tensor FP8 scale companion of `name`, or 1.0 when the tensor is
    /// not FP8 (the scale is only ever consulted by the FP8 GEMM path).
    pub(crate) fn weight_scale(&self, name: &str) -> Result<f32, ImageGenError> {
        for candidate in [
            format!("{name}_scale"),
            format!("{name}.weight_scale"),
            format!("{name}.scale"),
        ] {
            if self.contains(&candidate) {
                return self.f32_scalar(&candidate);
            }
        }
        Err(ImageGenError::Checkpoint(format!(
            "FP8 tensor '{name}' has no weight_scale companion ('{name}_scale' or '{name}.weight_scale')"
        )))
    }

    /// Counts `prefix.<n>.` blocks from the indexed names.
    pub(crate) fn count_blocks(&self, prefix: &str) -> Option<usize> {
        crate::flux2::config::count_blocks_internal(self.names(), prefix)
    }
}

/// A linear layer's weight in its stored native format, with its bias widened
/// to f32 and its FP8 scale read at load.
///
/// The weight buffer is an import of the shard mapping's byte range: BF16,
/// `F8_E4M3` and F32 weights all reach the device as the bytes the checkpoint
/// wrote, and the GEMM entry point is chosen from the header dtype. Only the
/// bias - a vector of at most a few tens of thousands of elements - is
/// widened, so the norm/bias rule and the zero-dequant rule for matrices
/// stay separate.
#[derive(Debug)]
pub(crate) struct Lin {
    /// Weight rows (`out_dim`, `in_dim`) in their stored format.
    pub w: DeviceBuffer,
    /// Stored format of `w`.
    pub kind: crate::flux2::ops::WeightKind,
    /// Epilogue multiplier (the FP8 tensor's `weight_scale`, else 1.0).
    pub scale: f32,
    /// Bias as f32, if the checkpoint carries one.
    pub bias: Option<DeviceBuffer>,
    /// Rows of the weight matrix.
    pub out_dim: usize,
    /// Columns of the weight matrix.
    pub in_dim: usize,
}

impl Lin {
    /// Loads `{name}.weight` (and `{name}.bias` when present) onto the ops'
    /// backend, refusing a dtype no GEMM entry point reads.
    pub(crate) fn load(store: &StStore, ops: &Ops, name: &str) -> Result<Self, ImageGenError> {
        let weight_name = format!("{name}.weight");
        let entry = store.require(&weight_name)?;
        let kind = match entry.dtype {
            StDType::Bf16 => crate::flux2::ops::WeightKind::Bf16,
            StDType::F8E4M3 => crate::flux2::ops::WeightKind::Fp8,
            StDType::F32 => crate::flux2::ops::WeightKind::F32,
            other => {
                return Err(ImageGenError::Checkpoint(format!(
                    "'{weight_name}' has dtype {other:?}, which no flux2 GEMM entry reads"
                )));
            }
        };
        let scale = if kind == crate::flux2::ops::WeightKind::Fp8 {
            store.weight_scale(&weight_name)?
        } else {
            1.0
        };
        let w = store.import(ops.backend(), &weight_name)?;
        let shape = &entry.shape;
        let (out_dim, in_dim) = (
            usize::try_from(*shape.first().ok_or_else(|| {
                ImageGenError::Checkpoint(format!("'{weight_name}' has empty shape"))
            })?)
            .unwrap_or(usize::MAX),
            usize::try_from(*shape.get(1).ok_or_else(|| {
                ImageGenError::Checkpoint(format!("'{weight_name}' is not a matrix"))
            })?)
            .unwrap_or(usize::MAX),
        );
        let bias_name = format!("{name}.bias");
        let bias = if store.contains(&bias_name) {
            let values = store.f32_vec(&bias_name)?;
            Some(ops.upload_f32("load bias", &values)?)
        } else {
            None
        };
        Ok(Self {
            w,
            kind,
            scale,
            bias,
            out_dim,
            in_dim,
        })
    }
}

/// Decodes one `F8_E4M3` byte to `f32` for the host fallback paths.
///
/// E4M3 with implicit leading one, bias 7, no infinities; 0x7F/0xFF is NaN in
/// the OCP spelling but never appears in a weight, and is mapped to NaN here
/// so a corrupt scale shows up as NaN rather than a plausible number.
pub(crate) fn f8e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0F;
    let frac = byte & 0x07;
    if exp == 0 {
        // Subnormals: 0.frac * 2^-6.
        return sign * f32::from(frac) * (1.0 / 64.0);
    }
    if exp == 0x0F && frac == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + f32::from(frac) / 8.0) * 2f32.powi(i32::from(exp) - 7)
}
