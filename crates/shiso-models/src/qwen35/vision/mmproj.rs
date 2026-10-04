//! The tower as a multimodal-projector GGUF: the `clip` container that ships
//! beside a quantized language model.
//!
//! It is the same tower the checkpoint shard and the standalone sidecar carry -
//! the same blocks, the same widths, the same numbers - written under a
//! different naming scheme and with three differences the manifest cannot be
//! asked to absorb, so this module resolves them before a tensor reaches the
//! shared upload loop:
//!
//! - **Names.** Every role is spelled differently (`v.blk.N.attn_qkv` for
//!   `blocks.N.attn.qkv`, `mm.0` for the merger's first projection, `v.post_ln`
//!   for the merger's norm). The manifest stays written in the checkpoint's
//!   spelling and [`gguf_name`] translates each lookup, so there is still one
//!   list of what a tower must hold.
//! - **The patch projection is split by frame.** The container stores one
//!   `[out, channel, y, x]` tensor per temporal slice (`v.patch_embd.weight`,
//!   `v.patch_embd.weight.1`). They are interleaved back into the
//!   framework-native `[out, channel, frame, y, x]` here, which is a spelling
//!   the manifest already accepts and the loader already knows how to reorder.
//! - **Mixed widths.** Projections are stored in half precision and the norms,
//!   biases and position table in single precision, while the encoder reads
//!   every weight at one width. The single-precision tensors are narrowed to
//!   the projections' width at load. A file stored entirely in single precision
//!   is narrowed to brain floats, which is lossless for the towers that were
//!   exported from a brain-float checkpoint and keeps the range either way.
//!
//! Projections are never dequantized: a block-quantized projector is refused by
//! name, as every other container's is.

use std::borrow::Cow;
use std::path::Path;

use shiso_core::DType;
use shiso_modelfile::{GgufFile, MetaValue, TensorSource};
use shiso_vision::{PatchGeometry, VisionError};

use super::dims::VisionDims;
use super::manifest::{AxisOrder, SIDECAR_PREFIX};

/// The architecture a projector file declares.
const ARCH: &str = "clip";

/// The projector type of the tower this family runs: a patch merger with no
/// deep-stack taps.
const PROJECTOR: &str = "qwen3vl_merger";

/// The patch projection's two temporal slices.
const PATCH_WEIGHT: &str = "v.patch_embd.weight";
/// The second temporal slice. Absent means a tower with one frame per patch.
const PATCH_WEIGHT_SECOND: &str = "v.patch_embd.weight.1";
/// The learned position table, whose row count is the tower's position count.
const POSITION_TABLE: &str = "v.position_embd.weight";

/// Name prefixes the vision tower's tensors carry in the container. A
/// projector that also carries another modality's tower writes it under a
/// prefix of its own, which the tensor count must not see.
const TOWER_PREFIXES: [&str; 2] = ["v.", "mm."];

/// Whether `file` is a projector GGUF rather than a standalone sidecar.
#[must_use]
pub(super) fn is_mmproj(file: &GgufFile) -> bool {
    file.architecture() == Some(ARCH)
}

/// An opened projector file, judged and with its dimensions derived.
#[derive(Debug)]
pub(super) struct MmprojTower {
    file: Box<GgufFile>,
    dims: VisionDims,
    /// The one width every tensor is uploaded at.
    width: DType,
}

impl MmprojTower {
    /// Judges the file and derives the tower's dimensions from its metadata.
    pub(super) fn open(file: GgufFile, path: &Path) -> Result<Self, VisionError> {
        check_projector(&file)?;
        let dims = read_dims(&file)?;
        let projection =
            TensorSource::find(&file, PATCH_WEIGHT).ok_or_else(|| VisionError::MissingTensor {
                name: PATCH_WEIGHT.to_owned(),
            })?;
        let width = match projection.dtype {
            DType::F16 => DType::F16,
            DType::BF16 | DType::F32 => DType::BF16,
            dtype => {
                return Err(VisionError::TensorDType {
                    name: PATCH_WEIGHT.to_owned(),
                    dtype,
                });
            }
        };
        tracing::info!(
            path = %path.display(),
            width = ?width,
            "the vision tower is a projector file"
        );
        Ok(Self {
            file: Box::new(file),
            dims,
            width,
        })
    }

    /// The tower's dimensions.
    pub(super) fn dims(&self) -> VisionDims {
        self.dims
    }

    /// Tower tensors the file declares, counting the split patch projection
    /// once, which is how the manifest counts it.
    pub(super) fn tensor_count(&self) -> usize {
        let declared = self
            .file
            .tensors()
            .iter()
            .filter(|desc| {
                TOWER_PREFIXES
                    .iter()
                    .any(|prefix| desc.name.starts_with(prefix))
            })
            .count();
        if self.has_second_slice() {
            declared.saturating_sub(1)
        } else {
            declared
        }
    }

    /// Width and declared shape of the tensor the manifest calls `name`.
    pub(super) fn describe(&self, name: &str) -> Result<(DType, Vec<u64>, AxisOrder), VisionError> {
        if is_patch_weight(name) {
            let first = self.find(PATCH_WEIGHT)?;
            // Stored `[x, y, channel, out]` fastest-first per slice; the
            // interleave below produces `[out, channel, frame, y, x]`.
            let [x, y, channels, out] = first.shape.dims() else {
                return Err(VisionError::TensorShape {
                    name: PATCH_WEIGHT.to_owned(),
                    declared: first.shape.dims().to_vec(),
                    expected: Vec::new(),
                });
            };
            let frames = if self.has_second_slice() { 2 } else { 1 };
            return Ok((
                self.width,
                vec![*out, *channels, frames, *y, *x],
                AxisOrder::SlowestFirst,
            ));
        }
        let desc = self.find(&gguf_name(name)?)?;
        Ok((
            self.width,
            desc.shape.dims().to_vec(),
            AxisOrder::FastestFirst,
        ))
    }

    /// The payload of the tensor the manifest calls `name`, at the upload
    /// width. Borrowed from the mapping when the file already stores it there.
    pub(super) fn bytes(&self, name: &str) -> Result<Cow<'_, [u8]>, VisionError> {
        if is_patch_weight(name) {
            return self.patch_bytes().map(Cow::Owned);
        }
        self.narrowed(&gguf_name(name)?)
    }

    /// One stored tensor at the upload width.
    fn narrowed(&self, stored: &str) -> Result<Cow<'_, [u8]>, VisionError> {
        let desc = self.find(stored)?;
        let view = TensorSource::get(self.file.as_ref(), stored).ok_or_else(|| {
            VisionError::MissingTensor {
                name: stored.to_owned(),
            }
        })?;
        convert(stored, desc.dtype, self.width, view.data)
    }

    /// The two temporal slices interleaved into one channel-slowest weight.
    ///
    /// Each slice is `[out, channel, y, x]` slowest-first; the result is
    /// `[out, channel, frame, y, x]`, so for every `(out, channel)` pair the
    /// `y * x` plane of frame 0 is followed by the same plane of frame 1.
    fn patch_bytes(&self) -> Result<Vec<u8>, VisionError> {
        let mut slices = vec![self.narrowed(PATCH_WEIGHT)?];
        if self.has_second_slice() {
            slices.push(self.narrowed(PATCH_WEIGHT_SECOND)?);
            if slices[0].len() != slices[1].len() {
                return Err(VisionError::TensorShape {
                    name: PATCH_WEIGHT_SECOND.to_owned(),
                    declared: self.find(PATCH_WEIGHT_SECOND)?.shape.dims().to_vec(),
                    expected: self.find(PATCH_WEIGHT)?.shape.dims().to_vec(),
                });
            }
        }
        let desc = self.find(PATCH_WEIGHT)?;
        let [x, y, ..] = desc.shape.dims() else {
            return Err(VisionError::MissingTensor {
                name: PATCH_WEIGHT.to_owned(),
            });
        };
        let element = usize::try_from(self.width.block_bytes().get()).unwrap_or(0);
        let plane = usize::try_from(x.saturating_mul(*y))
            .unwrap_or(0)
            .saturating_mul(element);
        if plane == 0 {
            return Err(VisionError::TowerFieldZero {
                field: "patch_size",
            });
        }
        let total = slices.iter().map(|slice| slice.len()).sum();
        let mut out = Vec::with_capacity(total);
        for at in (0..slices[0].len()).step_by(plane) {
            for slice in &slices {
                out.extend_from_slice(&slice[at..at + plane]);
            }
        }
        Ok(out)
    }

    /// Whether the patch projection is split across two frames.
    fn has_second_slice(&self) -> bool {
        TensorSource::find(self.file.as_ref(), PATCH_WEIGHT_SECOND).is_some()
    }

    /// The directory entry for a stored tensor, or its absence by name.
    fn find(&self, stored: &str) -> Result<&shiso_modelfile::TensorDesc, VisionError> {
        TensorSource::find(self.file.as_ref(), stored).ok_or_else(|| VisionError::MissingTensor {
            name: stored.to_owned(),
        })
    }
}

/// Whether the manifest name is the patch projection's weight.
fn is_patch_weight(name: &str) -> bool {
    name == format!("{SIDECAR_PREFIX}.patch_embed.proj.weight")
}

/// The container's spelling of a manifest name.
///
/// The manifest is built under [`SIDECAR_PREFIX`] for this container, so a
/// name under any other prefix, or a role this table does not know, is a
/// manifest this container cannot satisfy and is refused by name.
fn gguf_name(name: &str) -> Result<String, VisionError> {
    let missing = || VisionError::MissingTensor {
        name: name.to_owned(),
    };
    let role = name
        .strip_prefix(SIDECAR_PREFIX)
        .and_then(|rest| rest.strip_prefix('.'))
        .ok_or_else(missing)?;
    let fixed = match role {
        "patch_embed.proj.bias" => Some("v.patch_embd.bias"),
        "pos_embed.weight" => Some(POSITION_TABLE),
        "merger.norm.weight" => Some("v.post_ln.weight"),
        "merger.norm.bias" => Some("v.post_ln.bias"),
        "merger.linear_fc1.weight" => Some("mm.0.weight"),
        "merger.linear_fc1.bias" => Some("mm.0.bias"),
        "merger.linear_fc2.weight" => Some("mm.2.weight"),
        "merger.linear_fc2.bias" => Some("mm.2.bias"),
        _ => None,
    };
    if let Some(fixed) = fixed {
        return Ok(fixed.to_owned());
    }
    let rest = role.strip_prefix("blocks.").ok_or_else(missing)?;
    let (block, part) = rest.split_once('.').ok_or_else(missing)?;
    let (module, leaf) = part.rsplit_once('.').ok_or_else(missing)?;
    let module = match module {
        "norm1" => "ln1",
        "norm2" => "ln2",
        "attn.qkv" => "attn_qkv",
        "attn.proj" => "attn_out",
        "mlp.linear_fc1" => "ffn_up",
        "mlp.linear_fc2" => "ffn_down",
        _ => return Err(missing()),
    };
    Ok(format!("v.blk.{block}.{module}.{leaf}"))
}

/// Refuses a projector that is not this family's tower, by name.
fn check_projector(file: &GgufFile) -> Result<(), VisionError> {
    let meta = file.metadata();
    if meta.get_bool(&format!("{ARCH}.has_vision_encoder")) == Some(false) {
        return Err(VisionError::Unsupported {
            what: "this projector file carries no vision encoder",
        });
    }
    // Written under either key depending on the exporter's version.
    let projector = meta
        .get_str(&format!("{ARCH}.vision.projector_type"))
        .or_else(|| meta.get_str(&format!("{ARCH}.projector_type")));
    if projector != Some(PROJECTOR) {
        return Err(VisionError::Unsupported {
            what: "this projector file's vision tower is not a qwen3vl_merger tower",
        });
    }
    // A tower that taps intermediate blocks into the language model feeds it
    // rows this encoder never produces; loading it would answer about a
    // picture the model only partly saw.
    if let Some(taps) = meta.get_array(&format!("{ARCH}.vision.is_deepstack_layers")) {
        let any = taps
            .iter()
            .any(|value| matches!(value, Ok(MetaValue::Bool(true))));
        if any {
            return Err(VisionError::Unsupported {
                what: "a projector tower with deep-stack layers",
            });
        }
    }
    Ok(())
}

/// Derives the tower's dimensions from the projector's own metadata.
///
/// Two of them the container does not write as keys and they are read off the
/// tensors that embody them instead: the frames fused into one patch are the
/// patch projection's slices, and the position count is the position table's
/// row count.
fn read_dims(file: &GgufFile) -> Result<VisionDims, VisionError> {
    let meta = file.metadata();
    let read = |suffix: &str| -> Result<usize, VisionError> {
        let key = format!("{ARCH}.vision.{suffix}");
        meta.get_u32(&key)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(VisionError::TowerMetadata { key })
    };
    let temporal = if TensorSource::find(file, PATCH_WEIGHT_SECOND).is_some() {
        2
    } else {
        1
    };
    let positions = TensorSource::find(file, POSITION_TABLE)
        .and_then(|desc| desc.shape.dims().get(1).copied())
        .and_then(|rows| usize::try_from(rows).ok())
        .ok_or_else(|| VisionError::MissingTensor {
            name: POSITION_TABLE.to_owned(),
        })?;
    let geometry = PatchGeometry::new(read("patch_size")?, temporal, read("spatial_merge_size")?)?;
    VisionDims::new(
        read("block_count")?,
        read("embedding_length")?,
        read("attention.head_count")?,
        read("feed_forward_length")?,
        read("projection_dim")?,
        positions,
        geometry,
    )
}

/// `data`, stored as `from`, at width `to`.
///
/// Borrowed when the two agree. Every other pair goes through single
/// precision, which every width here widens to exactly, and rounds to nearest
/// even on the way down.
fn convert<'a>(
    name: &str,
    from: DType,
    to: DType,
    data: &'a [u8],
) -> Result<Cow<'a, [u8]>, VisionError> {
    if from == to {
        return Ok(Cow::Borrowed(data));
    }
    let refuse = || VisionError::TensorDType {
        name: name.to_owned(),
        dtype: from,
    };
    let wide: Vec<f32> = match from {
        DType::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DType::F16 => data
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        _ => return Err(refuse()),
    };
    let narrow: Vec<u8> = match to {
        DType::F16 => wide
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect(),
        DType::BF16 => wide
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect(),
        _ => return Err(refuse()),
    };
    Ok(Cow::Owned(narrow))
}

#[cfg(test)]
#[path = "mmproj_tests.rs"]
mod tests;
