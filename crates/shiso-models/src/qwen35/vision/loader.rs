//! Moving a vision tower's weights onto a device.
//!
//! Two containers ship a tower and they agree on almost nothing: one is a
//! shard inside a checkpoint directory whose dimensions live in a sibling
//! config, the other a standalone file that carries its own metadata and writes
//! its axes in the opposite order. What they do agree on is the tower - the
//! same blocks, the same roles, the same names under a different prefix - so
//! this module resolves the differences once, up front, and runs one upload
//! loop over one manifest.
//!
//! # Dimensions come from the file, contracts only judge them
//!
//! Nothing here computes a shape from a constant. The dimensions are read out
//! of whichever container was opened, the manifest is derived from those
//! dimensions, and every tensor is checked against the manifest before it is
//! uploaded. The published contract for the standalone tower is consulted only
//! to decide whether to accept the file at all - it never supplies a number the
//! loader then uses. That split is what lets one loader read two towers of
//! different widths without either of them being the special case.
//!
//! # Why the shape check is the point
//!
//! Every tensor in a mismatched tower is individually well-formed. Load one
//! without checking and the weights are on the device, the kernels run, and the
//! model describes pictures it was never shown. The check costs one comparison
//! per tensor at boot and is the only thing standing between those two
//! outcomes, which is why it happens before the upload rather than after.

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, DType, alloc};
use shiso_modelfile::modelopt::{ModeloptFile, Stored};
use shiso_modelfile::st_header::StDType;
use shiso_modelfile::{GgufFile, TensorSource};
use shiso_vision::{VisionError, VisionShard, VisionSource, open_gguf};

use super::dims::VisionDims;
use super::manifest::{
    AxisOrder, CHECKPOINT_PREFIX, PUBLISHED_SIDECAR, PatchRows, SIDECAR_PREFIX,
    SIDECAR_PREFIX_CANONICAL, SidecarContract, TowerManifest, TowerTensorSpec, TowerTree,
};

/// Formats the tower loader will move to a device.
///
/// A tower stored in a block-quantized format is a different loading problem
/// rather than a bigger one - the bytes are not rows, and the kernels that read
/// them are not the ones a float tower uses - so it is refused by name here
/// rather than uploaded and discovered later. This is the one place to widen if
/// such a tower ever ships.
const UPLOADABLE: &[DType] = &[DType::F32, DType::F16, DType::BF16];

/// The prefix a `modelopt` checkpoint writes its tower under.
///
/// The same spelling [`SIDECAR_PREFIX`] uses, and not a coincidence: both are
/// what `transformers` names the module, one written into a standalone file
/// and one left in the shard set it was exported from.
const MODELOPT_TOWER_PREFIX: &str = SIDECAR_PREFIX;

/// One header spelling as the format the rest of this file speaks.
///
/// The container's own enum is the union over two containers and is wider than
/// a tower can be; this narrows it, and a width the tower cannot carry is a
/// refusal naming the tensor rather than a cast.
fn st_dtype(name: &str, dtype: StDType) -> Result<DType, VisionError> {
    match dtype {
        StDType::Bf16 => Ok(DType::BF16),
        StDType::F16 => Ok(DType::F16),
        StDType::F32 => Ok(DType::F32),
        StDType::U8 | StDType::U32 | StDType::F8E4M3 => Err(VisionError::TensorDType {
            name: name.to_owned(),
            dtype: DType::I32,
        }),
    }
}

/// Metadata suffixes the standalone tower's dimensions are read from.
///
/// Read under the file's own architecture prefix, so a tower that renames
/// itself keeps the same sub-keys. Each is required: a tower missing one has no
/// derivable geometry, and defaulting it would produce a manifest that happens
/// to match some other tower.
const KEY_BLOCK_COUNT: &str = "block_count";
/// Metadata suffix holding the tower's internal width.
const KEY_EMBEDDING_LENGTH: &str = "embedding_length";
/// Metadata suffix holding the feed-forward inner width.
const KEY_FEED_FORWARD_LENGTH: &str = "feed_forward_length";
/// Metadata suffix holding the attention head count.
const KEY_HEAD_COUNT: &str = "attention.head_count";
/// Metadata suffix holding the width handed to the language model.
///
/// `projection_length`, and the `_length` is not a synonym: it is the key every
/// published tower of both vision families writes, and a tower that carries the
/// weights but not this name is a tower this loader refuses by name.
const KEY_PROJECTION_LENGTH: &str = "projection_length";
/// Metadata suffix holding the spatial patch edge.
const KEY_PATCH_SIZE: &str = "patch_size";
/// Metadata suffix holding the frames fused into one patch.
const KEY_TEMPORAL_PATCH_SIZE: &str = "temporal_patch_size";
/// Metadata suffix holding the merge block's side.
const KEY_SPATIAL_MERGE_SIZE: &str = "spatial_merge_size";
/// Metadata suffix holding the learned position table's row count.
const KEY_POSITION_COUNT: &str = "position_embedding_count";

/// One tower tensor, resident on a device.
///
/// Owns its allocation: dropping the tower frees every buffer in it, with no
/// teardown call to forget and no handle that outlives what it names.
#[derive(Debug)]
#[non_exhaustive]
pub struct TowerTensor {
    name: String,
    dtype: DType,
    elements: u64,
    buffer: shiso_core::DeviceBuffer,
}

impl TowerTensor {
    /// Fully qualified name the tensor was loaded under.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Storage format the container declared.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Elements the tensor holds, from the manifest rather than from its bytes.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.elements
    }

    /// The device allocation.
    #[must_use]
    pub fn buffer(&self) -> &shiso_core::DeviceBuffer {
        &self.buffer
    }

    /// A view of the whole tensor.
    #[must_use]
    pub fn view(&self) -> shiso_core::DeviceView<'_> {
        self.buffer.view()
    }
}

/// A vision tower on a device: its dimensions and every weight it needs.
#[derive(Debug)]
#[non_exhaustive]
pub struct VisionWeights {
    dims: VisionDims,
    prefix: String,
    tensors: TowerTree<TowerTensor>,
}

impl VisionWeights {
    /// The tower's dimensions, as the container declared them.
    #[must_use]
    pub fn dims(&self) -> VisionDims {
        self.dims
    }

    /// The name prefix the tower was found under.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The weights, in the tree the encoder walks.
    #[must_use]
    pub fn tensors(&self) -> &TowerTree<TowerTensor> {
        &self.tensors
    }

    /// Tensors resident on the device.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Device bytes the tower occupies.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.tensors.iter().map(|tensor| tensor.buffer.len()).sum()
    }
}

/// Loads a vision tower onto `backend`.
///
/// Returns a tower or an error, never `Ok(None)`: a source that was named and
/// yielded nothing is a load failure, and "no tower configured" is the absence
/// of a [`VisionSource`] at the call site rather than a value this returns.
///
/// # Errors
///
/// [`VisionError::NoVisionConfig`] when a checkpoint declares no tower;
/// [`VisionError::NoTowerPrefix`] when a standalone file carries neither
/// spelling of the prefix; [`VisionError::SidecarContract`] when a standalone
/// tower disagrees with the published contract for its architecture;
/// [`VisionError::MissingTensor`], [`VisionError::TensorShape`],
/// [`VisionError::TensorDType`] and [`VisionError::TensorCount`] when the file
/// does not hold the tower its own dimensions describe; and
/// [`VisionError::Upload`] when the device refuses a buffer.
pub fn load_vision_weights(
    backend: &Arc<dyn Backend>,
    source: &VisionSource,
) -> Result<VisionWeights, VisionError> {
    let tower = Tower::open(source)?;
    let dims = tower.dims();
    let prefix = tower.prefix().to_owned();
    let manifest = TowerManifest::build(&prefix, dims);

    let declared = tower.tensor_count();
    if declared != manifest.len() {
        return Err(VisionError::TensorCount {
            found: declared,
            expected: manifest.len(),
        });
    }

    tracing::info!(
        backend = backend.backend_name(),
        prefix = %prefix,
        depth = dims.depth(),
        hidden = dims.hidden(),
        tensors = manifest.len(),
        "loading a vision tower"
    );

    let tensors = manifest.try_map(|spec| tower.upload(backend, &spec))?;
    let weights = VisionWeights {
        dims,
        prefix,
        tensors,
    };
    tracing::info!(
        bytes = weights.device_bytes(),
        "the vision tower is resident"
    );
    Ok(weights)
}

/// An opened tower, whichever container it arrived in.
#[derive(Debug)]
enum Tower {
    /// A shard inside a checkpoint directory.
    Shard {
        /// The opened shard.
        shard: VisionShard,
        /// Dimensions, from the checkpoint's config.
        dims: VisionDims,
    },
    /// The tower inside a `modelopt` checkpoint's own shard set.
    Modelopt {
        /// The opened shard set, narrowed to the tower's prefix.
        file: Box<ModeloptFile>,
        /// Dimensions, from the checkpoint's config.
        dims: VisionDims,
    },
    /// A standalone tower file.
    Sidecar {
        /// The opened container.
        file: Box<GgufFile>,
        /// Dimensions, from the file's own metadata.
        dims: VisionDims,
        /// Which spelling of the prefix the tower was found under.
        prefix: &'static str,
    },
}

impl Tower {
    /// Opens whichever container the source names and derives its dimensions.
    fn open(source: &VisionSource) -> Result<Self, VisionError> {
        match source {
            VisionSource::OptiqDir(dir) => Self::open_checkpoint(dir),
            VisionSource::ModeloptDir(dir) => Self::open_modelopt(dir),
            VisionSource::SidecarGguf(path) => Self::open_sidecar(path),
        }
    }

    /// Opens the tower shard beside a checkpoint's language model.
    ///
    /// The container door reads the config and resolves the shard path; what
    /// is done here is the tower half: deriving the dimensions from the
    /// checkpoint's own vision block and refusing a checkpoint that has none.
    fn open_checkpoint(dir: &Path) -> Result<Self, VisionError> {
        let (config, shard_path) = shiso_vision::open_checkpoint(dir)?;
        let vcfg = config
            .vision_config()
            .map_err(|source| VisionError::TowerConfig {
                path: dir.join("config.json"),
                source,
            })?
            .ok_or_else(|| VisionError::NoVisionConfig {
                path: dir.to_path_buf(),
            })?;
        let dims = VisionDims::try_from(&vcfg)?;
        let shard = VisionShard::open(&shard_path)?;
        Ok(Self::Shard { shard, dims })
    }

    /// Opens the tower that rides in a `modelopt` checkpoint's own shard set.
    ///
    /// The dimensions come from the same place the `OptiQ` arm takes them - the
    /// checkpoint's `vision_config` - because they describe the same tower.
    /// What differs is only where the bytes are: `model.visual.*` interleaved
    /// through the language model's shards rather than in a file of its own,
    /// which is why the container below is the one that reads that shard set
    /// and not [`shiso_vision::VisionShard`], whose whole design is one file.
    ///
    /// The prefix is passed to the open so the 93,000 language-model tensors
    /// beside the tower are never described at all.
    fn open_modelopt(dir: &Path) -> Result<Self, VisionError> {
        let (config, _) = shiso_vision::open_checkpoint(dir)?;
        let vcfg = config
            .vision_config()
            .map_err(|source| VisionError::TowerConfig {
                path: dir.join("config.json"),
                source,
            })?
            .ok_or_else(|| VisionError::NoVisionConfig {
                path: dir.to_path_buf(),
            })?;
        let dims = VisionDims::try_from(&vcfg)?;
        let file = ModeloptFile::open(dir, &[MODELOPT_TOWER_PREFIX]).map_err(|source| {
            VisionError::TowerContainer {
                path: dir.to_path_buf(),
                source,
            }
        })?;
        Ok(Self::Modelopt {
            file: Box::new(file),
            dims,
        })
    }

    /// Opens a standalone tower file and judges it against the contract.
    fn open_sidecar(path: &Path) -> Result<Self, VisionError> {
        let file = open_gguf(path)?;
        let prefix = probe_prefix(&file, path)?;
        let dims = sidecar_dims(&file)?;
        // Judged AFTER the dimensions are derived, never before: the contract
        // decides whether this file is the tower it claims to be, and it can
        // only do that once there is something to compare.
        if file.architecture() == Some(PUBLISHED_SIDECAR.architecture) {
            check_contract(&PUBLISHED_SIDECAR, dims)?;
        }
        Ok(Self::Sidecar {
            file: Box::new(file),
            dims,
            prefix,
        })
    }

    /// The tower's dimensions.
    fn dims(&self) -> VisionDims {
        match self {
            Self::Shard { dims, .. } | Self::Modelopt { dims, .. } | Self::Sidecar { dims, .. } => {
                *dims
            }
        }
    }

    /// The prefix every tensor name carries.
    fn prefix(&self) -> &str {
        match self {
            Self::Shard { .. } => CHECKPOINT_PREFIX,
            // The same spelling a standalone tower's short prefix uses, which
            // is what this container writes.
            Self::Modelopt { .. } => MODELOPT_TOWER_PREFIX,
            Self::Sidecar { prefix, .. } => prefix,
        }
    }

    /// Tensors the container declares under this tower's prefix.
    fn tensor_count(&self) -> usize {
        match self {
            Self::Shard { shard, .. } => shard.len(),
            // Already narrowed to the prefix at open, so this is the tower's
            // own count and not the checkpoint's.
            Self::Modelopt { file, .. } => file.tensors().len(),
            Self::Sidecar { file, prefix, .. } => {
                let under = format!("{prefix}.");
                file.tensors()
                    .iter()
                    .filter(|desc| desc.name.starts_with(&under))
                    .count()
            }
        }
    }

    /// Checks one tensor against the manifest and moves it to the device.
    fn upload(
        &self,
        backend: &Arc<dyn Backend>,
        spec: &TowerTensorSpec,
    ) -> Result<TowerTensor, VisionError> {
        let (dtype, declared, order) = self.describe(spec.name())?;
        let Some(rows) = spec.spelling(&declared, order) else {
            return Err(VisionError::TensorShape {
                name: spec.name().to_owned(),
                declared,
                expected: spec.shape().to_vec(),
            });
        };
        if !UPLOADABLE.contains(&dtype) {
            return Err(VisionError::TensorDType {
                name: spec.name().to_owned(),
                dtype,
            });
        }
        let bytes = self.bytes(spec.name())?;
        let bytes = match rows {
            PatchRows::AsStored => bytes,
            // Transposed here rather than in the encoder, because the encoder
            // never learns which file a weight came from - and rather than in
            // the packer, because the preprocessor does not either. This is the
            // one place that knows both.
            PatchRows::ChannelSlowest => Cow::Owned(channel_minor(&bytes, spec, dtype)?),
        };
        upload(backend, spec, dtype, &bytes)
    }

    /// Format and declared shape of one tensor, with the axis order they are
    /// written in.
    fn describe(&self, name: &str) -> Result<(DType, Vec<u64>, AxisOrder), VisionError> {
        match self {
            Self::Shard { shard, .. } => {
                let entry = shard
                    .entry(name)
                    .ok_or_else(|| VisionError::MissingTensor {
                        name: name.to_owned(),
                    })?;
                Ok((entry.dtype, entry.shape.clone(), AxisOrder::SlowestFirst))
            }
            Self::Modelopt { file, .. } => {
                let tensor = file.get(name).ok_or_else(|| VisionError::MissingTensor {
                    name: name.to_owned(),
                })?;
                // Every tensor of this tower is plain `BF16`; a quantized one
                // would be a different loading problem rather than a bigger
                // one, so it is refused by name here rather than decoded.
                let Stored::Plain { dtype } = tensor.stored() else {
                    return Err(VisionError::TensorDType {
                        name: name.to_owned(),
                        dtype: DType::F32,
                    });
                };
                Ok((
                    st_dtype(name, dtype)?,
                    tensor.declared().to_vec(),
                    AxisOrder::SlowestFirst,
                ))
            }
            Self::Sidecar { file, .. } => {
                let desc = TensorSource::find(file.as_ref(), name).ok_or_else(|| {
                    VisionError::MissingTensor {
                        name: name.to_owned(),
                    }
                })?;
                Ok((
                    desc.dtype,
                    desc.shape.dims().to_vec(),
                    AxisOrder::FastestFirst,
                ))
            }
        }
    }

    /// One tensor's payload.
    ///
    /// Borrowed from the standalone container's mapping and owned from the
    /// shard, because the shard reads on demand and the container does not.
    /// The difference rides in the return type rather than being flattened to
    /// an owned buffer: flattening it would copy the whole tower a second time
    /// on the path that did not need to.
    fn bytes(&self, name: &str) -> Result<Cow<'_, [u8]>, VisionError> {
        match self {
            Self::Shard { shard, .. } => shard.read(name).map(Cow::Owned),
            Self::Modelopt { file, .. } => {
                let tensor = file.get(name).ok_or_else(|| VisionError::MissingTensor {
                    name: name.to_owned(),
                })?;
                file.weight_bytes(tensor)
                    .map(Cow::Borrowed)
                    .map_err(|source| VisionError::TowerContainer {
                        path: std::path::PathBuf::from(name),
                        source,
                    })
            }
            Self::Sidecar { file, .. } => TensorSource::get(file.as_ref(), name)
                .map(|view| Cow::Borrowed(view.data))
                .ok_or_else(|| VisionError::MissingTensor {
                    name: name.to_owned(),
                }),
        }
    }
}

/// Moves one checked tensor to the device.
/// Moves the channel from the slowest axis of a patch row to the fastest.
///
/// One output row of the patch projection is `channels` blocks of `patch`
/// elements as stored, and `patch` blocks of `channels` elements as the
/// preprocessor builds it: a transpose of each row, and nothing else. The
/// element count does not change, which is exactly why this cannot be left to a
/// shape check.
fn channel_minor(
    bytes: &[u8],
    spec: &TowerTensorSpec,
    dtype: DType,
) -> Result<Vec<u8>, VisionError> {
    let wrong = |what: &'static str| VisionError::TensorShape {
        name: format!("{} ({what})", spec.name()),
        declared: spec.shape().to_vec(),
        expected: spec.shape().to_vec(),
    };
    // `[out, channels, frame, y, x]`: the canonical spelling this is the
    // alternative to has the same five axes in a different order, so the widths
    // are read off the spelling that carries the channel second.
    let dims = spec
        .accepted()
        .iter()
        .find(|shape| shape.rows == PatchRows::ChannelSlowest)
        .ok_or_else(|| wrong("no channel-slowest spelling"))?;
    let [out, channels, rest @ ..] = dims.dims.as_slice() else {
        return Err(wrong("a patch spelling with fewer than three axes"));
    };
    let patch = rest.iter().product::<u64>();
    let width = u64::from(dtype.block_bytes().get());
    let row = channels.saturating_mul(patch).saturating_mul(width);
    let want = out.saturating_mul(row);
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != want {
        return Err(wrong("a payload that is not the shape's own size"));
    }
    let (channels, patch, width) = (
        usize::try_from(*channels).unwrap_or(0),
        usize::try_from(patch).unwrap_or(0),
        usize::try_from(width).unwrap_or(0),
    );
    let mut out_bytes = vec![0u8; bytes.len()];
    for (src_row, dst_row) in bytes
        .chunks_exact(channels * patch * width)
        .zip(out_bytes.chunks_exact_mut(channels * patch * width))
    {
        for c in 0..channels {
            for p in 0..patch {
                let from = (c * patch + p) * width;
                let to = (p * channels + c) * width;
                dst_row[to..to + width].copy_from_slice(&src_row[from..from + width]);
            }
        }
    }
    Ok(out_bytes)
}

fn upload(
    backend: &Arc<dyn Backend>,
    spec: &TowerTensorSpec,
    dtype: DType,
    bytes: &[u8],
) -> Result<TowerTensor, VisionError> {
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let fail = |source| VisionError::Upload {
        name: spec.name().to_owned(),
        source,
    };
    let buffer = alloc(backend, len).map_err(fail)?;
    backend.copy_h2d(buffer.view(), bytes).map_err(fail)?;
    Ok(TowerTensor {
        name: spec.name().to_owned(),
        dtype,
        elements: spec.elements(),
        buffer,
    })
}

/// Finds which spelling of the tower prefix a standalone file uses.
///
/// Probed on the patch projection, which every tower has exactly one of, rather
/// than on a block tensor: a truncated export that lost its blocks would still
/// answer to a block-zero probe if it kept one.
fn probe_prefix(file: &GgufFile, path: &Path) -> Result<&'static str, VisionError> {
    for prefix in [SIDECAR_PREFIX, SIDECAR_PREFIX_CANONICAL] {
        if TensorSource::find(file, &format!("{prefix}.patch_embed.proj.weight")).is_some() {
            return Ok(prefix);
        }
    }
    Err(VisionError::NoTowerPrefix {
        path: path.to_path_buf(),
        short: SIDECAR_PREFIX,
        canonical: SIDECAR_PREFIX_CANONICAL,
    })
}

/// Derives a standalone tower's dimensions from its own metadata.
fn sidecar_dims(file: &GgufFile) -> Result<VisionDims, VisionError> {
    let read = |suffix: &str| -> Result<usize, VisionError> {
        let value =
            TensorSource::arch_u32(file, suffix).ok_or_else(|| VisionError::TowerMetadata {
                key: suffix.to_owned(),
            })?;
        usize::try_from(value).map_err(|_| VisionError::TowerMetadata {
            key: suffix.to_owned(),
        })
    };
    let geometry = shiso_vision::PatchGeometry::new(
        read(KEY_PATCH_SIZE)?,
        read(KEY_TEMPORAL_PATCH_SIZE)?,
        read(KEY_SPATIAL_MERGE_SIZE)?,
    )?;
    VisionDims::new(
        read(KEY_BLOCK_COUNT)?,
        read(KEY_EMBEDDING_LENGTH)?,
        read(KEY_HEAD_COUNT)?,
        read(KEY_FEED_FORWARD_LENGTH)?,
        read(KEY_PROJECTION_LENGTH)?,
        read(KEY_POSITION_COUNT)?,
        geometry,
    )
}

/// Judges derived dimensions against a published contract.
fn check_contract(contract: &SidecarContract, dims: VisionDims) -> Result<(), VisionError> {
    for (field, required, got) in [
        ("block_count", contract.depth, dims.depth()),
        ("embedding_length", contract.hidden, dims.hidden()),
        (
            "feed_forward_length",
            contract.intermediate,
            dims.intermediate(),
        ),
        ("projection_length", contract.out_hidden, dims.out_hidden()),
        ("tensor_count", contract.tensors, dims.tensor_count()),
    ] {
        if required != got {
            return Err(VisionError::SidecarContract {
                architecture: contract.architecture,
                field,
                required,
                got,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
