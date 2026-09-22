//! The Qwen family's vision tower: dimensions, manifest, encoder, loader and
//! the wrapper behind the contract every family meets.
//!
//! The wrapper's encode is the sequence the server ran before the contract
//! existed - the same per-item encode, the same post-merge grid, the same
//! gather into one allocation - kept in order bit for bit, because the
//! committed vision goldens pin it: anything that would reorder it or change
//! an argument belongs in the encoder, not here.
//!
//! Everything below the wrapper moved here with it - the tower's dimensions,
//! its manifest, the encoder that launches the kernels and the loader that
//! fills them - because a tower is a family's, and the family is the thing
//! that knows what its weights mean. What stays shared lives in
//! `shiso-vision`: the preprocess, the containers, the contract.

use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, VisionGrid};
use shiso_vision::{
    EncodedRows, MediaPlacement, PreprocessSpec, VisionError, VisionItem, VisionSource,
    VisionStages, VisionTower,
};

mod dims;
mod encoder;
mod loader;
mod manifest;

pub use dims::VisionDims;
pub use encoder::VisionEncoder;
pub use loader::{TowerTensor, VisionWeights, load_vision_weights};
pub use manifest::{
    Affine, AxisOrder, CHECKPOINT_PREFIX, PUBLISHED_SIDECAR, SIDECAR_PREFIX,
    SIDECAR_PREFIX_CANONICAL, SidecarContract, TowerBlock, TowerManifest, TowerMerger,
    TowerTensorSpec, TowerTree,
};

/// A loaded Qwen vision tower.
///
/// Holds the weights, the encoder that drives them, and the preprocessing spec
/// derived from the tower's OWN geometry rather than from the language model's
/// config document. Those two agree for a matched pair, and when they do not,
/// the tower is the one that is right: it is the thing the patch rows are about
/// to be fed to.
#[derive(Debug)]
pub struct Qwen35Tower {
    weights: VisionWeights,
    encoder: VisionEncoder,
    spec: PreprocessSpec,
    /// The merger's block side, which is what turns a patch grid into the token
    /// grid the prompt was rendered with.
    merge: usize,
    /// Row width the merger projects to.
    out_hidden: u32,
}

impl Qwen35Tower {
    /// What this family's template writes for media, and what it expands to.
    pub const PLACEMENT: MediaPlacement = MediaPlacement::QWEN35;

    /// Opens the tower the operator named.
    ///
    /// A file is a standalone tower. A DIRECTORY is a checkpoint, and there are
    /// two ways a checkpoint ships its tower: `OptiQ` writes a shard of its own
    /// beside the language model, and `modelopt` leaves `model.visual.*`
    /// interleaved through the language model's own shards. Which one is
    /// decided by looking - the `OptiQ` shard is either there or it is not - and
    /// not by a second flag, because the operator has already said what they
    /// have by pointing at it.
    ///
    /// The probe is in that order because it is the cheaper question: the
    /// `OptiQ` layout names one file, and a `modelopt` checkpoint has no file at
    /// that path.
    ///
    /// # Errors
    ///
    /// Whatever [`load_vision_weights`] refuses the source for: a container
    /// that does not hold the tower its own dimensions describe, a tensor in a
    /// width the kernels do not read, a device that would not take it.
    pub fn open(backend: &Arc<dyn Backend>, path: &Path) -> Result<Self, VisionError> {
        let source = if path.is_dir() {
            Self::checkpoint_source(path)?
        } else {
            VisionSource::SidecarGguf(path.to_path_buf())
        };
        let weights = load_vision_weights(backend, &source)?;
        let dims = weights.dims();
        let geometry = dims.geometry();
        let merge = geometry.spatial_merge_size().max(1);
        let out_hidden = u32::try_from(dims.out_hidden()).unwrap_or(u32::MAX);
        Ok(Self {
            weights,
            encoder: VisionEncoder::new(),
            spec: PreprocessSpec::qwen35(geometry),
            merge,
            out_hidden,
        })
    }

    /// Which of the two checkpoint layouts this directory is.
    fn checkpoint_source(dir: &Path) -> Result<VisionSource, VisionError> {
        let (_, shard) = shiso_vision::open_checkpoint(dir)?;
        if shard.is_file() {
            return Ok(VisionSource::OptiqDir(dir.to_path_buf()));
        }
        Ok(VisionSource::ModeloptDir(dir.to_path_buf()))
    }

    /// Depth, width and output width, for the line the boot path logs.
    #[must_use]
    pub fn geometry(&self) -> (usize, usize, u32) {
        let dims = self.weights.dims();
        (dims.depth(), dims.hidden(), self.out_hidden)
    }

    /// Runs the whole graph over one item and returns its intermediate stages.
    ///
    /// # Errors
    ///
    /// [`VisionError`] if dimensions mismatch or device operations fail.
    pub fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        self.encoder.encode_stages(backend, &self.weights, item)
    }
}

impl VisionTower for Qwen35Tower {
    fn preprocess(&self) -> &PreprocessSpec {
        &self.spec
    }

    fn placement(&self) -> MediaPlacement {
        Self::PLACEMENT
    }

    fn out_hidden(&self) -> u32 {
        self.out_hidden
    }

    fn device_bytes(&self) -> u64 {
        self.weights.device_bytes()
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError> {
        let row_bytes = u64::from(self.out_hidden) * 4;
        let mut encoded: Vec<DeviceBuffer> = Vec::with_capacity(items.len());
        let mut grids = Vec::with_capacity(items.len());
        for item in items {
            let buffer = self.encoder.encode(backend, &self.weights, item)?;
            let (t_len, gh, gw) = item.grid();
            grids.push(VisionGrid::new(
                narrow(t_len),
                narrow(gh / self.merge),
                narrow(gw / self.merge),
            ));
            encoded.push(buffer);
        }
        EncodedRows::gather(backend, &encoded, grids, row_bytes)
    }

    fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        self.encode_stages(backend, item)
    }
}

/// Narrows one grid extent to the width a launch counts in.
///
/// Saturating rather than fallible: an extent past four billion is not a grid a
/// device could hold, and the placeholder-count check downstream refuses the
/// request either way with a message about the thing that is actually wrong.
fn narrow(extent: usize) -> u32 {
    u32::try_from(extent).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;
