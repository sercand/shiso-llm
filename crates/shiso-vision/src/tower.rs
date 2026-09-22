//! The seam between a family's vision tower and the server that serves it.
//!
//! # Why a trait and not a struct
//!
//! There was one tower, so there was one type, and the server named it
//! directly: it opened a Qwen encoder, read a Qwen merge factor off it, and
//! expanded a Qwen pad marker in the prompt. A second family does not fit
//! behind any of those - its patch grid pools three-wide rather than two, its
//! template writes a different marker, and it brackets the run of pads with two
//! more tokens. Adding it as a second branch would put an architecture test in
//! the scheduler, the renderer and the boot path, and each of those branches is
//! a place where a build serving one family silently renders for the other.
//!
//! So the tower is a contract and the family implements it. Everything the
//! server needs to know about a tower arrives through this trait: how to turn a
//! picture into patch rows ([`VisionTower::preprocess`]), what the template
//! writes and what it expands to ([`MediaPlacement`]), how wide the rows come
//! out, and how to run it.
//!
//! # Why the placement is data rather than a rendering hook
//!
//! A family could have been given a "render the media run" method. It would
//! have to be handed the vocabulary, the item's pad count and the token buffer,
//! and it could then write anything at all into the prompt - which is precisely
//! the thing the engine's media walk cannot survive. The walk finds media by
//! looking for runs of the placeholder id. Three strings and a count are enough
//! to describe every placement either family uses, they cannot express a run
//! the walk would not recognise, and they can be resolved to ids once at boot.

use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, VisionGrid};

use crate::config::PreprocessSpec;
use crate::error::VisionError;
use crate::item::VisionItem;

/// Where a family's template puts media, and what the renderer expands it to.
///
/// Spelled as markers rather than as token ids because a placement is a
/// property of the family and an id is a property of one checkpoint's
/// vocabulary. The resolution from one to the other happens once, at boot,
/// where a marker the vocabulary does not declare is a fact an operator can be
/// told about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaPlacement {
    /// The marker the template writes once per image; the renderer repeats it
    /// `pad_count` times.
    pub image_marker: &'static str,
    /// The marker for a clip, when the family has one.
    pub video_marker: Option<&'static str>,
    /// Tokens written before and after the repeated run, when the family wants
    /// them.
    pub wrap_image: Option<(&'static str, &'static str)>,
}

impl MediaPlacement {
    /// The Qwen family's placement: one marker per item, no bracket.
    ///
    /// Lives here until the Qwen tower moves into its own family crate, and
    /// moves with it when it does. It is a constant rather than two string
    /// literals at the call sites because the renderer and the scheduler both
    /// need it and a second spelling of either marker is a second answer that
    /// can disagree with the first.
    pub const QWEN35: Self = Self {
        image_marker: "<|image_pad|>",
        video_marker: Some("<|video_pad|>"),
        wrap_image: None,
    };

    /// Gemma 4's placement: one marker per image, bracketed, and no clip.
    ///
    /// The bracket is not decoration. This family's template writes
    /// `<|image>` … `<image|>` around the soft-token run, both tokens are in
    /// the vocabulary, and a prompt without them is a prompt the model was
    /// never trained on - which produces a fluent answer rather than a failure.
    pub const GEMMA4: Self = Self {
        image_marker: "<|image|>",
        video_marker: None,
        wrap_image: Some(("<|image>", "<image|>")),
    };
}

/// One request's encoder output, in the single block a splice can address.
///
/// The rows and the grids are one value because they are one fact: a block of
/// rows whose height nothing records is a block nobody can address, and a grid
/// list with no rows beside it describes a picture that was never encoded.
#[derive(Debug)]
pub struct EncodedRows {
    /// The whole block, `[sum of placeholder counts, out_hidden]` single
    /// precision, in prompt order.
    pub rows: DeviceBuffer,
    /// The items' post-merge grids, in the order the prompt refers to them.
    pub grids: Vec<VisionGrid>,
}

impl EncodedRows {
    /// Gathers per-item encoder outputs into one allocation, in prompt order.
    ///
    /// Every tower ends its encode this way, so the copy loop is here rather
    /// than duplicated per family: the map the engine builds is a row index
    /// into a single allocation, and a list of per-item buffers would need a
    /// base index the walk does not carry. Items are copied device-to-device,
    /// which is a few microseconds against the tens or hundreds of
    /// milliseconds the encode itself costs.
    ///
    /// `row_bytes` is the floor for the allocation, so that a request whose
    /// items encoded to nothing still yields a buffer a view can be taken over.
    ///
    /// # Errors
    ///
    /// [`VisionError::Encode`] naming the gather, carrying the device's own
    /// error, when the allocation or one of the copies fails.
    pub fn gather(
        backend: &Arc<dyn Backend>,
        parts: &[DeviceBuffer],
        grids: Vec<VisionGrid>,
        row_bytes: u64,
    ) -> Result<Self, VisionError> {
        let step = "gathering the encoded rows";
        let total = parts
            .iter()
            .map(DeviceBuffer::len)
            .fold(0u64, u64::saturating_add);
        let rows = shiso_core::alloc(backend, total.max(row_bytes))
            .map_err(|source| VisionError::Encode { step, source })?;
        let mut at = 0u64;
        for buffer in parts {
            let len = buffer.len();
            let dst = rows
                .slice(at, len)
                .map_err(|source| VisionError::Encode { step, source })?;
            backend
                .copy_d2d(dst, buffer.view())
                .map_err(|source| VisionError::Encode { step, source })?;
            at = at.saturating_add(len);
        }
        backend
            .sync()
            .map_err(|source| VisionError::Encode { step, source })?;
        Ok(Self { rows, grids })
    }

    /// Placeholder tokens the whole request occupies.
    #[must_use]
    pub fn pad_total(&self) -> u32 {
        self.grids
            .iter()
            .map(VisionGrid::pad_count)
            .fold(0u32, u32::saturating_add)
    }
}

/// Intermediate activation stages captured from a vision tower.
#[derive(Debug)]
pub struct VisionStages {
    /// Preprocessed patch rows fed into the tower.
    pub pixel_values: DeviceBuffer,
    /// Activations after the initial patch projection GEMM.
    pub patch_embed: DeviceBuffer,
    /// Activations after adding position embeddings.
    pub pos_added: DeviceBuffer,
    /// Outputs of each transformer block in the tower.
    pub blocks: Vec<DeviceBuffer>,
    /// Activations after pooling / spatial merge.
    pub pooled: DeviceBuffer,
    /// Activations after normalization before projection.
    pub normed: DeviceBuffer,
    /// Final projected activations.
    pub projected: DeviceBuffer,
}

/// A loaded vision tower, as everything above it sees one.
///
/// `Send + Sync` because the value is built on the boot thread and run on the
/// engine thread, and `&self` on the encode because a tower holds weights and
/// no scratch: two encodes of the same tower share nothing that could be
/// observed. A family whose encoder needs mutable state owns that decision -
/// it wraps it, rather than widening this contract and making every caller
/// hold a lock it does not need.
pub trait VisionTower: Send + Sync + std::fmt::Debug {
    /// How this tower wants a picture turned into patch rows.
    fn preprocess(&self) -> &PreprocessSpec;

    /// What this family's template writes for media, and what it expands to.
    fn placement(&self) -> MediaPlacement;

    /// Width of the rows the tower projects to, which must be the language
    /// model's residual width.
    fn out_hidden(&self) -> u32;

    /// Device bytes this tower's weights occupy.
    fn device_bytes(&self) -> u64;

    /// Encodes every item and gathers the rows in prompt order.
    ///
    /// `grids[i].pad_count()` rows per item, which is the same number of
    /// placeholder tokens item `i` occupies in the prompt. That equality is the
    /// contract the prefill splice leans on.
    ///
    /// # Errors
    ///
    /// Whatever the family's encoder refuses an item for, and
    /// [`VisionError::Encode`] for a device failure in the tower or in the
    /// gather.
    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError>;

    /// Encodes a single item and returns its intermediate device buffers.
    ///
    /// # Errors
    ///
    /// [`VisionError::Unsupported`] if the tower does not implement staged encoding,
    /// or [`VisionError::Encode`] on device failure.
    fn encode_stages(
        &self,
        _backend: &Arc<dyn Backend>,
        _item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        Err(VisionError::Unsupported {
            what: "staged encoding is not implemented for this vision tower",
        })
    }
}
