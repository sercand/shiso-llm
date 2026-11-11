//! What one job conditions the transformer on, and how it is kept.
//!
//! A text job's conditioning is the encoder's rows for its prompt. An
//! edit's is more: the encoder read the reference pictures too, so its rows
//! depend on them, and each picture also enters the transformer as VAE
//! latents between the text runs. [`JobPrompt`] is the prompt in either
//! form with the prefix layout it implies; [`Conditioning`] is what the
//! encoders produced for it; [`FeatureCache`] keeps the last few so a
//! repeated job needs no encoder at all. The cache is keyed by the prompt's
//! ids *and* a digest of the prepared pictures: the same words over other
//! pictures, or over none, are another conditioning.

use shiso_vision::generator::{ImageGenError, InputReferenceImage};

use super::edit::{MAX_REFERENCES, PreparedReference, check_count, prepare, references_hash};
use super::prompt::{EditPromptIds, PromptIds};
use super::rope::Segment;
use super::vae::LatentGrid;
use crate::diffusion::residency::Shortfall;

/// One job's prompt, as the encoder reads it.
#[derive(Debug, Clone)]
pub(crate) enum JobPrompt {
    /// Text alone.
    Text(PromptIds),
    /// Text over reference pictures.
    Edit {
        /// The ids, with each picture's run of image slots.
        ids: EditPromptIds,
        /// A digest of the prepared pictures ([`references_hash`]).
        references: [u8; 32],
        /// Each picture's latent grid, in order.
        grids: Vec<LatentGrid>,
    },
}

/// Decodes and resizes an edit's references, the count judged first so a
/// request that can never run costs no decode.
///
/// # Errors
/// [`ImageGenError::ReferenceImage`] for no references, more than the
/// model reads, or one that does not decode or has no condition size.
pub(crate) fn prepare_references(
    references: &[InputReferenceImage],
) -> Result<Vec<PreparedReference>, ImageGenError> {
    check_count(references.len(), MAX_REFERENCES)?;
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| prepare(reference, index))
        .collect()
}

impl JobPrompt {
    /// An edit's prompt over its prepared pictures.
    pub(crate) fn edit(ids: EditPromptIds, prepared: &[PreparedReference]) -> Self {
        Self::Edit {
            ids,
            references: references_hash(&prepared.iter().collect::<Vec<_>>()),
            grids: prepared
                .iter()
                .map(PreparedReference::latent_grid)
                .collect(),
        }
    }

    /// The full id sequence the encoder reads, system turn included.
    pub(crate) fn encoder_ids(&self) -> &[u32] {
        match self {
            Self::Text(ids) => &ids.ids,
            Self::Edit { ids, .. } => &ids.ids,
        }
    }

    /// Leading encoder rows the transformer never sees.
    pub(crate) fn dropped(&self) -> usize {
        match self {
            Self::Text(ids) => ids.drop,
            Self::Edit { ids, .. } => ids.drop,
        }
    }

    /// The prefix this prompt implies, over the rows after the dropped
    /// system turn: text runs, with each picture's latent grid where the
    /// encoder had its slots.
    pub(crate) fn layout(&self) -> Vec<Segment> {
        let (ids, grids) = match self {
            Self::Text(ids) => return vec![Segment::Text(ids.ids.len() - ids.drop)],
            Self::Edit { ids, grids, .. } => (ids, grids),
        };
        let mut layout = Vec::with_capacity(2 * grids.len() + 1);
        let mut cursor = ids.drop;
        for ((run, _), grid) in ids.images.iter().zip(grids) {
            if run.start > cursor {
                layout.push(Segment::Text(run.start - cursor));
            }
            layout.push(Segment::Image(*grid));
            cursor = run.end;
        }
        if ids.ids.len() > cursor {
            layout.push(Segment::Text(ids.ids.len() - cursor));
        }
        layout
    }

    /// Text rows of the prefix.
    pub(crate) fn text_rows(&self) -> usize {
        self.layout()
            .iter()
            .map(|segment| match segment {
                Segment::Text(n) => *n,
                Segment::Image(_) => 0,
            })
            .sum()
    }

    /// Every row of the prefix: text and condition-image tokens.
    pub(crate) fn prefix_tokens(&self) -> usize {
        self.layout().iter().map(|segment| segment.rows()).sum()
    }

    /// The reference pictures' latent grids; none for a text job.
    pub(crate) fn grids(&self) -> &[LatentGrid] {
        match self {
            Self::Text(_) => &[],
            Self::Edit { grids, .. } => grids,
        }
    }

    /// What this prompt's conditioning is cached under.
    pub(crate) fn key(&self) -> FeatureKey {
        let kept = self.encoder_ids()[self.dropped()..].to_vec();
        FeatureKey {
            kept,
            references: match self {
                Self::Text(_) => None,
                Self::Edit { references, .. } => Some(*references),
            },
        }
    }

    /// The text rows of the encoder's kept rows `(kept, width)`: all of
    /// them for a text job, the ones outside the pictures' slots for an
    /// edit (the transformer puts the pictures' latents there instead).
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] when `kept` is not the prompt's row count.
    pub(crate) fn text_features(
        &self,
        kept: &[f32],
        width: usize,
    ) -> Result<Vec<f32>, ImageGenError> {
        match self {
            Self::Text(_) => Ok(kept.to_vec()),
            Self::Edit { ids, .. } => text_rows_of(kept, &ids.image_mask_kept(), width),
        }
    }
}

/// The rows of `features` `(rows, width)` that `image_mask` does not mark.
///
/// # Errors
/// [`ImageGenError::Backend`] when the mask and the rows disagree.
pub(crate) fn text_rows_of(
    features: &[f32],
    image_mask: &[bool],
    width: usize,
) -> Result<Vec<f32>, ImageGenError> {
    if width == 0 || features.len() != image_mask.len() * width {
        return Err(ImageGenError::Backend(format!(
            "the encoder returned {} values for {} rows of {width}",
            features.len(),
            image_mask.len()
        )));
    }
    Ok(features
        .chunks_exact(width)
        .zip(image_mask)
        .filter(|(_, image)| !**image)
        .flat_map(|(row, _)| row.iter().copied())
        .collect())
}

/// What a job's conditioning is cached under: the kept prompt ids and, for
/// an edit, the digest of its prepared pictures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FeatureKey {
    /// The ids after the dropped system turn.
    pub kept: Vec<u32>,
    /// The pictures' digest; `None` for a text job.
    pub references: Option<[u8; 32]>,
}

/// What the encoders produced for one job.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Conditioning {
    /// The prefix's text rows in sequence order, `(text rows, context)`.
    pub text: Vec<f32>,
    /// Each reference picture's normalized latent tokens,
    /// `(cells, channels)`, in order; empty for a text job.
    pub images: Vec<Vec<f32>>,
}

/// The last few jobs' conditioning, most recent first.
#[derive(Debug)]
pub(crate) struct FeatureCache {
    entries: Vec<(FeatureKey, Conditioning)>,
    capacity: usize,
}

impl FeatureCache {
    /// An empty cache of at most `capacity` entries.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// The conditioning cached under `key`, which becomes the most recent.
    pub(crate) fn get(&mut self, key: &FeatureKey) -> Option<Conditioning> {
        let at = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(at);
        let found = entry.1.clone();
        self.entries.insert(0, entry);
        Some(found)
    }

    /// Caches `conditioning` under `key` as the most recent entry.
    pub(crate) fn put(&mut self, key: FeatureKey, conditioning: Conditioning) {
        self.entries.retain(|(k, _)| *k != key);
        self.entries.insert(0, (key, conditioning));
        self.entries.truncate(self.capacity);
    }
}

/// The refusal of an edit whose working memory does not fit: the budget is
/// the device's, and what moves the need is how many pictures the job
/// carries, so the message names both.
pub(crate) fn edit_refusal(short: Shortfall, references: usize) -> ImageGenError {
    ImageGenError::ReferenceImage(format!(
        "an edit with {references} reference images needs {} MiB of device memory and this \
         host has {} MiB for it; each reference adds its tokens to every step, so send fewer \
         references or a smaller output size",
        short.needed.div_ceil(1 << 20),
        short.budget >> 20
    ))
}
