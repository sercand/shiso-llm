//! One decoded piece of a prompt's media, whichever tower it is for.
//!
//! A prompt can interleave pictures and clips, and the rows each tower returns
//! are spliced at pad runs in prompt order. So the items travel as one list in
//! that order, tagged by kind, rather than as a list per tower: two lists would
//! need a second record of how they interleave, and a disagreement between the
//! two is rows landing in the wrong run.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use shiso_audio::{AudioClip, AudioTower};
use shiso_core::media::{MediaDigest, VisionGrid};
use shiso_vision::VisionItem;

use crate::media_cache::{CachedRows, MediaCache};
use crate::vision::VisionTower;

/// A decoded image, video or audio clip, ready for its tower - or one whose
/// tower output is already resident.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MediaItem {
    /// A still or a clip of frames, for the vision tower.
    Vision(VisionItem),
    /// Mel rows and their token count, for the audio tower.
    Audio(AudioClip),
    /// An item the request thread recognised, holding the rows its tower
    /// produced the last time it was seen; nothing was decoded. See
    /// [`crate::media_cache`] for why it holds the rows and not their digest.
    Encoded(EncodedItem),
}

/// An item whose tower output is resident, and the digest it was filed under.
#[derive(Debug, Clone)]
pub struct EncodedItem {
    /// The digest [`MediaItem::digest`] gave the item when it was decoded.
    pub digest: MediaDigest,
    /// The rows, shared with the cache that found them.
    pub rows: Arc<CachedRows>,
}

impl MediaItem {
    /// Pad tokens the item occupies in the prompt.
    ///
    /// `spatial_merge` is the vision tower's merge factor; a clip's count is
    /// fixed by its own tower when the clip was built, so it ignores it.
    #[must_use]
    pub fn pad_count(&self, spatial_merge: usize) -> usize {
        match self {
            Self::Vision(item) => item.pad_count(spatial_merge),
            Self::Audio(clip) => clip.tokens(),
            Self::Encoded(item) => usize::try_from(item.rows.grid().pad_count()).unwrap_or(0),
        }
    }

    /// The grid the item's tower lays its rows out in, which is also what
    /// decides the rotary positions of everything after it.
    ///
    /// Stated here, before any tower has run, because a prefix resume has to
    /// know where every picture it skips sat. It is the same division each
    /// tower performs on its way out - a still or clip's pre-merge patch grid
    /// over the merge factor, and a clip of audio as one row of its tokens.
    #[must_use]
    pub fn grid(&self, spatial_merge: usize) -> VisionGrid {
        let narrow = |value: usize| u32::try_from(value).unwrap_or(u32::MAX);
        match self {
            Self::Vision(item) => {
                let (t_len, gh, gw) = item.grid();
                let merge = spatial_merge.max(1);
                VisionGrid::new(narrow(t_len), narrow(gh / merge), narrow(gw / merge))
            }
            Self::Audio(clip) => VisionGrid::new(1, 1, narrow(clip.tokens())),
            Self::Encoded(item) => item.rows.grid(),
        }
    }

    /// A digest of exactly what the item's tower will be handed.
    ///
    /// The decoded values and their layout rather than the request's bytes:
    /// the same file can decode to different rows under a different resize,
    /// and two encodings of one picture decode to the same rows. What a cached
    /// prefix depends on is the rows, so that is what is hashed.
    ///
    /// Free for an [`MediaItem::Encoded`] item, which carries the digest it
    /// was filed under; the decoded kinds hash megabytes, which is why the
    /// request thread asks once per payload and remembers the answer.
    #[must_use]
    pub fn digest(&self) -> MediaDigest {
        let mut hash = Sha256::new();
        match self {
            Self::Encoded(item) => return item.digest,
            Self::Vision(item) => {
                let (t_len, gh, gw) = item.grid();
                hash.update(b"vision");
                for extent in [t_len, gh, gw, item.row_len()] {
                    hash.update(u64::try_from(extent).unwrap_or(u64::MAX).to_le_bytes());
                }
                hash_floats(&mut hash, item.as_flat());
            }
            Self::Audio(clip) => {
                hash.update(b"audio");
                hash.update(
                    u64::try_from(clip.tokens())
                        .unwrap_or(u64::MAX)
                        .to_le_bytes(),
                );
                hash_floats(&mut hash, clip.mel().as_slice());
            }
        }
        MediaDigest::new(hash.finalize().into())
    }

    /// Whether the item is for the audio tower.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        match self {
            Self::Audio(_) => true,
            Self::Vision(_) => false,
            Self::Encoded(item) => item.rows.is_audio(),
        }
    }
}

/// The towers a server was started with, handed to the engine thread as one.
///
/// One value because the thread encodes a request's items in prompt order and
/// needs whichever tower each one is for; handing them over separately would
/// be a second argument that could arrive without the first.
#[derive(Debug, Default)]
pub struct MediaTowers {
    /// The vision tower, when `--vision` named one.
    pub vision: Option<VisionTower>,
    /// The audio tower, when the `--vision` container carries one.
    pub audio: Option<Arc<dyn AudioTower>>,
    /// Where the towers' output is kept by content digest, shared with the
    /// renderer. `None` encodes every item of every request.
    pub cache: Option<Arc<MediaCache>>,
}

/// Feeds `values` to `hash` as little-endian bytes, a block at a time so a
/// megabyte payload is not copied whole to be hashed.
fn hash_floats(hash: &mut Sha256, values: &[f32]) {
    let mut block = Vec::with_capacity(4096);
    for chunk in values.chunks(1024) {
        block.clear();
        for value in chunk {
            block.extend_from_slice(&value.to_le_bytes());
        }
        hash.update(&block);
    }
}
