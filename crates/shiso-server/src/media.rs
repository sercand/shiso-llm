//! One decoded piece of a prompt's media, whichever tower it is for.
//!
//! A prompt can interleave pictures and clips, and the rows each tower returns
//! are spliced at pad runs in prompt order. So the items travel as one list in
//! that order, tagged by kind, rather than as a list per tower: two lists would
//! need a second record of how they interleave, and a disagreement between the
//! two is rows landing in the wrong run.

use std::sync::Arc;

use shiso_audio::{AudioClip, AudioTower};
use shiso_vision::VisionItem;

use crate::vision::VisionTower;

/// A decoded image, video or audio clip, ready for its tower.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MediaItem {
    /// A still or a clip of frames, for the vision tower.
    Vision(VisionItem),
    /// Mel rows and their token count, for the audio tower.
    Audio(AudioClip),
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
        }
    }

    /// Whether the item is for the audio tower.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
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
}
