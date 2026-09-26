//! The vision tower, as a running server holds it.
//!
//! # What this closes
//!
//! Everything either side of this file already existed. The renderer decodes a
//! request's pictures and expands each one's placeholder run from the tower's
//! own geometry; the engine splices encoder rows over those placeholders and
//! rotates them by grid coordinates instead of by a count. Between the two
//! there was nothing: no server ever constructed an encoder, so a request with
//! a picture attached was served as its text half with a warning in the log.
//! This is the middle.
//!
//! # Why this type is a wrapper and not the tower
//!
//! The tower itself belongs to the family that ships it - it is the thing that
//! knows how many patches pool into a token and what its template writes for a
//! picture - and it reaches here as a [`shiso_vision::VisionTower`]. What is
//! left for the server is the part that is about THIS process rather than about
//! any family: that the tower pairs with the language model beside it, and that
//! a request's rows come back in a value the prefill contract can borrow from.
//!
//! # Why the tower is loaded at boot or not at all
//!
//! `--vision` names it and its absence means text only. Loading lazily, on the
//! first request that carries a picture, would move a multi-second load and a
//! large allocation into one unlucky caller's latency and would turn a bad path
//! or a mismatched tower into a runtime failure that the operator sees hours
//! after starting the process. So it is opened before the listener binds, and a
//! server started without the flag pays nothing at all: no file is read, no
//! device memory is taken, and no encoder type is ever instantiated.
//!
//! # The width check, and why it is at boot
//!
//! A tower's merger projects to the language model's residual width. Two
//! checkpoints that disagree about that width still load, still encode and
//! still answer — the row count is right, the rows are the wrong shape, and the
//! answer is fluent prose about nothing. The comparison is one integer against
//! one integer and it is made once, where it can still stop the process.

use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, VisionGrid};
use shiso_vision::{EncodedRows, MediaPlacement, PreprocessSpec, VisionError, VisionItem};

/// Why a request's media could not be turned into embedding rows.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TowerError {
    /// The tower named on the command line would not load.
    #[error("the vision tower at {path} could not be loaded")]
    Load {
        /// The path the operator named.
        path: String,
        /// What the loader refused it for.
        #[source]
        source: VisionError,
    },
    /// The tower and the language model disagree about the residual width.
    #[error(
        "the vision tower projects to {tower}-wide rows but this checkpoint's residual \
         stream is {model} wide; they are not a pair"
    )]
    Width {
        /// What the tower's merger produces.
        tower: u32,
        /// What the language model expects.
        model: u32,
    },
    /// The tower ran and refused an item, or the device did.
    #[error("a media item could not be encoded")]
    Encode(#[source] VisionError),
}

/// One request's encoder output, owned for as long as its prefill needs it.
///
/// The rows and the grids are one value because they are one fact: a block of
/// rows whose height nothing records is a block nobody can address, and a grid
/// list with no rows beside it describes a picture that was never encoded. The
/// prefill contract borrows both out of here together.
#[derive(Debug)]
pub struct EncodedMedia {
    rows: DeviceBuffer,
    grids: Vec<VisionGrid>,
    /// How many of the rows are audio soft tokens, for the usage block.
    audio_pads: u32,
}

impl EncodedMedia {
    /// Takes ownership of a gathered block.
    #[must_use]
    pub fn from_rows(encoded: EncodedRows) -> Self {
        Self {
            rows: encoded.rows,
            grids: encoded.grids,
            audio_pads: 0,
        }
    }

    /// The same block, recording that `pads` of its rows are audio.
    #[must_use]
    pub const fn with_audio_pads(mut self, pads: u32) -> Self {
        self.audio_pads = pads;
        self
    }

    /// Rows that are audio soft tokens.
    #[must_use]
    pub const fn audio_pads(&self) -> u32 {
        self.audio_pads
    }

    /// The whole block, `[sum of placeholder counts, hidden]` single precision.
    #[must_use]
    pub fn rows(&self) -> &DeviceBuffer {
        &self.rows
    }

    /// The items' post-merge grids, in the order the prompt refers to them.
    #[must_use]
    pub fn grids(&self) -> &[VisionGrid] {
        &self.grids
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

/// A loaded vision tower and the facts this server needs about it.
///
/// Held by the one thread that owns the model, because what it produces is
/// device memory that thread is about to splice into a forward pass.
#[derive(Debug)]
pub struct VisionTower {
    inner: Box<dyn shiso_vision::VisionTower>,
}

impl VisionTower {
    /// Takes ownership of a tower a family's opener produced.
    ///
    /// There is deliberately no path-taking constructor here any more: which
    /// container a family's tower lives in, and how its dimensions are read
    /// back out, is that family's business, and a second opener in the server
    /// would be a second answer that can disagree with the first.
    #[must_use]
    pub fn from_tower(inner: Box<dyn shiso_vision::VisionTower>) -> Self {
        Self { inner }
    }

    /// Checks this tower against the language model it was loaded beside.
    ///
    /// # Errors
    ///
    /// [`TowerError::Width`] when the merger's output width is not the model's
    /// residual width. The two are separate checkpoints and nothing else in the
    /// process compares them.
    pub fn check_against(&self, model_hidden: u32) -> Result<(), TowerError> {
        let tower = self.inner.out_hidden();
        if tower == model_hidden {
            Ok(())
        } else {
            Err(TowerError::Width {
                tower,
                model: model_hidden,
            })
        }
    }

    /// How this tower wants a picture turned into patch rows.
    #[must_use]
    pub fn preprocess(&self) -> &PreprocessSpec {
        self.inner.preprocess()
    }

    /// What this family's template writes for media, and what it expands to.
    #[must_use]
    pub fn placement(&self) -> MediaPlacement {
        self.inner.placement()
    }

    /// The merger's block side, which is what turns a patch grid into the token
    /// grid the prompt was rendered with.
    #[must_use]
    pub fn spatial_merge_size(&self) -> usize {
        self.inner.preprocess().geometry.spatial_merge_size()
    }

    /// Row width the tower projects to.
    #[must_use]
    pub fn out_hidden(&self) -> u32 {
        self.inner.out_hidden()
    }

    /// Device bytes the tower's weights occupy.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.inner.device_bytes()
    }

    /// Runs the tower over `items` and returns its block as the tower built it.
    ///
    /// # Errors
    ///
    /// [`TowerError::Encode`] when the tower refuses an item or the device
    /// fails under it.
    pub fn encode_rows(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, TowerError> {
        self.inner
            .encode(backend, items)
            .map_err(TowerError::Encode)
    }

    /// Runs the tower over every item of one request and gathers the result.
    ///
    /// # Errors
    ///
    /// [`TowerError::Encode`] when the tower refuses an item or the device
    /// fails under it.
    pub fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedMedia, TowerError> {
        let encoded = self
            .inner
            .encode(backend, items)
            .map_err(TowerError::Encode)?;
        Ok(EncodedMedia::from_rows(encoded))
    }
}
