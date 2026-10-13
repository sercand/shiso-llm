//! The LFM2 vision-language family's vision tower.
//!
//! A `SigLIP2` "`NaFlex`" encoder - a patch projection, a learned 16 x 16
//! position grid resampled onto each picture's own patch grid, 27 pre-norm
//! blocks with biased projections and a tanh-GELU feed-forward, a final layer
//! norm - followed by the family's projector: a 2 x 2 pixel unshuffle that
//! folds four neighbouring patches into one row, then two biased linears with
//! the exact GELU between them, into the language model's width.
//!
//! [`loader`] reads the `clip` mmproj whose projector type is `lfm2`,
//! [`encoder`] runs the graph, [`positions`] holds the one piece of host
//! arithmetic (the position grid's antialiased resize), and [`Lfm2Tower`] is
//! the object the server holds behind [`shiso_vision::VisionTower`].
//!
//! # One tile per picture
//!
//! The family's processor has two layouts. A picture that fits the token band
//! is resized into it and becomes one run of soft tokens. A picture larger
//! than about twice the band's top is instead cut into 512-pixel tiles, each
//! encoded on its own, and a downscaled thumbnail of the whole is appended; the
//! prompt then interleaves a `<|img_row_R_col_C|>` marker before each tile's
//! run and `<|img_thumbnail|>` before the thumbnail's.
//!
//! This tower implements the first layout only. A picture the processor would
//! tile is resized into the band like any other, which is exactly the
//! thumbnail the tiled layout would have carried - the same picture, at the
//! same resolution, without the tiles beside it. It is the processor's own
//! `do_image_splitting = false` behaviour, a mode the checkpoint is trained to
//! read, not an approximation of the tiled one. What it costs is detail on
//! large pictures: a 2048 x 2048 photograph reaches the model as 512 x 512.
//! The tiled layout is several runs per picture with markers between them,
//! which the server's one-marker-one-run placement cannot express.

mod encoder;
mod loader;
mod positions;
#[cfg(test)]
mod tests;

pub use encoder::Lfm2Tower;

/// The projector type a `clip` mmproj of this family declares.
const PROJECTOR: &str = "lfm2";

/// Fewest merged tokens one picture may cost: the processor's
/// `min_image_tokens`. Not in the mmproj, which states the patch and the
/// unshuffle factor and nothing about the band, so it is this family's fact,
/// stated once.
pub(super) const MIN_IMAGE_TOKENS: u32 = 64;

/// Most merged tokens one picture may cost: the processor's
/// `max_image_tokens`.
pub(super) const MAX_IMAGE_TOKENS: u32 = 256;

/// The tower's widths and the one scalar its normalizations need.
///
/// Every number is read from the container's own metadata or from a tensor's
/// own shape; nothing here has a default, because a tower that loads on a
/// guessed width is a tower whose every later shape check agrees with the
/// guess.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Lfm2Dims {
    /// Transformer blocks.
    pub(super) depth: usize,
    /// Values in one residual row.
    pub(super) hidden: usize,
    /// Attention heads.
    pub(super) heads: usize,
    /// Values in one head: `hidden / heads`.
    pub(super) head_dim: usize,
    /// Width of one head's slot in the padded attention planes: `head_dim`
    /// rounded up to the tiled attention's 32-value slice. See the loader for
    /// why the padding is exact.
    pub(super) head_slot: usize,
    /// Values in one feed-forward row.
    pub(super) intermediate: usize,
    /// Width of the projector's hidden layer and of the rows the language
    /// model consumes; the shipped projector uses one width for both.
    pub(super) out_hidden: usize,
    /// Side of the patch the preprocessor cuts.
    pub(super) patch: usize,
    /// Values in one patch row: `3 x patch x patch`.
    pub(super) patch_dim: usize,
    /// Side of the patch block the unshuffle folds into one row.
    pub(super) merge: usize,
    /// Side of the learned position grid.
    pub(super) pos_side: usize,
    /// Epsilon inside every layer norm.
    pub(super) eps: f32,
}

impl Lfm2Dims {
    /// Width of one unshuffled row: `merge^2` patch rows side by side.
    pub(super) const fn merged(&self) -> usize {
        self.merge * self.merge * self.hidden
    }
}
