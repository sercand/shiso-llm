//! The geometry and the caller-supplied bounds the preprocessor runs under.
//!
//! Two values decide what a picture becomes: the tower's patch geometry, which
//! comes from the checkpoint and is fixed for the life of a process, and the
//! resolution bound, which is per-deployment and per-request. They are separate
//! types because they have separate lifetimes and separate owners - the first
//! is read out of `config.json` once, the second rides in from the CLI - and
//! folding them together is how a request-scoped override ends up baked into a
//! process-scoped value.
//!
//! Both are validated on the way in rather than on every use. A
//! [`PatchGeometry`] that exists is one whose products cannot overflow and
//! whose divisors cannot be zero, so the pixel loops below it are plain
//! arithmetic with nothing to re-check.

use std::num::{NonZeroU64, NonZeroUsize};

use shiso_core::VisionConfig;

use crate::error::VisionError;

/// Largest value any single geometry field may take.
///
/// Real towers declare 16 for the patch edge and 2 for both the temporal and
/// spatial merge factors, so this is roughly sixteen times the largest number
/// anyone has shipped. It is not a taste judgement: with every field at or
/// under this bound, `patch_size * spatial_merge_size` and
/// `3 * temporal_patch_size * patch_size^2` both fit in 32 bits, which is what
/// makes [`PatchGeometry::grid_unit`] and [`PatchGeometry::patch_dim`] plain
/// multiplications instead of checked ones.
const MAX_GEOMETRY_FIELD: usize = 256;

/// Normalization statistics that map the byte range onto `[-1, 1]`.
///
/// Both families this crate preprocesses for want this mapping, and they arrive
/// at it differently - see [`PreprocessSpec::gemma4`], which is where that
/// difference is written down. The numbers live on [`PreprocessSpec`] rather
/// than as a constant the pixel loop reads so that a family wanting something
/// else does not have to edit this one.
const HALF: [f32; 3] = [0.5, 0.5, 0.5];

/// Gemma 4's spatial patch edge, in pixels.
const GEMMA4_PATCH: usize = 16;

/// Gemma 4 has no temporal patch axis: a frame is a frame.
const GEMMA4_TEMPORAL: usize = 1;

/// Gemma 4's pooling kernel, which plays the part `spatial_merge_size` plays
/// for the Qwen towers: the side of the patch block that folds into one token.
const GEMMA4_POOL: usize = 3;

/// Gemma 4's patch budget: 280 soft tokens, each covering a 3x3 patch block.
const GEMMA4_MAX_PATCHES: u32 = 2520;

/// Decoder allocation budget, in bytes, for one image.
///
/// The `image` crate's own default is 512 MiB. On a machine whose GPU and CPU
/// share one pool that is a per-request budget competing directly with the KV
/// cache, and the bytes arrive over HTTP from a caller the server has no reason
/// to trust. 192 MiB still admits an 8000x8000 RGB photograph.
const DEFAULT_DECODED_BYTES: u64 = 192 * 1024 * 1024;

/// The tower's patch geometry: how pixels become patch rows.
///
/// Every field is non-zero and bounded by construction. That is the whole
/// point of the type: the preprocessor divides by all three of them and
/// multiplies two pairs of them together, and a checkpoint whose
/// `vision_config` omits a key parses as `0`, so without this boundary the
/// zero reaches a division inside a request task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PatchGeometry {
    patch: NonZeroUsize,
    temporal: NonZeroUsize,
    merge: NonZeroUsize,
}

impl PatchGeometry {
    /// Checks a geometry triple and returns it, or names the field that failed.
    ///
    /// # Errors
    ///
    /// [`VisionError::GeometryZero`] when a field is zero, and
    /// [`VisionError::GeometryTooLarge`] when one is above
    /// [`MAX_GEOMETRY_FIELD`]. Both name the offending field, because "the
    /// vision config is invalid" sends the operator reading the whole file.
    pub fn new(
        patch_size: usize,
        temporal_patch_size: usize,
        spatial_merge_size: usize,
    ) -> Result<Self, VisionError> {
        Ok(Self {
            patch: check("patch_size", patch_size)?,
            temporal: check("temporal_patch_size", temporal_patch_size)?,
            merge: check("spatial_merge_size", spatial_merge_size)?,
        })
    }

    /// Spatial patch edge, in pixels.
    #[must_use]
    pub const fn patch_size(&self) -> usize {
        self.patch.get()
    }

    /// Frames fused into one patch. A still fills this axis by replicating
    /// itself, which is why a still and a one-group video are the same shape.
    #[must_use]
    pub const fn temporal_patch_size(&self) -> usize {
        self.temporal.get()
    }

    /// Side of the patch block the merger folds into one output token.
    #[must_use]
    pub const fn spatial_merge_size(&self) -> usize {
        self.merge.get()
    }

    /// Pixel edge both target sides must be a multiple of.
    ///
    /// Snapping to `patch_size` alone is not enough: the merger consumes whole
    /// `spatial_merge_size` blocks of patches, so a grid that is a patch
    /// multiple but not a merge multiple loses its last row and column to
    /// integer division, and the pad-run arithmetic downstream then describes
    /// a picture one block smaller than the one that was encoded.
    #[must_use]
    pub const fn grid_unit(&self) -> usize {
        self.patch.get() * self.merge.get()
    }

    /// Values in one patch row: `3 channels x temporal_patch_size x patch^2`.
    #[must_use]
    pub const fn patch_dim(&self) -> usize {
        3 * self.temporal.get() * self.patch.get() * self.patch.get()
    }
}

impl TryFrom<&VisionConfig> for PatchGeometry {
    type Error = VisionError;

    /// Narrows a parsed checkpoint config to the three numbers preprocessing
    /// actually reads. The tower's width, depth and head count decide nothing
    /// about how pixels are laid out, so they are deliberately not carried
    /// here - a preprocessor that refused a config because `depth` was odd
    /// would be refusing for a reason that has nothing to do with its job.
    fn try_from(cfg: &VisionConfig) -> Result<Self, Self::Error> {
        Self::new(
            widen("patch_size", cfg.patch_size)?,
            widen("temporal_patch_size", cfg.temporal_patch_size)?,
            widen("spatial_merge_size", cfg.spatial_merge_size)?,
        )
    }
}

/// Widens one config field to a `usize`, naming it if the target cannot hold it.
fn widen(field: &'static str, value: u32) -> Result<usize, VisionError> {
    usize::try_from(value).map_err(|_| VisionError::GeometryWidth { field })
}

/// Rejects a zero or oversized geometry field by name.
fn check(field: &'static str, value: usize) -> Result<NonZeroUsize, VisionError> {
    if value > MAX_GEOMETRY_FIELD {
        return Err(VisionError::GeometryTooLarge {
            field,
            value,
            max: MAX_GEOMETRY_FIELD,
        });
    }
    NonZeroUsize::new(value).ok_or(VisionError::GeometryZero { field })
}

/// Which filter the resize runs through.
///
/// One variant, deliberately. The `image` crate has exactly one cubic filter,
/// `FilterType::CatmullRom`, and the reference processors' `resample = 3` -
/// "bicubic" in the PIL numbering - maps onto it. The variant is named after
/// the filter this crate actually runs rather than after the reference's word
/// for it, because a `Bicubic` variant that resolved to Catmull-Rom would be a
/// name promising a filter the crate does not have. A second variant belongs
/// here when a tower needs a filter this crate cannot yet run, not before; the
/// field exists now so that the intent travels with the spec instead of being
/// a constant buried in the resize call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// The Catmull-Rom cubic, which `image` provides and which the reference
    /// image processors spell "bicubic".
    CatmullRom,
}

/// How a decoded frame's target size is chosen.
///
/// The two rules do not differ by a parameter, they differ in what they are
/// trying to achieve, which is why this is an enum and not a pair of bounds.
/// One fits the picture under a resolution ceiling and refuses to invent
/// pixels; the other spends a fixed patch budget and will scale a small picture
/// UP to spend it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeRule {
    /// The Qwen-family rule: the long side is held under the caller's ceiling,
    /// the area under the caller's bound, both sides are snapped to
    /// [`PatchGeometry::grid_unit`], and nothing is ever scaled up.
    GridUnit,
    /// transformers' `get_aspect_ratio_preserving_size`: the largest
    /// aspect-preserving size that costs at most `max_patches` patches and
    /// whose sides are whole multiples of `patch_size * spatial_merge_size`.
    ///
    /// `factor = sqrt(max_patches * patch^2 / (h * w))`, and each side is
    /// `floor(factor * side / side_mult) * side_mult`. Upscaling is allowed:
    /// a small picture is grown to spend its budget, which is the opposite of
    /// what [`ResizeRule::GridUnit`] does.
    ///
    /// A side can floor to zero, and the reference handles that rather than
    /// returning it. The whole rule runs over two ranges of
    /// `transformers/models/gemma4/image_processing_gemma4.py` (transformers
    /// 5.17.0): the sizing arithmetic at `:46-55`, restated above, and the
    /// fallback at `:64-76`, quoted below - with each `min(...)` call folded
    /// onto one line from the source's four-line spread, a paraphrase of
    /// that expression and nothing else:
    ///
    /// ```text
    /// max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    /// if target_height == 0:
    ///     target_height = side_mult
    ///     target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    /// elif target_width == 0:
    ///     target_width = side_mult
    ///     target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    /// ```
    ///
    /// Both sides flooring to zero is impossible for a frame with any pixels in
    /// it - the two ideal sides multiply out to the budget itself, so they
    /// cannot both be under one unit - and the reference raises there. This
    /// crate returns one unit on each side instead: the caller is a request
    /// task, and a decoded frame that reached here already has pixels.
    PatchBudget {
        /// Patches the target may cost, counted before pooling.
        max_patches: u32,
    },
}

/// Everything about turning a decoded frame into patch rows that varies by
/// tower family.
///
/// This exists because the resize rule turned out not to be a property of the
/// preprocessor. Two towers with the same patch size want different target
/// sizes for the same picture, and the difference is not a tuning constant -
/// one rule caps resolution and the other spends a budget. Hard-wiring either
/// one into the preprocessor makes the other family's numbers unreachable
/// without an `if` on the architecture string, which is the shape this type
/// exists to avoid: the tower supplies its spec, and the preprocessor runs it.
///
/// It is [`Clone`] rather than [`Copy`] only because it is passed by reference
/// everywhere and a `Copy` struct with two float triples invites being copied
/// per pixel.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PreprocessSpec {
    /// The tower's patch geometry.
    pub geometry: PatchGeometry,
    /// How the target size is chosen.
    pub resize: ResizeRule,
    /// Per-channel mean subtracted after the `1/255` rescale.
    pub mean: [f32; 3],
    /// Per-channel divisor applied after the mean is subtracted.
    pub std: [f32; 3],
    /// The filter the resize runs through.
    pub resample: Resample,
}

impl PreprocessSpec {
    /// The Qwen-family spec today's [`VisionConfig`] implies.
    ///
    /// This is the compatibility path, and it has to reproduce the previous
    /// behaviour exactly: the committed vision goldens were recorded under the
    /// grid-unit rule with the `0.5` statistics, and a spec that differed by
    /// one number here would re-record them rather than fail.
    ///
    /// # Errors
    ///
    /// [`VisionError::GeometryZero`] and its siblings, naming the field, when
    /// the checkpoint's `vision_config` cannot drive a preprocessor. Reported
    /// here rather than per-image, because the geometry is a property of the
    /// checkpoint and a server that accepted it at boot should not be
    /// discovering it is unusable on the first picture.
    pub fn from_vision_config(cfg: &VisionConfig) -> Result<Self, VisionError> {
        Ok(Self::qwen35(PatchGeometry::try_from(cfg)?))
    }

    /// The Qwen-family spec over a geometry already in hand.
    ///
    /// The tower reads its geometry out of its own container and the config
    /// document states the same three numbers, so both doors lead here rather
    /// than each writing the rule down. Two spellings of "grid unit, half
    /// statistics, Catmull-Rom" is two things to keep in step, and the one that
    /// drifted would not fail: it would re-record the goldens.
    #[must_use]
    pub const fn qwen35(geometry: PatchGeometry) -> Self {
        Self {
            geometry,
            resize: ResizeRule::GridUnit,
            mean: HALF,
            std: HALF,
            resample: Resample::CatmullRom,
        }
    }

    /// Gemma 4's spec: a 16-pixel patch, no temporal axis, a 3x3 pooler, and a
    /// budget of 280 soft tokens.
    ///
    /// ## The statistics are `0.5`, and `processor_config.json` disagreeing is
    /// not a reason to change them
    ///
    /// The shipped checkpoint's `processor_config.json` declares
    /// `do_normalize: false` with `image_mean` `[0, 0, 0]` and `image_std`
    /// `[1, 1, 1]`, and the `Gemma4ImageProcessor` class defaults in
    /// transformers 5.17.0 say the same. Read on its own that says a pixel
    /// should arrive in `[0, 1]`, and it is wrong to read it on its own: the
    /// reference does the `[-1, 1]` mapping INSIDE the vision tower, not in the
    /// processor. From transformers 5.17.0,
    /// `models/gemma4/modeling_gemma4.py:616`, under a comment that says so
    /// outright:
    ///
    /// ```text
    /// # Gemma4 applies no normalization and instead scales in model code
    /// pixel_values = 2 * (pixel_values - 0.5)
    /// ```
    ///
    /// mlx-vlm 0.7.1 does the same in two places,
    /// `mlx_vlm/models/gemma4/vision.py:319` and
    /// `mlx_vlm/models/gemma4/vision.py:328`:
    ///
    /// ```text
    /// patches = 2 * (patches - 0.5)
    /// pixel_values = 2 * (pixel_values - 0.5)
    /// ```
    ///
    /// shiso's Gemma tower has no such step, so the mapping is folded in here
    /// instead - `(x - 0.5) / 0.5` is exactly `2 * (x - 0.5)`, so the pixels
    /// the tower multiplies are the same either way. Anyone "correcting" these
    /// back to `[0, 0, 0]` and `[1, 1, 1]` from the processor config would halve
    /// the contrast of every picture, and nothing would fail: every shape still
    /// agrees and the model answers fluently about what it was shown.
    ///
    /// Written out here rather than read from a `VisionConfig` because a
    /// Gemma checkpoint's vision block does not carry a `temporal_patch_size`
    /// or a patch budget; Task 4's loader is what will fill this in from the
    /// file, and until then this is the honest place for the numbers.
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "the three geometry constants are checked against the accepted range at compile time, so the expect below cannot fire"
    )]
    pub fn gemma4() -> Self {
        // The geometry constructor is fallible for configs read off disk. These
        // three are literals inside the accepted range, checked at compile time
        // so the expect below is unreachable rather than merely unlikely.
        const {
            assert!(GEMMA4_PATCH > 0 && GEMMA4_PATCH <= MAX_GEOMETRY_FIELD);
            assert!(GEMMA4_TEMPORAL > 0 && GEMMA4_TEMPORAL <= MAX_GEOMETRY_FIELD);
            assert!(GEMMA4_POOL > 0 && GEMMA4_POOL <= MAX_GEOMETRY_FIELD);
        }
        Self {
            geometry: PatchGeometry::new(GEMMA4_PATCH, GEMMA4_TEMPORAL, GEMMA4_POOL)
                .expect("Gemma 4's geometry is three constants checked at compile time"),
            resize: ResizeRule::PatchBudget {
                max_patches: GEMMA4_MAX_PATCHES,
            },
            mean: HALF,
            std: HALF,
            resample: Resample::CatmullRom,
        }
    }
}

/// What one image is allowed to cost.
///
/// `max_pixels` is an AREA, not an edge length, matching the convention the
/// reference image processors use where `longest_edge` and `shortest_edge` are
/// both pixel counts despite their names. It comes from the checkpoint's
/// preprocessor config or from the operator's override, and the operator's
/// value wins.
///
/// An absent bound is [`Option::None`] rather than a sentinel zero, so "the
/// operator declared no cap" and "the operator declared a cap of nothing" are
/// not the same value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageLimits {
    max_pixels: Option<NonZeroU64>,
    max_decoded_bytes: NonZeroU64,
}

impl ImageLimits {
    /// The area bound, if one was declared.
    #[must_use]
    pub const fn max_pixels(&self) -> Option<NonZeroU64> {
        self.max_pixels
    }

    /// Bytes the decoder may hold at once while decoding one image.
    #[must_use]
    pub const fn max_decoded_bytes(&self) -> u64 {
        self.max_decoded_bytes.get()
    }

    /// Declares an area bound, in pixels.
    ///
    /// The bound REPLACES the fallback long-side clamp rather than combining
    /// with it. Combining is the trap: intersecting the two means a checkpoint
    /// that permits 4096x4096 can never exceed the fallback's long side, so
    /// the model's own declared bound can only ever lower the resolution and
    /// never raise it, and detail-bearing inputs - documents, charts, dense
    /// screenshots - pay for that with no log line to say so.
    #[must_use]
    pub const fn with_max_pixels(mut self, max_pixels: NonZeroU64) -> Self {
        self.max_pixels = Some(max_pixels);
        self
    }

    /// Drops any area bound, restoring the fallback long-side clamp.
    #[must_use]
    pub const fn without_pixel_bound(mut self) -> Self {
        self.max_pixels = None;
        self
    }

    /// Sets the decoder's allocation budget for one image.
    #[must_use]
    pub const fn with_max_decoded_bytes(mut self, bytes: NonZeroU64) -> Self {
        self.max_decoded_bytes = bytes;
        self
    }
}

impl Default for ImageLimits {
    /// No declared area bound, and the conservative allocation budget.
    ///
    /// The default is what a checkpoint that ships no preprocessor config gets,
    /// so it has to be the historically safe posture rather than the permissive
    /// one.
    fn default() -> Self {
        Self {
            max_pixels: None,
            max_decoded_bytes: NonZeroU64::new(DEFAULT_DECODED_BYTES).unwrap_or(NonZeroU64::MIN),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageLimits, MAX_GEOMETRY_FIELD, PatchGeometry};
    use crate::error::VisionError;

    #[test]
    fn a_zero_geometry_field_names_itself() {
        for (ps, tp, sms, want) in [
            (0usize, 2usize, 2usize, "patch_size"),
            (16, 0, 2, "temporal_patch_size"),
            (16, 2, 0, "spatial_merge_size"),
        ] {
            match PatchGeometry::new(ps, tp, sms) {
                Err(VisionError::GeometryZero { field }) => assert_eq!(field, want),
                other => panic!("expected a named zero-geometry error, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_oversized_geometry_field_is_refused_by_name() {
        let too_big = MAX_GEOMETRY_FIELD + 1;
        match PatchGeometry::new(too_big, 2, 2) {
            Err(VisionError::GeometryTooLarge { field, value, max }) => {
                assert_eq!(field, "patch_size");
                assert_eq!(value, too_big);
                assert_eq!(max, MAX_GEOMETRY_FIELD);
            }
            other => panic!("expected an oversized-geometry error, got {other:?}"),
        }
    }

    #[test]
    fn the_shipped_geometry_derives_its_grid_unit_and_row_length() {
        let geom = PatchGeometry::new(16, 2, 2).expect("the shipped geometry is valid");
        assert_eq!(geom.grid_unit(), 32);
        assert_eq!(geom.patch_dim(), 3 * 2 * 16 * 16);
    }

    /// Every accepted geometry must keep both products inside 32 bits; that is
    /// the invariant `MAX_GEOMETRY_FIELD` exists to buy, and it is checked at
    /// the extreme rather than assumed.
    #[test]
    fn the_field_bound_keeps_both_products_in_32_bits() {
        let geom = PatchGeometry::new(MAX_GEOMETRY_FIELD, MAX_GEOMETRY_FIELD, MAX_GEOMETRY_FIELD)
            .expect("the bound itself is accepted");
        assert!(u32::try_from(geom.grid_unit()).is_ok());
        assert!(u32::try_from(geom.patch_dim()).is_ok());
    }

    /// The media token ids are an SSOT the prompt renderer and the wire
    /// adapters both read, and nothing downstream can tell a wrong one from a
    /// right one: the wrong id produces a prompt with no pad run, which
    /// generates fluently about nothing. Pinned here against the recorded
    /// checkpoint config so a parser change has to break a test rather than a
    /// deployment.
    #[test]
    fn the_media_token_ids_come_from_the_checkpoint_config() {
        let vcfg = crate::testing::vision_config();
        assert_eq!(u32::from(vcfg.image_token_id), 248_056);
        assert_eq!(u32::from(vcfg.video_token_id), 248_057);
        assert_eq!(u32::from(vcfg.vision_start_token_id), 248_053);
        assert_eq!(u32::from(vcfg.vision_end_token_id), 248_054);
    }

    /// The shipped checkpoint's geometry must reach this crate intact: the
    /// preprocessor reads three numbers out of it and every token count below
    /// depends on all three.
    #[test]
    fn the_recorded_checkpoint_yields_the_shipped_geometry() {
        let vcfg = crate::testing::vision_config();
        let geom = PatchGeometry::try_from(&vcfg).expect("the recorded geometry is valid");
        assert_eq!(geom.patch_size(), 16);
        assert_eq!(geom.temporal_patch_size(), 2);
        assert_eq!(geom.spatial_merge_size(), 2);
        assert_eq!(geom.grid_unit(), 32);
    }

    #[test]
    fn the_default_limits_declare_no_area_bound() {
        let limits = ImageLimits::default();
        assert!(limits.max_pixels().is_none());
        assert_eq!(limits.max_decoded_bytes(), 192 * 1024 * 1024);
    }
}
