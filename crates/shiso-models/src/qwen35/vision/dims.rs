//! The tower's dimensions, derived once from the checkpoint.
//!
//! Every shape in the encoder is a function of these ten numbers, and nothing
//! below this type is allowed to carry a literal. That is not style: the two
//! towers this crate loads have different widths, depths and feed-forward
//! sizes, so a hard-coded width would not be a shortcut, it would be a loader
//! that silently reads the wrong bytes out of the other checkpoint.
//!
//! The derived quantities live here too - the head width, the merger's input
//! width, the fused projection width, the position grid - rather than being
//! recomputed at each use. Each is a division or a product that can fail on a
//! malformed config, and doing them once, at the boundary, is what turns "the
//! head count does not divide the width" into a load error naming both numbers
//! instead of a silently truncated head.

use shiso_core::VisionConfig;

use shiso_vision::PatchGeometry;
use shiso_vision::VisionError;

/// Query, key and value, fused into one projection.
///
/// Named rather than written as `3` at the two places it is used: the fused
/// projection's width is the reason the tower's `qkv` tensor is three times the
/// hidden width, and a reader who meets a bare `3` next to a `1024` has to
/// guess which of the two it belongs to.
const QKV_PARTS: usize = 3;

/// Geometry of one vision tower.
///
/// Built only through [`VisionDims::try_from`], so a value that exists has
/// already had every division checked. There is no setter and no default: a
/// tower whose dimensions were half-filled and then corrected is exactly the
/// state this crate cannot detect later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct VisionDims {
    depth: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    intermediate: usize,
    out_hidden: usize,
    position_embeddings: usize,
    position_grid: usize,
    geometry: PatchGeometry,
}

impl VisionDims {
    /// Transformer blocks in the tower.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Width inside the tower.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    /// Attention heads per block.
    #[must_use]
    pub const fn heads(&self) -> usize {
        self.heads
    }

    /// Width of one attention head. Derived, never read from the config: a
    /// config that declared a head width disagreeing with `hidden / heads`
    /// would give two answers to one question.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Feed-forward inner width.
    #[must_use]
    pub const fn intermediate(&self) -> usize {
        self.intermediate
    }

    /// Width of the vectors the merger hands to the language model.
    #[must_use]
    pub const fn out_hidden(&self) -> usize {
        self.out_hidden
    }

    /// Rows in the learned position table.
    #[must_use]
    pub const fn position_embeddings(&self) -> usize {
        self.position_embeddings
    }

    /// Side of the square position grid the table is interpolated over.
    ///
    /// The table is stored flat, but it is a square grid of learned positions
    /// that the loader resamples to whatever grid an image actually produced.
    /// A table whose length is not a square has no grid to resample, which is
    /// why the side is computed and checked at load rather than assumed.
    #[must_use]
    pub const fn position_grid(&self) -> usize {
        self.position_grid
    }

    /// How pixels become patch rows.
    #[must_use]
    pub const fn geometry(&self) -> PatchGeometry {
        self.geometry
    }

    /// Values in one patch row, which is the patch projection's input width.
    #[must_use]
    pub const fn patch_dim(&self) -> usize {
        self.geometry.patch_dim()
    }

    /// Width of the fused query/key/value projection.
    #[must_use]
    pub const fn qkv_width(&self) -> usize {
        QKV_PARTS * self.hidden
    }

    /// Input width of the merger.
    ///
    /// The merger folds a `spatial_merge_size` square block of patch vectors
    /// into one token, so it consumes that many hidden vectors at once. This is
    /// the number that makes the shipped tower's merger 4096 wide off a 1024
    /// hidden width, and it is derived from the merge factor rather than
    /// written down, because the other tower's is not 4096.
    #[must_use]
    pub const fn merge_input(&self) -> usize {
        let block = self.geometry.spatial_merge_size();
        self.hidden * block * block
    }

    /// Tensors a complete tower of these dimensions holds.
    ///
    /// Twelve per block - two normalizations, a fused projection, an output
    /// projection and two feed-forward matrices, each with a bias - plus the
    /// patch projection, the position table and the merger's three layers.
    /// Derived rather than recorded, so it fails when a block's worth of
    /// tensors is missing AND when the block layout is not the one the rest of
    /// this loader assumed.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.depth * TENSORS_PER_BLOCK + TENSORS_OUTSIDE_BLOCKS
    }

    /// The same tower with its feed-forward inner width padded to `width`.
    ///
    /// For a container that quantized the feed-forward in 128-wide blocks and
    /// so stores it padded: the padded projections are what the weights
    /// were quantized as, and running them at the stored width is the exact
    /// computation, where cutting them back would drop the share of the
    /// quantization error the padding carries.
    ///
    /// # Errors
    ///
    /// [`VisionError::TowerTensorExtent`] when `width` is narrower than the
    /// configured width.
    pub fn with_padded_intermediate(self, width: usize) -> Result<Self, VisionError> {
        if width < self.intermediate {
            return Err(VisionError::TowerTensorExtent {
                name: "mlp.linear_fc1".to_owned(),
                got: width,
                want: self.intermediate,
            });
        }
        Ok(Self {
            intermediate: width,
            ..self
        })
    }
}

/// Tensors one transformer block holds; see [`VisionDims::tensor_count`].
pub(crate) const TENSORS_PER_BLOCK: usize = 12;

/// Tensors outside the blocks: the patch projection and its bias, the position
/// table, and the merger's normalization plus two layers with their biases.
pub(crate) const TENSORS_OUTSIDE_BLOCKS: usize = 9;

impl TryFrom<&VisionConfig> for VisionDims {
    type Error = VisionError;

    /// Derives the tower's dimensions from a parsed checkpoint config.
    ///
    /// # Errors
    ///
    /// [`VisionError::TowerFieldZero`] for a missing or zero field,
    /// [`VisionError::TowerNotDivisible`] when the head count does not divide
    /// the width, and [`VisionError::PositionGridNotSquare`] when the position
    /// table is not a square grid. Each names the numbers involved, because a
    /// tower that is off by one head is indistinguishable from a correct one
    /// once the weights are on the device.
    fn try_from(cfg: &VisionConfig) -> Result<Self, Self::Error> {
        let geometry = PatchGeometry::try_from(cfg)?;
        Self::new(
            widen("depth", cfg.depth)?,
            widen("hidden_size", cfg.hidden)?,
            widen("num_heads", cfg.num_heads)?,
            widen("intermediate_size", cfg.intermediate_size)?,
            widen("out_hidden_size", cfg.out_hidden)?,
            widen("num_position_embeddings", cfg.num_position_embeddings)?,
            geometry,
        )
    }
}

impl VisionDims {
    /// Checks and assembles a tower's dimensions.
    ///
    /// Public so a container that carries its dimensions as metadata rather
    /// than as a config document reaches the same checks: the sidecar tower and
    /// the checkpoint tower must not validate differently.
    ///
    /// # Errors
    ///
    /// See [`VisionDims::try_from`].
    pub fn new(
        depth: usize,
        hidden: usize,
        heads: usize,
        intermediate: usize,
        out_hidden: usize,
        position_embeddings: usize,
        geometry: PatchGeometry,
    ) -> Result<Self, VisionError> {
        for (field, value) in [
            ("depth", depth),
            ("hidden_size", hidden),
            ("num_heads", heads),
            ("intermediate_size", intermediate),
            ("out_hidden_size", out_hidden),
            ("num_position_embeddings", position_embeddings),
        ] {
            if value == 0 {
                return Err(VisionError::TowerFieldZero { field });
            }
        }
        if !hidden.is_multiple_of(heads) {
            return Err(VisionError::TowerNotDivisible {
                field: "hidden_size",
                value: hidden,
                divisor_field: "num_heads",
                divisor: heads,
            });
        }
        let position_grid = position_embeddings.isqrt();
        if position_grid * position_grid != position_embeddings {
            return Err(VisionError::PositionGridNotSquare {
                count: position_embeddings,
            });
        }
        Ok(Self {
            depth,
            hidden,
            heads,
            head_dim: hidden / heads,
            intermediate,
            out_hidden,
            position_embeddings,
            position_grid,
            geometry,
        })
    }
}

/// Widens one config field, naming it if the target cannot hold it.
fn widen(field: &'static str, value: u32) -> Result<usize, VisionError> {
    usize::try_from(value).map_err(|_| VisionError::GeometryWidth { field })
}

#[cfg(test)]
mod tests {
    use super::VisionDims;
    use shiso_vision::PatchGeometry;
    use shiso_vision::VisionError;

    fn geometry() -> PatchGeometry {
        PatchGeometry::new(16, 2, 2).expect("the shipped geometry")
    }

    /// The recorded checkpoint is the only statement of these numbers that is
    /// not this crate's own; everything downstream is derived from it.
    #[test]
    fn the_recorded_checkpoint_yields_the_shipped_tower() {
        let vcfg = crate::qwen35::vision::testing::vision_config();
        let dims = VisionDims::try_from(&vcfg).expect("the recorded tower is valid");
        assert_eq!(dims.depth(), 24);
        assert_eq!(dims.hidden(), 1024);
        assert_eq!(dims.heads(), 16);
        assert_eq!(dims.head_dim(), 64);
        assert_eq!(dims.intermediate(), 4096);
        assert_eq!(dims.out_hidden(), 2560);
        assert_eq!(dims.position_embeddings(), 2304);
        assert_eq!(dims.position_grid(), 48, "2304 is a 48x48 grid");
    }

    /// The derived widths are what the tensor shapes are checked against, so
    /// they are pinned against the measured checkpoint rather than against the
    /// formula that produced them.
    #[test]
    fn the_derived_widths_match_the_measured_tower() {
        let dims =
            VisionDims::try_from(&crate::qwen35::vision::testing::vision_config()).expect("valid");
        assert_eq!(dims.patch_dim(), 1536, "3 channels x 2 frames x 16x16");
        assert_eq!(dims.qkv_width(), 3072);
        assert_eq!(
            dims.merge_input(),
            4096,
            "1024 hidden folded over a 2x2 block"
        );
        assert_eq!(dims.tensor_count(), 24 * 12 + 9);
        assert_eq!(dims.tensor_count(), 297);
    }

    /// A head count that does not divide the width truncates every head by a
    /// fraction, and nothing downstream can see that it happened.
    #[test]
    fn a_head_count_that_does_not_divide_the_width_is_refused() {
        match VisionDims::new(24, 1024, 15, 4096, 2560, 2304, geometry()) {
            Err(VisionError::TowerNotDivisible {
                field,
                value,
                divisor,
                ..
            }) => {
                assert_eq!(field, "hidden_size");
                assert_eq!((value, divisor), (1024, 15));
            }
            other => panic!("expected a divisibility error, got {other:?}"),
        }
    }

    /// The position table is resampled as a square grid, so a length that is
    /// not a square has no grid to resample.
    #[test]
    fn a_position_table_that_is_not_square_is_refused() {
        assert!(matches!(
            VisionDims::new(24, 1024, 16, 4096, 2560, 2305, geometry()),
            Err(VisionError::PositionGridNotSquare { count: 2305 })
        ));
    }

    #[test]
    fn a_zero_tower_field_names_itself() {
        for (depth, hidden, heads, want) in [
            (0usize, 1024usize, 16usize, "depth"),
            (24, 0, 16, "hidden_size"),
            (24, 1024, 0, "num_heads"),
        ] {
            match VisionDims::new(depth, hidden, heads, 4096, 2560, 2304, geometry()) {
                Err(VisionError::TowerFieldZero { field }) => assert_eq!(field, want),
                other => panic!("expected a named zero-field error, got {other:?}"),
            }
        }
    }

    /// A second tower shape must derive different widths from the same code;
    /// the merger width in particular is the one a hard-coded 4096 would break.
    #[test]
    fn a_wider_tower_derives_its_own_widths() {
        let dims = VisionDims::new(27, 1152, 16, 4304, 2560, 2304, geometry())
            .expect("the sidecar tower shape is valid");
        assert_eq!(dims.head_dim(), 72);
        assert_eq!(dims.qkv_width(), 3456);
        assert_eq!(dims.merge_input(), 4608);
        assert_eq!(dims.tensor_count(), 27 * 12 + 9);
        assert_eq!(dims.tensor_count(), 333);
    }
}
