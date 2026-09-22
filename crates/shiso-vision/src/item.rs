//! The unit of vision input handed to the encoder.

use crate::error::VisionError;

/// Patch rows for one media item: a still image, or a video, ready for the
/// vision tower.
///
/// # One flat buffer, not a vector of vectors
///
/// The obvious shape for this is one `Vec<f32>` per temporal group. It is the
/// wrong one. A 1280x1280 image is roughly 6400 patch rows, so the nested form
/// is 6400 separate heap allocations that the encoder then has to walk and
/// gather before it can upload anything; flat plus a row stride uploads in a
/// single host-to-device copy and indexes by multiplication.
///
/// # Why the grid lives in the type
///
/// The pad run spliced into the prompt, the embedding rows the encoder returns
/// and the multimodal position stream all have to describe the same item. Carry
/// the grid in a parallel `groups_per_item` vector alongside the pixels and a
/// desync between them is silent - the request succeeds, the output is fluent,
/// and it is about the wrong picture. Here there is one buffer and one grid,
/// checked against each other once, at construction.
#[derive(Clone, PartialEq)]
#[non_exhaustive]
pub struct VisionItem {
    pixels: Box<[f32]>,
    row_len: usize,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
}

impl VisionItem {
    /// Binds a pixel buffer to the grid it describes, or refuses the pair.
    ///
    /// `pixels` holds `grid_t * grid_h * grid_w` rows of `row_len` values,
    /// laid out row-major: all of group 0's rows, then all of group 1's.
    ///
    /// # Errors
    ///
    /// [`VisionError::ItemEmpty`] when any extent is zero - an item that
    /// encodes to no rows is not a media reference - and
    /// [`VisionError::ItemShape`] when the buffer length disagrees with the
    /// grid. The second is the check this type exists for: it is the last
    /// place the shape can still be wrong loudly instead of quietly.
    pub fn new(
        pixels: Box<[f32]>,
        row_len: usize,
        grid_t: usize,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Self, VisionError> {
        if row_len == 0 || grid_t == 0 || grid_h == 0 || grid_w == 0 {
            return Err(VisionError::ItemEmpty {
                grid_t,
                grid_h,
                grid_w,
                row_len,
            });
        }
        let expected = grid_t
            .checked_mul(grid_h)
            .and_then(|n| n.checked_mul(grid_w))
            .and_then(|n| n.checked_mul(row_len));
        if expected != Some(pixels.len()) {
            return Err(VisionError::ItemShape {
                grid_t,
                grid_h,
                grid_w,
                row_len,
                expected: expected.unwrap_or(usize::MAX),
                got: pixels.len(),
            });
        }
        Ok(Self {
            pixels,
            row_len,
            grid_t,
            grid_h,
            grid_w,
        })
    }

    /// Every patch row, back to back, in the order the encoder consumes them.
    #[must_use]
    pub fn as_flat(&self) -> &[f32] {
        &self.pixels
    }

    /// One patch row, or `None` past the end.
    ///
    /// Bounds-checked rather than panicking: the index usually comes from a
    /// loop over [`VisionItem::rows`], but it can also come from a kernel
    /// dispatch that thinks the grid is a different shape, and that is exactly
    /// the case that should return nothing rather than read a neighbouring row.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.row_len)?;
        let end = start.checked_add(self.row_len)?;
        self.pixels.get(start..end)
    }

    /// Patch rows in the whole item, across every temporal group.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.grid_t * self.grid_h * self.grid_w
    }

    /// Values in one patch row: `3 channels x temporal_patch_size x patch^2`.
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.row_len
    }

    /// The pre-merge patch grid as `(t, h, w)`.
    #[must_use]
    pub fn grid(&self) -> (usize, usize, usize) {
        (self.grid_t, self.grid_h, self.grid_w)
    }

    /// Temporal extent, in groups. `1` for a still image.
    #[must_use]
    pub fn t_len(&self) -> usize {
        self.grid_t
    }

    /// Merged tokens this item occupies in the prompt.
    ///
    /// This is simultaneously the length of the item's pad run and the number
    /// of embedding rows the encoder will hand back for it. They are the same
    /// number by construction, and that identity is what the whole splice
    /// depends on - which is why there is one function rather than one on each
    /// side of the boundary.
    ///
    /// A zero merge size is clamped to one rather than dividing by it: the
    /// caller that passes zero has a broken config, and this is not where that
    /// is diagnosed.
    #[must_use]
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.grid_t * (self.grid_h / sms) * (self.grid_w / sms)
    }
}

/// Summarised rather than derived: the payload is megabytes of `f32`, and a
/// derived `Debug` dumps all of it into any log line or test failure that
/// happens to format one.
impl std::fmt::Debug for VisionItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionItem")
            .field("grid_t", &self.grid_t)
            .field("grid_h", &self.grid_h)
            .field("grid_w", &self.grid_w)
            .field("row_len", &self.row_len)
            .field("values", &self.pixels.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::VisionItem;
    use crate::error::VisionError;

    fn item(grid_t: usize, grid_h: usize, grid_w: usize, row_len: usize) -> VisionItem {
        let pixels = vec![0.0f32; grid_t * grid_h * grid_w * row_len].into_boxed_slice();
        VisionItem::new(pixels, row_len, grid_t, grid_h, grid_w).expect("a consistent grid")
    }

    /// The standard 2x2 merger folds a patch block into one embedding token.
    #[test]
    fn pad_count_uses_the_spatially_merged_grid() {
        assert_eq!(item(1, 64, 64, 1).pad_count(2), 32 * 32);
        assert_eq!(item(1, 40, 80, 1).pad_count(2), 20 * 40);
    }

    #[test]
    fn pad_count_is_identity_without_merging() {
        assert_eq!(item(1, 64, 64, 1).pad_count(1), 64 * 64);
        assert_eq!(item(1, 8, 12, 1).pad_count(1), 96);
    }

    #[test]
    fn pad_count_treats_a_zero_merge_as_one() {
        assert_eq!(item(1, 64, 64, 1).pad_count(0), 64 * 64);
    }

    /// Integer division truncates: 65/2 is 32, not 33. A partial merge block
    /// has no token, and rounding it up would put the pad run one ahead of the
    /// rows the encoder returns.
    #[test]
    fn pad_count_floors_partial_merge_blocks() {
        assert_eq!(item(1, 65, 64, 1).pad_count(2), 32 * 32);
    }

    /// A video is the same type with more than one group, so its token count
    /// is the still's multiplied by the temporal extent - not a second
    /// formula that can drift from this one.
    #[test]
    fn a_multi_group_item_multiplies_its_merged_plane() {
        let video = item(4, 4, 4, 1);
        assert_eq!(video.t_len(), 4);
        assert_eq!(video.pad_count(2), 4 * 4);
        assert_eq!(video.pad_count(1), 4 * 16);
    }

    #[test]
    fn a_buffer_that_does_not_match_its_grid_is_refused() {
        let pixels = vec![0.0f32; 10].into_boxed_slice();
        match VisionItem::new(pixels, 4, 1, 2, 2) {
            Err(VisionError::ItemShape { expected, got, .. }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 10);
            }
            other => panic!("expected a shape mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_zero_extent_is_not_an_item() {
        let empty = Vec::new().into_boxed_slice();
        assert!(matches!(
            VisionItem::new(empty, 4, 1, 0, 2),
            Err(VisionError::ItemEmpty { .. })
        ));
    }

    /// Rows are addressed by stride, and the last one ends exactly at the end
    /// of the buffer - the off-by-one that would read into a neighbour.
    #[test]
    fn rows_are_addressed_by_stride_and_stop_at_the_end() {
        let pixels: Vec<f32> = (0u8..12).map(f32::from).collect();
        let item = VisionItem::new(pixels.into_boxed_slice(), 3, 2, 1, 2)
            .expect("12 values as 4 rows of 3");
        assert_eq!(item.rows(), 4);
        assert_eq!(item.row(0), Some(&[0.0, 1.0, 2.0][..]));
        assert_eq!(item.row(3), Some(&[9.0, 10.0, 11.0][..]));
        assert_eq!(item.row(4), None);
        assert_eq!(item.as_flat().len(), 12);
        assert_eq!(item.grid(), (2, 1, 2));
    }

    /// The payload must never reach a log line or a failure message.
    #[test]
    fn debug_summarises_rather_than_dumping_the_payload() {
        let text = format!("{:?}", item(1, 2, 2, 8));
        assert!(text.contains("values: 32"), "{text}");
        assert!(
            !text.contains("0.0"),
            "the payload leaked into Debug: {text}"
        );
    }
}
