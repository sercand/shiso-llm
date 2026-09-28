//! How a model family spells its tensors in a safetensors export.
//!
//! A safetensors export can name tensors arbitrarily. This module defines the
//! [`NameMap`] trait and role descriptors so that the container loader remains
//! family-agnostic and never needs hard-coded tensor names.

use std::fmt::Debug;

use crate::reader::{GgufError, GgufResult};

/// Transform applied to a tensor's values on the way in.
///
/// Written as an enum rather than a flag so that adding a second one cannot be
/// done by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueOp {
    /// Widen to `f32` and change nothing else.
    Keep,
    /// Widen to `f32`, then replace each value `x` with `-exp(x)`.
    NegExp,
}

/// Which axis of a tensor is indexed by value head, and where along it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadAxis {
    /// The leading axis is exactly one entry per value head.
    PerHeadRows,
    /// The leading axis is `value_heads * value_head_dim` channels and nothing else.
    ValueRows,
    /// The leading axis is the fused `[q | k | v]` channel list, of which only
    /// the trailing `v` run is indexed by value head.
    QkvTailRows,
    /// Every row's trailing axis is `value_heads * value_head_dim` values.
    ValueColumns,
}

/// One translated tensor: what to call it and what to do to its values.
#[derive(Debug, Clone)]
pub struct Role {
    /// Name the binding layer looks the tensor up by.
    pub gguf: String,
    /// Value transform, meaningful only for unpacked tensors.
    pub op: ValueOp,
    /// Axis carrying the value-head index, `None` for every tensor the
    /// recurrent block's head order does not reach.
    pub heads: Option<HeadAxis>,
}

impl Role {
    /// A role whose values pass through untouched.
    #[must_use]
    pub fn plain(gguf: impl Into<String>) -> Self {
        Self {
            gguf: gguf.into(),
            op: ValueOp::Keep,
            heads: None,
        }
    }
}

/// How a family spells its tensors in a safetensors export, mapped onto the
/// GGUF role names the binders read.
pub trait NameMap: Send + Sync + Debug {
    /// The GGUF role for `name`, or `None` when the family does not import it.
    fn map_role(&self, name: &str) -> Option<Role>;

    /// How many tensors a complete export of this family translates to, for a
    /// checkpoint of `n_layers` slots with an attention layer every
    /// `full_attention_interval` of them.
    ///
    /// The loader refuses a directory that translates to any other number, so
    /// this is what turns "every tensor we looked for was there" into "there
    /// was nothing else there we should have looked for". A family that cannot
    /// state the count from those two numbers alone answers `None` and is
    /// checked by its binder instead; the default is `None`.
    ///
    /// `full_attention_interval` of zero means the family has no attention
    /// layers, which is how a purely recurrent variant is described.
    fn expected_tensors(&self, n_layers: u32, full_attention_interval: u32) -> Option<usize> {
        let _ = (n_layers, full_attention_interval);
        None
    }

    /// Translated names an export may or may not carry, left out of
    /// [`NameMap::expected_tensors`]: an output projection, for a family whose
    /// checkpoints tie it to the embedding in one size and store it in another.
    fn optional_tensors(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Identity name map that maps every tensor name to itself with no transformations.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityNames;

impl NameMap for IdentityNames {
    fn map_role(&self, name: &str) -> Option<Role> {
        Some(Role::plain(name))
    }
}

/// Shape of the recurrent block's head structure, read from the config.
///
/// Public because the reordering below is a fact about the ARCHITECTURE and
/// not about a container: every export of this family stores its value heads
/// the way `transformers` builds them - `repeat_interleave`, so value head `h`
/// belongs to key head `h / group` - and every kernel in this tree reads key
/// head `h % key_heads`. Two containers in this crate now have to apply the
/// same permutation, and a second implementation of it would be a second place
/// for a 32-head model to answer plausibly and wrongly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GdnHeads {
    /// Value heads per recurrent block.
    pub value_heads: usize,
    /// Query and key heads per recurrent block, which each value head shares.
    pub key_heads: usize,
    /// Channels in one value head.
    pub value_head_dim: usize,
    /// Channels in one query or key head.
    pub key_head_dim: usize,
}

impl GdnHeads {
    /// The four widths of a recurrent block's head structure.
    #[must_use]
    pub const fn new(
        value_heads: usize,
        key_heads: usize,
        value_head_dim: usize,
        key_head_dim: usize,
    ) -> Self {
        Self {
            value_heads,
            key_heads,
            value_head_dim,
            key_head_dim,
        }
    }

    /// Value heads that share one query/key head.
    fn group(self, tensor: &str) -> GgufResult<usize> {
        if self.key_heads == 0 || !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(GgufError::LayoutMismatch {
                tensor: tensor.into(),
                want: Box::from("value heads that divide among the key heads"),
                got: Box::from(format!(
                    "{} value heads over {} key heads",
                    self.value_heads, self.key_heads
                )),
            });
        }
        Ok(self.value_heads / self.key_heads)
    }

    /// Channels the fused projection spends on queries and keys together.
    fn qk_width(self) -> usize {
        2 * self.key_heads * self.key_head_dim
    }

    /// Checkpoint value head holding the one the binding layer calls `index`.
    fn source_head(self, index: usize, group: usize) -> usize {
        let key = index % self.key_heads;
        let member = index / self.key_heads;
        key * group + member
    }
}

/// A value-head reordering, resolved against one tensor's shape.
///
/// **What it is for.** `transformers` builds the recurrent block's query and
/// key heads with `repeat_interleave`, so checkpoint value head `k * group + m`
/// reads key head `k`; every kernel in this tree reads key head
/// `h % key_heads`. The two disagree for any model with more value heads than
/// key heads, and they disagree by a PERMUTATION - which leaves every
/// magnitude, every norm and every histogram in the model exactly where it was
/// and changes what the answer means. A stack loaded without this runs, is
/// finite everywhere, and answers fluent nonsense.
#[derive(Debug, Clone, Copy)]
pub struct HeadPermute {
    /// Independent runs, each holding one copy of the permuted region.
    outer: usize,
    /// Values at the start of each run that are not indexed by value head.
    prefix: usize,
    /// Values one value head owns.
    block: usize,
    /// Head structure the reordering is derived from.
    heads: GdnHeads,
    /// Value heads sharing one query/key head, checked once at construction.
    group: usize,
}

impl HeadPermute {
    /// Resolves `axis` against a decoded tensor of `rows` entries of `cols` values each.
    /// # Errors
    ///
    /// [`GgufError::LayoutMismatch`] when the shape is not the one `axis`
    /// describes, and when the value heads do not divide among the key heads.
    pub fn new(
        axis: HeadAxis,
        heads: GdnHeads,
        rows: usize,
        cols: usize,
        tensor: &str,
    ) -> GgufResult<Self> {
        let group = heads.group(tensor)?;
        let value_width = heads.value_heads * heads.value_head_dim;
        let (outer, prefix, block) = match axis {
            HeadAxis::PerHeadRows => {
                expect(
                    tensor,
                    rows == heads.value_heads,
                    "one row per value head",
                    rows,
                )?;
                (1, 0, cols)
            }
            HeadAxis::ValueRows => {
                expect(
                    tensor,
                    rows == value_width,
                    "one row per value channel",
                    rows,
                )?;
                (1, 0, heads.value_head_dim * cols)
            }
            HeadAxis::QkvTailRows => {
                let want = heads.qk_width() + value_width;
                expect(tensor, rows == want, "query, key and value rows", rows)?;
                (1, heads.qk_width() * cols, heads.value_head_dim * cols)
            }
            HeadAxis::ValueColumns => {
                expect(
                    tensor,
                    cols == value_width,
                    "one column per value channel",
                    cols,
                )?;
                (rows, 0, heads.value_head_dim)
            }
        };
        Ok(Self {
            outer,
            prefix,
            block,
            heads,
            group,
        })
    }

    /// Reorders `values` in place, or reports that it is not the size the
    /// geometry implies.
    ///
    /// Generic over `Copy` on purpose: the element it moves is whatever the
    /// container stores. An MLX export hands it decoded `f32`; a `modelopt`
    /// one hands it the raw FP8 bytes, because permuting rows of a matrix
    /// whose elements are one byte each is the same operation on the bytes.
    ///
    /// # Errors
    ///
    /// [`GgufError::SizeMismatch`] when `values` is not the length the
    /// geometry implies.
    pub fn apply<T: Copy>(&self, values: &mut [T], tensor: &str) -> GgufResult<()> {
        let region = self.heads.value_heads * self.block;
        let stride = self.prefix + region;
        let want = self.outer * stride;
        if values.len() != want {
            return Err(GgufError::SizeMismatch {
                tensor: tensor.into(),
                want,
                got: values.len(),
            });
        }

        let mut scratch = vec![values[0]; region];
        for outer in 0..self.outer {
            let start = outer * stride + self.prefix;
            let target = &mut values[start..start + region];
            scratch.copy_from_slice(target);
            for dest in 0..self.heads.value_heads {
                let src = self.heads.source_head(dest, self.group);
                let from = src * self.block;
                let to = dest * self.block;
                target[to..to + self.block].copy_from_slice(&scratch[from..from + self.block]);
            }
        }
        Ok(())
    }
}

/// Reports a shape that is not the one a [`HeadAxis`] variant describes.
fn expect(tensor: &str, ok: bool, want: &str, got: usize) -> GgufResult<()> {
    if ok {
        return Ok(());
    }
    Err(GgufError::LayoutMismatch {
        tensor: tensor.into(),
        want: Box::from(want),
        got: Box::from(format!("{got}")),
    })
}

#[cfg(test)]
mod tests {
    //! Granularity, written down rather than inferred.
    //!
    //! The value-head reordering is invisible in every shape it touches, so a
    //! permutation applied per scalar where it should be per head block, or on
    //! the columns where it should be on the rows, loads exactly as cleanly as
    //! the right one. These cases spell out the expected output for a tiny
    //! model - four value heads over two key heads, three channels each - so
    //! that each axis is pinned by an answer written in advance.

    use super::{GdnHeads, HeadAxis, HeadPermute};
    use crate::reader::GgufError;

    /// Four value heads over two key heads, three channels per head.
    const HEADS: GdnHeads = GdnHeads {
        value_heads: 4,
        key_heads: 2,
        value_head_dim: 3,
        key_head_dim: 3,
    };

    /// Permutes `values`, failing the test with the loader's own message.
    fn permuted(axis: HeadAxis, rows: usize, cols: usize, values: &[u32]) -> Vec<u32> {
        let permute = HeadPermute::new(axis, HEADS, rows, cols, "tensor")
            .expect("the shape is the one the axis describes");
        let mut values = values.to_vec();
        permute
            .apply(&mut values, "tensor")
            .expect("the buffer is the size the geometry implies");
        values
    }

    #[test]
    fn heads_are_de_interleaved_pairwise() {
        // The checkpoint keeps the pair belonging to key head 0 at positions
        // 0 and 1; the binding layer wants the first of every pair first, so
        // the second of key head 0 - checkpoint position 1 - lands at 2.
        assert_eq!(
            permuted(HeadAxis::PerHeadRows, 4, 1, &[0, 1, 2, 3]),
            [0, 2, 1, 3]
        );
    }

    #[test]
    fn a_per_head_row_moves_whole() {
        // Same permutation, but each head now owns a two-value row. Moving
        // scalars here would interleave halves of different heads.
        assert_eq!(
            permuted(HeadAxis::PerHeadRows, 4, 2, &[0, 1, 10, 11, 20, 21, 30, 31]),
            [0, 1, 20, 21, 10, 11, 30, 31]
        );
    }

    #[test]
    fn value_rows_move_in_blocks_of_a_head() {
        // Twelve channels, three per head: the block size is the head width,
        // not one row.
        assert_eq!(
            permuted(
                HeadAxis::ValueRows,
                12,
                1,
                &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ),
            [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
        );
    }

    #[test]
    fn the_query_and_key_rows_are_left_alone() {
        // Twelve query and key channels ahead of the twelve value ones. The
        // prefix is not indexed by value head and must come through untouched.
        let mut input: Vec<u32> = (0..24).collect();
        let want: Vec<u32> = (0..12)
            .chain([12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23])
            .collect();
        input = permuted(HeadAxis::QkvTailRows, 24, 1, &input);
        assert_eq!(input, want);
    }

    #[test]
    fn output_columns_are_permuted_within_every_row() {
        // The output projection consumes the heads, so the index is on its
        // columns: two rows of twelve, each rearranged the same way.
        let input: Vec<u32> = (0..24).collect();
        assert_eq!(
            permuted(HeadAxis::ValueColumns, 2, 12, &input),
            [
                0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, //
                12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23
            ]
        );
    }

    #[test]
    fn a_shape_the_axis_does_not_describe_is_refused() {
        // The reordering has no output shape of its own, so a tensor that has
        // drifted into another layout can only be caught here.
        let err = HeadPermute::new(HeadAxis::ValueRows, HEADS, 11, 1, "tensor")
            .expect_err("eleven rows is not four heads of three channels");
        assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
    }

    #[test]
    fn heads_that_do_not_divide_are_refused() {
        let heads = GdnHeads {
            value_heads: 5,
            ..HEADS
        };
        let err = HeadPermute::new(HeadAxis::PerHeadRows, heads, 5, 1, "tensor")
            .expect_err("five value heads do not divide among two key heads");
        assert!(matches!(err, GgufError::LayoutMismatch { .. }), "{err:?}");
    }
}
