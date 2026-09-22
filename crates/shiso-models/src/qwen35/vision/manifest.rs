//! Which tensors a vision tower holds, what shape each one must be, and the
//! shape of the tree they hang in.
//!
//! The manifest is DERIVED from [`VisionDims`], never recorded. That is the
//! whole reason it exists as a type: a loader that asked the file what it
//! contained and uploaded whatever came back would load a tower of the wrong
//! width without complaint, because every tensor in such a file is individually
//! well-formed. Here the dimensions state what must be present, the container
//! is asked for exactly that, and a disagreement names the tensor and both
//! shapes.
//!
//! # One tree, two payloads
//!
//! [`TowerTree`] is generic over what it holds for each tensor, and there are
//! exactly two useful fillings: a [`TowerTensorSpec`] before loading, and a
//! device buffer after. They are the same tree because the alternative - a
//! manifest laid out flat and a weight struct laid out by hand, matched up by
//! position - puts a silent reordering bug exactly where nothing can catch it.
//! The attention output projection and the block's normalizations are all
//! square and all the same width, so swapping them produces a tower that loads,
//! runs, and answers wrongly.
//!
//! # The two name spaces
//!
//! The same tower ships under two prefixes - one inside a checkpoint directory,
//! one in a standalone file - and the standalone form has a second, longer
//! spelling in some exports. Only the prefix differs; the role names under it
//! are identical, which is why the prefix is a parameter and the rest of this
//! module holds one copy of the layout.
//!
//! # Axis order
//!
//! Shapes here are written slowest-varying axis first, which is how a
//! checkpoint directory's header writes them. The standalone container writes
//! the same axes in the opposite order and means the same memory, so the
//! comparison takes the order as an argument rather than reversing at the call
//! site and hoping every call site remembers.

use crate::qwen35::vision::dims::{TENSORS_OUTSIDE_BLOCKS, TENSORS_PER_BLOCK, VisionDims};

/// Name prefix the tower carries inside a checkpoint directory.
pub const CHECKPOINT_PREFIX: &str = "vision_tower";

/// Name prefix the standalone tower file carries.
pub const SIDECAR_PREFIX: &str = "model.visual";

/// Longer spelling of [`SIDECAR_PREFIX`] that some exports write.
///
/// Probed second, and only when the short spelling is absent. Probing rather
/// than accepting both at once matters: a file carrying both prefixes holds two
/// towers, and taking whichever was looked up first would be a coin toss
/// between them.
pub const SIDECAR_PREFIX_CANONICAL: &str = "model.language_model.visual";

/// Colour channels a patch row carries.
const CHANNELS: u64 = 3;

/// Which end of a shape the fastest-varying axis is written at.
///
/// The two containers disagree, and for any square matrix - which is most of a
/// transformer block - the disagreement is invisible in the numbers. A
/// square-only test would pass under either convention, so the convention is
/// named and passed explicitly rather than inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisOrder {
    /// Slowest-varying axis first, as a checkpoint directory's header writes it.
    SlowestFirst,
    /// Fastest-varying axis first, as the standalone container writes it.
    FastestFirst,
}

/// Where the channel axis sits inside one row of a patch projection.
///
/// The patch projection is the one tensor in the tower whose row is not just a
/// width: it is `(channel, frame, y, x)` flattened, and the two published
/// exports of this tower flatten it in different orders. Same numbers, same
/// count, different arrangement - and the encoder is layout-agnostic by
/// construction, so a row read in the wrong order produces a scrambled picture
/// and no error anywhere. Which one a file holds is therefore recorded here and
/// acted on at load, never inferred later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PatchRows {
    /// The channel varies fastest, which is the order the preprocessor packs a
    /// patch row in. The bytes go to the device as they are.
    AsStored,
    /// The channel varies slowest - the framework-native convolution layout,
    /// which is what the standalone container carries because its exporter
    /// copies the checkpoint's bytes unchanged. Each output row is transposed
    /// at load.
    ChannelSlowest,
}

/// One spelling a tensor may legitimately be written in, and what it means.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct AcceptedShape {
    /// The shape, slowest-varying axis first.
    pub dims: Vec<u64>,
    /// Where the channel sits in a row written this way.
    pub rows: PatchRows,
}

/// One tensor the tower must hold.
///
/// A spec carries every shape the tensor may legitimately be written in, not
/// just one. That is a concession to exactly one tensor: the patch projection
/// is exported with its patch axes spelled out in either of two orders, or
/// already flattened into the row width. Everything else has a single accepted
/// spelling, and accepting a second one for them would be the loophole this
/// check exists to close.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TowerTensorSpec {
    name: String,
    accepted: Vec<AcceptedShape>,
}

impl TowerTensorSpec {
    /// Fully qualified tensor name, prefix included.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The canonical shape, slowest-varying axis first.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.accepted
            .first()
            .map_or(&[], |first| first.dims.as_slice())
    }

    /// Every shape this tensor may legitimately be written in.
    #[must_use]
    pub fn accepted(&self) -> &[AcceptedShape] {
        &self.accepted
    }

    /// Elements the tensor holds. Identical across the accepted spellings, so
    /// the canonical one answers for all of them.
    #[must_use]
    pub fn elements(&self) -> u64 {
        self.shape().iter().copied().product()
    }

    /// Whether `declared`, written in `order`, is a shape this spec accepts.
    ///
    /// Trailing unit axes are ignored on the declared side: they are a
    /// broadcasting artefact of the exporting framework rather than a dimension
    /// anything reads, and refusing a tower over one would be refusing it over
    /// punctuation.
    #[must_use]
    pub fn matches(&self, declared: &[u64], order: AxisOrder) -> bool {
        self.spelling(declared, order).is_some()
    }

    /// Which accepted spelling `declared` is, or `None` when it is none of
    /// them.
    ///
    /// The answer is not a yes: it says where the channel sits in a row written
    /// this way, which is what the loader has to act on. A caller that only
    /// asks whether the shape is legal gets a tower that loads and describes
    /// the wrong picture.
    #[must_use]
    pub fn spelling(&self, declared: &[u64], order: AxisOrder) -> Option<PatchRows> {
        let mut trimmed = declared;
        while trimmed.len() > 1 && trimmed.last() == Some(&1) {
            trimmed = &trimmed[..trimmed.len() - 1];
        }
        self.accepted
            .iter()
            .find(|shape| match order {
                AxisOrder::SlowestFirst => trimmed == shape.dims.as_slice(),
                AxisOrder::FastestFirst => {
                    trimmed.len() == shape.dims.len() && trimmed.iter().rev().eq(shape.dims.iter())
                }
            })
            .map(|shape| shape.rows)
    }
}

/// A weight and the bias applied after it.
///
/// Used for the normalizations as well as the projections: a normalization with
/// a scale and an offset is an affine map, and giving it its own near-identical
/// type would mean writing the walk twice.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Affine<T> {
    /// The matrix, or the normalization's scale vector.
    pub weight: T,
    /// The offset added afterwards.
    pub bias: T,
}

/// One transformer block of the tower.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TowerBlock<T> {
    /// Normalization before attention.
    pub norm1: Affine<T>,
    /// Fused query/key/value projection.
    pub qkv: Affine<T>,
    /// Attention output projection.
    pub proj: Affine<T>,
    /// Normalization before the feed-forward.
    pub norm2: Affine<T>,
    /// Feed-forward input projection.
    pub fc1: Affine<T>,
    /// Feed-forward output projection.
    pub fc2: Affine<T>,
}

/// The spatial merger that turns patch vectors into language-model tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TowerMerger<T> {
    /// Normalization over the merged block.
    pub norm: Affine<T>,
    /// First merger projection.
    pub fc1: Affine<T>,
    /// Projection into the language model's width.
    pub fc2: Affine<T>,
}

/// Every tensor of one tower, in the shape they hang in.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TowerTree<T> {
    /// Patch projection and its bias.
    pub patch_embed: Affine<T>,
    /// The learned position table.
    pub pos_embed: T,
    /// The transformer blocks, in depth order.
    pub blocks: Vec<TowerBlock<T>>,
    /// The merger.
    pub merger: TowerMerger<T>,
}

/// A tower's manifest: the tree, filled with what each tensor must be.
pub type TowerManifest = TowerTree<TowerTensorSpec>;

impl<T> Affine<T> {
    /// Rewrites both halves, weight first.
    fn try_map<U, E>(self, visit: &mut impl FnMut(T) -> Result<U, E>) -> Result<Affine<U>, E> {
        Ok(Affine {
            weight: visit(self.weight)?,
            bias: visit(self.bias)?,
        })
    }

    /// Both halves, weight first.
    fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.weight, &self.bias].into_iter()
    }
}

impl<T> TowerTree<T> {
    /// Rewrites every tensor in upload order, stopping at the first failure.
    ///
    /// The order is the tree's own order - patch projection, position table,
    /// blocks in depth order, merger - which is also the order the manifest was
    /// built in, so a tower is uploaded in the order its file lays it out.
    ///
    /// # Errors
    ///
    /// Whatever `visit` reports, unchanged and at the first tensor that fails.
    pub fn try_map<U, E>(
        self,
        mut visit: impl FnMut(T) -> Result<U, E>,
    ) -> Result<TowerTree<U>, E> {
        let visit = &mut visit;
        Ok(TowerTree {
            patch_embed: self.patch_embed.try_map(visit)?,
            pos_embed: visit(self.pos_embed)?,
            blocks: self
                .blocks
                .into_iter()
                .map(|block| {
                    Ok(TowerBlock {
                        norm1: block.norm1.try_map(visit)?,
                        qkv: block.qkv.try_map(visit)?,
                        proj: block.proj.try_map(visit)?,
                        norm2: block.norm2.try_map(visit)?,
                        fc1: block.fc1.try_map(visit)?,
                        fc2: block.fc2.try_map(visit)?,
                    })
                })
                .collect::<Result<Vec<_>, E>>()?,
            merger: TowerMerger {
                norm: self.merger.norm.try_map(visit)?,
                fc1: self.merger.fc1.try_map(visit)?,
                fc2: self.merger.fc2.try_map(visit)?,
            },
        })
    }

    /// Every tensor, in the same order [`TowerTree::try_map`] visits them.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.patch_embed
            .iter()
            .chain(std::iter::once(&self.pos_embed))
            .chain(self.blocks.iter().flat_map(|block| {
                block
                    .norm1
                    .iter()
                    .chain(block.qkv.iter())
                    .chain(block.proj.iter())
                    .chain(block.norm2.iter())
                    .chain(block.fc1.iter())
                    .chain(block.fc2.iter())
            }))
            .chain(self.merger.norm.iter())
            .chain(self.merger.fc1.iter())
            .chain(self.merger.fc2.iter())
    }

    /// Tensors in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len() * TENSORS_PER_BLOCK + TENSORS_OUTSIDE_BLOCKS
    }

    /// Whether the tree holds no tensors, which no real tower does.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

impl TowerManifest {
    /// Builds the manifest a tower of these dimensions must satisfy.
    #[must_use]
    pub fn build(prefix: &str, dims: VisionDims) -> Self {
        let hidden = dim(dims.hidden());
        let qkv = dim(dims.qkv_width());
        let ff = dim(dims.intermediate());
        let merge_in = dim(dims.merge_input());
        let out = dim(dims.out_hidden());
        let geometry = dims.geometry();
        let patch = dim(geometry.patch_size());
        let temporal = dim(geometry.temporal_patch_size());

        let spec = |name: String, shape: Vec<u64>| TowerTensorSpec {
            name,
            accepted: vec![AcceptedShape {
                dims: shape,
                rows: PatchRows::AsStored,
            }],
        };
        let vector = |name: String| spec(name, vec![hidden]);

        TowerTree {
            // The patch projection's input is spelled out as the axes the patch
            // row was built from rather than as their flat product: a tower
            // whose temporal depth disagrees with the preprocessor's would
            // otherwise match on the flat width alone and be wrong per frame.
            patch_embed: Affine {
                weight: TowerTensorSpec {
                    name: format!("{prefix}.patch_embed.proj.weight"),
                    accepted: vec![
                        AcceptedShape {
                            dims: vec![hidden, temporal, patch, patch, CHANNELS],
                            rows: PatchRows::AsStored,
                        },
                        // The framework-native convolution weight, which the
                        // standalone container carries because its exporter
                        // copies the checkpoint's bytes and only reverses the
                        // axis list. Same count, channel slowest.
                        AcceptedShape {
                            dims: vec![hidden, CHANNELS, temporal, patch, patch],
                            rows: PatchRows::ChannelSlowest,
                        },
                        AcceptedShape {
                            dims: vec![hidden, dim(dims.patch_dim())],
                            rows: PatchRows::AsStored,
                        },
                    ],
                },
                bias: vector(format!("{prefix}.patch_embed.proj.bias")),
            },
            pos_embed: spec(
                format!("{prefix}.pos_embed.weight"),
                vec![dim(dims.position_embeddings()), hidden],
            ),
            blocks: (0..dims.depth())
                .map(|block| {
                    let at = format!("{prefix}.blocks.{block}");
                    TowerBlock {
                        norm1: Affine {
                            weight: vector(format!("{at}.norm1.weight")),
                            bias: vector(format!("{at}.norm1.bias")),
                        },
                        qkv: Affine {
                            weight: spec(format!("{at}.attn.qkv.weight"), vec![qkv, hidden]),
                            bias: spec(format!("{at}.attn.qkv.bias"), vec![qkv]),
                        },
                        proj: Affine {
                            weight: spec(format!("{at}.attn.proj.weight"), vec![hidden, hidden]),
                            bias: vector(format!("{at}.attn.proj.bias")),
                        },
                        norm2: Affine {
                            weight: vector(format!("{at}.norm2.weight")),
                            bias: vector(format!("{at}.norm2.bias")),
                        },
                        fc1: Affine {
                            weight: spec(format!("{at}.mlp.linear_fc1.weight"), vec![ff, hidden]),
                            bias: spec(format!("{at}.mlp.linear_fc1.bias"), vec![ff]),
                        },
                        fc2: Affine {
                            weight: spec(format!("{at}.mlp.linear_fc2.weight"), vec![hidden, ff]),
                            bias: vector(format!("{at}.mlp.linear_fc2.bias")),
                        },
                    }
                })
                .collect(),
            merger: TowerMerger {
                norm: Affine {
                    weight: vector(format!("{prefix}.merger.norm.weight")),
                    bias: vector(format!("{prefix}.merger.norm.bias")),
                },
                fc1: Affine {
                    weight: spec(
                        format!("{prefix}.merger.linear_fc1.weight"),
                        vec![merge_in, merge_in],
                    ),
                    bias: spec(format!("{prefix}.merger.linear_fc1.bias"), vec![merge_in]),
                },
                fc2: Affine {
                    weight: spec(
                        format!("{prefix}.merger.linear_fc2.weight"),
                        vec![out, merge_in],
                    ),
                    bias: spec(format!("{prefix}.merger.linear_fc2.bias"), vec![out]),
                },
            },
        }
    }

    /// Every tensor name, in upload order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.iter().map(TowerTensorSpec::name).collect()
    }
}

/// Narrows a dimension for comparison against a container's declared shape.
///
/// Saturating rather than fallible: a dimension that does not fit sixty-four
/// bits cannot match any real tensor either, so it fails at the comparison with
/// the tensor's name attached instead of one layer earlier without it.
fn dim(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// The published dimensions of the standalone tower.
///
/// This is a CONTRACT, not a source of truth for the loader: the dimensions a
/// tower is loaded with always come from the file's own metadata, and this is
/// only what those dimensions are then checked against. The distinction
/// matters because the two are different failures - a file that declares its
/// real shape and disagrees with the published one is a different artefact,
/// while a file that declares one shape and stores another is corrupt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SidecarContract {
    /// Architecture the file must declare for this contract to apply.
    pub architecture: &'static str,
    /// Transformer blocks.
    pub depth: usize,
    /// Width inside the tower.
    pub hidden: usize,
    /// Feed-forward inner width.
    pub intermediate: usize,
    /// Width handed to the language model.
    pub out_hidden: usize,
    /// Tensors a complete tower holds.
    pub tensors: usize,
}

/// The one published standalone tower.
///
/// Every number here is a contract-validation constant and appears nowhere in a
/// computation: the loader derives its shapes from whatever the file declares
/// and consults these only to decide whether to accept the file at all.
pub const PUBLISHED_SIDECAR: SidecarContract = SidecarContract {
    architecture: "qwen38f-vision",
    depth: 27,
    hidden: 1152,
    intermediate: 4304,
    out_hidden: 2560,
    tensors: 27 * TENSORS_PER_BLOCK + TENSORS_OUTSIDE_BLOCKS,
};

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
