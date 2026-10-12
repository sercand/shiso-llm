//! Running the LFM2 vision tower.
//!
//! One picture's patch rows go in; the projected rows the language model
//! consumes come out. Between them: the patch projection, the resampled
//! position grid, the blocks, the final norm, the 2 x 2 unshuffle and the
//! projector. Every width comes from [`Lfm2Dims`], which the loader derived
//! from the container.
//!
//! # Whose kernels
//!
//! None of this tower's own. The projections, the norms and the attention are
//! the shared encoder set's ([`Ops`]): its GEMM reads every weight format the
//! file stores - `Q8_0`, half, single - with the bias in its epilogue, and its
//! tiled span attention runs the whole picture bidirectionally. The two
//! GELUs, the residual add and the unshuffle are the vision set's, which the
//! Qwen tower also drives. A tower that owned its kernels would be a tower
//! whose arithmetic drifts from its neighbours'.
//!
//! The shipped head is 72 wide and the span attention takes heads in whole
//! 32-value slices, so the attention runs over 96-wide head slots that the
//! loader zero-padded the projections into; the loader says why that is
//! exact. The vision set's one-row-per-group attention takes any width and
//! needs no padding, and it was about 70% of an encode; over the padded slots
//! a 576-patch picture encodes in about 165 ms on an M1 Max and a 936-patch
//! one in about 300 ms (the parity case prints the number on every run).
//!
//! # The unshuffle is the merge kernel, and that is a fact about orders
//!
//! The reference folds each 2 x 2 block of patches into one row as
//! `out[i][j][(dy * 2 + dx) * C + c] = in[2i + dy][2j + dx][c]`, rows of the
//! output row-major over the halved grid. That is exactly what the vision
//! set's spatial merge writes over row-major patch rows (block position `p`
//! walks the block row-major, `p = dy * merge + dx`), so the kernel the Qwen
//! merger uses is this projector's unshuffle without a line changed. The unit
//! test of the index mapping is in this module's tests.

#![expect(
    clippy::cast_precision_loss,
    reason = "head widths are a few dozen; their root is exact enough in f32"
)]

use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use shiso_core::{Backend, DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs, VisionGrid};
use shiso_vision::{
    EncodedRows, MediaPlacement, PatchGeometry, PreprocessSpec, VisionError, VisionItem,
    VisionStages, VisionTower,
};

use super::loader::{Lfm2Linear, Lfm2Norm, Lfm2VisionWeights, load_lfm2_weights};
use super::positions::resample;
use super::{Lfm2Dims, MAX_IMAGE_TOKENS, MIN_IMAGE_TOKENS};
use crate::audio_ops::{NormDesc, Ops, SpanAttnDesc};

/// Registry module the vision set's kernels live in.
const MODULE_VISION: &str = "vision";

/// Threads the vision set's elementwise kernels are launched with; they index
/// by this value, so it is the one block size they may be launched with.
const LINEAR_BLOCK: u32 = 256;

/// Widest head slot the span attention takes.
const MAX_HEAD_SLOT: usize = 512;

/// `vit_args_unary`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

/// `vit_args_add`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AddArgs {
    count: u32,
    period: u32,
}

/// `vit_args_merge`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    merged_rows: u32,
    hidden: u32,
    grid_h: u32,
    grid_w: u32,
    merge: u32,
}

/// The layouts the kernels read; a field added on one side only would bind
/// silently, so the widths are asserted.
const _: () = {
    assert!(size_of::<UnaryArgs>() == 4);
    assert!(size_of::<AddArgs>() == 8);
    assert!(size_of::<MergeArgs>() == 20);
};

/// An LFM2 vision tower: the loaded weights, the preprocessing contract this
/// family publishes, and the widths both sides read.
#[derive(Debug)]
pub struct Lfm2Tower {
    weights: Lfm2VisionWeights,
    spec: PreprocessSpec,
}

impl Lfm2Tower {
    /// How this family wants a picture marked in the prompt.
    pub const PLACEMENT: MediaPlacement = MediaPlacement::LFM2;

    /// Opens the mmproj at `path` as this family's tower.
    ///
    /// # Errors
    ///
    /// [`VisionError`] naming what the container lacks or holds at the wrong
    /// shape or format, and [`VisionError::Upload`] for a device failure.
    pub fn open(backend: &Arc<dyn Backend>, path: &Path) -> Result<Self, VisionError> {
        let weights = load_lfm2_weights(backend, path)?;
        let dims = weights.dims;
        if dims.head_slot > MAX_HEAD_SLOT {
            return Err(VisionError::HeadTooWide {
                head_dim: dims.head_dim,
                max: MAX_HEAD_SLOT,
            });
        }
        let geometry = PatchGeometry::new(dims.patch, 1, dims.merge)?;
        let spec = PreprocessSpec::lfm2(geometry, MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS);
        Ok(Self { weights, spec })
    }

    /// Runs the whole graph over one item and returns its intermediate stages:
    /// `pooled` is the unshuffled rows and `normed` the final norm's output.
    ///
    /// # Errors
    ///
    /// [`VisionError`] if the item does not fit the tower or a device
    /// operation fails.
    pub fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        let mut stages = Stages::default();
        Graph::new(backend, &self.weights, item)?.run(Some(&mut stages))?;
        stages.finish()
    }
}

impl VisionTower for Lfm2Tower {
    fn preprocess(&self) -> &PreprocessSpec {
        &self.spec
    }

    fn placement(&self) -> MediaPlacement {
        Self::PLACEMENT
    }

    fn out_hidden(&self) -> u32 {
        narrow(self.weights.dims.out_hidden)
    }

    fn device_bytes(&self) -> u64 {
        self.weights.device_bytes()
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError> {
        let merge = self.weights.dims.merge;
        let mut encoded = Vec::with_capacity(items.len());
        let mut grids = Vec::with_capacity(items.len());
        for item in items {
            let (t_len, grid_h, grid_w) = item.grid();
            encoded.push(Graph::new(backend, &self.weights, item)?.run(None)?);
            grids.push(VisionGrid::new(
                narrow(t_len),
                narrow(grid_h / merge),
                narrow(grid_w / merge),
            ));
        }
        let row_bytes = u64::from(self.out_hidden()) * 4;
        EncodedRows::gather(backend, &encoded, grids, row_bytes)
    }

    fn encode_stages(
        &self,
        backend: &Arc<dyn Backend>,
        item: &VisionItem,
    ) -> Result<VisionStages, VisionError> {
        Self::encode_stages(self, backend, item)
    }
}

/// Copies of the stages one encode passed through, for a parity case.
#[derive(Default)]
struct Stages {
    pixel_values: Option<DeviceBuffer>,
    patch_embed: Option<DeviceBuffer>,
    pos_added: Option<DeviceBuffer>,
    blocks: Vec<DeviceBuffer>,
    normed: Option<DeviceBuffer>,
    pooled: Option<DeviceBuffer>,
    projected: Option<DeviceBuffer>,
}

impl Stages {
    fn finish(self) -> Result<VisionStages, VisionError> {
        let missing = |what| VisionError::Unsupported { what };
        Ok(VisionStages {
            pixel_values: self.pixel_values.ok_or_else(|| missing("pixel_values"))?,
            patch_embed: self.patch_embed.ok_or_else(|| missing("patch_embed"))?,
            pos_added: self.pos_added.ok_or_else(|| missing("pos_added"))?,
            blocks: self.blocks,
            pooled: self.pooled.ok_or_else(|| missing("pooled"))?,
            normed: self.normed.ok_or_else(|| missing("normed"))?,
            projected: self.projected.ok_or_else(|| missing("projected"))?,
        })
    }
}

/// One item's encode: the device, the weights, the checked widths and the
/// activation buffers.
struct Graph<'a> {
    backend: &'a Arc<dyn Backend>,
    ops: Ops,
    weights: &'a Lfm2VisionWeights,
    item: &'a VisionItem,
    dims: Lfm2Dims,
    rows: usize,
    grid_h: usize,
    grid_w: usize,
}

impl<'a> Graph<'a> {
    /// Checks `item` against the tower before anything is allocated.
    fn new(
        backend: &'a Arc<dyn Backend>,
        weights: &'a Lfm2VisionWeights,
        item: &'a VisionItem,
    ) -> Result<Self, VisionError> {
        let dims = weights.dims;
        let (t_len, grid_h, grid_w) = item.grid();
        if t_len != 1 {
            return Err(VisionError::Unsupported {
                what: "the lfm2 vision tower reads stills; a temporal group is not one",
            });
        }
        if !grid_h.is_multiple_of(dims.merge) || !grid_w.is_multiple_of(dims.merge) {
            return Err(VisionError::ItemGridNotMergeable {
                grid_h,
                grid_w,
                merge: dims.merge,
            });
        }
        if item.row_len() != dims.patch_dim {
            return Err(VisionError::ItemRowWidth {
                expected: dims.patch_dim,
                got: item.row_len(),
            });
        }
        Ok(Self {
            backend,
            ops: Ops::new(Arc::clone(backend)),
            weights,
            item,
            dims,
            rows: grid_h * grid_w,
            grid_h,
            grid_w,
        })
    }

    /// Issues the whole tower and returns the projected rows,
    /// `[rows / merge^2][out_hidden]` f32.
    #[expect(
        clippy::many_single_char_names,
        reason = "q, k and v are the attention's own names, and x, h and t the residual stream, its norm and the scratch beside them"
    )]
    fn run(&self, mut stages: Option<&mut Stages>) -> Result<DeviceBuffer, VisionError> {
        let d = self.dims;
        let w = self.weights;
        let rows = self.rows;
        let n = rows * d.hidden;
        let x = self.buffer(n)?;
        let h = self.buffer(n)?;
        let t = self.buffer(n)?;
        // The attention planes are in padded head slots; see the loader.
        let planes = rows * d.heads * d.head_slot;
        let q = self.buffer(planes)?;
        let k = self.buffer(planes)?;
        let v = self.buffer(planes)?;
        let attn = self.buffer(planes)?;
        let mid = self.buffer(rows * d.intermediate)?;

        let pixels = self.upload("patch rows", self.item.as_flat())?;
        self.capture(&mut stages, |s| &mut s.pixel_values, &pixels)?;
        let patch = Mul::new(rows, d.hidden, d.patch_dim);
        self.linear("patch projection", &x, &pixels, &w.patch, patch)?;
        self.capture(&mut stages, |s| &mut s.patch_embed, &x)?;
        let positions = resample(&w.positions, d.pos_side, d.hidden, self.grid_h, self.grid_w);
        let positions = self.upload("position rows", &positions)?;
        self.add("position add", &x, &positions, n)?;
        self.capture(&mut stages, |s| &mut s.pos_added, &x)?;

        let slots = d.heads * d.head_slot;
        let into_heads = Mul::new(rows, slots, d.hidden);
        let from_heads = Mul::new(rows, d.hidden, slots);
        let widen = Mul::new(rows, d.intermediate, d.hidden);
        let narrow_back = Mul::new(rows, d.hidden, d.intermediate);
        for block in &w.blocks {
            self.norm("attention norm", &h, &x, &block.ln1)?;
            self.linear("query", &q, &h, &block.q, into_heads)?;
            self.linear("key", &k, &h, &block.k, into_heads)?;
            self.linear("value", &v, &h, &block.v, into_heads)?;
            self.attention(&attn, &q, &k, &v)?;
            self.linear("attention output", &t, &attn, &block.out, from_heads)?;
            self.add("attention residual", &x, &t, n)?;
            self.norm("feed-forward norm", &h, &x, &block.ln2)?;
            self.linear("feed-forward in", &mid, &h, &block.up, widen)?;
            self.unary(
                "kernel_vit_gelu_tanh",
                "block activation",
                &mid,
                rows * d.intermediate,
            )?;
            self.linear("feed-forward out", &t, &mid, &block.down, narrow_back)?;
            self.add("feed-forward residual", &x, &t, n)?;
            if let Some(s) = stages.as_deref_mut() {
                s.blocks.push(self.clone_buffer(&x)?);
            }
        }
        self.norm("final norm", &h, &x, &w.post_ln)?;
        self.capture(&mut stages, |s| &mut s.normed, &h)?;

        let tokens = rows / (d.merge * d.merge);
        let folded = self.buffer(tokens * d.merged())?;
        self.unshuffle(&folded, &h, tokens)?;
        self.capture(&mut stages, |s| &mut s.pooled, &folded)?;
        let inner = self.buffer(tokens * w.proj_hidden)?;
        let project_in = Mul::new(tokens, w.proj_hidden, d.merged());
        self.linear("projector in", &inner, &folded, &w.mm1, project_in)?;
        // The projector's activation is the exact GELU; the blocks' is the
        // tanh form. The two differ by about 1e-4 a value, and the layers
        // genuinely declare different ones.
        self.unary(
            "kernel_vit_gelu_erf",
            "projector activation",
            &inner,
            tokens * w.proj_hidden,
        )?;
        let out = self.buffer(tokens * d.out_hidden)?;
        let project_out = Mul::new(tokens, d.out_hidden, w.proj_hidden);
        self.linear("projector out", &out, &inner, &w.mm2, project_out)?;
        self.capture(&mut stages, |s| &mut s.projected, &out)?;
        self.backend.sync().map_err(|source| VisionError::Encode {
            step: "draining the encode",
            source,
        })?;
        Ok(out)
    }

    /// `out = input . W^T + b` over `shape.m` rows.
    fn linear(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        layer: &Lfm2Linear,
        shape: Mul,
    ) -> Result<(), VisionError> {
        let Mul { m, n, k } = shape;
        let encode = |source| VisionError::Encode { step, source };
        self.ops
            .gemm(
                layer.dtype,
                out.slice(0, bytes(m * n)).map_err(encode)?,
                input.slice(0, bytes(m * k)).map_err(encode)?,
                layer.weight.view(),
                Some(layer.bias.view()),
                (m, n, k),
            )
            .map_err(encode)
    }

    /// A layer norm over every row, `out` from `input`.
    fn norm(
        &self,
        step: &'static str,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        norm: &Lfm2Norm,
    ) -> Result<(), VisionError> {
        self.ops
            .layer_norm(&NormDesc {
                out: out.view(),
                input: input.view(),
                weight: Some(norm.weight.view()),
                bias: Some(norm.bias.view()),
                rows: self.rows,
                dim: self.dims.hidden,
                eps: self.dims.eps,
            })
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// Bidirectional attention over the whole picture, over the padded head
    /// slots, at the inverse root of the head's own width - not the slot's:
    /// the padding adds nothing to a score, so the scale is the reference's.
    fn attention(
        &self,
        out: &DeviceBuffer,
        q: &DeviceBuffer,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        let d = self.dims;
        self.ops
            .span_attention(&SpanAttnDesc {
                out: out.view(),
                q: q.view(),
                k: k.view(),
                v: v.view(),
                segments: None,
                n: self.rows,
                heads: d.heads,
                kv_heads: d.heads,
                head_dim: d.head_slot,
                left: self.rows,
                right: self.rows,
                scale: 1.0 / (d.head_dim as f32).sqrt(),
            })
            .map_err(|source| VisionError::Encode {
                step: "attention",
                source,
            })
    }

    /// The 2 x 2 unshuffle: `tokens` rows of `merge^2` patch rows each.
    fn unshuffle(
        &self,
        out: &DeviceBuffer,
        input: &DeviceBuffer,
        tokens: usize,
    ) -> Result<(), VisionError> {
        let args = MergeArgs {
            merged_rows: narrow(tokens),
            hidden: narrow(self.dims.hidden),
            grid_h: narrow(self.grid_h),
            grid_w: narrow(self.grid_w),
            merge: narrow(self.dims.merge),
        };
        self.launch(
            "unshuffle",
            "kernel_vit_spatial_merge",
            Extent3::linear(narrow(tokens)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
            ],
        )
    }

    /// `x[i] += addend[i]` over `count` values.
    fn add(
        &self,
        step: &'static str,
        x: &DeviceBuffer,
        addend: &DeviceBuffer,
        count: usize,
    ) -> Result<(), VisionError> {
        let args = AddArgs {
            count: narrow(count),
            period: narrow(count),
        };
        self.launch(
            step,
            "kernel_vit_add_inplace",
            Extent3::linear(narrow(count.div_ceil(LINEAR_BLOCK as usize))),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(addend.view()),
            ],
        )
    }

    /// One of the vision set's in-place activations over `count` values.
    fn unary(
        &self,
        entry: &'static str,
        step: &'static str,
        x: &DeviceBuffer,
        count: usize,
    ) -> Result<(), VisionError> {
        let args = UnaryArgs {
            count: narrow(count),
        };
        self.launch(
            step,
            entry,
            Extent3::linear(narrow(count.div_ceil(LINEAR_BLOCK as usize))),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(x.view()),
            ],
        )
    }

    /// One launch of the vision set.
    fn launch(
        &self,
        step: &'static str,
        entry: &'static str,
        grid: Extent3,
        block: Extent3,
        smem: u32,
        args: &[KernelArg<'_>],
    ) -> Result<(), VisionError> {
        self.backend
            .run(
                KernelRef::new(MODULE_VISION, entry),
                &LaunchArgs {
                    grid,
                    block,
                    smem_bytes: smem,
                    args,
                },
            )
            .map_err(|source| VisionError::Encode { step, source })
    }

    /// An f32 activation buffer of `count` values. Every buffer here is
    /// written in full before it is read, so none is cleared.
    fn buffer(&self, count: usize) -> Result<DeviceBuffer, VisionError> {
        self.ops
            .scratch_uninit(count)
            .map_err(|source| VisionError::Encode {
                step: "allocating the encode buffers",
                source,
            })
    }

    /// Host floats on the device.
    fn upload(&self, step: &'static str, values: &[f32]) -> Result<DeviceBuffer, VisionError> {
        let buffer = self.buffer(values.len())?;
        self.backend
            .copy_h2d(buffer.view(), bytemuck::cast_slice(values))
            .map_err(|source| VisionError::Encode { step, source })?;
        Ok(buffer)
    }

    /// A device copy of `buffer`, for a captured stage.
    fn clone_buffer(&self, buffer: &DeviceBuffer) -> Result<DeviceBuffer, VisionError> {
        let step = "capturing a stage";
        let copy = shiso_core::alloc(self.backend, buffer.len())
            .map_err(|source| VisionError::Encode { step, source })?;
        self.backend
            .copy_d2d(copy.view(), buffer.view())
            .map_err(|source| VisionError::Encode { step, source })?;
        Ok(copy)
    }

    /// Stores a copy of `buffer` in the stage `slot` names, when capturing.
    fn capture(
        &self,
        stages: &mut Option<&mut Stages>,
        slot: impl FnOnce(&mut Stages) -> &mut Option<DeviceBuffer>,
        buffer: &DeviceBuffer,
    ) -> Result<(), VisionError> {
        if let Some(s) = stages.as_deref_mut() {
            *slot(s) = Some(self.clone_buffer(buffer)?);
        }
        Ok(())
    }
}

/// The three widths of one projection: `m` rows, `n` outputs, `k` inputs.
///
/// A named triple rather than three positional counts: a call that swapped
/// the last two would launch at the right size and compute the wrong product.
#[derive(Debug, Clone, Copy)]
struct Mul {
    m: usize,
    n: usize,
    k: usize,
}

impl Mul {
    const fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }
}

/// Bytes `floats` f32 values take.
fn bytes(floats: usize) -> u64 {
    u64::try_from(floats).unwrap_or(u64::MAX).saturating_mul(4)
}

/// A device dimension as the kernels' 32-bit field.
///
/// Every width here is bounded by the token band and the tower's own sizes,
/// far inside 32 bits; the saturating arm is unreachable.
fn narrow(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}
