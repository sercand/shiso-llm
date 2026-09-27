//! Tests for running the vision tower.
//!
//! The device cases hold the whole tower to a naive host reference: a plain
//! `f32` transcription of the same forward pass, written from the tower's
//! description rather than from the kernels. That is the only gate this tower
//! has. There is no upstream shading-language source to diff the kernels
//! against, so nothing else stands between an ordering mistake and a model that
//! answers confidently about a picture it never saw.
//!
//! The same code runs at two dimension sets: the shipped tower's widths, and a
//! second set with a different width, head width and feed-forward size. One set
//! proves the tower runs. Two prove it is driven by the dimensions it was handed
//! rather than by a literal that happens to be right once - and the head width
//! in particular, which the rotation splits twice, is the number a single-set
//! test would never exercise.
//!
//! Device cases are `#[ignore]`-gated:
//! `cargo test -p shiso-vision --release -- --ignored`. Release matters here:
//! the host reference is a triple loop over the same widths the device runs.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the reference works in host floats over loop indices, and every index here is bounded by a small fixture dimension"
)]
#![expect(
    clippy::excessive_precision,
    reason = "the activation constants are written with the digits the tower's own description gives them, which round to the f32 the kernels hold"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the reference uses the tower's own names - q, k, v for the attention planes - and renaming them to satisfy a lint would make it a worse transcription"
)]

use std::sync::Arc;

use shiso_core::Backend;

use super::VisionEncoder;
use crate::qwen35::vision::dims::VisionDims;
use crate::qwen35::vision::loader::load_vision_weights;
use crate::qwen35::vision::manifest::TowerTree;
use crate::qwen35::vision::testing::{signal, synthetic_tower};
use shiso_vision::PatchGeometry;
use shiso_vision::VisionError;
use shiso_vision::VisionItem;
use shiso_vision::VisionSource;

/// Relative tolerance the device is held to.
///
/// Not a round number picked for comfort. It is calibrated, and the calibration
/// is the point of the case: the device and the reference disagree by about
/// 3e-6 relative typically, purely because the kernels reduce in a different
/// order than the reference loops do, with a measured cancellation tail of
/// 2.25e-5 worst-of-10240 in the two-temporal-groups case (every row of both
/// groups carries the same tail, so it is rounding, not a group boundary read
/// in the wrong place). The smallest mistake this file has to catch - the
/// merger running the blocks' activation instead of its own, which is two
/// characters at one call site - moves the output by about 1e-4 relative. The
/// bound sits between the two, near the noise end.
///
/// A looser bound is not a safer bound. At 3e-3, which is a perfectly ordinary
/// choice for a comparison like this, that activation swap passes every case
/// below. This project has already shipped one kernel that passed its own
/// oracle and made the model answer in fragments; a tolerance chosen by feel is
/// how that happens again.
const TOL: f32 = 3e-5;

/// The shipped tower's widths, at a depth of one block.
///
/// Depth is the one dimension deliberately not the shipped value: the reference
/// is a host triple loop, a block repeats the same arithmetic, and two dozen
/// copies of it would buy no coverage for a couple of gigabytes of fixture.
fn shipped_dims() -> VisionDims {
    VisionDims::new(
        1,
        1024,
        16,
        4096,
        2560,
        2304,
        PatchGeometry::new(16, 2, 2).expect("the shipped geometry"),
    )
    .expect("the shipped widths are valid")
}

/// A second set: a different width, a different head width, a different
/// feed-forward size and a different position grid.
fn second_dims() -> VisionDims {
    VisionDims::new(
        1,
        1152,
        16,
        4304,
        2560,
        1024,
        PatchGeometry::new(16, 2, 2).expect("the second geometry"),
    )
    .expect("the second widths are valid")
}

/// An open Metal device, or `None` on a machine without one.
#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the encoder cases");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

/// A patch item on the given grid, filled with the fixture signal.
fn fixture_item(
    dims: VisionDims,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    seed: u32,
) -> VisionItem {
    let row_len = dims.patch_dim();
    let pixels = signal(grid_t * grid_h * grid_w * row_len, seed);
    VisionItem::new(pixels.into_boxed_slice(), row_len, grid_t, grid_h, grid_w)
        .expect("the fixture grid and pixels agree")
}

// -- the host reference -----------------------------------------------------

/// A naive reading of the whole tower, in `f32`, with no device anywhere.
///
/// Every loop here is the simplest form of the operation it names. That is the
/// point: a reference that shared the kernels' blocking, their reduction order
/// or their address arithmetic would agree with them on a mistake.
struct Reference<'a> {
    dims: VisionDims,
    eps: f32,
    rope_base: f32,
    tree: &'a TowerTree<Vec<f32>>,
}

impl Reference<'_> {
    /// `out[m, n] = sum_k input[m, k] * weight[n, k] + bias[n]`.
    fn linear(
        input: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * n];
        for m in 0..rows {
            for col in 0..n {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += input[m * k + i] * weight[col * k + i];
                }
                out[m * n + col] = acc + bias[col];
            }
        }
        out
    }

    /// Mean and variance over the row, then the affine map.
    fn layer_norm(x: &[f32], rows: usize, dim: usize, w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let row = &x[r * dim..][..dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for i in 0..dim {
                out[r * dim + i] = (row[i] - mean) * rstd * w[i] + b[i];
            }
        }
        out
    }

    /// The tanh approximation, the blocks' activation.
    fn gelu_tanh(v: f32) -> f32 {
        let c = 0.797_884_560_802_865_4_f32;
        0.5 * v * (1.0 + (c * (v + 0.044_715 * v * v * v)).tanh())
    }

    /// The exact activation, the merger's.
    ///
    /// The shading language has no error function, so the kernel carries this
    /// same series and the comparison cannot check the series itself - only that
    /// the merger reaches it and the blocks do not. The series is bounded by its
    /// own published error, 1.5e-7 (Abramowitz and Stegun 7.1.26), which is three
    /// orders of magnitude tighter than the 1e-4 separating the two activations.
    fn gelu_erf(v: f32) -> f32 {
        let sign = if v < 0.0 { -1.0 } else { 1.0 };
        let x = (v * std::f32::consts::FRAC_1_SQRT_2).abs();
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erf = sign * (1.0 - poly * (-x * x).exp());
        0.5 * v * (1.0 + erf)
    }

    /// The learned position table, resampled onto this grid.
    ///
    /// Written as a second reading of the interpolation rather than a call into
    /// the encoder's: the sample positions and the corner weights are exactly
    /// the part that is easy to get subtly wrong and invisible afterwards.
    fn positions(&self, grid_h: usize, grid_w: usize) -> Vec<f32> {
        let hidden = self.dims.hidden();
        let side = self.dims.position_grid();
        let table = &self.tree.pos_embed;
        let mut out = vec![0.0f32; grid_h * grid_w * hidden];
        let at = |extent: usize, index: usize| -> (usize, usize, f32) {
            if extent <= 1 {
                return (0, 0, 0.0);
            }
            let step = (side - 1) as f32 / (extent - 1) as f32;
            let position = index as f32 * step;
            let low = position.floor();
            let low_index = (low as usize).min(side - 1);
            (low_index, (low_index + 1).min(side - 1), position - low)
        };
        for gh in 0..grid_h {
            let (y0, y1, dy) = at(grid_h, gh);
            for gw in 0..grid_w {
                let (x0, x1, dx) = at(grid_w, gw);
                let base = (gh * grid_w + gw) * hidden;
                for d in 0..hidden {
                    let v00 = table[(y0 * side + x0) * hidden + d];
                    let v01 = table[(y0 * side + x1) * hidden + d];
                    let v10 = table[(y1 * side + x0) * hidden + d];
                    let v11 = table[(y1 * side + x1) * hidden + d];
                    out[base + d] = (1.0 - dy) * (1.0 - dx) * v00
                        + (1.0 - dy) * dx * v01
                        + dy * (1.0 - dx) * v10
                        + dy * dx * v11;
                }
            }
        }
        out
    }

    /// Bidirectional attention within each temporal group, on already-rotated
    /// planes.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], rows: usize, plane: usize) -> Vec<f32> {
        let heads = self.dims.heads();
        let head_dim = self.dims.head_dim();
        let dim = heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; rows * dim];
        for n in 0..rows {
            let base = (n / plane) * plane;
            for h in 0..heads {
                let off = h * head_dim;
                let scores: Vec<f32> = (base..base + plane)
                    .map(|p| {
                        (0..head_dim)
                            .map(|d| q[n * dim + off + d] * k[p * dim + off + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let den: f32 = weights.iter().sum();
                for d in 0..head_dim {
                    let num: f32 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| w * v[(base + i) * dim + off + d])
                        .sum();
                    out[n * dim + off + d] = num / den;
                }
            }
        }
        out
    }

    /// Splits the fused projection and rotates the query and key in two axes.
    fn rope(
        &self,
        qkv: &[f32],
        rows: usize,
        grid_w: usize,
        plane: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let heads = self.dims.heads();
        let head_dim = self.dims.head_dim();
        let dim = heads * head_dim;
        let half = head_dim / 2;
        let per_axis = half / 2;
        let mut q = vec![0.0f32; rows * dim];
        let mut k = vec![0.0f32; rows * dim];
        let mut v = vec![0.0f32; rows * dim];
        for n in 0..rows {
            let local = n % plane;
            let row = (local / grid_w) as f32;
            let col = (local % grid_w) as f32;
            for h in 0..heads {
                let off = h * head_dim;
                for i in 0..head_dim {
                    v[n * dim + off + i] = qkv[n * 3 * dim + 2 * dim + off + i];
                }
                for i in 0..half {
                    let (freq, pos) = if i < per_axis {
                        (i, row)
                    } else {
                        (i - per_axis, col)
                    };
                    let theta = pos * self.rope_base.powf(-2.0 * freq as f32 / half as f32);
                    let (s, c) = (theta.sin(), theta.cos());
                    for (plane_out, src_base) in [(&mut q, 0), (&mut k, dim)] {
                        let a = qkv[n * 3 * dim + src_base + off + i];
                        let b = qkv[n * 3 * dim + src_base + off + i + half];
                        plane_out[n * dim + off + i] = a * c - b * s;
                        plane_out[n * dim + off + i + half] = b * c + a * s;
                    }
                }
            }
        }
        (q, k, v)
    }

    /// The whole tower, from patch rows to merged output.
    fn forward(&self, item: &VisionItem) -> Vec<f32> {
        let (grid_t, grid_h, grid_w) = item.grid();
        let plane = grid_h * grid_w;
        let rows = grid_t * plane;
        let hidden = self.dims.hidden();

        let mut state = Self::linear(
            item.as_flat(),
            rows,
            self.dims.patch_dim(),
            hidden,
            &self.tree.patch_embed.weight,
            &self.tree.patch_embed.bias,
        );
        let positions = self.positions(grid_h, grid_w);
        for (i, value) in state.iter_mut().enumerate() {
            *value += positions[i % (plane * hidden)];
        }
        for block in &self.tree.blocks {
            state = self.block(block, state, grid_w, plane);
        }
        self.merge(&state, grid_t, grid_h, grid_w)
    }

    /// One block: normalize, attend, add back; normalize, feed forward, add back.
    fn block(
        &self,
        block: &crate::qwen35::vision::manifest::TowerBlock<Vec<f32>>,
        mut state: Vec<f32>,
        grid_w: usize,
        plane: usize,
    ) -> Vec<f32> {
        let hidden = self.dims.hidden();
        let rows = state.len() / hidden;

        let normed = Self::layer_norm(
            &state,
            rows,
            hidden,
            &block.norm1.weight,
            &block.norm1.bias,
            self.eps,
        );
        let qkv = Self::linear(
            &normed,
            rows,
            hidden,
            3 * hidden,
            &block.qkv.weight,
            &block.qkv.bias,
        );
        let (q, k, v) = self.rope(&qkv, rows, grid_w, plane);
        let attended = self.attention(&q, &k, &v, rows, plane);
        let projected = Self::linear(
            &attended,
            rows,
            hidden,
            hidden,
            &block.proj.weight,
            &block.proj.bias,
        );
        for (slot, value) in state.iter_mut().enumerate() {
            *value += projected[slot];
        }

        let normed = Self::layer_norm(
            &state,
            rows,
            hidden,
            &block.norm2.weight,
            &block.norm2.bias,
            self.eps,
        );
        let inner = self.dims.intermediate();
        let mut wide = Self::linear(
            &normed,
            rows,
            hidden,
            inner,
            &block.fc1.weight,
            &block.fc1.bias,
        );
        for value in &mut wide {
            *value = Self::gelu_tanh(*value);
        }
        let back = Self::linear(
            &wide,
            rows,
            inner,
            hidden,
            &block.fc2.weight,
            &block.fc2.bias,
        );
        for (slot, value) in state.iter_mut().enumerate() {
            *value += back[slot];
        }
        state
    }

    /// The merger: normalize each patch row, fold the square block, then two
    /// layers with the exact activation between them.
    fn merge(&self, state: &[f32], grid_t: usize, grid_h: usize, grid_w: usize) -> Vec<f32> {
        let hidden = self.dims.hidden();
        let plane = grid_h * grid_w;
        let rows = grid_t * plane;
        let merge = self.dims.geometry().spatial_merge_size();
        let merger = &self.tree.merger;
        let normed = Self::layer_norm(
            state,
            rows,
            hidden,
            &merger.norm.weight,
            &merger.norm.bias,
            self.eps,
        );
        let merge_input = self.dims.merge_input();
        let tall = grid_h / merge;
        let wide = grid_w / merge;
        let merged_rows = grid_t * tall * wide;
        let mut folded = vec![0.0f32; merged_rows * merge_input];
        for o in 0..merged_rows {
            let group = o / (tall * wide);
            let local = o % (tall * wide);
            let (oh, ow) = (local / wide, local % wide);
            for slot in 0..merge * merge {
                let ph = oh * merge + slot / merge;
                let pw = ow * merge + slot % merge;
                let src = (group * plane + ph * grid_w + pw) * hidden;
                for d in 0..hidden {
                    folded[o * merge_input + slot * hidden + d] = normed[src + d];
                }
            }
        }
        let mut inner = Self::linear(
            &folded,
            merged_rows,
            merge_input,
            merge_input,
            &merger.fc1.weight,
            &merger.fc1.bias,
        );
        for value in &mut inner {
            *value = Self::gelu_erf(*value);
        }
        Self::linear(
            &inner,
            merged_rows,
            merge_input,
            self.dims.out_hidden(),
            &merger.fc2.weight,
            &merger.fc2.bias,
        )
    }
}

/// Runs the device encoder and the host reference over the same tower and item,
/// and reports the two.
fn encode_both(
    dims: VisionDims,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    seed: u32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let backend = device()?;
    let (file, values) = synthetic_tower(dims, seed);
    let weights = load_vision_weights(&backend, &VisionSource::SidecarGguf(file.path().into()))
        .expect("the synthetic tower loads");
    let item = fixture_item(dims, grid_t, grid_h, grid_w, seed ^ 0x5a5a);

    let encoder = VisionEncoder::new();
    let out = encoder
        .encode(&backend, &weights, &item)
        .expect("the encoder runs");
    let mut bytes = vec![0u8; usize::try_from(out.len()).expect("the output fits")];
    backend
        .copy_d2h(out.view(), &mut bytes)
        .expect("the output reads back");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_bits(u32::from_le_bytes(*w)))
        .collect();

    let reference = Reference {
        dims,
        eps: encoder.layer_norm_eps(),
        rope_base: encoder.rope_base(),
        tree: &values,
    };
    let want = reference.forward(&item);
    assert_eq!(
        got.len(),
        item.pad_count(dims.geometry().spatial_merge_size()) * dims.out_hidden(),
        "the encoder must emit one row per placeholder token",
    );
    Some((got, want))
}

/// Asserts the device agrees with the reference within [`TOL`].
fn agree(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = TOL * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= bound,
            "{what}: value {index} is {g}, the reference says {w} (tolerance {bound})",
        );
    }
}

// -- the cases --------------------------------------------------------------

#[test]
#[ignore = "requires a Metal device"]
fn a_block_and_the_merger_agree_with_the_reference_at_the_shipped_widths() {
    let Some((got, want)) = encode_both(shipped_dims(), 1, 2, 2, 101) else {
        return;
    };
    agree(&got, &want, "shipped widths");
}

#[test]
#[ignore = "requires a Metal device"]
fn a_block_and_the_merger_agree_with_the_reference_at_a_second_set_of_widths() {
    // Nothing about this set is the shipped one: a wider tower, a head width of
    // 72 rather than 64, a feed-forward size that is not a power of two and a
    // position table that is not the shipped square. If any of those numbers
    // were a literal in the encoder, only one of these two cases could pass.
    let Some((got, want)) = encode_both(second_dims(), 1, 2, 4, 202) else {
        return;
    };
    agree(&got, &want, "second widths");
}

#[test]
#[ignore = "requires a Metal device"]
fn a_clip_encodes_every_temporal_group_without_letting_them_mix() {
    // Two temporal groups on one grid. The reference bounds attention by the
    // group and reuses one grid's position rows for both, so a device that
    // attended across the boundary or walked the position rows off the end of
    // the first group would disagree here and nowhere else.
    let Some((got, want)) = encode_both(second_dims(), 2, 2, 4, 303) else {
        return;
    };
    agree(&got, &want, "two temporal groups");
}

// -- the refusals, which need no device --------------------------------------

/// A backend that refuses everything, for the cases that must fail before any
/// device work is attempted.
#[derive(Debug)]
struct NoDevice;

impl Backend for NoDevice {
    fn backend_name(&self) -> &'static str {
        "none"
    }

    fn alloc_raw(&self, _bytes: u64) -> shiso_core::CoreResult<shiso_core::RawBuffer> {
        Err(refusal("alloc_raw"))
    }

    fn free_raw(&self, _raw: shiso_core::RawBuffer) -> shiso_core::CoreResult<()> {
        Ok(())
    }

    fn copy_h2d(
        &self,
        _dst: shiso_core::DeviceView<'_>,
        _src: &[u8],
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_h2d"))
    }

    fn copy_d2h(
        &self,
        _src: shiso_core::DeviceView<'_>,
        _dst: &mut [u8],
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_d2h"))
    }

    fn copy_d2d(
        &self,
        _dst: shiso_core::DeviceView<'_>,
        _src: shiso_core::DeviceView<'_>,
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("copy_d2d"))
    }

    fn run(
        &self,
        _kernel: shiso_core::KernelRef,
        _args: &shiso_core::LaunchArgs<'_>,
    ) -> shiso_core::CoreResult<()> {
        Err(refusal("run"))
    }

    fn sync(&self) -> shiso_core::CoreResult<()> {
        Err(refusal("sync"))
    }
}

fn refusal(op: &'static str) -> shiso_core::CoreError {
    shiso_core::CoreError::Backend {
        backend: "none",
        op,
        source: Box::new(std::io::Error::other("this backend does nothing")),
    }
}

/// The shape checks have to run before anything is allocated, so they are
/// reachable on a backend that cannot allocate at all. Building a loaded tower
/// needs a device, so these check `Plan` directly - the type that holds every
/// one of them.
#[test]
fn a_grid_that_is_not_whole_merge_blocks_is_refused_by_name() {
    let dims = second_dims();
    let bad = fixture_item(dims, 1, 3, 4, 1);
    match super::Plan::new(dims, &bad) {
        Err(VisionError::ItemGridNotMergeable {
            grid_h,
            grid_w,
            merge,
        }) => assert_eq!((grid_h, grid_w, merge), (3, 4, 2)),
        other => panic!("expected a merge-block error, got {other:?}"),
    }
}

#[test]
fn an_item_preprocessed_for_another_tower_is_refused_by_width() {
    let dims = second_dims();
    let rows = 4;
    let mismatched = VisionItem::new(vec![0.0f32; rows * 64].into_boxed_slice(), 64, 1, 2, 2)
        .expect("the fixture grid and pixels agree");
    match super::Plan::new(dims, &mismatched) {
        Err(VisionError::ItemRowWidth { expected, got }) => {
            assert_eq!((expected, got), (dims.patch_dim(), 64));
        }
        other => panic!("expected a row-width error, got {other:?}"),
    }
}

#[test]
fn a_head_wider_than_the_attention_kernel_holds_is_refused() {
    let dims = VisionDims::new(
        1,
        1024,
        2,
        4096,
        2560,
        1024,
        PatchGeometry::new(16, 2, 2).expect("a geometry"),
    )
    .expect("the widths are valid");
    let bad = fixture_item(dims, 1, 2, 2, 1);
    match super::Plan::new(dims, &bad) {
        Err(VisionError::HeadTooWide { head_dim, max }) => assert_eq!((head_dim, max), (512, 256)),
        other => panic!("expected a head-width error, got {other:?}"),
    }
}

/// The encoder never hands a raw device error back: every failure names the step
/// it happened in, so an operator reading one line knows which part of the tower
/// stopped.
#[test]
fn a_device_failure_names_the_step_it_happened_in() {
    let backend: Arc<dyn Backend> = Arc::new(NoDevice);
    let encoder = VisionEncoder::new();
    let run = super::Run {
        backend: &backend,
        eps: encoder.layer_norm_eps(),
        rope_base: encoder.rope_base(),
        fast: super::Fast::SCALAR,
        operand: None,
        split: None,
    };
    match run.scratch("patch rows", 16) {
        Err(VisionError::Encode { step, .. }) => assert_eq!(step, "patch rows"),
        other => panic!("expected a named encode failure, got {other:?}"),
    }
}

/// The defaults are the numbers both descriptions of this tower state, and the
/// builders are the door for a tower that says otherwise.
#[test]
fn the_encoder_carries_the_two_constants_no_container_states() {
    let encoder = VisionEncoder::new();
    assert!((encoder.layer_norm_eps() - 1e-6).abs() < f32::EPSILON);
    assert!((encoder.rope_base() - 10_000.0).abs() < f32::EPSILON);
    let tuned = VisionEncoder::default()
        .with_layer_norm_eps(1e-5)
        .with_rope_base(50_000.0);
    assert!((tuned.layer_norm_eps() - 1e-5).abs() < f32::EPSILON);
    assert!((tuned.rope_base() - 50_000.0).abs() < f32::EPSILON);
}

/// The resample has to land on the table's corners exactly, and a one-patch grid
/// has no interval to spread across.
#[test]
fn the_position_sampler_lands_on_the_corners_and_survives_a_single_patch() {
    // Five patches across a table of five: first on row 0, last on row 4, and
    // the fraction is zero at both ends because both land on a table row.
    let first = super::sample(0, 5, 4.0, 5);
    assert_eq!((first.0, first.1), (0, 1));
    assert!(first.2.abs() < f32::EPSILON);
    let last = super::sample(4, 5, 4.0, 5);
    assert_eq!(
        (last.0, last.1),
        (4, 4),
        "the last patch lands on the last row"
    );
    assert!(last.2.abs() < f32::EPSILON);
    // A grid of one samples the first row rather than dividing by zero.
    let single = super::sample(0, 1, 4.0, 5);
    assert_eq!((single.0, single.1), (0, 1));
    assert!(single.2.abs() < f32::EPSILON);
    // Halfway between the corners of a five-row table is row two, exactly.
    let middle = super::sample(1, 3, 4.0, 5);
    assert_eq!((middle.0, middle.1), (2, 3));
    assert!(middle.2.abs() < f32::EPSILON);
}
