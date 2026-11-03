//! The FLUX.3 diffusion transformer: per-mode block stacks, then a shared
//! single-stream tail, with the time conditioning applied per mode.
//!
//! The graph follows the reference's `DiT` under the checkpoint's own tensor
//! names. Tokens belong to one of three modes (`image`, `image_cond`,
//! `text`). Each mode first runs its own stack of blocks (the two content
//! modes keep one stack each, the text its own) over its tokens alone, with
//! its own modulation; the modes are then concatenated per sample in
//! mode-name order and run through the shared single-stream blocks, where
//! every token is modulated by its own mode's vector. A block is parallel
//! attention and MLP under one gated residual:
//! `x + gate * (attn_out(attention) + mlp_out(swiglu(mlp_in(x_mod))))`,
//! with separate q, k and v projections, per-head RMS QK-norm and the
//! four-axis interleaved rotary embedding.
//!
//! With one sample each mode has a single time vector, so the per-token
//! modulation is applied per row segment with the existing modulate and
//! gated-residual entries; no per-token kernel exists or is needed. The
//! `image` vector carries the step's time, the `image_cond` and `text`
//! vectors carry time zero, and every mode adds the flow map's embedding of
//! the time the step jumps to.
//!
//! Weights run in their stored format, mixed per tensor (a block's q
//! projection may be `Q4_K` while its v projection is `Q6_K`); none is
//! dequantized into a duplicate buffer. An absent `image_cond` mode is
//! skipped outright: the reference pushes one phantom token through its
//! projections and never attends it, and no phantom token exists here to
//! leak.

use std::ops::Range;

use half::bf16;
use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::config::Flux3Params;
use super::pack::{Mode, Pack};
use crate::diffusion::lin::Lin;
use crate::diffusion::ops::{
    GemmCall, LayerNormCall, LinearResidualCall, ModulateCall, NormCall, Ops, Rope4dCall, Scratch,
    SdpaCall, SliceExt,
};
use crate::diffusion::weights::StStore;
use crate::flux2::rope::rope_4d_freqs;

/// `LayerNorm` and QK-norm epsilon of the reference blocks.
const NORM_EPS: f32 = 1e-6;
/// Width of the sinusoidal time embedding.
const TIME_DIM: usize = 256;
/// The reference's timestep scale, applied inside the embedding.
const TIME_FACTOR: f32 = 1000.0;
/// Rotary widths per axis; they sum to the head width.
const AXES_DIM: [usize; 4] = [32, 32, 32, 32];
/// Rotary base.
const THETA: f32 = 10_000.0;
/// Width of every head, which the attention kernels are built for.
const HEAD_DIM: usize = 128;

/// One transformer block's weights, per-mode or single-stream alike.
#[derive(Debug)]
pub(super) struct Block {
    q: Lin,
    k: Lin,
    v: Lin,
    attn_out: Lin,
    mlp_in: Lin,
    mlp_out: Lin,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
}

/// The loaded diffusion transformer.
#[derive(Debug)]
pub(crate) struct DiT {
    params: Flux3Params,
    emb_image: Lin,
    emb_cond: Lin,
    txt_in: Lin,
    time_in: [Lin; 2],
    ref_time_in: [Lin; 2],
    /// Per-mode early modulation, indexed `[image, image_cond, text]`.
    early_mod: [Lin; 3],
    /// Per-mode single-stream modulation, in the same order.
    single_mod: [Lin; 3],
    final_ada: Lin,
    final_linear: Lin,
    pub(super) image_blocks: Vec<Block>,
    cond_blocks: Vec<Block>,
    text_blocks: Vec<Block>,
    pub(super) single_blocks: Vec<Block>,
    rope_freqs: DeviceBuffer,
}

/// One forward pass's inputs. Rotary ids are `(rows, 4)` f32 device
/// buffers the caller builds once per generation.
pub(crate) struct DiTInput<'a> {
    /// `(image_rows, 128)` f32 noisy latent tokens.
    pub image: DeviceView<'a>,
    /// `(cond_rows, 128)` f32 reference tokens, or `None` without references.
    pub image_cond: Option<DeviceView<'a>>,
    /// `(text_rows, 20480)` f32 text conditioning.
    pub text: DeviceView<'a>,
    /// Rows of `image`.
    pub image_rows: usize,
    /// Rows of `image_cond` (0 when it is `None`).
    pub cond_rows: usize,
    /// Rows of `text`.
    pub text_rows: usize,
    /// The step's time, in `[0, 1]`.
    pub t: f32,
    /// The time the step jumps to (the flow map's target).
    pub t_ref: f32,
    /// Rotary ids of the image tokens.
    pub image_ids: DeviceView<'a>,
    /// Rotary ids of the reference tokens, present with `image_cond`.
    pub cond_ids: Option<DeviceView<'a>>,
    /// Rotary ids of the text tokens.
    pub text_ids: DeviceView<'a>,
}

/// The (shift, scale, gate) of one mode's modulation, three `hidden`-wide
/// vectors in one buffer.
pub(super) struct ModVec<'o> {
    pub(super) buffer: Scratch<'o>,
    pub(super) hidden: usize,
}

impl ModVec<'_> {
    fn part(&self, index: usize) -> Result<DeviceView<'_>, ImageGenError> {
        let bytes = self.hidden as u64 * 4;
        self.buffer.view().cut(index as u64 * bytes, bytes)
    }
}

/// Rows of a block's stream that share one modulation.
pub(super) struct Segment<'a, 'o> {
    pub(super) rows: Range<usize>,
    pub(super) modulation: &'a ModVec<'o>,
}

/// What a block runs over: the stream, its rotary ids, and the modulation
/// segments that tile its rows.
pub(super) struct Stream<'a, 'o> {
    pub(super) x: &'a DeviceBuffer,
    pub(super) rows: usize,
    pub(super) ids: DeviceView<'a>,
    pub(super) segments: Vec<Segment<'a, 'o>>,
}

/// A stream whose rows all share one modulation.
fn whole_stream<'a, 'o>(
    x: &'a DeviceBuffer,
    rows: usize,
    ids: DeviceView<'a>,
    modulation: &'a ModVec<'o>,
) -> Stream<'a, 'o> {
    Stream {
        x,
        rows,
        ids,
        segments: vec![Segment {
            rows: 0..rows,
            modulation,
        }],
    }
}

/// `rows` rows of `width` f32 starting at row `start` of `buffer`.
fn cut_rows(
    buffer: &DeviceBuffer,
    start: usize,
    rows: usize,
    width: usize,
) -> Result<DeviceView<'_>, ImageGenError> {
    let row_bytes = width as u64 * 4;
    buffer
        .view()
        .cut(start as u64 * row_bytes, rows as u64 * row_bytes)
}

/// Rounds to the nearest brain float, as the reference's bf16 time tensors do.
fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

/// The sinusoidal embedding of time `t`, as the reference computes it in
/// brain float: `t` and `1000 * t` are rounded to bf16, the frequencies
/// `exp(-ln(10000) * i / 128)` are f32, and the `[cos | sin]` result is
/// rounded to bf16 again.
#[expect(
    clippy::cast_precision_loss,
    reason = "frequency indices are below 128, exact in f32"
)]
pub(crate) fn time_embedding(t: f32) -> Vec<f32> {
    let t = round_bf16(TIME_FACTOR * round_bf16(t));
    let half = TIME_DIM / 2;
    let args: Vec<f32> = (0..half)
        .map(|i| t * (-(10_000f32).ln() * i as f32 / half as f32).exp())
        .collect();
    args.iter()
        .map(|a| a.cos())
        .chain(args.iter().map(|a| a.sin()))
        .map(round_bf16)
        .collect()
}

fn checkpoint(message: String) -> ImageGenError {
    ImageGenError::Checkpoint(message)
}

/// Refuses a linear whose shape is not the one the graph needs.
fn expect_shape(lin: &Lin, out_dim: usize, in_dim: usize, name: &str) -> Result<(), ImageGenError> {
    if lin.out_dim == out_dim && lin.in_dim == in_dim {
        Ok(())
    } else {
        Err(checkpoint(format!(
            "'{name}' is {}x{}, the FLUX.3 graph needs {out_dim}x{in_dim}",
            lin.out_dim, lin.in_dim
        )))
    }
}

fn load_lin(
    store: &StStore,
    ops: &Ops,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Lin, ImageGenError> {
    let lin = Lin::load(store, ops, name)?;
    expect_shape(&lin, out_dim, in_dim, name)?;
    Ok(lin)
}

fn load_norm(store: &StStore, ops: &Ops, name: &str) -> Result<DeviceBuffer, ImageGenError> {
    let values = store.f32_vec(name)?;
    if values.len() != HEAD_DIM {
        return Err(checkpoint(format!(
            "'{name}' holds {} scales for a {HEAD_DIM}-wide head",
            values.len()
        )));
    }
    ops.upload_f32("qk norm scale", &values)
}

fn load_block(
    store: &StStore,
    ops: &Ops,
    prefix: &str,
    hidden: usize,
    mlp: usize,
) -> Result<Block, ImageGenError> {
    let lin = |part: &str, out_dim: usize, in_dim: usize| {
        load_lin(store, ops, &format!("{prefix}.{part}"), out_dim, in_dim)
    };
    Ok(Block {
        q: lin("q_proj", hidden, hidden)?,
        k: lin("k_proj", hidden, hidden)?,
        v: lin("v_proj", hidden, hidden)?,
        attn_out: lin("attn_out", hidden, hidden)?,
        mlp_in: lin("mlp_in", 2 * mlp, hidden)?,
        mlp_out: lin("mlp_out", hidden, mlp)?,
        q_norm: load_norm(store, ops, &format!("{prefix}.norm.query_norm.scale"))?,
        k_norm: load_norm(store, ops, &format!("{prefix}.norm.key_norm.scale"))?,
    })
}

impl DiT {
    /// Uploads every weight of the checkpoint in `store` in its stored
    /// format, refusing a linear whose shape is not the one `params` named.
    ///
    /// # Errors
    /// [`ImageGenError::Checkpoint`] naming the missing or misshapen tensor.
    pub(crate) fn load(
        store: &StStore,
        ops: &Ops,
        params: &Flux3Params,
    ) -> Result<Self, ImageGenError> {
        let h = params.hidden;
        let mlp = params.mlp_hidden;
        if params.heads * HEAD_DIM != h {
            return Err(checkpoint(format!(
                "{} heads of {HEAD_DIM} do not make a hidden width of {h}",
                params.heads
            )));
        }
        let stack = |prefix: &str, depth: usize| -> Result<Vec<Block>, ImageGenError> {
            (0..depth)
                .map(|i| load_block(store, ops, &format!("{prefix}.{i}"), h, mlp))
                .collect()
        };
        let pair = |name: &str, input: usize| -> Result<[Lin; 2], ImageGenError> {
            Ok([
                load_lin(store, ops, &format!("{name}.in_layer"), h, input)?,
                load_lin(store, ops, &format!("{name}.out_layer"), h, h)?,
            ])
        };
        let modulations = |stream: &str| -> Result<[Lin; 3], ImageGenError> {
            let one = |mode: &str| load_lin(store, ops, &format!("{stream}.{mode}.lin"), 3 * h, h);
            Ok([one("image")?, one("image_cond")?, one("txt")?])
        };
        Ok(Self {
            params: params.clone(),
            emb_image: load_lin(store, ops, "emb_in.image", h, params.img_in)?,
            emb_cond: load_lin(store, ops, "emb_in.image_cond", h, params.img_in)?,
            txt_in: load_lin(store, ops, "txt_in", h, params.txt_in)?,
            time_in: pair("time_in", TIME_DIM)?,
            ref_time_in: pair("ref_time_in", TIME_DIM)?,
            early_mod: modulations("early_stream_modulations")?,
            single_mod: modulations("single_stream_modulations")?,
            final_ada: load_lin(store, ops, "final_layer.image.adaLN_modulation.1", 2 * h, h)?,
            final_linear: load_lin(store, ops, "final_layer.image.linear", params.img_in, h)?,
            image_blocks: stack("content_mode_blocks.image", params.depth_mode)?,
            cond_blocks: stack("content_mode_blocks.image_cond", params.depth_mode)?,
            text_blocks: stack("txt_mode_blocks", params.depth_mode)?,
            single_blocks: stack("single_blocks", params.depth_single)?,
            rope_freqs: ops.upload_f32("dit rope freqs", &rope_4d_freqs(&AXES_DIM, THETA))?,
        })
    }

    /// `out_layer(silu(in_layer(embedding)))`, one row.
    fn time_mlp<'o>(
        &self,
        ops: &'o Ops,
        layers: &[Lin; 2],
        embedding: &[f32],
    ) -> Result<Scratch<'o>, ImageGenError> {
        let h = self.params.hidden;
        let input = ops.upload_scratch("dit time embedding", embedding)?;
        let hidden = ops.alloc_f32("dit time hidden", h)?;
        ops.gemm(&GemmCall {
            step: "dit time in",
            out: hidden.view(),
            input: input.view(),
            weight: layers[0].weight(),
            m: 1,
            n: h,
            k: TIME_DIM,
        })?;
        ops.silu("dit time silu", hidden.view(), hidden.view(), h)?;
        let out = ops.alloc_f32("dit time out", h)?;
        ops.gemm(&GemmCall {
            step: "dit time out",
            out: out.view(),
            input: hidden.view(),
            weight: layers[1].weight(),
            m: 1,
            n: h,
            k: h,
        })?;
        Ok(out)
    }

    /// `time_in(embedding(t)) + ref_time_in(embedding(t_ref))`: the vector a
    /// mode with time `t` is conditioned on.
    fn mode_vector<'o>(
        &self,
        ops: &'o Ops,
        t: f32,
        reference: &Scratch<'o>,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let h = self.params.hidden;
        let own = self.time_mlp(ops, &self.time_in, &time_embedding(t))?;
        let sum = ops.alloc_f32("dit time vec", h)?;
        ops.add("dit time sum", sum.view(), own.view(), reference.view(), h)?;
        Ok(sum)
    }

    /// `lin(silu(vec))`: a mode's `(shift, scale, gate)` triple.
    fn modulation<'o>(
        &self,
        ops: &'o Ops,
        lin: &Lin,
        vec_silu: &Scratch<'o>,
    ) -> Result<ModVec<'o>, ImageGenError> {
        let h = self.params.hidden;
        let buffer = ops.alloc_f32("dit modulation", 3 * h)?;
        ops.gemm(&GemmCall {
            step: "dit modulation",
            out: buffer.view(),
            input: vec_silu.view(),
            weight: lin.weight(),
            m: 1,
            n: 3 * h,
            k: h,
        })?;
        Ok(ModVec { buffer, hidden: h })
    }

    fn silu_of<'o>(&self, ops: &'o Ops, vec: &Scratch<'o>) -> Result<Scratch<'o>, ImageGenError> {
        let h = self.params.hidden;
        let out = ops.alloc_f32("dit vec silu", h)?;
        ops.silu("dit vec silu", out.view(), vec.view(), h)?;
        Ok(out)
    }

    /// Projects `input` rows through `lin` into a fresh `(rows, out)` buffer.
    fn project<'o>(
        ops: &'o Ops,
        step: &'static str,
        lin: &Lin,
        input: DeviceView<'_>,
        rows: usize,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let out = ops.alloc_f32(step, rows * lin.out_dim)?;
        ops.gemm(&GemmCall {
            step,
            out: out.view(),
            input,
            weight: lin.weight(),
            m: rows,
            n: lin.out_dim,
            k: lin.in_dim,
        })?;
        Ok(out)
    }

    /// One block over a stream: modulated attention and MLP in parallel,
    /// under one gated residual per modulation segment.
    #[expect(
        clippy::too_many_lines,
        reason = "one block's straight-line graph: norm, modulate, project, normalize, rotate, attend, gated residuals"
    )]
    pub(super) fn run_block(
        &self,
        ops: &Ops,
        block: &Block,
        stream: &Stream<'_, '_>,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden;
        let heads = self.params.heads;
        let mlp = self.params.mlp_hidden;
        let rows = stream.rows;
        let row_bytes = h as u64 * 4;

        let xn = ops.alloc_f32("dit norm", rows * h)?;
        ops.layer_norm(&LayerNormCall {
            step: "dit norm",
            out: xn.view(),
            input: stream.x.view(),
            rows,
            dim: h,
            eps: NORM_EPS,
        })?;
        for segment in &stream.segments {
            let part = xn.view().cut(
                segment.rows.start as u64 * row_bytes,
                segment.rows.len() as u64 * row_bytes,
            )?;
            ops.modulate(&ModulateCall {
                step: "dit modulate",
                out: part,
                x: part,
                scale: segment.modulation.part(1)?,
                shift: segment.modulation.part(0)?,
                count: segment.rows.len() * h,
                period: h,
            })?;
        }

        let q = Self::project(ops, "dit q", &block.q, xn.view(), rows)?;
        let k = Self::project(ops, "dit k", &block.k, xn.view(), rows)?;
        let v = Self::project(ops, "dit v", &block.v, xn.view(), rows)?;
        for (step, tensor, scale) in [
            ("dit q norm", &q, &block.q_norm),
            ("dit k norm", &k, &block.k_norm),
        ] {
            ops.rmsnorm(&NormCall {
                step,
                out: tensor.view(),
                input: tensor.view(),
                weight: scale.view(),
                rows: rows * heads,
                dim: HEAD_DIM,
                eps: NORM_EPS,
            })?;
        }
        ops.rope_4d(&Rope4dCall {
            step: "dit rope",
            q: q.view(),
            k: k.view(),
            positions: stream.ids,
            freqs: self.rope_freqs.view(),
            tokens: rows,
            heads,
            head_dim: HEAD_DIM,
            freqs_per_axis: AXES_DIM[0] / 2,
            row_stride: h,
            q_offset: 0,
            k_offset: 0,
        })?;

        // Every row of the stream attends every other: one sample, no mask.
        let attn = ops.alloc_f32("dit attn", rows * h)?;
        ops.sdpa(&SdpaCall {
            step: "dit attention",
            out_txt: attn.view(),
            out_img: attn.view(),
            q_txt: q.view(),
            k_txt: k.view(),
            v_txt: v.view(),
            q_img: q.view(),
            k_img: k.view(),
            v_img: v.view(),
            q_offset: 0,
            k_offset: 0,
            v_offset: 0,
            cache_k: ops.dummy_view(),
            cache_v: ops.dummy_view(),
            cache_stride: h,
            txt_tokens: rows,
            img_tokens: 0,
            cache_tokens: 0,
            heads,
            head_dim: HEAD_DIM,
            txt_stride: h,
            img_stride: h,
            causal: false,
        })?;

        let gate_up = Self::project(ops, "dit mlp in", &block.mlp_in, xn.view(), rows)?;
        // `silu(gate) * up` once over every row, then read as a dense input.
        // The fused tile could form it while staging, but every one of the
        // `h / 64` column tiles would recompute the same silu over the whole
        // 3h-wide row: measured on an M4 at 4096 rows, the fused mlp_out
        // took 134 ms against 81 ms for this pass plus the dense tile, with
        // the same values (one `dit_silu` and one product per element).
        let act = ops.alloc_f32("dit mlp act", rows * mlp)?;
        ops.swiglu(
            "dit mlp act",
            act.view(),
            gate_up.view(),
            rows,
            mlp,
            2 * mlp,
        )?;
        drop(gate_up);
        for segment in &stream.segments {
            let rows_in = segment.rows.len();
            let start = segment.rows.start;
            let residual = cut_rows(stream.x, start, rows_in, h)?;
            let gate = segment.modulation.part(2)?;
            ops.linear_residual(&LinearResidualCall {
                step: "dit attn out",
                residual,
                gate,
                dense: Some(cut_rows(&attn, start, rows_in, h)?),
                dense_cols: h,
                swiglu: None,
                weight: block.attn_out.weight(),
                m: rows_in,
                n: h,
            })?;
            ops.linear_residual(&LinearResidualCall {
                step: "dit mlp out",
                residual,
                gate,
                dense: Some(cut_rows(&act, start, rows_in, mlp)?),
                dense_cols: mlp,
                swiglu: None,
                weight: block.mlp_out.weight(),
                m: rows_in,
                n: h,
            })?;
        }
        Ok(())
    }

    /// One denoising forward pass: the velocity of the image tokens as
    /// `(image_rows, 128)` f32.
    ///
    /// # Errors
    /// [`ImageGenError::Backend`] for inputs whose rows, views and optional
    /// reference parts disagree, and the device's own failures.
    #[expect(
        clippy::too_many_lines,
        reason = "one denoising step's graph, straight-line by construction"
    )]
    pub(crate) fn forward<'o>(
        &self,
        ops: &'o Ops,
        input: &DiTInput<'_>,
    ) -> Result<Scratch<'o>, ImageGenError> {
        let p = &self.params;
        let h = p.hidden;
        let (n_image, n_cond, n_text) = (input.image_rows, input.cond_rows, input.text_rows);
        let with_cond = n_cond > 0;
        if n_image == 0 || n_text == 0 {
            return Err(ImageGenError::Backend(
                "a FLUX.3 step needs image and text tokens".to_string(),
            ));
        }
        if with_cond != input.image_cond.is_some() || with_cond != input.cond_ids.is_some() {
            return Err(ImageGenError::Backend(format!(
                "{n_cond} reference rows with image_cond {} and its ids {}: all three must agree",
                if input.image_cond.is_some() {
                    "present"
                } else {
                    "absent"
                },
                if input.cond_ids.is_some() {
                    "present"
                } else {
                    "absent"
                },
            )));
        }

        // Time vectors. The flow map's target embedding joins every mode;
        // the image mode carries the step's time, the others time zero.
        let reference = self.time_mlp(ops, &self.ref_time_in, &time_embedding(input.t_ref))?;
        let vec_image = self.mode_vector(ops, input.t, &reference)?;
        let vec_other = self.mode_vector(ops, 0.0, &reference)?;
        let silu_image = self.silu_of(ops, &vec_image)?;
        let silu_other = self.silu_of(ops, &vec_other)?;
        let early_image = self.modulation(ops, &self.early_mod[0], &silu_image)?;
        let early_text = self.modulation(ops, &self.early_mod[2], &silu_other)?;
        let early_cond = if with_cond {
            Some(self.modulation(ops, &self.early_mod[1], &silu_other)?)
        } else {
            None
        };
        let single_image = self.modulation(ops, &self.single_mod[0], &silu_image)?;
        let single_text = self.modulation(ops, &self.single_mod[2], &silu_other)?;
        let single_cond = if with_cond {
            Some(self.modulation(ops, &self.single_mod[1], &silu_other)?)
        } else {
            None
        };

        // Token embeddings.
        let x_image = Self::project(ops, "dit emb image", &self.emb_image, input.image, n_image)?;
        let x_text = Self::project(ops, "dit txt in", &self.txt_in, input.text, n_text)?;
        let x_cond = match input.image_cond {
            Some(tokens) => Some(Self::project(
                ops,
                "dit emb cond",
                &self.emb_cond,
                tokens,
                n_cond,
            )?),
            None => None,
        };

        // Per-mode stacks, each over its own tokens.
        for depth in 0..p.depth_mode {
            self.run_block(
                ops,
                &self.image_blocks[depth],
                &whole_stream(&x_image, n_image, input.image_ids, &early_image),
            )?;
            if let (Some(x), Some(ids), Some(modulation)) =
                (x_cond.as_ref(), input.cond_ids, early_cond.as_ref())
            {
                self.run_block(
                    ops,
                    &self.cond_blocks[depth],
                    &whole_stream(x, n_cond, ids, modulation),
                )?;
            }
            self.run_block(
                ops,
                &self.text_blocks[depth],
                &whole_stream(&x_text, n_text, input.text_ids, &early_text),
            )?;
        }

        // The tail: the modes concatenated in mode-name order.
        let pack = Pack::single(n_image, n_cond, n_text);
        let total = pack.stream_len();
        let joint = ops.alloc_f32("dit joint", total * h)?;
        let joint_ids = ops.alloc_f32("dit joint ids", total * 4)?;
        let mut segments = Vec::new();
        let mut start = 0usize;
        for (mode, rows) in pack.single_stream_order() {
            let (x, ids, modulation) = match mode {
                Mode::Image => (Some(&x_image), Some(input.image_ids), Some(&single_image)),
                Mode::ImageCond => (x_cond.as_ref(), input.cond_ids, single_cond.as_ref()),
                Mode::Text => (Some(&x_text), Some(input.text_ids), Some(&single_text)),
            };
            let (Some(x), Some(ids), Some(modulation)) = (x, ids, modulation) else {
                return Err(ImageGenError::Backend(format!(
                    "{mode:?} has rows but no tokens, ids or modulation"
                )));
            };
            let len = rows.len();
            ops.copy(
                "dit joint rows",
                joint
                    .view()
                    .cut((start * h) as u64 * 4, (len * h) as u64 * 4)?,
                x.view(),
            )?;
            ops.copy(
                "dit joint ids",
                joint_ids
                    .view()
                    .cut((start * 4) as u64 * 4, (len * 4) as u64 * 4)?,
                ids,
            )?;
            segments.push(Segment {
                rows: start..start + len,
                modulation,
            });
            start += len;
        }
        let tail = Stream {
            x: &joint,
            rows: total,
            ids: joint_ids.view(),
            segments,
        };
        for block in &self.single_blocks {
            self.run_block(ops, block, &tail)?;
        }

        // The final layer over the image rows, which lead the stream.
        let image_rows = joint.view().cut(0, (n_image * h) as u64 * 4)?;
        let xn = ops.alloc_f32("dit final norm", n_image * h)?;
        ops.layer_norm(&LayerNormCall {
            step: "dit final norm",
            out: xn.view(),
            input: image_rows,
            rows: n_image,
            dim: h,
            eps: NORM_EPS,
        })?;
        let final_mod = ops.alloc_f32("dit final mod", 2 * h)?;
        ops.gemm(&GemmCall {
            step: "dit final mod",
            out: final_mod.view(),
            input: silu_image.view(),
            weight: self.final_ada.weight(),
            m: 1,
            n: 2 * h,
            k: h,
        })?;
        ops.modulate(&ModulateCall {
            step: "dit final modulate",
            out: xn.view(),
            x: xn.view(),
            scale: final_mod.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: final_mod.view(),
            count: n_image * h,
            period: h,
        })?;
        Self::project(ops, "dit head", &self.final_linear, xn.view(), n_image)
    }
}
