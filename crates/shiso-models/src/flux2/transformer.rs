//! The FLUX.2 unified diffusion transformer: double-stream and single-stream
//! blocks over a joint text+image sequence.
//!
//! The graph mirrors the reference `Flux2` module under the checkpoint's own
//! tensor names. `AdaLN` modulation is *shared*: one `Modulation` linear per
//! stream kind (`double_stream_modulation_img.lin`, `..._txt`,
//! `single_stream_modulation`) projects `silu` of the timestep embedding once
//! per forward and every block of that kind reuses the chunks - there are no
//! per-block modulation weights. Double blocks norm with weightless
//! `LayerNorm` (mean-subtracting, eps 1e-6), apply `QK-norm` to the q and k
//! heads before `RoPE`, and run `SwiGLU` MLPs whose gate is the *first* half
//! of the fused row. Single blocks fuse qkv and the MLP gate/value into one
//! `linear1` (qkv | gate | value), attend over the joint sequence, and carry
//! a single gated residual around `linear2(cat(attn, swiglu))`. The final
//! layer is `LayerNorm` + modulation + `final_layer.linear`, which is the
//! velocity head itself.
//!
//! Weights run in their stored native format: BF16 rows are read by the BF16
//! GEMM, `F8_E4M3` rows by the FP8 GEMM with the tensor's `weight_scale`
//! folded into the accumulator epilogue, and F32 rows by the F32 GEMM. No
//! weight is ever converted, duplicated, or dequantized into an F32 buffer.
//!
//! The reference-image KV path is the plan's: step 0 (`Extract`) runs the
//! reference tokens through every block and keeps their post-`RoPE` fused
//! qkv rows; later steps (`Cached`) project and modulate only the denoise
//! tokens and read the cached rows as extra attention keys and values.

use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::config::Flux2Params;
use super::kv_cache::Flux2KvCache;
use super::ops::{
    AddGatedCall, ConcatCall, GemmCall, LayerNormCall, ModulateCall, Ops, QkNormCall, Rope4dCall,
    SdpaCall, SliceExt,
};
use super::weights::{Lin, StStore};

/// `LayerNorm` / `QK-norm` epsilon the reference blocks use.
const NORM_EPS: f32 = 1e-6;

/// The reference's `time_factor`: the pipeline hands the transformer the raw
/// sigma and the 1000x scaling happens inside the sinusoidal embedding.
const TIME_FACTOR: f32 = 1000.0;

/// What the transformer should do with reference-token keys and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvMode {
    /// No reference tokens in play.
    None,
    /// Reference tokens run with the batch; cache their rows afterwards.
    Extract,
    /// Reference tokens are absent; attend over the cached rows instead.
    #[expect(
        dead_code,
        reason = "the reference-image KV path is implemented per plan; no wire format carries image input yet, so no step runs in cached mode"
    )]
    Cached,
}

/// A `SwiGLU` feedforward pair (`img_mlp.0` / `img_mlp.2`).
#[derive(Debug)]
struct Mlp {
    fc1: Lin,
    fc2: Lin,
}

/// One double-stream block: two attention/MLP branches and the `QK-norm`
/// scale vectors each branch's attention carries. The `AdaLN` modulation is
/// *not* here - it is shared per forward, as in the reference.
#[derive(Debug)]
struct DoubleBlock {
    img_qkv: Lin,
    img_q_norm: DeviceBuffer,
    img_k_norm: DeviceBuffer,
    img_proj: Lin,
    img_mlp: Mlp,
    txt_qkv: Lin,
    txt_q_norm: DeviceBuffer,
    txt_k_norm: DeviceBuffer,
    txt_proj: Lin,
    txt_mlp: Mlp,
}

/// One single-stream block: the fused `linear1` (qkv | MLP gate | MLP value)
/// and the `linear2` that reads attention and MLP output together.
#[derive(Debug)]
struct SingleBlock {
    linear1: Lin,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    linear2: Lin,
}

/// One denoising forward pass; everything the blocks read, in one place.
pub(crate) struct DiTInput<'a> {
    /// (`img_tokens`, `in_channels`) f32 latent sequence.
    pub latents: DeviceView<'a>,
    /// (`txt_tokens`, `context_in_dim`) f32 text conditioning.
    pub txt: DeviceView<'a>,
    /// The raw denoising sigma in [0, 1]; the reference's 1000x time factor
    /// is applied inside the sinusoidal embedding, not by the caller.
    pub timestep: f32,
    /// (`img_tokens`, 4) axis positions for the image tokens.
    pub img_positions: DeviceView<'a>,
    /// (`txt_tokens`, 4) axis positions for the text tokens.
    pub txt_positions: DeviceView<'a>,
    /// (`txt_tokens` + `img_tokens`, 4) joint axis positions for single blocks.
    pub joint_positions: DeviceView<'a>,
    pub txt_tokens: usize,
    pub img_tokens: usize,
    /// Reference tokens at the head of the image stream (0 unless KV is in play).
    pub ref_tokens: usize,
    pub kv_mode: KvMode,
}

/// Everything a block call needs beyond its own weights and the shared
/// modulation: the residual streams (two buffers for double blocks, the
/// joint buffer for single blocks), the `RoPE` tables, and the KV-cache
/// state. A named bundle because the block geometry travels together and no
/// block may take twelve arguments.
struct StepCtx<'a, 'b> {
    x_img: &'b DeviceBuffer,
    x_txt: &'b DeviceBuffer,
    freqs: &'b DeviceBuffer,
    input: &'b DiTInput<'a>,
    kv: &'b mut Flux2KvCache,
    refs: usize,
    n_img: usize,
    t_txt: usize,
}

/// One stream's attention preparation: the stream buffer, its `QK-norm`
/// scales, its fused qkv weight, and its half of the shared modulation.
struct PreCall<'b> {
    qkv: &'b Lin,
    q_norm: &'b DeviceBuffer,
    k_norm: &'b DeviceBuffer,
    x: &'b DeviceBuffer,
    /// The mod1 (shift, scale) pair of the shared double-stream modulation.
    mod1: &'b DeviceBuffer,
    tokens: usize,
}

/// The loaded diffusion transformer.
#[derive(Debug)]
pub(crate) struct DiT {
    params: Flux2Params,
    img_in: Lin,
    txt_in: Lin,
    time_in_in: Lin,
    time_in_out: Lin,
    double_mod_img: Lin,
    double_mod_txt: Lin,
    single_mod: Lin,
    final_ada: Lin,
    final_linear: Lin,
    double: Vec<DoubleBlock>,
    single: Vec<SingleBlock>,
    inner: usize,
    head_dim: usize,
    freqs_per_axis: usize,
}

/// The reference's `timestep_embedding` frequency table:
/// `exp(-ln(10000) * i / half)`.
#[expect(
    clippy::cast_precision_loss,
    reason = "frequency indices and the embedding half-width are small config-bounded dimensions"
)]
fn time_freq(i: usize, half: usize) -> f32 {
    (-(10_000f32).ln() * (i as f32) / half.max(1) as f32).exp()
}

/// Refuses a linear whose shape does not match the reference graph below;
/// a checkpoint with different widths is a different model, not a default.
fn expect_shape(lin: &Lin, out_dim: usize, in_dim: usize, name: &str) -> Result<(), ImageGenError> {
    if lin.out_dim != out_dim || lin.in_dim != in_dim {
        return Err(ImageGenError::Checkpoint(format!(
            "'{name}' is {}x{}, the reference graph needs {out_dim}x{in_dim}",
            lin.out_dim, lin.in_dim
        )));
    }
    Ok(())
}

/// Uploads one `QK-norm` scale vector, refusing a length that is not a head.
fn load_norm_scale(
    store: &StStore,
    ops: &Ops,
    name: &str,
    head_dim: usize,
) -> Result<DeviceBuffer, ImageGenError> {
    let values = store.f32_vec(name)?;
    if values.len() != head_dim {
        return Err(ImageGenError::Checkpoint(format!(
            "'{name}' holds {} scales for a {head_dim}-wide head",
            values.len()
        )));
    }
    ops.upload_f32("qk norm scale", &values)
}

impl DiT {
    /// Opens the transformer component and uploads every weight in its
    /// stored format.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "mlp_ratio * hidden is a layer width bounded by the model config; it fits usize"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "mlp_ratio is a small config float whose product with hidden is exact in f32"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "mlp_ratio and hidden size are positive, so the product is non-negative"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "checkpoint discovery is a flat checklist: entry linears, shared modulations, per-block weights"
    )]
    pub(crate) fn load(
        store: &StStore,
        ops: &Ops,
        params: &Flux2Params,
    ) -> Result<Self, ImageGenError> {
        if params.use_guidance_embed {
            return Err(ImageGenError::Checkpoint(
                "guidance embedding is not supported: no guidance_in weights are loaded"
                    .to_string(),
            ));
        }
        let h = params.hidden_size;
        let heads = params.num_heads;
        if heads == 0 || !h.is_multiple_of(heads) {
            return Err(ImageGenError::Checkpoint(format!(
                "hidden size {h} must divide evenly across {heads} heads"
            )));
        }
        let head_dim = h / heads;
        let freqs_per_axis = params.axes_dim.first().copied().unwrap_or(0) / 2;
        if freqs_per_axis == 0 || !params.axes_dim.iter().all(|&a| a / 2 == freqs_per_axis) {
            return Err(ImageGenError::Checkpoint(
                "flux2 RoPE requires equal, non-zero axis widths".to_string(),
            ));
        }
        if params.axes_dim.iter().sum::<usize>() != head_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "axes_dim sums to {}, expected the positional dim {head_dim}",
                params.axes_dim.iter().sum::<usize>()
            )));
        }
        let inner = (params.mlp_ratio * h as f32) as usize;

        let img_in = Lin::load(store, ops, "img_in")?;
        let txt_in = Lin::load(store, ops, "txt_in")?;
        let time_in_in = Lin::load(store, ops, "time_in.in_layer")?;
        let time_in_out = Lin::load(store, ops, "time_in.out_layer")?;
        expect_shape(&img_in, h, params.in_channels, "img_in")?;
        expect_shape(&txt_in, h, params.context_in_dim, "txt_in")?;
        // The reference's `timestep_embedding` width is fixed at 256, so the
        // embedder's input must be too.
        expect_shape(&time_in_in, h, 256, "time_in.in_layer")?;
        expect_shape(&time_in_out, h, h, "time_in.out_layer")?;

        // The shared modulations: one linear per stream kind, applied to
        // silu(vec) once per forward (reference `Modulation`).
        let double_mod_img = Lin::load(store, ops, "double_stream_modulation_img.lin")?;
        let double_mod_txt = Lin::load(store, ops, "double_stream_modulation_txt.lin")?;
        let single_mod = Lin::load(store, ops, "single_stream_modulation.lin")?;
        expect_shape(
            &double_mod_img,
            6 * h,
            h,
            "double_stream_modulation_img.lin",
        )?;
        expect_shape(
            &double_mod_txt,
            6 * h,
            h,
            "double_stream_modulation_txt.lin",
        )?;
        expect_shape(&single_mod, 3 * h, h, "single_stream_modulation.lin")?;

        // The final layer: `adaLN_modulation.1` is the (shift, scale) source
        // and `linear` is the velocity head itself.
        let final_ada = Lin::load(store, ops, "final_layer.adaLN_modulation.1")?;
        let final_linear = Lin::load(store, ops, "final_layer.linear")?;
        expect_shape(&final_ada, 2 * h, h, "final_layer.adaLN_modulation.1")?;
        expect_shape(&final_linear, params.in_channels, h, "final_layer.linear")?;

        let mut double = Vec::with_capacity(params.depth);
        for i in 0..params.depth {
            let b = format!("double_blocks.{i}.");
            let img_qkv = Lin::load(store, ops, &format!("{b}img_attn.qkv"))?;
            let img_proj = Lin::load(store, ops, &format!("{b}img_attn.proj"))?;
            let txt_qkv = Lin::load(store, ops, &format!("{b}txt_attn.qkv"))?;
            let txt_proj = Lin::load(store, ops, &format!("{b}txt_attn.proj"))?;
            expect_shape(&img_qkv, 3 * h, h, &format!("{b}img_attn.qkv"))?;
            expect_shape(&img_proj, h, h, &format!("{b}img_attn.proj"))?;
            expect_shape(&txt_qkv, 3 * h, h, &format!("{b}txt_attn.qkv"))?;
            expect_shape(&txt_proj, h, h, &format!("{b}txt_attn.proj"))?;
            let img_mlp = Mlp {
                fc1: Lin::load(store, ops, &format!("{b}img_mlp.0"))?,
                fc2: Lin::load(store, ops, &format!("{b}img_mlp.2"))?,
            };
            let txt_mlp = Mlp {
                fc1: Lin::load(store, ops, &format!("{b}txt_mlp.0"))?,
                fc2: Lin::load(store, ops, &format!("{b}txt_mlp.2"))?,
            };
            expect_shape(&img_mlp.fc1, 2 * inner, h, &format!("{b}img_mlp.0"))?;
            expect_shape(&img_mlp.fc2, h, inner, &format!("{b}img_mlp.2"))?;
            expect_shape(&txt_mlp.fc1, 2 * inner, h, &format!("{b}txt_mlp.0"))?;
            expect_shape(&txt_mlp.fc2, h, inner, &format!("{b}txt_mlp.2"))?;
            double.push(DoubleBlock {
                img_qkv,
                img_q_norm: load_norm_scale(
                    store,
                    ops,
                    &format!("{b}img_attn.norm.query_norm.scale"),
                    head_dim,
                )?,
                img_k_norm: load_norm_scale(
                    store,
                    ops,
                    &format!("{b}img_attn.norm.key_norm.scale"),
                    head_dim,
                )?,
                img_proj,
                img_mlp,
                txt_qkv,
                txt_q_norm: load_norm_scale(
                    store,
                    ops,
                    &format!("{b}txt_attn.norm.query_norm.scale"),
                    head_dim,
                )?,
                txt_k_norm: load_norm_scale(
                    store,
                    ops,
                    &format!("{b}txt_attn.norm.key_norm.scale"),
                    head_dim,
                )?,
                txt_proj,
                txt_mlp,
            });
        }
        let mut single = Vec::with_capacity(params.depth_single_blocks);
        for i in 0..params.depth_single_blocks {
            let b = format!("single_blocks.{i}.");
            let linear1 = Lin::load(store, ops, &format!("{b}linear1"))?;
            let linear2 = Lin::load(store, ops, &format!("{b}linear2"))?;
            // linear1 emits qkv (3h) plus the MLP gate and value halves
            // (inner each); linear2 reads attention plus the MLP activation.
            expect_shape(&linear1, 3 * h + 2 * inner, h, &format!("{b}linear1"))?;
            expect_shape(&linear2, h, h + inner, &format!("{b}linear2"))?;
            single.push(SingleBlock {
                linear1,
                q_norm: load_norm_scale(
                    store,
                    ops,
                    &format!("{b}norm.query_norm.scale"),
                    head_dim,
                )?,
                k_norm: load_norm_scale(store, ops, &format!("{b}norm.key_norm.scale"), head_dim)?,
                linear2,
            });
        }

        Ok(Self {
            params: params.clone(),
            img_in,
            txt_in,
            time_in_in,
            time_in_out,
            double_mod_img,
            double_mod_txt,
            single_mod,
            final_ada,
            final_linear,
            double,
            single,
            inner,
            head_dim,
            freqs_per_axis,
        })
    }

    /// Patchified output width of the final projection.
    pub(crate) fn out_width(&self) -> usize {
        self.final_linear.out_dim
    }

    /// Runs one denoising step, returning the velocity for the image tokens
    /// as (rows, `out_width`) f32. In `Cached` mode the image stream already
    /// excludes the reference tokens, so every row is a denoise token.
    #[expect(
        clippy::too_many_lines,
        reason = "one denoising step's block graph, straight-line by construction"
    )]
    pub(crate) fn forward(
        &self,
        ops: &Ops,
        input: &DiTInput<'_>,
        kv: &mut Flux2KvCache,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let p = &self.params;
        let h = p.hidden_size;

        let t_txt = input.txt_tokens;
        let n_img = input.img_tokens;
        let refs = if input.kv_mode == KvMode::None {
            0
        } else {
            input.ref_tokens
        };
        let freqs = ops.upload_f32(
            "dit rope freqs",
            &super::rope::rope_4d_freqs(&p.axes_dim, p.theta),
        )?;

        // Timestep embedding (the reference's `timestep_embedding`): the raw
        // sigma is scaled by 1000 here, the table is cos-then-sin, f32.
        let half = self.time_in_in.in_dim / 2;
        let t = input.timestep * TIME_FACTOR;
        let mut emb = Vec::with_capacity(2 * half);
        for i in 0..half {
            emb.push((t * time_freq(i, half)).cos());
        }
        for i in 0..half {
            emb.push((t * time_freq(i, half)).sin());
        }
        let emb_buf = ops.upload_f32("dit timestep", &emb)?;

        // `MLPEmbedder`: out_layer(silu(in_layer(emb))). The two GEMMs write
        // distinct buffers; an in-place m=1 GEMM would race its accumulator
        // against the input it reads.
        let temb1 = ops.alloc_f32("dit temb", h)?;
        ops.gemm(&GemmCall {
            step: "dit time in",
            out: temb1.view(),
            input: emb_buf.view(),
            weight: self.time_in_in.w.view(),
            weight_scale: self.time_in_in.scale,
            bias: self.time_in_in.bias.as_ref().map(DeviceBuffer::view),
            m: 1,
            n: h,
            k: self.time_in_in.in_dim,
            kind: self.time_in_in.kind,
        })?;
        ops.silu("dit time silu", temb1.view(), temb1.view(), h)?;
        let vec = ops.alloc_f32("dit vec", h)?;
        ops.gemm(&GemmCall {
            step: "dit time out",
            out: vec.view(),
            input: temb1.view(),
            weight: self.time_in_out.w.view(),
            weight_scale: self.time_in_out.scale,
            bias: self.time_in_out.bias.as_ref().map(DeviceBuffer::view),
            m: 1,
            n: h,
            k: h,
            kind: self.time_in_out.kind,
        })?;
        let vec_silu = ops.alloc_f32("dit vec silu", h)?;
        ops.silu("dit vec silu", vec_silu.view(), vec.view(), h)?;

        // Shared modulation, computed once per step: every double block
        // reuses the img/txt (shift, scale, gate) x 2 chunks, every single
        // block the (shift, scale, gate) triple, and the final layer its
        // (shift, scale) pair - the reference computes these once too.
        let mod_img = self.mod_vec(ops, &self.double_mod_img, &vec_silu, 6 * h, "dit mod img")?;
        let mod_txt = self.mod_vec(ops, &self.double_mod_txt, &vec_silu, 6 * h, "dit mod txt")?;
        let mod_single = self.mod_vec(ops, &self.single_mod, &vec_silu, 3 * h, "dit mod single")?;
        let mod_final = self.mod_vec(ops, &self.final_ada, &vec_silu, 2 * h, "dit mod final")?;

        let x_img = ops.alloc_f32("dit x_img", n_img * h)?;
        ops.gemm(&GemmCall {
            step: "dit img_in",
            out: x_img.view(),
            input: input.latents,
            weight: self.img_in.w.view(),
            weight_scale: self.img_in.scale,
            bias: self.img_in.bias.as_ref().map(DeviceBuffer::view),
            m: n_img,
            n: h,
            k: p.in_channels,
            kind: self.img_in.kind,
        })?;
        let x_txt = ops.alloc_f32("dit x_txt", t_txt * h)?;
        ops.gemm(&GemmCall {
            step: "dit txt_in",
            out: x_txt.view(),
            input: input.txt,
            weight: self.txt_in.w.view(),
            weight_scale: self.txt_in.scale,
            bias: self.txt_in.bias.as_ref().map(DeviceBuffer::view),
            m: t_txt,
            n: h,
            k: p.context_in_dim,
            kind: self.txt_in.kind,
        })?;

        for (i, block) in self.double.iter().enumerate() {
            let mut ctx = StepCtx {
                x_img: &x_img,
                x_txt: &x_txt,
                freqs: &freqs,
                input,
                kv: &mut *kv,
                refs,
                n_img,
                t_txt,
            };
            self.double_block(ops, block, i, &mod_img, &mod_txt, &mut ctx)?;
        }

        // Single blocks run the joint sequence in one buffer, txt first.
        // `copy_d2d` requires equal byte counts, so each copy spans exactly
        // the rows it writes, not the whole joint buffer.
        let s_total = t_txt + n_img;
        let joint = ops.alloc_f32("dit joint", s_total * h)?;
        let joint_txt = joint.view().cut(0, (t_txt * h) as u64 * 4)?;
        ops.copy("dit joint txt", joint_txt, x_txt.view())?;
        let joint_img = joint
            .view()
            .cut((t_txt * h) as u64 * 4, (n_img * h) as u64 * 4)?;
        ops.copy("dit joint img", joint_img, x_img.view())?;

        for (j, block) in self.single.iter().enumerate() {
            let mut ctx = StepCtx {
                x_img: &joint,
                x_txt: &joint,
                freqs: &freqs,
                input,
                kv: &mut *kv,
                refs,
                n_img,
                t_txt,
            };
            self.single_block(ops, block, j, &mod_single, &mut ctx)?;
        }

        // Final layer over the image rows of the joint stream: weightless
        // LayerNorm, the shared (shift, scale), then the velocity head.
        let joint_view = joint.view();
        let x_final = joint_view.cut((t_txt * h) as u64 * 4, (n_img * h) as u64 * 4)?;
        let xn = ops.alloc_f32("dit final norm", n_img * h)?;
        ops.layer_norm(&LayerNormCall {
            step: "dit final norm",
            out: xn.view(),
            input: x_final,
            rows: n_img,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&ModulateCall {
            step: "dit final modulate",
            out: xn.view(),
            x: xn.view(),
            scale: mod_final.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: mod_final.view(),
            count: n_img * h,
            period: h,
        })?;
        let out = ops.alloc_f32("dit velocity", n_img * self.final_linear.out_dim)?;
        ops.gemm(&GemmCall {
            step: "dit head",
            out: out.view(),
            input: xn.view(),
            weight: self.final_linear.w.view(),
            weight_scale: self.final_linear.scale,
            bias: self.final_linear.bias.as_ref().map(DeviceBuffer::view),
            m: n_img,
            n: self.final_linear.out_dim,
            k: h,
            kind: self.final_linear.kind,
        })?;
        Ok(out)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "a double block's attention and MLP branches for both streams, straight-line by construction"
    )]
    fn double_block(
        &self,
        ops: &Ops,
        block: &DoubleBlock,
        index: usize,
        mod_img: &DeviceBuffer,
        mod_txt: &DeviceBuffer,
        ctx: &mut StepCtx<'_, '_>,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let heads = self.params.num_heads;
        let head_dim = self.head_dim;
        let freqs_per_axis = self.freqs_per_axis;

        // Attention branch: LayerNorm -> modulate (mod1) -> fused qkv ->
        // QK-norm -> RoPE, image stream and text stream.
        let qkv_img = self.attention_pre(
            ops,
            &PreCall {
                qkv: &block.img_qkv,
                q_norm: &block.img_q_norm,
                k_norm: &block.img_k_norm,
                x: ctx.x_img,
                mod1: mod_img,
                tokens: ctx.n_img,
            },
        )?;
        let qkv_txt = self.attention_pre(
            ops,
            &PreCall {
                qkv: &block.txt_qkv,
                q_norm: &block.txt_q_norm,
                k_norm: &block.txt_k_norm,
                x: ctx.x_txt,
                mod1: mod_txt,
                tokens: ctx.t_txt,
            },
        )?;

        ops.rope_4d(&Rope4dCall {
            step: "dit img rope",
            q: qkv_img.view(),
            k: qkv_img.view(),
            positions: ctx.input.img_positions,
            freqs: ctx.freqs.view(),
            tokens: ctx.n_img,
            heads,
            head_dim,
            freqs_per_axis,
            row_stride: 3 * h,
        })?;
        ops.rope_4d(&Rope4dCall {
            step: "dit txt rope",
            q: qkv_txt.view(),
            k: qkv_txt.view(),
            positions: ctx.input.txt_positions,
            freqs: ctx.freqs.view(),
            tokens: ctx.t_txt,
            heads,
            head_dim,
            freqs_per_axis,
            row_stride: 3 * h,
        })?;

        // Full (unmasked) attention over the joint txt+img rows; with
        // num_ref = 0 the reference's causal_attn_fn is exactly this.
        let out_img = ops.alloc_f32("dit attn img", ctx.n_img * h)?;
        let out_txt = ops.alloc_f32("dit attn txt", ctx.t_txt * h)?;
        let (cache_k, cache_tokens) = Self::cached_kv(
            ops,
            Some(&ctx.kv.double_caches[index]),
            ctx.refs,
            ctx.input.kv_mode,
        )?;
        ops.sdpa(&SdpaCall {
            step: "dit attention",
            out_txt: out_txt.view(),
            out_img: out_img.view(),
            q_txt: qkv_txt.view(),
            k_txt: qkv_txt.view(),
            v_txt: qkv_txt.view(),
            q_img: qkv_img.view(),
            k_img: qkv_img.view(),
            v_img: qkv_img.view(),
            cache_k,
            cache_v: cache_k,
            cache_stride: 3 * h,
            txt_tokens: ctx.t_txt,
            img_tokens: ctx.n_img,
            cache_tokens,
            heads,
            head_dim,
            txt_stride: 3 * h,
            img_stride: 3 * h,
        })?;
        if ctx.input.kv_mode == KvMode::Extract && ctx.refs > 0 {
            let cache = ops.alloc_f32("dit kv cache", ctx.refs * 3 * h)?;
            ops.copy(
                "dit kv extract",
                cache.view(),
                qkv_img.view().cut(0, (ctx.refs * 3 * h) as u64 * 4)?,
            )?;
            ctx.kv.double_caches[index].kv_ref = Some(cache);
            ctx.kv.double_caches[index].row_stride = 3 * h;
            ctx.kv.num_ref_tokens = ctx.refs;
        }

        // Attention residual and MLP, image stream, then text stream. The
        // scratch row buffer is sized for whichever stream is longer.
        let rows = ctx.n_img.max(ctx.t_txt);
        let tmp = ops.alloc_f32("dit attn proj", rows * h)?;
        ops.gemm(&GemmCall {
            step: "dit img proj",
            out: tmp.view(),
            input: out_img.view(),
            weight: block.img_proj.w.view(),
            weight_scale: block.img_proj.scale,
            bias: block.img_proj.bias.as_ref().map(DeviceBuffer::view),
            m: ctx.n_img,
            n: h,
            k: h,
            kind: block.img_proj.kind,
        })?;
        ops.add_gated(&AddGatedCall {
            step: "dit img attn residual",
            out: ctx.x_img.view(),
            a: ctx.x_img.view(),
            gate: mod_img.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: ctx.n_img * h,
            period: h,
        })?;
        self.mlp(
            ops,
            &block.img_mlp,
            ctx.x_img,
            mod_img,
            ctx.n_img,
            "dit img mlp",
        )?;

        ops.gemm(&GemmCall {
            step: "dit txt proj",
            out: tmp.view(),
            input: out_txt.view(),
            weight: block.txt_proj.w.view(),
            weight_scale: block.txt_proj.scale,
            bias: block.txt_proj.bias.as_ref().map(DeviceBuffer::view),
            m: ctx.t_txt,
            n: h,
            k: h,
            kind: block.txt_proj.kind,
        })?;
        ops.add_gated(&AddGatedCall {
            step: "dit txt attn residual",
            out: ctx.x_txt.view(),
            a: ctx.x_txt.view(),
            gate: mod_txt.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: ctx.t_txt * h,
            period: h,
        })?;
        self.mlp(
            ops,
            &block.txt_mlp,
            ctx.x_txt,
            mod_txt,
            ctx.t_txt,
            "dit txt mlp",
        )?;
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "a single block's fused joint attention and gated MLP, straight-line by construction"
    )]
    fn single_block(
        &self,
        ops: &Ops,
        block: &SingleBlock,
        index: usize,
        shared: &DeviceBuffer,
        ctx: &mut StepCtx<'_, '_>,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let heads = self.params.num_heads;
        let head_dim = self.head_dim;
        let freqs_per_axis = self.freqs_per_axis;
        let inner = self.inner;
        // linear1 rows: [qkv (3h) | mlp gate (inner) | mlp value (inner)].
        let fused = 3 * h + 2 * inner;
        let joint = ctx.x_img;
        let s_total = ctx.t_txt + ctx.n_img;

        // LayerNorm -> modulate (shared shift/scale) -> fused linear1.
        let xn = ops.alloc_f32("dit single norm", s_total * h)?;
        ops.layer_norm(&LayerNormCall {
            step: "dit single norm",
            out: xn.view(),
            input: joint.view(),
            rows: s_total,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&ModulateCall {
            step: "dit single modulate",
            out: xn.view(),
            x: xn.view(),
            scale: shared.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: shared.view(),
            count: s_total * h,
            period: h,
        })?;
        let qkv = ops.alloc_f32("dit single linear1", s_total * fused)?;
        ops.gemm(&GemmCall {
            step: "dit single linear1",
            out: qkv.view(),
            input: xn.view(),
            weight: block.linear1.w.view(),
            weight_scale: block.linear1.scale,
            bias: block.linear1.bias.as_ref().map(DeviceBuffer::view),
            m: s_total,
            n: fused,
            k: h,
            kind: block.linear1.kind,
        })?;
        ops.qk_norm(&QkNormCall {
            step: "dit single qk norm",
            qkv: qkv.view(),
            q_scale: block.q_norm.view(),
            k_scale: block.k_norm.view(),
            tokens: s_total,
            heads,
            head_dim,
            row_stride: fused,
            eps: NORM_EPS,
        })?;
        ops.rope_4d(&Rope4dCall {
            step: "dit single rope",
            q: qkv.view(),
            k: qkv.view(),
            positions: ctx.input.joint_positions,
            freqs: ctx.freqs.view(),
            tokens: s_total,
            heads,
            head_dim,
            freqs_per_axis,
            row_stride: fused,
        })?;

        // Full attention over the joint rows (txt first), plus the cached
        // reference rows in `Cached` mode.
        let attn = ops.alloc_f32("dit single attn", s_total * h)?;
        let (cache_k, cache_tokens) = Self::cached_kv(
            ops,
            Some(&ctx.kv.single_caches[index]),
            ctx.refs,
            ctx.input.kv_mode,
        )?;
        ops.sdpa(&SdpaCall {
            step: "dit single attention",
            out_txt: attn.view(),
            out_img: attn.view(),
            q_txt: qkv.view(),
            k_txt: qkv.view(),
            v_txt: qkv.view(),
            q_img: qkv.view(),
            k_img: qkv.view(),
            v_img: qkv.view(),
            cache_k,
            cache_v: cache_k,
            cache_stride: fused,
            txt_tokens: s_total,
            img_tokens: 0,
            cache_tokens,
            heads,
            head_dim,
            txt_stride: fused,
            img_stride: fused,
        })?;
        if ctx.input.kv_mode == KvMode::Extract && ctx.refs > 0 {
            let cache = ops.alloc_f32("dit single kv cache", ctx.refs * fused)?;
            ops.copy(
                "dit single kv extract",
                cache.view(),
                qkv.view().cut(
                    (ctx.t_txt * fused) as u64 * 4,
                    (ctx.refs * fused) as u64 * 4,
                )?,
            )?;
            ctx.kv.single_caches[index].kv_ref = Some(cache);
            ctx.kv.single_caches[index].row_stride = fused;
            ctx.kv.num_ref_tokens = ctx.refs;
        }

        // MLP half of linear1: silu(gate) * value, the gate half starting
        // mid-row at column 3h of the fused buffer.
        let act = ops.alloc_f32("dit single act", s_total * inner)?;
        let gate_rows = qkv
            .view()
            .cut(3 * h as u64 * 4, (s_total * fused - 3 * h) as u64 * 4)?;
        ops.swiglu(
            "dit single swiglu",
            act.view(),
            gate_rows,
            s_total,
            inner,
            fused,
        )?;

        // linear2 reads cat(attn, swiglu) and the block carries a single
        // gated residual (the reference's `_out`).
        let lin2_in = ops.alloc_f32("dit single lin2 in", s_total * (h + inner))?;
        ops.concat_rows(&ConcatCall {
            step: "dit single concat",
            out: lin2_in.view(),
            a: attn.view(),
            b: act.view(),
            rows: s_total,
            cols_a: h,
            cols_b: inner,
        })?;
        let tmp = ops.alloc_f32("dit single proj", s_total * h)?;
        ops.gemm(&GemmCall {
            step: "dit single linear2",
            out: tmp.view(),
            input: lin2_in.view(),
            weight: block.linear2.w.view(),
            weight_scale: block.linear2.scale,
            bias: block.linear2.bias.as_ref().map(DeviceBuffer::view),
            m: s_total,
            n: h,
            k: h + inner,
            kind: block.linear2.kind,
        })?;
        ops.add_gated(&AddGatedCall {
            step: "dit single residual",
            out: joint.view(),
            a: joint.view(),
            gate: shared.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: s_total * h,
            period: h,
        })?;
        Ok(())
    }

    /// The cached reference rows to attend over in `Cached` mode, or the
    /// dummy binding and zero count otherwise.
    fn cached_kv<'c>(
        ops: &'c Ops,
        cache: Option<&'c super::kv_cache::LayerKvCache>,
        refs: usize,
        mode: KvMode,
    ) -> Result<(DeviceView<'c>, usize), ImageGenError> {
        match mode {
            KvMode::Cached => {
                let buf = cache.and_then(|c| c.kv_ref.as_ref()).ok_or_else(|| {
                    ImageGenError::Backend("KV cached step without a cached reference".to_string())
                })?;
                Ok((buf.view(), refs))
            }
            KvMode::None | KvMode::Extract => Ok((ops.dummy_view(), 0)),
        }
    }

    /// Shared `AdaLN` vector: silu(vec) @ modulation.lin, (width,) f32.
    fn mod_vec(
        &self,
        ops: &Ops,
        module: &Lin,
        vec_silu: &DeviceBuffer,
        width: usize,
        step: &'static str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let vec = ops.alloc_f32(step, width)?;
        ops.gemm(&GemmCall {
            step,
            out: vec.view(),
            input: vec_silu.view(),
            weight: module.w.view(),
            weight_scale: module.scale,
            bias: module.bias.as_ref().map(DeviceBuffer::view),
            m: 1,
            n: width,
            k: self.params.hidden_size,
            kind: module.kind,
        })?;
        Ok(vec)
    }

    /// Weightless `LayerNorm` + `AdaLN` modulation (mod1) + fused qkv +
    /// `QK-norm`, in that reference order. The `QK-norm` runs before `RoPE`
    /// because the reference norms the heads and then rotates them.
    fn attention_pre(&self, ops: &Ops, call: &PreCall<'_>) -> Result<DeviceBuffer, ImageGenError> {
        let PreCall {
            qkv,
            q_norm,
            k_norm,
            x,
            mod1,
            tokens,
        } = *call;
        let h = self.params.hidden_size;
        let xn = ops.alloc_f32("dit norm", tokens * h)?;
        ops.layer_norm(&LayerNormCall {
            step: "dit norm",
            out: xn.view(),
            input: x.view(),
            rows: tokens,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&ModulateCall {
            step: "dit modulate",
            out: xn.view(),
            x: xn.view(),
            scale: mod1.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: mod1.view(),
            count: tokens * h,
            period: h,
        })?;
        let out = ops.alloc_f32("dit qkv", tokens * 3 * h)?;
        ops.gemm(&GemmCall {
            step: "dit qkv",
            out: out.view(),
            input: xn.view(),
            weight: qkv.w.view(),
            weight_scale: qkv.scale,
            bias: qkv.bias.as_ref().map(DeviceBuffer::view),
            m: tokens,
            n: 3 * h,
            k: h,
            kind: qkv.kind,
        })?;
        ops.qk_norm(&QkNormCall {
            step: "dit qk norm",
            qkv: out.view(),
            q_scale: q_norm.view(),
            k_scale: k_norm.view(),
            tokens,
            heads: self.params.num_heads,
            head_dim: self.head_dim,
            row_stride: 3 * h,
            eps: NORM_EPS,
        })?;
        Ok(out)
    }

    /// `AdaLN` MLP branch: `LayerNorm` -> modulate (mod2) -> fc1
    /// (2 * inner out) -> `SwiGLU` (gate is the first half) -> fc2, gated
    /// residual add.
    fn mlp(
        &self,
        ops: &Ops,
        mlp: &Mlp,
        x: &DeviceBuffer,
        mod2: &DeviceBuffer,
        tokens: usize,
        step: &'static str,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let inner = self.inner;
        let xn = ops.alloc_f32(step, tokens * h)?;
        ops.layer_norm(&LayerNormCall {
            step,
            out: xn.view(),
            input: x.view(),
            rows: tokens,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&ModulateCall {
            step,
            out: xn.view(),
            x: xn.view(),
            scale: mod2.view().cut(4 * h as u64 * 4, h as u64 * 4)?,
            shift: mod2.view().cut(3 * h as u64 * 4, h as u64 * 4)?,
            count: tokens * h,
            period: h,
        })?;
        let fused = ops.alloc_f32(step, tokens * 2 * inner)?;
        ops.gemm(&GemmCall {
            step,
            out: fused.view(),
            input: xn.view(),
            weight: mlp.fc1.w.view(),
            weight_scale: mlp.fc1.scale,
            bias: mlp.fc1.bias.as_ref().map(DeviceBuffer::view),
            m: tokens,
            n: 2 * inner,
            k: h,
            kind: mlp.fc1.kind,
        })?;
        let act = ops.alloc_f32(step, tokens * inner)?;
        ops.swiglu(step, act.view(), fused.view(), tokens, inner, 2 * inner)?;
        let tmp = ops.alloc_f32(step, tokens * h)?;
        ops.gemm(&GemmCall {
            step,
            out: tmp.view(),
            input: act.view(),
            weight: mlp.fc2.w.view(),
            weight_scale: mlp.fc2.scale,
            bias: mlp.fc2.bias.as_ref().map(DeviceBuffer::view),
            m: tokens,
            n: h,
            k: inner,
            kind: mlp.fc2.kind,
        })?;
        ops.add_gated(&AddGatedCall {
            step,
            out: x.view(),
            a: x.view(),
            gate: mod2.view().cut(5 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: tokens * h,
            period: h,
        })
    }
}
