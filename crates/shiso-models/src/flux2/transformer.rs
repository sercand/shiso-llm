//! The FLUX.2 unified diffusion transformer: double-stream and single-stream
//! blocks over a joint text+image sequence.
//!
//! The graph follows the reference block layout under the checkpoint's own
//! tensor names: double blocks carry `img_attn.qkv`/`img_attn.proj`/
//! `img_mlp.0`/`img_mlp.2` and their `txt_` twins, single blocks carry the
//! fused `linear1` (q, k, v and the MLP gate in one row) and `linear2`.
//! `AdaLN` modulation comes from a per-block projection of the timestep
//! embedding, applied with the weightless `RMSNorm` the reference uses.
//!
//! Weights run in their stored native format: BF16 rows are read by the BF16
//! GEMM, `F8_E4M3` rows by the FP8 GEMM with the tensor's `weight_scale`
//! folded into the accumulator epilogue, and F32 rows by the F32 GEMM. No
//! weight is ever converted, duplicated, or dequantized into an F32 buffer.
//!
//! The reference-image KV path is the plan's: step 0 (`Extract`) runs the
//! reference tokens through every block and keeps their post-RoPE fused qkv
//! rows; later steps (`Cached`) project and modulate only the denoise tokens
//! and read the cached rows as extra attention keys and values.

use shiso_core::{DeviceBuffer, DeviceView};
use shiso_vision::generator::ImageGenError;

use super::config::Flux2Params;
use super::kv_cache::Flux2KvCache;
use super::ops::{GemmCall, NormCall, Ops, Rope4dCall, SdpaCall, SliceExt, WeightKind};
use super::weights::{Lin, StStore};

/// `RMSNorm` epsilon the reference blocks use.
const NORM_EPS: f32 = 1e-6;

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

#[derive(Debug)]
struct DoubleBlock {
    img_mod: Lin,
    img_qkv: Lin,
    img_proj: Lin,
    img_mlp: Mlp,
    txt_mod: Lin,
    txt_qkv: Lin,
    txt_proj: Lin,
    txt_mlp: Mlp,
}

#[derive(Debug)]
struct SingleBlock {
    module: Lin,
    linear1: Lin,
    linear2: Lin,
}

/// One denoising forward pass; everything the blocks read, in one place.
pub(crate) struct DiTInput<'a> {
    /// (`img_tokens`, `in_channels`) f32 latent sequence.
    pub latents: DeviceView<'a>,
    /// (`txt_tokens`, `context_in_dim`) f32 text conditioning.
    pub txt: DeviceView<'a>,
    /// Denoising timestep in [0, 1000].
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

/// Everything a block call needs beyond its own weights: the two residual
/// streams, the `RoPE` tables, and the KV-cache state. A named bundle because
/// the block geometry travels together and no block may take twelve arguments.
struct StepCtx<'a, 'b> {
    x_img: &'b DeviceBuffer,
    x_txt: &'b DeviceBuffer,
    freqs: &'b DeviceBuffer,
    input: &'b DiTInput<'a>,
    kv: &'b mut Flux2KvCache,
    refs: usize,
    n_img: usize,
    t_txt: usize,
    heads: usize,
    head_dim: usize,
    freqs_per_axis: usize,
}

/// The loaded diffusion transformer.
#[derive(Debug)]
pub(crate) struct DiT {
    params: Flux2Params,
    img_in: Lin,
    txt_in: Lin,
    time_in_in: Lin,
    time_in_out: Lin,
    time_freqs: Vec<f32>,
    double: Vec<DoubleBlock>,
    single: Vec<SingleBlock>,
    final_linear: Lin,
    proj_out: Lin,
    ones: DeviceBuffer,
    inner: usize,
}

impl DiT {
    /// Opens the transformer component and uploads every weight in its
    /// stored format.
    #[expect(
        clippy::cast_precision_loss,
        reason = "frequency-table indices and hidden size are small config-bounded dimensions"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "mlp_ratio * hidden is a layer width bounded by the model config; it fits usize"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "mlp_ratio and hidden size are positive, so the product is non-negative"
    )]
    pub(crate) fn load(
        store: &StStore,
        ops: &Ops,
        params: &Flux2Params,
    ) -> Result<Self, ImageGenError> {
        let img_in = Lin::load(store, ops, "img_in")?;
        let txt_in = Lin::load(store, ops, "txt_in")?;
        let time_in_in = Lin::load(store, ops, "time_in.in_layer")?;
        let time_in_out = Lin::load(store, ops, "time_in.out_layer")?;
        let time_freqs = if store.contains("time_proj.freqs") {
            store.f32_vec("time_proj.freqs")?
        } else {
            let half = time_in_in.in_dim / 2;
            (0..half)
                .map(|i| (-((10_000f32).ln()) * (i as f32) / (half.max(1)) as f32).exp())
                .collect()
        };

        let mut double = Vec::with_capacity(params.depth);
        for i in 0..params.depth {
            let b = format!("double_blocks.{i}.");
            double.push(DoubleBlock {
                img_mod: Lin::load(store, ops, &format!("{b}img_norm1.linear"))?,
                img_qkv: Lin::load(store, ops, &format!("{b}img_attn.qkv"))?,
                img_proj: Lin::load(store, ops, &format!("{b}img_attn.proj"))?,
                img_mlp: Mlp {
                    fc1: Lin::load(store, ops, &format!("{b}img_mlp.0"))?,
                    fc2: Lin::load(store, ops, &format!("{b}img_mlp.2"))?,
                },
                txt_mod: Lin::load(store, ops, &format!("{b}txt_norm1.linear"))?,
                txt_qkv: Lin::load(store, ops, &format!("{b}txt_attn.qkv"))?,
                txt_proj: Lin::load(store, ops, &format!("{b}txt_attn.proj"))?,
                txt_mlp: Mlp {
                    fc1: Lin::load(store, ops, &format!("{b}txt_mlp.0"))?,
                    fc2: Lin::load(store, ops, &format!("{b}txt_mlp.2"))?,
                },
            });
        }
        let mut single = Vec::with_capacity(params.depth_single_blocks);
        for i in 0..params.depth_single_blocks {
            let b = format!("single_blocks.{i}.");
            single.push(SingleBlock {
                module: Lin::load(store, ops, &format!("{b}norm.linear"))?,
                linear1: Lin::load(store, ops, &format!("{b}linear1"))?,
                linear2: Lin::load(store, ops, &format!("{b}linear2"))?,
            });
        }

        let final_linear = Lin::load(store, ops, "final_layer.linear")?;
        let proj_out = Lin::load(store, ops, "final_layer.proj_out")?;

        let hidden = params.hidden_size;
        let ones = ops.upload_f32("dit ones", &vec![1.0; hidden])?;
        let inner = (params.mlp_ratio * hidden as f32) as usize;

        Ok(Self {
            params: params.clone(),
            img_in,
            txt_in,
            time_in_in,
            time_in_out,
            time_freqs,
            double,
            single,
            final_linear,
            proj_out,
            ones,
            inner,
        })
    }

    /// Patchified output width of the final projection.
    pub(crate) fn out_width(&self) -> usize {
        self.proj_out.out_dim
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
        let heads = p.num_heads;
        let head_dim = h / heads.max(1);
        let freqs_per_axis = p.axes_dim.first().copied().unwrap_or(0) / 2;
        if !p.axes_dim.iter().all(|&a| a / 2 == freqs_per_axis) {
            return Err(ImageGenError::Checkpoint(
                "flux2 RoPE requires equal axis widths".to_string(),
            ));
        }
        let freqs = ops.upload_f32(
            "dit rope freqs",
            &super::rope::rope_4d_freqs(&p.axes_dim, p.theta),
        )?;

        let t_txt = input.txt_tokens;
        let n_img = input.img_tokens;
        let refs = if input.kv_mode == KvMode::None {
            0
        } else {
            input.ref_tokens
        };

        // Timestep embedding: host sinusoid (cos then sin of t * freq), then
        // the two-layer MLP. The two GEMMs write distinct buffers; an in-place
        // m=1 GEMM would race its accumulator against the input it reads.
        let half = self.time_in_in.in_dim / 2;
        let t = input.timestep;
        let mut emb = Vec::with_capacity(half * 2);
        for i in 0..half {
            let f = self.time_freq_freq(i, half);
            emb.push((t * f).cos());
        }
        for i in 0..half {
            let f = self.time_freq_freq(i, half);
            emb.push((t * f).sin());
        }
        let emb_buf = ops.upload_f32("dit timestep", &emb)?;
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
        let temb2 = ops.alloc_f32("dit temb2", h)?;
        ops.gemm(&GemmCall {
            step: "dit time out",
            out: temb2.view(),
            input: temb1.view(),
            weight: self.time_in_out.w.view(),
            weight_scale: self.time_in_out.scale,
            bias: self.time_in_out.bias.as_ref().map(DeviceBuffer::view),
            m: 1,
            n: h,
            k: h,
            kind: self.time_in_out.kind,
        })?;
        let temb_silu = ops.alloc_f32("dit temb silu", h)?;
        ops.silu("dit temb silu", temb_silu.view(), temb2.view(), h)?;

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
                heads,
                head_dim,
                freqs_per_axis,
            };
            self.double_block(ops, block, i, &temb_silu, &mut ctx)?;
        }

        // Single blocks run the joint sequence in one buffer.
        let s_total = t_txt + n_img;
        let joint = ops.alloc_f32("dit joint", s_total * h)?;
        ops.copy("dit joint txt", joint.view(), x_txt.view())?;
        ops.copy(
            "dit joint img",
            joint
                .view()
                .cut((t_txt * h) as u64 * 4, (n_img * h) as u64 * 4)?,
            x_img.view(),
        )?;
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
                heads,
                head_dim,
                freqs_per_axis,
            };
            self.single_block(ops, block, j, &temb_silu, &mut ctx)?;
        }

        // Final layer over the image rows of the joint stream.
        let joint_view = joint.view();
        let x_final = joint_view.cut((t_txt * h) as u64 * 4, (n_img * h) as u64 * 4)?;
        let vec = ops.alloc_f32("dit final vec", 2 * h)?;
        ops.gemm(&GemmCall {
            step: "dit final mod",
            out: vec.view(),
            input: temb_silu.view(),
            weight: self.final_linear.w.view(),
            weight_scale: self.final_linear.scale,
            bias: self.final_linear.bias.as_ref().map(DeviceBuffer::view),
            m: 1,
            n: 2 * h,
            k: h,
            kind: self.final_linear.kind,
        })?;
        let xn = ops.alloc_f32("dit final norm", n_img * h)?;
        ops.rmsnorm(&NormCall {
            step: "dit final norm",
            out: xn.view(),
            input: x_final,
            weight: self.ones.view(),
            rows: n_img,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&super::ops::ModulateCall {
            step: "dit final modulate",
            out: xn.view(),
            x: xn.view(),
            scale: vec.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: vec.view(),
            count: n_img * h,
            period: h,
        })?;
        let out = ops.alloc_f32("dit velocity", n_img * self.proj_out.out_dim)?;
        ops.gemm(&GemmCall {
            step: "dit proj_out",
            out: out.view(),
            input: xn.view(),
            weight: self.proj_out.w.view(),
            weight_scale: self.proj_out.scale,
            bias: self.proj_out.bias.as_ref().map(DeviceBuffer::view),
            m: n_img,
            n: self.proj_out.out_dim,
            k: h,
            kind: self.proj_out.kind,
        })?;
        Ok(out)
    }

    /// The i-th stored sinusoid frequency, or the reference formula when the
    /// checkpoint carries fewer frequencies than the embedding width needs.
    #[expect(
        clippy::cast_precision_loss,
        reason = "frequency indices and embedding half-width are small config-bounded dimensions"
    )]
    fn time_freq_freq(&self, i: usize, half: usize) -> f32 {
        self.time_freqs
            .get(i)
            .copied()
            .unwrap_or_else(|| (-((10_000f32).ln()) * (i as f32) / (half.max(1)) as f32).exp())
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
        temb_silu: &DeviceBuffer,
        ctx: &mut StepCtx<'_, '_>,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let img_vec = self.mod_vec(ops, &block.img_mod, temb_silu, 6 * h, "dit img mod")?;
        let txt_vec = self.mod_vec(ops, &block.txt_mod, temb_silu, 6 * h, "dit txt mod")?;

        let qkv_img = self.attention_pre(ops, &block.img_qkv, ctx.x_img, &img_vec, ctx.n_img)?;
        let qkv_txt = self.attention_pre(ops, &block.txt_qkv, ctx.x_txt, &txt_vec, ctx.t_txt)?;

        ops.rope_4d(&Rope4dCall {
            step: "dit img rope",
            q: qkv_img.view(),
            k: qkv_img.view(),
            positions: ctx.input.img_positions,
            freqs: ctx.freqs.view(),
            tokens: ctx.n_img,
            heads: ctx.heads,
            head_dim: ctx.head_dim,
            freqs_per_axis: ctx.freqs_per_axis,
            row_stride: 3 * h,
        })?;
        ops.rope_4d(&Rope4dCall {
            step: "dit txt rope",
            q: qkv_txt.view(),
            k: qkv_txt.view(),
            positions: ctx.input.txt_positions,
            freqs: ctx.freqs.view(),
            tokens: ctx.t_txt,
            heads: ctx.heads,
            head_dim: ctx.head_dim,
            freqs_per_axis: ctx.freqs_per_axis,
            row_stride: 3 * h,
        })?;

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
            heads: ctx.heads,
            head_dim: ctx.head_dim,
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
        ops.add_gated(&super::ops::AddGatedCall {
            step: "dit img attn residual",
            out: ctx.x_img.view(),
            a: ctx.x_img.view(),
            gate: img_vec.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: ctx.n_img * h,
            period: h,
        })?;
        self.mlp(
            ops,
            &block.img_mlp,
            ctx.x_img,
            &img_vec,
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
        ops.add_gated(&super::ops::AddGatedCall {
            step: "dit txt attn residual",
            out: ctx.x_txt.view(),
            a: ctx.x_txt.view(),
            gate: txt_vec.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: ctx.t_txt * h,
            period: h,
        })?;
        self.mlp(
            ops,
            &block.txt_mlp,
            ctx.x_txt,
            &txt_vec,
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
        temb_silu: &DeviceBuffer,
        ctx: &mut StepCtx<'_, '_>,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let fused = 7 * h;
        let joint = ctx.x_img;
        let s_total = ctx.t_txt + ctx.n_img;
        let vec = self.mod_vec(ops, &block.module, temb_silu, 6 * h, "dit single mod")?;

        let xn = ops.alloc_f32("dit single norm", s_total * h)?;
        ops.rmsnorm(&NormCall {
            step: "dit single norm",
            out: xn.view(),
            input: joint.view(),
            weight: self.ones.view(),
            rows: s_total,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&super::ops::ModulateCall {
            step: "dit single modulate",
            out: xn.view(),
            x: xn.view(),
            scale: vec.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: vec.view(),
            count: s_total * h,
            period: h,
        })?;
        let qkv = ops.alloc_f32("dit single qkv", s_total * fused)?;
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
        ops.rope_4d(&Rope4dCall {
            step: "dit single rope",
            q: qkv.view(),
            k: qkv.view(),
            positions: ctx.input.joint_positions,
            freqs: ctx.freqs.view(),
            tokens: s_total,
            heads: ctx.heads,
            head_dim: ctx.head_dim,
            freqs_per_axis: ctx.freqs_per_axis,
            row_stride: fused,
        })?;

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
            heads: ctx.heads,
            head_dim: ctx.head_dim,
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

        let tmp = ops.alloc_f32("dit single proj", s_total * h)?;
        ops.gemm(&GemmCall {
            step: "dit single linear2",
            out: tmp.view(),
            input: attn.view(),
            weight: block.linear2.w.view(),
            weight_scale: block.linear2.scale,
            bias: block.linear2.bias.as_ref().map(DeviceBuffer::view),
            m: s_total,
            n: h,
            k: h,
            kind: block.linear2.kind,
        })?;
        ops.add_gated(&super::ops::AddGatedCall {
            step: "dit single attn residual",
            out: joint.view(),
            a: joint.view(),
            gate: vec.view().cut(2 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: s_total * h,
            period: h,
        })?;

        // MLP: the gate/value rows of `linear1`, sliced out of the fused
        // weight matrix by row range in the stored element width.
        ops.rmsnorm(&NormCall {
            step: "dit single mlp norm",
            out: xn.view(),
            input: joint.view(),
            weight: self.ones.view(),
            rows: s_total,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&super::ops::ModulateCall {
            step: "dit single mlp modulate",
            out: xn.view(),
            x: xn.view(),
            scale: vec.view().cut(4 * h as u64 * 4, h as u64 * 4)?,
            shift: vec.view().cut(3 * h as u64 * 4, h as u64 * 4)?,
            count: s_total * h,
            period: h,
        })?;
        let elem_bytes = match block.linear1.kind {
            WeightKind::Fp8 => 1u64,
            WeightKind::Bf16 => 2,
            WeightKind::F32 => 4,
        };
        let mlp_rows = 4 * h;
        let linear1_view = block.linear1.w.view();
        let weight = linear1_view.cut(
            3 * h as u64 * h as u64 * elem_bytes,
            mlp_rows as u64 * h as u64 * elem_bytes,
        )?;
        let bias = match &block.linear1.bias {
            Some(b) => {
                let bias_view = b.view();
                Some(bias_view.cut(3 * h as u64 * 4, mlp_rows as u64 * 4)?)
            }
            None => None,
        };
        let mlp1 = ops.alloc_f32("dit single mlp1", s_total * mlp_rows)?;
        ops.gemm(&GemmCall {
            step: "dit single mlp1",
            out: mlp1.view(),
            input: xn.view(),
            weight,
            weight_scale: block.linear1.scale,
            bias,
            m: s_total,
            n: mlp_rows,
            k: h,
            kind: block.linear1.kind,
        })?;
        let act = ops.alloc_f32("dit single act", s_total * 2 * h)?;
        ops.swiglu(
            "dit single swiglu",
            act.view(),
            mlp1.view(),
            s_total,
            2 * h,
            mlp_rows,
        )?;
        ops.gemm(&GemmCall {
            step: "dit single mlp2",
            out: tmp.view(),
            input: act.view(),
            weight: block.linear2.w.view(),
            weight_scale: block.linear2.scale,
            bias: block.linear2.bias.as_ref().map(DeviceBuffer::view),
            m: s_total,
            n: h,
            k: 2 * h,
            kind: block.linear2.kind,
        })?;
        ops.add_gated(&super::ops::AddGatedCall {
            step: "dit single mlp residual",
            out: joint.view(),
            a: joint.view(),
            gate: vec.view().cut(5 * h as u64 * 4, h as u64 * 4)?,
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

    /// Per-block `AdaLN` vector: silu(temb) @ mod.linear + bias, (width,) f32.
    fn mod_vec(
        &self,
        ops: &Ops,
        module: &Lin,
        temb_silu: &DeviceBuffer,
        width: usize,
        step: &'static str,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let vec = ops.alloc_f32(step, width)?;
        ops.gemm(&GemmCall {
            step,
            out: vec.view(),
            input: temb_silu.view(),
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

    /// Weightless `RMSNorm` + `AdaLN` modulation + fused qkv projection.
    fn attention_pre(
        &self,
        ops: &Ops,
        qkv: &Lin,
        x: &DeviceBuffer,
        vec: &DeviceBuffer,
        tokens: usize,
    ) -> Result<DeviceBuffer, ImageGenError> {
        let h = self.params.hidden_size;
        let xn = ops.alloc_f32("dit norm", tokens * h)?;
        ops.rmsnorm(&NormCall {
            step: "dit norm",
            out: xn.view(),
            input: x.view(),
            weight: self.ones.view(),
            rows: tokens,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&super::ops::ModulateCall {
            step: "dit modulate",
            out: xn.view(),
            x: xn.view(),
            scale: vec.view().cut(h as u64 * 4, h as u64 * 4)?,
            shift: vec.view(),
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
        Ok(out)
    }

    /// `AdaLN` MLP: fc1 (2 * inner out) -> `SwiGLU` -> fc2, gated residual add.
    fn mlp(
        &self,
        ops: &Ops,
        mlp: &Mlp,
        x: &DeviceBuffer,
        vec: &DeviceBuffer,
        tokens: usize,
        step: &'static str,
    ) -> Result<(), ImageGenError> {
        let h = self.params.hidden_size;
        let inner = self.inner;
        let xn = ops.alloc_f32(step, tokens * h)?;
        ops.rmsnorm(&NormCall {
            step,
            out: xn.view(),
            input: x.view(),
            weight: self.ones.view(),
            rows: tokens,
            dim: h,
            eps: NORM_EPS,
        })?;
        ops.modulate(&super::ops::ModulateCall {
            step,
            out: xn.view(),
            x: xn.view(),
            scale: vec.view().cut(4 * h as u64 * 4, h as u64 * 4)?,
            shift: vec.view().cut(3 * h as u64 * 4, h as u64 * 4)?,
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
        ops.add_gated(&super::ops::AddGatedCall {
            step,
            out: x.view(),
            a: x.view(),
            gate: vec.view().cut(5 * h as u64 * 4, h as u64 * 4)?,
            b: tmp.view(),
            count: tokens * h,
            period: h,
        })
    }
}
