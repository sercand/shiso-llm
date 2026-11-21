//! Inkling-2 MTP / DSpark parallel block drafter (`mtp.safetensors`).
//!
//! Absorbs 4 target hidden-state taps (`[5, 23, 41, 48]`), projects per-layer
//! context KV via softmax-weighted `fuse_weight` + `dflare_norm`, runs a
//! 3-layer non-causal-in-block transformer over `[embed(anchor), mask_embed * 7]`,
//! and refines each drafted position left-to-right with the `rnn_hidden` Markov head.

#![expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::missing_panics_doc,
    clippy::needless_range_loop,
    reason = "MTP drafter executes parallel block attention and sequential RNN refinement"
)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use shiso_core::{
    Backend, CoreError, CoreResult, DeviceBuffer, DeviceView, Extent3, KernelArg, KernelRef,
    LaunchArgs, SessionId, TokenId, alloc,
};
use shiso_modelfile::st_header::StDType;

use super::config::{MtpConfig, TextConfig};
use super::ops::{self, LinearArgs, MODULE, NormArgs};
use super::weights::SafetensorsDir;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MtpMixArgs {
    rows: u32,
    stride_rows: u32,
    hidden: u32,
    n_taps: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MtpQkArgs {
    rows: u32,
    n_head: u32,
    head_dim: u32,
    pos0: u32,
    rope_theta: f32,
    eps: f32,
    kv_cap: u32,
    is_swa: u32,
    write_cache: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MtpAttnArgs {
    block_size: u32,
    n_head: u32,
    n_head_kv: u32,
    head_dim: u32,
    ctx_len: u32,
    kv_cap: u32,
    window_size: u32,
    is_swa: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MtpStoreArgs {
    rows: u32,
    width: u32,
    pos0: u32,
    kv_cap: u32,
    is_swa: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MtpRnnArgs {
    rows: u32,
    rank: u32,
}

#[derive(Debug)]
struct MtpLayerWeights {
    dflare_norm: DeviceBuffer,
    input_ln: DeviceBuffer,
    post_attn_ln: DeviceBuffer,
    q_proj: DeviceBuffer,
    k_proj: DeviceBuffer,
    v_proj: DeviceBuffer,
    o_proj: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    k_proj_ctx: DeviceBuffer,
    v_proj_ctx: DeviceBuffer,
    k_norm_ctx: DeviceBuffer,
    gate_proj: DeviceBuffer,
    up_proj: DeviceBuffer,
    down_proj: DeviceBuffer,
    is_local: bool,
}

#[derive(Debug)]
struct MtpLayerKv {
    k_ctx: DeviceBuffer,
    v_ctx: DeviceBuffer,
    kv_cap: u32,
}

/// The rows an absorb mixes and projects through, kept with the session so
/// nothing a queued launch reads is dropped under it.
#[derive(Debug)]
struct AbsorbScratch {
    cap_rows: u32,
    mixed: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
}

#[derive(Debug)]
struct MtpSessionState {
    ctx_len: u32,
    kv: Vec<MtpLayerKv>,
    /// Last committed `f_rnn` vector (`[hidden]`, F32), on the device: a draft
    /// reads it back beside its hidden rows, so absorbing never waits on the
    /// device.
    f_rnn: DeviceBuffer,
    scratch: Option<AbsorbScratch>,
}

/// Loaded Inkling-2 MTP / DSpark drafter on a [`Backend`].
#[derive(Debug)]
pub struct Inkling2MtpDrafter {
    backend: Arc<dyn Backend>,
    config: MtpConfig,
    text: TextConfig,
    /// Softmax-normalized `fuse_weight` rows (`[num_layers + 1][n_taps]`) uploaded per row.
    fuse_alpha: Vec<DeviceBuffer>,
    /// Mask embedding (`[hidden]`, F32 on device).
    mask_embed_f32: DeviceBuffer,
    layers: Vec<MtpLayerWeights>,
    final_norm: DeviceBuffer,
    markov_dflare_norm: DeviceBuffer,
    /// Host copies of the tiny sequential RNN Markov head weights for microsecond host steps:
    /// `markov_w1`: `[vocab, rank]` BF16, `rnn_proj_w`: `[3*rank, 2*rank + 2*H]` BF16,
    /// `rnn_proj_b`: `[3*rank]` F32, `hidden_out_w`: `[H, rank]` F32, `hidden_out_b`: `[H]` F32.
    markov_w1_bf16: Vec<u16>,
    rnn_proj_w_bf16: Vec<u16>,
    rnn_proj_b: Vec<f32>,
    hidden_out_w: Vec<f32>,
    hidden_out_b: Vec<f32>,
    sessions: HashMap<SessionId, MtpSessionState>,
}

fn bf16_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

fn bf16_slice_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Inkling2MtpDrafter {
    /// Loads `mtp.safetensors` from `path` onto `backend`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if the file or any tensor is missing or invalid.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: MtpConfig,
        text: TextConfig,
    ) -> CoreResult<Self> {
        let dir = SafetensorsDir::open_file(path)?;
        let n_taps = config.target_layer_ids.len();
        let n_stages = config.num_hidden_layers as usize;

        // Compute softmax(fuse_weight, dim=-1) per row and upload each row as F32.
        let (fw_bytes, _) = dir.tensor_bytes("fuse_weight", StDType::Bf16)?;
        let fw_f32 = bf16_slice_to_f32(fw_bytes);
        let mut fuse_alpha = Vec::with_capacity(n_stages + 1);
        for row in 0..=n_stages {
            let sl = &fw_f32[row * n_taps..(row + 1) * n_taps];
            let mx = sl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = sl.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let alpha: Vec<f32> = exps.iter().map(|v| v / sum).collect();
            let buf = alloc(backend, (alpha.len() * 4) as u64)?;
            backend.copy_h2d(buf.view(), cast_slice(&alpha))?;
            fuse_alpha.push(buf);
        }

        let (mask_bytes, _) = dir.tensor_bytes("mask_embed", StDType::Bf16)?;
        let mask_f32 = bf16_slice_to_f32(mask_bytes);
        let mask_embed_f32 = alloc(backend, (mask_f32.len() * 4) as u64)?;
        backend.copy_h2d(mask_embed_f32.view(), cast_slice(&mask_f32))?;

        let final_norm = dir.upload(backend, "norm.weight", StDType::Bf16)?;
        let markov_dflare_norm =
            dir.upload(backend, "markov_head.dflare_norm.weight", StDType::Bf16)?;

        let (mw1_bytes, _) = dir.tensor_bytes("markov_head.markov_w1.weight", StDType::Bf16)?;
        let markov_w1_bf16 = bf16_slice_to_u16(mw1_bytes);
        let (rp_w_bytes, _) = dir.tensor_bytes("markov_head.rnn_proj.weight", StDType::Bf16)?;
        let rnn_proj_w_bf16 = bf16_slice_to_u16(rp_w_bytes);
        let (rp_b_bytes, _) = dir.tensor_bytes("markov_head.rnn_proj.bias", StDType::Bf16)?;
        let rnn_proj_b = bf16_slice_to_f32(rp_b_bytes);
        let (ho_w_bytes, _) = dir.tensor_bytes("markov_head.hidden_out.weight", StDType::Bf16)?;
        let hidden_out_w = bf16_slice_to_f32(ho_w_bytes);
        let (ho_b_bytes, _) = dir.tensor_bytes("markov_head.hidden_out.bias", StDType::Bf16)?;
        let hidden_out_b = bf16_slice_to_f32(ho_b_bytes);

        let mut layers = Vec::with_capacity(n_stages);
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let is_local = config.local_layer_ids.contains(&i);
            layers.push(MtpLayerWeights {
                dflare_norm: dir.upload(
                    backend,
                    &format!("{p}.dflare_norm.weight"),
                    StDType::Bf16,
                )?,
                input_ln: dir.upload(
                    backend,
                    &format!("{p}.input_layernorm.weight"),
                    StDType::Bf16,
                )?,
                post_attn_ln: dir.upload(
                    backend,
                    &format!("{p}.post_attention_layernorm.weight"),
                    StDType::Bf16,
                )?,
                q_proj: dir.upload(
                    backend,
                    &format!("{p}.self_attn.q_proj.weight"),
                    StDType::Bf16,
                )?,
                k_proj: dir.upload(
                    backend,
                    &format!("{p}.self_attn.k_proj.weight"),
                    StDType::Bf16,
                )?,
                v_proj: dir.upload(
                    backend,
                    &format!("{p}.self_attn.v_proj.weight"),
                    StDType::Bf16,
                )?,
                o_proj: dir.upload(
                    backend,
                    &format!("{p}.self_attn.o_proj.weight"),
                    StDType::Bf16,
                )?,
                q_norm: dir.upload(
                    backend,
                    &format!("{p}.self_attn.q_norm.weight"),
                    StDType::Bf16,
                )?,
                k_norm: dir.upload(
                    backend,
                    &format!("{p}.self_attn.k_norm.weight"),
                    StDType::Bf16,
                )?,
                k_proj_ctx: dir.upload(
                    backend,
                    &format!("{p}.self_attn.k_proj_ctx.weight"),
                    StDType::Bf16,
                )?,
                v_proj_ctx: dir.upload(
                    backend,
                    &format!("{p}.self_attn.v_proj_ctx.weight"),
                    StDType::Bf16,
                )?,
                k_norm_ctx: dir.upload(
                    backend,
                    &format!("{p}.self_attn.k_norm_ctx.weight"),
                    StDType::Bf16,
                )?,
                gate_proj: dir.upload(
                    backend,
                    &format!("{p}.mlp.gate_proj.weight"),
                    StDType::Bf16,
                )?,
                up_proj: dir.upload(backend, &format!("{p}.mlp.up_proj.weight"), StDType::Bf16)?,
                down_proj: dir.upload(
                    backend,
                    &format!("{p}.mlp.down_proj.weight"),
                    StDType::Bf16,
                )?,
                is_local,
            });
        }

        Ok(Self {
            backend: Arc::clone(backend),
            config,
            text,
            fuse_alpha,
            mask_embed_f32,
            layers,
            final_norm,
            markov_dflare_norm,
            markov_w1_bf16,
            rnn_proj_w_bf16,
            rnn_proj_b,
            hidden_out_w,
            hidden_out_b,
            sessions: HashMap::new(),
        })
    }

    /// Parallel block size (1 anchor + `block_size - 1` drafted tokens).
    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.config.block_size
    }

    /// Allocates per-session draft KV caches.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on allocation failure.
    pub fn open_session(&mut self, session: SessionId, context_capacity: u32) -> CoreResult<()> {
        let kv_dim = u64::from(self.config.num_key_value_heads) * u64::from(self.config.head_dim);
        let mut kv = Vec::with_capacity(self.layers.len());
        for lw in &self.layers {
            let cap = if lw.is_local {
                self.config.sliding_window_size
            } else {
                context_capacity
            };
            let bytes = u64::from(cap) * kv_dim * 2;
            kv.push(MtpLayerKv {
                k_ctx: alloc(&self.backend, bytes)?,
                v_ctx: alloc(&self.backend, bytes)?,
                kv_cap: cap,
            });
        }
        self.sessions.insert(
            session,
            MtpSessionState {
                ctx_len: 0,
                kv,
                f_rnn: {
                    let buf = alloc(&self.backend, u64::from(self.config.hidden_size) * 4)?;
                    self.backend.fill_zero(buf.view())?;
                    buf
                },
                scratch: None,
            },
        );
        Ok(())
    }

    /// Releases a session's draft state.
    pub fn close_session(&mut self, session: SessionId) {
        self.sessions.remove(&session);
    }

    /// Absorbs `rows` newly committed target tap rows (`[n_taps, rows, H]`) into the
    /// draft context KV cache and updates `f_rnn` from row `rows - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on unknown session or device failure.
    pub fn absorb_taps(
        &mut self,
        session: SessionId,
        taps: DeviceView<'_>,
        stride_rows: u32,
        rows: u32,
        pos0: u32,
    ) -> CoreResult<()> {
        let Some(sess) = self.sessions.get_mut(&session) else {
            return Err(CoreError::UnknownSession(session));
        };
        let h = self.config.hidden_size;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let n_taps = self.config.target_layer_ids.len() as u32;
        if sess.scratch.as_ref().is_none_or(|sc| sc.cap_rows < rows) {
            let cap = rows.next_power_of_two();
            sess.scratch = Some(AbsorbScratch {
                cap_rows: cap,
                mixed: alloc(&self.backend, u64::from(cap) * u64::from(h) * 4)?,
                k: alloc(&self.backend, u64::from(cap) * u64::from(kv_dim) * 4)?,
                v: alloc(&self.backend, u64::from(cap) * u64::from(kv_dim) * 4)?,
            });
        }
        let scratch = sess.scratch.as_ref().expect("sized above");
        let (d_mixed, d_k, d_v) = (&scratch.mixed, &scratch.k, &scratch.v);
        let backend = self.backend.as_ref();

        let mix_args = MtpMixArgs {
            rows,
            stride_rows,
            hidden: h,
            n_taps,
            eps: self.config.rms_norm_eps,
        };

        for (i, (lw, kv)) in self.layers.iter().zip(&sess.kv).enumerate() {
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_mix_rmsnorm"),
                &LaunchArgs::new(
                    Extent3::linear(rows),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&mix_args)),
                        KernelArg::Buffer(taps),
                        KernelArg::Buffer(self.fuse_alpha[i].view()),
                        KernelArg::Buffer(lw.dflare_norm.view()),
                        KernelArg::Buffer(d_mixed.view()),
                    ],
                ),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows,
                    in_dim: h,
                    out_dim: kv_dim,
                },
                d_mixed.view(),
                lw.k_proj_ctx.view(),
                d_k.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows,
                    in_dim: h,
                    out_dim: kv_dim,
                },
                d_mixed.view(),
                lw.v_proj_ctx.view(),
                d_v.view(),
            )?;
            let qk_args = MtpQkArgs {
                rows,
                n_head: self.config.num_key_value_heads,
                head_dim: self.config.head_dim,
                pos0,
                rope_theta: self.config.rope_theta,
                eps: self.config.rms_norm_eps,
                kv_cap: kv.kv_cap,
                is_swa: u32::from(lw.is_local),
                write_cache: 1,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_qk_norm_rope"),
                &LaunchArgs::new(
                    Extent3::new(self.config.num_key_value_heads, rows, 1),
                    Extent3::linear(self.config.head_dim),
                    &[
                        KernelArg::Bytes(bytes_of(&qk_args)),
                        KernelArg::Buffer(d_k.view()),
                        KernelArg::Buffer(lw.k_norm_ctx.view()),
                        KernelArg::Buffer(kv.k_ctx.view()),
                    ],
                ),
            )?;
            // The values go to the cache as they are, narrowed on the device.
            let store = MtpStoreArgs {
                rows,
                width: kv_dim,
                pos0,
                kv_cap: kv.kv_cap,
                is_swa: u32::from(lw.is_local),
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_store_f16"),
                &LaunchArgs::new(
                    Extent3::linear((rows * kv_dim).div_ceil(256)),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&store)),
                        KernelArg::Buffer(d_v.view()),
                        KernelArg::Buffer(kv.v_ctx.view()),
                    ],
                ),
            )?;
        }

        // Compute `f_rnn` for row `rows - 1`
        let f_row_alpha = self.fuse_alpha.last().expect("f_rnn alpha row");
        backend.run(
            KernelRef::new(MODULE, "kernel_inkling2_mtp_mix_rmsnorm"),
            &LaunchArgs::new(
                Extent3::linear(rows),
                Extent3::linear(256),
                &[
                    KernelArg::Bytes(bytes_of(&mix_args)),
                    KernelArg::Buffer(taps),
                    KernelArg::Buffer(f_row_alpha.view()),
                    KernelArg::Buffer(self.markov_dflare_norm.view()),
                    KernelArg::Buffer(d_mixed.view()),
                ],
            ),
        )?;
        let last_off = u64::from(rows - 1) * u64::from(h) * 4;
        backend.copy_d2d(
            sess.f_rnn.view(),
            d_mixed.view().slice(last_off, u64::from(h) * 4)?,
        )?;
        sess.ctx_len = pos0 + rows;
        Ok(())
    }

    /// Proposes `depth` draft tokens (at most `block_size - 1`) behind `anchor`
    /// using the 3-layer parallel block backbone + sequential `rnn_hidden`
    /// Markov head. The backbone always runs the whole block - its positions
    /// attend to one another - and only the head's left-to-right refinement
    /// stops at `depth`.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on unknown session or device failure.
    pub fn draft_block(
        &self,
        session: SessionId,
        anchor: TokenId,
        depth: u32,
        embed_w: DeviceView<'_>,
        embed_norm_w: DeviceView<'_>,
        unembed_w: DeviceView<'_>,
    ) -> CoreResult<Vec<TokenId>> {
        let Some(sess) = self.sessions.get(&session) else {
            return Err(CoreError::UnknownSession(session));
        };
        let bs = self.config.block_size;
        let h = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let inter = self.config.intermediate_size;
        let backend = self.backend.as_ref();

        // Build [embed(anchor), mask_embed * (bs - 1)]
        let d_hidden = alloc(&self.backend, u64::from(bs) * u64::from(h) * 4)?;
        let d_tok = alloc(&self.backend, 4)?;
        let tok_raw = [anchor.get()];
        backend.copy_h2d(d_tok.view(), cast_slice(&tok_raw))?;
        ops::embed_rmsnorm_bf16(
            backend,
            NormArgs {
                rows: 1,
                hidden: h,
                eps: self.text.rms_norm_eps,
            },
            d_tok.view(),
            embed_w,
            embed_norm_w,
            d_hidden.view().slice(0, u64::from(h) * 4)?,
        )?;
        for slot in 1..bs {
            backend.copy_d2d(
                d_hidden
                    .view()
                    .slice(u64::from(slot) * u64::from(h) * 4, u64::from(h) * 4)?,
                self.mask_embed_f32.view(),
            )?;
        }

        let d_norm = alloc(&self.backend, u64::from(bs) * u64::from(h) * 4)?;
        let d_q = alloc(&self.backend, u64::from(bs) * u64::from(q_dim) * 4)?;
        let d_k = alloc(&self.backend, u64::from(bs) * u64::from(kv_dim) * 4)?;
        let d_v = alloc(&self.backend, u64::from(bs) * u64::from(kv_dim) * 4)?;
        let d_attn = alloc(&self.backend, u64::from(bs) * u64::from(q_dim) * 4)?;
        let d_out = alloc(&self.backend, u64::from(bs) * u64::from(h) * 4)?;
        let d_gate = alloc(&self.backend, u64::from(bs) * u64::from(inter) * 4)?;
        let d_up = alloc(&self.backend, u64::from(bs) * u64::from(inter) * 4)?;
        let dummy_cache = alloc(&self.backend, 4)?;

        let pos0 = sess.ctx_len;
        let attn_scale = 1.0 / (self.config.head_dim as f32).sqrt();

        for (lw, kv) in self.layers.iter().zip(&sess.kv) {
            ops::rmsnorm_bf16(
                backend,
                NormArgs {
                    rows: bs,
                    hidden: h,
                    eps: self.config.rms_norm_eps,
                },
                d_hidden.view(),
                lw.input_ln.view(),
                d_norm.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: h,
                    out_dim: q_dim,
                },
                d_norm.view(),
                lw.q_proj.view(),
                d_q.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: h,
                    out_dim: kv_dim,
                },
                d_norm.view(),
                lw.k_proj.view(),
                d_k.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: h,
                    out_dim: kv_dim,
                },
                d_norm.view(),
                lw.v_proj.view(),
                d_v.view(),
            )?;

            let q_args = MtpQkArgs {
                rows: bs,
                n_head: self.config.num_attention_heads,
                head_dim: self.config.head_dim,
                pos0,
                rope_theta: self.config.rope_theta,
                eps: self.config.rms_norm_eps,
                kv_cap: kv.kv_cap,
                is_swa: u32::from(lw.is_local),
                write_cache: 0,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_qk_norm_rope"),
                &LaunchArgs::new(
                    Extent3::new(self.config.num_attention_heads, bs, 1),
                    Extent3::linear(self.config.head_dim),
                    &[
                        KernelArg::Bytes(bytes_of(&q_args)),
                        KernelArg::Buffer(d_q.view()),
                        KernelArg::Buffer(lw.q_norm.view()),
                        KernelArg::Buffer(dummy_cache.view()),
                    ],
                ),
            )?;
            let k_args = MtpQkArgs {
                n_head: self.config.num_key_value_heads,
                ..q_args
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_qk_norm_rope"),
                &LaunchArgs::new(
                    Extent3::new(self.config.num_key_value_heads, bs, 1),
                    Extent3::linear(self.config.head_dim),
                    &[
                        KernelArg::Bytes(bytes_of(&k_args)),
                        KernelArg::Buffer(d_k.view()),
                        KernelArg::Buffer(lw.k_norm.view()),
                        KernelArg::Buffer(dummy_cache.view()),
                    ],
                ),
            )?;

            let attn_args = MtpAttnArgs {
                block_size: bs,
                n_head: self.config.num_attention_heads,
                n_head_kv: self.config.num_key_value_heads,
                head_dim: self.config.head_dim,
                ctx_len: sess.ctx_len,
                kv_cap: kv.kv_cap,
                window_size: self.config.sliding_window_size,
                is_swa: u32::from(lw.is_local),
                scale: attn_scale,
            };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_block_attn"),
                &LaunchArgs::new(
                    Extent3::new(self.config.num_attention_heads, bs, 1),
                    Extent3::linear(self.config.head_dim),
                    &[
                        KernelArg::Bytes(bytes_of(&attn_args)),
                        KernelArg::Buffer(d_q.view()),
                        KernelArg::Buffer(d_k.view()),
                        KernelArg::Buffer(d_v.view()),
                        KernelArg::Buffer(kv.k_ctx.view()),
                        KernelArg::Buffer(kv.v_ctx.view()),
                        KernelArg::Buffer(d_attn.view()),
                    ],
                ),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: q_dim,
                    out_dim: h,
                },
                d_attn.view(),
                lw.o_proj.view(),
                d_out.view(),
            )?;
            ops::add2_f32(
                backend,
                bs * h,
                d_hidden.view(),
                d_out.view(),
                d_hidden.view(),
            )?;

            // Dense SwiGLU MLP
            ops::rmsnorm_bf16(
                backend,
                NormArgs {
                    rows: bs,
                    hidden: h,
                    eps: self.config.rms_norm_eps,
                },
                d_hidden.view(),
                lw.post_attn_ln.view(),
                d_norm.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: h,
                    out_dim: inter,
                },
                d_norm.view(),
                lw.gate_proj.view(),
                d_gate.view(),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: h,
                    out_dim: inter,
                },
                d_norm.view(),
                lw.up_proj.view(),
                d_up.view(),
            )?;
            let mul_args = ops::AddArgs { count: bs * inter };
            backend.run(
                KernelRef::new(MODULE, "kernel_inkling2_mtp_swiglu_mul"),
                &LaunchArgs::new(
                    Extent3::linear(mul_args.count.div_ceil(256)),
                    Extent3::linear(256),
                    &[
                        KernelArg::Bytes(bytes_of(&mul_args)),
                        KernelArg::Buffer(d_gate.view()),
                        KernelArg::Buffer(d_up.view()),
                        KernelArg::Buffer(d_gate.view()),
                    ],
                ),
            )?;
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows: bs,
                    in_dim: inter,
                    out_dim: h,
                },
                d_gate.view(),
                lw.down_proj.view(),
                d_out.view(),
            )?;
            ops::add2_f32(
                backend,
                bs * h,
                d_hidden.view(),
                d_out.view(),
                d_hidden.view(),
            )?;
        }

        // Final norm on the `bs - 1` drafted positions (slots 1..bs)
        ops::rmsnorm_bf16(
            backend,
            NormArgs {
                rows: bs,
                hidden: h,
                eps: self.config.rms_norm_eps,
            },
            d_hidden.view(),
            self.final_norm.view(),
            d_hidden.view(),
        )?;

        // Read back the `bs - 1` normalized hidden states (tiny: 7 * 2304 * 4 = 64 KB)
        backend.sync()?;
        let mut hidden_bytes = vec![0u8; (u64::from(bs) * u64::from(h) * 4) as usize];
        backend.copy_d2h(d_hidden.view(), &mut hidden_bytes)?;
        let hidden_f32: &[f32] = cast_slice(&hidden_bytes);
        let mut f_rnn_bytes = vec![0u8; (h * 4) as usize];
        backend.copy_d2h(sess.f_rnn.view(), &mut f_rnn_bytes)?;
        let f_rnn: &[f32] = cast_slice(&f_rnn_bytes);

        // Sequential RNN-hidden Markov refinement over positions 1..bs:
        // At each step, add `hidden_out(latent)` into the hidden vector BEFORE projecting
        // through `unembed_w` (algebraically identical to adding `unembed @ hidden_out(latent)`
        // to base logits, and only costs 1 LM-head call per step instead of folding a 201024x256 matrix!).
        let rank = self.config.markov_rank as usize;
        let hs = h as usize;
        let in_rnn_w = 2 * rank + 2 * hs;
        let mut state = vec![0.0_f32; rank];
        let mut prev_tok = anchor.get() as usize;
        let mut drafted = Vec::with_capacity((bs - 1) as usize);

        let d_step_h = alloc(&self.backend, u64::from(h) * 4)?;
        let vocab = self.text.unpadded_vocab_size;
        let d_step_logits = alloc(&self.backend, u64::from(vocab) * 4)?;
        let mut logits_bytes = vec![0u8; (vocab * 4) as usize];

        for pos in 1..=depth.min(bs - 1) as usize {
            let h_pos = &hidden_f32[pos * hs..(pos + 1) * hs];
            // Build rnn input: [state, markov_w1[prev_tok], h_pos, f_rnn]
            let mut inp = Vec::with_capacity(in_rnn_w);
            inp.extend_from_slice(&state);
            let emb_off = prev_tok * rank;
            for r in 0..rank {
                inp.push(bf16_to_f32(self.markov_w1_bf16[emb_off + r]));
            }
            inp.extend_from_slice(h_pos);
            inp.extend_from_slice(f_rnn);

            let mut proj = self.rnn_proj_b.clone();
            for (o, p_out) in proj.iter_mut().enumerate() {
                let row_off = o * in_rnn_w;
                let mut acc = 0.0_f32;
                for (k, &iv) in inp.iter().enumerate() {
                    acc += bf16_to_f32(self.rnn_proj_w_bf16[row_off + k]) * iv;
                }
                *p_out += acc;
            }
            let mut latent = vec![0.0_f32; rank];
            for r in 0..rank {
                let g = sigmoid(proj[r]);
                let cand = proj[rank + r].tanh();
                let out_v = proj[2 * rank + r].tanh();
                state[r] = g * state[r] + (1.0 - g) * cand;
                latent[r] = out_v;
            }

            // refined_h = h_pos + hidden_out_w @ latent + hidden_out_b
            let mut refined_h = vec![0.0_f32; hs];
            for i in 0..hs {
                let mut acc = h_pos[i] + self.hidden_out_b[i];
                let row_off = i * rank;
                for r in 0..rank {
                    acc += self.hidden_out_w[row_off + r] * latent[r];
                }
                refined_h[i] = acc;
            }

            backend.copy_h2d(d_step_h.view(), cast_slice(&refined_h))?;
            ops::lm_head_softcap(
                backend,
                ops::LmHeadArgs {
                    rows: 1,
                    hidden: h,
                    vocab,
                    softcap: self.text.final_logit_softcapping,
                },
                d_step_h.view(),
                unembed_w,
                d_step_logits.view(),
            )?;
            backend.sync()?;
            backend.copy_d2h(d_step_logits.view(), &mut logits_bytes)?;
            let logits: &[f32] = cast_slice(&logits_bytes);
            let mut best_id = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_id = i;
                }
            }
            prev_tok = best_id;
            drafted.push(TokenId::new(best_id as u32));
        }

        Ok(drafted)
    }
}
