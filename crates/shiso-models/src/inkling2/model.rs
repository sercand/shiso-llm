//! Inkling-2 looped MoE model execution (`shiso_core::Model`).

#![expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    reason = "forward pass dispatches 48 logical looped layers with explicit tensor shapes"
)]

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use bytemuck::cast_slice;
use shiso_core::{
    Backend, CoreError, CoreResult, DecodeStep, DeviceBuffer, Logits, MediaModel, Model, ModelInfo,
    SessionId, Speculated, Speculating, TokenId, VisionPrefill, alloc,
};

use super::config::{Inkling2Config, LogicalLayer};
use super::mtp::Inkling2MtpDrafter;
use super::ops::{
    self, AttnResArgs, LinearArgs, LmHeadArgs, LoopCtrlArgs, LoopCtrlViews, MoeW2Args, MoeW13Args,
    Mxfp8Views, NormArgs, Nvfp4Views, QkvrPrepArgs, QkvrPrepViews, RelAttnArgs, RelAttnViews,
    RouteArgs, SconvDynamicArgs, SconvStaticArgs,
};
use super::weights::{
    AttnNormWeights, ExpertWeights, Inkling2TextWeights, KvConvWeights, SafetensorsDir,
    load_text_weights,
};

/// Per-logical-layer KV cache and short-convolution state for one session.
#[derive(Debug)]
struct LayerState {
    /// Key cache (`F16 [kv_cap, n_head_kv, head_dim]`).
    k_cache: DeviceBuffer,
    /// Value cache (`F16 [kv_cap, n_head_kv, head_dim]`).
    v_cache: DeviceBuffer,
    /// Slot capacity of `k_cache` / `v_cache`.
    kv_cap: u32,
    /// K short-conv lag state (`F32 [k - 1, kv_dim]`).
    k_conv_state: DeviceBuffer,
    /// V short-conv lag state (`F32 [k - 1, kv_dim]`).
    v_conv_state: DeviceBuffer,
    /// Post-attention short-conv lag state (`F32 [k - 1, H]`).
    attn_conv_state: DeviceBuffer,
    /// Post-MLP short-conv lag state (`F32 [k - 1, H]`).
    mlp_conv_state: DeviceBuffer,
}

/// Active session state.
#[derive(Debug)]
struct SessionState {
    /// Current sequence position (number of tokens committed so far).
    pos: u32,
    /// One state slot per logical layer (`0..num_logical_layers`).
    layers: Vec<LayerState>,
}

/// Reusable scratch buffers grown on demand for a given batch/chunk `rows`.
#[derive(Debug)]
struct Scratch {
    cap_rows: u32,
    tokens: DeviceBuffer,
    /// Residual block stack (`F32 [max_blocks, cap_rows, H]`).
    blocks: DeviceBuffer,
    /// Prelude snapshot of `blocks` for looped core passes.
    prelude_blocks: DeviceBuffer,
    /// Cross-pass read-only block bank from previous pass.
    bank_blocks: DeviceBuffer,
    /// Current partial block (`F32 [cap_rows, H]`).
    partial: DeviceBuffer,
    /// Prelude snapshot of `partial`.
    prelude_partial: DeviceBuffer,
    /// Previous pass final partial (`bank_last`).
    bank_last: DeviceBuffer,
    /// Up to 3 mixed attention inputs (`F32 [3, cap_rows, H]`).
    attn_in: DeviceBuffer,
    /// Cross-pass messages `(attn, mlp)` and `(k, v)` (`F32 [2, cap_rows, H]` each).
    msg_attn_mlp: DeviceBuffer,
    msg_kv: DeviceBuffer,
    /// Controlled / normed attention inputs (`F32 [3, cap_rows, H]`).
    norm_q: DeviceBuffer,
    norm_k: DeviceBuffer,
    norm_v: DeviceBuffer,
    /// Q, K, V, R, Gate, Sink, AttnOut, WoOut, AttnSconvOut.
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    k_conv: DeviceBuffer,
    v_conv: DeviceBuffer,
    k_dyn_w: DeviceBuffer,
    v_dyn_w: DeviceBuffer,
    r: DeviceBuffer,
    rel_bias: DeviceBuffer,
    gate: DeviceBuffer,
    sinks: DeviceBuffer,
    attn_out: DeviceBuffer,
    /// Split-K Flash-Decoding partials for long-context full-attention decode (`[n_head, 16, 2]` and `[n_head, 16, head_dim]`).
    split_ml: DeviceBuffer,
    split_out: DeviceBuffer,
    wo_out: DeviceBuffer,
    attn_sconv_out: DeviceBuffer,
    /// MLP scratch.
    mlp_in: DeviceBuffer,
    mlp_ctrl: DeviceBuffer,
    mlp_norm: DeviceBuffer,
    router_logits: DeviceBuffer,
    expert_ids: DeviceBuffer,
    expert_weights: DeviceBuffer,
    expert_counts: DeviceBuffer,
    expert_cols: DeviceBuffer,
    moe_act: DeviceBuffer,
    moe_w2_slots: DeviceBuffer,
    moe_out: DeviceBuffer,
    mlp_sconv_out: DeviceBuffer,
    /// Scratch bottleneck buffer for parallel loop controller (`F32 [cap_rows, rank]`).
    ctrl_z: DeviceBuffer,
    /// Final hidden & output logits (`F32 [cap_rows, unpadded_vocab]`).
    final_hidden: DeviceBuffer,
    logits: DeviceBuffer,
    /// Captured MTP tap rows if requested (`F32 [n_taps, cap_rows, H]`).
    mtp_taps: Option<DeviceBuffer>,
}

impl Scratch {
    fn alloc_for(
        backend: &Arc<dyn Backend>,
        cfg: &Inkling2Config,
        cap_rows: u32,
        n_taps: u32,
    ) -> CoreResult<Self> {
        let t = &cfg.text;
        let r = u64::from(cap_rows);
        let h = u64::from(t.hidden_size);
        let q_dim = u64::from(t.num_attention_heads) * u64::from(t.head_dim);
        let kv_dim = u64::from(t.num_key_value_heads) * u64::from(t.head_dim);
        let rel_dim = u64::from(t.num_attention_heads) * u64::from(t.d_rel);
        let rel_bias_dim = u64::from(t.num_attention_heads) * u64::from(t.rel_extent);
        let dyn_w_dim = u64::from(t.num_key_value_heads) * u64::from(t.sconv_kernel_size);
        let max_blocks = u64::from(
            t.num_hidden_layers
                .div_ceil(t.n_layers_per_attn_res_block)
                .max(4)
                + 2,
        );
        let f32b = 4u64;

        let mtp_taps = if n_taps > 0 {
            Some(alloc(backend, u64::from(n_taps) * r * h * f32b)?)
        } else {
            None
        };

        Ok(Self {
            cap_rows,
            tokens: alloc(backend, r * 4)?,
            blocks: alloc(backend, max_blocks * r * h * f32b)?,
            prelude_blocks: alloc(backend, max_blocks * r * h * f32b)?,
            bank_blocks: alloc(backend, max_blocks * r * h * f32b)?,
            partial: alloc(backend, r * h * f32b)?,
            prelude_partial: alloc(backend, r * h * f32b)?,
            bank_last: alloc(backend, r * h * f32b)?,
            attn_in: alloc(backend, 3 * r * h * f32b)?,
            msg_attn_mlp: alloc(backend, 2 * r * h * f32b)?,
            msg_kv: alloc(backend, 2 * r * h * f32b)?,
            norm_q: alloc(backend, r * h * f32b)?,
            norm_k: alloc(backend, r * h * f32b)?,
            norm_v: alloc(backend, r * h * f32b)?,
            q: alloc(backend, r * q_dim * f32b)?,
            k: alloc(backend, r * kv_dim * f32b)?,
            v: alloc(backend, r * kv_dim * f32b)?,
            k_conv: alloc(backend, r * kv_dim * f32b)?,
            v_conv: alloc(backend, r * kv_dim * f32b)?,
            k_dyn_w: alloc(backend, r * dyn_w_dim * f32b)?,
            v_dyn_w: alloc(backend, r * dyn_w_dim * f32b)?,
            r: alloc(backend, r * rel_dim * f32b)?,
            rel_bias: alloc(backend, r * rel_bias_dim * f32b)?,
            gate: alloc(backend, r * q_dim * f32b)?,
            sinks: alloc(backend, r * u64::from(t.num_attention_heads) * f32b)?,
            attn_out: alloc(backend, r * q_dim * f32b)?,
            split_ml: alloc(backend, u64::from(t.num_attention_heads) * 16 * 2 * f32b)?,
            split_out: alloc(
                backend,
                u64::from(t.num_attention_heads) * 16 * u64::from(t.head_dim) * f32b,
            )?,
            wo_out: alloc(backend, r * h * f32b)?,
            attn_sconv_out: alloc(backend, r * h * f32b)?,
            mlp_in: alloc(backend, r * h * f32b)?,
            mlp_ctrl: alloc(backend, r * h * f32b)?,
            mlp_norm: alloc(backend, r * h * f32b)?,
            router_logits: alloc(backend, r * u64::from(t.n_routed_experts) * f32b)?,
            expert_ids: alloc(backend, r * u64::from(t.num_experts_per_tok) * 4)?,
            expert_weights: alloc(backend, r * u64::from(t.num_experts_per_tok) * f32b)?,
            expert_counts: alloc(backend, u64::from(t.n_routed_experts) * 4)?,
            expert_cols: alloc(backend, u64::from(t.n_routed_experts) * r * 4)?,
            moe_act: alloc(
                backend,
                r * u64::from(t.num_experts_per_tok) * u64::from(t.intermediate_size) * f32b,
            )?,
            moe_w2_slots: alloc(backend, r * u64::from(t.num_experts_per_tok) * h * f32b)?,
            moe_out: alloc(backend, r * h * f32b)?,
            mlp_sconv_out: alloc(backend, r * h * f32b)?,
            ctrl_z: alloc(
                backend,
                r * u64::from(t.loop_config.map_or(64, |lc| lc.gate_rank)) * f32b,
            )?,
            final_hidden: alloc(backend, r * h * f32b)?,
            logits: alloc(
                backend,
                u64::from(cap_rows.clamp(8, 16)) * u64::from(t.unpadded_vocab_size) * f32b,
            )?,
            mtp_taps,
        })
    }
}

/// Loaded Inkling-2 looped MoE model on a [`Backend`].
#[derive(Debug)]
pub struct Inkling2Model {
    backend: Arc<dyn Backend>,
    config: Inkling2Config,
    info: ModelInfo,
    schedule: Vec<LogicalLayer>,
    weights: Inkling2TextWeights,
    sessions: HashMap<SessionId, SessionState>,
    next_session_id: NonZeroU64,
    scratch: Scratch,
    /// Optional target layer indices to capture into `scratch.mtp_taps`.
    tap_layers: Vec<u32>,
    /// Optional DSpark / MTP block drafter (`mtp.safetensors`).
    mtp: Option<Inkling2MtpDrafter>,
}

impl Inkling2Model {
    /// Loads an Inkling-2 checkpoint directory onto `backend` (without MTP).
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when configuration, shard files, or device allocations fail.
    pub fn load(backend: Arc<dyn Backend>, dir: &Path, context_capacity: u32) -> CoreResult<Self> {
        Self::load_with_options(backend, dir, context_capacity, false)
    }

    /// Loads an Inkling-2 checkpoint directory onto `backend`, optionally loading
    /// the bundled `mtp.safetensors` DSpark block drafter.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] when configuration, shard files, or device allocations fail.
    pub fn load_with_options(
        backend: Arc<dyn Backend>,
        dir: &Path,
        context_capacity: u32,
        enable_mtp: bool,
    ) -> CoreResult<Self> {
        let config = Inkling2Config::from_dir(dir)?;
        let cap = if context_capacity == 0 {
            return Err(CoreError::InvalidValue {
                key: "context_capacity",
                value: "0".to_owned(),
            });
        } else {
            context_capacity.min(config.text.model_max_length)
        };
        let st_dir = SafetensorsDir::open_model(dir)?;
        let weights = load_text_weights(&backend, &st_dir, &config)?;
        let schedule = config.text.schedule();
        let info = ModelInfo {
            context_capacity: cap,
            vocab: config.text.unpadded_vocab_size,
            hidden: config.text.hidden_size,
            n_layers: config.text.num_logical_layers(),
            stop_tokens: config.stop_tokens,
        };
        let mtp = if enable_mtp {
            let mtp_cfg = config.mtp.clone().ok_or_else(|| CoreError::InvalidValue {
                key: "mtp_config",
                value: "speculation requested but config.json has no mtp_config".to_owned(),
            })?;
            let mtp_path = dir.join("mtp.safetensors");
            Some(Inkling2MtpDrafter::load(
                &backend,
                &mtp_path,
                mtp_cfg,
                config.text.clone(),
            )?)
        } else {
            None
        };
        let tap_layers = if mtp.is_some() {
            config
                .mtp
                .as_ref()
                .map(|m| m.target_layer_ids.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let scratch = Scratch::alloc_for(&backend, &config, 16, tap_layers.len() as u32)?;
        Ok(Self {
            backend,
            config,
            info,
            schedule,
            weights,
            sessions: HashMap::new(),
            next_session_id: NonZeroU64::MIN,
            scratch,
            tap_layers,
            mtp,
        })
    }

    /// Returns the parsed configuration.
    #[must_use]
    pub fn config(&self) -> &Inkling2Config {
        &self.config
    }

    fn ensure_scratch(&mut self, rows: u32) -> CoreResult<()> {
        if rows > self.scratch.cap_rows {
            self.scratch = Scratch::alloc_for(
                &self.backend,
                &self.config,
                rows.next_power_of_two(),
                self.tap_layers.len() as u32,
            )?;
        }
        Ok(())
    }

    /// Runs `rows` tokens through all 48 logical layers for `session`.
    ///
    /// If `compute_last_logits` is true, computes logits for row `rows - 1` into
    /// `self.scratch.logits[0..vocab]`.
    fn forward_tokens(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        compute_last_logits: bool,
    ) -> CoreResult<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let rows = u32::try_from(tokens.len()).map_err(|_| CoreError::InvalidValue {
            key: "tokens",
            value: "batch exceeds u32".to_owned(),
        })?;
        self.ensure_scratch(rows)?;
        let h = self.config.text.hidden_size;
        let raw_ids: Vec<u32> = tokens.iter().map(|tok| tok.get()).collect();
        self.backend.copy_h2d(
            self.scratch.tokens.view().slice(0, u64::from(rows) * 4)?,
            cast_slice(&raw_ids),
        )?;
        ops::embed_rmsnorm_bf16(
            self.backend.as_ref(),
            NormArgs {
                rows,
                hidden: h,
                eps: self.config.text.rms_norm_eps,
            },
            self.scratch.tokens.view(),
            self.weights.embed.view(),
            self.weights.embed_norm.view(),
            self.scratch.partial.view(),
        )?;
        self.forward_already_embedded(session, rows, compute_last_logits, false)
    }

    /// Runs the 48-layer forward pass assuming `self.scratch.partial[0..rows]` already
    /// holds the input embeddings (either token embeddings or vision-spliced embeddings).
    fn forward_already_embedded(
        &mut self,
        session: SessionId,
        rows: u32,
        compute_logits: bool,
        all_rows_logits: bool,
    ) -> CoreResult<()> {
        let Some(sess) = self.sessions.get_mut(&session) else {
            return Err(CoreError::UnknownSession(session));
        };
        if sess.pos.saturating_add(rows) > self.info.context_capacity {
            return Err(CoreError::ContextOverflow {
                session,
                cap: self.info.context_capacity,
            });
        }

        let pos0 = sess.pos;
        let t = &self.config.text;
        let h = t.hidden_size;
        let r_u64 = u64::from(rows);
        let row_bytes = r_u64 * u64::from(h) * 4;
        let backend = self.backend.as_ref();

        let mut n_blocks = 0u32;
        let mut prelude_n_blocks = 0u32;
        let mut bank_n_blocks = 0u32;

        let sconv_static_scale = 1.0_f32 / ((t.sconv_kernel_size + 1) as f32).sqrt();
        let sconv_dyn_scale = 1.0_f32 / ((2 * t.sconv_kernel_size + 1) as f32).sqrt();
        let attn_scale = 1.0_f32 / (t.head_dim as f32).sqrt();

        for &ll in &self.schedule {
            let p = ll.physical_idx as usize;
            let l_idx = ll.logical_idx as usize;
            let lw = &self.weights.layers[p];
            let ls = &sess.layers[l_idx];

            // Capture MTP tap BEFORE layer `ll.logical_idx` (matches reference `for i, layer in enumerate(layers): if i in layers_to_capture: capture(partial)`)
            if let Some(taps_buf) = &self.scratch.mtp_taps
                && let Some(tap_pos) = self.tap_layers.iter().position(|&id| id == ll.logical_idx)
            {
                backend.copy_d2d(
                    taps_buf
                        .view()
                        .slice(tap_pos as u64 * row_bytes, row_bytes)?,
                    self.scratch.partial.view().slice(0, row_bytes)?,
                )?;
            }

            // Loop boundary management
            let mut has_messages = false;
            if ll.in_loop_core {
                let lc = self
                    .config
                    .text
                    .loop_config
                    .expect("in_loop_core implies loop_config");
                if ll.physical_idx == lc.core_start {
                    if ll.pass_idx == 0 {
                        prelude_n_blocks = n_blocks;
                        if n_blocks > 0 {
                            backend.copy_d2d(
                                self.scratch
                                    .prelude_blocks
                                    .view()
                                    .slice(0, u64::from(n_blocks) * row_bytes)?,
                                self.scratch
                                    .blocks
                                    .view()
                                    .slice(0, u64::from(n_blocks) * row_bytes)?,
                            )?;
                        }
                        backend.copy_d2d(
                            self.scratch.prelude_partial.view().slice(0, row_bytes)?,
                            self.scratch.partial.view().slice(0, row_bytes)?,
                        )?;
                    } else {
                        // bank = blocks[prelude_n_blocks + 1 .. n_blocks], bank_last = partial
                        let start_b = prelude_n_blocks + 1;
                        bank_n_blocks = n_blocks.saturating_sub(start_b);
                        if bank_n_blocks > 0 {
                            backend.copy_d2d(
                                self.scratch
                                    .bank_blocks
                                    .view()
                                    .slice(0, u64::from(bank_n_blocks) * row_bytes)?,
                                self.scratch.blocks.view().slice(
                                    u64::from(start_b) * row_bytes,
                                    u64::from(bank_n_blocks) * row_bytes,
                                )?,
                            )?;
                        }
                        backend.copy_d2d(
                            self.scratch.bank_last.view().slice(0, row_bytes)?,
                            self.scratch.partial.view().slice(0, row_bytes)?,
                        )?;
                        // restore prelude state
                        n_blocks = prelude_n_blocks;
                        if n_blocks > 0 {
                            backend.copy_d2d(
                                self.scratch
                                    .blocks
                                    .view()
                                    .slice(0, u64::from(n_blocks) * row_bytes)?,
                                self.scratch
                                    .prelude_blocks
                                    .view()
                                    .slice(0, u64::from(n_blocks) * row_bytes)?,
                            )?;
                        }
                        backend.copy_d2d(
                            self.scratch.partial.view().slice(0, row_bytes)?,
                            self.scratch.prelude_partial.view().slice(0, row_bytes)?,
                        )?;
                    }
                }

                if ll.pass_idx > 0 {
                    has_messages = true;
                    let q_row = u64::from(2 * (ll.physical_idx - lc.core_start));
                    let q_byte_off = q_row * u64::from(h) * 4;
                    let q_byte_len = 2 * u64::from(h) * 4;
                    let cp_q = self
                        .weights
                        .cross_pass_queries
                        .as_ref()
                        .expect("loop cross_pass_queries")
                        .view()
                        .slice(q_byte_off, q_byte_len)?;
                    let cp_kv_q = self
                        .weights
                        .cross_pass_kv_queries
                        .as_ref()
                        .expect("loop cross_pass_kv_queries")
                        .view()
                        .slice(q_byte_off, q_byte_len)?;

                    ops::attn_res(
                        backend,
                        AttnResArgs {
                            rows,
                            hidden: h,
                            n_blocks: bank_n_blocks,
                            n_queries: 2,
                            eps: t.rms_norm_eps,
                            has_addend: 0,
                        },
                        self.scratch.bank_blocks.view(),
                        self.scratch.bank_last.view(),
                        self.scratch.bank_last.view(),
                        cp_q,
                        self.scratch.msg_attn_mlp.view(),
                    )?;
                    ops::attn_res(
                        backend,
                        AttnResArgs {
                            rows,
                            hidden: h,
                            n_blocks: bank_n_blocks,
                            n_queries: 2,
                            eps: t.rms_norm_eps,
                            has_addend: 0,
                        },
                        self.scratch.bank_blocks.view(),
                        self.scratch.bank_last.view(),
                        self.scratch.bank_last.view(),
                        cp_kv_q,
                        self.scratch.msg_kv.view(),
                    )?;
                }
            }

            // Attention input via block attention residual
            let n_attn_q: u32 = if ll.is_local { 1 } else { 3 };
            if n_blocks == 0 {
                backend.copy_d2d(
                    self.scratch.attn_in.view().slice(0, row_bytes)?,
                    self.scratch.partial.view().slice(0, row_bytes)?,
                )?;
                if !ll.is_local {
                    backend.copy_d2d(
                        self.scratch.attn_in.view().slice(row_bytes, row_bytes)?,
                        self.scratch.partial.view().slice(0, row_bytes)?,
                    )?;
                    backend.copy_d2d(
                        self.scratch
                            .attn_in
                            .view()
                            .slice(2 * row_bytes, row_bytes)?,
                        self.scratch.partial.view().slice(0, row_bytes)?,
                    )?;
                }
            } else {
                let q_view = lw
                    .attn_res_query
                    .as_ref()
                    .expect("layer > 0 has attn_res_query")
                    .view();
                ops::attn_res(
                    backend,
                    AttnResArgs {
                        rows,
                        hidden: h,
                        n_blocks,
                        n_queries: n_attn_q,
                        eps: t.rms_norm_eps,
                        has_addend: 0,
                    },
                    self.scratch.blocks.view(),
                    self.scratch.partial.view(),
                    self.scratch.partial.view(),
                    q_view,
                    self.scratch.attn_in.view(),
                )?;
            }

            // Apply loop controllers (if pass > 0) and input RMSNorm(s)
            let (q_in_view, k_in_view, v_in_view) = if has_messages {
                let ctrl = lw.controllers.as_ref().expect("core layer has controllers");
                let lc = self.config.text.loop_config.expect("loop_config");
                let c_args = LoopCtrlArgs {
                    rows,
                    hidden: h,
                    rank: lc.gate_rank,
                    eps: t.rms_norm_eps,
                };
                let src_q = self.scratch.attn_in.view().slice(0, row_bytes)?;
                let src_k = if ll.is_local {
                    src_q
                } else {
                    self.scratch.attn_in.view().slice(row_bytes, row_bytes)?
                };
                let src_v = if ll.is_local {
                    src_q
                } else {
                    self.scratch
                        .attn_in
                        .view()
                        .slice(2 * row_bytes, row_bytes)?
                };
                let msg_q = self.scratch.msg_attn_mlp.view().slice(0, row_bytes)?;
                let msg_k = self.scratch.msg_kv.view().slice(0, row_bytes)?;
                let msg_v = self.scratch.msg_kv.view().slice(row_bytes, row_bytes)?;

                // Write controlled streams into norm_q/k/v then normalize in-place
                ops::loop_control(
                    backend,
                    c_args,
                    &LoopCtrlViews {
                        cur: src_q,
                        msg: msg_q,
                        down_w: ctrl.attn.down.view(),
                        gain_w: ctrl.attn.gain.view(),
                        forget_w: ctrl.attn.forget.view(),
                        z_scratch: self.scratch.ctrl_z.view(),
                        out: self.scratch.norm_q.view(),
                    },
                )?;
                ops::loop_control(
                    backend,
                    c_args,
                    &LoopCtrlViews {
                        cur: src_k,
                        msg: msg_k,
                        down_w: ctrl.k.down.view(),
                        gain_w: ctrl.k.gain.view(),
                        forget_w: ctrl.k.forget.view(),
                        z_scratch: self.scratch.ctrl_z.view(),
                        out: self.scratch.norm_k.view(),
                    },
                )?;
                ops::loop_control(
                    backend,
                    c_args,
                    &LoopCtrlViews {
                        cur: src_v,
                        msg: msg_v,
                        down_w: ctrl.v.down.view(),
                        gain_w: ctrl.v.gain.view(),
                        forget_w: ctrl.v.forget.view(),
                        z_scratch: self.scratch.ctrl_z.view(),
                        out: self.scratch.norm_v.view(),
                    },
                )?;

                let nargs = NormArgs {
                    rows,
                    hidden: h,
                    eps: t.rms_norm_eps,
                };
                match &lw.attn_norm {
                    AttnNormWeights::Shared(nw) => {
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_q.view(),
                            nw.view(),
                            self.scratch.norm_q.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_k.view(),
                            nw.view(),
                            self.scratch.norm_k.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_v.view(),
                            nw.view(),
                            self.scratch.norm_v.view(),
                        )?;
                    }
                    AttnNormWeights::Split { q, k, v } => {
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_q.view(),
                            q.view(),
                            self.scratch.norm_q.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_k.view(),
                            k.view(),
                            self.scratch.norm_k.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.norm_v.view(),
                            v.view(),
                            self.scratch.norm_v.view(),
                        )?;
                    }
                }
                (
                    self.scratch.norm_q.view(),
                    self.scratch.norm_k.view(),
                    self.scratch.norm_v.view(),
                )
            } else {
                let nargs = NormArgs {
                    rows,
                    hidden: h,
                    eps: t.rms_norm_eps,
                };
                match &lw.attn_norm {
                    AttnNormWeights::Shared(nw) => {
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.attn_in.view().slice(0, row_bytes)?,
                            nw.view(),
                            self.scratch.norm_q.view(),
                        )?;
                        (
                            self.scratch.norm_q.view(),
                            self.scratch.norm_q.view(),
                            self.scratch.norm_q.view(),
                        )
                    }
                    AttnNormWeights::Split { q, k, v } => {
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.attn_in.view().slice(0, row_bytes)?,
                            q.view(),
                            self.scratch.norm_q.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch.attn_in.view().slice(row_bytes, row_bytes)?,
                            k.view(),
                            self.scratch.norm_k.view(),
                        )?;
                        ops::rmsnorm_bf16(
                            backend,
                            nargs,
                            self.scratch
                                .attn_in
                                .view()
                                .slice(2 * row_bytes, row_bytes)?,
                            v.view(),
                            self.scratch.norm_v.view(),
                        )?;
                        (
                            self.scratch.norm_q.view(),
                            self.scratch.norm_k.view(),
                            self.scratch.norm_v.view(),
                        )
                    }
                }
            };

            // Fused Q, K, V, R, Gate, Sink mega-projection (1 launch instead of 6)
            let q_dim = t.num_attention_heads * t.head_dim;
            let kv_dim = t.num_key_value_heads * t.head_dim;
            let r_dim = t.num_attention_heads * t.d_rel;
            ops::qkvr_gate_sink_bf16(
                backend,
                ops::QkvrGateSinkArgs {
                    rows,
                    in_dim: h,
                    q_dim,
                    kv_dim,
                    r_dim,
                    n_head: t.num_attention_heads,
                },
                &ops::QkvrGateSinkViews {
                    x_q: q_in_view,
                    x_k: k_in_view,
                    x_v: v_in_view,
                    wq: lw.wq.view(),
                    wk: lw.wk.view(),
                    wv: lw.wv.view(),
                    wr: lw.wr.view(),
                    wgate: lw.gate_attn.view(),
                    wsink: lw.sink_proj.view(),
                    out_q: self.scratch.q.view(),
                    out_k: self.scratch.k.view(),
                    out_v: self.scratch.v.view(),
                    out_r: self.scratch.r.view(),
                    out_gate: self.scratch.gate.view(),
                    out_sink: self.scratch.sinks.view(),
                },
            )?;

            // K & V short convolutions
            match &lw.kv_conv {
                KvConvWeights::Static { k, v } => {
                    let sargs = SconvStaticArgs {
                        rows,
                        channels: kv_dim,
                        kernel_size: t.sconv_kernel_size,
                        scale: sconv_static_scale,
                    };
                    ops::short_conv_static(
                        backend,
                        sargs,
                        self.scratch.k.view(),
                        k.view(),
                        ls.k_conv_state.view(),
                        self.scratch.k_conv.view(),
                    )?;
                    ops::short_conv_static(
                        backend,
                        sargs,
                        self.scratch.v.view(),
                        v.view(),
                        ls.v_conv_state.view(),
                        self.scratch.v_conv.view(),
                    )?;
                }
                KvConvWeights::Dynamic {
                    k_static,
                    k_proj,
                    v_static,
                    v_proj,
                } => {
                    let dyn_out = t.num_key_value_heads * t.sconv_kernel_size;
                    ops::linear_bf16(
                        backend,
                        LinearArgs {
                            rows,
                            in_dim: h,
                            out_dim: dyn_out,
                        },
                        k_in_view,
                        k_proj.view(),
                        self.scratch.k_dyn_w.view(),
                    )?;
                    ops::linear_bf16(
                        backend,
                        LinearArgs {
                            rows,
                            in_dim: h,
                            out_dim: dyn_out,
                        },
                        v_in_view,
                        v_proj.view(),
                        self.scratch.v_dyn_w.view(),
                    )?;
                    let dargs = SconvDynamicArgs {
                        rows,
                        kv_heads: t.num_key_value_heads,
                        head_dim: t.head_dim,
                        kernel_size: t.sconv_kernel_size,
                        scale: sconv_dyn_scale,
                    };
                    ops::short_conv_dynamic(
                        backend,
                        dargs,
                        self.scratch.k.view(),
                        self.scratch.k_dyn_w.view(),
                        k_static.view(),
                        ls.k_conv_state.view(),
                        self.scratch.k_conv.view(),
                    )?;
                    ops::short_conv_dynamic(
                        backend,
                        dargs,
                        self.scratch.v.view(),
                        self.scratch.v_dyn_w.view(),
                        v_static.view(),
                        ls.v_conv_state.view(),
                        self.scratch.v_conv.view(),
                    )?;
                }
            }

            // QKVR prep (L2 clip, V norm, log scaling) + F16 KV cache store
            ops::qkvr_prep_store(
                backend,
                QkvrPrepArgs {
                    rows,
                    n_head: t.num_attention_heads,
                    n_head_kv: t.num_key_value_heads,
                    head_dim: t.head_dim,
                    d_rel: t.d_rel,
                    qk_clip: t.qk_l2_clip_threshold,
                    rel_clip: t.rel_l2_clip_threshold,
                    eps: t.rms_norm_eps,
                    is_full: u32::from(!ll.is_local),
                    log_n_floor: t.log_scaling_n_floor,
                    log_alpha: t.log_scaling_alpha,
                    pos0,
                    kv_cap: ls.kv_cap,
                    is_swa: u32::from(ll.is_local),
                },
                &QkvrPrepViews {
                    q: self.scratch.q.view(),
                    k: self.scratch.k_conv.view(),
                    v: self.scratch.v_conv.view(),
                    r: self.scratch.r.view(),
                    q_scale: lw.q_l2_scale.view(),
                    r_scale: lw.rel_l2_scale.view(),
                    k_cache: ls.k_cache.view(),
                    v_cache: ls.v_cache.view(),
                },
            )?;

            // Relative bias: R [rows * n_head, d_rel] @ rel_logits_proj_t^T [rel_extent, d_rel] -> [rows * n_head, rel_extent]
            ops::rel_bias_bf16(
                backend,
                LinearArgs {
                    rows: rows * t.num_attention_heads,
                    in_dim: t.d_rel,
                    out_dim: t.rel_extent,
                },
                self.scratch.r.view(),
                lw.rel_logits_proj_t.view(),
                self.scratch.rel_bias.view(),
            )?;

            // Relative-bias + per-query sink + output-gated attention
            // Use 16-way split-K Flash-Decoding for long-context full-attention decode.
            let attn_views = RelAttnViews {
                q: self.scratch.q.view(),
                rel_bias: self.scratch.rel_bias.view(),
                sinks: self.scratch.sinks.view(),
                gate: self.scratch.gate.view(),
                k_cache: ls.k_cache.view(),
                v_cache: ls.v_cache.view(),
                out: self.scratch.attn_out.view(),
            };
            if rows == 1 && !ll.is_local && pos0 >= 256 {
                ops::rel_attn_splitk(
                    backend,
                    ops::RelAttnSplitKArgs {
                        n_head: t.num_attention_heads,
                        n_head_kv: t.num_key_value_heads,
                        head_dim: t.head_dim,
                        rel_extent: t.rel_extent,
                        pos: pos0,
                        n_splits: 16,
                        scale: attn_scale,
                    },
                    &attn_views,
                    self.scratch.split_ml.view(),
                    self.scratch.split_out.view(),
                )?;
            } else {
                ops::rel_attn(
                    backend,
                    RelAttnArgs {
                        rows,
                        n_head: t.num_attention_heads,
                        n_head_kv: t.num_key_value_heads,
                        head_dim: t.head_dim,
                        rel_extent: t.rel_extent,
                        pos0,
                        kv_cap: ls.kv_cap,
                        window_size: t.sliding_window_size,
                        is_swa: u32::from(ll.is_local),
                        scale: attn_scale,
                    },
                    &attn_views,
                )?;
            }

            // Output projection + post-attention short conv
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows,
                    in_dim: q_dim,
                    out_dim: h,
                },
                self.scratch.attn_out.view(),
                lw.wo.view(),
                self.scratch.wo_out.view(),
            )?;
            ops::short_conv_static(
                backend,
                SconvStaticArgs {
                    rows,
                    channels: h,
                    kernel_size: t.sconv_kernel_size,
                    scale: sconv_static_scale,
                },
                self.scratch.wo_out.view(),
                lw.attn_sconv.view(),
                ls.attn_conv_state.view(),
                self.scratch.attn_sconv_out.view(),
            )?;

            // Block boundary check before MLP
            let is_boundary = ll.physical_idx % t.n_layers_per_attn_res_block == 0;
            if is_boundary {
                backend.copy_d2d(
                    self.scratch
                        .blocks
                        .view()
                        .slice(u64::from(n_blocks) * row_bytes, row_bytes)?,
                    self.scratch.partial.view().slice(0, row_bytes)?,
                )?;
                n_blocks += 1;
                backend.copy_d2d(
                    self.scratch.partial.view().slice(0, row_bytes)?,
                    self.scratch.attn_sconv_out.view().slice(0, row_bytes)?,
                )?;
            } else {
                ops::add2_f32(
                    backend,
                    rows * h,
                    self.scratch.partial.view(),
                    self.scratch.attn_sconv_out.view(),
                    self.scratch.partial.view(),
                )?;
            }

            // MLP input via block attention residual
            ops::attn_res(
                backend,
                AttnResArgs {
                    rows,
                    hidden: h,
                    n_blocks,
                    n_queries: 1,
                    eps: t.rms_norm_eps,
                    has_addend: 0,
                },
                self.scratch.blocks.view(),
                self.scratch.partial.view(),
                self.scratch.partial.view(),
                lw.mlp_res_query.view(),
                self.scratch.mlp_in.view(),
            )?;

            let mlp_pre_norm = if has_messages {
                let ctrl = lw.controllers.as_ref().expect("core layer has controllers");
                let lc = self.config.text.loop_config.expect("loop_config");
                let msg_mlp = self
                    .scratch
                    .msg_attn_mlp
                    .view()
                    .slice(row_bytes, row_bytes)?;
                ops::loop_control(
                    backend,
                    LoopCtrlArgs {
                        rows,
                        hidden: h,
                        rank: lc.gate_rank,
                        eps: t.rms_norm_eps,
                    },
                    &LoopCtrlViews {
                        cur: self.scratch.mlp_in.view(),
                        msg: msg_mlp,
                        down_w: ctrl.mlp.down.view(),
                        gain_w: ctrl.mlp.gain.view(),
                        forget_w: ctrl.mlp.forget.view(),
                        z_scratch: self.scratch.ctrl_z.view(),
                        out: self.scratch.mlp_ctrl.view(),
                    },
                )?;
                self.scratch.mlp_ctrl.view()
            } else {
                self.scratch.mlp_in.view()
            };

            ops::rmsnorm_bf16(
                backend,
                NormArgs {
                    rows,
                    hidden: h,
                    eps: t.rms_norm_eps,
                },
                mlp_pre_norm,
                lw.mlp_norm.view(),
                self.scratch.mlp_norm.view(),
            )?;

            // Router + Top-K
            ops::linear_bf16(
                backend,
                LinearArgs {
                    rows,
                    in_dim: h,
                    out_dim: t.n_routed_experts,
                },
                self.scratch.mlp_norm.view(),
                lw.router_weight.view(),
                self.scratch.router_logits.view(),
            )?;
            ops::route_topk(
                backend,
                RouteArgs {
                    rows,
                    n_experts: t.n_routed_experts,
                    top_k: t.num_experts_per_tok,
                    route_scale: t.route_scale,
                },
                self.scratch.router_logits.view(),
                lw.router_bias.view(),
                self.scratch.expert_ids.view(),
                self.scratch.expert_weights.view(),
            )?;

            // Routed MoE (NVFP4 or MXFP8): use token-grouped 8-row batched path
            // during prefill (`rows >= 4`) and low-latency direct path during decode.
            let w13_args = MoeW13Args {
                rows,
                in_dim: h,
                inter_dim: t.intermediate_size,
                top_k: t.num_experts_per_tok,
                gate_cap: t.mlp_gate_softcap,
                up_cap: t.mlp_up_softcap,
            };
            let w2_args = MoeW2Args {
                rows,
                inter_dim: t.intermediate_size,
                out_dim: h,
                top_k: t.num_experts_per_tok,
            };
            if rows >= 32 {
                ops::moe_bucket(
                    backend,
                    ops::MoeBucketArgs {
                        rows,
                        top_k: t.num_experts_per_tok,
                        n_experts: t.n_routed_experts,
                    },
                    self.scratch.expert_ids.view(),
                    self.scratch.expert_counts.view(),
                    self.scratch.expert_cols.view(),
                )?;
                let grp = ops::MoeGroupViews {
                    counts: self.scratch.expert_counts.view(),
                    cols: self.scratch.expert_cols.view(),
                    n_experts: t.n_routed_experts,
                };
                match &lw.experts {
                    ExpertWeights::Nvfp4(nv) => {
                        ops::moe_nvfp4_w13_grouped(
                            backend,
                            w13_args,
                            self.scratch.mlp_norm.view(),
                            &Nvfp4Views {
                                weight: nv.w13.view(),
                                scale: nv.w13_scale.view(),
                                amax: nv.w13_amax.view(),
                                denom: nv.w13_denom.view(),
                            },
                            self.scratch.expert_weights.view(),
                            &grp,
                            self.scratch.moe_act.view(),
                        )?;
                        ops::moe_nvfp4_w2_grouped(
                            backend,
                            w2_args,
                            self.scratch.moe_act.view(),
                            &Nvfp4Views {
                                weight: nv.w2.view(),
                                scale: nv.w2_scale.view(),
                                amax: nv.w2_amax.view(),
                                denom: nv.w2_denom.view(),
                            },
                            &grp,
                            self.scratch.moe_w2_slots.view(),
                        )?;
                    }
                    ExpertWeights::Mxfp8(mx) => {
                        ops::moe_mxfp8_w13_grouped(
                            backend,
                            w13_args,
                            self.scratch.mlp_norm.view(),
                            &Mxfp8Views {
                                weight: mx.w13.view(),
                                scale: mx.w13_scale.view(),
                            },
                            self.scratch.expert_weights.view(),
                            &grp,
                            self.scratch.moe_act.view(),
                        )?;
                        ops::moe_mxfp8_w2_grouped(
                            backend,
                            w2_args,
                            self.scratch.moe_act.view(),
                            &Mxfp8Views {
                                weight: mx.w2.view(),
                                scale: mx.w2_scale.view(),
                            },
                            &grp,
                            self.scratch.moe_w2_slots.view(),
                        )?;
                    }
                }
                ops::moe_reduce_topk(
                    backend,
                    w2_args,
                    self.scratch.moe_w2_slots.view(),
                    self.scratch.moe_out.view(),
                )?;
            } else {
                match &lw.experts {
                    ExpertWeights::Nvfp4(nv) => {
                        ops::moe_nvfp4_w13_swiglu(
                            backend,
                            w13_args,
                            self.scratch.mlp_norm.view(),
                            &Nvfp4Views {
                                weight: nv.w13.view(),
                                scale: nv.w13_scale.view(),
                                amax: nv.w13_amax.view(),
                                denom: nv.w13_denom.view(),
                            },
                            self.scratch.expert_ids.view(),
                            self.scratch.expert_weights.view(),
                            self.scratch.moe_act.view(),
                        )?;
                        ops::moe_nvfp4_w2_accum(
                            backend,
                            w2_args,
                            self.scratch.moe_act.view(),
                            &Nvfp4Views {
                                weight: nv.w2.view(),
                                scale: nv.w2_scale.view(),
                                amax: nv.w2_amax.view(),
                                denom: nv.w2_denom.view(),
                            },
                            self.scratch.expert_ids.view(),
                            self.scratch.moe_out.view(),
                        )?;
                    }
                    ExpertWeights::Mxfp8(mx) => {
                        ops::moe_mxfp8_w13_swiglu(
                            backend,
                            w13_args,
                            self.scratch.mlp_norm.view(),
                            &Mxfp8Views {
                                weight: mx.w13.view(),
                                scale: mx.w13_scale.view(),
                            },
                            self.scratch.expert_ids.view(),
                            self.scratch.expert_weights.view(),
                            self.scratch.moe_act.view(),
                        )?;
                        ops::moe_mxfp8_w2_accum(
                            backend,
                            w2_args,
                            self.scratch.moe_act.view(),
                            &Mxfp8Views {
                                weight: mx.w2.view(),
                                scale: mx.w2_scale.view(),
                            },
                            self.scratch.expert_ids.view(),
                            self.scratch.moe_out.view(),
                        )?;
                    }
                }
            }

            // Post-MLP short conv + residual add into partial
            ops::short_conv_static(
                backend,
                SconvStaticArgs {
                    rows,
                    channels: h,
                    kernel_size: t.sconv_kernel_size,
                    scale: sconv_static_scale,
                },
                self.scratch.moe_out.view(),
                lw.mlp_sconv.view(),
                ls.mlp_conv_state.view(),
                self.scratch.mlp_sconv_out.view(),
            )?;
            ops::add2_f32(
                backend,
                rows * h,
                self.scratch.partial.view(),
                self.scratch.mlp_sconv_out.view(),
                self.scratch.partial.view(),
            )?;
        }

        // Final block attention residual + final RMSNorm
        ops::attn_res(
            backend,
            AttnResArgs {
                rows,
                hidden: h,
                n_blocks,
                n_queries: 1,
                eps: t.rms_norm_eps,
                has_addend: 0,
            },
            self.scratch.blocks.view(),
            self.scratch.partial.view(),
            self.scratch.partial.view(),
            self.weights.output_res_query.view(),
            self.scratch.final_hidden.view(),
        )?;
        ops::rmsnorm_bf16(
            backend,
            NormArgs {
                rows,
                hidden: h,
                eps: t.rms_norm_eps,
            },
            self.scratch.final_hidden.view(),
            self.weights.final_norm.view(),
            self.scratch.final_hidden.view(),
        )?;

        // Capture final tap (logical_layers == 48) if requested
        let final_id = t.num_logical_layers();
        if let Some(taps_buf) = &self.scratch.mtp_taps
            && let Some(tap_pos) = self.tap_layers.iter().position(|&id| id == final_id)
        {
            backend.copy_d2d(
                taps_buf
                    .view()
                    .slice(tap_pos as u64 * row_bytes, row_bytes)?,
                self.scratch.final_hidden.view().slice(0, row_bytes)?,
            )?;
        }

        if compute_logits {
            if all_rows_logits {
                ops::lm_head_softcap(
                    backend,
                    LmHeadArgs {
                        rows,
                        hidden: h,
                        vocab: t.unpadded_vocab_size,
                        softcap: t.final_logit_softcapping,
                    },
                    self.scratch.final_hidden.view(),
                    self.weights.unembed.view(),
                    self.scratch.logits.view(),
                )?;
            } else {
                let last_row_off = u64::from(rows - 1) * u64::from(h) * 4;
                let one_row_bytes = u64::from(h) * 4;
                ops::lm_head_softcap(
                    backend,
                    LmHeadArgs {
                        rows: 1,
                        hidden: h,
                        vocab: t.unpadded_vocab_size,
                        softcap: t.final_logit_softcapping,
                    },
                    self.scratch
                        .final_hidden
                        .view()
                        .slice(last_row_off, one_row_bytes)?,
                    self.weights.unembed.view(),
                    self.scratch.logits.view(),
                )?;
            }
        }

        sess.pos += rows;
        if !all_rows_logits && let (Some(mtp), Some(taps)) = (&mut self.mtp, &self.scratch.mtp_taps)
        {
            mtp.absorb_taps(session, taps.view(), rows, rows, pos0)?;
        }
        Ok(())
    }
}

impl Model for Inkling2Model {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        let t = &self.config.text;
        let kv_dim = u64::from(t.num_key_value_heads) * u64::from(t.head_dim);
        let lags = u64::from(t.sconv_kernel_size.saturating_sub(1));
        let h = u64::from(t.hidden_size);
        let backend = &self.backend;

        let mut layers = Vec::with_capacity(self.schedule.len());
        for &ll in &self.schedule {
            let kv_cap = if ll.is_local {
                t.sliding_window_size
            } else {
                self.info.context_capacity
            };
            let kv_bytes = u64::from(kv_cap) * kv_dim * 2;
            let k_cache = alloc(backend, kv_bytes)?;
            let v_cache = alloc(backend, kv_bytes)?;
            let k_conv_state = alloc(backend, (lags * kv_dim * 4).max(4))?;
            let v_conv_state = alloc(backend, (lags * kv_dim * 4).max(4))?;
            let attn_conv_state = alloc(backend, (lags * h * 4).max(4))?;
            let mlp_conv_state = alloc(backend, (lags * h * 4).max(4))?;
            backend.fill_zero(k_conv_state.view())?;
            backend.fill_zero(v_conv_state.view())?;
            backend.fill_zero(attn_conv_state.view())?;
            backend.fill_zero(mlp_conv_state.view())?;
            layers.push(LayerState {
                k_cache,
                v_cache,
                kv_cap,
                k_conv_state,
                v_conv_state,
                attn_conv_state,
                mlp_conv_state,
            });
        }

        let id =
            SessionId::next(&mut self.next_session_id).ok_or_else(|| CoreError::InvalidValue {
                key: "session_id",
                value: "session id space exhausted".to_owned(),
            })?;
        self.sessions.insert(id, SessionState { pos: 0, layers });
        if let Some(mtp) = &mut self.mtp {
            mtp.open_session(id, self.info.context_capacity)?;
        }
        Ok(id)
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        if let Some(mtp) = &mut self.mtp {
            mtp.close_session(session);
        }
        self.sessions
            .remove(&session)
            .map(|_| ())
            .ok_or(CoreError::UnknownSession(session))
    }

    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    fn spec_mut(&mut self) -> Option<&mut dyn Speculating> {
        if self.mtp.is_some() { Some(self) } else { None }
    }

    fn suppress_tokens(&mut self, _ids: &[TokenId]) -> CoreResult<()> {
        Ok(())
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        const MAX_CHUNK: usize = 256;
        for chunk in tokens.chunks(MAX_CHUNK) {
            self.forward_tokens(session, chunk, false)?;
        }
        Ok(())
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        const MAX_CHUNK: usize = 256;
        if tokens.len() > MAX_CHUNK {
            let split = tokens.len() - MAX_CHUNK;
            for chunk in tokens[..split].chunks(MAX_CHUNK) {
                self.forward_tokens(session, chunk, false)?;
            }
            self.forward_tokens(session, &tokens[split..], true)?;
        } else {
            self.forward_tokens(session, tokens, true)?;
        }
        let row_bytes = self.info.logits_row_bytes();
        Logits::new(
            self.scratch.logits.view().slice(0, row_bytes)?,
            1,
            self.info.vocab,
        )
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        if steps.is_empty() {
            return Logits::new(self.scratch.logits.view().slice(0, 0)?, 0, self.info.vocab);
        }
        let row_bytes = self.info.logits_row_bytes();
        if steps.len() == 1 {
            self.forward_tokens(steps[0].session, &[steps[0].token], true)?;
            return Logits::new(
                self.scratch.logits.view().slice(0, row_bytes)?,
                1,
                self.info.vocab,
            );
        }
        let n = u32::try_from(steps.len()).map_err(|_| CoreError::InvalidValue {
            key: "steps",
            value: "batch exceeds u32".to_owned(),
        })?;
        self.ensure_scratch(n)?;
        let mut collected = alloc(&self.backend, u64::from(n) * row_bytes)?;
        for (i, step) in steps.iter().enumerate() {
            self.forward_tokens(step.session, &[step.token], true)?;
            self.backend.copy_d2d(
                collected.view().slice(i as u64 * row_bytes, row_bytes)?,
                self.scratch.logits.view().slice(0, row_bytes)?,
            )?;
        }
        // Copy collected rows back into `self.scratch.logits` so the returned Logits borrows self.
        self.backend.copy_d2d(
            self.scratch
                .logits
                .view()
                .slice(0, u64::from(n) * row_bytes)?,
            collected.view(),
        )?;
        let _ = &mut collected;
        Logits::new(
            self.scratch
                .logits
                .view()
                .slice(0, u64::from(n) * row_bytes)?,
            n,
            self.info.vocab,
        )
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.sessions.clear();
        self.backend.sync()
    }
}

impl MediaModel for Inkling2Model {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        if media.is_empty() {
            return self.prefill_last(session, tokens);
        }
        let Some(img_pad) = media.pads().image() else {
            return Err(CoreError::Unsupported {
                what: "media prefill without image placeholder token",
                backend: "inkling2",
            });
        };
        let rows = u32::try_from(tokens.len()).map_err(|_| CoreError::InvalidValue {
            key: "tokens",
            value: "prompt exceeds u32".to_owned(),
        })?;
        let pad_positions: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &tok)| (tok == img_pad).then_some(i))
            .collect();
        if pad_positions.len() as u64 != u64::from(media.row_count()) {
            return Err(CoreError::ShapeMismatch {
                field: "image placeholder count",
                expected: u64::from(media.row_count()),
                got: pad_positions.len() as u64,
            });
        }
        self.ensure_scratch(rows)?;
        let h = self.config.text.hidden_size;
        let row_bytes = u64::from(h) * 4;
        let raw_ids: Vec<u32> = tokens.iter().map(|tok| tok.get()).collect();
        self.backend.copy_h2d(
            self.scratch.tokens.view().slice(0, u64::from(rows) * 4)?,
            cast_slice(&raw_ids),
        )?;
        ops::embed_rmsnorm_bf16(
            self.backend.as_ref(),
            NormArgs {
                rows,
                hidden: h,
                eps: self.config.text.rms_norm_eps,
            },
            self.scratch.tokens.view(),
            self.weights.embed.view(),
            self.weights.embed_norm.view(),
            self.scratch.partial.view(),
        )?;
        // Splice encoded image rows into placeholder positions
        for (src_idx, &tok_pos) in pad_positions.iter().enumerate() {
            let src = media.rows().slice(src_idx as u64 * row_bytes, row_bytes)?;
            let dst = self
                .scratch
                .partial
                .view()
                .slice(tok_pos as u64 * row_bytes, row_bytes)?;
            self.backend.copy_d2d(dst, src)?;
        }
        // Run the 48-layer forward pass on the spliced embeddings
        self.forward_already_embedded(session, rows, true, false)?;
        let logits_bytes = self.info.logits_row_bytes();
        Logits::new(
            self.scratch.logits.view().slice(0, logits_bytes)?,
            1,
            self.info.vocab,
        )
    }
}

impl Speculating for Inkling2Model {
    fn speculate(&mut self, session: SessionId, first: TokenId) -> CoreResult<Speculated> {
        let Some(mtp) = &self.mtp else {
            return Err(CoreError::Unsupported {
                what: "speculation without loaded mtp.safetensors",
                backend: "inkling2",
            });
        };
        let drafts = mtp.draft_block(
            session,
            first,
            self.weights.embed.view(),
            self.weights.embed_norm.view(),
            self.weights.unembed.view(),
        )?;
        let mut block = Vec::with_capacity(1 + drafts.len());
        block.push(first);
        block.extend_from_slice(&drafts);
        let block_len = block.len() as u32;

        let pos0 = self
            .sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?
            .pos;

        // Run target verify pass over all `block_len` tokens and compute logits for every row.
        self.ensure_scratch(block_len)?;
        let raw_ids: Vec<u32> = block.iter().map(|tok| tok.get()).collect();
        self.backend.copy_h2d(
            self.scratch
                .tokens
                .view()
                .slice(0, u64::from(block_len) * 4)?,
            cast_slice(&raw_ids),
        )?;
        ops::embed_rmsnorm_bf16(
            self.backend.as_ref(),
            NormArgs {
                rows: block_len,
                hidden: self.config.text.hidden_size,
                eps: self.config.text.rms_norm_eps,
            },
            self.scratch.tokens.view(),
            self.weights.embed.view(),
            self.weights.embed_norm.view(),
            self.scratch.partial.view(),
        )?;
        self.forward_already_embedded(session, block_len, true, true)?;

        // Greedy-verify each row's logits against `drafts`
        let vocab = self.info.vocab as usize;
        let total_bytes = (block_len as usize) * vocab * 4;
        let mut logits_bytes = vec![0u8; total_bytes];
        self.backend.sync()?;
        self.backend.copy_d2h(
            self.scratch.logits.view().slice(0, total_bytes as u64)?,
            &mut logits_bytes,
        )?;
        let logits_all: &[f32] = cast_slice(&logits_bytes);

        let mut accepted = 0u32;
        let mut emitted = Vec::with_capacity(block.len());
        for row in 0..block.len() {
            let row_slice = &logits_all[row * vocab..(row + 1) * vocab];
            let mut best_id = 0u32;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in row_slice.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_id = i as u32;
                }
            }
            let target_tok = TokenId::new(best_id);
            emitted.push(target_tok);
            if row < drafts.len()
                && target_tok == drafts[row]
                && !self.info.stop_tokens.contains(target_tok)
            {
                accepted += 1;
            } else {
                break;
            }
        }

        // Rewind session position to committed prefix `pos0 + 1 + accepted`
        if let Some(sess) = self.sessions.get_mut(&session) {
            sess.pos = pos0 + 1 + accepted;
        }
        if let (Some(mtp), Some(taps)) = (&mut self.mtp, &self.scratch.mtp_taps) {
            mtp.absorb_taps(session, taps.view(), block_len, 1 + accepted, pos0)?;
        }

        Ok(Speculated {
            produced: emitted,
            drafted: drafts.len() as u32,
            accepted,
        })
    }
}

impl shiso_engine::Checkpointed for Inkling2Model {
    fn kv_enable(&mut self, _config: &shiso_engine::CkptConfig) -> CoreResult<()> {
        Err(CoreError::Unsupported {
            what: "disk KV checkpointing",
            backend: "inkling2",
        })
    }

    fn kv_checkpoint_enabled(&self) -> bool {
        false
    }

    /// `None` means cold start (no stored prefix matched; not a refusal).
    fn kv_resume(
        &mut self,
        session: SessionId,
        _prompt: &[TokenId],
    ) -> CoreResult<Option<shiso_engine::ResumeInfo<'_>>> {
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        Ok(None)
    }

    fn kv_save(
        &mut self,
        session: SessionId,
        _reason: shiso_engine::Reason,
    ) -> CoreResult<shiso_engine::SaveOutcome> {
        if !self.sessions.contains_key(&session) {
            return Err(CoreError::UnknownSession(session));
        }
        Ok(shiso_engine::SaveOutcome::Disabled)
    }

    fn kv_shutdown_sweep(&mut self) -> usize {
        0
    }
}
