//! Safetensors weight loader for Inkling-2 checkpoints.
//!
//! Opens `model-00001-of-00001.safetensors` (or sharded checkpoints via
//! `model.safetensors.index.json`) using `shiso_modelfile::st_header::Shard`,
//! validates every tensor's dtype and shape, and uploads weights directly to
//! device memory without dequantizing NVFP4 or MXFP8 routed experts.

#![expect(
    clippy::doc_markdown,
    clippy::too_many_lines,
    reason = "checkpoint weight loader maps tensor names directly to device buffers"
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use shiso_core::{Backend, CoreError, CoreResult, DeviceBuffer, alloc};
use shiso_modelfile::st_header::{RawTensor, Shard, StDType};

use super::config::Inkling2Config;

/// Multi-shard safetensors directory reader.
pub struct SafetensorsDir {
    shards: Vec<Shard>,
    index: HashMap<String, (usize, StDType, Vec<u64>, [u64; 2])>,
}

impl std::fmt::Debug for SafetensorsDir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetensorsDir")
            .field("shards", &self.shards.len())
            .field("tensors", &self.index.len())
            .finish()
    }
}

impl SafetensorsDir {
    /// Opens a single-shard or multi-shard safetensors checkpoint directory.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] or [`CoreError::InvalidValue`] if shard files are missing or invalid.
    pub fn open_model(dir: &Path) -> CoreResult<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let shard_paths: Vec<PathBuf> = if index_path.is_file() {
            #[derive(Deserialize)]
            struct IndexDoc {
                weight_map: HashMap<String, String>,
            }
            let raw = std::fs::read_to_string(&index_path)?;
            let doc: IndexDoc =
                serde_json::from_str(&raw).map_err(|err| CoreError::InvalidValue {
                    key: "model.safetensors.index.json",
                    value: err.to_string(),
                })?;
            let mut unique: Vec<String> = doc.weight_map.into_values().collect();
            unique.sort();
            unique.dedup();
            unique.into_iter().map(|name| dir.join(name)).collect()
        } else {
            let single = dir.join("model.safetensors");
            if single.is_file() {
                vec![single]
            } else {
                vec![dir.join("model-00001-of-00001.safetensors")]
            }
        };
        Self::open_files(&shard_paths)
    }

    /// Opens a single safetensors file (such as `mtp.safetensors`).
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] or [`CoreError::InvalidValue`] on unreadable or malformed files.
    pub fn open_file(path: &Path) -> CoreResult<Self> {
        Self::open_files(&[path.to_path_buf()])
    }

    fn open_files(paths: &[PathBuf]) -> CoreResult<Self> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for (shard_idx, path) in paths.iter().enumerate() {
            let (shard, raw_tensors): (Shard, Vec<(String, RawTensor)>) =
                Shard::open_untagged(path).map_err(|err| CoreError::InvalidValue {
                    key: "safetensors",
                    value: format!("{}: {err}", path.display()),
                })?;
            for (name, rt) in raw_tensors {
                let dtype = rt
                    .parsed_dtype(&name)
                    .map_err(|err| CoreError::InvalidValue {
                        key: "safetensors_dtype",
                        value: err.to_string(),
                    })?;
                index.insert(name, (shard_idx, dtype, rt.shape, rt.data_offsets));
            }
            shards.push(shard);
        }
        Ok(Self { shards, index })
    }

    /// Returns `true` if `name` is present in the opened shards.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Returns the raw payload bytes, dtype, and shape of tensor `name`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] if the tensor is missing or has an unexpected dtype.
    pub fn tensor_bytes(&self, name: &str, expected_dtype: StDType) -> CoreResult<(&[u8], &[u64])> {
        let Some((shard_idx, dtype, shape, offsets)) = self.index.get(name) else {
            return Err(CoreError::InvalidValue {
                key: "missing_tensor",
                value: name.to_owned(),
            });
        };
        if *dtype != expected_dtype {
            return Err(CoreError::InvalidValue {
                key: "tensor_dtype",
                value: format!("{name}: expected {expected_dtype:?}, got {dtype:?}"),
            });
        }
        let bytes = self.shards[*shard_idx]
            .payload(offsets[0], offsets[1])
            .map_err(|err| CoreError::InvalidValue {
                key: "tensor_payload",
                value: format!("{name}: {err}"),
            })?;
        Ok((bytes, shape.as_slice()))
    }

    /// Uploads tensor `name` directly to a new device buffer.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] or [`CoreError::Backend`] on mismatch or device failure.
    pub fn upload(
        &self,
        backend: &Arc<dyn Backend>,
        name: &str,
        expected_dtype: StDType,
    ) -> CoreResult<DeviceBuffer> {
        let (bytes, _shape) = self.tensor_bytes(name, expected_dtype)?;
        let buf = alloc(backend, bytes.len().max(4) as u64)?;
        backend.copy_h2d(buf.view(), bytes)?;
        Ok(buf)
    }
}

/// NVFP4 routed expert buffers for one physical layer.
#[derive(Debug)]
pub struct Nvfp4ExpertWeights {
    /// `U8 [E, 2*inter, H/2]`
    pub w13: DeviceBuffer,
    /// `F8_E4M3 [E, 2*inter, H/16]`
    pub w13_scale: DeviceBuffer,
    /// `F32 [E]`
    pub w13_amax: DeviceBuffer,
    /// `F32 [1]`
    pub w13_denom: DeviceBuffer,
    /// `U8 [E, H, inter/2]`
    pub w2: DeviceBuffer,
    /// `F8_E4M3 [E, H, inter/16]`
    pub w2_scale: DeviceBuffer,
    /// `F32 [E]`
    pub w2_amax: DeviceBuffer,
    /// `F32 [1]`
    pub w2_denom: DeviceBuffer,
}

/// MXFP8 (L20–23) routed expert buffers.
#[derive(Debug)]
pub struct Mxfp8ExpertWeights {
    /// `F8_E4M3 [E, 2*inter, H]`
    pub w13: DeviceBuffer,
    /// `U8 [E, 2*inter, H/32]`
    pub w13_scale: DeviceBuffer,
    /// `F8_E4M3 [E, H, inter]`
    pub w2: DeviceBuffer,
    /// `U8 [E, H, inter/32]`
    pub w2_scale: DeviceBuffer,
}

/// Routed MoE expert weights for one physical layer.
#[derive(Debug)]
pub enum ExpertWeights {
    /// NVFP4 (L0–19): packed E2M1 nibbles + per-16 E4M3 scales + per-expert amax + scalar denom.
    Nvfp4(Box<Nvfp4ExpertWeights>),
    /// MXFP8 (L20–23): E4M3 weights + per-32 UE8M0 power-of-two scales.
    Mxfp8(Box<Mxfp8ExpertWeights>),
}

/// Input normalization weights for one physical layer's attention block.
#[derive(Debug)]
pub enum AttnNormWeights {
    /// Local sliding-window layer: single shared `attn_norm.weight`.
    Shared(DeviceBuffer),
    /// Global full-attention layer: three separate norms for Q, K, and V streams.
    Split {
        /// Q stream norm.
        q: DeviceBuffer,
        /// K stream norm.
        k: DeviceBuffer,
        /// V stream norm.
        v: DeviceBuffer,
    },
}

/// K/V short convolution weights for one physical layer.
#[derive(Debug)]
pub enum KvConvWeights {
    /// Static depthwise conv (`[kv_dim, 1, k]`) used in local SWA layers.
    Static {
        /// Key conv weights (`BF16 [kv_dim, k]`).
        k: DeviceBuffer,
        /// Value conv weights (`BF16 [kv_dim, k]`).
        v: DeviceBuffer,
    },
    /// Dynamic input-conditioned conv used in full-attention layers.
    Dynamic {
        /// Static K taps (`BF16 [k, kv_dim]`).
        k_static: DeviceBuffer,
        /// Dynamic K tap projection (`BF16 [kv_heads * k, H]`).
        k_proj: DeviceBuffer,
        /// Static V taps (`BF16 [k, kv_dim]`).
        v_static: DeviceBuffer,
        /// Dynamic V tap projection (`BF16 [kv_heads * k, H]`).
        v_proj: DeviceBuffer,
    },
}

/// Rank-64 cross-pass message controller weights for one module (`attn`, `k`, `v`, or `mlp`).
#[derive(Debug)]
pub struct ControllerWeights {
    /// `BF16 [rank, 2*H]`
    pub down: DeviceBuffer,
    /// `BF16 [H, rank]`
    pub gain: DeviceBuffer,
    /// `BF16 [H, rank]`
    pub forget: DeviceBuffer,
}

/// Four cross-pass controllers attached to each layer in the looped core (`L6..L17`).
#[derive(Debug)]
pub struct LoopControllers {
    /// Query / attention input controller.
    pub attn: ControllerWeights,
    /// Key input controller.
    pub k: ControllerWeights,
    /// Value input controller.
    pub v: ControllerWeights,
    /// MLP input controller.
    pub mlp: ControllerWeights,
}

/// All device-resident weights for one physical decoder layer.
#[derive(Debug)]
pub struct PhysicalLayerWeights {
    /// Attention input norm(s).
    pub attn_norm: AttnNormWeights,
    /// Query projection (`BF16 [n_head * head_dim, H]`).
    pub wq: DeviceBuffer,
    /// Key projection (`BF16 [n_head_kv * head_dim, H]`).
    pub wk: DeviceBuffer,
    /// Value projection (`BF16 [n_head_kv * head_dim, H]`).
    pub wv: DeviceBuffer,
    /// Relative position projection (`BF16 [n_head * d_rel, H]`).
    pub wr: DeviceBuffer,
    /// Output gate projection (`BF16 [n_head * head_dim, H]`).
    pub gate_attn: DeviceBuffer,
    /// Per-query attention sink projection (`BF16 [n_head, H]`).
    pub sink_proj: DeviceBuffer,
    /// Output projection (`BF16 [H, n_head * head_dim]`).
    pub wo: DeviceBuffer,
    /// Q L2-clip learned scale (`BF16 [n_head, head_dim]`).
    pub q_l2_scale: DeviceBuffer,
    /// Relative L2-clip learned scale (`BF16 [n_head, d_rel]`).
    pub rel_l2_scale: DeviceBuffer,
    /// Relative logits projection transposed to `[rel_extent, d_rel]` (`BF16`) for `linear_bf16`.
    pub rel_logits_proj_t: DeviceBuffer,
    /// K/V short convolution weights.
    pub kv_conv: KvConvWeights,
    /// Post-attention short convolution (`BF16 [H, k]`).
    pub attn_sconv: DeviceBuffer,
    /// MLP input norm (`BF16 [H]`).
    pub mlp_norm: DeviceBuffer,
    /// Router weight (`BF16 [E, H]`).
    pub router_weight: DeviceBuffer,
    /// Router correction bias (`F32 [E]`).
    pub router_bias: DeviceBuffer,
    /// Routed MoE experts (NVFP4 or MXFP8).
    pub experts: ExpertWeights,
    /// Post-MLP short convolution (`BF16 [H, k]`).
    pub mlp_sconv: DeviceBuffer,
    /// Optional block attention residual query before attention (`F32 [1 or 3, H]`), absent on layer 0.
    pub attn_res_query: Option<DeviceBuffer>,
    /// Block attention residual query before MLP (`F32 [H]`).
    pub mlp_res_query: DeviceBuffer,
    /// Optional cross-pass loop controllers (present on `core_start..core_end`).
    pub controllers: Option<LoopControllers>,
}

/// Complete text backbone weights on device.
#[derive(Debug)]
pub struct Inkling2TextWeights {
    /// Token embedding table (`BF16 [vocab_size, H]`).
    pub embed: DeviceBuffer,
    /// Post-embedding RMSNorm (`BF16 [H]`).
    pub embed_norm: DeviceBuffer,
    /// Physical decoder layers (`0..num_hidden_layers`).
    pub layers: Vec<PhysicalLayerWeights>,
    /// Final block attention residual query (`F32 [H]`).
    pub output_res_query: DeviceBuffer,
    /// Final RMSNorm (`BF16 [H]`).
    pub final_norm: DeviceBuffer,
    /// Output projection / LM head (`BF16 [vocab_size, H]`).
    pub unembed: DeviceBuffer,
    /// Cross-pass `(attn, mlp)` queries for looped core (`F32 [2 * core_len, H]`).
    pub cross_pass_queries: Option<DeviceBuffer>,
    /// Cross-pass `(k, v)` queries for looped core (`F32 [2 * core_len, H]`).
    pub cross_pass_kv_queries: Option<DeviceBuffer>,
}

/// Transposes a BF16 `[rows, cols]` matrix on the host to `[cols, rows]` before upload.
fn upload_bf16_transposed(
    backend: &Arc<dyn Backend>,
    dir: &SafetensorsDir,
    name: &str,
) -> CoreResult<DeviceBuffer> {
    let (bytes, shape) = dir.tensor_bytes(name, StDType::Bf16)?;
    if shape.len() != 2 {
        return Err(CoreError::InvalidValue {
            key: "transpose_shape",
            value: format!("{name}: expected 2D tensor, got {shape:?}"),
        });
    }
    let rows = usize::try_from(shape[0]).map_err(|_| CoreError::InvalidValue {
        key: "shape",
        value: name.to_owned(),
    })?;
    let cols = usize::try_from(shape[1]).map_err(|_| CoreError::InvalidValue {
        key: "shape",
        value: name.to_owned(),
    })?;
    let mut transposed = vec![0u8; bytes.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * 2;
            let dst = (c * rows + r) * 2;
            transposed[dst] = bytes[src];
            transposed[dst + 1] = bytes[src + 1];
        }
    }
    let buf = alloc(backend, transposed.len().max(4) as u64)?;
    backend.copy_h2d(buf.view(), &transposed)?;
    Ok(buf)
}

fn load_controller(
    backend: &Arc<dyn Backend>,
    dir: &SafetensorsDir,
    prefix: &str,
) -> CoreResult<ControllerWeights> {
    Ok(ControllerWeights {
        down: dir.upload(backend, &format!("{prefix}.down_weight"), StDType::Bf16)?,
        gain: dir.upload(backend, &format!("{prefix}.gain_weight"), StDType::Bf16)?,
        forget: dir.upload(backend, &format!("{prefix}.forget_weight"), StDType::Bf16)?,
    })
}

/// Loads all text backbone weights onto `backend`.
///
/// # Errors
///
/// Returns [`CoreError`] when required tensors are missing, mistyped, or cannot be allocated.
pub fn load_text_weights(
    backend: &Arc<dyn Backend>,
    dir: &SafetensorsDir,
    cfg: &Inkling2Config,
) -> CoreResult<Inkling2TextWeights> {
    let t = &cfg.text;
    let embed = dir.upload(backend, "model.llm.embed.weight", StDType::Bf16)?;
    let embed_norm = dir.upload(backend, "model.llm.embed_norm.weight", StDType::Bf16)?;
    let final_norm = dir.upload(backend, "model.llm.norm.weight", StDType::Bf16)?;
    let unembed = dir.upload(backend, "model.llm.unembed.weight", StDType::Bf16)?;
    let output_res_query = dir.upload(backend, "model.llm.attn_res.output_query", StDType::F32)?;

    let (cross_pass_queries, cross_pass_kv_queries) = if t.loop_config.is_some() {
        (
            Some(dir.upload(
                backend,
                "model.llm.middle_cycle_cross_pass_queries",
                StDType::F32,
            )?),
            Some(dir.upload(
                backend,
                "model.llm.middle_cycle_cross_pass_kv_queries",
                StDType::F32,
            )?),
        )
    } else {
        (None, None)
    };

    let mut layers = Vec::with_capacity(t.num_hidden_layers as usize);
    for l in 0..t.num_hidden_layers {
        let lp = format!("model.llm.layers.{l}");
        let is_local = t.is_local(l);

        let attn_norm = if is_local {
            AttnNormWeights::Shared(dir.upload(
                backend,
                &format!("{lp}.attn_norm.weight"),
                StDType::Bf16,
            )?)
        } else {
            AttnNormWeights::Split {
                q: dir.upload(backend, &format!("{lp}.q_input_norm.weight"), StDType::Bf16)?,
                k: dir.upload(backend, &format!("{lp}.k_input_norm.weight"), StDType::Bf16)?,
                v: dir.upload(backend, &format!("{lp}.v_input_norm.weight"), StDType::Bf16)?,
            }
        };

        let kv_conv = if is_local {
            KvConvWeights::Static {
                k: dir.upload(backend, &format!("{lp}.attn.k_sconv.weight"), StDType::Bf16)?,
                v: dir.upload(backend, &format!("{lp}.attn.v_sconv.weight"), StDType::Bf16)?,
            }
        } else {
            KvConvWeights::Dynamic {
                k_static: dir.upload(
                    backend,
                    &format!("{lp}.attn.k_sconv.static_w"),
                    StDType::Bf16,
                )?,
                k_proj: dir.upload(
                    backend,
                    &format!("{lp}.attn.k_sconv.weight_proj"),
                    StDType::Bf16,
                )?,
                v_static: dir.upload(
                    backend,
                    &format!("{lp}.attn.v_sconv.static_w"),
                    StDType::Bf16,
                )?,
                v_proj: dir.upload(
                    backend,
                    &format!("{lp}.attn.v_sconv.weight_proj"),
                    StDType::Bf16,
                )?,
            }
        };

        let w13_name = format!("{lp}.mlp.experts.w13_weight");
        let experts = if dir.contains(&format!("{lp}.mlp.experts.w13_weight_amax")) {
            ExpertWeights::Nvfp4(Box::new(Nvfp4ExpertWeights {
                w13: dir.upload(backend, &w13_name, StDType::U8)?,
                w13_scale: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w13_weight_scale"),
                    StDType::F8E4M3,
                )?,
                w13_amax: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w13_weight_amax"),
                    StDType::F32,
                )?,
                w13_denom: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w13_weight_outer_scale_denominator"),
                    StDType::F32,
                )?,
                w2: dir.upload(backend, &format!("{lp}.mlp.experts.w2_weight"), StDType::U8)?,
                w2_scale: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w2_weight_scale"),
                    StDType::F8E4M3,
                )?,
                w2_amax: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w2_weight_amax"),
                    StDType::F32,
                )?,
                w2_denom: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w2_weight_outer_scale_denominator"),
                    StDType::F32,
                )?,
            }))
        } else {
            ExpertWeights::Mxfp8(Box::new(Mxfp8ExpertWeights {
                w13: dir.upload(backend, &w13_name, StDType::F8E4M3)?,
                w13_scale: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w13_weight_scale"),
                    StDType::U8,
                )?,
                w2: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w2_weight"),
                    StDType::F8E4M3,
                )?,
                w2_scale: dir.upload(
                    backend,
                    &format!("{lp}.mlp.experts.w2_weight_scale"),
                    StDType::U8,
                )?,
            }))
        };

        let attn_res_query = if l > 0 {
            Some(dir.upload(
                backend,
                &format!("model.llm.attn_res.attn_queries.{}", l - 1),
                StDType::F32,
            )?)
        } else {
            None
        };
        let mlp_res_query = dir.upload(
            backend,
            &format!("model.llm.attn_res.mlp_queries.{l}"),
            StDType::F32,
        )?;

        let controllers = match t.loop_config {
            Some(lc) if l >= lc.core_start && l < lc.core_end => Some(LoopControllers {
                attn: load_controller(backend, dir, &format!("{lp}.controllers.attn"))?,
                k: load_controller(backend, dir, &format!("{lp}.controllers.k"))?,
                v: load_controller(backend, dir, &format!("{lp}.controllers.v"))?,
                mlp: load_controller(backend, dir, &format!("{lp}.controllers.mlp"))?,
            }),
            _ => None,
        };

        layers.push(PhysicalLayerWeights {
            attn_norm,
            wq: dir.upload(backend, &format!("{lp}.attn.wq_du.weight"), StDType::Bf16)?,
            wk: dir.upload(backend, &format!("{lp}.attn.wk_dv.weight"), StDType::Bf16)?,
            wv: dir.upload(backend, &format!("{lp}.attn.wv_dv.weight"), StDType::Bf16)?,
            wr: dir.upload(backend, &format!("{lp}.attn.wr_du.weight"), StDType::Bf16)?,
            gate_attn: dir.upload(
                backend,
                &format!("{lp}.attn.gate_attn.weight"),
                StDType::Bf16,
            )?,
            sink_proj: dir.upload(
                backend,
                &format!("{lp}.attn.attn_sink_proj.weight"),
                StDType::Bf16,
            )?,
            wo: dir.upload(backend, &format!("{lp}.attn.wo_ud.weight"), StDType::Bf16)?,
            q_l2_scale: dir.upload(
                backend,
                &format!("{lp}.attn.q_l2_clip.scale"),
                StDType::Bf16,
            )?,
            rel_l2_scale: dir.upload(
                backend,
                &format!("{lp}.attn.rel_l2_clip.scale"),
                StDType::Bf16,
            )?,
            rel_logits_proj_t: upload_bf16_transposed(
                backend,
                dir,
                &format!("{lp}.attn.rel_logits_proj.proj"),
            )?,
            kv_conv,
            attn_sconv: dir.upload(backend, &format!("{lp}.attn_sconv.weight"), StDType::Bf16)?,
            mlp_norm: dir.upload(backend, &format!("{lp}.mlp_norm.weight"), StDType::Bf16)?,
            router_weight: dir.upload(backend, &format!("{lp}.mlp.gate.weight"), StDType::Bf16)?,
            router_bias: dir.upload(backend, &format!("{lp}.mlp.gate.bias"), StDType::F32)?,
            experts,
            mlp_sconv: dir.upload(backend, &format!("{lp}.mlp_sconv.weight"), StDType::Bf16)?,
            attn_res_query,
            mlp_res_query,
            controllers,
        });
    }

    Ok(Inkling2TextWeights {
        embed,
        embed_norm,
        layers,
        output_res_query,
        final_norm,
        unembed,
        cross_pass_queries,
        cross_pass_kv_queries,
    })
}
