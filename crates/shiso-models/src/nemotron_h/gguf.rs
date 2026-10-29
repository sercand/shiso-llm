//! The GGUF half of the loader: the same stack as the MLX half, out of a GGUF
//! export whose routed experts are `IQ4_NL` blocks and whose other matrices
//! are `q8_0`.
//!
//! # Where the routed experts go
//!
//! They are most of the file - 16.5 of a 19.7 GB export - and the reason the
//! export is read on a machine with less memory than that. So they are never
//! wrapped from the mapping: either the whole bank is imported (when the
//! device can hold it) or an [`ExpertStream`] reads one expert at a time from
//! the file with the page cache bypassed, and the stage multiplies through the
//! stream's slot map. Each expert is one contiguous slice of its projection's
//! tensor - the expert index is the outermost dimension - so a read is a single
//! positioned read.
//!
//! # What is converted at load
//!
//! Only the vectors: the container stores the state-space decay as
//! `-exp(A_log)` where the engine reads `A_log`, and the group norm as
//! `[groups, inner / groups]`, which is the same floats as the flat
//! `[inner]` the engine reads.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::LayerIdx;
use shiso_engine::attn::Attn;
use shiso_engine::draft::{DraftParts, HeadNorm};
use shiso_engine::ffn::Ffn;
use shiso_engine::layers::{Block, LayerWeights, Routed, Weights};
use shiso_engine::moe_biased::{
    self, BiasedMoe, ExpertStack, RouterGuess, RowIq4NlStack, StackStore,
};
use shiso_engine::projection::Weight;
use shiso_engine::sparse::weights::WeightPart;
use shiso_engine::sparse::{ExpertSource, ExpertStream};
use shiso_engine::ssm::Ssm;
use shiso_modelfile::GgufFile;
use shiso_modelfile::st_header::open_uncached;

use crate::nemotron_h::config::{Mixer, NemotronHConfig};
use crate::qwen4exp::mlx::{f32_bytes, upload};

/// Bytes one `IQ4_NL` block of 32 values occupies.
const IQ4NL_BLOCK_BYTES: u64 = 18;

/// The routed expert banks of a GGUF export, read straight from the file.
#[derive(Debug)]
pub struct GgufExperts {
    file: File,
    /// Per routed layer, the file offsets of its up and down banks.
    offsets: Vec<[u64; 2]>,
    /// Bytes one expert's up (and down) projection occupies.
    expert_bytes: u64,
}

impl GgufExperts {
    /// Locates every routed bank of `blocks`, in order, in the export at
    /// `path`, already parsed as `file`.
    ///
    /// # Errors
    ///
    /// [`CoreError::ConfigMissing`] naming an absent bank,
    /// [`CoreError::InvalidValue`] for one stored in another format, and
    /// [`CoreError::ShapeMismatch`] when a bank is not `experts` slices of
    /// one expert's width, or the up and down slices differ in size.
    pub fn open(path: &Path, file: &GgufFile, blocks: &[u32], experts: u32) -> CoreResult<Self> {
        let mut offsets = Vec::with_capacity(blocks.len());
        let mut expert_bytes = None;
        for block in blocks {
            let mut pair = [0u64; 2];
            for (slot, role) in pair.iter_mut().zip(["ffn_up_exps", "ffn_down_exps"]) {
                let name = format!("blk.{block}.{role}.weight");
                let desc = file
                    .find(&name)
                    .ok_or_else(|| CoreError::ConfigMissing { key: name.clone() })?;
                if desc.dtype != DType::IQ4_NL {
                    return Err(CoreError::InvalidValue {
                        key: "routed expert format",
                        value: format!("{name} is {:?}, not IQ4_NL", desc.dtype),
                    });
                }
                let per = desc.bytes / u64::from(experts.max(1));
                if per * u64::from(experts) != desc.bytes
                    || expert_bytes.is_some_and(|have| have != per)
                {
                    return Err(CoreError::ShapeMismatch {
                        field: "routed expert slice bytes",
                        expected: expert_bytes.unwrap_or(per) * u64::from(experts),
                        got: desc.bytes,
                    });
                }
                expert_bytes = Some(per);
                *slot = desc.abs_offset;
            }
            offsets.push(pair);
        }
        Ok(Self {
            file: open_uncached(path)?,
            offsets,
            expert_bytes: expert_bytes.unwrap_or(0),
        })
    }

    /// Routed layers this source reads.
    #[must_use]
    pub fn layers(&self) -> u32 {
        u32::try_from(self.offsets.len()).unwrap_or(u32::MAX)
    }
}

impl ExpertSource for GgufExperts {
    fn parts(&self) -> &'static [WeightPart] {
        &[WeightPart::Whole]
    }

    fn matrices(&self) -> usize {
        2
    }

    fn layout(&self) -> Vec<u64> {
        vec![self.expert_bytes; 2]
    }

    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()> {
        let base = self
            .offsets
            .get(layer as usize)
            .and_then(|pair| pair.get(array))
            .ok_or(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: self.offsets.len() as u64,
            })?;
        self.file
            .read_exact_at(dst, base + u64::from(expert) * self.expert_bytes)?;
        Ok(())
    }
}

/// Where a load puts the routed expert banks.
#[derive(Debug, Clone)]
pub enum ExpertPlacement {
    /// Imported whole: every expert resident.
    Resident,
    /// Read through this stream, one routed layer per stream layer in order.
    Streamed {
        /// The stream.
        stream: Arc<ExpertStream>,
        /// Whether each routed layer guesses the next one's experts and reads
        /// them while the device computes.
        prefetch: bool,
    },
}

/// Every resident weight of a GGUF export, slot by slot.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first absent tensor,
/// [`CoreError::InvalidValue`] for one in a format this loader does not take,
/// and whatever the imports report.
pub(crate) fn weights_gguf(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    config: &NemotronHConfig,
    placement: &ExpertPlacement,
) -> CoreResult<Weights> {
    let loader = Loader { file, backend };
    let shape = config.shape();
    // Every routed layer's router on the host, for the layer before it to
    // guess with, when the banks are streamed and guessing is on.
    let routers: Vec<HostRouter> = match placement {
        ExpertPlacement::Streamed { prefetch: true, .. } => routed_blocks(config)
            .iter()
            .map(|block| {
                Ok((
                    loader
                        .floats(&format!("blk.{block}.ffn_gate_inp.weight"))?
                        .into(),
                    loader
                        .floats(&format!("blk.{block}.exp_probs_b.bias"))?
                        .into(),
                ))
            })
            .collect::<CoreResult<_>>()?,
        _ => Vec::new(),
    };
    let hidden = shape.hidden;
    let mut layers = Vec::with_capacity(config.slots().len());
    let (mut ssm_rank, mut attn_rank, mut routed_rank) = (0u32, 0u32, 0u32);
    for (index, slot) in config.slots().iter().enumerate() {
        let il = LayerIdx::new(u32::try_from(index).unwrap_or(u32::MAX));
        let base = format!("blk.{}", slot.mixer);
        let (block, ordinal) = match config.mixers()[slot.mixer as usize] {
            Mixer::Mamba => {
                ssm_rank += 1;
                (Block::Ssm(loader.ssm(&base, config)?), ssm_rank - 1)
            }
            Mixer::Attention => {
                attn_rank += 1;
                (Block::Attn(loader.attn(&base, config)?), attn_rank - 1)
            }
            Mixer::Moe => {
                return Err(CoreError::InvalidValue {
                    key: "feed_forward_length",
                    value: format!("slot {index} opens on routed block {}", slot.mixer),
                });
            }
        };
        let (ffn_norm, ffn, moe) = match slot.ffn {
            None => (None, None, None),
            Some(block) => {
                let at = format!("blk.{block}");
                let (shared, mut routed) = loader.moe(&at, config, placement, routed_rank)?;
                routed.guess = routers
                    .get(routed_rank as usize + 1)
                    .map(|(weights, bias)| RouterGuess {
                        layer: routed_rank + 1,
                        weights: Arc::clone(weights),
                        bias: Arc::clone(bias),
                    });
                routed_rank += 1;
                (
                    Some(loader.vector(&format!("{at}.attn_norm.weight"))?),
                    Some(shared),
                    Some(Routed::Biased(routed)),
                )
            }
        };
        layers.push(LayerWeights {
            input_norm: loader.vector(&format!("{base}.attn_norm.weight"))?,
            post_attn_norm: None,
            ffn_norm,
            post_ffn_norm: None,
            ffn,
            moe,
            block,
            ordinal,
            kv_rank: shape.kv_owner_rank(il),
            layer_scalar: 1.0,
            ple: None,
        });
    }
    Ok(Weights {
        token_embd: loader.matrix("token_embd.weight", hidden, shape.vocab)?,
        output_norm: loader.vector("output_norm.weight")?,
        lm_head: Some(loader.matrix("output.weight", hidden, shape.vocab)?),
        ple_model_proj: None,
        ple_proj_norm: None,
        layers,
    })
}

/// The multi-token prediction block as a draft head: `eh_proj` over the
/// normed token embedding and the normed target state, one attention and one
/// biased routed sub-layer, and a centring output norm before the target's
/// own vocabulary projection.
///
/// The head reads the target's FINAL-normed state, so the target's output
/// norm is its hidden pre-norm; its routed banks are routed layer `rank` of
/// the stream when they are streamed.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] naming the first absent tensor, and whatever
/// the imports report.
pub(crate) fn draft_parts(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    config: &NemotronHConfig,
    placement: &ExpertPlacement,
    block: u32,
    rank: u32,
) -> CoreResult<DraftParts> {
    let loader = Loader { file, backend };
    let hidden = config.shape().hidden;
    let at = |role: &str| format!("blk.{block}.{role}");
    let (ffn, moe) = loader.moe(&format!("blk.{block}"), config, placement, rank)?;
    let shape = config.shape().moe.ok_or(CoreError::ShapeMismatch {
        field: "router geometry",
        expected: 1,
        got: 0,
    })?;
    Ok(DraftParts {
        fc: loader.matrix(&at("nextn.eh_proj.weight"), 2 * hidden, hidden)?,
        norm_embed: loader.vector(&at("nextn.enorm.weight"))?,
        norm_hidden: loader.vector(&at("nextn.hnorm.weight"))?,
        input_norm: loader.vector(&at("attn_norm.weight"))?,
        ffn_norm: loader.vector(&at("post_attention_norm.weight"))?,
        output_norm: loader.vector(&at("nextn.shared_head_norm.weight"))?,
        head_norm: HeadNorm::Layer,
        hidden_prenorm: Some(loader.vector("output_norm.weight")?),
        attn: loader.attn(&format!("blk.{block}"), config)?,
        ffn,
        moe: Some((Routed::Biased(moe), shape)),
    })
}

/// One routed layer's router weights and selection bias, on the host.
type HostRouter = (Arc<[f32]>, Arc<[f32]>);

/// The routed blocks of a stack, in the order the stream numbers them.
#[must_use]
pub fn routed_blocks(config: &NemotronHConfig) -> Vec<u32> {
    config.slots().iter().filter_map(|slot| slot.ffn).collect()
}

/// The export and the device, for the imports below.
struct Loader<'f> {
    file: &'f GgufFile,
    backend: &'f Arc<dyn Backend>,
}

impl Loader<'_> {
    fn desc(&self, name: &str) -> CoreResult<&shiso_modelfile::reader::TensorDesc> {
        self.file
            .find(name)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: name.to_owned(),
            })
    }

    /// A stored matrix `[out_dim, in_dim]`, wrapped from the mapping in its own
    /// format. Row-exact: this family speculates, and a verify pass's rows
    /// must sum exactly as the serial steps they check.
    fn matrix(&self, name: &str, in_dim: u32, out_dim: u32) -> CoreResult<Weight> {
        let desc = self.desc(name)?;
        Weight::import(
            self.backend,
            desc.dtype,
            in_dim,
            out_dim,
            self.file.mapping(),
            self.file.range_of(desc),
        )
        .map(Weight::row_exact)
    }

    /// An `f32` tensor's values.
    fn floats(&self, name: &str) -> CoreResult<Vec<f32>> {
        let view = self
            .file
            .get(name)
            .ok_or_else(|| CoreError::ConfigMissing {
                key: name.to_owned(),
            })?;
        match view.dtype {
            DType::F32 => {
                view.as_f32()
                    .map(<[f32]>::to_vec)
                    .map_err(|err| CoreError::InvalidValue {
                        key: "gguf vector",
                        value: format!("{name}: {err}"),
                    })
            }
            DType::BF16 => Ok(view
                .data
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()),
            other => Err(CoreError::InvalidValue {
                key: "gguf vector",
                value: format!("{name} is {other:?}, not f32 or bf16"),
            }),
        }
    }

    /// An `f32` tensor uploaded as it is.
    fn vector(&self, name: &str) -> CoreResult<DeviceBuffer> {
        upload(self.backend, f32_bytes(&self.floats(name)?))
    }

    /// One state-space block.
    fn ssm(&self, base: &str, config: &NemotronHConfig) -> CoreResult<Ssm> {
        let shape = config.shape();
        let s = shape.ssm.ok_or(CoreError::ShapeMismatch {
            field: "state-space geometry",
            expected: 1,
            got: 0,
        })?;
        let at = |role: &str| format!("{base}.{role}");
        // The container stores the decay itself, `-exp(A_log)`; the scan reads
        // its log and negates the exponential, so it is converted back once.
        let a_log: Vec<f32> = self
            .floats(&at("ssm_a"))?
            .into_iter()
            .map(|a| (-a).ln())
            .collect();
        Ok(Ssm {
            in_proj: self.matrix(&at("ssm_in.weight"), shape.hidden, s.in_width())?,
            conv_weight: self.vector(&at("ssm_conv1d.weight"))?,
            conv_bias: self.vector(&at("ssm_conv1d.bias"))?,
            dt_bias: self.vector(&at("ssm_dt.bias"))?,
            a_log: upload(self.backend, f32_bytes(&a_log))?,
            d: self.vector(&at("ssm_d"))?,
            norm: self.vector(&at("ssm_norm.weight"))?,
            out_proj: self.matrix(&at("ssm_out.weight"), s.inner(), shape.hidden)?,
        })
    }

    /// One attention block: four projections and no norms.
    fn attn(&self, base: &str, config: &NemotronHConfig) -> CoreResult<Attn> {
        let shape = config.shape();
        let widths = shape.attn.first().ok_or(CoreError::ShapeMismatch {
            field: "attention slots",
            expected: 1,
            got: 0,
        })?;
        let q = widths.q_heads * widths.head_dim;
        let kv = widths.kv_heads * widths.head_dim;
        let at = |role: &str| format!("{base}.{role}.weight");
        Ok(Attn {
            q: self.matrix(&at("attn_q"), shape.hidden, q)?,
            k: Some(self.matrix(&at("attn_k"), shape.hidden, kv)?),
            v: Some(self.matrix(&at("attn_v"), shape.hidden, kv)?),
            output: self.matrix(&at("attn_output"), q, shape.hidden)?,
            q_norm: None,
            k_norm: None,
        })
    }

    /// One routed block: the shared expert and the routed half, whose banks
    /// are routed layer `rank` of the stream when they are streamed.
    fn moe(
        &self,
        base: &str,
        config: &NemotronHConfig,
        placement: &ExpertPlacement,
        rank: u32,
    ) -> CoreResult<(Ffn, BiasedMoe)> {
        let shape = config.shape();
        let moe = shape.moe.ok_or(CoreError::ShapeMismatch {
            field: "router geometry",
            expected: 1,
            got: 0,
        })?;
        let hidden = shape.hidden;
        let at = |role: &str| format!("{base}.{role}");
        let shared = Ffn {
            gate: None,
            up: self.matrix(&at("ffn_up_shexp.weight"), hidden, shape.intermediate)?,
            down: self.matrix(&at("ffn_down_shexp.weight"), shape.intermediate, hidden)?,
        };
        let router_values = self.floats(&at("ffn_gate_inp.weight"))?;
        let router = Weight::from_buffer(
            DType::F32,
            hidden,
            moe.experts,
            upload(self.backend, f32_bytes(&router_values))?,
        );
        let stack = |role: &str, matrix: usize, in_dim: u32, out_dim: u32| -> CoreResult<_> {
            let store = match placement {
                ExpertPlacement::Resident => {
                    let desc = self.desc(&at(role))?;
                    StackStore::Resident(import_host(
                        self.backend,
                        self.file.mapping(),
                        self.file.range_of(desc),
                    )?)
                }
                ExpertPlacement::Streamed { stream, .. } => StackStore::Streamed {
                    stream: Arc::clone(stream),
                    matrix,
                    layer: rank,
                },
            };
            let stack = ExpertStack::Iq4Nl(RowIq4NlStack {
                store,
                in_dim,
                out_dim,
            });
            moe_biased::check_stack(&stack)?;
            Ok(stack)
        };
        Ok((
            shared,
            BiasedMoe {
                router,
                bias: self.vector(&at("exp_probs_b.bias"))?,
                scale: config.routed_scale(),
                renormalize: config.renormalize(),
                up: stack("ffn_up_exps.weight", 0, hidden, moe.intermediate)?,
                down: stack("ffn_down_exps.weight", 1, moe.intermediate, hidden)?,
                guess: None,
            },
        ))
    }
}

/// Bytes one expert's projection of `rows` rows over `in_dim` inputs occupies.
#[must_use]
pub const fn iq4nl_expert_bytes(rows: u32, in_dim: u32) -> u64 {
    rows as u64 * (in_dim as u64 / 32) * IQ4NL_BLOCK_BYTES
}

#[cfg(test)]
#[path = "gguf_tests.rs"]
mod tests;
