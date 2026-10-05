//! The model type: a checkpoint of this family, loaded and answering the shared
//! contract.
//!
//! Everything here is identity. Which file the weights came from, what its
//! tensors are called, which token ends a turn, which formats a kernel in this
//! build can read - those are facts about this family, and they are the only
//! facts this crate holds. The stack that runs them is
//! [`shiso_engine::sparse::SparseStack`], and every method below forwards to
//! it.
//!
//! # One constructor, and what it removes
//!
//! [`Qwen4ExpModel::load`] either returns a model whose weights are on the
//! device or it returns an error. There is no `new` plus `load` pair and no
//! state that means "not loaded yet", so no caller downstream has to ask.
//!
//! # What a pass reads, and where each of those things comes from
//!
//! The three methods of the shared contract that run a chunk forward to
//! [`SparseStack::prefill_chunk`], [`SparseStack::prefill_last`] and
//! [`SparseStack::decode_batch`], which claim the cache rows, plan the pass,
//! start its n-gram gather, write its host inputs, land the gathered rows,
//! issue every step and advance the session.
//!
//! One of the buffers a pass reads is not a weight and not an activation: the
//! n-gram rows. Every checkpoint of this family folds an n-gram block into one
//! slot, and that block decodes a row per (position, hash head) out of a table
//! of hundreds of millions of rows that is never made resident. [`load`] opens
//! a [`shiso_ple::PleStream`] over that tensor and the stack gathers through
//! it, so the rows a launch reads are rows this process fetched for those
//! positions.
//!
//! That is the whole of the difference from the previous state of this file,
//! and it is worth naming: the buffer was allocated and never written, so every
//! launch of a pass succeeded and one layer's entire contribution to the
//! residual was decoded out of whatever the allocator last left at that
//! address. There is no typed refusal above these three methods any more,
//! because the two buffers it named are both written - the n-gram rows here,
//! and a session's recurrent slots when it opens.
//!
//! # What still refuses, and what has not been checked
//!
//! A decode batch of more than one sequence runs: the six kernels that carry
//! per-sequence state take a device array of one pointer per row, the pass
//! packs those arrays a step, and the pooled indexer key pass - the one with no
//! batched sibling - is launched once per row over that row's own page table.
//! What a batch still refuses is a session opened for speculation, because the
//! batched kernels declare no per-position checkpoint slab; and a device that
//! cannot hand out the addresses of its own allocations, because that is what
//! the arrays hold.
//!
//! A template kernel this build captured no entry name for is refused by name
//! as well, which on a host that never compiled the kernels is every trellis
//! projection.
//!
//! What has *not* been checked is everything a device would answer. No kernel
//! of this family has run anywhere in this workspace, so the claims that hold
//! are the ones about the host half: which launches a chunk makes, in what
//! order, over which buffers, and with what in those buffers. Numbers are owed
//! to a machine with the hardware.
//!
//! [`load`]: Qwen4ExpModel::load

use std::path::Path;
use std::sync::Arc;

use shiso_core::DType;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, import_host};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::ledger::{self, MemTag};
use shiso_core::media::{MediaModel, VisionPrefill};
use shiso_core::model::{
    DecodeStep, Logits, Model, ModelInfo, PromptRun, Speculated, Speculating, StopTokens,
};

use shiso_engine::sparse::headslice;
use shiso_engine::sparse::ple::NgramTable;
use shiso_engine::sparse::stack::{Pass, SparseOptions, SparseStack};
use shiso_engine::sparse::weights::{
    DraftHead, ExpertBank, Indexer, NgramBlock, Recurrent, SparseAttn, SparseBlock, SparseLayer,
    SparseWeights, StreamMix, Weight,
};
use shiso_modelfile::{GgufError, GgufFile, TensorView};
use shiso_ple::{PleRowType, PleStream};

use crate::qwen4exp::binding::{
    AttnRoles, BlockRoles, DraftBinding, ExpertRoles, MixRoles, ModelBinding, NGRAM_BIAS,
    NGRAM_TABLE, NgramRoles, RecurrentRoles, bind_draft, bind_qwen4exp,
};
use crate::qwen4exp::config::Qwen4ExpConfig;
use crate::qwen4exp::{exl3, mlx};

/// The formats a bound projection may be stored as.
///
/// A tensor outside this list ends the load rather than selecting a slower path:
/// a model that quietly ran a dequantized copy of one tensor would produce
/// logits nobody can attribute to a kernel.
const SUPPORTED: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Q8_0,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q4_K,
    DType::Q6_K,
    DType::Exl3K4,
    DType::Exl3K5,
    DType::Exl3K6,
];

/// A loaded checkpoint of this family.
#[derive(Debug)]
pub struct Qwen4ExpModel {
    stack: SparseStack,
    info: ModelInfo,
    /// Storage format of the layer-zero routed expert gate.
    ///
    /// Kept because it is the one thing a loaded model knows about how its
    /// weights were quantized that survives the load: the binding is dropped
    /// once the tensors are resident, and the checkpoint tier has to be able to
    /// say what width the rows it stores were produced at long afterwards.
    expert_gate: DType,
}

impl Qwen4ExpModel {
    /// Opens a checkpoint and puts it on the device.
    ///
    /// Weights are imported rather than copied: the file mapping is handed to
    /// the backend, which wraps it where the device can read host memory in
    /// place and copies where it cannot. The engine never branches on which of
    /// the two happened.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] for a file that cannot be opened,
    /// [`CoreError::ConfigMissing`] naming the first tensor it does not carry,
    /// [`CoreError::Unsupported`] for a tensor whose format no kernel reads, and
    /// whatever [`SparseStack::new`] reports.
    pub fn load(
        backend: &Arc<dyn Backend>,
        path: &Path,
        config: &Qwen4ExpConfig,
        opts: SparseOptions,
    ) -> CoreResult<Self> {
        Self::load_with_draft(backend, path, None, config, opts)
    }

    /// The same, with a draft head read from a sidecar beside the checkpoint.
    ///
    /// The head is a second model over the same residual stream and the same
    /// vocabulary, so it is loaded against the target's own geometry and
    /// refused when it does not match. `None` is a build that decodes one token
    /// a step.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpModel::load`], plus [`CoreError::UnknownArch`] when the
    /// sidecar is not a draft head and [`CoreError::ShapeMismatch`] when it is
    /// one for a different model.
    pub fn load_with_draft(
        backend: &Arc<dyn Backend>,
        path: &Path,
        draft: Option<&Path>,
        config: &Qwen4ExpConfig,
        opts: SparseOptions,
    ) -> CoreResult<Self> {
        Self::load_with(backend, path, draft, None, config, opts)
    }

    /// The same, with the corpus counts a narrowed draft vocabulary is chosen
    /// from.
    ///
    /// `counts` is one `id count` pair per line. Absent is the whole head,
    /// which is correct and costs a draft most of what a target step costs.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpModel::load_with_draft`], plus [`CoreError::Backend`] when
    /// the counts file cannot be read.
    pub fn load_with(
        backend: &Arc<dyn Backend>,
        path: &Path,
        draft: Option<&Path>,
        counts: Option<&Path>,
        config: &Qwen4ExpConfig,
        opts: SparseOptions,
    ) -> CoreResult<Self> {
        let file = GgufFile::open_shared(path).map_err(open_failure)?;
        let binding = bind_qwen4exp(&file, config)?;
        let ngram = {
            let _tag = ledger::scope(MemTag::WeightsEmbedding);
            open_ngram(backend, &file, path, &binding, config, opts)?
        };
        let expert_gate = binding
            .layers
            .first()
            .map_or(binding.token_embd.dtype, |layer| layer.experts.gate.dtype);
        let mut weights = import(backend, &file, &binding, opts.pack_halves)?;
        if let Some(sidecar) = draft {
            let _tag = ledger::scope(MemTag::DraftWeights);
            // Its own mapping, dropped with the target's: the head's tensors
            // take handles on it exactly as the target's do on the checkpoint.
            let head = GgufFile::open_shared(sidecar).map_err(open_failure)?;
            let bound = bind_draft(&head, config)?;
            let rank = config.shape().count(shiso_engine::shape::LayerKind::Attn);
            weights.draft = Some(import_draft(
                backend,
                &head,
                &bound,
                rank,
                opts.pack_halves,
            )?);
            drop(bound);
            drop(head);
            if let Some(list) = counts {
                narrow_draft_head(backend, &mut weights, list, config.shape().vocab)?;
            }
        }
        // The mapping is dropped here on purpose: every weight that needed it
        // holds its own handle on it, so the bytes stay mapped for exactly as
        // long as some device buffer names them and no longer.
        drop(binding);
        drop(file);

        let stack = SparseStack::new(backend, config.shape().clone(), weights, opts, Some(ngram))?;
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
            expert_gate,
        })
    }

    /// Opens an MLX checkpoint directory and puts it on the device.
    ///
    /// `residency` chooses where the routed banks live; `None` decides from
    /// the device: the banks are imported in place when they fit in the
    /// device's working set beside everything else with `reserve` bytes to
    /// spare for the cache and activations, and are streamed from disk into as
    /// many expert slots as do fit otherwise.
    ///
    /// # Errors
    ///
    /// Whatever [`mlx::MlxCheckpoint::open`], [`Qwen4ExpConfig::from_mlx`],
    /// the imports and [`SparseStack::new`] report, and
    /// [`CoreError::Unsupported`] when not even a streamed bank fits.
    pub fn load_mlx(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        hf: &shiso_core::config::HfConfig,
        opts: SparseOptions,
        residency: Option<mlx::ExpertResidency>,
        reserve: u64,
    ) -> CoreResult<Self> {
        let ck = mlx::MlxCheckpoint::open(dir)?;
        let config = Qwen4ExpConfig::from_mlx(hf, &ck)?;
        let ngram = {
            let _tag = ledger::scope(MemTag::WeightsEmbedding);
            mlx::open_ngram(&ck, &config, &opts)?
        };
        let sized = residency.is_none();
        let residency = match residency {
            Some(chosen) => chosen,
            None => choose_residency(backend.as_ref(), &ck, reserve)?,
        };
        tracing::info!(?residency, "routed expert banks");
        let weights = mlx::import(backend, &ck, &config, residency)?;
        let expert_gate = DType::MlxAffine4Gs64;
        let free = device_free(backend.as_ref());
        let mut stack =
            SparseStack::new(backend, config.shape().clone(), weights, opts, Some(ngram))?;
        if let mlx::ExpertResidency::Streamed { slots } = residency {
            let slots = if sized {
                beside_stack(backend.as_ref(), free, slots, mlx::slot_bytes())
            } else {
                slots
            };
            let stream = mlx::open_stream(backend, &ck, slots, opts.prefill_rows)?;
            stack.attach_expert_stream(stream)?;
        }
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
            expert_gate,
        })
    }

    /// Opens the reference export's directory - brain-float dense weights,
    /// FP8 experts and n-gram table - and puts it on the device.
    ///
    /// The routed experts always go through the expert stream. `slots` fixes
    /// the pool; `None` sizes it to the device - every expert, preloaded, when
    /// they all fit beside the dense weights with `reserve` to spare, and as
    /// many as fit otherwise.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpModel::load_mlx`].
    pub fn load_reference(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        hf: &shiso_core::config::HfConfig,
        opts: SparseOptions,
        slots: Option<u32>,
        reserve: u64,
    ) -> CoreResult<Self> {
        Self::load_streamed(
            backend,
            dir,
            hf,
            opts,
            &StreamedLoad {
                slots,
                reserve,
                draft: false,
                counts: None,
            },
            StreamedExport::Reference,
        )
    }

    /// Opens NVIDIA's `modelopt` export - the reference export's layout with
    /// NVFP4 routed experts - and puts it on the device; see
    /// [`crate::qwen4exp::nvfp4`]. Pool sizing is [`Qwen4ExpModel::load_reference`]'s;
    /// `load.draft` binds the export's own draft head, and `load.counts`
    /// narrows its vocabulary as [`Qwen4ExpModel::load_with`] does.
    ///
    /// # Errors
    ///
    /// As [`Qwen4ExpModel::load_reference`], plus whatever the draft head's
    /// import reports.
    pub fn load_nvfp4(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        hf: &shiso_core::config::HfConfig,
        opts: SparseOptions,
        load: &StreamedLoad<'_>,
    ) -> CoreResult<Self> {
        Self::load_streamed(backend, dir, hf, opts, load, StreamedExport::Nvfp4)
    }

    /// The body of the two streamed-expert exports, which differ only in the
    /// routed codec and in what their dense matrices become.
    fn load_streamed(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        hf: &shiso_core::config::HfConfig,
        opts: SparseOptions,
        load: &StreamedLoad<'_>,
        export: StreamedExport,
    ) -> CoreResult<Self> {
        use crate::qwen4exp::config::{EXPERTS, N_LAYERS};
        let (slots, reserve) = (load.slots, load.reserve);
        let ck = match export {
            StreamedExport::Reference => crate::qwen4exp::fp8::open(dir)?,
            StreamedExport::Nvfp4 => crate::qwen4exp::nvfp4::open(dir)?,
        };
        let config = Qwen4ExpConfig::from_reference(hf, &ck)?;
        let ngram = {
            let _tag = ledger::scope(MemTag::WeightsEmbedding);
            crate::qwen4exp::fp8::open_ngram(backend, &ck, &config, &opts)?
        };
        let all = EXPERTS * N_LAYERS;
        let sized = slots.is_none();
        let slot_bytes = export.slot_bytes();
        let slots = if let Some(slots) = slots {
            slots
        } else {
            {
                let total = backend.memory_info().map_or(u64::MAX, |info| info.total);
                let rest = ck.resident_bytes().saturating_sub(export.expert_bytes());
                // The dense weights are counted twice. They are wrapped file
                // pages, read whole every step, and the system evicts wrapped
                // pages first when the working set is full: sized to the last
                // byte, the pool pushed them out and every step read them back
                // from disk (measured on a 64 GB machine: 0.3 tok/s, against
                // 2.9 with this margin).
                let room = total.saturating_sub(rest.saturating_mul(2).saturating_add(reserve));
                let fit = u32::try_from(room / slot_bytes).unwrap_or(u32::MAX);
                if fit < EXPERTS {
                    return Err(CoreError::Unsupported {
                        what: "a device whose working set cannot hold one layer's routed experts \
                               beside the dense weights",
                        backend: "qwen4exp",
                    });
                }
                fit.min(all)
            }
        };
        tracing::info!(slots, all, export = ?export, "routed expert pool of the streamed export");
        let mut weights = crate::qwen4exp::fp8::import(backend, &ck, &config, export.layout())?;
        if load.draft {
            let rank = config.shape().count(shiso_engine::shape::LayerKind::Attn);
            weights.draft = Some(crate::qwen4exp::fp8::import_draft(
                backend,
                &ck,
                rank,
                export.layout(),
            )?);
            if let Some(list) = load.counts {
                narrow_draft_head(backend, &mut weights, list, config.shape().vocab)?;
            }
        }
        let free = device_free(backend.as_ref());
        let mut stack =
            SparseStack::new(backend, config.shape().clone(), weights, opts, Some(ngram))?;
        // A pool that holds every expert keeps them all; only one sized to the
        // device gives back what the stack took.
        let slots = if sized && slots < all {
            beside_stack(backend.as_ref(), free, slots, slot_bytes)
        } else {
            slots
        };
        let stream = match export {
            StreamedExport::Reference => {
                crate::qwen4exp::fp8::open_stream(backend, &ck, slots, opts.prefill_rows)?
            }
            StreamedExport::Nvfp4 => {
                crate::qwen4exp::nvfp4::open_stream(backend, &ck, slots, opts.prefill_rows)?
            }
        };
        stack.attach_expert_stream(stream)?;
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
            expert_gate: export.layout().experts,
        })
    }

    /// Opens the trellis export's directory as it is published and puts it on
    /// the device, its routed banks resident.
    ///
    /// `ngram` is the n-gram table file, which the export keeps beside its
    /// shards and the caller names; nothing here looks for it by name.
    ///
    /// The export carries its own draft head; `draft` loads it, and `counts`
    /// narrows its vocabulary as [`Qwen4ExpModel::load_with`] does. The
    /// trellis kernels exist on one backend, and a device without them refuses
    /// the first projection the stack is built over.
    ///
    /// # Errors
    ///
    /// Whatever [`exl3::open`], [`exl3::NgramFile::open`],
    /// [`Qwen4ExpConfig::from_exl3`], the imports and [`SparseStack::new`]
    /// report, and [`CoreError::Backend`] when the counts file cannot be read.
    pub fn load_exl3(
        backend: &Arc<dyn Backend>,
        dir: &Path,
        ngram: &Path,
        hf: &shiso_core::config::HfConfig,
        draft: bool,
        counts: Option<&Path>,
        opts: SparseOptions,
    ) -> CoreResult<Self> {
        let ck = exl3::open(dir)?;
        let table = exl3::NgramFile::open(ngram)?;
        let config = Qwen4ExpConfig::from_exl3(hf, dir, &table)?;
        let ngram = {
            let _tag = ledger::scope(MemTag::WeightsEmbedding);
            exl3::open_ngram(backend, &table, &config, &opts)?
        };
        let mut weights = exl3::import(backend, &ck, &config, opts.pack_halves)?;
        if draft {
            let rank = config.shape().count(shiso_engine::shape::LayerKind::Attn);
            weights.draft = Some(exl3::import_draft(backend, &ck, rank, opts.pack_halves)?);
            if let Some(list) = counts {
                narrow_draft_head(backend, &mut weights, list, config.shape().vocab)?;
            }
        }
        let expert_gate = weights
            .layers
            .first()
            .map_or(DType::Exl3K4, |layer| layer.experts.gate.dtype);
        drop(ck);
        let stack = SparseStack::new(backend, config.shape().clone(), weights, opts, Some(ngram))?;
        Ok(Self {
            info: ModelInfo {
                context_capacity: stack.context_capacity(),
                vocab: config.shape().vocab,
                hidden: config.shape().hidden,
                n_layers: config.shape().n_layers(),
                stop_tokens: StopTokens::new(&config.stop_tokens())?,
            },
            stack,
            expert_gate,
        })
    }

    /// Storage format of the layer-zero routed expert gate.
    #[must_use]
    pub const fn expert_gate_dtype(&self) -> DType {
        self.expert_gate
    }

    /// The stack this model forwards to.
    #[must_use]
    pub const fn stack(&self) -> &SparseStack {
        &self.stack
    }

    /// The stack this model forwards to, mutably.
    pub const fn stack_mut(&mut self) -> &mut SparseStack {
        &mut self.stack
    }

    /// The ordered work one chunk would do.
    ///
    /// The one entry point that reaches the whole of the forward pass on this
    /// half of the port. A caller checking that a chunk is dispatchable - every
    /// weight has a tier, every launch has a grid, every island has a key -
    /// asks this and reads the answer.
    ///
    /// # Errors
    ///
    /// Whatever [`SparseStack::plan_chunk`] reports.
    pub fn plan_chunk(
        &self,
        session: SessionId,
        tokens: &[TokenId],
        logit_rows: u32,
    ) -> CoreResult<Pass> {
        self.stack.plan_chunk(session, tokens, logit_rows)
    }
}

/// Where an MLX checkpoint's routed banks go on this device.
///
/// Resident when every language-model weight, the banks included, fits in the
/// device's working set with `reserve` bytes left; otherwise streamed, into as
/// many slots as fit once the rest of the weights and the reserve are
/// accounted for. The n-gram table is never counted: it is streamed either
/// way.
/// What the device reports free, or `None` where it reports nothing.
fn device_free(backend: &dyn Backend) -> Option<u64> {
    backend.memory_info().map(|info| info.free)
}

/// A pool sized to the device, less what the stack took after it was sized.
///
/// The pool's size has to be known before the weights are imported - it
/// decides whether the banks are imported at all - so the scratch and cache
/// the stack then allocates for its widest pass come out of the working set
/// the pool was sized to fill. At a server's prefill chunk that is several
/// gigabytes, and with it left in, the pool and the stack together overran
/// the working set of a 64 GB machine: every step waited on the system's
/// swap, 0.46 tok/s against 5 to 6 with the stack's bytes given back. `free`
/// is what the device reported free before the stack was built; the floor is
/// one layer's experts, below which the stream refuses to open anyway.
fn beside_stack(backend: &dyn Backend, free: Option<u64>, slots: u32, slot_bytes: u64) -> u32 {
    let (Some(before), Some(after)) = (free, device_free(backend)) else {
        return slots;
    };
    let taken = before.saturating_sub(after);
    let give_back = u32::try_from(taken.div_ceil(slot_bytes.max(1))).unwrap_or(u32::MAX);
    let fitted = slots
        .saturating_sub(give_back)
        .max(crate::qwen4exp::config::EXPERTS.min(slots));
    if fitted < slots {
        tracing::info!(
            stack_gib = taken >> 30,
            slots = fitted,
            was = slots,
            "the expert pool gives back what the stack's scratch and cache took"
        );
    }
    fitted
}

fn choose_residency(
    backend: &dyn Backend,
    ck: &mlx::MlxCheckpoint,
    reserve: u64,
) -> CoreResult<mlx::ExpertResidency> {
    let total = backend.memory_info().map_or(u64::MAX, |info| info.total);
    let banks = mlx::expert_bytes();
    let rest = ck.resident_bytes().saturating_sub(banks);
    let fixed = rest.saturating_add(reserve);
    if fixed.saturating_add(banks) <= total {
        return Ok(mlx::ExpertResidency::Resident);
    }
    let room = total.saturating_sub(fixed);
    let slots = u32::try_from(room / mlx::slot_bytes()).unwrap_or(u32::MAX);
    let all = crate::qwen4exp::config::EXPERTS * crate::qwen4exp::config::N_LAYERS;
    if slots < crate::qwen4exp::config::EXPERTS {
        return Err(CoreError::Unsupported {
            what: "a device whose working set cannot hold one layer's routed experts beside \
                   the rest of the weights",
            backend: "qwen4exp",
        });
    }
    tracing::info!(
        total_gib = total >> 30,
        other_weights_gib = rest >> 30,
        reserve_gib = reserve >> 30,
        banks_gib = banks >> 30,
        slots,
        "the routed banks do not fit; streaming them from disk"
    );
    Ok(mlx::ExpertResidency::Streamed {
        slots: slots.min(all),
    })
}

/// Opens the streamed n-gram table this checkpoint gathers its rows from.
///
/// The table is a tensor of the container and is never made resident: what is
/// opened here is a reader over the same file, positioned at that tensor's own
/// offset. Its bias is the other half - a tensor small enough to be resident,
/// which the trellis codec finishes every decoded weight with and the
/// block-quantized codec has none of - and it goes to the device as well as to
/// the host, because the host copy is what the reader validates against and the
/// device copy is what the decode kernel reads.
///
/// # Errors
///
/// [`CoreError::ConfigMissing`] for a checkpoint with no table, or a
/// trellis-coded one with no bias; [`CoreError::ShapeMismatch`] when the
/// per-head row counts do not add up to the rows the tensor holds; whatever the
/// reader reports about the table's own numbers; and [`CoreError::Io`] when the
/// file cannot be opened a second time.
fn open_ngram(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    path: &Path,
    binding: &ModelBinding<'_>,
    config: &Qwen4ExpConfig,
    opts: SparseOptions,
) -> CoreResult<NgramTable> {
    let ple = config.shape().ple.ok_or(CoreError::ShapeMismatch {
        field: "n-gram slot",
        expected: 1,
        got: 0,
    })?;
    let table = binding
        .ngram_table
        .as_ref()
        .ok_or_else(|| CoreError::ConfigMissing {
            key: NGRAM_TABLE.to_owned(),
        })?;
    // The container states the codec in the tensor's own type, which is why
    // this is a lookup rather than a build-time choice: both codecs ship.
    let row_type = if table.dtype == DType::Exl3Ngram6 {
        PleRowType::TrellisK6
    } else {
        PleRowType::BlockNl4
    };
    let bias = match (row_type, binding.ngram_bias.as_ref()) {
        (PleRowType::TrellisK6, None) => {
            return Err(CoreError::ConfigMissing {
                key: NGRAM_BIAS.to_owned(),
            });
        }
        (PleRowType::TrellisK6, Some(view)) => Some(view),
        // A bias beside a codec that takes no such pointer is a tensor nothing
        // would read, and the tier below refuses it rather than carrying it.
        (PleRowType::BlockNl4 | PleRowType::MlxAffine4 | PleRowType::Fp8E4M3 { .. }, _) => None,
    };
    let params =
        config
            .ngram()
            .params(ple, row_type, bias.map(|view| Arc::new(halves(view.data))))?;
    // The tensor's second dimension is how many rows the file holds; the head
    // vocabulary sizes are how many the hash can name. The second must fit
    // inside the first, or an identifier past the end is a row read out of the
    // next tensor. It need not equal it: the head tables tile the table without
    // gaps (`PleParams::new` checks that), and the published checkpoint rounds
    // the stored row count up past the last head - 320,001,536 rows behind
    // 320,001,446 addressable ones - so an equality test refuses a file that is
    // correct.
    let rows = table.shape.dims().get(1).copied().unwrap_or(0);
    if params.total_rows() > rows {
        return Err(CoreError::ShapeMismatch {
            field: "n-gram table rows",
            expected: rows,
            got: params.total_rows(),
        });
    }
    let offset = u64::try_from(range_of(file, table)?.start).unwrap_or(u64::MAX);
    let shard = std::fs::File::open(path).map_err(CoreError::Io)?;
    let stream = PleStream::open(
        shard,
        offset,
        rows,
        params,
        opts.ngram_cache_bytes,
        &opts.ngram_reader,
    )?;
    let resident_bias = bias
        .map(|view| import_host(backend, file.mapping(), range_of(file, view)?))
        .transpose()?;
    NgramTable::new(table.dtype, Arc::new(stream), resident_bias)
}

/// Reads a half-precision tensor's bytes as the words the reader's codec wants.
///
/// The bias is consumed as bit patterns rather than as floats: the codec
/// converts them itself, in the rounding the device kernel uses, and a host-side
/// widening here would be a second conversion the two sides could disagree
/// about.
fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// Moves a draft head's tensors to the device.
///
/// `rank` is the head's rank among attention slots, which for a head is always
/// one past the target's last: its planes are the extra rank the pool was
/// opened with, and naming any other would make the head read and overwrite a
/// target layer's cache.
fn import_draft(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    binding: &DraftBinding<'_>,
    rank: u32,
    pack: bool,
) -> CoreResult<DraftHead> {
    let BlockRoles::Attn(roles) = &binding.layer.block else {
        return Err(CoreError::Unsupported {
            what: "a draft head whose block is not the attention block",
            backend: "qwen4exp",
        });
    };
    Ok(DraftHead {
        norm_embed: vector(backend, file, &binding.norm_embed)?,
        norm_hidden: vector(backend, file, &binding.norm_hidden)?,
        fc_embed: weight(backend, file, &binding.fc_embed)?,
        fc_hidden: weight(backend, file, &binding.fc_hidden)?,
        layer: SparseLayer {
            attn_mix: import_mix(backend, file, &binding.layer.attn_mix, pack)?,
            ffn_mix: import_mix(backend, file, &binding.layer.ffn_mix, pack)?,
            block: SparseBlock::Attn(import_attn(backend, file, roles)?),
            experts: import_experts(backend, file, &binding.layer.experts, pack)?,
            ngram: None,
            ordinal: rank,
        },
        output_mix: import_mix(backend, file, &binding.output_mix, pack)?,
        head: None,
    })
}

/// Moves every bound tensor to the device.
fn import(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    binding: &ModelBinding<'_>,
    pack: bool,
) -> CoreResult<SparseWeights> {
    let mut gdn_rank = 0;
    let mut attn_rank = 0;
    let mut layers = Vec::with_capacity(binding.layers.len());
    for layer in &binding.layers {
        let (block, ordinal) = match &layer.block {
            BlockRoles::Recurrent(roles) => {
                let rank = gdn_rank;
                gdn_rank += 1;
                let _tag = ledger::scope(MemTag::WeightsRecurrent);
                (
                    SparseBlock::Recurrent(import_recurrent(backend, file, roles, pack)?),
                    rank,
                )
            }
            BlockRoles::Attn(roles) => {
                let rank = attn_rank;
                attn_rank += 1;
                let _tag = ledger::scope(MemTag::WeightsAttention);
                (SparseBlock::Attn(import_attn(backend, file, roles)?), rank)
            }
        };
        let mix = ledger::scope(MemTag::WeightsStreamMix);
        let attn_mix = import_mix(backend, file, &layer.attn_mix, pack)?;
        let ffn_mix = import_mix(backend, file, &layer.ffn_mix, pack)?;
        drop(mix);
        let experts = {
            let _tag = ledger::scope(MemTag::WeightsExperts);
            import_experts(backend, file, &layer.experts, pack)?
        };
        let ngram = {
            let _tag = ledger::scope(MemTag::WeightsEmbedding);
            layer
                .ngram
                .as_ref()
                .map(|roles| import_ngram(backend, file, roles, pack))
                .transpose()?
        };
        layers.push(SparseLayer {
            attn_mix,
            ffn_mix,
            block,
            experts,
            ngram,
            ordinal,
        });
    }
    let token_embd = {
        let _tag = ledger::scope(MemTag::WeightsEmbedding);
        weight(backend, file, &binding.token_embd)?
    };
    let _tag = ledger::scope(MemTag::WeightsHead);
    let lm_head = binding
        .output
        .as_ref()
        .map(|view| weight(backend, file, view))
        .transpose()?;
    // The mix that folds the streams back into one row is part of the head.
    let output_mix = import_mix(backend, file, &binding.output_mix, pack)?;
    Ok(SparseWeights {
        token_embd,
        lm_head,
        output_mix,
        layers,
        draft: None,
        expert_map: None,
    })
}

/// Moves one stream mix to the device.
fn import_mix(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    roles: &MixRoles<'_>,
    pack: bool,
) -> CoreResult<StreamMix> {
    Ok(StreamMix {
        norm: vector(backend, file, &roles.norm)?,
        down: projection(backend, file, &roles.down, pack)?,
        up: projection(backend, file, &roles.up, pack)?,
        inject: roles
            .inject
            .as_ref()
            .map(|view| projection(backend, file, view, pack))
            .transpose()?,
    })
}

/// Moves one routed feed-forward to the device.
fn import_experts(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    roles: &ExpertRoles<'_>,
    pack: bool,
) -> CoreResult<ExpertBank> {
    Ok(ExpertBank {
        router: projection(backend, file, &roles.router, pack)?,
        gate: weight(backend, file, &roles.gate)?,
        up: weight(backend, file, &roles.up)?,
        down: weight(backend, file, &roles.down)?,
        shared_router: projection(backend, file, &roles.shared_router, pack)?,
        shared_gate: weight(backend, file, &roles.shared_gate)?,
        shared_up: weight(backend, file, &roles.shared_up)?,
        shared_down: weight(backend, file, &roles.shared_down)?,
    })
}

/// Moves one attention slot to the device.
fn import_attn(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    roles: &AttnRoles<'_>,
) -> CoreResult<SparseAttn> {
    Ok(SparseAttn {
        q: weight(backend, file, &roles.q)?,
        k: weight(backend, file, &roles.k)?,
        v: weight(backend, file, &roles.v)?,
        output: weight(backend, file, &roles.output)?,
        q_norm: vector(backend, file, &roles.q_norm)?,
        k_norm: vector(backend, file, &roles.k_norm)?,
        indexer: Indexer {
            q: weight(backend, file, &roles.idx_q)?,
            k: weight(backend, file, &roles.idx_k)?,
            q_norm: vector(backend, file, &roles.idx_q_norm)?,
            k_norm: vector(backend, file, &roles.idx_k_norm)?,
        },
    })
}

/// Moves one recurrent slot to the device.
fn import_recurrent(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    roles: &RecurrentRoles<'_>,
    pack: bool,
) -> CoreResult<Recurrent> {
    Ok(Recurrent {
        qkv: weight(backend, file, &roles.qkv)?,
        gate: weight(backend, file, &roles.gate)?,
        alpha: projection(backend, file, &roles.alpha, pack)?,
        beta: projection(backend, file, &roles.beta, pack)?,
        out: weight(backend, file, &roles.out)?,
        conv1d: vector(backend, file, &roles.conv1d)?,
        a: vector(backend, file, &roles.a)?,
        dt_bias: vector(backend, file, &roles.dt_bias)?,
        norm: vector(backend, file, &roles.norm)?,
    })
}

/// Moves the n-gram block to the device.
fn import_ngram(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    roles: &NgramRoles<'_>,
    pack: bool,
) -> CoreResult<NgramBlock> {
    Ok(NgramBlock {
        key: projection(backend, file, &roles.key, pack)?,
        value: projection(backend, file, &roles.value, pack)?,
        norm_key: vector(backend, file, &roles.norm_key)?,
        norm_query: vector(backend, file, &roles.norm_query)?,
        norm_conv: vector(backend, file, &roles.norm_conv)?,
        conv1d: vector(backend, file, &roles.conv1d)?,
    })
}

/// Imports one projection matrix, refusing a format this family does not ship.
///
/// Two checks, and they are not the same one twice. This list is the family's
/// own: these are the formats a checkpoint of this family is expected to store,
/// and anything else means the file is not what it claims. The engine then
/// refuses, separately, a format for which no projection exists in this build. A
/// file can pass the first and fail the second - that is a build that cannot run
/// a legitimate checkpoint, which is a different problem and deserves a
/// different message.
fn weight(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    view: &TensorView<'_>,
) -> CoreResult<Weight> {
    if !SUPPORTED.contains(&view.dtype) {
        return Err(CoreError::Unsupported {
            what: "a weight stored in a format this family's loader does not accept",
            backend: "qwen4exp",
        });
    }
    // A third dimension is the routed expert bank: `[k, n, E]` is E matrices of
    // the same shape in one tensor, not a matrix of a different shape. Reading
    // only the first two would size the import at one expert's bytes and refuse
    // the tensor - which is exactly what it did.
    let dims = view.shape.dims();
    let in_dim = extent(dims.first().copied().unwrap_or(0))?;
    let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
    let slots = extent(dims.get(2).copied().unwrap_or(1))?;
    let range = range_of(file, view)?;
    let weight = Weight::import(
        backend,
        view.dtype,
        in_dim,
        out_dim,
        slots,
        file.mapping(),
        range.clone(),
    )?;
    release(backend, file, range);
    Ok(weight)
}

/// Imports one projection matrix, repacking a half-precision one to twelve
/// bits a weight when `pack` asks for it (see
/// [`shiso_engine::sparse::pack12`]): lossless, and a quarter fewer bytes for
/// every decode step to stream. A matrix the packing does not cover is
/// imported as it is stored.
fn projection(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    view: &TensorView<'_>,
    pack: bool,
) -> CoreResult<Weight> {
    let dims = view.shape.dims();
    if pack && view.dtype == DType::F16 && dims.len() == 2 {
        let in_dim = extent(dims.first().copied().unwrap_or(0))?;
        let out_dim = extent(dims.get(1).copied().unwrap_or(1))?;
        let range = range_of(file, view)?;
        let mapping = file.mapping();
        let stored =
            AsRef::<[u8]>::as_ref(&*mapping)
                .get(range.clone())
                .ok_or(CoreError::OutOfBounds {
                    offset: range.start as u64,
                    len: (range.end - range.start) as u64,
                    buffer_bytes: 0,
                })?;
        if let Some(packed) = Weight::packed_half(backend, in_dim, out_dim, stored)? {
            release(backend, file, range);
            return Ok(packed);
        }
    }
    weight(backend, file, view)
}

/// Hands an imported tensor's file pages back to the kernel, where the device
/// copied them.
///
/// The checkpoint is 105 GB and its weights 62; kept cached beside the device's
/// copies, the file's pages are what the next allocation has to reclaim, which
/// measured on a GB10 as 17 s of a 48 s load spent inside the allocator.
fn release(backend: &Arc<dyn Backend>, file: &GgufFile, range: std::ops::Range<usize>) {
    if backend.imports_copy() {
        file.release_cached(range);
    }
}

/// Imports one weight vector, which every format stores as plain `f32`.
fn vector(
    backend: &Arc<dyn Backend>,
    file: &GgufFile,
    view: &TensorView<'_>,
) -> CoreResult<DeviceBuffer> {
    if view.dtype != DType::F32 {
        return Err(CoreError::Unsupported {
            what: "a norm or bias vector stored as anything but single precision",
            backend: "qwen4exp",
        });
    }
    let range = range_of(file, view)?;
    let buffer = import_host(backend, file.mapping(), range.clone())?;
    release(backend, file, range);
    Ok(buffer)
}

/// The file range one bound tensor occupies.
fn range_of(file: &GgufFile, view: &TensorView<'_>) -> CoreResult<std::ops::Range<usize>> {
    let desc = file.find(view.name).ok_or(CoreError::InvalidValue {
        key: "tensor name",
        value: format!("{} was bound but is not in the directory", view.name),
    })?;
    Ok(file.range_of(desc))
}

/// Narrows one file dimension to the width a launch counts in.
fn extent(dim: u64) -> CoreResult<u32> {
    u32::try_from(dim).map_err(|_| CoreError::ShapeMismatch {
        field: "tensor extent",
        expected: u64::from(u32::MAX),
        got: dim,
    })
}

/// Presents a checkpoint-file failure in the core vocabulary.
fn open_failure(source: GgufError) -> CoreError {
    CoreError::Backend {
        backend: "qwen4exp",
        op: "load",
        source: Box::new(source),
    }
}

impl Model for Qwen4ExpModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.stack.open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack.close_session(session)
    }

    /// Advances the session by a prompt chunk that produces no logits.
    ///
    /// A chunk wider than one pass is split by the stack rather than refused,
    /// and the split is invisible in the result: the recurrent state, the
    /// n-gram window and the cache carry the prefix across an internal
    /// boundary exactly as across a caller's.
    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.stack.prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.stack.decode_batch(steps)
    }

    /// The stack's batched prompt pass, which refuses everything but scoring
    /// tails it can read byte-for-byte as their solo reads.
    fn prefill_last_many(&mut self, runs: &[PromptRun<'_>]) -> CoreResult<Logits<'_>> {
        let pairs = runs
            .iter()
            .map(|run| (run.session, run.tokens))
            .collect::<Vec<_>>();
        self.stack.prefill_segments(&pairs)
    }

    fn spec_mut(&mut self) -> Option<&mut dyn Speculating> {
        // The head is what makes a session speculative, and a build without one
        // answers no here rather than one verify pass later.
        self.stack.has_draft_head().then_some(self)
    }

    /// The stack masks every logits row on the device right after the
    /// vocabulary projection, so the list goes to it: the verify's device
    /// pick and every row the scheduler reads back then both skip it.
    fn suppress_tokens(&mut self, ids: &[TokenId]) -> CoreResult<()> {
        self.stack.suppress_tokens(ids)
    }

    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.stack.teardown()
    }
}

/// The media path, which this family has because its stack carries one.
///
/// One method and no chunked sibling, for the reason the contract states: the
/// stack's walk over a prompt's placeholder runs is built per call, so a prompt
/// handed over in pieces would restart the running multimodal position and the
/// encoder row the splice has reached. The stack still splits the prompt into
/// device-sized chunks underneath, sharing one walk across them, which is what
/// makes the single call cost no more than the chunked one would have.
///
/// The tower itself is not this crate's. This family's checkpoint carries one
/// under the same `vision_tower` prefix and at the same widths as the dense
/// hybrid's, so the server opens it through that family's loader; what is this
/// family's is the seam below, which is the stack's.
impl MediaModel for Qwen4ExpModel {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.stack.prefill_last_media(session, tokens, media)
    }
}

/// The speculative path, which this family has exactly when a draft head was
/// loaded beside the checkpoint.
impl Speculating for Qwen4ExpModel {
    fn speculate(&mut self, session: SessionId, first: TokenId) -> CoreResult<Speculated> {
        let eos = self.info.stop_tokens.primary();
        self.stack.speculate(session, first, eos)
    }

    fn speculate_batch(&mut self, steps: &[(SessionId, TokenId)]) -> CoreResult<Vec<Speculated>> {
        let eos = self.info.stop_tokens.primary();
        match steps {
            [(session, first)] => Ok(vec![self.stack.speculate(*session, *first, eos)?]),
            _ => self.stack.speculate_batch(steps, eos),
        }
    }
}

/// Narrows the draft head's vocabulary projection to the blocks `counts` says
/// are worth keeping.
///
/// Reported rather than refused when the counts do not narrow anything: a list
/// that covers the whole head is an operator's file being wider than they
/// thought, and drafting through the whole head is the answer that still works.
fn narrow_draft_head(
    backend: &Arc<dyn Backend>,
    weights: &mut SparseWeights,
    counts: &Path,
    vocab: u32,
) -> CoreResult<()> {
    let text = std::fs::read_to_string(counts).map_err(|source| CoreError::Backend {
        backend: "qwen4exp",
        op: "reading the draft vocabulary counts",
        source: source.to_string().into(),
    })?;
    let blocks = headslice::blocks_from_counts(&text, vocab, headslice::DEFAULT_IDS);
    if blocks.is_empty() {
        // No logger in this crate: the caller that supplied the path is the one
        // that can say so, and it gets the whole head - which works.
        return Ok(());
    }
    let slice = headslice::build(backend, weights.output_matrix(), &blocks)?;
    if let Some(head) = weights.draft.as_mut() {
        head.head = Some(slice);
    }
    Ok(())
}

/// The two exports whose routed experts are always streamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamedExport {
    /// Brain-float dense weights, block-scaled FP8 experts.
    Reference,
    /// The same layout with NVFP4 experts, NVIDIA's `modelopt` export.
    Nvfp4,
}

impl StreamedExport {
    /// What the import makes of the export's dense matrices and banks.
    const fn layout(self) -> crate::qwen4exp::fp8::ImportLayout {
        match self {
            Self::Reference => crate::qwen4exp::fp8::ImportLayout::REFERENCE,
            Self::Nvfp4 => crate::qwen4exp::fp8::ImportLayout {
                dense: crate::qwen4exp::fp8::DenseFormat::Half,
                experts: DType::Nvfp4Gs16,
            },
        }
    }

    /// Bytes one routed expert occupies in the pool.
    fn slot_bytes(self) -> u64 {
        match self {
            Self::Reference => crate::qwen4exp::fp8::slot_bytes(),
            Self::Nvfp4 => crate::qwen4exp::nvfp4::slot_bytes(),
        }
    }

    /// Bytes of every layer's routed banks.
    fn expert_bytes(self) -> u64 {
        match self {
            Self::Reference => crate::qwen4exp::fp8::expert_bytes(),
            Self::Nvfp4 => crate::qwen4exp::nvfp4::expert_bytes(),
        }
    }
}

/// How a streamed-expert export is loaded: its pool, and whether its draft
/// head is bound.
#[derive(Debug, Clone, Copy)]
pub struct StreamedLoad<'a> {
    /// Slots in the routed pool; `None` sizes it to the device.
    pub slots: Option<u32>,
    /// Bytes the sizing leaves free beyond the dense weights.
    pub reserve: u64,
    /// Whether the export's own draft head is imported.
    pub draft: bool,
    /// The draft vocabulary's counts file, which narrows the head's output.
    pub counts: Option<&'a Path>,
}
