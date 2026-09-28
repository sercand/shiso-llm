//! What this family tells the checkpoint tier about itself.
//!
//! The tier's mechanism - the payload, the prefix scan, the save and the resume -
//! belongs to [`shiso_engine`], because none of it is a fact about one checkpoint
//! family. Two things are, and they are here: which tag a stored file is written
//! under, and how wide the weights that produced its rows were.
//!
//! # Why this file exists at all
//!
//! `shiso_server::model_factory::Loaded` carries a `Box<dyn Checkpointed>`, not
//! a `Box<dyn Model>`, and that is deliberate: every family this build serves
//! can store and resume a prefix, and the module that says so also says what
//! the alternative would cost - "a family that genuinely cannot checkpoint is a
//! reason to widen this into an enum of the two, not a reason to narrow every
//! arm to the weaker contract". This family is not that family. Its stack is
//! [`shiso_engine::HybridStack`], the same type the dense hybrid's impl
//! forwards to, so the whole of the contract below is the same delegation that
//! family already wrote, and the enum stays unwidened.
//!
//! Both facts are derived, never configured. A tag that a deployment could set
//! would let two families claim the same one in a shared directory, and a
//! quantization that a caller could state would let it be stated wrong - at
//! which point a 2-bit build reads a 4-bit build's key/value rows and answers
//! fluently from numbers that were never computed for its weights.

use shiso_core::DType;
use shiso_core::error::CoreResult;
use shiso_core::ids::{SessionId, TokenId};
use shiso_engine::CacheStats;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_engine::{CkptIdentity, PumpReport, Reason};

use crate::qwen35moe::model::Qwen35MoeModel;

/// The tag a checkpoint written by this family carries.
///
/// A shared cache directory stays unambiguous only while no two families claim
/// the same tag, so the low ids are treated as taken by the on-disk format that
/// predates this crate, the dense family took 5, the sparse family 6, Gemma 4
/// 7, and this family takes the one after. It is a compile-time constant
/// because it identifies the *shape* the bytes assume, not the file they came
/// from: two checkpoints of this family have the same tag and a different
/// digest.
pub const QWEN35MOE_MODEL_ID: u8 = 8;

/// The storage every routed expert of this family is in.
///
/// The sibling families read this back off a loaded weight, because theirs is a
/// property of the particular container an operator pointed at - a `Q4_K` GGUF
/// and a `Q8_0` one are both that family. This family has no such choice to
/// read: `Qwen35MoeModel::load` refuses, by the offending tensor's name, any
/// expert that is not NVFP4, so a loaded model of this family that is in any
/// other storage does not exist. The constant is therefore the same derivation
/// the siblings make, taken at the point where it is actually decided.
const EXPERT_STORAGE: DType = DType::Nvfp4Gs16;

/// Quantization width a checkpoint's key/value rows are stored under, read off
/// the storage the weights that produce them are in.
///
/// Four-bit formats report four and everything else reports two, which is the
/// split the disk tier's own header accepts - a build whose weights are neither
/// ends up unable to checkpoint at all, which is the right outcome for bytes
/// nothing downstream can check. NVFP4 is four-bit codes under a shared `e4m3`
/// byte per group of sixteen, so it reports four; FP8, which is what this
/// family's *dense* projections are in, is neither and reports two.
#[must_use]
pub const fn quant_bits_of(gate: DType) -> u8 {
    match gate {
        DType::Nvfp4Gs16 | DType::Q4_0 | DType::Q4_K | DType::IQ4_NL => 4,
        _ => 2,
    }
}

impl Qwen35MoeModel {
    /// How a checkpoint written by this loaded model is tagged.
    #[must_use]
    pub const fn ckpt_identity(&self) -> CkptIdentity {
        CkptIdentity::new(QWEN35MOE_MODEL_ID, quant_bits_of(EXPERT_STORAGE))
    }
}

impl Checkpointed for Qwen35MoeModel {
    fn kv_cache_stats(&self) -> Option<CacheStats> {
        self.stack().cache_stats()
    }

    fn kv_cache(&self) -> Option<std::sync::Arc<shiso_engine::KvCache>> {
        self.stack().cache()
    }

    fn kv_enable(&mut self, config: &CkptConfig) -> CoreResult<()> {
        let identity = self.ckpt_identity();
        self.stack_mut().ckpt_enable(config, identity)
    }

    fn kv_checkpoint_enabled(&self) -> bool {
        self.stack().ckpt_enabled()
    }

    // `Ok(None)` is the trait's cold start, not a refusal: this impl forwards the
    // stack's answer without narrowing it.
    fn kv_resume(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        self.stack_mut().ckpt_resume(session, prompt)
    }

    fn kv_save(&mut self, session: SessionId, reason: Reason) -> CoreResult<SaveOutcome> {
        self.stack_mut().ckpt_save(session, reason)
    }

    fn kv_pump(&mut self, max_bytes: u64) -> CoreResult<PumpReport> {
        self.stack().ckpt_pump(max_bytes)
    }

    fn kv_drain(&mut self, timeout: std::time::Duration) -> bool {
        self.stack().ckpt_drain(timeout)
    }

    fn kv_observe_prefill(&mut self, tokens: u32, seconds: f64) {
        if let Some(cache) = self.stack().cache() {
            cache.observe_prefill(tokens, seconds);
        }
    }

    fn kv_shutdown_sweep(&mut self) -> usize {
        self.stack_mut().ckpt_shutdown_sweep()
    }
}

#[cfg(test)]
mod tests {
    use super::{EXPERT_STORAGE, QWEN35MOE_MODEL_ID, quant_bits_of};
    use shiso_core::DType;

    #[test]
    fn nvfp4_reports_four_and_this_family_s_dense_storage_reports_two() {
        assert_eq!(quant_bits_of(DType::Nvfp4Gs16), 4);
        assert_eq!(quant_bits_of(DType::Q4_K), 4);
        assert_eq!(quant_bits_of(DType::Fp8E4M3), 2);
        assert_eq!(quant_bits_of(DType::F16), 2);
    }

    /// The tag this family writes is the one its own loader's refusal implies.
    ///
    /// If the loader ever gains a second expert storage, this is the assertion
    /// that has to be revisited rather than a silent 4 that is now sometimes
    /// wrong.
    #[test]
    fn the_width_is_the_one_the_loader_refuses_anything_but() {
        assert_eq!(EXPERT_STORAGE, DType::Nvfp4Gs16);
        assert_eq!(quant_bits_of(EXPERT_STORAGE), 4);
    }

    #[test]
    fn no_two_families_claim_the_same_tag() {
        // The same check `crates/shiso-models/src/qwen4exp/ckpt.rs` makes over
        // the set that existed when it was written, repeated here over the set
        // that exists now. It is over the whole set rather than this family
        // against the others for the reason that file gives: a per-family test
        // can only say that its own family is distinct from the ones that
        // existed when it was written, which is exactly the check a newly added
        // family skips.
        let ids = [
            crate::qwen35::QWEN35_MODEL_ID,
            crate::qwen4exp::QWEN4EXP_MODEL_ID,
            crate::gemma4::GEMMA4_MODEL_ID,
            QWEN35MOE_MODEL_ID,
            crate::nemotron_h::NEMOTRON_H_MODEL_ID,
        ];
        for (rank, id) in ids.iter().enumerate() {
            for other in &ids[rank + 1..] {
                assert_ne!(id, other, "two families claim tag {id}");
            }
        }
    }
}
