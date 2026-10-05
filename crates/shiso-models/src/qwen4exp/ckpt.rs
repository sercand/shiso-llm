//! What this family tells the checkpoint tier about itself.
//!
//! The tier's mechanism - the payload, the prefix scan, the save and the resume -
//! belongs to the engine, because none of it is a fact about one checkpoint
//! family. Two things are, and they are here: which tag a stored file is written
//! under, and how wide the weights that produced its rows were.
//!
//! Both are derived, never configured. A tag that a deployment could set would
//! let two families claim the same one in a shared directory, and a quantization
//! that a caller could state would let it be stated wrong - at which point a
//! build reads another build's key/value rows and answers fluently from numbers
//! that were never computed for its weights.
//!
//! # What the engine answers and what this file adds
//!
//! The `Checkpointed` bodies below forward to
//! [`shiso_engine::sparse::ckpt`], which is the sparse stack's own half of the
//! tier: the blob's content, the commit that registers a span, and the resume
//! that stands one back up. Nothing about the *policy* is restated - the floor,
//! the prefix scan and the refusal vocabulary are the dense family's and this
//! family's alike, and are [`shiso_engine::ckpt`]'s.

use shiso_core::DType;
use shiso_core::error::CoreResult;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::media::MediaKey;
use shiso_engine::CacheStats;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_engine::{CkptIdentity, PumpReport, Reason};

use crate::qwen4exp::model::Qwen4ExpModel;

/// The tag a checkpoint written by this family carries.
///
/// A shared cache directory stays unambiguous only while no two families claim
/// the same tag, so the low ids are treated as taken by the on-disk format that
/// predates this crate, the dense family took the next one, and this family
/// takes the one after. It is a compile-time constant because it identifies the
/// *shape* the bytes assume, not the file they came from: two checkpoints of
/// this family have the same tag and a different digest.
pub const QWEN4EXP_MODEL_ID: u8 = 6;

/// Quantization width a checkpoint's key/value rows are stored under, read off
/// the weights that produce them.
///
/// The layer-zero routed expert gate is the witness: it is present in every slot
/// of every checkpoint of this family, and its format is the one the bulk of the
/// weights are stored in. Four-bit formats report four and everything else
/// reports two, which is the split the disk tier's own header accepts - a build
/// whose weights are neither ends up unable to checkpoint at all, which is the
/// right outcome for bytes nothing downstream can check.
#[must_use]
pub const fn quant_bits_of(gate: DType) -> u8 {
    match gate {
        DType::Nvfp4Gs16 | DType::Q4_0 | DType::Q4_K | DType::IQ4_NL | DType::Exl3K4 => 4,
        _ => 2,
    }
}

impl Qwen4ExpModel {
    /// How a checkpoint written by this loaded model is tagged.
    #[must_use]
    pub const fn ckpt_identity(&self) -> CkptIdentity {
        CkptIdentity::new(QWEN4EXP_MODEL_ID, quant_bits_of(self.expert_gate_dtype()))
    }
}

impl Checkpointed for Qwen4ExpModel {
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

    fn kv_set_hints(&mut self, session: SessionId, hints: &[u32]) -> CoreResult<()> {
        self.stack_mut().ckpt_set_hints(session, hints)
    }

    fn kv_set_scoring(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack_mut().ckpt_set_scoring(session)
    }

    fn kv_set_media_key(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        key: MediaKey<'_>,
    ) -> CoreResult<()> {
        self.stack_mut().ckpt_set_media_key(session, prompt, key)
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
    use super::{QWEN4EXP_MODEL_ID, quant_bits_of};
    use shiso_core::DType;

    #[test]
    fn a_four_bit_expert_gate_reports_four_and_everything_else_reports_two() {
        assert_eq!(quant_bits_of(DType::Exl3K4), 4);
        assert_eq!(quant_bits_of(DType::Q4_K), 4);
        assert_eq!(quant_bits_of(DType::Exl3K5), 2);
        assert_eq!(quant_bits_of(DType::Q8_0), 2);
    }

    #[test]
    fn no_two_families_claim_the_same_tag() {
        // A shared cache directory is only unambiguous while every tag differs,
        // and a cross-load would answer from another architecture's rows without
        // any sign of trouble. Asserted here, over the whole set, rather than
        // once per family: a per-family test can only ever say that the family
        // it belongs to is distinct from the ones that existed when it was
        // written, which is exactly the check a newly added family skips.
        let ids = [
            crate::qwen35::QWEN35_MODEL_ID,
            QWEN4EXP_MODEL_ID,
            crate::gemma4::GEMMA4_MODEL_ID,
            crate::qwen35moe::QWEN35MOE_MODEL_ID,
        ];
        for (rank, id) in ids.iter().enumerate() {
            for other in &ids[rank + 1..] {
                assert_ne!(id, other, "two families claim tag {id}");
            }
        }
    }
}
