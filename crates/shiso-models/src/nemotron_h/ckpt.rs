//! Prefix checkpoints for this family: the stack's own store, under this
//! family's tag.

use shiso_core::error::CoreResult;
use shiso_core::ids::{SessionId, TokenId};
use shiso_engine::CacheStats;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_engine::{CkptIdentity, PumpReport, Reason};

use crate::nemotron_h::model::NemotronHModel;

/// The tag a checkpoint written by this family carries: the one after the
/// Qwen 3.5 mixture-of-experts family's 8.
pub const NEMOTRON_H_MODEL_ID: u8 = 9;

/// Quantization width the tag records: every matrix of the container is
/// 4-bit affine.
const QUANT_BITS: u8 = 4;

impl NemotronHModel {
    /// How a checkpoint written by this loaded model is tagged.
    #[must_use]
    pub const fn ckpt_identity(&self) -> CkptIdentity {
        CkptIdentity::new(NEMOTRON_H_MODEL_ID, QUANT_BITS)
    }
}

impl Checkpointed for NemotronHModel {
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

    fn kv_set_hints(&mut self, session: SessionId, hints: &[u32]) -> CoreResult<()> {
        self.stack_mut().ckpt_set_hints(session, hints)
    }

    fn kv_set_scoring(&mut self, session: SessionId) -> CoreResult<()> {
        self.stack_mut().ckpt_set_scoring(session)
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
