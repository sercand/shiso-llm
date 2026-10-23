//! Prefix checkpoints for this family: the stack's own store, under this
//! family's tag.

use shiso_core::error::CoreResult;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::media::MediaKey;
use shiso_engine::CacheStats;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_engine::{CkptIdentity, PumpReport, Reason};

use crate::llama::model::LlamaModel;

/// The tag a checkpoint written by this family carries: the one after LFM2's 10.
pub const LLAMA_MODEL_ID: u8 = 11;

impl LlamaModel {
    /// How a checkpoint written by this loaded model is tagged: the family,
    /// and the width its feed-forward was stored at, so an `f16` export and a
    /// `q8_0` one never resume each other's rows.
    #[must_use]
    pub const fn ckpt_identity(&self) -> CkptIdentity {
        CkptIdentity::new(LLAMA_MODEL_ID, self.quant_bits())
    }
}

impl Checkpointed for LlamaModel {
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

    fn kv_set_media_key(
        &mut self,
        session: SessionId,
        prompt: &[TokenId],
        key: MediaKey<'_>,
    ) -> CoreResult<()> {
        self.stack_mut().ckpt_set_media_key(session, prompt, key)
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
