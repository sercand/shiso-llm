//! What this family tells the checkpoint tier about itself.
//!
//! The tier's mechanism - the payload, the prefix scan, the save and the resume -
//! belongs to [`shiso_engine`], because none of it is a fact about one checkpoint
//! family. Two things are, and they are here: which tag a stored file is written
//! under, and how wide the weights that produced its rows were.
//!
//! Both are derived, never configured. A tag that a deployment could set would let
//! two families claim the same one in a shared directory, and a quantization that a
//! caller could state would let it be stated wrong - at which point a 2-bit build
//! reads a 4-bit build's key/value rows and answers fluently from numbers that were
//! never computed for its weights.

use shiso_core::DType;
use shiso_core::error::CoreResult;
use shiso_core::ids::{SessionId, TokenId};
use shiso_engine::CacheStats;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_engine::{CkptIdentity, PumpReport, Reason};

use crate::qwen35::model::Qwen35Model;

/// The tag a checkpoint written by this family carries.
///
/// A shared cache directory stays unambiguous only while no two families claim the
/// same tag, so the low ids are treated as taken by the on-disk format that predates
/// this crate and this family takes the next one. It is a compile-time constant
/// because it identifies the *shape* the bytes assume, not the file they came from:
/// two checkpoints of this family have the same tag and a different digest.
pub const QWEN35_MODEL_ID: u8 = 5;

/// Quantization width a checkpoint's key/value rows are stored under, read off the
/// weights that produce them.
///
/// The layer-zero feed-forward gate is the witness: it is present in every slot of
/// every checkpoint of this family, and its format is the one the bulk of the weights
/// are stored in. Four-bit formats report four and everything else reports two, which
/// is the split the disk tier's own header accepts - a build whose weights are
/// neither ends up unable to checkpoint at all, which is the right outcome for bytes
/// nothing downstream can check.
#[must_use]
pub const fn quant_bits_of(gate: DType) -> u8 {
    match gate {
        DType::Nvfp4Gs16 | DType::Q4_0 | DType::Q4_K | DType::IQ4_NL => 4,
        _ => 2,
    }
}

impl Qwen35Model {
    /// How a checkpoint written by this loaded model is tagged.
    #[must_use]
    pub const fn ckpt_identity(&self) -> CkptIdentity {
        CkptIdentity::new(QWEN35_MODEL_ID, quant_bits_of(self.ffn_gate_dtype()))
    }
}

impl Checkpointed for Qwen35Model {
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
    use super::quant_bits_of;
    use shiso_core::DType;

    #[test]
    fn a_four_bit_gate_reports_four_and_everything_else_reports_two() {
        // The two widths the disk tier accepts, and the rule that chooses between
        // them. A checkpoint whose gate is neither still reports a width the header
        // accepts, which is what keeps the decision out of the store.
        assert_eq!(quant_bits_of(DType::Q4_K), 4);
        assert_eq!(quant_bits_of(DType::Q4_0), 4);
        assert_eq!(quant_bits_of(DType::IQ4_NL), 4);
        assert_eq!(quant_bits_of(DType::Nvfp4Gs16), 4);
        assert_eq!(quant_bits_of(DType::Q2_K), 2);
        assert_eq!(quant_bits_of(DType::Q8_0), 2);
        assert_eq!(quant_bits_of(DType::F16), 2);
    }
}
