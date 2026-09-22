//! The cross-crate type contracts, named once and checked by the compiler.
//!
//! Every contract here is a type alias plus one `const _` that assigns the real
//! function to it. Nothing is executed that could pass for the wrong reason: if
//! a signature drifts, this file stops compiling, and the failure names the
//! seam rather than a downstream symptom three crates away.
//!
//! # Why a file like this exists
//!
//! The plan review that produced the workspace rules found four separate cases
//! of two task plans naming the same thing differently - a cache type spelled
//! two ways, prompt tokens typed `&[i32]` on one side of a seam and `&[u32]` on
//! the other, three incompatible shapes of device view, and a sampler whose
//! parameters had split into a parameter struct plus a separate seed. Each was
//! found by reading, which is the expensive way. A compile-time contract is the
//! cheap way, and it is what stops the fifth.
//!
//! # What is asserted
//!
//! - **One token type.** `TokenId` is the only spelling of a token on every
//!   seam it crosses: the vocabulary that produces it, the model that prefills
//!   and decodes with it, the checkpoint trait that saves and resumes paths of
//!   it, and the cache that keys on it. A cast anywhere along that path would
//!   have to change one of the signatures below.
//! - **Object safety.** `Model`, `Backend` and `Checkpointed` stay reachable as
//!   `dyn`, because the server holds boxed models and buffers hold a shared
//!   backend. A default method with a generic parameter would take that away,
//!   and the first sign of it would be a scheduler that no longer compiles.
//! - **One device-view shape.** Every byte movement speaks `DeviceView`,
//!   including the logits row a sampler reads.
//! - **One sampling shape.** Draw parameters and generator state are two
//!   arguments, not one bundle and not three.
//! - **The IR counts tokens, it does not name them.** The neutral layer carries
//!   text and integer counts; the single place text becomes a `TokenId` is the
//!   server's vocabulary. That is what keeps a token id from acquiring a second
//!   representation on the wire side.
//! - **The cache API is the one its callers call.** Plan, look up, commit - so
//!   a rename in the store surfaces here rather than as a rewrite in three
//!   callers.

use shiso_core::error::CoreResult;
use shiso_core::ids::{Seed, SessionId, TokenId};
use shiso_core::model::{DecodeStep, Logits, SamplingParams};
use shiso_core::{Backend, DeviceView, Model};
use shiso_engine::{Checkpointed, Reason, ResumeInfo, SaveOutcome};
use shiso_ir::Usage;
use shiso_kvstore::{CommitOpts, Frontier, KvCache, KvError, Lookup, Resume};
use shiso_server::tokenizer::{TokenizerError, Vocabulary};

// ---------------------------------------------------------------------------
// object safety
// ---------------------------------------------------------------------------
// Writing `dyn Trait` in a type position is the whole assertion: a trait that
// has stopped being dyn-compatible cannot be written there at all.

const _: Option<&dyn Model> = None;
const _: Option<&dyn Backend> = None;
const _: Option<&dyn Checkpointed> = None;

// ---------------------------------------------------------------------------
// TokenId: vocabulary -> model -> checkpoint -> cache
// ---------------------------------------------------------------------------

/// Text in, token ids out. This is the one place in the workspace where that
/// conversion happens, and the ids it yields are the core newtype rather than a
/// raw integer the next stage would have to reinterpret.
type Encode = fn(&Vocabulary, &str, bool) -> Result<Vec<TokenId>, TokenizerError>;
const _: Encode = Vocabulary::encode;

/// The way back, over the same token type.
type Decode = fn(&Vocabulary, &[TokenId]) -> Result<String, TokenizerError>;
const _: Decode = Vocabulary::decode;

/// Prefill takes the tokens the vocabulary produced, unconverted.
type PrefillChunk = fn(&mut (dyn Model + 'static), SessionId, &[TokenId]) -> CoreResult<()>;
const _: PrefillChunk = <dyn Model>::prefill_chunk;

/// The prefill that also yields the frontier row, borrowed from the model.
type PrefillLast =
    for<'m> fn(&'m mut (dyn Model + 'static), SessionId, &[TokenId]) -> CoreResult<Logits<'m>>;
const _: PrefillLast = <dyn Model>::prefill_last;

/// A decode batch is a slice of steps, each one session plus one token - not a
/// pair of parallel slices a caller could get out of step.
type DecodeBatch =
    for<'m> fn(&'m mut (dyn Model + 'static), &[DecodeStep]) -> CoreResult<Logits<'m>>;
const _: DecodeBatch = <dyn Model>::decode_batch;

/// A step is read apart rather than built: `SessionId` has no public
/// constructor by design, so the field types are pinned by a projection the
/// compiler has to type-check.
fn decode_step_fields(step: DecodeStep) -> (SessionId, TokenId) {
    (step.session, step.token)
}
const _: fn(DecodeStep) -> (SessionId, TokenId) = decode_step_fields;

/// A resume borrows from the model it stood up, which is why the returned
/// lifetime is the receiver's.
type KvResume = for<'c> fn(
    &'c mut (dyn Checkpointed + 'static),
    SessionId,
    &[u8],
) -> CoreResult<Option<ResumeInfo<'c>>>;
const _: KvResume = <dyn Checkpointed>::kv_resume;

/// A save is keyed by the token path the session stands on, in the same type.
type KvSave = fn(
    &mut (dyn Checkpointed + 'static),
    SessionId,
    &[TokenId],
    &[u8],
    Reason,
) -> CoreResult<SaveOutcome>;
const _: KvSave = <dyn Checkpointed>::kv_save;

/// The shutdown sweep asks its caller for each session's text and token path,
/// because the engine has no tokenizer of its own.
type KvSweep = fn(
    &mut (dyn Checkpointed + 'static),
    &mut dyn FnMut(SessionId) -> Option<(Vec<TokenId>, Vec<u8>)>,
) -> usize;
const _: KvSweep = <dyn Checkpointed>::kv_shutdown_sweep;

/// What a resume hands back: an owned path in the same token type, plus the two
/// counts a caller needs to know how much prefill it does not owe.
fn resume_info_fields(info: ResumeInfo<'_>) -> (u32, u64, Vec<TokenId>) {
    (info.tokens_reused, info.text_bytes, info.tokens)
}
const _: fn(ResumeInfo<'_>) -> (u32, u64, Vec<TokenId>) = resume_info_fields;

/// What the cache would reuse for a prompt, without taking anything.
type Plan = fn(&KvCache, &[TokenId], f64) -> Lookup;
const _: Plan = KvCache::plan;

/// The taking version, keyed on the same tokens.
type CacheLookup = fn(&KvCache, &[TokenId], f64) -> Result<Option<Resume>, KvError>;
const _: CacheLookup = KvCache::lookup;

/// Writing a span back, again keyed on the same tokens.
type Commit =
    fn(&KvCache, &mut Frontier, &[TokenId], u32, CommitOpts<'static>, f64) -> Result<(), KvError>;
const _: Commit = KvCache::commit;

// ---------------------------------------------------------------------------
// the shapes that had drifted before
// ---------------------------------------------------------------------------

/// One device view for every byte movement. Three shapes of this is the defect
/// these three lines pin shut.
type CopyIn = fn(&(dyn Backend + 'static), DeviceView<'_>, &[u8]) -> CoreResult<()>;
const _: CopyIn = <dyn Backend>::copy_h2d;

/// The read-back half.
type CopyOut = fn(&(dyn Backend + 'static), DeviceView<'_>, &mut [u8]) -> CoreResult<()>;
const _: CopyOut = <dyn Backend>::copy_d2h;

/// And device to device, in the same currency.
type CopyAcross = fn(&(dyn Backend + 'static), DeviceView<'_>, DeviceView<'_>) -> CoreResult<()>;
const _: CopyAcross = <dyn Backend>::copy_d2d;

/// A logits row is that same view, so a sampler reads what a copy wrote.
type LogitsRow = fn(&Logits<'static>, u32) -> CoreResult<DeviceView<'static>>;
const _: LogitsRow = Logits::row;

/// Draw parameters and generator state stay two arguments. A single bundle
/// would mean the seed travelled inside the parameters, which is how a
/// "deterministic" run quietly starts sharing a stream between requests.
type SampleLogits = fn(&[f32], &SamplingParams, &mut Seed) -> TokenId;
const _: SampleLogits = shiso_engine::sample_logits;

/// The greedy draw takes no parameters and no seed at all, which is what makes
/// it the reproducible one.
const _: fn(&[f32]) -> TokenId = shiso_engine::argmax;

/// Greediness is a property of the parameters, asked of them rather than
/// tracked beside them.
const _: fn(&SamplingParams) -> bool = SamplingParams::is_greedy;

/// The neutral layer counts tokens and never names one. A `TokenId` appearing
/// in the IR would be a second public representation of a token, and the wire
/// adapters would start converting between them.
fn usage_counts(usage: &Usage) -> (u64, u64, u64, u64) {
    (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.cached_prompt_tokens,
        usage.reasoning_tokens,
    )
}
const _: fn(&Usage) -> (u64, u64, u64, u64) = usage_counts;

/// The contracts above are checked by the compiler, so this file building at
/// all is the result. The test exists so a target that stopped being built
/// shows up as a missing test rather than as silence.
#[test]
fn the_contracts_above_are_compiled() {
    assert_eq!(u32::from(TokenId::new(7)), 7, "TokenId is a u32 newtype");
    assert_eq!(
        usage_counts(&Usage::default()),
        (0, 0, 0, 0),
        "a fresh usage block counts nothing"
    );
}

/// The rig opens whatever device this machine has.
///
/// Bound to the platform rather than to a backend: this file's own history is
/// three lanes that bound `MetalBackend` and therefore skipped, in silence, on
/// the only machine that had a device.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip is opt-in by environment, so the reason it did not run belongs on stderr"
)]
fn the_rig_opens_the_device_this_platform_has() {
    let Ok(model) = std::env::var("SHISO_TEST_MODEL") else {
        eprintln!("SKIP: SHISO_TEST_MODEL names no checkpoint");
        return;
    };
    let path = std::path::PathBuf::from(model);
    if !path.exists() {
        eprintln!("SKIP: {} does not exist", path.display());
        return;
    }
    let rig = shiso_bench::decode::Rig::open(&path);
    assert!(
        rig.is_ok(),
        "the rig must open on this platform, got {:?}",
        rig.err()
    );
}

/// The `MoE` rig opens the 23.4 GB checkpoint and takes one step.
///
/// `--ignored`, because it moves 23.4 GB and this machine kills long loads
/// under memory pressure.
#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
#[expect(
    clippy::print_stderr,
    reason = "the skip is opt-in by environment, so the reason it did not run belongs on stderr"
)]
fn the_moe_rig_opens_and_steps() {
    let Ok(dir) = std::env::var("SHISO_TEST_MODEL_QWEN35MOE") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_QWEN35MOE names no directory");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!("SKIP: {} is not a directory", dir.display());
        return;
    }
    let mut rig = shiso_bench::decode::Rig::open_qwen35moe(&dir).expect("the MoE checkpoint opens");
    let ms =
        shiso_bench::decode::decode_ms_per_step(&mut rig, shiso_bench::measure::Protocol::single())
            .expect("a decode number");
    assert!(ms > 0.0 && ms < 1000.0, "a plausible step time, got {ms}");
}
