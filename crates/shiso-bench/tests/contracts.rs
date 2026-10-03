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
use shiso_engine::{Checkpointed, Reason, ResumeInfo, ResumeSource, SaveOutcome};
use shiso_ir::Usage;
use shiso_kvstore::{CommitOpts, Frontier, KvCache, KvError, Lookup, PersistOutcome, Resume};
use shiso_models::flux2::config::ImageModelDesc;
use shiso_server::tokenizer::{TokenizerError, Vocabulary};
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageOutputFormat,
    ImageStepEvent, InputReferenceImage,
};

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
/// lifetime is the receiver's. It is keyed by the prompt's token path.
type KvResume = for<'c> fn(
    &'c mut (dyn Checkpointed + 'static),
    SessionId,
    &[TokenId],
) -> CoreResult<Option<ResumeInfo<'c>>>;
const _: KvResume = <dyn Checkpointed>::kv_resume;

/// A save names only the session and why: the path is the session's own
/// history, so there is no caller-supplied one to disagree with its state.
type KvSave = fn(&mut (dyn Checkpointed + 'static), SessionId, Reason) -> CoreResult<SaveOutcome>;
const _: KvSave = <dyn Checkpointed>::kv_save;

/// The shutdown sweep stores every open session from its own history.
type KvSweep = fn(&mut (dyn Checkpointed + 'static)) -> usize;
const _: KvSweep = <dyn Checkpointed>::kv_shutdown_sweep;

/// What a resume hands back: the reach, the owned path in the same token type,
/// and where it came from.
fn resume_info_fields(info: ResumeInfo<'_>) -> (u32, Vec<TokenId>, ResumeSource) {
    (info.tokens_reused, info.tokens, info.source)
}
const _: fn(ResumeInfo<'_>) -> (u32, Vec<TokenId>, ResumeSource) = resume_info_fields;

/// The cache's own resume: the tree, then the disk behind it, keyed by the
/// same tokens a plan is.
type CacheResume = fn(&KvCache, &[TokenId], f64) -> Result<Option<Resume>, KvError>;
const _: CacheResume = KvCache::resume;

/// Queueing a frontier's checkpoint for disk names the frontier and a reason,
/// and answers in the store's own outcome.
type CachePersist = fn(&KvCache, &Frontier, Reason) -> Result<PersistOutcome, KvError>;
const _: CachePersist = KvCache::persist;

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

// ---------------------------------------------------------------------------
// Speech.
//
// The recognizer, the diarizer and the audio tower are held as trait objects
// by the server's audio worker and engine thread, and each family opens its
// own; losing object safety, or a second token type on the recognizer's
// seam, would surface here rather than in the worker.

const _: Option<&dyn shiso_audio::SpeechRecognizer> = None;
const _: Option<&dyn shiso_audio::RecognizerStream> = None;
const _: Option<&dyn shiso_audio::Diarizer> = None;
const _: Option<&dyn shiso_audio::DiarizationPass> = None;
const _: Option<&dyn shiso_audio::AudioTower> = None;

/// A recognized token is the one `TokenId`, at an encoder frame.
fn token_event_fields(event: shiso_audio::TokenEvent) -> (TokenId, u32) {
    (event.token, event.frame)
}
const _: fn(shiso_audio::TokenEvent) -> (TokenId, u32) = token_event_fields;

/// A stream is fed mel frames and answers timed tokens.
type RecognizerPush = fn(
    &mut (dyn shiso_audio::RecognizerStream + 'static),
    &shiso_audio::MelFrames,
) -> Result<Vec<shiso_audio::TokenEvent>, shiso_audio::AudioError>;
const _: RecognizerPush = <dyn shiso_audio::RecognizerStream>::push;

/// A chunking recognizer is told where its chunks end, in mel frames from the
/// stream's start, and says whether it honours that.
fn stream_cuts(options: shiso_audio::StreamOptions) -> Option<Vec<u32>> {
    options.cuts
}
const _: fn(shiso_audio::StreamOptions) -> Option<Vec<u32>> = stream_cuts;
type ChunksAtCuts = fn(&(dyn shiso_audio::SpeechRecognizer + 'static)) -> bool;
const _: ChunksAtCuts = <dyn shiso_audio::SpeechRecognizer>::chunks_at_cuts;

/// A stream's options carry the context as the recognizer's own ids, from
/// the encoder a recognizer with one hands the server.
const _: fn(
    shiso_audio::Language,
    usize,
    Option<Vec<u32>>,
    shiso_audio::ContextIds,
) -> shiso_audio::StreamOptions = |language, lookahead, cuts, context| shiso_audio::StreamOptions {
    language,
    lookahead,
    cuts,
    context,
};

/// A stream may name the language it read out of the model's words, and a
/// chunking stream reports where its chunks ended, which ends its words.
type StreamLanguage =
    for<'a> fn(&'a (dyn shiso_audio::RecognizerStream + 'static)) -> Option<&'a str>;
const _: StreamLanguage = <dyn shiso_audio::RecognizerStream>::language;
type ChunkEnds = for<'a> fn(&'a (dyn shiso_audio::RecognizerStream + 'static)) -> &'a [u32];
const _: ChunkEnds = <dyn shiso_audio::RecognizerStream>::chunk_ends;

/// The two chunk plans the worker and the chunking stream share: the
/// diarizer's turns over the mel frames, and the silence rule.
type TurnCuts =
    fn(&shiso_audio::transcript::SpeakerActivity, &shiso_audio::MelFrames, f64) -> Vec<usize>;
const _: TurnCuts = shiso_audio::chunks::turn_cuts;
type SilenceCuts = fn(&shiso_audio::MelFrames, usize, usize, usize) -> Vec<usize>;
const _: SilenceCuts = shiso_audio::chunks::silence_cuts;

/// A clip recognised offline is planned whole into spans, which a batching
/// recognizer transcribes a batch at a time; one that does not batch says 0.
type PlanOffline = fn(
    &shiso_audio::MelFrames,
    Option<&shiso_audio::transcript::SpeakerActivity>,
    f64,
    &shiso_audio::chunks::OfflineRule,
) -> Result<Vec<shiso_audio::chunks::Span>, shiso_audio::AudioError>;
const _: PlanOffline = shiso_audio::chunks::plan_offline;
const _: fn(bool) -> shiso_audio::chunks::OfflineRule =
    shiso_audio::chunks::OfflineRule::for_10ms_mel;
const _: fn(usize, usize, bool) -> shiso_audio::chunks::Span =
    |start, end, transcribe| shiso_audio::chunks::Span {
        start,
        end,
        transcribe,
    };
type BatchesOffline = fn(&(dyn shiso_audio::SpeechRecognizer + 'static)) -> usize;
const _: BatchesOffline = <dyn shiso_audio::SpeechRecognizer>::batches_offline;
type OpenOffline = fn(
    &(dyn shiso_audio::SpeechRecognizer + 'static),
    &shiso_audio::StreamOptions,
    Vec<shiso_audio::chunks::Span>,
) -> Result<Box<dyn shiso_audio::OfflineJob>, shiso_audio::AudioError>;
const _: OpenOffline = <dyn shiso_audio::SpeechRecognizer>::open_offline;
type OfflineStep = fn(
    &mut (dyn shiso_audio::OfflineJob + 'static),
    &shiso_audio::MelFrames,
) -> Result<Option<shiso_audio::OfflineBatch>, shiso_audio::AudioError>;
const _: OfflineStep = <dyn shiso_audio::OfflineJob>::step;
const _: fn(Vec<shiso_audio::SpanResult>) -> shiso_audio::OfflineBatch =
    |spans| shiso_audio::OfflineBatch { spans };
const _: fn(
    shiso_audio::chunks::Span,
    Vec<shiso_audio::TokenEvent>,
    u32,
    Option<String>,
) -> shiso_audio::SpanResult = |span, tokens, chunk_end, language| shiso_audio::SpanResult {
    span,
    tokens,
    chunk_end,
    language,
};

/// A diarizer answers one whole clip with one activity block.
type Diarize = fn(
    &(dyn shiso_audio::Diarizer + 'static),
    &shiso_audio::MelFrames,
) -> Result<shiso_audio::transcript::SpeakerActivity, shiso_audio::AudioError>;
const _: Diarize = <dyn shiso_audio::Diarizer>::diarize;

/// An audio tower's rows are the vision towers' block type, so the engine
/// splices both through one path.
type AudioEncode = fn(
    &(dyn shiso_audio::AudioTower + 'static),
    &std::sync::Arc<dyn Backend>,
    &[shiso_audio::AudioClip],
) -> Result<shiso_vision::EncodedRows, shiso_audio::AudioError>;
const _: AudioEncode = <dyn shiso_audio::AudioTower>::encode;

/// The openers the server calls, one per role, each trying every family.
type OpenRecognizer =
    fn(
        &std::path::Path,
        &shiso_models::audio_registry::RecognizerDesc<'_>,
        &std::sync::Arc<dyn Backend>,
    ) -> Result<std::sync::Arc<dyn shiso_audio::SpeechRecognizer>, shiso_audio::AudioError>;
const _: OpenRecognizer = shiso_models::audio_registry::open_speech_recognizer;
/// The recognizer's open-time knobs: the tower file and the offline batch.
const _: fn(Option<&std::path::Path>, usize) -> shiso_models::audio_registry::RecognizerDesc<'_> =
    |mmproj, offline_batch| shiso_models::audio_registry::RecognizerDesc {
        mmproj,
        offline_batch,
    };
type OpenDiarizer =
    fn(
        &std::path::Path,
        &std::sync::Arc<dyn Backend>,
    ) -> Result<std::sync::Arc<dyn shiso_audio::Diarizer>, shiso_audio::AudioError>;
const _: OpenDiarizer = shiso_models::audio_registry::open_diarizer;

// ---------------------------------------------------------------------------
// Image generation.
//
// The wire handlers build a job, the server's image worker holds the
// generator as a trait object and answers the job with encoded bytes, and
// each family opens through the registry; losing object safety, or a field
// type drifting on the job, the image or the service description, would
// surface here rather than in the wire adapters.

const _: Option<&dyn ImageGenerator> = None;

/// The opener the server calls: a described checkpoint and a shared backend
/// answer with the trait object the worker holds.
type OpenImageGenerator = fn(
    &ImageModelDesc,
    &std::sync::Arc<dyn Backend>,
) -> Result<std::sync::Arc<dyn ImageGenerator>, ImageGenError>;
const _: OpenImageGenerator = shiso_models::image_registry::open_image_generator;

/// A generation takes the validated job and an optional progress callback
/// and answers with the encoded image - the one door between the worker and
/// a family.
type Generate = fn(
    &(dyn ImageGenerator + 'static),
    &ImageGenerationJob,
    Option<&dyn Fn(ImageStepEvent)>,
) -> Result<GeneratedImage, ImageGenError>;
const _: Generate = <dyn ImageGenerator>::generate;

/// A job read apart: prompt, pixel dimensions and step count as u32, the
/// core `Seed` newtype rather than a raw integer the pipeline would have to
/// reinterpret, the requested encoding, and any reference images for
/// image-to-image work.
type JobFields = (
    String,
    u32,
    u32,
    u32,
    Seed,
    ImageOutputFormat,
    Option<u32>,
    Vec<InputReferenceImage>,
);
fn image_generation_job_fields(job: ImageGenerationJob) -> JobFields {
    (
        job.prompt,
        job.width,
        job.height,
        job.steps,
        job.seed,
        job.output_format,
        job.output_compression,
        job.input_references,
    )
}
const _: fn(ImageGenerationJob) -> JobFields = image_generation_job_fields;

/// A finished generation read apart: encoded bytes with the MIME string and
/// the pixel and token counts the wire response reports.
type GeneratedImageFields = (Vec<u8>, &'static str, u32, u32, u32, u32);
fn generated_image_fields(image: GeneratedImage) -> GeneratedImageFields {
    (
        image.bytes,
        image.media_type,
        image.width,
        image.height,
        image.prompt_tokens,
        image.completion_tokens,
    )
}
const _: fn(GeneratedImage) -> GeneratedImageFields = generated_image_fields;

/// What the worker thread is started with, read apart: the generator as the
/// trait object the server holds, its wire name, and the queue ceiling.
type ImageServiceDescFields = (std::sync::Arc<dyn ImageGenerator>, String, usize);
fn image_service_desc_fields(
    desc: shiso_server::image::ImageServiceDesc,
) -> ImageServiceDescFields {
    (desc.generator, desc.model_id, desc.max_queue)
}
const _: fn(shiso_server::image::ImageServiceDesc) -> ImageServiceDescFields =
    image_service_desc_fields;

/// Every variant of the generator's error, named: the wire handlers map
/// them onto HTTP statuses, so a rename or a removed variant must stop
/// here rather than in a downstream match arm. The match is exhaustive in
/// both directions: a variant added upstream makes it non-exhaustive.
fn image_gen_error_variants(error: &ImageGenError) -> (u8, u32, u32) {
    match error {
        ImageGenError::EmptyPrompt => (0, 0, 0),
        ImageGenError::InvalidDimensions { width, height } => (1, *width, *height),
        ImageGenError::UnsupportedFormat(_) => (2, 0, 0),
        ImageGenError::ReferenceImage(_) => (3, 0, 0),
        ImageGenError::Backend(_) => (4, 0, 0),
        ImageGenError::Encoding(_) => (5, 0, 0),
        ImageGenError::Checkpoint(_) => (6, 0, 0),
    }
}
const _: fn(&ImageGenError) -> (u8, u32, u32) = image_gen_error_variants;

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
