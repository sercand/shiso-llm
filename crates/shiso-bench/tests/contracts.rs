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
use shiso_core::model::{DecodeStep, Logits, PromptRun, SamplingParams};
use shiso_core::{
    Backend, DeviceView, LaunchSample, Model, ProfileRegion, ProfileReport, Profiler,
};
use shiso_engine::{Checkpointed, Reason, ResumeInfo, ResumeSource, SaveOutcome};
use shiso_ir::Usage;
use shiso_kvstore::{CommitOpts, Frontier, KvCache, KvError, Lookup, PersistOutcome, Resume};
use shiso_models::image_registry::{ImageFamily, ImageModelDesc};
use shiso_server::tokenizer::{TokenizerError, Vocabulary};
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits,
    ImageOutputFormat, ImageStepEvent, InputReferenceImage,
};

use shiso_hub::{HostProfile, HubError, ModelRef, ProgressEvent, Recipe};

// The profiling surface the bench, the kernel bench and the server all build
// on. Pinned through wrappers: `ProfileRegion`'s lifetime is a parameter of
// its impl, so its methods do not coerce to a higher-ranked fn pointer.
fn profile_through(backend: &dyn Backend) -> Option<CoreResult<ProfileReport>> {
    let profiler: &dyn Profiler = backend.as_profiler()?;
    Some(ProfileRegion::open(profiler).and_then(ProfileRegion::finish))
}
const _: fn(&dyn Backend) -> Option<CoreResult<ProfileReport>> = profile_through;
type FromSamples = fn(&[LaunchSample], u64) -> ProfileReport;
const _: FromSamples = ProfileReport::from_samples;
type ReportJson = fn(&ProfileReport) -> serde_json::Value;
const _: ReportJson = ProfileReport::to_json;

// The group-scaled weights a model crate's loader assembles for the sparse
// engine: the MXFP8 dense projection, whose format id the engine's tier
// ladder matches on and whose two arrays are the codes and the group scales,
// and the resident NVFP4 bank a draft head's experts are held in.
const _: () = assert!(shiso_core::DType::Mxfp8E4M3G32.gguf_id() == 213);
type Mxfp8Bytes = fn(u32, u32) -> Option<(u64, u64)>;
const _: Mxfp8Bytes = shiso_engine::sparse::weights::mxfp8_bytes;
type Mxfp8Weight = fn(
    u32,
    u32,
    [shiso_core::DeviceBuffer; 2],
) -> CoreResult<shiso_engine::sparse::weights::Weight>;
const _: Mxfp8Weight = shiso_engine::sparse::weights::Weight::mxfp8;
type Nvfp4Bank = fn(
    u32,
    u32,
    u32,
    [shiso_core::DeviceBuffer; 3],
) -> CoreResult<shiso_engine::sparse::weights::Weight>;
const _: Nvfp4Bank = shiso_engine::sparse::weights::Weight::nvfp4;

const _: Option<&Recipe> = None;
const _: Option<&ModelRef> = None;
const _: Option<&HostProfile> = None;
const _: Option<&ProgressEvent> = None;
const _: Option<&HubError> = None;

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

/// A batched prompt pass names each run's session beside its tokens - not two
/// parallel slices - and borrows its rows from the model, as `prefill_last`
/// does.
type PrefillLastMany =
    for<'m> fn(&'m mut (dyn Model + 'static), &[PromptRun<'_>]) -> CoreResult<Logits<'m>>;
const _: PrefillLastMany = <dyn Model>::prefill_last_many;

fn prompt_run_fields(run: PromptRun<'_>) -> (SessionId, &[TokenId]) {
    (run.session, run.tokens)
}
const _: for<'a> fn(PromptRun<'a>) -> (SessionId, &'a [TokenId]) = prompt_run_fields;

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

/// The cache element: one family-neutral type from the command line to the
/// pool. The server converts its flag into it, the engine options carry it,
/// the sparse stack's page geometry and attention slots take it, and its width
/// and codec name are what the disk tier files a page under - so a second
/// spelling anywhere along that path would be a page sized at one element and
/// filed under another.
fn kv_elem_codes(kv: shiso_core::page::KvElem) -> (u32, &'static str, u8) {
    use shiso_core::page::KvElem;
    // Exhaustive on purpose: an element added without a row here is a build
    // failure, not a silent f16.
    let tag = match kv {
        KvElem::F16 => 0,
        KvElem::Fp8E4m3 => 1,
    };
    (kv.bytes(), kv.codec(), tag)
}
const _: fn(shiso_core::page::KvElem) -> (u32, &'static str, u8) = kv_elem_codes;
const _: fn(shiso_server::cli::KvCacheDtype) -> shiso_core::page::KvElem = Into::into;
const _: fn(&shiso_engine::EngineOptions) -> shiso_core::page::KvElem = |opts| opts.kv_elem;
const _: fn(shiso_core::page::KvElem) -> shiso_engine::sparse::paged::PagePlanes =
    |kv| shiso_engine::sparse::paged::PagePlanes {
        indexer: true,
        draft: true,
        kv,
    };
const _: fn(shiso_engine::sparse::qsa::AttnSlot) -> shiso_core::page::KvElem = |slot| slot.kv;

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

/// Both decode entry points take the operator's policy by reference. Video's is
/// optional - `None` is the off state - and audio's is not, because audio has no
/// opt-in flag.
type DecodeAudio = fn(
    &[u8],
    &shiso_audio::DecodeLimits,
    &shiso_audio::AudioDecodePolicy,
) -> Result<shiso_audio::Pcm, shiso_audio::AudioError>;
const _: DecodeAudio = shiso_audio::decode_audio;
type DecodeClip = fn(
    &[u8],
    f64,
    Option<&shiso_vision::VideoDecodePolicy>,
) -> Result<shiso_vision::DecodedClip, shiso_vision::VisionError>;
const _: DecodeClip = shiso_vision::decode_clip;

/// The OS decoder surface the video chain drives: opened from a codec
/// configuration record, fed samples in decode order, drained at the end. Only
/// `Unavailable` and `Unsupported` let the chain move on.
type HwOpen = fn(
    &shiso_media_sys::HwDecoderDesc<'_>,
) -> Result<shiso_media_sys::HwDecoder, shiso_media_sys::HwError>;
const _: HwOpen = shiso_media_sys::HwDecoder::open;
type HwDecode = fn(
    &mut shiso_media_sys::HwDecoder,
    &[u8],
    shiso_media_sys::HwTimestamp,
    bool,
) -> Result<Vec<shiso_media_sys::HwFrame>, shiso_media_sys::HwError>;
const _: HwDecode = shiso_media_sys::HwDecoder::decode;

/// The demuxer reads a container into one video track whose samples are ranges
/// of the request buffer.
type Demux =
    for<'a> fn(
        &'a [u8],
    )
        -> Result<shiso_vision::demux::DemuxedTrack<'a>, shiso_vision::demux::DemuxError>;
const _: Demux = shiso_vision::demux::open;

/// A decoder's refusals, as the chain tells them apart.
fn hw_error_kind(error: &shiso_media_sys::HwError) -> u8 {
    match error {
        shiso_media_sys::HwError::Unavailable { .. } => 0,
        shiso_media_sys::HwError::Unsupported { .. } => 1,
        shiso_media_sys::HwError::Failed { .. } => 2,
        _ => 3,
    }
}

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

/// What the model accepts and defaults to: the trait's required method, the
/// struct's six fields, and the family detector the registry dispatches on.
const _: fn(&(dyn ImageGenerator + 'static)) -> ImageLimits = <dyn ImageGenerator>::limits;
type ImageLimitsFields = (u32, u32, u32, (u32, u32), u32, Option<u32>);
fn image_limits_fields(limits: ImageLimits) -> ImageLimitsFields {
    (
        limits.size_multiple,
        limits.min_side,
        limits.max_side,
        limits.default_size,
        limits.default_steps,
        limits.fixed_steps,
    )
}
const _: fn(ImageLimits) -> ImageLimitsFields = image_limits_fields;
const _: fn(&ImageGenerationJob, &ImageLimits) -> Result<(), ImageGenError> =
    ImageGenerationJob::validate;
const _: fn(&ImageModelDesc) -> Result<ImageFamily, ImageGenError> =
    shiso_models::image_registry::detect_image_family;

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

/// What the server hands the registry to open a checkpoint, read apart: the
/// four component paths and the assets directory, then the residency policy
/// and the byte reserve left to the host, then the speed approximations: the
/// half-size draft steps and the last step's reused block span. The schedule
/// is not here: it is the checkpoint's own documents. The server
/// builds these across the crate seam from its flags.
type ImageModelDescFields = (
    std::path::PathBuf,
    Option<std::path::PathBuf>,
    Option<std::path::PathBuf>,
    Option<std::path::PathBuf>,
    Option<std::path::PathBuf>,
    shiso_models::image_registry::ImageResidency,
    u64,
    u32,
    Option<(u32, u32)>,
);
fn image_model_desc_fields(desc: ImageModelDesc) -> ImageModelDescFields {
    (
        desc.model_path,
        desc.text_encoder_path,
        desc.vae_path,
        desc.tokenizer_path,
        desc.assets_dir,
        desc.residency,
        desc.reserve_bytes,
        desc.draft_steps,
        desc.final_reuse,
    )
}
const _: fn(ImageModelDesc) -> ImageModelDescFields = image_model_desc_fields;

/// The residency policy is exhaustive in all three of its choices.
fn residency_choices(residency: shiso_models::image_registry::ImageResidency) -> u8 {
    use shiso_models::image_registry::ImageResidency;
    match residency {
        ImageResidency::Auto => 0,
        ImageResidency::Resident => 1,
        ImageResidency::Swap => 2,
    }
}
const _: fn(shiso_models::image_registry::ImageResidency) -> u8 = residency_choices;

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

/// Every variant of the generator's error, named with its payload type:
/// the wire handlers map them onto HTTP statuses and quote the payloads
/// into the response, so a rename, a removed variant or a retyped payload
/// must stop here rather than in a downstream match arm. The match is
/// exhaustive in both directions: a variant added upstream makes it
/// non-exhaustive.
fn image_gen_error_variants(error: &ImageGenError) -> (u8, u32, u32) {
    match error {
        ImageGenError::EmptyPrompt => (0, 0, 0),
        ImageGenError::PromptTooLong { tokens, max } => (9, *tokens, *max),
        ImageGenError::InvalidDimensions {
            width,
            height,
            multiple,
            min,
            max,
        } => {
            let _: (&u32, &u32, &u32) = (multiple, min, max);
            (1, *width, *height)
        }
        ImageGenError::StepsFixed { steps, fixed } => (8, *steps, *fixed),
        ImageGenError::DraftTakesEveryStep { draft, steps } => (13, *draft, *steps),
        ImageGenError::DecodeTooLarge {
            width,
            height,
            needed_mib,
            budget_mib,
        } => {
            let _: (&u64, &u64) = (needed_mib, budget_mib);
            (11, *width, *height)
        }
        ImageGenError::TooLargeForMemory {
            width,
            height,
            needed_mib,
            budget_mib,
            largest_side,
        } => {
            let _: (&u64, &u64, &u32) = (needed_mib, budget_mib, largest_side);
            (12, *width, *height)
        }
        ImageGenError::InvalidSteps { steps } => (7, *steps, 0),
        ImageGenError::UnsupportedFormat(msg) => {
            let _: &String = msg;
            (2, 0, 0)
        }
        ImageGenError::ReferenceImage(msg) => {
            let _: &String = msg;
            (3, 0, 0)
        }
        ImageGenError::Backend(msg) => {
            let _: &String = msg;
            (4, 0, 0)
        }
        ImageGenError::Encoding(msg) => {
            let _: &String = msg;
            (5, 0, 0)
        }
        ImageGenError::Checkpoint(msg) => {
            let _: &String = msg;
            (6, 0, 0)
        }
    }
}
const _: fn(&ImageGenError) -> (u8, u32, u32) = image_gen_error_variants;

/// The typed image cores as embedders and the server call them: the IR they
/// take and return, the four entry points (blocking and sink-streaming, for a
/// generation and for an edit), and the floor they check first. A drift in any
/// of them stops here.
mod image_core {
    use shiso_ir::image::{Edit, Generate, Image, Images, Step, Usage};
    use shiso_runtime::error::AppError;
    use shiso_runtime::image::{
        MAX_IMAGES_PER_REQUEST, MAX_STEPS_PER_IMAGE, edit, edit_stream, generate, generate_stream,
        preflight, validate, validate_edit, validate_edit_shape,
    };
    use shiso_runtime::state::AppState;

    /// The entry points are async and the streaming ones take a sink that
    /// answers whether anyone is still reading; the futures are named so the
    /// compiler checks both.
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signature"
    )]
    fn generate_future(
        state: &AppState,
        request: Generate,
    ) -> impl Future<Output = Result<Images, AppError>> + '_ {
        generate(state, request)
    }
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signature"
    )]
    fn generate_stream_future(
        state: &AppState,
        request: Generate,
    ) -> impl Future<Output = Result<(), AppError>> + '_ {
        generate_stream(state, request, |_: Step| true)
    }
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signature"
    )]
    fn edit_future(
        state: &AppState,
        request: Edit,
    ) -> impl Future<Output = Result<Images, AppError>> + '_ {
        edit(state, request)
    }
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signature"
    )]
    fn edit_stream_future(
        state: &AppState,
        request: Edit,
    ) -> impl Future<Output = Result<(), AppError>> + '_ {
        edit_stream(state, request, |_: Step| true)
    }
    const _: fn(&Generate) -> Result<(), AppError> = validate;
    const _: fn(&Edit) -> Result<(), AppError> = validate_edit;
    type EditShape = fn(&[Vec<u8>], Option<usize>) -> Result<(), AppError>;
    const _: EditShape = validate_edit_shape;
    type Preflight = fn(&AppState, &Generate, &[Vec<u8>]) -> Result<(), AppError>;
    const _: Preflight = preflight;
    const _: [usize; 1] = [MAX_IMAGES_PER_REQUEST];
    const _: [u32; 1] = [MAX_STEPS_PER_IMAGE];

    type GenerateFields = (
        String,
        Option<usize>,
        Option<String>,
        Option<String>,
        Option<u32>,
        Option<u64>,
        Option<String>,
        Option<u32>,
    );
    fn generate_fields(request: Generate) -> GenerateFields {
        (
            request.prompt,
            request.n,
            request.size,
            request.aspect_ratio,
            request.steps,
            request.seed,
            request.output_format,
            request.output_compression,
        )
    }
    const _: fn(Generate) -> GenerateFields = generate_fields;

    type EditFields = (Generate, Vec<Vec<u8>>);
    fn edit_fields(request: Edit) -> EditFields {
        (request.generate, request.references)
    }
    const _: fn(Edit) -> EditFields = edit_fields;

    type ImagesFields = (u64, Vec<Image>, Usage);
    fn images_fields(images: Images) -> ImagesFields {
        (images.created, images.images, images.usage)
    }
    const _: fn(Images) -> ImagesFields = images_fields;

    fn image_fields(image: Image) -> (Vec<u8>, String) {
        (image.bytes, image.media_type)
    }
    const _: fn(Image) -> (Vec<u8>, String) = image_fields;

    fn usage_fields(usage: Usage) -> (u32, u32, u32) {
        (
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens(),
        )
    }
    const _: fn(Usage) -> (u32, u32, u32) = usage_fields;

    /// The stream's three events and what each carries.
    type StepFields = Option<(u32, Vec<u8>, u64, Image, Usage)>;
    fn step_fields(step: Step) -> StepFields {
        match step {
            Step::Admitted => None,
            Step::Partial { index, bytes } => Some((
                index,
                bytes,
                0,
                Image {
                    bytes: Vec::new(),
                    media_type: String::new(),
                },
                Usage::default(),
            )),
            Step::Completed {
                created,
                image,
                usage,
            } => Some((0, Vec::new(), created, image, usage)),
        }
    }
    const _: fn(Step) -> StepFields = step_fields;
}

// ---------------------------------------------------------------------------
// Embeddings.
//
// The wire handler builds an IR request, the service turns each input into
// the family-neutral parts the model reads, the worker holds the model as a
// trait object, and each family opens through the registry. The text side of
// a part is a `TokenId` list - the vocabulary's own spelling - so a token-id
// input and a tokenized string cross this seam as one type.

const _: Option<&dyn shiso_models::embedder::Embedder> = None;

/// The opener the server calls: a described checkpoint and a shared backend
/// answer with the trait object the worker holds.
type OpenEmbedder =
    fn(
        &shiso_models::embedder::EmbedderDesc<'_>,
        &std::sync::Arc<dyn Backend>,
    )
        -> Result<Box<dyn shiso_models::embedder::Embedder>, shiso_models::embedder::EmbedError>;
const _: OpenEmbedder = shiso_models::embedder::open_embedder;

/// One input in, one pooled vector and its per-modality positions out.
type Embed = fn(
    &mut (dyn shiso_models::embedder::Embedder + 'static),
    &[shiso_models::embedder::EmbedPart],
) -> Result<shiso_models::embedder::Embedding, shiso_models::embedder::EmbedError>;
const _: Embed = <dyn shiso_models::embedder::Embedder>::embed;

/// A text part is the vocabulary's own token type.
fn embed_text_part(ids: Vec<TokenId>) -> shiso_models::embedder::EmbedPart {
    shiso_models::embedder::EmbedPart::Text(ids)
}
const _: fn(Vec<TokenId>) -> shiso_models::embedder::EmbedPart = embed_text_part;

/// The IR request read apart: model, inputs, the width the caller asked
/// for, the encoding and the task name. A token-id input stays `u32` on the
/// IR side - the IR counts tokens and does not name them.
type EmbedRequestFields = (
    String,
    Vec<shiso_ir::EmbedInput>,
    Option<u32>,
    shiso_ir::EmbedEncoding,
    Option<String>,
);
fn embed_request_fields(request: shiso_ir::EmbedRequest) -> EmbedRequestFields {
    (
        request.model,
        request.inputs,
        request.dimensions,
        request.encoding,
        request.task,
    )
}
const _: fn(shiso_ir::EmbedRequest) -> EmbedRequestFields = embed_request_fields;
const _: fn(Vec<u32>) -> shiso_ir::EmbedInput = shiso_ir::EmbedInput::Tokens;

/// The truncation every surface shares: a vector and a width in, a unit
/// vector out.
type Matryoshka = fn(&[f32], usize) -> Result<Vec<f32>, shiso_models::embedder::EmbedError>;
const _: Matryoshka = shiso_models::embedder::matryoshka;

/// The typed decision core as embedders and the server call it: the IR it
/// takes and returns, `decide`, and the pure steps the server's scheduler
/// builds the decisions route from. A drift in any of them stops here.
mod decision {
    use std::sync::Arc;

    use shiso_core::{ReadoutPrompt, SchemaQuestion, TokenId};
    use shiso_ir::decision::{
        Answer, AnswerKind, Choice, ChoiceValue, Input, Level, Part, Question, QuestionKind,
        ReadoutState, Request, Response, Usage,
    };
    use shiso_runtime::decision::score::{
        RunScore, ScoreError, ScoreRun, Scored, fold_head, fold_readout, readout_runs,
    };
    use shiso_runtime::decision::{core, decide, scored_answer};
    use shiso_runtime::error::AppError;
    use shiso_runtime::state::AppState;

    /// `decide` answers an IR request from the model a state serves. The
    /// future is named so the compiler checks the async signature.
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signature"
    )]
    fn decide_future(
        state: &Arc<AppState>,
        request: Request,
    ) -> impl Future<Output = Result<Response, AppError>> + '_ {
        decide(state, request)
    }

    type RequestFields = (
        Input,
        ReadoutState,
        Vec<Question>,
        Option<f64>,
        Option<bool>,
        bool,
    );
    fn request_fields(request: Request) -> RequestFields {
        (
            request.input,
            request.readout_state,
            request.questions,
            request.temperature,
            request.preserve_thinking,
            request.token_ids,
        )
    }
    const _: fn(Request) -> RequestFields = request_fields;

    /// Every shape of input, with its payload types. Exhaustive: a shape
    /// added makes this non-exhaustive.
    fn input_shapes(input: Input) -> u8 {
        match input {
            Input::Text(text) => {
                let _: String = text;
                0
            }
            Input::Blocks(parts) => {
                let _: Vec<Part> = parts;
                1
            }
            Input::Pictured { media, text } => {
                let _: (Vec<shiso_ir::MediaSource>, String) = (media, text);
                2
            }
        }
    }
    const _: fn(Input) -> u8 = input_shapes;

    fn part_shapes(part: Part) -> u8 {
        match part {
            Part::Text(text) => {
                let _: String = text;
                0
            }
            Part::Image(source) => {
                let _: shiso_ir::MediaSource = source;
                1
            }
        }
    }
    const _: fn(Part) -> u8 = part_shapes;

    fn readout_state_shapes(state: ReadoutState) -> u8 {
        match state {
            ReadoutState::Input => 0,
            ReadoutState::Absent => 1,
            ReadoutState::Structured { compact, indented } => {
                let _: (String, String) = (compact, indented);
                2
            }
        }
    }
    const _: fn(ReadoutState) -> u8 = readout_state_shapes;

    type QuestionFields = (String, String, QuestionKind, bool);
    fn question_fields(question: Question) -> QuestionFields {
        (
            question.name,
            question.instructions,
            question.kind,
            question.probability_answer,
        )
    }
    const _: fn(Question) -> QuestionFields = question_fields;

    /// Every kind of question, with its payload types. Exhaustive: a kind
    /// added makes this non-exhaustive.
    fn question_kinds(kind: QuestionKind) -> u8 {
        match kind {
            QuestionKind::Choice(choices) => {
                let _: Vec<Choice> = choices;
                0
            }
            QuestionKind::YesNo { yes, no } => {
                let _: (Option<String>, Option<String>) = (yes, no);
                1
            }
            QuestionKind::Score(levels) => {
                let _: Vec<Level> = levels;
                2
            }
        }
    }
    const _: fn(QuestionKind) -> u8 = question_kinds;

    type ChoiceFields = (ChoiceValue, Option<String>);
    fn choice_fields(choice: Choice) -> ChoiceFields {
        (choice.value, choice.description)
    }
    const _: fn(Choice) -> ChoiceFields = choice_fields;

    type LevelFields = (String, Option<String>);
    fn level_fields(level: Level) -> LevelFields {
        (level.label, level.description)
    }
    const _: fn(Level) -> LevelFields = level_fields;

    fn choice_values(value: ChoiceValue) -> u8 {
        match value {
            ChoiceValue::Str(text) => {
                let _: String = text;
                0
            }
            ChoiceValue::Bool(flag) => {
                let _: bool = flag;
                1
            }
        }
    }
    const _: fn(ChoiceValue) -> u8 = choice_values;

    type AnswerFields = (
        String,
        AnswerKind,
        bool,
        ChoiceValue,
        Vec<(ChoiceValue, f64)>,
        f64,
        f64,
        Option<u32>,
        Option<f64>,
        Option<Vec<u32>>,
        Option<Vec<u32>>,
    );
    fn answer_fields(answer: Answer) -> AnswerFields {
        (
            answer.name,
            answer.kind,
            answer.probability_answer,
            answer.choice,
            answer.probabilities,
            answer.confidence,
            answer.label_mass,
            answer.level,
            answer.expected_score,
            answer.prompt_token_ids,
            answer.label_token_ids,
        )
    }
    const _: fn(Answer) -> AnswerFields = answer_fields;

    fn answer_kinds(kind: AnswerKind) -> u8 {
        match kind {
            AnswerKind::Choice => 0,
            AnswerKind::YesNo => 1,
            AnswerKind::Score => 2,
        }
    }
    const _: fn(AnswerKind) -> u8 = answer_kinds;

    type ResponseFields = (Vec<Answer>, Usage);
    fn response_fields(response: Response) -> ResponseFields {
        (response.answers, response.usage)
    }
    const _: fn(Response) -> ResponseFields = response_fields;

    type UsageFields = (u64, u64);
    fn usage_fields(usage: Usage) -> UsageFields {
        (usage.input_tokens, usage.cached_tokens)
    }
    const _: fn(Usage) -> UsageFields = usage_fields;

    const _: fn(Question, Vec<f64>, f64, usize) -> Answer = scored_answer;

    /// The shared validators every wire calls while parsing.
    const _: fn(Option<&str>) -> Result<(), AppError> = core::check_model;
    const _: fn(&str) -> Result<(), AppError> = core::check_input;
    const _: fn(&[Question], bool) -> Result<(), AppError> = core::check_ids;
    const _: fn(&str, &str, bool) -> Result<(), AppError> = core::check_question_text;
    const _: fn(&[Choice], &str, usize) -> Result<(), AppError> = core::check_choice_options;
    const _: fn(&[Level], &str, usize) -> Result<(), AppError> = core::check_score_levels;
    const _: fn(Option<f64>) -> Result<f64, AppError> = core::check_temperature;

    /// Every error variant a caller matches on, with its payload type. The
    /// enum is `non_exhaustive`, so the wildcard is the only way to stay
    /// total; the named arms are what is pinned.
    fn score_error_variants(error: &ScoreError) -> u8 {
        match error {
            ScoreError::Layout(message) => {
                let _: &String = message;
                0
            }
            ScoreError::Engine(message) => {
                let _: &String = message;
                1
            }
            _ => u8::MAX,
        }
    }
    const _: fn(&ScoreError) -> u8 = score_error_variants;

    /// The pure steps the server's scheduler builds its decisions route from.
    type ReadoutRuns = fn(&ReadoutPrompt, &[SchemaQuestion]) -> Result<Vec<ScoreRun>, ScoreError>;
    type FoldReadout =
        fn(&ReadoutPrompt, &[ScoreRun], &[RunScore], f64) -> Result<Vec<Scored>, ScoreError>;
    type FoldHead = fn(&[Vec<f32>], &[Vec<usize>], f64) -> Result<Vec<Scored>, ScoreError>;
    const _: ReadoutRuns = readout_runs;
    const _: FoldReadout = fold_readout;
    const _: FoldHead = fold_head;

    type ScoreRunFields = (Vec<TokenId>, Vec<TokenId>, u32);
    fn score_run_fields(run: ScoreRun) -> ScoreRunFields {
        (run.tokens, run.labels, run.shared_prefix)
    }
    const _: fn(ScoreRun) -> ScoreRunFields = score_run_fields;

    type RunScoreFields = (Vec<f64>, u64);
    fn run_score_fields(score: RunScore) -> RunScoreFields {
        (score.label_logprobs, score.cached_tokens)
    }
    const _: fn(RunScore) -> RunScoreFields = run_score_fields;

    /// The literal-aware encoder the vocabulary's decision pieces and chat
    /// prompts share.
    type Encode = fn(
        &tokenizers::Tokenizer,
        &str,
        &[std::ops::Range<usize>],
    ) -> Result<Vec<TokenId>, tokenizers::Error>;
    const _: Encode = shiso_runtime::decision::literal::encode_piece;
    const _: Encode = shiso_runtime::decision::literal::encode_with_literals;
}

/// The hub half of the embedder API: find a model offline, pull it, and hold its
/// lock.
mod embedder_hub {
    use std::path::{Path, PathBuf};

    use shiso_hub::HubConfig;
    use shiso_runtime::hub::{
        InstalledModel, PullError, config_for_home, resolve, resolve_serve_dir, resolve_serve_model,
    };
    use shiso_runtime::options::{ExplicitlySet, ServeOptions};

    const _: fn(&str, &HubConfig) -> Result<Option<InstalledModel>, PullError> = resolve;
    type Pull =
        fn(&str, &HubConfig, &mut dyn FnMut(u64, u64) -> bool) -> Result<InstalledModel, PullError>;
    const _: Pull = shiso_runtime::hub::pull;
    const _: fn(&Path) -> HubConfig = config_for_home;
    const _: fn(&Path, &mut ServeOptions, &ExplicitlySet) -> Result<(), PullError> =
        resolve_serve_dir;
    type ResolveServe = fn(
        &str,
        &mut ServeOptions,
        &ExplicitlySet,
        &HubConfig,
    ) -> Result<Option<InstalledModel>, PullError>;
    const _: ResolveServe = resolve_serve_model;

    /// What an installed model tells its holder; the lock is private.
    fn installed_fields(model: &InstalledModel) -> (&PathBuf, u64) {
        let InstalledModel {
            dir, total_bytes, ..
        } = model;
        (dir, *total_bytes)
    }
    const _: fn(&InstalledModel) -> (&PathBuf, u64) = installed_fields;

    /// What a pull or an open can fail with.
    fn pull_error_variants(error: &PullError) -> u8 {
        match error {
            PullError::NoDevice(_) => 0,
            PullError::UnknownModel(_) => 1,
            PullError::InvalidReference { .. } => 2,
            PullError::ComponentFlagConflict(_) => 3,
            PullError::NotInstalled(_) => 4,
            PullError::Inconsistent(_) => 5,
            PullError::Lock { .. } => 6,
            PullError::Cancelled => 7,
            PullError::Request(_) => 8,
            PullError::Install(_) => 9,
            PullError::Hub(_) => 10,
            _ => u8::MAX,
        }
    }
    const _: fn(&PullError) -> u8 = pull_error_variants;
}

/// The in-process facade: the open path, the three capabilities, and the rules
/// that make one handle safe to share.
mod runtime_facade {
    use std::future::Future;
    use std::path::{Path, PathBuf};

    use shiso_ir::image::{Edit, Generate, Images, Step};
    use shiso_ir::{Request, Response, StreamEvent};
    use shiso_runtime::boot::EngineThread;
    use shiso_runtime::error::AppError;
    use shiso_runtime::options::{ExplicitlySet, ServeOptions};
    use shiso_runtime::state::AppState;
    use shiso_runtime::{OpenError, Runtime, RuntimeOptions, Source};

    /// Where a model comes from. Exhaustive: a source added makes this
    /// non-exhaustive.
    fn source_shapes(source: Source<'_>) -> u8 {
        match source {
            Source::Registry(name) => {
                let _: &str = name;
                0
            }
            Source::Dir(dir) => {
                let _: &Path = dir;
                1
            }
        }
    }
    const _: fn(Source<'_>) -> u8 = source_shapes;

    type OptionsFields = (Option<PathBuf>, bool, ServeOptions, ExplicitlySet);
    fn options_fields(options: RuntimeOptions) -> OptionsFields {
        // Exhaustive: a field added makes this fail to compile.
        let RuntimeOptions {
            home,
            pull_if_missing,
            serve,
            explicit,
        } = options;
        (home, pull_if_missing, serve, explicit)
    }
    const _: fn(RuntimeOptions) -> OptionsFields = options_fields;
    const _: fn() -> RuntimeOptions = RuntimeOptions::default;

    /// What an open can fail with.
    fn open_error_variants(error: &OpenError) -> u8 {
        match error {
            OpenError::Pull(_) => 0,
            OpenError::Config(_) => 1,
            OpenError::Boot(_) => 2,
            OpenError::NothingLoaded => 3,
            OpenError::Executor(_) => 4,
            OpenError::WarmUp(_) => 5,
            _ => u8::MAX,
        }
    }
    const _: fn(&OpenError) -> u8 = open_error_variants;

    /// The open path, blocking, with a pull's progress callback.
    type Open = for<'a, 'b> fn(
        Source<'a>,
        RuntimeOptions,
        &'b mut dyn FnMut(u64, u64) -> bool,
    ) -> Result<Runtime, OpenError>;
    const _: Open = Runtime::open;
    const _: fn(AppState, Option<EngineThread>) -> Result<Runtime, OpenError> = Runtime::from_state;
    const _: fn(&Runtime) -> bool = Runtime::has_language_model;
    const _: fn(&Runtime) -> bool = Runtime::has_image_model;

    /// The blocking calls.
    const _: fn(&Runtime, Request) -> Result<Response, AppError> = Runtime::generate_blocking;
    const _: fn(
        &Runtime,
        shiso_ir::decision::Request,
    ) -> Result<shiso_ir::decision::Response, AppError> = Runtime::decide_blocking;

    /// The async calls, named so the compiler checks their signatures and that
    /// every future is `Send`: a caller spawns them on a multi-thread executor.
    #[expect(
        dead_code,
        reason = "named only so the compiler checks the async signatures"
    )]
    fn generate_future(
        runtime: &Runtime,
        request: Request,
    ) -> impl Future<Output = Result<Response, AppError>> + Send {
        runtime.generate(request)
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn generate_stream_future(
        runtime: &Runtime,
        request: Request,
    ) -> impl Future<Output = Result<(), AppError>> + Send {
        runtime.generate_stream(request, |_: StreamEvent| true)
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn images_future(runtime: &Runtime) -> impl Future<Output = Result<Images, AppError>> + Send {
        runtime.images(blank_generate())
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn images_stream_future(
        runtime: &Runtime,
    ) -> impl Future<Output = Result<(), AppError>> + Send {
        runtime.images_stream(blank_generate(), |_: Step| true)
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn edit_image_future(
        runtime: &Runtime,
    ) -> impl Future<Output = Result<Images, AppError>> + Send {
        runtime.edit_image(blank_edit())
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn edit_image_stream_future(
        runtime: &Runtime,
    ) -> impl Future<Output = Result<(), AppError>> + Send {
        runtime.edit_image_stream(blank_edit(), |_: Step| true)
    }
    #[expect(dead_code, reason = "named only so the compiler checks the signature")]
    fn decide_future(
        runtime: &Runtime,
    ) -> impl Future<Output = Result<shiso_ir::decision::Response, AppError>> + Send {
        runtime.decide(blank_decision())
    }

    fn blank_generate() -> Generate {
        Generate {
            prompt: String::new(),
            n: None,
            size: None,
            aspect_ratio: None,
            steps: None,
            seed: None,
            output_format: None,
            output_compression: None,
        }
    }
    fn blank_edit() -> Edit {
        Edit {
            generate: blank_generate(),
            references: Vec::new(),
        }
    }
    fn blank_decision() -> shiso_ir::decision::Request {
        shiso_ir::decision::Request {
            input: shiso_ir::decision::Input::Text(String::new()),
            readout_state: shiso_ir::decision::ReadoutState::Input,
            questions: Vec::new(),
            temperature: None,
            preserve_thinking: None,
            token_ids: false,
        }
    }

    /// One handle serves every thread.
    const fn shared<T: Send + Sync>() {}
    const _: () = shared::<Runtime>();

    /// Not `Clone`: sharing is an `Arc`, and Drop joins the engine thread once,
    /// when the last owner lets go. If `Runtime` were `Clone`, `Runtime: Clone`
    /// would make both impls below apply and the inference ambiguous, which
    /// does not compile.
    const _: fn() = || {
        trait AmbiguousIfClone<A> {
            fn some_item() {}
        }
        impl<T: ?Sized> AmbiguousIfClone<()> for T {}
        impl<T: Clone> AmbiguousIfClone<u8> for T {}
        let _ = <Runtime as AmbiguousIfClone<_>>::some_item;
    };
}

/// The contracts above are checked by the compiler, so this file building at
/// all is the result. The test exists so a target that stopped being built
/// shows up as a missing test rather than as silence.
#[test]
fn the_contracts_above_are_compiled() {
    assert_eq!(u32::from(TokenId::new(7)), 7, "TokenId is a u32 newtype");
    assert_eq!(
        usage_counts(&Usage::default()),
        (0, 0, 0, 0),
        "a fresh usage block counts nothing",
    );
    assert_eq!(
        hw_error_kind(&shiso_media_sys::HwError::Failed {
            detail: String::new()
        }),
        2,
        "a failed decoder is not a decline"
    );
    assert_eq!(
        kv_elem_codes(shiso_core::page::KvElem::default()),
        (2, "f16", 0),
        "the default cache element is f16, under the codec name existing disk files carry"
    );
    let parsed_ref = ModelRef::parse("qwen3.5-9b:metal-mlx4").expect("valid model ref");
    assert_eq!(parsed_ref.model.as_str(), "qwen3.5-9b");
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

// ---------------------------------------------------------------------------
// The Clef backbone.
//
// Every Clef GGUF declares one architecture whatever backbone it carries, so
// the server's factory asks the container which stack it holds before it tries
// an arm. `None` is "not a Clef GGUF", which every other checkpoint is; a
// change to that answer would send a non-Clef file through the refusal.

/// The probe the factory calls on the checkpoint path.
type ClefBackboneAt = fn(&std::path::Path) -> CoreResult<Option<shiso_models::clef::ClefBackbone>>;
const _: ClefBackboneAt = shiso_models::clef::ClefBackbone::at;

/// The dense backbone's arm: a document and a container in, a model whose
/// head answers the decision routes out.
type LlamaFromHf =
    fn(&shiso_core::HfConfig, &std::path::Path) -> CoreResult<shiso_models::llama::LlamaConfig>;
const _: LlamaFromHf = shiso_models::llama::LlamaConfig::from_hf;

type LlamaLoad = fn(
    &std::sync::Arc<dyn Backend>,
    &std::path::Path,
    &shiso_models::llama::LlamaConfig,
    shiso_engine::EngineOptions,
) -> CoreResult<shiso_models::llama::LlamaModel>;
const _: LlamaLoad = shiso_models::llama::LlamaModel::load;

// ---------------------------------------------------------------------------
// The CPU device.
//
// A plain CPU answers `None` to `max_block_threads` and exposes no optional
// capability trait, so the engine launches the kernels listed below by name.
// The table was measured by wrapping the Metal device that way and running the
// llama Clef model through every pass it makes; a model change that reaches
// for another entry must add it here and to `shiso-cpu` in the same commit.

/// How a test opens the CPU device.
type CpuOpen = fn(shiso_cpu::CpuOptions) -> CoreResult<std::sync::Arc<shiso_cpu::CpuBackend>>;
const _: CpuOpen = shiso_cpu::CpuBackend::new;

/// Every `(module, entry)` the engine launches for the llama Clef model on a
/// plain device. The spec's table has 19 rows; the engine also picks the
/// `mul_mv_ext` `r1_3`, `r1_4` and `r1_5` variants for 3-8 decode rows, which
/// the measured 2-session batch never reached, so there are 22.
const CPU_INVENTORY: [(&str, &str); 22] = [
    ("generic", "kernel_get_rows_q8_0_f32"),
    ("generic", "kernel_rms_norm_mul_f32_4"),
    ("generic", "kernel_add_rms_norm_mul_f32_4"),
    ("generic", "kernel_swiglu_flat_f32"),
    ("generic", "kernel_mul_mv_q8_0_f32"),
    ("generic", "kernel_mul_mv_ext_q8_0_f32_r1_2"),
    ("generic", "kernel_mul_mv_ext_q8_0_f32_r1_3"),
    ("generic", "kernel_mul_mv_ext_q8_0_f32_r1_4"),
    ("generic", "kernel_mul_mv_ext_q8_0_f32_r1_5"),
    ("qwen35", "kernel_mul_mm_q8_0_f32"),
    ("dense", "kernel_dense_q_rope"),
    ("dense", "kernel_dense_kv_rope_store"),
    ("dense", "kernel_dense_attn_decode"),
    ("dense", "kernel_dense_attn_prefill"),
    ("audio", "kernel_audio_gemv_q8_0w"),
    ("audio", "kernel_audio_gemm_q8_0w"),
    ("audio", "kernel_audio_gemm64_q8_0w"),
    ("audio", "kernel_audio_split_sum_f32"),
    ("audio", "kernel_audio_layer_norm_f32"),
    ("audio", "kernel_audio_window_attn_f32"),
    ("audio", "kernel_audio_unary_f32"),
    ("audio", "kernel_audio_axpy_f32"),
];

#[test]
fn every_inventory_entry_resolves_on_the_cpu_device() {
    let missing: Vec<_> = CPU_INVENTORY
        .iter()
        .filter(|(module, entry)| !shiso_cpu::has_entry(module, entry))
        .collect();
    assert!(missing.is_empty(), "no CPU kernel for {missing:?}");
}

// The biased routed block's stacks are one enum over the stored codecs, and a
// streamed IQ4_NL stack is checked against its stream's slot width by the same
// function the loader calls on a resident one.
const _: fn(&shiso_engine::moe_biased::ExpertStack) -> CoreResult<()> =
    shiso_engine::moe_biased::check_stack;
const _: fn(&shiso_engine::moe_biased::RowIq4NlStack) -> u64 =
    shiso_engine::moe_biased::RowIq4NlStack::expert_bytes;
