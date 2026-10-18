//! The command line, which is the whole configuration surface.
//!
//! There is no config file format and no free-floating environment reads
//! scattered through the crate: every knob is a flag here, and the environment
//! is a per-flag fallback declared on the flag itself, so `--help` is an
//! exhaustive list of what this binary can be told. A flag always wins over its
//! variable, because an operator who typed something meant it.
//!
//! # Why the vision knobs are flags and not a config block
//!
//! Each of them is a policy decision an operator makes about this deployment -
//! whether a remote address may be fetched, whether a subprocess may be spawned,
//! how much wall clock one decode may burn - and a policy decision that lives in
//! a file nobody reads is a policy decision nobody made. They map one to one
//! onto the types the media layer takes, and [`ServeArgs::video_decode_policy`]
//! and [`ServeArgs::audio_decode_policy`] are the only places that mapping
//! happens.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use shiso_audio::AudioDecodePolicy;
use shiso_parsers::{ReasoningScheme, ToolCallScheme};
pub use shiso_router::cli::RouterArgs;
#[cfg(feature = "decode-ffmpeg")]
use shiso_vision::FfmpegPolicy;
use shiso_vision::{ImageLimits, VideoDecodePolicy};

use clap::parser::ValueSource;
use shiso_hub::recipe::{ExpertResidency, ServerDefaults};
/// The default listen address. Loopback, so an unconfigured server is not on the
/// network.
pub const DEFAULT_BIND: &str = "127.0.0.1";
/// The default port.
pub const DEFAULT_PORT: u16 = 8090;
/// The default context, in tokens.
pub const DEFAULT_CTX: u32 = 4096;
/// The default floor below which the disk tier stores and reads nothing.
pub const DEFAULT_KVSTORE_MIN_TOKENS: u32 = shiso_engine::CKPT_MIN_TOKENS_DEFAULT;
/// The default device memory kept for the towers' output, in mebibytes.
pub const DEFAULT_VISION_CACHE_MB: u64 = 1024;
/// The default host staging the disk tier's writes may hold, in mebibytes.
pub const DEFAULT_KVSTORE_STAGING_MB: u64 = 1024;
/// The default checkpoint pages one scheduler tick copies for the disk tier, in
/// mebibytes.
pub const DEFAULT_KVSTORE_PUMP_MB: u64 = 64;
/// The default time a shutdown waits for queued disk writes, in seconds.
pub const DEFAULT_KVSTORE_DRAIN_SECS: u64 = 60;
/// The default ids the turn replay keeps across finished turns: about 32 MiB
/// of ids and text, room for sixteen conversations at a 262 144-token context.
pub const DEFAULT_TURN_REPLAY_TOKENS: usize = 4 * 1024 * 1024;
/// Seconds between the engine's periodic info lines.
pub const DEFAULT_LOG_STATS_INTERVAL_SECS: u64 = 10;
/// The default pixel ceiling one image may occupy after resizing.
pub const DEFAULT_VISION_MAX_PIXELS: usize = 1280 * 1280;
/// The default row-cache budget for a per-layer embedding table, in mebibytes.
pub const DEFAULT_PLE_CACHE_MB: u64 = 512;
/// Bytes a `--gpu-memory-utilization` pool leaves for what is allocated after
/// the model is built and is not otherwise counted: the math libraries'
/// workspaces, captured decode graphs, allocator granularity.
///
/// A margin, not a measurement: an undercount here is a failed allocation
/// under load, and 2 GiB is a small price against a pool of tens.
pub const KV_FIT_HEADROOM_BYTES: u64 = 2 << 30;

/// `shiso`, the inference server.
#[derive(Debug, clap::Parser)]
#[command(name = "shiso", version, about = "shiso inference server")]
pub struct Cli {
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// The subcommands this binary offers.
///
/// One today. It is a subcommand rather than a bare flag set because the second
/// one - a tokenizer probe, a checkpoint dump - should not force every existing
/// invocation to be rewritten.
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Serve a checkpoint over HTTP.
    Serve(Box<ServeArgs>),
    /// Run an LLM router proxying requests to third-party or local backends.
    Router(Box<RouterArgs>),
    /// Download a model from the registry.
    Pull(PullArgs),
    /// List installed models, or the registry's catalog with `--remote`.
    List(ListArgs),
    /// Describe a model's recipe and variants.
    Info(InfoArgs),
    /// Re-hash an installed model against its recipe.
    Verify(VerifyArgs),
    /// Remove an installed model.
    Rm(RmArgs),
}

/// Everything `shiso serve` accepts.
#[derive(Debug, clap::Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each one is a separate on-off policy an operator sets on the command line, and a state machine over four independent switches would be sixteen states nobody asked for"
)]
pub struct ServeArgs {
    /// Language model checkpoint to serve. Either a checkpoint file or a
    /// checkpoint directory. Optional when `--asr-model` is given: a server
    /// can serve transcription alone.
    #[arg(short = 'm', long, env = "SHISO_MODEL")]
    pub model: Option<PathBuf>,
    /// Model reference (registry recipe) or path.
    ///
    /// If it contains `/` or starts with `.` or `~`, it is treated as a path (`--model`).
    /// Otherwise, it resolves to an installed registry model variant.
    #[arg(value_name = "MODEL", conflicts_with = "model")]
    pub model_ref: Option<String>,
    /// The name this checkpoint answers to on the wire. Defaults to the last
    /// component of `--model`, with any extension removed.
    ///
    /// Worth setting whenever that default is a poor name, which is exactly when
    /// the checkpoint is a *directory*: a file is usually named after the model,
    /// while a directory is as often named after the export that produced it, and
    /// a server that answers to `mlx` tells a client nothing about what it is
    /// running. The default is left alone rather than made cleverer because there
    /// is no way to tell the two kinds of directory name apart from the outside -
    /// and a server that renamed itself on a rule the operator did not know would
    /// be worse than one with a dull name.
    #[arg(long, env = "SHISO_MODEL_NAME")]
    pub model_name: Option<String>,
    /// The architecture config. Defaults to `config.json` beside `--model`.
    ///
    /// Required in practice wherever the checkpoint directory holds only the
    /// weights file: the geometry is read from the config, never guessed from
    /// the weights.
    #[arg(long, env = "SHISO_CONFIG")]
    pub config: Option<PathBuf>,
    /// The vocabulary. Defaults to `tokenizer.json` beside the config.
    #[arg(long, env = "SHISO_TOKENIZER")]
    pub tokenizer: Option<PathBuf>,
    /// The chat template. Defaults to `chat_template.jinja` beside the config,
    /// and then to the `chat_template` key of `tokenizer_config.json`.
    ///
    /// A checkpoint that ships neither still serves: the informational routes
    /// answer, and the generation routes say that no template was found rather
    /// than inventing a prompt format the model has never seen.
    #[arg(long, env = "SHISO_CHAT_TEMPLATE")]
    pub chat_template: Option<PathBuf>,
    /// Sampling defaults. Defaults to `generation_config.json` beside the config.
    ///
    /// Its `temperature`, `top_k`, `top_p` and `min_p` are what a caller who
    /// names no sampling knob gets, and each knob a caller names overrides its
    /// own. A missing file means neutral sampling; a file that does not parse
    /// stops the server at boot.
    #[arg(long, env = "SHISO_GENERATION_CONFIG")]
    pub generation_config: Option<PathBuf>,
    /// Address to listen on.
    #[arg(long, env = "SHISO_BIND", default_value = DEFAULT_BIND)]
    pub bind: String,
    /// Port to listen on.
    #[arg(short = 'p', long, env = "SHISO_PORT", default_value_t = DEFAULT_PORT)]
    pub port: u16,
    /// Tokens one session may hold, prompt included.
    #[arg(short = 'c', long, env = "SHISO_CTX", default_value_t = DEFAULT_CTX)]
    pub ctx: u32,
    /// Sequences that may be resident at once.
    ///
    /// Also the sessions whose state a `--gpu-memory-utilization` pool sets
    /// aside before it takes the rest. More than one decode pass carries are
    /// decoded in several passes a step.
    #[arg(long, env = "SHISO_MAX_NUM_SEQS", default_value_t = crate::scheduler::DEFAULT_MAX_NUM_SEQS)]
    pub max_num_seqs: usize,
    /// Writes one GPU-time-per-kernel report per request into this directory,
    /// as `<kind>-<pid>-<id>.json`.
    ///
    /// Every launch is submitted and timed on its own while a request runs, so
    /// serving is slower: a development tool, not a serving mode. Needs
    /// `--max-num-seqs 1`, so a report is one request's kernels.
    #[arg(long, env = "SHISO_PROFILE_DIR")]
    pub profile_dir: Option<PathBuf>,
    /// Prompt tokens one prefill chunk covers when nothing else is decoding.
    ///
    /// Wide, because with nobody to starve the only thing a narrow chunk buys is
    /// more passes, and the sparse family reads every routed expert once per
    /// pass. A stack splits a chunk wider than its own widest pass itself.
    #[arg(
        long,
        env = "SHISO_SOLO_PREFILL_CHUNK",
        default_value_t = crate::scheduler::DEFAULT_SOLO_CHUNK,
        conflicts_with = "max_num_batched_tokens"
    )]
    pub solo_prefill_chunk: usize,
    /// Prompt tokens one prefill quantum adds to a step while other sequences
    /// are decoding.
    ///
    /// The trade: a wider quantum brings a waiting prompt's first token sooner,
    /// and makes every decoding sequence's step that much longer.
    #[arg(
        long,
        env = "SHISO_MIXED_PREFILL_QUANTUM",
        default_value_t = crate::scheduler::DEFAULT_MIXED_QUANTUM,
        conflicts_with = "max_num_batched_tokens"
    )]
    pub mixed_prefill_quantum: usize,
    /// Tokens one scheduler step may carry, prompt and decode rows together.
    ///
    /// Sets both prefill widths at once: the whole budget when nothing is
    /// decoding, and what the decode rows leave of it when something is. The
    /// decode rows are counted at their worst, `--max-num-seqs` times
    /// `1 + --spec-k`, so a budget that fits them never lets a mixed step grow
    /// past it. Refused beside either width, since the two would disagree.
    #[arg(long, env = "SHISO_MAX_NUM_BATCHED_TOKENS")]
    pub max_num_batched_tokens: Option<usize>,
    /// Tokens the KV cache holds, shared by every session.
    ///
    /// The pool is allocated whole at load, so this is device memory spent up
    /// front: `--ctx` caps one session, and this caps all of them together.
    /// Unset, and without `--gpu-memory-utilization`, the pool holds 131072.
    #[arg(
        long,
        env = "SHISO_KV_CACHE_TOKENS",
        conflicts_with = "gpu_memory_utilization"
    )]
    pub kv_cache_tokens: Option<u64>,
    /// Share of the device's memory this process may fill, in (0, 1]; the KV
    /// cache gets whatever of it the weights, the scratch and the sessions'
    /// state leave.
    ///
    /// Counted against the device's total and against everything already on
    /// it, other processes included - which on a device that shares memory with
    /// its host is the rest of the machine. A share that leaves less than one
    /// `--ctx` context refuses to start, with the numbers.
    #[arg(long, env = "SHISO_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: Option<f64>,
    /// The element the KV cache stores attention keys and values as: `f16`,
    /// or `fp8` (E4M3 at a static scale of 1, also spelled `fp8_e4m3`).
    ///
    /// `fp8` halves the attention key and value bytes of every cached token,
    /// so `--gpu-memory-utilization` and `--kv-cache-tokens` buy more context
    /// for the same memory; the indexer's planes stay f16. Values are rounded
    /// to a 3-bit mantissa when stored, saturate at 448 and flush below 2^-10.
    /// Only the sparse stack on CUDA has the kernels for it; any other model or
    /// device refuses to start rather than serving f16 under this name.
    #[arg(
        long,
        env = "SHISO_KV_CACHE_DTYPE",
        value_enum,
        default_value_t = KvCacheDtype::F16
    )]
    pub kv_cache_dtype: KvCacheDtype,
    /// The precision prompt (prefill) activations enter the block-scaled
    /// weights' products at: `a16`, or `lowp`.
    ///
    /// `a16` keeps every prompt activation at half precision against the
    /// decoded weights - the reference numerics. `lowp` quantizes a prompt
    /// pass's activations to the weight's own format - NVFP4 for the routed
    /// NVFP4 experts, MXFP8 for the dense MXFP8 projections - and runs those
    /// products on the GPU's block-scaled tensor-core instructions: a faster
    /// prefill, at a measurable cost in accuracy (a few percent of a row on
    /// the experts). Decode steps and speculative verifies keep the
    /// half-precision kernels either way. Only the sparse stack on a CUDA
    /// build with kernels for a family-specific architecture (`sm_121a`, a
    /// GB10) has the kernels; anywhere else `lowp` refuses to start.
    #[arg(
        long,
        env = "SHISO_PREFILL_ACTIVATIONS",
        value_enum,
        default_value_t = PrefillActivations::A16
    )]
    pub prefill_activations: PrefillActivations,
    /// Recurrent-state checkpoints the prefix cache may hold, shared by every
    /// session. Zero holds none.
    ///
    /// A checkpoint is what lets a request resume partway through a prompt it
    /// shares with an earlier one instead of prefilling it again; a model with
    /// no recurrent layers keeps only a logits row in each. Slots fill lazily,
    /// but the pool is sized as if all of them were full, because on a device
    /// that shares memory with its host they come out of the same memory: every
    /// slot here is a slot's worth of bytes the KV cache does not get, and a
    /// count that leaves no room for one `--ctx` context refuses to start.
    #[arg(
        long,
        env = "SHISO_KV_CHECKPOINTS",
        default_value_t = shiso_engine::EngineOptions::new().kv_ckpt_slots
    )]
    pub kv_checkpoints: u32,
    /// Directory for the cache's disk tier. Unset means nothing is persisted.
    #[arg(long, env = "SHISO_KVSTORE_DIR")]
    pub kvstore_dir: Option<PathBuf>,
    /// Budget for the disk tier, in mebibytes. Unset is twice the key/value
    /// pool's bytes: enough for every resident session and as many again that
    /// memory gave up.
    #[arg(long, env = "SHISO_KVSTORE_BUDGET_MB")]
    pub kvstore_budget_mb: Option<u64>,
    /// Checkpoints shorter than this many tokens are neither stored on disk nor
    /// looked up there: below it a file costs more to write and read than the
    /// prefill it saves.
    #[arg(long, env = "SHISO_KVSTORE_MIN_TOKENS", default_value_t = DEFAULT_KVSTORE_MIN_TOKENS)]
    pub kvstore_min_tokens: u32,
    /// Host memory the disk tier's write-behind may hold in copied pages, in
    /// mebibytes.
    #[arg(long, env = "SHISO_KVSTORE_STAGING_MB", default_value_t = DEFAULT_KVSTORE_STAGING_MB)]
    pub kvstore_staging_mb: u64,
    /// Checkpoint pages one scheduler tick copies off the device for the disk
    /// writer, in mebibytes: what the write-behind costs a decoding stream per
    /// step.
    #[arg(long, env = "SHISO_KVSTORE_PUMP_MB", default_value_t = DEFAULT_KVSTORE_PUMP_MB)]
    pub kvstore_pump_mb: u64,
    /// How long a shutdown waits for queued disk writes before abandoning them,
    /// in seconds. Keep it under the service manager's stop timeout.
    #[arg(long, env = "SHISO_KVSTORE_DRAIN_SECS", default_value_t = DEFAULT_KVSTORE_DRAIN_SECS)]
    pub kvstore_drain_secs: u64,
    /// Ids the server keeps from finished turns, so the next prompt in a
    /// conversation carries the ids the model generated rather than a
    /// re-encoding of their text that can split differently and miss the
    /// prefix cache. Zero keeps none.
    #[arg(long, env = "SHISO_TURN_REPLAY_TOKENS", default_value_t = DEFAULT_TURN_REPLAY_TOKENS)]
    pub turn_replay_tokens: usize,
    /// Seconds between the engine's info lines of prefill and decode rates,
    /// speculation acceptance and cache use, printed while requests run and
    /// once as they finish. Zero turns the line off.
    #[arg(long, env = "SHISO_LOG_STATS_INTERVAL", default_value_t = DEFAULT_LOG_STATS_INTERVAL_SECS)]
    pub log_stats_interval: u64,
    /// Replay a decode step's launches from a captured device graph.
    ///
    /// On by default, and off is a diagnostic rather than a second way to
    /// serve: a replay issues the same launches with the same arguments, so the
    /// two arms are meant to answer identically and a difference between them
    /// is a bug in the capture. `--no-decode-graphs` is what an operator turns
    /// on to find out which side of that line a wrong answer is on.
    #[arg(long, env = "SHISO_NO_DECODE_GRAPHS", default_value_t = false)]
    pub no_decode_graphs: bool,
    /// Speculative decode depth. Zero is plain decode.
    ///
    /// Read only by the arm of the model factory whose architecture implements
    /// speculation; a knob one architecture reads belongs to that
    /// architecture's arm rather than to a union-of-all-models options struct.
    #[arg(long, env = "SHISO_SPEC_K", default_value_t = 0)]
    pub spec_k: u32,
    /// Prompt-lookup drafts: tokens a step may propose by finding the text
    /// being written earlier in the context. Zero is off.
    ///
    /// Taken in place of the draft head's chain only when it is longer and
    /// starts with the head's whole chain, so it only ever extends the head's
    /// guesses; a rejected proposal is not made again from the
    /// same place. The verify pass is `1 + max(--spec-k, this)` rows at the
    /// widest, which is what the per-session rollback records are sized for.
    /// Read by the sparse family's arm, and only beside `--spec-k`.
    #[arg(long, env = "SHISO_SPEC_LOOKUP", default_value_t = 0)]
    pub spec_lookup: u32,
    /// Capture the speculative verify passes as device graphs, one per row
    /// count and session, as the single-row step already is.
    ///
    /// Off, a verify pass is encoded launch by launch every step: thousands of
    /// launches whose host cost shows as gaps between them. On, the first pass
    /// at each width pays a capture. Read by the sparse family's arm.
    #[arg(long, env = "SHISO_CAPTURE_VERIFY", default_value_t = false)]
    pub capture_verify: bool,
    /// Capture passes over several sessions as device graphs, one per distinct
    /// pass: its sessions in row order and every launch's arguments.
    ///
    /// Off, a decode batch or a batched verify is encoded launch by launch
    /// every step, and at four sessions the device idled 6 ms of each 58 ms
    /// step waiting on the host. Read by the sparse family's arm.
    #[arg(long, env = "SHISO_CAPTURE_BATCH", default_value_t = false)]
    pub capture_batch: bool,
    /// Preserve the ordinary target sampling distribution instead of accepting
    /// target-matching greedy drafts directly.
    ///
    /// Off by default: opportunistic speculation accepts matching greedy drafts
    /// even at non-zero temperature. When set, non-greedy sequences fall back
    /// to ordinary batched decoding on models without a two-phase verifier.
    #[arg(long, env = "SHISO_MTP_EXACT_SAMPLING", default_value_t = false)]
    pub mtp_exact_sampling: bool,
    /// Keep half-precision projections as the checkpoint stores them rather
    /// than repacking them to twelve bits a weight at load.
    ///
    /// The repack is lossless - every weight decodes to its stored half - and
    /// a decode step streams a quarter fewer of those bytes; this is the way
    /// back to the stored form for a comparison. Read by the sparse family's
    /// arm.
    #[arg(long, env = "SHISO_NO_PACK_HALVES", default_value_t = false)]
    pub no_pack_halves: bool,
    /// The draft head: a sidecar weights file beside the checkpoint.
    ///
    /// Absent means one token a step. Present, and with `--spec-k` above zero,
    /// a decode step proposes that many tokens and verifies them in one pass.
    /// The two go together: a head with no depth drafts nothing, and a depth
    /// with no head has nothing to draft with, so either alone is refused
    /// rather than quietly ignored.
    #[arg(long, env = "SHISO_DRAFT_MODEL")]
    pub draft_model: Option<PathBuf>,
    /// Corpus token counts the draft's vocabulary head is narrowed from.
    ///
    /// One `id count` pair per line. Absent, a draft projects through the whole
    /// vocabulary, which is correct and costs it most of what a target step
    /// costs. A token the narrowed head cannot reach is a guess the target
    /// rejects - a verify row that was paid for either way - so this trades
    /// acceptance for speed and never correctness.
    #[arg(long, env = "SHISO_DRAFT_VOCAB")]
    pub draft_vocab: Option<PathBuf>,

    /// Device memory kept for the towers' output, in mebibytes: a picture seen
    /// before is neither decoded nor encoded again while its rows are here.
    ///
    /// A picture's rows are one per placeholder token, a few megabytes each,
    /// so the default holds a long agent session's screenshots. It is set aside
    /// before the key/value pool is sized, and only when `--vision` names a
    /// tower.
    #[arg(long, env = "SHISO_VISION_CACHE_MB", default_value_t = DEFAULT_VISION_CACHE_MB)]
    pub vision_cache_mb: u64,

    /// Row-cache budget for the per-layer embedding table, in mebibytes.
    ///
    /// Read only by families with a per-layer embedding table; every other
    /// architecture's arm ignores it. The table is streamed off the checkpoint
    /// rather than held on the device, and this is what a decode step is allowed
    /// to keep of it between steps. Zero disables the cache, which reads every
    /// row from disk and is the slow, correct answer.
    #[arg(long, env = "SHISO_PLE_CACHE_MB", default_value_t = DEFAULT_PLE_CACHE_MB)]
    pub ple_cache_mb: u64,
    /// Hold the whole per-layer embedding table in host memory.
    ///
    /// Read only by families with a per-layer embedding table. Off by
    /// default, because the table is the largest single tensor these
    /// checkpoints carry and a server that paid for all of it at startup would
    /// be slower to boot and no faster to answer a short run. On, a step never
    /// touches the disk, at the cost of about 2.3 GB more resident memory for
    /// the E4B table, on top of the 5.1 GB mapping the checkpoint itself
    /// costs.
    #[arg(long, env = "SHISO_PLE_RESIDENT", default_value_t = false)]
    pub ple_resident: bool,

    /// Where a sparse checkpoint's routed expert banks live: imported beside
    /// the other weights, or streamed from disk into a pool of expert slots.
    ///
    /// Read only by the sparse family's MLX directory. `auto` imports them
    /// when every weight fits in the device's working set with
    /// `--expert-reserve-gb` to spare, and streams them otherwise; the other
    /// two insist, and `resident` on a device that cannot hold them fails at
    /// the first allocation that does not fit.
    #[arg(long, env = "SHISO_EXPERT_RESIDENCY", value_enum, default_value_t = ExpertResidencyChoice::Auto)]
    pub expert_residency: ExpertResidencyChoice,
    /// Gibibytes of the device's working set the automatic residency choice
    /// leaves for the cache, the activations and the n-gram rows.
    ///
    /// Larger is a smaller expert pool and a larger cache; the pool is sized
    /// from what is left.
    #[arg(long, env = "SHISO_EXPERT_RESERVE_GB", default_value_t = 6)]
    pub expert_reserve_gb: u64,
    /// Expert slots when the banks are streamed; zero sizes the pool from the
    /// working set.
    ///
    /// One slot holds one expert of one layer, 2.6 MiB for the published
    /// checkpoint.
    #[arg(long, env = "SHISO_EXPERT_SLOTS", default_value_t = 0)]
    pub expert_slots: u32,

    /// The per-layer n-gram embedding table, for a checkpoint that keeps it in
    /// a file of its own: the sparse family's trellis export.
    ///
    /// Required by that checkpoint and refused beside any other: the table is
    /// read from here and never looked for by name beside `--model`.
    #[arg(long, env = "SHISO_NGRAM")]
    pub ngram: Option<PathBuf>,

    /// The vision tower: a sidecar weights file, a multimodal-projector GGUF
    /// (`mmproj-*.gguf`), or a checkpoint directory whose config names its
    /// tower shard.
    ///
    /// Absent means text only, and the encoder is never loaded.
    #[arg(long, env = "SHISO_VISION")]
    pub vision: Option<PathBuf>,
    /// Pixel ceiling one image may occupy after resizing.
    #[arg(long, env = "SHISO_VISION_MAX_PIXELS", default_value_t = DEFAULT_VISION_MAX_PIXELS)]
    pub vision_max_pixels: usize,
    /// Allow fetching media named by an http or https address.
    ///
    /// Off by default, and the refusal happens before any network call: a
    /// server that fetches whatever a request names is a request-forgery
    /// primitive that happens to also caption pictures.
    #[arg(
        long,
        env = "SHISO_VISION_ALLOW_REMOTE_IMAGES",
        default_value_t = false
    )]
    pub vision_allow_remote_images: bool,
    /// Allow decoding video other than GIF: the OS decoder, then the pure-Rust
    /// one, then ffmpeg when `--ffmpeg-path` is set.
    #[arg(long, env = "SHISO_VIDEO_ALLOW_DECODE", default_value_t = false)]
    pub video_allow_decode: bool,
    /// The ffmpeg binary, for the containers and codecs nothing else in this
    /// build decodes. A bare name is resolved on `PATH`; an absolute path pins
    /// a known build. There is no default: without it ffmpeg is never loaded or
    /// spawned, for video or for audio.
    #[cfg(feature = "decode-ffmpeg")]
    #[arg(long, env = "SHISO_FFMPEG_PATH")]
    pub ffmpeg_path: Option<PathBuf>,
    /// Frames a second to sample a clip at.
    #[arg(long, env = "SHISO_VIDEO_FPS", default_value_t = shiso_vision::DEFAULT_FPS)]
    pub video_fps: f64,
    /// Frames one clip may contribute.
    #[arg(long, env = "SHISO_VIDEO_MAX_FRAMES", default_value_t = shiso_vision::DEFAULT_MAX_FRAMES)]
    pub video_max_frames: usize,
    /// Wall-clock budget for one decode, in seconds.
    #[arg(long, env = "SHISO_VIDEO_DECODE_TIMEOUT_S", default_value_t = 20)]
    pub video_decode_timeout_s: u64,
    /// The longest audio clip one request may carry, in seconds.
    ///
    /// Checked on the decoded length, before any mel frame is computed. A
    /// tower's own limit still applies beneath it.
    #[arg(long, env = "SHISO_AUDIO_MAX_SECONDS", default_value = "7200")]
    pub audio_max_seconds: NonZeroU32,
    /// The largest transcription upload, in MiB. Only the transcription
    /// routes take it; every other route keeps the JSON body limit.
    #[arg(long, env = "SHISO_AUDIO_MAX_UPLOAD_MB", default_value = "512")]
    pub audio_max_upload_mb: NonZeroU32,
    /// Speech recognizer checkpoint: an export directory or a GGUF file.
    /// Serves `/v1/audio/transcriptions`.
    #[arg(long, env = "SHISO_ASR_MODEL")]
    pub asr_model: Option<PathBuf>,
    /// The audio-tower GGUF (an `mmproj-*.gguf`) for a recognizer whose GGUF
    /// export keeps its tower apart from the text model. Without it the one
    /// mmproj beside `--asr-model` with the same quantisation suffix is taken,
    /// and none or several is refused. Needs `--asr-model`.
    #[arg(long, env = "SHISO_ASR_MMPROJ")]
    pub asr_mmproj: Option<PathBuf>,
    /// The name the recognizer answers to in a transcription's `model` field.
    /// Defaults to the last component of `--asr-model`, extension removed.
    #[arg(long, env = "SHISO_ASR_ID")]
    pub asr_id: Option<String>,
    /// Speaker diarizer checkpoint, for `response_format=diarized_json`.
    /// Needs `--asr-model`.
    #[arg(long, env = "SHISO_DIARIZATION_MODEL")]
    pub diarization_model: Option<PathBuf>,
    /// Encoder frames of future context the recognizer uses: one of the
    /// lookaheads the checkpoint was trained for, checked when it loads. More
    /// is more accurate and later to emit.
    #[arg(long, env = "SHISO_ASR_LOOKAHEAD", default_value_t = 13)]
    pub asr_lookahead: usize,
    /// Transcriptions advanced at once; the rest wait in the queue.
    #[arg(long, env = "SHISO_AUDIO_MAX_STREAMS", default_value_t = 4)]
    pub audio_max_streams: usize,
    /// Chunks an offline transcription decodes at once, on a recognizer that
    /// batches: one weight read per step serves this many. Streaming
    /// requests decode one at a time whatever it is.
    #[arg(long, env = "SHISO_ASR_BATCH", default_value_t = 8)]
    pub asr_batch: usize,
    /// Whether an offline transcription with a diarizer loaded leaves out
    /// silences of 3 s or more where no speaker is active. Skipped audio
    /// yields no text: speech the diarizer misses there is lost.
    /// `--asr-skip-silence=false` transcribes them.
    #[arg(
        long,
        env = "SHISO_ASR_SKIP_SILENCE",
        default_value_t = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true,
        action = clap::ArgAction::Set
    )]
    pub asr_skip_silence: bool,
    /// The shortest clip, in seconds, that a non-streaming transcription
    /// plans whole and decodes in batches, on a recognizer that batches;
    /// shorter clips keep the chunk-by-chunk path. 0 sends every
    /// non-streaming transcription offline.
    #[arg(
        long,
        env = "SHISO_ASR_OFFLINE_MIN_SECS",
        default_value_t = crate::audio::DEFAULT_OFFLINE_MIN_SECS
    )]
    pub asr_offline_min_secs: u32,
    /// The most tokens a transcription's `keywords` and `prompt` may take in
    /// the recognizer's prompt, once worded and tokenized; a request past it
    /// is refused. At most the recognizer's own cap (512 for the Qwen3-ASR
    /// family); every chunk's prefill carries them.
    #[arg(long, env = "SHISO_ASR_MAX_CONTEXT_TOKENS", default_value_t = 256)]
    pub asr_max_context_tokens: usize,
    /// Image generation model to serve: a checkpoint file or a checkpoint
    /// directory. Serves the image generation routes.
    #[arg(long, env = "SHISO_IMAGE_MODEL")]
    pub image_model: Option<PathBuf>,
    /// The name the image generator answers to on the wire. Defaults to the
    /// name the loaded checkpoint reports for itself.
    #[arg(long, env = "SHISO_IMAGE_MODEL_ID")]
    pub image_model_id: Option<String>,
    /// Directory holding shared image assets (`text_encoder`, `vae`,
    /// `tokenizer`) when they are not collocated with `--image-model`.
    #[arg(long, env = "SHISO_IMAGE_ASSETS_DIR")]
    pub image_assets_dir: Option<PathBuf>,
    /// Explicit path to the text encoder weights when they are split from
    /// `--image-model`. Overrides where the locator would otherwise look.
    #[arg(long, env = "SHISO_IMAGE_TEXT_ENCODER")]
    pub image_text_encoder: Option<PathBuf>,
    /// Explicit path to the VAE weights when they are split from
    /// `--image-model`. Overrides where the locator would otherwise look.
    #[arg(long, env = "SHISO_IMAGE_VAE")]
    pub image_vae: Option<PathBuf>,
    /// Explicit path to the tokenizer when it is split from `--image-model`.
    /// Overrides where the locator would otherwise look.
    #[arg(long, env = "SHISO_IMAGE_TOKENIZER")]
    pub image_tokenizer: Option<PathBuf>,
    /// Image requests that may wait in the queue beyond the one generating;
    /// the next is refused with a 503 a retry can fix.
    #[arg(long, env = "SHISO_IMAGE_MAX_QUEUE", default_value_t = 16)]
    pub image_max_queue: usize,
    /// Embedding requests that may wait in the queue beyond the one running,
    /// when `--model` is an embedding checkpoint; the next is refused with a
    /// 503 a retry can fix.
    #[arg(long, env = "SHISO_EMBEDDING_MAX_QUEUE", default_value_t = 64)]
    pub embedding_max_queue: usize,
    /// Denoising steps a request that names none gets. More is slower and,
    /// past the point the checkpoint was distilled for, not better.
    /// Unset, the loaded model's own default (or fixed schedule) is used.
    #[arg(long, env = "SHISO_IMAGE_DEFAULT_STEPS")]
    pub image_default_steps: Option<u32>,
    /// The image size a request that names none gets, as `WxH`
    /// (e.g. 1024x1024). Both sides must sit on the loaded model's grid and
    /// within its bounds; unset, the model's own default size is used.
    #[arg(long, env = "SHISO_IMAGE_DEFAULT_SIZE")]
    pub image_default_size: Option<String>,
    /// Which image pipeline components stay loaded across jobs: `auto`
    /// keeps all of them when they fit the device's working set beside a
    /// job and otherwise swaps; `resident` refuses a job that does not fit;
    /// `swap` lets the text encoder and the transformer take turns.
    #[arg(
        long,
        env = "SHISO_IMAGE_RESIDENCY",
        value_enum,
        default_value_t = ImageResidencyChoice::Auto
    )]
    pub image_residency: ImageResidencyChoice,
    /// GiB of the device working set the image pipeline leaves to the rest
    /// of the host when it plans what stays loaded.
    #[arg(long, env = "SHISO_IMAGE_RESERVE_GB", default_value_t = 1.5)]
    pub image_reserve_gb: f64,
    /// Leading denoising steps of every image job run at half the
    /// requested resolution; the job then finishes at its own size. Each
    /// such step costs about a quarter of a full one, at some cost in fine
    /// detail. 0 runs every step at full size. A job must keep at least one
    /// full-size step, and a model without a draft stage refuses a nonzero
    /// count at boot.
    #[arg(long, env = "SHISO_IMAGE_DRAFT_STEPS", default_value_t = 0)]
    pub image_draft_steps: u32,
    /// Transformer blocks `START:END` (end exclusive) the last denoising
    /// step of every image job skips, adding instead the residual the step
    /// before recorded over the same blocks. Saves `(END - START) / blocks`
    /// of that step at some cost in fidelity. The job's last two steps must
    /// both run at full size (see `--image-draft-steps`). Unset, every
    /// block of every step runs.
    #[arg(long, env = "SHISO_IMAGE_FINAL_REUSE", value_parser = parse_block_span)]
    pub image_final_reuse: Option<BlockSpan>,
    /// Which reasoning parser to use on model output.
    #[arg(
        long,
        env = "SHISO_REASONING_PARSER",
        value_enum,
        default_value_t = ParserChoice::Auto
    )]
    pub reasoning_parser: ParserChoice,
    /// Which tool-call parser to use on model output.
    #[arg(
        long,
        env = "SHISO_TOOL_CALL_PARSER",
        value_enum,
        default_value_t = ParserChoice::Auto
    )]
    pub tool_call_parser: ParserChoice,
}

/// A span of transformer blocks, `START:END` with `END` exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSpan {
    /// The first block in the span.
    pub start: u32,
    /// One past the last block in the span.
    pub end: u32,
}

/// Parses `START:END` into a non-empty [`BlockSpan`].
fn parse_block_span(text: &str) -> Result<BlockSpan, String> {
    let (start, end) = text
        .split_once(':')
        .ok_or_else(|| format!("'{text}' is not START:END"))?;
    let parse = |v: &str| {
        v.trim()
            .parse::<u32>()
            .map_err(|e| format!("'{v}' in '{text}': {e}"))
    };
    let (start, end) = (parse(start)?, parse(end)?);
    if start >= end {
        return Err(format!("'{text}' is empty: START must be below END"));
    }
    Ok(BlockSpan { start, end })
}

/// A setting that parses but cannot be served.
///
/// The scheduler's two prefill widths, resolved from the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillWidths {
    /// Prompt tokens one chunk covers when nothing else is decoding.
    pub solo_chunk: usize,
    /// Prompt tokens one quantum adds to a step while others decode.
    pub mixed_quantum: usize,
}

/// Separate from clap's own parse failures because these are cross-field or
/// range checks that clap has no vocabulary for, and separate from
/// [`anyhow`](https://docs.rs/anyhow) because this is library code: the binary
/// edge is the only place an opaque error is the right shape.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A value that must be positive was zero.
    #[error("--{flag} must be greater than zero")]
    Zero {
        /// The flag, spelled as it is typed.
        flag: &'static str,
    },
    /// The checkpoint path has no file name to take a model id from.
    #[error("--model {path} has no file name to use as a model id")]
    NamelessModel {
        /// The path as it was given.
        path: String,
    },
    /// Neither a language model, a recognizer, nor an image model was named.
    #[error("nothing to serve: give --model, --asr-model, or --image-model")]
    NothingToServe,
    /// A diarizer was named with no recognizer to attribute speech from.
    #[error("--diarization-model needs --asr-model: a diarizer labels a transcript")]
    DiarizerWithoutRecognizer,
    /// An audio tower was named with no recognizer to load it into.
    #[error("--asr-mmproj needs --asr-model: it is the tower of the recognizer's text model")]
    MmprojWithoutRecognizer,
    /// A share that is not in `(0, 1]`.
    #[error("--{flag} {value} is not a share in (0, 1]")]
    NotAShare {
        /// The flag, spelled as it is typed.
        flag: &'static str,
        /// The value as it was given.
        value: f64,
    },
    /// A KV pool that cannot hold one full context.
    #[error("--kv-cache-tokens {tokens} cannot serve --ctx {ctx}: it must be at least --ctx")]
    PoolBelowContext {
        /// The pool as it was given.
        tokens: u64,
        /// The context one session may reach.
        ctx: u32,
    },
    /// A KV pool of more pages than a pool can count.
    #[error("--kv-cache-tokens {tokens} is more than a pool can count")]
    PoolTooLarge {
        /// The pool as it was given.
        tokens: u64,
    },
    /// A step budget the decode rows alone fill, leaving no prefill.
    #[error(
        "--max-num-batched-tokens {budget} leaves no room for a prefill: \
         {decode_rows} decode rows (--max-num-seqs x (1 + --spec-k)) can fill it"
    )]
    BudgetBelowDecodes {
        /// The budget as it was given.
        budget: usize,
        /// The worst-case decode rows of one step.
        decode_rows: usize,
    },
    /// A prefill width that is not a whole number of the recurrence's runs.
    #[error(
        "--{flag} {value} is not a multiple of {grid}: a prompt chunk must end on \
         the recurrent blocks' run grid, or where a prompt is cut changes its answer"
    )]
    OffGrid {
        /// The flag, spelled as it is typed.
        flag: &'static str,
        /// The width as it was given.
        value: usize,
        /// The grid it must be a multiple of.
        grid: usize,
    },
    /// One of the two settings speculation needs was given without the other.
    #[error("--{given} without --{missing}: speculation needs a draft head and a depth")]
    HalfSpeculative {
        /// The flag that was given.
        given: &'static str,
        /// The one it needs beside it.
        missing: &'static str,
    },
    /// `--profile-dir` with more than one sequence resident at once.
    #[error(
        "--profile-dir needs --max-num-seqs 1 (got {max_num_seqs}): a report covers one request's kernels"
    )]
    ProfileNeedsOneSequence {
        /// The value given.
        max_num_seqs: usize,
    },
    /// `--profile-dir` with the speech service, whose worker interleaves
    /// streams.
    #[error(
        "--profile-dir cannot be combined with --asr-model: the audio worker interleaves streams, so a report could not be one request's"
    )]
    ProfileWithAudio,
    /// An `--image-reserve-gb` that is negative or not a number.
    #[error("--image-reserve-gb {value} must be a finite number of GiB, at least 0")]
    BadImageReserve {
        /// The value as it was given.
        value: f64,
    },
    /// An `--image-default-size` that is not a servable `WxH`.
    #[error("--image-default-size {value} is not servable: {reason}")]
    BadImageSize {
        /// The value as it was given.
        value: String,
        /// Why it cannot be served, from the size grammar.
        reason: String,
    },
}

impl ServeArgs {
    /// Checks the settings that clap cannot.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Zero`] for a context, port or budget of zero - the cache
    /// silently substitutes a default for a zero budget, and a server that
    /// quietly served a different setting than the one it was given is worse
    /// than one that refuses to start - [`ConfigError::NamelessModel`] when
    /// the checkpoint path cannot name a model, and
    /// [`ConfigError::HalfSpeculative`] when a draft head was given with no
    /// depth to use it at. A depth with no head is left to the family arm,
    /// which is the only party that knows whether that family has a default.
    /// [`ConfigError::ProfileNeedsOneSequence`] and
    /// [`ConfigError::ProfileWithAudio`] when `--profile-dir` is given where a
    /// report could not be one request's kernels.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.ctx == 0 {
            return Err(ConfigError::Zero { flag: "ctx" });
        }
        if self.port == 0 {
            return Err(ConfigError::Zero { flag: "port" });
        }
        if self.kvstore_budget_mb == Some(0) {
            return Err(ConfigError::Zero {
                flag: "kvstore-budget-mb",
            });
        }
        for (value, flag) in [
            (u64::from(self.kvstore_min_tokens), "kvstore-min-tokens"),
            (self.kvstore_staging_mb, "kvstore-staging-mb"),
            (self.kvstore_pump_mb, "kvstore-pump-mb"),
            (self.kvstore_drain_secs, "kvstore-drain-secs"),
            (self.vision_cache_mb, "vision-cache-mb"),
        ] {
            if value == 0 {
                return Err(ConfigError::Zero { flag });
            }
        }
        if self.vision_max_pixels == 0 {
            return Err(ConfigError::Zero {
                flag: "vision-max-pixels",
            });
        }
        if self.max_num_seqs == 0 {
            return Err(ConfigError::Zero {
                flag: "max-num-seqs",
            });
        }
        self.kv_pool_pages()?;
        self.prefill_widths()?;
        if let Some(value) = self.gpu_memory_utilization
            && !(value > 0.0 && value <= 1.0)
        {
            return Err(ConfigError::NotAShare {
                flag: "gpu-memory-utilization",
                value,
            });
        }
        // A path with no last component cannot name a model, and a server whose
        // model list is empty answers every request with "no such model". The
        // explicit name is the escape hatch: an operator who supplied one has
        // already answered the question this check asks.
        match (&self.model, &self.asr_model, &self.image_model) {
            (None, None, None) => return Err(ConfigError::NothingToServe),
            (Some(model), _, _) if self.model_name.is_none() && model.file_stem().is_none() => {
                return Err(ConfigError::NamelessModel {
                    path: model.display().to_string(),
                });
            }
            _ => {}
        }
        if self.diarization_model.is_some() && self.asr_model.is_none() {
            return Err(ConfigError::DiarizerWithoutRecognizer);
        }
        if self.asr_mmproj.is_some() && self.asr_model.is_none() {
            return Err(ConfigError::MmprojWithoutRecognizer);
        }
        if self.audio_max_streams == 0 {
            return Err(ConfigError::Zero {
                flag: "audio-max-streams",
            });
        }
        if self.asr_batch == 0 {
            return Err(ConfigError::Zero { flag: "asr-batch" });
        }
        if self.asr_max_context_tokens == 0 {
            return Err(ConfigError::Zero {
                flag: "asr-max-context-tokens",
            });
        }
        if self.embedding_max_queue == 0 {
            return Err(ConfigError::Zero {
                flag: "embedding-max-queue",
            });
        }
        self.image_queue()?;
        // A generation with no steps has nothing to denoise: the pipeline
        // refuses it by name, and an operator who set the default to zero
        // should hear about it at boot rather than on the first request.
        if self.image_default_steps == Some(0) {
            return Err(ConfigError::Zero {
                flag: "image-default-steps",
            });
        }
        // The default size is parsed at boot, not at request time: a default
        // the first caller trips over is an operator mistake found the
        // expensive way. The wire clamps a caller's odd size; an operator's
        // setting is refused rather than quietly rounded, because a default
        // that silently moved is not the default that was configured.
        // Only the shape is judged here; whether the loaded model's grid and
        // bounds admit the pixels is judged at boot, when the model is known.
        self.image_reserve_bytes()?;
        if let Some(value) = &self.image_default_size
            && let Err(error) = crate::image::size::parse_exact_size(value)
        {
            return Err(ConfigError::BadImageSize {
                value: value.clone(),
                reason: error.to_string(),
            });
        }
        // A head with no depth is refused here: it is a gigabyte of weights
        // made resident that nothing will ever ask a question, and no arm can
        // turn that into anything useful.
        //
        // A depth with NO head is not refused here, and that is deliberate.
        // Whether it is an error depends on the family: the dense arm defaults
        // it to the sidecar's own `mtp_file`, which is where that family has
        // always published its head, so `--spec-k 3` alone is a complete
        // instruction there. The sparse family has no such default and
        // `model_factory` refuses it by name instead. The check therefore
        // belongs to whichever arm can answer it, and what must NOT happen -
        // a server that quietly decodes one token a step while an operator
        // believes it is speculating - is prevented in both arms rather than
        // here.
        if self.draft_model.is_some() && self.spec_k == 0 {
            return Err(ConfigError::HalfSpeculative {
                given: "draft-model",
                missing: "spec-k",
            });
        }
        self.validate_profile()
    }

    /// `--profile-dir` needs every report to be one request's kernels: one
    /// resident sequence, and no audio worker interleaving streams.
    fn validate_profile(&self) -> Result<(), ConfigError> {
        if self.profile_dir.is_none() {
            return Ok(());
        }
        if self.max_num_seqs != 1 {
            return Err(ConfigError::ProfileNeedsOneSequence {
                max_num_seqs: self.max_num_seqs,
            });
        }
        if self.asr_model.is_some() {
            return Err(ConfigError::ProfileWithAudio);
        }
        Ok(())
    }

    /// How wide the scheduler's prefill chunks are, from the two width flags or
    /// from `--max-num-batched-tokens`.
    ///
    /// # Errors
    ///
    /// [`ConfigError::Zero`] for a width or budget of zero, which would never
    /// finish a prompt, and [`ConfigError::BudgetBelowDecodes`] for a budget
    /// the worst-case decode rows fill on their own.
    pub fn prefill_widths(&self) -> Result<PrefillWidths, ConfigError> {
        let Some(budget) = self.max_num_batched_tokens else {
            if self.solo_prefill_chunk == 0 {
                return Err(ConfigError::Zero {
                    flag: "solo-prefill-chunk",
                });
            }
            if self.mixed_prefill_quantum == 0 {
                return Err(ConfigError::Zero {
                    flag: "mixed-prefill-quantum",
                });
            }
            // A budget's widths are what is left of it, and the scheduler rounds
            // those down to the grid itself; widths typed as widths are refused
            // instead, because a width that silently shrank is not the one
            // configured.
            for (flag, value) in [
                ("solo-prefill-chunk", self.solo_prefill_chunk),
                ("mixed-prefill-quantum", self.mixed_prefill_quantum),
            ] {
                if !value.is_multiple_of(crate::scheduler::PREFILL_GRID) {
                    return Err(ConfigError::OffGrid {
                        flag,
                        value,
                        grid: crate::scheduler::PREFILL_GRID,
                    });
                }
            }
            return Ok(PrefillWidths {
                solo_chunk: self.solo_prefill_chunk,
                mixed_quantum: self.mixed_prefill_quantum,
            });
        };
        if budget == 0 {
            return Err(ConfigError::Zero {
                flag: "max-num-batched-tokens",
            });
        }
        let rows_per_seq = usize::try_from(self.spec_k.max(self.spec_lookup))
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        let decode_rows = self.max_num_seqs.saturating_mul(rows_per_seq);
        if budget <= decode_rows {
            return Err(ConfigError::BudgetBelowDecodes {
                budget,
                decode_rows,
            });
        }
        Ok(PrefillWidths {
            solo_chunk: budget,
            mixed_quantum: budget - decode_rows,
        })
    }

    /// The pool `--kv-cache-tokens` asks for, in pages; `None` when it was not
    /// given.
    ///
    /// # Errors
    ///
    /// [`ConfigError::PoolBelowContext`] for a pool that cannot hold one
    /// `--ctx` context - it would evict the span a session is standing on -
    /// and [`ConfigError::PoolTooLarge`] for more pages than a pool counts.
    pub fn kv_pool_pages(&self) -> Result<Option<u32>, ConfigError> {
        let Some(tokens) = self.kv_cache_tokens else {
            return Ok(None);
        };
        if tokens < u64::from(self.ctx) {
            return Err(ConfigError::PoolBelowContext {
                tokens,
                ctx: self.ctx,
            });
        }
        let pages = tokens.div_ceil(u64::from(shiso_core::page::PAGE_TOKENS));
        u32::try_from(pages)
            .map(Some)
            .map_err(|_| ConfigError::PoolTooLarge { tokens })
    }

    /// The share of the device `--gpu-memory-utilization` sizes the pool to,
    /// with the sessions and the late allocations it sets aside; `None` when it
    /// was not given.
    ///
    /// The reserve is the vision tower and `--vision-cache-mb`, when
    /// `--vision` names a file - both are allocated after the model, so the
    /// stack cannot see them - and
    /// [`KV_FIT_HEADROOM_BYTES`]. A tower that lives inside a checkpoint
    /// directory is not counted beyond the headroom.
    ///
    /// # Errors
    ///
    /// Whatever [`shiso_engine::KvFit::new`] reports for a share outside
    /// `(0, 1]` or no sessions, which [`ServeArgs::validate`] has already
    /// refused by flag name.
    pub fn kv_fit(&self) -> shiso_core::CoreResult<Option<shiso_engine::KvFit>> {
        let Some(fraction) = self.gpu_memory_utilization else {
            return Ok(None);
        };
        let sessions = u32::try_from(self.max_num_seqs).unwrap_or(u32::MAX);
        shiso_engine::KvFit::new(fraction, sessions, self.late_reserve_bytes()).map(Some)
    }

    /// The limit a pool given as a token count - `--kv-cache-tokens`, or the
    /// default - is held to: the same sessions and reserve a
    /// `--gpu-memory-utilization` pool sets aside. `None` when a share sizes
    /// the pool instead.
    ///
    /// # Errors
    ///
    /// Whatever [`shiso_engine::KvCheck::new`] reports for no sessions, which
    /// [`ServeArgs::validate`] has already refused by flag name.
    pub fn kv_check(&self) -> shiso_core::CoreResult<Option<shiso_engine::KvCheck>> {
        if self.gpu_memory_utilization.is_some() {
            return Ok(None);
        }
        let sessions = u32::try_from(self.max_num_seqs).unwrap_or(u32::MAX);
        shiso_engine::KvCheck::new(sessions, self.late_reserve_bytes()).map(Some)
    }

    /// Bytes allocated after the stack is built that it cannot see: the vision
    /// tower and the cache of its output, when `--vision` names a file, and
    /// [`KV_FIT_HEADROOM_BYTES`].
    fn late_reserve_bytes(&self) -> u64 {
        let tower = self
            .vision
            .as_deref()
            .and_then(|path| std::fs::metadata(path).ok())
            .filter(std::fs::Metadata::is_file)
            .map_or(0, |meta| {
                meta.len()
                    .saturating_add(self.vision_cache_mb.saturating_mul(1 << 20))
            });
        tower.saturating_add(KV_FIT_HEADROOM_BYTES)
    }

    /// The name this checkpoint answers to on the wire.
    ///
    /// `--model-name` when it was given, and otherwise the file stem, which is
    /// what an operator typed and therefore what they will put in a client's
    /// model field. A checkpoint with no file name has already been refused by
    /// [`ServeArgs::validate`]; the fallback here keeps the accessor total rather
    /// than giving it a second failure mode.
    #[must_use]
    pub fn model_id(&self) -> String {
        if let Some(name) = &self.model_name {
            return name.clone();
        }
        self.model.as_deref().and_then(Path::file_stem).map_or_else(
            || "model".to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        )
    }

    /// The name the recognizer answers to, when one is configured.
    #[must_use]
    pub fn asr_id(&self) -> Option<String> {
        if let Some(name) = &self.asr_id {
            return Some(name.clone());
        }
        let path = self.asr_model.as_deref()?;
        Some(path.file_stem().map_or_else(
            || "asr".to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        ))
    }

    /// The operator's explicit image defaults; a flag left unset stays `None`
    /// so the loaded model's own value applies.
    ///
    /// # Errors
    ///
    /// [`ConfigError::BadImageSize`] for a size `validate` would already have
    /// refused; the parse is kept fallible so no caller has to assume
    /// `validate` ran.
    pub fn image_overrides(&self) -> Result<crate::image::ImageOverrides, ConfigError> {
        let size = self
            .image_default_size
            .as_deref()
            .map(|value| {
                crate::image::size::parse_exact_size(value).map_err(|error| {
                    ConfigError::BadImageSize {
                        value: value.to_owned(),
                        reason: error.to_string(),
                    }
                })
            })
            .transpose()?;
        Ok(crate::image::ImageOverrides {
            steps: self.image_default_steps,
            size,
        })
    }

    /// Refuses an image queue of zero: every request would be turned away.
    fn image_queue(&self) -> Result<(), ConfigError> {
        if self.image_max_queue == 0 {
            return Err(ConfigError::Zero {
                flag: "image-max-queue",
            });
        }
        Ok(())
    }

    /// `--image-reserve-gb` in bytes.
    ///
    /// # Errors
    ///
    /// [`ConfigError::BadImageReserve`] for a negative or non-finite value.
    pub fn image_reserve_bytes(&self) -> Result<u64, ConfigError> {
        let gb = self.image_reserve_gb;
        if !gb.is_finite() || gb < 0.0 {
            return Err(ConfigError::BadImageReserve { value: gb });
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a finite, non-negative GiB count; anything past u64 saturates"
        )]
        let bytes = (gb * f64::from(1u32 << 30)) as u64;
        Ok(bytes)
    }

    /// Where the architecture config is read from.
    ///
    /// Beside the checkpoint, and for a checkpoint *directory* that means inside
    /// it rather than beside it: a directory checkpoint carries its own
    /// `config.json`, which is how the directory loader knows its geometry in the
    /// first place, so its parent is one level too high. The test is on the
    /// filesystem rather than on the spelling because that is the same question
    /// the checkpoint loader asks when it decides which container to open, and
    /// the two must not be able to disagree about what kind of thing `--model`
    /// is.
    #[must_use]
    pub fn config_path(&self) -> PathBuf {
        self.config.clone().unwrap_or_else(|| {
            let model = self.model.as_deref().unwrap_or_else(|| Path::new("."));
            let beside = if model.is_dir() {
                model
            } else {
                model.parent().unwrap_or_else(|| Path::new("."))
            };
            beside.join("config.json")
        })
    }

    /// Where the chat template is read from, when it is a file of its own.
    #[must_use]
    pub fn chat_template_path(&self) -> PathBuf {
        self.chat_template.clone().unwrap_or_else(|| {
            let config = self.config_path();
            config
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("chat_template.jinja")
        })
    }

    /// Where the sampling defaults are read from.
    #[must_use]
    pub fn generation_config_path(&self) -> PathBuf {
        self.generation_config.clone().unwrap_or_else(|| {
            let config = self.config_path();
            config
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("generation_config.json")
        })
    }

    /// Where the tokenizer's own configuration lives, which is the second place
    /// a chat template is found.
    #[must_use]
    pub fn tokenizer_config_path(&self) -> PathBuf {
        let config = self.config_path();
        config
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer_config.json")
    }

    /// Where the vocabulary is read from.
    ///
    /// Beside the config rather than beside the checkpoint: a weights file and
    /// the documents that describe it frequently live in different directories,
    /// and the config is the one the operator has already had to point at.
    #[must_use]
    pub fn tokenizer_path(&self) -> PathBuf {
        self.tokenizer.clone().unwrap_or_else(|| {
            let config = self.config_path();
            config
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("tokenizer.json")
        })
    }

    /// The image limits these flags describe.
    #[must_use]
    pub fn image_limits(&self) -> ImageLimits {
        let pixels = u64::try_from(self.vision_max_pixels).unwrap_or(u64::MAX);
        NonZeroU64::new(pixels).map_or_else(ImageLimits::default, |pixels| {
            ImageLimits::default().with_max_pixels(pixels)
        })
    }

    /// The video decode policy these flags describe, or `None` when video
    /// decoding is off.
    ///
    /// `None` *is* the off state, which is why this returns an option rather
    /// than a policy with a disabled flag inside it: there is then no value
    /// that means "configured but not allowed" for a later caller to
    /// misinterpret.
    #[must_use]
    pub fn video_decode_policy(&self) -> Option<VideoDecodePolicy> {
        if !self.video_allow_decode {
            return None;
        }
        let mut policy = VideoDecodePolicy::default();
        if let (Some(seconds), Some(frames)) = (
            NonZeroU64::new(self.video_decode_timeout_s),
            NonZeroUsize::new(self.video_max_frames),
        ) {
            policy = VideoDecodePolicy::new(seconds, frames);
        }
        #[cfg(feature = "decode-ffmpeg")]
        if let Some(ffmpeg) = self.ffmpeg_policy() {
            let ffmpeg = match NonZeroUsize::new(self.video_max_frames) {
                Some(frames) => ffmpeg.with_max_frames(frames),
                None => ffmpeg,
            };
            policy = policy.with_ffmpeg(ffmpeg);
        }
        Some(policy)
    }

    /// The ffmpeg policy `--ffmpeg-path` describes, or `None` when no binary
    /// was named.
    #[cfg(feature = "decode-ffmpeg")]
    #[must_use]
    pub fn ffmpeg_policy(&self) -> Option<FfmpegPolicy> {
        let path = self.ffmpeg_path.clone()?;
        let mut policy = FfmpegPolicy::new(path);
        if let Some(seconds) = NonZeroU64::new(self.video_decode_timeout_s) {
            policy = policy.with_timeout_secs(seconds);
        }
        Some(policy)
    }

    /// The transcription routes' body limit, in bytes.
    #[must_use]
    pub fn audio_max_upload_bytes(&self) -> usize {
        usize::try_from(self.audio_max_upload_mb.get())
            .unwrap_or(usize::MAX)
            .saturating_mul(1024 * 1024)
    }

    /// The audio decode limits these flags describe.
    #[must_use]
    pub const fn audio_limits(&self) -> shiso_audio::DecodeLimits {
        shiso_audio::DecodeLimits::new(self.audio_max_seconds)
    }

    /// How audio containers other than WAV are decoded.
    ///
    /// There is no audio opt-in flag: any container the build decodes without
    /// a subprocess is accepted, and the ffmpeg step exists only when
    /// `--ffmpeg-path` names a binary. The binary and the timeout are shared
    /// with video: one ffmpeg per server.
    #[must_use]
    pub fn audio_decode_policy(&self) -> AudioDecodePolicy {
        let mut policy = AudioDecodePolicy::default();
        if let Some(seconds) = NonZeroU64::new(self.video_decode_timeout_s) {
            policy = AudioDecodePolicy::new(seconds);
        }
        #[cfg(feature = "decode-ffmpeg")]
        if let Some(ffmpeg) = self.ffmpeg_policy() {
            policy = policy.with_ffmpeg(ffmpeg);
        }
        policy
    }
}

/// `--kv-cache-dtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum KvCacheDtype {
    /// Half precision, two bytes an element.
    F16,
    /// FP8 E4M3 at a static scale of 1, one byte an element.
    #[value(alias = "fp8_e4m3")]
    Fp8,
}

impl From<KvCacheDtype> for shiso_core::page::KvElem {
    fn from(choice: KvCacheDtype) -> Self {
        match choice {
            KvCacheDtype::F16 => Self::F16,
            KvCacheDtype::Fp8 => Self::Fp8E4m3,
        }
    }
}

/// `--prefill-activations`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PrefillActivations {
    /// Half-precision prompt activations against the decoded weights.
    A16,
    /// Prompt activations quantized to the weight's format, on the
    /// block-scaled tensor-core instructions.
    Lowp,
    /// MXFP8 prompt activations into the dense projections only; the routed
    /// experts keep half-precision activations.
    Dense8,
}

impl From<PrefillActivations> for shiso_engine::sparse::matmul::PrefillActivations {
    fn from(choice: PrefillActivations) -> Self {
        match choice {
            PrefillActivations::A16 => Self::A16,
            PrefillActivations::Lowp => Self::Lowp,
            PrefillActivations::Dense8 => Self::Dense8,
        }
    }
}

/// Where the routed expert banks live, as the operator asked for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExpertResidencyChoice {
    /// Resident when they fit, streamed when they do not.
    Auto,
    /// Imported beside the other weights.
    Resident,
    /// Streamed from disk into a pool of expert slots.
    Stream,
}

/// `--image-residency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ImageResidencyChoice {
    /// Resident when everything fits, else swap.
    Auto,
    /// Everything resident; a job that does not fit is refused.
    Resident,
    /// The text encoder and the transformer take turns.
    Swap,
}

impl From<ImageResidencyChoice> for shiso_models::image_registry::ImageResidency {
    fn from(choice: ImageResidencyChoice) -> Self {
        match choice {
            ImageResidencyChoice::Auto => Self::Auto,
            ImageResidencyChoice::Resident => Self::Resident,
            ImageResidencyChoice::Swap => Self::Swap,
        }
    }
}

/// The parser choice specified via CLI flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ParserChoice {
    /// Use the model family's default parser.
    Auto,
    /// Disable the parser (emit raw output as prose).
    None,
    /// Use Gemma 4 tags.
    Gemma4,
    /// Use Qwen 3 tags.
    Qwen3,
    /// Use Qwen 3.5 tags: Qwen 3's reasoning tags around an XML call body.
    Qwen35,
}

impl ParserChoice {
    /// Resolves this choice to a [`ReasoningScheme`] given the family default.
    #[must_use]
    pub fn reasoning(self, default: ReasoningScheme) -> ReasoningScheme {
        match self {
            Self::Auto => default,
            Self::None => ReasoningScheme::None,
            Self::Gemma4 => ReasoningScheme::Gemma4,
            Self::Qwen3 | Self::Qwen35 => ReasoningScheme::Qwen3,
        }
    }

    /// Resolves this choice to a [`ToolCallScheme`] given the family default.
    #[must_use]
    pub fn tools(self, default: ToolCallScheme) -> ToolCallScheme {
        match self {
            Self::Auto => default,
            Self::None => ToolCallScheme::None,
            Self::Gemma4 => ToolCallScheme::Gemma4,
            Self::Qwen3 => ToolCallScheme::Qwen3,
            Self::Qwen35 => ToolCallScheme::Qwen35,
        }
    }
}
/// Output format for download progress (§6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum ProgressStyle {
    /// Bars on TTY, plain text when redirected.
    #[default]
    Auto,
    /// Interactive multi-bar progress display.
    Bars,
    /// Plain line-oriented progress logging.
    Plain,
    /// NDJSON stream for programmatic integration.
    Json,
}

/// Shared hub arguments for registry and downloading commands (§6).
#[derive(Debug, Clone, clap::Args)]
pub struct HubArgs {
    /// Local storage home directory.
    #[arg(long, env = "SHISO_HOME", default_value = "~/.shiso")]
    pub home: String,

    /// Static HTTPS registry URL.
    #[arg(
        long,
        env = "SHISO_REGISTRY_URL",
        default_value = "https://hub.shiso.run/registry/v1"
    )]
    pub registry_url: String,

    /// Hugging Face endpoint URL.
    #[arg(
        long,
        env = "SHISO_HF_ENDPOINT",
        default_value = "https://huggingface.co"
    )]
    pub hf_endpoint: String,

    /// Hugging Face bearer token.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Maximum concurrent file downloads.
    #[arg(long, env = "SHISO_PULL_JOBS", default_value_t = 4)]
    pub jobs: usize,

    /// Progress reporting style (`auto`, `bars`, `plain`, `json`).
    #[arg(long, env = "SHISO_PROGRESS", default_value = "auto")]
    pub progress: ProgressStyle,
}

/// Arguments for `shiso pull <ref>`.
#[derive(Debug, Clone, clap::Args)]
pub struct PullArgs {
    /// Model reference to download (`<name-or-alias>[:<variant-key>]`).
    pub model_ref: String,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso list [--remote] [--task <t>] [--tag <t>]`.
#[derive(Debug, Clone, clap::Args)]
pub struct ListArgs {
    /// Query remote registry index instead of local installations.
    #[arg(long)]
    pub remote: bool,
    /// Filter models by task.
    #[arg(long)]
    pub task: Option<String>,
    /// Filter models by tag.
    #[arg(long)]
    pub tag: Option<String>,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso info <ref>`.
#[derive(Debug, Clone, clap::Args)]
pub struct InfoArgs {
    /// Model reference to describe.
    pub model_ref: String,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso verify <ref> [--repair]`.
#[derive(Debug, Clone, clap::Args)]
pub struct VerifyArgs {
    /// Model reference to verify.
    pub model_ref: String,
    /// Re-download corrupt or missing files.
    #[arg(long)]
    pub repair: bool,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Arguments for `shiso rm <ref> [--force]`.
#[derive(Debug, Clone, clap::Args)]
pub struct RmArgs {
    /// Model reference to remove (`name` removes all variants, `name:key` removes one).
    pub model_ref: String,
    /// Remove even if model is in use by a running server.
    #[arg(short, long)]
    pub force: bool,
    /// Common hub configuration options.
    #[command(flatten)]
    pub hub: HubArgs,
}

/// Classification of a positional model argument (§6.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArgKind {
    /// Path to a checkpoint file or directory.
    Path(PathBuf),
    /// Model reference from the registry.
    RegistryRef(String),
}

/// Classifies a positional model argument as a path or registry reference (§6.6).
#[must_use]
pub fn classify_model_arg(arg: &str) -> ModelArgKind {
    if arg.contains('/') || arg.starts_with('.') || arg.starts_with('~') {
        ModelArgKind::Path(PathBuf::from(arg))
    } else {
        ModelArgKind::RegistryRef(arg.to_string())
    }
}

/// Merges recipe `server_defaults` into `args` respecting precedence:
/// `explicit CLI flag > SHISO_* env var > recipe server_defaults > built-in default` (§6.6).
///
/// Overwrites a field only when `matches.value_source(id) == Some(ValueSource::DefaultValue)`
/// (or when `matches` is `None`).
pub fn apply_recipe_defaults(
    args: &mut ServeArgs,
    matches: Option<&clap::ArgMatches>,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| defaulted(matches, id);

    if let Some(ctx) = defaults.ctx
        && check_override("ctx")
    {
        args.ctx = u32::try_from(ctx).unwrap_or(args.ctx);
        tracing::info!(ctx = args.ctx, "applied recipe default for ctx");
    }

    if let Some(max_num_seqs) = defaults.max_num_seqs
        && check_override("max_num_seqs")
    {
        args.max_num_seqs = max_num_seqs;
        tracing::info!(
            max_num_seqs = args.max_num_seqs,
            "applied recipe default for max_num_seqs"
        );
    }

    if let Some(spec_k) = defaults.spec_k
        && check_override("spec_k")
    {
        args.spec_k = u32::try_from(spec_k).unwrap_or(args.spec_k);
        tracing::info!(spec_k = args.spec_k, "applied recipe default for spec_k");
    }

    if let Some(turn_replay_tokens) = defaults.turn_replay_tokens
        && check_override("turn_replay_tokens")
    {
        args.turn_replay_tokens = turn_replay_tokens;
        tracing::info!(
            turn_replay_tokens = args.turn_replay_tokens,
            "applied recipe default for turn_replay_tokens"
        );
    }

    if let Some(vision_max_pixels) = defaults.vision_max_pixels
        && check_override("vision_max_pixels")
    {
        args.vision_max_pixels = vision_max_pixels;
        tracing::info!(
            vision_max_pixels = args.vision_max_pixels,
            "applied recipe default for vision_max_pixels"
        );
    }

    if let Some(video_fps) = defaults.video_fps
        && check_override("video_fps")
    {
        args.video_fps = f64::from(video_fps);
        tracing::info!(
            video_fps = args.video_fps,
            "applied recipe default for video_fps"
        );
    }

    if let Some(video_max_frames) = defaults.video_max_frames
        && check_override("video_max_frames")
    {
        args.video_max_frames = video_max_frames;
        tracing::info!(
            video_max_frames = args.video_max_frames,
            "applied recipe default for video_max_frames"
        );
    }

    if let Some(asr_lookahead) = defaults.asr_lookahead
        && check_override("asr_lookahead")
    {
        args.asr_lookahead = asr_lookahead;
        tracing::info!(
            asr_lookahead = args.asr_lookahead,
            "applied recipe default for asr_lookahead"
        );
    }

    if let Some(asr_batch) = defaults.asr_batch
        && check_override("asr_batch")
    {
        args.asr_batch = asr_batch;
        tracing::info!(
            asr_batch = args.asr_batch,
            "applied recipe default for asr_batch"
        );
    }

    apply_engine_defaults(args, matches, defaults);
}

/// Whether flag `id` still holds its built-in default, which is the only case
/// a recipe default may replace.
fn defaulted(matches: Option<&clap::ArgMatches>, id: &str) -> bool {
    matches.is_none_or(|m| m.value_source(id) == Some(ValueSource::DefaultValue))
}

/// The recipe defaults that tune the engine rather than the request surface:
/// expert residency, speculation and capture, and the memory and prefill
/// budgets.
fn apply_engine_defaults(
    args: &mut ServeArgs,
    matches: Option<&clap::ArgMatches>,
    defaults: &ServerDefaults,
) {
    let check_override = |id: &str| defaulted(matches, id);

    if let Some(residency) = defaults.expert_residency
        && check_override("expert_residency")
    {
        args.expert_residency = match residency {
            ExpertResidency::Auto => ExpertResidencyChoice::Auto,
            ExpertResidency::Resident => ExpertResidencyChoice::Resident,
            ExpertResidency::Stream => ExpertResidencyChoice::Stream,
        };
        tracing::info!(?residency, "applied recipe default for expert_residency");
    }

    if let Some(kv) = defaults.kv_cache_dtype
        && check_override("kv_cache_dtype")
    {
        args.kv_cache_dtype = match kv {
            shiso_hub::recipe::KvCacheDtype::F16 => KvCacheDtype::F16,
            shiso_hub::recipe::KvCacheDtype::Fp8 => KvCacheDtype::Fp8,
        };
        tracing::info!(?kv, "applied recipe default for kv_cache_dtype");
    }

    if let Some(reserve) = defaults.expert_reserve_gb
        && check_override("expert_reserve_gb")
    {
        args.expert_reserve_gb = reserve;
        tracing::info!(reserve, "applied recipe default for expert_reserve_gb");
    }

    if let Some(slots) = defaults.expert_slots
        && check_override("expert_slots")
    {
        args.expert_slots = slots;
        tracing::info!(slots, "applied recipe default for expert_slots");
    }

    if let Some(spec_lookup) = defaults.spec_lookup
        && check_override("spec_lookup")
    {
        args.spec_lookup = spec_lookup;
        tracing::info!(spec_lookup, "applied recipe default for spec_lookup");
    }

    if let Some(capture_verify) = defaults.capture_verify
        && check_override("capture_verify")
    {
        args.capture_verify = capture_verify;
        tracing::info!(capture_verify, "applied recipe default for capture_verify");
    }

    if let Some(capture_batch) = defaults.capture_batch
        && check_override("capture_batch")
    {
        args.capture_batch = capture_batch;
        tracing::info!(capture_batch, "applied recipe default for capture_batch");
    }

    // An `Option` flag has no default value source, so "not given" is `None`.
    if let Some(fraction) = defaults.gpu_memory_utilization
        && args.gpu_memory_utilization.is_none()
    {
        args.gpu_memory_utilization = Some(fraction);
        tracing::info!(
            fraction,
            "applied recipe default for gpu_memory_utilization"
        );
    }

    if let Some(chunk) = defaults.solo_prefill_chunk
        && check_override("solo_prefill_chunk")
    {
        args.solo_prefill_chunk = chunk;
        tracing::info!(chunk, "applied recipe default for solo_prefill_chunk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_choice_resolves_against_defaults() {
        assert_eq!(
            ParserChoice::Auto.reasoning(ReasoningScheme::Gemma4),
            ReasoningScheme::Gemma4
        );
        assert_eq!(
            ParserChoice::None.reasoning(ReasoningScheme::Gemma4),
            ReasoningScheme::None
        );
        assert_eq!(
            ParserChoice::Qwen3.tools(ToolCallScheme::Gemma4),
            ToolCallScheme::Qwen3
        );
        // Qwen 3.5 shares Qwen 3's reasoning tags and not its call body, so the
        // one choice resolves to two different answers.
        assert_eq!(
            ParserChoice::Qwen35.reasoning(ReasoningScheme::Gemma4),
            ReasoningScheme::Qwen3
        );
        assert_eq!(
            ParserChoice::Qwen35.tools(ToolCallScheme::Gemma4),
            ToolCallScheme::Qwen35
        );
    }
}
