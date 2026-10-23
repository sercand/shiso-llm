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
//! onto the types the media layer takes, and [`ServeOptions::video_decode_policy`]
//! and [`ServeOptions::audio_decode_policy`] are the only places that mapping
//! happens.

use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use shiso_audio::AudioDecodePolicy;
use shiso_parsers::{ReasoningScheme, ToolCallScheme};
#[cfg(feature = "decode-ffmpeg")]
use shiso_vision::FfmpegPolicy;
use shiso_vision::{ImageLimits, VideoDecodePolicy};

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

/// Everything `shiso serve` configures, with no command line in it.
///
/// With the `clap` feature on this derives `clap::Args`, so the binary flattens
/// it into its parser; an embedder builds it from [`Default`] and sets the
/// fields it cares about. The field list exists once: the clap attributes are
/// `cfg_attr`s on the fields themselves rather than a mirror struct.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each one is a separate on-off policy an operator sets on the command line, and a state machine over four independent switches would be sixteen states nobody asked for"
)]
pub struct ServeOptions {
    /// Language model checkpoint to serve. Either a checkpoint file or a
    /// checkpoint directory. Optional when `--asr-model` is given: a server
    /// can serve transcription alone.
    #[cfg_attr(feature = "clap", arg(short = 'm', long, env = "SHISO_MODEL"))]
    pub model: Option<PathBuf>,
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
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_MODEL_NAME"))]
    pub model_name: Option<String>,
    /// The architecture config. Defaults to `config.json` beside `--model`.
    ///
    /// Required in practice wherever the checkpoint directory holds only the
    /// weights file: the geometry is read from the config, never guessed from
    /// the weights.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_CONFIG"))]
    pub config: Option<PathBuf>,
    /// The vocabulary. Defaults to `tokenizer.json` beside the config.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_TOKENIZER"))]
    pub tokenizer: Option<PathBuf>,
    /// The chat template. Defaults to `chat_template.jinja` beside the config,
    /// and then to the `chat_template` key of `tokenizer_config.json`.
    ///
    /// A checkpoint that ships neither still serves: the informational routes
    /// answer, and the generation routes say that no template was found rather
    /// than inventing a prompt format the model has never seen.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_CHAT_TEMPLATE"))]
    pub chat_template: Option<PathBuf>,
    /// Sampling defaults. Defaults to `generation_config.json` beside the config.
    ///
    /// Its `temperature`, `top_k`, `top_p` and `min_p` are what a caller who
    /// names no sampling knob gets, and each knob a caller names overrides its
    /// own. A missing file means neutral sampling; a file that does not parse
    /// stops the server at boot.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_GENERATION_CONFIG"))]
    pub generation_config: Option<PathBuf>,
    /// Address to listen on.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_BIND", default_value = DEFAULT_BIND))]
    pub bind: String,
    /// Port to listen on.
    #[cfg_attr(feature = "clap", arg(short = 'p', long, env = "SHISO_PORT"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_PORT))]
    pub port: u16,
    /// Tokens one session may hold, prompt included.
    #[cfg_attr(feature = "clap", arg(short = 'c', long, env = "SHISO_CTX"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_CTX))]
    pub ctx: u32,
    /// Sequences that may be resident at once.
    ///
    /// Also the sessions whose state a `--gpu-memory-utilization` pool sets
    /// aside before it takes the rest. More than one decode pass carries are
    /// decoded in several passes a step.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_MAX_NUM_SEQS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = crate::scheduler::DEFAULT_MAX_NUM_SEQS))]
    pub max_num_seqs: usize,
    /// Writes one GPU-time-per-kernel report per request into this directory,
    /// as `<kind>-<pid>-<id>.json`.
    ///
    /// Every launch is submitted and timed on its own while a request runs, so
    /// serving is slower: a development tool, not a serving mode. Needs
    /// `--max-num-seqs 1`, so a report is one request's kernels.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_PROFILE_DIR"))]
    pub profile_dir: Option<PathBuf>,
    /// Prompt tokens one prefill chunk covers when nothing else is decoding.
    ///
    /// Wide, because with nobody to starve the only thing a narrow chunk buys is
    /// more passes, and the sparse family reads every routed expert once per
    /// pass. A stack splits a chunk wider than its own widest pass itself.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_SOLO_PREFILL_CHUNK",
        default_value_t = crate::scheduler::DEFAULT_SOLO_CHUNK,
        conflicts_with = "max_num_batched_tokens"
    ))]
    pub solo_prefill_chunk: usize,
    /// Prompt tokens one prefill quantum adds to a step while other sequences
    /// are decoding.
    ///
    /// The trade: a wider quantum brings a waiting prompt's first token sooner,
    /// and makes every decoding sequence's step that much longer.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_MIXED_PREFILL_QUANTUM",
        default_value_t = crate::scheduler::DEFAULT_MIXED_QUANTUM,
        conflicts_with = "max_num_batched_tokens"
    ))]
    pub mixed_prefill_quantum: usize,
    /// Tokens one scheduler step may carry, prompt and decode rows together.
    ///
    /// Sets both prefill widths at once: the whole budget when nothing is
    /// decoding, and what the decode rows leave of it when something is. The
    /// decode rows are counted at their worst, `--max-num-seqs` times
    /// `1 + --spec-k`, so a budget that fits them never lets a mixed step grow
    /// past it. Refused beside either width, since the two would disagree.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_MAX_NUM_BATCHED_TOKENS"))]
    pub max_num_batched_tokens: Option<usize>,
    /// Tokens the KV cache holds, shared by every session.
    ///
    /// The pool is allocated whole at load, so this is device memory spent up
    /// front: `--ctx` caps one session, and this caps all of them together.
    /// Unset, and without `--gpu-memory-utilization`, the pool holds 131072.
    #[cfg_attr(
        feature = "clap",
        arg(
            long,
            env = "SHISO_KV_CACHE_TOKENS",
            conflicts_with = "gpu_memory_utilization"
        )
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
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_GPU_MEMORY_UTILIZATION"))]
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
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_KV_CACHE_DTYPE",
        value_enum,
        default_value_t = KvCacheDtype::F16
    ))]
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
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_PREFILL_ACTIVATIONS",
        value_enum,
        default_value_t = PrefillActivations::A16
    ))]
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
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_KV_CHECKPOINTS",
        default_value_t = shiso_engine::EngineOptions::new().kv_ckpt_slots
    ))]
    pub kv_checkpoints: u32,
    /// Directory for the cache's disk tier. Unset means nothing is persisted.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_DIR"))]
    pub kvstore_dir: Option<PathBuf>,
    /// Budget for the disk tier, in mebibytes. Unset is twice the key/value
    /// pool's bytes: enough for every resident session and as many again that
    /// memory gave up.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_BUDGET_MB"))]
    pub kvstore_budget_mb: Option<u64>,
    /// Checkpoints shorter than this many tokens are neither stored on disk nor
    /// looked up there: below it a file costs more to write and read than the
    /// prefill it saves.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_MIN_TOKENS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_KVSTORE_MIN_TOKENS))]
    pub kvstore_min_tokens: u32,
    /// Host memory the disk tier's write-behind may hold in copied pages, in
    /// mebibytes.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_STAGING_MB"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_KVSTORE_STAGING_MB))]
    pub kvstore_staging_mb: u64,
    /// Checkpoint pages one scheduler tick copies off the device for the disk
    /// writer, in mebibytes: what the write-behind costs a decoding stream per
    /// step.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_PUMP_MB"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_KVSTORE_PUMP_MB))]
    pub kvstore_pump_mb: u64,
    /// How long a shutdown waits for queued disk writes before abandoning them,
    /// in seconds. Keep it under the service manager's stop timeout.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_KVSTORE_DRAIN_SECS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_KVSTORE_DRAIN_SECS))]
    pub kvstore_drain_secs: u64,
    /// Ids the server keeps from finished turns, so the next prompt in a
    /// conversation carries the ids the model generated rather than a
    /// re-encoding of their text that can split differently and miss the
    /// prefix cache. Zero keeps none.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_TURN_REPLAY_TOKENS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_TURN_REPLAY_TOKENS))]
    pub turn_replay_tokens: usize,
    /// Seconds between the engine's info lines of prefill and decode rates,
    /// speculation acceptance and cache use, printed while requests run and
    /// once as they finish. Zero turns the line off.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_LOG_STATS_INTERVAL"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_LOG_STATS_INTERVAL_SECS))]
    pub log_stats_interval: u64,
    /// Which device to run on: `auto`, or one this build compiled.
    ///
    /// `auto` takes the GPU when it opens and the CPU otherwise, logging why
    /// when it passed over a GPU this build has. Naming a device that is not
    /// usable is an error rather than a silent run elsewhere.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_DEVICE", value_enum, default_value_t = crate::device::DeviceChoice::Auto)
    )]
    pub device: crate::device::DeviceChoice,
    /// Replay a decode step's launches from a captured device graph.
    ///
    /// On by default, and off is a diagnostic rather than a second way to
    /// serve: a replay issues the same launches with the same arguments, so the
    /// two arms are meant to answer identically and a difference between them
    /// is a bug in the capture. `--no-decode-graphs` is what an operator turns
    /// on to find out which side of that line a wrong answer is on.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_NO_DECODE_GRAPHS", default_value_t = false)
    )]
    pub no_decode_graphs: bool,
    /// Speculative decode depth. Zero is plain decode.
    ///
    /// Read only by the arm of the model factory whose architecture implements
    /// speculation; a knob one architecture reads belongs to that
    /// architecture's arm rather than to a union-of-all-models options struct.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_SPEC_K", default_value_t = 0))]
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
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_SPEC_LOOKUP", default_value_t = 0)
    )]
    pub spec_lookup: u32,
    /// Capture the speculative verify passes as device graphs, one per row
    /// count and session, as the single-row step already is.
    ///
    /// Off, a verify pass is encoded launch by launch every step: thousands of
    /// launches whose host cost shows as gaps between them. On, the first pass
    /// at each width pays a capture. Read by the sparse family's arm.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_CAPTURE_VERIFY", default_value_t = false)
    )]
    pub capture_verify: bool,
    /// Capture passes over several sessions as device graphs, one per distinct
    /// pass: its sessions in row order and every launch's arguments.
    ///
    /// Off, a decode batch or a batched verify is encoded launch by launch
    /// every step, and at four sessions the device idled 6 ms of each 58 ms
    /// step waiting on the host. Read by the sparse family's arm.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_CAPTURE_BATCH", default_value_t = false)
    )]
    pub capture_batch: bool,
    /// Preserve the ordinary target sampling distribution instead of accepting
    /// target-matching greedy drafts directly.
    ///
    /// Off by default: opportunistic speculation accepts matching greedy drafts
    /// even at non-zero temperature. When set, non-greedy sequences fall back
    /// to ordinary batched decoding on models without a two-phase verifier.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_MTP_EXACT_SAMPLING", default_value_t = false)
    )]
    pub mtp_exact_sampling: bool,
    /// Keep half-precision projections as the checkpoint stores them rather
    /// than repacking them to twelve bits a weight at load.
    ///
    /// The repack is lossless - every weight decodes to its stored half - and
    /// a decode step streams a quarter fewer of those bytes; this is the way
    /// back to the stored form for a comparison. Read by the sparse family's
    /// arm.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_NO_PACK_HALVES", default_value_t = false)
    )]
    pub no_pack_halves: bool,
    /// The draft head: a sidecar weights file beside the checkpoint.
    ///
    /// Absent means one token a step. Present, and with `--spec-k` above zero,
    /// a decode step proposes that many tokens and verifies them in one pass.
    /// The two go together: a head with no depth drafts nothing, and a depth
    /// with no head has nothing to draft with, so either alone is refused
    /// rather than quietly ignored.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_DRAFT_MODEL"))]
    pub draft_model: Option<PathBuf>,
    /// Corpus token counts the draft's vocabulary head is narrowed from.
    ///
    /// One `id count` pair per line. Absent, a draft projects through the whole
    /// vocabulary, which is correct and costs it most of what a target step
    /// costs. A token the narrowed head cannot reach is a guess the target
    /// rejects - a verify row that was paid for either way - so this trades
    /// acceptance for speed and never correctness.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_DRAFT_VOCAB"))]
    pub draft_vocab: Option<PathBuf>,

    /// Device memory kept for the towers' output, in mebibytes: a picture seen
    /// before is neither decoded nor encoded again while its rows are here.
    ///
    /// A picture's rows are one per placeholder token, a few megabytes each,
    /// so the default holds a long agent session's screenshots. It is set aside
    /// before the key/value pool is sized, and only when `--vision` names a
    /// tower.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_VISION_CACHE_MB"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_VISION_CACHE_MB))]
    pub vision_cache_mb: u64,

    /// Row-cache budget for the per-layer embedding table, in mebibytes.
    ///
    /// Read only by families with a per-layer embedding table; every other
    /// architecture's arm ignores it. The table is streamed off the checkpoint
    /// rather than held on the device, and this is what a decode step is allowed
    /// to keep of it between steps. Zero disables the cache, which reads every
    /// row from disk and is the slow, correct answer.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_PLE_CACHE_MB"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_PLE_CACHE_MB))]
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
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_PLE_RESIDENT", default_value_t = false)
    )]
    pub ple_resident: bool,

    /// Where a sparse checkpoint's routed expert banks live: imported beside
    /// the other weights, or streamed from disk into a pool of expert slots.
    ///
    /// Read only by the sparse family's MLX directory. `auto` imports them
    /// when every weight fits in the device's working set with
    /// `--expert-reserve-gb` to spare, and streams them otherwise; the other
    /// two insist, and `resident` on a device that cannot hold them fails at
    /// the first allocation that does not fit.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_EXPERT_RESIDENCY", value_enum)
    )]
    #[cfg_attr(feature = "clap", arg(default_value_t = ExpertResidencyChoice::Auto))]
    pub expert_residency: ExpertResidencyChoice,
    /// Gibibytes of the device's working set the automatic residency choice
    /// leaves for the cache, the activations and the n-gram rows.
    ///
    /// Larger is a smaller expert pool and a larger cache; the pool is sized
    /// from what is left.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_EXPERT_RESERVE_GB", default_value_t = 6)
    )]
    pub expert_reserve_gb: u64,
    /// Expert slots when the banks are streamed; zero sizes the pool from the
    /// working set.
    ///
    /// One slot holds one expert of one layer, 2.6 MiB for the published
    /// checkpoint.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_EXPERT_SLOTS", default_value_t = 0)
    )]
    pub expert_slots: u32,

    /// The per-layer n-gram embedding table, for a checkpoint that keeps it in
    /// a file of its own: the sparse family's trellis export.
    ///
    /// Required by that checkpoint and refused beside any other: the table is
    /// read from here and never looked for by name beside `--model`.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_NGRAM"))]
    pub ngram: Option<PathBuf>,

    /// The vision tower: a sidecar weights file, a multimodal-projector GGUF
    /// (`mmproj-*.gguf`), or a checkpoint directory whose config names its
    /// tower shard.
    ///
    /// Absent means text only, and the encoder is never loaded.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_VISION"))]
    pub vision: Option<PathBuf>,
    /// Pixel ceiling one image may occupy after resizing.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_VISION_MAX_PIXELS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = DEFAULT_VISION_MAX_PIXELS))]
    pub vision_max_pixels: usize,
    /// Allow fetching media named by an http or https address.
    ///
    /// Off by default, and the refusal happens before any network call: a
    /// server that fetches whatever a request names is a request-forgery
    /// primitive that happens to also caption pictures.
    #[cfg_attr(
        feature = "clap",
        arg(
            long,
            env = "SHISO_VISION_ALLOW_REMOTE_IMAGES",
            default_value_t = false
        )
    )]
    pub vision_allow_remote_images: bool,
    /// Allow decoding video other than GIF: the OS decoder, then the pure-Rust
    /// one, then ffmpeg when `--ffmpeg-path` is set.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_VIDEO_ALLOW_DECODE", default_value_t = false)
    )]
    pub video_allow_decode: bool,
    /// The ffmpeg binary, for the containers and codecs nothing else in this
    /// build decodes. A bare name is resolved on `PATH`; an absolute path pins
    /// a known build. There is no default: without it ffmpeg is never loaded or
    /// spawned, for video or for audio.
    #[cfg(feature = "decode-ffmpeg")]
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_FFMPEG_PATH"))]
    pub ffmpeg_path: Option<PathBuf>,
    /// Frames a second to sample a clip at.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_VIDEO_FPS"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = shiso_vision::DEFAULT_FPS))]
    pub video_fps: f64,
    /// Frames one clip may contribute.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_VIDEO_MAX_FRAMES"))]
    #[cfg_attr(feature = "clap", arg(default_value_t = shiso_vision::DEFAULT_MAX_FRAMES))]
    pub video_max_frames: usize,
    /// Wall-clock budget for one decode, in seconds.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_VIDEO_DECODE_TIMEOUT_S", default_value_t = 20)
    )]
    pub video_decode_timeout_s: u64,
    /// The longest audio clip one request may carry, in seconds.
    ///
    /// Checked on the decoded length, before any mel frame is computed. A
    /// tower's own limit still applies beneath it.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_AUDIO_MAX_SECONDS", default_value = "7200")
    )]
    pub audio_max_seconds: NonZeroU32,
    /// The largest transcription upload, in MiB. Only the transcription
    /// routes take it; every other route keeps the JSON body limit.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_AUDIO_MAX_UPLOAD_MB", default_value = "512")
    )]
    pub audio_max_upload_mb: NonZeroU32,
    /// The largest image edit request body, in MiB. Only the image edit
    /// routes take it; every other route keeps the JSON body limit.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_IMAGE_MAX_UPLOAD_MB", default_value = "64")
    )]
    pub image_max_upload_mb: NonZeroU32,
    /// Speech recognizer checkpoint: an export directory or a GGUF file.
    /// Serves `/v1/audio/transcriptions`.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_ASR_MODEL"))]
    pub asr_model: Option<PathBuf>,
    /// The audio-tower GGUF (an `mmproj-*.gguf`) for a recognizer whose GGUF
    /// export keeps its tower apart from the text model. Without it the one
    /// mmproj beside `--asr-model` with the same quantisation suffix is taken,
    /// and none or several is refused. Needs `--asr-model`.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_ASR_MMPROJ"))]
    pub asr_mmproj: Option<PathBuf>,
    /// The name the recognizer answers to in a transcription's `model` field.
    /// Defaults to the last component of `--asr-model`, extension removed.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_ASR_ID"))]
    pub asr_id: Option<String>,
    /// Speaker diarizer checkpoint, for `response_format=diarized_json`.
    /// Needs `--asr-model`.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_DIARIZATION_MODEL"))]
    pub diarization_model: Option<PathBuf>,
    /// Encoder frames of future context the recognizer uses: one of the
    /// lookaheads the checkpoint was trained for, checked when it loads. More
    /// is more accurate and later to emit.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_ASR_LOOKAHEAD", default_value_t = 13)
    )]
    pub asr_lookahead: usize,
    /// Transcriptions advanced at once; the rest wait in the queue.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_AUDIO_MAX_STREAMS", default_value_t = 4)
    )]
    pub audio_max_streams: usize,
    /// Chunks an offline transcription decodes at once, on a recognizer that
    /// batches: one weight read per step serves this many. Streaming
    /// requests decode one at a time whatever it is.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_ASR_BATCH", default_value_t = 8)
    )]
    pub asr_batch: usize,
    /// Whether an offline transcription with a diarizer loaded leaves out
    /// silences of 3 s or more where no speaker is active. Skipped audio
    /// yields no text: speech the diarizer misses there is lost.
    /// `--asr-skip-silence=false` transcribes them.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_ASR_SKIP_SILENCE",
        default_value_t = true,
        num_args = 0..=1,
        default_missing_value = "true",
        require_equals = true,
        action = clap::ArgAction::Set
    ))]
    pub asr_skip_silence: bool,
    /// The shortest clip, in seconds, that a non-streaming transcription
    /// plans whole and decodes in batches, on a recognizer that batches;
    /// shorter clips keep the chunk-by-chunk path. 0 sends every
    /// non-streaming transcription offline.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_ASR_OFFLINE_MIN_SECS",
        default_value_t = crate::audio::DEFAULT_OFFLINE_MIN_SECS
    ))]
    pub asr_offline_min_secs: u32,
    /// The most tokens a transcription's `keywords` and `prompt` may take in
    /// the recognizer's prompt, once worded and tokenized; a request past it
    /// is refused. At most the recognizer's own cap (512 for the Qwen3-ASR
    /// family); every chunk's prefill carries them.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_ASR_MAX_CONTEXT_TOKENS", default_value_t = 256)
    )]
    pub asr_max_context_tokens: usize,
    /// Image generation model to serve: a checkpoint file or a checkpoint
    /// directory. Serves the image generation routes.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_MODEL"))]
    pub image_model: Option<PathBuf>,
    /// The name the image generator answers to on the wire. Defaults to the
    /// name the loaded checkpoint reports for itself.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_MODEL_ID"))]
    pub image_model_id: Option<String>,
    /// Directory holding shared image assets (`text_encoder`, `vae` or
    /// `ae.safetensors`, `tokenizer`) when they are not collocated with
    /// `--image-model`.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_ASSETS_DIR"))]
    pub image_assets_dir: Option<PathBuf>,
    /// Explicit path to the text encoder weights when they are split from
    /// `--image-model`. Overrides where the locator would otherwise look.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_TEXT_ENCODER"))]
    pub image_text_encoder: Option<PathBuf>,
    /// Explicit path to the VAE weights when they are split from
    /// `--image-model`. Overrides where the locator would otherwise look.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_VAE"))]
    pub image_vae: Option<PathBuf>,
    /// Explicit path to the tokenizer when it is split from `--image-model`.
    /// Overrides where the locator would otherwise look.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_TOKENIZER"))]
    pub image_tokenizer: Option<PathBuf>,
    /// Image requests that may wait in the queue beyond the one generating;
    /// the next is refused with a 503 a retry can fix.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_IMAGE_MAX_QUEUE", default_value_t = 16)
    )]
    pub image_max_queue: usize,
    /// Embedding requests that may wait in the queue beyond the one running,
    /// when `--model` is an embedding checkpoint; the next is refused with a
    /// 503 a retry can fix.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_EMBEDDING_MAX_QUEUE", default_value_t = 64)
    )]
    pub embedding_max_queue: usize,
    /// Denoising steps a request that names none gets. More is slower and,
    /// past the point the checkpoint was distilled for, not better.
    /// Unset, the loaded model's own default (or fixed schedule) is used:
    /// 4 for FLUX.2-klein, 16 for the distilled FLUX.3 Klein.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_DEFAULT_STEPS"))]
    pub image_default_steps: Option<u32>,
    /// The image size a request that names none gets, as `WxH`
    /// (e.g. 1024x1024). Both sides must sit on the loaded model's grid and
    /// within its bounds; unset, the model's own default size is used
    /// (1024x1024 on a 16-pixel grid for FLUX.2-klein and FLUX.3 Klein, whose
    /// sides run 64 to 4096).
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_DEFAULT_SIZE"))]
    pub image_default_size: Option<String>,
    /// Which image pipeline components stay loaded across jobs: `auto`
    /// keeps all of them when they fit the device's working set beside a
    /// job and otherwise swaps; `resident` refuses a job that does not fit;
    /// `swap` lets the text encoder and the transformer take turns (the
    /// FLUX.3 Klein and Qwen-Image-2.1 pipelines; FLUX.2-klein refuses it).
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_IMAGE_RESIDENCY",
        value_enum,
        default_value_t = ImageResidencyChoice::Auto
    ))]
    pub image_residency: ImageResidencyChoice,
    /// GiB of the device working set the image pipeline leaves to the rest
    /// of the host when it plans what stays loaded.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_IMAGE_RESERVE_GB", default_value_t = 1.5)
    )]
    pub image_reserve_gb: f64,
    /// Leading denoising steps of every image job run at half the
    /// requested resolution; the job then finishes at its own size. Each
    /// such step costs about a quarter of a full one, at some cost in fine
    /// detail. 0 runs every step at full size. A job must keep at least one
    /// full-size step, and a model without a draft stage refuses a nonzero
    /// count at boot.
    #[cfg_attr(
        feature = "clap",
        arg(long, env = "SHISO_IMAGE_DRAFT_STEPS", default_value_t = 0)
    )]
    pub image_draft_steps: u32,
    /// Transformer blocks `START:END` (end exclusive) the last denoising
    /// step of every image job skips, adding instead the residual the step
    /// before recorded over the same blocks. Saves `(END - START) / blocks`
    /// of that step at some cost in fidelity. The job's last two steps must
    /// both run at full size (see `--image-draft-steps`). Unset, every
    /// block of every step runs.
    #[cfg_attr(feature = "clap", arg(long, env = "SHISO_IMAGE_FINAL_REUSE"))]
    #[cfg_attr(feature = "clap", arg(value_parser = parse_block_span))]
    pub image_final_reuse: Option<BlockSpan>,
    /// Which reasoning parser to use on model output.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_REASONING_PARSER",
        value_enum,
        default_value_t = ParserChoice::Auto
    ))]
    pub reasoning_parser: ParserChoice,
    /// Which tool-call parser to use on model output.
    #[cfg_attr(feature = "clap", arg(
        long,
        env = "SHISO_TOOL_CALL_PARSER",
        value_enum,
        default_value_t = ParserChoice::Auto
    ))]
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
#[cfg(feature = "clap")]
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

/// The option names a caller set itself, as opposed to the ones that still hold
/// their built-in default.
///
/// A name is a field of [`ServeOptions`] spelled as the field (`"ctx"`,
/// `"max_num_seqs"`), which is also the id clap gives the flag. The recipe merge
/// asks it before replacing a value: a registry recipe may tune a setting the
/// operator left alone, never one they chose. The command line builds it from
/// what it parsed - a flag or its `SHISO_*` variable counts as set; an embedder
/// passes the names it set, or nothing, and [`Default`] is the empty set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExplicitlySet(std::collections::BTreeSet<String>);

impl ExplicitlySet {
    /// Whether `name` was set explicitly.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }

    /// Records that `name` was set explicitly.
    pub fn insert(&mut self, name: impl Into<String>) {
        self.0.insert(name.into());
    }

    /// Whether nothing was set explicitly.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<S: Into<String>> FromIterator<S> for ExplicitlySet {
    fn from_iter<I: IntoIterator<Item = S>>(names: I) -> Self {
        Self(names.into_iter().map(Into::into).collect())
    }
}

/// The built-in audio length cap, in seconds; the `default_value` above.
const DEFAULT_AUDIO_MAX_SECONDS: NonZeroU32 = NonZeroU32::new(7200).expect("7200 is not zero");
/// The built-in transcription upload cap, in MiB.
const DEFAULT_AUDIO_MAX_UPLOAD_MB: NonZeroU32 = NonZeroU32::new(512).expect("512 is not zero");
/// The built-in image edit body cap, in MiB.
const DEFAULT_IMAGE_MAX_UPLOAD_MB: NonZeroU32 = NonZeroU32::new(64).expect("64 is not zero");

impl Default for ServeOptions {
    /// The values the command line gives a flag nobody passed.
    fn default() -> Self {
        Self {
            model: None,
            model_name: None,
            config: None,
            tokenizer: None,
            chat_template: None,
            generation_config: None,
            bind: DEFAULT_BIND.to_owned(),
            port: DEFAULT_PORT,
            ctx: DEFAULT_CTX,
            max_num_seqs: crate::scheduler::DEFAULT_MAX_NUM_SEQS,
            profile_dir: None,
            solo_prefill_chunk: crate::scheduler::DEFAULT_SOLO_CHUNK,
            mixed_prefill_quantum: crate::scheduler::DEFAULT_MIXED_QUANTUM,
            max_num_batched_tokens: None,
            kv_cache_tokens: None,
            gpu_memory_utilization: None,
            kv_cache_dtype: KvCacheDtype::F16,
            prefill_activations: PrefillActivations::A16,
            kv_checkpoints: shiso_engine::EngineOptions::new().kv_ckpt_slots,
            kvstore_dir: None,
            kvstore_budget_mb: None,
            kvstore_min_tokens: DEFAULT_KVSTORE_MIN_TOKENS,
            kvstore_staging_mb: DEFAULT_KVSTORE_STAGING_MB,
            kvstore_pump_mb: DEFAULT_KVSTORE_PUMP_MB,
            kvstore_drain_secs: DEFAULT_KVSTORE_DRAIN_SECS,
            turn_replay_tokens: DEFAULT_TURN_REPLAY_TOKENS,
            log_stats_interval: DEFAULT_LOG_STATS_INTERVAL_SECS,
            device: crate::device::DeviceChoice::Auto,
            no_decode_graphs: false,
            spec_k: 0,
            spec_lookup: 0,
            capture_verify: false,
            capture_batch: false,
            mtp_exact_sampling: false,
            no_pack_halves: false,
            draft_model: None,
            draft_vocab: None,
            vision_cache_mb: DEFAULT_VISION_CACHE_MB,
            ple_cache_mb: DEFAULT_PLE_CACHE_MB,
            ple_resident: false,
            expert_residency: ExpertResidencyChoice::Auto,
            expert_reserve_gb: 6,
            expert_slots: 0,
            ngram: None,
            vision: None,
            vision_max_pixels: DEFAULT_VISION_MAX_PIXELS,
            vision_allow_remote_images: false,
            video_allow_decode: false,
            #[cfg(feature = "decode-ffmpeg")]
            ffmpeg_path: None,
            video_fps: shiso_vision::DEFAULT_FPS,
            video_max_frames: shiso_vision::DEFAULT_MAX_FRAMES,
            video_decode_timeout_s: 20,
            audio_max_seconds: DEFAULT_AUDIO_MAX_SECONDS,
            audio_max_upload_mb: DEFAULT_AUDIO_MAX_UPLOAD_MB,
            image_max_upload_mb: DEFAULT_IMAGE_MAX_UPLOAD_MB,
            asr_model: None,
            asr_mmproj: None,
            asr_id: None,
            diarization_model: None,
            asr_lookahead: 13,
            audio_max_streams: 4,
            asr_batch: 8,
            asr_skip_silence: true,
            asr_offline_min_secs: crate::audio::DEFAULT_OFFLINE_MIN_SECS,
            asr_max_context_tokens: 256,
            image_model: None,
            image_model_id: None,
            image_assets_dir: None,
            image_text_encoder: None,
            image_vae: None,
            image_tokenizer: None,
            image_max_queue: 16,
            embedding_max_queue: 64,
            image_default_steps: None,
            image_default_size: None,
            image_residency: ImageResidencyChoice::Auto,
            image_reserve_gb: 1.5,
            image_draft_steps: 0,
            image_final_reuse: None,
            reasoning_parser: ParserChoice::Auto,
            tool_call_parser: ParserChoice::Auto,
        }
    }
}

impl ServeOptions {
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
    /// `(0, 1]` or no sessions, which [`ServeOptions::validate`] has already
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
    /// [`ServeOptions::validate`] has already refused by flag name.
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
    /// [`ServeOptions::validate`]; the fallback here keeps the accessor total rather
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

    /// The image edit routes' body limit, in bytes.
    #[must_use]
    pub fn image_max_upload_bytes(&self) -> usize {
        usize::try_from(self.image_max_upload_mb.get())
            .unwrap_or(usize::MAX)
            .saturating_mul(1024 * 1024)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum KvCacheDtype {
    /// Half precision, two bytes an element.
    F16,
    /// FP8 E4M3 at a static scale of 1, one byte an element.
    #[cfg_attr(feature = "clap", value(alias = "fp8_e4m3"))]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum ExpertResidencyChoice {
    /// Resident when they fit, streamed when they do not.
    Auto,
    /// Imported beside the other weights.
    Resident,
    /// Streamed from disk into a pool of expert slots.
    Stream,
}

/// `--image-residency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
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
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_options_default_validates() {
        // Nothing to serve is the one thing the defaults cannot supply; with a
        // checkpoint named, nothing else is out of range, so an embedder starts
        // from `default()` and sets only what it needs.
        assert!(matches!(
            ServeOptions::default().validate(),
            Err(ConfigError::NothingToServe)
        ));
        let options = ServeOptions {
            model: Some(PathBuf::from("/models/m.gguf")),
            ..ServeOptions::default()
        };
        options.validate().expect("the defaults are valid");
        assert!(ExplicitlySet::default().is_empty());
    }

    #[test]
    fn explicitly_set_remembers_names() {
        let set: ExplicitlySet = ["ctx", "port"].into_iter().collect();
        assert!(set.contains("ctx"));
        assert!(!set.contains("spec_k"));
    }

    /// The hand-written [`Default`] and the clap defaults are two spellings of
    /// one list; this is what keeps them one list.
    #[cfg(feature = "clap")]
    #[test]
    #[expect(
        clippy::print_stderr,
        reason = "a test that cannot run says on stderr why it skipped"
    )]
    fn default_matches_what_clap_gives_an_empty_command_line() {
        use clap::Parser as _;

        #[derive(Debug, clap::Parser)]
        struct Probe {
            #[command(flatten)]
            options: ServeOptions,
        }
        // `try_parse_from` reads `SHISO_*`; a developer shell with one set would
        // make this test about the shell, and could make the parse itself fail,
        // so look before parsing.
        if std::env::vars().any(|(name, _)| name.starts_with("SHISO_")) {
            eprintln!("SKIP: a SHISO_* variable is set and would override the defaults");
            return;
        }
        let parsed = Probe::try_parse_from(["probe"]).expect("an empty command line parses");
        assert_eq!(parsed.options, ServeOptions::default());
    }

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
