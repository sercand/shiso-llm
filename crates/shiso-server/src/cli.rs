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
//! onto the types the media layer takes, and [`ServeArgs::ffmpeg_policy`] is the
//! only place that mapping happens.

use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};

use shiso_parsers::{ReasoningScheme, ToolCallScheme};
use shiso_vision::{FfmpegPolicy, ImageLimits};

/// The default listen address. Loopback, so an unconfigured server is not on the
/// network.
pub const DEFAULT_BIND: &str = "127.0.0.1";
/// The default port.
pub const DEFAULT_PORT: u16 = 8090;
/// The default context, in tokens.
pub const DEFAULT_CTX: u32 = 4096;
/// The default disk-tier budget, in mebibytes.
pub const DEFAULT_KVSTORE_BUDGET_MB: u64 = 4096;
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
}

/// Everything `shiso serve` accepts.
#[derive(Debug, clap::Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each one is a separate on-off policy an operator sets on the command line, and a state machine over four independent switches would be sixteen states nobody asked for"
)]
pub struct ServeArgs {
    /// Checkpoint to serve. Either a checkpoint file or a checkpoint directory.
    #[arg(short = 'm', long, env = "SHISO_MODEL")]
    pub model: PathBuf,
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
    /// Budget for the disk tier, in mebibytes.
    #[arg(long, env = "SHISO_KVSTORE_BUDGET_MB", default_value_t = DEFAULT_KVSTORE_BUDGET_MB)]
    pub kvstore_budget_mb: u64,
    /// Bearer tokens that may call the inference routes, comma separated.
    ///
    /// `hide_env_values` is set: a token must never appear in `--help`, in a
    /// clap error message, or in a usage dump, and clap echoes an environment
    /// variable's *value* in all three unless told not to.
    #[arg(
        long,
        env = "SHISO_AUTH_TOKENS",
        value_delimiter = ',',
        hide_env_values = true
    )]
    pub auth_tokens: Vec<String>,
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
    /// Taken in place of the draft head's chain only when it is longer and its
    /// first token is the head's own first guess, which is the evidence that
    /// the text is an echo; a rejected proposal is not made again from the
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

    /// The vision tower: a sidecar weights file, or a directory holding one.
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
    /// Allow decoding video by spawning the configured decoder.
    #[arg(long, env = "SHISO_VIDEO_ALLOW_FFMPEG", default_value_t = false)]
    pub video_allow_ffmpeg: bool,
    /// The decoder binary. A bare name is resolved on `PATH`; an absolute path
    /// pins a known build.
    #[arg(long, env = "SHISO_VIDEO_FFMPEG_PATH", default_value = "ffmpeg")]
    pub video_ffmpeg_path: PathBuf,
    /// Frames a second to sample a clip at.
    #[arg(long, env = "SHISO_VIDEO_FPS", default_value_t = shiso_vision::DEFAULT_FPS)]
    pub video_fps: f64,
    /// Frames one clip may contribute.
    #[arg(long, env = "SHISO_VIDEO_MAX_FRAMES", default_value_t = shiso_vision::DEFAULT_MAX_FRAMES)]
    pub video_max_frames: usize,
    /// Wall-clock budget for one decode, in seconds.
    #[arg(long, env = "SHISO_VIDEO_DECODE_TIMEOUT_S", default_value_t = 20)]
    pub video_decode_timeout_s: u64,
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
    /// One of the two settings speculation needs was given without the other.
    #[error("--{given} without --{missing}: speculation needs a draft head and a depth")]
    HalfSpeculative {
        /// The flag that was given.
        given: &'static str,
        /// The one it needs beside it.
        missing: &'static str,
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
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.ctx == 0 {
            return Err(ConfigError::Zero { flag: "ctx" });
        }
        if self.port == 0 {
            return Err(ConfigError::Zero { flag: "port" });
        }
        if self.kvstore_budget_mb == 0 {
            return Err(ConfigError::Zero {
                flag: "kvstore-budget-mb",
            });
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
        if self.model_name.is_none() && self.model.file_stem().is_none() {
            return Err(ConfigError::NamelessModel {
                path: self.model.display().to_string(),
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
    /// The reserve is the vision tower, when `--vision` names a file - it is
    /// loaded after the model, so the stack cannot see it - and
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
    /// tower, when `--vision` names a file, and [`KV_FIT_HEADROOM_BYTES`].
    fn late_reserve_bytes(&self) -> u64 {
        let tower = self
            .vision
            .as_deref()
            .and_then(|path| std::fs::metadata(path).ok())
            .filter(std::fs::Metadata::is_file)
            .map_or(0, |meta| meta.len());
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
        self.model.file_stem().map_or_else(
            || "model".to_owned(),
            |stem| stem.to_string_lossy().into_owned(),
        )
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
            let beside = if self.model.is_dir() {
                self.model.as_path()
            } else {
                self.model.parent().unwrap_or_else(|| Path::new("."))
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

    /// The decode policy these flags describe, or `None` when video decoding is
    /// off.
    ///
    /// `None` *is* the off state, which is why this returns an option rather
    /// than a policy with a disabled flag inside it: there is then no value
    /// that means "configured but not allowed" for a later caller to
    /// misinterpret.
    #[must_use]
    pub fn ffmpeg_policy(&self) -> Option<FfmpegPolicy> {
        if !self.video_allow_ffmpeg {
            return None;
        }
        let mut policy = FfmpegPolicy::new(self.video_ffmpeg_path.clone());
        if let Some(frames) = NonZeroUsize::new(self.video_max_frames) {
            policy = policy.with_max_frames(frames);
        }
        if let Some(seconds) = NonZeroU64::new(self.video_decode_timeout_s) {
            policy = policy.with_timeout_secs(seconds);
        }
        Some(policy)
    }

    /// The bearer tokens, with blank entries dropped.
    ///
    /// A trailing comma in `--auth-tokens a,b,` yields an empty string, and an
    /// empty token would authenticate a request that sent `Authorization:
    /// Bearer ` with nothing after it.
    #[must_use]
    pub fn auth_tokens(&self) -> Vec<String> {
        self.auth_tokens
            .iter()
            .map(|token| token.trim().to_owned())
            .filter(|token| !token.is_empty())
            .collect()
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
