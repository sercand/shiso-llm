//! Baselines, records, and the arithmetic that grades one against the other.
//!
//! # The gate never measures
//!
//! A five-per-cent threshold applied to one sample taken on whatever machine
//! happens to be running the gate is not a gate; it is a coin toss with a
//! plausible name. So measurement and grading are split:
//!
//! - measuring happens **only** on a pinned machine, through `record`, which
//!   writes a JSON record into `records/` and refuses to write one whose
//!   repetitions disagreed by more than [`SPREAD_MAX`];
//! - a human promotes a record into `baseline/` by copying it, in a commit
//!   whose message cites the old and new numbers;
//! - the gate, `check`, reads committed JSON and does arithmetic. It needs no
//!   model, no device and no network, it writes nothing, and it produces the
//!   same verdict on every machine.
//!
//! # What makes a record stale
//!
//! A record describes one tree. It keeps standing only while the sources that
//! produced its numbers are byte-for-byte what they were, so every record
//! carries a digest over those sources ([`source_digest`]) and the gate
//! recomputes it. A documentation change, a new test, a new baseline or a new
//! record leaves the digest alone and the record keeps standing; a change under
//! any crate's `src`, under `kernels/`, or to the manifests, lockfile or
//! toolchain pin changes it and the gate demands a re-record.
//!
//! The digest is content, not commit range, for two reasons. It catches an
//! uncommitted edit, which a comparison against the last commit cannot see at
//! all. And it lets the commit that lands a record together with the code the
//! record measures be green, which a commit-range rule cannot: under that rule
//! a record is stale the instant it is committed beside its own sources.
//! Ancestry is still checked - a record has to name a commit this history
//! actually contains - it just is not the staleness test.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::measure::{Protocol, SPREAD_MAX};

/// The regression threshold: a measured cost may exceed its baseline by this
/// factor before the gate fails.
///
/// Five per cent is wide enough to absorb the residual noise a quiesced pinned
/// machine still has after the protocol's stabilisers, and narrow enough that a
/// real regression - a kernel that lost its fast path, a copy that became a
/// round trip - lands well outside it. It is never widened to make a run pass.
pub const GATE_RATIO: f64 = 1.05;

/// The schema version this build writes.
pub const SCHEMA_VERSION: u32 = 2;

/// What a record or a baseline describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    /// Throughput on an Apple GPU.
    Metal,
    /// Throughput on an NVIDIA GPU.
    Cuda,
    /// Checkpoint resume latency, which is read from the recorder's file rather
    /// than measured here.
    Resume,
}

impl Target {
    /// The spelling used in a filename and in the JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Resume => "resume",
        }
    }

    /// The baseline file this target is graded against.
    #[must_use]
    pub const fn baseline_file(self) -> &'static str {
        match self {
            // Named for `hw.model`, which is what the record's own `machine`
            // field carries, so the file and its contents agree about which
            // machine produced the numbers. The plan document calls this file
            // `m1max-...`: it was written expecting an M1 Max dev box and the
            // measurements were taken on a Mac16,10. A baseline whose name
            // claims different silicon than it was recorded on is a trap for
            // whoever compares against it next.
            Self::Metal => "mac16-10-qwen35.json",
            // This port's own numbers from the box that has an NVIDIA
            // device, in the same machine-keyed shape as every other baseline
            // the gate grades. The reference implementation's figures live
            // beside it in a file of their own: they are provenance for a
            // per-frontier comparison, not a baseline this arithmetic reads.
            //
            // It read `nvidia-qwen4exp.json` until Phase 8 - a v1 naming
            // accident, from when the only CUDA record in the tree measured
            // the sparse checkpoint and carried no family key at all, so it
            // read as [`DEFAULT_FAMILY`] and the file it was graded against
            // was named for what it had really measured. Phase 8 recorded the
            // dense hybrid on that device for the first time, and the two
            // stopped being harmlessly conflated the moment they were two
            // different sets of numbers under one key: a Qwen3.5-4B decode was
            // graded against a Qwen3.8-Flash-Next one and failed at ratio
            // 1.0510 and 1.4524 for no reason either model is responsible for.
            // The historical record now names `qwen4exp` in its own `family`
            // field, which is what it always measured, so it keeps grading
            // against `nvidia-qwen4exp.json` exactly as it did.
            Self::Cuda => "nvidia-qwen35.json",
            Self::Resume => "mac16-10-qwen35-resume.json",
        }
    }

    /// The two halves of a baseline filename, either side of the family.
    ///
    /// Read off the name the default family is filed under, so a derived name
    /// for a pair nobody has recorded sits in the same shape as the names that
    /// do exist: the machine stem this target's first baseline was named for,
    /// and whatever the name says after the family - which for the resume
    /// target is what it measured, and for the device targets is nothing.
    #[must_use]
    pub const fn baseline_parts(self) -> (&'static str, &'static str) {
        match self {
            Self::Metal => ("mac16-10", ""),
            Self::Cuda => ("nvidia", ""),
            Self::Resume => ("mac16-10", "-resume"),
        }
    }

    /// Every target the gate knows, in the order it reports them.
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Metal, Self::Cuda, Self::Resume]
    }
}

/// The family a record describes when it does not say.
///
/// Every record written before the schema carried a family was a dense-hybrid
/// one, so reading the absence as this name is what keeps those files meaning
/// what they meant when they were written.
pub const DEFAULT_FAMILY: &str = "qwen35";

/// The Gemma 4 family, as a record spells it.
pub const GEMMA4_FAMILY: &str = "gemma4";
/// The Gemma 4 MLX family, as a record spells it.
pub const GEMMA4_MLX_FAMILY: &str = "gemma4-mlx";

/// The sparse hybrid family, as a record spells it when it names the family
/// explicitly rather than relying on the default.
///
/// A value this gate's own `--family` accepts. Until Phase 8 it was not one:
/// every CUDA record predated the family key and so read as
/// [`DEFAULT_FAMILY`], which put the sparse checkpoint's numbers under the
/// dense hybrid's name. The record that did that now carries this spelling in
/// its own `family` field, so the pair is named for what it measures and
/// [`baseline_file`] derives `nvidia-qwen4exp.json` for it - the same file it
/// has always been graded against.
pub const QWEN4EXP_FAMILY: &str = "qwen4exp";

/// The baseline file one (target, family) pair is graded against.
///
/// A target is no longer enough on its own: one machine can hold a record for
/// two families on the same device, and grading both against one file would
/// have the second family's numbers overwrite the first's the moment either was
/// promoted. So every pair names a file of its own, unless the pair already is
/// another one under an older name:
///
/// - the default family keeps the two Mac historical names, unchanged. The
///   NVIDIA one does not: it read `nvidia-qwen4exp.json` until Phase 8, from
///   when every CUDA record predated the family key and the only one that
///   existed had measured the sparse checkpoint. Recording the dense hybrid on
///   that device made the conflation load-bearing - two checkpoints' numbers
///   under one key, the second graded against the first - so the default CUDA
///   pair is named for its own family now and `(Cuda, "qwen4exp")` derives the
///   file it always had. Nothing about the sparse family's grading moved: the
///   record that measured it says so in its own `family` field;
/// - a pair somebody has recorded is named for the `hw.model` of the machine
///   that recorded it first, which is the convention every file here follows;
/// - a pair nobody has recorded yet gets a derived name, built from the target's
///   own stem and the family. Nothing is at that path, so the gate reports the
///   pair as ungraded rather than grading it against a file that describes other
///   weights. Whoever records that pair first replaces the derived name with one
///   naming their machine, in the commit that lands the baseline - which is what
///   the Metal arm below is.
#[must_use]
pub fn baseline_file(target: Target, family: &str) -> String {
    if family == DEFAULT_FAMILY {
        return target.baseline_file().to_owned();
    }
    if target == Target::Metal && family == GEMMA4_FAMILY {
        return "macbookpro18-4-gemma4.json".to_owned();
    }
    if target == Target::Metal && family == GEMMA4_MLX_FAMILY {
        return "macbookpro18-4-gemma4-mlx.json".to_owned();
    }
    // `(Cuda, "qwen4exp")` needs no arm of its own: the derived name below is
    // `nvidia-qwen4exp.json`, which is the file that pair has always been
    // graded against. It had one until Phase 8, when the CUDA default pair was
    // named for the sparse checkpoint and a derived name would have collided
    // with it.
    let (stem, tail) = target.baseline_parts();
    format!("{stem}-{family}{tail}.json")
}

/// The throughput half of a baseline entry, keyed by machine in the file.
///
/// The five v1 field names are kept and keep their meaning - tokens per second
/// for each phase, the token counts they were measured over, and the resident
/// set an external observer reported - so a reader written against the first
/// schema still works. What v2 adds is the evidence behind them: the protocol
/// that produced the number, the per-step medians the throughput is derived
/// from, and the spread across repetitions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerfEntry {
    /// Schema version; absent means the first one.
    #[serde(default = "schema_v1")]
    pub schema_version: u32,
    /// Hardware identifier, and the key this entry is filed under.
    pub machine: String,
    /// Operating system version.
    #[serde(default)]
    pub os: String,
    /// The day the numbers were taken.
    #[serde(default)]
    pub date: String,
    /// The commit the measured build came from.
    #[serde(default)]
    pub git_rev: String,
    /// The checkpoint file's name.
    #[serde(default)]
    pub model: String,
    /// The device family the numbers describe.
    #[serde(default)]
    pub backend: String,
    /// Context the session was opened with.
    #[serde(default)]
    pub ctx: u32,
    /// Prompt length the prefill phase ran.
    pub prefill_tokens: u32,
    /// Prompt tokens per second, derived from [`Metrics::prefill_ms_median`].
    pub prefill_tokens_per_sec: f64,
    /// Steps the decode phase ran per repetition.
    pub decode_tokens: u32,
    /// Decoded tokens per second, derived from
    /// [`Metrics::decode_ms_step_median`].
    pub decode_tokens_per_sec: f64,
    /// Peak resident set, when an external observer measured it; zero means it
    /// was not measured on this run.
    #[serde(default)]
    pub peak_rss_bytes: u64,
    /// The stabilisers the run used. Absent in v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<Protocol>,
    /// The medians the throughput figures are derived from. Absent in v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    /// Agreement across repetitions. Absent in v1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spread: Option<Spread>,
}

/// Default for a file written before the schema carried a version.
const fn schema_v1() -> u32 {
    1
}

/// The two medians every throughput gate is computed from.
///
/// Milliseconds rather than tokens per second because the gate asks whether a
/// cost grew: a ratio over a cost is monotone in the direction the threshold is
/// written, and a ratio over a rate is not.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Median milliseconds per decode step, across repetitions.
    pub decode_ms_step_median: f64,
    /// Median milliseconds for one prefill of the recorded prompt.
    pub prefill_ms_median: f64,
}

/// Agreement across the outer repetitions, per phase.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Spread {
    /// Slowest over fastest, minus one, for the decode phase.
    pub decode: f64,
    /// The same for the prefill phase.
    pub prefill: f64,
}

impl Spread {
    /// Whether both phases held the noise policy.
    #[must_use]
    pub fn steady(&self) -> bool {
        self.decode <= SPREAD_MAX && self.prefill <= SPREAD_MAX
    }

    /// The wider of the two.
    #[must_use]
    pub fn widest(&self) -> f64 {
        self.decode.max(self.prefill)
    }
}

/// The resume entry, as the checkpoint bench writes it.
///
/// This crate never produces these numbers; it reads them. The bench that does
/// produce them owns the protocol and the byte-exactness check, and this file
/// is consumed exactly as recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResumeEntry {
    /// Hardware identifier, and the key this entry is filed under.
    pub machine: String,
    /// The checkpoint file's name.
    #[serde(default)]
    pub model: String,
    /// Context the sessions were opened with.
    #[serde(default)]
    pub ctx: u32,
    /// Prompt length both runs used.
    pub prompt_tokens: u32,
    /// Wall time of the cold prefill, in milliseconds.
    pub cold_prefill_ms: f64,
    /// Wall time of the resume that replaced it, in milliseconds.
    pub resume_ms: f64,
    /// Tokens the resume stood on.
    pub tokens_reused: u32,
    /// Whether both runs continued identically. A resume that is fast and
    /// different is not a resume, so the gate requires this before it compares
    /// any timing.
    pub continuation_tokens_equal: bool,
}

/// One measurement record, as written into `records/`.
///
/// The payload under `perf` or `resume` is exactly the object a baseline file
/// holds under a machine key, so promoting a record is a copy rather than a
/// translation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// What was measured.
    pub target: Target,
    /// The checkpoint family that was measured.
    ///
    /// A free string rather than an enumeration, and defaulted rather than
    /// required, so that every record written before this key existed still
    /// parses and still means what it meant: the family the gate had at the
    /// time, which is [`DEFAULT_FAMILY`].
    #[serde(default = "default_family")]
    pub family: String,
    /// The commit the measured build came from.
    ///
    /// Carried on the record itself rather than read out of the payload,
    /// because one of the payloads is written by another bench and does not
    /// have the field.
    pub git_rev: String,
    /// The day the record was taken.
    pub date: String,
    /// Digest over the sources that produced these numbers.
    pub source_digest: String,
    /// The throughput payload, for the device targets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perf: Option<PerfEntry>,
    /// The resume payload, for the resume target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resume: Option<ResumeEntry>,
    /// The fetch-latency section, when the run exercised the streaming tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ple: Option<crate::ple::PleSection>,
    /// What another implementation measured on the same file, beside this run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<Reference>,
    /// How the per-layer embedding table was held: `streamed` or `resident`.
    ///
    /// Absent for a family that has no such table, which is why it is an option
    /// rather than a word with a default: a record of a family with no per-layer
    /// table claiming either holding would be describing something that does not
    /// exist. For a family that has one, the holding is a property of the run and
    /// not of the checkpoint, and a record that did not say which one produced
    /// its number would be a number nobody can reproduce.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ple_holding: Option<String>,
}

/// What an independent implementation answered on the same checkpoint, recorded
/// beside this port's own numbers.
///
/// Informational and never graded. The gate compares this port against its own
/// history, because that is the only comparison whose two sides ran the same
/// protocol; a threshold against another implementation's figure would fail the
/// moment that implementation got faster, which is not a regression in anything
/// here. What this section is for is the reader: a number with no scale beside
/// it cannot be judged at all, and the scale for a decode rate is what the same
/// weights do on the same machine under a mature runtime.
///
/// The protocols differ and the record says so by carrying `args`: the other
/// tool's prompt length and generation count are its own, not this protocol's.
/// `n_prompt` and `n_gen` restate the same two numbers out of `args` as plain
/// fields, for a reader that wants them without parsing a command line; they
/// are `Option` because a record written before this key existed has only
/// `args` to say what the tool ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    /// The tool that produced the figures.
    pub tool: String,
    /// The commit that tool was built from.
    pub commit: String,
    /// Its decoded tokens per second.
    pub decode_tok_s: f64,
    /// Its prompt tokens per second.
    pub prefill_tok_s: f64,
    /// The arguments it was run with, so the figures can be reproduced.
    pub args: String,
    /// The prompt length the tool's `pp<n>` row ran, when the record carries
    /// one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_prompt: Option<u32>,
    /// The generation count the tool's `tg<n>` row ran, when the record
    /// carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gen: Option<u32>,
}

/// Default for a record written before the schema carried a family.
fn default_family() -> String {
    DEFAULT_FAMILY.to_owned()
}

impl Record {
    /// The machine key this record is graded against.
    #[must_use]
    pub fn machine(&self) -> &str {
        match (&self.perf, &self.resume) {
            (Some(perf), _) => perf.machine.as_str(),
            (None, Some(resume)) => resume.machine.as_str(),
            (None, None) => "",
        }
    }

    /// The filename this record belongs under: the target, the family when it
    /// is not the default one, the day it was taken, and enough of the revision
    /// to name it.
    ///
    /// The family is in the name because two families can be recorded on one
    /// device on one day from one commit, and without it the second record
    /// would be written over the first. It is left out for the default family
    /// so that the names of the records already in the tree keep describing the
    /// records already in the tree - the gate reads the family out of the file's
    /// contents, never out of its name, so the two spellings cost nothing.
    #[must_use]
    pub fn file_name(&self) -> String {
        let rev: String = self.git_rev.chars().take(12).collect();
        let date = self.date.replace('-', "");
        let target = self.target.as_str();
        if self.family == DEFAULT_FAMILY {
            format!("{target}-{date}-{rev}.json")
        } else {
            format!("{target}-{}-{date}-{rev}.json", self.family)
        }
    }
}

/// A baseline file: one entry per machine, so a run on one machine never
/// erases another's numbers.
pub type Baseline<T> = BTreeMap<String, T>;

/// Why a record cannot be used.
///
/// Every variant means the same thing operationally - go back to the pinned
/// machine and record again - but they are separate so the message says which
/// of them happened, and so a caller can tell a missing record from a changed
/// one without parsing prose.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Stale {
    /// The record names a commit this history does not contain.
    #[error("the record names revision {0}, which this history does not contain")]
    UnknownRevision(String),
    /// The measured sources changed after the record was taken.
    #[error(
        "the {target:?} {family} record's measured sources changed: recorded against \
         {recorded}, this tree is {current}"
    )]
    SourceChanged {
        /// The target the stale record was taken against.
        target: Target,
        /// The family the stale record measured.
        family: String,
        /// Digest the record was taken against.
        recorded: String,
        /// Digest of the tree being gated.
        current: String,
    },
    /// The record's payload does not match the target it claims.
    #[error("the {0:?} record carries no payload for the target it names")]
    PayloadMissing(Target),
    /// The record or its baseline predates the recorded protocol.
    #[error("the {0:?} record or its baseline predates the recorded protocol; record again")]
    NoProtocol(Target),
    /// The repetitions disagreed by more than the noise policy allows.
    #[error("the repetitions disagreed by {spread:.4}, past the noise policy")]
    Unsteady {
        /// The widest spread the record carries.
        spread: f64,
    },
    /// No record at all for a target that has a baseline.
    #[error("no record for {0:?}, which has a baseline to be graded against")]
    NoRecord(Target),
    /// The baseline has no entry for the machine the record was taken on.
    #[error("the {target:?} {family} baseline has no entry for {machine}")]
    NoBaselineEntry {
        /// The target whose baseline was consulted.
        target: Target,
        /// The family whose baseline was consulted. One device now carries two
        /// of them, and a line that named only the device would not say which
        /// file is missing an entry.
        family: String,
        /// The machine the record names.
        machine: String,
    },
}

/// The tree the gate is being run against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeState {
    /// Digest over the measured sources as they stand now.
    pub digest: String,
    /// Whether the record's revision is one this history contains.
    pub revision_known: bool,
}

/// Whether a record still describes the tree being gated.
///
/// Pure, so the rule can be exercised without a repository.
///
/// # Errors
///
/// A [`Stale`] describing the first rule the record failed.
pub fn still_stands(record: &Record, tree: &TreeState) -> Result<(), Stale> {
    if !tree.revision_known {
        return Err(Stale::UnknownRevision(record.git_rev.clone()));
    }
    if record.source_digest != tree.digest {
        return Err(Stale::SourceChanged {
            target: record.target,
            family: record.family.clone(),
            recorded: record.source_digest.clone(),
            current: tree.digest.clone(),
        });
    }
    match record.target {
        Target::Metal | Target::Cuda => {
            let perf = record
                .perf
                .as_ref()
                .ok_or(Stale::PayloadMissing(record.target))?;
            let spread = perf.spread.ok_or(Stale::NoProtocol(record.target))?;
            if !spread.steady() {
                return Err(Stale::Unsteady {
                    spread: spread.widest(),
                });
            }
            if perf.metrics.is_none() {
                return Err(Stale::NoProtocol(record.target));
            }
        }
        Target::Resume => {
            record
                .resume
                .as_ref()
                .ok_or(Stale::PayloadMissing(record.target))?;
        }
    }
    Ok(())
}

/// One (target, family) the gate grades, or reports it cannot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    /// The device or the latency the pair describes.
    pub target: Target,
    /// The checkpoint family.
    pub family: String,
}

/// One gated quantity, measured against its baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// What was compared.
    pub name: String,
    /// The recorded baseline cost.
    pub baseline: f64,
    /// The cost this record carries.
    pub measured: f64,
}

impl Comparison {
    /// Measured over baseline. Infinite when the baseline is not a positive
    /// cost, which fails the gate rather than dividing by zero quietly.
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.baseline > 0.0 {
            self.measured / self.baseline
        } else {
            f64::INFINITY
        }
    }

    /// Whether the measured cost is within the threshold.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.ratio() <= GATE_RATIO
    }
}

/// An assertion that carries no baseline: it either holds or it does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invariant {
    /// What was asserted.
    pub name: String,
    /// Whether it held.
    pub held: bool,
}

/// What the gate concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Every gated quantity is within its threshold.
    Pass,
    /// At least one is not.
    Regression,
    /// The gate could not be run: no usable record.
    Stale,
}

impl Verdict {
    /// The process exit code.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Regression => 1,
            Self::Stale => 2,
        }
    }
}

/// Everything one `check` run concluded, in the order it is printed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// Ratio comparisons against a baseline.
    pub comparisons: Vec<Comparison>,
    /// Assertions with no baseline behind them.
    pub invariants: Vec<Invariant>,
    /// Reasons a target could not be graded.
    pub stale: Vec<Stale>,
    /// Pairs that have neither a baseline nor a record, and so are not gated on
    /// this machine's history yet.
    ///
    /// A pair rather than a target: two families on one device are two things
    /// that can be ungated, and a list of targets would report the device twice
    /// with nothing to tell the two entries apart.
    pub absent: Vec<Pair>,
}

impl Report {
    /// The verdict, and so the exit code.
    ///
    /// Staleness outranks a regression: a record the gate cannot trust cannot
    /// be the evidence for either answer.
    #[must_use]
    pub fn verdict(&self) -> Verdict {
        if !self.stale.is_empty() {
            Verdict::Stale
        } else if self.comparisons.iter().any(|c| !c.passed())
            || self.invariants.iter().any(|i| !i.held)
        {
            Verdict::Regression
        } else {
            Verdict::Pass
        }
    }
}

/// Grades one throughput record against its baseline entry.
///
/// `family` names the checkpoint the pair describes and appears in the
/// comparison's name: one device now carries two of them, and a line that said
/// only which device it graded would be a line a reader cannot act on.
pub fn grade_perf(
    target: Target,
    family: &str,
    record: &PerfEntry,
    baseline: &PerfEntry,
    report: &mut Report,
) {
    let (Some(measured), Some(base)) = (record.metrics, baseline.metrics) else {
        report.stale.push(Stale::NoProtocol(target));
        return;
    };
    let name = target.as_str();
    report.comparisons.push(Comparison {
        name: format!("{name} {family} decode ms/step"),
        baseline: base.decode_ms_step_median,
        measured: measured.decode_ms_step_median,
    });
    report.comparisons.push(Comparison {
        name: format!("{name} {family} prefill ms"),
        baseline: base.prefill_ms_median,
        measured: measured.prefill_ms_median,
    });
}

/// Where the baselines live.
#[must_use]
pub fn baseline_dir() -> PathBuf {
    crate_root().join("baseline")
}

/// Where measurement records live.
#[must_use]
pub fn records_dir() -> PathBuf {
    crate_root().join("records")
}

/// This crate's own directory.
fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The workspace root, which is what the source digest is taken over.
#[must_use]
pub fn workspace_root() -> PathBuf {
    crate_root().join("..").join("..")
}

/// The source paths whose contents a measured number depends on.
///
/// Crate sources and manifests, the kernel trees, the lockfile and the
/// toolchain pin. Tests, documentation, baselines and records are deliberately
/// outside the set: none of them can change what a kernel costs, and putting
/// them in would mean a docs commit demanded a device run.
const MEASURED_ROOTS: [&str; 5] = [
    "crates",
    "kernels",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
];

/// A digest over every measured source under `root`.
///
/// Content-addressed and order-stable: paths are visited in sorted order and
/// both the path and the bytes go into the hash, so a rename is a change.
///
/// # Errors
///
/// Nothing: a path that cannot be read contributes its absence, which changes
/// the digest exactly as an edit would. A gate that refused to run because one
/// file was unreadable would be a gate that could be disabled by deleting a
/// file.
#[must_use]
pub fn source_digest(root: &Path) -> String {
    let mut bytes = Vec::new();
    for entry in MEASURED_ROOTS {
        let path = root.join(entry);
        if path.is_file() {
            // A root named outright is measured by being named; the filter
            // below is for deciding what a directory walk picks up.
            absorb(&path, root, &mut bytes);
        } else {
            collect(&path, root, &mut bytes);
        }
    }
    crate::replay::fnv_digest(&bytes)
}

/// Folds one path - a file, or every measured file under a directory - into the
/// digest input.
fn collect(path: &Path, root: &Path, out: &mut Vec<u8>) {
    if path.is_file() {
        if measured(path) {
            absorb(path, root, out);
        }
        return;
    }
    let Ok(dir) = std::fs::read_dir(path) else {
        return;
    };
    let mut children: Vec<PathBuf> = dir.filter_map(|e| e.ok().map(|e| e.path())).collect();
    children.sort();
    for child in children {
        collect(&child, root, out);
    }
}

/// Folds one file's path and contents into the digest input.
fn absorb(path: &Path, root: &Path, out: &mut Vec<u8>) {
    let relative = path.strip_prefix(root).unwrap_or(path);
    out.extend_from_slice(relative.to_string_lossy().as_bytes());
    out.push(0);
    if let Ok(bytes) = std::fs::read(path) {
        out.extend_from_slice(&bytes);
    }
    out.push(0);
}

/// Whether one file's contents can change a measured number.
///
/// A crate's `src` tree and its manifest can; its `tests`, its `baseline` and
/// its `records` cannot.
fn measured(path: &Path) -> bool {
    let text = path.to_string_lossy();
    if text.contains("/target/") || text.contains("/records/") || text.contains("/baseline/") {
        return false;
    }
    if text.ends_with("Cargo.toml") || text.ends_with("Cargo.lock") {
        return true;
    }
    if text.contains("/src/") {
        return true;
    }
    // Everything under the kernel trees counts: a kernel source, its manifest
    // and its generated headers all reach the device.
    text.contains("/kernels/")
}
