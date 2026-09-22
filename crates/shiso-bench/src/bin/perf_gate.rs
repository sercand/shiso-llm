//! The performance gate: one command that records, one that grades.
//!
//! ```text
//! perf_gate record --target metal|cuda|resume [--family qwen35|gemma4]
//!                  [--model PATH] [--config PATH] [--reference llama-bench.json]
//!                  [--ple-resident]
//! perf_gate check [--family qwen35|gemma4]
//! ```
//!
//! # One device, two families
//!
//! A record says which family it measured, and the gate grades every (target,
//! family) pair a record exists for against a baseline file of its own. A record
//! written before that key existed reads as the family the gate had then, so the
//! history keeps grading exactly as it did. `check` with no `--family` grades all
//! of them, which is what the tree's own gate runs: a family that went ungraded
//! because nobody named it would read as a family with nothing wrong with it.
//!
//! A record may also carry a `reference` section - what another implementation
//! measured on the same checkpoint file, folded in from its own JSON output by
//! `--reference`. Nothing grades against it. It is there so a reader can tell a
//! slow number from a normal one, which a figure with no scale beside it cannot
//! say.
//!
//! `record` runs only on a pinned machine, because that is the only place a
//! number means anything. It measures, checks that the repetitions agreed, and
//! writes a JSON record into `records/`. It refuses to write a record whose
//! repetitions disagreed by more than the noise policy allows: a wide spread
//! means the machine was busy, and the answer to that is to quiesce it and run
//! again, never to record the number anyway.
//!
//! `check` is the gate. It reads committed JSON, recomputes a digest over the
//! sources the records were taken against, and does arithmetic. No model, no
//! device, no network, and it writes nothing at all. Exit 0 when every gated
//! quantity is within its threshold, 1 when one is not, and 2 when there is no
//! record it can trust - which means somebody has to go back to the pinned
//! machine, not that the code is fine.
//!
//! Promoting a record into a baseline is a human copying a file, in a commit
//! whose message cites the old and the new numbers. Nothing here ever writes a
//! baseline: a gate that can move its own reference is not a gate.
//!
//! # The NVIDIA side
//!
//! Recorded, and gating. The box is a DGX Spark carrying a GB10, and the
//! checkpoint is the sparse hybrid one. Unlike the Metal target, this one has
//! no pinned default: it takes both the checkpoint and its config document,
//! because it loads whichever family the document names and a default would
//! pick one for a checkpoint that is the other.
//!
//! ```text
//! cargo run --release -p shiso-bench --bin perf_gate -- record --target cuda \
//!     --model <checkpoint>.gguf --config <sidecar>/config.json
//! ```
//!
//! On a machine without the device this refuses with a typed error and writes
//! nothing. Building it needs `SHISO_CUDA_COMPILE=1`, like everything else that
//! reaches this device: without it the module registry is empty and the first
//! template dispatch is a refusal about a missing capture, not a number.
//!
//! Promoting a record is still a human copying a file: the record's `perf`
//! payload goes into `baseline/nvidia-qwen4exp.json` under the machine's key,
//! committed beside the record with the numbers cited in the message.
//!
//! **What is not here.** A per-frontier comparison against the reference
//! implementation's own numbers - decode and prefill at contexts from 2,048 to
//! 32,768 - needs a record shaped like a frontier table rather than one point,
//! and needs `baseline/reference-q4e.json`, which nobody has produced. The
//! runbook for producing it lived here and was written before the device
//! existed; what it asked for is two of the reference's benches merged into
//! that file, with the clock speed recorded because a throughput figure taken
//! at a different one is a different figure, and the first generated step kept
//! for context and never gated because it is the graph-capture step this
//! protocol drops. The command lines name an upstream project and so live in
//! the provenance section of the workspace README, which is the one file in
//! this repository allowed to name it.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};

use shiso_bench::baseline::{
    Baseline, DEFAULT_FAMILY, GEMMA4_FAMILY, GEMMA4_MLX_FAMILY, Metrics, Pair, PerfEntry,
    QWEN4EXP_FAMILY, Record, Reference, Report, ResumeEntry, SCHEMA_VERSION, Spread, Stale, Target,
    TreeState, baseline_dir, baseline_file, grade_perf, records_dir, source_digest, still_stands,
    workspace_root,
};
use shiso_bench::decode::{CTX, PREFILL_TOKENS, Rig, RigError, decode_ms_per_step};
use shiso_bench::measure::{Protocol, Summary, tokens_per_sec};
use shiso_bench::prefill::prefill_ms;
use shiso_cuda::cuda_available;
use shiso_models::gemma4::PleOptions;

/// The gate's command line.
#[derive(Debug, Parser)]
#[command(about = "records and grades measured performance", long_about = None)]
struct Cli {
    /// What to do.
    #[command(subcommand)]
    command: Command,
}

/// The two halves: measuring, and grading what was measured.
#[derive(Debug, Subcommand)]
enum Command {
    /// Measures on this machine and writes a record. Pinned machines only.
    Record(RecordArgs),
    /// Grades the committed records against the committed baselines.
    Check {
        /// Grade only this family's pairs. Every family that has a record is
        /// graded when this is absent, which is what the tree's own gate wants:
        /// a family left ungraded because nobody named it on the command line
        /// would read as a family with nothing wrong with it.
        #[arg(long)]
        family: Option<FamilyArg>,
    },
}

/// What one `record` run was asked to measure.
#[derive(Debug, clap::Args)]
struct RecordArgs {
    /// What to measure.
    #[arg(long)]
    target: TargetArg,
    /// Which checkpoint family to measure.
    #[arg(long, default_value = "qwen35")]
    family: FamilyArg,
    /// The checkpoint to measure, when it is not where the environment says.
    #[arg(long)]
    model: Option<PathBuf>,
    /// The checkpoint's own config document. Required by the NVIDIA target and
    /// by every Metal family but the dense-hybrid one, both of which read the
    /// family off it; the dense-hybrid Metal arm reads a recorded profile
    /// instead and ignores this.
    #[arg(long)]
    config: Option<PathBuf>,
    /// A JSON array written by `llama-bench`, folded into the record as the
    /// reference section. Informational: nothing here grades against it.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Hold the whole per-layer table in memory instead of streaming it, for a
    /// family that has one. The record says which holding produced its number.
    #[arg(long)]
    ple_resident: bool,
}

/// The targets, as the command line spells them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TargetArg {
    /// Throughput on an Apple GPU.
    Metal,
    /// Throughput on an NVIDIA GPU.
    Cuda,
    /// Checkpoint resume latency, read from the bench that records it.
    Resume,
}

impl From<TargetArg> for Target {
    fn from(arg: TargetArg) -> Self {
        match arg {
            TargetArg::Metal => Self::Metal,
            TargetArg::Cuda => Self::Cuda,
            TargetArg::Resume => Self::Resume,
        }
    }
}

/// The families a run may be pointed at, as the command line spells them.
///
/// An enumeration here and a string in the record: the command line is a place
/// a typo should be refused, and the schema is a place a file written by an
/// older or newer build has to keep parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FamilyArg {
    /// The dense hybrid this gate's first baselines were taken on.
    Qwen35,
    /// Gemma 4 E4B, which carries a per-layer embedding table.
    Gemma4,
    /// Gemma 4 E4B on the MLX affine container.
    Gemma4Mlx,
    /// The sparse hybrid, which the CUDA records written before this key
    /// existed measured under the default family's name.
    ///
    /// Spelled out because clap would derive `qwen4-exp` from the variant and
    /// the record's own field says `qwen4exp`. Every other variant's derived
    /// spelling already equals its [`FamilyArg::as_str`]; a command line that
    /// would not accept the string the file it writes contains is a trap for
    /// whoever reads one and types the other.
    #[value(name = "qwen4exp")]
    Qwen4Exp,
}

impl FamilyArg {
    /// The spelling the record carries.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Qwen35 => DEFAULT_FAMILY,
            Self::Gemma4 => GEMMA4_FAMILY,
            Self::Gemma4Mlx => GEMMA4_MLX_FAMILY,
            Self::Qwen4Exp => QWEN4EXP_FAMILY,
        }
    }
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Record(args) => match record(&args) {
            Ok(path) => {
                tracing::info!(path = %path.display(), "record written");
                ExitCode::SUCCESS
            }
            Err(error) => {
                tracing::error!("{error:#}");
                ExitCode::FAILURE
            }
        },
        Command::Check { family } => match check(family.map(FamilyArg::as_str)) {
            Ok(code) => code,
            Err(error) => {
                tracing::error!("{error:#}");
                // A gate that cannot read its own inputs is stale, not green.
                ExitCode::from(2)
            }
        },
    }
}

/// Measures one target and writes its record.
fn record(args: &RecordArgs) -> Result<PathBuf> {
    let target = Target::from(args.target);
    let family = args.family.as_str();
    let model = args.model.as_deref();
    let config = args.config.as_deref();
    let mut record = match target {
        Target::Metal => record_metal(args.family, model, config, args.ple_resident)?,
        Target::Cuda => record_cuda(model, config)?,
        Target::Resume => record_resume()?,
    };
    family.clone_into(&mut record.family);
    // Only a family that has a per-layer table can have held one, so this is
    // absent rather than defaulted everywhere else.
    if args.family == FamilyArg::Gemma4 || args.family == FamilyArg::Gemma4Mlx {
        record.ple_holding = Some(holding(args.ple_resident).to_owned());
    }
    if let Some(path) = args.reference.as_deref() {
        record.reference = Some(reference(path)?);
    }
    let dir = records_dir();
    std::fs::create_dir_all(&dir).context("the records directory")?;
    let path = dir.join(record.file_name());
    let text = serde_json::to_string_pretty(&record).context("rendering the record")?;
    std::fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Measures throughput on the local NVIDIA device.
///
/// The checkpoint's config document is required rather than defaulted: this
/// target loads whichever family the document names, and a default would pick
/// one for a checkpoint that is the other.
fn record_cuda(model: Option<&Path>, config: Option<&Path>) -> Result<Record> {
    if !cuda_available() {
        return Err(RigError::NoDevice { backend: "CUDA" }.into());
    }
    let (Some(path), Some(config)) = (model, config) else {
        return Err(RigError::Unrecordable {
            target: "cuda",
            why: "this target needs --model and --config; there is no pinned default for it",
        }
        .into());
    };
    let rig = Rig::open_cuda(path, config)?;
    throughput(Target::Cuda, "cuda", rig)
}

/// Measures throughput on the local Metal device.
///
/// The dense-hybrid arm loads through the recorded shape profile, as it always
/// has: its baselines were taken that way and a number is comparable only to
/// one taken the same way. Every other family loads through the dispatcher off
/// the checkpoint's own config document, which is why that document is required
/// for them and ignored for the first. The sparse family is in the second
/// group and has never been recorded on a Metal device; it is here so the
/// family key has one spelling on both targets rather than a match this
/// function refuses.
fn record_metal(
    family: FamilyArg,
    model: Option<&Path>,
    config: Option<&Path>,
    resident: bool,
) -> Result<Record> {
    match family {
        FamilyArg::Qwen35 => {
            let path = model.map_or_else(model_path, Path::to_path_buf);
            let rig = Rig::open(&path)?;
            throughput(Target::Metal, "metal", rig)
        }
        FamilyArg::Gemma4 | FamilyArg::Gemma4Mlx | FamilyArg::Qwen4Exp => {
            let (Some(path), Some(config)) = (model, config) else {
                return Err(RigError::Unrecordable {
                    target: "metal",
                    why: "this family needs --model and --config; the recorded profile beside the \
                          test vectors describes another one",
                }
                .into());
            };
            // The holding the server uses unless it is told otherwise. A record
            // taken with the table resident measures a different machine state,
            // so the caller says which one it wants and the record says which
            // one it got.
            let ple = PleOptions {
                resident,
                ..PleOptions::default()
            };
            let rig = Rig::open_with_ple(path, config, ple)?;
            throughput(Target::Metal, "metal", rig)
        }
    }
}

/// The word a record uses for how the per-layer table was held.
const fn holding(resident: bool) -> &'static str {
    if resident { "resident" } else { "streamed" }
}

/// Reads a `llama-bench` JSON array and folds it into a reference section.
///
/// The tool writes one object per test, and the two this reads are told apart
/// by which of `n_prompt` and `n_gen` is non-zero - the same way the tool's own
/// table labels them `pp<n>` and `tg<n>`. A file that carries neither is a file
/// from some other run, and it is refused rather than recorded as zeroes.
fn reference(path: &Path) -> Result<Reference> {
    let rows: Vec<serde_json::Value> = read_json(path)?;
    let mut prefill = None;
    let mut decode = None;
    for row in &rows {
        let rate = number(row, "avg_ts");
        if number(row, "n_prompt") > 0.0 {
            prefill = Some(rate);
        } else if number(row, "n_gen") > 0.0 {
            decode = Some(rate);
        }
    }
    let (Some(prefill_tok_s), Some(decode_tok_s)) = (prefill, decode) else {
        bail!(
            "{} carries no prompt row and no generation row; it is not a llama-bench result for \
             this protocol",
            path.display()
        );
    };
    // The command line is reconstructed from the rows rather than passed in, so
    // the record cannot claim arguments the figures did not come from: every
    // part of it is a field the tool itself wrote beside the rates.
    let widest = |key: &str| {
        rows.iter()
            .map(|row| number(row, key))
            .fold(0.0_f64, f64::max)
    };
    let first = rows.first();
    let n_prompt = widest("n_prompt");
    let n_gen = widest("n_gen");
    let args = format!(
        "-m {} -p {} -n {} -ngl {} -o json",
        first
            .map(|row| text(row, "model_filename"))
            .unwrap_or_default(),
        n_prompt,
        n_gen,
        widest("n_gpu_layers"),
    );
    Ok(Reference {
        tool: "llama-bench".to_owned(),
        commit: first
            .map(|row| text(row, "build_commit"))
            .unwrap_or_default(),
        decode_tok_s,
        prefill_tok_s,
        args,
        n_prompt: as_count(n_prompt),
        n_gen: as_count(n_gen),
    })
}

/// Turns one of `reference`'s widest-row numbers into the plain count it
/// names, or `None` when the row it came from was absent - which is what a
/// missing field reads as in `widest`, and a benchmark tool that ran a prompt
/// or a generation of zero tokens is a tool that did not run.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a llama-bench token count is a small non-negative integer that fits a u32; \
              `widest` only ever hands this a value read straight off such a count"
)]
fn as_count(value: f64) -> Option<u32> {
    (value > 0.0).then_some(value as u32)
}

/// One numeric field of a JSON object, or zero when it is absent.
///
/// Zero rather than a refusal because the fields this reads are what tells the
/// two rows apart: a row with no `n_prompt` is a row that prompted nothing.
fn number(row: &serde_json::Value, key: &str) -> f64 {
    row.get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0)
}

/// One string field of a JSON object, or the empty string when it is absent.
fn text(row: &serde_json::Value, key: &str) -> String {
    row.get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

/// Runs the recorded protocol over an open rig and folds it into a record.
///
/// One body for both devices: what a repetition measures is the protocol's,
/// not the backend's, and two copies of it would be two definitions of what a
/// record means.
fn throughput(target: Target, backend: &str, mut rig: Rig) -> Result<Record> {
    let protocol = Protocol::standard();

    // Discarded, not measured: the first repetition of a process pays for
    // pipeline state and first-touch of the weights, and it costs about three
    // times what the steady state does. Keeping it would put a number in the
    // record that no later run can reproduce.
    for _ in 0..protocol.warmup_reps {
        decode_ms_per_step(&mut rig, protocol)?;
        prefill_ms(&mut rig, protocol)?;
        tracing::info!("warmup repetition discarded");
    }

    let mut decode_reps = Vec::new();
    let mut prefill_reps = Vec::new();
    for rep in 0..protocol.reps {
        let decode = decode_ms_per_step(&mut rig, protocol)?;
        let prefill = prefill_ms(&mut rig, protocol)?;
        tracing::info!(
            rep,
            decode_ms_per_step = decode,
            prefill_ms = prefill,
            "rep"
        );
        decode_reps.push(decode);
        prefill_reps.push(prefill);
    }
    let name = rig.model_name().to_owned();
    rig.teardown()?;

    let decode = summarize(&decode_reps, "decode")?;
    let prefill = summarize(&prefill_reps, "prefill")?;
    let spread = Spread {
        decode: decode.spread,
        prefill: prefill.spread,
    };
    if !spread.steady() {
        bail!(
            "the repetitions disagreed by {:.3}% (decode {:.3}%, prefill {:.3}%), past the noise \
             policy: quiesce the machine and run again. Nothing was written.",
            spread.widest() * 100.0,
            spread.decode * 100.0,
            spread.prefill * 100.0
        );
    }

    let entry = PerfEntry {
        schema_version: SCHEMA_VERSION,
        machine: machine(),
        os: os_version(),
        date: date(),
        git_rev: git_rev(),
        model: name,
        backend: backend.to_owned(),
        ctx: CTX,
        prefill_tokens: PREFILL_TOKENS,
        prefill_tokens_per_sec: tokens_per_sec(PREFILL_TOKENS, prefill.value),
        decode_tokens: protocol.decode_steps,
        decode_tokens_per_sec: tokens_per_sec(1, decode.value),
        // Zero means "not measured here": the resident-set figure comes from an
        // external observer of the process, because reading it from inside
        // would add a dependency for a number the operating system already has.
        peak_rss_bytes: 0,
        protocol: Some(protocol),
        metrics: Some(Metrics {
            decode_ms_step_median: decode.value,
            prefill_ms_median: prefill.value,
        }),
        spread: Some(spread),
    };
    Ok(Record {
        target,
        // Overwritten by the caller, which is the one that knows which family
        // it pointed this at. Written here rather than threaded through as an
        // argument because nothing in this function measures a family: it runs
        // a protocol over an already-open rig.
        family: DEFAULT_FAMILY.to_owned(),
        git_rev: git_rev(),
        date: date(),
        source_digest: source_digest(&workspace_root()),
        perf: Some(entry),
        resume: None,
        ple: None,
        reference: None,
        ple_holding: None,
    })
}

/// Folds one phase's repetitions.
fn summarize(reps: &[f64], phase: &str) -> Result<Summary> {
    Summary::over(reps).with_context(|| format!("the {phase} phase produced no repetition"))
}

/// Builds a resume record from the file the checkpoint bench wrote.
///
/// Nothing is measured: this reads the numbers that bench recorded and files
/// them as a record so the gate has something to grade. Re-measuring here would
/// be a second implementation of a protocol that already has one.
fn record_resume() -> Result<Record> {
    let path = baseline_dir().join(Target::Resume.baseline_file());
    let file = shiso_bench::kv::read(&path)?;
    let machine = machine();
    let entry = file
        .get(&machine)
        .or_else(|| file.values().next())
        .cloned()
        .with_context(|| format!("{} carries no entry", path.display()))?;
    Ok(Record {
        target: Target::Resume,
        family: DEFAULT_FAMILY.to_owned(),
        git_rev: git_rev(),
        date: date(),
        source_digest: source_digest(&workspace_root()),
        perf: None,
        resume: Some(entry),
        ple: None,
        reference: None,
        ple_holding: None,
    })
}

/// Grades every (target, family) pair that has a record or a baseline, and
/// reports.
///
/// `only` narrows the run to one family; `None` grades all of them. The pairs
/// are not a fixed list: every target is gated for the default family, because
/// that is the pair the tree's history is written in and a missing record there
/// is the thing this gate exists to notice, and every other pair exists exactly
/// when a record for it does.
fn check(only: Option<&str>) -> Result<ExitCode> {
    let root = workspace_root();
    let digest = source_digest(&root);
    let mut report = Report::default();
    let all = every_record()?;

    for (target, family) in pairs(&all, only) {
        let baseline_path = baseline_dir().join(baseline_file(target, &family));
        let records: Vec<Record> = all
            .iter()
            .filter(|record| record.target == target && record.family == family)
            .cloned()
            .collect();
        if !baseline_path.is_file() {
            if records.is_empty() {
                report.absent.push(Pair {
                    target,
                    family: family.clone(),
                });
            } else {
                report.stale.push(Stale::NoBaselineEntry {
                    target,
                    family: family.clone(),
                    machine: records
                        .first()
                        .map(|r| r.machine().to_owned())
                        .unwrap_or_default(),
                });
            }
            continue;
        }
        let Some(record) = freshest(&records, &digest, &mut report, target) else {
            continue;
        };
        grade(target, &family, &record, &baseline_path, &mut report)?;
    }

    let verdict = report.verdict();
    print_report(&report, &digest);
    Ok(ExitCode::from(verdict.code()))
}

/// The pairs one `check` run grades, in the order it reports them.
fn pairs(records: &[Record], only: Option<&str>) -> Vec<(Target, String)> {
    let mut pairs: Vec<(Target, String)> = Vec::new();
    for target in Target::all() {
        let mut families: Vec<String> = vec![DEFAULT_FAMILY.to_owned()];
        for record in records.iter().filter(|record| record.target == target) {
            if !families.contains(&record.family) {
                families.push(record.family.clone());
            }
        }
        families.sort();
        for family in families {
            if only.is_none_or(|wanted| wanted == family) {
                pairs.push((target, family));
            }
        }
    }
    pairs
}

/// Grades one record against the baseline file for its pair.
fn grade(
    target: Target,
    family: &str,
    record: &Record,
    baseline_path: &Path,
    report: &mut Report,
) -> Result<()> {
    let machine = record.machine().to_owned();
    match target {
        Target::Metal | Target::Cuda => {
            let file: Baseline<PerfEntry> = read_json(baseline_path)?;
            let Some(entry) = file.get(&machine) else {
                report.stale.push(Stale::NoBaselineEntry {
                    target,
                    family: family.to_owned(),
                    machine: machine.clone(),
                });
                return Ok(());
            };
            let Some(measured) = record.perf.as_ref() else {
                report.stale.push(Stale::PayloadMissing(target));
                return Ok(());
            };
            grade_perf(target, family, measured, entry, report);
        }
        Target::Resume => {
            let file: Baseline<ResumeEntry> = read_json(baseline_path)?;
            let Some(entry) = file.get(&machine) else {
                report.stale.push(Stale::NoBaselineEntry {
                    target,
                    family: family.to_owned(),
                    machine: machine.clone(),
                });
                return Ok(());
            };
            let Some(measured) = record.resume.as_ref() else {
                report.stale.push(Stale::PayloadMissing(target));
                return Ok(());
            };
            shiso_bench::kv::grade(measured, entry, report);
        }
    }
    if let (Some(measured), Some(base)) =
        (record.ple.as_ref(), baseline_ple(baseline_path, &machine)?)
    {
        shiso_bench::ple::grade_ple(measured, &base, report);
    }
    Ok(())
}

/// The fetch-latency section of a baseline file, when it has one.
///
/// `Ok(None)` means this machine's baseline records no fetch-latency section,
/// which is not a refusal: there is simply nothing to gate against yet.
fn baseline_ple(path: &Path, machine: &str) -> Result<Option<shiso_bench::ple::PleSection>> {
    let value: serde_json::Value = read_json(path)?;
    let section = value
        .as_object()
        .and_then(|map| map.get(machine))
        .and_then(|entry| entry.get("ple"))
        .cloned();
    match section {
        Some(section) => Ok(Some(serde_json::from_value(section)?)),
        None => Ok(None),
    }
}

/// Every record in the tree, newest first.
///
/// Ordered by the file's own modification time, not by its name: a record is
/// named for its target, its date and a git revision, and a revision is a hash,
/// so sorting by name orders by hash as soon as two records share a date. Every
/// record is tried either way, but the reason reported when none of them stands
/// is the first one's - so a name-ordered list blames whichever record happened
/// to sort highest, which is the kind of diagnosis that sends someone
/// re-recording the wrong thing.
///
/// Read once and filtered per pair rather than selected by filename: which pair
/// a record describes is a fact inside it, and a filter over the name would be a
/// second, weaker statement of the same thing - one that a record named for one
/// family and recorded for another would defeat silently.
fn every_record() -> Result<Vec<Record>> {
    let dir = records_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(Vec::new());
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    paths.sort_by_key(|path| {
        std::cmp::Reverse(
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok()
                .map(|time| {
                    time.duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs()
                }),
        )
    });
    let mut records = Vec::new();
    for path in paths {
        records.push(read_json(&path)?);
    }
    Ok(records)
}

/// The newest record that still describes this tree, or `None` with the reason
/// recorded in the report.
fn freshest(
    records: &[Record],
    digest: &str,
    report: &mut Report,
    target: Target,
) -> Option<Record> {
    if records.is_empty() {
        report.stale.push(Stale::NoRecord(target));
        return None;
    }
    let mut first_reason = None;
    for record in records {
        let tree = TreeState {
            digest: digest.to_owned(),
            revision_known: revision_known(&record.git_rev),
        };
        match still_stands(record, &tree) {
            Ok(()) => return Some(record.clone()),
            Err(reason) => {
                first_reason.get_or_insert(reason);
            }
        }
    }
    if let Some(reason) = first_reason {
        report.stale.push(reason);
    }
    None
}

/// Reads and parses one JSON file.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Prints the verdict.
#[expect(
    clippy::print_stdout,
    reason = "the gate's verdict is its output, and a caller reads it from stdout"
)]
fn print_report(report: &Report, digest: &str) {
    println!("perf gate: sources {digest}");
    for comparison in &report.comparisons {
        let ratio = comparison.ratio();
        let verdict = if comparison.passed() { "ok" } else { "FAIL" };
        println!(
            "  {verdict:>4}  {name}: {measured:.4} vs {baseline:.4} baseline, ratio {ratio:.4}",
            name = comparison.name,
            measured = comparison.measured,
            baseline = comparison.baseline,
        );
    }
    for invariant in &report.invariants {
        let verdict = if invariant.held { "ok" } else { "FAIL" };
        println!("  {verdict:>4}  {}", invariant.name);
    }
    for pair in &report.absent {
        println!(
            "  --    {} {}: no baseline and no record, so nothing is gated yet",
            pair.target.as_str(),
            pair.family,
        );
    }
    for stale in &report.stale {
        println!("  STALE {stale}");
    }
    println!("perf gate: {:?}", report.verdict());
}

/// The checkpoint a run measures, when the command line did not name one.
fn model_path() -> PathBuf {
    PathBuf::from(
        std::env::var("SHISO_TEST_MODEL")
            .expect("SHISO_TEST_MODEL must name the checkpoint to measure"),
    )
}

/// Whether this history contains the named commit.
fn revision_known(rev: &str) -> bool {
    if rev.is_empty() {
        return false;
    }
    read_command("git", &["merge-base", "--is-ancestor", rev, "HEAD"]).is_some()
}

/// The machine's own model identifier, which is the key a record is filed
/// under.
fn machine() -> String {
    // A baseline is keyed by this string, so `unknown` is not a fallback: it is
    // a key that matches every machine that could not name itself, which is the
    // one thing a per-machine baseline must not do. Each platform is asked the
    // question it answers, and a machine that answers none is reported as one
    // that did not, deliberately unmergeable with any recorded entry.
    read_command("sysctl", &["-n", "hw.model"])
        .or_else(|| read_file("/sys/devices/virtual/dmi/id/product_name"))
        .or_else(|| read_file("/proc/device-tree/model"))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// The operating system version the run happened on.
fn os_version() -> String {
    if let Some(version) = read_command("sw_vers", &["-productVersion"]) {
        return format!("macOS {version}");
    }
    read_file("/etc/os-release")
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("PRETTY_NAME="))
                .map(|name| name.trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// A file's contents with the trailing whitespace and NULs taken off, or `None`
/// when it is absent or unreadable.
///
/// The device tree's model string is NUL-terminated, which is why the trim is
/// not just whitespace: left on, it travels into a JSON key.
fn read_file(path: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// The commit the build came from, so a number can be traced to a tree.
fn git_rev() -> String {
    read_command("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned())
}

/// The day the record was taken.
fn date() -> String {
    read_command("date", &["-u", "+%Y-%m-%d"]).unwrap_or_else(|| "unknown".to_owned())
}

/// One line of a command's output, or `None` when it failed.
///
/// Run from the workspace root, so the revision questions are asked of this
/// repository rather than of whatever directory the gate was invoked from.
fn read_command(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .current_dir(workspace_root())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    Some(text.trim().to_owned())
}
