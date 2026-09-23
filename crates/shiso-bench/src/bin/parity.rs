//! The parity runner: replay a fixture against a checkpoint, or capture one.
//!
//! ```text
//! cargo run -p shiso-bench --release --bin parity -- \
//!   --model "$SHISO_TEST_MODEL" --backend metal \
//!   --vectors test-vectors/goldens/qwen35-local-golden.vec [--case ID]... [--strict] \
//!   [--capture OUT.vec --label STR] [--json OUT.json] [--dump-rows DIR] \
//!   [--dump-layers DIR]
//! ```
//!
//! # Rows for an exact KL
//!
//! `--dump-rows DIR` also writes each case's whole logits row, as
//! `DIR/<case-id>.f16.npy` plus `DIR/index.json`, in the layout the independent
//! reference writes (see [`shiso_bench::rows`]). It works beside a capture or a
//! replay, and like a capture it refuses to replace anything already there.
//!
//! # Residual rows for a layer bisection
//!
//! `--dump-layers DIR` writes, for each case's frontier position, the residual
//! streams after the embedding broadcast and after both halves of every layer,
//! as `DIR/<case-id>/{embed,L{nn}.mixer,L{nn}}.f32.npy` plus `DIR/index.json`
//! (see [`shiso_bench::layers`]). Only the sparse family's stack has the tap it
//! reads. The case's final pass runs with device graphs withheld, so every
//! island is encoded directly and the tap can read between them; the launches
//! are the ones a replay would issue, so the row graded beside the dump is the
//! row an untapped run produces. Like the row dump it refuses to replace
//! anything already there.
//!
//! # Replay refuses more than it compares
//!
//! Before a single logit is graded the run checks that it is comparing like
//! with like: the same vocabulary file the capture used, a prompt that is at
//! least as long as the frontier the case declares, and a device that actually
//! opened. Each of those is a typed refusal rather than a comparison, because a
//! parity run that silently graded a different thing would report the most
//! misleading result a test can produce - a pass.
//!
//! # Capture writes once
//!
//! A capture refuses to overwrite an existing fixture. A recorded answer is only
//! worth anything if somebody read it before it was committed, and a capture
//! that could quietly replace yesterday's file would make "the golden moved"
//! indistinguishable from "somebody re-ran the capture".

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use shiso_bench::compare::{Budget, FLOOR, STRICT, Verdict, compare};
use shiso_bench::compare::{TopSlice, top_k};
use shiso_bench::layers::LayerDump;
use shiso_bench::replay::{Device, Harness, Vocabulary};
use shiso_bench::rows::{IndexEntry, RowDump};
use shiso_bench::run::{REPLAY_CHUNK, case_tokens, prompts_digest, run_case_layers, run_case_row};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};

/// Replay or capture a golden-vector fixture against a local checkpoint.
#[derive(Debug, Parser)]
#[command(name = "parity", about, long_about = None)]
struct Args {
    /// The checkpoint to run. Defaults to `SHISO_TEST_MODEL`.
    #[arg(long, env = "SHISO_TEST_MODEL")]
    model: PathBuf,

    /// The device to run on: `metal` or `cuda`.
    #[arg(long, default_value = "metal")]
    backend: String,

    /// The fixture to replay, or the skeleton to capture from.
    #[arg(long)]
    vectors: PathBuf,

    /// Replay only these cases. Repeatable; all cases when absent.
    #[arg(long = "case")]
    cases: Vec<String>,

    /// Grade at the same-source budget instead of the cross-build floor.
    #[arg(long)]
    strict: bool,

    /// Write the completed fixture here instead of grading.
    #[arg(long)]
    capture: Option<PathBuf>,

    /// What this capture is: usually a commit and a machine.
    #[arg(long)]
    label: Option<String>,

    /// Also write a machine-readable summary here.
    #[arg(long)]
    json: Option<PathBuf>,

    /// The directory holding `tokenizer.json`. Defaults to
    /// `SHISO_TEST_MODEL_OPTIQ`.
    #[arg(long, env = "SHISO_TEST_MODEL_OPTIQ")]
    tokenizer: PathBuf,

    /// The model configuration the checkpoint is loaded against.
    #[arg(long)]
    config: Option<PathBuf>,

    /// During a capture, also decode this many greedy tokens per case and print
    /// them.
    ///
    /// Not recorded anywhere: it exists so the person committing a fixture can
    /// read what the checkpoint actually says before deciding that this is the
    /// answer future runs are held to. A golden nobody read is a golden that
    /// locks in whatever was wrong on the day.
    #[arg(long, default_value_t = 0)]
    preview: usize,

    /// Also write each case's full logits row here: `<case-id>.f16.npy` plus
    /// `index.json`.
    ///
    /// The directory is created if needed; a row or index already in it is a
    /// refusal, checked before the checkpoint loads.
    #[arg(long)]
    dump_rows: Option<PathBuf>,

    /// Also write each case's residual streams at every layer boundary here:
    /// `<case-id>/embed.f32.npy`, `<case-id>/L{nn}.mixer.f32.npy` and
    /// `<case-id>/L{nn}.f32.npy`, plus `index.json`.
    ///
    /// Sparse-family checkpoints only. The directory is created if needed; an
    /// index or a case directory already in it is a refusal, checked before
    /// the checkpoint loads.
    #[arg(long)]
    dump_layers: Option<PathBuf>,

    /// Prefill width in tokens: everything before a case's last token is fed
    /// in pieces this wide, then the last token alone.
    ///
    /// The default is the replay width every recorded fixture was graded at.
    /// A wider one puts the prefill's chunk boundaries where a server's are
    /// (8192 for the sparse family), so a frontier on either side of one tells
    /// whether crossing it moves the row. It changes what is measured, not the
    /// fixture, so a capture made at another width says so in its label.
    #[arg(long, default_value_t = REPLAY_CHUNK, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    chunk: usize,
}

/// What every captured fixture says about itself, in the file, in these words.
///
/// It is written into the artifact rather than only into a README because the
/// artifact is what somebody will be looking at when they decide what a failing
/// replay means.
const CAPTURE_NOTE: &str = "\
WHAT THIS FILE IS: a regression lock. It was captured by the same implementation
that replays it, so it proves that this build answers what the build that
captured it answered. It proves nothing about whether either answer is correct.
A parity claim against an independent implementation needs a fixture captured
from that implementation, on hardware that can run it.
WHEN IT FAILS: the answer moved. Find out why. Widening a budget to make this
pass converts a real finding into a permanently weaker test.";

/// Prints one line of the run's report.
#[expect(
    clippy::print_stderr,
    reason = "the per-case report is this binary's output and its shape is part of the format"
)]
fn report(line: &str) {
    eprintln!("{line}");
}

fn main() -> Result<()> {
    // The replay is the only runner that can put this port and the reference on
    // the same tokens, so it is where a divergence gets localised - and the
    // executor's step dump is a `tracing` record, which without a subscriber
    // goes nowhere at all.
    //
    // `SHISO_LOG` and not `RUST_LOG`, for the reason the server gives: every
    // knob this workspace reads is prefixed, and a level that came from an
    // unprefixed variable is the one nobody finds by listing the prefix.
    // Default `error`, because this binary's own output is the per-case report
    // on stderr and an unasked-for `info` would bury it.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("SHISO_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();
    let device = Device::parse(&args.backend)?;
    let fixture = vecfile::read(&args.vectors)
        .with_context(|| format!("reading {}", args.vectors.display()))?;
    let base_dir = args
        .vectors
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);

    let Body::Local(all_cases) = &fixture.body else {
        bail!(
            "{} is a hosted-logprob fixture. That grammar records token bytes and \
             log-probabilities, which this engine cannot produce and this runner therefore \
             cannot replay; it is parsed here so the shape is not invented later.",
            args.vectors.display()
        );
    };
    let cases = select(all_cases, &args.cases)?;
    let ctx = cases.iter().map(|c| c.ctx).max().unwrap_or(0);

    let vocabulary = Vocabulary::load(&args.tokenizer.join("tokenizer.json"))?;
    let provenance = Provenance {
        vocabulary: vocabulary.digest()?,
        prompts: prompts_digest(&base_dir, &cases)?,
    };
    if args.capture.is_none() {
        provenance.check(&fixture)?;
    }

    // Opened before the checkpoint so a used directory costs a second, not a
    // load and a prefill.
    let mut dumps = Dumps::open(&args, &cases)?;

    let config = args.config.clone().unwrap_or_else(default_config);
    let mut harness = Harness::open(device, &args.model, &config, ctx)
        .with_context(|| format!("opening {} for {}", args.model.display(), device.as_str()))?;
    report(&format!(
        "shiso-parity: {} cases, backend {}, ctx {ctx}, prefill chunk {}, vocabulary {}, prompts {}",
        cases.len(),
        device.as_str(),
        args.chunk,
        provenance.vocabulary,
        provenance.prompts,
    ));
    match (&dumps.layers, harness.sparse_layers()) {
        (Some(_), None) => {
            harness.close().context("releasing the device")?;
            bail!(
                "--dump-layers reads the sparse stack's residual tap, and {} is not a \
                 sparse-family checkpoint",
                args.model.display()
            );
        }
        (_, n) => dumps.n_layers = n.unwrap_or(0),
    }

    let outcome = if let Some(out) = args.capture.clone() {
        capture(
            &mut harness,
            &vocabulary,
            &base_dir,
            &cases,
            &CaptureDesc {
                args: &args,
                provenance: &provenance,
                out: &out,
            },
            &mut dumps,
        )
    } else {
        replay(
            &mut harness,
            &vocabulary,
            &base_dir,
            &cases,
            args.strict,
            args.chunk,
            &mut dumps,
        )
    };
    harness.close().context("releasing the device")?;
    dumps.finish()?;
    let verdicts = outcome?;

    if let Some(path) = &args.json {
        write_summary(path, device, &args.model, &verdicts)?;
    }
    let failed = verdicts.iter().filter(|v| !v.passed()).count();
    if failed > 0 {
        bail!(
            "{failed} of {} cases did not hold. A mismatch is a bug to find, not a budget to \
             widen: dump the diverging row, walk the per-op ladder, and name the kernel.",
            verdicts.len()
        );
    }
    Ok(())
}

/// The cases this run touches, in fixture order.
fn select<'a>(all: &'a [LocalCase], wanted: &[String]) -> Result<Vec<&'a LocalCase>> {
    if wanted.is_empty() {
        if all.is_empty() {
            bail!("the fixture declares no cases");
        }
        return Ok(all.iter().collect());
    }
    let mut chosen = Vec::with_capacity(wanted.len());
    for id in wanted {
        let case = all
            .iter()
            .find(|c| &c.id == id)
            .with_context(|| format!("no case named {id} in this fixture"))?;
        chosen.push(case);
    }
    Ok(chosen)
}

/// The two inputs that decide which tokens a case means.
///
/// Recorded in a fixture and rechecked on every replay. Neither is a security
/// measure; both answer the one question a parity failure cannot answer for
/// itself - "am I even comparing the same thing?".
#[derive(Debug)]
struct Provenance {
    /// Digest of the `tokenizer.json` the prompts are encoded with.
    vocabulary: String,
    /// Digest of the prompt files the selected cases name.
    prompts: String,
}

impl Provenance {
    /// Refuses a replay whose inputs are not the capture's inputs.
    ///
    /// A different vocabulary encodes the prompt differently, and an edited
    /// prompt moves the frontier; either one moves every logit. Grading that
    /// would produce a parity failure whose stated cause - the kernels - is the
    /// one thing that did not change.
    ///
    /// A fixture that records neither digest is replayed anyway: the header is
    /// an extension, and a file written before it existed is still a valid
    /// fixture. A fixture that records one and disagrees is refused.
    fn check(&self, fixture: &VecFile) -> Result<()> {
        for (key, mine, fix) in [
            ("vocabulary_digest", &self.vocabulary, "--tokenizer"),
            ("prompts_digest", &self.prompts, "the prompt files"),
        ] {
            let Some(recorded) = fixture.headers.get(key) else {
                continue;
            };
            if recorded != mine {
                bail!(
                    "this fixture records {key} {recorded}; this run computes {mine}. \
                     {fix} is not what the capture used, so the comparison would not be one."
                );
            }
        }
        Ok(())
    }
}

/// The dumps a run writes beside its verdicts, whichever are open.
#[derive(Debug)]
struct Dumps {
    /// Full logits rows.
    rows: Option<RowDump>,
    /// Residual rows at every layer boundary.
    layers: Option<LayerDump>,
    /// Layers of the loaded stack, which the layer dump checks each tap
    /// against; zero until a sparse stack is loaded.
    n_layers: u32,
}

impl Dumps {
    /// Opens the directories the command line names, refusing any that
    /// already hold one of these cases.
    fn open(args: &Args, cases: &[&LocalCase]) -> Result<Self> {
        let ids = || cases.iter().map(|c| c.id.as_str());
        let rows = match &args.dump_rows {
            Some(dir) => {
                let dump = RowDump::create(dir)
                    .with_context(|| format!("opening {} for rows", dir.display()))?;
                dump.check_free(ids())?;
                Some(dump)
            }
            None => None,
        };
        let layers = match &args.dump_layers {
            Some(dir) => {
                let dump = LayerDump::create(dir)
                    .with_context(|| format!("opening {} for layers", dir.display()))?;
                dump.check_free(ids())?;
                Some(dump)
            }
            None => None,
        };
        Ok(Self {
            rows,
            layers,
            n_layers: 0,
        })
    }

    /// Writes each open dump's index.
    ///
    /// Called whether or not every case ran, so a long run that died at its
    /// last case still leaves what it wrote readable.
    fn finish(self) -> Result<()> {
        if let Some(dump) = self.rows {
            let index = dump.finish().context("writing the row index")?;
            report(&format!(
                "shiso-parity: wrote rows, indexed in {}",
                index.display()
            ));
        }
        if let Some(dump) = self.layers {
            let index = dump.finish().context("writing the layer index")?;
            report(&format!(
                "shiso-parity: wrote layers, indexed in {}",
                index.display()
            ));
        }
        Ok(())
    }
}

/// Runs one case, writes its rows when a dump is open, and returns its slice.
fn produce(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    case: &LocalCase,
    chunk: usize,
    dumps: &mut Dumps,
) -> Result<TopSlice> {
    let (produced, tap) = if dumps.layers.is_some() {
        let (row, tap) = run_case_layers(harness, vocabulary, base_dir, case, chunk)?;
        (row, Some(tap))
    } else {
        (
            run_case_row(harness, vocabulary, base_dir, case, chunk)?,
            None,
        )
    };
    // The index names the row's 0-based position, one less than the count of
    // tokens the case feeds; see `shiso_bench::rows`.
    let entry = || -> Result<IndexEntry> {
        let frontier = produced
            .consumed
            .checked_sub(1)
            .context("a case with frontier 0 reads no row")?;
        Ok(IndexEntry {
            prompt_file: case.prompt.to_string_lossy().into_owned(),
            frontier: u32::try_from(frontier)?,
            prompt_tokens: u32::try_from(produced.prompt_tokens)?,
        })
    };
    if let Some(dump) = dumps.rows.as_mut() {
        dump.write(&case.id, &produced.row, entry()?)?;
    }
    if let (Some(layers), Some(tap)) = (dumps.layers.as_mut(), tap) {
        layers.write(&case.id, &tap, dumps.n_layers, entry()?)?;
    }
    Ok(top_k(
        &produced.row,
        usize::try_from(case.top_count).unwrap_or(0),
    ))
}

/// Grades every case and returns the verdicts.
fn replay(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    cases: &[&LocalCase],
    strict: bool,
    chunk: usize,
    dumps: &mut Dumps,
) -> Result<Vec<Verdict>> {
    let budget: &Budget = if strict { &STRICT } else { &FLOOR };
    let mut verdicts = Vec::with_capacity(cases.len());
    for case in cases {
        if case.is_skeleton() {
            bail!(
                "case {} carries no recorded entries. That is a capture skeleton; run with \
                 --capture to fill it.",
                case.id
            );
        }
        let produced = produce(harness, vocabulary, base_dir, case, chunk, dumps)?;
        let verdict = compare(&case.id, &case.tops, &produced, budget);
        report(&verdict.report_line());
        for failure in &verdict.failures {
            report(&format!("shiso-parity:   {failure}"));
        }
        verdicts.push(verdict);
    }
    Ok(verdicts)
}

/// What a capture records beside the rows it runs: where the fixture goes and
/// what its header says.
#[derive(Debug, Clone, Copy)]
struct CaptureDesc<'a> {
    /// The command line, for the label, backend, model and preview.
    args: &'a Args,
    /// The digests the fixture records.
    provenance: &'a Provenance,
    /// Where the completed fixture is written.
    out: &'a Path,
}

/// Fills a skeleton and writes the completed fixture.
fn capture(
    harness: &mut Harness,
    vocabulary: &Vocabulary,
    base_dir: &Path,
    cases: &[&LocalCase],
    desc: &CaptureDesc<'_>,
    dumps: &mut Dumps,
) -> Result<Vec<Verdict>> {
    let CaptureDesc {
        args,
        provenance,
        out,
    } = *desc;
    if out.exists() {
        bail!(
            "{} already exists. A recorded answer is only worth what the person who read it put \
             into it, so this refuses to replace one; move the old file aside deliberately.",
            out.display()
        );
    }
    let label = args
        .label
        .clone()
        .context("a capture needs --label: a fixture that cannot say where it came from cannot be argued with")?;

    let mut filled = Vec::with_capacity(cases.len());
    for case in cases {
        let produced = produce(harness, vocabulary, base_dir, case, args.chunk, dumps)?;
        let preview = vocabulary.decode(&[shiso_core::ids::TokenId::new(
            produced.first().map_or(0, |e| e.token),
        )])?;
        report(&format!(
            "shiso-parity: captured case {} top1={} ({preview:?}) entries={}",
            case.id,
            produced.first().map_or(u32::MAX, |e| e.token),
            produced.len(),
        ));
        if args.preview > 0 {
            let tokens = case_tokens(vocabulary, base_dir, case)?;
            let continued = harness.greedy_after(&tokens, args.chunk, args.preview)?;
            let text = vocabulary.decode(&continued)?;
            report(&format!("shiso-parity:   continuation {text:?}"));
        }
        let mut case = (*case).clone();
        case.top_count = u32::try_from(produced.len()).unwrap_or(0);
        case.tops = produced;
        filled.push(case);
    }

    let mut headers = BTreeMap::new();
    headers.insert("label".to_owned(), label);
    headers.insert("captured_from".to_owned(), "shiso".to_owned());
    headers.insert("backend".to_owned(), args.backend.clone());
    headers.insert(
        "vocabulary_digest".to_owned(),
        provenance.vocabulary.clone(),
    );
    headers.insert("prompts_digest".to_owned(), provenance.prompts.clone());
    headers.insert(
        "model_file".to_owned(),
        args.model
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
    );
    headers.insert(
        "model_bytes".to_owned(),
        std::fs::metadata(&args.model)
            .map(|m| m.len())
            .unwrap_or_default()
            .to_string(),
    );
    let file = VecFile {
        headers,
        notes: CAPTURE_NOTE.lines().map(str::to_owned).collect(),
        body: Body::Local(filled),
    };
    std::fs::write(out, vecfile::write(&file))
        .with_context(|| format!("writing {}", out.display()))?;
    report(&format!(
        "shiso-parity: wrote {} - read its continuation before committing it",
        out.display()
    ));
    Ok(Vec::new())
}

/// Writes the machine-readable half of a run's report.
fn write_summary(path: &Path, device: Device, model: &Path, verdicts: &[Verdict]) -> Result<()> {
    let cases: Vec<serde_json::Value> = verdicts
        .iter()
        .map(|v| {
            serde_json::json!({
                "case": v.id,
                "budget": v.budget,
                "reference_top1": v.reference_top1,
                "candidate_top1": v.candidate_top1,
                "top5_overlap": v.overlap[0],
                "top20_overlap": v.overlap[1],
                "top64_overlap": v.overlap[2],
                "top20_max_abs": v.max_abs,
                "passed": v.passed(),
                "failures": v.failures.iter().map(ToString::to_string).collect::<Vec<_>>(),
            })
        })
        .collect();
    let summary = serde_json::json!({
        "backend": device.as_str(),
        "model": model.file_name().map(|n| n.to_string_lossy().into_owned()),
        "cases": cases,
    });
    std::fs::write(path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// The model configuration a run uses when the caller names none.
fn default_config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/qwen35-config.json")
}
