//! The draft head of NVIDIA's mixed-precision Qwen3.8-Flash-Next export,
//! teacher-forced, against the library reference's own draft head over the
//! same checkpoint.
//!
//! The reference (`scripts/reference/qwen4exp_modelopt_mixed_ref.py mtp`)
//! assembles the checkpoint's `mtp.*` head out of the Python library's own
//! decoder-layer, attention, sparse-MoE and stream-mix modules, every weight
//! dequantized to brain floats and every activation kept in brain floats, and
//! runs it at every position `t` of a prompt: the target's four-stream residual
//! at `t` beside token `t + 1`, attending over the head's own rows `<= t`. It
//! records the head's top 16 log-probabilities for token `t + 2` and the
//! target's own greedy pick there. Nothing in it shares a kernel with this port.
//!
//! Four runs per prompt, through the head's real pass
//! ([`shiso_engine::sparse::stack::SparseStack::draft`]):
//!
//! - **own input**: the target decodes the prompt a token at a time and the
//!   head drafts from the row each step stages, exactly as a speculative step's
//!   first draft does. Its first-position acceptance (`p1`, the head's top 1
//!   against the target's own pick) is the number the served reference reports
//!   per position, and the target's own residual is held against the
//!   reference's. Every position, but every head row written by a one-row pass.
//! - **speculative steps**: the target in verify-shaped passes of one to five
//!   rows, the head's cache rewritten by the multi-row catch-up pass a served
//!   step runs, a pick at each group's last position.
//! - **after prompt**: at sampled positions, the prompt path - whose chunks fill
//!   the head's cache with their own wide passes - then one draft.
//!
//!   A head that agrees in the first run and not in these two is wrong in its
//!   multi-row passes, which only a served step exercises.
//! - **reference input**: the head is handed the reference's own four-stream
//!   rows, so a disagreement belongs to the head alone. At the fixture's probe
//!   positions every stage the head's buffers expose is compared - the normed
//!   hidden term, its projection, the token term's projection, the layer's
//!   input, the residual after the attention half, the layer's output and the
//!   collapsed row - which bisects a defect to one stage in one run.
//!
//!   What the reference's own rounding looks like, measured by running its
//!   head in single precision against the recorded brain-float one
//!   (`mtp-precision`): about 1.6e-3 on the normed hidden term, 3e-3 on the
//!   projections and the layer input, 6e-3 to 1.3e-2 after the attention half
//!   at `t = 0` - where attention is one row and only the value path counts -
//!   and 1e-2 to 6e-2 from `t = 1` on, where the scores count, carried through
//!   to the layer output and the collapsed row (up to 0.15 on a few rows of
//!   `list`). A stage within those of the reference is this port agreeing
//!   with single precision; one well past them is a defect.
//!
//! ```text
//! SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX=<the export directory> cargo test -p shiso-bench \
//!   --release --test mtp_qwen4exp_nvfp4mx -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `SHISO_MTP_DUMP_DIR=<dir>` also writes this port's stage rows at the probe
//! positions, in the reference's own probe-file layout (`<case>.probes.f32`;
//! the two stages the head's buffers do not expose are NaN), for offline
//! comparison.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_bench::compare::TopSlice;
use shiso_bench::replay::{Device, Vocabulary, open_backend};
use shiso_bench::run::logprob_top_k;
use shiso_core::page::KvElem;
use shiso_core::{Backend, DecodeStep, HfConfig, Model, SessionId, TokenId};
use shiso_engine::EngineOptions;
use shiso_engine::sparse::bind::Operand;
use shiso_engine::sparse::capture;
use shiso_engine::sparse::draft::DraftProbe;
use shiso_engine::sparse::matmul::PrefillActivations;
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::recurrent::ChunkPolicy;
use shiso_engine::sparse::scratch::Buf;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::{EXPERTS, N_LAYERS, Qwen4ExpModel, StreamedLoad};

/// The reference records, captured from the library's own modules over the
/// same checkpoint.
const FIXTURE: &str = "references/qwen38-flash-fp4local-hf-mtp.json";

/// The depth of the recorded top.
const TOP: usize = 16;

/// The session ceiling: the fixture's prompts are a few hundred tokens.
const CTX: u32 = 2048;

/// The head's top-1 agreement with the reference's, on the reference's own
/// input, below which the head is wrong rather than rounded differently.
///
/// Set from the reference's own rounding: the same head run in single
/// precision (`qwen4exp_modelopt_mixed_ref.py mtp-precision`) agrees with the
/// recorded brain-float head on 0.992 (math), 0.994 (story) and 0.998 (list) of
/// positions, and this port keeps single-precision activations. Before the
/// head's collapse bound its own mix it agreed on 0.924 of math's.
const HEAD_TOP1_FLOOR: f64 = 0.97;

#[expect(clippy::print_stderr, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    eprintln!("SKIP: {why}");
}

#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The export directory, or `None` when this box has none.
fn model_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX").ok()?);
    path.is_dir().then_some(path)
}

/// One position of a reference case.
struct RefPos {
    /// The token the head is handed (the prompt's `t + 1`).
    next: TokenId,
    /// The target's own greedy pick for `t + 2`.
    target: u32,
    /// The head's top log-probabilities for `t + 2`, highest first.
    draft: Vec<(u32, f32)>,
}

/// One reference case.
struct RefCase {
    name: String,
    tokens: Vec<TokenId>,
    positions: Vec<RefPos>,
    /// The reference's four-stream rows for every position, f32.
    hidden: Vec<f32>,
    /// The stage rows at the probe positions, every stage concatenated.
    probes: Vec<f32>,
    probe_positions: Vec<u32>,
    p1: f64,
}

/// The fixture.
struct Fixture {
    cases: Vec<RefCase>,
    /// Every stage the reference records, with its width in residual widths.
    stages: Vec<(String, usize)>,
    hidden: usize,
    streams: usize,
    vocabulary_digest: String,
}

fn u32_of(v: &serde_json::Value) -> u32 {
    u32::try_from(v.as_u64().expect("an unsigned integer")).expect("a 32-bit value")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the fixture's log-probabilities are single-precision values printed in JSON"
)]
fn parse(path: &Path) -> Fixture {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: serde_json::Value = serde_json::from_str(&text).expect("the fixture parses");
    assert_eq!(
        doc["format"].as_str(),
        Some("shiso-mtp-teacher-forced-v1"),
        "not a teacher-forced draft-head fixture"
    );
    assert!(
        doc["layers"].is_null(),
        "this fixture was recorded from a truncated target, which is a plumbing run"
    );
    let dir = path.parent().expect("the fixture sits in a directory");
    let hidden = usize::try_from(doc["hidden_size"].as_u64().expect("hidden_size")).expect("usize");
    let streams = usize::try_from(doc["streams"].as_u64().expect("streams")).expect("usize");
    let stages = doc["stages"]
        .as_array()
        .expect("stages")
        .iter()
        .map(|s| {
            (
                s[0].as_str().expect("a stage name").to_owned(),
                usize::try_from(s[1].as_u64().expect("a stage width")).expect("usize"),
            )
        })
        .collect();
    let cases = doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| {
            let positions = c["positions"]
                .as_array()
                .expect("positions")
                .iter()
                .map(|p| RefPos {
                    next: TokenId::new(u32_of(&p["next"])),
                    target: u32_of(&p["target"]),
                    draft: p["draft"]
                        .as_array()
                        .expect("draft")
                        .iter()
                        .map(|e| (u32_of(&e[0]), e[1].as_f64().expect("a logprob") as f32))
                        .collect(),
                })
                .collect();
            RefCase {
                name: c["name"].as_str().expect("name").to_owned(),
                tokens: c["prompt_tokens"]
                    .as_array()
                    .expect("prompt_tokens")
                    .iter()
                    .map(|v| TokenId::new(u32_of(v)))
                    .collect(),
                positions,
                hidden: read_f32(&dir.join(c["hidden_file"].as_str().expect("hidden_file"))),
                probes: read_f32(&dir.join(c["probes_file"].as_str().expect("probes_file"))),
                probe_positions: c["probe_positions"]
                    .as_array()
                    .expect("probe_positions")
                    .iter()
                    .map(u32_of)
                    .collect(),
                p1: c["p1"].as_f64().expect("p1"),
            }
        })
        .collect();
    Fixture {
        cases,
        stages,
        hidden,
        streams,
        vocabulary_digest: doc["vocabulary_digest"]
            .as_str()
            .expect("vocabulary_digest")
            .to_owned(),
    }
}

/// Whether an optional switch variable is set to `1`.
fn switch(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// The export, loaded as the server serves it with its draft head.
///
/// Two of the server's knobs change the head's arithmetic and are off by
/// default here as they are there: `SHISO_MTP_KV_FP8=1` stores the cache (the
/// head's own rows included) as FP8, as `--kv-cache-dtype fp8` does, and
/// `SHISO_MTP_PREFILL_LOWP=1` runs prompt passes with low-precision
/// activations, as `--prefill-activations lowp` does. A served configuration
/// with either on is measured by setting it here too.
fn load(backend: &Arc<dyn Backend>, dir: &Path) -> Qwen4ExpModel {
    let text = std::fs::read_to_string(dir.join("config.json")).expect("the config reads");
    let hf = HfConfig::from_json_str(&text).expect("the config parses");
    let mut engine = EngineOptions::new();
    engine.ctx = CTX;
    if switch("SHISO_MTP_KV_FP8") {
        engine.kv_elem = KvElem::Fp8E4m3;
    }
    let prefill_activations = if switch("SHISO_MTP_PREFILL_LOWP") {
        PrefillActivations::Lowp
    } else {
        PrefillActivations::A16
    };
    report(&format!(
        "cache {:?}, prompt activations {prefill_activations:?}",
        engine.kv_elem
    ));
    let entries = capture::capture(backend.as_ref());
    let paths = capture::optional_paths(backend.as_ref());
    // The server's own options for this export at `--spec-k 4`, less the
    // prompt lookup: the head's numbers are what is measured.
    let opts = SparseOptions {
        engine,
        spec: SpecDepth::new(4),
        head_drafts: 4,
        pack_halves: true,
        entries,
        co_resident_blocks: capture::co_resident_blocks(backend.as_ref(), entries),
        tensor_core: paths.tensor_core,
        tiled_attention: paths.tiled_attention,
        prompt_matmul: paths.prompt_matmul,
        route_sort: paths.route_sort,
        prefill_activations,
        chunk: ChunkPolicy {
            available: paths.chunked_recurrence,
            fused: paths.fused_recurrence,
            ..ChunkPolicy::new()
        },
        ..SparseOptions::default()
    };
    Qwen4ExpModel::load_modelopt_mixed(
        backend,
        dir,
        &hf,
        opts,
        &StreamedLoad {
            slots: Some(EXPERTS * N_LAYERS),
            reserve: 0,
            draft: true,
            counts: None,
        },
    )
    .expect("the export and its draft head load onto the device")
}

/// The first `len` f32 values of a scratch buffer.
fn read_scratch(
    backend: &Arc<dyn Backend>,
    model: &Qwen4ExpModel,
    buf: Buf,
    len: usize,
) -> Vec<f32> {
    let view = model
        .stack()
        .scratch_view(buf)
        .and_then(|v| v.slice(0, len as u64 * 4))
        .expect("the scratch buffer holds a row");
    let mut bytes = vec![0u8; len * 4];
    backend
        .copy_d2h(view, &mut bytes)
        .expect("the row reads back");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

/// Writes one f32 row into row zero of the residual.
fn write_residual(backend: &Arc<dyn Backend>, model: &Qwen4ExpModel, row: &[f32]) {
    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = model
        .stack()
        .scratch_view(Buf::Residual)
        .and_then(|v| v.slice(0, bytes.len() as u64))
        .expect("the residual holds a row");
    backend.copy_h2d(view, &bytes).expect("the row uploads");
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    u32::try_from(best).expect("a vocabulary row")
}

/// `||a - b|| / ||b||`, accumulated in double precision.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

/// Per-position agreement between this port's draft rows and the reference's.
#[derive(Default)]
struct Tally {
    rows: u32,
    top1_ref: u32,
    overlap: f64,
    max_gap: f32,
    top1_gap: f64,
    top1_gap_rows: u32,
    p1_ref_target: u32,
    p1_own_target: u32,
    own_target_rows: u32,
    ref_p1: u32,
}

impl Tally {
    fn add(&mut self, ours: &TopSlice, reference: &RefPos, own_target: Option<u32>) {
        self.rows += 1;
        let mine = ours.first().map(|e| e.token);
        let theirs = reference.draft.first().map(|e| e.0);
        self.top1_ref += u32::from(mine.is_some() && mine == theirs);
        let shared = reference
            .draft
            .iter()
            .filter(|(t, _)| ours.iter().any(|e| e.token == *t))
            .count();
        let (shared, depth) = (
            u32::try_from(shared).unwrap_or(u32::MAX),
            u32::try_from(reference.draft.len().max(1)).unwrap_or(u32::MAX),
        );
        self.overlap += f64::from(shared) / f64::from(depth);
        for (t, lp) in &reference.draft {
            if let Some(e) = ours.iter().find(|e| e.token == *t) {
                self.max_gap = self.max_gap.max((e.logit - lp).abs());
            }
        }
        if let (Some(a), Some(b)) = (ours.first(), reference.draft.first())
            && a.token == b.0
        {
            self.top1_gap += f64::from((a.logit - b.1).abs());
            self.top1_gap_rows += 1;
        }
        self.p1_ref_target += u32::from(mine == Some(reference.target));
        self.ref_p1 += u32::from(theirs == Some(reference.target));
        if let Some(target) = own_target {
            self.own_target_rows += 1;
            self.p1_own_target += u32::from(mine == Some(target));
        }
    }

    fn frac(n: u32, d: u32) -> f64 {
        f64::from(n) / f64::from(d.max(1))
    }

    fn top1(&self) -> f64 {
        Self::frac(self.top1_ref, self.rows)
    }

    fn line(&self, label: &str) -> String {
        let own = if self.own_target_rows > 0 {
            format!(
                ", p1 vs own target {:.3}",
                Self::frac(self.p1_own_target, self.own_target_rows)
            )
        } else {
            String::new()
        };
        format!(
            "{label}: {} rows; head top-1 = reference head top-1 {:.3}, top-{TOP} overlap {:.3}, \
             max logprob gap on shared ids {:.3}, mean top-1 gap {:.4}; p1 vs reference target \
             {:.3}{own} (reference head's own p1 {:.3})",
            self.rows,
            self.top1(),
            self.overlap / f64::from(self.rows.max(1)),
            self.max_gap,
            self.top1_gap / f64::from(self.top1_gap_rows.max(1)),
            Self::frac(self.p1_ref_target, self.rows),
            Self::frac(self.ref_p1, self.rows),
        )
    }
}

/// The head's pass with its vocabulary row read back, as top log-probabilities.
fn draft_row(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    session: SessionId,
    token: TokenId,
    at: u32,
    hidden: Operand,
) -> TopSlice {
    let vocab = model.info().vocab as usize;
    model
        .stack_mut()
        .draft(session, &[token], &[at], hidden, true)
        .expect("the draft pass runs")
        .expect("a picking pass returns a token");
    logprob_top_k(&read_scratch(backend, model, Buf::Logits, vocab), TOP)
}

/// The own-input run: the target a token at a time, the head from each step's
/// staged row.
fn own_input(backend: &Arc<dyn Backend>, model: &mut Qwen4ExpModel, case: &RefCase, fx: &Fixture) {
    let width = fx.hidden * fx.streams;
    let vocab = model.info().vocab as usize;
    let session = model.open_session().expect("a session opens");
    // The first token alone: its pass stages the row the head's first draft
    // reads, and the head's cache is filled position by position below.
    model
        .prefill_last(session, &case.tokens[..1])
        .expect("the first token runs");
    let mut tally = Tally::default();
    let mut target_agree = 0u32;
    let (mut h_sum, mut h_max) = (0.0f64, 0.0f64);
    for (t, pos) in case.positions.iter().enumerate() {
        let at = u32::try_from(t).expect("a position");
        // The target's own row at `t`, before the head overwrites the residual.
        let own = read_scratch(backend, model, Buf::Residual, width);
        let reference = &case.hidden[t * width..(t + 1) * width];
        let err = rel_l2(&own, reference);
        h_sum += err;
        h_max = h_max.max(err);
        let ours = draft_row(backend, model, session, pos.next, at, Operand::DraftRow);
        // The target reads token t+1 and picks for t+2.
        let row = {
            let logits = model
                .decode_batch(&[DecodeStep {
                    session,
                    token: pos.next,
                }])
                .expect("the target step runs");
            let view = logits.row(0).expect("a logits row");
            let mut bytes = vec![0u8; vocab * 4];
            backend
                .copy_d2h(view, &mut bytes)
                .expect("the row reads back");
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect::<Vec<f32>>()
        };
        let own_target = argmax(&row);
        target_agree += u32::from(own_target == pos.target);
        tally.add(&ours, pos, Some(own_target));
    }
    model.close_session(session).expect("the session closes");
    let rows = f64::from(u32::try_from(case.positions.len().max(1)).expect("a row count"));
    report(&tally.line(&format!("{} own input", case.name)));
    report(&format!(
        "{} own input: target pick = reference target pick {:.3}; target four-stream row vs \
         reference: mean relative L2 {:.4}, max {:.4}",
        case.name,
        f64::from(target_agree) / rows,
        h_sum / rows,
        h_max
    ));
}

/// The verify group sizes the speculative-step run cycles through: every row
/// count a verify at `--spec-k 4` commits.
const GROUPS: [usize; 5] = [1, 2, 3, 4, 5];

/// The speculative-step run: the target in verify-shaped passes of one to five
/// rows, the head's cache rewritten over each group by one multi-row pass from
/// the target's own rows, and the head drafting at the group's last position
/// from the staged row - the order [`shiso_engine::sparse::stack::SparseStack::speculate`]
/// runs them in, with every group accepted. Only each group's last position
/// is a pick, so this covers a sample of positions; what it adds over the
/// own-input run is that the head's cache was written by the multi-row passes a
/// served step writes it with.
fn speculative_steps(backend: &Arc<dyn Backend>, model: &mut Qwen4ExpModel, case: &RefCase) {
    let count = case.tokens.len();
    let session = model.open_session().expect("a session opens");
    model
        .prefill_last(session, &case.tokens[..1])
        .expect("the first token runs");
    let mut tally = Tally::default();
    let ours = draft_row(
        backend,
        model,
        session,
        case.tokens[1],
        0,
        Operand::DraftRow,
    );
    tally.add(&ours, &case.positions[0], None);
    let (mut start, mut group) = (1usize, 0usize);
    while start + 1 < count {
        let rows = GROUPS[group % GROUPS.len()].min(count - 1 - start);
        group += 1;
        let stack = model.stack_mut();
        stack
            .step_rows(session, &case.tokens[start..start + rows])
            .expect("the verify-shaped pass runs");
        let last = u32::try_from(rows - 1).expect("a row");
        stack
            .stage_draft_row(session, last)
            .expect("the row stages");
        if rows > 1 {
            let positions: Vec<u32> = (start..start + rows - 1)
                .map(|p| u32::try_from(p).expect("a position"))
                .collect();
            stack
                .draft(
                    session,
                    &case.tokens[start + 1..start + rows],
                    &positions,
                    Operand::Scratch(Buf::Residual),
                    false,
                )
                .expect("the catch-up pass runs");
        }
        let t = start + rows - 1;
        let at = u32::try_from(t).expect("a position");
        let ours = draft_row(
            backend,
            model,
            session,
            case.tokens[t + 1],
            at,
            Operand::DraftRow,
        );
        tally.add(&ours, &case.positions[t], None);
        start += rows;
    }
    model.close_session(session).expect("the session closes");
    report(&tally.line(&format!("{} speculative steps", case.name)));
}

/// Positions the after-prompt run samples: one prompt pass each.
const PROMPT_SAMPLE_STRIDE: usize = 24;

/// The after-prompt run: at sampled positions `t`, a fresh session reads the
/// prompt up to `t` through the prompt path - whose chunks fill the head's
/// cache with its wide catch-up passes - and the head drafts once from the
/// staged row, as the first draft of a served turn does.
fn after_prompt(backend: &Arc<dyn Backend>, model: &mut Qwen4ExpModel, case: &RefCase) {
    let n = case.positions.len();
    let mut sample: Vec<usize> = (8..n).step_by(PROMPT_SAMPLE_STRIDE).collect();
    sample.push(n - 1);
    sample.dedup();
    let mut tally = Tally::default();
    for t in sample {
        let session = model.open_session().expect("a session opens");
        model
            .prefill_last(session, &case.tokens[..=t])
            .expect("the prompt runs");
        let at = u32::try_from(t).expect("a position");
        let ours = draft_row(
            backend,
            model,
            session,
            case.tokens[t + 1],
            at,
            Operand::DraftRow,
        );
        tally.add(&ours, &case.positions[t], None);
        model.close_session(session).expect("the session closes");
    }
    report(&tally.line(&format!("{} after prompt", case.name)));
}

/// The reference-input run, with the stage comparison at the probe positions.
/// Returns the head's top-1 agreement with the reference head.
#[expect(
    clippy::too_many_lines,
    reason = "one position's three passes and their reads are one sequence: \
              the residual is rewritten before each pass and read after it"
)]
fn reference_input(
    backend: &Arc<dyn Backend>,
    model: &mut Qwen4ExpModel,
    case: &RefCase,
    fx: &Fixture,
    dump: Option<&Path>,
) -> f64 {
    let width = fx.hidden * fx.streams;
    let session = model.open_session().expect("a session opens");
    let probe_width: usize = fx.stages.iter().map(|(_, w)| w * fx.hidden).sum();
    let mut dumped = Vec::new();
    let mut tally = Tally::default();
    for (t, pos) in case.positions.iter().enumerate() {
        let at = u32::try_from(t).expect("a position");
        let input = &case.hidden[t * width..(t + 1) * width];
        let probe = case.probe_positions.iter().position(|p| *p == at);
        let mut stages: Vec<(&str, Vec<f32>)> = Vec::new();
        if probe.is_some() {
            write_residual(backend, model, input);
            model
                .stack_mut()
                .draft_probe(
                    session,
                    pos.next,
                    at,
                    Operand::Scratch(Buf::Residual),
                    DraftProbe::LayerInput,
                )
                .expect("the layer-input probe runs");
            stages.push((
                "fc_embed",
                read_scratch(backend, model, Buf::Mixed, fx.hidden),
            ));
            stages.push((
                "layer_in",
                read_scratch(backend, model, Buf::Residual, width),
            ));
            write_residual(backend, model, input);
            model
                .stack_mut()
                .draft_probe(
                    session,
                    pos.next,
                    at,
                    Operand::Scratch(Buf::Residual),
                    DraftProbe::PostAttention,
                )
                .expect("the post-attention probe runs");
            stages.push((
                "post_attn",
                read_scratch(backend, model, Buf::Residual, width),
            ));
        }
        write_residual(backend, model, input);
        let ours = draft_row(
            backend,
            model,
            session,
            pos.next,
            at,
            Operand::Scratch(Buf::Residual),
        );
        tally.add(&ours, pos, None);
        let Some(k) = probe else {
            continue;
        };
        stages.push((
            "normed_hidden",
            read_scratch(backend, model, Buf::DraftNormed, width),
        ));
        stages.push((
            "fc_hidden",
            read_scratch(backend, model, Buf::DraftHidden, width),
        ));
        stages.push((
            "layer_out",
            read_scratch(backend, model, Buf::Residual, width),
        ));
        stages.push(("mixed", read_scratch(backend, model, Buf::Mixed, fx.hidden)));
        let reference = &case.probes[k * probe_width..(k + 1) * probe_width];
        let mut offset = 0usize;
        let mut line = format!("{} t={at} stages vs reference (relative L2):", case.name);
        let mut row = Vec::with_capacity(probe_width);
        for (name, w) in &fx.stages {
            let span = w * fx.hidden;
            let want = &reference[offset..offset + span];
            offset += span;
            match stages.iter().find(|(n, _)| n == name) {
                Some((_, got)) => {
                    let _ = write!(line, " {name} {:.2e}", rel_l2(got, want));
                    row.extend_from_slice(got);
                }
                None => row.extend(std::iter::repeat_n(f32::NAN, span)),
            }
        }
        let _ = write!(
            line,
            "; top-1 ours {:?} reference {:?}",
            ours.first().map(|e| (e.token, e.logit)),
            pos.draft.first()
        );
        report(&line);
        dumped.extend(row);
    }
    model.close_session(session).expect("the session closes");
    report(&tally.line(&format!(
        "{} reference input (reference p1 recorded {:.3})",
        case.name, case.p1
    )));
    if let Some(dir) = dump {
        let path = dir.join(format!("{}.probes.f32", case.name));
        let bytes: Vec<u8> = dumped.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        report(&format!("wrote {}", path.display()));
    }
    tally.top1()
}

#[test]
#[ignore = "needs the local Qwen3.8-Flash-Next FP4-Local export and a device"]
fn the_draft_head_agrees_with_the_library_reference_head() {
    let Some(dir) = model_dir() else {
        skip("SHISO_TEST_MODEL_QWEN4EXP_NVFP4MX names no export directory");
        return;
    };
    let fixture_path = vectors_dir().join(FIXTURE);
    if !fixture_path.is_file() {
        skip(&format!(
            "{} is absent: record it with scripts/reference/qwen4exp_modelopt_mixed_ref.py mtp",
            fixture_path.display()
        ));
        return;
    }
    if !shiso_cuda::cuda_available() {
        skip("no NVIDIA device on this machine");
        return;
    }
    let fx = parse(&fixture_path);
    let vocabulary =
        Vocabulary::load(&dir.join("tokenizer.json")).expect("the export's vocabulary loads");
    assert_eq!(
        vocabulary.digest().expect("the vocabulary reads"),
        fx.vocabulary_digest,
        "this is not the tokenizer.json the reference encoded with"
    );
    let dump = std::env::var("SHISO_MTP_DUMP_DIR").ok().map(PathBuf::from);
    if let Some(d) = dump.as_ref() {
        std::fs::create_dir_all(d).expect("the dump directory exists");
    }
    let backend = open_backend(Device::Cuda).expect("the device opens");
    let mut model = load(&backend, &dir);
    let mut failures = Vec::new();
    for case in &fx.cases {
        assert_eq!(
            case.hidden.len(),
            case.positions.len() * fx.hidden * fx.streams,
            "case {}: the reference's rows and positions disagree",
            case.name
        );
        own_input(&backend, &mut model, case, &fx);
        speculative_steps(&backend, &mut model, case);
        after_prompt(&backend, &mut model, case);
        let top1 = reference_input(&backend, &mut model, case, &fx, dump.as_deref());
        if top1 < HEAD_TOP1_FLOOR {
            failures.push(format!(
                "{}: the head's top 1 agrees with the reference head's at {top1:.3} of positions \
                 on the reference's own input",
                case.name
            ));
        }
    }
    model.teardown().expect("the device releases");
    assert!(
        failures.is_empty(),
        "this port's draft head disagrees with the library reference's over the same weights \
         and the same input: a defect to locate, never a budget to widen.\n{}",
        failures.join("\n")
    );
}
