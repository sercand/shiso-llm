//! Gemma 4 against upstream llama.cpp's own frontier row.
//!
//! This is the only file in this tree that can say the port is *wrong*. Every
//! other parity case on this machine compares shiso against a recording shiso
//! made, which locks regression and proves nothing about correctness. Here the
//! recording was made by a different implementation of the same checkpoint -
//! `llama-server` from upstream llama.cpp, and `mlx-lm` for the mlx lane - so a
//! disagreement is a defect in one of the two stacks rather than a changed
//! answer.
//!
//! The text lanes run on whichever device the platform has: Metal on macOS and
//! CUDA elsewhere, chosen by [`DEVICE`]. The fixtures do not care, because they
//! were captured from the two reference implementations rather than from a
//! device, and the budgets they are graded at are measurements of the gap
//! between implementations rather than of a backend.
//!
//! Run it with the server stopped - a device runs one workload at a time:
//!
//! ```text
//! SHISO_CUDA_COMPILE=1 cargo test -p shiso-bench --release \
//!   --test parity_gemma4 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! # What is being compared, and in which units
//!
//! `llama-server` exposes no raw logit. With `post_sampling_probs` false it
//! softmaxes the whole logits row and returns log-probabilities, so the fixture
//! stores log-probabilities and says so in a `# values logprob` header. The
//! replay converts its own row the same way, over the full vocabulary rather
//! than over the recorded sixty-four: see [`logprob_top_k`] for why the constant
//! has to come from the whole row.
//!
//! # What makes the two sides the same question
//!
//! Four things, all of them asserted rather than assumed. The prompt files are
//! digested and checked against the header, so an edited prompt is named instead
//! of measured. The token count is checked against the case's frontier. The ids
//! themselves are digested and checked, because two tokenizers can agree about
//! how many tokens a prompt is and disagree about which.
//!
//! And the sequence begins with the model's own opening token. `llama-server`
//! tokenizes the prompt text its own way, which for this family prepends
//! `bos_token_id`; the sidecar `tokenizer.json`'s post-processor is
//! `single: [$A]` with no special tokens, so the `tokenizers` crate adds nothing
//! at either `special` setting. The fixture records the id it was given and this
//! replay prepends it, cross-checking it against `text_config.bos_token_id` in
//! the shipped `config.json`. That is what makes the golden grade the regime the
//! model is actually run in; a fixture with no `leading_token` header is refused
//! rather than replayed without one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_bench::compare::{
    FLOOR, GEMMA4_VISION_TOWER_GGUF_BUDGET, GEMMA4_VISION_TOWER_MLX_BUDGET, MLX_REFERENCE,
    REFERENCE, TopSlice, compare, decode_row,
};
use shiso_bench::replay::{Device, Harness, Vocabulary};
use shiso_bench::run::{
    REPLAY_CHUNK, case_tokens_with_leading, logprob_top_k, prompts_digest, tokens_digest,
};
use shiso_bench::vecfile::{self, Body, LocalCase, VecFile};
use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::Model;
use shiso_engine::EngineOptions;
use shiso_metal::metal_available;
use shiso_models::gemma4::Gemma4Tower;
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};
use shiso_vision::VisionTower;

/// The committed reference fixture.
const GOLDEN: &str = "references/gemma4-reference-golden.vec";

/// The committed greedy trajectory.
const GREEDY_LOCK: &str = "references/gemma4-greedy-64.json";

/// The committed vision tower reference fixture.
const VISION_TOWER_GOLDEN: &str = "vision/gemma4-vision-tower.vec";
/// The committed MLX reference fixture.
const MLX_GOLDEN: &str = "references/gemma4-mlx-golden.vec";

/// The committed MLX greedy trajectory.
const MLX_GREEDY_LOCK: &str = "references/gemma4-mlx-greedy-64.json";

/// How many leading greedy ids have to agree.
///
/// Not sixty-four: a greedy trajectory over a prompt with no turn markup passes
/// through near-ties, and two implementations that agree about a row to within
/// a hundredth of a nat can still resolve one of those the other way and diverge
/// for good afterwards. Eight is the length over which a divergence means the
/// forward pass disagrees rather than a tie broke differently. The rest is
/// reported, and the index of the first difference with it, because *where* two
/// trajectories part is the number worth reading.
const GREEDY_MIN_AGREEMENT: usize = 8;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Prints a measurement, which is recorded as well as compared.
#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// Where the committed fixtures live.
fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The geometry document this family loads against.
fn config_path() -> PathBuf {
    vectors_dir().join("loader/gemma4-config.json")
}

/// The container, or `None` when this box has none.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.is_file().then_some(path)
}

/// The sidecar vocabulary, or `None` when it is absent.
///
/// Absent and unreadable are different answers. A box without the sidecar cannot
/// run these cases and skips, which is honest. A box that *has* the file and
/// cannot load it has a broken checkpoint directory, and skipping there would
/// report "nothing to run" for what is really "this vocabulary is corrupt" - so
/// that fails loudly, naming the path and what the loader said.
fn vocabulary() -> Option<Vocabulary> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_SIDECAR").ok()?);
    let path = dir.join("tokenizer.json");
    if !path.is_file() {
        return None;
    }
    match Vocabulary::load(&path) {
        Ok(vocabulary) => Some(vocabulary),
        Err(why) => panic!(
            "the sidecar vocabulary at {} is present and will not load: {why}. That is a broken \
             checkpoint directory, not a reason to skip.",
            path.display()
        ),
    }
}

/// The committed fixture, parsed.
fn golden() -> VecFile {
    vecfile::read(&vectors_dir().join(GOLDEN)).expect("the committed reference fixture parses")
}

/// Its cases.
fn golden_cases(file: &VecFile) -> Vec<LocalCase> {
    match &file.body {
        Body::Local(cases) => cases.clone(),
        Body::Official(_) | Body::Tower(_) => panic!("the reference fixture is a token-id fixture"),
    }
}

/// The compute device this platform's text lane runs on.
///
/// A constant rather than a flag. A box has one of these two devices and never
/// both, so a run-time selector would only be a way to ask for the one that is
/// not there; the fixtures, the budgets and the greedy locks are the same on
/// either side because they were captured from llama.cpp and `mlx-lm` rather
/// than from a device.
#[cfg(target_os = "macos")]
const DEVICE: Device = Device::Metal;
/// The compute device this platform's text lane runs on.
#[cfg(not(target_os = "macos"))]
const DEVICE: Device = Device::Cuda;

/// Whether [`DEVICE`] is present on this box.
fn device_available() -> bool {
    match DEVICE {
        Device::Metal => metal_available(),
        Device::Cuda => shiso_cuda::cuda_available(),
    }
}

/// Opens a harness on [`DEVICE`] at `ctx`, or reports why it could not.
fn device_harness(ctx: u32) -> Option<Harness> {
    let model = model_path()?;
    if !device_available() {
        return None;
    }
    match Harness::open(DEVICE, &model, &config_path(), ctx) {
        Ok(harness) => Some(harness),
        Err(why) => {
            skip(&format!("the harness would not open: {why}"));
            None
        }
    }
}

/// The frontier row for `tokens` with the per-layer table held the given way.
///
/// The replay harness holds that table the way an unconfigured server does, and
/// there is no knob on it, on purpose. This loads the family directly instead,
/// because the claim below is precisely about the knob: the streamed table reads
/// a row off the disk per token while the resident one indexes a two-gigabyte
/// device tensor, and two paths to the same numbers have to produce the same
/// bits.
///
/// The prompt is fed exactly the way [`Harness::frontier_row`] feeds it - every
/// token but the last in chunks of [`REPLAY_CHUNK`], then the last one alone -
/// because the chunk decomposition is itself a route through the kernels and a
/// row produced by a different decomposition differs in its last bits. Comparing
/// against one of those would measure the chunking rather than the table.
fn frontier_row_with(ple: PleOptions, tokens: &[TokenId], ctx: u32) -> Vec<f32> {
    let path = model_path().expect("the caller checked the container is present");
    let text = std::fs::read_to_string(config_path()).expect("the geometry document reads");
    let document = HfConfig::from_json_str(&text).expect("the geometry document parses");
    let family = Gemma4Config::from_hf(&document, &path).expect("the e4b geometry");
    let backend: Arc<dyn Backend> =
        shiso_bench::replay::open_backend(DEVICE).expect("the device this lane runs on");
    let mut opts = EngineOptions::new();
    opts.ctx = ctx;
    let mut model =
        Gemma4Model::load(&backend, &path, &family, opts, ple).expect("the container loads");

    let row_bytes = usize::try_from(model.info().logits_row_bytes()).unwrap_or(0);
    let mut bytes = vec![0u8; row_bytes];
    let session = model.open_session().expect("a session opens");
    let split = tokens.len().saturating_sub(1);
    let (head, tail) = tokens.split_at(split);
    for piece in head.chunks(REPLAY_CHUNK) {
        model.prefill_chunk(session, piece).expect("a chunk runs");
    }
    let last = model.prefill_last(session, tail).expect("the prompt runs");
    backend
        .copy_d2h(last.row(0).expect("the frontier row"), &mut bytes)
        .expect("the frontier row reads back");
    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
    decode_row(&bytes)
}

/// The header value a case's token digest is recorded under.
fn tokens_digest_key(case: &LocalCase) -> String {
    format!("tokens_digest_{}", case.id)
}

/// The id every sequence in the fixture begins with.
///
/// Recorded by the capture and required here: the reference prepends the model's
/// own opening token and the sidecar `tokenizer.json` never does, so a replay
/// that did not put it back would evaluate a prefix the reference never saw. A
/// fixture without the header is refused rather than replayed BOS-less, because
/// that silently grades a regime the model is not run in.
fn leading_token(file: &VecFile) -> TokenId {
    let recorded = file
        .headers
        .get("leading_token")
        .expect(
            "this fixture declares no leading_token. A capture from llama.cpp always has one \
             (the model's bos_token_id); replaying without it would evaluate a sequence the \
             reference never saw. Recapture with scripts/capture_gemma4_golden.sh.",
        )
        .parse::<u32>()
        .expect("the leading_token header is a token id");

    // Cross-checked against the model's own document, so the header cannot drift
    // into naming some other id and still be believed.
    let text = std::fs::read_to_string(config_path()).expect("the geometry document reads");
    let document = HfConfig::from_json_str(&text).expect("the geometry document parses");
    let declared = document
        .section("text_config")
        .expect("the document carries text_config")
        .get("bos_token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .expect("text_config declares a bos_token_id");
    assert_eq!(
        recorded, declared,
        "the fixture says sequences begin with {recorded} and the model's config.json declares \
         bos_token_id {declared}. One of them is describing a different checkpoint."
    );
    TokenId::new(recorded)
}

/// Checks that the fixture and this machine agree about what the case *is*,
/// then returns the tokens it means.
fn agreed_tokens(file: &VecFile, vocabulary: &Vocabulary, case: &LocalCase) -> Vec<TokenId> {
    let tokens =
        case_tokens_with_leading(vocabulary, &vectors_dir(), case, Some(leading_token(file)))
            .expect("the prompt encodes");
    assert_eq!(
        u32::try_from(tokens.len()).unwrap_or(u32::MAX),
        case.frontier,
        "case {}: this vocabulary makes the prompt {} tokens and the reference evaluated {}. The \
         two sides are not answering the same question, so any comparison of their rows would be \
         measuring the tokenizer.",
        case.id,
        tokens.len(),
        case.frontier
    );
    let key = tokens_digest_key(case);
    let recorded = file.headers.get(&key).unwrap_or_else(|| {
        panic!(
            "case {}: the fixture carries no `{key}` header, so nothing here can tell that the \
             two sides tokenized this prompt into the same ids - only that they produced the \
             same count, which two different tokenizations can also do. A capture always writes \
             it; recapture with scripts/capture_gemma4_golden.sh.",
            case.id
        )
    });
    assert_eq!(
        &tokens_digest(&tokens),
        recorded,
        "case {}: the two vocabularies agree about how many tokens this prompt is and disagree \
         about which. Decode both sides and find the first differing position before looking at \
         a single kernel.",
        case.id
    );
    tokens
}

/// Grades one produced slice against the fixture at both budgets, reporting each.
fn grade(case: &LocalCase, produced: &TopSlice, failures: &mut Vec<String>) {
    for budget in [&FLOOR, &REFERENCE] {
        let verdict = compare(&case.id, &case.tops, produced, budget);
        report(&verdict.report_line());
        if !verdict.passed() {
            failures.push(format!("{}: {:?}", verdict.report_line(), verdict.failures));
        }
    }
}

// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs the local Gemma 4 container, the sidecar vocabulary and a device"]
fn two_identical_runs_produce_bit_identical_logits() {
    let (Some(vocabulary), Some(mut harness)) = (vocabulary(), device_harness(1024)) else {
        skip("no local Gemma 4 container, vocabulary or device");
        return;
    };
    let file = golden();
    let cases = golden_cases(&file);
    let case = cases
        .iter()
        .find(|c| c.id == "short_code_completion")
        .expect("the fixture carries this case");
    let tokens = agreed_tokens(&file, &vocabulary, case);

    let first = harness
        .frontier_row(&tokens, REPLAY_CHUNK)
        .expect("the first run");
    let second = harness
        .frontier_row(&tokens, REPLAY_CHUNK)
        .expect("the second run");
    harness.close().expect("the device releases");

    assert_eq!(
        first.len(),
        second.len(),
        "two runs of one prompt produce rows of one width"
    );
    let differing = first
        .iter()
        .zip(&second)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} logits moved between two identical runs. The reference comparison \
         below assumes a produced row is a fact rather than a sample; if this fails, it is not.",
        first.len()
    );
    report(&format!(
        "gemma4 determinism: {} logits, bit-identical across two runs",
        first.len()
    ));
}

#[test]
#[ignore = "needs the local Gemma 4 container, the sidecar vocabulary and a device"]
fn the_independent_reference_row_replays_at_the_reference_budget() {
    let file = golden();
    assert_eq!(
        file.headers.get("values").map(String::as_str),
        Some("logprob"),
        "this fixture does not declare log-probabilities, so converting the produced row to one \
         would compare two different quantities"
    );
    assert_eq!(
        file.headers.get("captured_from").map(String::as_str),
        Some("reference"),
        "a fixture this port captured of itself cannot be graded at the reference budget"
    );
    let cases = golden_cases(&file);
    let refs: Vec<&LocalCase> = cases.iter().collect();
    let digest = prompts_digest(&vectors_dir(), &refs).expect("the prompt files read");
    assert_eq!(
        file.headers.get("prompts_digest").map(String::as_str),
        Some(digest.as_str()),
        "a prompt file changed since the capture, so this would compare different tokens"
    );

    let Some(vocabulary) = vocabulary() else {
        skip("no sidecar vocabulary");
        return;
    };
    assert_eq!(
        Some(&vocabulary.digest().expect("the vocabulary reads")),
        file.headers.get("vocabulary_digest"),
        "this is not the tokenizer.json the capture encoded with - or the fixture carries no \
         `vocabulary_digest` header at all, in which case it cannot say which one it was. A \
         capture always writes it; recapture with scripts/capture_gemma4_golden.sh."
    );
    let ctx = cases.iter().map(|c| c.ctx).max().unwrap_or(8192);
    let Some(mut harness) = device_harness(ctx) else {
        skip("no local Gemma 4 container or device");
        return;
    };

    let mut failures = Vec::new();
    for case in &cases {
        let tokens = agreed_tokens(&file, &vocabulary, case);
        let row = harness
            .frontier_row(&tokens, REPLAY_CHUNK)
            .expect("the case runs against the container");
        let produced = logprob_top_k(&row, usize::try_from(case.top_count).unwrap_or(0));
        grade(case, &produced, &mut failures);
    }
    harness.close().expect("the device releases");

    assert!(
        failures.is_empty(),
        "this port disagrees with upstream llama.cpp about the same checkpoint on the same \
         device. That is a defect to locate, never a budget to widen: name the case and the \
         diverging rank, dump that logits row, and walk the operators until the first one that \
         moved names its kernel.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Gemma 4 container, the sidecar vocabulary and a device"]
fn the_resident_per_layer_table_answers_exactly_what_the_streamed_one_answers() {
    let (Some(vocabulary), Some(mut harness)) = (vocabulary(), device_harness(1024)) else {
        skip("no local Gemma 4 container, vocabulary or device");
        return;
    };
    let file = golden();
    let cases = golden_cases(&file);
    let case = cases
        .iter()
        .find(|c| c.id == "short_code_completion")
        .expect("the fixture carries this case");
    let tokens = agreed_tokens(&file, &vocabulary, case);

    let streamed = harness
        .frontier_row(&tokens, REPLAY_CHUNK)
        .expect("the streamed run");
    harness.close().expect("the device releases");
    let resident = frontier_row_with(
        PleOptions {
            resident: true,
            ..PleOptions::default()
        },
        &tokens,
        1024,
    );

    assert_eq!(
        streamed.len(),
        resident.len(),
        "the two holdings of one table produce rows of one width"
    );
    let differing = streamed
        .iter()
        .zip(&resident)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} logits differ between the streamed and the resident per-layer table. \
         Two paths to the same numbers dequantize the same rows; a nearly-identical row means one \
         of them reads the block format differently, which no token sequence is guaranteed to \
         show.",
        streamed.len()
    );
    report(&format!(
        "gemma4 per-layer table: {} logits, bit-identical streamed against resident",
        streamed.len()
    ));

    // Free, and worth taking: the resident holding is graded against the
    // independent reference too, so the claim is not merely that the two
    // holdings agree with each other.
    let produced = logprob_top_k(&resident, usize::try_from(case.top_count).unwrap_or(0));
    let mut failures = Vec::new();
    grade(case, &produced, &mut failures);
    assert!(
        failures.is_empty(),
        "the resident holding disagrees with upstream llama.cpp:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Gemma 4 container, the sidecar vocabulary and a device"]
fn the_greedy_trajectory_follows_the_reference_one() {
    let (Some(vocabulary), Some(mut harness)) = (vocabulary(), device_harness(1024)) else {
        skip("no local Gemma 4 container, vocabulary or device");
        return;
    };
    let recorded = std::fs::read_to_string(vectors_dir().join(GREEDY_LOCK))
        .expect("the recorded greedy trajectory is committed");
    let lock: serde_json::Value =
        serde_json::from_str(&recorded).expect("the recorded trajectory parses");
    let prompt = lock["prompt"]
        .as_str()
        .expect("the record names its prompt");
    let want: Vec<u32> = lock["tokens"]
        .as_array()
        .expect("the record carries a token array")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("a token is a number")).expect("a token is u32"))
        .collect();
    // The same leading token the cases carry, and recorded here too so this file
    // is self-contained; the fixture's copy is what the config is checked
    // against, and the two must not disagree.
    let leading = u32::try_from(
        lock["leading_token"]
            .as_u64()
            .expect("the record names the id the reference prepended"),
    )
    .expect("a token is u32");
    assert_eq!(
        leading,
        leading_token(&golden()).get(),
        "the greedy lock and the reference golden were captured with different opening tokens"
    );

    let mut tokens = vec![TokenId::new(leading)];
    tokens.extend(
        vocabulary
            .encode(prompt, false)
            .expect("the prompt encodes"),
    );
    let produced = harness
        .greedy_after(&tokens, REPLAY_CHUNK, want.len())
        .expect("the continuation runs");
    harness.close().expect("the device releases");
    let ids: Vec<u32> = produced.iter().map(|t| t.get()).collect();

    let agreed = ids.iter().zip(&want).take_while(|(a, b)| a == b).count();
    report(&format!(
        "gemma4 greedy lock: {agreed}/{} ids agree with llama.cpp{}",
        want.len(),
        if agreed == want.len() {
            String::new()
        } else {
            format!(
                "; first difference at {agreed}, shiso {:?} against reference {:?}",
                ids.get(agreed),
                want.get(agreed)
            )
        }
    ));
    report(&format!(
        "gemma4 greedy lock text: {:?}",
        vocabulary.decode(&produced).unwrap_or_default()
    ));
    assert!(
        agreed >= GREEDY_MIN_AGREEMENT,
        "the two stacks part after {agreed} greedy ids, inside the {GREEDY_MIN_AGREEMENT} a \
         resolved near-tie cannot explain. shiso produced {ids:?}; llama.cpp produced {want:?}. \
         Compare the frontier rows at step {agreed} before looking anywhere else."
    );
}

/// The captured vision reference parses, and says which picture it looked at.
///
/// # This does not replay, and cannot as captured
///
/// It reads the fixture, checks it carries a full sixty-four-deep row, and
/// prints the top of it. It never opens the tower, never encodes the picture and
/// never produces a row of this port's own to compare - so it is not evidence
/// that this port's vision path is right, and its earlier name said it was.
///
/// The obstacle is in the fixture rather than in the work of writing a replay.
/// `scripts/capture_gemma4_vision_golden.sh` asks `llama-server`'s chat endpoint
/// for the row, so the prompt that produced it is the one *llama.cpp* rendered:
/// its template, and its own count of placeholders for a 224x224 picture, which
/// is 107 tokens in all. This port renders the same request as 274. Two
/// different prompts have two different frontier rows whatever either stack does
/// with them, so grading one against the other would measure the two
/// tokenizations and report it as a kernel.
///
/// Closing this needs a decision rather than a patch: either both sides are made
/// to render the same placeholder run and the fixture is recaptured, or the
/// capture drops to a completions request whose prompt both sides are handed
/// verbatim. Until one of those happens, the Gemma 4 vision path's only
/// correctness evidence is
/// `shiso_models::gemma4::vision::encoder::tests` - the graph against a host
/// reference at synthetic widths, and the two containers' towers against each
/// other - and neither of those is an independent implementation.
/// The Gemma 4 vision tower output rows replay against mlx-vlm's reference.
///
/// Grades 4096 sampled output rows from the 256x2560 projected block, on both
/// the MLX container (at [`GEMMA4_VISION_TOWER_MLX_BUDGET`]) and the mmproj
/// GGUF (at [`GEMMA4_VISION_TOWER_GGUF_BUDGET`]).
#[test]
#[ignore = "needs the local Gemma 4 container, mmproj and a device"]
#[expect(
    clippy::too_many_lines,
    reason = "replays against both MLX and GGUF containers"
)]
fn the_independent_vision_tower_reference_replays() {
    // Through [`DEVICE`], like every other case in this file. It named
    // `MetalBackend` outright until Phase 8B, on the premise that "the tower's
    // kernels exist only in the Metal set" - true when it was written, and false
    // the moment `kernels/cuda/vision/vit.cu` gave the `vision` module the nine
    // entries the encoder dispatches. On a CUDA box the old form skipped, which
    // reads exactly like a pass.
    if !device_available() {
        skip("no device for this platform's lane");
        return;
    }
    let backend: Arc<dyn Backend> = match shiso_bench::replay::open_backend(DEVICE) {
        Ok(b) => b,
        Err(err) => {
            skip(&format!("opening the {} backend: {err}", DEVICE.as_str()));
            return;
        }
    };
    let fixture_path = vectors_dir().join(VISION_TOWER_GOLDEN);
    if !fixture_path.is_file() {
        skip("no gemma4-vision-tower.vec fixture");
        return;
    }
    let file = vecfile::read(&fixture_path).expect("the vision tower fixture parses");
    let Body::Tower(cases) = &file.body else {
        panic!("expected Body::Tower in {fixture_path:?}");
    };
    let case = cases
        .iter()
        .find(|c| c.id == "gemma4_vision_quadrants")
        .expect("fixture carries gemma4_vision_quadrants");

    let mlx = mlx_dir();
    let mmproj = mmproj_path();
    if mlx.is_none() && mmproj.is_none() {
        skip("neither Gemma 4 MLX directory nor mmproj GGUF found");
        return;
    }

    let image_path = fixture_path
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
        .join(&case.image);
    let image_bytes = std::fs::read(&image_path)
        .unwrap_or_else(|err| panic!("reading image {}: {err}", image_path.display()));

    if let Some(mlx_path) = mlx {
        let tower = Gemma4Tower::open(&backend, &mlx_path).expect("open Gemma 4 MLX tower");
        let item = shiso_vision::preprocess_image(
            &image_bytes,
            tower.preprocess(),
            shiso_vision::ImageLimits::default(),
        )
        .expect("preprocess image");
        let stages = tower.encode_stages(&backend, &item).expect("encode stages");

        let len = usize::try_from(stages.projected.len()).expect("len fits in usize");
        let mut bytes = vec![0u8; len];
        backend
            .copy_d2h(stages.projected.view(), &mut bytes)
            .expect("copy projected");
        let candidate_floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut max_diff = 0.0f32;
        for sample in &case.samples {
            let diff = (sample.value - candidate_floats[sample.pos]).abs();
            max_diff = max_diff.max(diff);
        }
        report(&format!(
            "gemma4 vision tower (mlx): {} samples, max |Δ| = {max_diff:.6} (budget {})",
            case.samples.len(),
            GEMMA4_VISION_TOWER_MLX_BUDGET
        ));
        assert!(
            max_diff <= GEMMA4_VISION_TOWER_MLX_BUDGET,
            "mlx vision tower exceeded budget: {max_diff} > {GEMMA4_VISION_TOWER_MLX_BUDGET}"
        );
    }

    if let Some(mmproj_path) = mmproj {
        let tower = Gemma4Tower::open(&backend, &mmproj_path).expect("open Gemma 4 mmproj tower");
        let item = shiso_vision::preprocess_image(
            &image_bytes,
            tower.preprocess(),
            shiso_vision::ImageLimits::default(),
        )
        .expect("preprocess image");
        let stages = tower.encode_stages(&backend, &item).expect("encode stages");

        let len = usize::try_from(stages.projected.len()).expect("len fits in usize");
        let mut bytes = vec![0u8; len];
        backend
            .copy_d2h(stages.projected.view(), &mut bytes)
            .expect("copy projected");
        let candidate_floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let mut max_diff = 0.0f32;
        for sample in &case.samples {
            let diff = (sample.value - candidate_floats[sample.pos]).abs();
            max_diff = max_diff.max(diff);
        }
        report(&format!(
            "gemma4 vision tower (gguf): {} samples, max |Δ| = {max_diff:.6} (budget {})",
            case.samples.len(),
            GEMMA4_VISION_TOWER_GGUF_BUDGET
        ));
        assert!(
            max_diff <= GEMMA4_VISION_TOWER_GGUF_BUDGET,
            "gguf vision tower exceeded budget: {max_diff} > {GEMMA4_VISION_TOWER_GGUF_BUDGET}"
        );
    }
}

fn mmproj_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MMPROJ").ok()?);
    path.is_file().then_some(path)
}
fn mlx_dir() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX").ok()?);
    path.is_dir().then_some(path)
}

fn mlx_golden() -> VecFile {
    vecfile::read(&vectors_dir().join(MLX_GOLDEN))
        .expect("the committed MLX reference fixture parses")
}

/// Opens a harness on [`DEVICE`] over the mlx directory, or says why it could not.
fn device_harness_mlx(ctx: u32) -> Option<Harness> {
    let dir = mlx_dir()?;
    if !device_available() {
        return None;
    }
    let config = dir.join("config.json");
    match Harness::open(DEVICE, &dir, &config, ctx) {
        Ok(harness) => Some(harness),
        Err(why) => {
            skip(&format!("the MLX harness would not open: {why}"));
            None
        }
    }
}

/// The mlx container against `mlx-lm`'s own frontier row.
///
/// Graded at [`MLX_REFERENCE`] rather than [`REFERENCE`], and the difference is
/// the point rather than a concession: `mlx-lm` carries its activations in
/// `bfloat16` and this port carries them in `f32`, so the two rows differ by
/// more than reduction order can explain and no amount of work on these kernels
/// would close it. The same container graded against llama.cpp - which computes
/// the way this port does - is held to [`REFERENCE`] by
/// [`the_independent_reference_row_replays_at_the_reference_budget`] and meets
/// it. Both verdicts are printed on every run, so the day the gap closes is
/// visible without editing anything.
#[test]
#[ignore = "needs the local Gemma 4 MLX directory, vocabulary and a device"]
fn the_independent_mlx_reference_row_replays_at_the_cross_precision_budget() {
    let file = mlx_golden();
    assert_eq!(
        file.headers.get("values").map(String::as_str),
        Some("logprob"),
        "this fixture does not declare log-probabilities"
    );
    let cases = golden_cases(&file);
    let refs: Vec<&LocalCase> = cases.iter().collect();
    let digest = prompts_digest(&vectors_dir(), &refs).expect("the prompt files read");
    assert_eq!(
        file.headers.get("prompts_digest").map(String::as_str),
        Some(digest.as_str()),
        "a prompt file changed since the capture"
    );

    let Some(vocabulary) = vocabulary() else {
        skip("no sidecar vocabulary");
        return;
    };
    let ctx = cases.iter().map(|c| c.ctx).max().unwrap_or(8192);
    let Some(mut harness) = device_harness_mlx(ctx) else {
        skip("no local Gemma 4 MLX container or device");
        return;
    };

    let mut failures = Vec::new();
    for case in &cases {
        let tokens = agreed_tokens(&file, &vocabulary, case);
        let row = harness
            .frontier_row(&tokens, REPLAY_CHUNK)
            .expect("the case runs against the container");
        let produced = logprob_top_k(&row, usize::try_from(case.top_count).unwrap_or(0));
        for budget in [&FLOOR, &MLX_REFERENCE] {
            let verdict = compare(&case.id, &case.tops, &produced, budget);
            report(&verdict.report_line());
            if !verdict.passed() {
                failures.push(format!("{}: {:?}", verdict.report_line(), verdict.failures));
            }
        }
        // Reported and not graded: this is the budget the same container meets
        // against llama.cpp, and watching how far the bf16 reference sits from
        // it is worth a line even though failing on it would be failing on the
        // reference's number format.
        report(&compare(&case.id, &case.tops, &produced, &REFERENCE).report_line());
    }
    harness.close().expect("the device releases");

    assert!(
        failures.is_empty(),
        "this port disagrees with mlx-lm about the same checkpoint on the same device, by more \n\
         than the two stacks' float widths account for. That is a defect to locate, never a \n\
         budget to widen: name the case and the diverging rank, dump that logits row, and walk \n\
         the operators until the first one that moved names its kernel.\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs the local Gemma 4 MLX directory, vocabulary and a device"]
fn the_mlx_greedy_trajectory_follows_the_reference_one() {
    let (Some(vocabulary), Some(mut harness)) = (vocabulary(), device_harness_mlx(1024)) else {
        skip("no local Gemma 4 MLX container, vocabulary or device");
        return;
    };
    let recorded = std::fs::read_to_string(vectors_dir().join(MLX_GREEDY_LOCK))
        .expect("the recorded MLX greedy trajectory is committed");
    let lock: serde_json::Value =
        serde_json::from_str(&recorded).expect("the recorded trajectory parses");
    let prompt = lock["prompt"]
        .as_str()
        .expect("the record names its prompt");
    let want: Vec<u32> = lock["tokens"]
        .as_array()
        .expect("the record carries a token array")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("a token is a number")).expect("a token is u32"))
        .collect();
    let leading = u32::try_from(
        lock["leading_token"]
            .as_u64()
            .expect("the record names the id the reference prepended"),
    )
    .expect("a token is u32");

    let mut tokens = vec![TokenId::new(leading)];
    tokens.extend(
        vocabulary
            .encode(prompt, false)
            .expect("the prompt encodes"),
    );
    let produced = harness
        .greedy_after(&tokens, REPLAY_CHUNK, want.len())
        .expect("the continuation runs");
    harness.close().expect("the device releases");
    let ids: Vec<u32> = produced.iter().map(|t| t.get()).collect();

    let agreed = ids.iter().zip(&want).take_while(|(a, b)| a == b).count();
    report(&format!(
        "gemma4 mlx greedy lock: {agreed}/{} ids agree with mlx-lm{}",
        want.len(),
        if agreed == want.len() {
            String::new()
        } else {
            format!(
                "; first difference at {agreed}, shiso {:?} against reference {:?}",
                ids.get(agreed),
                want.get(agreed)
            )
        }
    ));
    report(&format!(
        "gemma4 mlx greedy lock text: {:?}",
        vocabulary.decode(&produced).unwrap_or_default()
    ));
    assert!(
        agreed >= GREEDY_MIN_AGREEMENT,
        "the two stacks part after {agreed} greedy ids: shiso {ids:?}; mlx-lm {want:?}"
    );
}
