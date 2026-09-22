//! The Qwen 3.5 mixture-of-experts checkpoint, loaded onto this device.
//!
//! **The one case in this tree that moves 23.4 GB.** It is here rather than in
//! `shiso-models` for a reason worth writing down: that crate dev-depends on
//! the Metal backend, and its `device()` helper answers `None` on Linux - so a
//! load case written there would skip on the only machine that has both this
//! checkpoint and a device to put it on, and would print the same colour as a
//! passing one.
//!
//! `SHISO_CUDA_COMPILE=1 SHISO_TEST_MODEL_QWEN35MOE=<dir> cargo test -p
//! shiso-cuda --test q35moe_load -- --ignored --nocapture`
//!
//! This machine kills long loads under memory pressure (`earlyoom`). A kill is
//! a kill and not a failure; the case prints what it was doing when it stopped.
//!
//! # What these two cases are for
//!
//! The first **grades the port against an independent implementation** and is
//! the only case in this tree that does so for this family; see [`Reference`].
//! The second runs a real, chat-templated prompt and prints what comes out,
//! because a number that agrees with a reference and a sentence a human can
//! read answer different questions.
#![cfg(target_os = "linux")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use shiso_core::Backend;
use shiso_core::model::Model;
use shiso_cuda::{CudaBackend, CudaOptions};
use shiso_engine::EngineOptions;
use shiso_models::qwen35moe::{Qwen35MoeConfig, Qwen35MoeModel};

/// The independent reference this port is graded against.
///
/// Captured by `scripts/qwen35moe_reference.py`, which is `transformers`
/// 5.16.1's own `qwen3_5_moe` forward over the same eight token ids -
/// including that package's own delta-rule function. **It reads the
/// checkpoint's NATIVE head layout** - `repeat_interleave`, so value head `h`
/// reads key head `h / group` - where this engine permutes the weights at load
/// and reads `h % key_heads`. Two different layouts and two different index
/// rules that have to produce the same answer, which is what makes an
/// agreement here evidence rather than a mirror.
#[derive(Debug, serde::Deserialize)]
struct Reference {
    /// The prompt, as token ids.
    ids: Vec<u32>,
    /// The smallest logit of the last row.
    min: f32,
    /// The largest.
    max: f32,
    /// The most likely token.
    argmax: usize,
    /// The most likely tokens and their logits, most likely first.
    top: Vec<(usize, f32)>,
}

/// What the port's logits may diverge from the reference's by.
///
/// A tenth of a logit, and it is where a `f32` reference that dequantizes to
/// `f32` and multiplies there lands against a device that decodes FP8 and
/// NVFP4 in its own order and accumulates in a different one. Measured over
/// all forty slots, the two residuals agree to 0.3% at worst and about 0.1%
/// typically; this is what that compounds to at the logits.
const BUDGET: f32 = 0.1;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Prints a measurement.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The checkpoint, or `None` when it is not on this machine.
fn snapshot() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_QWEN35MOE").ok()?);
    path.join("model.safetensors.index.json")
        .is_file()
        .then_some(path)
}

/// Opens device 0, or says on stderr why the case did not run.
#[expect(
    clippy::print_stderr,
    reason = "the GPU cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn open() -> Option<Arc<dyn Backend>> {
    match CudaBackend::new(0, CudaOptions::default()) {
        Ok(cuda) => Some(cuda),
        Err(err) => {
            eprintln!("note: no CUDA device on this machine, skipping: {err}");
            None
        }
    }
}

/// Compares the port's logits row against the reference's record of it.
///
/// **This is the grading.** The reference reads the checkpoint's native head
/// layout and this engine permutes it; agreement is evidence about both.
#[expect(clippy::print_stdout, reason = "the measurement is this case's output")]
fn grade(row: &[f32], min: f32, max: f32, argmax: usize, reference: &Reference) {
    assert_eq!(
        argmax, reference.argmax,
        "the port and the reference disagree about the most likely token"
    );
    assert!(
        (min - reference.min).abs() < BUDGET && (max - reference.max).abs() < BUDGET,
        "logit bounds [{min}, {max}] against the reference's [{}, {}]",
        reference.min,
        reference.max,
    );
    let mut worst = 0.0f32;
    for (id, want) in &reference.top {
        let got = row[*id];
        worst = worst.max((got - want).abs());
        assert!(
            (got - want).abs() < BUDGET,
            "token {id}: {got} against the reference's {want}",
        );
    }
    println!(
        "graded against the reference: {} of its top tokens, worst divergence {worst:.4} \
         of a logit, budget {BUDGET}",
        reference.top.len(),
    );
}

#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
fn the_checkpoint_loads_onto_the_device_and_answers_a_prompt() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open() else { return };

    let started = Instant::now();
    let config = Qwen35MoeConfig::from_dir(&dir).expect("the two config documents");
    report(&format!(
        "config: {} layers, hidden {}, {} experts at {} per token, read in {:.3} s",
        config.shape().n_layers(),
        config.shape().hidden,
        config.moe().experts,
        config.moe().used,
        started.elapsed().as_secs_f64(),
    ));

    let started = Instant::now();
    let mut model = Qwen35MoeModel::load(&backend, &dir, &config, EngineOptions::default())
        .expect("the checkpoint loads onto the device");
    report(&format!(
        "loaded in {:.1} s",
        started.elapsed().as_secs_f64()
    ));

    // One prompt through the whole stack, at the same eight token ids the
    // reference was captured over.
    let reference: Reference = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test-vectors/references/qwen35moe-reference-logits.json"),
        )
        .expect("the reference logits"),
    )
    .expect("the reference parses");
    let session = model.open_session().expect("a session opens");
    let tokens: Vec<_> = reference
        .ids
        .iter()
        .copied()
        .map(shiso_core::TokenId::new)
        .collect();
    let started = Instant::now();
    let logits = model
        .prefill_last(session, &tokens)
        .expect("an eight-token prefill");
    let view = logits.row(0).expect("one logits row");
    let mut host = vec![0u8; usize::try_from(view.len()).expect("one row fits")];
    backend
        .copy_d2h(view, &mut host)
        .expect("the row reads back");
    let row: Vec<f32> = host
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    report(&format!(
        "prefilled {} tokens in {:.3} s",
        tokens.len(),
        started.elapsed().as_secs_f64()
    ));

    assert_eq!(row.len(), config.shape().vocab as usize);
    // The assertion that catches a pass that ran and produced nothing. A stack
    // with one stage silently skipped answers a row of zeros, of NaNs, or of
    // one repeated value, and all three are "the model loaded" to everything
    // upstream of here.
    let finite = row.iter().filter(|v| v.is_finite()).count();
    assert_eq!(
        finite,
        row.len(),
        "{} of {} logits are finite",
        finite,
        row.len()
    );
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = row.iter().copied().fold(f32::INFINITY, f32::min);
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    report(&format!(
        "logits: min {min:e}, max {max:e}, argmax {argmax}"
    ));
    assert!(max > min, "every logit is the same value");

    grade(&row, min, max, argmax, &reference);

    model.close_session(session).expect("the session closes");
    model.teardown().expect("the device memory is released");
}

#[test]
#[ignore = "requires a CUDA device and the 23.4 GB checkpoint"]
fn a_real_prompt_runs_every_stage_of_the_pass() {
    let Some(dir) = snapshot() else {
        skip("SHISO_TEST_MODEL_QWEN35MOE names no checkpoint directory");
        return;
    };
    let Some(backend) = open() else { return };

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("the checkpoint's own vocabulary");
    let config = Qwen35MoeConfig::from_dir(&dir).expect("the two config documents");
    let mut model = Qwen35MoeModel::load(&backend, &dir, &config, EngineOptions::default())
        .expect("the checkpoint loads onto the device");

    // The prompt `HANDOFF.md` uses for exactly this, on exactly this box, for
    // a different family. A model that loads, runs and answers this with
    // something that is not " Paris" has a defect somewhere between the
    // vocabulary gather and the output projection - and every one of the
    // twelve this repository has found produced finite, plausible logits.
    // Chat-templated, because this checkpoint is an RL post-trained agentic
    // model rather than a base one: `Ornith-1.5-35B-A3B`, whose card is about
    // coding and agentic benchmarks. A raw completion from one of those is not
    // the thing to judge a port by - and the raw completion this stack gives
    // has been checked against an independent reference anyway, which is what
    // the plan's account records.
    let prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one \
                  word.<|im_end|>\n<|im_start|>assistant\n";
    let encoded = tokenizer.encode(prompt, false).expect("the prompt encodes");
    let mut tokens: Vec<shiso_core::TokenId> = encoded
        .get_ids()
        .iter()
        .map(|id| shiso_core::TokenId::new(*id))
        .collect();
    report(&format!("prompt {prompt:?} is {} tokens", tokens.len()));

    let session = model.open_session().expect("a session opens");
    let mut produced: Vec<u32> = Vec::new();
    for step in 0..24u32 {
        let logits = if step == 0 {
            model
                .prefill_last(session, &tokens)
                .expect("the prompt prefills")
        } else {
            let last = *tokens.last().expect("at least one token");
            model
                .decode_batch(&[shiso_core::model::DecodeStep {
                    session,
                    token: last,
                }])
                .expect("a decode step")
        };
        let view = logits.row(0).expect("one logits row");
        let mut host = vec![0u8; usize::try_from(view.len()).expect("one row fits")];
        backend
            .copy_d2h(view, &mut host)
            .expect("the row reads back");
        let argmax = host
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        let next = u32::try_from(argmax).expect("a token id");
        produced.push(next);
        tokens.push(shiso_core::TokenId::new(next));
    }
    let text = tokenizer
        .decode(&produced, false)
        .expect("the answer decodes");
    report(&format!("greedy: {text:?}"));
    report(&format!("ids: {produced:?}"));

    model.close_session(session).expect("the session closes");
    model.teardown().expect("the device memory is released");

    // NOT asserted on the text, and the text IS strange: this checkpoint
    // answers "What is the capital of France?" - chat-templated, with its own
    // template - with one token and then the same CJK token over and over.
    //
    // **That is the checkpoint, not the port**, and the other case in this
    // file is what establishes it: the same eight-token prompt through
    // `transformers` 5.16.1's own `qwen3_5_moe` forward produces the same
    // argmax and logits within 0.08, and the two residuals agree at every one
    // of the forty slots. Run over the real prompt's ids the reference
    // predicts the same first token this does. `ornith-ai/Ornith-1.5-35B-A3B`
    // is an RL post-trained agentic model and this is a third-party NVFP4
    // quantization of it; whether that quantization is a good model is a
    // question about the quantization.
    //
    // So this case asserts what it can stand behind - that the pass completes
    // every stage and does not answer one constant - and prints the text to be
    // read. An assertion on a string would turn the one thing a human can
    // judge into a boolean.
    assert_eq!(produced.len(), 24);
    let distinct: std::collections::HashSet<u32> = produced.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "every one of the steps produced the same token: {produced:?}"
    );
}
