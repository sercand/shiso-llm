//! What the E4B checkpoint answers, and the claim that where its per-layer rows
//! were held did not change the answer.
//!
//! Two cases over one prompt. The first renders the family's turn markup around
//! a question with one uncontroversial answer, greedily decodes sixteen tokens
//! and reads them back as text: a model whose weights all reached the device but
//! whose rope pairing, norm scheme or key/value sharing is wrong still produces
//! finite logits, so the only assertion that catches it is one about meaning.
//!
//! The second is the reason the per-layer table has two holdings at all. The
//! streamed table reads a row off the disk per token; the resident one indexes a
//! two-gigabyte tensor on the device. They are two paths to the same numbers, so
//! the sixteen ids must match AND the first logits row must be bit-identical - a
//! nearly-identical row would mean the two paths dequantize differently, which is
//! a difference no token sequence is guaranteed to show.
//!
//! Both need the container, the sidecar vocabulary and a Metal device, so both
//! are `#[ignore]`d and skip with a printed note when any is missing.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::HfConfig;
use shiso_core::backend::Backend;
use shiso_core::ids::TokenId;
use shiso_core::model::{DecodeStep, Model};
use shiso_engine::{EngineOptions, argmax};
use shiso_metal::{MetalBackend, metal_available};
use shiso_models::gemma4::{Gemma4Config, Gemma4Model, PleOptions};

/// The prompt, in this family's own turn markup.
///
/// Written out rather than rendered through a template engine: the template
/// ships with the checkpoint and this is the one branch of it a question with no
/// system turn and no tools takes. `<bos>` is spelled in the string because the
/// sidecar vocabulary's post-processor adds nothing, so the only way the
/// sequence starts with id 2 is for the text to say so.
const PROMPT: &str = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";

/// Steps the greedy run takes.
const STEPS: usize = 16;

/// Vocabulary size, which is the width of one logits row.
const VOCAB: u32 = 262_144;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports what a run produced, for the eye reading the log.
#[expect(clippy::print_stdout, reason = "the decoded answer is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The local container, overridable, or `None` when it is not on this machine.
fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4").ok()?);
    path.is_file().then_some(path)
}

/// The vocabulary the prompt is encoded with.
///
/// Read from the sibling checkpoint directory rather than committed beside the
/// test: the file is tens of megabytes, which is not a thing to carry in a
/// source tree for one prompt.
fn tokenizer_path() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX").ok()?);
    let path = dir.join("tokenizer.json");
    path.is_file().then_some(path)
}

/// The E4B geometry, read from the shipped document.
fn profile() -> Gemma4Config {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/loader/gemma4-config.json");
    let document =
        HfConfig::from_file(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    Gemma4Config::from_hf(&document, Path::new("/nonexistent")).expect("the e4b geometry")
}

/// A context small enough that the cache is not what these cases measure.
fn options() -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = 1024;
    opts.kv_pool_pages = 64;
    opts
}

/// The prompt's ids, with exactly one `<bos>` at the front.
fn encode(tokenizer: &tokenizers::Tokenizer) -> Vec<TokenId> {
    let encoded = tokenizer.encode(PROMPT, true).expect("the prompt encodes");
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    assert_eq!(
        ids.first().copied(),
        Some(2),
        "the rendered prompt starts with <bos>: {ids:?}"
    );
    assert_ne!(
        ids.get(1).copied(),
        Some(2),
        "the vocabulary's post-processor added a second <bos>: {ids:?}"
    );
    ids.into_iter().map(TokenId::new).collect()
}

/// The highest-scoring id of one logits row, read out of its bytes.
fn greedy(row: &[u8], scratch: &mut [f32]) -> TokenId {
    for (slot, bytes) in scratch.iter_mut().zip(row.as_chunks::<4>().0) {
        *slot = f32::from_le_bytes(*bytes);
    }
    argmax(scratch)
}

/// What one greedy run produced: the ids, and the bytes of its first row.
struct Run {
    /// The ids the run chose, in order.
    ids: Vec<u32>,
    /// The prompt's logits row, exactly as the device held it.
    first_row: Vec<u8>,
}

/// Loads the checkpoint with the given per-layer holding and decodes `STEPS`
/// greedy tokens after the prompt.
fn decode_greedy(prompt: &[TokenId], ple: PleOptions) -> Run {
    let path = checkpoint().expect("caller checked the container is present");
    let config = profile();
    let backend: Arc<dyn Backend> = MetalBackend::new().expect("a Metal device");

    let started = std::time::Instant::now();
    let mut model =
        Gemma4Model::load(&backend, &path, &config, options(), ple).expect("the container loads");
    report(&format!(
        "load ({}): {:.3}s",
        if ple.resident { "resident" } else { "streamed" },
        started.elapsed().as_secs_f64()
    ));

    let vocab = usize::try_from(VOCAB).expect("a vocabulary fits a host index");
    let mut row = vec![0u8; vocab * 4];
    let mut scratch = vec![0f32; vocab];

    let session = model.open_session().expect("a session opens");
    let decoding = std::time::Instant::now();
    let last = model
        .prefill_last(session, prompt)
        .expect("the prompt runs");
    backend
        .copy_d2h(last.row(0).expect("the prompt's row"), &mut row)
        .expect("the prompt's row reads back");
    let first_row = row.clone();

    // The prompt's own logits row already names the first token, so the loop
    // that follows runs one fewer decode than there are tokens to report. The
    // two counts are kept apart rather than conflated, because a rate that
    // divided the token count by a window containing one fewer step would
    // overstate itself by a sixteenth.
    let mut ids = Vec::with_capacity(STEPS);
    let mut token = greedy(&row, &mut scratch);
    ids.push(token.get());
    let mut decodes = 0usize;
    while ids.len() < STEPS {
        let out = model
            .decode_batch(&[DecodeStep { session, token }])
            .expect("a decode step runs");
        backend
            .copy_d2h(out.row(0).expect("the step's row"), &mut row)
            .expect("the step's row reads back");
        token = greedy(&row, &mut scratch);
        ids.push(token.get());
        decodes += 1;
    }
    assert_eq!(decodes, STEPS - 1, "one token comes out of the prefill");
    let elapsed = decoding.elapsed().as_secs_f64();
    let produced = ids.len();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of sixteen is exact in a double"
    )]
    let rate = produced as f64 / elapsed;
    report(&format!(
        "prefill + {decodes} decode steps ({}): {elapsed:.3}s for {produced} tokens, \
         {rate:.2} tok/s over that window",
        if ple.resident { "resident" } else { "streamed" }
    ));

    model.close_session(session).expect("the session closes");
    model.teardown().expect("the model tears down");
    Run { ids, first_row }
}

#[test]
#[ignore = "needs the local checkpoint, the sidecar vocabulary and a Metal device"]
fn the_streamed_table_answers_the_question_and_the_resident_one_answers_identically() {
    let Some(_container) = checkpoint() else {
        skip("no local Gemma 4 container");
        return;
    };
    let Some(vocabulary) = tokenizer_path() else {
        skip("no local sidecar vocabulary");
        return;
    };
    if !metal_available() {
        skip("no Metal device");
        return;
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&vocabulary).expect("the vocabulary loads");
    let prompt = encode(&tokenizer);

    let streamed = decode_greedy(&prompt, PleOptions::default());
    let text = tokenizer
        .decode(&streamed.ids, true)
        .expect("the answer decodes");
    report(&format!("streamed answer: {text:?}"));
    assert!(
        text.contains("Paris"),
        "the model names the capital of France; it said {text:?} (ids {:?})",
        streamed.ids
    );

    let resident = decode_greedy(
        &prompt,
        PleOptions {
            resident: true,
            ..PleOptions::default()
        },
    );
    let resident_text = tokenizer
        .decode(&resident.ids, true)
        .expect("the answer decodes");
    report(&format!("resident answer: {resident_text:?}"));

    assert_eq!(
        streamed.ids, resident.ids,
        "the two holdings of one table produced different continuations"
    );
    assert!(
        streamed.first_row == resident.first_row,
        "the two holdings of one table produced different logits for the same prompt, \
         so they decode its rows differently"
    );
}
