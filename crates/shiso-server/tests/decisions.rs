//! `POST /v1/decisions` driven through the router: one prefill per question,
//! the labels scored from a scripted logits row, nothing generated.
//!
//! The model behind the engine answers prefills with whole rows this file
//! scripts as numbers, so every probability assertion is arithmetic a reader
//! can check against the same row rather than a constant copied from the
//! server's own math.
//!
//! The vocabulary is its own rather than `openai.rs`'s: a label can only be
//! scored if it is one token at the end of the rendered prompt, so the words
//! that answer - the letters, the digits, `yes`, `no` - must exist as single
//! tokens here, and the template must end at a whitespace boundary so a label
//! encodes as its own chunk. It is `WordLevel` over `Whitespace`, which means
//! an unknown word collapses to `<unk>` (id 0): every word of the two prompt
//! bodies pinned by the prompt-shape test is therefore in the vocabulary too,
//! so a wording change moves at least one recorded id. What this tokenizer
//! cannot see is whitespace *between* chunks - newlines and spaces encode the
//! same - and that seam is pinned by the wording goldens beside the builder.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_core::{SamplingParams, Seed, TokenId};
use shiso_server::router::build_router;
use shiso_server::scheduler::{GenerateRequest, SchedulerConfig};
use shiso_server::state::AppState;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::jinja_helpers::ToolJsonStyle;
use shiso_server::tokenizer::render_prompt::PromptRenderer;
use shiso_vision::{ImageLimits, MediaPlacement};
use tower::ServiceExt as _;

/// The fake model's logits width: a scripted row must cover every logit the
/// scoring pass reads, or the tail of a previous row rides into the
/// log-sum-exp.
const VOCAB: usize = 64;

/// The context every state is built with unless a test wants a smaller one.
const CTX: u32 = 256;

/// The words the vocabulary defines, in id order: a word's index is its id.
/// The tests name labels by word, so the two views cannot drift apart.
const WORDS: &[&str] = &[
    "<unk>", // 0: every unknown chunk collapses here
    "<|user|>",
    "<|end|>",
    "<|assistant|>",
    "A",
    "B",
    "C",
    "D",
    "0",
    "1",
    "2",
    "yes",
    "no",
    // The words of the two prompt bodies the shape test pins. Each is one
    // whitespace chunk of them, so each claims an id and wording moves ids.
    "the",
    "input",
    "Question:",
    "pick",
    "one",
    "A:",
    "red",
    "B:",
    "blue",
    "Answer",
    "with",
    "letter",
    "of",
    "option",
    "only.",
    "Is",
    "following",
    "true?",
    "ready",
    "yes:",
    "confirmed",
    "no:",
    "denied",
    "or",
];

/// The id of one word of the test vocabulary.
fn word(text: &str) -> usize {
    WORDS
        .iter()
        .position(|candidate| *candidate == text)
        .expect("the word is in the test vocabulary")
}

/// The template: markers and message separated by spaces, so every prompt word
/// is its own whitespace chunk, and one newline after the assistant opener -
/// the boundary that lets a label encode as one fresh token after the prompt.
const TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|> ",
    "{% for part in m.content %}{{ part.text }}{% endfor %} <|end|> ",
    "{% endfor %}<|assistant|>\n",
);

/// The prompt this template puts around one user message's body.
fn rendered_prompt(body: &str) -> String {
    format!("<|user|> {body} <|end|> <|assistant|>\n")
}

/// A vocabulary of `WORDS` as one-token words, built the way `openai.rs`
/// builds its own: through the document the library loads.
fn test_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    for (id, text) in WORDS.iter().enumerate() {
        vocab.insert((*text).to_owned(), json!(id));
    }
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": [],
        "normalizer": Value::Null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    })
    .to_string();
    let inner = tokenizers::Tokenizer::from_str(&document).expect("the vocabulary loads");
    Arc::new(Vocabulary::from_tokenizer(inner))
}

/// A server whose engine answers the next prefills with these whole rows,
/// in order, with `ctx` tokens of room. The audit comes back because the
/// prompt-shape test reads the ids the model was asked to read.
fn state_with_rows(rows: Vec<Vec<f32>>, ctx: u32) -> (Arc<AppState>, Arc<common::Audit>) {
    let vocabulary = test_vocabulary();
    let (engine, _join, audit) = common::fake_engine_rows(rows, Some(Arc::clone(&vocabulary)));
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    let state = Arc::new(
        AppState::new(engine, "test-model".to_owned(), ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    );
    (state, audit)
}

/// A full logits row: every slot `base`, the named slots their values. A row
/// with no named slots is the flat row whose answer is uniform, which is what
/// a test that only asserts on prompts wants.
fn row_with(base: f32, slots: &[(usize, f32)]) -> Vec<f32> {
    let mut row = vec![base; VOCAB];
    for (slot, value) in slots {
        row[*slot] = *value;
    }
    row
}

/// The log-sum-exp of `row` in `f64`, computed here by the same formula so the
/// assertions are independent of the engine's arithmetic.
fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |best, v| best.max(*v));
    f64::from(max)
        + row
            .iter()
            .map(|v| (f64::from(*v) - f64::from(max)).exp())
            .sum::<f64>()
            .ln()
}

/// The softmax of `logprobs / temperature`, computed here as well, from the
/// row's own numbers.
fn softmax(logprobs: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = logprobs.iter().map(|lp| lp / temperature).collect();
    let max = scaled
        .iter()
        .fold(f64::NEG_INFINITY, |best, value| best.max(*value));
    let exps: Vec<f64> = scaled.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|value| value / sum).collect()
}

/// The `probabilities` of one answer, as `f64`.
fn probabilities(answer: &Value) -> Vec<f64> {
    answer["probabilities"]
        .as_array()
        .expect("an answer carries its probabilities")
        .iter()
        .map(|p| p.as_f64().expect("a probability is a number"))
        .collect()
}

/// The answer of one question id.
fn answer_of(body: &Value, id: &str) -> Value {
    body["answers"][id].clone()
}

/// One POST with a JSON body.
fn post(path: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds")
}

/// Sends one request and reads the whole answer as JSON.
async fn call(state: &Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let body = if bytes.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(Value::Null)
    };
    (status, body)
}

/// The smallest body that answers: one yes/no question over one input.
fn valid_body() -> Value {
    json!({
        "input": "the input",
        "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
    })
}

/// The choice question the shape test pins: its exact version-1 body.
const CHOICE_BODY: &str =
    "the input\n\nQuestion: pick one\nA: red\nB: blue\nAnswer with the letter of one option only.";

/// The yes/no question the shape test pins: its exact version-1 body.
const YES_NO_BODY: &str = "the input\n\nIs the following true? ready\nyes: confirmed\nno: denied\nAnswer with yes or no only.";

/// The ids of one expected prompt, encoded through the test vocabulary.
fn prompt_ids(body: &str) -> Vec<u32> {
    test_vocabulary()
        .encode(&rendered_prompt(body), false)
        .expect("the prompt encodes")
        .into_iter()
        .map(TokenId::get)
        .collect()
}

// --- the scored answers -----------------------------------------------------

#[tokio::test]
async fn a_choice_question_answers_the_scripted_argmax() {
    // B stands above A in the scripted row, so the answer is B's option name
    // and the probabilities are the label softmax of exactly that row.
    let row = row_with(0.0, &[(word("A"), 2.0), (word("B"), 4.0)]);
    let (state, audit) = state_with_rows(vec![row.clone()], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "model": "echo-me",
                "input": "the input",
                "questions": [{
                    "id": "q1",
                    "type": "choice",
                    "question": "pick one",
                    "options": ["first option", "second option"],
                }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "decisions");
    assert_eq!(body["model"], "echo-me", "the model string is echoed");
    assert_eq!(body["prompt_format_version"], 1);
    let keys: Vec<&str> = body["answers"]
        .as_object()
        .expect("answers is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["q1"], "the answers keep the request's order");
    let q1 = answer_of(&body, "q1");
    assert_eq!(q1["type"], "choice");
    assert_eq!(
        q1["choice"], "second option",
        "the top label names its option"
    );

    let norm = logsumexp(&row);
    let lp = [2.0 - norm, 4.0 - norm];
    let want = softmax(&lp, 1.0);
    let got = probabilities(&q1);
    assert_eq!(got.len(), 2);
    for (index, (got, want)) in got.iter().zip(&want).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "label {index}: {got} is not {want}: {body}"
        );
    }
    let mass: f64 = lp.iter().map(|lp| lp.exp()).sum();
    assert!(
        (q1["label_mass"].as_f64().expect("a mass") - mass).abs() < 1e-5,
        "{body}"
    );

    let prompt_tokens = body["usage"]["prompt_tokens"]
        .as_u64()
        .expect("the accounting counts the prompt");
    let prompts = audit.prefill_prompts.lock();
    assert_eq!(prompts.len(), 1, "one question, one prefill");
    assert_eq!(
        prompt_tokens,
        u64::try_from(prompts[0].len()).expect("a small prompt"),
        "the count is the prompt that was read: {body}"
    );
    assert_eq!(body["usage"]["completion_tokens"], 0, "nothing generated");
    assert_eq!(body["usage"]["total_tokens"], prompt_tokens);
}

#[tokio::test]
async fn a_score_question_returns_the_expectation() {
    let row = row_with(0.0, &[(word("0"), 1.0), (word("1"), 2.0), (word("2"), 3.0)]);
    let (state, _audit) = state_with_rows(vec![row.clone()], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [{
                    "id": "s1",
                    "type": "score",
                    "question": "how bad",
                    "levels": ["none", "some", "lots"],
                }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "test-model", "absent model names the server");
    let s1 = answer_of(&body, "s1");
    assert_eq!(s1["type"], "score");

    let norm = logsumexp(&row);
    let lp = [1.0 - norm, 2.0 - norm, 3.0 - norm];
    let want: f64 = softmax(&lp, 1.0)
        .iter()
        .enumerate()
        .map(|(index, p)| f64::from(u32::try_from(index).expect("a small index")) * p)
        .sum();
    let got = s1["score"].as_f64().expect("a score");
    assert!(
        (got - want).abs() < 1e-5,
        "the expectation Σ i·p_i is {want}, got {got}: {body}"
    );
    assert!(
        s1.get("choice").is_none() && s1.get("answer").is_none(),
        "{body}"
    );
}

#[tokio::test]
async fn a_yes_no_question_answers_with_a_label() {
    let row = row_with(0.0, &[(word("yes"), 3.0)]);
    let (state, _audit) = state_with_rows(vec![row.clone()], CTX);
    let (status, body) = call(&state, post("/v1/decisions", &valid_body())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let q1 = answer_of(&body, "q1");
    assert_eq!(q1["type"], "yes_no");
    assert_eq!(q1["answer"], "yes");

    let norm = logsumexp(&row);
    let want = softmax(&[3.0 - norm, 0.0 - norm], 1.0);
    let got = probabilities(&q1);
    for (got, want) in got.iter().zip(&want) {
        assert!((got - want).abs() < 1e-5, "{body}");
    }
}

#[tokio::test]
async fn questions_are_scored_against_prompts_carrying_the_shared_input() {
    // Every question is scored from its own prompt: the shared input, one
    // blank line, then that question's version-1 block. The ids below are the
    // encoding of the exact rendered bytes, so the wording is pinned from the
    // wire to the model.
    let (state, audit) = state_with_rows(vec![row_with(0.5, &[]), row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [
                    {"id": "q1", "type": "choice", "question": "pick one",
                     "options": ["red", "blue"]},
                    {"id": "q2", "type": "yes_no", "question": "ready",
                     "yes": "confirmed", "no": "denied"},
                ],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want_choice = prompt_ids(CHOICE_BODY);
    let want_yes_no = prompt_ids(YES_NO_BODY);
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(got.len(), 2, "one prompt per question: {got:?}");
    assert!(
        got.contains(&want_choice),
        "the choice prompt was not read: {got:?}"
    );
    assert!(
        got.contains(&want_yes_no),
        "the yes/no prompt was not read: {got:?}"
    );
    let total = want_choice.len() + want_yes_no.len();
    assert_eq!(
        body["usage"]["prompt_tokens"],
        json!(u64::try_from(total).expect("a small prompt")),
        "the accounting is the two prompts together: {body}"
    );
}

/// The key-sorted compact rendering of `{"b": 1, "a": [1, 2]}`: the bytes
/// the shared render rule owes a container input, pinned here in the order
/// the server - not the client - chooses.
const CONTAINER_TEXT: &str = "{\"a\":[1,2],\"b\":1}";

#[tokio::test]
async fn a_container_input_is_scored_from_its_key_sorted_rendering() {
    // End to end, not just at the render helper: a container `input`
    // reaches the model as compact JSON with keys sorted, so the same
    // logical object spells one prompt whatever order its client sent -
    // and the audited ids are exactly those of that rendering in the
    // shared template.
    let (state, audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": {"b": 1, "a": [1, 2]},
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want = prompt_ids(&format!(
        "{CONTAINER_TEXT}\n\nIs the following true? ready\nAnswer with yes or no only."
    ));
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(
        got,
        vec![want],
        "the model read the key-sorted bytes: {got:?}"
    );
}

#[tokio::test]
async fn temperature_rescales_the_probabilities() {
    // The label log-probabilities come from the model; the temperature divides
    // them before the softmax, so T = 2 halves the gap between the labels.
    let row = row_with(0.0, &[(word("A"), 2.0), (word("B"), 4.0)]);
    let (state, _audit) = state_with_rows(vec![row.clone()], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "temperature": 2.0,
                "questions": [{
                    "id": "q1", "type": "choice", "question": "pick one",
                    "options": ["red", "blue"],
                }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let norm = logsumexp(&row);
    let want = softmax(&[2.0 - norm, 4.0 - norm], 2.0);
    let got = probabilities(&answer_of(&body, "q1"));
    for (got, want) in got.iter().zip(&want) {
        assert!(
            (got - want).abs() < 1e-5,
            "T=2 probabilities must be softmax(lp/2): {body}"
        );
    }
}

#[tokio::test]
async fn return_prompt_token_ids_adds_both_id_lists() {
    let row = row_with(0.0, &[(word("no"), 3.0)]);
    let (state, audit) = state_with_rows(vec![row], CTX);
    let mut body = valid_body();
    body["return_prompt_token_ids"] = json!(true);
    let (status, replied) = call(&state, post("/v1/decisions", &body)).await;
    assert_eq!(status, StatusCode::OK, "{replied}");
    let q1 = answer_of(&replied, "q1");
    assert_eq!(
        q1["prompt_token_ids"],
        json!(audit.prefill_prompts.lock()[0]),
        "the ids the model read, not a recount"
    );
    assert_eq!(
        q1["label_token_ids"],
        json!([word("yes"), word("no")]),
        "the label ids in the question's own order"
    );

    // Without the flag neither list is in the answer.
    let (quiet, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (_status, replied) = call(&quiet, post("/v1/decisions", &valid_body())).await;
    let q1 = answer_of(&replied, "q1");
    assert!(
        q1.get("prompt_token_ids").is_none() && q1.get("label_token_ids").is_none(),
        "{replied}"
    );
}

// --- refusals ----------------------------------------------------------------

#[tokio::test]
async fn unknown_fields_are_refused_everywhere() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let cases = vec![
        json!({"input": "the input", "questions": [{"id": "q1", "type": "yes_no",
                "question": "ready"}], "nope": true}),
        json!({"input": "the input", "questions": [{"id": "q1", "type": "yes_no",
                "question": "ready", "bogus": 1}]}),
        json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
                "question": "pick one",
                "options": [{"name": "red", "bogus": 1}, {"name": "blue"}]}]}),
    ];
    for case in cases {
        let (status, body) = call(&state, post("/v1/decisions", &case)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case} -> {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    }
    // The complaint names the key the caller misspelled or invented.
    let (_status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({"input": "the input", "questions": [], "nope": true}),
        ),
    )
    .await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("nope")),
        "{body}"
    );
    // The same refusal inside an option object, which is a hand-written
    // parse rather than a derived struct, must also name the invented key.
    let (_status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
                    "question": "pick one",
                    "options": [{"name": "red", "bogus": 1}, {"name": "blue"}]}]}),
        ),
    )
    .await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("bogus")),
        "{body}"
    );

    // And the keys an option object really does have still parse: the full
    // spelling and the name-only spelling alike.
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
                    "question": "pick one",
                    "options": [{"name": "red", "description": "a color"}, {"name": "blue"}]}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn prompt_format_version_other_than_one_is_refused() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let mut body = valid_body();
    body["prompt_format_version"] = json!(2);
    let (status, replied) = call(&state, post("/v1/decisions", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{replied}");
    assert_eq!(replied["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn validation_matrix() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let options = |names: &[&str]| -> Vec<Value> { names.iter().map(|name| json!(name)).collect() };
    let mut wide = Vec::new();
    for index in 0..27 {
        wide.push(json!(format!("opt {index}")));
    }
    let mut tall = Vec::new();
    for index in 0..11 {
        tall.push(json!(format!("level {index}")));
    }
    let cases: Vec<(Value, Option<(&str, &str)>)> = vec![
        (
            json!({"input": "   ", "questions": [{"id": "q1", "type": "yes_no",
             "question": "ready"}]}),
            Some(("param", "input")),
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
             "question": "  ", "options": ["red", "blue"]}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "", "type": "yes_no",
             "question": "ready"}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [
             {"id": "q1", "type": "yes_no", "question": "a"},
             {"id": "q1", "type": "yes_no", "question": "b"}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
             "question": "pick", "options": wide}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
             "question": "pick", "options": options(&["only"])}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "score",
             "question": "how bad", "levels": ["ok", " "]}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
             "question": "pick", "options": options(&["Red", "red"])}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "score",
             "question": "how bad", "levels": tall}]}),
            None,
        ),
        (
            json!({"input": "the input", "temperature": 0.0, "questions":
             [{"id": "q1", "type": "yes_no", "question": "ready"}]}),
            Some(("param", "temperature")),
        ),
        (
            json!({"model": "base:lora", "input": "the input", "questions":
             [{"id": "q1", "type": "yes_no", "question": "ready"}]}),
            Some(("param", "model")),
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "choice",
             "options": ["red", "blue"]}]}),
            None,
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q1", "type": "score",
             "question": "how bad", "levels": ["a", "b"], "options": ["x", "y"]}]}),
            Some(("message", "options")),
        ),
        (
            json!({"input": "the input", "questions": [{"id": "q9", "type": "mystery",
             "question": "ready"}]}),
            Some(("message", "q9")),
        ),
    ];
    for (case, checked) in cases {
        let (status, body) = call(&state, post("/v1/decisions", &case)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case} -> {body}");
        assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
        match checked {
            Some(("param", want)) => assert_eq!(body["error"]["param"], want, "{body}"),
            Some(("message", want)) => assert!(
                body["error"]["message"]
                    .as_str()
                    .is_some_and(|message| message.contains(want)),
                "{body} must name {want}"
            ),
            Some(_) => unreachable!("the table only pairs param or message checks"),
            None => {}
        }
    }
}

// --- content-block `input`: the wire split, refused or kept by `validate` ---

#[tokio::test]
async fn a_malformed_block_array_is_refused_naming_the_element() {
    // One element looks like a block, so the array is blocks and element 1
    // is held to the block grammar (Review Focus 1): the typo'd `img` type
    // is refused with a 400 naming `input` and the element's index, never
    // fallen back to a scored compact-JSON prompt.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [{"type": "text", "text": "a"}, {"type": "img"}],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert_eq!(body["error"]["param"], "input", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains('1')),
        "the refusal must name index 1: {body}"
    );
}

#[tokio::test]
async fn audio_and_video_blocks_are_refused_naming_the_type() {
    // The scoring endpoints take images only in v1: the refusal parses far
    // enough to name the block's type.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    for kind in ["audio", "video"] {
        let (status, body) = call(
            &state,
            post(
                "/v1/decisions",
                &json!({
                    "input": [{"type": kind, "source": {}}],
                    "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{kind} -> {body}");
        assert_eq!(body["error"]["param"], "input", "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(kind)),
            "the refusal must name {kind}: {body}"
        );
    }
}

#[tokio::test]
async fn an_all_unknown_types_array_stays_a_container() {
    // No element carries a known `type`, so the array is a JSON container by
    // design (spec ruling 1) and scores as today's key-sorted compact JSON.
    let (state, audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [{"type": "photo", "url": "x"}],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want = prompt_ids(
        "{\"type\":\"photo\",\"url\":\"x\"}\n\nIs the following true? ready\nAnswer with yes or no only.",
    );
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(
        got,
        vec![want],
        "the model read the compact-JSON rendering: {got:?}"
    );
}

#[tokio::test]
async fn an_object_with_a_type_key_is_never_blocks() {
    // Review Focus 2: a top-level object is never blocks, however much it
    // spells like an image block - it stays a container and scores as
    // key-sorted compact JSON.
    let (state, audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want = prompt_ids(
        "{\"source\":{\"data\":\"aGk=\",\"media_type\":\"image/png\",\"type\":\"base64\"},\"type\":\"image\"}\n\nIs the following true? ready\nAnswer with yes or no only.",
    );
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(
        got,
        vec![want],
        "the model read the compact-JSON rendering: {got:?}"
    );
}

#[tokio::test]
async fn label_checks_refuse_a_question_whose_label_is_not_one_token() {
    // Five score levels need the labels 0..4. This vocabulary defines 0..2,
    // and a WordLevel vocabulary maps every unknown word to `<unk>` - one
    // token, so the one-token arm passes - which leaves the distinctness arm:
    // "3" and "4" collapse to the same id, and a label pointing at another
    // label's token could never win. The refusal names the question.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [{
                    "id": "q5", "type": "score", "question": "how bad",
                    "levels": ["a", "b", "c", "d", "e"],
                }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("q5")),
        "the refusal must name the question: {body}"
    );
}

/// `TEMPLATE` with the answer position glued flush to a word: the prompt
/// then ends `<|assistant|>Answer`, with no whitespace boundary after it.
/// This test vocabulary splits at punctuation as well as whitespace -
/// `<|assistant|>` alone is already three chunks and `>` ends one - so a
/// template that merely dropped the newline after the opener would still
/// leave a label its own chunk; ending the prompt on a word is what makes
/// the tail swallow a label.
const GLUED_TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|> ",
    "{% for part in m.content %}{{ part.text }}{% endfor %} <|end|> ",
    "{% endfor %}<|assistant|>Answer",
);

/// A server on [`GLUED_TEMPLATE`], otherwise identical to `state_with_rows`.
fn glued_state(rows: Vec<Vec<f32>>) -> (Arc<AppState>, Arc<common::Audit>) {
    let vocabulary = test_vocabulary();
    let (engine, _join, audit) = common::fake_engine_rows(rows, Some(Arc::clone(&vocabulary)));
    let renderer = PromptRenderer::new(
        GLUED_TEMPLATE,
        ToolJsonStyle::Compact,
        Arc::clone(&vocabulary),
    )
    .expect("the test template compiles");
    let state = Arc::new(
        AppState::new(engine, "test-model".to_owned(), CTX)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    );
    (state, audit)
}

#[tokio::test]
async fn label_checks_refuse_a_label_that_would_merge_into_the_prompt_tail() {
    // The join arm of the label checks: a label that is one fresh token on
    // its own but merges with the prompt's last chunk when the two are
    // encoded together could never be read at the answer position. Here the
    // tail word swallows it - `Answer` alone is the vocabulary's own token,
    // but `AnswerA` encodes as one unknown chunk - so the refusal is a 400
    // naming the question, in the same shape as the other label proofs.
    let (state, _audit) = glued_state(vec![row_with(0.5, &[])]);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [{
                    "id": "q9", "type": "choice", "question": "pick one",
                    "options": ["red", "blue"],
                }],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("q9")),
        "the refusal must name the question: {body}"
    );
}

#[tokio::test]
async fn context_overflow_names_the_input_param() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], 12);
    let (status, body) = call(&state, post("/v1/decisions", &valid_body())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    assert_eq!(body["error"]["param"], "input");
    assert_eq!(body["error"]["n_ctx"], 12);
    assert!(
        body["error"]["n_prompt_tokens"]
            .as_u64()
            .is_some_and(|n| n >= 12),
        "{body}"
    );
}

#[tokio::test]
async fn chat_template_kwargs_keys_are_mapped_or_refused() {
    let (state, _audit) = state_with_rows(
        vec![row_with(0.5, &[]), row_with(0.5, &[]), row_with(0.5, &[])],
        CTX,
    );
    for (kwargs, allowed) in [
        (json!({"enable_thinking": false}), true),
        (json!({"enable_thinking": null}), true),
        (json!({"enable_thinking": true}), false),
        (json!({"custom": 1}), false),
        (json!({"reasoning_effort": "wild"}), false),
        (json!({"reasoning_effort": "low"}), true),
        (json!({"preserve_thinking": true}), true),
    ] {
        let mut body = valid_body();
        body["chat_template_kwargs"] = kwargs.clone();
        let (status, replied) = call(&state, post("/v1/decisions", &body)).await;
        if allowed {
            assert_eq!(status, StatusCode::OK, "{kwargs} -> {replied}");
        } else {
            assert_eq!(status, StatusCode::BAD_REQUEST, "{kwargs} -> {replied}");
            assert_eq!(replied["error"]["type"], "invalid_request_error");
        }
    }
    // The refusal quotes the key the caller is holding.
    let (_status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({"input": "the input", "questions": [],
                    "chat_template_kwargs": {"custom": 1}}),
        ),
    )
    .await;
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("custom")),
        "{body}"
    );
}

#[tokio::test]
async fn an_overloaded_engine_surfaces_as_503() {
    // The engine thread is parked inside a gated decode step, so nothing
    // drains the queue; one score takes the only slot and waits for its
    // answer, and the next is refused once its admission window runs out.
    let gate = common::Gate::closed();
    let cfg = SchedulerConfig {
        max_num_seqs: 1,
        max_queued: 1,
        queue_admission: Duration::from_millis(20),
        ..SchedulerConfig::new()
    };
    let (engine, _join, audit) = common::scheduled_engine(vec![10], cfg, Some(Arc::clone(&gate)));
    let vocabulary = test_vocabulary();
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    let state = Arc::new(
        AppState::new(engine.clone(), "test-model".to_owned(), CTX)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    );

    let _stream = engine
        .generate(GenerateRequest::new(
            Arc::from(vec![TokenId::new(10)]),
            SamplingParams {
                temperature: 0.0,
                ..SamplingParams::neutral()
            },
            Seed::new(7),
            4,
        ))
        .await
        .expect("the generation is admitted");
    // Wait until the model has read the prompt, then give the engine thread
    // its next tick to enter the gated decode step.
    common::wait_for_calls(&audit, 1).await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // A score that takes the queue slot and never gets an answer...
    let router = build_router(Arc::clone(&state));
    tokio::spawn(async move {
        let _ = router.oneshot(post("/v1/decisions", &valid_body())).await;
    });
    common::wait_for_queued(&engine, 2).await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    // ...makes the next one a refusal the client can back off from.
    let (status, body) = call(&state, post("/v1/decisions", &valid_body())).await;
    gate.open();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(body["error"]["code"], "server_overloaded");
}

// --- image content blocks: the scored media path -----------------------------

/// The image marker's id in `media_vocabulary`: past every word of `WORDS`
/// and inside the scripted row's width.
const PAD: u32 = 40;

/// The vocabulary the media tests render with: every word of `WORDS` plus
/// Qwen's image marker as a declared added token. Added rather than ordinary
/// because that is what lets the renderer fan one marker out to a pad run.
fn media_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    for (id, text) in WORDS.iter().enumerate() {
        vocab.insert((*text).to_owned(), json!(id));
    }
    vocab.insert(MediaPlacement::QWEN35.image_marker.to_owned(), json!(PAD));
    let added = vec![json!({
        "id": PAD,
        "content": MediaPlacement::QWEN35.image_marker,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true,
    })];
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
        "normalizer": Value::Null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": Value::Null,
        "decoder": Value::Null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
    })
    .to_string();
    let inner = tokenizers::Tokenizer::from_str(&document).expect("the media vocabulary loads");
    Arc::new(Vocabulary::from_tokenizer(inner))
}

/// `TEMPLATE` with the image branch a family's template carries: one marker
/// per picture, text parts verbatim. For a text-only message it renders
/// byte-identically to `TEMPLATE`.
const MEDIA_TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|> ",
    "{% for part in m.content %}",
    "{% if part.type == 'image' %}<|image_pad|>{% else %}{{ part.text }}{% endif %}",
    "{% endfor %} <|end|> ",
    "{% endfor %}<|assistant|>\n",
);

/// [`GLUED_TEMPLATE`] with the same image branch.
const GLUED_MEDIA_TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|> ",
    "{% for part in m.content %}",
    "{% if part.type == 'image' %}<|image_pad|>{% else %}{{ part.text }}{% endif %}",
    "{% endfor %} <|end|> ",
    "{% endfor %}<|assistant|>Answer",
);

/// A server in the shape a `--vision` server has: a renderer with the tower's
/// geometry attached, and an engine holding a fake tower and a model that can
/// read media. `glued` picks the template whose prompt swallows a label.
fn media_state(rows: Vec<Vec<f32>>, ctx: u32, glued: bool) -> (Arc<AppState>, Arc<common::Audit>) {
    let vocabulary = media_vocabulary();
    let (engine, _join, audit, _media_calls) =
        common::fake_media_engine(rows, Some(Arc::clone(&vocabulary)), false);
    let template = if glued {
        GLUED_MEDIA_TEMPLATE
    } else {
        MEDIA_TEMPLATE
    };
    let renderer = PromptRenderer::new(template, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the media template compiles")
        .with_tower(
            &common::media_spec(),
            MediaPlacement::QWEN35,
            ImageLimits::default(),
        );
    let state = Arc::new(
        AppState::new(engine, "test-model".to_owned(), ctx)
            .with_vocabulary(vocabulary)
            .with_renderer(Arc::new(renderer)),
    );
    (state, audit)
}

/// One base64 image block as the wire spells it.
fn image_block(data: &str) -> Value {
    json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
}

/// The ids the model should read for a media prompt: the rendered text
/// encoded through the media vocabulary, with every marker occurrence fanned
/// out to `pads` pad ids - the expansion the renderer owes the prompt.
fn expanded_ids(vocabulary: &Arc<Vocabulary>, text_with_markers: &str, pads: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    for token in vocabulary
        .encode(text_with_markers, false)
        .expect("the prompt encodes")
    {
        if token.get() == PAD {
            ids.extend(std::iter::repeat_n(PAD, pads));
        } else {
            ids.push(token.get());
        }
    }
    ids
}

/// The yes/no question block, as `prompt_body` spells it for this body.
const YES_NO_BLOCK: &str = "Is the following true? ready\nAnswer with yes or no only.";

#[tokio::test]
async fn an_image_request_scores_the_expanded_prompt() {
    // Whitespace golden 1: the caller's blocks come first, the question block
    // last, joined by the pinned blank line because the last caller block is
    // text. The marker reaches the model fanned to its pad run.
    let png = common::png_data_uri(96, 96);
    let (state, audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png), {"type": "text", "text": "urgent?"}],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want = expanded_ids(
        &media_vocabulary(),
        &format!("<|user|> <|image_pad|>urgent?\n\n{YES_NO_BLOCK} <|end|> <|assistant|>\n"),
        4,
    );
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(
        got,
        vec![want],
        "the model read one marker fanned to four pads, the question block last: {got:?}"
    );
}

#[tokio::test]
async fn a_trailing_image_block_pins_the_question_block_without_a_blank_line() {
    // Whitespace golden 2: the last caller block is an image, so the question
    // block follows the marker as plain wording - no blank line, because the
    // pad run already ends the text side of the seam.
    let png = common::png_data_uri(96, 96);
    let (state, audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [{"type": "text", "text": "urgent?"}, image_block(&png)],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let want = expanded_ids(
        &media_vocabulary(),
        &format!("<|user|> urgent?<|image_pad|>{YES_NO_BLOCK} <|end|> <|assistant|>\n"),
        4,
    );
    let got: Vec<Vec<u32>> = audit.prefill_prompts.lock().clone();
    assert_eq!(
        got,
        vec![want],
        "the question block followed the marker with no blank line: {got:?}"
    );
}

#[tokio::test]
async fn an_image_only_input_is_valid() {
    // "Blank" means no text *and* no images (spec ruling 2): a picture alone
    // is a thing to score.
    let png = common::png_data_uri(96, 96);
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png)],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn the_round_trip_check_is_skipped_with_media_and_enforced_without() {
    // With media the text spells one marker where the ids hold a whole pad
    // run, so the full-text re-encode cannot match by construction - and the
    // request still scores, because the seam check carries the proof instead.
    let png = common::png_data_uri(96, 96);
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png), {"type": "text", "text": "urgent?"}],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Without media, a marker typed into the input is text: the ids the
    // engine would read are not the re-encode of the text the caller is
    // told about, and the shared round-trip refusal still fires.
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input <|image_pad|>",
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("did not survive re-encoding")),
        "{body}"
    );
}

#[tokio::test]
async fn the_seam_check_still_refuses_a_glued_label_with_media() {
    // Review Focus 4: the seam proof moves to the tail after the last span,
    // it does not switch off. A template that ends the prompt on a word
    // swallows the label whatever the prompt began with.
    let png = common::png_data_uri(96, 96);
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, true);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [{"type": "text", "text": "urgent?"}, image_block(&png)],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("yes") && message.contains("one fresh token")),
        "the refusal must name the label and the proof: {body}"
    );

    // Two adjacent images: the proof runs against the tail after the second
    // marker, and the glued tail swallows the label just the same.
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png), image_block(&png)],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("one fresh token")),
        "{body}"
    );
}

#[tokio::test]
async fn an_over_budget_image_is_a_context_overflow() {
    // The context guard measures the expanded prompt, so a picture's pad run
    // is charged like any other text - and blamed on `input`.
    let png = common::png_data_uri(96, 96);
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], 12, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png)],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "context_length_exceeded", "{body}");
    assert_eq!(body["error"]["param"], "input", "{body}");
}

#[tokio::test]
async fn return_prompt_token_ids_with_media_returns_expanded_ids() {
    // Review Focus 5: the ids a caller gets back are the ids the model read,
    // pad runs included, so their count is the usage number.
    let png = common::png_data_uri(96, 96);
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [image_block(&png), {"type": "text", "text": "urgent?"}],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
                "return_prompt_token_ids": true,
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ids: Vec<u64> = answer_of(&body, "q1")["prompt_token_ids"]
        .as_array()
        .expect("the answer carries its prompt ids")
        .iter()
        .map(|id| id.as_u64().expect("an id is a number"))
        .collect();
    let usage = body["usage"]["prompt_tokens"].as_u64().expect("usage");
    assert_eq!(
        u64::try_from(ids.len()).expect("a prompt fits"),
        usage,
        "the returned ids are the usage number"
    );
    assert!(
        ids.windows(4).any(|window| window == [u64::from(PAD); 4]),
        "the pad run is in the returned ids: {ids:?}"
    );
}

#[tokio::test]
async fn a_remote_image_url_is_refused_without_the_flag() {
    // The renderer's policy refusal is now reachable from decisions: the
    // block parses, and the render is where the operator's flag is missed.
    let (state, _audit) = media_state(vec![row_with(0.5, &[])], CTX, false);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [
                    {"type": "text", "text": "urgent?"},
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}},
                ],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("--vision-allow-remote-images")),
        "the refusal must name the flag: {body}"
    );
}

#[tokio::test]
async fn a_valid_image_block_without_a_vision_tower_is_refused() {
    // The block is well-formed and the wire accepts it; the refusal is the
    // renderer's: a text-only deployment cannot score pixels and says so,
    // naming the missing tower rather than mangling the picture into text.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let png = common::png_data_uri(96, 96);
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": [
                    {"type": "text", "text": "urgent?"},
                    image_block(&png),
                ],
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("no vision tower")),
        "the refusal must name the missing tower: {body}"
    );
}
