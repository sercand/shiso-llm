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
