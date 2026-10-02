//! `POST /v1/systemone` driven through the router: the same scoring engine as
//! `/v1/decisions` behind the official System One wire — map-form questions
//! keyed by id, `noul`/`choice`/`score`, relaxed schema at the edges (blank
//! ids, blank state, lists of one, unknown top-level fields ignored), strict
//! where an answer could go wrong, and schema violations answering 422 with a
//! location where the decisions route answers 400.
//!
//! The harness is `decisions.rs`'s, copied rather than shared: the model
//! behind the engine answers prefills with whole rows scripted as numbers, so
//! every probability assertion is arithmetic a reader can check against the
//! same row. Row-to-question pairing goes through the audit rather than a
//! fixed order — the audit push and the scripted-row pop happen in one prefill
//! call — so the assertions hold whatever order the scheduler ran the
//! concurrent prefills in.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_core::TokenId;
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::jinja_helpers::ToolJsonStyle;
use shiso_server::tokenizer::render_prompt::PromptRenderer;
use tower::ServiceExt as _;

/// The fake model's logits width: a scripted row must cover every logit the
/// scoring pass reads, or the tail of a previous row rides into the
/// log-sum-exp.
const VOCAB: usize = 64;

/// The context every state is built with.
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
    // The words of the prompt bodies the pairing below names. Each is one
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

/// A vocabulary of `WORDS` as one-token words, built the way `decisions.rs`
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
/// answer assertions pair each prompt with the row that answered it.
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
/// a test that only asserts on shapes wants.
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

/// The label log-probabilities of `labels` under `row`: each label's logit
/// minus the row's log-sum-exp, from the row's own numbers.
fn label_probs(row: &[f32], labels: &[&str]) -> Vec<f64> {
    let norm = logsumexp(row);
    labels
        .iter()
        .map(|label| f64::from(row[word(label)]) - norm)
        .collect()
}

/// The mass the labels carry together: `Σ exp(logprob)`.
fn mass(logprobs: &[f64]) -> f64 {
    logprobs.iter().map(|lp| lp.exp()).sum()
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

/// The choice confidence recomputed from the answer's own probabilities:
/// `clamp01((n·max − 1) / (n − 1))`, `1.0` for one label.
fn choice_confidence(probabilities: &[f64]) -> f64 {
    match probabilities.len() {
        0 => 0.0,
        1 => 1.0,
        n => {
            let top = probabilities
                .iter()
                .fold(f64::NEG_INFINITY, |best, p| best.max(*p));
            let width = f64::from(u32::try_from(n).expect("a small label count"));
            ((width * top - 1.0) / (width - 1.0)).clamp(0.0, 1.0)
        }
    }
}

/// The score confidence recomputed from the answer's own probabilities:
/// `max(0, 1 − spread / uniform_spread)`.
fn score_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let top = probabilities
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
            if *p > best.1 { (index, *p) } else { best }
        })
        .0;
    let spread: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| p * f64::from(u32::try_from(index.abs_diff(top)).expect("a small index")))
        .sum();
    let mid = f64::from(u32::try_from(n - 1).expect("a small count")) / 2.0;
    let uniform_spread: f64 = (0..n)
        .map(|index| (f64::from(u32::try_from(index).expect("a small index")) - mid).abs())
        .sum::<f64>()
        / f64::from(u32::try_from(n).expect("a small count"));
    (1.0 - spread / uniform_spread).max(0.0)
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

/// Asserts one number field of an answer against the value the scripted row
/// implies.
fn assert_close(got: &Value, want: f64, what: &str, body: &Value) {
    let got = got.as_f64().expect("the field carries a number");
    assert!(
        (got - want).abs() < 1e-5,
        "{what}: want {want}, got {got}: {body}"
    );
}

/// Asserts an answer's probabilities against the values the scripted row
/// implies, label by label.
fn assert_probabilities(answer: &Value, want: &[f64], body: &Value) {
    let got = probabilities(answer);
    assert_eq!(got.len(), want.len(), "{body}");
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "label {index}: {got} is not {want}: {body}"
        );
    }
}

/// The shape every 422 refusal shares on this surface when the body itself
/// did not parse: serde's own line and column name the spot in the bytes.
fn assert_schema_violation(status: StatusCode, body: &Value, names: &str) {
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "invalid_schema", "{body}");
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "422 is still a bad request in the client's eyes: {body}"
    );
    let at = body["error"]["at"]
        .as_str()
        .expect("a schema refusal carries a location");
    assert!(
        at.starts_with("line ") && at.contains(" column "),
        "the location is serde's line and column, got {at}: {body}"
    );
    if !names.is_empty() {
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(names)),
            "the refusal must name {names}: {body}"
        );
    }
}

/// The shape of a 422 that `validate` raises rather than the body parse: the
/// schema class, with a location naming the field path the caller must fix
/// (`questions`, or `questions.<id>`).
fn assert_schema_violation_at(status: StatusCode, body: &Value, at: &str, names: &str) {
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "invalid_schema", "{body}");
    assert_eq!(
        body["error"]["type"], "invalid_request_error",
        "422 is still a bad request in the client's eyes: {body}"
    );
    assert_eq!(body["error"]["at"], at, "{body}");
    if !names.is_empty() {
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(names)),
            "the refusal must name {names}: {body}"
        );
    }
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

/// One POST with raw bytes, so a test can send something that is not JSON.
fn post_raw(path: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_owned()))
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

/// The smallest body that answers: one noul question over one state.
fn valid_body() -> Value {
    json!({
        "state": "the input",
        "questions": {"q1": {"type": "noul", "instructions": "ready"}},
    })
}

// --- the scored answers -----------------------------------------------------

#[tokio::test]
// The brief pins this test's name and its scope: every answer kind and a
// noul question in one request, because "all four in one call" is the claim.
#[expect(
    clippy::too_many_lines,
    reason = "one request exercises every answer kind, and each kind's assertions travel with it"
)]
async fn a_full_request_answers_every_question_kind_and_noul() {
    // Four questions, four scripted rows, and every answer asserted against
    // the row that answered *its* prompt - the audit push and the row pop
    // share one prefill call, so index `i` of one belongs to index `i` of the
    // other whatever order the scheduler ran the prefills in.
    let rows = vec![
        row_with(0.0, &[(word("A"), 2.0), (word("B"), 4.0)]),
        row_with(0.0, &[(word("0"), 1.0), (word("1"), 2.0), (word("2"), 3.0)]),
        row_with(0.0, &[(word("yes"), 3.0)]),
        row_with(0.0, &[(word("yes"), 2.0)]),
    ];
    let (state, audit) = state_with_rows(rows.clone(), CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "model": "echo-me",
                "state": "the input",
                "questions": {
                    "q1": {"type": "choice", "instructions": "pick one",
                           "criteria": {"red": null, "blue": null}},
                    "s1": {"type": "score", "instructions": "how bad",
                           "criteria": ["none", "some", "lots"]},
                    "q2": {"type": "noul", "instructions": "ready",
                           "criteria": {"true": "confirmed", "false": "denied"}},
                    "n1": {"type": "noul", "instructions": "ready"},
                },
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["model"], "test-model",
        "the served model answers; the request's name is never echoed"
    );
    assert_eq!(body["state"], "the input", "the state is echoed");
    let keys: Vec<&str> = body["answers"]
        .as_object()
        .expect("answers is an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        ["q1", "s1", "q2", "n1"],
        "the answers come back in the request map's insertion order"
    );

    let prompts = audit.prefill_prompts.lock();
    assert_eq!(prompts.len(), 4, "one prefill per question and criterion");
    let logprobs_answering = |text: &str, labels: &[&str]| -> Vec<f64> {
        let want = prompt_ids(text);
        let at = prompts
            .iter()
            .position(|prompt| *prompt == want)
            .expect("the question's prompt was read");
        label_probs(&rows[at], labels)
    };

    let q1 = answer_of(&body, "q1");
    assert_eq!(q1["type"], "choice");
    assert_eq!(q1["choice"], "blue", "the top label names its option");
    assert!(
        q1.get("answer").is_none() && q1.get("score").is_none(),
        "{body}"
    );
    let lp = logprobs_answering(CHOICE_BODY, &["A", "B"]);
    let want = softmax(&lp, 1.0);
    assert_probabilities(&q1, &want, &body);
    assert_close(
        &q1["confidence"],
        choice_confidence(&want),
        "q1 confidence",
        &body,
    );
    assert_close(&q1["x_label_mass"], mass(&lp), "q1 x_label_mass", &body);
    assert!(
        q1.get("label_mass").is_none(),
        "this wire renames the mass to x_label_mass: {body}"
    );

    let s1 = answer_of(&body, "s1");
    assert_eq!(s1["type"], "score");
    assert_eq!(
        s1["legend"],
        json!(["none", "some", "lots"]),
        "the score answer echoes its legend: {body}"
    );
    let lp = logprobs_answering(SCORE_BODY, &["0", "1", "2"]);
    let want = softmax(&lp, 1.0);
    assert_probabilities(&s1, &want, &body);
    let score: f64 = want
        .iter()
        .enumerate()
        .map(|(index, p)| f64::from(u32::try_from(index).expect("a small index")) * p)
        .sum();
    assert_close(&s1["score"], score, "s1 score", &body);
    assert_close(
        &s1["confidence"],
        score_confidence(&want),
        "s1 confidence",
        &body,
    );
    assert_close(&s1["x_label_mass"], mass(&lp), "s1 x_label_mass", &body);

    let q2 = answer_of(&body, "q2");
    assert_eq!(q2["type"], "yes_no");
    let lp = logprobs_answering(YES_NO_BODY, &["yes", "no"]);
    let want = softmax(&lp, 1.0);
    // The official wire has no label-word answer for a binary question: a
    // `noul` answers with the yes probability, so the answer is the number
    // the old noul criterion carried, not the word "yes".
    assert_close(&q2["answer"], want[0], "q2 answer", &body);
    assert_probabilities(&q2, &want, &body);
    assert_close(
        &q2["confidence"],
        choice_confidence(&want),
        "q2 confidence",
        &body,
    );
    assert_close(&q2["x_label_mass"], mass(&lp), "q2 x_label_mass", &body);

    // The design: "the `noul` answer is the yes probability" - the `answer`
    // field itself is that number (entry zero of the probabilities, the
    // labels being `yes`, `no` in that order). q2 above is a noul too; its
    // `criteria` wording changes the prompt, not the answer shape.
    let n1 = answer_of(&body, "n1");
    assert_eq!(
        n1["type"], "yes_no",
        "a noul question answers as a yes/no question"
    );
    let lp = logprobs_answering(NOUL_BODY, &["yes", "no"]);
    let want = softmax(&lp, 1.0);
    assert_close(&n1["answer"], want[0], "n1 answer", &body);
    assert_probabilities(&n1, &want, &body);
    assert!(want[0] > want[1], "the scripted row favours yes");
    assert_close(
        &n1["confidence"],
        choice_confidence(&want),
        "n1 confidence",
        &body,
    );
    assert_close(&n1["x_label_mass"], mass(&lp), "n1 x_label_mass", &body);

    let prompt_tokens: usize = prompts.iter().map(Vec::len).sum();
    assert_eq!(
        body["usage"]["input_tokens"],
        json!(u64::try_from(prompt_tokens).expect("a small count")),
        "the accounting sums every question's prompt: {body}"
    );
    assert_eq!(body["usage"]["output_tokens"], 0, "nothing generated");
    assert_eq!(
        body["usage"]["total_tokens"], body["usage"]["input_tokens"],
        "{body}"
    );
}

// --- the schema's shape -----------------------------------------------------

#[tokio::test]
async fn unknown_top_level_fields_are_ignored_and_nested_ones_refused_422() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);

    // A System One client stamps its whole document into the call; a key at
    // the top level changes no scored byte, so it costs nothing.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"q1": {"type": "noul", "instructions": "ready"}},
                "extra": 1,
                "client": {"stamp": true},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    // Inside a question the same key would be one the caller believes is
    // scored, so it is refused - with the question's key as the location.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"q1": {"type": "noul", "instructions": "ready",
                                     "bogus": 1}},
            }),
        ),
    )
    .await;
    assert_schema_violation_at(status, &body, "questions.q1", "bogus");

    // And inside the noul `criteria` object, which is its own strict shape.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"q1": {"type": "noul", "instructions": "ready",
                                     "criteria": {"true": "confirmed", "bogus": 2}}},
            }),
        ),
    )
    .await;
    assert_schema_violation_at(status, &body, "questions.q1", "bogus");

    // The same nested key on the decisions route is a 400: the schema shape is
    // the one axis the two routes disagree on.
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [{"id": "q1", "type": "yes_no", "question": "ready",
                               "bogus": 1}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["code"], "invalid_request", "{body}");
    assert!(body["error"].get("at").is_none(), "{body}");
}

#[tokio::test]
async fn decisions_only_fields_are_refused_when_not_null() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[]); 4], CTX);
    let knobs = [
        ("temperature", json!(1.0)),
        ("prompt_format_version", json!(1)),
        ("return_prompt_token_ids", json!(true)),
    ];
    for (field, value) in knobs {
        // Stamped as null, the knob costs nothing...
        let mut body = valid_body();
        body[field] = json!(null);
        let (status, replied) = call(&state, post("/v1/systemone", &body)).await;
        assert_eq!(status, StatusCode::OK, "{field}=null -> {replied}");
        // ...and with any other value it asks this route to be the other one.
        let mut body = valid_body();
        body[field] = value.clone();
        let (status, replied) = call(&state, post("/v1/systemone", &body)).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{field}={value} -> {replied}"
        );
        assert_eq!(replied["error"]["code"], "invalid_request", "{replied}");
        assert_eq!(replied["error"]["param"], field, "{replied}");
    }
    // Any other value means *any* value: the field's schema type is "any JSON",
    // its rule is "absent or null", so even a nonsense type is the 400 refusal
    // rather than a 422 schema violation.
    let mut body = valid_body();
    body["temperature"] = json!("warm");
    let (status, replied) = call(&state, post("/v1/systemone", &body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{replied}");
    assert_eq!(replied["error"]["param"], "temperature", "{replied}");
}

#[tokio::test]
// The brief pins this test's name and its scope: every relaxation the schema
// states, proven answerable against one server rather than six.
async fn relaxed_bounds_and_blanks() {
    // A one-option choice answers: its single label is as confident as an
    // answer gets, and the decisions route's minimum of two is not asked here.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"q1": {"type": "choice", "instructions": "pick one",
                                     "criteria": {"only one": null}}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"]["q1"]["choice"], "only one");
    assert_close(
        &body["answers"]["q1"]["confidence"],
        1.0,
        "a one-label choice is fully confident",
        &body,
    );

    // A one-level score answers, at its only level.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"s1": {"type": "score", "instructions": "how bad",
                                     "criteria": ["none"]}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"]["s1"]["legend"], json!(["none"]));
    assert_close(
        &body["answers"]["s1"]["score"],
        0.0,
        "the only level",
        &body,
    );
    assert_close(
        &body["answers"]["s1"]["confidence"],
        1.0,
        "a one-level score is fully confident",
        &body,
    );

    // An empty state answers and echoes back blank, and a state that is
    // simply absent is the same answer.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[]); 2], CTX);
    let mut body = valid_body();
    body["state"] = json!("");
    let (status, replied) = call(&state, post("/v1/systemone", &body)).await;
    assert_eq!(status, StatusCode::OK, "{replied}");
    assert_eq!(replied["state"], "", "{replied}");
    let (status, replied) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({"questions": {"q1": {"type": "noul", "instructions": "ready"}}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{replied}");
    assert_eq!(replied["state"], "", "{replied}");

    // A blank map key is a blank id and answers under ""; the map wire
    // cannot repeat one - JSON keys are unique - so that is the whole
    // blank-id story this surface has.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, replied) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"": {"type": "noul", "instructions": "ready"}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{replied}");
    assert_eq!(replied["answers"][""]["type"], "yes_no", "{replied}");

    // A blank question text answers: its line drops from the version-1 block,
    // which is why the wording builder carries the blank branches at all.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[]); 2], CTX);
    let (status, replied) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {
                    "q1": {"type": "choice", "instructions": "",
                           "criteria": {"red": null, "blue": null}},
                    "q2": {"type": "noul", "instructions": "   "},
                },
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{replied}");
}

#[tokio::test]
async fn strict_bounds_still_refuse() {
    // The relaxation is about blanks and list minima, not about answers that
    // could not be read: the alphabet's twenty-six, the digit's ten, unique
    // names, and the served model are refused on this route exactly as on
    // the decisions one - and as 400s, never 422s. A field a question's type
    // has no place for is no longer a bound but the schema's own refusal
    // (the unknown-key test above pins it as a 422), and the map wire
    // cannot deliver a duplicate id: its keys are unique by construction.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[]); 2], CTX);
    let wide: Value = (0..27)
        .map(|index| (format!("opt {index}"), Value::Null))
        .collect();
    let tall: Vec<Value> = (0..11)
        .map(|index| json!(format!("level {index}")))
        .collect();
    let cases: Vec<(Value, &str)> = vec![
        (
            json!({"state": "the input", "questions": {"q1": {"type": "choice",
                 "instructions": "pick", "criteria": wide}}}),
            "26",
        ),
        (
            json!({"state": "the input", "questions": {"q1": {"type": "score",
                 "instructions": "how bad", "criteria": tall}}}),
            "10",
        ),
        (
            json!({"state": "the input", "questions": {"q1": {"type": "choice",
                 "instructions": "pick", "criteria": {"Red": null, "red": null}}}}),
            "used more than once",
        ),
        (
            json!({"model": "base:lora", "state": "the input", "questions":
                 {"q1": {"type": "noul", "instructions": "ready"}}}),
            "base:lora",
        ),
    ];
    for (case, want) in cases {
        let (status, body) = call(&state, post("/v1/systemone", &case)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case} -> {body}");
        assert_eq!(body["error"]["code"], "invalid_request", "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains(want)),
            "the refusal must name {want}: {body}"
        );
    }

    // A blank map key answers under "", and a blank `instructions` answers
    // with the bare lead line - and the flat row ties, so the yes
    // probability the noul answer carries is exactly one half.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"": {"type": "noul", "instructions": "ready"}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"][""]["type"], "yes_no", "{body}");
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"n1": {"type": "noul", "instructions": ""}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_close(
        &body["answers"]["n1"]["answer"],
        0.5,
        "a tied row answers the noul question at a half",
        &body,
    );
}

/// The key-sorted compact rendering of `{"b": 1, "a": [1, 2]}`: the bytes
/// the shared render rule owes a container state, pinned here in the order
/// the server - not the client - chooses.
const CONTAINER_TEXT: &str = "{\"a\":[1,2],\"b\":1}";

#[tokio::test]
async fn a_container_state_is_scored_from_its_key_sorted_rendering() {
    // The decisions route pins this at its `input`; the systemone `state`
    // rides the same render rule, end to end: the audited prompt ids are
    // exactly those of the key-sorted compact bytes in the shared
    // template, and the echoed state is that same text.
    let (state, audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": {"b": 1, "a": [1, 2]},
                "questions": {"q1": {"type": "noul", "instructions": "ready"}},
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
    assert_eq!(
        body["state"], CONTAINER_TEXT,
        "the echoed state is the rendered text: {body}"
    );
}

#[tokio::test]
async fn malformed_json_is_422_not_400() {
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);

    // Yesterday's array spelling is a schema violation naming `questions`:
    // the cutover removed the dialect, and a caller posting the old body
    // gets the document-fix class with the field to fix, not a 500 or a
    // silently-empty answer.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": [{"id": "q1", "type": "noul", "instructions": "ready"}],
            }),
        ),
    )
    .await;
    assert_schema_violation_at(status, &body, "questions", "questions");

    // A known field spelled with the wrong type inside a question is a
    // schema violation here, located at the question's key, where the
    // decisions route's typed parse answers its own spelling of the same
    // mistake 400.
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {"q1": {"type": 5, "instructions": "ready"}},
            }),
        ),
    )
    .await;
    assert_schema_violation_at(status, &body, "questions.q1", "type");
    let (status, body) = call(&state, post_raw("/v1/systemone", r#"{"questions": ["#)).await;
    assert_schema_violation(status, &body, "");

    // The two routes disagree on this axis alone: bytes this one calls a
    // schema violation, the decisions route calls a 400.
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "questions": [{"id": "q1", "type": 5, "question": "ready"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = call(&state, post_raw("/v1/decisions", r#"{"questions": ["#)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn a_content_block_state_is_refused_422() {
    // The systemone `state` is text-only: a content-block array - even a
    // well-formed one, the refusal is by shape (Review Focus 6) - is refused
    // with the surface's schema class naming `state`, rather than
    // compact-JSON'ing a base64 blob into a scored prompt (spec ruling 6).
    // The assertion is spelled out rather than through
    // `assert_schema_violation` because that helper pins serde's
    // `line L column C` location, and this refusal names the field.
    let (state, _audit) = state_with_rows(vec![row_with(0.5, &[])], CTX);
    for block in [
        json!({"type": "text", "text": "a"}),
        json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}),
    ] {
        let (status, body) = call(
            &state,
            post(
                "/v1/systemone",
                &json!({
                    "state": [block],
                    "questions": {"q1": {"type": "noul", "instructions": "ready"}},
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
        assert_eq!(body["error"]["code"], "invalid_schema", "{body}");
        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "422 is still a bad request in the client's eyes: {body}"
        );
        assert_eq!(body["error"]["at"], "state", "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("state")),
            "the refusal must name state: {body}"
        );
    }
}

// --- prompt pairing ----------------------------------------------------------

/// The choice question's exact version-1 body, as the shared builder renders
/// it; `decisions.rs` pins these bytes, this file uses them to find which
/// scripted row answered which prompt.
const CHOICE_BODY: &str =
    "the input\n\nQuestion: pick one\nA: red\nB: blue\nAnswer with the letter of one option only.";

/// The score question's exact version-1 body.
const SCORE_BODY: &str = "the input\n\nQuestion: how bad\n0: none\n1: some\n2: lots\nAnswer with the number of one level only.";

/// The yes/no question's exact version-1 body.
const YES_NO_BODY: &str = "the input\n\nIs the following true? ready\nyes: confirmed\nno: denied\nAnswer with yes or no only.";

/// The noul question's exact version-1 body: the yes/no block without the
/// label-wording lines.
const NOUL_BODY: &str = "the input\n\nIs the following true? ready\nAnswer with yes or no only.";

/// The ids of one expected prompt, encoded through the test vocabulary.
fn prompt_ids(body: &str) -> Vec<u32> {
    test_vocabulary()
        .encode(&rendered_prompt(body), false)
        .expect("the prompt encodes")
        .into_iter()
        .map(TokenId::get)
        .collect()
}
