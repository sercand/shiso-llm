//! The decision routes over a checkpoint that answers from its answer-slot
//! rows through a family readout rather than a head.
//!
//! The readout here is a fake with the shape a real one has: the state as a
//! shared trunk, each question's wording as its branch, and an option read
//! from a group of tokens - a yes is `yes` or `A` - so the max-pooling is
//! visible in the numbers. The engine answers each prefill with a scripted
//! row, and the audit pairs each prompt with the row that answered it, so
//! every probability is arithmetic over the row a reader can check.

mod common;

use std::str::FromStr as _;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_core::{
    DecisionLayoutError, DecisionReadout, PromptEncoder, ReadoutPrompt, ReadoutQuestion,
    ReadoutRecord, SchemaAnswer, StateSpelling, TokenId,
};
use shiso_server::router::build_router;
use shiso_server::state::AppState;
use shiso_server::tokenizer::Vocabulary;
use tower::ServiceExt as _;

/// The fake model's logits width.
const VOCAB: usize = 64;

/// The context every state is built with.
const CTX: u32 = 256;

/// The words the vocabulary defines, in id order.
const WORDS: &[&str] = &[
    "<unk>", "A", "B", "C", "0", "1", "2", "yes", "no", "the", "input", "ready", "pick", "level",
];

fn word(text: &str) -> u32 {
    u32::try_from(
        WORDS
            .iter()
            .position(|candidate| *candidate == text)
            .expect("the word is in the test vocabulary"),
    )
    .expect("a small id")
}

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

/// The fake readout. It records the state text each decision was laid out
/// over, so a test can see how the route spelled it.
#[derive(Debug, Default)]
struct FakeReadout {
    states: Mutex<Vec<Option<String>>>,
}

fn ids(text: &str, encoder: &dyn PromptEncoder) -> Result<Vec<TokenId>, DecisionLayoutError> {
    encoder
        .encode(text, &[])
        .map_err(DecisionLayoutError::Encode)
}

impl DecisionReadout for FakeReadout {
    fn state_spelling(&self) -> StateSpelling {
        StateSpelling::Indented
    }

    fn encode(
        &self,
        encoder: &dyn PromptEncoder,
        record: &ReadoutRecord<'_>,
    ) -> Result<ReadoutPrompt, DecisionLayoutError> {
        self.states
            .lock()
            .expect("the lock")
            .push(record.state.map(str::to_owned));
        let trunk = match record.state {
            Some(state) if state.starts_with('{') => ids("level", encoder)?,
            Some(state) => ids(state, encoder)?,
            None => Vec::new(),
        };
        let one = |text: &str| TokenId::new(word(text));
        let questions = record
            .questions
            .iter()
            .map(|question| {
                let options = match &question.answer {
                    // `A` is a second spelling of yes: the answer is the
                    // better of the two.
                    SchemaAnswer::YesNo { .. } => vec![vec![one("yes"), one("A")], vec![one("no")]],
                    SchemaAnswer::Choice(options) => ["A", "B", "C"]
                        .iter()
                        .take(options.len())
                        .map(|code| vec![one(code)])
                        .collect(),
                    SchemaAnswer::Score(levels) => ["0", "1", "2"]
                        .iter()
                        .take(levels.len())
                        .map(|digit| vec![one(digit)])
                        .collect(),
                };
                Ok(ReadoutQuestion {
                    branch: ids(&question.wording, encoder)?,
                    options,
                })
            })
            .collect::<Result<_, DecisionLayoutError>>()?;
        Ok(ReadoutPrompt { trunk, questions })
    }
}

fn state_with_rows(rows: Vec<Vec<f32>>) -> (Arc<AppState>, Arc<common::Audit>, Arc<FakeReadout>) {
    let vocabulary = test_vocabulary();
    let (engine, _join, audit) = common::fake_engine_rows(rows, Some(Arc::clone(&vocabulary)));
    let readout = Arc::new(FakeReadout::default());
    let mut state = AppState::new(engine, "test-model".to_owned(), CTX).with_vocabulary(vocabulary);
    let layout: Arc<dyn DecisionReadout> = readout.clone();
    state.decision_readout = Some(layout);
    (Arc::new(state), audit, readout)
}

fn row_with(slots: &[(&str, f32)]) -> Vec<f32> {
    let mut row = vec![0.0; VOCAB];
    for (slot, value) in slots {
        row[word(slot) as usize] = *value;
    }
    row
}

fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |best, v| best.max(*v));
    f64::from(max)
        + row
            .iter()
            .map(|v| (f64::from(*v) - f64::from(max)).exp())
            .sum::<f64>()
            .ln()
}

fn logprob(row: &[f32], label: &str) -> f64 {
    f64::from(row[word(label) as usize]) - logsumexp(row)
}

fn softmax(values: &[f64], temperature: f64) -> Vec<f64> {
    let scaled: Vec<f64> = values.iter().map(|v| v / temperature).collect();
    let max = scaled.iter().fold(f64::NEG_INFINITY, |b, v| b.max(*v));
    let exps: Vec<f64> = scaled.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

fn post(path: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("a POST request builds")
}

async fn call(state: &Arc<AppState>, request: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(Arc::clone(state))
        .oneshot(request)
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn close(got: &Value, want: f64, body: &Value) {
    let got = got.as_f64().expect("a number");
    assert!((got - want).abs() < 1e-6, "want {want}, got {got}: {body}");
}

/// The row that answered `prompt`, found through the audit.
fn row_for<'r>(audit: &common::Audit, rows: &'r [Vec<f32>], prompt: &[&str]) -> &'r [f32] {
    let want: Vec<u32> = prompt.iter().map(|text| word(text)).collect();
    let prompts = audit.prefill_prompts.lock();
    let at = prompts
        .iter()
        .position(|read| *read == want)
        .expect("the question's prompt was read whole: the trunk, then its branch");
    &rows[at]
}

#[tokio::test]
async fn each_question_reads_its_own_prompt_and_pools_its_spellings() {
    let rows = vec![
        // Whichever question reads first gets this row, the other the next:
        // the pairing below goes through the audit, not the order.
        row_with(&[("yes", 1.0), ("A", 3.0), ("no", 2.0), ("B", 0.5)]),
        row_with(&[("yes", 0.5), ("A", 1.0), ("no", 2.5), ("B", 2.0)]),
    ];
    let (state, audit, readout) = state_with_rows(rows.clone());
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": "the input",
                "questions": {
                    "n1": {"type": "noul", "instructions": "ready"},
                    "c1": {"type": "choice", "instructions": "pick",
                           "criteria": {"red": null, "blue": null}},
                },
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        *readout.states.lock().expect("the lock"),
        vec![Some("the input".to_owned())]
    );

    // The noul: yes is the better of `yes` and `A`, no is `no`.
    let row = row_for(&audit, &rows, &["the", "input", "ready"]);
    let yes = logprob(row, "yes").max(logprob(row, "A"));
    let want = softmax(&[yes, logprob(row, "no")], 1.0);
    close(&body["answers"]["n1"]["noul"], want[0], &body);

    // The choice: one code each.
    let row = row_for(&audit, &rows, &["the", "input", "pick"]);
    let want = softmax(&[logprob(row, "A"), logprob(row, "B")], 1.0);
    close(
        &body["answers"]["c1"]["probabilities"]["red"],
        want[0],
        &body,
    );
    close(
        &body["answers"]["c1"]["probabilities"]["blue"],
        want[1],
        &body,
    );

    // Usage counts the trunk once and each branch once.
    assert_eq!(body["usage"]["input_tokens"], 4, "{body}");
}

#[tokio::test]
async fn a_structured_state_reaches_the_layout_in_the_familys_spelling() {
    let rows = vec![row_with(&[("yes", 1.0)])];
    let (state, _audit, readout) = state_with_rows(rows);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({
                "state": {"b": 1, "a": "é"},
                "questions": {"n1": {"type": "noul", "instructions": "ready"}},
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        *readout.states.lock().expect("the lock"),
        vec![Some("{\n  \"b\": 1,\n  \"a\": \"é\"\n}".to_owned())],
        "indented, in the caller's key order, non-ASCII as itself"
    );
}

#[tokio::test]
async fn the_decisions_route_reads_the_same_rows_at_the_callers_temperature() {
    let rows = vec![row_with(&[("0", 1.0), ("1", 3.0), ("2", 2.0)])];
    let (state, audit, _readout) = state_with_rows(rows.clone());
    let (status, body) = call(
        &state,
        post(
            "/v1/decisions",
            &json!({
                "input": "the input",
                "temperature": 2.0,
                "questions": [{"name": "s", "type": "score", "instructions": "level",
                               "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}]}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let row = row_for(&audit, &rows, &["the", "input", "level"]);
    let want = softmax(
        &[logprob(row, "0"), logprob(row, "1"), logprob(row, "2")],
        2.0,
    );
    let got: Vec<f64> = body["answers"][0]["probabilities"]
        .as_array()
        .map(|values| {
            values
                .iter()
                .filter_map(|entry| entry["probability"].as_f64())
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(got.len(), 3, "{body}");
    for (got, want) in got.iter().zip(&want) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}: {body}");
    }
}

#[tokio::test]
async fn a_prompt_past_the_context_is_refused_before_it_is_read() {
    #[derive(Debug)]
    struct TooLong;
    impl DecisionReadout for TooLong {
        fn state_spelling(&self) -> StateSpelling {
            StateSpelling::CompactSorted
        }
        fn encode(
            &self,
            _encoder: &dyn PromptEncoder,
            record: &ReadoutRecord<'_>,
        ) -> Result<ReadoutPrompt, DecisionLayoutError> {
            Err(DecisionLayoutError::PromptTooLong {
                tokens: record.budget + 1,
                budget: record.budget,
            })
        }
    }
    let vocabulary = test_vocabulary();
    let (engine, _join, audit) =
        common::fake_engine_rows(Vec::new(), Some(Arc::clone(&vocabulary)));
    let mut state = AppState::new(engine, "test-model".to_owned(), CTX).with_vocabulary(vocabulary);
    state.decision_readout = Some(Arc::new(TooLong));
    let state = Arc::new(state);
    let (status, body) = call(
        &state,
        post(
            "/v1/systemone",
            &json!({"state": "the input", "questions": {"n": {"type": "noul", "instructions": "ready"}}}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(audit.prefill_prompts.lock().is_empty(), "nothing was read");
}
