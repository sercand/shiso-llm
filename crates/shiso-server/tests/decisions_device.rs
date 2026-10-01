//! The decision API against a real checkpoint and a real device.
//!
//! `decisions.rs` and `systemone.rs` script the logits rows, so they pin the
//! arithmetic but nothing about the tokenizer: a fake vocabulary makes every
//! label one token by construction. This is the other half. Against a real
//! export it settles what the fakes cannot see: that the version-1 prompt
//! survives a real tokenizer's round trip, that the letters, digits and
//! `yes`/`no` are each one fresh token after a real chat template's opener,
//! and that the loaded model actually reads the ticket - the choice question
//! answers the obviously-correct option and the billing question answers no.
//!
//! One boot, one ticket, both routes: the answers come from prefills only,
//! so the expensive part (the checkpoint) is shared and the cheap part (eight
//! scored prompts) is the claim. The router is called in process, the way the
//! per-surface tests call it; a socket would test hyper again, which `e2e.rs`
//! already does.
//!
//! Run it with an export named:
//!
//! ```text
//! SHISO_TEST_MODEL=/path/to/Qwen3.5-4B \
//!   cargo test -p shiso-server --features metal --test decisions_device -- --ignored --nocapture
//! ```

#![cfg(feature = "metal")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use shiso_core::Backend;
use shiso_server::cli::{Cli, Command, ServeArgs};
use shiso_server::router::build_router;
use shiso_server::scheduler::{SchedulerConfig, spawn_engine_with_checkpoints};
use shiso_server::state::AppState;
use tower::ServiceExt as _;

/// Prints the test's own report; a skipped case says why and returns.
#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

/// The checkpoint and config to boot, when this machine has an export.
///
/// `SHISO_TEST_MODEL` names either a checkpoint file - then the config comes
/// from `SHISO_TEST_MODEL_OPTIQ`, exactly as `e2e.rs` reads the pair - or an
/// export directory, which is its own checkpoint and carries its own
/// `config.json`; a root that holds several exports resolves to their `mlx`
/// directory, the way the runbooks point at one.
fn export() -> Option<(PathBuf, PathBuf)> {
    let root = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    if root.is_file() {
        let dir = PathBuf::from(std::env::var("SHISO_TEST_MODEL_OPTIQ").ok()?);
        let config = dir.join("config.json");
        return config.is_file().then_some((root, config));
    }
    for candidate in [&root, &root.join("mlx")] {
        let config = candidate.join("config.json");
        if candidate.is_dir() && config.is_file() {
            return Some((candidate.clone(), config));
        }
    }
    None
}

/// The command line this test would have been given.
fn args(model: &Path, config: &Path) -> ServeArgs {
    let (model, config) = (model.display().to_string(), config.display().to_string());
    let line = [
        "shiso", "serve", "--model", &model, "--config", &config, "--ctx", "2048",
    ];
    let cli = <Cli as clap::Parser>::try_parse_from(line).expect("the command line parses");
    let Command::Serve(serve) = cli.command else {
        panic!("expected Command::Serve");
    };
    *serve
}

/// A booted server: the state a router holds, and the engine thread under it.
struct Stack {
    /// What every request is routed against.
    state: Arc<AppState>,
    /// The engine thread, joined once every queue reference is gone.
    engine: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
}

impl Stack {
    /// Loads the checkpoint and starts the coordinator, the way the binary
    /// boots it - the same sequence `e2e.rs` walks, stopped short of a socket.
    fn boot(model: &Path, config: &Path) -> Self {
        let args = args(model, config);
        args.validate().expect("the command line is servable");
        let backend: Arc<dyn Backend> = shiso_metal::MetalBackend::new().expect("a Metal device");
        let parsed = shiso_core::HfConfig::from_file(&args.config_path()).expect("the config");
        let loaded = shiso_server::model_factory::load_model(
            &backend,
            args.model.as_deref().expect("this test passes --model"),
            &parsed,
            &args,
        )
        .expect("the checkpoint loads");
        let model = loaded.model;
        let defaults = loaded.defaults;
        let vocabulary = Arc::new(
            shiso_server::tokenizer::Vocabulary::from_file(&args.tokenizer_path())
                .expect("the vocabulary loads"),
        );
        let template = std::fs::read_to_string(args.chat_template_path())
            .expect("the checkpoint ships a chat template");
        let renderer = shiso_server::tokenizer::render_prompt::PromptRenderer::new(
            &template,
            shiso_server::tokenizer::jinja_helpers::ToolJsonStyle::Compact,
            Arc::clone(&vocabulary),
        )
        .expect("the chat template compiles")
        .with_thinking_default(defaults.thinking_default);

        let model_id = args.model_id();
        let (engine, join) = spawn_engine_with_checkpoints(
            backend,
            model,
            shiso_server::media::MediaTowers::default(),
            Some(Arc::clone(&vocabulary)),
            &model_id,
            args.ctx,
            SchedulerConfig::new(),
            (defaults.reasoning, defaults.tools),
        );
        let state = Arc::new(
            AppState::new(engine, model_id, args.ctx)
                .with_vocabulary(vocabulary)
                .with_renderer(Arc::new(renderer)),
        );
        Self {
            state,
            engine: join,
        }
    }

    /// Releases the queue and joins the engine thread.
    fn shutdown(self) {
        drop(self.state);
        self.engine
            .join()
            .expect("the engine thread does not panic")
            .expect("the model tears down cleanly");
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

/// Calls one route in process and reads the whole answer as JSON.
async fn post_json(state: &Arc<AppState>, path: &str, body: &Value) -> Value {
    let response = build_router(Arc::clone(state))
        .oneshot(post(path, body))
        .await
        .expect("the router answers");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    let body: Value = serde_json::from_slice(&bytes).expect("the answer is JSON");
    assert_eq!(status, StatusCode::OK, "{path} answered {body}");
    body
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

/// The shape every answer must have whatever the model thought: the
/// probabilities are a distribution over the labels and the label mass -
/// `mass_key` is each wire's own name for it - is a real probability. The
/// values themselves are the model's business; these bounds are arithmetic,
/// not an opinion about the ticket.
fn check_shape(answer: &Value, mass_key: &str, label: &str) {
    let probs = probabilities(answer);
    assert!(!probs.is_empty(), "{label}: no probabilities in {answer}");
    for (index, p) in probs.iter().enumerate() {
        assert!(
            *p >= 0.0 && *p <= 1.0,
            "{label}: probability {index} is {p}: {answer}"
        );
    }
    let sum: f64 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "{label}: probabilities sum to {sum}: {answer}"
    );
    let mass = answer[mass_key]
        .as_f64()
        .unwrap_or_else(|| panic!("{label}: no {mass_key} in {answer}"));
    assert!(
        mass > 0.0 && mass <= 1.0,
        "{label}: {mass_key} {mass} is not a probability: {answer}"
    );
}

/// The expected level of a score answer must sit inside the scale it was
/// given - the arithmetic bound, not an assertion about the ticket.
fn check_score_range(answer: &Value, label: &str) {
    let levels = probabilities(answer).len();
    let top = u32::try_from(levels - 1).expect("levels are single digits");
    let score = answer["score"]
        .as_f64()
        .unwrap_or_else(|| panic!("{label}: no score in {answer}"));
    assert!(
        score >= 0.0 && score <= f64::from(top),
        "{label}: score {score} is off a scale of {levels} levels: {answer}"
    );
}

/// The ticket both routes are asked about, and the questions asked of it.
/// The correct answers here are the obvious ones on purpose: this is a smoke
/// case against a real model, so it claims only what no reasonable reading
/// of the ticket could dispute.
const TICKET: &str = "The login page shows a spinner forever after entering the correct password.";

fn choice_question() -> Value {
    json!({
        "id": "kind", "type": "choice", "question": "What kind of report is this?",
        "options": ["a bug", "a feature request", "a question about billing"],
    })
}

fn score_question() -> Value {
    json!({
        "id": "urgency", "type": "score", "question": "How urgent is this report?",
        "levels": ["not urgent", "moderately urgent", "urgent"],
    })
}

/// The decisions surface: one prefill per question, every kind answered.
async fn decisions(state: &Arc<AppState>) {
    let body = post_json(
        state,
        "/v1/decisions",
        &json!({
            "input": TICKET,
            "questions": [choice_question(), score_question(),
                          {"id": "billing", "type": "yes_no",
                           "question": "Is this about billing?"}],
        }),
    )
    .await;
    assert_eq!(body["object"], "decisions", "{body}");
    assert_eq!(body["prompt_format_version"], 1, "{body}");

    let kind = &body["answers"]["kind"];
    assert_eq!(kind["type"], "choice", "{body}");
    assert_eq!(
        kind["choice"], "a bug",
        "the obvious reading of the ticket wins the argmax: {body}"
    );
    check_shape(kind, "label_mass", "decisions/kind");

    let urgency = &body["answers"]["urgency"];
    assert_eq!(urgency["type"], "score", "{body}");
    check_score_range(urgency, "decisions/urgency");
    check_shape(urgency, "label_mass", "decisions/urgency");

    let billing = &body["answers"]["billing"];
    assert_eq!(billing["type"], "yes_no", "{body}");
    assert_eq!(
        billing["answer"], "no",
        "the ticket is not about billing: {body}"
    );
    check_shape(billing, "label_mass", "decisions/billing");

    let prompt_tokens = body["usage"]["prompt_tokens"]
        .as_u64()
        .expect("the accounting counts the prompts");
    assert!(prompt_tokens > 0, "three prefills counted nothing: {body}");
    assert_eq!(
        body["usage"]["completion_tokens"], 0,
        "nothing was generated on this surface: {body}"
    );
    say("ok: /v1/decisions answered the ticket from a real model");
}

/// The systemone surface: the same ticket, the relaxed wire, confidences.
async fn systemone(state: &Arc<AppState>) {
    let body = post_json(
        state,
        "/v1/systemone",
        &json!({
            "state": TICKET,
            "questions": [choice_question(), score_question()],
            "noul": {"criteria": [{"id": "billing",
                                   "question": "Is this about billing?"}]},
        }),
    )
    .await;
    assert_eq!(body["state"], TICKET, "the state is echoed: {body}");

    let kind = &body["answers"]["kind"];
    assert_eq!(kind["choice"], "a bug", "{body}");
    let confidence = kind["confidence"]
        .as_f64()
        .expect("a choice answer carries its confidence");
    assert!(
        confidence > 0.5,
        "a three-option answer this tight read as a near-tie: {body}"
    );
    check_shape(kind, "x_label_mass", "systemone/kind");

    let urgency = &body["answers"]["urgency"];
    assert!(
        urgency["legend"].is_array(),
        "a score answer echoes its legend: {body}"
    );
    check_score_range(urgency, "systemone/urgency");
    check_shape(urgency, "x_label_mass", "systemone/urgency");

    let billing = &body["answers"]["billing"];
    // The noul answer is the yes probability itself (the design), so the
    // number, not a label word, carries "not about billing": below a half,
    // equal to the first probability, with `no` above `yes` in the row.
    let answer = billing["answer"]
        .as_f64()
        .expect("a noul answer is the yes probability, a number");
    assert!(
        answer < 0.5,
        "the noul criterion answered yes ({answer}) over a ticket that is \
         not about billing: {body}"
    );
    let probs = probabilities(billing);
    assert_eq!(probs.len(), 2, "a binary criterion: {body}");
    let (yes, no) = (probs[0], probs[1]);
    assert!(
        (answer - yes).abs() < 1e-5,
        "the answer is the yes probability {yes}, got {answer}: {body}"
    );
    assert!(
        no > yes,
        "the criterion leans no (yes {yes}, no {no}): {body}"
    );
    check_shape(billing, "x_label_mass", "systemone/billing");
    say("ok: /v1/systemone answered the same ticket from the same model");
}

/// One boot, one ticket, both routes, on a real checkpoint and device.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the local checkpoint and a Metal device"]
async fn both_routes_answer_one_ticket_against_a_real_model() {
    let Some((model, config)) = export() else {
        say(
            "SKIP: SHISO_TEST_MODEL names no usable export (a file needs \
             SHISO_TEST_MODEL_OPTIQ; a directory needs a config.json in it or \
             in its mlx/ subdirectory)",
        );
        return;
    };
    if !shiso_metal::metal_available() {
        say("SKIP: no Metal device");
        return;
    }

    let stack = Stack::boot(&model, &config);
    say(&format!("booted {}", model.display()));
    let state = Arc::clone(&stack.state);

    // A ceiling on the session rather than on each request: a device that
    // wedges should end this with a diagnosis, not with a runner still
    // waiting. Scoring generates nothing, so minutes are ample.
    let session = async {
        decisions(&state).await;
        systemone(&state).await;
    };
    let finished = tokio::time::timeout(Duration::from_secs(600), session).await;

    // The engine thread exits when the last queue reference goes, so this
    // clone must be gone before the join - `e2e.rs` never had one to release.
    drop(state);
    stack.shutdown();
    finished.expect("the session finished inside its deadline");
}
