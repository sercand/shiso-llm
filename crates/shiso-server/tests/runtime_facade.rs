//! `shiso_runtime::Runtime` over fake engines: what it serves, what it refuses,
//! and how it behaves under the three kinds of caller (a plain thread, a
//! blocking-pool thread, an async worker).
//!
//! The model and the image generator are in-memory fakes, so these pin the
//! facade's own discipline - capability errors, the blocking rule, the drop
//! order - not any arithmetic. The device cases are `runtime_device.rs`.

mod common;

use std::str::FromStr as _;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use serde_json::{Value, json};
use shiso_ir::decision::{Choice, ChoiceValue, Input, Question, QuestionKind, ReadoutState};
use shiso_ir::image::{Generate, Step};
use shiso_ir::{Message, Request, StreamEvent};
use shiso_runtime::{OpenError, Runtime, RuntimeOptions, Source};
use shiso_server::error::AppError;
use shiso_server::image::{ImageService, ImageServiceDesc};
use shiso_server::scheduler::SchedulerConfig;
use shiso_server::state::AppState;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::jinja_helpers::ToolJsonStyle;
use shiso_server::tokenizer::render_prompt::PromptRenderer;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits, ImageStepEvent,
};

/// How long a step that must happen may take.
const GUARD: Duration = Duration::from_secs(20);

/// The fake model's logits width.
const VOCAB: usize = 64;

/// The context of every state here.
const CTX: u32 = 256;

const TEMPLATE: &str = concat!(
    "{% for m in messages %}<|{{ m.role }}|> ",
    "{% for part in m.content %}{{ part.text }}{% endfor %} <|end|> ",
    "{% endfor %}<|assistant|>\n",
);

/// Ids 0..=7 are named, the rest are `w<id>`: scripted ids decode to words and
/// the labels `A`..`D` are one token each.
fn vocabulary() -> Arc<Vocabulary> {
    let named = [
        "<unk>",
        "<|user|>",
        "<|end|>",
        "<|assistant|>",
        "A",
        "B",
        "C",
        "D",
    ];
    let mut vocab = serde_json::Map::new();
    for id in 0..VOCAB {
        let text = named
            .get(id)
            .map_or_else(|| format!("w{id}"), |name| (*name).to_owned());
        vocab.insert(text, json!(id));
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

/// A language-model runtime over `engine`, with a chat template.
fn llm_runtime(
    engine: shiso_server::scheduler::EngineHandle,
    join: std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
) -> Runtime {
    let vocabulary = vocabulary();
    let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, Arc::clone(&vocabulary))
        .expect("the test template compiles");
    let state = AppState::new(engine, "test-model".to_owned(), CTX)
        .with_vocabulary(vocabulary)
        .with_renderer(Arc::new(renderer));
    Runtime::from_state(state, Some(join)).expect("a runtime over a model")
}

fn scripted(script: Vec<u32>) -> Runtime {
    let (engine, join, _teardowns) = common::fake_engine(script, Some(vocabulary()));
    llm_runtime(engine, join)
}

fn scored(rows: Vec<Vec<f32>>) -> Runtime {
    let (engine, join, _audit) = common::fake_engine_rows(rows, Some(vocabulary()));
    llm_runtime(engine, join)
}

#[derive(Debug)]
struct MockGenerator;

impl ImageGenerator for MockGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }
    fn limits(&self) -> ImageLimits {
        ImageLimits {
            size_multiple: 16,
            min_side: 256,
            max_side: 2048,
            default_size: (512, 512),
            default_steps: 2,
            fixed_steps: None,
        }
    }
    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        Ok(GeneratedImage {
            bytes: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 3,
            completion_tokens: 4,
        })
    }
}

/// A runtime that serves only an image model.
fn image_runtime() -> Runtime {
    let service = ImageService::start(ImageServiceDesc {
        generator: Arc::new(MockGenerator),
        model_id: "mock-flux".to_owned(),
        max_queue: 4,
        profile: None,
    })
    .expect("the image service starts");
    Runtime::from_state(AppState::without_language_model().with_image(service), None)
        .expect("a runtime over an image model")
}

fn picture() -> Generate {
    Generate {
        prompt: "a lighthouse".to_owned(),
        n: Some(1),
        size: Some("256x256".to_owned()),
        aspect_ratio: None,
        steps: Some(1),
        seed: Some(1),
        output_format: None,
        output_compression: None,
    }
}

fn chat() -> Request {
    let mut request = Request::new("test-model", vec![Message::user("w9 w10")]);
    request.max_tokens = 8;
    request.sampling.temperature = Some(0.0);
    request
}

fn pick_one() -> shiso_ir::decision::Request {
    let choice = |value: &str| Choice {
        value: ChoiceValue::Str(value.to_owned()),
        description: None,
    };
    shiso_ir::decision::Request {
        input: Input::Text("w9 w10".to_owned()),
        readout_state: ReadoutState::Input,
        questions: vec![Question {
            name: "q".to_owned(),
            instructions: "w11".to_owned(),
            kind: QuestionKind::Choice(vec![choice("first"), choice("second")]),
            probability_answer: false,
        }],
        temperature: None,
        preserve_thinking: None,
        token_ids: false,
    }
}

/// A row whose `A` and `B` slots differ, so the answer is `second`.
fn row() -> Vec<f32> {
    let mut row = vec![0.0; VOCAB];
    row[4] = 2.0;
    row[5] = 4.0;
    row
}

#[test]
fn a_runtime_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Runtime>();
}

#[test]
fn options_default_to_no_download() {
    let options = RuntimeOptions::default();
    assert!(
        !options.pull_if_missing,
        "opening never reaches the network unasked"
    );
    assert!(options.home.is_none() && options.explicit.is_empty());
}

#[test]
fn a_state_with_nothing_loaded_is_refused() {
    let error =
        Runtime::from_state(AppState::without_language_model(), None).expect_err("nothing to run");
    assert!(matches!(error, OpenError::NothingLoaded), "{error:?}");
}

#[test]
fn a_language_model_alone_has_no_image_model() {
    let runtime = scripted(vec![1]);
    assert!(runtime.has_language_model() && !runtime.has_image_model());
    let error = futures::executor::block_on(runtime.images(picture())).expect_err("no images");
    assert!(matches!(error, AppError::NoImageModel), "{error:?}");
}

#[test]
fn an_image_model_alone_has_no_language_model() {
    let runtime = image_runtime();
    assert!(!runtime.has_language_model() && runtime.has_image_model());
    let error = runtime.generate_blocking(&chat()).expect_err("no model");
    assert!(matches!(error, AppError::NoModel), "{error:?}");
    let error = runtime.decide_blocking(pick_one()).expect_err("no model");
    assert!(matches!(error, AppError::NoModel), "{error:?}");

    let images = futures::executor::block_on(runtime.images(picture())).expect("an image");
    assert_eq!(images.images.len(), 1);
    assert_eq!(images.images[0].media_type, "image/png");
}

#[test]
fn image_steps_reach_the_sink_and_false_stops() {
    let runtime = image_runtime();
    let mut steps = Vec::new();
    futures::executor::block_on(runtime.images_stream(picture(), |step| {
        steps.push(step);
        true
    }))
    .expect("streams");
    assert!(matches!(steps.first(), Some(Step::Admitted)), "{steps:?}");
    assert!(
        matches!(steps.last(), Some(Step::Completed { .. })),
        "{steps:?}"
    );

    let mut calls = 0;
    futures::executor::block_on(runtime.images_stream(picture(), |_| {
        calls += 1;
        false
    }))
    .expect("a sink that stops is not an error");
    assert_eq!(calls, 1);
}

#[test]
fn a_request_with_no_questions_is_refused() {
    let runtime = scored(vec![row()]);
    let mut request = pick_one();
    request.questions.clear();
    let refused = |error: AppError| {
        assert!(
            matches!(&error, AppError::BadRequest { param: Some(param), code: "invalid_request", .. } if param == "questions")
                && error.to_string().contains("at least one question"),
            "{error:?}"
        );
    };
    refused(
        runtime
            .decide_blocking(request.clone())
            .expect_err("no questions"),
    );
    refused(futures::executor::block_on(runtime.decide(request)).expect_err("no questions"));
}

/// The async methods work under an executor that is not tokio's, and the
/// blocking ones agree with them.
#[test]
fn async_and_blocking_answers_agree_on_a_foreign_executor() {
    let runtime = scored(vec![row(), row()]);
    let asynchronous =
        futures::executor::block_on(runtime.decide(pick_one())).expect("decide answers");
    let blocking = runtime.decide_blocking(pick_one()).expect("decide answers");
    assert_eq!(asynchronous, blocking);
    let answer = &blocking.answers[0];
    assert_eq!(answer.choice, ChoiceValue::Str("second".to_owned()));
    let total: f64 = answer.probabilities.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9, "{total}");
}

#[test]
fn generate_answers_blocking_async_and_streamed() {
    let runtime = scripted(vec![10, 11, 12]);
    let response = futures::executor::block_on(runtime.generate(&chat())).expect("generates");
    assert_eq!(response.message.text(), "w10 w11 w12");

    let again = runtime.generate_blocking(&chat()).expect("generates");
    assert_eq!(again.message.text(), response.message.text());

    let mut events = Vec::new();
    futures::executor::block_on(runtime.generate_stream(&chat(), |event| {
        events.push(event);
        true
    }))
    .expect("streams");
    assert!(
        matches!(events.first(), Some(StreamEvent::Start { .. })),
        "{events:?}"
    );
    assert!(
        matches!(events.last(), Some(StreamEvent::Stop { .. })),
        "{events:?}"
    );

    let mut seen = 0;
    futures::executor::block_on(runtime.generate_stream(&chat(), |_| {
        seen += 1;
        false
    }))
    .expect("a sink that stops is not an error");
    assert_eq!(seen, 1);
}

/// Runs `work` on a thread of its own and fails the test if it does not finish
/// in [`GUARD`]: a hang is a failure, not a stuck suite.
fn within_guard<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = done.send(work());
    });
    finished
        .recv_timeout(GUARD)
        .expect("the call finished without panicking or hanging")
}

fn said(response: &shiso_ir::Response) -> String {
    response.message.text()
}

/// A blocking call answers from every kind of caller: a plain thread, a
/// blocking-pool thread, a task and a `block_on` main future, on a
/// multi-thread runtime. From an async worker it blocks that worker only.
#[test]
fn a_blocking_call_answers_from_every_caller_on_a_multi_thread_runtime() {
    within_guard(|| {
        let runtime = Arc::new(scripted(vec![10, 11, 12]));
        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("a runtime");
        tokio_runtime.block_on(async {
            let in_task = {
                let runtime = Arc::clone(&runtime);
                tokio::spawn(async move { runtime.generate_blocking(&chat()) })
                    .await
                    .expect("the task does not panic")
            };
            assert_eq!(said(&in_task.expect("answers in a task")), "w10 w11 w12");

            let in_main = runtime.generate_blocking(&chat());
            assert_eq!(said(&in_main.expect("answers in main")), "w10 w11 w12");

            let in_pool = {
                let runtime = Arc::clone(&runtime);
                tokio::task::spawn_blocking(move || runtime.generate_blocking(&chat()))
                    .await
                    .expect("the blocking task does not panic")
            };
            assert_eq!(said(&in_pool.expect("answers in the pool")), "w10 w11 w12");
        });
        let in_thread = {
            let runtime = Arc::clone(&runtime);
            std::thread::spawn(move || runtime.generate_blocking(&chat()))
                .join()
                .expect("the thread does not panic")
        };
        assert_eq!(
            said(&in_thread.expect("answers on a thread")),
            "w10 w11 w12"
        );
        // The last handle goes on a plain thread: it joins the engine.
        drop(runtime);
    });
}

/// The same on a current-thread runtime, where blocking the only thread must
/// still finish because the call runs on the Runtime's own executor.
#[test]
fn a_blocking_call_answers_from_a_current_thread_runtime() {
    within_guard(|| {
        let runtime = Arc::new(scripted(vec![10, 11, 12]));
        let tokio_runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        tokio_runtime.block_on(async {
            let in_task = {
                let runtime = Arc::clone(&runtime);
                tokio::spawn(async move { runtime.generate_blocking(&chat()) })
                    .await
                    .expect("the task does not panic")
            };
            assert_eq!(said(&in_task.expect("answers in a task")), "w10 w11 w12");
            let in_main = runtime.generate_blocking(&chat());
            assert_eq!(said(&in_main.expect("answers in main")), "w10 w11 w12");
        });
        drop(tokio_runtime);
        drop(runtime);
    });
}

/// A sink runs inside the Runtime's own tokio context, so a task it spawns
/// lands on the Runtime's only worker. A blocking call there would wait for the
/// worker that has to run it; it returns an error instead.
#[test]
fn a_blocking_call_on_the_runtimes_own_worker_is_an_error() {
    let code = within_guard(|| {
        let runtime = Arc::new(scripted(vec![10, 11, 12]));
        let mut spawned = Vec::new();
        {
            let inner = Arc::clone(&runtime);
            futures::executor::block_on(runtime.generate_stream(&chat(), |event| {
                if matches!(event, StreamEvent::Start { .. }) {
                    let inner = Arc::clone(&inner);
                    spawned.push(tokio::spawn(
                        async move { inner.generate_blocking(&chat()) },
                    ));
                }
                true
            }))
            .expect("streams");
        }
        let error = futures::executor::block_on(spawned.remove(0))
            .expect("the task does not panic")
            .expect_err("refused, not hung");
        let code = error.code();
        drop(spawned);
        drop(runtime);
        code
    });
    assert_eq!(code, "blocking_on_runtime_worker");
}

/// Capability errors reach a blocking caller in a task as typed errors too.
#[test]
fn a_blocking_call_in_a_task_returns_the_capability_error() {
    within_guard(|| {
        let runtime = Arc::new(image_runtime());
        let tokio_runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime");
        let error = tokio_runtime.block_on(async move {
            tokio::spawn(async move { runtime.generate_blocking(&chat()) })
                .await
                .expect("the task does not panic")
        });
        assert!(matches!(error, Err(AppError::NoModel)), "{error:?}");
    });
}

/// The same runtime answers an async call made from a tokio task: the call
/// borrows nothing of tokio's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_calls_work_from_a_tokio_task() {
    let runtime = Arc::new(scripted(vec![10, 11, 12]));
    let answer = {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move { runtime.generate(&chat()).await })
            .await
            .expect("no panic")
    };
    assert_eq!(answer.expect("generates").message.text(), "w10 w11 w12");
    // The last handle goes on a thread that may block: the engine is joined.
    tokio::task::spawn_blocking(move || drop(runtime))
        .await
        .expect("the drop does not panic");
}

/// A call whose future was abandoned while the model was inside a step must not
/// make the drop hang or return early: the drop joins the engine thread, which
/// finishes the step it is in first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropping_after_an_abandoned_call_waits_for_the_engine_and_returns() {
    let gate = common::Gate::closed();
    let (engine, join, _audit) = common::scheduled_engine(
        vec![10; 40],
        SchedulerConfig::new(),
        Some(Arc::clone(&gate)),
    );
    let runtime = Arc::new(llm_runtime(engine, join));

    let (cancel, cancelled) = futures::channel::oneshot::channel::<()>();
    let caller = {
        let runtime = Arc::clone(&runtime);
        std::thread::spawn(move || {
            let request = chat();
            let call = Box::pin(runtime.generate(&request));
            let raced = futures::executor::block_on(futures::future::select(call, cancelled));
            matches!(raced, futures::future::Either::Right(_))
        })
    };
    gate.wait_for_arrivals(1).await;
    // The model is inside a step; abandon the call that is waiting on it.
    cancel.send(()).expect("the caller is waiting");
    assert!(
        caller.join().expect("the caller does not panic"),
        "the call was abandoned"
    );

    let (released_tx, released) = mpsc::channel();
    let drop_thread = std::thread::spawn(move || {
        drop(runtime);
        let _ = released_tx.send(());
    });
    assert!(
        released.recv_timeout(Duration::from_millis(300)).is_err(),
        "the drop returned with the engine inside a step"
    );
    gate.open();
    released
        .recv_timeout(GUARD)
        .expect("the drop returns once the engine is released");
    drop_thread.join().expect("the drop thread does not panic");
}

#[test]
fn opening_what_is_not_installed_is_a_typed_error() {
    let home = tempfile::tempdir().expect("a temp dir");
    let options = RuntimeOptions {
        home: Some(home.path().to_owned()),
        ..RuntimeOptions::default()
    };
    let error = Runtime::open(Source::Registry("nope"), options, &mut |_, _| true)
        .expect_err("not installed");
    assert!(
        matches!(
            &error,
            OpenError::Pull(shiso_runtime::hub::PullError::NotInstalled(_))
        ),
        "{error:?}"
    );
}

/// Opening from a task does not panic or hang; it fails here only because the
/// directory holds no model.
#[test]
fn opening_from_an_async_task_proceeds() {
    let error = within_guard(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("a runtime");
        let dir = tempfile::tempdir().expect("a temp dir");
        let path = dir.path().to_owned();
        runtime.block_on(async move {
            tokio::spawn(async move {
                Runtime::open(
                    Source::Dir(&path),
                    RuntimeOptions::default(),
                    &mut |_, _| true,
                )
                .err()
            })
            .await
            .expect("no panic")
        })
    });
    assert!(matches!(error, Some(OpenError::Boot(_))), "{error:?}");
}

/// A blocking-pool thread may block, so `open` gets as far as the model home.
#[tokio::test]
async fn opening_from_a_blocking_pool_thread_is_allowed() {
    let home = tempfile::tempdir().expect("a temp dir");
    let options = RuntimeOptions {
        home: Some(home.path().to_owned()),
        ..RuntimeOptions::default()
    };
    let error = tokio::task::spawn_blocking(move || {
        Runtime::open(Source::Registry("nope"), options, &mut |_, _| true)
    })
    .await
    .expect("no panic")
    .expect_err("not installed");
    assert!(matches!(error, OpenError::Pull(_)), "{error:?}");
}
