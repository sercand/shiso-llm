//! The image service: a job goes in, an image comes back; a full queue is
//! refused with a typed 503; a streaming caller sees its step events.
//!
//! The generator is mocked here: what these tests pin is the service's own
//! discipline - the queue, the refusals, the progress forwarding - not the
//! diffusion math, which is `shiso-models`' business.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use shiso_core::ids::Seed;
use shiso_server::error::AppError;
use shiso_server::image::{ImageService, ImageServiceDesc};
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageOutputFormat,
    ImageStepEvent,
};

/// The encoded bytes a mock hands back: a PNG signature, no pixels.
fn png_image(width: u32, height: u32) -> GeneratedImage {
    GeneratedImage {
        bytes: vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A],
        media_type: "image/png",
        width,
        height,
        prompt_tokens: 10,
        completion_tokens: 1024,
    }
}

/// A job with everything but the prompt and the step count at servable values.
fn job(prompt: &str, steps: u32) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: prompt.to_owned(),
        width: 512,
        height: 512,
        steps,
        seed: Seed::new(1234),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    }
}

#[derive(Debug)]
struct MockGenerator;
impl ImageGenerator for MockGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
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
            prompt_tokens: 10,
            completion_tokens: 1024,
        })
    }
}

/// A generator that reports it has started and then waits until the test
/// releases it: how a full queue is made deterministic.
#[derive(Debug)]
struct BlockingGenerator {
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    gate: Mutex<mpsc::Receiver<()>>,
}

impl BlockingGenerator {
    fn new(started: tokio::sync::oneshot::Sender<()>, gate: mpsc::Receiver<()>) -> Self {
        Self {
            started: Mutex::new(Some(started)),
            gate: Mutex::new(gate),
        }
    }
}

impl ImageGenerator for BlockingGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }
    fn generate(
        &self,
        job: &ImageGenerationJob,
        _progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        if let Some(sender) = self
            .started
            .lock()
            .expect("the started lock is never poisoned")
            .take()
        {
            let _gone = sender.send(());
        }
        // Waits for the test's token; when the test drops the sender the wait
        // ends and the job completes.
        let _gone = self
            .gate
            .lock()
            .expect("the gate lock is never poisoned")
            .recv();
        Ok(png_image(job.width, job.height))
    }
}

/// A generator that emits one event per denoising step, as the real pipeline
/// does.
#[derive(Debug)]
struct SteppingGenerator;
impl ImageGenerator for SteppingGenerator {
    fn model_id(&self) -> &'static str {
        "mock-flux"
    }
    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        if let Some(progress) = progress {
            for step in 0..job.steps {
                progress(ImageStepEvent {
                    step,
                    total_steps: job.steps,
                    partial_image: None,
                });
            }
        }
        Ok(png_image(job.width, job.height))
    }
}

#[tokio::test]
async fn image_service_submits_and_completes_job() {
    let generator = Arc::new(MockGenerator);
    let service = ImageService::start(ImageServiceDesc {
        generator,
        model_id: "mock-flux".to_string(),
        max_queue: 4,
    })
    .expect("starts image service");

    let job = ImageGenerationJob {
        prompt: "a sunset over mountains".to_string(),
        width: 512,
        height: 512,
        steps: 4,
        seed: Seed::new(1234),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    };

    let result = service.generate(job).await.expect("generates image");
    assert_eq!(&result.bytes[..4], &[0x89, b'P', b'N', b'G']);
    assert_eq!(result.width, 512);
}

#[tokio::test]
async fn image_service_streams_step_events() {
    let service = ImageService::start(ImageServiceDesc {
        generator: Arc::new(SteppingGenerator),
        model_id: "mock-flux".to_string(),
        max_queue: 4,
    })
    .expect("starts image service");

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let result = service
        .generate_stream(job("a red door", 4), tx)
        .await
        .expect("generates image");
    assert_eq!(result.height, 512);

    // The events were sent before the reply was, so by the time the answer
    // arrives every step is already in the channel.
    let mut steps = Vec::new();
    while let Ok(event) = rx.try_recv() {
        steps.push(event.step);
    }
    assert_eq!(steps, vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn image_service_refuses_when_the_queue_is_full() {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let generator = Arc::new(BlockingGenerator::new(started_tx, release_rx));
    let service = Arc::new(
        ImageService::start(ImageServiceDesc {
            generator,
            model_id: "mock-flux".to_string(),
            max_queue: 1,
        })
        .expect("starts image service"),
    );

    // The worker takes the first job off the queue and blocks inside the
    // generator, so the queue's one slot is empty and stays empty.
    let first = {
        let service = Arc::clone(&service);
        tokio::spawn(async move { service.generate(job("first", 4)).await })
    };
    started_rx.await.expect("the worker started the first job");

    // With the worker busy, the queue's one slot admits exactly one of the
    // next two jobs and the other is refused. `#[tokio::test]` runs a
    // current-thread runtime, so the two tasks send in spawn order.
    let second = {
        let service = Arc::clone(&service);
        tokio::spawn(async move { service.generate(job("second", 4)).await })
    };
    let third = {
        let service = Arc::clone(&service);
        tokio::spawn(async move { service.generate(job("third", 4)).await })
    };

    // Let everything finish: one token for the blocked job, and dropping the
    // sender ends the queued one's wait.
    release_tx.send(()).expect("the generator is waiting");
    drop(release_tx);

    let (first, second, third) = tokio::join!(first, second, third);
    let outcomes = [
        first.expect("the first task runs"),
        second.expect("the second task runs"),
        third.expect("the third task runs"),
    ];
    let refused: Vec<&AppError> = outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            Err(error @ AppError::Overloaded { .. }) => Some(error),
            _ => None,
        })
        .collect();
    assert_eq!(
        refused.len(),
        1,
        "exactly one of three jobs meets a full queue: {outcomes:?}"
    );
    assert_eq!(
        refused[0].status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        "a full queue is a refusal a retry can fix, not a 500"
    );
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_ok()).count(),
        2,
        "the running job and the queued one are both served"
    );
}

#[tokio::test]
async fn image_service_refuses_an_invalid_job_before_queueing_it() {
    let service = ImageService::start(ImageServiceDesc {
        generator: Arc::new(MockGenerator),
        model_id: "mock-flux".to_string(),
        max_queue: 4,
    })
    .expect("starts image service");

    let error = service
        .generate(job("   ", 4))
        .await
        .expect_err("a blank prompt is refused");
    assert_eq!(
        error.status(),
        axum::http::StatusCode::BAD_REQUEST,
        "a job the generator would refuse is the request's fault: {error}"
    );
}
