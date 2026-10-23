//! The typed image cores, driven with IR and no HTTP: what an embedding
//! application sees.

use std::sync::{Arc, Mutex};

use shiso_core::ids::Seed;
use shiso_ir::image::{Edit, Generate, Step, Usage};
use shiso_runtime::error::AppError;
use shiso_runtime::image::{ImageService, ImageServiceDesc};
use shiso_runtime::state::AppState;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageLimits, ImageStepEvent,
    InputReferenceImage,
};

const LIMITS: ImageLimits = ImageLimits {
    size_multiple: 16,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 3,
    fixed_steps: None,
};

/// A generator that records its jobs and reports one partial per step.
#[derive(Debug, Default)]
struct Recording {
    jobs: Mutex<Vec<ImageGenerationJob>>,
}

impl ImageGenerator for Recording {
    fn model_id(&self) -> &'static str {
        "mock"
    }

    fn limits(&self) -> ImageLimits {
        LIMITS
    }

    fn max_references(&self) -> usize {
        2
    }

    fn default_size_for_references(
        &self,
        _references: &[InputReferenceImage],
    ) -> Result<(u32, u32), ImageGenError> {
        Ok((512, 768))
    }

    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        self.jobs.lock().expect("lock").push(job.clone());
        if let Some(progress) = progress {
            for step in 0..job.steps {
                progress(ImageStepEvent {
                    step,
                    total_steps: job.steps,
                    partial_image: Some(vec![u8::try_from(step).unwrap_or(0)]),
                });
            }
        }
        Ok(GeneratedImage {
            bytes: vec![7, 7],
            media_type: "image/png",
            width: job.width,
            height: job.height,
            prompt_tokens: 1,
            completion_tokens: 10,
        })
    }
}

fn state() -> (AppState, Arc<Recording>) {
    let generator = Arc::new(Recording::default());
    let boxed: Arc<dyn ImageGenerator> = generator.clone();
    let service = ImageService::start(ImageServiceDesc {
        generator: boxed,
        model_id: "mock".to_owned(),
        max_queue: 4,
        profile: None,
    })
    .expect("starts");
    (
        AppState::without_language_model().with_image(service),
        generator,
    )
}

fn prompt(text: &str) -> Generate {
    Generate {
        prompt: text.to_owned(),
        n: None,
        size: None,
        aspect_ratio: None,
        steps: None,
        seed: None,
        output_format: None,
        output_compression: None,
    }
}

fn bad_request_param(error: &AppError) -> String {
    assert!(matches!(error, AppError::BadRequest { .. }), "{error:?}");
    error.to_string()
}

#[tokio::test]
async fn generate_makes_n_images_with_seed_plus_index_and_summed_usage() {
    let (state, generator) = state();
    let request = Generate {
        n: Some(2),
        seed: Some(40),
        size: Some("512x512".to_owned()),
        ..prompt("a cat")
    };
    let images = shiso_runtime::image::generate(&state, request)
        .await
        .expect("generates");
    assert_eq!(images.images.len(), 2);
    assert_eq!(images.images[0].media_type, "image/png");
    assert_eq!(images.images[0].bytes, vec![7, 7]);
    assert_eq!(
        images.usage,
        Usage {
            prompt_tokens: 2,
            completion_tokens: 20
        }
    );
    assert_eq!(images.usage.total_tokens(), 22);
    let jobs = generator.jobs.lock().expect("lock").clone();
    let seeds: Vec<_> = jobs.iter().map(|job| job.seed).collect();
    assert_eq!(seeds, vec![Seed::new(40), Seed::new(41)]);
    assert_eq!(
        (jobs[0].width, jobs[0].height, jobs[0].steps),
        (512, 512, 3)
    );
}

#[tokio::test]
async fn generate_stream_steps_count_partials_across_images_and_skip_the_last_step() {
    let (state, _generator) = state();
    let request = Generate {
        n: Some(2),
        ..prompt("a cat")
    };
    let mut steps = Vec::new();
    shiso_runtime::image::generate_stream(&state, request, |step| {
        steps.push(step);
        true
    })
    .await
    .expect("streams");
    // Three steps per image; the last step's partial is the finished image, so
    // two partials per image, numbered straight across the images.
    let summary: Vec<String> = steps
        .iter()
        .map(|step| match step {
            Step::Admitted => "admitted".to_owned(),
            Step::Partial { index, bytes } => format!("partial {index} {bytes:?}"),
            Step::Completed { usage, .. } => format!("completed {}", usage.completion_tokens),
        })
        .collect();
    assert_eq!(
        summary,
        [
            "admitted",
            "partial 0 [0]",
            "partial 1 [1]",
            "completed 10",
            "admitted",
            "partial 2 [0]",
            "partial 3 [1]",
            "completed 20",
        ]
    );
}

#[tokio::test]
async fn a_sink_that_stops_ends_the_stream_without_an_error() {
    let (state, generator) = state();
    let request = Generate {
        n: Some(3),
        ..prompt("a cat")
    };
    let mut seen = 0;
    shiso_runtime::image::generate_stream(&state, request, |_| {
        seen += 1;
        false
    })
    .await
    .expect("a gone reader is not a failure");
    assert_eq!(seen, 1);
    // Only the admitted picture was asked for.
    assert!(generator.jobs.lock().expect("lock").len() <= 1);
}

#[tokio::test]
async fn a_refusal_comes_back_before_the_sink_is_ever_called() {
    let (state, _generator) = state();
    let mut called = false;
    let error = shiso_runtime::image::generate_stream(
        &state,
        Generate {
            size: Some("wide".to_owned()),
            ..prompt("a cat")
        },
        |_| {
            called = true;
            true
        },
    )
    .await
    .expect_err("refused");
    assert!(!called);
    bad_request_param(&error);
}

#[tokio::test]
async fn the_ir_floor_is_checked_first() {
    let (state, generator) = state();
    for (request, wanted) in [
        (
            Generate {
                n: Some(0),
                ..prompt("x")
            },
            "n must be between 1 and 10",
        ),
        (
            Generate {
                n: Some(11),
                ..prompt("x")
            },
            "n must be between 1 and 10",
        ),
        (
            Generate {
                steps: Some(101),
                ..prompt("x")
            },
            "steps must be between 1 and 100",
        ),
        (
            Generate {
                output_compression: Some(0),
                ..prompt("x")
            },
            "output_compression must be between 1 and 100",
        ),
    ] {
        let error = shiso_runtime::image::generate(&state, request)
            .await
            .expect_err("refused");
        assert!(bad_request_param(&error).contains(wanted), "{error}");
    }
    assert!(generator.jobs.lock().expect("lock").is_empty());
}

#[tokio::test]
async fn a_malformed_request_is_a_400_ahead_of_the_missing_model() {
    let state = AppState::without_language_model();
    let malformed = Generate {
        size: Some("wide".to_owned()),
        ..prompt("x")
    };
    let error = shiso_runtime::image::generate(&state, malformed)
        .await
        .expect_err("refused");
    bad_request_param(&error);
    let error = shiso_runtime::image::generate(&state, prompt("x"))
        .await
        .expect_err("no model");
    assert!(matches!(error, AppError::NoImageModel), "{error:?}");
}

#[tokio::test]
async fn an_edit_without_a_size_takes_the_one_its_references_imply() {
    let (state, generator) = state();
    let edit = Edit {
        generate: prompt("make it blue"),
        references: vec![vec![1, 2, 3]],
    };
    let images = shiso_runtime::image::edit(&state, edit)
        .await
        .expect("edits");
    assert_eq!(images.images.len(), 1);
    let jobs = generator.jobs.lock().expect("lock").clone();
    assert_eq!((jobs[0].width, jobs[0].height), (512, 768));
    assert_eq!(jobs[0].input_references.len(), 1);
    assert_eq!(jobs[0].input_references[0].bytes, vec![1, 2, 3]);
}

#[tokio::test]
async fn an_edit_is_refused_for_what_its_shape_or_model_cannot_take() {
    let (state, generator) = state();
    let none = Edit {
        generate: prompt("x"),
        references: Vec::new(),
    };
    let error = shiso_runtime::image::edit(&state, none)
        .await
        .expect_err("none");
    assert!(bad_request_param(&error).contains("at least one reference"));
    let two = Edit {
        generate: Generate {
            n: Some(2),
            ..prompt("x")
        },
        references: vec![vec![1]],
    };
    let error = shiso_runtime::image::edit(&state, two)
        .await
        .expect_err("n");
    assert!(bad_request_param(&error).contains("n must be 1 for an edit, got 2"));
    let many = Edit {
        generate: prompt("x"),
        references: vec![vec![1], vec![2], vec![3]],
    };
    let error = shiso_runtime::image::edit_stream(&state, many, |_| true)
        .await
        .expect_err("too many");
    assert!(bad_request_param(&error).contains("accepts at most 2"));
    assert!(generator.jobs.lock().expect("lock").is_empty());
}
