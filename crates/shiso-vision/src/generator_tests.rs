use super::*;
use crate::generator::ImageLimits;
use shiso_core::ids::Seed;

const LIMITS: ImageLimits = ImageLimits {
    size_multiple: 16,
    min_side: 256,
    max_side: 2048,
    default_size: (1024, 1024),
    default_steps: 4,
    fixed_steps: None,
};

#[test]
fn output_format_mime_types() {
    assert_eq!(ImageOutputFormat::Png.mime_type(), "image/png");
    assert_eq!(ImageOutputFormat::Jpeg.mime_type(), "image/jpeg");
}

#[test]
fn output_format_from_str() {
    assert_eq!(
        "png".parse::<ImageOutputFormat>().expect("parses"),
        ImageOutputFormat::Png
    );
    assert_eq!(
        "jpeg".parse::<ImageOutputFormat>().expect("parses"),
        ImageOutputFormat::Jpeg
    );
    assert_eq!(
        "jpg".parse::<ImageOutputFormat>().expect("parses"),
        ImageOutputFormat::Jpeg
    );
    assert!("webp".parse::<ImageOutputFormat>().is_err());
    assert!("gif".parse::<ImageOutputFormat>().is_err());
}

#[test]
fn job_dimension_validation() {
    let job = ImageGenerationJob {
        prompt: "test prompt".to_string(),
        width: 1024,
        height: 1024,
        steps: 4,
        seed: Seed::new(42),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    };
    assert!(job.validate(&LIMITS).is_ok());

    let invalid_job = ImageGenerationJob {
        prompt: String::new(),
        width: 1024,
        height: 1024,
        steps: 4,
        seed: Seed::new(42),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    };
    assert!(matches!(
        invalid_job.validate(&LIMITS),
        Err(ImageGenError::EmptyPrompt)
    ));
}

#[test]
fn job_refuses_a_zero_step_count_as_invalid_input() {
    // A job with no steps has no answer to give, and it is the request's
    // fault rather than a device failure: the refusal is its own variant so
    // the wire can answer 400, not 500.
    let job = ImageGenerationJob {
        prompt: "test prompt".to_string(),
        width: 1024,
        height: 1024,
        steps: 0,
        seed: Seed::new(42),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    };
    assert!(matches!(
        job.validate(&LIMITS),
        Err(ImageGenError::InvalidSteps { steps: 0 })
    ));
}

fn job_of(width: u32, height: u32, steps: u32) -> ImageGenerationJob {
    ImageGenerationJob {
        prompt: "test prompt".to_string(),
        width,
        height,
        steps,
        seed: Seed::new(42),
        output_format: ImageOutputFormat::Png,
        output_compression: None,
        input_references: Vec::new(),
    }
}

#[test]
fn dimensions_are_judged_against_the_models_grid_and_bounds() {
    let coarse = ImageLimits {
        size_multiple: 32,
        ..LIMITS
    };
    let error = job_of(1000, 1000, 4)
        .validate(&coarse)
        .expect_err("off the grid");
    assert_eq!(
        error.to_string(),
        "width and height must be multiples of 32 between 256 and 2048, got 1000x1000"
    );
    assert!(job_of(1008, 1008, 4).validate(&LIMITS).is_ok());
    assert!(job_of(1008, 1008, 4).validate(&coarse).is_err());
    assert!(job_of(2064, 1024, 4).validate(&LIMITS).is_err());
    assert!(job_of(128, 1024, 4).validate(&LIMITS).is_err());
}

#[test]
fn a_fixed_schedule_refuses_any_other_step_count() {
    let fixed = ImageLimits {
        fixed_steps: Some(8),
        ..LIMITS
    };
    let error = job_of(1024, 1024, 4).validate(&fixed).expect_err("fixed");
    assert_eq!(
        error.to_string(),
        "this model's schedule is fixed at 8 steps, got 4"
    );
    assert!(job_of(1024, 1024, 8).validate(&fixed).is_ok());
}

#[test]
fn memory_refusal_names_the_largest_side_or_says_none_fits() {
    let refusal = |largest_side| {
        ImageGenError::TooLargeForMemory {
            width: 1024,
            height: 1024,
            needed_mib: 7320,
            budget_mib: 778,
            largest_side,
        }
        .to_string()
    };
    assert!(refusal(1504).ends_with("the largest square image that fits is 1504x1504"));
    let none = refusal(0);
    assert!(!none.contains("0x0"), "{none}");
    assert!(none.contains("no image size fits"), "{none}");
}
