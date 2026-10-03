use super::*;
use shiso_core::ids::Seed;

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
    assert!(job.validate().is_ok());

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
        invalid_job.validate(),
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
        job.validate(),
        Err(ImageGenError::InvalidSteps { steps: 0 })
    ));
}
