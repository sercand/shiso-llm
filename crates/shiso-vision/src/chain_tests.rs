//! The chain's fall-through rules, against fake steps.

use super::*;
use crate::video::DecodedFrame;

/// A step that always answers the same way.
struct Fake {
    name: &'static str,
    answer: fn() -> Result<Attempt, VisionError>,
}

impl VideoStep for Fake {
    fn name(&self) -> &'static str {
        self.name
    }

    fn decode(&self, _: &StepInput<'_>) -> Result<Attempt, VisionError> {
        (self.answer)()
    }
}

fn frame() -> DecodedFrame {
    DecodedFrame::from_rgb8(1, 1, vec![0, 0, 0]).expect("1x1 rgb")
}

fn run(steps: &[&dyn VideoStep]) -> Result<DecodedClip, VisionError> {
    let policy = VideoDecodePolicy::default();
    run_chain(steps, &StepInput::new(b"x", Container::Mp4, 2.0, &policy))
}

#[test]
fn an_unsupported_step_falls_through_to_the_next() {
    let skip = Fake {
        name: "os",
        answer: || Ok(Attempt::Unsupported("no session".into())),
    };
    let done = Fake {
        name: "ffmpeg",
        answer: || DecodedClip::new(vec![frame()], 2.0).map(Attempt::Done),
    };
    assert_eq!(run(&[&skip, &done]).expect("decodes").frames().len(), 1);
}

#[test]
fn a_failed_step_stops_the_chain() {
    let fail = Fake {
        name: "os",
        answer: || {
            Err(VisionError::DecoderFailed {
                detail: "bad slice".into(),
            })
        },
    };
    let never = Fake {
        name: "ffmpeg",
        answer: || panic!("the chain must not retry after a failure"),
    };
    assert!(matches!(
        run(&[&fail, &never]),
        Err(VisionError::DecoderFailed { .. })
    ));
}

#[test]
fn every_step_unsupported_lists_what_was_tried() {
    let a = Fake {
        name: "os",
        answer: || Ok(Attempt::Unsupported("a".into())),
    };
    let b = Fake {
        name: "rust",
        answer: || Ok(Attempt::Unsupported("b".into())),
    };
    let error = run(&[&a, &b]).expect_err("nothing decodes");
    let VisionError::NoVideoDecoder { tried, .. } = &error else {
        panic!("wrong error: {error}");
    };
    assert_eq!(tried, &["os", "rust"]);
    let text = error.to_string();
    assert!(text.contains("os, rust"), "{text}");
}

#[test]
fn an_empty_chain_says_nothing_was_tried() {
    let error = run(&[]).expect_err("no decoder");
    assert!(matches!(
        error,
        VisionError::NoVideoDecoder { ref tried, .. } if tried.is_empty()
    ));
}
