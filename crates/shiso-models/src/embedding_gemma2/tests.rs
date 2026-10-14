//! The family against `transformers`' own `EmbeddingGemma2Model`.
//!
//! The fixtures under `test-vectors/references/embedding-gemma2/` are an
//! independent reference: sentence-transformers over the safetensors
//! checkpoint in f32, captured by `scripts/reference/embedding_gemma2_reference.py`.
//! Each case records the exact ids the reference processor produced, so what
//! is compared here is the model and not two tokenizers: the text spans are
//! fed as those ids, the media as the same bytes the reference decoded.
//!
//! Needs a GPU device and `SHISO_TEST_MODEL_EMBEDDING_GEMMA2` (a GGUF), with
//! `SHISO_TEST_MMPROJ_EMBEDDING_GEMMA2` (the projector) for the media cases.
//! A case whose checkpoint is not given says so and is skipped, which reads
//! the same as a pass in the test output: read the lines it prints.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, TokenId};
use shiso_vision::{DecodedFrame, ImageLimits, preprocess_image, preprocess_video};

use super::{EmbeddingGemma2, Markers};
use crate::embedder::{EmbedPart, Embedder, EmbedderDesc, matryoshka};

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why, and the measured distance is the report"
)]
fn say(line: &str) {
    println!("{line}");
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/embedding-gemma2")
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .expect("the fixture is json")
}

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .expect("an array of numbers")
        .iter()
        .map(|v| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the fixture holds f32 values"
            )]
            let f = v.as_f64().expect("a number") as f32;
            f
        })
        .collect()
}

fn ids(case: &serde_json::Value) -> Vec<TokenId> {
    case["input_ids"]
        .as_array()
        .expect("input ids")
        .iter()
        .map(|v| TokenId::new(u32::try_from(v.as_u64().expect("an id")).expect("fits")))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// The media a case carries, decoded as the server would decode it.
struct Media {
    images: Vec<shiso_vision::VisionItem>,
    frames: Vec<shiso_vision::VisionItem>,
    audio: Vec<shiso_audio::AudioClip>,
}

fn media(model: &EmbeddingGemma2, case: &serde_json::Value) -> Media {
    let mut out = Media {
        images: Vec::new(),
        frames: Vec::new(),
        audio: Vec::new(),
    };
    let Some(file) = case["media"]["file"].as_str() else {
        return out;
    };
    let bytes = std::fs::read(fixtures().join(file)).expect("the media fixture");
    let frontend = model.frontend();
    let ext = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    if ext == "png" {
        let spec = frontend.image.as_ref().expect("a vision tower");
        out.images
            .push(preprocess_image(&bytes, spec, ImageLimits::default()).expect("preprocesses"));
    } else if ext == "wav" {
        let tower = frontend.audio.as_ref().expect("an audio tower");
        let pcm = shiso_audio::decode_audio(
            &bytes,
            &shiso_audio::DecodeLimits::new(std::num::NonZeroU32::new(60).expect("nonzero")),
            &shiso_audio::AudioDecodePolicy::default(),
        )
        .expect("decodes");
        out.audio
            .push(shiso_audio::clip_for(tower.as_ref(), &pcm).expect("frames"));
    } else if ext == "mp4" {
        // The reference's own sampled frames, rgb24, so the decoder is not
        // what is compared: `[frames][height][width][3]`.
        let shape = case["video"]["sampled_frames_rgb"]["shape"]
            .as_array()
            .expect("the sampled frames' shape");
        let dim = |i: usize| usize::try_from(shape[i].as_u64().expect("a size")).expect("fits");
        let (n, h, w) = (dim(0), dim(1), dim(2));
        let raw = std::fs::read(fixtures().join("video-synthetic-frames.u8")).expect("frames");
        let spec = &frontend.video.as_ref().expect("a vision tower").frames;
        for frame in raw.chunks_exact(h * w * 3).take(n) {
            let frame = DecodedFrame::from_rgb8(
                u32::try_from(w).expect("fits"),
                u32::try_from(h).expect("fits"),
                frame.to_vec(),
            )
            .expect("a frame");
            out.frames.push(
                preprocess_video(std::slice::from_ref(&frame), spec, ImageLimits::default())
                    .expect("preprocesses"),
            );
        }
    }
    out
}

/// The reference's ids, back as the parts the server would have sent: text
/// spans between the sequence's own markers, and each media run as its item.
fn parts(markers: &Markers, ids: &[TokenId], mut media: Media) -> Vec<EmbedPart> {
    assert_eq!(
        ids.first(),
        Some(&markers.bos),
        "the reference opens with bos"
    );
    assert_eq!(
        ids.last(),
        Some(&markers.eos),
        "the reference closes with eos"
    );
    let body = &ids[1..ids.len() - 1];
    let mut out = Vec::new();
    let mut text = Vec::new();
    let mut images = media.images.drain(..);
    let mut frames = media.frames.drain(..);
    let mut audio = media.audio.drain(..);
    let mut i = 0;
    while i < body.len() {
        let id = body[i];
        let opens = id == markers.image_open || id == markers.audio_open;
        if !opens {
            text.push(id);
            i += 1;
            continue;
        }
        if !text.is_empty() {
            out.push(EmbedPart::Text(std::mem::take(&mut text)));
        }
        let close = if id == markers.image_open {
            markers.image_close
        } else {
            markers.audio_close
        };
        let end = i + body[i..]
            .iter()
            .position(|t| *t == close)
            .expect("a closed run");
        let run = body[i + 1];
        if run == markers.image {
            out.push(EmbedPart::Image(images.next().expect("an image per run")));
        } else if run == markers.video {
            let frame = frames.next().expect("a frame per run");
            match out.last_mut() {
                Some(EmbedPart::Video(clip)) => clip.push(frame),
                _ => out.push(EmbedPart::Video(vec![frame])),
            }
        } else {
            out.push(EmbedPart::Audio(audio.next().expect("a clip per run")));
        }
        i = end + 1;
    }
    if !text.is_empty() {
        out.push(EmbedPart::Text(text));
    }
    out
}

/// Cosine floors per container. The BF16 container holds the reference's
/// own weights, so what is left is f32 summation order; the `Q8_0` one
/// rounds every matrix to eight bits on top.
const FLOORS: [(&str, &str, f64); 2] = [
    ("bf16", "SHISO_TEST_MODEL_EMBEDDING_GEMMA2", 0.9999),
    ("q8_0", "SHISO_TEST_MODEL_EMBEDDING_GEMMA2_Q8", 0.999),
];

/// The floor a video case is held to; measured 0.99962 in BF16.
const VIDEO_FLOOR: f64 = 0.999;

const CASES: [&str; 7] = [
    "query-northern-lights",
    "doc-northern-lights",
    "plain-multilingual",
    "code",
    "image-synthetic",
    "audio-speech",
    "video-synthetic",
];

fn device() -> Option<Arc<dyn Backend>> {
    crate::audio_ops::testing::device()
}

#[test]
#[ignore = "requires a GPU device and the checkpoint"]
fn every_case_lands_where_the_reference_puts_it() {
    let Some(backend) = device() else {
        say("SKIP: no GPU device");
        return;
    };
    let mmproj = std::env::var_os("SHISO_TEST_MMPROJ_EMBEDDING_GEMMA2").map(PathBuf::from);
    let mut ran = 0;
    for (name, var, floor) in FLOORS {
        let Some(path) = std::env::var_os(var).map(PathBuf::from) else {
            say(&format!("SKIP {name}: {var} is unset"));
            continue;
        };
        let mut model = EmbeddingGemma2::open(
            &EmbedderDesc {
                model: &path,
                mmproj: mmproj.as_deref(),
                sidecar: None,
            },
            &backend,
        )
        .expect("the model opens");
        for case_name in CASES {
            let case = json(&fixtures().join(format!("{case_name}.json")));
            if case["media"]["file"].is_string() && mmproj.is_none() {
                say(&format!(
                    "SKIP {name} {case_name}: SHISO_TEST_MMPROJ_EMBEDDING_GEMMA2 is unset"
                ));
                continue;
            }
            let decoded = media(&model, &case);
            let input = parts(model.markers(), &ids(&case), decoded);
            let out = model.embed(&input).expect("embeds");
            assert_eq!(
                u64::from(out.usage.total()),
                case["n_tokens"].as_u64().expect("a count"),
                "{name} {case_name}: the layout's length"
            );
            let reference = floats(&case["embedding"]);
            let ours = matryoshka(&out.vector, 768).expect("full width");
            let cos = cosine(&ours, &reference);
            let mut line = format!("{name} {case_name}: cos {cos:.6}");
            for dim in [512usize, 256, 128] {
                let cut = matryoshka(&out.vector, dim).expect("a listed width");
                let theirs = floats(&case["truncated"][dim.to_string()]);
                let _ = write!(line, ", {dim}d {:.6}", cosine(&cut, &theirs));
            }
            say(&line);
            // A video's frames are small pictures grown to their budget, so
            // the resize filter - this tree's Catmull-Rom against the
            // reference's PIL bicubic - moves them further than a still. Fed
            // the reference's own patch rows the pass measured 1.000000.
            let floor = if case_name.starts_with("video") {
                floor.min(VIDEO_FLOOR)
            } else {
                floor
            };
            assert!(
                cos >= floor,
                "{name} {case_name}: cosine {cos} under {floor}"
            );
            ran += 1;
        }
    }
    say(&format!("{ran} cases compared"));
    assert!(ran > 0 || std::env::var_os("SHISO_TEST_MODEL_EMBEDDING_GEMMA2").is_none());
}

/// Every text case in one batch: packed into one pass, each must still land
/// where the reference puts it alone - packing is invisible or it is wrong.
#[test]
#[ignore = "requires a GPU device and the checkpoint"]
fn packed_inputs_land_where_each_lands_alone() {
    let Some(backend) = device() else {
        say("SKIP: no GPU device");
        return;
    };
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_EMBEDDING_GEMMA2").map(PathBuf::from)
    else {
        say("SKIP: SHISO_TEST_MODEL_EMBEDDING_GEMMA2 is unset");
        return;
    };
    let mut model = EmbeddingGemma2::open(
        &EmbedderDesc {
            model: &path,
            mmproj: None,
            sidecar: None,
        },
        &backend,
    )
    .expect("the model opens");
    let names = CASES
        .iter()
        .copied()
        .filter(|c| !json(&fixtures().join(format!("{c}.json")))["media"]["file"].is_string())
        .collect::<Vec<_>>();
    let cases: Vec<_> = names
        .iter()
        .map(|c| json(&fixtures().join(format!("{c}.json"))))
        .collect();
    let inputs: Vec<Vec<EmbedPart>> = cases
        .iter()
        .map(|case| {
            parts(
                model.markers(),
                &ids(case),
                Media {
                    images: Vec::new(),
                    frames: Vec::new(),
                    audio: Vec::new(),
                },
            )
        })
        .collect();
    let out = model.embed_batch(&inputs).expect("embeds");
    assert_eq!(out.len(), cases.len());
    for ((name, case), got) in names.iter().zip(&cases).zip(&out) {
        let cos = cosine(
            &matryoshka(&got.vector, 768).expect("full"),
            &floats(&case["embedding"]),
        );
        say(&format!("packed {name}: cos {cos:.6}"));
        assert!(cos >= 0.99999, "packed {name}: cosine {cos}");
    }
}
