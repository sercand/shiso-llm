//! Which spans a step takes and which plans a job refuses, on the host; and
//! the offline job against the stream over one long clip, on the device.

use std::sync::Arc;

use shiso_audio::chunks::{OfflineRule, Span, plan_offline};
use shiso_audio::{
    DecodeLimits, Language, MelFrames, MelFrontend, SpeechRecognizer, StreamOptions, TokenEvent,
    decode_audio,
};

use super::{check_plan, next_batch};
use crate::qwen3_asr::{first_outside, offline_batch};

fn span(start: usize, end: usize, transcribe: bool) -> Span {
    Span {
        start,
        end,
        transcribe,
    }
}

#[test]
fn a_step_takes_the_next_transcribed_spans_and_passes_the_skipped_ones() {
    let spans = [
        span(0, 10, true),
        span(10, 20, false),
        span(20, 30, true),
        span(30, 40, true),
        span(40, 50, false),
        span(50, 60, false),
        span(60, 70, true),
    ];
    assert_eq!(next_batch(&spans, 0, 2), (vec![0, 2], 3));
    assert_eq!(next_batch(&spans, 3, 2), (vec![3, 6], 7));
    assert_eq!(next_batch(&spans, 7, 2), (vec![], 7));
    // A batch wider than what is left takes the rest.
    assert_eq!(next_batch(&spans, 0, 8), (vec![0, 2, 3, 6], 7));
    // A plan that is all silence has nothing to transcribe.
    assert_eq!(next_batch(&[span(0, 900, false)], 0, 8), (vec![], 1));
}

#[test]
fn a_plan_with_a_gap_an_overlap_or_an_empty_span_is_refused() {
    check_plan(&[span(0, 5, true), span(5, 9, false)]).expect("tiles");
    check_plan(&[]).expect("an empty clip plans nothing");
    for bad in [
        vec![span(1, 5, true)],
        vec![span(0, 5, true), span(6, 9, true)],
        vec![span(0, 5, true), span(4, 9, true)],
        vec![span(0, 5, true), span(5, 5, true)],
    ] {
        assert!(check_plan(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn a_weight_outside_the_audio_set_keeps_the_single_path() {
    let names = |alone: &[bool]| {
        alone
            .iter()
            .enumerate()
            .map(|(i, &a)| (format!("layers.{i}.q"), a))
            .collect::<Vec<_>>()
    };
    assert_eq!(first_outside(names(&[true, true, true])), None);
    assert_eq!(
        first_outside(names(&[true, false, false])).as_deref(),
        Some("layers.1.q")
    );
    assert_eq!(offline_batch(8, None), 8);
    assert_eq!(offline_batch(8, Some("layers.1.q")), 0);
}

#[test]
#[ignore = "requires a GPU device"]
fn an_engine_projected_weight_does_not_sum_its_rows_alone() {
    use crate::audio_ops::{Linear, LinearWeight};
    use shiso_core::{DType, HostBytes};
    use shiso_engine::projection::Weight;

    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    // One row of a 256-wide K-quant block: the engine's own kernels, whose
    // tiling follows the row count.
    let host: Arc<dyn HostBytes> = Arc::new(vec![0_u8; 144]);
    let weight = Weight::import(&backend, DType::Q4_K, 256, 1, host, 0..144).expect("imports");
    let projected = Linear {
        weight: LinearWeight::Projected(weight),
        bias: None,
    };
    assert!(!projected.rows_sum_alone());
}

/// The text `events` spell in `asr`'s vocabulary.
fn text_of(asr: &crate::qwen3_asr::Qwen3Asr, events: &[TokenEvent]) -> String {
    let mut detok = shiso_audio::Detokenizer::new(Arc::clone(asr.vocab()));
    for e in events {
        detok.push(e.token.get());
    }
    detok.flush();
    detok.text().to_owned()
}

/// The model's pooled memories: the streams' count, and whether a batch's is
/// held.
fn pooled(asr: &crate::qwen3_asr::Qwen3Asr) -> (usize, bool) {
    use std::sync::PoisonError;
    let streams = asr
        .model
        .spare
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .len();
    let batch = asr
        .model
        .batch_spare
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some();
    (streams, batch)
}

#[test]
#[ignore = "requires a GPU device and SHISO_TEST_MODEL_QWEN3ASR"]
fn offline_text_equals_streaming_text_on_long_dialogue() {
    use crate::qwen3_asr::config::tests::say;

    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR") else {
        say("SKIP: SHISO_TEST_MODEL_QWEN3ASR is unset");
        return;
    };
    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    let path = std::path::Path::new(&root).join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf");
    let asr = crate::qwen3_asr::open(&path, None, &backend)
        .expect("loads")
        .expect("this family")
        .with_offline_batch(8);
    assert_eq!(
        asr.batches_offline(),
        8,
        "every Q8_0 weight is the audio set's"
    );
    let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/audio/clips/long-dialogue.wav");
    let bytes = std::fs::read(&wav).expect("the clip");
    let limits = DecodeLimits::new(std::num::NonZeroU32::new(600).expect("non-zero"));
    let pcm = decode_audio(&bytes, &limits, None).expect("decodes");
    let mel = MelFrontend::new(asr.mel_spec().clone())
        .expect("a frontend")
        .compute(pcm.samples());
    let (n, valid) = (mel.n_mels(), mel.valid());
    let mel = MelFrames::from_rows(mel.as_slice()[..valid * n].to_vec(), n).expect("rows");
    let options = StreamOptions {
        language: Language::Auto,
        lookahead: 0,
        cuts: None,
    };

    // The stream, pushed as the server pushes.
    let mut stream = asr.open_stream(&options).expect("opens");
    let mut streamed = Vec::new();
    for rows in mel.as_slice().chunks(1_600 * n) {
        let piece = MelFrames::from_rows(rows.to_vec(), n).expect("rows");
        streamed.extend(stream.push(&piece).expect("pushes"));
    }
    streamed.extend(stream.finish().expect("finishes"));
    let (stream_ends, stream_language) = (
        stream.chunk_ends().to_vec(),
        stream.language().map(str::to_owned),
    );
    drop(stream);
    assert_eq!(
        pooled(&asr),
        (1, false),
        "the closed stream's memory is pooled"
    );

    // The offline job over the silence rule's plan: the stream's own cuts.
    let spans = plan_offline(&mel, None, 0.01, &OfflineRule::for_10ms_mel(true)).expect("plans");
    let mut job = asr.open_offline(&options, spans.clone()).expect("opens");
    assert_eq!(
        pooled(&asr),
        (1, false),
        "an open offline job takes no stream's memory"
    );
    let mut offline: Vec<TokenEvent> = Vec::new();
    let mut ends = Vec::new();
    let mut languages = Vec::new();
    let mut steps = 0;
    let started = std::time::Instant::now();
    while let Some(batch) = job.step(&mel).expect("steps") {
        steps += 1;
        assert!(batch.spans.len() <= 8);
        for result in batch.spans {
            assert!(result.span.transcribe);
            offline.extend(result.tokens);
            ends.push(result.chunk_end);
            languages.push(result.language);
        }
    }
    assert_eq!(
        pooled(&asr),
        (1, true),
        "between steps the job holds nothing: its tower and decoder memory is pooled"
    );
    drop(job);
    assert_eq!(pooled(&asr), (1, true));
    let text = |events: &[TokenEvent]| text_of(&asr, events);
    say(&format!(
        "{} spans in {steps} steps ({:.2} s), {} events; offline {:?}",
        spans.len(),
        started.elapsed().as_secs_f64(),
        offline.len(),
        text(&offline)
    ));
    assert_eq!(text(&offline), text(&streamed));
    assert_eq!(offline, streamed, "the same ids at the same frames");
    assert_eq!(ends, stream_ends);
    assert_eq!(
        languages.iter().flatten().next().map(String::as_str),
        stream_language.as_deref()
    );
}
