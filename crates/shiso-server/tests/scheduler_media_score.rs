//! A `ScoreRequest` carrying media: the encode step runs on admission, the
//! rows reach the prefill, and the reply is answered exactly once either way.
//!
//! The generation path already encodes a request's media before the slot reads
//! its prompt (`open` runs `encode_media`); scoring must mirror it, because a
//! score slot that reached the prefill with pad tokens and no rows would read
//! placeholders as text. The fakes are the shared media ones in `common`: a
//! model that records WHICH prefill entry point it was asked for, and a tower
//! that either hands over rows sized to the item's grid or refuses.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::{Value, json};
use shiso_core::TokenId;
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_server::error::AppError;
use shiso_server::media::MediaItem;
use shiso_server::scheduler::ScoreRequest;
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::message_preprocess::{MediaPolicy, preprocess};
use shiso_vision::{ImageLimits, MediaPlacement};
use std::str::FromStr as _;

/// The fake's vocabulary width.
const VOCAB: usize = 64;

/// The image marker's id in the test vocabulary below.
const PAD: u32 = 2;

/// A full logits row: `[1.0, 2.0, 0, ...]` over the fake vocabulary.
fn label_row() -> Vec<f32> {
    let mut row = vec![0f32; VOCAB];
    row[0] = 1.0;
    row[1] = 2.0;
    row
}

/// The log-sum-exp of `row`, computed here in `f64` by the same formula the
/// scoring terminus states, so the assertion is independent arithmetic.
fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().fold(f32::NEG_INFINITY, |best, v| best.max(*v));
    f64::from(max)
        + row
            .iter()
            .map(|v| (f64::from(*v) - f64::from(max)).exp())
            .sum::<f64>()
            .ln()
}

/// A vocabulary declaring Qwen's image marker as an added token, so the
/// coordinator resolves a pad id and the splice has a run to write into.
fn qwen_vocabulary() -> Arc<Vocabulary> {
    let mut vocab = serde_json::Map::new();
    vocab.insert("<unk>".to_owned(), json!(0));
    vocab.insert("word".to_owned(), json!(1));
    vocab.insert(MediaPlacement::QWEN35.image_marker.to_owned(), json!(PAD));
    let added = vec![json!({
        "id": PAD,
        "content": MediaPlacement::QWEN35.image_marker,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
        "special": true,
    })];
    let document = json!({
        "version": "1.0",
        "truncation": Value::Null,
        "padding": Value::Null,
        "added_tokens": added,
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

/// One decoded picture and its digest, through the same preprocess the
/// renderer runs - so the item's grid and the prompt's pad run cannot drift.
fn one_picture() -> (MediaItem, shiso_core::media::MediaDigest) {
    let spec = common::media_spec();
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(common::png_data_uri(
            96, 96,
        )))],
    );
    let policy = MediaPolicy {
        vision: Some(&spec),
        limits: ImageLimits::default(),
        ..MediaPolicy::default()
    };
    let prepared =
        preprocess(&[], std::slice::from_ref(&message), &policy).expect("the picture preprocesses");
    assert_eq!(prepared.media.len(), 1);
    assert_eq!(
        prepared.pad_counts,
        vec![4],
        "a 96x96 picture is four pads here"
    );
    (
        prepared.media.into_iter().next().expect("one item"),
        prepared.digests.into_iter().next().expect("one digest"),
    )
}

/// An engine over the media-capable fake, `fail_tower` deciding whether the
/// vision tower encodes or refuses.
fn media_engine(rows: Vec<Vec<f32>>, fail_tower: bool) -> common::MediaEngine {
    common::fake_media_engine(rows, Some(qwen_vocabulary()), fail_tower)
}

/// A prompt of `word`, four pads, `word` - the ids the renderer would hand the
/// scheduler for one 96x96 picture between two words.
fn padded_prompt() -> Arc<[TokenId]> {
    Arc::from(
        [1, PAD, PAD, PAD, PAD, 1]
            .iter()
            .copied()
            .map(TokenId::new)
            .collect::<Vec<_>>(),
    )
}

fn labels() -> Arc<[TokenId]> {
    Arc::from(vec![TokenId::new(0), TokenId::new(1)])
}

#[tokio::test]
async fn a_score_request_with_media_reaches_the_terminus() {
    let (item, digest) = one_picture();
    let (engine, join, audit, media_calls) = media_engine(vec![label_row()], false);

    let outcome = engine
        .score(
            ScoreRequest::new(padded_prompt(), labels())
                .with_media(Arc::from(vec![item]), Arc::from(vec![digest])),
        )
        .await
        .expect("the score is answered");

    // Media attached: the encode step ran on admission and the prefill went
    // through the media entry point, exactly once.
    assert_eq!(media_calls.load(Ordering::SeqCst), 1);
    // The prompt was prefilled in one piece - `indivisible_prompt` kept the
    // pad run out of the planner's chunking.
    assert_eq!(
        audit.prefill_chunks(),
        vec![6],
        "{:?}",
        audit.prefill_chunks()
    );
    assert_eq!(
        audit.prefill_prompts.lock().clone(),
        vec![vec![1, PAD, PAD, PAD, PAD, 1]],
        "the model read the prompt with its pad run intact"
    );
    // And the answer is the scripted row's label log-probabilities, unchanged
    // by the media path.
    let norm = logsumexp(&label_row());
    let expected = [1.0 - norm, 2.0 - norm];
    for (got, want) in outcome.label_logprobs.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-5,
            "label logprob {got} is not {want}"
        );
    }

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn a_media_encode_failure_fails_the_score_reply() {
    let (item, digest) = one_picture();
    let (engine, join, _audit, media_calls) = media_engine(vec![label_row()], true);

    let outcome = engine
        .score(
            ScoreRequest::new(padded_prompt(), labels())
                .with_media(Arc::from(vec![item]), Arc::from(vec![digest])),
        )
        .await;

    // The tower's refusal rides the score channel as the reply's own error -
    // the failed slot retires and its still-held reply carries the failure -
    // and the prompt was never read.
    assert!(outcome.is_err(), "the encode failure must fail the reply");
    assert_eq!(media_calls.load(Ordering::SeqCst), 0);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn media_on_a_server_without_a_tower_is_the_callers_error() {
    let (item, digest) = one_picture();
    let (engine, join, audit, media_calls) =
        common::fake_media_engine_without_tower(vec![label_row()], Some(qwen_vocabulary()));

    let error = engine
        .score(
            ScoreRequest::new(padded_prompt(), labels())
                .with_media(Arc::from(vec![item]), Arc::from(vec![digest])),
        )
        .await
        .expect_err("a picture with no tower to look at it is refused");

    // Refused before a session opened, as a request the caller or the
    // operator can fix - not as a server fault - and naming the flag.
    assert!(matches!(error, AppError::BadRequest { .. }), "{error:?}");
    assert!(error.to_string().contains("--vision"), "{error}");
    assert_eq!(media_calls.load(Ordering::SeqCst), 0);
    assert!(audit.prefill_chunks().is_empty(), "nothing was read");

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn a_score_without_media_is_unchanged() {
    let (engine, join, audit, media_calls) = media_engine(vec![label_row()], false);

    let outcome = engine
        .score(ScoreRequest::new(
            Arc::from(vec![TokenId::new(1), TokenId::new(1)]),
            labels(),
        ))
        .await
        .expect("the score is answered");

    assert_eq!(
        media_calls.load(Ordering::SeqCst),
        0,
        "the plain path stayed plain"
    );
    assert_eq!(audit.prefill_chunks(), vec![2]);
    assert_eq!(outcome.label_logprobs.len(), 2);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

/// Three questions over one picture, as a decision request asks them: one
/// prompt each, every one opening with the same words and pad run.
fn sibling_prompts() -> Vec<Arc<[TokenId]>> {
    (0..3u32)
        .map(|question| {
            Arc::from(
                // Each tail its own word, clear of the pad id.
                [1, PAD, PAD, PAD, PAD, 1, 1, 10 + question]
                    .iter()
                    .copied()
                    .map(TokenId::new)
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

#[tokio::test]
async fn sibling_questions_over_one_keyed_picture_read_it_once() {
    let (item, digest) = one_picture();
    let (engine, join, _audit, media_calls) =
        common::fake_keyed_media_engine(vec![label_row(); 3], Some(qwen_vocabulary()));
    let items: Arc<[MediaItem]> = Arc::from(vec![item]);
    let digests: Arc<[shiso_core::media::MediaDigest]> = Arc::from(vec![digest]);

    let outcomes = futures::future::join_all(sibling_prompts().into_iter().map(|prompt| {
        engine.score(
            ScoreRequest::new(prompt, labels())
                .with_media(Arc::clone(&items), Arc::clone(&digests))
                .with_shared_prefix(7),
        )
    }))
    .await;
    let cached: Vec<u64> = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("every question is answered").cached_tokens)
        .collect();

    // The first question reads the picture and leaves the state after the
    // shared words filed under the picture's content; the other two stand on
    // it and read only their own tail - the picture is never read again.
    assert_eq!(cached, vec![0, 7, 7], "{cached:?}");
    assert_eq!(media_calls.load(Ordering::SeqCst), 1);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}

#[tokio::test]
async fn another_picture_under_the_same_pad_run_never_resumes_past_it() {
    let (item, digest) = one_picture();
    let (engine, join, _audit, media_calls) =
        common::fake_keyed_media_engine(vec![label_row(); 2], Some(qwen_vocabulary()));
    let items: Arc<[MediaItem]> = Arc::from(vec![item]);
    let prompt = sibling_prompts().remove(0);

    let first = engine
        .score(
            ScoreRequest::new(Arc::clone(&prompt), labels())
                .with_media(Arc::clone(&items), Arc::from(vec![digest]))
                .with_shared_prefix(7),
        )
        .await
        .expect("the first picture is answered");
    assert_eq!(first.cached_tokens, 0);

    // The same ids, pad run and all, under another picture's digest: what was
    // filed past the first picture must not serve this one. The word before
    // the run is the only prefix the two share.
    let other = shiso_core::media::MediaDigest::new([0xab; 32]);
    let second = engine
        .score(
            ScoreRequest::new(prompt, labels())
                .with_media(items, Arc::from(vec![other]))
                .with_shared_prefix(7),
        )
        .await
        .expect("the second picture is answered");
    assert!(
        second.cached_tokens <= 1,
        "resumed {} tokens of another picture's prefix",
        second.cached_tokens
    );
    assert_eq!(
        media_calls.load(Ordering::SeqCst),
        2,
        "the second picture was read"
    );

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
