//! A `ScoreRequest` carrying media: the encode step runs on admission, the
//! rows reach the prefill, and the reply is answered exactly once either way.
//!
//! The generation path already encodes a request's media before the slot reads
//! its prompt (`open` runs `encode_media`); scoring must mirror it, because a
//! score slot that reached the prefill with pad tokens and no rows would read
//! placeholders as text. The fakes here are the minimum that can tell those
//! two paths apart: a model that records WHICH prefill entry point it was
//! asked for, and a tower that either hands over rows sized to the item's grid
//! or refuses.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use base64::Engine as _;
use serde_json::{Value, json};
use shiso_core::media::{MediaDigest, VisionGrid};
use shiso_core::{
    Backend, CoreError, CoreResult, DecodeStep, Logits, MediaModel, Model, ModelInfo, SessionId,
    TokenId, VisionPrefill,
};
use shiso_engine::PersistRefusal;
use shiso_engine::Reason;
use shiso_engine::ckpt::{Checkpointed, CkptConfig, ResumeInfo, SaveOutcome};
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_server::media::{MediaItem, MediaTowers};
use shiso_server::scheduler::{
    EngineHandle, SchedulerConfig, ScoreRequest, spawn_engine_with_checkpoints,
};
use shiso_server::tokenizer::Vocabulary;
use shiso_server::tokenizer::message_preprocess::{MediaPolicy, preprocess};
use shiso_vision::{
    EncodedRows, ImageLimits, MediaPlacement, PatchGeometry, PreprocessSpec, VisionError,
    VisionItem,
};
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

/// A one-token-spec geometry: a 96x96 picture is a 6x6 patch grid pooled 3x3
/// into four soft tokens, so the prompt's pad run is four long and the
/// indivisible claim is about more than one token.
fn spec() -> PreprocessSpec {
    PreprocessSpec::qwen35(PatchGeometry::new(16, 1, 3).expect("a 3x3-pooled 16-pixel patch"))
}

/// A solid-colour PNG, as the data URI a request would carry.
fn png_data_uri(width: u32, height: u32) -> String {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb([40, 90, 160]));
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(image)
        .write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("a solid-colour PNG encodes");
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes)
    )
}

/// One decoded picture and its digest, through the same preprocess the
/// renderer runs - so the item's grid and the prompt's pad run cannot drift.
fn one_picture() -> (MediaItem, MediaDigest) {
    let spec = spec();
    let message = Message::new(
        Role::User,
        vec![Block::Image(MediaSource::classify(png_data_uri(96, 96)))],
    );
    let policy = MediaPolicy {
        vision: Some(&spec),
        limits: ImageLimits::default(),
        ..MediaPolicy::default()
    };
    let prepared =
        preprocess(&[], std::slice::from_ref(&message), &policy).expect("the picture preprocesses");
    assert_eq!(prepared.media.len(), 1);
    assert_eq!(prepared.pad_counts, vec![4], "a 96x96 picture is four pads here");
    (
        prepared.media.into_iter().next().expect("one item"),
        prepared.digests.into_iter().next().expect("one digest"),
    )
}

/// A tower that hands over rows sized to each item's own post-merge grid -
/// the contract the splice leans on - or refuses every encode.
#[derive(Debug)]
struct FakeTower {
    spec: PreprocessSpec,
    fail: bool,
}

impl shiso_vision::VisionTower for FakeTower {
    fn preprocess(&self) -> &PreprocessSpec {
        &self.spec
    }

    fn placement(&self) -> MediaPlacement {
        MediaPlacement::QWEN35
    }

    fn out_hidden(&self) -> u32 {
        // The fake model's residual width; the rows are `[pads, hidden]`.
        8
    }

    fn device_bytes(&self) -> u64 {
        0
    }

    fn encode(
        &self,
        backend: &Arc<dyn Backend>,
        items: &[VisionItem],
    ) -> Result<EncodedRows, VisionError> {
        if self.fail {
            return Err(VisionError::Encode {
                step: "fake",
                source: CoreError::Unsupported {
                    what: "the fake tower was told to refuse",
                    backend: "fake",
                },
            });
        }
        let merge = self.spec.geometry.spatial_merge_size().max(1);
        let mut grids = Vec::with_capacity(items.len());
        let mut pads = 0u64;
        for item in items {
            let (t, gh, gw) = item.grid();
            let grid = VisionGrid::new(
                u32::try_from(t).unwrap_or(u32::MAX),
                u32::try_from(gh / merge).unwrap_or(u32::MAX),
                u32::try_from(gw / merge).unwrap_or(u32::MAX),
            );
            pads += u64::from(grid.pad_count());
            grids.push(grid);
        }
        let rows = shiso_core::alloc(backend, pads * 8 * 4).expect("the fake allocates");
        Ok(EncodedRows { rows, grids })
    }
}

/// The scripted model with a media path: `prefill_media` is counted, and both
/// entry points record through the shared audit like `common::FakeModel` does.
#[derive(Debug)]
struct FakeMediaModel {
    inner: common::FakeModel,
    audit: Arc<common::Audit>,
    media_calls: Arc<AtomicUsize>,
}

impl Model for FakeMediaModel {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn open_session(&mut self) -> CoreResult<SessionId> {
        self.inner.open_session()
    }

    fn close_session(&mut self, session: SessionId) -> CoreResult<()> {
        self.inner.close_session(session)
    }

    fn prefill_chunk(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        self.inner.prefill_chunk(session, tokens)
    }

    fn prefill_last(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<Logits<'_>> {
        self.inner.prefill_last(session, tokens)
    }

    fn decode_batch(&mut self, steps: &[DecodeStep]) -> CoreResult<Logits<'_>> {
        self.inner.decode_batch(steps)
    }

    fn teardown(&mut self) -> CoreResult<()> {
        self.inner.teardown()
    }

    fn media_mut(&mut self) -> Option<&mut dyn MediaModel> {
        Some(self)
    }
}

impl MediaModel for FakeMediaModel {
    fn prefill_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        _media: &VisionPrefill<'_>,
    ) -> CoreResult<Logits<'_>> {
        self.media_calls.fetch_add(1, Ordering::SeqCst);
        // The scripted row answers through the plain entry point, which also
        // records the prompt: a media read and a text read differ by the
        // counter, not by what the audit's prompt list shows.
        self.inner.prefill_last(session, tokens)
    }
}

impl Checkpointed for FakeMediaModel {
    fn kv_enable(&mut self, _config: &CkptConfig) -> CoreResult<()> {
        Ok(())
    }

    fn kv_checkpoint_enabled(&self) -> bool {
        false
    }

    fn kv_resume(
        &mut self,
        _session: SessionId,
        _prompt: &[TokenId],
    ) -> CoreResult<Option<ResumeInfo<'_>>> {
        Ok(None)
    }

    fn kv_save(&mut self, _session: SessionId, _reason: Reason) -> CoreResult<SaveOutcome> {
        Ok(SaveOutcome::Refused(PersistRefusal::NoCheckpoint))
    }

    fn kv_set_hints(&mut self, _session: SessionId, _hints: &[u32]) -> CoreResult<()> {
        Ok(())
    }

    fn kv_shutdown_sweep(&mut self) -> usize {
        0
    }
}

/// An engine over the media-capable fake, with `fail_tower` deciding whether
/// the vision tower encodes or refuses. The media-call counter and the audit
/// come back; the model does not - it lives on the engine thread now.
fn media_engine(
    rows: Vec<Vec<f32>>,
    fail_tower: bool,
) -> (
    EngineHandle,
    std::thread::JoinHandle<Result<(), shiso_server::scheduler::EngineFailure>>,
    Arc<common::Audit>,
    Arc<AtomicUsize>,
) {
    let backend: Arc<dyn Backend> = common::FakeBackend::new();
    let (model, audit) = common::FakeModel::new(&backend, Vec::new());
    let media_calls = Arc::new(AtomicUsize::new(0));
    let model = FakeMediaModel {
        inner: model.with_prefill_rows(rows),
        audit: Arc::clone(&audit),
        media_calls: Arc::clone(&media_calls),
    };
    let tower = shiso_server::vision::VisionTower::from_tower(Box::new(FakeTower {
        spec: spec(),
        fail: fail_tower,
    }));
    let (engine, join) = spawn_engine_with_checkpoints(
        backend,
        Box::new(model),
        MediaTowers {
            vision: Some(tower),
            ..MediaTowers::default()
        },
        Some(qwen_vocabulary()),
        "fake-model",
        common::CAPACITY,
        SchedulerConfig::new(),
        (
            shiso_parsers::ReasoningScheme::None,
            shiso_parsers::ToolCallScheme::None,
        ),
    );
    (engine, join, audit, media_calls)
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
    assert_eq!(audit.prefill_chunks(), vec![6], "{:?}", audit.prefill_chunks());
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
async fn a_score_without_media_is_unchanged() {
    let (engine, join, audit, media_calls) = media_engine(vec![label_row()], false);

    let outcome = engine
        .score(ScoreRequest::new(
            Arc::from(vec![TokenId::new(1), TokenId::new(1)]),
            labels(),
        ))
        .await
        .expect("the score is answered");

    assert_eq!(media_calls.load(Ordering::SeqCst), 0, "the plain path stayed plain");
    assert_eq!(audit.prefill_chunks(), vec![2]);
    assert_eq!(outcome.label_logprobs.len(), 2);

    drop(engine);
    join.join()
        .expect("the engine thread does not panic")
        .expect("the loop exits cleanly");
}
