//! Prompts that carry a picture, through the hybrid stack's checkpoint tier and
//! its rotation, on a device that is a `HashMap`.
//!
//! Two promises are settled here. The cache files a picture's prompt under the
//! picture's content and never under its placeholder run, which is one id
//! whatever the picture is: a prompt keyed by one picture is stored and resumed
//! past it, a different picture under the same placeholders resumes nothing,
//! and an unkeyed one is never stored. And a session standing past a picture -
//! because it read one or because a resume stood it there - rotates every later
//! token on the three-stream rule, where its rotary position trails its
//! sequence position; a text-only session never leaves the one-dimensional
//! rule. Which rotation a pass reached is what the launch log shows; whether
//! the rotation's arithmetic is right is the kernel cases' business.

use shiso_core::media::{MediaDigest, MediaKey, MediaSpan, PadTokens, VisionGrid, VisionPrefill};
use shiso_core::{Backend, DecodeStep, Model, SessionId, TokenId, alloc};
use shiso_engine::ckpt::Checkpointed;
use shiso_engine::{CkptConfig, CkptOptions, Reason, SaveOutcome};

#[path = "common/qwen35_fixture.rs"]
mod fixture;

use fixture::{FakeBackend, HIDDEN, loaded, tokens};

/// The placeholder the fixture's pictures are rendered with.
const IMG: TokenId = TokenId::new(100);
/// Text tokens ahead of the picture.
const LEAD: u32 = 6;

/// A four-by-four picture: sixteen placeholders that advance the rotary
/// stream by four, so text after it trails its sequence position by twelve.
fn grid() -> VisionGrid {
    VisionGrid::new(1, 4, 4)
}

/// Text, the picture's run, text.
fn prompt() -> Vec<TokenId> {
    let mut prompt = tokens(1, LEAD);
    prompt.extend(std::iter::repeat_n(IMG, 16));
    prompt.extend(tokens(20, 4));
    prompt
}

/// The picture's span in [`prompt`].
fn span() -> MediaSpan {
    MediaSpan::new(LEAD, grid())
}

/// The key path of [`prompt`] for the picture whose digest is `seed`
/// repeated: the run written in that picture's key ids.
fn key_path(seed: u8) -> Vec<TokenId> {
    let digest = MediaDigest::new([seed; 32]);
    let mut path = prompt();
    let start = usize::try_from(LEAD).expect("a small offset");
    for (k, slot) in path[start..start + 16].iter_mut().enumerate() {
        *slot = digest.key_id(u32::try_from(k).expect("a small index"));
    }
    path
}

/// A tier that stores anything four tokens or longer; the shipped floor is
/// more than a fixture prompt holds.
fn options() -> CkptOptions {
    let mut opts = CkptOptions::new();
    opts.min_tokens = 4;
    opts.budget_mb = Some(64);
    opts.staging_mb = 16;
    opts
}

/// Reads [`prompt`] into `session` with the picture spliced, keyed by
/// `seed`'s picture when there is one.
fn read_picture(
    fake: &std::sync::Arc<FakeBackend>,
    model: &mut shiso_models::qwen35::Qwen35Model,
    session: SessionId,
    key: Option<u8>,
) {
    if let Some(seed) = key {
        let path = key_path(seed);
        let spans = [span()];
        let key = MediaKey::new(&path, &spans).expect("a key that rewrites only its run");
        model
            .kv_set_media_key(session, &prompt(), key)
            .expect("the hybrid stack files a media prompt under its content");
    }
    let backend: std::sync::Arc<dyn Backend> = fake.clone();
    let rows = alloc(&backend, 16 * u64::from(HIDDEN) * 4).expect("an encoder block");
    let grids = [grid()];
    let media = VisionPrefill::new(rows.view(), &grids, PadTokens::new(Some(IMG), None), HIDDEN)
        .expect("a block whose rows match the picture");
    model
        .media_mut()
        .expect("the family has a media path")
        .prefill_media(session, &prompt(), &media)
        .expect("the picture's prompt reads");
}

/// Whether the launches since `mark` reached the three-stream rotation.
fn rotated_on_three_streams(fake: &FakeBackend, mark: usize) -> bool {
    fake.launches_since(mark)
        .iter()
        .any(|launch| launch.entry.ends_with("_mrope"))
}

#[test]
fn a_keyed_picture_prompt_is_stored_and_resumed_past_its_picture() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, fake, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(options()))
        .expect("the checkpoint directory opens");

    let first = model.open_session().expect("a session opens");
    read_picture(&fake, &mut model, first, Some(7));
    let outcome = model
        .kv_save(first, Reason::Cold)
        .expect("the store answers");
    assert!(
        !matches!(outcome, SaveOutcome::CarriesMedia),
        "a keyed picture prompt is stored like a text one: {outcome:?}"
    );

    let second = model.open_session().expect("a session opens");
    let path = key_path(7);
    let spans = [span()];
    model
        .kv_set_media_key(
            second,
            &prompt(),
            MediaKey::new(&path, &spans).expect("the same picture's key"),
        )
        .expect("the key is accepted before anything is read");
    let reused = model
        .kv_resume(second, &prompt())
        .expect("the resume answers")
        .map(|info| info.tokens_reused)
        .expect("the same picture resumes");
    assert!(
        reused >= span().end(),
        "the resume stands past the picture, at {reused}"
    );

    // The resumed session never read the picture itself; its rotation past
    // it comes from the shift the resume recomputed from the key.
    let mark = fake.launch_mark();
    model
        .decode_batch(&[DecodeStep {
            session: second,
            token: TokenId::new(3),
        }])
        .expect("a decode step runs");
    assert!(
        rotated_on_three_streams(&fake, mark),
        "a token after a resumed picture rotates where its rotary stream stands"
    );
}

#[test]
fn another_picture_under_the_same_placeholders_resumes_nothing_past_the_text() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, fake, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(options()))
        .expect("the checkpoint directory opens");

    let first = model.open_session().expect("a session opens");
    read_picture(&fake, &mut model, first, Some(7));
    model
        .kv_save(first, Reason::Cold)
        .expect("the store answers");

    let second = model.open_session().expect("a session opens");
    let path = key_path(8);
    let spans = [span()];
    model
        .kv_set_media_key(
            second,
            &prompt(),
            MediaKey::new(&path, &spans).expect("another picture's key"),
        )
        .expect("the key is accepted");
    let reused = model
        .kv_resume(second, &prompt())
        .expect("the resume answers")
        .map_or(0, |info| info.tokens_reused);
    assert!(
        reused <= span().start(),
        "the same placeholder ids must not serve another picture's rows: resumed {reused}"
    );
}

#[test]
fn an_unkeyed_picture_prompt_is_never_stored() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, fake, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(options()))
        .expect("the checkpoint directory opens");

    let session = model.open_session().expect("a session opens");
    read_picture(&fake, &mut model, session, None);
    let outcome = model
        .kv_save(session, Reason::Cold)
        .expect("the store answers");
    assert!(
        matches!(outcome, SaveOutcome::CarriesMedia),
        "a placeholder run cannot tell two pictures apart: {outcome:?}"
    );
}

#[test]
fn text_after_a_picture_rotates_on_three_streams_and_a_text_session_never_does() {
    let (_dir, fake, mut model) = loaded();

    let pictured = model.open_session().expect("a session opens");
    read_picture(&fake, &mut model, pictured, None);
    let mark = fake.launch_mark();
    model
        .prefill_chunk(pictured, &tokens(30, 3))
        .expect("more text reads");
    assert!(
        rotated_on_three_streams(&fake, mark),
        "a text chunk after a picture rotates at its trailing rotary position"
    );
    let mark = fake.launch_mark();
    model
        .decode_batch(&[DecodeStep {
            session: pictured,
            token: TokenId::new(3),
        }])
        .expect("a decode step runs");
    assert!(
        rotated_on_three_streams(&fake, mark),
        "so does every decoded token"
    );

    let text = model.open_session().expect("a session opens");
    model
        .prefill_last(text, &tokens(1, 8))
        .expect("a text prompt reads");
    let mark = fake.launch_mark();
    model
        .decode_batch(&[DecodeStep {
            session: text,
            token: TokenId::new(3),
        }])
        .expect("a decode step runs");
    assert!(
        !rotated_on_three_streams(&fake, mark),
        "a session that read no picture keeps the one-dimensional rotation"
    );
}

/// Sibling questions over one picture, as a decision request asks them: the
/// first read cuts where the picture ends, the siblings resume there, and
/// their text tails read together in one batched pass that rotates on the
/// three-stream rule, as each one's solo read would.
#[test]
fn sibling_questions_stand_at_the_pictures_end_and_read_their_tails_together() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, fake, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(options()))
        .expect("the checkpoint directory opens");
    // Where the questions part: two tokens past the picture, so the plain
    // grid point below it lies inside the picture.
    let split = span().end() + 2;
    let mark_sibling = |model: &mut shiso_models::qwen35::Qwen35Model, session| {
        model
            .kv_set_scoring(session)
            .expect("the session is marked as scoring");
        model
            .kv_set_hints(session, &[LEAD, split])
            .expect("the boundaries are recorded");
    };

    let first = model.open_session().expect("a session opens");
    mark_sibling(&mut model, first);
    read_picture(&fake, &mut model, first, Some(7));

    let path = key_path(7);
    let spans = [span()];
    let mut siblings = Vec::new();
    for _ in 0..2 {
        let session = model.open_session().expect("a session opens");
        model
            .kv_set_media_key(
                session,
                &prompt(),
                MediaKey::new(&path, &spans).expect("the same picture's key"),
            )
            .expect("the key is accepted before anything is read");
        mark_sibling(&mut model, session);
        let reused = model
            .kv_resume(session, &prompt())
            .expect("the resume answers")
            .map_or(0, |info| info.tokens_reused);
        assert_eq!(
            reused,
            span().end(),
            "a sibling stands where the shared picture ends"
        );
        siblings.push(session);
    }

    let tail = &prompt()[span().end() as usize..];
    let runs: Vec<shiso_core::PromptRun<'_>> = siblings
        .iter()
        .map(|session| shiso_core::PromptRun {
            session: *session,
            tokens: tail,
        })
        .collect();
    let mark = fake.launch_mark();
    let logits = model
        .prefill_last_many(&runs)
        .expect("siblings past their picture read their tails in one pass");
    assert_eq!(logits.rows(), 2, "one answer row per sibling");
    assert!(
        rotated_on_three_streams(&fake, mark),
        "a tail past a picture rotates at its trailing rotary position"
    );
}
