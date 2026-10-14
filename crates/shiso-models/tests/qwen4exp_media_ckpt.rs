//! Sibling questions over one picture, through the sparse stack's checkpoint
//! tier, on a device that is a `HashMap`.
//!
//! A decision request asks each question from its own prompt, every one
//! opening with the same text and picture. The scheduler keys each by the
//! picture's content, marks it as scoring, and names two boundaries: where the
//! picture starts - what the agent's next step, with a new picture, stands on -
//! and where the questions part, just past the picture. The first question's
//! read has to leave a checkpoint at the second of those, or every sibling
//! reads the picture again. That boundary sits a few tokens past the picture,
//! so the grid point beneath it lies inside the picture: the stop is taken at
//! the boundary itself rather than dropped.

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use shiso_core::Model as _;
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::media::{
    MediaDigest, MediaKey, MediaModel as _, MediaSpan, PadTokens, VisionGrid, VisionPrefill,
};
use shiso_core::{Backend, alloc};
use shiso_engine::ckpt::{Checkpointed as _, CkptConfig, CkptOptions};
use shiso_models::qwen4exp::Qwen4ExpModel;

/// The placeholder the fixture's picture is written in: the vocabulary's last
/// id, which no text token below uses.
const IMG: TokenId = TokenId::new(63);
/// Text ahead of the picture; its start is off the recurrence's grid.
const LEAD: u32 = 70;
/// The picture: eight by eight placeholders.
const SIDE: u32 = 8;
/// Where the questions part: two tokens past the picture.
const SPLIT: u32 = LEAD + SIDE * SIDE + 2;

fn grid() -> VisionGrid {
    VisionGrid::new(1, SIDE, SIDE)
}

fn span() -> MediaSpan {
    MediaSpan::new(LEAD, grid())
}

/// The text, the picture, two shared tokens, and `question`'s own tail.
fn prompt(question: u32) -> Vec<TokenId> {
    let mut tokens: Vec<TokenId> = (0..LEAD).map(|i| TokenId::new(i % 60)).collect();
    tokens.extend(std::iter::repeat_n(IMG, (SIDE * SIDE) as usize));
    tokens.extend([TokenId::new(1), TokenId::new(2)]);
    tokens.extend((0..10).map(|i| TokenId::new((question * 7 + i) % 60)));
    tokens
}

/// `prompt(question)` with the picture's run written in its key ids.
fn key_path(question: u32) -> Vec<TokenId> {
    let digest = MediaDigest::new([3; 32]);
    let mut path = prompt(question);
    let start = LEAD as usize;
    for (k, slot) in path[start..start + (SIDE * SIDE) as usize]
        .iter_mut()
        .enumerate()
    {
        *slot = digest.key_id(u32::try_from(k).expect("a small index"));
    }
    path
}

fn config(dir: &std::path::Path) -> CkptConfig {
    let mut options = CkptOptions::new();
    options.min_tokens = 8;
    options.budget_mb = Some(256);
    options.staging_mb = 16;
    CkptConfig::new(dir.to_path_buf()).with_options(options)
}

/// Marks `session` as the scheduler marks a keyed sibling question.
fn as_sibling(model: &mut Qwen4ExpModel, session: SessionId, question: u32) {
    let path = key_path(question);
    let spans = [span()];
    model
        .kv_set_media_key(
            session,
            &prompt(question),
            MediaKey::new(&path, &spans).expect("a key that rewrites only its run"),
        )
        .expect("the sparse stack files a media prompt under its content");
    model
        .kv_set_scoring(session)
        .expect("the session is marked as scoring");
    model
        .kv_set_hints(session, &[LEAD, SPLIT])
        .expect("the boundaries are recorded");
}

#[test]
fn a_sibling_question_resumes_past_the_shared_picture() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");

    let first = model.open_session().expect("a session opens");
    as_sibling(&mut model, first, 0);
    let backend: std::sync::Arc<dyn Backend> = backend;
    let rows = alloc(
        &backend,
        u64::from(SIDE * SIDE) * u64::from(fixture::HIDDEN) * 4,
    )
    .expect("an encoder block");
    let grids = [grid()];
    let media = VisionPrefill::new(
        rows.view(),
        &grids,
        PadTokens::new(Some(IMG), None),
        fixture::HIDDEN,
    )
    .expect("a block whose rows match the picture");
    model
        .prefill_media(first, &prompt(0), &media)
        .expect("the first question reads the picture");

    let second = model.open_session().expect("a session opens");
    as_sibling(&mut model, second, 1);
    let reused = model
        .kv_resume(second, &prompt(1))
        .expect("the resume answers")
        .map_or(0, |info| info.tokens_reused);
    assert!(
        reused >= span().end(),
        "a sibling must stand past the shared picture, not re-read it: resumed {reused} of a \
         picture ending at {}",
        span().end()
    );
}
