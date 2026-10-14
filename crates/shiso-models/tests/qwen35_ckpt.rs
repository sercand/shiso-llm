//! The checkpoint tier end to end, on a device that is a `HashMap`.
//!
//! What a checkpoint promises is not "a resume works" but something stricter: the
//! session that comes back is a *byte image* of the one that went in. Nothing along
//! the path interprets a number - the recurrent state is a copy, the key/value pages
//! are a copy, and the frontier logits row is the row the original run produced - so
//! the assertion is byte equality and not a tolerance.
//!
//! # How a fake device makes that provable
//!
//! Every buffer is a byte vector the test can read and write directly. So the
//! recurrent stack and the pool are filled with a seeded pattern *before* the prefill
//! that snapshots them, which is what makes "equal after a resume" a statement about
//! the payload rather than about two buffers that were both zero.
//!
//! # Why there are two models
//!
//! One saves and a second, over its own fresh device, resumes. A resume back into the
//! same process would compare a pool against bytes that were never cleared out of it,
//! which proves nothing at all. The two are loaded from identical synthetic files with
//! identical options, so their devices hand out the same buffer ids in the same order
//! and a buffer in one has a counterpart in the other.

use std::collections::HashMap;
use std::time::Duration;

use shiso_core::Model;
use shiso_engine::ckpt::Checkpointed;
use shiso_engine::{CkptConfig, CkptOptions, EngineOptions, PersistRefusal, Reason, SaveOutcome};

#[path = "common/qwen35_fixture.rs"]
mod fixture;

use fixture::{
    ATTN_HEAD, CKPT_SLOTS, CTX, FakeBackend, KV_HEADS, LOGITS_ROW_BYTES, POOL_PAGES,
    RECURRENT_SLOTS, loaded, loaded_with, tokens,
};

/// Tokens the fixture prompt holds. Eight, which clears the floor below.
const PROMPT_TOKENS: u32 = 8;

/// Bytes of one page inside one key/value allocation, which is the range a restore
/// writes into a cold pool: every position of a page, one f16 per key channel.
fn kv_run_bytes() -> usize {
    usize::try_from(shiso_core::page::PAGE_TOKENS * KV_HEADS * ATTN_HEAD * 2)
        .expect("the fixture page fits this host")
}

/// Entry files the store holds.
fn entries(store: &std::path::Path) -> usize {
    let Ok(fans) = store.join("ckpt").read_dir() else {
        return 0;
    };
    fans.flatten()
        .filter_map(|fan| fan.path().read_dir().ok())
        .map(Iterator::count)
        .sum()
}

/// Every queued write lands.
fn drain(model: &mut shiso_models::qwen35::Qwen35Model) {
    assert!(
        model.kv_drain(Duration::from_secs(10)),
        "the writer caught up"
    );
}

/// A tier that stores anything four tokens or longer.
///
/// The shipped floor is 512, which no fixture prompt could reach; lowering it is the
/// only thing these tests change about the policy, and it is the floor rather than
/// the mechanism.
fn ckpt_options() -> CkptOptions {
    let mut opts = CkptOptions::new();
    opts.min_tokens = 4;
    opts.budget_mb = Some(64);
    opts.staging_mb = 16;
    opts
}

/// Options with a context of `ctx` tokens.
fn options_at(ctx: u32) -> EngineOptions {
    let mut opts = EngineOptions::default();
    opts.ctx = ctx;
    opts.kv_pool_pages = POOL_PAGES;
    opts.kv_ckpt_slots = CKPT_SLOTS;
    opts
}

/// A model with its checkpoint tier open over `store`, its whole device filled with a
/// pattern, and one session whose recurrent stack is filled with another.
///
/// Returns the model, its device, the session, and the ids of the recurrent buffers
/// the session owns - which are the ones a resume has to reproduce.
fn saver(
    store: &std::path::Path,
) -> (
    tempfile::TempDir,
    std::sync::Arc<FakeBackend>,
    shiso_models::qwen35::Qwen35Model,
    shiso_core::SessionId,
    Vec<u64>,
) {
    let (dir, fake, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store).with_options(ckpt_options()))
        .expect("the checkpoint directory opens");

    // Everything the load allocated, including the key/value pool: a payload that
    // dropped the pages would otherwise restore into a pool that was already right.
    for (seed, id) in fake.live_ids().into_iter().enumerate() {
        fake.scribble(
            id,
            0x5eed_0000 + u64::try_from(seed).expect("a small index"),
        );
    }

    let mark = fake.allocation_mark();
    let session = model.open_session().expect("a session opens");
    let owned = fake.allocated_since(mark);
    // The conv window and state matrix of every recurrent slot, in the order
    // `Session::open` allocates them. The page table is the entry after them and is
    // deliberately left alone: it is uploaded by the engine, not carried by a payload.
    let recurrent: Vec<u64> = owned[..2 * RECURRENT_SLOTS].to_vec();
    for (seed, id) in recurrent.iter().enumerate() {
        fake.scribble(
            *id,
            0xf00d_0000 + u64::try_from(seed).expect("a small index"),
        );
    }
    (dir, fake, model, session, recurrent)
}

/// Prefills the fixture prompt - whose last chunk queues its own checkpoint -
/// asks for the frontier to be stored, and drains the queue.
fn store_prefix(
    model: &mut shiso_models::qwen35::Qwen35Model,
    session: shiso_core::SessionId,
) -> SaveOutcome {
    let path = tokens(1, PROMPT_TOKENS);
    model
        .prefill_last(session, &path)
        .expect("the prompt prefills");
    let outcome = model
        .kv_save(session, Reason::Cold)
        .expect("the store answers");
    drain(model);
    outcome
}

#[test]
fn a_resumed_session_is_a_byte_image_of_the_one_that_was_saved() {
    let store = tempfile::TempDir::new().expect("a temporary directory");

    let (_dir, saved_device, mut saved, session, recurrent) = saver(store.path());
    let outcome = store_prefix(&mut saved, session);
    assert!(
        matches!(outcome, SaveOutcome::AlreadyStored(_)),
        "the prompt's end was queued by its own commit: {outcome:?}"
    );

    // A second model over its own device, which has never run anything.
    let (_dir_b, device, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let before: HashMap<u64, Vec<u8>> = device
        .live_ids()
        .into_iter()
        .map(|id| (id, device.contents(id)))
        .collect();
    let mark = device.allocation_mark();
    let fresh = model.open_session().expect("a session opens");
    let owned = device.allocated_since(mark);

    let info = model
        .kv_resume(fresh, &tokens(1, PROMPT_TOKENS))
        .expect("the resume answers")
        .expect("the stored prefix is found");
    assert_eq!(
        info.tokens_reused, PROMPT_TOKENS,
        "the whole stored path is reused"
    );
    assert_eq!(
        info.tokens,
        tokens(1, PROMPT_TOKENS),
        "the token path comes back as it was stored"
    );
    let row = info
        .frontier_logits
        .expect("a resume of the whole prompt carries its row");
    assert_eq!(row.rows(), 1);
    let row_buffer = row.view().id().get();

    // The recurrent stack: the Decision that the payload carries it, as a test. If
    // anyone drops the conv windows or the state matrices from the payload, this is
    // what fails, and it fails loudly rather than by degrading an answer.
    for (from, into) in recurrent.iter().zip(owned[..2 * RECURRENT_SLOTS].iter()) {
        assert_eq!(
            device.contents(*into),
            saved_device.contents(*from),
            "a recurrent buffer came back with different bytes than it went in with"
        );
    }

    // The frontier logits row, which is why no step is re-run on a resume.
    let width = usize::try_from(LOGITS_ROW_BYTES).expect("a fixture row fits this host");
    assert_eq!(
        device.contents(row_buffer)[..width],
        saved_device.contents(row_buffer)[..width],
        "the row a resumed session stands on is the row the saved run produced"
    );

    // The key/value pages. Everything the load allocated is still exactly as it was
    // except the buffers the restore wrote into: the pool, and the logits row above.
    let touched: Vec<u64> = before
        .iter()
        .filter(|(id, was)| device.contents(**id) != **was)
        .map(|(id, _)| *id)
        .filter(|id| *id != row_buffer)
        .collect();
    assert!(
        !touched.is_empty(),
        "a restore that wrote no page bytes restored no key/value rows"
    );
    for id in touched {
        assert_eq!(
            device.contents(id)[..kv_run_bytes()],
            saved_device.contents(id)[..kv_run_bytes()],
            "a restored page does not hold the bytes that were stored for it"
        );
    }
}

#[test]
fn a_prompt_the_stored_path_does_not_begin_is_a_cold_start() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    assert!(store_prefix(&mut saved, session).stored().is_some());

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");

    // A prompt the stored path is not a prefix of: its keys name other files.
    assert!(
        model
            .kv_resume(fresh, &tokens(40, PROMPT_TOKENS))
            .expect("the resume answers")
            .is_none(),
        "a prompt the stored entry does not begin is a cold start"
    );
    // A prompt shorter than the stored path cannot resume from its end either.
    assert!(
        model
            .kv_resume(fresh, &tokens(1, 5))
            .expect("the resume answers")
            .is_none()
    );
}

#[test]
fn a_longer_stored_prefix_beats_a_shorter_one() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    // Two entries over the same conversation: a short one and the whole prompt.
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    let short_path = tokens(1, 4);
    saved
        .prefill_chunk(session, &short_path)
        .expect("the first half prefills");
    // Stored at the frontier, whether the chunk's own commit already queued
    // it or the save has to take the checkpoint itself.
    let head = saved
        .kv_save(session, Reason::Cold)
        .expect("the store answers");
    assert!(head.stored().is_some(), "{head:?}");
    saved
        .prefill_last(session, &tokens(5, 4))
        .expect("the rest prefills");
    drain(&mut saved);
    assert_eq!(entries(store.path()), 2);

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");
    let info = model
        .kv_resume(fresh, &tokens(1, PROMPT_TOKENS))
        .expect("the resume answers")
        .expect("a stored prefix is found");
    assert_eq!(
        info.tokens_reused, PROMPT_TOKENS,
        "the deepest entry on the path is the one read"
    );
}

#[test]
fn every_refusal_says_which_one_it_is() {
    let store = tempfile::TempDir::new().expect("a temporary directory");

    // Disabled: no directory was ever opened.
    let (_dir, _device, mut cold) = loaded();
    let session = cold.open_session().expect("a session opens");
    let path = tokens(1, PROMPT_TOKENS);
    cold.prefill_last(session, &path).expect("the prompt runs");
    assert_eq!(
        cold.kv_save(session, Reason::Cold)
            .expect("a disabled tier still answers"),
        SaveOutcome::Disabled,
        "a stack nobody gave a directory to says so rather than failing"
    );
    assert!(!cold.kv_checkpoint_enabled());
    // Nothing persists without a directory, but the checkpoints a commit
    // admits are kept in memory whether or not one is open, so a resume is
    // still answered - from memory, never from a disk tier this stack lacks.
    let source = cold
        .kv_resume(session, &path)
        .expect("a disabled tier still answers")
        .map(|info| info.source);
    assert!(
        source.is_none_or(|source| source == shiso_engine::ckpt::ResumeSource::Memory),
        "a stack with no directory resumes from memory or not at all: {source:?}"
    );

    // Below the floor, which is the one intrinsic gate this tier owns.
    let (_dir, _device, mut model, session, _recurrent) = saver(store.path());
    let short = tokens(1, 2);
    model
        .prefill_last(session, &short)
        .expect("two tokens prefill");
    assert_eq!(
        model
            .kv_save(session, Reason::Cold)
            .expect("the store answers"),
        SaveOutcome::Refused(PersistRefusal::BelowMinTokens { tokens: 2, min: 4 })
    );
    drain(&mut model);
    assert_eq!(
        entries(store.path()),
        0,
        "not one refusal left a file behind"
    );
}

#[test]
fn a_path_prefilled_under_a_larger_context_is_not_read_into_a_smaller_one() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    assert!(store_prefix(&mut saved, session).stored().is_some());

    // The same build, started with half the context. The context is a layout
    // word, so the entry is named for a build this one is not: it is never
    // found, and it is kept, because the build it was written for can read it.
    let (_dir_b, _device_b, mut small) = loaded_with(options_at(CTX / 2));
    small
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = small.open_session().expect("a session opens");
    assert!(
        small
            .kv_resume(fresh, &tokens(1, PROMPT_TOKENS))
            .expect("the resume answers")
            .is_none(),
        "an entry written under a larger context is not read into a smaller one"
    );
    assert_eq!(entries(store.path()), 1, "the other build's file is kept");
}

#[test]
fn a_corrupt_entry_is_a_cold_start_and_is_unlinked() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    assert!(store_prefix(&mut saved, session).stored().is_some());
    drop(saved);

    // Flip one byte of the state section: the checksum no longer adds up.
    for fan in store.path().join("ckpt").read_dir().expect("the directory") {
        for file in fan.expect("a fan").path().read_dir().expect("files") {
            let path = file.expect("a file").path();
            let mut bytes = std::fs::read(&path).expect("readable");
            let last = bytes.len() - 1;
            bytes[last] ^= 0x5A;
            std::fs::write(&path, bytes).expect("writable");
        }
    }

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");
    assert!(
        model
            .kv_resume(fresh, &tokens(1, PROMPT_TOKENS))
            .expect("the resume answers")
            .is_none(),
        "a corrupt entry is a cold start"
    );
    drain(&mut model);
    assert_eq!(
        entries(store.path()),
        0,
        "a file this build can never read is unlinked rather than left to be refused again"
    );
}

#[test]
fn the_shutdown_sweep_stores_every_open_session() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut model, first, _recurrent) = saver(store.path());
    let path = tokens(1, PROMPT_TOKENS);
    model.prefill_last(first, &path).expect("the prompt runs");

    let second = model.open_session().expect("a second session opens");
    let other = tokens(20, PROMPT_TOKENS);
    model.prefill_last(second, &other).expect("the prompt runs");

    let third = model.open_session().expect("a third session opens");
    model.prefill_last(third, &path).expect("the prompt runs");

    // The third stands on the first one's path: already stored, and counted.
    assert_eq!(model.kv_shutdown_sweep(), 3);
    drain(&mut model);
    assert_eq!(entries(store.path()), 2);
}

#[test]
fn a_sweep_without_a_directory_writes_nothing() {
    let (_dir, _device, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let path = tokens(1, PROMPT_TOKENS);
    model.prefill_last(session, &path).expect("the prompt runs");
    assert_eq!(
        model.kv_shutdown_sweep(),
        0,
        "a stack with no checkpoint directory has nowhere to sweep to"
    );
}

/// A plain conversation stops at a message end inside a chunk, exactly where it
/// falls and whatever the tier's floor, so its next turn resumes there: the
/// shipped floor is as wide as a chunk, and a floor counted from the chunk's
/// start would leave no message end inside one that could ever clear it.
#[test]
fn a_conversation_stops_at_its_message_end_under_the_shipped_floor() {
    // Off the recurrence's run grid on purpose: the stop is the boundary itself.
    const TURN: u32 = 13;
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _fake, mut model) = loaded();
    let mut opts = ckpt_options();
    opts.min_tokens = CkptOptions::new().min_tokens;
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(opts))
        .expect("the checkpoint directory opens");

    let first = model.open_session().expect("a session opens");
    model
        .kv_set_hints(first, &[TURN])
        .expect("the boundary is recorded");
    model
        .prefill_last(first, &tokens(1, 40))
        .expect("the first turn reads");

    let mut next = tokens(1, TURN);
    next.extend(tokens(200, 20));
    let second = model.open_session().expect("a session opens");
    let reused = model
        .kv_resume(second, &next)
        .expect("the resume answers")
        .map_or(0, |info| info.tokens_reused);
    assert_eq!(reused, TURN, "the next turn stands on the message end");
}
