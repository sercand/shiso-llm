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

use shiso_core::{Model, TokenId};
use shiso_engine::ckpt::Checkpointed;
use shiso_engine::{CkptConfig, CkptOptions, EngineOptions, Reason, SaveOutcome};
use shiso_kvstore::{Incoming, PathWords, SpillConfig, SpillStore};

#[path = "common/qwen35_fixture.rs"]
mod fixture;

use fixture::{
    ATTN_HEAD, CKPT_SLOTS, CTX, FakeBackend, KV_HEADS, LOGITS_ROW_BYTES, POOL_PAGES,
    RECURRENT_SLOTS, loaded, loaded_with, tokens,
};

/// The rendered text a stored entry is keyed by. Only the engine's caller knows what
/// a request rendered to, so it arrives as bytes and nothing here tokenizes it.
const PROMPT: &[u8] = b"<|im_start|>user\nThe capital of France is<|im_end|>\n";

/// Tokens the fixture prompt stands for. Eight, which clears the floor below.
const PROMPT_TOKENS: u32 = 8;

/// Bytes of one page inside one key/value allocation, which is the range a restore
/// writes into a cold pool: every position of a page, one f16 per key channel.
fn kv_run_bytes() -> usize {
    usize::try_from(shiso_core::page::PAGE_TOKENS * KV_HEADS * ATTN_HEAD * 2)
        .expect("the fixture page fits this host")
}

/// Bytes of the rendered prompt, as an entry's header counts them.
fn prompt_bytes() -> u64 {
    u64::try_from(PROMPT.len()).expect("the fixture prompt is short")
}

/// A tier that stores anything four tokens or longer.
///
/// The shipped floor is 512, which no fixture prompt could reach; lowering it is the
/// only thing these tests change about the policy, and it is the floor rather than
/// the mechanism.
fn ckpt_options() -> CkptOptions {
    let mut opts = CkptOptions::new();
    opts.min_tokens = 4;
    opts.budget_mb = 64;
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

/// Prefills the fixture prompt and stores the session at its frontier.
fn store_prefix(
    model: &mut shiso_models::qwen35::Qwen35Model,
    session: shiso_core::SessionId,
) -> SaveOutcome {
    let path = tokens(1, PROMPT_TOKENS);
    model
        .prefill_last(session, &path)
        .expect("the prompt prefills");
    model
        .kv_save(session, &path, PROMPT, Reason::Cold)
        .expect("the store answers")
}

#[test]
fn a_resumed_session_is_a_byte_image_of_the_one_that_was_saved() {
    let store = tempfile::TempDir::new().expect("a temporary directory");

    let (_dir, saved_device, mut saved, session, recurrent) = saver(store.path());
    let outcome = store_prefix(&mut saved, session);
    assert!(
        matches!(outcome, SaveOutcome::Written(_)),
        "the prefix is written: {outcome:?}"
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
        .kv_resume(fresh, PROMPT)
        .expect("the resume answers")
        .expect("the stored prefix is found");
    assert_eq!(
        info.tokens_reused, PROMPT_TOKENS,
        "the whole stored path is reused"
    );
    assert_eq!(
        info.text_bytes,
        prompt_bytes(),
        "the entry covers the whole rendered prompt"
    );
    assert_eq!(
        info.tokens,
        tokens(1, PROMPT_TOKENS),
        "the token path comes back as it was stored"
    );
    assert_eq!(info.frontier_logits.rows(), 1);
    let row_buffer = info.frontier_logits.view().id().get();

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
fn a_request_whose_text_is_not_the_stored_one_is_a_cold_start() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    assert!(store_prefix(&mut saved, session).stored().is_some());

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");

    // A prompt the stored entry is not a prefix of. The key is the hash of the text,
    // so a different rendering simply names a file that is not there.
    assert!(
        model
            .kv_resume(
                fresh,
                b"<|im_start|>user\nSomething else entirely<|im_end|>\n"
            )
            .expect("the resume answers")
            .is_none(),
        "a text the stored entry does not begin is a cold start"
    );
    // A text shorter than the stored prefix cannot have it as a prefix either.
    assert!(
        model
            .kv_resume(fresh, &PROMPT[..8])
            .expect("the resume answers")
            .is_none()
    );
}

#[test]
fn a_longer_stored_prefix_beats_a_shorter_one() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    // Two entries over the same conversation: a short one and the whole prompt.
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    let short_text = &PROMPT[..20];
    let short_path = tokens(1, 4);
    saved
        .prefill_chunk(session, &short_path)
        .expect("the first half prefills");
    // A chunk that is not the last one leaves no frontier row, so the short entry is
    // stored from its own session standing at its own frontier.
    let head = saved
        .kv_save(session, &short_path, short_text, Reason::Cold)
        .expect("the store answers");
    let tail_path = tokens(5, 4);
    saved
        .prefill_last(session, &tail_path)
        .expect("the rest prefills");
    let whole: Vec<TokenId> = tokens(1, PROMPT_TOKENS);
    let full = saved
        .kv_save(session, &whole, PROMPT, Reason::Cold)
        .expect("the store answers");
    assert!(
        full.stored().is_some(),
        "the whole prompt is stored: {full:?}"
    );

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");
    let info = model
        .kv_resume(fresh, PROMPT)
        .expect("the resume answers")
        .expect("a stored prefix is found");
    assert_eq!(
        info.tokens_reused, PROMPT_TOKENS,
        "the entry covering the most text is the one read, not the first one found"
    );
    // The short store is only interesting if it happened at all; a mid-prefill chunk
    // carries no frontier row, so it is refused for that and not silently skipped.
    assert!(
        matches!(
            head,
            SaveOutcome::NoFrontierLogits | SaveOutcome::Written(_)
        ),
        "{head:?}"
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
        cold.kv_save(session, &path, PROMPT, Reason::Cold)
            .expect("a disabled tier still answers"),
        SaveOutcome::Disabled,
        "a stack nobody gave a directory to says so rather than failing"
    );
    assert!(!cold.kv_checkpoint_enabled());
    assert!(
        cold.kv_resume(session, PROMPT)
            .expect("a disabled tier still answers")
            .is_none()
    );

    let (_dir, _device, mut model, session, _recurrent) = saver(store.path());
    assert!(model.kv_checkpoint_enabled());
    model
        .prefill_last(session, &path)
        .expect("the prompt prefills");

    // Not at the frontier: the right number of the wrong tokens, and the wrong number.
    let wrong = tokens(40, PROMPT_TOKENS);
    assert_eq!(
        model
            .kv_save(session, &wrong, PROMPT, Reason::Cold)
            .expect("the store answers"),
        SaveOutcome::NotAtFrontier {
            tokens: PROMPT_TOKENS,
            written: PROMPT_TOKENS
        },
        "a path the session is not standing on is refused even at the right length"
    );
    assert_eq!(
        model
            .kv_save(session, &path[..3], PROMPT, Reason::Cold)
            .expect("the store answers"),
        SaveOutcome::NotAtFrontier {
            tokens: 3,
            written: PROMPT_TOKENS
        }
    );

    // Empty text: nothing could ever name the entry.
    assert_eq!(
        model
            .kv_save(session, &path, b"", Reason::Cold)
            .expect("the store answers"),
        SaveOutcome::EmptyText
    );

    // Below the floor, which is the one intrinsic gate this tier owns.
    let (_dir, _device, mut strict, strict_session, _r) = saver(store.path());
    let mut high = ckpt_options();
    high.min_tokens = 4096;
    strict
        .kv_enable(&CkptConfig::new(store.path()).with_options(high))
        .expect_err("the tier cannot be reopened under a running session");
    let short = tokens(1, 2);
    strict
        .prefill_last(strict_session, &short)
        .expect("two tokens prefill");
    assert_eq!(
        strict
            .kv_save(strict_session, &short, PROMPT, Reason::Cold)
            .expect("the store answers"),
        SaveOutcome::BelowMinTokens { tokens: 2, min: 4 }
    );

    assert!(
        store
            .path()
            .read_dir()
            .expect("the directory exists")
            .count()
            == 0,
        "not one refusal left a file behind"
    );
}

#[test]
fn a_path_prefilled_under_a_larger_context_is_not_read_into_a_smaller_one() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    assert!(store_prefix(&mut saved, session).stored().is_some());

    // The same build, started with half the context. The stored path fits, but the
    // entry was written for a cache this one is not, so it is refused - and kept,
    // because the build it was written for can still read it.
    let (_dir_b, _device_b, mut small) = loaded_with(options_at(CTX / 2));
    small
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = small.open_session().expect("a session opens");
    assert!(
        small
            .kv_resume(fresh, PROMPT)
            .expect("the resume answers")
            .is_none(),
        "an entry written under a larger context is not read into a smaller one"
    );
    assert_eq!(
        store
            .path()
            .read_dir()
            .expect("the directory exists")
            .count(),
        1,
        "a disagreement with this configuration keeps the file"
    );
}

#[test]
fn a_payload_describing_a_stack_this_build_does_not_have_is_unlinked() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut saved, session, _recurrent) = saver(store.path());
    let id = store_prefix(&mut saved, session)
        .stored()
        .expect("the prefix is stored");

    // Rewrite the stored payload so its header claims a recurrent stack of a
    // different depth. Nothing about the envelope changes, so the scan still matches
    // it by name and the refusal has to come from the payload's own geometry.
    {
        let mut disk = SpillStore::open(&SpillConfig::new(store.path(), 64))
            .expect("the directory opens directly");
        let entry = disk
            .read(id)
            .expect("the entry reads")
            .expect("the entry is there");
        let mut payload = entry.payload.clone();
        let mut words: PathWords = bytemuck::pod_read_unaligned(&payload[..size_of::<PathWords>()]);
        words.n_gdn_layers += 1;
        payload[..size_of::<PathWords>()].copy_from_slice(bytemuck::bytes_of(&words));
        assert!(disk.forget(id).expect("the original is removed"));
        disk.write(&Incoming {
            text: &entry.text,
            tokens: entry.header.tokens,
            quant_bits: entry.header.quant_bits,
            model_id: entry.header.model_id,
            ctx_size: entry.header.ctx_size,
            reason: Reason::Cold,
            ext_flags: 0,
            payload: &payload,
        })
        .expect("the forged entry is written");
    }

    let (_dir_b, _device_b, mut model) = loaded();
    model
        .kv_enable(&CkptConfig::new(store.path()).with_options(ckpt_options()))
        .expect("the same directory opens");
    let fresh = model.open_session().expect("a session opens");
    assert!(
        model
            .kv_resume(fresh, PROMPT)
            .expect("the resume answers")
            .is_none(),
        "a payload shaped for another stack is a cold start"
    );
    assert_eq!(
        store
            .path()
            .read_dir()
            .expect("the directory exists")
            .count(),
        0,
        "a file this build can never read is unlinked rather than left to be refused again"
    );
}

#[test]
fn the_shutdown_sweep_stores_every_session_it_is_given_text_for() {
    let store = tempfile::TempDir::new().expect("a temporary directory");
    let (_dir, _device, mut model, first, _recurrent) = saver(store.path());
    let path = tokens(1, PROMPT_TOKENS);
    model.prefill_last(first, &path).expect("the prompt runs");

    let second = model.open_session().expect("a second session opens");
    let other = tokens(20, PROMPT_TOKENS);
    model.prefill_last(second, &other).expect("the prompt runs");

    let third = model.open_session().expect("a third session opens");
    model.prefill_last(third, &path).expect("the prompt runs");

    let mut text_of = |session| {
        if session == first {
            Some((path.clone(), PROMPT.to_vec()))
        } else if session == second {
            Some((
                other.clone(),
                b"<|im_start|>user\nsomething else<|im_end|>\n".to_vec(),
            ))
        } else {
            // A session the caller has no rendered text for is skipped, not failed:
            // the engine has no tokenizer and cannot invent one.
            None
        }
    };
    assert_eq!(
        model.kv_shutdown_sweep(&mut text_of),
        2,
        "the two sessions with text are stored and the third is skipped"
    );
    assert_eq!(
        store
            .path()
            .read_dir()
            .expect("the directory exists")
            .count(),
        2
    );
}

#[test]
fn a_sweep_without_a_directory_writes_nothing() {
    let (_dir, _device, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let path = tokens(1, PROMPT_TOKENS);
    model.prefill_last(session, &path).expect("the prompt runs");
    let mut text_of = |_| Some((path.clone(), PROMPT.to_vec()));
    assert_eq!(
        model.kv_shutdown_sweep(&mut text_of),
        0,
        "a stack with no checkpoint directory has nowhere to sweep to"
    );
}
