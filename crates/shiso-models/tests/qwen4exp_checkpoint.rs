//! Storing a session's prefix and standing a fresh one back up on it.
//!
//! Every store here is written behind: a commit queues the checkpoint, and the
//! cases drain the queue before they reopen the directory, which is what a
//! server does on its way down.
//!
//! The tier's policy - the floor, the prefix scan, the refusal vocabulary - is
//! the engine's and is tested there. What is tested here is the part that is
//! this family's own: that a blob carries everything a sparse resume cannot
//! reconstruct, that a resumed session is a byte image of the one that was
//! saved, and that a build which speculates and one which does not refuse to
//! read each other's files.
//!
//! # What the fake device makes possible
//!
//! The fixture's backend is a `HashMap` of byte vectors, so the device-to-host
//! reads a snapshot makes and the host-to-device writes a resume makes really
//! move bytes. A test can therefore write a recognisable pattern into a
//! session's state, store it, and assert that a *different* session standing up
//! from the file holds the same bytes - which is the only assertion that
//! distinguishes a resume from a zeroed buffer that answers anyway.
//!
//! # Nothing here has run on a GPU
//!
//! The bytes these cases move are bytes a test wrote, not bytes a kernel
//! produced: no kernel in this family has ever executed. What is asserted is the
//! bookkeeping around them.

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use std::time::Duration;

use shiso_core::Model as _;
use shiso_core::ids::{SessionId, TokenId};
use shiso_engine::ckpt::{Checkpointed as _, CkptConfig, CkptOptions, ResumeSource, SaveOutcome};
use shiso_engine::sparse::mtp::SpecDepth;
use shiso_engine::sparse::stack::SparseOptions;
use shiso_models::qwen4exp::{QWEN4EXP_MODEL_ID, Qwen4ExpModel};

/// Tokens the cases prefill, comfortably over the floor they lower.
const PROMPT: u32 = 64;

/// The floor the cases run under.
///
/// Lowered from the shipped default so a case can prefill a prompt in one chunk
/// on a fixture whose whole context is a thousand tokens. Lowering a floor is
/// not weakening a check: the floor is a cost decision about which files are
/// worth writing, and every other gate is left exactly as it ships.
fn config(dir: &std::path::Path) -> CkptConfig {
    let mut options = CkptOptions::new();
    options.min_tokens = 8;
    options.budget_mb = Some(256);
    options.staging_mb = 16;
    CkptConfig::new(dir.to_path_buf()).with_options(options)
}

/// Tokens `0..n`, which is a path no other case writes.
fn path_of(n: u32) -> Vec<TokenId> {
    (0..n).map(TokenId::new).collect()
}

/// A shared boundary the cases put a checkpoint at: one run of the
/// recurrence's grid in, because a boundary between two grid points is moved
/// back onto the grid before a chunk stops at it.
const BOUNDARY: u32 = 64;

/// `n` tokens cycling through the fixture's 64-id vocabulary, so a prompt can
/// run past a [`BOUNDARY`] without leaving it.
fn cycle_of(n: u32) -> Vec<TokenId> {
    (0..n).map(|i| TokenId::new(i % 64)).collect()
}

/// Every queued write lands.
fn drain(model: &mut Qwen4ExpModel) {
    assert!(
        model.kv_drain(Duration::from_secs(10)),
        "the writer caught up"
    );
}

/// Fills every recurrent slot, the n-gram window and - when the session has one -
/// the staged draft row with `value`.
fn paint(model: &Qwen4ExpModel, session: SessionId, value: u8) {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    let mut views = Vec::new();
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        views.push(conv);
        views.push(state);
    }
    views.extend(live.ngram_window());
    views.extend(live.pending_draft());
    for view in views {
        let bytes = vec![value; usize::try_from(view.len()).expect("a fixture range fits")];
        stack
            .backend()
            .copy_h2d(view, &bytes)
            .expect("the fake device accepts a write");
    }
}

/// Whether every piece of restorable state holds `want` end to end.
fn state_is(model: &Qwen4ExpModel, session: SessionId, want: u8) -> bool {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    let mut views = Vec::new();
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        views.push(conv);
        views.push(state);
    }
    views.extend(live.ngram_window());
    views.extend(live.pending_draft());
    views.into_iter().all(|view| {
        let mut bytes = vec![0_u8; usize::try_from(view.len()).expect("a fixture range fits")];
        stack
            .backend()
            .copy_d2h(view, &mut bytes)
            .expect("the fake device answers a read");
        bytes.iter().all(|byte| *byte == want)
    })
}

/// A loaded model with the tier open over `dir`, plus one session that ended a
/// prompt of [`PROMPT`] tokens with its state painted `value`, drained to disk.
fn stored(
    dir: &tempfile::TempDir,
    opts: SparseOptions,
    value: u8,
) -> (tempfile::TempDir, Qwen4ExpModel, SessionId) {
    let (weights, _backend, mut model) = fixture::loaded_with(opts);
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over a fresh directory");
    let session = model.open_session().expect("a session opens");
    let tokens = path_of(PROMPT);
    model
        .stack_mut()
        .reserve(session, PROMPT)
        .expect("the pool has room for the prompt");
    if opts.spec.is_speculative() {
        model
            .stack_mut()
            .stage_draft_row(session, 0)
            .expect("a speculative session stages its row");
    }
    paint(&model, session, value);
    // `true` is what says this chunk ended a prompt, and `Some(0)` names the
    // row a resume from here would sample. The commit queues the checkpoint for
    // disk on its own; a save afterwards finds it already queued.
    model
        .stack_mut()
        .advance(session, &tokens, true, Some(0))
        .expect("the span is registered");
    let outcome = model
        .kv_save(session, shiso_engine::Reason::Cold)
        .expect("the store accepted the save");
    assert!(
        matches!(outcome, SaveOutcome::AlreadyStored(_)),
        "the prompt's end was queued by its own commit: {outcome:?}"
    );
    drain(&mut model);
    (weights, model, session)
}

#[test]
fn a_stored_prefix_stands_a_fresh_session_up_on_the_bytes_it_was_saved_with() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, model, _session) = stored(&dir, fixture::options(), 0xA7);
    drop(model);

    let (_weights, _backend, mut fresh) = fixture::loaded_with(fixture::options());
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over the directory the first model wrote");
    let session = fresh.open_session().expect("a session opens");
    // A different pattern first, so a resume that wrote nothing would be visible
    // as the pattern this session started with rather than as zeros.
    paint(&fresh, session, 0x11);

    let reused = {
        let info = fresh
            .kv_resume(session, &path_of(PROMPT))
            .expect("the resume did not fail")
            .expect("the stored prefix was found");
        assert_eq!(info.tokens, path_of(PROMPT), "the token path came back");
        assert_eq!(info.tokens_reused, PROMPT);
        assert_eq!(info.source, ResumeSource::Disk);
        assert!(
            info.frontier_logits.is_some(),
            "a resume covering the whole prompt hands back the row to sample from"
        );
        info.tokens_reused
    };
    assert_eq!(reused, PROMPT);
    assert!(
        state_is(&fresh, session, 0xA7),
        "the resumed session holds the bytes the saved one held, not the ones it \
         started with"
    );
}

#[test]
fn a_resume_restores_the_staged_draft_row_and_not_only_the_recurrent_stack() {
    // The row the draft head reads on its first step. A resume that dropped it
    // would answer, and would answer differently on exactly that step - which is
    // the one a stored prefix exists to make cheap.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let opts = fixture::speculative_options(4);
    let (_weights, model, _session) = stored(&dir, opts, 0x5E);
    drop(model);

    let (_weights, _backend, mut fresh) = fixture::loaded_with(opts);
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = fresh.open_session().expect("a session opens");
    assert!(
        fresh
            .stack()
            .session(session)
            .expect("open")
            .pending_draft()
            .is_none(),
        "a fresh session has staged nothing"
    );
    fresh
        .kv_resume(session, &path_of(PROMPT))
        .expect("the resume did not fail")
        .expect("the stored prefix was found");
    let stack = fresh.stack();
    let live = stack.session(session).expect("open");
    assert!(
        live.staged_row_current(),
        "the restored row drafts from the resumed frontier"
    );
    let row = live
        .pending_draft()
        .expect("the resumed session can draft on its first step");
    let mut bytes = vec![0_u8; usize::try_from(row.len()).expect("a fixture range fits")];
    stack
        .backend()
        .copy_d2h(row, &mut bytes)
        .expect("the fake device answers a read");
    assert!(bytes.iter().all(|byte| *byte == 0x5E));
}

#[test]
fn a_prompt_that_shares_an_opening_with_an_earlier_one_is_still_stored() {
    // Every rendered prompt shares at least its template's opening with the
    // prompts before it, so the cache's branch point lies behind its end. The
    // end of a prompt is admitted anyway: it is what the same prompt, or the
    // next turn of its conversation, resumes from.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over a fresh directory");
    let first = model.open_session().expect("a session opens");
    model
        .prefill_last(first, &cycle_of(2 * BOUNDARY))
        .expect("the first prompt prefills");
    let second = model.open_session().expect("a second session opens");
    let mut tokens = cycle_of(BOUNDARY);
    tokens.extend((0..BOUNDARY).map(|i| TokenId::new(63 - i % 64)));
    model
        .prefill_last(second, &tokens)
        .expect("the second prompt prefills");
    let outcome = model
        .kv_save(second, shiso_engine::Reason::Cold)
        .expect("the store answered");
    assert!(
        outcome.stored().is_some(),
        "the second prompt was not stored: {outcome:?}"
    );
    drain(&mut model);
    let stats = model
        .kv_cache_stats()
        .and_then(|stats| stats.disk)
        .expect("a disk tier is open");
    // Both prompt ends, and the point the second prompt branched off the first.
    assert_eq!(stats.entries, 3, "{stats:?}");
}

#[test]
fn a_build_that_speculates_will_not_read_a_file_written_by_one_that_does_not() {
    // The two blobs are different lengths, because the draft row is in one and
    // not the other, and the layout word that says so is part of every key: the
    // speculating build never even finds the other build's entry.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, model, _session) = stored(&dir, fixture::options(), 0x22);
    drop(model);

    let (_weights, _backend, mut speculating) =
        fixture::loaded_with(fixture::speculative_options(4));
    speculating
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = speculating.open_session().expect("a session opens");
    assert!(
        speculating
            .kv_resume(session, &path_of(PROMPT))
            .expect("a refusal to read is a cold start, not a failure")
            .is_none(),
        "a file whose state section is a different length was read anyway"
    );
}

/// Bytes of one raw indexer key ring row, mirroring `IDX_DIM` f16 elements.
const RING_ROW: usize = 128 * 2;
/// Rows of one rank's raw key ring, mirroring `kernels::IDX_RING`.
const RING_ROWS: usize = 16;

/// Writes `value(rank, slot)` into every byte of every row of the session's raw
/// key ring.
fn paint_ring(model: &Qwen4ExpModel, session: SessionId, value: impl Fn(usize, usize) -> u8) {
    let stack = model.stack();
    let ring = stack
        .session(session)
        .expect("the session is open")
        .indexer_rings();
    let len = usize::try_from(ring.len()).expect("a fixture range fits");
    let bytes = (0..len)
        .map(|at| {
            let row = at / RING_ROW;
            value(row / RING_ROWS, row % RING_ROWS)
        })
        .collect::<Vec<_>>();
    stack
        .backend()
        .copy_h2d(ring, &bytes)
        .expect("the fake device accepts a write");
}

#[test]
fn a_resume_restores_the_raw_keys_of_the_frontier_block_and_zeros_every_other_ring_row() {
    // The page carries no raw indexer key, so the blob is the only place the
    // raw keys of the block a frontier stands inside survive. At 67 tokens an
    // attention rank stands after 64, 65 and 66 - ring slots 0, 1 and 2 - and a
    // draft head, which has written only up to the position before the
    // frontier, after 64 and 65. Everything else in the ring is dead and comes
    // back as zeros, so two checkpoints of one state are one byte image.
    const AT: u32 = 67;
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let opts = fixture::speculative_options(4);
    {
        let (_weights, _backend, mut model) = fixture::loaded_with(opts);
        model
            .kv_enable(&config(dir.path()))
            .expect("the tier opens over a fresh directory");
        let session = model.open_session().expect("a session opens");
        model
            .stack_mut()
            .reserve(session, AT)
            .expect("the pool has room for the prompt");
        model
            .stack_mut()
            .stage_draft_row(session, 0)
            .expect("a speculative session stages its row");
        paint(&model, session, 0x3C);
        paint_ring(&model, session, |rank, slot| {
            u8::try_from(1 + rank * RING_ROWS + slot).expect("a few rows")
        });
        model
            .stack_mut()
            .advance(session, &path_of(AT), true, Some(0))
            .expect("the span is registered");
        drain(&mut model);
    }

    let (_weights, _backend, mut fresh) = fixture::loaded_with(opts);
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over the directory the first model wrote");
    let session = fresh.open_session().expect("a session opens");
    // A pattern no live row holds, so a restore that skipped the ring - or
    // skipped zeroing the dead rows - shows.
    paint_ring(&fresh, session, |_, _| 0xEE);
    let info = fresh
        .kv_resume(session, &path_of(AT))
        .expect("the resume did not fail")
        .expect("the stored prefix was found");
    assert_eq!(info.tokens_reused, AT);

    let stack = fresh.stack();
    // The fixture's checkpoint carries no head, so its ring is the attention
    // slots' alone; a head's rank, the last, is covered by the engine's own
    // `RingLayout` cases.
    let head = stack.has_draft_head();
    let ring = stack.session(session).expect("open").indexer_rings();
    let mut bytes = vec![0_u8; usize::try_from(ring.len()).expect("a fixture range fits")];
    stack
        .backend()
        .copy_d2h(ring, &mut bytes)
        .expect("the fake device answers a read");
    let ranks = bytes.len() / (RING_ROW * RING_ROWS);
    assert_eq!(ranks, 2 + usize::from(head), "one ring per attention slot");
    for (row, chunk) in bytes.chunks_exact(RING_ROW).enumerate() {
        let (rank, slot) = (row / RING_ROWS, row % RING_ROWS);
        let live = if head && rank + 1 == ranks {
            0..2
        } else {
            0..3
        };
        let want = if live.contains(&slot) {
            u8::try_from(1 + row).expect("a few rows")
        } else {
            0
        };
        assert!(
            chunk.iter().all(|byte| *byte == want),
            "rank {rank} slot {slot} came back {:#04x}, expected {want:#04x}",
            chunk[0]
        );
    }
}

#[test]
fn a_finished_turn_is_stored_at_the_last_position_it_wrote() {
    // The next turn's prompt is this one plus the answer plus a new message, so
    // the checkpoint worth having is at the end of the answer - not the end of
    // the prompt, and not one past it: the last sampled token was never run.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let answered = {
        let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
        model
            .kv_enable(&config(dir.path()))
            .expect("the tier opens");
        let session = model.open_session().expect("a session opens");
        model
            .prefill_last(session, &path_of(PROMPT))
            .expect("the prompt prefills");
        let mut path = path_of(PROMPT);
        for step in 0..5u32 {
            let token = TokenId::new(500 + step);
            model
                .stack_mut()
                .reserve(session, 1)
                .expect("the pool has room for a row");
            model
                .stack_mut()
                .advance(session, &[token], false, Some(0))
                .expect("the step is registered");
            path.push(token);
        }
        paint(&model, session, 0x3C);
        let outcome = model
            .kv_save(session, shiso_engine::Reason::Continued)
            .expect("the store answered");
        assert!(
            matches!(outcome, SaveOutcome::Queued(_)),
            "the turn's end was queued: {outcome:?}"
        );
        drain(&mut model);
        path
    };

    let (_weights, _backend, mut fresh) = fixture::loaded_with(fixture::options());
    fresh
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let session = fresh.open_session().expect("a session opens");
    let mut next_turn = answered.clone();
    next_turn.extend((900..910).map(TokenId::new));
    let info = fresh
        .kv_resume(session, &next_turn)
        .expect("not an error")
        .expect("the turn's end was found");
    assert_eq!(
        info.tokens_reused,
        u32::try_from(answered.len()).expect("short"),
        "the resume reaches the end of the answer"
    );
    assert!(
        info.frontier_logits.is_none(),
        "a resume short of the prompt's end prefills the rest and needs no row"
    );
    assert!(state_is(&fresh, session, 0x3C));
}

#[test]
fn a_message_boundary_earns_a_checkpoint_a_shorter_prompt_resumes_from() {
    // A system prompt shared by every session of an agent ends at a message
    // boundary. The first prompt through it leaves a checkpoint there, so the
    // second - which shares nothing past it - still skips it.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let first = model.open_session().expect("a session opens");
    model
        .kv_set_hints(first, &[BOUNDARY])
        .expect("the session is open");
    model
        .prefill_last(first, &cycle_of(2 * BOUNDARY))
        .expect("the prompt prefills");
    let second = model.open_session().expect("a second session opens");
    let mut other = cycle_of(BOUNDARY);
    other.extend((700..740).map(TokenId::new));
    let info = model
        .kv_resume(second, &other)
        .expect("not an error")
        .expect("the boundary's checkpoint was found");
    assert_eq!(info.tokens_reused, BOUNDARY);
    assert_eq!(info.source, ResumeSource::Memory);
}

#[test]
fn the_point_a_prompt_branches_off_a_cached_path_earns_a_checkpoint() {
    // Same opening, different question: the second prompt stops a chunk where
    // it leaves the first one's path, and a third that shares that opening
    // resumes there.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let first = model.open_session().expect("a session opens");
    model
        .prefill_last(first, &cycle_of(2 * BOUNDARY))
        .expect("the first prompt prefills");
    let with_question = |question: u32| {
        let mut tokens = cycle_of(BOUNDARY);
        // The fixture's vocabulary is 64 ids wide.
        tokens.extend((0..20).map(|i| TokenId::new((question + i) % 64)));
        tokens
    };
    let second = model.open_session().expect("a second session opens");
    model
        .prefill_last(second, &with_question(50))
        .expect("the second prompt prefills");
    let third = model.open_session().expect("a third session opens");
    let info = model
        .kv_resume(third, &with_question(10))
        .expect("not an error")
        .expect("the branch point's checkpoint was found");
    assert_eq!(info.tokens_reused, BOUNDARY);
}

#[test]
fn nothing_is_stored_until_a_directory_is_opened() {
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    assert!(!model.kv_checkpoint_enabled());
    let session = model.open_session().expect("a session opens");
    assert_eq!(
        model
            .kv_save(session, shiso_engine::Reason::Cold)
            .expect("asking is not an error"),
        SaveOutcome::Disabled,
        "a process that was never told where to put a cache must not invent a place"
    );
    assert!(
        model
            .kv_resume(session, &path_of(PROMPT))
            .expect("asking is not an error")
            .is_none()
    );
}

#[test]
fn the_tier_is_chosen_before_anything_stands_on_a_session() {
    // A session holds page references into the pool the old cache owns, and
    // turning the tier on builds a new one.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    let _session = model.open_session().expect("a session opens");
    let error = model
        .kv_enable(&config(dir.path()))
        .expect_err("the tier cannot open under a live session");
    assert!(
        format!("{error:?}").contains("sessions are already running"),
        "the refusal names why: {error:?}"
    );
}

#[test]
fn a_shutdown_sweep_stores_every_open_session() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    for first in [0u32, 1000] {
        let session = model.open_session().expect("a session opens");
        let tokens: Vec<TokenId> = (first..first + PROMPT).map(TokenId::new).collect();
        model
            .stack_mut()
            .reserve(session, PROMPT)
            .expect("the pool has room");
        model
            .stack_mut()
            .advance(session, &tokens, true, Some(0))
            .expect("the span is registered");
    }
    assert_eq!(model.kv_shutdown_sweep(), 2);
}

#[test]
fn this_family_tags_its_files_with_its_own_identity() {
    // The tag is part of every key, and the quantization is read off the
    // weights rather than stated by a caller.
    let (_weights, _backend, model) = fixture::loaded_with(fixture::options());
    let identity = model.ckpt_identity();
    assert_eq!(identity.model_id, QWEN4EXP_MODEL_ID);
    assert_eq!(
        identity.quant_bits,
        shiso_models::qwen4exp::quant_bits_of(model.expert_gate_dtype())
    );
    assert_ne!(
        SpecDepth::NONE.get(),
        1,
        "a sanity check on the depth newtype"
    );
}

/// A decode step produces a logits row and ends no prompt, and the difference
/// is not one the commit can work out for itself.
///
/// A snapshot is the whole recurrent stack read back to the host, one
/// device-to-host copy per slot. Reading "this span is a boundary worth
/// resuming from" off "this span produced a logits row" makes every decoded
/// token take one - and because a snapshot changes no answer, nothing that
/// asserts on output can see it. So the assertion is on the reads.
#[test]
fn a_span_that_ends_no_prompt_is_not_snapshotted() {
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, backend, mut model) = fixture::loaded_with(SparseOptions::new());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens over a fresh directory");
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, PROMPT)
        .expect("the pool has room for the prompt");
    model
        .stack_mut()
        .advance(session, &path_of(PROMPT), true, Some(0))
        .expect("the prompt's last chunk is registered");
    let after_prompt = backend.read_count();
    assert!(
        after_prompt > 0,
        "a prompt's last chunk took no snapshot at all"
    );

    // Four steps, each with the row the sampler reads and none of them a
    // boundary a later request would resume from.
    for step in 0..4u32 {
        model
            .stack_mut()
            .reserve(session, 1)
            .expect("the pool has room for a row");
        model
            .stack_mut()
            .advance(session, &[TokenId::new(PROMPT + step)], false, Some(0))
            .expect("the step is registered");
    }
    assert_eq!(
        backend.read_count(),
        after_prompt,
        "a decode step read the session's state back off the device"
    );
}

#[test]
fn a_prompt_that_ends_off_the_grid_is_checkpointed_on_the_grid_beneath_its_end() {
    // A prompt of two grid runs and a bit more: its final pass stops between
    // two grid points, so its checkpoint is taken at the last one it passed,
    // inside the pass. A longer prompt continues from there as a cold prefill
    // of it would; the same prompt again re-reads the few positions past it.
    let dir = tempfile::TempDir::new().expect("a cache directory");
    let (_weights, _backend, mut model) = fixture::loaded_with(fixture::options());
    model
        .kv_enable(&config(dir.path()))
        .expect("the tier opens");
    let prompt = cycle_of(2 * BOUNDARY + 36);
    let first = model.open_session().expect("a session opens");
    model
        .prefill_last(first, &prompt)
        .expect("the prompt prefills");

    let longer = {
        let mut tokens = prompt.clone();
        tokens.extend(cycle_of(20));
        tokens
    };
    let second = model.open_session().expect("a second session opens");
    let info = model
        .kv_resume(second, &longer)
        .expect("not an error")
        .expect("the grid checkpoint was found");
    assert_eq!(
        info.tokens_reused,
        2 * BOUNDARY,
        "a longer prompt resumes on the grid"
    );

    let third = model.open_session().expect("a third session opens");
    let info = model
        .kv_resume(third, &prompt)
        .expect("not an error")
        .expect("the grid checkpoint was found");
    assert_eq!(
        info.tokens_reused,
        2 * BOUNDARY,
        "a repeat resumes on the grid too"
    );
    assert!(
        info.frontier_logits.is_none(),
        "short of the prompt's end, the rest is prefilled and no row is handed back"
    );
}
