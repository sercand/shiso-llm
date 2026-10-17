//! The model's lifecycle, proven against a device that is a `HashMap`.
//!
//! Everything here runs on every machine: the model is loaded from a synthetic
//! file the shared fixture writes, and the device under it records what it was
//! asked to do instead of doing it. That is enough to settle the questions the
//! forward pass cannot answer for itself - which buffers a session owns, when
//! they come back, what a session says when it runs out of context - because all
//! of them are decided by bookkeeping the host does before any kernel reads a
//! byte.
//!
//! See `common/fixture.rs` for the device, the synthetic checkpoint and why both
//! are shaped the way they are.

use shiso_core::model::DecodeStep;
use shiso_core::{CoreError, Model, TokenId};

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use fixture::{ATTN_SLOTS, CTX, HIDDEN, N_LAYERS, RECURRENT_SLOTS, VOCAB, loaded, tokens};
use shiso_engine::sparse::stack::MAX_DECODE_BATCH;

#[test]
fn a_loaded_model_reports_the_five_numbers_a_scheduler_reads() {
    let (_dir, _fake, model) = loaded();
    let info = model.info();
    assert_eq!(info.n_layers, N_LAYERS);
    assert_eq!(info.hidden, HIDDEN);
    assert_eq!(info.vocab, VOCAB);
    assert_eq!(info.context_capacity, CTX);
    assert_eq!(info.stop_tokens.primary().get(), VOCAB - 1);
}

#[test]
fn a_session_owns_its_recurrent_stack_and_its_n_gram_window_and_closing_it_hands_them_back() {
    let (_dir, fake, mut model) = loaded();

    let mark = fake.allocation_mark();
    let session = model.open_session().expect("a session opens");
    let owned = fake.allocated_since(mark);

    // A convolution window and a state matrix per recurrent slot, the n-gram
    // block's own dilated window, the raw indexer key ring, and one page
    // table. Nothing else is per session: the key/value rows and the indexer's
    // pooled plane belong to the shared pool, which is why opening a session
    // is kilobytes rather than a cache.
    assert_eq!(
        owned.len(),
        2 * RECURRENT_SLOTS + 3,
        "a session allocates two buffers per recurrent slot, an n-gram window, a raw key ring \
         and a page table"
    );
    assert!(
        owned.iter().all(|id| fake.is_live(*id)),
        "every buffer a session claimed is live while it is open"
    );

    model.close_session(session).expect("the session closes");
    assert!(
        owned.iter().all(|id| !fake.is_live(*id)),
        "closing a session gives every buffer it owned back"
    );
    assert_eq!(fake.double_frees(), 0, "no buffer was released twice");

    let err = model
        .close_session(session)
        .expect_err("a closed session is no longer known");
    assert!(
        matches!(err, CoreError::UnknownSession(id) if id == session),
        "{err:?}"
    );
    assert_eq!(fake.double_frees(), 0, "a second close frees nothing twice");
}

#[test]
fn the_cache_pool_allocates_three_planes_for_every_attention_slot() {
    let (_dir, fake, model) = loaded();
    // Keys, values and the indexer's pooled key rows; the raw keys are each
    // session's ring, not the pool's. All three have to be present for a span
    // to be reusable, because a block that resumed on a span without its
    // pooled keys would select blocks from rows another session wrote.
    let live = fake.live_count();
    assert!(
        live >= 3 * ATTN_SLOTS,
        "{live} live allocations is fewer than the three planes of {ATTN_SLOTS} attention slots"
    );
    drop(model);
}

#[test]
fn a_prompt_past_the_context_is_refused_by_the_session_that_cannot_hold_it() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let err = model
        .stack_mut()
        .reserve(session, CTX + 1)
        .expect_err("a chunk wider than the context");
    assert!(
        matches!(err, CoreError::ContextOverflow { cap, .. } if cap == CTX),
        "{err:?}"
    );
}

#[test]
fn reserving_rows_claims_pages_and_says_the_table_was_republished() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    // The first claim moves a frontier that held nothing, so the table changes.
    assert!(
        model
            .stack_mut()
            .reserve(session, 4)
            .expect("the pool has pages"),
        "the first claim publishes a table"
    );
    // Asking for the same rows again claims nothing new, so the table the
    // kernels read is the one they were already handed - which is what lets a
    // captured launch keep naming it.
    assert!(
        !model
            .stack_mut()
            .reserve(session, 4)
            .expect("the pool has pages"),
        "a second claim of rows the session already holds republishes nothing"
    );
    model
        .stack_mut()
        .advance(session, &tokens(0, 4), false, None)
        .expect("the path grows");
}

#[test]
fn teardown_releases_every_allocation_and_a_second_one_is_harmless() {
    let (_dir, fake, mut model) = loaded();
    let a = model.open_session().expect("a session opens");
    let b = model.open_session().expect("a second session opens");
    assert_ne!(a, b, "two sessions are two handles");

    model.teardown().expect("the model tears down");
    assert_eq!(
        fake.live_count(),
        0,
        "teardown releases the weights, the pool, the scratch and every session"
    );
    assert_eq!(fake.double_frees(), 0);
    model.teardown().expect("a second teardown reports nothing");
    assert_eq!(fake.double_frees(), 0);
}

#[test]
fn every_execution_path_reaches_the_device() {
    // The counterpart of what this test used to assert. It used to say that
    // nothing was ever dispatched, because the two paths below were a typed
    // refusal; now they issue a pass, and the thing worth pinning is that they
    // reach the device at all rather than finding a second quiet refusal
    // somewhere below.
    let (_dir, fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    model
        .prefill_chunk(session, &tokens(0, 4))
        .expect("a prompt chunk runs");
    let after_prefill = fake.launch_count();
    assert!(after_prefill > 0, "a prefill chunk issued nothing");

    model
        .prefill_last(session, &tokens(4, 4))
        .expect("the prompt's final chunk runs");
    assert!(
        fake.launch_count() > after_prefill,
        "the final chunk issued nothing"
    );
}

#[test]
fn a_chunk_that_cannot_be_planned_issues_nothing() {
    // The order the pass is run in is what this asserts: the plan comes before
    // the first launch, so a chunk whose geometry is wrong leaves the device
    // untouched instead of half-run.
    let (_dir, fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let err = model
        .prefill_chunk(session, &[])
        .expect_err("an empty chunk is not a chunk");
    assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    assert_eq!(fake.launch_count(), 0);
}

#[test]
fn the_execution_paths_no_longer_refuse_and_what_is_left_is_a_bound_not_a_gap() {
    // This test has always been for one property: what still refuses names a
    // gap that is really still open, and names none of the ones that have been
    // closed. Every gap it used to name was a buffer *nothing wrote* - the
    // streamed n-gram rows, a session's recurrent slots, and then the per-row
    // pointer arrays the batched kernels dereference. All three are written, so
    // the first half of this test is that the paths run, a prompt and a batch
    // of two unrelated sequences alike.
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    model
        .prefill_chunk(session, &tokens(0, 4))
        .expect("a prompt chunk runs");
    let second = model.open_session().expect("a second session opens");
    let logits = model
        .decode_batch(&[
            DecodeStep {
                session,
                token: TokenId::new(1),
            },
            DecodeStep {
                session: second,
                token: TokenId::new(2),
            },
        ])
        .expect("two unrelated sequences decode in one pass");
    assert_eq!(logits.rows(), 2);
    assert_eq!(logits.vocab(), VOCAB);

    // What is left refuses a *width* rather than a missing buffer: the logits
    // block is as tall as a batch may be wide, and a ninth row would have
    // nowhere to be written. So the message may name the block, and may not
    // claim that anything is ungathered, uncleared or otherwise left to nobody.
    let wide = (0..=MAX_DECODE_BATCH)
        .map(|index| DecodeStep {
            session,
            token: TokenId::new(index + 1),
        })
        .collect::<Vec<_>>();
    let err = model
        .decode_batch(&wide)
        .expect_err("a batch taller than the logits block has nowhere to write its last row");
    assert!(
        matches!(err, CoreError::Unsupported { what, .. }
            if what.contains("logits block")
                && !what.contains("nothing")
                && !what.contains("gathers")
                && !what.contains("cleared")
                && !what.contains("per-row")
                && !what.contains("token id")
                && !what.contains("rope-table")),
        "{err:?}"
    );
}

#[test]
fn a_prompt_and_the_decode_step_after_it_each_return_one_row_and_advance_the_session() {
    // The whole of the execution call path, stated as the two things a caller
    // reads: a logits block of the shape the sampler expects, and a session
    // that now stands on the tokens it was given. The session is advanced
    // *after* the pass, so a chunk that failed would leave it standing on a
    // prefix something really computed.
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let prompt = tokens(0, 5);
    let logits = model
        .prefill_last(session, &prompt)
        .expect("the prompt's final chunk runs");
    assert_eq!(logits.rows(), 1);
    assert_eq!(logits.vocab(), VOCAB);
    assert_eq!(
        model
            .stack()
            .session(session)
            .expect("the session is live")
            .written(),
        5
    );

    let logits = model
        .decode_batch(&[DecodeStep {
            session,
            token: TokenId::new(7),
        }])
        .expect("a decode step runs");
    assert_eq!(logits.rows(), 1);
    assert_eq!(
        model
            .stack()
            .session(session)
            .expect("the session is live")
            .written(),
        6,
        "a decode step is one more token of this session's path"
    );
}

#[test]
fn a_prompt_wider_than_one_pass_is_split_and_the_session_stands_on_all_of_it() {
    // The split is the stack's, not the caller's, and it is invisible in the
    // result: the recurrent state, the n-gram window and the cache carry the
    // prefix across an internal boundary exactly as across a caller's.
    let (_dir, fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let wide = shiso_engine::sparse::stack::MAX_PREFILL_CHUNK + 7;
    let prompt = tokens(0, wide);
    model
        .prefill_last(session, &prompt)
        .expect("a prompt wider than one pass runs");
    assert_eq!(
        model
            .stack()
            .session(session)
            .expect("the session is live")
            .written(),
        wide
    );
    assert!(fake.launch_count() > 0);
}
