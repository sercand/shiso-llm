//! The model's lifecycle, proven against a device that is a `HashMap`.
//!
//! Everything here runs on every machine: the model is loaded from a synthetic file
//! the shared fixture writes, and the device under it records what it was asked to do
//! instead of doing it. That is enough to settle the questions the forward pass cannot
//! answer for itself - which buffers a session owns, when they come back, which row of
//! a decode batch belongs to which session, and what a session says when it runs out
//! of context - because all four are decided by bookkeeping the host does before any
//! kernel reads a byte.
//!
//! See `common/fixture.rs` for the device, the synthetic checkpoint and why both are
//! shaped the way they are.

use std::collections::HashSet;
use std::sync::Arc;

use tempfile::TempDir;

use shiso_core::{Backend, CoreError, DecodeStep, Model, TokenId};
use shiso_models::qwen35::Qwen35Model;

#[path = "common/qwen35_fixture.rs"]
mod fixture;

use fixture::{
    CTX, FakeBackend, LOGITS_ROW_BYTES, Launch, RECURRENT_SLOTS, VOCAB, loaded, options,
    tiny_profile, tokens, write_fixture,
};
// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

#[test]
fn a_session_owns_its_recurrent_stack_and_closing_it_hands_every_buffer_back() {
    let (_dir, fake, mut model) = loaded();

    let mark = fake.allocation_mark();
    let session = model.open_session().expect("a session opens");
    let owned = fake.allocated_since(mark);

    // A recurrent slot's convolution window and its state matrix, three slots
    // over, plus the one page table the attention slot addresses the cache
    // through. Nothing else is per session: the key/value rows belong to the
    // shared pool, which is why opening a session is kilobytes and not a cache.
    assert_eq!(
        owned.len(),
        2 * RECURRENT_SLOTS + 1,
        "a session allocates a conv window and a state per recurrent slot, and one page table"
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

    // Closing the same session again is a question about an id, not a second
    // release: the buffers are already gone and the model must not try again.
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
fn a_middle_prefill_chunk_produces_no_logits_and_the_last_one_does() {
    let (_dir, fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");

    // The middle chunk advances the session and nothing else. It cannot be
    // checked by what it returns - the signature has nothing to return - so it is
    // checked by what it did not launch.
    model
        .prefill_chunk(session, &tokens(1, 4))
        .expect("a middle chunk advances the session");
    let before_last = fake.launch_mark();

    let logits = model
        .prefill_last(session, &tokens(5, 4))
        .expect("the last chunk produces a row");
    assert_eq!(logits.rows(), 1, "a prompt ends in exactly one logits row");
    assert_eq!(logits.vocab(), VOCAB);
    assert_eq!(
        logits.view().len(),
        LOGITS_ROW_BYTES,
        "the block handed back is one row wide"
    );
    let block = logits.view().id().get();

    let all = fake.launches_since(0);
    let (early, late) = all.split_at(before_last);
    assert!(
        early.iter().all(|launch| launch.span_in(block).is_none()),
        "no launch wrote the logits block before the chunk that asked for a row"
    );
    let writes: Vec<&Launch> = late
        .iter()
        .filter(|launch| launch.span_in(block).is_some())
        .collect();
    assert_eq!(
        writes.len(),
        1,
        "the last chunk projects exactly one row, not one per token"
    );
    assert_eq!(
        writes[0]
            .span_in(block)
            .expect("the write was found")
            .offset,
        0,
        "a prompt's row is row zero of the block"
    );
}

/// Every launch that reaches a member's own buffers reaches exactly one
/// member's, and the launches of any one entry walk the members in order.
///
/// A batch is ONE pass, so "everything between the previous logits row and this
/// one" is no longer the work that produced this row - the rows share every
/// projection, every norm and every softmax, which is the whole point. What is
/// still per row is the recurrent state: a state matrix belongs to one
/// conversation and no launch can fold two of them together. So the
/// attribution is asserted where it still exists, and more precisely than
/// segmenting the log could.
fn every_state_launch_belongs_to_one_member(log: &[Launch], owned: &[HashSet<u64>]) {
    let members = owned.len();
    let mut per_entry: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for launch in log {
        let touched: Vec<usize> = owned
            .iter()
            .enumerate()
            .filter(|(_, buffers)| launch.touches_any(buffers))
            .map(|(index, _)| index)
            .collect();
        match touched.as_slice() {
            [] => {}
            [one] => match per_entry.iter_mut().find(|(name, _)| *name == launch.entry) {
                Some((_, seen)) => seen.push(*one),
                None => per_entry.push((launch.entry, vec![*one])),
            },
            several => panic!(
                "a launch of {} was handed {} members' state at once",
                launch.entry,
                several.len()
            ),
        }
    }
    assert!(
        !per_entry.is_empty(),
        "a batch over {members} members reaches their recurrent state somewhere"
    );
    for (entry, seen) in &per_entry {
        assert_eq!(
            seen.len() % members,
            0,
            "{entry} ran {} times over {members} members",
            seen.len()
        );
        for (at, member) in seen.iter().enumerate() {
            assert_eq!(
                *member,
                at % members,
                "launch {at} of {entry} belongs to member {}, not {member}",
                at % members
            );
        }
    }
}

#[test]
fn each_decode_row_is_the_work_of_the_session_that_asked_for_it() {
    let (_dir, fake, mut model) = loaded();

    // Three sessions, and the buffers each one owns, so a launch can be
    // attributed to a session by the memory it was handed.
    let mut sessions = Vec::new();
    let mut owned: Vec<HashSet<u64>> = Vec::new();
    for _ in 0..3 {
        let mark = fake.allocation_mark();
        let session = model.open_session().expect("a session opens");
        owned.push(fake.allocated_since(mark).into_iter().collect());
        sessions.push(session);
    }
    // Distinct prompts, so the three sessions stand at different positions and a
    // row that came from the wrong one would be reading a different prefix.
    for (index, session) in sessions.iter().enumerate() {
        let first = u32::try_from(index).expect("three fits") * 8;
        model
            .prefill_chunk(*session, &tokens(first, 4))
            .expect("each session takes a prompt");
    }

    let steps: Vec<DecodeStep> = sessions
        .iter()
        .enumerate()
        .map(|(index, session)| DecodeStep {
            session: *session,
            token: TokenId::new(u32::try_from(index).expect("three fits") + 40),
        })
        .collect();

    let mark = fake.launch_mark();
    let logits = model.decode_batch(&steps).expect("the batch decodes");
    assert_eq!(logits.rows(), 3, "one row per step, in the caller's order");
    for index in 0..3_u32 {
        let row = logits.row(index).expect("the row exists");
        assert_eq!(
            row.offset(),
            u64::from(index) * LOGITS_ROW_BYTES,
            "row {index} sits at its own offset in the block"
        );
        assert_eq!(row.len(), LOGITS_ROW_BYTES);
    }
    let block = logits.view().id().get();

    let log = fake.launches_since(mark);
    let writes: Vec<usize> = log
        .iter()
        .enumerate()
        .filter(|(_, launch)| launch.span_in(block).is_some())
        .map(|(index, _)| index)
        .collect();
    assert_eq!(writes.len(), 3, "each step writes exactly one row");
    for (row, at) in writes.iter().copied().enumerate() {
        let offset = log[at].span_in(block).expect("the write was found").offset;
        assert_eq!(
            offset,
            u64::try_from(row).expect("three fits") * LOGITS_ROW_BYTES,
            "the rows are written in the order the caller listed its steps"
        );
    }

    every_state_launch_belongs_to_one_member(&log, &owned);

    // And the page table: one packed block for the whole batch, belonging to no
    // member. Which row of it each member reads is a kernel-side fact, graded
    // on a device by `shiso-cuda --test q35_attn` and end to end by
    // `--test qwen35_batch`; what this can see is that the launches were not
    // handed one member's own table, which is the defect that reading them all
    // from `owned` would be.
    let tables: Vec<u64> = log
        .iter()
        .filter(|launch| launch.entry == "kernel_q35_attn_decode")
        .map(|launch| launch.spans[5].id)
        .collect();
    assert!(
        !tables.is_empty(),
        "a decode batch runs the attention softmax"
    );
    assert!(
        tables.windows(2).all(|pair| pair[0] == pair[1]),
        "every attention launch of one pass reads one packed block"
    );
    assert!(
        owned.iter().all(|buffers| !buffers.contains(&tables[0])),
        "the packed block is the pass's own, not a member's page table"
    );
}

#[test]
fn a_decode_batch_naming_an_unknown_session_advances_nobody() {
    let (_dir, fake, mut model) = loaded();
    let live = model.open_session().expect("a session opens");
    let ghost = model.open_session().expect("a second session opens");
    model.close_session(ghost).expect("the second one closes");

    let mark = fake.launch_mark();
    let err = model
        .decode_batch(&[
            DecodeStep {
                session: live,
                token: TokenId::new(1),
            },
            DecodeStep {
                session: ghost,
                token: TokenId::new(2),
            },
        ])
        .expect_err("a batch naming a closed session is refused");
    assert!(
        matches!(err, CoreError::UnknownSession(id) if id == ghost),
        "{err:?}"
    );
    // The refusal is whole: the first member must not have been advanced by a
    // token the caller never receives a row for.
    assert_eq!(
        fake.launches_since(mark).len(),
        0,
        "a batch that cannot run in full runs not at all"
    );
}

#[test]
fn a_session_that_would_run_past_its_context_says_so_instead() {
    let (_dir, _fake, mut model) = loaded();
    assert_eq!(model.info().context_capacity, CTX);
    let session = model.open_session().expect("a session opens");

    // Exactly the capacity is allowed; the position after it is not.
    model
        .prefill_chunk(session, &tokens(0, CTX))
        .expect("a prompt may fill the context exactly");
    let err = model
        .decode_batch(&[DecodeStep {
            session,
            token: TokenId::new(7),
        }])
        .expect_err("one more position is past the cap");
    assert!(
        matches!(err, CoreError::ContextOverflow { session: id, cap } if id == session && cap == CTX),
        "{err:?}"
    );
}

#[test]
fn teardown_releases_the_device_from_the_top_down() {
    let (_dir, fake, mut model) = loaded();
    let resident = fake.allocated_since(0);
    let first_built = *resident.first().expect("the load allocated something");
    let last_built = *resident.last().expect("the load allocated something");

    let mark = fake.allocation_mark();
    let older = model.open_session().expect("a session opens");
    let older_buffers = fake.allocated_since(mark);
    let mark = fake.allocation_mark();
    let newer = model.open_session().expect("a second session opens");
    let newer_buffers = fake.allocated_since(mark);
    assert_ne!(older, newer);

    model.teardown().expect("teardown reports cleanly");
    assert_eq!(fake.live_count(), 0, "teardown leaves nothing allocated");
    assert_eq!(fake.double_frees(), 0, "nothing was released twice");

    let rank = |id: u64| fake.release_rank(id);
    let latest = |ids: &[u64]| ids.iter().copied().map(rank).max().expect("a group");
    let earliest = |ids: &[u64]| ids.iter().copied().map(rank).min().expect("a group");

    // Sessions before the model they stand on, and the newer session before the
    // older one: a session holds page references the pool underneath it is about
    // to be asked to release.
    assert!(
        latest(&newer_buffers) < earliest(&older_buffers),
        "the session opened last is released first"
    );
    assert!(
        latest(&older_buffers) < earliest(&resident),
        "every session is released before the model's own memory"
    );
    // And inside the load, the last thing built is released before the first: the
    // scratch goes before the cache, the cache before the pool, the pool before
    // the weights.
    assert!(
        rank(last_built) < rank(first_built),
        "the model releases its own memory in reverse construction order"
    );

    // Teardown is answerable twice and the drop that follows adds no second
    // release: the whole point of the explicit call is that it reports, and a
    // `Drop` that freed again would turn a reported teardown into a fault.
    model.teardown().expect("a second teardown is a no-op");
    drop(model);
    assert_eq!(fake.double_frees(), 0, "drop after teardown frees nothing");
}

#[test]
fn a_boxed_model_drives_a_whole_turn_without_naming_its_type() {
    let dir = TempDir::new().expect("a temporary directory");
    let path = write_fixture(&dir);
    let profile = tiny_profile();
    let fake = FakeBackend::new();
    let backend: Arc<dyn Backend> = Arc::<FakeBackend>::clone(&fake);

    // The scheduler owns a `Box<dyn Model>`, so object safety is proven by
    // building one and driving a turn through it rather than by inspection.
    let mut model: Box<dyn Model> = Box::new(
        Qwen35Model::load(&backend, &path, &profile, options())
            .expect("the synthetic checkpoint loads"),
    );

    assert_eq!(model.info().context_capacity, CTX);
    assert_eq!(
        model.info().vocab,
        VOCAB,
        "the trait object reports the geometry it was loaded at"
    );

    let session = model.open_session().expect("a session opens");
    model
        .prefill_chunk(session, &tokens(1, 3))
        .expect("a middle chunk runs");
    let logits = model
        .prefill_last(session, &tokens(4, 3))
        .expect("the last chunk produces a row");
    assert_eq!(logits.rows(), 1);
    let logits = model
        .decode_batch(&[DecodeStep {
            session,
            token: TokenId::new(9),
        }])
        .expect("a step decodes");
    assert_eq!(logits.rows(), 1);
    model.close_session(session).expect("the session closes");
    model.teardown().expect("teardown reports cleanly");
    assert_eq!(fake.live_count(), 0, "the boxed model released everything");
}
