//! Speculative decode as a session sees it: what a step stages, and what a
//! rejected draft puts back.
//!
//! The arithmetic over tokens - how far a draft is bounded, how much of it the
//! target agreed with - is checked in the engine's own unit tests, where it
//! needs nothing. What is checked here is the half that touches a session's
//! device state, and the device under it is the fixture's: a `HashMap` of byte
//! vectors, so a copy really moves bytes and a test can read them back.
//!
//! # Why this matters more than it looks
//!
//! A rollback that restores the wrong record does not fail. It leaves a session
//! standing on recurrent state for a position it never committed, and the next
//! token comes out of that state fluently. The only way to see it is to put
//! distinguishable bytes in every record and assert which one arrived - which is
//! what every case below does.
//!
//! # Nothing here has run on a GPU
//!
//! The records these cases restore from are written by the sequential recurrent
//! kernel and the n-gram convolution, neither of which has ever executed. What
//! is asserted is that the plan hands those kernels the slab and the record
//! count, and that the copy back reads the record the accept rule named.

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use shiso_core::Model as _;
use shiso_core::ids::SessionId;
use shiso_engine::sparse::bind::Operand;
use shiso_engine::sparse::mtp::{Rollback, SpecDepth};
use shiso_engine::sparse::plan::Step;
use shiso_engine::sparse::stack::SparseStack;
use shiso_models::qwen4exp::Qwen4ExpModel;

/// Drafts the speculative cases run at.
const DRAFTS: u32 = 4;

/// Fills every byte of one device range with `value`.
fn fill(model: &Qwen4ExpModel, view: shiso_core::DeviceView<'_>, value: u8) {
    let bytes = vec![value; usize::try_from(view.len()).expect("a fixture range fits this host")];
    model
        .stack()
        .backend()
        .copy_h2d(view, &bytes)
        .expect("the fake device accepts a whole-range write");
}

/// Reads one device range back.
fn read(model: &Qwen4ExpModel, view: shiso_core::DeviceView<'_>) -> Vec<u8> {
    let mut bytes =
        vec![0_u8; usize::try_from(view.len()).expect("a fixture range fits this host")];
    model
        .stack()
        .backend()
        .copy_d2h(view, &mut bytes)
        .expect("the fake device answers a whole-range read");
    bytes
}

/// Puts a distinguishable byte in every per-position record and a different one
/// in every live buffer, so that after a rollback the bytes say which record
/// arrived.
///
/// Record `r` is filled with `0x10 + r`; the live buffers are filled with
/// `0xFF`, which is the state a verify pass would have left them in after
/// running past the last token the target agreed with.
fn seed(model: &Qwen4ExpModel, session: SessionId, depth: SpecDepth) {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        fill(model, conv, 0xFF);
        fill(model, state, 0xFF);
        let (conv_ckpt, state_ckpt) = live
            .recurrent_rollback(ordinal)
            .expect("a speculative session carries records");
        seed_records(model, conv_ckpt, conv.len(), depth);
        seed_records(model, state_ckpt, state.len(), depth);
    }
    let window = live
        .ngram_window()
        .expect("the fixture has an n-gram block");
    fill(model, window, 0xFF);
    let records = live
        .ngram_window_rollback()
        .expect("a speculative session carries n-gram records");
    seed_records(model, records, window.len(), depth);
}

/// Fills a slab of `depth` records of `stride` bytes, record `r` with `0x10 + r`.
fn seed_records(
    model: &Qwen4ExpModel,
    slab: shiso_core::DeviceView<'_>,
    stride: u64,
    depth: SpecDepth,
) {
    assert_eq!(
        slab.len(),
        stride * u64::from(depth.rollback_slots()),
        "a slab holds exactly one record per drafted row"
    );
    for record in 0..depth.rollback_slots() {
        let at = u64::from(record) * stride;
        let byte = 0x10_u8.saturating_add(u8::try_from(record).expect("fewer than 256 records"));
        fill(
            model,
            slab.slice(at, stride).expect("the record is in range"),
            byte,
        );
    }
}

/// Asserts that every piece of per-position state holds `want` end to end.
fn assert_state_is(model: &Qwen4ExpModel, session: SessionId, want: u8) {
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    for ordinal in 0..u32::try_from(live.recurrent_slots()).expect("a few slots") {
        let (conv, state) = live.recurrent(ordinal).expect("the slot exists");
        assert!(
            read(model, conv).iter().all(|byte| *byte == want),
            "slot {ordinal}'s convolution window is not record {want:#x}"
        );
        assert!(
            read(model, state).iter().all(|byte| *byte == want),
            "slot {ordinal}'s state matrix is not record {want:#x}"
        );
    }
    let window = live
        .ngram_window()
        .expect("the fixture has an n-gram block");
    assert!(
        read(model, window).iter().all(|byte| *byte == want),
        "the n-gram window is not record {want:#x}"
    );
}

/// A loaded speculative model with one open session.
fn speculative() -> (tempfile::TempDir, Qwen4ExpModel, SessionId) {
    let (dir, _backend, mut model) = fixture::loaded_with(fixture::speculative_options(DRAFTS));
    let session = model.open_session().expect("a session opens");
    (dir, model, session)
}

#[test]
fn a_draft_the_target_rejected_outright_leaves_the_session_on_the_token_it_committed() {
    // Nothing accepted, so the session has to stand where the one committed
    // token left it - record zero - and on no position the drafts reached.
    let (_dir, model, session) = speculative();
    let depth = SpecDepth::new(DRAFTS);
    seed(&model, session, depth);
    let before_written = model
        .stack()
        .session(session)
        .expect("the session is open")
        .written();

    let rollback = Rollback::new(0, DRAFTS).expect("nothing was accepted");
    model
        .stack()
        .spec_rollback(session, rollback)
        .expect("a speculative session can roll back");

    assert_state_is(&model, session, 0x10);
    assert_eq!(
        model
            .stack()
            .session(session)
            .expect("the session is open")
            .written(),
        before_written,
        "a rollback restores state and moves no position: the caller's own \
         commit is what advances a session"
    );
}

#[test]
fn a_partly_accepted_draft_restores_the_record_of_the_last_row_the_target_agreed_with() {
    let (_dir, model, session) = speculative();
    seed(&model, session, SpecDepth::new(DRAFTS));
    let rollback = Rollback::new(3, DRAFTS).expect("one of four was rejected");
    model
        .stack()
        .spec_rollback(session, rollback)
        .expect("a speculative session can roll back");
    // Record three, not record four: the slab holds one record per drafted row,
    // indexed by the row it was written after.
    assert_state_is(&model, session, 0x13);
}

#[test]
fn every_piece_of_per_position_state_is_restored_and_not_only_the_recurrent_slots() {
    // The n-gram block's dilated window advances per position exactly as a
    // recurrent slot does. Restoring the slots and leaving it behind would put
    // the stack on two different positions at once, and would still answer.
    let (_dir, model, session) = speculative();
    seed(&model, session, SpecDepth::new(DRAFTS));
    model
        .stack()
        .spec_rollback(session, Rollback::new(1, DRAFTS).expect("a rejection"))
        .expect("a speculative session can roll back");
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    let window = live
        .ngram_window()
        .expect("the fixture has an n-gram block");
    assert!(
        read(&model, window).iter().all(|byte| *byte == 0x11),
        "the n-gram window was left where the rejected drafts put it"
    );
}

#[test]
fn a_session_that_does_not_speculate_refuses_a_rollback_rather_than_doing_nothing() {
    // A silent success here would be the worst outcome: a caller that thought it
    // had rolled back, on a session that never recorded anything to roll back
    // to, keeps generating from state the target never produced.
    let (_dir, _backend, mut model) = fixture::loaded();
    let session = model.open_session().expect("a session opens");
    let error = model
        .stack()
        .spec_rollback(session, Rollback::new(0, 1).expect("a rejection"))
        .expect_err("a plain-decode session has no records");
    assert!(
        format!("{error:?}").contains("speculation"),
        "the refusal names what is missing: {error:?}"
    );
}

#[test]
fn a_session_that_does_not_speculate_carries_none_of_the_rollback_state() {
    // The slabs are absent rather than empty. A zero-length allocation is still
    // an address, and a kernel handed one would write through it.
    let (_dir, _backend, mut model) = fixture::loaded();
    let session = model.open_session().expect("a session opens");
    let stack = model.stack();
    let live = stack.session(session).expect("the session is open");
    assert_eq!(live.spec_depth(), SpecDepth::NONE);
    assert!(live.recurrent_rollback(0).is_err());
    assert!(live.ngram_window_rollback().is_none());
    assert!(live.pending_draft().is_none());
}

#[test]
fn a_staged_draft_row_is_a_copy_of_the_residual_and_not_a_view_of_it() {
    // The shared activation scratch is the next session's to use, so a staged
    // row that was a view of it would be whatever the next chunk wrote.
    let (_dir, _backend, mut model) = fixture::loaded_with(fixture::speculative_options(DRAFTS));
    let session = model.open_session().expect("a session opens");
    assert!(
        model
            .stack()
            .session(session)
            .expect("open")
            .pending_draft()
            .is_none(),
        "a fresh session has staged nothing, so there is no draft to make"
    );

    let residual = model
        .stack()
        .scratch_view(shiso_engine::sparse::scratch::Buf::Residual)
        .expect("the stack is resident");
    fill(&model, residual, 0x5A);
    model
        .stack_mut()
        .stage_draft_row(session, 0)
        .expect("a speculative session stages a row");

    {
        let stack = model.stack();
        let staged = stack
            .session(session)
            .expect("open")
            .pending_draft()
            .expect("a row was staged");
        assert!(read(&model, staged).iter().all(|byte| *byte == 0x5A));
    }

    // Overwriting the scratch the way the next session's chunk would must not
    // reach the staged row.
    let residual = model
        .stack()
        .scratch_view(shiso_engine::sparse::scratch::Buf::Residual)
        .expect("the stack is resident");
    fill(&model, residual, 0x00);
    let stack = model.stack();
    let staged = stack
        .session(session)
        .expect("open")
        .pending_draft()
        .expect("the row is still staged");
    assert!(
        read(&model, staged).iter().all(|byte| *byte == 0x5A),
        "the staged row followed the scratch it was copied out of"
    );
}

#[test]
fn a_discarded_draft_row_stops_being_offered_to_the_drafter() {
    let (_dir, _backend, mut model) = fixture::loaded_with(fixture::speculative_options(DRAFTS));
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .stage_draft_row(session, 0)
        .expect("a speculative session stages a row");
    assert!(
        model
            .stack()
            .session(session)
            .expect("open")
            .pending_draft()
            .is_some()
    );
    model
        .stack_mut()
        .discard_draft_row(session)
        .expect("a live session forgets its staged row");
    assert!(
        model
            .stack()
            .session(session)
            .expect("open")
            .pending_draft()
            .is_none(),
        "without a staged row there is no draft and therefore no proposal"
    );
}

// ---------------------------------------------------------------------------
// what the plan hands the kernels that write the records
// ---------------------------------------------------------------------------

/// The last two operands of every launch of `entry` in a one-row pass planned at
/// `depth`.
fn checkpoint_args(
    stack: &SparseStack,
    session: SessionId,
    entry: &str,
) -> Vec<(Operand, Operand)> {
    let ids = vec![shiso_core::TokenId::new(1)];
    let pass = stack
        .plan_chunk(session, &ids, 1)
        .expect("a row is planned");
    pass.plan
        .steps()
        .iter()
        .filter(|step| step.name() == entry)
        .map(|step| {
            let operands = step.operands().expect("the list fits a launch");
            let len = operands.len();
            (
                *operands.get(len - 2).expect("a checkpoint slab argument"),
                *operands.get(len - 1).expect("a record count argument"),
            )
        })
        .collect::<Vec<_>>()
}

#[test]
fn a_speculative_pass_hands_the_state_writing_kernels_a_slab_and_the_record_count() {
    let (_dir, _backend, mut model) = fixture::loaded_with(fixture::speculative_options(DRAFTS));
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, 1)
        .expect("the cache has room");

    let conv = checkpoint_args(model.stack(), session, "q4e_gdn_conv_kernel");
    assert!(!conv.is_empty(), "the fixture has recurrent slots");
    for (slot, (slab, cap)) in conv.iter().enumerate() {
        assert_eq!(
            (*slab, *cap),
            (
                Operand::RecurrentConvCkpt(u32::try_from(slot).expect("a few slots")),
                Operand::U32(DRAFTS)
            ),
            "slot {slot}'s convolution is handed its own slab and the draft depth"
        );
    }

    let recurrence = checkpoint_args(model.stack(), session, "q4e_gdn_recurrent_kernel");
    for (slot, (slab, cap)) in recurrence.iter().enumerate() {
        assert_eq!(
            (*slab, *cap),
            (
                Operand::RecurrentStateCkpt(u32::try_from(slot).expect("a few slots")),
                Operand::U32(DRAFTS)
            )
        );
    }

    let ple = checkpoint_args(model.stack(), session, "q4e_ple_conv_kernel");
    assert_eq!(
        ple.as_slice(),
        [(Operand::NgramWindowCkpt, Operand::U32(DRAFTS))],
        "the n-gram convolution records per-position windows too"
    );
}

#[test]
fn a_plain_decode_pass_hands_them_a_null_and_a_zero() {
    // Not a slab with a zero count: the session has no records allocated, so the
    // pointer has to be null and the count has to agree with it.
    let (_dir, _backend, mut model) = fixture::loaded();
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, 1)
        .expect("the cache has room");
    for entry in [
        "q4e_gdn_conv_kernel",
        "q4e_gdn_recurrent_kernel",
        "q4e_ple_conv_kernel",
    ] {
        for (slab, cap) in checkpoint_args(model.stack(), session, entry) {
            assert_eq!(
                (slab, cap),
                (Operand::Null, Operand::U32(0)),
                "{entry} was handed a checkpoint slab on a pass that drafts nothing"
            );
        }
    }
}

#[test]
fn the_slab_a_launch_is_handed_is_the_one_the_rollback_reads_from() {
    // The two halves of a rollback are written in different files - the plan
    // names the slab, the copy back reads it - and nothing but this compares
    // them. A launch pointed at one allocation and a rollback reading another
    // would record correctly and restore nothing.
    let (_dir, _backend, mut model) = fixture::loaded_with(fixture::speculative_options(DRAFTS));
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, 1)
        .expect("the cache has room");
    let ids = vec![shiso_core::TokenId::new(1)];
    let pass = model
        .stack()
        .plan_chunk(session, &ids, 1)
        .expect("a row is planned");
    let binder = model.stack().binder(session).expect("a live session binds");
    let slab = pass
        .plan
        .steps()
        .iter()
        .find_map(|step| match step {
            Step::Kernel { entry, args, .. } if entry.name() == "q4e_gdn_conv_kernel" => {
                let operands = args.operands().ok()?;
                let len = operands.len();
                binder.buffer(*operands.get(len - 2)?).ok()
            }
            _ => None,
        })
        .expect("the first recurrent slot's convolution is planned");
    let stack = model.stack();
    let (records, _state) = stack
        .session(session)
        .expect("open")
        .recurrent_rollback(0)
        .expect("a speculative session carries records");
    assert_eq!(
        (slab.id().get(), slab.offset(), slab.len()),
        (records.id().get(), records.offset(), records.len()),
        "the slab the launch writes is the slab the rollback reads"
    );
}
