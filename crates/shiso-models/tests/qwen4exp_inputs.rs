//! What is actually in the three buffers a pass reads before any weight does.
//!
//! These are the pass's only host-supplied arguments - the token ids the row
//! gather indexes with, the positions every cache write and every rotation is
//! addressed by, and the rotary table the attention and indexer blocks rotate
//! against - and an unwritten one of them does not fail: it gathers row zero for
//! every token and rotates every position to nothing, and the launches all
//! succeed. So every assertion below is on *contents*. A test that only checked
//! a copy had happened would pass on a writer that filled the buffers with
//! zeros, which is exactly the state this replaces.
//!
//! Each buffer is read back through the binder rather than by looking the
//! allocation up directly, so what is asserted is the range a launch would be
//! handed, not a range that merely happens to sit beside it.
//!
//! See `common/fixture.rs` for the synthetic checkpoint and why its widths are
//! what they are.

use shiso_core::Backend as _;
use shiso_core::{CoreError, DeviceView, Model as _, TokenId};
use shiso_engine::sparse::bind::Operand;
use shiso_engine::sparse::inputs::{MROPE_AXES, MROPE_PREFIX_BYTES};
use shiso_engine::sparse::scratch::Buf;

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use fixture::{FakeBackend, VOCAB, loaded};

/// Positions one rotary table row carries.
fn axes() -> usize {
    usize::try_from(MROPE_AXES).expect("a row is three positions wide")
}

/// Positions the rotary table carries in front of a chunk's first row.
fn lookback() -> usize {
    let row_bytes = u64::from(MROPE_AXES) * 4;
    usize::try_from(MROPE_PREFIX_BYTES / row_bytes).expect("the prefix is a handful of rows")
}

/// The first `count` `int32` of one device range, read back off the fake device.
fn read_i32(backend: &FakeBackend, view: DeviceView<'_>, count: usize) -> Vec<i32> {
    let bytes = u64::try_from(count * 4).expect("a row of this width fits");
    let row = view.slice(0, bytes).expect("the buffer holds the row");
    let mut raw = vec![0_u8; count * 4];
    backend
        .copy_d2h(row, &mut raw)
        .expect("the device reads back");
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

/// The first position of each rotary table row, which is what makes a table of
/// text rows readable as a list of positions.
fn table_positions(table: &[i32]) -> Vec<i32> {
    table.chunks(axes()).map(|row| row[0]).collect()
}

#[test]
fn the_id_buffer_holds_the_tokens_the_caller_passed_in_the_order_they_were_passed() {
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    // Deliberately not consecutive and deliberately repeating: a writer that
    // filled the row with its own index, or with zeros, would agree with a
    // consecutive chunk and disagree with this one.
    let chunk: Vec<TokenId> = [41_u32, 5, 5, 60, 0]
        .into_iter()
        .map(TokenId::new)
        .collect();
    model
        .stack_mut()
        .write_inputs(session, &chunk)
        .expect("the inputs are written");

    let binder = model.stack().binder(session).expect("a live session binds");
    let view = binder
        .buffer(Operand::Scratch(Buf::Ids))
        .expect("the gather's id operand resolves");
    assert_eq!(
        read_i32(&backend, view, chunk.len()),
        [41, 5, 5, 60, 0],
        "the row gather indexes the vocabulary with the caller's own ids"
    );
}

#[test]
fn the_position_buffer_holds_the_sessions_own_frontier_positions() {
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let first = fixture::tokens(1, 4);
    model
        .stack_mut()
        .write_inputs(session, &first)
        .expect("the inputs are written");
    {
        let binder = model.stack().binder(session).expect("a live session binds");
        let view = binder
            .buffer(Operand::Scratch(Buf::Positions))
            .expect("the attention's position operand resolves");
        assert_eq!(read_i32(&backend, view, 4), [0, 1, 2, 3]);
    }

    // A second chunk is not a second copy of the first. Every cache row this
    // chunk writes is addressed by these numbers, so a writer that restarted at
    // zero would overwrite the prompt with its own continuation.
    model
        .stack_mut()
        .advance(session, &first, false, None)
        .expect("the path grows");
    let second = fixture::tokens(9, 3);
    model
        .stack_mut()
        .write_inputs(session, &second)
        .expect("the inputs are written");
    let binder = model.stack().binder(session).expect("a live session binds");
    let view = binder
        .buffer(Operand::Scratch(Buf::Positions))
        .expect("the attention's position operand resolves");
    assert_eq!(read_i32(&backend, view, 3), [4, 5, 6]);
}

#[test]
fn the_rotary_table_holds_the_lookback_rows_in_front_of_the_chunks_own() {
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let prompt = fixture::tokens(1, 7);
    model
        .stack_mut()
        .advance(session, &prompt, false, None)
        .expect("the path grows");
    let chunk = fixture::tokens(20, 3);
    model
        .stack_mut()
        .write_inputs(session, &chunk)
        .expect("the inputs are written");

    let binder = model.stack().binder(session).expect("a live session binds");
    let whole = binder
        .buffer(Operand::Scratch(Buf::Mrope))
        .expect("the pooled-key pass's operand resolves");
    let rows = lookback() + chunk.len();
    let table = read_i32(&backend, whole, rows * axes());
    // Four positions reaching back behind the chunk, then the chunk's own
    // three. The pooled-key pass rounds a position down to the compression
    // boundary and indexes this table from its front, so the rows behind the
    // chunk are the ones it reads for the block its first token falls inside.
    assert_eq!(table_positions(&table), [3, 4, 5, 6, 7, 8, 9]);
    for row in table.chunks(axes()) {
        assert!(
            row.windows(2).all(|pair| pair[0] == pair[1]),
            "a token that is not part of a picture has one position: {row:?}"
        );
    }
}

#[test]
fn the_two_ways_the_plan_binds_the_rotary_table_start_at_different_rows() {
    // The pooled-key pass binds the table whole and everything else binds it
    // past the prefix. Both ranges are inside one buffer, so filling only one
    // of the two regions - or laying the rows out the other way round - would
    // still produce a launch that succeeds, with every query rotated by the
    // lookback.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let prompt = fixture::tokens(1, 11);
    model
        .stack_mut()
        .advance(session, &prompt, false, None)
        .expect("the path grows");
    model
        .stack_mut()
        .write_inputs(session, &fixture::tokens(30, 2))
        .expect("the inputs are written");

    let binder = model.stack().binder(session).expect("a live session binds");
    let whole = binder
        .buffer(Operand::Scratch(Buf::Mrope))
        .expect("the pooled-key pass's operand resolves");
    let past = binder
        .buffer(Operand::ScratchFrom(Buf::Mrope, MROPE_PREFIX_BYTES))
        .expect("the query rotation's operand resolves");

    assert_eq!(
        table_positions(&read_i32(&backend, whole, axes())),
        [7],
        "bound whole, the table starts four positions behind the chunk"
    );
    assert_eq!(
        table_positions(&read_i32(&backend, past, axes())),
        [11],
        "bound past the prefix, it starts at the chunk's own first position"
    );

    // And that second number is the one the position row names for row zero,
    // which is what makes the rotation and the cache write agree about where
    // this token is.
    let positions = binder
        .buffer(Operand::Scratch(Buf::Positions))
        .expect("the position operand resolves");
    assert_eq!(read_i32(&backend, positions, 1), [11]);
}

#[test]
fn a_token_the_vocabulary_has_no_row_for_is_refused_rather_than_clamped() {
    // Two of the three gather entries clamp an id outside the table to row zero
    // and the third reads past the table. This is the last place the ids are
    // still ids, so it is the only place the range can be checked at all.
    let (_dir, _backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let err = model
        .stack_mut()
        .write_inputs(session, &[TokenId::new(VOCAB)])
        .expect_err("this table has no row VOCAB");
    assert!(matches!(err, CoreError::InvalidValue { .. }), "{err:?}");
}

#[test]
fn an_empty_or_over_wide_chunk_is_refused_the_way_planning_one_is() {
    let (_dir, _backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let err = model
        .stack_mut()
        .write_inputs(session, &[])
        .expect_err("a chunk of no tokens writes nothing");
    assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
}

#[test]
fn the_shared_contracts_own_prefill_path_is_what_writes_the_inputs() {
    // The writer is reached through `prefill_chunk` rather than only through
    // the stack method beneath it, which is what keeps it on the path a caller
    // actually takes: an id this vocabulary has no row for is reported by
    // `prefill_chunk` rather than gathered as row zero inside the pass.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let chunk = fixture::tokens(12, 3);
    model
        .prefill_chunk(session, &chunk)
        .expect("the chunk runs");

    let binder = model.stack().binder(session).expect("a live session binds");
    let ids = binder
        .buffer(Operand::Scratch(Buf::Ids))
        .expect("the gather's id operand resolves");
    let want: Vec<i32> = chunk
        .iter()
        .map(|token| i32::try_from(*token).expect("a fixture id fits"))
        .collect();
    assert_eq!(read_i32(&backend, ids, chunk.len()), want);
}

/// A fresh session's recurrent state is cleared, not merely allocated.
///
/// The device allocator hands back whatever the last tenant of that address
/// left behind. A recurrent slot is read before it is written on a session's
/// first chunk, so a slot that is merely allocated folds a stranger's state
/// into the answer - and answers anyway, with finite numbers, which is why
/// nothing else here would catch it. The fake device fills every allocation
/// with `POISON` for exactly this reason: a double that zeroes cannot fail the
/// way a device does.
#[test]
fn a_fresh_session_starts_from_zeroed_recurrent_state() {
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let binder = model.stack().binder(session).expect("a live session binds");

    let mut checked = 0;
    for slot in 0..u32::try_from(model.stack().recurrent_slots()).expect("slot count fits") {
        for operand in [Operand::RecurrentConv(slot), Operand::RecurrentState(slot)] {
            let view = binder.buffer(operand).expect("the slot binds");
            let words = read_i32(&backend, view, 16);
            assert!(
                words.iter().all(|w| *w == 0),
                "slot {slot}'s {operand:?} was handed to a launch unzeroed: {words:?}"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "the fixture declares no recurrent slot, so this asserts nothing"
    );

    if let Ok(window) = binder.buffer(Operand::NgramWindow) {
        let words = read_i32(&backend, window, 16);
        assert!(
            words.iter().all(|w| *w == 0),
            "the n-gram window was handed to a launch unzeroed: {words:?}"
        );
    }
}

/// The same, for the per-position records a speculative session rolls back from.
///
/// The recurrent slabs were cleared and the n-gram one beside them was not,
/// which is the asymmetry this pins: a rollback copies all of them back into
/// live state, so one of them holding a stranger's bytes puts the session at a
/// position no pass ever computed - and it answers.
#[test]
fn a_speculating_sessions_rollback_records_start_cleared_too() {
    let (_dir, backend, mut model) = fixture::loaded_with(fixture::speculative_options(4));
    let session = model.open_session().expect("a session opens");
    let binder = model.stack().binder(session).expect("a live session binds");

    let mut checked = 0;
    for slot in 0..u32::try_from(model.stack().recurrent_slots()).expect("slot count fits") {
        for operand in [
            Operand::RecurrentConvCkpt(slot),
            Operand::RecurrentStateCkpt(slot),
        ] {
            let view = binder.buffer(operand).expect("the slab binds");
            let words = read_i32(&backend, view, 16);
            assert!(
                words.iter().all(|w| *w == 0),
                "slot {slot}'s {operand:?} was handed to a launch unzeroed: {words:?}"
            );
            checked += 1;
        }
    }
    let view = binder
        .buffer(Operand::NgramWindowCkpt)
        .expect("this fixture folds in an n-gram block");
    let words = read_i32(&backend, view, 16);
    assert!(
        words.iter().all(|w| *w == 0),
        "the n-gram rollback records were handed to a launch unzeroed: {words:?}"
    );
    checked += 1;
    assert!(checked > 0);
}
