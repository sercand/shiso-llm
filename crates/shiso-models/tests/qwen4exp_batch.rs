//! What a decode batch of more than one sequence actually hands the device.
//!
//! The planning half of this lives in `pass.rs`: which kernels a batched pass
//! dispatches, and which operand each of their per-row parameters names. This
//! binary asks the question that survives a correct plan - what is *in* the
//! per-row pointer arrays when the launches read them.
//!
//! # Why the contents and not the fact of a write
//!
//! The bug this whole path exists to close is not a missing write. It is one
//! session's page table answered for every row: a batch planned the scalar way
//! read one sequence's cache and handed the result back as the other sequences'
//! logits, and every launch of it succeeded. A packer that filled all eight
//! slots of an array with row zero's pointer is that same bug wearing a new
//! hat, and a test that only checked "the array was written" would pass on it.
//!
//! So every assertion below is on the bytes: slot `i` holds row `i`'s own
//! buffer, and two rows of one batch never hold the same pointer. The fake
//! device poisons a fresh allocation with `0xA5` rather than zeroing it, so a
//! slot nothing wrote reads as poison rather than as a plausible null.
//!
//! # Nothing here has run on a GPU
//!
//! No kernel of this family has executed anywhere in this workspace. What is
//! checked is the bookkeeping the kernels depend on and none of which fails
//! loudly on real hardware; the arithmetic is owed to a machine with the
//! hardware.

use shiso_core::{
    Backend as _, DEVICE_POINTER_BYTES, DecodeStep, DeviceView, Model as _, SessionId, TokenId,
};
use shiso_engine::sparse::rows::{Layout, PtrArray};
use shiso_engine::sparse::scratch::Buf;

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use fixture::{FakeBackend, POISON, RECURRENT_SLOTS, fake_address, loaded};

/// Opens `rows` sessions, each holding one claimed cache row.
fn open_rows(model: &mut shiso_models::qwen4exp::Qwen4ExpModel, rows: u32) -> Vec<SessionId> {
    (0..rows)
        .map(|_| model.open_session().expect("a session opens"))
        .collect()
}

/// The steps a batch of `sessions` decodes, one distinct token each.
fn steps(sessions: &[SessionId]) -> Vec<DecodeStep> {
    sessions
        .iter()
        .enumerate()
        .map(|(row, session)| DecodeStep {
            session: *session,
            token: TokenId::new(u32::try_from(row).expect("a row fits") + 1),
        })
        .collect()
}

/// One device range, read back off the fake device as raw bytes.
fn read_bytes(backend: &FakeBackend, view: DeviceView<'_>, len: usize) -> Vec<u8> {
    let span = view
        .slice(0, u64::try_from(len).expect("the span fits"))
        .expect("the buffer holds the span");
    let mut raw = vec![0_u8; len];
    backend
        .copy_d2h(span, &mut raw)
        .expect("the device reads back");
    raw
}

/// The pointers one packed array holds, in row order.
fn array(pack: &[u8], layout: Layout, which: PtrArray) -> Vec<u64> {
    let (offset, len) = layout
        .span(which)
        .unwrap_or_else(|| panic!("{which:?} is an array this stack packs"));
    let start = usize::try_from(offset).expect("the offset fits");
    let width = usize::try_from(len).expect("the width fits");
    pack[start..start + width]
        .chunks_exact(usize::try_from(DEVICE_POINTER_BYTES).expect("eight"))
        .map(|slot| u64::from_ne_bytes(slot.try_into().expect("one pointer")))
        .collect()
}

/// Runs one batch and hands back the packed arrays as bytes.
fn packed(rows: u32) -> (Vec<u8>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>) {
    let (_dir, backend, mut model) = loaded();
    let sessions = open_rows(&mut model, rows);
    model
        .decode_batch(&steps(&sessions))
        .expect("a batch of unrelated sequences decodes");

    let layout = layout(rows);
    let stack = model.stack();
    let view = stack
        .scratch_view(Buf::RowPointers)
        .expect("the pack is an activation buffer");
    let bytes = read_bytes(
        &backend,
        view,
        usize::try_from(layout.bytes()).expect("the pack fits"),
    );
    // What each array *should* hold, read straight off the sessions.
    let want_pages = sessions
        .iter()
        .map(|id| fake_address(stack.session(*id).expect("a live session").pages()))
        .collect::<Vec<_>>();
    let want_states = (0..u32::try_from(RECURRENT_SLOTS).expect("the slot count fits"))
        .map(|slot| {
            sessions
                .iter()
                .map(|id| {
                    let live = stack.session(*id).expect("a live session");
                    fake_address(live.recurrent(slot).expect("the slot exists").1)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let want_windows = sessions
        .iter()
        .map(|id| {
            fake_address(
                stack
                    .session(*id)
                    .expect("a live session")
                    .ngram_window()
                    .expect("this stack folds in an n-gram block"),
            )
        })
        .collect::<Vec<_>>();
    let pack = bytes;
    drop(model);
    (pack, want_pages, want_states, want_windows)
}

/// The layout the fixture's geometry packs at `rows` rows.
fn layout(rows: u32) -> Layout {
    Layout {
        rows,
        slots: u32::try_from(RECURRENT_SLOTS).expect("the slot count fits"),
        ngram: true,
        // The record arrays come after every array read here, so whether the
        // fixture's sessions carry them moves none of these spans.
        rollback: false,
    }
}

#[test]
fn the_page_table_array_holds_each_rows_own_table_and_never_one_table_repeated() {
    let rows = 3;
    let (pack, want_pages, _, _) = packed(rows);
    let held = array(&pack, layout(rows), PtrArray::Pages);
    assert_eq!(held.len(), rows as usize);
    // Row by row against the table the session itself holds. This is the
    // assertion the whole change exists for: a packer that wrote row zero's
    // table into every slot passes every structural check and fails here.
    assert_eq!(held, want_pages);
    // And the three are three, which is the same statement made without
    // reference to what they should be - so it still bites if the expectation
    // above were ever computed from the pack rather than from the sessions.
    let mut distinct = held.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        rows as usize,
        "two rows of one batch share a page table: {held:?}"
    );
}

#[test]
fn every_recurrent_slots_state_array_holds_one_state_per_row() {
    // Six recurrent slots in the fixture, each with its own array: a pack that
    // wrote one slot's states into every slot's array would give three rows
    // three different pointers and still be wrong, which is why the check is
    // per slot rather than per row.
    let rows = 3;
    let (pack, _, want_states, _) = packed(rows);
    for (slot, want) in want_states.iter().enumerate() {
        let slot = u32::try_from(slot).expect("the slot fits");
        let held = array(&pack, layout(rows), PtrArray::RecurrentState(slot));
        assert_eq!(&held, want, "recurrent slot {slot}");
        let mut distinct = held.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), rows as usize, "recurrent slot {slot}");
    }
}

#[test]
fn the_n_gram_window_array_holds_one_window_per_row() {
    let rows = 2;
    let (pack, _, _, want_windows) = packed(rows);
    let held = array(&pack, layout(rows), PtrArray::NgramWindows);
    assert_eq!(held, want_windows);
    assert_ne!(held[0], held[1]);
}

#[test]
fn no_slot_of_the_pack_a_launch_reads_is_left_as_the_allocators_poison() {
    // The pack is allocated once and filled per step. A slot the packer skipped
    // holds 0xA5A5A5A5A5A5A5A5, which is an address no kernel may dereference
    // and which a test that only counted transfers would never see.
    let rows = 3;
    let (pack, _, _, _) = packed(rows);
    let poison = u64::from_ne_bytes([POISON; 8]);
    let used = usize::try_from(layout(rows).bytes()).expect("the pack fits");
    for (index, slot) in pack[..used]
        .chunks_exact(usize::try_from(DEVICE_POINTER_BYTES).expect("eight"))
        .enumerate()
    {
        let value = u64::from_ne_bytes(slot.try_into().expect("one pointer"));
        assert_ne!(value, poison, "slot {index} of the pack was never written");
        assert_ne!(value, 0, "slot {index} of the pack is a null pointer");
    }
}

#[test]
fn each_row_of_a_batch_decodes_at_its_own_sequences_position() {
    // Three sessions at three different frontiers, so the position row is three
    // unrelated numbers rather than three consecutive ones. A batch that wrote
    // the positions the way a chunk does - one frontier plus an index - would
    // put every row but the first at a position in the first row's sequence.
    //
    // The prompt lengths are deliberately *not* one, two and three: those would
    // make the right answer and the wrong one the same three numbers, and the
    // test would pass on the bug. One, three and seven do not.
    let (_dir, backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 3);
    for (index, session) in sessions.iter().enumerate() {
        let count = [1_u32, 3, 7][index];
        model
            .prefill_chunk(*session, &fixture::tokens(1, count))
            .expect("a prompt runs");
    }
    model
        .decode_batch(&steps(&sessions))
        .expect("a batch of unrelated sequences decodes");

    let stack = model.stack();
    let view = stack
        .scratch_view(Buf::Positions)
        .expect("the positions are an activation buffer");
    let raw = read_bytes(&backend, view, 3 * 4);
    let held = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|slot| i32::from_le_bytes(*slot))
        .collect::<Vec<_>>();
    // One, three and seven tokens of prompt, so the three rows decode at
    // positions one, three and seven.
    assert_eq!(held, [1, 3, 7]);
}

#[test]
fn every_row_of_a_batch_advances_its_own_session_by_its_own_token() {
    let (_dir, _backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 3);
    let batch = steps(&sessions);
    model
        .decode_batch(&batch)
        .expect("a batch of unrelated sequences decodes");
    let stack = model.stack();
    for (row, step) in batch.iter().enumerate() {
        let live = stack.session(step.session).expect("a live session");
        assert_eq!(live.written(), 1, "row {row} did not advance");
        assert_eq!(
            live.history(),
            &[step.token],
            "row {row} recorded another row's token"
        );
    }
}

#[test]
fn each_rows_pooled_key_pass_reads_its_own_lookback_and_not_the_first_rows() {
    // The rotary table of a batched pass has two regions. The first is one row
    // per batch row, where every `3 * t` reader finds row `t`. The second is one
    // group per row - that row's own four lookback positions and then its own -
    // because the pooled key pass indexes backwards from the row it was
    // launched for, and the position behind row one is a position in row one's
    // sequence rather than row zero's.
    //
    // Nothing fails if the second region is missing or shared: row one's pooled
    // key is simply rotated at a position out of another sequence, and only
    // past the selection budget where the indexer is consulted at all.
    use shiso_engine::sparse::inputs::{MROPE_AXES, MROPE_GROUP_ROWS, batch_lookback_offset};

    let (_dir, backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 3);
    for (index, session) in sessions.iter().enumerate() {
        let count = u32::try_from(index).expect("the row fits") * 4 + 4;
        model
            .prefill_chunk(*session, &fixture::tokens(1, count))
            .expect("a prompt runs");
    }
    // The positions the batch is *about to* occupy, read before it runs: the
    // pass advances every row by one, and the table it wrote names where each
    // row decoded rather than where it now stands.
    let at = sessions
        .iter()
        .map(|id| {
            i64::from(
                model
                    .stack()
                    .session(*id)
                    .expect("a live session")
                    .written(),
            )
        })
        .collect::<Vec<_>>();
    model
        .decode_batch(&steps(&sessions))
        .expect("a batch of unrelated sequences decodes");

    let axes = usize::try_from(MROPE_AXES).expect("three axes");
    let group = usize::try_from(MROPE_GROUP_ROWS).expect("a group is five rows");
    let stack = model.stack();
    let table = stack
        .scratch_view(Buf::Mrope)
        .expect("the rotary table is an activation buffer");
    for (row, position) in at.iter().copied().enumerate() {
        let offset = batch_lookback_offset(3, u32::try_from(row).expect("the row fits"));
        let span = table
            .slice(offset, (group * axes * 4) as u64)
            .expect("the group is inside the table");
        let mut raw = vec![0_u8; group * axes * 4];
        backend
            .copy_d2h(span, &mut raw)
            .expect("the device reads back");
        let held = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|slot| i64::from(i32::from_le_bytes(*slot)))
            .collect::<Vec<_>>();
        // Four lookback rows then the row's own, all three axes agreeing
        // because a token that is not part of a picture has one position.
        for (back, chunk) in held.chunks_exact(axes).enumerate() {
            let lookback = i64::try_from(group - 1 - back).expect("a group is five rows");
            let want = position - lookback;
            assert_eq!(
                chunk, [want; 3],
                "row {row} group entry {back} names a position out of another sequence"
            );
        }
    }
}

#[test]
fn every_pooled_key_launch_of_a_batch_reaches_the_device_with_its_own_rows_page_table() {
    // The plan-level check lives in `pass.rs`: the operand is `PagesOf(row)`.
    // This is the same statement after the binder has resolved it, which is
    // where a resolver that answered every row with the first session's table
    // would show up - the operands would still read correctly and the launches
    // would all address one sequence's cache.
    let (_dir, backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 2);
    model
        .decode_batch(&steps(&sessions))
        .expect("a batch of unrelated sequences decodes");

    let stack = model.stack();
    let want = sessions
        .iter()
        .map(|id| {
            let view = stack.session(*id).expect("a live session").pages();
            (view.id().get(), view.offset())
        })
        .collect::<Vec<_>>();
    assert_ne!(want[0], want[1], "the two sessions share a page table");

    let launches = backend
        .launches()
        .into_iter()
        .filter(|launch| launch.entry == "q4e_idx_ring_pool_kernel")
        .collect::<Vec<_>>();
    let pooled = launches
        .iter()
        .map(|launch| match launch.args[5] {
            fixture::Arg::Buffer(id, offset, _) => (id, offset),
            other => panic!("the pooled pass binds a page table: {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pooled.len(),
        2 * fixture::ATTN_SLOTS,
        "the pooled pass is launched once per row per attention slot"
    );
    // Two attention slots, each launching row zero then row one.
    for pair in pooled.as_chunks::<2>().0 {
        assert_eq!(pair[0], want[0]);
        assert_eq!(pair[1], want[1]);
    }
    // And each launch reaches its own row's raw key ring, at its own rank:
    // a resolver that answered every row with the first session's ring would
    // pool one sequence's block out of another's raw keys.
    let rings = launches
        .iter()
        .map(|launch| match launch.args[1] {
            fixture::Arg::Buffer(id, offset, _) => (id, offset),
            other => panic!("the pooled pass binds a ring: {other:?}"),
        })
        .collect::<Vec<_>>();
    for (slot, pair) in (0_u32..).zip(rings.as_chunks::<2>().0) {
        for (row, got) in pair.iter().enumerate() {
            let view = stack
                .session(sessions[row])
                .expect("a live session")
                .indexer_ring(slot)
                .expect("the ring has a section per attention rank");
            assert_eq!(
                *got,
                (view.id().get(), view.offset()),
                "slot {slot} row {row}"
            );
        }
    }
}

#[test]
fn an_operand_naming_the_passs_one_session_is_refused_against_a_batch() {
    // The net under the planner. Every launch in a batched pass is supposed to
    // name a per-row array or a row, and nothing in the tree binds
    // `Operand::Pages` there - but a planner that forgot to widen one launch
    // would bind it, and answering with the first row's table is precisely the
    // bug. So the resolver refuses instead, and this is the test that keeps the
    // refusal from being a branch nothing reaches.
    use shiso_engine::sparse::bind::Operand;

    let (_dir, _backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 2);
    let stack = model.stack();
    let batched = stack
        .binder_rows(&sessions)
        .expect("a two-row binder is built");
    let error = batched
        .arg(Operand::Pages)
        .expect_err("a batch has one page table per row");
    assert!(format!("{error:?}").contains("one session"), "{error:?}");
    // The same operand against one session is the ordinary answer, so the
    // refusal above is about the batch rather than about the operand.
    let one = stack
        .binder_rows(&sessions[..1])
        .expect("a one-row binder is built");
    one.arg(Operand::Pages)
        .expect("one session's pass names its own page table");
    // And the addressed form answers per row, which is what the pooled pass
    // binds.
    for (row, session) in sessions.iter().enumerate() {
        let row = u32::try_from(row).expect("the row fits");
        let want = stack.session(*session).expect("a live session").pages();
        match batched
            .arg(Operand::PagesOf(row))
            .expect("every row of the batch has a page table")
        {
            shiso_core::KernelArg::Buffer(view) => {
                assert_eq!((view.id(), view.offset()), (want.id(), want.offset()));
            }
            other => panic!("a page table is a buffer: {other:?}"),
        }
    }
}

#[test]
fn a_pass_planned_for_one_width_is_refused_against_another() {
    // The per-row arrays are laid out at the width the pass was planned for, so
    // issuing a three-row plan against two sessions would read one slot past
    // the end of every array - and read it as an address.
    let (_dir, _backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 3);
    let pass = model
        .stack()
        .plan_batch(&sessions)
        .expect("a three-row batch plans");
    let error = model
        .stack()
        .execute_rows(&sessions[..2], &pass)
        .expect_err("a three-row plan is not a two-row batch");
    assert!(
        matches!(error, shiso_core::CoreError::ShapeMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn the_layout_the_pack_is_sized_at_and_the_one_it_is_written_at_agree() {
    // Two derivations of one layout. The buffer is sized before any session
    // exists, so its half asks the geometry; the pack and the binder both hold
    // sessions rather than a shape, so their half asks the sessions. They agree
    // by construction - a session allocates one state pair per recurrent slot
    // the geometry names, and an n-gram window exactly when the geometry folds
    // one in - and this is the test that keeps "by construction" true. The
    // one field they may differ in is the rollback records: the geometry
    // cannot know whether a session will be opened for speculation, so it
    // sizes for them, and these sessions were not.
    let (_dir, _backend, mut model) = loaded();
    let sessions = open_rows(&mut model, 4);
    let stack = model.stack();
    let live = sessions
        .iter()
        .map(|id| stack.session(*id).expect("a live session"))
        .collect::<Vec<_>>();
    let from_rows = Layout::of_rows(&live);
    let from_shape = Layout::of(&fixture::tiny_shape(), from_rows.rows);
    assert!(from_shape.rollback && !from_rows.rollback);
    assert_eq!(
        from_rows,
        Layout {
            rollback: false,
            ..from_shape
        }
    );
    assert!(from_shape.bytes() >= from_rows.bytes());
    assert_eq!(from_rows.rows, 4);
    assert_eq!(
        from_rows.slots,
        u32::try_from(RECURRENT_SLOTS).expect("the slot count fits")
    );
    assert!(from_rows.ngram);
}
