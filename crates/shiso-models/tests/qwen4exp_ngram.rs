//! The one buffer a pass reads that is neither a weight nor an activation.
//!
//! Every checkpoint of this family folds an n-gram block into one slot, and
//! that block decodes a row per (position, hash head) out of a table of
//! hundreds of millions of rows that is never made resident. The buffer those
//! rows land in is allocated at load and read by a launch, and until the
//! streaming tier was wired to it nothing wrote it: every launch of the pass
//! succeeded and one whole layer's contribution to the residual was decoded out
//! of whatever was last at that address.
//!
//! That is why every assertion here is on *contents*, and why the fake device
//! poisons a fresh allocation with `0xA5` rather than zeroing it. A test that
//! only checked that a copy had happened would pass on a tier that uploaded
//! zeros - which is indistinguishable from the bug.
//!
//! See `common/fixture.rs` for the synthetic checkpoint, its n-gram table, and
//! the byte each of that table's rows holds at each position.

use shiso_core::Backend as _;
use shiso_core::{DecodeStep, DeviceView, Model as _, TokenId};
use shiso_engine::sparse::bind::Operand;
use shiso_ple::{PleParams, PleRowType, codec};

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use fixture::{
    FakeBackend, PLE_HEADS, PLE_ROW_BYTES, POISON, loaded, ple_row_byte, tiny_ngram, tiny_shape,
    tokens,
};

/// The reader parameters the fixture's table would be opened with.
fn params() -> PleParams {
    let ple = tiny_shape()
        .ple
        .expect("the fixture folds in an n-gram block");
    tiny_ngram()
        .params(ple, PleRowType::BlockNl4, None)
        .expect("the fixture's table numbers are consistent")
}

/// The row identifiers a chunk starting at position zero names.
fn row_ids(path: &[TokenId], first: u32, rows: u32) -> Vec<u64> {
    let params = params();
    let mut ids = vec![0_u64; (rows * PLE_HEADS) as usize];
    codec::row_ids(&params, path, first, rows, &mut ids).expect("the hash runs over the path");
    ids
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

#[test]
fn the_staged_rows_are_the_table_rows_the_hash_named_and_not_the_poison_under_them() {
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let chunk = tokens(3, 4);
    model
        .prefill_chunk(session, &chunk)
        .expect("a four-token chunk runs");

    let binder = model.stack().binder(session).expect("a live session binds");
    let staged = binder
        .buffer(Operand::NgramRows)
        .expect("the stack holds a table, so the rows bind");
    let stride = usize::try_from(PLE_ROW_BYTES).expect("a row fits this host");
    let ids = row_ids(&chunk, 0, 4);
    let raw = read_bytes(&backend, staged, ids.len() * stride);

    assert!(
        !raw.iter().all(|byte| *byte == POISON),
        "the staging still holds the byte a fresh allocation is poisoned with, \
         so nothing was uploaded into it"
    );
    for (row, &id) in ids.iter().enumerate() {
        for j in 0..stride {
            assert_eq!(
                raw[row * stride + j],
                ple_row_byte(id, j),
                "row {row} of the gather is not table row {id} at byte {j}"
            );
        }
    }
}

#[test]
fn a_second_chunk_stages_its_own_positions_rows_rather_than_the_first_chunks() {
    // The staging is one buffer for the life of the stack - its address cannot
    // move, because a captured launch bakes it in - so the thing that has to be
    // true per pass is that its contents were replaced. A tier that gathered
    // once and never again would pass the test above and fail this one.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let first = tokens(3, 4);
    let second = tokens(30, 4);
    model
        .prefill_chunk(session, &first)
        .expect("the first chunk runs");
    model
        .prefill_chunk(session, &second)
        .expect("the second chunk runs");

    let mut path = first.clone();
    path.extend_from_slice(&second);
    // The second chunk's positions are four through seven, and its grams reach
    // back into the first chunk, which is why the hash is given the whole path
    // rather than the chunk.
    let ids = row_ids(&path, 4, 4);
    assert_ne!(
        ids,
        row_ids(&path, 0, 4),
        "the fixture's two chunks name the same rows, so this asserts nothing"
    );

    let binder = model.stack().binder(session).expect("a live session binds");
    let staged = binder.buffer(Operand::NgramRows).expect("the rows bind");
    let stride = usize::try_from(PLE_ROW_BYTES).expect("a row fits this host");
    let raw = read_bytes(&backend, staged, ids.len() * stride);
    for (row, &id) in ids.iter().enumerate() {
        assert_eq!(
            &raw[row * stride..row * stride + stride],
            &(0..stride).map(|j| ple_row_byte(id, j)).collect::<Vec<_>>()[..],
            "row {row} of the second chunk's gather is not table row {id}"
        );
    }
}

#[test]
fn the_decode_launch_reads_the_staging_the_rows_were_uploaded_into() {
    // Contents alone are not enough: a tier that uploaded into a buffer the
    // launch does not read would satisfy every assertion above. So this pins
    // the other half - that the range the decode kernel is handed is the same
    // allocation the upload went to.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    model
        .prefill_chunk(session, &tokens(3, 4))
        .expect("a chunk runs");

    let binder = model.stack().binder(session).expect("a live session binds");
    let staged = binder.buffer(Operand::NgramRows).expect("the rows bind");
    let decode = backend
        .launches()
        .into_iter()
        .find(|launch| launch.entry == "q4e_ple_dequant_kernel")
        .expect("the pass decodes the gathered rows exactly once");
    // `out, rows, head_dim, n_heads, row_bytes, n_rows`
    assert_eq!(
        decode.args[1],
        fixture::Arg::Buffer(staged.id().get(), staged.offset(), staged.len()),
        "the decode launch was pointed at something other than the staging"
    );
}

#[test]
fn a_pass_whose_gather_was_never_issued_is_refused_rather_than_decoded() {
    // The planner starts a gather and the runner takes it out of the pass, so a
    // pass that still holds one is a pass whose rows nobody fetched. Issuing it
    // would bind the staging anyway and decode a layer out of the previous
    // chunk's rows, with every launch succeeding.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let chunk = tokens(3, 4);
    model
        .stack_mut()
        .reserve(session, 4)
        .expect("the cache has room");
    let pass = model.plan_chunk(session, &chunk, 1).expect("a chunk plans");
    assert!(
        !pass.gather.is_empty(),
        "the fixture's geometry folds in an n-gram block, so a pass carries a gather"
    );
    let err = model
        .stack()
        .execute(session, &pass)
        .expect_err("an ungathered pass cannot be issued");
    assert!(
        matches!(err, shiso_core::CoreError::Unsupported { what, .. }
            if what.contains("gathered")),
        "{err:?}"
    );
    assert_eq!(
        backend.launch_count(),
        0,
        "the refusal came before the first launch"
    );
}

#[test]
fn the_rows_land_behind_the_first_layer_and_ahead_of_the_launch_that_decodes_them() {
    // The reads are awaited at the island that first reads the staging, not
    // before the pass: the device runs the prologue and the layers ahead of it
    // while the host reads. What has to hold is the stream order - the upload
    // after every launch that does not read the staging and before the one
    // that decodes it - for a prompt chunk and for a decode step alike.
    let (_dir, backend, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let staged = {
        let binder = model.stack().binder(session).expect("a live session binds");
        let view = binder.buffer(Operand::NgramRows).expect("the rows bind");
        view.id().get()
    };
    model
        .prefill_chunk(session, &tokens(3, 4))
        .expect("a chunk runs");
    for step in 0..2 {
        let mark = backend.launch_count();
        if step == 0 {
            model
                .prefill_chunk(session, &tokens(30, 4))
                .expect("a second chunk runs");
        } else {
            model
                .decode_batch(&[DecodeStep {
                    session,
                    token: TokenId::new(9),
                }])
                .expect("a decode step runs");
        }
        let launches = backend.launches();
        let decode = (mark..launches.len())
            .find(|at| launches[*at].entry == "q4e_ple_dequant_kernel")
            .expect("the pass decodes the gathered rows");
        let landed = backend
            .uploads()
            .into_iter()
            .filter(|upload| upload.buffer == staged && upload.launches >= mark)
            .collect::<Vec<_>>();
        assert_eq!(landed.len(), 1, "one upload of the rows per pass");
        let upload = landed[0];
        assert!(
            upload.launches > mark,
            "the rows landed before anything was launched, so nothing overlapped the reads"
        );
        assert!(
            upload.launches <= decode,
            "the rows landed after the launch that decodes them"
        );
        for launch in &launches[mark..upload.launches] {
            assert!(
                !launch
                    .args
                    .iter()
                    .any(|arg| matches!(arg, fixture::Arg::Buffer(id, _, _) if *id == staged)),
                "{} read the staging before the rows landed",
                launch.entry
            );
        }
    }
}
