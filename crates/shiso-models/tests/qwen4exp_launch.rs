//! What the device is handed when a planned pass is issued.
//!
//! The companion to the planning tests: those ask what a pass decides, and this
//! asks what actually reaches a device when those decisions are carried out.
//! The device here records rather than computes, so nothing below is a claim
//! about a number a kernel would produce. What it is a claim about is the
//! bookkeeping every kernel depends on and none of which fails loudly on real
//! hardware - which allocation a launch was pointed at, how many blocks it
//! asked for, how much shared memory it reserved, and in what order the
//! reservations and the launches that read them happened.
//!
//! # Nothing here has run on a GPU
//!
//! No kernel in this family has ever executed. These tests drive the same code
//! path a device would be driven through and read back what it was asked to
//! do; the arithmetic on the other side of that seam is owed to a machine with
//! the hardware.

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use std::sync::Arc;

use fixture::{Arg, FakeBackend, Recorded};

use shiso_core::CoreError;
use shiso_core::Model as _;
use shiso_engine::sparse::plan::Step;

/// Runs a one-token pass and returns the trace.
///
/// Every step of a pass is issued: the fake device answers to all three
/// capabilities a pass can ask for, and the table it is loaded with stands in
/// for the entry names a build captures off its compiled modules. A step that
/// refuses here is a failure, which is the point - the refusals this used to
/// step over were the gaps, and a helper that kept stepping over them would
/// hide the next one.
fn trace(tokens: u32) -> (Arc<FakeBackend>, Vec<Recorded>) {
    let (dir, backend, mut model) = fixture::loaded();
    let session = model.open_session().expect("a session opens");
    let ids = fixture::tokens(1, tokens);
    model
        .stack_mut()
        .reserve(session, tokens)
        .expect("the cache has room for one chunk");
    let pass = model
        .plan_chunk(session, &ids, 1)
        .expect("a chunk of this width is planned");
    let exec = model
        .stack()
        .executor(session)
        .expect("a live session has an executor");
    for step in pass.plan.steps() {
        exec.run_step(step)
            .unwrap_or_else(|err| panic!("{} is issuable and was refused: {err:?}", step.name()));
    }
    let recorded = backend.launches();
    drop(model);
    drop(dir);
    (backend, recorded)
}

#[test]
fn every_launch_the_pass_issues_reaches_the_device_with_a_grid_and_a_block() {
    let (_backend, recorded) = trace(1);
    assert!(
        recorded.len() > 200,
        "a pass over eight layers issues more than this: {}",
        recorded.len()
    );
    for launch in &recorded {
        assert!(
            launch.grid.volume() > 0 && launch.block.volume() > 0,
            "{} asked for an empty launch",
            launch.entry
        );
        assert!(
            !launch.module.is_empty(),
            "{} was fetched from no module",
            launch.entry
        );
    }
}

#[test]
fn the_row_gather_opens_the_pass_and_reads_the_table_its_stored_format_names() {
    // The fixture's vocabulary table is half precision, so the entry is the
    // half-precision gather and its fan-out argument is one: the residual
    // broadcast is the launch after it, and a gather that fanned out here too
    // would write the residual four times over.
    let (_backend, recorded) = trace(3);
    let gather = recorded.first().expect("a pass opens with something");
    assert_eq!(gather.entry, "embed_tokens_hc_kernel");
    // `out, tokens, w, n_vocab, n_tokens, n_embd, n_hc`
    assert_eq!(gather.args.len(), 7);
    assert!(matches!(gather.args[0], Arg::Buffer(..)));
    assert!(matches!(gather.args[1], Arg::Buffer(..)));
    assert!(matches!(gather.args[2], Arg::Buffer(..)));
    assert_eq!(gather.args[3], Arg::U32(fixture::VOCAB));
    assert_eq!(gather.args[4], Arg::U32(3));
    assert_eq!(gather.args[5], Arg::U32(fixture::HIDDEN));
    assert_eq!(gather.args[6], Arg::U32(1));
    // The ids and the table are different allocations, and so is the
    // destination: a gather handed its own destination as the table reads
    // whatever the last chunk left there.
    let (Arg::Buffer(dst, ..), Arg::Buffer(ids, ..), Arg::Buffer(table, ..)) =
        (gather.args[0], gather.args[1], gather.args[2])
    else {
        panic!("the first three arguments are ranges");
    };
    assert_ne!(dst, ids);
    assert_ne!(dst, table);
    assert_ne!(ids, table);
    let expected = (u64::from(3 * fixture::HIDDEN)).div_ceil(256);
    assert_eq!(u64::from(gather.grid.x), expected);
}

#[test]
fn a_template_this_build_captured_no_name_for_is_the_one_refusal_left() {
    // Every tier is issuable; what is not is a template instantiation whose
    // mangled name no build has read off a module. That is a missing build
    // artifact rather than a missing kernel, and the refusal says so.
    use shiso_engine::sparse::kernels::{Entry, EntryTable, TemplateArgs};

    let empty = EntryTable::default();
    let err = empty
        .resolve(Entry::template(
            shiso_engine::sparse::kernels::MOD_EXL3,
            shiso_engine::sparse::kernels::EXL3_GEMM,
            TemplateArgs::Trellis { bits: 4, shape: 2 },
        ))
        .expect_err("an uncaptured instantiation has no name");
    assert!(
        matches!(err, CoreError::Unsupported { what, .. } if what.contains("never captured")),
        "{err:?}"
    );
}

#[test]
fn the_pooled_key_pass_launches_a_truncating_divide_plus_one_and_not_a_ceiling() {
    // The trap the kernel set's own table records: the two agree everywhere
    // except at a multiple of the pooling ratio, which is the case that
    // matters, because it is the only one where they disagree. At four rows a
    // ceiling launches one block and the launcher launches two.
    let (_backend, recorded) = trace(4);
    let pool = recorded
        .iter()
        .find(|launch| launch.entry == "q4e_idx_ring_pool_kernel")
        .expect("an attention layer keeps the pooled plane current");
    assert_eq!(pool.grid.x, 4 / 4 + 1);
    assert_eq!(pool.block.x, 128);

    let (_backend, recorded) = trace(3);
    let pool = recorded
        .iter()
        .find(|launch| launch.entry == "q4e_idx_ring_pool_kernel")
        .expect("an attention layer keeps the pooled plane current");
    // Three rows complete no block, and the launcher still asks for the one
    // that covers the partial block the next token starts.
    assert_eq!(pool.grid.x, 1);
}

#[test]
fn the_cache_store_is_bounded_in_rows_and_not_in_pages() {
    // The store refuses a row past the pool, and the bound it is handed is the
    // pool in *positions*. Handing it the page count is the same refusal with
    // the bound a factor of PAGE_TOKENS too low, so every write from the first
    // session whose page id reaches `pool_pages / PAGE_TOKENS` lands nowhere:
    // its attention then reads a plane of zeros and answers fluently out of a
    // residual the cache never contributed to. Nothing fails, which is why this
    // is asserted on the argument rather than on an outcome.
    let (_backend, recorded) = trace(1);
    let store = recorded
        .iter()
        .find(|launch| launch.entry == "q4e_qsa_store_kv_kernel")
        .expect("an attention layer stores its rows");
    // `k_cache, v_cache, k, v, kw, head_dim, n_head_kv, n_rot, rope_base, pos,
    //  mrope, sec_t, sec_h, sec_w, pages, pool_slots, eps`
    assert_eq!(
        store.args[15],
        Arg::U32(fixture::POOL_PAGES * shiso_core::page::PAGE_TOKENS),
    );
}

#[test]
fn the_residual_broadcast_is_handed_the_two_widths_and_the_row_count_in_that_order() {
    let (_backend, recorded) = trace(1);
    let init = recorded
        .iter()
        .find(|launch| launch.entry == "q4e_hc_init_kernel")
        .expect("a pass opens with the broadcast");
    // `res, embed, n_embd, n_hc, n_tok`. Two ranges then three counts, and the
    // counts in the kernel's own order: a stream count where the residual width
    // belongs writes a fraction of the buffer and nothing fails.
    assert_eq!(init.args.len(), 5);
    assert!(matches!(init.args[0], Arg::Buffer(..)));
    assert!(matches!(init.args[1], Arg::Buffer(..)));
    assert_eq!(init.args[2], Arg::U32(fixture::HIDDEN));
    assert_eq!(init.args[3], Arg::U32(fixture::STREAMS));
    assert_eq!(init.args[4], Arg::U32(1));
    // And the two ranges are different allocations: a broadcast that read its
    // own destination would be a residual of whatever the buffer held.
    let (Arg::Buffer(dst, ..), Arg::Buffer(src, ..)) = (init.args[0], init.args[1]) else {
        panic!("the first two arguments are ranges");
    };
    assert_ne!(dst, src);
}

#[test]
fn the_recurrent_normalizations_are_one_kernel_at_two_offsets_in_one_buffer() {
    let (_backend, recorded) = trace(1);
    let norms = recorded
        .iter()
        .filter(|launch| launch.entry == "q4e_gdn_l2norm_kernel")
        .collect::<Vec<_>>();
    // Six recurrent slots in the fixture's eight, two normalizations each.
    assert_eq!(norms.len(), 12);
    for pair in norms.chunks(2) {
        let [query, key] = pair else {
            panic!("the normalizations come in pairs")
        };
        assert_eq!(query.args[0], key.args[0], "both read the fused projection");
        assert_eq!(query.args[4], Arg::U32(0));
        let Arg::U32(offset) = key.args[4] else {
            panic!("the offset is a count")
        };
        assert!(offset > 0, "the key half starts past the query half");
    }
}

#[test]
fn the_routed_stage_reserves_its_staging_before_anything_reads_it() {
    let (dir, backend, mut model) = fixture::loaded();
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, 1)
        .expect("the cache has room");
    let pass = model
        .plan_chunk(session, &fixture::tokens(1, 1), 1)
        .expect("a chunk of this width is planned");
    let exec = model
        .stack()
        .executor(session)
        .expect("a live session has an executor");
    // The reservation is the pass's first step, before any island: growing it
    // later would move an address a captured island has baked in, and a
    // reservation attempted mid capture is refused by the device outright.
    let first = pass.plan.steps().first().expect("a pass has steps");
    assert!(
        matches!(first, Step::Reserve { bytes } if *bytes > 0),
        "the pass opens by reserving its cooperative scratch: {first:?}"
    );
    let first_island = pass.islands.first().expect("a pass has islands").start;
    assert!(
        first_island > 0,
        "the reservation sits in front of the first island"
    );
    for step in pass.plan.steps() {
        if let Step::Reserve { .. } = step {
            exec.run_step(step).expect("the fake device reserves");
        }
    }
    let reserved = backend.reservations();
    // One for the whole pass rather than one per layer, and wide enough for
    // the lock chain plus the widest staging any launch in it carves.
    assert_eq!(reserved.len(), 1);
    let bytes = reserved[0];
    assert!(
        bytes > shiso_engine::sparse::exl3::LOCK_BYTES,
        "the reservation covers the lock chain and staging behind it: {bytes}"
    );
    drop(model);
    drop(dir);
}

#[test]
fn a_template_dispatch_is_a_named_refusal_on_a_build_that_captured_no_entry_names() {
    use shiso_engine::sparse::kernels::{Elem, EntryTable, TemplateArgs};

    let empty = EntryTable::default();
    let entry = shiso_engine::sparse::kernels::Entry::template(
        shiso_engine::sparse::kernels::MOD_HC,
        "q4e_hc_norm_kernel",
        TemplateArgs::Elem(Elem::F32),
    );
    let err = empty
        .resolve(entry)
        .expect_err("a build with no captured names cannot resolve a template");
    assert!(
        matches!(err, CoreError::Unsupported { what, .. } if what.contains("never captured")),
        "{err:?}"
    );
}
