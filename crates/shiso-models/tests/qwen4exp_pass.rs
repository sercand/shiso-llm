//! What one pass through the stack is made of, and in what order.
//!
//! This is the whole forward pass as this half of the port can state it: every
//! launch a chunk would make, its grid, its shared memory, and which of them
//! belong to which captured island. None of it needs a device, and all of it is
//! what a port gets wrong - so it is asserted here rather than left for the
//! machine that can run the kernels to discover.
//!
//! See `common/fixture.rs` for the synthetic checkpoint and why its widths are
//! what they are.

use shiso_core::Model;
use shiso_engine::sparse::island::Island;

#[path = "common/qwen4exp_fixture.rs"]
mod fixture;

use fixture::{N_LAYERS, PLE_LAYER, loaded, loaded_with, speculative_options, tokens};

/// Opens `rows` sessions and claims one cache row for each.
///
/// Claiming matters here: a session that has never reserved has a page table
/// the pool has not published, and what a batch addresses per row is exactly
/// that table.
fn open_rows(
    model: &mut shiso_models::qwen4exp::Qwen4ExpModel,
    rows: u32,
) -> Vec<shiso_core::SessionId> {
    (0..rows)
        .map(|_| {
            let session = model.open_session().expect("a session opens");
            model
                .stack_mut()
                .reserve(session, 1)
                .expect("the cache has room for a row");
            session
        })
        .collect()
}

/// The names of the steps one island covers.
fn island_names(pass: &shiso_engine::sparse::stack::Pass, index: usize) -> Vec<&'static str> {
    let span = &pass.islands[index];
    pass.plan.steps()[span.start..span.start + span.len]
        .iter()
        .map(shiso_engine::sparse::plan::Step::name)
        .collect()
}

#[test]
fn a_pass_is_two_islands_a_layer_in_mixer_then_experts_order() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");

    assert_eq!(
        pass.islands.len(),
        2 * N_LAYERS as usize,
        "every layer is captured as two islands"
    );
    for (index, span) in pass.islands.iter().enumerate() {
        let layer = u32::try_from(index / 2).expect("the layer count fits");
        let half = if index % 2 == 0 {
            Island::Mixer
        } else {
            Island::Experts
        };
        assert_eq!(span.key.layer.get(), layer, "island {index}");
        assert_eq!(span.key.island, half, "island {index}");
        assert!(span.len > 0, "island {index} encodes nothing");
    }
}

#[test]
fn the_islands_tile_the_plan_without_a_gap_or_an_overlap() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");

    let first = &pass.islands[0];
    let last = &pass.islands[pass.islands.len() - 1];
    // The embedding gather and the residual broadcast come before the first
    // island; the collapsing mix and the vocabulary projection come after the
    // last one. Everything between them belongs to exactly one island.
    assert!(first.start > 0, "the pass opens outside any island");
    assert!(
        last.start + last.len < pass.plan.len(),
        "the pass closes outside any island"
    );
    for pair in pass.islands.windows(2) {
        assert_eq!(
            pair[0].start + pair[0].len,
            pair[1].start,
            "two islands leave a gap or overlap"
        );
    }
}

#[test]
fn the_n_gram_block_runs_inside_the_mixer_island_of_its_own_layer_and_nowhere_else() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");

    assert_eq!(
        pass.plan.count("q4e_ple_gated_value_kernel"),
        1,
        "exactly one slot folds in the n-gram block"
    );
    let home = 2 * PLE_LAYER as usize;
    assert!(
        island_names(&pass, home).contains(&"q4e_ple_gated_value_kernel"),
        "the block belongs to the mixer island of its own layer"
    );
    // And it is the first thing that island does: it reads and rewrites the
    // residual, so a mix that ran before it would mix a residual it was about
    // to change.
    assert_eq!(
        island_names(&pass, home).first(),
        Some(&"q4e_ple_dequant_kernel")
    );
}

#[test]
fn the_normalization_chain_breaks_in_exactly_three_places() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");

    // A mixer island launches the stream norm only when the combine before it
    // did not leave one behind. That happens at the first island of the pass and
    // at the island whose n-gram block rewrote the residual in between - and
    // nowhere else, because the fused combine does both jobs in one pass.
    //
    // The n-gram block runs three norms of its own inside its island, which is
    // why that island is counted by its mix rather than by its total.
    for layer in 0..N_LAYERS {
        let names = island_names(&pass, 2 * layer as usize);
        let norms = names.iter().filter(|n| **n == "q4e_hc_norm_kernel").count();
        let own_ngram = layer == PLE_LAYER;
        let expected = match (layer, own_ngram) {
            (_, true) => 4,  // three inside the block, one for the mix
            (0, false) => 1, // the first island of the pass
            _ => 0,
        };
        assert_eq!(norms, expected, "the mixer island of layer {layer}");
    }
    // The feed-forward island never normalizes: nothing can come between a
    // layer's token mixer and its own feed-forward.
    for layer in 0..N_LAYERS {
        let names = island_names(&pass, 2 * layer as usize + 1);
        assert_eq!(
            names.iter().filter(|n| **n == "q4e_hc_norm_kernel").count(),
            0,
            "the feed-forward island of layer {layer}"
        );
    }
}

#[test]
fn every_island_closes_with_the_fused_residual_update() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");
    for index in 0..pass.islands.len() {
        assert_eq!(
            island_names(&pass, index).last(),
            Some(&"q4e_hc_combine_norm_kernel"),
            "island {index}"
        );
    }
    assert_eq!(
        pass.plan.count("q4e_hc_combine_norm_kernel"),
        2 * N_LAYERS as usize
    );
}

#[test]
fn a_chunk_that_produces_no_logits_does_not_run_the_widest_matmul_in_the_model() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let with = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a chunk that produces a row");
    let without = model
        .plan_chunk(session, &tokens(0, 4), 0)
        .expect("a chunk that produces none");
    assert!(
        without.plan.len() < with.plan.len(),
        "a chunk with no logits row skips the collapsing mix and the projection"
    );
    // The collapsing mix is the one that stands in for an output norm, so its
    // absence is what says the tail was skipped.
    assert_eq!(
        without.plan.count("q4e_hc_collapse_kernel"),
        with.plan.count("q4e_hc_collapse_kernel") - 1
    );
}

#[test]
fn more_than_one_logit_row_brings_back_the_greedy_pick_rather_than_the_rows() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let one = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("one row plans");
    assert!(!one.plan.launches("q4e_argmax_rows_kernel"));
    let verify = model
        .plan_chunk(session, &tokens(0, 4), 4)
        .expect("a verify batch plans");
    assert_eq!(verify.plan.count("q4e_argmax_rows_kernel"), 1);
}

#[test]
fn a_decode_step_is_capturable_and_a_prefill_chunk_is_not() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    assert!(
        model
            .plan_chunk(session, &tokens(0, 1), 1)
            .expect("one row plans")
            .capturable
    );
    assert!(
        !model
            .plan_chunk(session, &tokens(0, 4), 1)
            .expect("four rows plan")
            .capturable
    );
}

#[test]
fn two_sessions_never_share_a_captured_island() {
    let (_dir, _fake, mut model) = loaded();
    let a = model.open_session().expect("a session opens");
    let b = model.open_session().expect("a second session opens");
    let pass_a = model.plan_chunk(a, &tokens(0, 1), 1).expect("a plans");
    let pass_b = model.plan_chunk(b, &tokens(9, 1), 1).expect("b plans");
    for (island_a, island_b) in pass_a.islands.iter().zip(&pass_b.islands) {
        assert_eq!(island_a.key.layer, island_b.key.layer);
        assert_ne!(
            island_a.key.pack().expect("the key fits"),
            island_b.key.pack().expect("the key fits"),
            "two sessions' islands would replay into each other's recurrent state"
        );
    }
}

#[test]
fn the_selection_is_off_until_the_context_outgrows_the_indexer_budget() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let cold = model.plan_chunk(session, &tokens(0, 1), 1).expect("plans");
    assert!(!cold.selected, "a cold session reads its whole context");
    assert!(cold.plan.launches("q4e_qsa_attention_split_kernel"));
    assert!(!cold.plan.launches("q4e_idx_score_kernel"));

    // Walk the session past the budget. Nothing is executed; the frontier is
    // what the selection is decided from.
    for _ in 0..80 {
        model
            .stack_mut()
            .advance(session, &tokens(0, 1), false, None)
            .expect("the path grows");
    }
    let warm = model.plan_chunk(session, &tokens(0, 1), 1).expect("plans");
    assert!(warm.selected, "a context past the budget selects blocks");
    assert!(warm.plan.launches("q4e_idx_score_tc_kernel"));
    assert!(warm.plan.launches("q4e_idx_topk_kernel"));
    assert!(warm.plan.launches("q4e_idx_expand_kernel"));
    assert!(!warm.plan.launches("q4e_qsa_attention_split_kernel"));

    // And the two are different captures at the same row count, which is what
    // the sparse half of the variant is for.
    for (cold_island, warm_island) in cold.islands.iter().zip(&warm.islands) {
        assert_eq!(cold_island.key.rows, warm_island.key.rows);
        assert_ne!(cold_island.key.variant(), warm_island.key.variant());
    }
}

#[test]
fn the_indexer_stores_its_keys_on_every_pass_and_selects_on_only_some() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let cold = model.plan_chunk(session, &tokens(0, 1), 1).expect("plans");
    // Two attention slots, each storing a raw key row and pooling it. Skipping
    // this would leave a hole in the table for the positions this pass wrote.
    assert_eq!(cold.plan.count("q4e_idx_store_k_kernel"), 2);
    assert_eq!(cold.plan.count("q4e_idx_pool_kernel"), 2);
}

#[test]
fn a_prefill_chunk_s_routed_experts_run_weight_stationary_and_combine_with_the_shared_one() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    // Forty tokens: wider than any decode step, verify pass or decode batch.
    let pass = model
        .plan_chunk(session, &tokens(0, 40), 1)
        .expect("a forty-token chunk plans");
    assert_eq!(pass.plan.count("trellis experts"), 0);
    assert_eq!(pass.plan.count("q4e_shared_add_kernel"), 0);
    assert_eq!(pass.plan.count("q4e_moe_combine_kernel"), 0);
    let names = island_names(&pass, 1);
    let at = |name: &str| {
        names
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("{name} runs"))
    };
    let order = [
        at("q4e_moe_route_kernel"),
        at("q4e_moe_sort_scatter_kernel"),
        at("exl3_moe_ws_invert"),
        at("exl3_moe_ws_prep_tok"),
        at("exl3_moe_ws_gateup"),
        at("exl3_moe_ws_down_h"),
        at("q4e_swiglu_kernel"),
        at("exl3_moe_ws_combine"),
    ];
    assert!(
        order.windows(2).all(|pair| pair[0] < pair[1]),
        "route, sort, the sort's inverse, the rows, gate and up, down, the shared \
         expert, and the combine that adds both, in that order: {order:?}"
    );
}

#[test]
fn the_grouped_expert_stage_has_its_staging_reserved_outside_every_island() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 3), 1)
        .expect("a three-token chunk plans");
    // The reservation is one step, and it is the pass's first: it cannot be
    // made while a stream is capturing, and every layer's fused expert stage
    // sits inside an island that may be. An undersized staging is silent
    // corruption rather than a failed launch, so the size is the widest any
    // launch in the pass carves rather than the one in front of it.
    assert_eq!(pass.plan.count("stream scratch"), 1);
    assert_eq!(
        pass.plan.names().first().copied(),
        Some("stream scratch"),
        "the reservation opens the pass"
    );
    assert!(
        !island_names(&pass, 1).contains(&"stream scratch"),
        "no island holds a reservation"
    );
    let names = pass.plan.names();
    let reserve = names
        .iter()
        .position(|n| *n == "stream scratch")
        .expect("the staging is reserved");
    let experts = names
        .iter()
        .position(|n| *n == "trellis pair")
        .expect("the experts run");
    assert!(reserve < experts);
}

#[test]
fn the_recurrent_slots_take_the_sequential_kernel_at_every_row_count_a_session_runs() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    // Six recurrent slots in eight, and the chunked kernel's floor is far above
    // any row count a decode or a verify reaches.
    for rows in [1_u32, 4, 9] {
        let pass = model
            .plan_chunk(session, &tokens(0, rows), 1)
            .expect("a chunk plans");
        assert_eq!(
            pass.plan.count("q4e_gdn_recurrent_kernel"),
            6,
            "{rows} rows"
        );
        assert_eq!(pass.plan.count("q4e_gdn_tc_prep_kernel"), 0, "{rows} rows");
    }
    // A wide prefill chunk does reach the chunked form, on tensor cores, and
    // its convolution runs parallel over tokens.
    let wide = model
        .plan_chunk(session, &tokens(0, 128), 1)
        .expect("a wide chunk plans");
    assert_eq!(wide.plan.count("q4e_gdn_tc_prep_kernel"), 6);
    assert_eq!(wide.plan.count("q4e_gdn_tc_scan_kernel"), 6);
    assert_eq!(wide.plan.count("q4e_gdn_conv_par_kernel"), 6);
    assert_eq!(wide.plan.count("q4e_gdn_conv_kernel"), 0);
    assert_eq!(wide.plan.count("q4e_gdn_chunk_kernel"), 0);
    assert_eq!(wide.plan.count("q4e_gdn_recurrent_kernel"), 0);
}

#[test]
fn a_pass_carries_the_n_gram_gather_it_will_need() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let pass = model
        .plan_chunk(session, &tokens(0, 4), 1)
        .expect("a four-token chunk plans");
    assert_eq!(
        pass.gather.len(),
        1,
        "one sequence's chunk is one gather over its consecutive positions"
    );
    let gather = &pass.gather[0];
    assert_eq!(gather.first(), 0);
    assert_eq!(gather.rows(), 4);
    // Four heads a position at this fixture's gram count.
    assert_eq!(gather.row_count(), 4 * 4);
}

/// The six kernels that carry per-sequence state through one batched launch.
const BATCHED: [&str; 5] = [
    "q4e_gdn_conv_batch_kernel",
    "q4e_gdn_recurrent_batch_kernel",
    "q4e_qsa_store_kv_batch_kernel",
    "q4e_idx_store_k_batch_kernel",
    "q4e_ple_conv_batch_kernel",
];

/// Their scalar siblings, each of which reads one session's state.
const SCALAR: [&str; 5] = [
    "q4e_gdn_conv_kernel",
    "q4e_gdn_recurrent_kernel",
    "q4e_qsa_store_kv_kernel",
    "q4e_idx_store_k_kernel",
    "q4e_ple_conv_kernel",
];

#[test]
fn a_decode_batch_of_one_is_one_sequences_own_step_and_takes_the_scalar_kernels() {
    // One row belongs to one sequence, so nothing about it is unrelated: the
    // scalar kernels address its state directly and a one-slot pointer array
    // would say what the buffer already says.
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .reserve(session, 1)
        .expect("the cache has room for a row");

    let one = model
        .stack()
        .plan_batch(&[session])
        .expect("a single-row batch is one session's own decode step");
    assert!(one.plan.launches("q4e_hc_init_kernel"));
    for entry in BATCHED {
        assert!(
            !one.plan.launches(entry),
            "{entry} is dispatched for a batch of one, which has no rows to address apart"
        );
    }
    for entry in SCALAR {
        assert!(one.plan.launches(entry), "{entry} is not dispatched");
    }
}

#[test]
fn a_wider_batch_dispatches_every_batched_kernel_and_none_of_their_scalar_siblings() {
    // The failure this rules out is the original one: the scalar kernels
    // planned over one session's page table and one session's recurrent state,
    // with the result handed back as the other rows' logits. A plan that still
    // named a scalar sibling would be exactly that, and every launch of it
    // would succeed.
    let (_dir, _fake, mut model) = loaded();
    let sessions = open_rows(&mut model, 3);
    let pass = model
        .stack()
        .plan_batch(&sessions)
        .expect("a three-row batch plans");

    for entry in BATCHED {
        assert!(pass.plan.launches(entry), "{entry} is not dispatched");
    }
    for entry in SCALAR {
        assert!(
            !pass.plan.launches(entry),
            "{entry} reads one session's state and a batch has one per row"
        );
    }
    // The pooled key pass has no batched sibling, so it is launched once per
    // row rather than once for the batch - and each launch covers one position.
    assert_eq!(
        pass.plan.count("q4e_idx_pool_kernel"),
        3 * fixture::ATTN_SLOTS,
        "the pooled key pass is planned per row per attention slot"
    );
}

#[test]
fn a_batched_pass_binds_a_per_row_array_where_a_chunk_binds_one_sessions_buffer() {
    use shiso_engine::sparse::bind::Operand;

    let (_dir, _fake, mut model) = loaded();
    let sessions = open_rows(&mut model, 2);
    let pass = model
        .stack()
        .plan_batch(&sessions)
        .expect("a two-row batch plans");

    // Position by position, against the kernel's own parameter list: the state
    // pointer of the convolution and the recurrence, and the page table of the
    // three cache-addressing entries. `sparse_kernels.rs` checks that each of
    // these positions is a pointer in the vendored declaration; this checks
    // that the pointer is the *array* rather than one row's buffer.
    for (entry, index, want) in [
        ("q4e_gdn_conv_batch_kernel", 1, Operand::RecurrentConvs(0)),
        (
            "q4e_gdn_recurrent_batch_kernel",
            1,
            Operand::RecurrentStates(0),
        ),
        ("q4e_ple_conv_batch_kernel", 1, Operand::NgramWindows),
        ("q4e_qsa_store_kv_batch_kernel", 14, Operand::PageTables),
        ("q4e_idx_store_k_batch_kernel", 3, Operand::PageTables),
    ] {
        let step = pass
            .plan
            .steps()
            .iter()
            .find(|step| step.name() == entry)
            .unwrap_or_else(|| panic!("{entry} is dispatched"));
        let ops = step.operands().expect("the list fits a launch");
        assert_eq!(ops[index], want, "{entry} parameter {index}");
    }

    // And the pooled pass, which has no array: each of its launches names one
    // row's page table, and the two launches name different rows.
    let pooled = pass
        .plan
        .steps()
        .iter()
        .filter(|step| step.name() == "q4e_idx_pool_kernel")
        .map(|step| step.operands().expect("the list fits a launch")[4])
        .collect::<Vec<_>>();
    assert_eq!(pooled[0], Operand::PagesOf(0));
    assert_eq!(pooled[1], Operand::PagesOf(1));

    // The attention is the single-row kernel, once per row, over that row's
    // own page table: a row of a batch attends exactly as a single-row step
    // of its sequence does.
    let attended = pass
        .plan
        .steps()
        .iter()
        .filter(|step| step.name() == "q4e_qsa_attention_split_kernel")
        .map(|step| step.operands().expect("the list fits a launch")[8])
        .collect::<Vec<_>>();
    assert_eq!(attended.len(), 2 * fixture::ATTN_SLOTS);
    assert_eq!(attended[0], Operand::PagesOf(0));
    assert_eq!(attended[1], Operand::PagesOf(1));
}

#[test]
fn a_batched_island_carries_no_capture_slot_at_all() {
    use shiso_engine::sparse::island::IslandContext;

    let (_dir, _fake, mut model) = loaded();
    let sessions = open_rows(&mut model, 2);
    let batch = model
        .stack()
        .plan_batch(&sessions)
        .expect("a two-row batch plans");
    assert!(!batch.capturable);
    for span in &batch.islands {
        assert_eq!(span.key.context, IslandContext::Batch);
        assert_eq!(
            span.key.pack().expect("a batch has no slot"),
            None,
            "a slot named after one of the batch's sessions would be replayed \
             for a different batch containing it"
        );
    }

    // One session's own step still has one, which is what makes the absence
    // above a decision rather than a capability nothing has.
    let one = model
        .stack()
        .plan_batch(&sessions[..1])
        .expect("a single-row batch plans");
    assert!(one.capturable);
    assert!(
        one.islands[0]
            .key
            .pack()
            .expect("a session's key fits")
            .is_some()
    );
}

#[test]
fn a_batch_that_names_one_session_twice_is_refused_rather_than_raced() {
    let (_dir, _fake, mut model) = loaded();
    let sessions = open_rows(&mut model, 2);
    let doubled = [sessions[0], sessions[1], sessions[0]];
    let error = model
        .stack()
        .plan_batch(&doubled)
        .expect_err("two rows on one session is two rows writing one state");
    let text = format!("{error:?}");
    assert!(text.contains("more than one row"), "{text}");
}

#[test]
fn the_vocabulary_projection_reads_the_chunks_last_rows_and_not_its_first() {
    // The single widest matmul in the model runs on the rows that get sampled,
    // which are the chunk's *last* ones - so it starts where they do inside the
    // collapsed block. A projection from the front of that block would have the
    // right count, the right width and the wrong tokens: a model that answers
    // fluently and ignores the end of its own prompt, with no diagnostic
    // anywhere.
    use shiso_engine::sparse::bind::Operand;
    use shiso_engine::sparse::plan::Step;
    use shiso_engine::sparse::scratch::Buf;

    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    let rows = 6_u32;
    let pass = model
        .plan_chunk(session, &tokens(0, rows), 1)
        .expect("a six-token chunk plans");

    // The projection is the last thing that reads the collapsed block, whichever
    // tier it took.
    let source = pass
        .plan
        .steps()
        .iter()
        .rev()
        .find_map(|step| match step {
            Step::Blas { a, .. } => Some(*a),
            Step::Kernel { args, .. } | Step::Trellis { args, .. } => {
                args.operands().ok()?.iter().copied().find(|op| {
                    matches!(
                        op,
                        Operand::Scratch(Buf::Mixed) | Operand::ScratchFrom(Buf::Mixed, _)
                    )
                })
            }
            _ => None,
        })
        .expect("the pass projects the collapsed block");

    let want = u64::from(rows - 1) * u64::from(fixture::HIDDEN) * 4;
    assert_eq!(
        source,
        Operand::ScratchFrom(Buf::Mixed, want),
        "the projection starts at the wrong row of the collapsed block"
    );
}

/// Every target pass takes the weight-stationary launches, whatever its width.
///
/// A single decode step, a verify pass's width and a prefill chunk all route
/// through the same kernels, because a row's arithmetic there does not depend
/// on how many rows share the pass - which is what keeps a plain step, a
/// verify and a decode batch answering alike. The grouped launches reduce in
/// another order, so a target pass that took them at one width and not at
/// another would drift between plain and speculative decoding.
#[test]
fn every_target_pass_width_routes_its_experts_weight_stationary() {
    let (_dir, _fake, mut model) = loaded();
    let session = model.open_session().expect("a session opens");
    for rows in [1, 4, 40] {
        let pass = model
            .plan_chunk(session, &tokens(0, rows), 1)
            .expect("the chunk plans");
        assert_eq!(pass.plan.count("q4e_moe_route_kernel"), N_LAYERS as usize);
        assert_eq!(
            pass.plan.count("exl3_moe_ws_gateup"),
            N_LAYERS as usize,
            "{rows} rows"
        );
        assert_eq!(pass.plan.count("trellis fan-out"), 0, "{rows} rows");
        assert_eq!(pass.plan.count("trellis experts"), 0, "{rows} rows");
        assert_eq!(
            pass.plan.count("q4e_moe_sort_scan_kernel"),
            N_LAYERS as usize,
            "{rows} rows"
        );
    }
}

/// The first frontier at which a single-row step of a session attends over a
/// selection rather than its whole context.
///
/// Found by walking a session one position at a time rather than computed from
/// the fixture's budget, so the cases below state the rule - each row as its
/// own decode step would - and not the arithmetic that implements it.
fn first_selected_frontier(model: &mut shiso_models::qwen4exp::Qwen4ExpModel) -> u32 {
    let session = model.open_session().expect("a session opens");
    for frontier in 0..1_000 {
        if model
            .plan_chunk(session, &tokens(0, 1), 1)
            .expect("plans")
            .selected
        {
            model.close_session(session).expect("the session closes");
            return frontier;
        }
        model
            .stack_mut()
            .advance(session, &tokens(0, 1), false, None)
            .expect("the path grows");
    }
    panic!("no frontier within reach selects");
}

/// The row counts `name`'s launches cover, read off their grids.
///
/// `axis` names the grid dimension that is the row count, which is x for the
/// split kernel and the top-k and z for the tensor-core attention.
fn rows_of(pass: &shiso_engine::sparse::stack::Pass, name: &str, axis: char) -> Vec<u32> {
    pass.plan
        .steps()
        .iter()
        .filter_map(|step| match step {
            shiso_engine::sparse::plan::Step::Kernel { entry, launch, .. }
                if entry.name() == name =>
            {
                Some(if axis == 'x' {
                    launch.grid.x
                } else {
                    launch.grid.z
                })
            }
            _ => None,
        })
        .collect()
}

/// A verify pass that starts two rows short of the budget attends those two
/// rows over the whole context, as the decode steps that would otherwise have
/// produced them do, and only the rows past the budget over a selection.
///
/// Deciding the selection once for the whole pass, from its last row, sent the
/// early rows through the selected-block kernel instead. Below the budget that
/// selection is every visible key, so the answer was the same attention summed
/// by a different kernel - close, not equal - and the key and value rows those
/// positions wrote in every later layer differed from a plain decode's. A
/// speculative greedy run then left a plain one's text wherever a later pick
/// was a near tie.
#[test]
fn a_pass_that_crosses_the_indexer_budget_attends_each_row_as_its_own_step_would() {
    let (_dir, _fake, mut model) = loaded();
    let budget = first_selected_frontier(&mut model);
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(session, &tokens(0, budget - 2), false, None)
        .expect("the path grows");
    let pass = model.plan_chunk(session, &tokens(0, 4), 4).expect("plans");
    assert!(pass.selected, "the pass's last row is past the budget");
    assert_eq!(
        rows_of(&pass, "q4e_qsa_attention_split_kernel", 'x'),
        vec![2; fixture::ATTN_SLOTS],
        "the two rows below the budget attend densely, in every attention slot"
    );
    assert_eq!(
        rows_of(&pass, "q4e_qsa_attention_tc_kernel", 'z'),
        vec![2; fixture::ATTN_SLOTS],
        "the two rows past it attend over their selections"
    );
    assert_eq!(
        rows_of(&pass, "q4e_idx_topk_kernel", 'x'),
        vec![2; fixture::ATTN_SLOTS],
        "and only those two rows are selected for"
    );
    assert!(
        !pass.capturable,
        "a crossing pass is planned once per sequence and never replayed"
    );
}

/// A pass whose rows are all on one side of the budget is one launch, as
/// before: the split is for the pass that straddles it and for nothing else.
#[test]
fn a_pass_wholly_past_the_budget_attends_in_one_launch() {
    let (_dir, _fake, mut model) = loaded();
    let budget = first_selected_frontier(&mut model);
    let session = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(session, &tokens(0, budget), false, None)
        .expect("the path grows");
    let pass = model.plan_chunk(session, &tokens(0, 4), 4).expect("plans");
    assert!(pass.selected);
    assert!(rows_of(&pass, "q4e_qsa_attention_split_kernel", 'x').is_empty());
    assert_eq!(
        rows_of(&pass, "q4e_qsa_attention_tc_kernel", 'z'),
        vec![4; fixture::ATTN_SLOTS]
    );
}

/// The same rule for a batch's verify: each run is split where its own rows
/// cross its own budget, and a run wholly past it is not.
#[test]
fn each_run_of_a_segmented_verify_crosses_the_budget_where_its_own_rows_do() {
    let (_dir, _fake, mut model) = loaded_with(speculative_options(3));
    let budget = first_selected_frontier(&mut model);
    let crossing = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(crossing, &tokens(0, budget - 1), false, None)
        .expect("the path grows");
    let beyond = model.open_session().expect("a session opens");
    model
        .stack_mut()
        .advance(beyond, &tokens(7, budget + 9), false, None)
        .expect("the path grows");
    let pass = model
        .stack()
        .plan_segments(&[crossing, beyond], &[4, 4])
        .expect("plans");
    assert_eq!(
        rows_of(&pass, "q4e_qsa_attention_split_kernel", 'x'),
        vec![1; fixture::ATTN_SLOTS],
        "the crossing run's one row below the budget attends densely"
    );
    assert_eq!(
        rows_of(&pass, "q4e_qsa_attention_tc_kernel", 'z'),
        [3, 4].repeat(fixture::ATTN_SLOTS),
        "the rest of the crossing run, and the whole of the other, attend over selections"
    );
}
