//! What the fixture reader accepts, what it refuses, and - the part that
//! matters - what the comparator refuses.
//!
//! Nothing here needs a device or a checkpoint, so all of it runs in the default
//! gate on every machine.
//!
//! # The mutation battery is the point of this file
//!
//! A parity harness that only fails on a crash or a `NaN` is worth very little,
//! because the two defects this project has actually shipped were neither. One
//! was a kernel computing the wrong activation; the other was a preprocessor
//! writing the right numbers along the wrong axis. Both produced output that was
//! finite, plausibly shaped, and wrong, and both passed every shape check and
//! every unit test in the tree.
//!
//! So the battery below takes the *real* recorded slice from the committed
//! golden and damages it the way those defects would have: a uniform shift, a
//! scale, a non-uniform wobble, an off-by-one in the ranks, a transposed block,
//! a near-tie swapped. Each mutation asserts which budget catches it. That is
//! the harness's sensitivity, measured rather than asserted, and it is what a
//! reviewer should read to decide whether this file is doing any work.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use shiso_bench::compare::{FLOOR, STRICT, TopSlice, compare, top_k};
use shiso_bench::vecfile::{self, Body, Grammar, Mode, TopEntry, VecError};

/// Where the committed fixtures live.
fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors")
}

/// The recorded slice of one case of the committed golden.
fn golden_slice(case_id: &str) -> TopSlice {
    let path = vectors_dir().join("goldens/qwen35-local-golden.vec");
    let file = vecfile::read(&path).expect("the committed golden parses");
    let Body::Local(cases) = file.body else {
        panic!("the committed golden is a token-id fixture");
    };
    cases
        .into_iter()
        .find(|c| c.id == case_id)
        .unwrap_or_else(|| panic!("the committed golden carries a case named {case_id}"))
        .tops
}

/// Parses text that is expected to fail, returning the error.
fn reject(text: &str) -> VecError {
    vecfile::parse(Path::new("fixture.vec"), text).expect_err("this fixture must be refused")
}

// ---------------------------------------------------------------------------
// Grammar conformance
// ---------------------------------------------------------------------------

#[test]
fn the_token_id_grammar_parses_to_the_cases_and_entries_it_spells() {
    let path = vectors_dir().join("grammar-samples/local-logit.vec");
    let file = vecfile::read(&path).expect("the sample parses");
    let Body::Local(cases) = &file.body else {
        panic!("this sample is written in the token-id grammar");
    };
    assert_eq!(cases.len(), 2, "the sample declares two cases");

    let first = &cases[0];
    assert_eq!(first.id, "sample_text");
    assert_eq!(first.mode, Mode::Text);
    assert_eq!(first.ctx, 64);
    assert_eq!(first.frontier, 32);
    assert_eq!(first.top_count, 3);
    assert_eq!(first.tops.len(), 3);
    assert_eq!(first.tops[0].token, 4371);
    assert!((first.tops[0].logit - 46.916_035).abs() < 1e-5);

    let second = &cases[1];
    assert_eq!(
        second.mode,
        Mode::Rendered,
        "the second case is pre-rendered"
    );
    assert!(
        (second.tops[1].logit + 3.25).abs() < f32::EPSILON,
        "a negative logit round-trips exactly, got {}",
        second.tops[1].logit
    );

    assert_eq!(
        file.headers.get("label").map(String::as_str),
        Some("parser-conformance-only"),
        "a header comment carries provenance and survives parsing"
    );
}

#[test]
fn the_token_byte_grammar_decodes_hex_to_the_bytes_it_names() {
    let path = vectors_dir().join("grammar-samples/official-logprob.vec");
    let file = vecfile::read(&path).expect("the sample parses");
    let Body::Official(cases) = &file.body else {
        panic!("this sample is written in the token-byte grammar");
    };
    let steps = &cases[0].steps;
    assert_eq!(steps.len(), 2);
    assert_eq!(
        steps[0].selected, b"Ada",
        "416461 is the UTF-8 for \"Ada\", and the reader has to say so"
    );
    assert_eq!(steps[0].tops[0].token, b"Ada");
    assert_eq!(steps[0].tops[1].token, b"Adabo");
    assert_eq!(steps[1].selected, b" Lov");
    assert!(
        steps[0].tops[0].logprob.abs() < f32::EPSILON,
        "the selected token's log-probability is zero in this sample"
    );
}

#[test]
fn a_fixture_survives_a_round_trip_through_the_writer_byte_for_byte() {
    for sample in [
        "grammar-samples/local-logit.vec",
        "grammar-samples/official-logprob.vec",
    ] {
        let path = vectors_dir().join(sample);
        let parsed = vecfile::read(&path).expect("the sample parses");
        let rendered = vecfile::write(&parsed);
        let reparsed =
            vecfile::parse(&path, &rendered).expect("what the writer produced parses again");
        assert_eq!(
            parsed, reparsed,
            "{sample}: writing and re-reading must not change a fixture"
        );
        assert_eq!(
            rendered,
            vecfile::write(&reparsed),
            "{sample}: the writer's output must be a function of its input alone"
        );
    }
}

#[test]
fn the_committed_golden_is_a_complete_fixture_and_not_a_skeleton() {
    let file = vecfile::read(&vectors_dir().join("goldens/qwen35-local-golden.vec"))
        .expect("the committed golden parses");
    let Body::Local(cases) = &file.body else {
        panic!("the committed golden is a token-id fixture");
    };
    assert_eq!(
        cases.len(),
        5,
        "the golden records every case of the skeleton"
    );
    for case in cases {
        assert!(
            !case.is_skeleton(),
            "case {} carries no entries, so the capture did not complete",
            case.id
        );
        assert_eq!(
            case.tops.len(),
            64,
            "case {} records a 64-wide slice",
            case.id
        );
    }
    for key in [
        "label",
        "model_bytes",
        "vocabulary_digest",
        "prompts_digest",
    ] {
        assert!(
            file.headers.contains_key(key),
            "a golden that cannot say its {key} cannot be argued with"
        );
    }
    assert!(
        file.notes.iter().any(|n| n.contains("regression lock")),
        "the file has to say in its own text what it does and does not prove"
    );
}

#[test]
fn the_capture_skeleton_declares_cases_and_records_nothing() {
    let file = vecfile::read(&vectors_dir().join("goldens/qwen35-local-golden.skeleton.vec"))
        .expect("the skeleton parses");
    let Body::Local(cases) = &file.body else {
        panic!("the skeleton is a token-id fixture");
    };
    assert_eq!(cases.len(), 5);
    assert!(
        cases.iter().all(vecfile::LocalCase::is_skeleton),
        "a skeleton is a capture request, so it carries no recorded entries"
    );
}

#[test]
fn the_two_grammars_are_told_apart_by_their_first_line() {
    assert_eq!(
        Grammar::LocalLogit.header(),
        "# shiso-local-logit-vectors-v1"
    );
    assert!(matches!(
        reject("# something-else-v1\n"),
        VecError::UnknownGrammar { .. }
    ));
}

// ---------------------------------------------------------------------------
// What the reader refuses
// ---------------------------------------------------------------------------

#[test]
fn every_malformed_line_produces_the_error_that_names_it() {
    let header = "# shiso-local-logit-vectors-v1\n";
    let case = "case c text 64 32 p.txt 2\n";

    let out_of_order = reject(&format!("{header}{case}top 0 1 1.0\ntop 2 2 0.5\n"));
    assert!(
        matches!(&out_of_order, VecError::Invalid { detail, .. } if detail.contains("sequential")),
        "ranks must be sequential from zero, got {out_of_order}"
    );

    let too_deep = {
        let mut text = "# shiso-local-logit-vectors-v1\ncase c text 4096 32 p.txt 128\n".to_owned();
        for rank in 0..129 {
            let _ = writeln!(text, "top {rank} {rank} 1.0");
        }
        reject(&text)
    };
    assert!(
        matches!(&too_deep, VecError::Invalid { detail, .. } if detail.contains("more than 128")),
        "a top-k block wider than the readers' buffer is refused, got {too_deep}"
    );

    let frontier_too_big = reject(&format!("{header}case c text 32 32 p.txt 1\ntop 0 1 1.0\n"));
    assert!(
        matches!(&frontier_too_big, VecError::Invalid { detail, .. } if detail.contains("ctx must exceed frontier")),
        "a frontier that fills the context leaves no room to decode, got {frontier_too_big}"
    );

    let not_hex = reject("# shiso-official-logprob-vectors-v1\ncase c 16 1 p.txt\nstep 0 zz 1\n");
    assert!(
        matches!(
            &not_hex,
            VecError::Malformed {
                keyword: "step",
                ..
            }
        ),
        "token bytes that are not hex name the step line, got {not_hex}"
    );

    let odd_hex = reject("# shiso-official-logprob-vectors-v1\ncase c 16 1 p.txt\nstep 0 416 1\n");
    assert!(
        matches!(&odd_hex, VecError::Malformed { detail, .. } if detail.contains("even-length")),
        "half a byte is not a byte, got {odd_hex}"
    );

    let short_case = reject(&format!("{header}case c text 64 32\n"));
    assert!(
        matches!(
            &short_case,
            VecError::Malformed {
                keyword: "case",
                ..
            }
        ),
        "a case line missing a column names the case line, got {short_case}"
    );

    let count_disagrees = reject(&format!("{header}{case}top 0 1 1.0\n"));
    assert!(
        matches!(&count_disagrees, VecError::Invalid { detail, .. } if detail.contains("declares 2")),
        "a case that declares more entries than it carries was truncated, got {count_disagrees}"
    );

    let orphan = reject(&format!("{header}top 0 1 1.0\n"));
    assert!(
        matches!(&orphan, VecError::Invalid { detail, .. } if detail.contains("before any case")),
        "entries have to belong to a case, got {orphan}"
    );

    let unknown = reject(&format!("{header}{case}tops 0 1 1.0\n"));
    assert!(
        matches!(&unknown, VecError::Invalid { detail, .. } if detail.contains("unknown keyword")),
        "a keyword the grammar does not have is a typo, not a comment, got {unknown}"
    );
}

#[test]
fn a_non_finite_logit_is_refused_at_the_door() {
    let refused =
        reject("# shiso-local-logit-vectors-v1\ncase c text 64 32 p.txt 1\ntop 0 1 NaN\n");
    assert!(
        matches!(&refused, VecError::Invalid { detail, .. } if detail.contains("not finite")),
        "a fixture carrying NaN compares equal to nothing and unequal to nothing, got {refused}"
    );
}

// ---------------------------------------------------------------------------
// The top-k extractor
// ---------------------------------------------------------------------------

#[test]
fn the_top_k_extractor_breaks_ties_the_way_the_greedy_sampler_does() {
    let row = [1.0_f32, 5.0, 5.0, 2.0, 5.0];
    let slice = top_k(&row, 3);
    assert_eq!(
        slice.iter().map(|e| e.token).collect::<Vec<_>>(),
        vec![1, 2, 4],
        "three entries share the maximum, so the lower id has to win each tie"
    );
    assert_eq!(
        slice[0].token,
        shiso_engine::argmax(&row).get(),
        "rank zero and the sampler's choice are the same token or the fixture is lying"
    );
}

#[test]
fn the_top_k_extractor_drops_the_entries_a_sampler_could_not_choose() {
    let row = [1.0_f32, f32::NAN, 7.0, f32::INFINITY];
    let slice = top_k(&row, 4);
    assert_eq!(
        slice.iter().map(|e| e.token).collect::<Vec<_>>(),
        vec![2, 0],
        "a row's non-finite entries are not candidates and must not become ranks"
    );
}

// ---------------------------------------------------------------------------
// The mutation battery
// ---------------------------------------------------------------------------

/// Applies `f` to every recorded logit, keeping the ids and the rank order.
fn perturbed(slice: &TopSlice, f: impl Fn(usize, f32) -> f32) -> TopSlice {
    slice
        .iter()
        .enumerate()
        .map(|(rank, e)| TopEntry {
            token: e.token,
            logit: f(rank, e.logit),
        })
        .collect()
}

#[test]
fn an_undamaged_slice_passes_both_budgets() {
    let slice = golden_slice("long_memory_archive");
    for budget in [&FLOOR, &STRICT] {
        let verdict = compare("control", &slice, &slice, budget);
        assert!(
            verdict.passed(),
            "a slice compared against itself must pass {}: {:?}",
            budget.name,
            verdict.failures
        );
        assert_eq!(
            verdict.max_abs.to_bits(),
            0.0_f32.to_bits(),
            "a slice compared against itself moved by nothing at all"
        );
    }
}

#[test]
fn a_uniform_shift_of_one_hundredth_of_a_nat_is_caught_by_the_same_source_budget() {
    // The shape of a scaling or epsilon bug: every logit moves by the same
    // small amount, the ranks do not move at all, and the decoded text is
    // identical. A token-sequence golden would not notice this for a thousand
    // steps. The strict budget notices it immediately.
    let slice = golden_slice("long_memory_archive");
    let damaged = perturbed(&slice, |_, l| l + 0.01);
    let strict = compare("uniform-shift", &slice, &damaged, &STRICT);
    assert!(
        !strict.passed(),
        "a 0.01 shift has to fail the same-source budget"
    );
    assert!(strict.max_abs > 1e-3);
    assert_eq!(
        strict.reference_top1, strict.candidate_top1,
        "and it has to fail on the logits, because the winner did not move"
    );

    let floor = compare("uniform-shift", &slice, &damaged, &FLOOR);
    assert!(
        floor.passed(),
        "the cross-build floor is deliberately blind to this, which is why the strict budget exists"
    );
}

#[test]
fn a_wrong_activation_sized_wobble_is_caught_by_both_budgets() {
    // What a kernel computing the wrong activation looks like from here: every
    // logit moves, by a different amount, in a way that mostly preserves the
    // ordering near the top. Finite, plausible, and wrong.
    let slice = golden_slice("long_code_audit");
    let damaged = perturbed(&slice, |rank, l| {
        let wobble = f32::from(u16::try_from(rank % 7).unwrap_or(0)) - 3.0;
        l + wobble * 4.0
    });
    for budget in [&FLOOR, &STRICT] {
        let verdict = compare("wobble", &slice, &damaged, budget);
        assert!(
            !verdict.passed(),
            "a multi-nat non-uniform wobble must fail {}: {}",
            budget.name,
            verdict.report_line()
        );
    }
}

#[test]
fn a_transposed_block_keeps_every_number_and_still_fails() {
    // The shape of the packing defect: every value is present and correct, and
    // every one of them is in the wrong place. Nothing is NaN, nothing is out of
    // range, and the set of tokens is exactly right - so a test that compared
    // sets, or sums, or ranges would pass. Comparing rank by rank does not.
    let slice = golden_slice("long_memory_archive");
    let mut damaged = slice.clone();
    for block in damaged.chunks_mut(8) {
        block.reverse();
    }
    let verdict = compare("transposed", &slice, &damaged, &FLOOR);
    assert!(
        !verdict.passed(),
        "a block-reversed slice must fail even the loose budget: {}",
        verdict.report_line()
    );
    assert_ne!(
        verdict.reference_top1, verdict.candidate_top1,
        "reversing the first block moves the winner, which is the loudest possible failure"
    );

    let ids: std::collections::BTreeSet<u32> = slice.iter().map(|e| e.token).collect();
    let damaged_ids: std::collections::BTreeSet<u32> = damaged.iter().map(|e| e.token).collect();
    assert_eq!(
        ids, damaged_ids,
        "the damaged slice holds exactly the same tokens, which is what makes this the hard case"
    );
}

#[test]
fn an_off_by_one_in_the_ranks_fails_on_the_winner() {
    // A slice shifted up by one: rank n carries what rank n+1 should. Every
    // logit is a real logit this model produced, in descending order, from the
    // right row. Only the alignment is wrong.
    let slice = golden_slice("short_code_completion");
    let damaged: TopSlice = slice.iter().skip(1).copied().collect();
    let verdict = compare("off-by-one", &slice, &damaged, &FLOOR);
    assert!(!verdict.passed(), "{}", verdict.report_line());
    assert_ne!(verdict.reference_top1, verdict.candidate_top1);
}

#[test]
fn two_adjacent_near_ties_swapped_are_caught_as_a_broken_ordering() {
    let slice = golden_slice("short_italian_fact");
    let mut damaged = slice.clone();
    damaged.swap(1, 2);
    let verdict = compare("swap", &slice, &damaged, &STRICT);
    assert!(
        !verdict.passed(),
        "a slice whose ranks disagree with its own logits is broken, not merely different: {}",
        verdict.report_line()
    );
}

#[test]
fn a_collapsed_row_is_refused_rather_than_graded() {
    let slice = golden_slice("short_reasoning_plain");
    let flat = perturbed(&slice, |_, _| 1.5);
    let verdict = compare("collapsed", &slice, &flat, &FLOOR);
    assert!(
        verdict
            .failures
            .iter()
            .any(|f| f.to_string().contains("same score")),
        "a row where every entry scores the same carries no answer at all: {:?}",
        verdict.failures
    );
}

#[test]
fn one_case_is_never_mistaken_for_another_which_is_what_makes_the_fixture_worth_keeping() {
    // If two recorded cases were close enough to pass for each other, the file
    // would be measuring the tokenizer rather than the model. They are not, and
    // this is the test that says so.
    let a = golden_slice("long_memory_archive");
    let b = golden_slice("long_code_audit");
    let verdict = compare("cross-case", &a, &b, &FLOOR);
    assert!(
        !verdict.passed(),
        "two different prompts must not produce interchangeable slices: {}",
        verdict.report_line()
    );
}

#[test]
fn the_budgets_are_the_two_values_they_are_documented_as() {
    // A diff that changes one of these is a review stop, and this is what makes
    // such a diff visible in the test output rather than only in the source.
    assert_eq!(STRICT.overlap_min, [5, 20, 64]);
    assert_eq!(STRICT.max_abs.to_bits(), 1e-3_f32.to_bits());
    assert_eq!(FLOOR.overlap_min, [4, 15, 40]);
    assert_eq!(FLOOR.max_abs.to_bits(), 8.0_f32.to_bits());
}
