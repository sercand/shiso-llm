//! The gate's arithmetic, checked without a model, a device or a repository.
//!
//! Everything the gate decides is a pure function of committed JSON, so all of
//! it can be exercised here: the percentile math against distributions whose
//! answers are known by hand, the ratio threshold at its exact boundary, the
//! staleness rule in each of the states it distinguishes, and the refusal to
//! record a noisy run.
//!
//! The last test runs the gate binary itself and asserts that it wrote nothing.
//! That is the one property no unit test can stand in for: a gate that can
//! write is a gate that can move its own reference.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use shiso_bench::baseline::{
    Baseline, Comparison, DEFAULT_FAMILY, GATE_RATIO, GEMMA4_FAMILY, Metrics, PerfEntry,
    QWEN4EXP_FAMILY, Record, Reference, Report, ResumeEntry, SCHEMA_VERSION, Spread, Stale, Target,
    TreeState, Verdict, baseline_file, grade_perf, still_stands,
};
use shiso_bench::kv;
use shiso_bench::measure::{Protocol, SPREAD_MAX, median, spread};
use shiso_bench::ple::WaitHistogram;

/// A histogram of ten buckets, each ten microseconds wide.
fn histogram(samples: &[u64]) -> WaitHistogram {
    let mut hist = WaitHistogram::new(10, 10);
    for sample in samples {
        hist.record(*sample);
    }
    hist
}

#[test]
fn an_empty_distribution_has_no_quantile() {
    let hist = histogram(&[]);
    assert_eq!(hist.p99(), None);
    assert_eq!(hist.mean_ms(), None);
    assert_eq!(hist.max_ms(), None);
}

#[test]
fn one_sample_is_its_own_every_quantile() {
    let hist = histogram(&[5]);
    assert_eq!(hist.count(), 1);
    // Reported at the bucket's upper edge but never past the largest wait
    // actually seen, so a lone sample reports itself rather than its bucket.
    assert_eq!(hist.p50(), Some(0.005));
    assert_eq!(hist.p99(), Some(0.005));
    assert_eq!(hist.p999(), Some(0.005));
}

#[test]
fn ties_collapse_to_the_one_value() {
    let hist = histogram(&[7, 7, 7, 7]);
    assert_eq!(hist.p50(), Some(0.007));
    assert_eq!(hist.p99(), Some(0.007));
    assert_eq!(hist.mean_ms(), Some(0.007));
}

#[test]
fn an_odd_count_takes_the_middle_rank() {
    // Ranks: ceil(0.50 * 3) = 2, which is the second sample, in bucket 1.
    let hist = histogram(&[5, 15, 25]);
    assert_eq!(hist.p50(), Some(0.020));
    assert_eq!(hist.p99(), Some(0.025));
}

#[test]
fn an_even_count_takes_the_upper_of_the_two_middles() {
    // Nearest-rank, not interpolation: ceil(0.50 * 4) = 2.
    let hist = histogram(&[5, 15, 25, 35]);
    assert_eq!(hist.p50(), Some(0.020));
    assert_eq!(hist.p99(), Some(0.035));
}

#[test]
fn a_sample_on_a_bucket_boundary_reports_the_edge_above_it() {
    // Ten microseconds is the first microsecond of bucket one, and bucket one
    // is reported at twenty. Rounding the other way would let a regression of
    // up to one bucket width hide inside the rounding.
    let hist = histogram(&[10, 95]);
    assert_eq!(hist.p50(), Some(0.020));
}

#[test]
fn a_wait_past_the_last_bucket_reports_the_largest_seen() {
    let hist = histogram(&[5, 5_000]);
    assert_eq!(hist.overflow(), 1);
    assert_eq!(hist.p99(), Some(5.0));
    assert_eq!(hist.max_ms(), Some(5.0));
}

#[test]
fn the_mean_is_over_every_sample_including_the_overflow() {
    let hist = histogram(&[0, 10, 5_000]);
    assert_eq!(hist.count(), 3);
    let mean = hist.mean_ms().expect("three samples have a mean");
    assert!((mean - 5.010 / 3.0).abs() < 1e-12, "mean was {mean}");
}

#[test]
fn the_median_of_an_even_count_is_the_midpoint() {
    assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    assert_eq!(median(&[3.0, 1.0]), Some(2.0));
    assert_eq!(median(&[]), None);
}

#[test]
fn the_spread_is_the_slowest_over_the_fastest() {
    let value = spread(&[100.0, 102.0]).expect("two samples have a spread");
    assert!((value - 0.02).abs() < 1e-12, "spread was {value}");
    assert_eq!(spread(&[5.0]), Some(0.0));
    assert_eq!(spread(&[]), None);
}

#[test]
fn the_ratio_gate_passes_at_exactly_the_threshold_and_fails_just_past_it() {
    let at = Comparison {
        name: "decode".to_owned(),
        baseline: 100.0,
        measured: 100.0 * GATE_RATIO,
    };
    assert!(at.passed(), "ratio {} should pass", at.ratio());
    let past = Comparison {
        name: "decode".to_owned(),
        baseline: 100.0,
        measured: 105.01,
    };
    assert!(!past.passed(), "ratio {} should fail", past.ratio());
}

#[test]
fn a_baseline_of_zero_fails_rather_than_dividing_quietly() {
    let comparison = Comparison {
        name: "decode".to_owned(),
        baseline: 0.0,
        measured: 1.0,
    };
    assert!(!comparison.passed());
    assert!(comparison.ratio().is_infinite());
}

/// A record whose numbers match its baseline exactly.
fn perf_record(digest: &str, spread: Spread) -> Record {
    Record {
        target: Target::Metal,
        family: DEFAULT_FAMILY.to_owned(),
        git_rev: "0123456789abcdef".to_owned(),
        date: "2026-09-15".to_owned(),
        source_digest: digest.to_owned(),
        perf: Some(perf_entry(spread)),
        resume: None,
        ple: None,
        reference: None,
        ple_holding: None,
    }
}

/// The payload a record and a baseline share.
fn perf_entry(spread: Spread) -> PerfEntry {
    PerfEntry {
        schema_version: SCHEMA_VERSION,
        machine: "Test16,1".to_owned(),
        os: "test".to_owned(),
        date: "2026-09-15".to_owned(),
        git_rev: "0123456789abcdef".to_owned(),
        model: "test.gguf".to_owned(),
        backend: "metal".to_owned(),
        ctx: 4096,
        prefill_tokens: 64,
        prefill_tokens_per_sec: 50.0,
        decode_tokens: 128,
        decode_tokens_per_sec: 6.0,
        peak_rss_bytes: 0,
        protocol: Some(Protocol::standard()),
        metrics: Some(Metrics {
            decode_ms_step_median: 160.0,
            prefill_ms_median: 1280.0,
        }),
        spread: Some(spread),
    }
}

/// The spread a quiet machine produces.
fn steady() -> Spread {
    Spread {
        decode: 0.005,
        prefill: 0.004,
    }
}

#[test]
fn a_record_stands_when_the_measured_sources_are_unchanged() {
    // A documentation-only change leaves the digest alone, which is the whole
    // point: the gate must be green on a commit that changed nothing measured.
    let record = perf_record("abc123", steady());
    let tree = TreeState {
        digest: "abc123".to_owned(),
        revision_known: true,
    };
    assert_eq!(still_stands(&record, &tree), Ok(()));
}

#[test]
fn a_record_goes_stale_when_a_measured_source_changes() {
    let record = perf_record("abc123", steady());
    let tree = TreeState {
        digest: "def456".to_owned(),
        revision_known: true,
    };
    let Err(Stale::SourceChanged {
        target,
        family,
        recorded,
        current,
    }) = still_stands(&record, &tree)
    else {
        panic!("a changed source must make the record stale");
    };
    assert_eq!(target, Target::Metal);
    assert_eq!(family, DEFAULT_FAMILY);
    assert_eq!(recorded, "abc123");
    assert_eq!(current, "def456");
}

#[test]
fn a_record_from_another_history_is_refused() {
    let record = perf_record("abc123", steady());
    let tree = TreeState {
        digest: "abc123".to_owned(),
        revision_known: false,
    };
    assert!(matches!(
        still_stands(&record, &tree),
        Err(Stale::UnknownRevision(_))
    ));
}

#[test]
fn a_noisy_record_is_not_gradeable() {
    let noisy = Spread {
        decode: SPREAD_MAX + 0.001,
        prefill: 0.001,
    };
    assert!(!noisy.steady());
    let record = perf_record("abc123", noisy);
    let tree = TreeState {
        digest: "abc123".to_owned(),
        revision_known: true,
    };
    assert!(matches!(
        still_stands(&record, &tree),
        Err(Stale::Unsteady { .. })
    ));
}

#[test]
fn a_record_without_the_protocol_cannot_be_graded() {
    let mut record = perf_record("abc123", steady());
    if let Some(perf) = record.perf.as_mut() {
        perf.metrics = None;
    }
    let tree = TreeState {
        digest: "abc123".to_owned(),
        revision_known: true,
    };
    assert!(matches!(
        still_stands(&record, &tree),
        Err(Stale::NoProtocol(Target::Metal))
    ));
}

#[test]
fn an_unchanged_measurement_grades_as_a_pass() {
    let mut report = Report::default();
    grade_perf(
        Target::Metal,
        DEFAULT_FAMILY,
        &perf_entry(steady()),
        &perf_entry(steady()),
        &mut report,
    );
    assert_eq!(report.comparisons.len(), 2);
    assert_eq!(report.verdict(), Verdict::Pass);
    assert_eq!(report.verdict().code(), 0);
}

#[test]
fn a_decode_step_that_got_slower_than_the_threshold_is_a_regression() {
    let mut measured = perf_entry(steady());
    if let Some(metrics) = measured.metrics.as_mut() {
        metrics.decode_ms_step_median *= 1.06;
    }
    let mut report = Report::default();
    grade_perf(
        Target::Metal,
        DEFAULT_FAMILY,
        &measured,
        &perf_entry(steady()),
        &mut report,
    );
    assert_eq!(report.verdict(), Verdict::Regression);
    assert_eq!(report.verdict().code(), 1);
}

#[test]
fn staleness_outranks_a_regression() {
    let mut report = Report::default();
    report.stale.push(Stale::NoRecord(Target::Metal));
    assert_eq!(report.verdict(), Verdict::Stale);
    assert_eq!(report.verdict().code(), 2);
}

/// A resume entry with a plausible shape.
fn resume_entry(resume_ms: f64) -> ResumeEntry {
    ResumeEntry {
        machine: "Test16,1".to_owned(),
        model: "test.gguf".to_owned(),
        ctx: 2048,
        prompt_tokens: 1024,
        cold_prefill_ms: 17_920.0,
        resume_ms,
        tokens_reused: 1024,
        continuation_tokens_equal: true,
    }
}

#[test]
fn a_resume_that_is_slower_than_its_cold_prefill_fails_its_invariant() {
    let mut record = resume_entry(24.5);
    record.cold_prefill_ms = 20.0;
    let mut report = Report::default();
    kv::grade(&record, &resume_entry(24.5), &mut report);
    assert_eq!(report.verdict(), Verdict::Regression);
}

#[test]
fn a_resume_that_continued_differently_fails_before_any_timing_matters() {
    let mut record = resume_entry(1.0);
    record.continuation_tokens_equal = false;
    let mut report = Report::default();
    kv::grade(&record, &resume_entry(24.5), &mut report);
    assert!(report.comparisons.iter().all(Comparison::passed));
    assert_eq!(report.verdict(), Verdict::Regression);
}

#[test]
fn the_v1_field_names_still_parse() {
    // The first schema, verbatim. A reader written against it must keep
    // working, which means the names and their meanings do not move.
    let text = r#"{
      "Mac16,10": {
        "machine": "Mac16,10", "os": "macOS 26.6.2", "date": "2026-09-14",
        "git_rev": "3fbda30", "model": "Qwen3.5-4B-Q4_K_M.gguf", "backend": "metal",
        "ctx": 4096, "prefill_tokens": 64, "prefill_tokens_per_sec": 49.96,
        "decode_tokens": 128, "decode_tokens_per_sec": 6.12, "peak_rss_bytes": 0
      }
    }"#;
    let file: Baseline<PerfEntry> = serde_json::from_str(text).expect("the first schema parses");
    let entry = file.get("Mac16,10").expect("one machine");
    assert_eq!(entry.schema_version, 1);
    assert_eq!(entry.prefill_tokens, 64);
    assert!(entry.metrics.is_none(), "v1 carries no medians to gate on");
}

#[test]
fn a_record_round_trips_through_its_own_schema() {
    let record = perf_record("abc123", steady());
    let text = serde_json::to_string(&record).expect("a record renders");
    let back: Record = serde_json::from_str(&text).expect("and parses back");
    assert_eq!(record, back);
}

#[test]
fn a_record_written_before_the_family_key_reads_as_the_family_it_was() {
    // Every record in the tree at the time this key was added measured the
    // dense hybrid, so the absence of the key has to keep meaning that one. A
    // default of anything else would silently re-file the whole history under a
    // pair that has no baseline.
    let text = r#"{
        "target": "metal",
        "git_rev": "0123456789abcdef",
        "date": "2026-09-15",
        "source_digest": "abc123"
    }"#;
    let record: Record = serde_json::from_str(text).expect("a pre-family record parses");
    assert_eq!(record.family, DEFAULT_FAMILY);
    assert_eq!(record.reference, None);
    assert_eq!(record.file_name(), "metal-20260915-0123456789ab.json");
}

#[test]
fn a_second_family_on_one_device_is_a_separate_baseline_and_a_separate_file() {
    // The two halves of keeping two families on one machine apart: the record
    // that names one is not written over the record that names the other, and
    // neither is graded against the other's baseline.
    let mut record = perf_record("abc123", steady());
    record.family = GEMMA4_FAMILY.to_owned();
    assert_eq!(
        record.file_name(),
        "metal-gemma4-20260915-0123456789ab.json"
    );
    assert_ne!(
        record.file_name(),
        perf_record("abc123", steady()).file_name()
    );

    assert_eq!(
        baseline_file(Target::Metal, DEFAULT_FAMILY),
        Target::Metal.baseline_file(),
        "the pair that already had a baseline keeps the file it was promoted into"
    );
    assert_ne!(
        baseline_file(Target::Metal, GEMMA4_FAMILY),
        baseline_file(Target::Metal, DEFAULT_FAMILY)
    );
    assert_eq!(
        baseline_file(Target::Cuda, DEFAULT_FAMILY),
        Target::Cuda.baseline_file(),
        "the pair that already had a baseline keeps the file it was promoted into"
    );
    assert_eq!(
        baseline_file(Target::Cuda, DEFAULT_FAMILY),
        "nvidia-qwen35.json",
        "the default CUDA pair is named for the family it measures. It was named for the sparse \
         checkpoint until Phase 8, from when the only CUDA record in the tree had measured that \
         checkpoint and carried no family key - harmless until a real dense-hybrid record existed \
         beside it, and a Qwen3.5-4B graded against a Qwen3.8-Flash-Next the moment one did"
    );
}

#[test]
fn no_family_can_fall_through_onto_another_familys_baseline() {
    // The trap this closes: a pair with no baseline of its own resolving to the
    // file of whichever family happened to be recorded on that target first. A
    // Gemma 4 record from the NVIDIA box would then have been graded against the
    // sparse checkpoint's numbers, under the same machine key, and passed or
    // failed for reasons that have nothing to do with either model.
    for target in Target::all() {
        let derived = baseline_file(target, GEMMA4_FAMILY);
        assert!(
            derived.contains(GEMMA4_FAMILY),
            "{target:?} + gemma4 resolved to {derived}, which does not name the family"
        );
        assert_ne!(derived, baseline_file(target, DEFAULT_FAMILY));
        assert_ne!(derived, target.baseline_file());
        for other in Target::all() {
            assert_ne!(
                derived,
                other.baseline_file(),
                "{target:?} + gemma4 resolved onto {other:?}'s own baseline"
            );
        }
    }
    // And the two the review asked for by name, spelled out.
    assert_eq!(
        baseline_file(Target::Cuda, GEMMA4_FAMILY),
        "nvidia-gemma4.json"
    );
    assert_eq!(
        baseline_file(Target::Resume, GEMMA4_FAMILY),
        "mac16-10-gemma4-resume.json"
    );
    assert_eq!(
        baseline_file(Target::Metal, GEMMA4_FAMILY),
        "macbookpro18-4-gemma4.json"
    );
}

#[test]
fn qwen4exp_is_its_own_cuda_pair_and_keeps_the_file_it_always_had() {
    // (Cuda, "qwen4exp") resolved onto the default pair's file until Phase 8,
    // because the default pair's file WAS the sparse checkpoint's: the only
    // CUDA record in the tree had measured it and predated the family key, so
    // it read as the dense hybrid and the baseline it was graded against was
    // named for what it had really measured. That conflation cost nothing
    // while no dense-hybrid CUDA record existed and cost a false failure the
    // moment one did.
    //
    // Both halves are settled the same way: the historical record now names
    // `qwen4exp` in its own `family` field, and the default CUDA pair is named
    // for the family it measures. So this pair needs no special arm - the
    // derived name is the file it has always been graded against - and the two
    // pairs are two files.
    assert_eq!(
        baseline_file(Target::Cuda, QWEN4EXP_FAMILY),
        "nvidia-qwen4exp.json",
        "the sparse family keeps the file its numbers have always been in"
    );
    assert_ne!(
        baseline_file(Target::Cuda, QWEN4EXP_FAMILY),
        baseline_file(Target::Cuda, DEFAULT_FAMILY),
        "two checkpoints measured on one device are two baselines, not one"
    );
}

#[test]
fn a_record_says_which_holding_produced_its_number() {
    // Absent for a family with no per-layer table, and round-tripped when set,
    // so a resident run cannot be read as the streamed one it is not comparable
    // with.
    let mut record = perf_record("abc123", steady());
    assert_eq!(record.ple_holding, None);
    record.family = GEMMA4_FAMILY.to_owned();
    record.ple_holding = Some("resident".to_owned());
    let text = serde_json::to_string(&record).expect("a record with a holding renders");
    let back: Record = serde_json::from_str(&text).expect("and parses back");
    assert_eq!(back.ple_holding.as_deref(), Some("resident"));
}

#[test]
fn a_reference_section_survives_the_round_trip_and_gates_nothing() {
    let mut record = perf_record("abc123", steady());
    record.reference = Some(Reference {
        tool: "llama-bench".to_owned(),
        commit: "c6824a9e".to_owned(),
        decode_tok_s: 22.14,
        prefill_tok_s: 296.38,
        args: "-m model.gguf -p 512 -n 128 -ngl 99 -o json".to_owned(),
        n_prompt: Some(512),
        n_gen: Some(128),
    });
    let text = serde_json::to_string(&record).expect("a record with a reference renders");
    let back: Record = serde_json::from_str(&text).expect("and parses back");
    assert_eq!(record, back);
    assert_eq!(back.reference.as_ref().and_then(|r| r.n_prompt), Some(512));
    assert_eq!(back.reference.and_then(|r| r.n_gen), Some(128));

    // Nothing the reference carries reaches the arithmetic: grading a record
    // against its baseline produces the same two comparisons with or without it.
    let mut with = Report::default();
    grade_perf(
        Target::Metal,
        DEFAULT_FAMILY,
        &perf_entry(steady()),
        &perf_entry(steady()),
        &mut with,
    );
    assert_eq!(with.comparisons.len(), 2);
    assert_eq!(with.verdict(), Verdict::Pass);
}

#[test]
fn any_committed_baseline_parses() {
    // A baseline that is not committed yet is the perf gate's "nothing is gated
    // yet" state, not a failure. One that IS committed has to parse.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("baseline");
    let perf = dir.join(Target::Metal.baseline_file());
    if perf.exists() {
        let text = std::fs::read_to_string(&perf).expect("the throughput baseline reads");
        let file: Baseline<PerfEntry> = serde_json::from_str(&text).expect("and parses");
        assert!(!file.is_empty(), "a baseline with no machine gates nothing");
    }

    let resume = dir.join(Target::Resume.baseline_file());
    if !resume.exists() {
        return;
    }
    let file = kv::read(&resume).expect("the resume baseline parses");
    for entry in file.values() {
        assert!(
            entry.resume_ms < entry.cold_prefill_ms,
            "a recorded resume that cost more than its cold prefill is not a resume"
        );
    }
}

#[test]
fn the_gate_writes_nothing() {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let before = fingerprint(&crate_dir.join("baseline"), &crate_dir.join("records"));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_perf_gate"))
        .arg("check")
        .output()
        .expect("the gate runs");
    let after = fingerprint(&crate_dir.join("baseline"), &crate_dir.join("records"));
    assert_eq!(
        before,
        after,
        "the gate changed its own inputs:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let code = out.status.code().expect("the gate exits with a code");
    assert!(
        (0..=2).contains(&code),
        "the gate exited {code}, which is not one of pass, regression or stale:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// Every file under the given directories, with its bytes.
fn fingerprint(baseline: &Path, records: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut all = BTreeMap::new();
    for dir in [baseline, records] {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if let Ok(bytes) = std::fs::read(entry.path()) {
                all.insert(entry.path(), bytes);
            }
        }
    }
    all
}
