//! Every `metal/qwen35/*.metal` and `metal/qwen35moe/*.metal` file leaves the
//! math mode where it found it.
//!
//! `compile_library` (`src/compile.rs`) concatenates the whole kernel library
//! into one translation unit and compiles it with fast math as the build
//! default. A handful of q35 kernels need bit-identical output across their
//! lane-width variants, so they opt out for the span that has to stay exact
//! with `#pragma METAL fp math_mode(safe)` and opt back in with
//! `#pragma METAL fp math_mode(fast)`. MSL has no push/pop for that pragma:
//! it takes effect at the line it appears on and holds until the next one, so
//! a `safe` region that is not closed before end of file does not stay inside
//! that file. It runs on into whatever source `full_source` appended next -
//! silently, because there is no compiler diagnostic for "still safe at EOF".
//!
//! A device test could reveal that (two variants of a *later* file coming out
//! bit-different from each other because they were compiled safe instead of
//! fast, or the reverse), but every device test in this crate that could
//! notice is `#[ignore]`d by default, so a build with no attached GPU would
//! ship the regression. This case needs none of that: it reads the same
//! source files `compile_library` embeds and checks, per file, that every
//! `safe` is matched by a `fast` before the file ends. Pure host text
//! scanning, so it runs in every build, on every platform, with no device
//! and no compile of the library at all.
//!
//! # Why per-file rather than one count over the whole set
//!
//! Balancing safe and fast pragmas across all files together would pass on a
//! source that opens a region in one file and closes it in the next -
//! technically balanced, actually still the exact bug this guards against,
//! since `SOURCES` order (`src/compile.rs`) is what decides which files sit
//! between the open and the close. Checking file by file is checking the unit
//! MSL actually respects no boundary for: string concatenation. A file that
//! is correct on its own is correct regardless of where `SOURCES` puts it.

use std::fs;
use std::path::Path;

/// Whether `source` restores the build's default fast math by the time it
/// ends.
///
/// Walks every `#pragma METAL fp math_mode(...)` line in file order and
/// tracks which mode is in force, the same way the Metal compiler would
/// reading the concatenated library top to bottom. A file with no pragma at
/// all trivially ends in fast mode, which is `full_source`'s prelude state
/// and exactly what a file that never touches the math mode should leave
/// standing.
///
/// A line is matched by its pragma text with leading whitespace stripped, not
/// by a full-line comparison, so a trailing comment on the same line (there
/// are none today, but nothing requires there never be one) would not hide
/// the pragma from this scan the way it would from a naive `==`.
fn ends_in_fast_math(source: &str) -> bool {
    let mut in_safe_region = false;
    for line in source.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#pragma METAL fp math_mode(safe)") {
            in_safe_region = true;
        } else if trimmed.starts_with("#pragma METAL fp math_mode(fast)") {
            in_safe_region = false;
        }
    }
    !in_safe_region
}

/// How many `safe` and `fast` pragma lines a source carries, in that order.
///
/// Reported alongside [`ends_in_fast_math`]'s verdict so a failure message
/// says *how* unbalanced a file is - two `safe`s and one `fast` reads
/// differently from one `safe` and zero `fast`s, even though both leave the
/// file in a safe region at EOF.
fn pragma_counts(source: &str) -> (usize, usize) {
    source.lines().fold((0, 0), |(safe, fast), line| {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#pragma METAL fp math_mode(safe)") {
            (safe + 1, fast)
        } else if trimmed.starts_with("#pragma METAL fp math_mode(fast)") {
            (safe, fast + 1)
        } else {
            (safe, fast)
        }
    })
}

/// Every `.metal` file under `metal/qwen35/` and `metal/qwen35moe/` ends back
/// at fast math.
///
/// Reads each directory rather than naming files, so a future kernel file
/// added under either is covered the day it lands rather than the day
/// somebody remembers to add it to a list here. The `checked > 0` guard is
/// per directory and not over the pair: a mistyped path that finds nothing
/// would otherwise be carried by the other directory's files and the scan
/// would pass having scanned half of what it claims.
#[test]
fn every_qwen35_metal_file_restores_fast_math_by_eof() {
    for module in ["qwen35", "qwen35moe"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../kernels/metal")
            .join(module);
        let entries = fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("{} reads as a directory: {err}", dir.display()));

        let mut checked = 0usize;
        for entry in entries {
            let entry = entry.expect("a directory entry resolves");
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("metal") {
                continue;
            }
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{} reads: {err}", path.display()));

            checked += 1;
            let (safe, fast) = pragma_counts(&source);
            assert!(
                ends_in_fast_math(&source),
                "{} does not restore fast math by end of file (saw {safe} `safe` \
                 pragma(s) and {fast} `fast` pragma(s)); MSL has no push/pop for \
                 `#pragma METAL fp math_mode`, so this would silently change the \
                 math mode of every q35, vision, dense and qwen35moe source \
                 `full_source` concatenates after this file (see compile_library \
                 in src/compile.rs)",
                path.display(),
            );
        }
        assert!(
            checked > 0,
            "{} produced no .metal files to check - the scan itself is broken",
            dir.display(),
        );
    }
}

/// A file that never mentions the math mode is not "balanced" by an absence
/// of evidence; it is balanced because there is nothing to unbalance.
///
/// Guards [`ends_in_fast_math`] and [`pragma_counts`] against a change that
/// makes an empty pragma count read as unsafe by construction, which would
/// turn every ordinary kernel file into a spurious failure.
#[test]
fn a_source_with_no_pragma_reads_as_balanced() {
    let source = "kernel void some_kernel() {}\n";
    assert!(ends_in_fast_math(source));
    assert_eq!(pragma_counts(source), (0, 0));
}

/// A region opened and never closed is caught, not merely a count mismatch.
///
/// This is the regression the whole file exists for: a `fast` line dropped
/// from the end of a `safe` region. [`pragma_counts`] alone would report
/// `(1, 0)`, which is already wrong, but this case pins the verdict function
/// that actually gates the real test, not just the counter that annotates it.
#[test]
fn an_unclosed_safe_region_does_not_end_in_fast_math() {
    let source = "\
#pragma METAL fp math_mode(safe)
kernel void some_kernel() {}
";
    assert!(!ends_in_fast_math(source));
    assert_eq!(pragma_counts(source), (1, 0));
}

/// A region that is opened and closed again restores fast math, even with
/// ordinary source on either side of the pair.
#[test]
fn a_closed_safe_region_ends_in_fast_math() {
    let source = "\
kernel void before() {}
#pragma METAL fp math_mode(safe)
kernel void exact() {}
#pragma METAL fp math_mode(fast)
kernel void after() {}
";
    assert!(ends_in_fast_math(source));
    assert_eq!(pragma_counts(source), (1, 1));
}

/// A pragma mentioned only inside a `//` comment does not count: this is what
/// keeps the scan from being fooled by the prose several of these files
/// already carry pointing readers at the real pragma elsewhere.
#[test]
fn a_pragma_named_only_in_a_comment_is_not_counted() {
    let source = "\
// See #pragma METAL fp math_mode(safe) in the sibling file.
kernel void some_kernel() {}
";
    assert!(ends_in_fast_math(source));
    assert_eq!(pragma_counts(source), (0, 0));
}
