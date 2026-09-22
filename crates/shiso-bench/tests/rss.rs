//! The resident-set ceiling one measured run has to stay under.
//!
//! Measured from outside the process on purpose. Reading the figure from inside
//! would add a dependency for a number the operating system already keeps, and
//! the number that matters is the whole run's - the load, the pool, and both
//! phases - not the part a self-report could see.
//!
//! This lives beside the runner rather than beside the engine's other model tests
//! because `CARGO_BIN_EXE_*` names only the binaries of the crate a test belongs
//! to, and a test that guessed at a path under the build directory would break on
//! the first profile change.

use std::path::PathBuf;

/// The ceiling: six gibibytes for the whole run.
const CEILING: u64 = 6_442_450_944;

/// Prints the skip notice; the caller returns immediately afterwards.
#[expect(clippy::print_stdout, reason = "the skip report is a test's output")]
fn skip(why: &str) {
    println!("SKIP: {why}");
}

/// Reports the measured figure, which is recorded rather than only compared.
#[expect(clippy::print_stdout, reason = "the measurement is a test's output")]
fn report(what: &str) {
    println!("{what}");
}

/// The checkpoint the runner measures.
fn model_path() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("SHISO_TEST_MODEL").ok()?);
    path.is_file().then_some(path)
}

#[test]
#[ignore = "needs the local checkpoint and a Metal device"]
fn one_measured_run_stays_under_the_resident_set_ceiling() {
    let Some(_model) = model_path() else {
        skip("no local checkpoint");
        return;
    };
    if !shiso_metal::metal_available() {
        skip("no Metal device");
        return;
    }
    let out = std::process::Command::new("/usr/bin/time")
        .arg("-l")
        .arg(env!("CARGO_BIN_EXE_qwen35"))
        .output()
        .expect("the runner starts");
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the runner failed:\n{text}");
    let peak = text
        .lines()
        .find_map(|line| {
            let rest = line.trim().strip_suffix("maximum resident set size")?;
            rest.trim().parse::<u64>().ok()
        })
        .expect("the timer reported a maximum resident set size");
    report(&format!("peak resident set: {peak} bytes"));
    assert!(
        peak < CEILING,
        "the run held {peak} bytes, past the {CEILING}-byte ceiling"
    );
}
