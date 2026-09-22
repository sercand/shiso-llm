//! Link hints for the vendor math libraries this backend calls.
//!
//! Two libraries are linked and one deliberately is not.
//!
//! * `cublas` and `cublasLt` are reached through generated bindings that expect
//!   the symbols at link time, so their search paths have to be emitted here.
//! * The driver library is *not* linked. The bindings dlopen it at first use,
//!   which is what lets a host without a GPU still produce a runnable binary
//!   that fails with a driver error instead of failing to start at all.
//! * The CUDA runtime is not linked either: nothing here calls a runtime-API
//!   symbol. Device-to-device copies go through the driver's own async copy.
//!
//! On any non-Linux target this script emits nothing at all and returns before
//! the first link line. That is what keeps a developer build on macOS from
//! asking for libraries that do not exist on the machine: the whole backend
//! compiles against its stub arm there, and a stub has nothing to link against.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The kernel lane's skip switch is read by no code here, but a build that
    // flips it is a build whose kernel set changed, and re-running costs
    // nothing next to silently keeping a stale link line.
    println!("cargo:rerun-if-env-changed=SHISO_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Early return before any link line: a target without the vendor libraries
    // must not be told to look for them.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    for dir in search_dirs() {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cublas");
}

/// Where a distribution or a hand-rolled install puts the vendor libraries.
///
/// Ordered most specific first, so an explicitly configured toolchain wins over
/// whatever the distribution happens to ship.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for key in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(root) = env::var(key) {
            let root = PathBuf::from(root);
            dirs.push(root.join("lib64"));
            dirs.push(root.join("lib"));
        }
    }
    dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
    dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
    dirs.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    dirs
}
