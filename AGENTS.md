# AGENTS.md

Guidance for coding agents working in this repository. `README.md` covers what
shiso is and how to build and run it; this file covers the rules a change has to
follow. Most of them are enforced by `scripts/check.sh` and
`scripts/api_gate.sh`, so breaking one fails the gate. None of them is fixed by
loosening the check that found it.

## Commands

Run everything from the repository root so `rust-toolchain.toml` (1.98.1)
applies. Pass `--locked` because `Cargo.lock` is committed.

```sh
bash scripts/check.sh                  # the gate, scoped to crates changed since master + dependents
bash scripts/check.sh shiso-router     # these crates + their dependents
bash scripts/check.sh --all            # the whole workspace (root manifest, lockfile, gate edits do this anyway)
bash scripts/check.sh --no-dependents  # inner loop only: skips dependents, so not a gate
bash scripts/check.sh --only clippy,test   # or --skip doc; steps: fmt tree clippy test doc deny libcuda perf
bash scripts/api_gate.sh               # standalone: tree rules plus contracts, fmt, clippy, deny

# individual steps (macOS needs the metal feature; Linux uses cuda or none)
SHISO_SKIP_BUILD=1 cargo test --workspace --locked --features shiso-server/metal
cargo clippy --workspace --all-targets --locked --features shiso-server/metal -- -D warnings
cargo fmt --all --check
cargo doc --workspace --no-deps --locked --features shiso-server/metal
cargo test -p <crate> --test <name>          # a single test target
```

- While iterating, run the crate and the test target you are changing
  (`cargo test -p <crate> --test <name>`, or `bash scripts/check.sh <crate>
  --no-dependents --only clippy,test`), not the whole gate. The full gate
  builds and links the workspace's ~190 test executables, which is tens of
  minutes on a laptop, so run it once, when a change is done. It prints each
  step's duration at the end.
- The device-opening Metal kernel cases are one test target:
  `cargo test -p shiso-metal --features metal --test kernels <file>:: -- --ignored`.
  Add a new kernel file to `crates/shiso-metal/tests/kernels.rs`, or it is not
  built.
- `check.sh` already runs everything in `api_gate.sh` (it calls it with
  `--tree-only` and runs the cargo steps itself), so there is no need to run
  both. `scripts/scope.py` decides the scope; there is no CI, so there is
  no separate full run before merging unless you ask for `--all`.
- `SHISO_SKIP_BUILD=1` stubs out the CUDA kernel compile. A real compile needs
  `SHISO_CUDA_COMPILE=1` and nvcc; `SHISO_CUDA_ARCH` overrides the target
  architectures.
- The `metal` and `cuda` features of `shiso-server` are mutually exclusive.
  Never use `--all-features`.
- Build the server with `--release` whenever it has to run on a device.
- On macOS, `target/debug/shiso` must not link libcuda. The gate checks this.
- The perf gate (`cargo run -p shiso-bench --bin perf_gate -- check`) only does
  arithmetic over committed records. Exit code 2 means a record needs to be
  re-recorded on the pinned machine. It is not a regression.

## Tests and fixtures

- Tests that need a GPU are `#[ignore]` and gated to their platform. Tests that
  need a real checkpoint read its path from `SHISO_TEST_MODEL*` variables
  (`SHISO_TEST_MODEL`, `SHISO_TEST_MODEL_GEMMA4`, `SHISO_TEST_MODEL_QWEN35MOE`,
  and others) and say on stderr why they skipped. A skipped test and a passing
  one look the same in the output, so never report a device test as passing
  unless it actually ran.
- `test-vectors/` is gitignored. Read `test-vectors/README.md` before touching
  a fixture. Goldens there are **regression locks**: they show the build answers
  what it answered yesterday, not that it is correct. Only fixtures under
  `references/` come from an independent implementation.
- In `shiso-bench`, no parity path records a duration and no perf path records
  a logit.
- Kernel cases live in `shiso-kerneltest` and are backend-neutral. Each backend
  keeps only thin `#[test]` wrappers. `shiso-kerneltest` must never depend on
  `shiso-metal` or `shiso-cuda`.

## Layering

- `shiso-core`, `shiso-engine` and `shiso-kvstore` name **no model family**:
  no family name in a type, module or variant. A family's config, tensor roles
  and stop tokens live in `crates/shiso-models/src/<family>/`. Adding a family
  should touch no file below `shiso-models`.
- The engine talks to devices only through `shiso_core::Backend`
  (`Arc<dyn Backend>`). Nothing above a backend crate names it.
- `shiso-ir` is the narrow waist. Each wire format (`shiso-server/src/openai`,
  `anthropic`) owns its own serde structs and converts them to and from the IR.
  The IR types carry no serde derives.
- Only the scheduler thread calls methods on the model.
- `shiso-modelfile` describes files and never converts them: weights are
  borrowed from the mapping, not copied or dequantized, wherever the container
  allows it.

## Code rules (enforced)

The lint settings in `[workspace.lints]` in `Cargo.toml` are the single source
of truth. Every crate sets `[lints] workspace = true`. Do not add crate-level
`#![deny(warnings)]` (rule 6).

- **Unsafe:** denied everywhere except `shiso-modelfile` (mmap),
  `shiso-metal` (objc2), `shiso-cuda` (driver FFI), `shiso-ple` (io_uring) and
  `shiso-media-sys` (OS video decoders). Each of those relaxes the lint with a
  stated `reason`. Every `unsafe` block and `unsafe impl` needs a `// SAFETY:`
  comment directly above it. Adding a sixth crate to that set is a design
  decision; raise it with the user, don't just do it.
- **Docs:** `missing_docs` is deny, so every public item needs a doc comment.
  Broken intra-doc links are deny.
- **Lint suppressions:** use `#[expect(..., reason = "...")]`, never `#[allow]`.
- **No** `todo!`, `unimplemented!`, `dbg!`, `println!`/`eprintln!` (use
  `tracing`), `mem::forget`, `process::exit`. Avoid `unwrap()`; `expect("why")`
  is fine.
- **Ids are newtypes** at every boundary (rule 10). No `type FooId = u32`, no
  `pub id: u32` fields, no bare `pub fn free(` (rule 9: every handle has an
  owner). Don't compare identity by address with `ptr::eq`.
- **`Result<Option<T>>`** is allowed only when the comment above the signature
  says what `None` means.
- **Argument count:** `clippy::too_many_arguments` is not allowed anywhere. A
  function that needs many arguments takes a named `...Desc` struct instead.
- **Upstream names** (other projects) appear only in `PROVENANCE.md` and the
  README's Acknowledgements, never in code or comments (rule 15). A new
  workspace member needs a row in `PROVENANCE.md`, or the gate fails. Copied or
  ported code gets a row there saying where it came from and under what
  license.
- **Cross-crate types** are pinned in `crates/shiso-bench/tests/contracts.rs`.
- **Dependencies:** declare them in `[workspace.dependencies]` and reference
  them with `{ workspace = true }`. `cargo deny check` bans git dependencies
  and unknown registries.
  Linux-only dependencies (cudarc, io-uring) go in target tables, not optional
  features.

## Style

- Comments explain *why*: the invariant, the failure a choice prevents, what
  a number was measured on. Match the surrounding density and tone; the existing
  crate and module docs (`//!`) are the model to follow.
- Errors are typed (`thiserror`) in libraries. `anyhow` is only used at the
  binary boundary in `shiso-server`.
- Refuse bad input loudly rather than substituting a default. A zero budget, or
  a speculation depth without a draft head, is an error, not a fallback.
- Every server knob is a clap flag with a `SHISO_*` env var in
  `crates/shiso-runtime/src/options.rs` (`ServeOptions`; the clap attributes sit
  behind the runtime's `clap` feature). Don't add scattered `std::env::var`
  reads.
- Don't edit vendored kernels in `kernels/cuda/mmq` and `kernels/cuda/exl3`
  without updating the file's row in that directory's `VENDOR.md`.
