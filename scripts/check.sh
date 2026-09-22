#!/usr/bin/env bash
# The local gate. There is no CI - this script is it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Kernel build crates stub themselves out unless a real compile is requested.
# This wins over SHISO_CUDA_IMPL, so a checkout without a CUDA toolkit still
# builds and tests.
export SHISO_SKIP_BUILD="${SHISO_SKIP_BUILD:-1}"

# Backend selection is explicit because Cargo features are target-independent:
# macOS builds pass --features metal.
FEATURES=()
if [ "$(uname -s)" = "Darwin" ]; then
    FEATURES=(--features shiso-server/metal)
fi

cargo check --workspace --locked "${FEATURES[@]}"
cargo test --workspace --locked "${FEATURES[@]}"
cargo clippy --workspace --all-targets --locked "${FEATURES[@]}" -- -D warnings
cargo fmt --all --check

# Doc links are checked because nothing else checks them: clippy does not read
# them, and six had already rotted into naming items that moved or were deleted
# - a link that resolves to nothing reads as a citation and is worse than no
# link at all. `broken_intra_doc_links` is denied in the workspace lints.
cargo doc --workspace --no-deps --locked "${FEATURES[@]}"

if command -v cargo-deny >/dev/null 2>&1; then
    cargo deny check
else
    echo "check: cargo-deny not installed, skipping (cargo install cargo-deny --locked)"
fi

# On macOS nothing may link libcuda.
if [ "$(uname -s)" = "Darwin" ] && [ -x target/debug/shiso ]; then
    if otool -L target/debug/shiso | grep -q libcuda; then
        echo "FAIL: target/debug/shiso links libcuda" >&2
        exit 1
    fi
fi

# The rules stop being advice here: the unsafe-exception set, the documented
# SAFETY note on every unsafe block, the cross-crate type contracts, the
# forbidden shapes, and the PROVENANCE.md table.
bash scripts/api_gate.sh

# The perf gate is arithmetic over committed records - it needs no model and no
# device, and it never measures. Exit 2 means a record no longer covers the tree
# it is being read against, which is a re-record on the pinned machine, not a
# regression.
cargo run --quiet -p shiso-bench --bin perf_gate -- check

echo "check: all gates green"
