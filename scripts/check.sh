#!/usr/bin/env bash
# The local gate. There is no CI - this script is it.
#
#   scripts/check.sh                   crates changed since master, plus their dependents
#   scripts/check.sh shiso-router      these crates, plus their dependents
#   scripts/check.sh --all             the whole workspace
#   scripts/check.sh --no-dependents   only the changed or named crates
#   scripts/check.sh --only clippy,test
#   scripts/check.sh --skip doc,deny
#
# Steps, in order: fmt tree clippy test doc deny libcuda perf
#
# The default scope is sound for everything the gate grades: a change is
# confined to the crates whose files it touched and the crates built against
# them, and a change to anything they all share (a root manifest, the lockfile,
# the toolchain pin, the gate itself) widens it to the whole workspace.
# scripts/scope.py says how. `--no-dependents` is not a gate - it skips the
# crates most likely to notice an API change - so use it while iterating and
# drop it before you call a change done.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '4,11p' "$0" | sed 's/^# \{0,1\}//'; }

ALL_STEPS="fmt tree clippy test doc deny libcuda perf"
scope_args=()
only=""
skip=""
all=0
while [ $# -gt 0 ]; do
    case "$1" in
        --all) all=1 ;;
        --no-dependents) scope_args+=(--no-dependents) ;;
        --base) scope_args+=(--base "$2"); shift ;;
        --only) only="$2"; shift ;;
        --skip) skip="$2"; shift ;;
        -h | --help) usage; exit 0 ;;
        -*) echo "check: unknown option $1" >&2; usage >&2; exit 2 ;;
        *) scope_args+=("$1") ;;
    esac
    shift
done
for s in ${only//,/ } ${skip//,/ }; do
    case " $ALL_STEPS " in
        *" $s "*) ;;
        *) echo "check: unknown step '$s' (steps: $ALL_STEPS)" >&2; exit 2 ;;
    esac
done
want() {
    if [ -n "$only" ]; then
        case ",$only," in *",$1,"*) ;; *) return 1 ;; esac
    fi
    case ",$skip," in *",$1,"*) return 1 ;; esac
    return 0
}

# Kernel build crates stub themselves out unless a real compile is requested.
# This wins over SHISO_CUDA_IMPL, so a checkout without a CUDA toolkit still
# builds and tests.
export SHISO_SKIP_BUILD="${SHISO_SKIP_BUILD:-1}"

if [ "$all" -eq 1 ]; then
    scope_all=1
    scope_deny=1
    crates=""
else
    scope=$(python3 -I scripts/scope.py "${scope_args[@]+"${scope_args[@]}"}")
    scope_all=$(printf '%s\n' "$scope" | sed -n 's/^all=//p')
    scope_deny=$(printf '%s\n' "$scope" | sed -n 's/^deny=//p')
    crates=$(printf '%s\n' "$scope" | sed -n 's/^crates=//p')
    changed=$(printf '%s\n' "$scope" | sed -n 's/^changed=//p')
fi

# Backend selection is explicit because Cargo features are target-independent:
# macOS builds pass --features metal. The feature belongs to shiso-server, so it
# can only be named when shiso-server is in the build.
macos=0
[ "$(uname -s)" = "Darwin" ] && macos=1
if [ "$scope_all" = "1" ]; then
    echo "check: scope: whole workspace"
    PKGS=(--workspace)
    has_server=1
elif [ -n "$crates" ]; then
    echo "check: scope: changed: $changed"
    echo "check:        checking: $crates"
    PKGS=()
    has_server=0
    for c in $crates; do
        PKGS+=(-p "$c")
        [ "$c" = "shiso-server" ] && has_server=1
    done
else
    echo "check: scope: no crate changed; running the tree-wide checks only"
    PKGS=()
    has_server=0
fi
FEATURES=()
if [ "$macos" -eq 1 ] && [ "$has_server" -eq 1 ]; then
    FEATURES=(--features shiso-server/metal)
fi

# Each step reports how long it took, so a slow gate says which step to look at
# instead of leaving a half-hour guess.
step_name=""
step_start=$SECONDS
timings=""
end_step() {
    if [ -n "$step_name" ]; then
        timings="$timings$(printf '  %-10s %4ds' "$step_name" $((SECONDS - step_start)))
"
    fi
}
step() {
    end_step
    step_name="$1"
    step_start=$SECONDS
    printf '\n== check: %s\n' "$1"
}

if want fmt; then
    step fmt
    cargo fmt --all --check
fi

# The rules stop being advice here: the unsafe-exception set, the documented
# SAFETY note on every unsafe block, the forbidden shapes, and the PROVENANCE.md
# table. These read the tree rather than build it, so they run whatever the
# scope.
if want tree; then
    step "tree rules (api_gate.sh --tree-only)"
    bash scripts/api_gate.sh --tree-only
fi

tested=0
has_bench=0
case " $crates " in *" shiso-bench "*) has_bench=1 ;; esac
[ "$scope_all" = "1" ] && has_bench=1

if [ ${#PKGS[@]} -gt 0 ]; then
    # Clippy type-checks every target it lints, so a separate `cargo check`
    # would only repeat it.
    if want clippy; then
        step clippy
        cargo clippy "${PKGS[@]}" --all-targets --locked "${FEATURES[@]+"${FEATURES[@]}"}" -- -D warnings
    fi
    # This includes crates/shiso-bench/tests/contracts.rs whenever any crate
    # those contracts name is in scope, since shiso-bench depends on all of them.
    if want test; then
        step test
        cargo test "${PKGS[@]}" --locked "${FEATURES[@]+"${FEATURES[@]}"}"
        tested=1
    fi
    # Doc links are checked because nothing else checks them: clippy does not
    # read them, and six had already rotted into naming items that moved or were
    # deleted - a link that resolves to nothing reads as a citation and is worse
    # than no link at all. `broken_intra_doc_links` is denied in the workspace
    # lints.
    if want doc; then
        step doc
        cargo doc "${PKGS[@]}" --no-deps --locked "${FEATURES[@]+"${FEATURES[@]}"}"
    fi
fi

# Only the lockfile, the manifests and deny.toml feed this, and it fetches the
# advisory database every time, so a change that touched none of them skips it.
if want deny && [ "$scope_deny" = "1" ]; then
    step deny
    if command -v cargo-deny >/dev/null 2>&1; then
        cargo deny check
    else
        echo "check: cargo-deny is not installed; install it with: cargo install cargo-deny --locked" >&2
        exit 1
    fi
fi

# On macOS nothing may link libcuda. The binary is fresh only when the test step
# just built shiso-server, so that is when it is read.
if want libcuda && [ "$macos" -eq 1 ] && [ "$has_server" -eq 1 ] && [ "$tested" -eq 1 ]; then
    step "libcuda link check"
    if otool -L target/debug/shiso | grep -q libcuda; then
        echo "FAIL: target/debug/shiso links libcuda" >&2
        exit 1
    fi
fi

# The perf gate is arithmetic over committed records - it needs no model and no
# device, and it never measures. Exit 2 means a record no longer covers the tree
# it is being read against, which is a re-record on the pinned machine, not a
# regression.
#
# The test step builds this binary with the same feature set as everything
# else, so it is run from there. `cargo run -p shiso-bench` would resolve
# features for that one package, which is a different build of half the
# workspace - two minutes of compiling for a program that runs in one second.
if want perf && [ "$has_bench" -eq 1 ]; then
    step perf
    if [ "$tested" -eq 1 ]; then
        target/debug/perf_gate check
    else
        cargo run --quiet --locked -p shiso-bench --bin perf_gate -- check
    fi
fi

end_step
printf '\ncheck: step timings\n%s' "$timings"
printf '\ncheck: all gates green\n'
