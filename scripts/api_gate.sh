#!/usr/bin/env bash
# The API and soundness gate: the point where the workspace rules stop being
# advice and become a build failure.
#
# Everything here was a rule somebody could have broken quietly. Undocumented
# public items, a fifth crate reaching for unsafe, an upstream name left in a
# comment, a public surface that changed without anyone seeing it in a diff -
# none of these break a test, and all of them are cheap to catch by reading the
# tree. So this script reads the tree.
#
# It reports every failure it finds rather than stopping at the first, because
# the thing a gate is worst at is telling you about one problem at a time.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
bad() { printf 'api_gate: FAIL  %s\n' "$*" >&2; fail=1; }
ok() { printf 'api_gate: ok    %s\n' "$*"; }
note() { printf 'api_gate: NOTE  %s\n' "$*"; }
section() { printf '\n== %s\n' "$*"; }

# The crates permitted to relax `unsafe_code`, and the reason each one needs it.
# A fifth name here is a design decision, not an oversight, so the set is
# asserted rather than counted.
UNSAFE_CRATES="shiso-cuda shiso-modelfile shiso-metal shiso-ple"

# ---------------------------------------------------------------------------
section "missing_docs is deny"
# ---------------------------------------------------------------------------
# The lint SSOT is the workspace manifest. It was `warn` through the build-out
# so a half-finished crate did not block a commit; shipping with an
# undocumented public item is a different matter, and clippy only enforces what
# the manifest asks for.
if grep -qE '^missing_docs = "deny"$' Cargo.toml; then
    ok "[workspace.lints.rust] missing_docs = \"deny\""
else
    bad "Cargo.toml does not set missing_docs = \"deny\" in [workspace.lints.rust]"
fi

# ---------------------------------------------------------------------------
section "unsafe audit"
# ---------------------------------------------------------------------------
# Three claims, and the third is the one worth having: the set of crates that
# may write unsafe, a stated reason on each relaxation, and a SAFETY comment on
# every unsafe block. Clippy's `undocumented_unsafe_blocks` already denies the
# third - this is the belt-and-braces check that nobody wrote their way past it
# with an expectation attribute.
relaxing=""
for lib in crates/*/src/lib.rs; do
    crate=$(basename "$(dirname "$(dirname "$lib")")")
    # Attributes wrap over several lines; flattening the file lets one pattern
    # see a whole attribute, including a cfg_attr that carries the relaxation
    # inside it.
    attrs=$(tr '\n' ' ' < "$lib" | grep -oE '#!\[[^]]*unsafe_code[^]]*\]')
    while IFS= read -r attr; do
        [ -n "$attr" ] || continue
        case "$attr" in
            *allow*unsafe_code* | *expect*unsafe_code*) ;;
            *) continue ;;
        esac
        printf '%s' "$attr" | grep -qE '(allow|expect)\([[:space:]]*unsafe_code' || continue
        relaxing="$relaxing $crate"
        if ! printf '%s' "$attr" | grep -qE 'reason[[:space:]]*='; then
            bad "$crate relaxes unsafe_code without a reason: $attr"
        fi
    done <<EOF
$attrs
EOF
done
relaxing=$(printf '%s\n' $relaxing | sort -u | tr '\n' ' ')
expected=$(printf '%s\n' $UNSAFE_CRATES | sort -u | tr '\n' ' ')
if [ "$relaxing" = "$expected" ]; then
    ok "exactly these crates relax unsafe_code: $expected"
else
    bad "the set of crates relaxing unsafe_code drifted"
    bad "  expected: $expected"
    bad "  found:    $relaxing"
fi

# A SAFETY note counts when it sits in the contiguous comment run directly above
# the block. Distance is not the test - several of these notes are eight lines
# long, because the invariant they state is a filesystem-level one no signature
# can express - so the scan walks the whole run rather than peeking at one line.
undocumented=$(awk '
FNR == 1 { delete line }
{ line[FNR] = $0 }
/unsafe[[:space:]]*(\{|impl)/ {
    t = $0
    sub(/^[[:space:]]+/, "", t)
    if (t ~ /^(\/\/|\*)/) next
    ok = 0
    for (i = FNR - 1; i >= 1; i--) {
        u = line[i]
        sub(/^[[:space:]]+/, "", u)
        if (u == "") continue
        if (u ~ /^\/\//) { if (u ~ /^\/\/[[:space:]]*SAFETY:/) ok = 1; continue }
        break
    }
    if (!ok) printf "%s:%d:%s\n", FILENAME, FNR, $0
}
' $(find crates -name '*.rs'))
if [ -z "$undocumented" ]; then
    ok "every unsafe block and unsafe impl carries a // SAFETY: note"
else
    bad "unsafe without a // SAFETY: note directly above it:"
    printf '%s\n' "$undocumented" >&2
fi

# Printed, never gated: a jump in these numbers is a review signal, and a
# threshold on them would only teach people to spread the same unsafe wider.
printf 'api_gate: unsafe blocks per crate (printed for review, never gated)\n'
for dir in crates/*/; do
    crate=$(basename "$dir")
    n=$(grep -rhoE 'unsafe[[:space:]]*(\{|impl)' "$dir" --include='*.rs' | wc -l | tr -d ' ')
    [ "$n" = "0" ] || printf '  %-20s %s\n' "$crate" "$n"
done

# ---------------------------------------------------------------------------
section "no upstream provenance in the code (rule 15)"


# The rule only works if the attribution actually landed somewhere, so the
# Provenance table must have a row naming every workspace member.
provenance=$(awk '/^## Provenance$/ { on = 1; next } /^## / { on = 0 } on' README.md)
members=$(awk '/^members = \[/ { on = 1; next } /^\]/ { on = 0 } on' Cargo.toml |
    sed -e 's/[",]//g' -e 's#.*/##' -e 's/^[[:space:]]*//' -e '/^$/d')
missing=""
for crate in $members; do
    printf '%s' "$provenance" | grep -q "crates/$crate" || missing="$missing $crate"
done
if [ -z "$missing" ]; then
    ok "README.md section Provenance names every workspace member"
else
    bad "README.md section Provenance has no row for:$missing"
fi

# ---------------------------------------------------------------------------
section "no forbidden shapes"
# ---------------------------------------------------------------------------
# Each of these is a defect the plan review actually found, written down as the
# shape it took so it cannot come back unnoticed.
forbidden() {
    local label="$1" pattern="$2"
    local found
    found=$(grep -rnE "$pattern" crates/ --include='*.rs')
    if [ -z "$found" ]; then
        ok "$label"
    else
        bad "$label"
        printf '%s\n' "$found" >&2
    fi
}
forbidden "no raw integer id aliases (rule 10: newtypes at every boundary)" \
    'type [A-Za-z0-9_]*Id[[:space:]]*=[[:space:]]*u(32|64)'
forbidden "no public id fields and no bare free() (rule 9: no handles without ownership)" \
    'pub id: u(32|64)|pub fn free\('
forbidden "no ptr::eq (identity by address is not identity here)" \
    'ptr::eq'
forbidden "no crate-level deny(warnings) (rule 6: [workspace.lints] is the SSOT)" \
    '#!\[deny\(warnings\)\]'

# `Result<Option<T>>` is two refusals in one type, and it is legitimate only
# where `None` is an answer rather than a failure. That distinction lives in
# prose, so the gate requires the prose: the comment run above the signature has
# to say what `None` means. Reviewed by hand once; kept honest by the compiler
# of record here, which is a human reading the sentence a new site had to write.
unjustified=$(awk '
FNR == 1 { delete line }
{ line[FNR] = $0 }
/Result<Option<|CoreResult<Option</ {
    t = $0
    sub(/^[[:space:]]+/, "", t)
    if (t ~ /^(\/\/|\*)/) next
    start = 0
    for (i = FNR; i >= 1 && i > FNR - 16; i--) {
        u = line[i]
        sub(/^[[:space:]]+/, "", u)
        if (u ~ /^(pub(\([a-z:]+\))?[[:space:]]+)?(default[[:space:]]+)?(const[[:space:]]+)?(async[[:space:]]+)?(unsafe[[:space:]]+)?fn[[:space:]]/) { start = i; break }
    }
    if (start == 0) { printf "%s:%d: no signature found above this site\n", FILENAME, FNR; next }
    ok = 0
    for (i = start - 1; i >= 1; i--) {
        u = line[i]
        sub(/^[[:space:]]+/, "", u)
        if (u == "" || u ~ /^#\[/) continue
        if (u ~ /^\/\//) { if (u ~ /None/) ok = 1; continue }
        break
    }
    if (!ok) printf "%s:%d:%s\n", FILENAME, start, line[start]
}
' $(find crates -path '*/src/*' -name '*.rs'))
if [ -z "$unjustified" ]; then
    ok "every Result<Option<..>> site says what None means"
else
    bad "Result<Option<..>> with no comment saying why None is not a refusal:"
    printf '%s\n' "$unjustified" >&2
fi

# Not a grep: `clippy::too_many_arguments` is deliberately absent from the
# workspace allow-list, so a public function that grows a fifth positional
# argument fails the clippy step below and has to become a named struct.
ok "argument-count rule is clippy's (too_many_arguments is not allowed anywhere)"

# ---------------------------------------------------------------------------
section "public API snapshots"
# ---------------------------------------------------------------------------
# The four crates every other crate builds against. Nothing here is published,
# so this is not about semver: it is about making a public-surface change show
# up in the diff of the change that makes it, which is the cheapest review of
# API drift there is.
API_CRATES="shiso-core shiso-modelfile shiso-kvstore shiso-ir"
for crate in $API_CRATES; do
    [ -f "api/$crate.txt" ] || bad "api/$crate.txt is missing - record it with cargo public-api"
done
if ! command -v cargo-public-api >/dev/null 2>&1; then
    note "SKIPPED the snapshot diff for $API_CRATES: cargo-public-api is not installed."
    note "      Install it with: cargo install cargo-public-api --locked"
    note "      The committed snapshots were checked for presence only, not for drift."
elif ! rustup toolchain list 2>/dev/null | grep -q '^nightly'; then
    note "SKIPPED the snapshot diff for $API_CRATES: cargo-public-api needs a nightly"
    note "      toolchain to emit rustdoc JSON. Install it with: rustup toolchain install nightly"
    note "      The committed snapshots were checked for presence only, not for drift."
else
    snapdir=$(mktemp -d)
    for crate in $API_CRATES; do
        if ! cargo public-api -p "$crate" --omit blanket-impls,auto-trait-impls \
            > "$snapdir/$crate.txt" 2> "$snapdir/$crate.err"; then
            bad "cargo public-api failed for $crate:"
            tail -20 "$snapdir/$crate.err" >&2
            continue
        fi
        if diff -u "api/$crate.txt" "$snapdir/$crate.txt" > "$snapdir/$crate.diff"; then
            ok "api/$crate.txt matches the crate's public surface"
        else
            bad "the public surface of $crate changed without the snapshot being re-recorded."
            bad "  Re-record with: cargo public-api -p $crate --omit blanket-impls,auto-trait-impls > api/$crate.txt"
            head -60 "$snapdir/$crate.diff" >&2
        fi
    done
    rm -rf "$snapdir"
fi

# ---------------------------------------------------------------------------
section "cross-crate type contracts"
# ---------------------------------------------------------------------------
# One file names every type that crosses a task boundary, and fails to compile
# if any of them drifts. Running it is almost beside the point - the assertions
# are type annotations, so the build is the test - but a compiled-and-not-run
# contract is one nobody notices has stopped being built.
export SHISO_SKIP_BUILD="${SHISO_SKIP_BUILD:-1}"
if cargo test -p shiso-bench --test contracts --locked; then
    ok "crates/shiso-bench/tests/contracts.rs compiles and passes"
else
    bad "the cross-crate type contracts no longer hold"
fi

# ---------------------------------------------------------------------------
section "format, lint, dependencies, file size"
# ---------------------------------------------------------------------------
# scripts/check.sh runs these too. They are repeated here so this script is the
# whole story: a gate that only works when it is called from another gate is one
# more thing to get wrong.
FEATURES=()
if [ "$(uname -s)" = "Darwin" ]; then
    FEATURES=(--features shiso-server/metal)
fi

run() {
    if "$@"; then
        ok "$*"
    else
        bad "$*"
    fi
}
run cargo fmt --all --check
run cargo clippy --workspace --all-targets --locked "${FEATURES[@]}" -- -D warnings
if command -v cargo-deny >/dev/null 2>&1; then
    run cargo deny check
else
    bad "cargo-deny is not installed; install it with: cargo install cargo-deny --locked"
fi

# ---------------------------------------------------------------------------
echo
if [ "$fail" -ne 0 ]; then
    echo "api_gate: FAILED" >&2
    echo "  None of these is fixed by loosening the check that found it." >&2
    exit 1
fi
echo "api_gate: OK"
