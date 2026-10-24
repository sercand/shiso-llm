#!/usr/bin/env python3
"""Which workspace crates a change can affect - the scope of a scoped gate.

    scope.py [--base REF] [--no-dependents] [CRATE ...]

Prints `key=value` lines for scripts/check.sh to read:

    all=1|0            the change reaches every crate (a root manifest, the
                       toolchain pin, the gate itself)
    deny=1|0           the dependency graph or its policy changed
    changed=a b        crates whose own files changed (or the ones named)
    crates=a b c       changed plus everything that depends on them

With no CRATE arguments the change is read from git: everything that differs
from the merge-base with `--base` (default master), committed or not, plus
untracked files. Named crates skip git entirely.

A changed crate's dependents are found in two steps. First every member that
reaches it through normal or build dependencies, transitively: their libraries
are rebuilt against the change. Then every member with a dev dependency on any
of those: its tests are. A dev edge does not propagate further, because the
crate behind it has a changed test binary and an unchanged library - following
it anyway takes `shiso-engine` to thirteen of nineteen crates instead of eight,
by way of a test-only cycle through `shiso-models`.
"""

import argparse
import json
import subprocess
import sys

# Files outside any crate that a crate reads at build or test time. A change
# under one of these prefixes is a change to every crate listed. `test-vectors/`
# is not here because it is gitignored and so never appears in a diff.
EXTRA_EDGES = {
    "kernels/": ["shiso-kernels", "shiso-engine"],
    "kernels/metal/": ["shiso-metal"],
    "crates/shiso-registry/tests/fixtures/": ["shiso-hub", "shiso-server"],
}

# Root files whose change can alter any crate's build. The gate scripts are here
# because a gate edited in the same change it grades would otherwise grade
# itself with a scope it just narrowed.
EVERYTHING = {
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "scripts/check.sh",
    "scripts/api_gate.sh",
    "scripts/scope.py",
}
EVERYTHING_PREFIXES = (".cargo/",)

# Files that change what `cargo deny` reads.
DENY_INPUTS = {"Cargo.toml", "Cargo.lock", "deny.toml"}


def git(*args):
    out = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return out.stdout


def changed_files(base):
    merge_base = git("merge-base", "HEAD", base).strip()
    files = set(git("diff", "--name-only", merge_base).split())
    files |= set(git("ls-files", "--others", "--exclude-standard").split())
    return files


def workspace():
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads(out.stdout)
    names = {p["name"] for p in meta["packages"]}
    lib_users = {n: set() for n in names}
    test_users = {n: set() for n in names}
    dirs = {}
    for p in meta["packages"]:
        crate_dir = p["manifest_path"].rsplit("/", 2)[-2]
        dirs[f"crates/{crate_dir}/"] = p["name"]
        for d in p["dependencies"]:
            if d["name"] in names:
                users = test_users if d["kind"] == "dev" else lib_users
                users[d["name"]].add(p["name"])
    return names, lib_users, test_users, dirs


def affected(seeds, lib_users, test_users):
    rebuilt = set(seeds)
    todo = list(seeds)
    while todo:
        for up in lib_users[todo.pop()]:
            if up not in rebuilt:
                rebuilt.add(up)
                todo.append(up)
    retested = set(rebuilt)
    for crate in rebuilt:
        retested |= test_users[crate]
    return retested


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="master")
    ap.add_argument("--no-dependents", action="store_true")
    ap.add_argument("crates", nargs="*")
    args = ap.parse_args()

    names, lib_users, test_users, dirs = workspace()
    everything = False
    deny = False
    changed = set()

    if args.crates:
        unknown = [c for c in args.crates if c not in names]
        if unknown:
            sys.exit(f"scope: not a workspace member: {' '.join(unknown)}")
        changed = set(args.crates)
    else:
        for f in changed_files(args.base):
            if f in EVERYTHING or f.startswith(EVERYTHING_PREFIXES):
                everything = True
            if f in DENY_INPUTS:
                deny = True
            for prefix, crates in EXTRA_EDGES.items():
                if f.startswith(prefix):
                    changed.update(crates)
            for prefix, crate in dirs.items():
                if f.startswith(prefix):
                    changed.add(crate)

    if everything:
        changed = set(names)
        deny = True
    crates = changed if args.no_dependents else affected(changed, lib_users, test_users)

    print(f"all={int(everything)}")
    print(f"deny={int(deny)}")
    print(f"changed={' '.join(sorted(changed))}")
    print(f"crates={' '.join(sorted(crates))}")


if __name__ == "__main__":
    main()
