#!/usr/bin/env python3
"""Find the first layer where two per-layer residual dumps part ways.

Reads two directories in the layout of `scripts/reference/LAYER_DUMP.md`, one
from the independent reference (`qwen4exp_fp8_ref.py capture --layer-dump-dir`)
and one from the engine (`parity --dump-layers`):

    <dir>/index.json                     case id -> {"frontier", "prompt_tokens", ...}
    <dir>/<case>/embed.f32.npy           [streams, hidden] f32
    <dir>/<case>/L{nn}.mixer.f32.npy     after layer nn's token-mixer combine
    <dir>/<case>/L{nn}.f32.npy           after layer nn's feed-forward combine

For every case both sides hold, and every point in pass order (embed, L00.mixer,
L00, L01.mixer, ...), it prints the relative L2 error ||cand - ref|| / ||ref||
and the cosine, over the whole row and per stream, and names the first point
whose error jumps.

The jump rule. Let e_k be the whole-row relative error at point k in pass order,
and m_k = max(e_0 .. e_{k-1}, floor). Point k is a jump when

    e_k > floor  and  e_k > ratio * m_k

with `--jump-ratio` (default 2) and `--jump-floor` (default 0.01). A quantized
build's error enters at every layer and accumulates, so it grows by a modest
factor per point and never doubles its own running maximum in one step; a
kernel defect adds its whole error at one point. The floor keeps the rule from
firing on the first few points, where the error is tiny and noisy (bf16
rounding of the embedding against exact f32 is already a few 1e-3), and the
running maximum rather than the previous point keeps one lucky small error from
making the next ordinary one look like a jump. A non-finite value anywhere is
reported as the finding regardless of the rule. The largest single-step growth
factor is printed too, so a near miss is visible.

The point names the component: a jump at `L{nn}.mixer` is in the token mixer
(recurrent or attention, and on the n-gram layer its block too) or the mixer's
hyper-connection mix and combine; a jump at `L{nn}` is in the routed experts or
their mix and combine.

    python3 scripts/reference/bisect_layers.py REF_DIR CAND_DIR [--case ID ...]
    python3 scripts/reference/bisect_layers.py --self-test

Exit status: 0 when no case jumps, 1 when one does, 2 for a refusal (mismatched
indexes, missing files, shape disagreement).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POINT = re.compile(r"^(embed|L(\d+)(\.mixer)?)\.f32\.npy$")


class Refusal(Exception):
    """The two dumps cannot be compared as asked."""


@dataclass
class Row:
    name: str
    rel: float
    cos: float
    stream_rel: list[float]
    stream_cos: list[float]
    finite: bool


def points_in(case_dir: Path) -> list[str]:
    """The point names under `case_dir`, in pass order."""
    names = []
    for p in case_dir.iterdir():
        m = POINT.match(p.name)
        if m:
            names.append(m)
    def key(m):
        if m.group(1) == "embed":
            return (-1, 0)
        return (int(m.group(2)), 0 if m.group(3) else 1)
    return [m.group(0)[: -len(".f32.npy")] for m in sorted(names, key=key)]


def load(path: Path) -> np.ndarray:
    a = np.load(path, allow_pickle=False)
    if a.dtype != np.dtype("<f4") or a.ndim != 2:
        raise Refusal(f"{path}: expected a 2-D <f4 array, found {a.dtype} {a.shape}")
    return a


def compare_point(name: str, ref: np.ndarray, cand: np.ndarray) -> Row:
    if ref.shape != cand.shape:
        raise Refusal(f"{name}: reference {ref.shape}, candidate {cand.shape}")
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    finite = bool(np.isfinite(r).all() and np.isfinite(c).all())

    def rel(a, b):
        n = np.linalg.norm(a)
        return float(np.linalg.norm(b - a) / n) if n > 0 else float("inf")

    def cos(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a.ravel(), b.ravel()) / n) if n > 0 else float("nan")

    return Row(
        name=name,
        rel=rel(r, c) if finite else float("nan"),
        cos=cos(r, c) if finite else float("nan"),
        stream_rel=[rel(r[s], c[s]) if finite else float("nan") for s in range(r.shape[0])],
        stream_cos=[cos(r[s], c[s]) if finite else float("nan") for s in range(r.shape[0])],
        finite=finite,
    )


def first_jump(rows: list[Row], ratio: float, floor: float) -> tuple[int | None, str]:
    """Index of the first jumping point and why, under the module's rule."""
    running = 0.0
    for k, row in enumerate(rows):
        if not row.finite:
            return k, "non-finite values"
        base = max(running, floor)
        if row.rel > floor and row.rel > ratio * base:
            return k, f"error {row.rel:.3e} is {row.rel / base:.1f}x the running max {base:.3e}"
        running = max(running, row.rel)
    return None, ""


def largest_step(rows: list[Row]) -> tuple[int | None, float]:
    best, at = 0.0, None
    for k in range(1, len(rows)):
        prev = rows[k - 1].rel
        if rows[k].finite and prev > 0 and np.isfinite(prev):
            f = rows[k].rel / prev
            if f > best:
                best, at = f, k
    return at, best


def read_index(d: Path) -> dict:
    p = d / "index.json"
    if not p.is_file():
        raise Refusal(f"{p} is missing")
    return json.loads(p.read_text())


def compare_case(ref_dir: Path, cand_dir: Path, case: str) -> list[Row]:
    rd, cd = ref_dir / case, cand_dir / case
    names = points_in(rd)
    cand_names = points_in(cd)
    if names != cand_names:
        missing = sorted(set(names) ^ set(cand_names))
        raise Refusal(f"case {case}: the two sides hold different points ({missing[:6]}...)")
    if not names or names[0] != "embed":
        raise Refusal(f"case {case}: no embed file")
    return [compare_point(n, load(rd / f"{n}.f32.npy"), load(cd / f"{n}.f32.npy")) for n in names]


def table(case: str, rows: list[Row], jump: int | None, why: str, layers_only: bool) -> str:
    streams = len(rows[0].stream_rel)
    head = (f"{'point':<11} {'rel_l2':>9} {'cos':>10}  "
            + " ".join(f"{'s' + str(s) + '_rel':>9}" for s in range(streams))
            + f"  {'min_s_cos':>10} {'step':>6}")
    out = [f"== {case}", head]
    for k, row in enumerate(rows):
        if layers_only and row.name.endswith(".mixer"):
            continue
        prev = rows[k - 1].rel if k else float("nan")
        step = row.rel / prev if k and prev > 0 else float("nan")
        mark = "  <== first jump" if k == jump else ""
        out.append(
            f"{row.name:<11} {row.rel:9.3e} {row.cos:10.7f}  "
            + " ".join(f"{e:9.3e}" for e in row.stream_rel)
            + f"  {min(row.stream_cos):10.7f} {step:6.2f}{mark}"
        )
    at, f = largest_step(rows)
    if jump is None:
        out.append(f"-> no jump; largest single-step growth {f:.2f}x at "
                   f"{rows[at].name if at is not None else '-'}; final error {rows[-1].rel:.3e}")
    else:
        out.append(f"-> FIRST JUMP at {rows[jump].name}: {why}")
    return "\n".join(out)


def run(ref_dir: Path, cand_dir: Path, cases: list[str], ratio: float, floor: float,
        layers_only: bool = False, emit=print) -> dict:
    ri, ci = read_index(ref_dir), read_index(cand_dir)
    both = [c for c in ri if c in ci]
    if cases:
        absent = [c for c in cases if c not in both]
        if absent:
            raise Refusal(f"cases not in both indexes: {absent}")
        both = cases
    if not both:
        raise Refusal("the two indexes share no case")
    for c in both:
        for key in ("frontier", "prompt_tokens"):
            if key in ri[c] and key in ci[c] and ri[c][key] != ci[c][key]:
                raise Refusal(f"case {c}: {key} is {ri[c][key]} in the reference and "
                              f"{ci[c][key]} in the candidate; they are not the same row")
    summary = {}
    for c in both:
        rows = compare_case(ref_dir, cand_dir, c)
        jump, why = first_jump(rows, ratio, floor)
        emit(table(c, rows, jump, why, layers_only))
        summary[c] = {
            "first_jump": rows[jump].name if jump is not None else None,
            "why": why,
            "final_rel": rows[-1].rel,
            "rows": [{"point": r.name, "rel": r.rel, "cos": r.cos,
                      "stream_rel": r.stream_rel, "stream_cos": r.stream_cos} for r in rows],
        }
    return summary


def self_test() -> None:
    rng = np.random.default_rng(7)
    streams, hidden, n_layers = 4, 64, 12
    names = ["embed"] + [n for i in range(n_layers) for n in (f"L{i:02d}.mixer", f"L{i:02d}")]

    def make(root: Path, arrays: dict[str, dict[str, np.ndarray]]) -> None:
        index = {}
        for case, pts in arrays.items():
            (root / case).mkdir(parents=True)
            for n, a in pts.items():
                np.save(root / case / f"{n}.f32.npy", a.astype("<f4"), allow_pickle=False)
            index[case] = {"frontier": 9, "prompt_tokens": 10}
        (root / "index.json").write_text(json.dumps(index))

    ref, smooth, jumpy = {}, {}, {}
    for k, n in enumerate(names):
        r = rng.standard_normal((streams, hidden)) * (1 + 0.1 * k)
        noise = rng.standard_normal((streams, hidden))
        noise /= np.linalg.norm(noise)
        # A smooth profile: 0.5% at the first point, growing 8% per point.
        e = 0.005 * 1.08 ** k
        ref[n] = r
        smooth[n] = r + noise * e * np.linalg.norm(r)
        # The same, with a defect that enters at L05.mixer and carries on.
        j = e * (6.0 if k >= names.index("L05.mixer") else 1.0)
        jumpy[n] = r + noise * j * np.linalg.norm(r)

    quiet = lambda _line: None  # noqa: E731
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        make(t / "ref", {"a": ref, "b": ref})
        make(t / "smooth", {"a": smooth, "b": ref})
        make(t / "jumpy", {"a": jumpy, "b": smooth})
        s = run(t / "ref", t / "smooth", [], 2.0, 0.01, emit=quiet)
        assert s["a"]["first_jump"] is None, s["a"]["why"]
        assert s["b"]["first_jump"] is None and s["b"]["final_rel"] == 0.0
        assert all(abs(r["cos"] - 1.0) < 1e-12 for r in s["b"]["rows"])
        first = s["a"]["rows"][0]["rel"]
        assert abs(first - 0.005) < 1e-4, first
        s = run(t / "ref", t / "jumpy", ["a"], 2.0, 0.01, emit=quiet)
        assert s["a"]["first_jump"] == "L05.mixer", s["a"]
        # Pass order, not listing order: L00 comes after L00.mixer, L10 after L09.
        assert [r["point"] for r in s["a"]["rows"]] == names
        # Non-finite is the finding wherever it is.
        bad = dict(smooth)
        bad["L02"] = smooth["L02"].copy()
        bad["L02"][1, 3] = np.nan
        make(t / "nan", {"a": bad})
        s = run(t / "ref", t / "nan", ["a"], 2.0, 0.01, emit=quiet)
        assert s["a"]["first_jump"] == "L02" and "non-finite" in s["a"]["why"]
        # Refusals: a different row, a missing point, a different shape.
        make(t / "moved", {"a": smooth})
        idx = json.loads((t / "moved" / "index.json").read_text())
        idx["a"]["frontier"] = 8
        (t / "moved" / "index.json").write_text(json.dumps(idx))
        for cand, what in (("moved", "frontier"),):
            try:
                run(t / "ref", t / cand, ["a"], 2.0, 0.01, emit=quiet)
                raise AssertionError(f"{what} mismatch was not refused")
            except Refusal:
                pass
        short = {n: a for n, a in smooth.items() if n != "L07"}
        make(t / "short", {"a": short})
        try:
            run(t / "ref", t / "short", ["a"], 2.0, 0.01, emit=quiet)
            raise AssertionError("a missing point was not refused")
        except Refusal:
            pass
        wide = {n: np.concatenate([a, a], axis=1) for n, a in smooth.items()}
        make(t / "wide", {"a": wide})
        try:
            run(t / "ref", t / "wide", ["a"], 2.0, 0.01, emit=quiet)
            raise AssertionError("a shape mismatch was not refused")
        except Refusal:
            pass
    print("self-test ok: smooth profile passes, a 6x defect at L05.mixer is named, "
          "NaN is named, three refusals hold")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ref_dir", nargs="?", type=Path)
    p.add_argument("cand_dir", nargs="?", type=Path)
    p.add_argument("--case", action="append", default=[])
    p.add_argument("--jump-ratio", type=float, default=2.0)
    p.add_argument("--jump-floor", type=float, default=0.01)
    p.add_argument("--layers-only", action="store_true",
                   help="print only the post-layer rows (the jump is still found over all points)")
    p.add_argument("--json", type=Path, help="also write the per-point numbers here")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if args.ref_dir is None or args.cand_dir is None:
        p.error("REF_DIR and CAND_DIR are required")
    try:
        summary = run(args.ref_dir, args.cand_dir, args.case, args.jump_ratio, args.jump_floor,
                      args.layers_only)
    except Refusal as e:
        print(f"refusing: {e}", file=sys.stderr)
        sys.exit(2)
    if args.json:
        if args.json.exists():
            print(f"refusing: {args.json} already exists", file=sys.stderr)
            sys.exit(2)
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    sys.exit(1 if any(s["first_jump"] for s in summary.values()) else 0)


if __name__ == "__main__":
    main()
