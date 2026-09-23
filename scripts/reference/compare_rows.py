#!/usr/bin/env python3
"""Grade a candidate's full logits rows against a reference's.

Both directories hold the shared row layout:

    <dir>/<case-id>.f16.npy   numpy .npy v1, dtype '<f2', shape [vocab],
                              raw pre-softmax logits at the case's frontier
    <dir>/index.json          case id -> {"prompt_file", "frontier" (0-based
                              position of the row), "prompt_tokens"}

The reference side is written by the FP8 reference driver, the candidate side
by `shiso-bench parity --dump-rows`. Per case this reports:

- top-1 match, and top-5 / top-20 overlap (|ref top-k & cand top-k|);
- exact KL(ref || cand) over the full softmax, in float64 with log-sum-exp;
- max |delta logit| over the reference's top-20, both raw and after each row's
  own log-sum-exp is subtracted (the second is shift-invariant);
- the candidate's offset from the reference over those twenty, so a row that
  is the reference plus a constant reads as "shifted", not as "wrong".

It never compares leading elements or sums: a sum over 248k near-zero-mean
logits cancels, and the leading elements of a vocabulary are rare tokens whose
logits say nothing about the distribution.

This is a report, not a gate: it exits 0 whatever the KL is. It exits 1 only
when it refuses a comparison (cases whose index entries disagree, rows of
different lengths, non-finite values), because a comparison of two different
things would be the most misleading output it could print.

Run with numpy only, e.g. ~/work/exllamav3/.venv/bin/python:

    compare_rows.py REF_DIR CAND_DIR [--json OUT.json]
    compare_rows.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, asdict

import numpy as np

TOP_KS = (5, 20)
# The window the logit deltas and the shift are measured over.
DELTA_K = 20
# An offset this large with a residual this small is called a constant shift.
# The residual bound is a few f16 ulps at logit magnitude ~16 (ulp 2^-6), so a
# pure shift survives the half-precision round trip on both sides.
SHIFT_MIN = 1e-3
SHIFT_RESIDUAL = 0.05


class Refusal(Exception):
    """A comparison that would not compare like with like."""


@dataclass
class CaseReport:
    case: str
    prompt: str
    frontier: int
    prompt_tokens: int
    ref_top1: int
    cand_top1: int
    top1_match: bool
    top5_overlap: int
    top20_overlap: int
    kl: float
    max_abs_logit: float
    max_abs_logprob: float
    offset: float
    shifted: bool


def load_index(directory: str) -> dict:
    path = os.path.join(directory, "index.json")
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict) or not index:
        raise Refusal(f"{path}: expected a non-empty object of cases")
    for case, entry in index.items():
        for key in ("prompt_file", "frontier", "prompt_tokens"):
            if key not in entry:
                raise Refusal(f"{path}: case {case} has no {key}")
    return index


def load_row(directory: str, case: str) -> np.ndarray:
    path = os.path.join(directory, f"{case}.f16.npy")
    row = np.load(path, allow_pickle=False)
    if row.dtype != np.dtype("<f2"):
        raise Refusal(f"{path}: dtype {row.dtype}, expected <f2")
    if row.ndim != 1:
        raise Refusal(f"{path}: shape {row.shape}, expected one dimension")
    if not np.isfinite(row).all():
        bad = int((~np.isfinite(row)).sum())
        raise Refusal(f"{path}: {bad} non-finite logits")
    return row


def log_softmax(row: np.ndarray) -> np.ndarray:
    """log p over the whole row, in float64, via log-sum-exp."""
    x = row.astype(np.float64)
    m = x.max()
    return x - (m + math.log(np.exp(x - m).sum()))


def kl_divergence(ref: np.ndarray, cand: np.ndarray) -> float:
    """Exact KL(ref || cand) over the full softmax, in nats."""
    lr = log_softmax(ref)
    lc = log_softmax(cand)
    p = np.exp(lr)
    kl = float((p * (lr - lc)).sum())
    # Rounding can leave an identical pair a hair below zero.
    return max(kl, 0.0)


def top_k(row: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values, highest first; ties by lower id."""
    x = row.astype(np.float64)
    part = np.argpartition(-x, k - 1)[:k]
    return part[np.lexsort((part, -x[part]))]


def compare_case(case: str, ref_entry: dict, cand_entry: dict,
                 ref: np.ndarray, cand: np.ndarray) -> CaseReport:
    for key in ("frontier", "prompt_tokens"):
        if ref_entry[key] != cand_entry[key]:
            raise Refusal(
                f"case {case}: reference {key} {ref_entry[key]}, candidate "
                f"{cand_entry[key]}; the rows are not at the same position")
    if os.path.basename(ref_entry["prompt_file"]) != os.path.basename(cand_entry["prompt_file"]):
        raise Refusal(
            f"case {case}: reference prompt {ref_entry['prompt_file']}, candidate "
            f"{cand_entry['prompt_file']}")
    if ref.shape != cand.shape:
        raise Refusal(f"case {case}: reference row {ref.shape}, candidate {cand.shape}")

    ref_tops = {k: top_k(ref, k) for k in TOP_KS}
    cand_tops = {k: top_k(cand, k) for k in TOP_KS}
    overlaps = {k: len(set(ref_tops[k].tolist()) & set(cand_tops[k].tolist())) for k in TOP_KS}

    window = top_k(ref, DELTA_K)
    r64 = ref.astype(np.float64)
    c64 = cand.astype(np.float64)
    delta = c64[window] - r64[window]
    offset = float(np.median(delta))
    residual = float(np.abs(delta - offset).max())
    lr = log_softmax(ref)
    lc = log_softmax(cand)
    return CaseReport(
        case=case,
        prompt=os.path.basename(ref_entry["prompt_file"]),
        frontier=int(ref_entry["frontier"]),
        prompt_tokens=int(ref_entry["prompt_tokens"]),
        ref_top1=int(ref_tops[5][0]),
        cand_top1=int(cand_tops[5][0]),
        top1_match=bool(ref_tops[5][0] == cand_tops[5][0]),
        top5_overlap=overlaps[5],
        top20_overlap=overlaps[20],
        kl=kl_divergence(ref, cand),
        max_abs_logit=float(np.abs(delta).max()),
        max_abs_logprob=float(np.abs(lc[window] - lr[window]).max()),
        offset=offset,
        shifted=abs(offset) > SHIFT_MIN and residual < SHIFT_RESIDUAL,
    )


def compare_dirs(ref_dir: str, cand_dir: str) -> tuple[list[CaseReport], list[str]]:
    ref_index = load_index(ref_dir)
    cand_index = load_index(cand_dir)
    notes = []
    only_ref = sorted(set(ref_index) - set(cand_index))
    only_cand = sorted(set(cand_index) - set(ref_index))
    if only_ref:
        notes.append(f"only in reference ({len(only_ref)}): {', '.join(only_ref)}")
    if only_cand:
        notes.append(f"only in candidate ({len(only_cand)}): {', '.join(only_cand)}")
    common = sorted(set(ref_index) & set(cand_index),
                    key=lambda c: (os.path.basename(ref_index[c]["prompt_file"]),
                                   ref_index[c]["frontier"], c))
    if not common:
        raise Refusal("the two directories share no case id")
    reports = []
    for case in common:
        reports.append(compare_case(case, ref_index[case], cand_index[case],
                                    load_row(ref_dir, case), load_row(cand_dir, case)))
    return reports, notes


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def print_report(reports: list[CaseReport], notes: list[str], out=sys.stdout) -> None:
    w = max(len("case"), *(len(r.case) for r in reports))
    head = (f"{'case':<{w}}  {'frontier':>8}  {'of':>6}  {'ref':>7}  {'cand':>7}  top1  "
            f"top5  top20  {'KL(ref||cand)':>13}  {'max|dlogit|':>11}  {'max|dlogp|':>10}  "
            f"{'offset':>8}")
    print(head, file=out)
    print("-" * len(head), file=out)
    for r in reports:
        print(f"{r.case:<{w}}  {r.frontier:>8}  {r.prompt_tokens:>6}  {r.ref_top1:>7}  "
              f"{r.cand_top1:>7}  {'yes' if r.top1_match else 'NO':>4}  {r.top5_overlap:>2}/5  "
              f"{r.top20_overlap:>2}/20  {r.kl:>13.6g}  {r.max_abs_logit:>11.4f}  "
              f"{r.max_abs_logprob:>10.4f}  {r.offset:>+8.4f}{'  shifted' if r.shifted else ''}",
              file=out)
    kls = [r.kl for r in reports]
    top1 = sum(r.top1_match for r in reports)
    print(file=out)
    print(f"cases {len(reports)}  top-1 {top1}/{len(reports)} ({100.0 * top1 / len(reports):.1f}%)  "
          f"KL mean {np.mean(kls):.6g}  p50 {percentile(kls, 50):.6g}  "
          f"p99 {percentile(kls, 99):.6g}  max {max(kls):.6g}", file=out)
    shifted = [r.case for r in reports if r.shifted]
    if shifted:
        print(f"shifted by a constant (KL and max|dlogp| are unaffected): {', '.join(shifted)}",
              file=out)
    for note in notes:
        print(note, file=out)

    print(file=out)
    print("KL by frontier, per prompt", file=out)
    by_prompt: dict[str, list[CaseReport]] = {}
    for r in reports:
        by_prompt.setdefault(r.prompt, []).append(r)
    for prompt, rows in sorted(by_prompt.items()):
        print(f"  {prompt} ({rows[0].prompt_tokens} tokens)", file=out)
        for r in sorted(rows, key=lambda r: r.frontier):
            print(f"    {r.frontier:>8}  KL {r.kl:<12.6g}  top1 {'yes' if r.top1_match else 'NO'}",
                  file=out)


# --- self-test ---------------------------------------------------------------

def _write_dir(directory: str, rows: dict[str, np.ndarray], frontier: int = 14) -> None:
    os.makedirs(directory)
    index = {}
    for i, (case, row) in enumerate(rows.items()):
        np.save(os.path.join(directory, f"{case}.f16.npy"), row.astype("<f2"))
        index[case] = {"prompt_file": "../prompts/synthetic.txt",
                       "frontier": frontier + i, "prompt_tokens": 64}
    with open(os.path.join(directory, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)


def self_test() -> int:
    rng = np.random.default_rng(0)
    vocab = 248_320
    # Multiples of 1/32 in [-12, 12]: exact in f16, and still exact after the
    # +4 shift below, so the shifted pair differs by exactly a constant.
    base = np.round(np.clip(rng.normal(0.0, 3.0, vocab), -12, 12) * 32) / 32
    base = base.astype("<f2")
    order = np.argsort(-base.astype(np.float64), kind="stable")
    permuted = base.copy()
    permuted[order[0]], permuted[order[9]] = base[order[9]], base[order[0]]
    noisy = (base.astype(np.float64) + rng.normal(0.0, 0.05, vocab)).astype("<f2")

    failures = []

    def check(cond: bool, what: str) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")
        if not cond:
            failures.append(what)

    # KL itself, on a vocabulary small enough to do by hand.
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.25, 0.5, 0.25])
    expect = 0.5 * math.log(2.0) + 0.25 * math.log(0.5)
    got = kl_divergence(np.log(p).astype("<f2"), np.log(q).astype("<f2"))
    check(abs(got - expect) < 2e-3, f"hand KL {got:.6f} vs {expect:.6f}")

    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = os.path.join(tmp, "ref")
        cand_dir = os.path.join(tmp, "cand")
        _write_dir(ref_dir, {"identical": base, "shift": base, "permuted": base, "noisy": base})
        _write_dir(cand_dir, {"identical": base,
                              "shift": (base.astype(np.float64) + 4.0).astype("<f2"),
                              "permuted": permuted, "noisy": noisy})
        reports, notes = compare_dirs(ref_dir, cand_dir)
        by = {r.case: r for r in reports}
        print_report(reports, notes)
        print()

        r = by["identical"]
        check(r.kl == 0.0 and r.top1_match and r.top20_overlap == 20 and r.max_abs_logit == 0.0,
              f"identical: KL {r.kl}, top-1 match, 20/20, max|dlogit| 0")
        r = by["shift"]
        check(r.kl < 1e-12 and r.top1_match and r.top20_overlap == 20,
              f"constant shift: KL {r.kl:.3g}, top-1 match, 20/20")
        check(r.shifted and abs(r.offset - 4.0) < 1e-9 and r.max_abs_logprob < 1e-9,
              f"constant shift: flagged, offset {r.offset:+.4f}, max|dlogp| {r.max_abs_logprob:.3g}")
        r = by["permuted"]
        check(not r.top1_match and r.cand_top1 == int(order[9]) and r.kl > 0.0,
              f"permuted top-1: mismatch, cand top-1 {r.cand_top1}, KL {r.kl:.4g}")
        check(r.top20_overlap == 20, "permuted top-1: the top-20 sets still agree")
        r = by["noisy"]
        check(0.0 < r.kl < 0.05 and not r.shifted, f"noise: small positive KL {r.kl:.4g}, not shifted")

        # A candidate at another position is refused, not graded.
        bad_dir = os.path.join(tmp, "bad")
        _write_dir(bad_dir, {"identical": base}, frontier=99)
        try:
            compare_dirs(ref_dir, bad_dir)
            check(False, "mismatched frontier is refused")
        except Refusal as e:
            check("frontier" in str(e), f"mismatched frontier is refused ({e})")

    print(f"self-test: {'PASS' if not failures else 'FAIL (' + str(len(failures)) + ')'}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", nargs="?", help="the reference row directory")
    parser.add_argument("candidate", nargs="?", help="the candidate row directory")
    parser.add_argument("--json", help="also write the per-case report here")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic checks and exit")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.reference or not args.candidate:
        parser.error("REFERENCE and CANDIDATE are required without --self-test")
    try:
        reports, notes = compare_dirs(args.reference, args.candidate)
    except Refusal as e:
        print(f"compare_rows: refused: {e}", file=sys.stderr)
        return 1
    print_report(reports, notes)
    if args.json:
        if os.path.exists(args.json):
            print(f"compare_rows: {args.json} exists; not replacing it", file=sys.stderr)
            return 1
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"reference": args.reference, "candidate": args.candidate,
                       "notes": notes, "cases": [asdict(r) for r in reports]}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
