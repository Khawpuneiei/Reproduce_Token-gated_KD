"""Aggregate arms across seed roots: per-seed metrics, mean, and a paired
bootstrap (over test questions) of seed-averaged correctness vs. KD."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ARMS = ["kd", "token_gate", "sample_gate", "taid", "token_gate_literal", "sft"]
METRICS = ["accuracy", "ece", "ece_seq", "ece_min"]


def load_correct(root: Path, arm: str):
    p = root / arm / "eval" / "predictions.jsonl"
    if not p.exists():
        return None
    return [float(json.loads(x)["correct"]) for x in p.read_text(encoding="utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("results/seeds.md"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    kd_runs = [c for c in (load_correct(r, "kd") for r in args.roots) if c]
    kd_mean = [sum(v) / len(v) for v in zip(*kd_runs)] if kd_runs else None
    lines = ["| Arm | Seeds | Acc per seed | Acc mean | ΔAcc vs KD, seed-avg (95% CI) | ECE ans mean | ECE seq mean | ECE min-tok mean |",
             "|---|---|---|---|---|---|---|---|"]
    rows = []
    for arm in ARMS:
        evs = [json.loads((r / arm / "eval" / "eval_summary.json").read_text())
               for r in args.roots if (r / arm / "eval" / "eval_summary.json").exists()]
        if not evs:
            continue
        mean = {m: sum(e[m] for e in evs if e.get(m) is not None) / max(1, sum(e.get(m) is not None for e in evs))
                for m in METRICS}
        runs = [c for c in (load_correct(r, arm) for r in args.roots) if c]
        ci = ""
        if arm != "kd" and kd_mean and len(runs) == len(kd_runs):
            a = [sum(v) / len(v) for v in zip(*runs)]
            rng = random.Random(0)
            k = len(a)
            diffs = sorted(sum(a[i] - kd_mean[i] for i in idx) / k
                           for idx in ([rng.randrange(k) for _ in range(k)] for _ in range(2000)))
            d = (sum(a) - sum(kd_mean)) / k
            ci = f"{d:+.3f} [{diffs[50]:+.3f}, {diffs[1949]:+.3f}]"
        per = ", ".join(f"{e['accuracy']:.3f}" for e in evs)
        rows.append(dict(arm=arm, seeds=len(evs), **{f"{m}_mean": mean[m] for m in METRICS}))
        lines.append(f"| {arm} | {len(evs)} | {per} | {mean['accuracy']:.3f} | {ci or '—'} | {mean['ece']:.3f} | "
                     f"{mean['ece_seq']:.3f} | {mean['ece_min']:.3f} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
