"""Collect run receipts into results/results.md, results/results.json and figures."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

LABELS = {
    "base_student": "Base student (no training)",
    "teacher": "Teacher Qwen2.5-1.5B-Instruct",
    "sft": "SFT on GSM8K solutions",
    "kd": "Vanilla KL-KD (ungated)",
    "token_gate": "Token-gated KD (sharp tau)",
    "sample_gate": "Sample-gated KD (sharp tau)",
    "token_gate_literal": "Token-gated KD (literal tau=1)",
    "taid": "TAID",
}
ORDER = list(LABELS)


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def paired_bootstrap(a: list[bool], b: list[bool], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of acc(a) - acc(b) over the same questions."""
    rng = random.Random(seed)
    k = len(a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        diffs.append(sum(a[i] - b[i] for i in idx) / k)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def correct_list(root: Path, name: str):
    p = root / name / "eval" / "predictions.jsonl"
    if not p.exists():
        return None
    return [json.loads(line)["correct"] for line in p.read_text(encoding="utf-8").splitlines()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    kd_corr = correct_list(args.root, "kd")
    for name in ORDER:
        ev = load(args.root / name / "eval" / "eval_summary.json")
        if ev is None:
            continue
        tr = load(args.root / name / "training_summary.json") or {}
        row = dict(arm=name, label=LABELS[name], accuracy=ev["accuracy"], ece=ev["ece"], brier=ev["brier"],
                   mean_confidence=ev["mean_confidence"], format_rate=ev["format_rate"], n=ev["n"],
                   eval_minutes=ev["eval_seconds"] / 60, eval_vram_gb=ev["peak_vram_gb"],
                   gating_function=tr.get("gating_function"), gate_level=tr.get("gate_level"),
                   tau=tr.get("tau") if tr.get("gate_level") else None,
                   train_minutes=tr["train_seconds"] / 60 if tr else None,
                   train_vram_gb=tr.get("peak_vram_gb"), teacher_fwd_tokens=tr.get("teacher_fwd_tokens"),
                   effective_teacher_tokens=tr.get("effective_teacher_tokens"),
                   reply_tokens=tr.get("reply_tokens"), final_taid_t=tr.get("final_taid_t"))
        corr = correct_list(args.root, name)
        kd = load(args.root / "kd" / "eval" / "eval_summary.json")
        if kd and name != "kd" and corr and kd_corr and len(corr) == len(kd_corr):
            row["acc_gap_vs_kd"] = ev["accuracy"] - kd["accuracy"]
            row["acc_gap_ci95"] = paired_bootstrap(corr, kd_corr)
            row["ece_gap_vs_kd"] = ev["ece"] - kd["ece"]
        rows.append(row)
    (args.out / "results.json").write_text(json.dumps(rows, indent=2))

    def f(v, fmt):
        return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)

    lines = ["| Arm | Gate | Acc | ΔAcc vs KD (95% CI) | ECE ↓ | ΔECE vs KD | Brier | Teacher tokens (fwd / effective) | Train min | Train VRAM GB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = r.get("acc_gap_ci95")
        gap = f"{r['acc_gap_vs_kd']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—"
        gate = f"{r['gate_level']}, τ={r['tau']:.3g}" if r["gate_level"] else "—"
        toks = (f"{r['teacher_fwd_tokens']:,} / {r['effective_teacher_tokens']:,.0f}"
                if r["teacher_fwd_tokens"] else "—")
        lines.append(f"| {r['label']} | {gate} | {r['accuracy']:.3f} | {gap} | {r['ece']:.3f} | "
                     f"{f(r.get('ece_gap_vs_kd'), '+.3f')} | {r['brier']:.3f} | {toks} | "
                     f"{f(r['train_minutes'], '.1f')} | {f(r['train_vram_gb'], '.2f')} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in ORDER:
        lp = args.root / name / "train_log.jsonl"
        if not lp.exists():
            continue
        recs = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]
        steps = [r["step"] for r in recs]
        axes[0].plot(steps, [r["kl"] for r in recs], label=name)
        axes[1].plot(steps, [r["mean_gate"] for r in recs], label=name)
        if name == "taid":
            axes[2].plot(steps, [r["taid_t"] for r in recs], label="TAID t")
    axes[0].set(title="Unweighted KL(teacher||student) on train batches", xlabel="step")
    axes[1].set(title="Mean gate weight", xlabel="step")
    axes[2].set(title="TAID interpolation t", xlabel="step")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "training_curves.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in rows:
        ax.scatter(r["ece"], r["accuracy"])
        ax.annotate(r["arm"], (r["ece"], r["accuracy"]), fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set(xlabel="ECE (answer-token confidence, 10 bins) ↓", ylabel="GSM8K test accuracy ↑",
           title="Accuracy vs calibration")
    fig.tight_layout()
    fig.savefig(args.out / "accuracy_vs_ece.png", dpi=120)


if __name__ == "__main__":
    main()
