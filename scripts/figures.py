"""Static README figures (results/figures/*.png) from run receipts.

Colors: validated categorical palette (fixed slot per arm, never cycled);
references (teacher, untrained student) in neutral ink. Every mark is directly
labeled because three slots sit below 3:1 contrast on the light surface.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tgkd.answers import extract_answer, is_correct  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
COLOR = {  # fixed slot order: blue, orange, aqua, yellow, magenta, green
    "kd": "#2a78d6", "token_gate": "#eb6834", "sample_gate": "#1baf7a",
    "taid": "#eda100", "token_gate_literal": "#e87ba4", "sft": "#008300",
    "teacher": INK2, "base_student": MUTED,
}
LABEL = {
    "kd": "Vanilla KD", "token_gate": "Token gate", "sample_gate": "Sample gate", "taid": "TAID",
    "token_gate_literal": "Token gate, literal τ=1", "sft": "SFT on teacher text",
    "teacher": "Teacher 1.5B", "base_student": "Untrained student",
}
KD_ARMS = ["kd", "sample_gate", "token_gate_literal", "token_gate", "taid"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def jl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def ev(root: Path, arm: str):
    p = root / arm / "eval" / "eval_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def preds(root: Path, arm: str):
    p = root / arm / "eval" / "predictions.jsonl"
    return jl(p) if p.exists() else None


def bootstrap_ci(a, b, n=2000):
    import random
    rng = random.Random(0)
    k = len(a)
    d = sorted(sum(a[i] - b[i] for i in idx) / k for idx in ([rng.randrange(k) for _ in range(k)] for _ in range(n)))
    return (sum(a) - sum(b)) / k, d[int(0.025 * n)], d[int(0.975 * n)]


def save(fig, out: Path, name: str):
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def fig_accuracy(roots, out):
    rows = []
    for arm in ["sft", "sample_gate", "kd", "token_gate_literal", "taid", "token_gate"]:
        accs = [e["accuracy"] for e in (ev(r, arm) for r in roots) if e]
        rows.append((arm, accs))
    rows.sort(key=lambda r: sum(r[1]) / len(r[1]))
    teacher, base = ev(roots[0], "teacher")["accuracy"], ev(roots[0], "base_student")["accuracy"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for y, (arm, accs) in enumerate(rows):
        m = sum(accs) / len(accs)
        if len(accs) > 1:
            ax.plot([min(accs) * 100, max(accs) * 100], [y, y], color=COLOR[arm], lw=2, alpha=0.45,
                    solid_capstyle="round")
        for a in accs:
            ax.scatter(a * 100, y, s=46, facecolor=SURFACE, edgecolor=COLOR[arm], linewidth=2, zorder=3)
        ax.scatter(m * 100, y, s=120, marker="D", color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{m * 100:.1f}%" + ("" if len(accs) > 1 else "  (1 seed)"), (max(accs) * 100, y),
                    xytext=(10, -4), textcoords="offset points", color=INK, fontsize=9.5)
    ax.axvline(base * 100, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(base * 100, len(rows) - 0.35, f"untrained student {base * 100:.1f}%", color=INK2, fontsize=9,
            ha="right", va="bottom")
    ax.set_yticks(range(len(rows)), [LABEL[a] for a, _ in rows])
    ax.set_xlim(42.5, 50)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("GSM8K test accuracy (%, 1,319 questions, greedy)")
    ax.set_title(f"No KD variant beats the untrained student  (teacher: {teacher * 100:.1f}%, off-scale)")
    ax.grid(axis="y", visible=False)
    ax.text(42.6, -0.55, "◆ mean   ○ individual seeds   — seed range", color=INK2, fontsize=8.5)
    save(fig, out, "fig1_accuracy.png")


def fig_delta(roots, out):
    kd_runs = [[x["correct"] for x in preds(r, "kd")] for r in roots]
    kd_mean = [sum(v) / len(v) for v in zip(*kd_runs)]
    rows = []
    for arm in ["sample_gate", "sft", "token_gate_literal", "taid", "token_gate"]:
        runs = [[x["correct"] for x in p] for p in (preds(r, arm) for r in roots) if p]
        base = kd_mean if len(runs) == len(kd_runs) else kd_runs[0]
        a = [sum(v) / len(v) for v in zip(*runs)]
        rows.append((arm, len(runs), *bootstrap_ci(a, base)))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    for y, (arm, nseed, d, lo, hi) in enumerate(reversed(rows)):
        solid = nseed > 1
        ax.plot([lo * 100, hi * 100], [y, y], color=COLOR[arm], lw=2.5, alpha=1 if solid else 0.55,
                solid_capstyle="round")
        ax.scatter(d * 100, y, s=70, color=COLOR[arm] if solid else SURFACE, edgecolor=COLOR[arm],
                   linewidth=2, zorder=3)
        tag = "2 seeds" if solid else "1 seed, vs KD seed 0"
        ax.annotate(f"{d * 100:+.1f} pt  [{lo * 100:+.1f}, {hi * 100:+.1f}]   {tag}", (max(hi * 100, 0.6), y),
                    xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1.2)
    ax.text(0, len(rows) - 0.4, " vanilla KD", color=INK2, fontsize=9, va="bottom")
    ax.set_yticks(range(len(rows)), [LABEL[r[0]] for r in reversed(rows)])
    ax.set_xlim(-5, 7.5)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Accuracy difference vs vanilla KD (points), paired bootstrap 95% CI over questions")
    ax.set_title("Token gating and TAID fall below KD; sample gating ties it")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig2_delta_vs_kd.png")


def fig_efficiency(roots, out):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    offsets = {"kd": (8, 6), "taid": (8, -12), "token_gate_literal": (-10, 10), "token_gate": (8, -4),
               "sample_gate": (8, 4)}
    for arm in KD_ARMS:
        trs = [json.loads((r / arm / "training_summary.json").read_text()) for r in roots
               if (r / arm / "training_summary.json").exists()]
        accs = [ev(r, arm)["accuracy"] for r in roots if ev(r, arm)]
        frac = trs[0]["effective_teacher_tokens"] / trs[0]["teacher_fwd_tokens"] * 100
        m = sum(accs) / len(accs) * 100
        ax.scatter(frac, m, s=110, color=COLOR[arm], edgecolor=SURFACE, linewidth=2, zorder=3)
        ha = "right" if offsets[arm][0] < 0 else "left"
        ax.annotate(f"{LABEL[arm]}  {m:.1f}%", (frac, m), xytext=offsets[arm], textcoords="offset points",
                    fontsize=9.5, ha=ha, color=INK)
    base = ev(roots[0], "base_student")["accuracy"] * 100
    ax.axhline(base, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(42, base + 0.12, f"untrained student {base:.1f}%", color=INK2, fontsize=9)
    ax.set_xlim(40, 108)
    ax.set_ylim(44, 49.4)
    ax.set_xlabel("Effective teacher tokens used, Σg / tokens scored (%)")
    ax.set_ylabel("Accuracy (%), mean over seeds")
    ax.set_title("Sample gating keeps KD accuracy with half the teacher signal")
    save(fig, out, "fig3_teacher_tokens.png")


def fig_gate(entropy_file: Path, out):
    d = json.loads(entropy_file.read_text())
    h, hs, lv = d["token_entropy"], d["sample_mean_entropy"], d["log_vocab"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    floor = 1e-6
    logh = [math.log10(max(x, floor)) for x in h]
    a1.hist(logh, bins=48, color=COLOR["kd"], edgecolor=SURFACE, linewidth=1)
    med = sorted(h)[len(h) // 2]
    a1.axvline(math.log10(med), color=INK, lw=1.2)
    a1.text(math.log10(med) + 0.08, a1.get_ylim()[1] * 0.92, f"median {med:.4f} nats", fontsize=9)
    a1.axvline(math.log10(lv), color=INK2, lw=1.2, ls=(0, (4, 3)))
    a1.text(math.log10(lv) - 0.08, a1.get_ylim()[1] * 0.78, "log|V| = 11.93", fontsize=9, ha="right", color=INK2)
    a1.set_xticks(range(-6, 2), [f"1e{k}" for k in range(-6, 2)])
    a1.set_xlabel("Teacher entropy per reply token (nats, log scale; ≤1e-6 clipped)")
    a1.set_ylabel("Tokens")
    a1.set_title(f"Teacher entropy on its own text ({len(h):,} held-out tokens)")
    a1.grid(axis="x", visible=False)

    xs = [10 ** (k / 50) for k in range(-300, 60)]
    curves = [("token_gate_literal", 1.0, "literal τ=1\n(mean gate 0.986)"),
              ("sample_gate", d["tau_sample"], f"sample τ={d['tau_sample']:.3g}"),
              ("token_gate", d["tau_token"], f"token τ={d['tau_token']:.2g}")]
    for arm, tau, lab in curves:
        g = [max(0.1, math.exp(-x / (tau * lv))) for x in xs]
        a2.plot(xs, g, color=COLOR[arm], lw=2.2)
        x_half = tau * lv * math.log(2)  # where the unfloored gate crosses 0.5
        a2.annotate(lab, (x_half, 0.5), xytext=(-8, 0), textcoords="offset points", fontsize=9, color=INK,
                    ha="right", va="center")
    a2.axhline(0.1, color=INK2, lw=1, ls=(0, (2, 2)))
    a2.text(1.2e-6, 0.12, "floor g_min = 0.1", fontsize=8.5, color=INK2)
    a2.plot([max(x, floor) for x in hs], [0.02] * len(hs), "|", color=COLOR["sample_gate"], ms=10, mew=1.4)
    a2.text(1.2e-6, 0.03, "| per-sample mean entropy", fontsize=8.5, color=INK2)
    a2.set_xscale("log")
    a2.set_xlim(1e-6, 12)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("Teacher entropy H (nats, log scale)")
    a2.set_ylabel("Gate g = max(0.1, exp(−H / (τ·log|V|)))")
    a2.set_title("Gate functions (τ set so the median unit gets 0.5)")
    fig.tight_layout(w_pad=3)
    save(fig, out, "fig4_gate_function.png")


def fig_reliability(root, out):
    arms = ["teacher", "kd", "sample_gate", "token_gate", "taid"]
    fig, axes = plt.subplots(1, len(arms), figsize=(15, 3.6), sharey=True)
    for ax, arm in zip(axes, arms):
        p = preds(root, arm)
        conf = [x["seq_confidence"] for x in p]
        corr = [x["correct"] for x in p]
        lo, width = 0.5, 0.05  # all mass sits above 0.5; 10 bins of 0.05 for display (ECE uses 10 × 0.1)
        bins = [[] for _ in range(10)]
        for c, y in zip(conf, corr):
            if c >= lo:
                bins[min(9, int((c - lo) / width))].append(y)
        ax.plot([lo, 1], [lo, 1], color=INK2, lw=1, ls=(0, (4, 3)))
        for b, ys in enumerate(bins):
            if len(ys) < 10:
                continue
            ax.bar(lo + b * width + width / 2, sum(ys) / len(ys), width=width * 0.92, color=COLOR[arm],
                   edgecolor=SURFACE, linewidth=1.5)
        e = ev(root, arm)
        ax.set_title(f"{LABEL[arm]}\nacc {e['accuracy']*100:.1f}% · ECE {e['ece_seq']:.3f}", fontsize=10.5)
        ax.set_xlim(lo, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Sequence confidence")
        ax.text(lo + 0.015, 0.92, f"mean conf {sum(conf) / len(conf):.2f}", fontsize=8.5, color=INK2, va="top")
    axes[0].set_ylabel("Fraction correct")
    fig.suptitle("Reliability (geometric-mean token probability, seed 0; bins with ≥10 questions). "
                 "Students are equally overconfident, so ECE tracks accuracy.",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, out, "fig5_reliability.png")


def fig_flips(root, out):
    kd = [x["correct"] for x in preds(root, "kd")]
    rows = []
    for arm in ["base_student", "sft", "sample_gate", "token_gate_literal", "taid", "token_gate"]:
        c = [x["correct"] for x in preds(root, arm)]
        fixed = sum(1 for a, b in zip(c, kd) if a and not b)
        broken = sum(1 for a, b in zip(c, kd) if b and not a)
        rows.append((arm, fixed, broken))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for y, (arm, fixed, broken) in enumerate(reversed(rows)):
        ax.barh(y, -broken, color="#e34948", height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, fixed, color=COLOR["kd"], height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.text(-broken - 3, y, f"{broken}", ha="right", va="center", fontsize=9)
        ax.text(fixed + 3, y, f"{fixed}", ha="left", va="center", fontsize=9)
        ax.text(222, y, f"net {fixed - broken:+d}", ha="right", va="center", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(rows)), [LABEL[r[0]] for r in reversed(rows)])
    ax.set_xlim(-180, 225)
    ax.text(-90, len(rows) - 0.35, "KD right → arm wrong", ha="center", fontsize=9, color=INK2)
    ax.text(90, len(rows) - 0.35, "KD wrong → arm right", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(-0.6, len(rows))
    ax.set_xlabel("Questions (of 1,319) whose outcome differs from vanilla KD, seed 0")
    lo_n = min(min(f, b) for _, f, b in rows)
    hi_n = max(max(f, b) for _, f, b in rows)
    ax.set_title(f"Each arm flips {lo_n}–{hi_n} questions each way vs KD; the net effects are small")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig6_flips_vs_kd.png")


def fig_training(root, out):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 3.9))
    for arm in KD_ARMS:
        recs = jl(root / arm / "train_log.jsonl")
        s = [r["step"] for r in recs]
        a1.plot(s, [r["kl"] for r in recs], color=COLOR[arm], lw=2, label=LABEL[arm])
        if arm in ("token_gate", "sample_gate", "token_gate_literal"):
            a2.plot(s, [r["mean_gate"] for r in recs], color=COLOR[arm], lw=2)
            a2.annotate(LABEL[arm], (s[-1], recs[-1]["mean_gate"]), xytext=(5, -3), textcoords="offset points",
                        fontsize=8.5, color=INK)
    a1.set_xlim(0, 105)
    a1.legend(fontsize=8.5, loc="upper right", handlelength=1.6)
    a1.set_title("Unweighted KL(teacher‖student) on train batches")
    a1.set_xlabel("optimizer step")
    a2.set_xlim(0, 150)
    a2.set_ylim(0, 1.05)
    a2.set_title("Mean gate weight per batch")
    a2.set_xlabel("optimizer step")
    recs = jl(root / "taid" / "train_log.jsonl")
    s = [r["step"] for r in recs]
    total = recs[-1]["total_steps"]
    a3.plot([0, total], [0.4, 1.0], color=INK2, lw=1.2, ls=(0, (4, 3)))
    a3.text(total * 0.38, 0.42, "dashed: linear target 0.4 → 1.0 (hidden under t)\n"
            "adaptive term ≤5e-4/step never overtakes it", color=INK2, fontsize=9)
    a3.plot(s, [r["taid_t"] for r in recs], color=COLOR["taid"], lw=2.2)
    a3.set_ylim(0.35, 1.03)
    a3.set_title("TAID t followed the linear schedule exactly")
    a3.set_xlabel("optimizer step")
    fig.tight_layout(w_pad=2.5)
    save(fig, out, "fig7_training.png")


def fig_ablation(out, gt_root: Path, tgen_root: Path, inst_root: Path):
    gt = preds(gt_root, "kd")
    gt_acc = sum(is_correct(extract_answer(x["completion"])[0], x["gold"]) for x in gt) / len(gt)
    setups = [
        ("Reference text\nbase student\n(320 tok)", gt_acc, ev(gt_root, "kd")["truncated_rate"]),
        ("Teacher text\nbase student", ev(tgen_root, "kd")["accuracy"], ev(tgen_root, "kd")["truncated_rate"]),
        ("Teacher text\nInstruct student\n(final)", ev(inst_root, "kd")["accuracy"], ev(inst_root, "kd")["truncated_rate"]),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = range(len(setups))
    for ax, k, title, color in ((a1, 1, "Vanilla KD accuracy", COLOR["kd"]),
                                (a2, 2, "Generations hitting the token limit", "#e34948")):
        vals = [s[k] * 100 for s in setups]
        ax.bar(xs, vals, width=0.55, color=color, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
        ax.set_xticks(list(xs), [s[0] for s in setups], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Fixing the data (teacher-written text) and the stop token (Instruct student) made KD usable",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, out, "fig8_setup_ablation.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", type=Path, nargs="+",
                    default=[Path("outputs/runs_instruct"), Path("outputs/runs_instruct_s1")])
    ap.add_argument("--entropy", type=Path, default=Path("results/teacher_entropy_calib.json"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_accuracy(args.roots, args.out)
    fig_delta(args.roots, args.out)
    fig_efficiency(args.roots, args.out)
    fig_gate(args.entropy, args.out)
    fig_reliability(args.roots[0], args.out)
    fig_flips(args.roots[0], args.out)
    fig_training(args.roots[0], args.out)
    fig_ablation(args.out, Path("outputs/runs"), Path("outputs/runs_tgen"), args.roots[0])


if __name__ == "__main__":
    main()
