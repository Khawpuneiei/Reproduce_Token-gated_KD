# Reproduce 3 dev log: TAID and GRACE-style token-gated KD

A running record of what was inspected and decided, and what evidence exists.

## 2026-10-02: Setup

### Brief

- Source: `reproduce-3-taid-grace-gate.md`. Option A: TAID (Shing et al., arXiv 2501.16937,
  SakanaAI/TAID), scaled to Qwen2.5-1.5B → 0.5B. Option B: a DIY GRACE-style token gate
  `g_i = exp(−H_i / log|V|)` on standard KD, token-level vs sample-level, on GSM8K.
- Metrics: GSM8K accuracy and ECE vs ungated KD, plus teacher tokens used. Record the gating
  function, token vs sample setting, VRAM, time, and the accuracy/ECE gap.
- Both options were implemented. They share one trainer, so every arm sees identical data,
  order, LoRA config and schedule.

### Constraints found

- GPU: RTX 4060 Laptop, **8 GiB** (brief assumes 24 GB). See `docs/compute-plan.md`.
  The full-scale plan does not fit in 12 h, so it was downscaled to 3,000 train examples, LoRA, and 1 epoch.
- The GPU was busy with the user's Distill `run_matrix` (3 seeds, ~4.7 h left). The user chose to
  pause it. Its parent processes were stopped after its in-flight eval wrote a receipt; that job is
  resumable. A separate session then restarted the seed-17 remainder (~40 min). This run queued behind it
  rather than interfering.
- The Hugging Face snapshot symlink failed (WinError 1314, no symlink privilege). Only `config.json`
  was affected, and it was copied from its blob.
- The TAID repo's scripts are Lightning plus multi-GPU. Only the loss and the `t` schedule were ported
  (`tgkd/losses.py`), matching `src/distil_losses/taid.py`:
  logit-space interpolation `softmax((1−t)·sg(z_s) + t·z_t)`, CE to the student,
  `t_start=0.4, alpha=5e-4, beta=0.99`, and a monotone momentum-adaptive update. Deviation: `update_t`
  runs once per optimizer step on that step's mean loss. The original runs it per Lightning step
  with no grad accumulation.

### Decisions

- Inputs are built with the student tokenizer and an explicit system prompt. The Instruct template's
  default system prompt differs, and the token IDs were verified identical across the two tokenizers.
- Softmaxes run over the 151,665 real tokens. The padded rows 151,665–151,935 are excluded.
- Off-policy KD on GSM8K reference solutions with `<<calc>>` annotations removed. All KD arms use
  pure distillation loss (no CE mixing), as TAID does.
- Gated losses are normalized by the gate sum. Gating changes which tokens dominate, not the
  effective learning rate.
- "Teacher tokens used" is reported two ways: reply tokens the teacher scored, and the effective
  count Σg.
- Confidence for ECE is the joint probability of the final-answer number's tokens under the
  model's own greedy decode. It is recorded during generation by a logits processor. ECE uses
  10 equal-width bins.

### Gate calibration (pre-registered rule, held-out train slice)

`scripts/calibrate_tau.py` covers 64 examples outside the training subset, 6,284 reply tokens:

- Teacher entropy is tiny: median 0.11 nats, p90 1.45, p99 2.51. log|V| = 11.93.
- **The literal gate `exp(−H/log|V|)` is nearly a no-op.** Mean 0.964, minimum 0.736.
  It is kept as its own arm (`token_gate_literal`).
- A sharpened gate `exp(−H/(τ·log|V|))` uses τ chosen so the median unit's gate is 0.5.
  Token level: τ = 0.0133 (37% of tokens at the floor). Sample level, on mean reply entropy:
  τ = 0.0518 (sample gates 0.18–0.75). Using the token τ at sample level would clamp more than half
  the samples to the floor, so each granularity gets its own τ under the same rule.
- Floor g_min = 0.1 on every gate (brief pitfall, EGRSD).

### Implementation and checks

- Memory: logits exist only at reply positions, in 256-row chunks, with a detached hidden-state leaf,
  followed by a single backward pass through the transformer.
- 18 unit tests (`tests/test_core.py`) cover gates, losses, the TAID schedule, answer extraction,
  token-span confidence, ECE, and collation masks.
  A test caught `0·(−inf) = NaN` in the entropy when a logit is −inf. Fixed with a `where` mask.
- CPU smoke test (student self-distillation, 2 steps, every arm) passed. KL starts at 0 as expected
  (LoRA B = 0). It also showed the TAID update was averaging the logging window rather than the step; fixed.
- First GPU smoke test stalled at 0% utilization with 7.9 GB used. Another process was holding the GPU,
  and WDDM silently paged into shared memory. Both scripts now call
  `torch.cuda.set_per_process_memory_fraction(0.92)`, so an overflow raises OOM instead of crawling.

## 2026-10-03: First run and a change of KD data

### GPU contention and power

- The user chose to let the Distill matrix finish first. The "Apply 2" session ran seeds 18 and 19
  and then added a 100k-budget run. This driver waited (≥7 GB free for 3 straight minutes) and
  started at 01:22.
- The laptop was unplugged at 00:02 and slept from 00:03 to 00:13. The waiting driver paused and resumed
  cleanly. The user was notified.

### Memory and speed, measured

- Micro-batch 4 OOMed (7.12 GB allocated, at the teacher-entropy chunk). The automatic fallback,
  micro 2 × accum 8 with gradient checkpointing, peaks at **5.43 GB**.
- KD throughput is **~1,250–1,300 tok/s** (teacher forward included). That is about 5× the planning
  estimate, so a 2,000-example arm trains in **6.7 min**.

### Finding: KD on reference solutions makes the student degenerate (ablation kept)

The `kd` arm trained on GSM8K reference solutions (`outputs/runs/kd`) scored 30.7% accuracy, with
**89.6% of generations truncated at 320 tokens** and only 25.9% using `####`.
Generations open in the teacher's verbose style ("To determine… 1. …"), then drift into the base
model's habits: Python blocks with missing `)`/`}`, and loops such as `\boxed{3.0. norge`. The 137
answers that finished score 46.7%.

Cause: under teacher forcing on the terse reference text, the teacher's distribution at the first reply
tokens favors its own phrasing. The student learns to *start* like the teacher but never sees how
that style continues. This is the off-policy style mismatch that GKD and DistiLLM address. TAID's own
data (UltraChat) is also model-written.

Change: following that practice, KD now runs on **teacher-generated solutions**. Teacher greedy
decoding (512 new tokens) runs on the same seeded prompt order. Only finished, correct solutions
that use `####` are kept. Pool positions 0–2,799 feed training (first 2,000 kept); 2,800–2,899 feed τ
calibration. The calibration rule is unchanged (median unit gate 0.5) and is re-run on the held-out
teacher solutions, before training. Training max length rises to 768 and eval to 512 new tokens.
The reference-solution KD result is kept as an ablation.

### Teacher data and re-calibration

- The first generation pass kept only 1 of 32 solutions. The teacher answers correctly but writes
  `\boxed{}` and ignores the `####` instruction. Extraction now uses: `####` → last `\boxed{}` →
  last number, for both the filter and evaluation.
  A test passed by accident: a non-raw `"\boxed{"` is a backspace escape, so the branch never fired
  and the tests passed via the last-number fallback. It now uses raw strings, with a test the
  fallback cannot satisfy (`\boxed{7}. Check: 7 + 1 = 8` → 7).
  Re-scored with `\boxed` support, the reference-solution KD ablation is **35.6%** (was 30.7%).
  Format rate 61%; 90% still truncated at 320 tokens.
- Generation: 2,900 prompts in 40 min, 768,969 teacher tokens. Kept **1,619 / 2,800** train
  (57.8%) and **58 / 100** calibration solutions. All 1,619 are used, below the planned 2,000.
- Calibration on the 58 held-out teacher solutions (16,616 tokens): on its own text the teacher is
  nearly deterministic. Median H = 0.0021 nats, p90 0.68.
  Same rule (median unit gate 0.5): **token τ = 2.51e-4** (42% of tokens at the 0.1 floor) and
  **sample τ = 0.0190** (sample gates 0.25–0.75). The literal τ = 1 gate averages 0.986.
- Driver relaunched at 02:21 on `outputs/runs_tgen`, micro 2 × accum 8, eval at 512 new tokens.

### Finding: the base student cannot emit `<|im_end|>`, so the student is switched to Instruct

- KD on teacher data with the base student (`outputs/runs_tgen/kd`): accuracy 44.7%, format 91%,
  but **81% truncated at 512 tokens**. After the `\boxed{}` answer the student emits `𬭶`
  where `<|im_end|>` belongs, then loops or restarts the question. The earlier `norge` junk is the
  same failure.
- Cause: in Qwen2.5-0.5B **base**, the tied output row for `<|im_end|>` has cosine ≈ 1.000 with a
  cluster of untrained token rows (Thai pieces; `𬭶` at 0.997). LoRA does not train embeddings, so
  no hidden state can separate `<|im_end|>` from that cluster.
- Fix: the student is now **Qwen2.5-0.5B-Instruct** (max cosine from `<|im_end|>` to any other row
  is 0.957, and the model emits it natively). This still matches the brief's "1.5B → 0.5B".
  Teacher data and τ (teacher-only quantities) are unchanged. Base-student runs are kept under
  `outputs/runs` and `outputs/runs_tgen` as ablations.
- Bug: `training_summary.json` failed on a `Path` argument (`json.dumps(..., default=str)`). The
  `kd` arm was rerun.
- Driver relaunched at 02:56 on `outputs/runs_instruct`.

### Final runs (Instruct student, teacher data)

- Seed 0, all arms, plus base-student and teacher evals: 02:56–05:06. The KD eval was then re-run
  to add the sequence-level confidences. It reproduced 46.32% exactly (greedy decoding is deterministic).
- Mid-run change: answer-token confidence is ≈ 0.98 for every model, because the final number is
  copied from the solution, so its ECE ≈ 1 − accuracy. That metric is kept as the pre-specified
  primary. Two sequence-level confidences recorded from the same decode were added: the geometric-mean
  token probability and the min token probability. KD seed 0 was re-evaluated to include them.
- Seed 1 for kd, token_gate, sample_gate, taid: 05:16–06:24.
- Outcome (two-seed means): KD 46.9, sample gate 47.4 (+0.5, CI includes 0, 48% effective teacher
  tokens), token gate 44.7 (−2.2, lower in both seeds), TAID 45.2 (−1.7). Literal-τ gate 45.9
  and SFT on teacher text 47.3 (one seed). The untrained student scores 48.7 and the teacher 72.2.
- The KD arms do not differ in mean confidence (sequence ≈ 0.89, min-token ≈ 0.27), so every ECE
  difference tracks accuracy. No calibration improvement can be claimed.
- TAID: with 102 steps the adaptive increment (≤ 5e-4·(1−t)) is negligible next to the linear target
  (≈ 0.006 per step), so `t` followed the linear ramp. Its final t was 0.994.
- Report fix: the SFT arm was labeled "GSM8K solutions" but trained on teacher solutions. Its KL
  and gate panels were also removed from the training-curves figure.
- `rm -rf` of the superseded `outputs/runs*` folders was blocked by a safety check and was not
  retried. They are git-ignored and can be deleted by hand: `outputs/runs` (reference-text KD,
  base student), `outputs/runs_tgen` (teacher-text KD, base student), `outputs/gpu_smoke`,
  `outputs/cpu_smoke`, `outputs/report_check`.
