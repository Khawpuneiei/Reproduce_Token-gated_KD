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
