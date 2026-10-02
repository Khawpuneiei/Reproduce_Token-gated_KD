# Reproduce 3: TAID and GRACE-style token-gated KD on GSM8K

A single-GPU (8 GB) reproduction comparing **ungated KL distillation**, a **GRACE-style
teacher-entropy gate** at token and sample level, and **TAID** (Shing et al., 2025,
[arXiv 2501.16937](https://arxiv.org/abs/2501.16937), [SakanaAI/TAID](https://github.com/SakanaAI/TAID)).
Teacher: Qwen2.5-1.5B-Instruct. Student: Qwen2.5-0.5B. Dataset: GSM8K.

The design log is in [DEVLOG.md](DEVLOG.md). The 12-hour compute calculation and downscale are in
[docs/compute-plan.md](docs/compute-plan.md).

## What is compared

| Arm | Loss on reply tokens | Gate |
|---|---|---|
| `kd` | KL(p_T ‖ q_S) | none (baseline) |
| `token_gate` | Σ g_i·KL_i / Σ g_i | token: g_i = max(0.1, exp(−H_i / (τ·log\|V\|))), τ = 0.0133 |
| `sample_gate` | same | sample: one g per sequence from its mean H, τ = 0.0518 |
| `taid` | CE(p_t, q_S), p_t = softmax((1−t)·sg(z_S) + t·z_T) | adaptive t: 0.4 → 1.0 |
| `token_gate_literal`* | as token_gate | τ = 1, the literal exp(−H/log\|V\|) |
| `sft`* | CE on reference solutions | — |

\* Run only if time allowed after the core arms (see results).

H_i is the **teacher's** predictive entropy at reply position i, computed in fp32 over the 151,665 real
vocabulary rows. τ was fixed *before training* on a held-out slice, using one rule for both
granularities: the median unit's gate equals 0.5. The literal τ = 1 gate averages 0.964 on GSM8K
because teacher entropy is tiny next to log|V| = 11.93, so it barely gates
([calibration](DEVLOG.md#gate-calibration-pre-registered-rule-held-out-train-slice)).

Shared setup: 2,000 GSM8K train examples (fixed seed and order), 1 epoch, LoRA r=64/α=128 on all
linear layers, lr 2e-4 cosine, effective batch 16, max length 512, bf16. Evaluation covers all 1,319
GSM8K test questions, greedy, 320 new tokens. **ECE** uses 10 bins on the joint probability of the
final-answer tokens in the model's own decode.

## Results

_Filled in from `results/results.md` after the run._

## Run it

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m scripts.calibrate_tau                    # writes outputs/calibration/tau.json
python -m scripts.run_all --wait-for-gpu 7000      # resumable; skips finished receipts
```

Single steps:

```powershell
python -m scripts.train_kd --arm token_gate --tau 0.0133 --output-dir outputs/runs/token_gate
python -m scripts.evaluate --adapter outputs/runs/token_gate/adapter --output-dir outputs/runs/token_gate/eval
python -m scripts.report --root outputs/runs
```

## Layout

- `tgkd/gates.py`: stable entropy, token and sample gates, τ calibration
- `tgkd/losses.py`: per-token KL, CE, and the TAID loss plus `t` schedule (ported)
- `tgkd/answers.py`: answer extraction, answer-token confidence, ECE, Brier
- `tgkd/data.py`: GSM8K formatting and reply masks
- `scripts/`: `train_kd`, `evaluate`, `calibrate_tau`, `run_all`, `report`

## Deviations from the papers

- LoRA instead of full fine-tuning, and 2,000 examples × 1 epoch. Both are forced by 8 GB and
  12 h (see the compute plan).
- TAID's `t` update runs once per optimizer step on that step's mean loss, instead of per
  Lightning step without accumulation. The loss and schedule otherwise match `src/distil_losses/taid.py`.
- Off-policy distillation on GSM8K reference solutions (`<<calc>>` annotations removed), not
  teacher-generated text.
- The GRACE paper (arXiv 2601.22709) targets VLM quantization. Its gating idea is reimplemented here
  for text KD; the paper's code is not used.
