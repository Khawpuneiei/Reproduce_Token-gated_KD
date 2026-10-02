# Compute plan: fitting Reproduce 3 into 12 hours on an 8 GB laptop GPU

## Hardware observed (2026-10-02)

| | |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop, 8 GiB VRAM, WDDM, 80 W cap (runs ~77–79 °C, ~40 W under load) |
| Driver / CUDA | 537.53 / 12.2, PyTorch 2.5.1+cu121 |
| RAM | 16 GiB |
| Brief's assumption | single 24 GB GPU, 6–12 GPU-h |

## Throughput measured on this GPU (from the sibling Distill project logs)

- LoRA SFT, Qwen2.5-0.5B, 300k tokens at max length 1024: 693 s, about **430 tok/s**.
- Greedy GSM8K generation, 0.5B, 500 questions, batch 32, 512 new tokens: **4–6 min**.

KD also runs a frozen 1.5B teacher forward pass (≈ 2·1.5B FLOP/token, about the cost of the
student's 6·0.5B forward+backward) and a 152k-vocab KL. Expected KD speed: **200–250 tok/s**.

## Full-scale estimate (does not fit)

| Item | Estimate |
|---|---|
| GSM8K train, 7,473 examples × ~190 tokens (chat template) | ~1.4M tokens / epoch |
| One KD arm, 1 epoch at ~220 tok/s | ~1.8 h |
| 4 KD arms (vanilla, token gate, sample gate, TAID) | ~7.2 h |
| Full GSM8K test (1,319 q) × 7 models | ~1.5 h |
| GPU occupied by another job at start | 1.7–4.7 h |
| **Total** | **> 12 h** |

Memory also rules out the TAID recipe as written. Full fine-tuning a 0.5B student with AdamW
needs ~7 GB, and the bf16 teacher adds 3.1 GB, which is more than 8 GB.

## Downscaled plan (what was run)

- Teacher **Qwen2.5-1.5B-Instruct** (frozen, bf16). Student **Qwen2.5-0.5B** base. Identical
  tokenizer, verified.
- Student trained with **LoRA r=64, α=128 on all linear layers** instead of full FT.
- Logits only at reply positions, in 256-row chunks, with a detached hidden-state leaf.
  The (batch × seq × 152k) tensor never exists.
- **2,000** GSM8K train examples (fixed seed, same order for every arm), 1 epoch, max length 512,
  micro-batch 4 × grad-accum 4 (125 optimizer steps). Micro-batch 8 OOMed at the 7.36 GB cap.
- Eval: **all 1,319** GSM8K test questions, greedy, 320 new tokens.
- Core: 4 trained arms (KD, token gate, sample gate, TAID) + base student + teacher. The literal-τ gate and SFT run only if finished before 06:30.

The GPU was shared with the user's Distill matrix run (3 seeds). The user chose to let it finish
first. `run_all --wait-for-gpu 7000` waits for 3 straight minutes of ≥7 GB free VRAM, so the driver
started on its own when the GPU was released. Because of that wait, the training set was cut from
3,000 to 2,000 examples and the core arms from 6 to 4.

Measured actuals are recorded in the README results section.
