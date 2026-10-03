"""Measure teacher reply-token entropy on a calibration slice and fix tau.

Pre-registered rule: for each gate granularity, tau is chosen so that the
median unit's (unfloored) gate is 0.5 -- the median reply token for the token
gate, the median sequence (mean reply-token entropy) for the sample gate. Calibration uses examples *outside* the training subset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tgkd.data import collate, encode_example, select_train_rows
from tgkd.gates import entropy_from_logits, tau_for_median_gate, token_gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("outputs/calibration/tau.json"))
    p.add_argument("--data-file", type=Path, help="jsonl of held-out question/answer (teacher solutions)")
    p.add_argument("--dump", type=Path, help="also save raw token / per-sample entropies (json) for figures")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = len(tok)
    if args.data_file:
        calib = [json.loads(x) for x in args.data_file.read_text(encoding="utf-8").splitlines() if x.strip()][: args.n]
    else:
        rows = select_train_rows(list(load_dataset("openai/gsm8k", "main", split="train")), 10**9, args.data_seed)
        calib = rows[args.train_size:args.train_size + args.n]  # disjoint from the training subset
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype).to(args.device).eval()
    ents, per_sample = [], []
    for i in range(0, len(calib), 8):
        b = collate([encode_example(tok, r["question"], r["answer"], 512) for r in calib[i:i + 8]], tok.pad_token_id)
        with torch.no_grad():
            h = model.model(input_ids=b["input_ids"].to(args.device),
                            attention_mask=b["attention_mask"].to(args.device)).last_hidden_state
            m = b["target_mask"].to(args.device)
            e = entropy_from_logits(model.lm_head(h[m])[:, :vocab]).cpu()
        ents.append(e)
        counts = m.sum(dim=1).tolist()
        for part in torch.split(e, counts):
            per_sample.append(float(part.mean()))
    ents = torch.cat(ents)
    tau = tau_for_median_gate(ents, vocab, 0.5)
    q = {f"p{int(k * 100)}": float(ents.quantile(k)) for k in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
    lit = token_gate(ents, vocab, 1.0, 0.1)
    sharp = token_gate(ents, vocab, tau, 0.1)
    ps = torch.tensor(per_sample)
    tau_sample = tau_for_median_gate(ps, vocab, 0.5)
    out = dict(n_examples=len(calib), n_tokens=len(ents), log_vocab=math.log(vocab), entropy_mean=float(ents.mean()),
               entropy_quantiles=q, tau_median_half=tau, tau_sample_median_half=tau_sample,
               literal_gate=dict(mean=float(lit.mean()), min=float(lit.min()), frac_below_0_9=float((lit < 0.9).float().mean())),
               sharp_gate=dict(mean=float(sharp.mean()), frac_at_floor=float((sharp <= 0.1 + 1e-6).float().mean()),
                               frac_below_0_5=float((sharp < 0.5).float().mean())),
               sample_mean_entropy=dict(min=float(ps.min()), median=float(ps.median()), max=float(ps.max())),
               sample_gate_at_token_tau=dict(min=float(token_gate(ps, vocab, tau, 0.1).min()),
                                             max=float(token_gate(ps, vocab, tau, 0.1).max())),
               sample_gate_at_sample_tau=dict(min=float(token_gate(ps, vocab, tau_sample, 0.1).min()),
                                              max=float(token_gate(ps, vocab, tau_sample, 0.1).max())))
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps({"token_entropy": [round(float(x), 6) for x in ents],
                                         "sample_mean_entropy": [round(x, 6) for x in per_sample],
                                         "log_vocab": math.log(vocab), "tau_token": tau, "tau_sample": tau_sample}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
