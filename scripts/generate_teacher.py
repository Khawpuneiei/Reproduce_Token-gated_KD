"""Generate teacher solutions for KD (sequence-level data, token-level KD on top).

Distilling on GSM8K reference solutions gives a style mismatch: at the first
reply tokens the teacher's distribution follows its own verbose style, so the
student learns to start verbosely but is never trained on how such text
continues, and its generations degenerate (see DEVLOG). Following GKD/DistiLLM
and TAID's own data, KD is run on the teacher's greedy solutions, keeping only
correct, finished ones.

Prompts come from the same seeded shuffle as before: positions [0, train_pool)
feed training, [train_pool, train_pool + calib_pool) feed tau calibration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tgkd.answers import extract_answer, has_final_format, is_correct
from tgkd.data import gold_answer, prompt_text, select_train_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-pool", type=int, default=2800)
    p.add_argument("--calib-pool", type=int, default=100)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/teacher_data"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.92)

    tok = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    rows = select_train_rows(list(load_dataset("openai/gsm8k", "main", split="train")), 10**9, args.data_seed)
    rows = rows[: args.train_pool + args.calib_pool]

    started, gen_tokens, kept = time.time(), 0, {"train": [], "calib": []}
    stats = {"train": [0, 0], "calib": [0, 0]}
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        enc = tok([prompt_text(tok, r["question"]) for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=stop_ids,
                                 pad_token_id=tok.pad_token_id)
        for j, (r, ids) in enumerate(zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist())):
            split = "train" if i + j < args.train_pool else "calib"
            n = next((k for k, t in enumerate(ids) if t in stop_ids), None)
            stats[split][1] += 1
            if n is None:  # unfinished
                continue
            gen_tokens += n
            text = tok.decode(ids[:n]).strip()
            pred, _ = extract_answer(text)
            if has_final_format(text) and is_correct(pred, gold_answer(r["answer"])):
                stats[split][0] += 1
                kept[split].append(dict(question=r["question"], answer=text, gold=gold_answer(r["answer"])))
        print(f"  {min(i + args.batch_size, len(rows))}/{len(rows)} kept train={len(kept['train'])} "
              f"calib={len(kept['calib'])}", flush=True)

    for split, items in kept.items():
        with (args.out_dir / f"teacher_{split}.jsonl").open("w", encoding="utf-8") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary = dict(teacher=args.teacher, seconds=time.time() - started, teacher_generated_tokens=gen_tokens,
                   train_kept=stats["train"][0], train_prompts=stats["train"][1],
                   calib_kept=stats["calib"][0], calib_prompts=stats["calib"][1],
                   max_new_tokens=args.max_new_tokens)
    (args.out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
