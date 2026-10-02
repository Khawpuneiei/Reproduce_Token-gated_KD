"""Greedy GSM8K-test evaluation: accuracy, answer-token confidence, ECE, Brier.

Confidence = joint probability of the final-answer number's tokens under the
model's own greedy decode. A logits processor records log p of each emitted
token during generation, so no second forward pass is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from tgkd.answers import (answer_confidence, has_final_format, brier, expected_calibration_error, extract_answer, is_correct,
                          span_token_indices)
from tgkd.data import gold_answer, prompt_text


class RecordGreedyLogprob(LogitsProcessor):
    def __init__(self):
        self.steps: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values)
        return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.set_per_process_memory_fraction(0.92)  # OOM rather than WDDM shared-memory spill
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if cuda else torch.float32,
                                                 attn_implementation="sdpa").to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.eval()
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    records = []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        enc = tok([prompt_text(tok, r["question"]) for r in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(args.device)
        rec = RecordGreedyLogprob()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=stop_ids,
                                 pad_token_id=tok.pad_token_id, logits_processor=LogitsProcessorList([rec]))
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        lps = torch.stack(rec.steps, dim=1).tolist()
        for r, ids, lp in zip(chunk, gen, lps):
            n = next((k for k, t in enumerate(ids) if t in stop_ids), len(ids))
            pieces = [tok.decode([t]) for t in ids[:n]]
            text = "".join(pieces)
            pred, span = extract_answer(text)
            conf = answer_confidence(lp[:n], span_token_indices(pieces, span)) if span else 0.0
            gold = gold_answer(r["answer"])
            seq_lp = lp[:n] if n else [float("-inf")]
            records.append(dict(question=r["question"], gold=gold, pred=pred, correct=is_correct(pred, gold),
                                confidence=conf, seq_confidence=math.exp(sum(seq_lp) / len(seq_lp)),
                                min_token_prob=math.exp(min(seq_lp)), has_hash=has_final_format(text), truncated=(n == len(ids)),
                                gen_tokens=n, completion=text))
        print(f"  {len(records)}/{len(rows)} acc={sum(x['correct'] for x in records) / len(records):.3f}",
              flush=True)

    conf = [x["confidence"] for x in records]
    corr = [x["correct"] for x in records]
    summary = dict(model=args.model, adapter=str(args.adapter) if args.adapter else None, n=len(records),
                   accuracy=sum(corr) / len(corr), ece=expected_calibration_error(conf, corr),
                   brier=brier(conf, corr), mean_confidence=sum(conf) / len(conf),
                   ece_seq=expected_calibration_error([x["seq_confidence"] for x in records], corr),
                   brier_seq=brier([x["seq_confidence"] for x in records], corr),
                   mean_seq_confidence=sum(x["seq_confidence"] for x in records) / len(records),
                   ece_min=expected_calibration_error([x["min_token_prob"] for x in records], corr),
                   brier_min=brier([x["min_token_prob"] for x in records], corr),
                   mean_min_token_prob=sum(x["min_token_prob"] for x in records) / len(records),
                   format_rate=sum(x["has_hash"] for x in records) / len(records),
                   truncated_rate=sum(x["truncated"] for x in records) / len(records),
                   mean_gen_tokens=sum(x["gen_tokens"] for x in records) / len(records),
                   eval_seconds=time.time() - started,
                   peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0,
                   max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    (args.output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("accuracy", "ece", "brier", "format_rate", "eval_seconds")}))


if __name__ == "__main__":
    main()
