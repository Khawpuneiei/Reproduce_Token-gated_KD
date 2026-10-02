"""Train one arm: sft | kd | token_gate | sample_gate | taid.

Student: LoRA on all linear layers of Qwen2.5-0.5B. Teacher: frozen
Qwen2.5-1.5B-Instruct in bf16. Logits are only formed at reply positions, in
chunks, with a detached hidden-state leaf so the 152k-vocab tensors never
exist for the whole micro-batch at once.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tgkd.data import collate, encode_example, select_train_rows
from tgkd.gates import entropy_from_logits, sample_gate, token_gate
from tgkd.losses import TAID, ce_per_token, kl_per_token

ARMS = ("sft", "kd", "token_gate", "sample_gate", "taid")


def peak_gb(cuda: bool) -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--train-file", type=Path, help="jsonl with question/answer (teacher solutions); default GSM8K references")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--tau", type=float, default=1.0, help="gate sharpness; 1.0 = literal exp(-H/log|V|)")
    p.add_argument("--g-min", type=float, default=0.1, help="gate floor")
    p.add_argument("--taid-t-start", type=float, default=0.4)
    p.add_argument("--taid-alpha", type=float, default=5e-4)
    p.add_argument("--taid-beta", type=float, default=0.99)
    p.add_argument("--chunk", type=int, default=256, help="reply positions per logits chunk")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    dtype = torch.bfloat16 if cuda else torch.float32
    if cuda:
        # fail with OOM instead of silently spilling into WDDM shared system memory
        torch.cuda.set_per_process_memory_fraction(0.92)
    tok = AutoTokenizer.from_pretrained(args.student)
    vocab = len(tok)  # real tokens; padded embedding rows are excluded from every softmax

    if args.train_file:
        rows = [json.loads(x) for x in args.train_file.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = rows[: args.train_size]  # already in the seeded order
    else:
        rows = select_train_rows(list(load_dataset("openai/gsm8k", "main", split="train")), args.train_size,
                                 args.data_seed)
    examples = [encode_example(tok, r["question"], r["answer"], args.max_length) for r in rows]
    batches = [examples[i:i + args.micro_batch] for i in range(0, len(examples), args.micro_batch)]
    total_steps = math.ceil(len(batches) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=dtype,
                                                   attn_implementation="sdpa").to(dev)
    student.config.use_cache = False
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    inner = student.get_base_model()
    teacher = None
    if args.arm != "sft":
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=dtype,
                                                       attn_implementation="sdpa").to(dev).eval()
        teacher.requires_grad_(False)
        assert teacher.config.vocab_size >= vocab
    taid = TAID(args.taid_t_start, 1.0, args.taid_alpha, args.taid_beta) if args.arm == "taid" else None

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(args.warmup_frac * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (
        0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, total_steps - warm))))))

    log_path = args.output_dir / "train_log.jsonl"
    log_path.write_text("")
    counters = dict(seq_tokens=0, reply_tokens=0, teacher_fwd_tokens=0, effective_teacher_tokens=0.0)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    window = dict(loss=0.0, kl=0.0, gate=0.0, ent=0.0, n=0, tokens=0, t0=time.time())
    step, micro, done = 0, 0, False
    step_loss, step_micro = 0.0, 0  # TAID's update uses the loss of the current optimizer step only
    student.train()
    order = list(range(len(batches)))
    for epoch in range(args.epochs):
        if epoch:
            random.Random(args.seed + epoch).shuffle(order)
        for bi in order:
            batch = {k: v.to(dev) for k, v in collate(batches[bi], tok.pad_token_id).items()}
            mask = batch["target_mask"]
            n = int(mask.sum())
            targets = batch["input_ids"].roll(-1, dims=1)[mask]
            row_of = mask.nonzero()[:, 0]

            h = inner.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state[mask]
            h_leaf = h.detach().requires_grad_()
            chunks = [slice(i, min(i + args.chunk, n)) for i in range(0, n, args.chunk)]

            th, weights, ent = None, None, None
            if teacher is not None:
                with torch.no_grad():
                    th = teacher.model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).last_hidden_state[mask]
                    ent = torch.cat([entropy_from_logits(teacher.lm_head(th[c])[:, :vocab]) for c in chunks])
                    if args.arm == "token_gate":
                        weights = token_gate(ent, vocab, args.tau, args.g_min)
                    elif args.arm == "sample_gate":
                        b = mask.shape[0]
                        dense = torch.zeros(b, mask.shape[1], device=dev)
                        dense[mask] = ent
                        weights = sample_gate(dense, mask, vocab, args.tau, args.g_min)[mask]
            denom = float(weights.sum()) if weights is not None else float(n)

            loss_val, kl_val = 0.0, 0.0
            for c in chunks:
                zs = inner.lm_head(h_leaf[c])[:, :vocab]
                if args.arm == "sft":
                    per = ce_per_token(zs, targets[c])
                else:
                    with torch.no_grad():
                        zt = teacher.lm_head(th[c])[:, :vocab]
                    if args.arm == "taid":
                        per = taid.per_token(zs, zt)
                        with torch.no_grad():
                            kl_val += float(kl_per_token(zs.detach(), zt).sum())
                    else:
                        per = kl_per_token(zs, zt)
                        kl_val += float(per.detach().sum())
                w = weights[c] if weights is not None else None
                part = (per * w).sum() if w is not None else per.sum()
                (part / denom / args.grad_accum).backward()
                loss_val += float(part.detach())
            h.backward(h_leaf.grad)

            loss_val /= denom
            counters["seq_tokens"] += int(batch["attention_mask"].sum())
            counters["reply_tokens"] += n
            if teacher is not None:
                counters["teacher_fwd_tokens"] += n
                counters["effective_teacher_tokens"] += float(weights.sum()) if weights is not None else n
            window["loss"] += loss_val
            step_loss += loss_val
            step_micro += 1
            window["kl"] += kl_val / n
            window["gate"] += float(weights.mean()) if weights is not None else 1.0
            window["ent"] += float(ent.mean()) if ent is not None else 0.0
            window["n"] += 1
            window["tokens"] += int(batch["attention_mask"].sum())
            micro += 1
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if taid is not None:
                taid.update(step_loss / step_micro, step, total_steps)
            step_loss, step_micro = 0.0, 0
            if step % args.log_every == 0 or step == total_steps:
                el = time.time() - window["t0"]
                rec = dict(step=step, total_steps=total_steps, loss=window["loss"] / window["n"],
                           kl=window["kl"] / window["n"], mean_gate=window["gate"] / window["n"],
                           teacher_entropy=window["ent"] / window["n"], lr=sched.get_last_lr()[0],
                           tok_per_s=window["tokens"] / el, peak_vram_gb=peak_gb(cuda),
                           elapsed_s=time.time() - started, taid_t=taid.t if taid else None)
                with log_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
                window = dict(loss=0.0, kl=0.0, gate=0.0, ent=0.0, n=0, tokens=0, t0=time.time())
            if step >= total_steps:
                done = True
                break
        if done:
            break

    student.save_pretrained(args.output_dir / "adapter")
    summary = dict(vars(args), output_dir=str(args.output_dir), steps=step, examples=len(examples),
                   vocab=vocab, train_seconds=time.time() - started,
                   peak_vram_gb=peak_gb(cuda),
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30 if cuda else 0.0,
                   gating_function=(f"g=max({args.g_min}, exp(-H_teacher/({args.tau}*log|V|)))"
                                    if args.arm in ("token_gate", "sample_gate") else None),
                   gate_level={"token_gate": "token", "sample_gate": "sample"}.get(args.arm),
                   final_taid_t=taid.t if taid else None, **counters)
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("steps", "train_seconds", "peak_vram_gb")}))


if __name__ == "__main__":
    main()
