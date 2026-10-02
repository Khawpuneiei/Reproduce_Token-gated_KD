"""Resumable driver: train + evaluate every arm, then the base student and teacher.

Each step runs in its own process (GPU memory is released between steps) and is
skipped when its receipt file already exists. With --wait-for-gpu it blocks
until enough VRAM is free, so it can queue behind another job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def free_vram_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def run(cmd: list[str], log: Path, check: bool = True) -> bool:
    started = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(cmd[2:])}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        if check:
            raise SystemExit(f"step failed ({code}); see {log}")
        print(f"    failed ({code}) after {time.time() - started:.0f}s", flush=True)
        return False
    print(f"    done in {time.time() - started:.0f}s", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/runs_instruct"))
    p.add_argument("--report-dir", type=Path, default=Path("results"))
    p.add_argument("--tau-file", type=Path, default=Path("outputs/calibration/tau_tgen.json"))
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--train-file", type=Path, default=Path("outputs/teacher_data/teacher_train.jsonl"),
                   help="empty string = GSM8K reference solutions")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--seed", type=int, default=0, help="LoRA init / shuffling seed (data subset is fixed)")
    p.add_argument("--skip-reference", action="store_true", help="do not evaluate the base student and teacher")
    p.add_argument("--arms", nargs="+", default=["kd", "token_gate", "sample_gate", "taid"])
    p.add_argument("--extra-arms", nargs="*", default=["token_gate_literal", "sft"],
                   help="run after the core report only if the clock is before --extra-deadline")
    p.add_argument("--extra-deadline", default="06:30", help="HH:MM local time")
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--wait-for-gpu", type=int, default=0, help="MiB of free VRAM to wait for before starting")
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    args = p.parse_args()

    if args.wait_for_gpu:
        stable = 0  # require 3 consecutive minutes free so we don't slip between another job's steps
        while stable < 3:
            free = free_vram_mib()
            stable = stable + 1 if free >= args.wait_for_gpu else 0
            if stable == 0:
                print(f"[{time.strftime('%H:%M:%S')}] waiting for GPU: {free} MiB free", flush=True)
            time.sleep(60)
        print(f"[{time.strftime('%H:%M:%S')}] GPU free, starting", flush=True)

    py = sys.executable
    calib = json.loads(args.tau_file.read_text())
    tau, tau_sample = calib["tau_median_half"], calib["tau_sample_median_half"]
    log = args.root / "run_all.log"
    args.root.mkdir(parents=True, exist_ok=True)
    data = ["--train-file", str(args.train_file)] if str(args.train_file) not in ("", ".") else []
    common = ["--train-size", str(args.train_size), "--micro-batch", str(args.micro_batch),
              "--grad-accum", str(args.grad_accum), "--student", args.student, "--seed", str(args.seed), *data]
    eval_extra = ["--max-new-tokens", str(args.max_new_tokens)]
    eval_extra += ["--limit", str(args.eval_limit)] if args.eval_limit else []
    arm_flags = {
        "kd": ["--arm", "kd"],
        "token_gate": ["--arm", "token_gate", "--tau", f"{tau:.6f}", "--g-min", "0.1"],
        "sample_gate": ["--arm", "sample_gate", "--tau", f"{tau_sample:.6f}", "--g-min", "0.1"],
        "token_gate_literal": ["--arm", "token_gate", "--tau", "1.0", "--g-min", "0.1"],
        "taid": ["--arm", "taid"],
        "sft": ["--arm", "sft"],
    }
    def train_and_eval(name: str) -> None:
        out = args.root / name
        if not (out / "training_summary.json").exists():
            cmd = [py, "-m", "scripts.train_kd", *arm_flags[name], *common, "--output-dir", str(out)]
            if not run(cmd, log, check=False):
                # OOM fallback: same effective batch (16), smaller micro-batch + activation checkpointing
                cmd = [py, "-m", "scripts.train_kd", *arm_flags[name], "--train-size", str(args.train_size), *data,
                       "--student", args.student, "--seed", str(args.seed),
                       "--micro-batch", "2", "--grad-accum", "8", "--gradient-checkpointing",
                       "--output-dir", str(out)]
                run(cmd, log)
        if not (out / "eval" / "eval_summary.json").exists():
            run([py, "-m", "scripts.evaluate", "--model", args.student, "--adapter", str(out / "adapter"), "--output-dir",
                 str(out / "eval"), *eval_extra], log)

    for name in args.arms:
        train_and_eval(name)
    refs = () if args.skip_reference else (("base_student", args.student), ("teacher", "Qwen/Qwen2.5-1.5B-Instruct"))
    for name, model in refs:
        out = args.root / name / "eval"
        if not (out / "eval_summary.json").exists():
            run([py, "-m", "scripts.evaluate", "--model", model, "--output-dir", str(out), *eval_extra], log)
    run([py, "-m", "scripts.report", "--root", str(args.root), "--out", str(args.report_dir)], log)
    for name in args.extra_arms:
        if time.strftime("%H:%M") >= args.extra_deadline and time.strftime("%H:%M") < "12:00":
            print(f"[{time.strftime('%H:%M:%S')}] past {args.extra_deadline}; skipping extra arm {name}", flush=True)
            continue
        train_and_eval(name)
        run([py, "-m", "scripts.report", "--root", str(args.root), "--out", str(args.report_dir)], log)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
