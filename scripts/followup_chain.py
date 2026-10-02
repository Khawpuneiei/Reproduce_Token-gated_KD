"""One-off follow-up chain for this session's run (kept for the record).

Waits for the first driver to exit, re-evaluates seed-0 KD so it carries the
sequence-level confidence metrics added mid-run, trains/evaluates a second seed
of the four core arms, then rebuilds the reports.
"""

import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
WAIT_PID = int(sys.argv[1])


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def run(cmd):
    print(f"[{time.strftime('%H:%M:%S')}] {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd).returncode
    print(f"    exit {code}", flush=True)
    if code:
        raise SystemExit(code)


while alive(WAIT_PID):
    time.sleep(30)

kd_eval = Path("outputs/runs_instruct/kd/eval/eval_summary.json")
if kd_eval.exists() and "ece_seq" not in kd_eval.read_text():
    kd_eval.rename(kd_eval.with_name("eval_summary_v1_answer_conf_only.json"))
run([PY, "-m", "scripts.run_all", "--arms", "kd", "--extra-arms", "--skip-reference",
     "--train-size", "5000", "--micro-batch", "2", "--grad-accum", "8", "--root", "outputs/runs_instruct"])
run([PY, "-m", "scripts.run_all", "--arms", "kd", "token_gate", "sample_gate", "taid", "--extra-arms",
     "--skip-reference", "--seed", "1", "--train-size", "5000", "--micro-batch", "2", "--grad-accum", "8",
     "--root", "outputs/runs_instruct_s1", "--report-dir", "results/seed1"])
run([PY, "-m", "scripts.report", "--root", "outputs/runs_instruct", "--out", "results"])
run([PY, "-m", "scripts.aggregate_seeds", "--roots", "outputs/runs_instruct", "outputs/runs_instruct_s1"])
print("chain done", flush=True)
