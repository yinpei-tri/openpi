"""Progress reporter for the val-MSE sweep. Prints a periodic status snapshot: how many
checkpoints are done, which are running (+ their batch progress), completed metrics so far, and
ETA. Reads only the on-disk artifacts the sweep writes (result JSONs + per-job logs + master
log) — it does NOT touch the running jobs.

Run once (single snapshot):     .venv/bin/python scripts/report_val_sweep.py
Loop every N seconds:           .venv/bin/python scripts/report_val_sweep.py --loop 600
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import time

# Paths + job count are env-overridable so the reporter follows whatever the sweep was pointed at
# (SWEEP_OPENPI_ROOT / SWEEP_OUT_DIR mirror run_val_sweep.py; SWEEP_TOTAL_JOBS sizes the ETA).
OPENPI = pathlib.Path(os.environ.get("SWEEP_OPENPI_ROOT", "/home/yinpei.dai/openpi"))
OUT_DIR = pathlib.Path(os.environ.get("SWEEP_OUT_DIR", str(OPENPI / "eval_results" / "valmse_results")))
JOBLOG_DIR = OUT_DIR / "joblogs"
MASTER = OPENPI / "_evallogs" / "sweep_master.log"
TOTAL_JOBS = int(os.environ.get("SWEEP_TOTAL_JOBS", "40"))
BATCHES_PER_JOB = 512

_BATCH_RE = re.compile(r"batch (\d+)/(\d+)")


def _short(name: str) -> str:
    return name.replace("m0717-50k-bs512-", "").replace("_granfine_verbsimp", "")


def snapshot() -> str:
    lines = []
    ts = time.strftime("%H:%M:%S")
    # completed results
    results = sorted(f for f in OUT_DIR.glob("*.json") if f.name != "SUMMARY.json")
    done = []
    for f in results:
        try:
            d = json.loads(f.read_text())
            for s in d.get("steps", []):
                done.append((d["exp_name"], s))
        except Exception:
            pass
    # Running jobs are read from the MASTER log (the current scheduler's authoritative record):
    # a job that was `launch`ed but not yet `DONE`/`FAIL`ed and has no result JSON. This avoids
    # counting stale joblogs from a previous sweep run (the master log is truncated per launch).
    launched, reaped = [], set()
    complete_line = None
    if MASTER.exists():
        for ln in MASTER.read_text().splitlines():
            m = re.search(r"launch gpu\d+: (\S+)/(\d+)", ln)
            if m:
                launched.append(f"{m.group(1)}__{m.group(2)}")
            m = re.search(r"(?:DONE|FAIL\S*) gpu\d+: (\S+)/(\d+)", ln)
            if m:
                reaped.add(f"{m.group(1)}__{m.group(2)}")
            if "SWEEP COMPLETE" in ln:
                complete_line = ln

    # A job is genuinely running if launched, not reaped, and no result JSON yet. Read its current
    # batch from its (current-sweep) joblog.
    done_names = {f.stem for f in results}
    running = []
    for name in launched:
        if name in reaped or name in done_names:
            continue
        cur = 0
        lg = JOBLOG_DIR / f"{name}.log"
        if lg.exists():
            m = None
            for m in _BATCH_RE.finditer(lg.read_text()):
                pass
            cur = int(m.group(1)) if m else 0
        running.append((name, cur))

    lines.append(f"===== val-MSE sweep report @ {ts} =====")
    lines.append(f"DONE: {len(done)}/{TOTAL_JOBS}   RUNNING: {len(running)}   "
                 f"PENDING: {max(0, TOTAL_JOBS - len(done) - len(running))}")
    if complete_line:
        lines.append(complete_line.strip())

    if running:
        lines.append("--- running (batch/512) ---")
        for name, cur in sorted(running, key=lambda x: -x[1]):
            pct = 100 * cur / BATCHES_PER_JOB
            lines.append(f"  {_short(name):<40s} {cur:>3d}/512 ({pct:4.0f}%)")

    if done:
        lines.append("--- completed metrics (action_mse | prog_acc | prog_mae | mode) ---")
        for exp, s in sorted(done, key=lambda x: (x[0], x[1].get("step") or 0)):
            acc = s.get("progress_acc")
            mae = s.get("progress_mae")
            acc_s = f"{acc:.3f}" if acc is not None else "  -  "
            mae_s = f"{mae:.4f}" if mae is not None else "  -   "
            lines.append(f"  {_short(exp):<40s} step {str(s.get('step')):>5s}  "
                         f"mse={s.get('action_mse'):.4f}  acc={acc_s}  mae={mae_s}  "
                         f"({s.get('progress_mode')})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="repeat every N seconds (0 = one shot)")
    args = ap.parse_args()
    while True:
        print(snapshot(), flush=True)
        if not args.loop:
            break
        # stop looping once the sweep is complete
        if MASTER.exists() and "SWEEP COMPLETE" in MASTER.read_text():
            print("[reporter] sweep complete — stopping loop.", flush=True)
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
