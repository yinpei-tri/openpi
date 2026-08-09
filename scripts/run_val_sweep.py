"""Scheduler for the System1 val-MSE sweep across all LOCAL checkpoints + GPUs.

Runs eval_val_mse.py for every (experiment, step) checkpoint found locally under checkpoints/,
16 at a time (2 per GPU on an 8x80GB box, XLA fraction 0.45), each pinned to its GPU. All 40
checkpoints (v1-v8 x 5 steps) are already downloaded, so there is NO S3 polling / no dynamic
enqueue — one clean pass, then exit. Results are written per-job so nothing is lost if a single
job dies; the sweep is resumable (a restart skips checkpoints whose result JSON already exists).

Determinism: every job uses the SAME num_workers + seed (0), so all checkpoints see the identical
val samples in the identical order (see the eval's determinism note).

Run (openpi env, from repo root); nohup so it survives disconnects:
    nohup .venv/bin/python scripts/run_val_sweep.py > _evallogs/sweep_master.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

# --- config ---
# Repo root + ckpt root are env-overridable so the sweep runs on any box (the default is the
# original workstation layout). SWEEP_CKPT_ROOT lets the ckpts live outside the repo entirely.
OPENPI = pathlib.Path(os.environ.get("SWEEP_OPENPI_ROOT", "/home/yinpei.dai/openpi"))
CKPT_ROOT = pathlib.Path(os.environ.get("SWEEP_CKPT_ROOT", str(OPENPI / "checkpoints")))
# Shards + output dir are overridable via env so the SAME scheduler runs both the val sweep and
# the train-set sweep (same 40 ckpts, same 32768-sample eval). Defaults = the val sweep.
VAL_SHARDS = os.environ.get(
    "SWEEP_SHARDS", str(OPENPI / "robocasa_training_data" / "system1_midset_0717_val" / "shards"))
OUT_DIR = pathlib.Path(os.environ.get("SWEEP_OUT_DIR", str(OPENPI / "eval_results" / "valmse_results")))
JOBLOG_DIR = OUT_DIR / "joblogs"

NUM_BATCHES = 512          # 512 * 64 = 32768 samples/ckpt
BATCH_SIZE = 64
NUM_WORKERS = 2
FLOW_STEPS = 10
# All overridable via env so the same script runs the big 8-GPU sweep OR a small serial run.
XLA_FRAC = os.environ.get("SWEEP_XLA_FRAC", "0.45")
N_GPUS = int(os.environ.get("SWEEP_N_GPUS", "8"))
SLOTS_PER_GPU = int(os.environ.get("SWEEP_SLOTS_PER_GPU", "2"))
MAX_CONCURRENT = N_GPUS * SLOTS_PER_GPU
# GPU ids to actually use (default 0..N_GPUS-1). Job on slot s -> GPUS[(s//SLOTS_PER_GPU)].
GPUS = [int(x) for x in os.environ.get("SWEEP_GPUS", ",".join(str(i) for i in range(N_GPUS))).split(",")]
# Skip step ckpts below this (e.g. 20000 to drop the noisy 10000). 0 = keep all.
MIN_STEP = int(os.environ.get("SWEEP_MIN_STEP", "0"))
# Which run dirs to sweep; overridable to target a subset (e.g. just v11/v12). Glob, sorted below.
EXP_GLOB = os.environ.get("SWEEP_EXP_GLOB", "m0717-50k-bs512-v*__*")
# Stagger launches: 16 simultaneous JIT compiles thrash the CPUs (compile balloons to ~200s+ and
# GPUs sit idle). Launch at most one new job per this many seconds so compiles spread out — once
# a job is past its ~90-150s compile it's GPU-bound and barely touches CPU, so 16 hold fine.
LAUNCH_STAGGER_S = 45
# Per-process CPU thread caps: XLA compile + numpy/OMP each grab ALL cores by default; 16 procs on
# a 256-core box => a ~1000 load-average thread explosion where nothing finishes. Cap so 16 coexist.
THREAD_CAP = "8"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def exp_dirs() -> list[pathlib.Path]:
    return sorted(d for d in CKPT_ROOT.glob(EXP_GLOB) if d.is_dir())


def step_dirs(exp_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted((d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()
                   and (d / "params").is_dir()), key=lambda d: int(d.name))


def make_jobs(exp_dir: pathlib.Path) -> list[dict]:
    """One job per step; skip steps whose result JSON already exists (resumable)."""
    jobs = []
    for sd in step_dirs(exp_dir):
        if MIN_STEP and int(sd.name) < MIN_STEP:
            continue   # e.g. drop the noisy 10000 step
        out = OUT_DIR / f"{exp_dir.name}__{sd.name}.json"
        if out.exists():
            log(f"skip (done): {exp_dir.name}/{sd.name}")
            continue
        jobs.append(dict(exp=exp_dir.name, step=sd.name, run_dir=str(exp_dir), out=str(out)))
    return jobs


def launch(job: dict, gpu: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = XLA_FRAC
    env["ROBOCASA_VAL_SHARDS"] = VAL_SHARDS
    env["PYTHONUNBUFFERED"] = "1"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
        env[k] = THREAD_CAP
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false").strip()
    env["TF_NUM_INTEROP_THREADS"] = "2"
    cmd = [
        str(OPENPI / ".venv" / "bin" / "python"), str(OPENPI / "scripts" / "eval_val_mse.py"),
        "--run-dir", job["run_dir"], "--only-step", job["step"],
        "--num-batches", str(NUM_BATCHES), "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(NUM_WORKERS), "--flow-steps", str(FLOW_STEPS),
        "--out", job["out"],
    ]
    fh = open(JOBLOG_DIR / f"{job['exp']}__{job['step']}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(OPENPI))
    p._logfh = fh
    log(f"launch gpu{gpu}: {job['exp']}/{job['step']}  (pid {p.pid})")
    return p


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JOBLOG_DIR.mkdir(parents=True, exist_ok=True)

    exps = exp_dirs()
    pending: list[dict] = []
    for d in exps:
        pending += make_jobs(d)
    total = len(pending)
    log(f"sweep start: {len(exps)} exps, {total} jobs to run "
        f"({NUM_BATCHES}x{BATCH_SIZE}={NUM_BATCHES*BATCH_SIZE} samples/ckpt, "
        f"{MAX_CONCURRENT} slots @ XLA {XLA_FRAC})")
    for d in exps:
        log(f"  {d.name}: steps {[s.name for s in step_dirs(d)]}")

    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    last_launch = 0.0
    done, failed = [], []
    t0 = time.time()

    def free_slots():
        return [s for s in range(MAX_CONCURRENT) if s not in running]

    while pending or running:
        # reap finished
        for slot, (p, job) in list(running.items()):
            if p.poll() is not None:
                p._logfh.close()
                ok = (p.returncode == 0 and pathlib.Path(job["out"]).exists())
                (done if ok else failed).append(f"{job['exp']}/{job['step']}")
                mins = (time.time() - t0) / 60
                log(f"{'DONE' if ok else 'FAIL(rc=%s)' % p.returncode} gpu{GPUS[slot//SLOTS_PER_GPU]}: "
                    f"{job['exp']}/{job['step']}  [{len(done)}/{total} done, {len(failed)} failed, "
                    f"{len(pending)} pending, {len(running)-1} running, {mins:.0f}min]")
                del running[slot]

        # staggered launch into free slots (one per LAUNCH_STAGGER_S)
        slots = free_slots()
        if slots and pending and (time.time() - last_launch) >= LAUNCH_STAGGER_S:
            job = pending.pop(0)
            running[slots[0]] = (launch(job, gpu=GPUS[slots[0] // SLOTS_PER_GPU]), job)
            last_launch = time.time()

        time.sleep(10)

    log(f"SWEEP COMPLETE in {(time.time()-t0)/60:.0f}min: {len(done)}/{total} done, {len(failed)} failed")
    if failed:
        log("failed: " + ", ".join(failed))
    # aggregate all per-job results into one summary
    summary = []
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name == "SUMMARY.json":
            continue
        try:
            d = json.loads(f.read_text())
            for s in d.get("steps", []):
                summary.append(dict(exp=d["exp_name"], **s))
        except Exception:
            pass
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {OUT_DIR / 'SUMMARY.json'} ({len(summary)} results)")


if __name__ == "__main__":
    main()
