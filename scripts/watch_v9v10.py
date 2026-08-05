"""Hourly watcher: wait for v9/v10 checkpoints to finish training on S3, then download + eval.

For each of v9, v10: every POLL_INTERVAL, check whether its 49999 step is FINALIZED on S3
(params/ present AND assets/robocasa_system1/norm_stats.json present — norm_stats is written
last, so it's the readiness signal). When ready:
  1. download steps 20000..49999 (skip 10000) to checkpoints/<exp>/ (flattened layout),
  2. launch the val-MSE sweep for JUST that exp on GPU1 (serial, XLA 0.6, steps>=20000),
  3. stop watching that version.

Runs standalone (nohup) so it survives disconnects — does NOT rely on the agent being awake.
Independent of the v11/v12 eval already running on GPU0.

    nohup .venv/bin/python scripts/watch_v9v10.py > _evallogs/watch_v9v10.log 2>&1 &
"""
from __future__ import annotations

import os
import subprocess
import time

OPENPI = "/home/yinpei.dai/openpi"
S3 = "s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints"
CKPT_ROOT = f"{OPENPI}/checkpoints"
POLL_INTERVAL = 3600          # 1 hour
STEPS = ["20000", "30000", "40000", "49999"]   # exclude 10000
EVAL_GPU = "1"
XLA_FRAC = "0.6"
# The run-dir names on S3 (NOTE the "-vla" suffix). v9-vla = progact_noexec,
# v10-vla = progact_nocond. Still training as of writing (49999 not finalized).
VERSIONS = ["m0717-50k-bs512-v9-vla", "m0717-50k-bs512-v10-vla"]


def log(m):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}", flush=True)


def _ls(prefix):
    r = subprocess.run(["aws", "s3", "ls", prefix], capture_output=True, text=True, timeout=180)
    return r.stdout if r.returncode == 0 else ""


def exp_name(v):
    """Resolve the exp-name dir under <v>/pi05_robocasa_system1/, or None if not there yet."""
    out = _ls(f"{S3}/{v}/pi05_robocasa_system1/")
    for ln in out.splitlines():
        if "PRE" in ln:
            return ln.split()[-1].rstrip("/")
    return None


def ready_49999(v, exp):
    """True if 49999 is finalized: params/ present AND norm_stats.json present."""
    base = f"{S3}/{v}/pi05_robocasa_system1/{exp}/49999"
    has_params = bool(_ls(f"{base}/params/").strip())
    has_norm = bool(_ls(f"{base}/assets/robocasa_system1/norm_stats.json").strip())
    return has_params and has_norm


def download(v, exp):
    """Sync steps 20000..49999 into checkpoints/<exp>/<step>/ (flattened)."""
    for s in STEPS:
        src = f"{S3}/{v}/pi05_robocasa_system1/{exp}/{s}"
        dest = f"{CKPT_ROOT}/{exp}/{s}"
        log(f"  syncing {exp}/{s} ...")
        r = subprocess.run(["aws", "s3", "sync", src, dest, "--only-show-errors"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            log(f"  SYNC FAILED {exp}/{s}: {r.stderr[-300:]}")
            return False
    log(f"  downloaded {exp} ({len(STEPS)} steps)")
    return True


def launch_eval(exp):
    """Launch the val-MSE sweep for just this exp on GPU1 (serial, XLA 0.6, steps>=20000)."""
    env = dict(os.environ)
    env.update(
        SWEEP_EXP_GLOB=exp,                 # exact dir name = target only this exp
        SWEEP_OUT_DIR=f"{OPENPI}/eval_results/valmse_results",
        SWEEP_SHARDS=f"{OPENPI}/robocasa_training_data/system1_midset_0717_val/shards",
        SWEEP_N_GPUS="1", SWEEP_SLOTS_PER_GPU="1", SWEEP_GPUS=EVAL_GPU,
        SWEEP_XLA_FRAC=XLA_FRAC, SWEEP_MIN_STEP="20000",
    )
    logf = open(f"{OPENPI}/_evallogs/{exp}_valmse_master.log", "w")
    subprocess.Popen([f"{OPENPI}/.venv/bin/python", f"{OPENPI}/scripts/run_val_sweep.py"],
                     env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=OPENPI)
    log(f"  LAUNCHED val-MSE for {exp} on GPU{EVAL_GPU} (XLA {XLA_FRAC}, steps>=20000)")


def main():
    pending = set(VERSIONS)
    log(f"watcher start: waiting for {sorted(pending)} (poll every {POLL_INTERVAL//60}min, eval on GPU{EVAL_GPU})")
    while pending:
        for v in sorted(pending):
            exp = exp_name(v)
            if not exp:
                log(f"{v}: exp dir not on S3 yet")
                continue
            if ready_49999(v, exp):
                log(f"{v}: 49999 FINALIZED ({exp}) -> download + eval")
                if download(v, exp):
                    launch_eval(exp)
                    pending.discard(v)
            else:
                log(f"{v}: {exp} exists, 49999 not finalized yet")
        if pending:
            time.sleep(POLL_INTERVAL)
    log("watcher done: both v9 + v10 downloaded + eval launched. exiting.")


if __name__ == "__main__":
    main()
