---
name: sagemaker-job-monitor
description: >-
  Monitor and analyze an openpi SageMaker JAX training job (AWS Batch service job),
  single-node OR multi-node. Use when the user asks to check a job's status, watch it
  until it schedules/starts/finishes, pull its CloudWatch logs, or analyze training
  health (step progress, checkpoint saves, failure/deadlock detection; plus data-sharding
  fingerprints for multi-node). Also the reference for the submitted_jobs/ record layout.
  Triggers: "check the job", "is it running", "watch the job", "analyze the log", "did it
  start training", "monitor <job/exp name>".
---

# SageMaker job monitor (openpi JAX training)

openpi training jobs are submitted via `scripts/sagemaker/launch.py` as **AWS Batch service
jobs** (`SubmitServiceJob`) on `fss-*` queues. Key facts that trip people up:

- `aws sagemaker describe-training-job` returns **"resource not found"** — these are NOT
  classic SageMaker training jobs. Use `aws batch list-service-jobs --job-status <ST>`
  (service jobs require an explicit status filter; there is no "list all").
- Ambient `AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN` env vars **override** the SSO profile in
  boto3/CLI → `AccessDenied`. Always `unset` them first (the script does this).
- Logs go to CloudWatch group **`/aws/sagemaker/TrainingJobs`**, streams
  `AWSBatch<jobname-hash>/algo-N-<ts>` where **algo-1 = rank 0** (does the S3 checkpoint
  sync); multi-node adds **algo-2 = rank 1**. The `<hash>` is not the timestamp, so match
  streams by name prefix or by a body string (the job monitor does the latter).
- CloudWatch ingestion lags real time by ~2–3 min; a job can be RUNNING while the newest
  log event is minutes old. Don't conclude "stuck" from a stale tail — poll again.

## Where jobs are recorded (submitted_jobs/ layout)

Every submission writes a record folder keyed by the **semantic experiment name**
(`sanitize(exp_name)`), matching the S3 checkpoint layout:

```
scripts/sagemaker/submitted_jobs/<semantic>/
    job.json         # machine-readable: job_name, queue, image, args, checkpoint_s3_dir, ...
    job.md           # human-readable summary
    norm_stats.json  # exact norm stats baked into that job's image (reproducibility)
```

`job.json.job_name` is the timestamped CloudWatch name (what you pass to the monitor);
`job.json.checkpoint_s3_dir` is the exact S3 path. To find a run, look under its semantic
folder — e.g. `submitted_jobs/subset0713-30k-progact/job.json`.

S3 layout (from `job.json.checkpoint_s3_dir`):
```
<s3_prefix>/<semantic>/<config>/<exp_name>/<step>/{params,assets[,train_state]}/   # checkpoints
<s3_prefix>/_artifacts/<job_name>/output.tar.gz                                    # SageMaker artifacts (off the ckpt root)
```

## Usage

Run the helper (job substring = any unique part of the CloudWatch job name, usually the
timestamp like `07-15-15-10-37`, or the semantic name):

```bash
bash .claude/skills/sagemaker-job-monitor/scripts/monitor.sh <job_substring> <mode>
```

Modes:
- `status` (default) — current batch status (RUNNABLE / STARTING / RUNNING / SUCCEEDED / FAILED).
- `logs` — pull the node(s') logs and print the health-signal summary.
- `watch` — poll until the job leaves the queue and, once RUNNING, until real training
  steps advance (success) or a failure signal appears. Runs long; launch it with
  `run_in_background: true` and analyze when it completes.

Env overrides: `QUEUE` (skip autodetect), `PROFILE` (default `sagemaker`), `REGION`
(default `us-west-2`), `INTERVAL` (watch seconds, default 120), `MAXPOLL` (default 120),
`NODES` (streams to show; default 2 — single-node just shows its 1), `MATCH` (body string
to pin this job's streams; defaults to `<job_substring>`), `STREAM_HASH` (pin the exact
`AWSBatch<hash>` prefix if known, to skip scanning other users' streams).

For a long watch, prefer running the script as a **background Bash task** so you're
notified on completion rather than blocking, then run `logs` mode to analyze.

## Adding a new job to monitor

When a new job is submitted, its record appears at `submitted_jobs/<semantic>/job.json`.
To monitor it, pass any unique substring of `job.json.job_name` (the timestamp is easiest)
to the helper. No registration step — the monitor discovers the job by scanning the batch
queues and CloudWatch streams for that substring.

## What the health signals mean (this is the analysis)

Report these explicitly:

1. **Topology formed** — `Running on: ... jax process 0/1 ... global devices 8` (single-node)
   or `0/2 ... global devices 16` (multi-node) on each node. For multi-node, missing/one-sided
   ⇒ `jax.distributed` didn't connect (coordinator/EFA/port).

2. **Data sharding (MULTI-NODE ONLY)** — `[data-dist] proc=0 per-shard image means=[...]`
   vs `proc=1`. The two lists **MUST DIFFER**; identical ⇒ the per-node shard split failed
   (both nodes read the same data — the frozen-`process_index` bug). Also check
   `[data-shard-split] jax_proc=N/2` (N/1 = the torch worker didn't see the process count).
   Single-node has no second proc to compare — skip this signal.

3. **Deadlock check (mainly MULTI-NODE)** — `==========Tentative run completed==========`
   should appear, then the **real** run logs `Step 0, 1, 2 …` after it. A multi-node run that
   prints tentative Step 0 then hangs with `Shutdown barrier ... 1/2 tasks reached` /
   `DEADLINE_EXCEEDED` is the classic desync (one process died/diverged; the real error is
   in the OTHER node's log — read both algo-1 and algo-2).

4. **Training progress** — `Step N: flow_loss=… grad_norm=…` advancing; `grad_norm` finite
   (~0.5). NaN/inf ⇒ bad data or (multi-node) collective. For RoboCasa also watch
   `progress_*` metrics (acc/mae/precision/recall/f1 depending on the progress mode).

5. **Checkpoints** — EMA-only runs (the default policy) save `params/` only (no
   `train_state/`), one self-contained `ocdbt.process_0`, no `*.orbax-checkpoint-tmp-*`.
   Verify in S3: `aws s3 ls <checkpoint_s3_dir>/<step>/` (path from `job.json`).

6. **Failure strings to grep**: `Traceback`, `Shutdown barrier`, `Aborted (core dumped)`,
   `non-trivial shardings for numpy inputs` (restore→sharded-jit bug), `wandb ... resume`,
   `final checkpoint upload failed` (the entrypoint fails the job if the checkpoint never
   reached S3 — this means training ran but the checkpoint may be lost).
