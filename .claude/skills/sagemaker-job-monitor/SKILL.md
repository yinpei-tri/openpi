---
name: sagemaker-job-monitor
description: >-
  Monitor and analyze an openpi SageMaker multi-node JAX training job (AWS Batch
  service job). Use when the user asks to check a job's status, watch it until it
  schedules/starts/finishes, pull its CloudWatch logs, or analyze the training
  health signals (data-sharding fingerprints, training-step progress, checkpoint
  saves, deadlock/failure detection). Triggers: "check the job", "is it running",
  "watch the job", "analyze the log", "did it start training", "monitor <job name>".
---

# SageMaker job monitor (openpi multi-node JAX)

These training jobs are submitted via `scripts/sagemaker/launch.py` as **AWS Batch
service jobs** (`SubmitServiceJob`) on `fss-*` queues. Key facts that trip people up:

- `aws sagemaker describe-training-job` returns **"resource not found"** — these are NOT
  classic SageMaker training jobs. Use `aws batch list-service-jobs --job-status <ST>`
  (service jobs require an explicit status filter; there is no "list all").
- Ambient `AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN` env vars **override** the SSO profile in
  boto3/CLI → `AccessDenied`. Always `unset` them first (the script does this).
- Logs go to CloudWatch group **`/aws/sagemaker/TrainingJobs`**, streams
  `AWSBatch<jobname-hash>/algo-N-<ts>` where **algo-1 = rank 0** (does S3 sync),
  **algo-2 = rank 1**. The `<hash>` is not the timestamp, so match streams by name prefix.
- CloudWatch ingestion lags real time by ~2–3 min; a job can be RUNNING while the newest
  log event is minutes old. Don't conclude "stuck" from a stale tail — poll again.

## Usage

Run the helper (job substring = any unique part of the name, usually the timestamp):

```bash
bash .claude/skills/sagemaker-job-monitor/scripts/monitor.sh <job_substring> <mode>
```

Modes:
- `status` (default) — current batch status (RUNNABLE / STARTING / RUNNING / SUCCEEDED / FAILED).
- `logs` — pull both nodes' logs and print the health-signal summary.
- `watch` — poll until the job leaves the queue and, once RUNNING, until real training
  steps advance (success) or a failure signal appears. Runs long; launch it with
  `run_in_background: true` and analyze when it completes.

Env overrides: `QUEUE` (skip autodetect), `PROFILE` (default `sagemaker`), `REGION`
(default `us-west-2`), `INTERVAL` (watch seconds, default 120), `MAXPOLL` (default 120).

For a long watch, prefer running the script as a **background Bash task** so you're
notified on completion rather than blocking, then run `logs` mode to analyze.

## What the health signals mean (this is the analysis)

Report these explicitly; they are the whole point of monitoring a multi-node run:

1. **Mesh formed** — `Running on: ... jax process 0/2 ... global devices 16` on both nodes.
   Missing/one-sided ⇒ `jax.distributed` didn't connect (coordinator/EFA/port).

2. **Data sharding — `[data-dist] proc=0 per-shard image means=[...]` vs `proc=1`.**
   The two lists **MUST DIFFER**. Identical values ⇒ the per-node shard split failed
   (both nodes read the same data — the frozen-`process_index` bug). This is the #1 thing
   to verify on any multi-node run. Also check `[data-shard-split] jax_proc=N/2` (not N/1;
   N/1 means the torch worker didn't see the distributed process count).

3. **Deadlock check** — `==========Tentative run completed==========` should appear, then
   the **real** run logs `Step 0, 1, 2 …` AFTER it. A run that prints the tentative Step 0
   then hangs with `Shutdown barrier ... 1/2 tasks reached` / `DEADLINE_EXCEEDED` is the
   classic multi-node desync (one process died/diverged; the barrier is the symptom, the
   real error is in the OTHER node's log — always read both algo-1 and algo-2).

4. **Training progress** — `Step N: flow_loss=… grad_norm=…` advancing; `grad_norm` finite
   and comparable to single-node (~0.5). NaN/inf ⇒ bad collective or data.

5. **Checkpoints** — for EMA-only fresh runs (the current policy, see the
   `multinode-libero-no-resume-policy` memory), each save writes `params/` only (no
   `train_state/`), one self-contained `ocdbt.process_0`, no `*.orbax-checkpoint-tmp-*`.
   Verify in S3 with `aws s3 ls <checkpoint_s3_uri>/<config>/<exp>/<step>/`.

6. **Failure strings to grep**: `Traceback`, `Shutdown barrier`, `Aborted (core dumped)`,
   `non-trivial shardings for numpy inputs` (restore→sharded-jit bug), `wandb ... resume`.

## S3 checkpoint location

`launch.py` prints it at submit; the layout is
`<output.s3_prefix>/<job_name>/<config>/<exp_name>/<step>/{params,assets[,train_state]}/`.
List steps: `aws s3 ls s3://.../<job_name>/<config>/<exp_name>/ --profile sagemaker`.
