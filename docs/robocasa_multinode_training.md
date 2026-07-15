# RoboCasa System1 — single-node vs multi-node JAX training

How to train `pi05_robocasa_system1` on SageMaker in **either** topology, and how the
multi-node JAX plumbing works. The multi-node mechanism (JAX distributed init, per-process
data split, `process_allgather` checkpointing) was first verified on a 2-node LIBERO job
(`pi05_libero_wds` on the `multinode-libero` branch); this doc ports it to RoboCasa.

**Single-node runs are unaffected by the multi-node code.** Every multi-node path is
gated on `jax.process_count() > 1` (or the presence of SageMaker's `resourceconfig.json`),
and the single-node data-shard order, checkpoint format, and launcher behavior are
byte-for-byte identical to before the multi-node work (verified — see "Correctness").

---

## Summary — the two topologies at a glance

| | **Single-node** | **Multi-node** |
| --- | --- | --- |
| launch config | `config_robocasa_jax.yaml` | `config_robocasa_multinode.yaml` |
| entrypoint | `sm_entrypoint_jax.sh` | `sm_entrypoint_robocasa_multinode_jax.sh` |
| `instance.count` | 1 | 2 (→ 16 H100s) |
| processes | 1 (owns 8 GPUs) | 1 **per node** (each owns its 8 GPUs) |
| mesh | `(1, 8)` — FSDP-8 | `(2, 8)` — data-parallel ×2, FSDP-8 within a node |
| `--batch-size` (global) | 128 | 256 (= 128/node) |
| `--fsdp-devices` | 8 | 8 (shard **within** a node; never 16) |
| data split | torch workers only | **by `jax.process_index()`** then torch workers |
| checkpoint | orbax device-sharded write | `process_allgather` → process-0 full write |
| S3 sync | this node | **rank 0 only** |
| wandb | this process | **process 0 only** (metrics already global) |
| resume | supported (`--resume`, needs `save_optimizer`) | **NOT supported** (EMA-only, fresh start) |

Both paths save **EMA-only checkpoints** by default (`save_optimizer=False`): the saved
`params/` item is the EMA (inference) weights, `train_state/` is omitted. That is the
checkpoint you deploy.

---

## How the JAX process model works

We train JAX only. The model runs as **one process per node**, and each process owns all
its local GPUs:

- **Single-node**: one process, 8 GPUs, mesh `(1, 8)` — pure FSDP-8.
- **Multi-node**: one process per node. The processes form ONE global device mesh
  (16 devices for 2×8) via
  `jax.distributed.initialize(coordinator_address, num_processes, process_id)`, which MUST
  run before any JAX device op (`train.py` calls it in `__main__`, before the tentative run
  brings up XLA). No-op single-node.

So the SageMaker estimator sets `distribution={}` (SageMaker launches the container once
per node and gets out of the way; the JAX process forms the mesh itself). `launch.py` gates
this on the `_jax.sh` entrypoint **suffix**, so both the single-node (`sm_entrypoint_jax.sh`)
and multi-node (`sm_entrypoint_robocasa_multinode_jax.sh`) entrypoints get the JAX treatment
(self-managed S3 checkpoint sync, no framework launcher).

---

## Run it

### Single-node (8×H100)
```bash
python scripts/sagemaker/launch.py --config scripts/sagemaker/config_robocasa_jax.yaml \
    --training.exp_name=run1 \
    [--training.extra_args="--fsdp-devices=8 --batch-size=128 --ema-decay=0.9999 --num-train-steps=400000 ..."] \
    [--dry-run]
```

### Multi-node (2×8 H100 = 16)
```bash
python scripts/sagemaker/launch.py --config scripts/sagemaker/config_robocasa_multinode.yaml \
    --training.exp_name=full400k_mn \
    [--dry-run]
```
The multinode config already sets `instance.count=2`, the multinode entrypoint, and
`--batch-size=256 --fsdp-devices=8`. `--dry-run` first to sanity-check the resolved job.

### Batch-size / FSDP rules (multi-node)
- `batch_size % jax.device_count() == 0` — `device_count` is **GLOBAL** (16 on 2 nodes).
- `jax.device_count() % fsdp_devices == 0` — `--fsdp-devices=8` → mesh `(2, 8)`: shard the
  model 8-way WITHIN each node (all-gather on NVLink) and data-parallel across the 2 nodes
  (only the gradient all-reduce crosses the inter-node link). **Do NOT set `--fsdp-devices=16`** —
  that would all-gather params over the slow inter-node link every layer.
- Global batch 256 = 128/node. (256 doesn't fit on ONE 8×80GB node with EMA — that is the
  whole reason for 2 nodes.)

---

## The pieces (what changed, and where)

| Component | Role |
| --- | --- |
| `src/openpi/training/distributed.py` | **New.** `maybe_init_distributed()` resolves coordinator/rank from `resourceconfig.json` (or `JAX_COORDINATOR_ADDRESS/JAX_NUM_PROCESSES/JAX_PROCESS_ID`) and calls `jax.distributed.initialize`. No-op single-node. `is_primary()` = process 0. |
| `scripts/train.py` | Calls `maybe_init_distributed()` in `__main__` **before** any JAX op (the tentative run brings up XLA). W&B init + `wandb.log` are process-0-only and non-fatal (`log_wandb`). A `[data-dist]` print proves the global batch is sharded + holds distinct data per GPU. |
| `src/openpi/training/robocasa_webdataset.py` | Splits shards by `jax.process_index()` (disjoint per node) then torch worker + reservoir shuffle. `(proc_idx, proc_cnt)` are **frozen at construction**. `debug_shard_split=True` logs the per-feeder slice. |
| `src/openpi/training/data_loader.py` | Allows multi-process loading for **iterable** WebDataset loaders (rejects map-style, which would replicate data). Each process feeds its LOCAL batch; `make_array_from_process_local_data` assembles the global sharded array. `local_batch_size = batch_size // process_count`. |
| `src/openpi/training/checkpoints.py` | `process_allgather` gathers the (replicated) model to host numpy before saving → orbax writes ONE self-contained `ocdbt.process_0/` (no shared FS needed). Gated on `process_count > 1`. |
| `scripts/sagemaker/sm_entrypoint_robocasa_multinode_jax.sh` | **New.** RoboCasa data handling (FastFile mount OR download-to-EBS + anchors) + `jax.distributed` coordinator port + **rank-0-only** S3 checkpoint sync. |
| `scripts/sagemaker/config_robocasa_multinode.yaml` | **New.** `instance.count=2`, multinode entrypoint, `--fsdp-devices=8 --batch-size=256`. |
| `scripts/sagemaker/launch.py` | `is_jax = endswith("_jax.sh")` (covers both JAX entrypoints); buildx fallback; managed-checkpoint sync OFF for all self-managing (JAX) entrypoints. |

---

## Correctness — why each part is safe

### Data must be split by process
Each JAX process assembles its portion of the global batch from its OWN local data, then
`make_array_from_process_local_data` stitches them into one globally-sharded array. This
is only correct if each process reads a **disjoint** shard set — otherwise nodes would
train on duplicated samples.

The RoboCasa loader guarantees this with a specific ordering (in `_worker_shards`):
1. shuffle all shards with a **FIXED seed** (same on every process, every epoch) and slice
   `shards[proc_idx::proc_cnt]` → each process owns a **PERMANENT disjoint** set;
2. reshuffle that fixed slice by `seed + epoch` for order diversity across passes;
3. slice by torch worker within the node.

The fixed-seed-**before**-slice ordering is what keeps ownership disjoint **even when hosts
drift to different epochs** (epochs advance independently per host; a naive
`shuffle(seed+epoch)` then slice would overlap under skew). `(proc_idx, proc_cnt)` are
frozen at construction because torch DataLoader workers are fresh interpreters where
`jax.process_index()` would return 0/1.

**Single-node** (`proc_cnt == 1`) takes a separate branch that is byte-for-byte the
pre-multinode code (fixed-seed slice `[0::1]` is identity; the multi-node branch is never
entered). Verified: the single-node shard order is identical to the historical ordering,
and the 2-proc split is disjoint + complete for every epoch including skew.

### Checkpoint without a shared filesystem
`save_state` calls `multihost_utils.process_allgather(params, tiled=True)` so every host
materializes the full (replicated across the data-parallel node axis) model as host numpy.
orbax then sees NON-distributed arrays and writes a single self-contained `ocdbt.process_0/`
from the primary host — no `ocdbt.process_1/`, no cross-node coordination, no manifest
referencing another node's shard. The entrypoint's **rank-0-only** `aws s3 sync` uploads
that complete checkpoint. With `save_optimizer=False` only `params` (EMA) is gathered, not
the ~20GB optimizer state.

Single-node: `process_count()==1`, so the gather block is **skipped** and orbax writes the
device-sharded arrays exactly as before. EMA-only save/reload round-trip is verified to
restore every param tensor (correct shapes + values) with `train_state/` correctly absent.

### wandb metrics are already global — logged once
`ptrain_step` is `jax.jit`'d with `out_shardings=(train_state, replicated)`. The metrics
dict is the replicated output, so every reduction inside the step (`jnp.mean(loss)`,
`optax.global_norm(grads)`, and every progress `jnp.mean`/`jnp.sum` — including the binary
head's `tp/fp/fn/tn` and the class/bin counters) is computed by XLA over the **whole global
batch** (all 16 devices) and replicated to process 0. So process 0 logs precision / recall /
F1 / MAE over the full global batch with **no extra gather**. Non-primary processes call
`wandb.init(disabled)` (no N duplicate runs, no wandb barrier that could deadlock a collective).

---

## What to watch in the logs (both nodes emit these)

- `Distributed: initializing jax.distributed | coordinator=<algo-1>:12355 num_processes=2 process_id=0/1`
  then `initialized ... global_devices=16` — the mesh formed across both hosts.
- `Running on: <host> | jax process 0/2 | local devices 8 | global devices 16`.
- `[data-dist] proc=0/2 ... n_addressable_shards=8 devices=16 mesh=(2, 8)` on each process,
  and `per-shard image means=[...]` — the fingerprints **MUST differ between proc 0 and
  proc 1**; identical values mean both nodes loaded duplicate data (the shard split lost the
  process identity).
- `[data-shard-split] jax_proc=0/2 torch_worker=... -> N/1055 shards` (enable via
  `--data.debug-shard-split` / the loader's `debug_shard_split`). `jax_proc=0/1` here
  (instead of 0/2, 1/2) is the bug signature: the worker didn't see the process count.
- `grad_norm` per step — finite and comparable to single-node. NaN / wildly different ⇒
  the cross-node collective is misconfigured.
- Checkpoint sync lines on **rank 0 only** (`algo-1`); the other host logs
  "Not rank 0: skipping S3 checkpoint sync."

---

## Known friction point (untested boundary)

The cross-host mesh formation (coordinator handshake over EFA, port `SM_MASTER_PORT`,
default 12355) can only be confirmed by a real 2-node submission — it can't be tested on a
single-node box. If the job hangs at `jax.distributed.initialize`, check: coordinator
address/port reachability between hosts (EFA, security group, `SM_MASTER_PORT`),
`resourceconfig.json` parsing (hosts list, `current_host`), and `NCCL_DEBUG=INFO` output.
Everything else (loader split, sharding, distributed-init wiring, rank-0 sync, launcher,
batch math, checkpoint round-trip) is verified single-node.

## Not supported on multi-node

- **Resume.** The trainer + loader reject a multi-node resume (`resume_shards_consumed>0`
  with `process_count>1` raises). Multi-node is EMA-only, fresh-start. Resume stays a
  single-node feature (and needs `save_optimizer=True`).
- **Map-style datasets.** Only iterable WebDataset loaders (RoboCasa/LIBERO) split by
  process; map-style loaders raise under `process_count>1` (they would replicate data).
