# Multi-node JAX training (LIBERO WebDataset example)

How to train `pi05_libero_wds` across **2 SageMaker nodes** (16 GPUs, global batch 256),
and how the multi-node JAX plumbing works. Single-node runs are unaffected — every
multi-node code path is gated on `jax.process_count() > 1` / the presence of SageMaker's
`resourceconfig.json`.

## TL;DR — submit the 2-node smoke

```bash
git checkout multinode-libero
GIT_LFS_SKIP_SMUDGE=1 uv sync --group sagemaker
aws sso login --profile sagemaker
# copy scripts/sagemaker/secrets.env over (gitignored: WANDB_API_KEY + HF_TOKEN)

uv run --group sagemaker scripts/sagemaker/launch.py \
    --config scripts/sagemaker/config_libero_multinode.yaml
# --dry-run first to sanity-check; must build the image on a BuildKit-capable machine.
```

This queues a 2× p5 (H100) job: `instance.count=2`, `--fsdp-devices=8 --batch-size=256`.
LIBERO shards stream from `s3://.../preprocessed/libero/shards`.

## How JAX multi-node differs from PyTorch

- **PyTorch**: one process PER GPU (`torchrun` spawns 8/node); DDP/FSDP wire them via NCCL.
- **JAX (this repo)**: one process PER NODE — each process owns all local GPUs. The
  processes form ONE global device mesh (16 devices for 2×8) via
  `jax.distributed.initialize(coordinator_address, num_processes, process_id)`, which
  MUST run before any JAX device op. That's the only thing single-node was missing.

So the JAX SageMaker estimator sets `distribution={}` (NOT `torch_distributed`) even for
multi-node — `launch.py` gates this on the `*_jax.sh` entrypoint suffix.

## The pieces

| Component | Role |
| --- | --- |
| `src/openpi/training/distributed.py` | `maybe_init_distributed()` — resolves coordinator/rank from `resourceconfig.json` (or `JAX_COORDINATOR_ADDRESS/JAX_NUM_PROCESSES/JAX_PROCESS_ID` env) and calls `jax.distributed.initialize`. No-op single-node. `is_primary()` = process 0. |
| `scripts/train.py` | Calls it before any JAX op; logs `process_index/count` + device counts; `[data-dist]` print proves the global batch is sharded across all devices. |
| `src/openpi/training/libero_webdataset.py` | Streaming LIBERO loader; **splits shards by `jax.process_index()` then torch worker** (disjoint per node) + reservoir shuffle. `[data-shard-split]` debug print per feeder. |
| `src/openpi/training/data_loader.py` | Multi-process allowed for iterable WebDataset loaders; each process feeds its LOCAL batch, `make_array_from_process_local_data` assembles the global sharded array. |
| `scripts/sagemaker/sm_entrypoint_libero_jax.sh` | One process/node; streams shards from S3 (no data channel); **rank-0-only** S3 checkpoint sync; `SM_MASTER_PORT` coordinator port. |
| `scripts/sagemaker/config_libero_multinode.yaml` | `instance.count=2`, libero entrypoint, `--fsdp-devices=8 --batch-size=256`. |

## Why data must be split by process (correctness)

Each JAX process assembles its portion of the global batch from its OWN local data, then
`make_array_from_process_local_data` stitches them into one globally-sharded array. This
is only correct if each process reads a **disjoint** data shard — otherwise nodes would
train on duplicated samples. The LIBERO loader guarantees this: it shuffles all shards
with the SAME seed on every process (so the order agrees), then takes
`shards[process_index::process_count]`. Verified disjoint + complete for 2 processes.

Map-style datasets do NOT split by process, so `data_loader.py` still rejects them under
`process_count > 1` — only iterable WebDataset loaders are multi-node-safe.

## Batch-size / FSDP rules

- `batch_size % jax.device_count() == 0` — device_count is GLOBAL (16 on 2 nodes).
- `jax.device_count() % fsdp_devices == 0` — `--fsdp-devices=8` → mesh `(2, 8)`: shard the
  model 8-way WITHIN each node (all-gather on NVLink) and data-parallel across the 2 nodes
  (only the gradient all-reduce crosses the inter-node link). This is cheaper than 16-way
  cross-node FSDP, which would all-gather params over the slow inter-node link every layer.
- Global batch 256 = 128/node. (256 does not fit on ONE 8×80GB node with EMA — that was
  the whole reason for 2 nodes; see the RoboCasa memory notes.)

## What to watch in the logs (both nodes emit these)

- `Distributed: initializing jax.distributed | coordinator=<algo-1>:12355 num_processes=2 process_id=0/1`
  then `initialized ... global_devices=16` — the mesh formed across both hosts.
- `[data-dist] proc=0/2 ... n_addressable_shards=8 ... mesh={batch:2, fsdp:8}` on each
  process — the global batch is sharded across all 16 devices (8 addressable per node).
  The per-shard `image means` fingerprint MUST differ between proc=0 and proc=1; identical
  values mean both nodes loaded duplicate data (the shard split lost the process identity).
- `[data-shard-split] jax_proc=0/2 torch_worker=... -> N/1055 shards` — each feeder's
  disjoint shard slice. `jax_proc=0/1` here (instead of 0/2 and 1/2) is the bug signature:
  the worker subprocess didn't see the distributed process count.
- `grad_norm` per step — should be finite and comparable to single-node (~0.4–0.8). NaN
  or wildly different ⇒ the cross-node collective is misconfigured.
- Checkpoint sync lines appear on **rank 0 only** (`algo-1`); the other host logs
  "Not rank 0: skipping S3 checkpoint sync."

## Known friction point (untested boundary)

The cross-host mesh formation (coordinator handshake over EFA, port `SM_MASTER_PORT`) is
the one thing that can only be confirmed by a real 2-node submission — it can't be tested
on a single-node box. If the job hangs at `jax.distributed.initialize`, look at:
- coordinator address / port reachability between the SageMaker hosts (EFA, security
  group, `SM_MASTER_PORT`),
- `resourceconfig.json` parsing (hosts list, current_host),
- set `NCCL_DEBUG=INFO` (already on) to see the collective init.

Everything up to that boundary (loader, sharding, distributed-init wiring, rank-0 sync,
launcher, batch math) is verified single-node.

## Single-node smoke (this box, for reference)

```bash
# stage a few shards locally, then:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_libero_wds \
    --exp-name=libero_wds_smoke \
    --data.shards=/path/to/local/libero/shards \
    --batch-size=128 --fsdp-devices=8 \
    --num-train-steps=6 --save-interval=1000 --log-interval=1 \
    --no-wandb-enabled --overwrite
```
Confirms the loader + transforms + distributed no-op path (logs `process 0/1`, 8 devices).
