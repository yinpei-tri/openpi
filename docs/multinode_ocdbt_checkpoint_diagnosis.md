# Multi-host orbax checkpoint failure — root cause & fix

Diagnosis of the 2-node SageMaker JAX run (`pi05_libero_wds`, `multinode-libero` branch)
whose S3 checkpoints are missing `params/ocdbt.process_1/` and fail to restore.

All code cited below is read from the branch via `git show origin/multinode-libero:<path>`.

---

## Symptom (ground truth)

Checkpoints at
`s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints/yinpeidai-pi05-libero-wds-multinode-2026-07-12-17-46-17/pi05_libero_wds/multinode_full/{10000,20000,29999}/`
contain only node 0's OCDBT shard:

- present: `params/ocdbt.process_0/`, `params/d/`, `params/manifest.ocdbt`,
  `params/_METADATA`, `params/_sharding`, `assets/.../norm_stats.json`,
  `_CHECKPOINT_METADATA` (~6.2 GB total, correct for ~3B bf16 params).
- absent: `params/ocdbt.process_1/`.

`restore_params(...)` on the byte-complete download fails with:
`ValueError: NOT_FOUND: chunk {15} stored at ".../encoder_norm.bias.value/15" in OCDBT
database ... is missing` — process 1's chunks are simply not in the stored data.
Download is byte-complete (17/17 objects) → process 1's data never reached S3.

---

## Summary

`params/ocdbt.process_1/` is missing because the multi-node JAX entrypoint writes each
node's orbax output to that node's **own local** `/opt/ml/checkpoints`, then relies on
SageMaker's managed checkpoint sync to *union* both nodes' local dirs into one S3
prefix. That union did not happen for node 1 (`algo-2`). The design's core assumption —
"managed `/opt/ml/checkpoints` acts as the shared store orbax multi-host needs" — is
stated verbatim in the code and is the bug.

---

## Confirmed from code

### 1. Orbax writes to a per-node local path, not a shared FS

- `scripts/sagemaker/sm_entrypoint_libero_jax.sh:51` — `CKPT_DIR="/opt/ml/checkpoints"`;
  `:74` passes it as `--checkpoint-base-dir`.
- `src/openpi/training/checkpoints.py:47` resolves that path
  (`epath.Path(checkpoint_dir).resolve()`); `:64-76` builds a vanilla
  `ocp.CheckpointManager` rooted there — no OCDBT remote/coordinator config, no
  shared-store option. So on each node, `jax.process_index()`'s `ArrayHandler` writes its
  `ocdbt.process_<id>/` shard into *that node's local disk*.
- With 2 nodes: process 0's shard lands on `algo-1:/opt/ml/checkpoints/...`, process 1's
  on `algo-2:/opt/ml/checkpoints/...` — two different physical disks.

### 2. The design explicitly depends on SageMaker managed sync to merge them (the flawed assumption)

- `checkpoints.py:28-31` (`_disable_array_metadata_store` docstring): *"On SageMaker there
  is no shared POSIX FS across nodes... Each host still writes its own param shards; the
  shared /opt/ml/checkpoints (S3-synced) is the common store."*
- `sm_entrypoint_libero_jax.sh:44-50`: *"we use SageMaker's MANAGED /opt/ml/checkpoints —
  both nodes' copies sync bidirectionally to the same S3 location, acting as the common
  store."*
- `scripts/sagemaker/launch.py:304-309, 346-347`: for the `*_jax.sh` multinode entrypoint,
  `self_manages_ckpt=False`, so managed sync stays ON with
  `checkpoint_local_path=/opt/ml/checkpoints` and `checkpoint_s3_uri=<one prefix>` (`:297`)
  shared by both nodes. There is **no** self-managed `aws s3 sync` in the multinode
  entrypoint (contrast the single-node `sm_entrypoint_jax.sh`, which self-syncs a plain
  EBS dir with a rank-0 guard).

### 3. Finalize does not verify process 1's shard is present

The `_disable_array_metadata_store()` change (`checkpoints.py:20-41`) deliberately removes
orbax's cross-host base-dir wait so non-primary hosts don't block on a dir the primary
created on a *different* disk. Consequence: process 0 writes `manifest.ocdbt`, `d/`,
`_METADATA`, `_sharding`, `_CHECKPOINT_METADATA` and finalizes/commits (renames tmp→final)
based purely on **its own local view**, never checking that `ocdbt.process_1/` exists in
the store. So process 0 "successfully commits" a step whose chunk data owned by process 1
is only on node 1's disk. This matches the observed artifact set (all primary metadata +
`ocdbt.process_0/`, no `ocdbt.process_1/`) and the restore error. `assets/norm_stats.json`
being present also fits: `CallbackHandler.save` writes it `if jax.process_index()==0`, i.e.
from node 0 — the node that reached S3.

**Conclusion (code-confirmed):** the layout is a standard multi-process OCDBT that is only
complete when every `ocdbt.process_<id>/` co-exists in one directory. Nothing in the code
path ever places node 1's shard into the S3 prefix.

---

## Must verify empirically (cannot be settled from repo code)

The *exact* reason node 1's local dir didn't reach S3 is a property of SageMaker's managed
checkpoint sync, not of this repo:

- whether managed checkpoint sync runs on **all** instances or **master-only** for this job
  type, and
- whether concurrent per-node syncs to the *same* `checkpoint_s3_uri` race / last-writer-win
  / prune each other.

Empirical check on the next run: inspect CloudWatch logs for `algo-2` for any
checkpoint-upload activity, and/or exec onto `algo-2` and confirm
`/opt/ml/checkpoints/.../ocdbt.process_1/` exists locally at save time (it will) but never
appears in S3. That distinguishes "master-only sync" from "both synced but raced." Either
way the code-level fix below removes the dependency entirely.

---

## Fix — options evaluated

- **(a) Real shared FS (FSx for Lustre / EFS on both nodes) + one rank-0 sync.**
  Orbax-idiomatic (multi-host orbax is *designed* for a shared FS). Correct/robust for any
  mesh/optimizer config. Cost: standing up FSx, VPC/subnet/SG wiring, `FileSystemInput`
  mount — real infra for a 2-node smoke.
- **(b) Per-node self-managed `aws s3 sync` on ALL nodes.** Directly patches the confirmed
  gap, no new infra. But reintroduces a multi-writer S3 union: per-node syncs must **never**
  use `--delete` (robocasa `final_sync` does — each node would delete the other's
  `ocdbt.process_N/`), and must be sequenced after `checkpoint_manager.wait_until_finished()`.
  Workable but race-prone.
- **(c) Native orbax/tensorstore S3 backend.** All processes write directly to one `s3://`
  kvstore. Elegant, but orbax's atomic finalize relies on directory-rename semantics S3
  lacks; least battle-tested. Risky.
- **(d) Single-host (process-0) save of fully-replicated params.** Emit only
  `ocdbt.process_0/`, which the already-proven rank-0 self-managed sync uploads completely.
  Eliminates the shared-FS assumption instead of working around it.

**Key fact that makes (d) especially attractive here:** with `--fsdp-devices=8` on 16
global devices, `sharding.make_mesh` (`src/openpi/training/sharding.py:22`) builds a
`(2, 8)` mesh — FSDP shards params 8-way *within* each node, and the two nodes are
**data-parallel replicas**. So params are already fully replicated across the two nodes;
process 0 alone holds a complete copy. Orbax only split the *write duty* across processes
for dedup; the data is redundant. And `save_optimizer=False` (config default at
`config.py:724`, not overridden in `config_libero_multinode.yaml`) means only ~6 GB of
params is saved. A process-0-only save loses nothing.

---

## Recommendation: (d), with (a) as the future-proof alternative

Recommend **(d)**. It reuses the sync path already proven single-node, adds no AWS infra,
and removes the entire class of "union two nodes' dirs in S3 without them deleting each
other" problems that (b) and the current design suffer from. Choose (a) later only if you
move to true full-sharding (`--fsdp-devices=16`, params not node-replicated) *and* start
checkpointing optimizer state at a size where gathering to one host is undesirable —
neither applies to the current run.

### Concrete changes implied by (d)

1. **Trainer save path (`checkpoints.py:save_state`)**: before `checkpoint_manager.save`,
   replicate the saved pytree to a fully-replicated sharding (a
   `jax.jit(lambda x: x, out_shardings=NamedSharding(mesh, PartitionSpec()))`, or
   `jax.experimental.multihost_utils.process_allgather` → host numpy). Once arrays are
   host/replicated, orbax's `ArrayHandler` writes them only from the primary, producing a
   single `ocdbt.process_0/`. (Under fsdp=8 this is effectively a no-op since data is
   already node-replicated; it just guarantees process 0 is byte-complete regardless of
   future mesh choices.) The exact orbax knob (host-numpy pytree vs. single-replica
   handler) is an implementation choice; the mechanism is "make the save single-writer."
2. **Entrypoint (`sm_entrypoint_libero_jax.sh`)**: switch to the single-node model — point
   `--checkpoint-base-dir` at a plain EBS dir (e.g. `/opt/ml/local_checkpoints`, as
   `sm_entrypoint_jax.sh` does) and run the **rank-0-only** periodic + final `aws s3 sync`
   (the proven robocasa block). Since only `ocdbt.process_0/` exists, rank-0 upload is
   complete.
3. **`launch.py`**: flip the multinode libero entrypoint to the self-managed branch — set
   `self_manages_ckpt=True` for it (managed sync OFF, `CHECKPOINT_S3_URI` drives the
   entrypoint's own sync). Currently `:309` restricts `self_manages_ckpt` to exactly
   `sm_entrypoint_jax.sh`; extend it to the libero entrypoint too.
4. The `_disable_array_metadata_store()` hack can stay (harmless) or be removed; with a
   single writer the cross-host coordination it worked around no longer occurs.

### Validation — minimal 2-node save+restore smoke before any long run

- Submit a 2-node job with `--num-train-steps=2 --save-interval=1 --keep-period=1` on
  `pi05_libero_wds` (or a tiny debug config) so a checkpoint is written under the
  multi-host mesh.
- After exit, `aws s3 ls --recursive` the step dir; assert the primary metadata is present
  and there is exactly one `ocdbt.process_0/` and **no** `ocdbt.process_1/` (proves
  single-writer save).
- Download the step dir byte-complete and run `openpi.models.model.restore_params(...)` on
  a CPU box — must succeed with no `NOT_FOUND: chunk ... missing`. (The current failure
  reproduces exactly this call, so a clean restore is the definitive pass.)
- Optionally diff a few restored tensors against a single-node save of the same init for
  parity.

Only after that restore passes should a 30k-step run be trusted.

---

## Key files (all via `git show origin/multinode-libero:<path>`)

- `src/openpi/training/checkpoints.py` — `_disable_array_metadata_store` (20-41),
  `initialize_checkpoint_dir` (43-86, esp. 47 / 64-76), `save_state` (89-121),
  `CallbackHandler.save` (process-0 guard).
- `scripts/sagemaker/sm_entrypoint_libero_jax.sh` — `CKPT_DIR=/opt/ml/checkpoints` (51),
  managed-store rationale (44-50), `--checkpoint-base-dir` (74).
- `scripts/sagemaker/sm_entrypoint_jax.sh` — the working single-node self-managed
  `aws s3 sync` pattern to reuse (periodic + `final_sync`, with the `--delete` caveat and
  rank-0 guard).
- `scripts/sagemaker/launch.py` — `is_jax` / `self_manages_ckpt` (308-309), managed-sync
  wiring (346-347), shared `checkpoint_s3_uri` (297).
- `scripts/sagemaker/config_libero_multinode.yaml` — `fsdp-devices=8`, batch 256,
  `checkpoint_local_path=/opt/ml/checkpoints`.
- `src/openpi/training/sharding.py:17-23` — `(2,8)` mesh proving params are node-replicated
  (basis for (d)).
- `src/openpi/training/distributed.py` — process_id / `is_primary` resolution.
