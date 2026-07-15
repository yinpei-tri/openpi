# System1 (RoboCasa pi0.5) — training setup

Subgoal-conditioned **pi0.5** + a **progress head**, trained on RoboCasa WebDataset
shards. This doc covers the data, the configs, and how to run training in each
environment: bare-metal, local Docker, an 8×A100 box (S3 data), and SageMaker.

> **Multi-node training:** the SageMaker section below is **single-node** (one 8×GPU node).
> For **multi-node** (2 nodes / 16 GPUs, global batch 256) — how to launch it, how the JAX
> process/data/checkpoint plumbing works, and the single-vs-multi-node comparison table —
> see **[docs/robocasa_multinode_training.md](robocasa_multinode_training.md)**. Single-node
> runs are unaffected by the multi-node code (every multi-node path is gated on
> `jax.process_count() > 1`).

## TL;DR — run on the 8×A100 box, streaming data from S3

```bash
cd <openpi-repo>
# 1. AWS creds in env (so the WebDataset loader's boto3 can read s3://)
eval "$(aws --profile sagemaker configure export-credentials --format env)"
export AWS_DEFAULT_REGION=us-west-2
# 2. point the config at the S3 shards
export ROBOCASA_SHARDS_DIR=s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/shards
# 3. norm stats must exist locally for this config (see "Norm stats" below):
#    assets/pi05_robocasa_system1_noanchor/robocasa_system1/norm_stats.json
# 4. train — 8 GPUs, FSDP across all 8, batch divisible by 8
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  .venv/bin/python scripts/train.py pi05_robocasa_system1_noanchor \
    --exp-name=a100_run1 \
    --batch-size=64 --fsdp-devices=8 --ema-decay=None \
    --num-train-steps=50000 --save-interval=5000 --overwrite
```

A100 80GB has far more memory than the dev A6000s, so you have headroom: larger
`--batch-size`, and you can re-enable EMA / anchors (see "Memory & knobs").

## The data

- **Format**: per-frame, globally-shuffled WebDataset tar shards + a deduped anchor
  store. Built by RoboAnnotator `producers/preprocess_robocasa_to_tar.py`. Full field
  schema is in the dataset's `SCHEMA.md`.
- **Local**: `/home/yinpeidai/RoboAnnotator/data/robocasa_system1/` (16GB, 594 shards,
  303,755 frames, 350 episodes).
- **S3**: `s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/`
  (`shards/`, `anchors/`, `meta.json`, `SCHEMA.md`).
- The loader (`openpi/training/robocasa_webdataset.py`) reads `ROBOCASA_SHARDS_DIR`
  (local dir OR `s3://`); it derives the `anchors/` sibling automatically. For S3 it
  uses boto3 (a core dep) and reads creds from the standard AWS env vars.

## The configs (`src/openpi/training/config.py`)

| config | use | images | progress head | EMA |
| --- | --- | --- | --- | --- |
| `pi05_robocasa_system1` | full design | 3 current + 3 anchor (6) | yes (shallow_transformer) | 0.999 |
| `pi05_robocasa_system1_noanchor` | **fast baseline** (anchor=none) | 3 current | yes | None |
| `pi05_robocasa_system1_debug` | tiny smoke | 3 current | yes | None |

All: pi05, `action_horizon=20`, **14-d lean state** (eef_pos3 + eef_rot_6D6 +
gripper_width1 + base_rel_xy2 + base_yaw_sincos2), **11-d lean action** (sim 12-d
minus the always-0 torso) padded to 32, quantile norm, fixed LR 5e-5, weight-loader =
released `pi05_base` (progress head + anchor role-emb kept fresh). Shards store BOTH
the lean arrays (training default) AND the full raw state(16)/action(12) + frame-0
`raw_state_ref`, so the lean recipe can be re-derived at load without re-sharding.

`ROBOCASA_SHARDS_DIR` env overrides the shards path for ALL three (default = the local
host path), so the same config runs bare-metal / Docker / A100 / SageMaker.

### Default data recipe + ablations
The two non-debug configs default to a MIXED subgoal recipe:
`subgoal_level=mixed`, `p_milestone=0.5` (milestone vs child), `p_detail=0.5`
(terse vs detailed phrasing), `subgoal_action_pad=subtask` (action zero-pads at the
chosen subgoal boundary so System1 learns to stop/settle for System2's hand-off).

Ablations are CLI overrides on a single config (no per-ablation config files). The
trainer AUTO-APPENDS a deterministic settings tag to `--exp_name` (so checkpoint dirs
+ wandb names never collide and are self-describing; `--resume` still works since the
tag is deterministic). E.g. `--exp_name run1` becomes
`run1__lvl-mixed_pm0.5_pd0.5_pad-subtask_anchor_prog-shallow_transformer`.

Override flags (verified):

| ablation | flag |
| --- | --- |
| subgoal level | `--data.subgoal-level {milestone,child,mixed}` (+ `--data.p-milestone FLOAT`) |
| phrasing | `--data.p-detail FLOAT` (0=terse only, 0.5=mixed) |
| action pad | `--data.subgoal-action-pad {subtask,episode}` |
| task goal in prompt | `--data.include-task-goal` (prepend whole-task goal → `Task: <task>; Current Subgoal: <subgoal>, State: …`; default off to keep System1 composable). Both task + subgoal are lowercased. |
| anchor state in prompt | `--data.include-anchor-state` (append subgoal-start lean state → state 14→28, rendered as a second labeled prompt segment `… State: <current>; Initial State: <anchor>;` so the ints carry the proprioceptive before/after delta; pairs with anchors). Recompute norm stats after toggling. |
| progress readout | `--model.progress-readout {shallow_transformer,mean_pool,prefix_token}` |
| progress head off | `--model.no-use-progress-head` |
| progress VLM-insulation | `--model.progress-stop-gradient` (insulate the VLM; default = OFF / gradients flow). Off is the default because the prefix's final-layer output — what the head reads — gets NO action-loss gradient (the action expert reads the prefix only via attention K/V, not its last-layer output), so insulating it would leave the readout representation at pretrained init. Letting progress gradient flow (weight 0.5) trains it. |
| progress loss weight | `--model.progress-loss-weight FLOAT` (default 0.5) |
| progress head width | `--model.progress-hidden INT` (default 512; head_dim = hidden/num_heads) |

For the ANCHOR ablation use the dedicated configs (anchor is set in BOTH model and
data, which must agree): `pi05_robocasa_system1` (anchor on, 6-img) vs
`pi05_robocasa_system1_noanchor` (anchor off, 3-img) — don't flip `--*.use-anchor-images`
on one config.

## Parameter reference (all knobs + defaults)

Single source of truth for everything we tuned. Defaults are the resolved values on the
non-debug configs (`pi05_robocasa_system1_noanchor`, and `_system1` which additionally
sets `use_anchor_images=True`). "Re-shard?" = whether changing it needs the producer to
re-run (most knobs are load-time / loss-time and do NOT).

### Data recipe (`--data.*`, `RoboCasaDataConfig`)
| param | default | meaning | re-shard? |
| --- | --- | --- | --- |
| `subgoal_level` | `mixed` | condition on milestone / child / mixed per sample | no |
| `p_milestone` | `0.5` | P(milestone vs child) when `mixed` | no |
| `p_detail` | `0.5` | P(detailed phrasing vs terse subgoal) | no |
| `subgoal_action_pad` | `subtask` | settle-pad action chunk at subgoal end (`subtask`) vs episode end (`episode`) | no |
| `include_base_pose` | `True` | include relative base pose (x/y + yaw sin/cos, 4-d) in lean state → 14-d vs 10-d | no¹ |
| `include_task_goal` | `True` | prepend whole-task goal → `Task: <goal>; Current Subgoal: <sg>` (both lowercased) | no |
| `include_anchor_state` | `True` | append anchor (subgoal-start) state → `Initial State: …; Current State: …` (state 14→28) | no² |
| `include_metadata` | `True` | emit the `Scope: milestone\|step` conditioning line | no |
| `recompute_lean_from_raw` | `True` | recompute lean from raw at load (same path as inference) + assert == baked lean (drift guard) | no |
| `use_anchor_images` | `False`³ | add 3 anchor camera views (before/after for progress) | no |
| `shuffle_buffer` / `shuffle_initial` | `16000` / `1000` | reservoir shuffle on top of the producer's global shard shuffle | no |

¹ load-time recompute supports it, but norm stats are dimension-specific — recompute stats.
² needs the `anchor_state_*` / `raw_anchor_state_*` arrays (present since the latest shards);
  changes state width 14→28 so **recompute norm stats** (auto-tiled ×2, see Norm stats).
³ `True` on `pi05_robocasa_system1` (the 6-image anchor config).

### Model (`--model.*`, `Pi0Config`)
| param | default | meaning |
| --- | --- | --- |
| `action_horizon` | `20` | 1 s @ 20 fps action chunk |
| `max_token_len` | `256` | worst-case prompt (task+detail+anchor-state+scope) ≈ 220 tok; 256 leaves headroom |
| `use_progress_head` | `True` | subgoal-completion state-value head (System2 reads it) |
| `progress_readout` | `shallow_transformer` | `[PROG]` token attends prefix; vs `mean_pool` / `prefix_token` |
| `progress_stop_gradient` | `False` | let progress loss train the VLM (prefix final-layer gets no action-loss grad, so insulating leaves it stale) |
| `progress_loss_weight` | `0.5` | total = flow_loss + 0.5·progress_huber |
| `progress_k` | `1.0` | target = `frac**k`; 1 = linear (uniform signal, no early dead zone) |
| `progress_hidden` / `_num_layers` / `_num_heads` | `512` / `2` / `8` | shallow-transformer head shape (head_dim 64) |

### Prompt format (the literal text the model conditions on)
```
Task: <task goal>; Current Subgoal: <subgoal>          # task goal omitted if include_task_goal=False
Scope: milestone|step                                  # only if include_metadata
Initial State: <14 anchor ints>; Current State: <14 ints>;   # "State: <14 ints>;" if include_anchor_state=False
Action:
```
Metadata is append-only (`Scope` now; `Quality`/`Mistake` reserved for future offline-RL
data). The `\n` separators are preserved for RoboCasa only (`preserve_newlines`); all
other datasets keep stock pi05 `, State:`.

### Training schedule (`TrainConfig`)
| param | default | note |
| --- | --- | --- |
| LR | `5e-5` fixed | warmup 500, peak == decay (no schedule) |
| `num_train_steps` | `20_000` | ≈4 epochs over 303,755 frames @ bs 64 (debug config uses more) |
| `batch_size` | per-run | divisible by visible GPU count |
| `ema_decay` | `0.999` | drop (`--ema-decay=None`) to fit small GPUs |
| `progress LR` | same as backbone | deliberately NOT separate (decided against a `progress_lr_mult`) |

## Key training args

- `--fsdp-devices=N` — shard the ~3B-param model across N GPUs. **Must divide the
  visible GPU count**; with N == #GPUs there's 1 data-parallel group (pure FSDP).
- `--batch-size=B` — **must be divisible by the visible GPU count**. On A100/H100 with
  NVLink, a bigger batch ≈ proportionally more throughput — use a large batch.
- `--ema-decay=None` — disables EMA (frees a full param-copy of memory). Needed to fit
  the dev A6000s; on A100 80GB you can leave EMA on (drop this flag).
- `--num-train-steps`, `--save-interval`, `--overwrite`, `--resume`.
- `--exp-name=` — names the checkpoint dir: `checkpoints/<config>/<exp-name>/<step>/`.

### Epoch math (303,755 frames)
`steps/epoch = 303755 / batch_size`. batch 64 → ~4,747/epoch → 4 epochs ≈ 19k steps;
batch 128 → ~2,373/epoch → 4 epochs ≈ 9.5k steps.

## Memory & knobs (dev A6000 48GB vs A100 80GB)

- Dev box (3×A6000, no NVLink) needed `--fsdp-devices=2 --ema-decay=None` + a 3-image
  config to fit, and fsdp=2 was much faster than fsdp=3 (topology: fsdp=3 dragged the
  all-gather over a slow link). Profiling: step time was compute-bound (data_wait
  0.03s), batch- and image-count-independent — pure FSDP comms + remat double-forward.
- **8×A100 80GB**: ~1.7× the memory/GPU + NVLink. Run `--fsdp-devices=8`, large
  `--batch-size` (e.g. 128/256, divisible by 8), keep EMA, and you can use the full
  6-image `pi05_robocasa_system1` (anchor) config. Expect far better throughput.
- Gradient checkpointing (`nn.remat(nothing_saveable)`) is always on in gemma — roughly
  doubles the forward (recompute in backward). Intrinsic; fine.

## Norm stats

Recipe-dependent (15-dim lean state + episode-padded action quantiles), stored at
`assets/<config>/robocasa_system1/norm_stats.json`. Already present for the three
configs. To recompute (e.g. if you change the state recipe):
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compute_norm_stats.py \
    --config-name pi05_robocasa_system1_noanchor --max-frames 60000
```
(Works with `ROBOCASA_SHARDS_DIR` pointing at local OR s3.)

## Local Docker (matches the SageMaker container)

```bash
eval "$(aws --profile sagemaker configure export-credentials --format env)"
HOST_UID=$(id -u) HOST_GID=$(id -g) \
NVIDIA_VISIBLE_DEVICES=0,1 \
ROBOCASA_SHARDS_DIR=s3://tri-ml-datasets-uw2/yinpeidai/data/robocasa_system1/shards \
AWS_DEFAULT_REGION=us-west-2 \
TRAIN_CONFIG=pi05_robocasa_system1_noanchor EXP_NAME=docker_run \
TRAIN_ARGS="--batch-size=2 --fsdp-devices=2 --ema-decay=None --num-train-steps=200 --overwrite" \
WANDB_MODE=offline \
  docker compose -f scripts/docker/train.compose.yml up
```
The compose bind-mounts local data at `/data/robocasa_system1` (used when
`ROBOCASA_SHARDS_DIR` is left at its local default) and forwards `AWS_*` for s3://.
Code is bind-mounted at `/app` — source edits need no rebuild; only dep/Dockerfile
changes need `docker compose ... build`.

## SageMaker (8×H100 p5)

```bash
python scripts/sagemaker/launch.py --config scripts/sagemaker/config_robocasa_jax.yaml \
    --training.exp_name=run1 \
    [--training.extra_args="--fsdp-devices=8 --batch-size=64 --ema-decay=None --num-train-steps=50000 --save-interval=5000"] \
    [--dry-run]
```
Builds + pushes the JAX image (`image.sm_entrypoint=sm_entrypoint_jax.sh`), mounts two
FastFile channels (`robocasa` dataset, `base_ckpt` JAX orbax), runs the single-process
JAX trainer. Checkpoints sync continuously to
`s3://.../openpi/checkpoints/<job_name>/<config>/<exp-name>/<step>/`. Download with the
`aws s3 sync` command launch.py prints.

## Checkpoints

`checkpoints/<config>/<exp-name>/<step>/` (local); on SageMaker `/opt/ml/checkpoints`
syncs to the S3 path above. Each holds orbax `params/` + `train_state/` + `assets/`
(norm stats). Resume with `--resume`.

## Phase 4 (closed-loop eval) — inference anchor policy [design note, not yet built]

How the anchor (images + state) is chosen at inference so it MATCHES training, where
the anchor is the conditioning subgoal's START frame.

**When System2 is queried** (issues / re-issues a subgoal):
1. System1 self-reports **high progress** (it thinks the subgoal is done), or
2. **timeout** (a max-steps budget for the subgoal elapsed).
Plus the **first frame** is always an anchor — System2 initializes the first subgoal there.

**Anchor update rule:** reset the anchor only when the **subgoal CHANGES**, NOT on every
query. If System2 re-issues the SAME subgoal (judges it unfinished), the anchor stays put
so "Initial State" keeps meaning "where this subgoal began" — matching training. When the
subgoal changes, snapshot the new anchor at that hand-off timestep.

**What the eval adapter must persist per episode:**
- **Current anchor** (held FIXED for the whole subgoal, like the constant training anchor):
  the 3 camera frames (anchor images) AND the raw 16-d state (anchor state). Snapshotted
  at each subgoal change.
- **Episode frame-0 base reference** ⚠️ the trap: the producer computes BOTH the current
  AND anchor lean states relative to the SAME episode-frame-0 base pose (`base_xy_ref` /
  `base_yaw_ref`). At inference, capture the first frame's raw base pose ONCE and reuse it
  as the reference for current AND every anchor for the entire episode. Making base
  relative to the subgoal-start instead would silently diverge from training. (Only base
  xy + yaw use the reference; eef_pos/rot/gripper are reference-free.)

**Per query, compute** via the SAME `robocasa_policy.lean_state_from_raw` training uses:
`current_lean = lean_state_from_raw(current_raw, ref=frame0)` and
`anchor_lean = lean_state_from_raw(anchor_raw, ref=frame0)`; then `RobocasaInputs` builds
the 28-d state + tiled-norm + tokenized prompt exactly as in training. First subgoal:
anchor == current (delta 0), matching the frame-0 training samples.

## Watch out

- **boto3** is required at train time for s3:// shards (a core dep now; rebuild any old
  image that predates this).
- **Docker GPU on snap-docker**: after a host driver upgrade, regenerate the CDI spec
  (`sudo nvidia-ctk cdi generate --output=/var/snap/docker/<rev>/etc/cdi/nvidia.yaml;
  sudo snap restart docker`).
- **`/tmp` is volatile** on the dev box — write logs under the repo (`.local_output/`).
- S3 first-batch latency: the reservoir buffer (`shuffle_buffer`/`shuffle_initial`)
  fills from S3 before step 0 (~1-2 min). Lower `shuffle_initial` for quick smokes.

## TODO / known limitations

- **Multi-dir shard pooling + anchors (fix BEFORE pooling dirs).** `--data.shards`
  accepts a comma-separated list of dirs (e.g. `normal` + `failure_augmented`), pooled
  into one dataset. But `_read_anchor` (`training/robocasa_webdataset.py`) does NOT bind
  a sample back to its source dir's anchor store — it searches the anchor roots in order
  and takes the FIRST that has `<flat_id>/f<frame>.jpg`. If two pooled dirs share a
  `flat_id`, a sample can get the WRONG rollout's "before" frame, silently training the
  progress head on a mismatched anchor. **Single-dir runs are unaffected.** Fix: carry
  the source-dir index from `_worker_shards` → sample → `_read_anchor` so each sample
  resolves its anchor in its own dir's store.
- **`shuffle_initial` is currently a no-op.** `WebDatasetConfig.shuffle_initial` is
  plumbed through but the `__iter__` reservoir loop only reads `shuffle_buffer` (emits
  once the buffer reaches `shuffle_buffer`). To speed up smoke-test startup, lower
  `shuffle_buffer` (env `ROBOCASA_SHUFFLE_BUFFER`), not `shuffle_initial` — or wire
  `shuffle_initial` in to emit at a smaller warm-up threshold.
- **Drift-verify skips on a dim mismatch.** `RobocasaInputs._to_lean_state` only asserts
  recompute==baked when shapes match; a dim mismatch (the exact drift it guards against)
  short-circuits instead of raising. Make it raise on a shape mismatch.
