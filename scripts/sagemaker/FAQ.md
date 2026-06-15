# SageMaker training FAQ

A running list of questions that came up while wiring up the SageMaker
path for `pi05_libero` PyTorch training. Skim before submitting your
first job.

---

## 1. Is `launch.py` compatible with local directories?

**Not as-is, but the workaround is `--local`.**

Three things block plain "swap S3 path for a local path":

1. `TrainingInput(s3_data=...)` only accepts `s3://...` URIs — a local
   path fails SageMaker validation before submission.
2. The remote instance has no access to your laptop's filesystem; that's
   exactly what S3 is bridging.
3. `output.s3_prefix` and `checkpoint_s3_uri` must also be `s3://` for
   the same reason — SageMaker only mirrors `/opt/ml/checkpoints/` to S3.

If you really want local paths, use `launch.py --local`. The SageMaker
local backend accepts `file://` URIs for inputs (must pair with
`input_mode: File`, not `FastFile`).

For day-to-day local iteration, prefer the bind-mount workflow in
`LOCAL_TESTING.md` — it's faster than the local SageMaker backend.

---

## 2. Should I upload the libero data and openpi pretrained ckpts now?

**Yes**, before submitting your first job. Order:

1. Three artifacts on disk, ~52 GB total: LeRobot libero (33 GB),
   pi05_base_pytorch (~7 GB), pi05_base JAX (~12 GB), assets (~20 KB).
2. Run `uv run --group sagemaker scripts/sagemaker/upload_data_to_s3.py` (uploads the two
   training channels — libero and base_ckpt). Idempotent: re-running
   only diffs new/changed files.
3. Optional opt-in mirrors: `--channels base_jax libero_pytorch`. Not
   mounted into the training job; useful for in-cloud JAX→PyTorch
   conversion or eval/resume baselines.
4. Sanity-check the upload landed where you expect:
   ```bash
   aws s3 ls s3://tri-ml-datasets-uw2/yinpeidai/data/libero/meta/ --profile sagemaker
   aws s3 ls s3://.../released_ckpt/openpi-assets/checkpoints/pi05_base_pytorch/ --profile sagemaker
   ```

Norm-stats assets are baked into the Docker image (~20 KB, not worth a
channel), so no upload is needed for them.

---

## 3. How can we build the docker and test the training locally without
   pushing to SageMaker?

See `LOCAL_TESTING.md` for the full recipe. Two flavors:

- Plain `docker run` with bind-mounted local artifacts (fastest, no S3,
  no SageMaker runtime). Good for >95% of iteration.
- `launch.py --local` which exercises the SageMaker LocalSession runtime.
  Slower, only useful for debugging SageMaker-specific issues.

For both, build with `make -C scripts/sagemaker docker-build-sm` (no push).

---

## 4. What do `HF_LEROBOT_HOME`, `HF_HOME`, `OPENPI_DATA_HOME` do? Does the
   job download data from S3 to local disk?

**The job does NOT download anything from S3.** SageMaker mounts each
channel as a FUSE filesystem at `/opt/ml/input/data/<channel>/` via the
FastFile driver before the container starts. The trainer just opens
local file paths; it never sees an `s3://` URL. Reads stream blocks
from S3 on-demand and cache them in memory / NVMe.

Each env var serves a specific purpose:

- **`HF_LEROBOT_HOME` (required)** — `LeRobotDataset(repo_id="...")`
  reads from `$HF_LEROBOT_HOME/<repo_id>/{data,meta}/`. We set it so
  LeRobot doesn't fall back to `~/.cache/huggingface/lerobot/`, find
  nothing, and try to `snapshot_download` from HF Hub.
- **`HF_HOME` (defensive)** — Redirects HF Hub's general cache off the
  read-only image layers. Catches stray `AutoTokenizer.from_pretrained`
  / `snapshot_download` calls that would otherwise crash on a read-only
  filesystem.
- **`OPENPI_DATA_HOME` (defensive)** — Same idea for openpi's own
  `maybe_download` cache (defaults to `~/.cache/openpi`). Not used in
  the happy path (we pass `--pytorch-weight-path` and `--assets-base-dir`
  explicitly), but safe to have set.

Data flow summary:

| Artifact | Source | How it's read |
| --- | --- | --- |
| libero parquet | S3 `libero` channel | FastFile FUSE → on-demand block fetch |
| pi05_base ckpt | S3 `base_ckpt` channel | FastFile FUSE → loaded once at start |
| norm_stats.json | Baked into image | Plain file read, no S3 |
| Output ckpts | `/opt/ml/checkpoints/` | SageMaker syncs **to** S3 continuously |

---

## 5. Which part of the Dockerfile mounts the S3 libero data into
   `/opt/ml/input/data/libero`?

**None of it. SageMaker does the mount, not Docker.**

The Dockerfile only declares a name on line 69:

```dockerfile
ENV HF_LEROBOT_HOME=/opt/ml/input/data/libero
```

That's a *promise* about where the data will appear, not a mount. The
real chain is:

```
config.yaml channels.libero.s3_uri
   → launch.py inputs["libero"] = TrainingInput(...)
       → SageMaker control plane (FastFile FUSE driver)
           → /opt/ml/input/data/libero/   (visible inside the container)
               → HF_LEROBOT_HOME points there → LeRobot reads files
```

The channel name in the `inputs` dict (`"libero"`) is what determines
the directory name. Change `inputs={"foo": ...}` and SageMaker would
mount it at `/opt/ml/input/data/foo/`.

This is also why **the same image works for any S3 layout** — the
container is agnostic. Editing `config.yaml` to point at a different S3
bucket needs no image rebuild.

The flip side: outside SageMaker, nothing mounts. That's why local
testing uses bind-mounts (`-v ...:/opt/ml/input/data/libero:ro`) to
emulate what SageMaker would have done.

---

## 6. Should we change `repo_id="physical-intelligence/libero"` since
   the S3 layout doesn't have that prefix?

**No. The symlink in `sm_entrypoint.sh` handles it.**

S3 channel mount layout:
```
/opt/ml/input/data/libero/{data,meta}/...
```

What LeRobot expects (because `repo_id="physical-intelligence/libero"`):
```
$HF_LEROBOT_HOME/physical-intelligence/libero/{data,meta}/...
```

`sm_entrypoint.sh:39-43` builds that view in `/tmp` instead of changing
the config:

```bash
LEROBOT_ROOT="/tmp/lerobot_root"
mkdir -p "$LEROBOT_ROOT/physical-intelligence"
ln -sfn "$LIBERO_DIR" "$LEROBOT_ROOT/physical-intelligence/libero"
export HF_LEROBOT_HOME="$LEROBOT_ROOT"
```

Why not change `repo_id`?
- **It's also the asset_id key.** `AssetsConfig(asset_id="physical-
  intelligence/libero")` is what makes the trainer look for norm stats at
  `assets/pi05_libero/physical-intelligence/libero/norm_stats.json`.
- **It's also the HF Hub fallback identifier.** Renaming would orphan
  the dataset from the public registry.
- **The DGX/local config uses the same `repo_id`** with the canonical
  cache layout. Changing it would force two divergent configs.
- **The symlink is free.** One `ln -s` at job start, ~1 ms.
