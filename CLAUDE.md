# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

`openpi` is the open-source release of Physical Intelligence's robotics VLA (vision-language-action) models: π₀ (flow-matching), π₀-FAST (autoregressive with FAST tokenizer), and π₀.₅. It contains both **JAX/Flax NNX** (canonical) and **PyTorch** implementations of the models, training scripts, and a websocket-based policy server for remote inference. The repo targets Linux with NVIDIA GPUs only (Ubuntu 22.04 tested); Python is pinned to 3.11.

## Environment & Common Commands

Dependencies are managed by **uv** (workspace mode — `openpi-client` is a member package under `packages/`). Submodules must be initialized before `uv sync`.

```bash
# First-time setup (LFS skip is required because the lerobot dep uses LFS).
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Always run Python via uv so the project venv is used.
uv run scripts/train.py <config_name> --exp-name=<name>
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<dir>
uv run scripts/compute_norm_stats.py --config-name <name>   # required before training a new config

# Tests (mirrors CI in .github/workflows/test.yml).
uv run pytest --strict-markers -m "not manual"
uv run pytest src/openpi/models/pi0_test.py                  # single file
uv run pytest src/openpi/models/pi0_test.py::TestName::test  # single test
# `manual` marker is registered in pyproject.toml — those tests are skipped by default.

# Lint / format (matches pre-commit hooks; install hooks via `pre-commit install`).
uv run ruff check .
uv run ruff format .
```

`pyproject.toml` configures `testpaths = ["src", "scripts", "packages"]`, so pytest discovery covers all three. `src/openpi/conftest.py` auto-falls back to JAX's CPU backend (sets `JAX_PLATFORMS=cpu`) when no GPU is detected, so tests run on CPU-only machines.

### Memory / GPU knobs

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` — set before training to give JAX more GPU memory (default 75%).
- `--fsdp-devices=N` (TrainConfig field `fsdp_devices`) — shard model across N devices to reduce per-GPU memory.
- `OPENPI_DATA_HOME` — override the cache dir for downloaded checkpoints (`~/.cache/openpi` by default). Checkpoints live in `gs://openpi-assets/...` and are auto-downloaded via `openpi.shared.download.maybe_download`.

## Architecture

### Config-driven everything

Training, inference, and the policy server are all driven by a single registry of named `TrainConfig` instances at `src/openpi/training/config.py:_CONFIGS` (e.g. `pi0_aloha`, `pi05_droid`, `pi05_libero`, `debug`). The list is exposed via `tyro.extras.overridable_config_cli`, so CLI invocations look like `train.py <config_name> --field value`. To add support for a new robot/dataset, you typically:

1. Write `<robot>_policy.py` in `src/openpi/policies/` with `Inputs` / `Outputs` transform classes (maps env dict ↔ normalized `model.Observation` / `model.Actions`).
2. Add or reuse a `DataConfigFactory` (e.g. `LeRobotAlohaDataConfig`, `SimpleDataConfig`) in `training/config.py`.
3. Append a `TrainConfig(name=..., model=..., data=...)` entry to `_CONFIGS`.

`TrainConfig` is the single source of truth — `model`, `weight_loader`, `data`, `optimizer`, `lr_schedule`, `freeze_filter`, `ema_decay`, `fsdp_devices`, `pytorch_weight_path`, etc. all live there.

### Model layout (JAX vs PyTorch)

- `src/openpi/models/` — JAX/Flax NNX implementations: `pi0.py` (flow-matching), `pi0_fast.py` (autoregressive), `gemma.py`/`siglip.py`/`vit.py` building blocks, `lora.py`, `tokenizer.py`. JAX is the canonical/training reference.
- `src/openpi/models_pytorch/` — PyTorch port (`pi0_pytorch.py`, `gemma_pytorch.py`). It depends on **patched** Hugging Face Transformers files committed under `src/openpi/models_pytorch/transformers_replace/`. After `uv sync`, copy them into the venv (the README documents this) before running PyTorch:
  ```bash
  cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
  ```
  This mutates the uv cache (hardlinks); use `uv cache clean transformers` to undo. PyTorch path **does not yet support** π₀-FAST, mixed precision, FSDP, LoRA, or EMA.

`policy_config.create_trained_policy` auto-detects PyTorch checkpoints by the presence of `model.safetensors` in the checkpoint dir; otherwise it loads JAX `params/` via Orbax. The two backends share the same `TrainConfig`, transforms, normalization, and `Policy` wrapper — only weight loading and the forward pass differ.

### Data pipeline (transforms)

A `DataConfig` (returned by a `DataConfigFactory.create(...)`) carries three transform groups applied in order during inference and training:

1. `repack_transforms` — rename/reshape raw env keys into the canonical schema.
2. `data_transforms` — robot-specific (e.g. `AlohaInputs`, `DroidInputs`).
3. `Normalize` (using `norm_stats.json` from the assets dir) — quantile or std normalization.
4. `model_transforms` — model-specific (tokenization, padding to `action_horizon`, etc.).

Outputs run in reverse. `compute_norm_stats.py` produces the `norm_stats.json` that lives under `assets/<config_name>/<asset_id>/`. See `docs/norm_stats.md` for the “reload pretrain norm stats” pattern, which matters when fine-tuning on a robot that was in the pretraining mix.

### Training scripts

- `scripts/train.py` — JAX trainer. Uses `nnx.split`/`nnx.merge`, `optax`, FSDP via `openpi.training.sharding`, Orbax checkpoints in `openpi.training.checkpoints`, wandb logging.
- `scripts/train_pytorch.py` — PyTorch trainer (single-GPU and `torchrun` multi-GPU/multi-node). Loads from `pytorch_weight_path` set in the config.
- `examples/convert_jax_model_to_pytorch.py` — required to use PyTorch finetuning starting from a JAX checkpoint.

### Policy serving

`scripts/serve_policy.py` wraps a `Policy` in `openpi.serving.websocket_policy_server` (port 8000 by default). Clients live in `packages/openpi-client/` (a uv workspace member, separately publishable) and stream observations/actions over a websocket. `EnvMode` (`aloha`, `aloha_sim`, `droid`, `libero`) selects a default checkpoint when `--policy.dir` isn't supplied. See `docs/remote_inference.md` for the on-robot integration pattern.

### Examples

`examples/<env>/` contains end-to-end recipes (data conversion, eval scripts, Dockerfiles) for `libero`, `droid`, `aloha_real`, `aloha_sim`, `ur5`, plus a `simple_client/` for testing without a robot. The LIBERO and DROID examples are the most fully fleshed out; new robot integrations should follow their structure.

## Conventions

- Ruff config in `pyproject.toml` uses a wide rule set (B, RUF, PERF, etc.), `line-length = 120`, py311 target. Isort is **force-single-line** with sections sorted within. `third_party/`, `docker/`, and `models_pytorch/transformers_replace/*` are excluded from lint.
- `dependency-groups`: `dev` (pytest, ruff, pre-commit, ipykernel, matplotlib, pynvml) and `rlds` (TF-CPU 2.15, dlimp — needs `uv venv --python 3.11` first because TF only ships cp311 wheels). Use `uv sync --group rlds` only when you actually need DROID RLDS data loading.
- `tool.uv.override-dependencies` pins `ml-dtypes` and `tensorstore`; do not change without verifying JAX/Orbax compatibility.
- Type-checking runtime: many model paths use `beartype` via `openpi.shared.array_typing` (`@at.typecheck`, `at.Array`, jaxtyping shapes). Preserve those decorators when editing — they catch shape bugs early.
