# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Eval outputs

`examples/libero/main.py` writes per-run results into a directory you control:

- `--run-name <NAME>` — name of this run (defaults to a `run_YYYYMMDD_HHMMSS` UTC stamp).
- `--results-root <PATH>` — root directory for all runs (default `data/libero/results`).

Layout per run:

```
<results_root>/<run_name>/<task_suite>/
├── results.json           # run metadata + per-task + per-episode success
└── videos/
    └── task<id>_ep<idx>_<task_desc>_{success,failure}.mp4
```

`results.json` is rewritten atomically after every episode (so a crash mid-suite still leaves a valid file) and includes:

- top-level: `run_name`, `task_suite`, `seed`, `host`, `port`, `replan_steps`, `total_episodes`, `total_successes`, `total_success_rate`, `completed`, `updated_at`
- `per_task`: list of `{task_id, task_description, episodes, successes, success_rate}`
- `per_episode`: list of `{task_id, episode_idx, success, steps, video}` (video path is relative to the run dir)

Suggested `--run-name` values when comparing the four pi05_libero variants:

| Variant                             | Suggested `--run-name`     |
|-------------------------------------|----------------------------|
| Released JAX checkpoint             | `pi05_libero_jax_release`  |
| Released checkpoint, PyTorch port   | `pi05_libero_pt_release`   |
| Your finetuned JAX checkpoint       | `pi05_libero_jax_ft`       |
| Your finetuned PyTorch checkpoint   | `pi05_libero_pt_ft`        |

## tmux launcher (server + client in one session)

`examples/libero/run_eval_tmux.sh` opens a two-pane tmux session: pane 0 runs `scripts/serve_policy.py`, pane 1 runs `examples/libero/main.py` against it. The client polls the server until it's ready, so no startup coordination is required.

```bash
# Default: released JAX pi05_libero on libero_spatial, both procs on GPU 0
./examples/libero/run_eval_tmux.sh

# Released PyTorch port (after examples/convert_jax_model_to_pytorch.py)
./examples/libero/run_eval_tmux.sh \
  --run-name pi05_libero_pt_release \
  --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

# Finetuned checkpoint, server on GPU 1, libero_10 suite, alternate port
./examples/libero/run_eval_tmux.sh \
  --run-name pi05_libero_jax_ft \
  --checkpoint-dir ./checkpoints/pi05_libero/<exp>/<step> \
  --task-suite libero_10 \
  --server-gpu 1 --client-gpu 0 --port 8001
```

Available flags (each also accepts the matching env var):

| Flag                  | Default                                                            | Notes                                                  |
|-----------------------|--------------------------------------------------------------------|--------------------------------------------------------|
| `--run-name`          | `pi05_libero_jax_release`                                          | Used both as `main.py --run-name` and tmux session     |
| `--session`           | `libero_<run-name>_<task-suite>`                                   | tmux session id; pass `--replace` to kill an existing one |
| `--task-suite`        | `libero_spatial`                                                   | `libero_{spatial,object,goal,10,90}`                   |
| `--config`            | `pi05_libero`                                                      | training config registered in `_CONFIGS`               |
| `--checkpoint-dir`    | `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`            | JAX/PyTorch backend auto-detected                      |
| `--server-gpu`        | `0`                                                                | `CUDA_VISIBLE_DEVICES` for the policy server           |
| `--client-gpu`        | same as `--server-gpu`                                             | `CUDA_VISIBLE_DEVICES` for MuJoCo / libero client      |
| `--port`              | `8010`                                                             | use a unique port per concurrent session               |
| `--host`              | `127.0.0.1`                                                        | client target host                                     |
| `--results-root`      | `data/libero/results`                                              | passed through to `main.py`                            |
| `--num-trials`        | `50`                                                               | rollouts per task                                      |
| `--seed`              | `7`                                                                | eval seed                                              |
| `--mujoco-gl`         | `egl`                                                              | switch to `glx` if EGL errors                          |

The launcher also sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` for the server pane. Each pane stays open after the process exits so logs remain visible.

```bash
tmux attach -t libero_pi05_libero_jax_release_libero_spatial   # attach
tmux kill-session -t libero_pi05_libero_jax_release_libero_spatial   # abort
```

Prereqs: the libero py3.8 venv at `examples/libero/.venv` (set up via the steps above) and `tmux` on `$PATH`.

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
