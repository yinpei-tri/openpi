# RoboCasa System1 — train / serve / eval workflow (across repos)

The System1 (RoboCasa pi0.5 + progress head) work spans **two git repos**, split by
the Python environment each piece needs. This doc is the map: what lives where, the
end-to-end flow, and — importantly — the **commit rule**.

> **Related, deeper docs:**
> - Training details (data, configs, knobs, envs): [robocasa_system1_training.md](robocasa_system1_training.md)
> - Multi-node training: [robocasa_multinode_training.md](robocasa_multinode_training.md)
> - Norm stats: [norm_stats.md](norm_stats.md)
> - Eval harness internals: `robocasa/robocasa/scripts/eval/README.md`

---

## ⚠️ Commit rule: always commit BOTH repos together

A single logical change to the System1 eval loop usually touches **both** repos at once
(the driver/policy code in `openpi`, the sim-reset/criteria code in `robocasa`). They are
**separate repos with separate histories** — there is no submodule, so nothing links them
automatically.

**When you finish a change, commit in both repos in the same work session**, with matching
messages so the two histories stay in sync:

```bash
# openpi
cd /home/yinpei.dai/openpi     && git add <files> && git commit

# robocasa
cd /home/yinpei.dai/robocasa   && git add <files> && git commit
```

Why not a submodule? A git submodule is only a pinned-SHA *pointer* — editing the child
still forces a commit in the child's history **plus** a pointer-bump commit in the parent,
i.e. *more* friction, not less, and histories never actually merge. So we keep two plain
repos and rely on this discipline instead.

Why can't it all live in one repo? The sim eval harness imports `robocasa` / `robosuite` /
`mujoco` and runs in the **`robocasa` micromamba env**; openpi's `.venv` can't even import
`robocasa`. The environment boundary is a hard wall — the harness has to live in `robocasa`.

---

## Repo / environment map

| Piece | Path | Env | Purpose |
| --- | --- | --- | --- |
| Training | `openpi/scripts/train.py` | openpi `.venv` | Train pi0.5 + progress head on WebDataset shards |
| Configs | `openpi/src/openpi/training/config.py` | openpi `.venv` | `pi05_robocasa_system1` (+ `_noanchor`, `_debug`) |
| Tag ↔ config | `openpi/scripts/train.py` (`robocasa_exp_tag`, `robocasa_config_from_tag`, `resolve_robocasa_config`) | openpi `.venv` | One root config + CLI overrides; the settings tag in the ckpt dir name (and `config.json`) reconstructs the exact per-ckpt config |
| Serving | `openpi/scripts/serve_policy.py` | openpi `.venv` | Websocket policy server; `--policy.config auto` resolves config from the ckpt |
| Rollout driver | `openpi/examples/robocasa/subtask_eval.py` | openpi `.venv` | Talks to the policy server; writes the rollout tree |
| Rollout GUI | `openpi/examples/robocasa/subtask_eval_gui.py` | any (Flask) | Self-contained viewer for the rollout tree (HTML/JS embedded) |
| **Sim eval harness** | `robocasa/robocasa/scripts/eval/` | **robocasa micromamba env** | Reset sim to a subgoal start, roll out, per-subtask readouts |

Envs:
```bash
OPENPI_PY=/home/yinpei.dai/openpi/.venv/bin/python
ROBOCASA_PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
```

---

## Config reconstruction (why the ckpt dir name matters)

All System1 ablations train from **one** config (`pi05_robocasa_system1`) + CLI overrides;
the differences are encoded in the exp-name **settings tag**, e.g.
`progreg_granfine_verbsimp_noexec_nostate`. So a checkpoint dir looks like:

```
checkpoints/pi05_robocasa_system1/<exp>__<tag>/<step>/
```

Two ways inference recovers the exact config from a checkpoint (see
`resolve_robocasa_config`):
1. **`config.json`** — written by training into both the run root **and** each finalized
   step dir (`<step>/config.json`), so a single uploaded step dir is self-describing.
2. **dir-name tag** — fallback for older ckpts with no `config.json` (parses `__<tag>`).

Tag grammar: `prog{cls,reg,act,none}_gran{fine,crse,both}_verb{simp,rich,both}[_<deviation>...]`
(deviations: `nocond, noexec, noestl, nostate, noanchorstate, notask, noanchor, nogrip`).

---

## End-to-end flow

### 1. Train (openpi env)
```bash
cd /home/yinpei.dai/openpi
export ROBOCASA_SHARDS_DIR=<local shards dir OR s3://...>
$OPENPI_PY scripts/train.py pi05_robocasa_system1 \
    --exp-name=<run> \
    --model.progress-mode=continuous \        # progreg (or default progcls, or --data.progress-as-action)
    --data.no-include-state \                  # any ablation flags -> become tag tokens
    --batch-size=64 --fsdp-devices=8 \
    --num-train-steps=50000 --save-interval=5000 --overwrite
# -> checkpoints/pi05_robocasa_system1/<run>__<tag>/<step>/  (+ config.json in each step dir)
```
See [robocasa_system1_training.md](robocasa_system1_training.md) for data setup, S3
streaming, norm stats, memory knobs, and multi-node.

### 2. Serve the policy (openpi env)
```bash
$OPENPI_PY scripts/serve_policy.py --port 8010 policy:checkpoint \
    --policy.dir <ckpt>/<run>__<tag>/<step>     # --policy.config defaults to 'auto'
```

### 3. Roll out per-subtask (openpi env, needs the server up)
```bash
$OPENPI_PY examples/robocasa/subtask_eval.py \
    --host 127.0.0.1 --port 8010 \
    --data-root <data_annotation> --episode-list <eps.txt> \
    --out-root subtask_rollouts --method reg   # method = subdir label under out-root
# -> subtask_rollouts/<method>/<episode>/child<NN>_<primitive>/{clean.mp4, anchor_*.jpg, steps.json}
```

The oracle (Phase-1, no policy) path runs in the **robocasa env** — either the same driver
with `--oracle`, or the harness module directly:
```bash
$ROBOCASA_PY -m robocasa.scripts.eval.eval_subtask_oracle \
    --data-root <data_annotation> --episode-list <eps.txt>
```

### 4. View the rollouts (any env)
```bash
$OPENPI_PY examples/robocasa/subtask_eval_gui.py --rollout-root subtask_rollouts --port 8092
# (GUI uses --rollout-root; the driver above uses --out-root -- same tree)
```

### 5. Judge success — **not** in-harness
The harness never self-judges. Rollout videos + logs are reviewed in the GUI, and
**Gemini decides subtask completion** separately.

---

## Data / artifacts are git-ignored (never commit)

`robocasa_training_data/`, `subtask_rollouts/`, `checkpoints/`, `replay_check_results/`,
`*.mp4`, and stats `*.json`/`*.png` are data, not code — keep them out of both repos.
Stage files explicitly (`git add <path>`), never `git add .`.
