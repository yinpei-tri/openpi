# RoboCasa System1 — Evaluation & Rollout Labeling

This document is the single reference for how we evaluate RoboCasa System1 VLA checkpoints
(subgoal-conditioned π₀.₅) **and** — the larger goal — how we produce **reliable per-span
success/failure labels** on rollouts. It covers the evaluation levels, the commands, how success is
decided at each level, the GUI, and the known limitations.

All evals run in the **robocasa micromamba env** (`/home/yinpei.dai/micromamba/envs/robocasa/bin/python`)
and talk to a policy served from the **openpi `.venv`** via `scripts/serve_policy.py` (websocket).
Ground-truth episodes + dense subgoal annotations come from the RoboAnnotator interface
(shared at `/home/<user>/data/data_annotation`, mirrored to
`s3://tri-ml-datasets-uw2/yinpeidai/robocasa_dataset_preprocessed/data_annotation`; subgoal
method `g_subgoal_gemini3_final_batch`).

Results for the current target-split run live under **`eval_results/`**:
- `eval_results/episode/<method>/`    — episode-level open-loop
- `eval_results/finestep/<method>/`   — fine-step (per-child-subgoal) eval + Gemini/sim-check
- `eval_results/milestone/<method>/`  — milestone-level (oracle-referenced sim-check)
- `eval_results/valmse_results/`, `.../trainmse_results/` — offline MSE curves
- each also has an `oracle` method = the ground-truth reference rollout.

The `eval_results/` tree is mirrored to
**`s3://tri-ml-datasets-uw2/yinpeidai/system1_eval_results`** (and browsed via the
RoboAnnotator portal's `/system1_eval` proxy). Checkpoints live under
`s3://tri-ml-datasets-uw2/yinpeidai/openpi/checkpoints`.

```sh
# pull / push the System1 eval results
aws s3 sync s3://tri-ml-datasets-uw2/yinpeidai/system1_eval_results eval_results
```

---

## Why this exists: reliable labels for offline RL (the north star)

The end goal is **not just a leaderboard** — it's to gather rollouts with a **clear success/failure
label per span**, then train System1 with **offline RL** where the success/failure label is a
**conditioning token** (so the model learns from ALL data, successes and failures alike; pi0.7-style
"train on everything" improves performance). **The label is the product.** Everything below is in
service of making that label trustworthy.

**The nested labeling hierarchy** (consistent by construction — inner labels compose into outer):

| level | primitives labeled | signal | role |
|---|---|---|---|
| **fine-step** | grasp, place/release, press/turn **only** | oracle-state reference, per child span | dense, high-precision labels on spans where a sim predicate is trustworthy |
| **milestone** | ALL primitives (grasp/pick/place/open/close/turn/nav/…) | oracle-state reference, per milestone | semantically-complete labels; settling makes them reliable |
| **episode** | — | compose milestone verdicts + env `_check_success` at the end | task-level success |

**Label-reliability principles (non-negotiable for RL conditioning):**
1. **Precision ≫ coverage.** A *wrong* label is actively harmful (teaches System1 to reproduce a
   failure under the "success" token). A *missing* label is harmless — emit `unknown` and either skip
   conditioning that span or give it a distinct `unknown` token. Never trade precision for coverage.
2. **Three-way verdict, always:** `success / failure / unknown`. `unknown` is a first-class label.
3. **Oracle-STATE reference, not action-replay.** Reference rollouts set recorded MuJoCo *states*
   frame-by-frame (stable, exact GT), NOT replayed actions (open-loop, drifts with sim noise → wrong
   reference end-state). Confirmed: CloseBlenderLid oracle `sim_success_final` flipped False→True when
   switched from action- to state-replay.
4. **Single-reference caveat.** End-state predicates (grasp contact, in-receptacle, joint qpos,
   fixture flag) are canonical — a valid alternate solution reaches the same end-state, no false
   negative. **navigate** is the exception (a different-but-valid pose could be mislabeled `failure`)
   → keep it strict/`unknown`-leaning. Milestone/fine labels are RELATIVE to GT; episode
   `_check_success` is ABSOLUTE task success — keep them as separate label fields, don't collapse.
5. **Calibration gate.** A predicate is trustworthy enough to become a training label ONLY if the
   **oracle's own rollout labels its own spans ≈100% success**. If the oracle "fails" its own span,
   the predicate is wrong, not the trajectory.

---

## Serving a checkpoint

The server reconstructs the config from the checkpoint (`config.json`, else the dir-name ablation
tag `progcls`/`progreg`/`progact` + deviations `noexec`/`nocond`/…), so `--policy.config=auto` needs
no per-variant config.

```bash
# openpi .venv, one GPU (80GB; low mem-fraction is plenty — the run is sim/CPU-bound).
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/serve_policy.py --port 8060 \
    policy:checkpoint \
    --policy.dir checkpoints/m0717-50k-bs512-v4__progreg_granfine_verbsimp_noexec/49999
```
Two servers per 80GB GPU is fine. The eval clients take
`--norm-stats checkpoints/<exp>/49999/assets/robocasa_system1/norm_stats.json`.

---

## Level 0 — Validation / Train MSE  *(offline, no sim)*

**Measures** pure action-prediction error (predicted vs recorded GT action chunk) on val + train
shards. No env, no rollout. **No success signal** — an MSE proxy only; a low-MSE model can still fail
the task. **From** `eval_results/{valmse,trainmse}_results/*.json` (sweep
`scripts/run_val_sweep.py` + `run_train_after_val.sh`; `SWEEP_OUT_DIR` overrides). **GUI:** `/val_mse`
curves; `/stats` final-step MSE.

---

## Level 1 — EPISODE-level (open-loop)  *(`examples/robocasa/episode_eval.py`)*

**Measures** whole-episode task success, closed-loop. Reset **once** to the first subgoal's start,
then roll the policy **continuously** through the ordered subgoal list (sim state carries over, no
reset); feed each subgoal's prompt; advance on the shared **STOP RULE** (progress ≥ `--stop-progress`
AND action-quiescence over `--stop-window`, see `stop_criterion.py`), mimicking System2 handing
subgoals to System1 one-by-one.

**Success signal (ground truth):** the RoboCasa env's own **`_check_success()`** at the end →
per-episode `episode_success`. Also records per-subgoal `advanced` (stop fired) vs `timeout` (budget
consumed, rule never fired). **The most trustworthy absolute signal we have.** In the labeling
hierarchy, episode success also = compose the milestone verdicts.

```bash
PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  $PY examples/robocasa/episode_eval.py \
    --episode-list eps.txt --host 127.0.0.1 --port 8060 \
    --norm-stats checkpoints/<exp>/49999/assets/robocasa_system1/norm_stats.json \
    --out-root eval_results/episode --method v4_progreg_noexec
# --oracle replays recorded actions as a reference (NOTE: action-replay, superseded by state-replay
# in milestone eval; for episode-level the env _check_success is what matters).
```
Flags: `--replan-steps 16 --horizon-mult 2.0 --max-steps-cap 400 --settle-steps 10`,
`--stop-progress 0.95 --stop-eps 0.02 --stop-window 5`.

**Per-subgoal budget (`--budget-formula`, all levels).** Selectable so old runs stay reproducible:
- `legacy` (**default**) = `min(cap, max(HORIZON, round(mult × span_len)))` — the formula the
  **v1..v12 target-split eval used**. Hard cap can fall BELOW span_len for long spans (a 1222-frame
  span capped at 400 → guaranteed spurious timeout).
- `longsafe` = `max(HORIZON, min(a + cap, round(mult × a)))`, `a = max(span_len, est_len)` — 2× for
  short/normal spans, `+cap` slack for long spans so budget always exceeds the span.
Recorded in each run's `index.json` (`budget_formula`). **Existing `eval_results` used `legacy`.**

**Output** `episode/<method>/<episode_flat>/` per-subgoal `clean.mp4` + `steps.npz` +
`steps_meta.json`; `episode.json` (`episode_success`, per-subgoal `advanced`/`stop_reason`);
per-method `index.json`. **GUI:** `/episode`; `/stats` #2 (per-split rates) + #2b (per-task).

---

## Level 2 — MILESTONE-level (oracle-referenced sim-check)  *(`examples/robocasa/milestone_eval.py` + `milestone_sim_check.py`)*

**The reliable, sim-grounded label for ALL primitives.** A milestone ("pick up the pan", "open the
fridge door", "place the ice into the glass") is a semantically-complete state change whose motion
has **settled** by its end — so the sim predicate is reliable, unlike the fine-step check which fires
mid-motion (see limitations). **86% of milestones end in a checkable manipulation goal** (only
hold/stir/scrub/dump stay `unknown`), and oracle REF resolvability is ~97%.

**Rollout structure** — for each milestone of an episode:
1. `reset_to(milestone_start_frame)` (full reset so fixtures/objects are placed) — clean GT start,
   isolating the milestone from upstream error (chosen over re-scoring the continuous episode
   rollout, which would inherit early-milestone failures).
2. **Policy:** roll CONTINUOUSLY through the milestone's child subgoals (per-child prompt, STOP-rule
   handoff, `--budget-formula` per child). **Oracle:** STATE-DRIVEN replay — step recorded MuJoCo
   states frame-by-frame across the span, rendering each frame (produces the `clean.mp4` reference
   video). State-replay reproduces the exact stable GT; action-replay would drift.
3. Settled end: **oracle** captures the REF state; **policy** loads the matching oracle REF (by
   episode_id + milestone_index) and evaluates the goal predicate against it.

Milestone "goal primitive" = its last non-move_to/navigate/reach/retract/hold child (grasp + trailing
lift/move ⇒ `pick_up`). **Object resolution by gripper CONTACT, not text** (held-most-frames), robust
to synonyms ("pan" == env object `vegetable_container`); receptacle = the object the held object ends
up `check_obj_in_receptacle` with. Both stored in the REF so the policy check doesn't re-parse.

**Per-goal predicates** (verdict success/failure/unknown; all compared to the oracle REF at the
settled milestone-end — **no self-chosen absolute thresholds where an oracle scalar exists**):
- **grasp** — `check_contact(gripper, obj)` AND `|finger_opening − oracle| ≤ 0.01` (matching the
  oracle finger state implies the right closure; rejects a closed EMPTY gripper).
- **pick_up** — grasp AND object lifted: rollout Δz ≥ `0.5 ×` oracle Δz (vs milestone-start z).
- **place / release** — `check_obj_in_receptacle(objA, recepB)` AND gripper→obj dist ≥
  `oracle_dist − 0.05` (released/retracted as far as GT; mirrors episode `_check_success` design).
- **open/close/pull/push** — per-door joint qpos `|joint − oracle_joint| ≤ 0.10` (settled value IS
  the target, whichever side; per-door for multi-door fixtures like FridgeFrenchDoor).
- **turn / press** — fixture on/off flag == oracle's (faucet water_on / stove knobs / `turned_on`).
- **navigate / search** — base xy within `0.30 m` of oracle end-xy AND `cos(Δyaw vs oracle) ≥ 0.90`
  (referenced to the oracle's *achieved* pose, sidestepping NavigateKitchen's over-strict fixed
  target; recovers nav as checkable — it was always `unknown` at fine level).
- **hold / stir / scrub / dump / other** — `unknown` (no clean predicate).

```bash
PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
# 1) oracle reference (state-driven, no server) — CAPTURES per-milestone REF into episode.json:
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  $PY examples/robocasa/milestone_eval.py --episode-list eps.txt --oracle \
    --out-root eval_results/milestone --method oracle
# 2) policy (LOADS the oracle refs by episode_id, compares):
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
  $PY examples/robocasa/milestone_eval.py --episode-list eps.txt --host 127.0.0.1 --port 8060 \
    --norm-stats checkpoints/<exp>/49999/assets/robocasa_system1/norm_stats.json \
    --out-root eval_results/milestone --method v4_progreg_noexec --oracle-method oracle
```
**Output** `milestone/<method>/<episode_flat>/m<NN>_<goalprim>/` (`clean.mp4` + `steps.npz`);
`episode.json` per-milestone `milestone_sim_check` (verdict/rule/target/detail) + `ref` (oracle only)
+ `sim_success_final`. **GUI:** `/milestone`.

**Calibration gate:** run the oracle, then run the policy-check ON the oracle rollout against its own
REF — decided-success must be ≈100%. (Contrast: the fine-step child-level check gives oracle only
74.8%.)

---

## Level 3 — FINE-STEP (per-child-subgoal)  *(`examples/robocasa/subtask_eval.py` → to be narrowed; GUI `/finestep`)*

**Measures** per-child-subgoal success in isolation: hard-reset to each child's exact start, roll the
policy closed-loop with that child's prompt (+ full training conditioning) for a `--budget-formula`
budget. Isolates "can the policy do THIS step" from error accumulation.

**Intended scope going forward:** fine-step labels should be trusted ONLY for **grasp, place/release,
press/turn** — the spans whose end-state predicate is reliable at the child boundary — using the same
oracle-STATE reference + comparison as milestone level. move_to/navigate and the settle-sensitive
open/close/pull spans are LEFT to the milestone level (they false-negative at the child boundary; see
limitations). This is a narrowing of the current `subtask_sim_check`, not a new mechanism.

**Current signals recorded per child span** (legacy, being narrowed):
1. `subtask_sim_check` — sim verdict (grasp/fixture-open-close-turn-press/place-release; else
   `unknown`). **Known to false-negative** on push/pull/close/grasp at the span boundary.
2. `sim_success_final` — env `_check_success()` at the rollout end (meaningful at the terminal span).
3. **Gemini** — VLM judge on oracle clip + policy clip → success/failure/uncertain, for spans
   sim-check leaves `unknown`.

```bash
# rollouts (writes fine-step tree):
$PY examples/robocasa/subtask_eval.py --episode-list eps.txt --host 127.0.0.1 --port 8060 \
    --norm-stats <ckpt>/assets/robocasa_system1/norm_stats.json \
    --out-root eval_results/finestep --method v4_progreg_noexec
# Gemini batch judge (Vertex GCS batch, 50% price, gemini-3.6-flash):
ROBOANNOTATOR=/home/yinpei.dai/RoboAnnotator \
  $PY scripts/gemini_judge_subtasks_batch.py --rollout-root eval_results/finestep \
    --oracle-method oracle --methods v4_progreg_noexec,v9_progact_noexec \
    --model gemini-3.6-flash --chunk-size 2000 --skip-existing
```
Gemini gates (no query, $0): `retract` moves + net-eef-displacement < `--move-gate` 0.04 m.
Per-span `<sub_dir>/judge/{gemini.json,response.txt,response_meta.json,inputs/}`; per-method rollup
`<method>/gemini_summary.json`. **Requires `gcloud auth login`** (batch upload uses the gcloud CLI
token, not just ADC; an expired token fails the upload before any job/cost). **GUI:** `/finestep`;
`/stats` #3.

---

## GUI

`examples/robocasa/subtask_eval_gui.py` serves the eval roots. Pages: `/finestep` `/milestone`
`/episode` `/val_mse` `/stats`, plus `/` landing. Launch:
```bash
$PY examples/robocasa/subtask_eval_gui.py \
    --finestep-root eval_results/finestep \
    --milestone-root eval_results/milestone \
    --episode-root  eval_results/episode --port 9091
```
Lazy loading: the episode dropdown reads only `index.json` (∪ dir-scan) — full per-episode doc loads
on click. `/stats` #3 reads per-method `gemini_summary.json` (three-way: success rate =
success/(success+failure), uncertain shown separately, gated excluded); oracle falls back to the
per-span sim-check verdict.

---

## Known limitations (read before trusting the numbers)

- **Fine-step span-boundary false-negatives** — the child-level sim-check evaluates the predicate at
  the child's END frame, but the state change completes in the NEXT child (grasp ends before lift;
  push/pull/close/turn end before the joint settles). Spurious `failure` even on a perfect oracle
  replay. Evidence: oracle episode-level = **94.5%** but oracle fine-step decided = only **74.8%**
  (push 33%, pull 53%, close 69%, grasp 84%); on models, sim-check vs Gemini disagree ~64–68% and the
  disagreements are overwhelmingly `sim=failure/gemini=success` (push 117:5, pull 92:41, close 37:2,
  grasp 95:60) = sim false-negatives, not model failures. **This is exactly why the MILESTONE level
  exists** — at milestone end the motion has settled, so the same predicates are reliable. Fine-step
  is therefore narrowed to grasp/place/release/press/turn only.

- **NavigateKitchen `_check_success` is too strict** (episode-level) — needs base within 0.20 m of
  target AND cos(Δyaw) ≥ 0.98 simultaneously; the GT demo itself clears by ~0.01 m. The milestone nav
  predicate sidesteps this by referencing the oracle's achieved pose (0.30 m / cos ≥ 0.90).

- **Single-oracle-reference for navigate** — a valid alternate parking pose could be labeled
  `failure`. Keep nav strict/`unknown`-leaning for RL labels (principle #4).

- **Fusion (fine-step /stats #3) not wired** — #3 currently shows pure-Gemini for models, pure
  sim-check for oracle. A naive "sim-check overrides Gemini" fusion would propagate the ~250+
  false-negatives above; the milestone level supersedes the need for it.
