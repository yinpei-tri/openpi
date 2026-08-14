"""COMBINED System2 + System1 closed-loop eval on RoboCasa.

System2 (Qwen3.5-4B planner, vLLM server) decides WHAT to do; System1 (pi0.5, JAX policy server)
decides HOW to move. One loop:

    PLAN (cold)   tiled scene image                  -> <plan> milestone checklist
    EXEC turn 1   checklist + tiled scene image      -> <subgoal> + <estimated_step>
      |
      +-> S1 SEGMENT  prompt = "Task: <goal>; Current Subgoal: <S2 subgoal>"
      |               anchor = FIRST frame of THIS segment; executed_step starts at 0
      |               budget = estimated_step * horizon_mult (capped); ends on the shared
      |               STOP RULE (progress >= thresh AND action quiescence) or budget
      |
    EXEC turn n   checklist + CLIP of the segment just run + privileged env signals
                  -> next <subgoal>, or <judge>task_finish  => STOP

Design decisions (confirmed with the user, not inferred):
  * anchor images/state = the first frame of the current S1 segment (i.e. the last frame of the
    previous one); BOTH the anchor and ``executed_step`` reset on every new S2 turn.
  * segment budget is S2-driven: ``estimated_step * horizon_mult`` capped at ``max_steps_cap``
    (no ground-truth span lengths leak into the loop).
  * ``Current task status`` is fed LIVE from ``env._check_success()``. System2's system prompt
    tells it to judge ``task_finish`` only when that reads ``finished``, so withholding it would
    prevent termination. It is a privileged (GT) bit by design -- recorded as such in the output.
  * termination: S2 ``task_finish`` OR ``--max-turns`` OR env success already true.
  * clip frame selection: near-static frames are dropped for motion subgoals, but KEPT for
    wait/hold subgoals where stillness is the content (see sys2_client.build_clip_frames).

EVERYTHING is recorded. Per turn we save the S2 system+user prompt, the raw response and every
parsed tag, the exact media the model saw, the S1 prompt, per-step actions/progress/state, the
anchor image, the raw rollout video and the compacted 4-fps clip. See ``turn.json`` /
``episode.json`` below and the /combine GUI.

NO ANNOTATIONS ARE READ. The annotation files were the TRAINING data for System1 and System2;
at rollout time System2 supplies the subgoals and System1 the actions. Everything this script
needs comes from the raw LeRobot dataset and the env itself:
  * ``reset_to(states[0])``           -- the episode's own first frame (recorded scene + start),
  * ``ep_meta['lang']``               -- the task goal handed to System2's planner,
  * ``env._check_success()``          -- the episode-success metric AND the privileged
                                         "Current task status" line System2 reads.
The recorded ACTIONS are never used; System1 generates every action.

Run (robocasa micromamba env; needs a System2 vLLM server AND a System1 policy server):
    MUJOCO_GL=egl "$ROBOCASA_PY" examples/robocasa/combined_eval.py \
        --lerobot-dir "$ROBOCASA_LEROBOT_ROOT/v1.0/target/atomic/CloseFridge/20250816/lerobot" \
        --episodes 0 --method s2s1_progreg \
        --s1-port 8060 --s2-port 8100 \
        --norm-stats <ckpt>/assets/robocasa_system1/norm_stats.json \
        --out-root eval_results/combine
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np
from openpi_client import websocket_client_policy as _wcp
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.scripts.eval.subtask_env import base_reference
from robocasa.scripts.eval.subtask_env import images_from_obs
from robocasa.scripts.eval.subtask_env import lerobot_action_to_sim
from robocasa.scripts.eval.subtask_env import make_camera_env
from robocasa.scripts.eval.subtask_env import raw_state_from_obs
from stop_criterion import StopConfig
from stop_criterion import StopTracker
from stop_criterion import action_eef_base_norm
from stop_criterion import gripper_width

# Sibling imports (this file runs as a script; examples/robocasa is sys.path[0]).
import subtask_eval as SE
import sys2_client as S2C
import sys2_rules as SR

HORIZON = SE.HORIZON
SIM_GRIP_IDX = SE.SIM_GRIP_IDX
SIM_CTRL_IDX = SE.SIM_CTRL_IDX
import robocasa.utils.lerobot_utils as LU


# ---------------------------------------------------------------------------
# PATHS. Nothing here is hardcoded to one machine's layout. Resolution order:
#
#   1. explicit env var  -- REPO_ROOT / DATA_DIR / SYS1_CKPT_DIR / SYS1_RESULTS_DIR /
#                           ROBOCASA_LEROBOT_ROOT (exported by 05b-shared-bashrc.sh on the
#                           shared-filesystem setup)
#   2. derived from THIS FILE's location -- the repo is <repo_root>/openpi, so REPO_ROOT is the
#      parent of the openpi checkout. Works for /shared/openpi, ~/openpi, or anywhere else.
#   3. DATA_DIR: the first existing candidate of <repo_root>/data, <repo_root>/../data,
#      ~/data -- so the classic "repo + sibling data dir" layout keeps working untouched.
#
# Nothing assumes a shared filesystem, a particular mount name, or where venvs live: the
# interpreter running this file is whatever the caller chose (repo-local .venv, ~/venvs, conda).
_OPENPI_REPO = Path(__file__).resolve().parents[2]      # <repo_root>/openpi/examples/robocasa/..


def env_path(var: str, default: str | Path) -> Path:
    """Path from ``$var``, else ``default``. Empty string counts as unset."""
    return Path(os.environ.get(var) or default).expanduser()


def _first_existing(*cands: Path, fallback: Path) -> Path:
    for c in cands:
        if c.is_dir():
            return c
    return fallback


OPENPI_REPO = env_path("OPENPI_REPO", _OPENPI_REPO)
REPO_ROOT = env_path("REPO_ROOT", OPENPI_REPO.parent)
DATA_DIR = env_path("DATA_DIR", _first_existing(
    REPO_ROOT / "data", OPENPI_REPO.parent / "data", Path.home() / "data",
    fallback=REPO_ROOT / "data"))
SYS1_CKPT_DIR = env_path("SYS1_CKPT_DIR", DATA_DIR / "sys1_ckpts")
SYS1_RESULTS_DIR = env_path("SYS1_RESULTS_DIR", DATA_DIR / "sys1_eval_results")
ROBOCASA_DATASET = env_path("ROBOCASA_LEROBOT_ROOT", DATA_DIR / "robocasa_dataset")


def _write_json(path: Path, obj) -> None:
    # The tmp name carries the PID: the fleet runs 8 workers that all re-merge the SAME
    # index.json, and a fixed ".tmp" made them share one scratch path -- whoever renamed first
    # unlinked it out from under the others, so the losers died with FileNotFoundError on rename
    # (taking their episode down with them). Per-PID tmp names make concurrent writers
    # independent; the rename itself is still atomic, so a reader sees old or new, never partial.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=1, default=str))
        tmp.replace(path)   # atomic: a crash mid-run still leaves a valid file
    finally:
        tmp.unlink(missing_ok=True)   # never leave scratch behind on an error path


def _short_method_name(s1_dir: str | None, s2_dir: str | None, suffix: str = "") -> str:
    """Derive a compact run name that names BOTH systems, e.g.

        s1-progreg270k_s2-qwen35-4b-full-ep3-11416

    ``suffix`` tags a non-default variant onto the end (``-memory``), so a variant sweep lands in
    its OWN results dir instead of appending into the cold-plan run's numbers.

    System1 side: the progress-head tag + the *checkpoint step* (rounded to the nearest 1k), so
    sibling steps of one run get distinct names instead of colliding on the run's training scale
    (``f0717-270k-bs512-progreg_granfine_verbsimp_noexec/240000`` -> ``progreg240k``).
    System2 side: model + tuner + epoch from the run dir, plus the checkpoint step
    (``system2-full-0804-qwen35-4b-gb192-full-...-ep3/checkpoint-11416``
     -> ``qwen35-4b-full-ep3-11416``).
    Unparseable parts degrade to the raw dir name rather than silently vanishing, so a name is
    always traceable back to a checkpoint.
    """
    def s1_short(d: str | None) -> str:
        """progress head + checkpoint step + the ablation tags that still carry information.

        The scale comes from the STEP the checkpoint dir is named after, not from the ``-270k-``
        in the run name: the latter is the run's total training length and is identical for every
        step of that run, so evaluating 210000/240000/269999 would produce one name and three
        sweeps would overwrite each other's results. Steps round to the nearest 1k
        (269999 -> ``270k``), which keeps names stable for the off-by-one final checkpoints.
        Only if the dir is not a step (a run dir passed directly) do we fall back to the run
        name's scale.

        ``-noexec`` is dropped: every current checkpoint has it, so it distinguishes nothing.
        ``-noanchor``/``-noanchorstate`` (they always co-occur) collapse to a single ``-noanchor``.
        """
        if not d:
            return "s1-unknown"
        pp = Path(d)
        is_step = pp.name.isdigit()
        exp = pp.parent.name if is_step else pp.name
        head = next((h for h in ("progreg", "progact", "progcls") if h in exp), None)
        if is_step:
            scale = f"{round(int(pp.name) / 1000)}k"
        else:
            m = re.search(r"-(\d+k)-", exp)
            scale = m.group(1) if m else ""
        if not head:
            return f"s1-{exp}"
        tags = "-noanchor" if ("noanchor" in exp or "noanchorstate" in exp) else ""
        return f"s1-{head}{scale}{tags}"

    def s2_short(d: str | None) -> str:
        if not d:
            return "s2-unknown"
        pp = Path(d)
        step = pp.name.split("-")[-1] if pp.name.startswith("checkpoint-") else None
        run = pp.parent.name if step else pp.name
        model = re.search(r"(qwen[\d.]*(?:vl)?-?\d+b)", run, re.I)
        # Tuner: LoRA wins over "full". Every run dir -- LoRA ones included -- is prefixed
        # "system2-full-", so testing "full" first labelled every LoRA run as full. The rank is part
        # of the tag because runs otherwise differing only in r32/r64 would derive the same name and
        # overwrite each other's results.
        rank = re.search(r"lora-r(\d+)", run, re.I)
        tuner = f"lora{rank.group(1)}" if rank else ("lora" if "lora" in run.lower() else "full")
        ep = re.search(r"-(ep\d+)", run)
        bits = [b for b in (model.group(1).lower().replace(".", "") if model else None,
                            tuner, ep.group(1) if ep else None, step) if b]
        return "s2-" + ("-".join(bits) if bits else run)

    return f"{s1_short(s1_dir)}_{s2_short(s2_dir)}{suffix}"



def merge_combine_index(method_dir: Path) -> dict:
    """Union every index_parts/*.json into one index.json (idempotent, crash-safe).

    Safe to call from concurrent workers: each writes its OWN part file, and the merge only ever
    rewrites the aggregate. A torn/partial read of one part is skipped rather than aborting the
    merge, so a worker still mid-write cannot corrupt the index.
    """
    parts_dir = method_dir / "index_parts"
    if not parts_dir.is_dir():
        return {}
    by_ep: dict[str, dict] = {}
    for f in sorted(parts_dir.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 - a part being written right now; next merge picks it up
            continue
        for e in doc.get("episodes", []):
            if e.get("episode_id"):
                by_ep[e["episode_id"]] = e          # later part wins (a re-run overwrites)
    eps = sorted(by_ep.values(), key=lambda e: str(e.get("episode_id")))
    ok = sum(1 for e in eps if e.get("episode_success"))
    secs = [e.get("seconds") for e in eps if isinstance(e.get("seconds"), (int, float))]
    idx = {
        "eval_kind": "combine", "method": method_dir.name,
        "n_episodes": len(eps), "n_success": ok,
        "success_rate": (ok / len(eps)) if eps else None,
        "total_seconds": round(sum(secs), 1) if secs else None,
        "avg_seconds_per_episode": round(sum(secs) / len(secs), 2) if secs else None,
        "episodes": eps,
    }
    _write_json(method_dir / "index.json", idx)
    return idx


# ---------------------------------------------------------------------------
# VENDORED BENCHMARK TABLES (inlined so the eval is a single self-contained file --
# no sidecar JSON to keep in sync or forget to commit).
#
# TARGET_MAX_SUBGOAL_TURNS: per-task worst-case number of System2 subgoal turns, measured
# over the target split (System2 expected_results). "subgoal turn" EXCLUDES the task_begin
# unroll and the task_finish turn -- hence TURN_DEF_OFFSET below. Source:
# robocasa_target_maxturns.json (50 tasks, max_subgoal_turns 2..35).
TARGET_MAX_SUBGOAL_TURNS: dict[str, int] = {
    'ArrangeBreadBasket': 15,
    'ArrangeTea': 16,
    'BreadSelection': 10,
    'CategorizeCondiments': 10,
    'CloseBlenderLid': 8,
    'CloseFridge': 6,
    'CloseToasterOvenDoor': 2,
    'CoffeeSetupMug': 5,
    'CuttingToolSelection': 7,
    'DeliverStraw': 13,
    'GarnishPancake': 14,
    'GatherTableware': 17,
    'GetToastedBread': 11,
    'HeatKebabSandwich': 22,
    'KettleBoiling': 8,
    'LoadDishwasher': 13,
    'MakeIceLemonade': 21,
    'NavigateKitchen': 2,
    'OpenCabinet': 5,
    'OpenDrawer': 2,
    'OpenStandMixerHead': 2,
    'PackIdenticalLunches': 35,
    'PanTransfer': 9,
    'PickPlaceCounterToCabinet': 7,
    'PickPlaceCounterToStove': 6,
    'PickPlaceDrawerToCounter': 9,
    'PickPlaceSinkToCounter': 6,
    'PickPlaceToasterToCounter': 6,
    'PortionHotDogs': 26,
    'PreSoakPan': 13,
    'PrepareCoffee': 9,
    'RecycleBottlesByType': 14,
    'RinseSinkBasin': 8,
    'ScrubCuttingBoard': 7,
    'SearingMeat': 17,
    'SeparateFreezerRack': 17,
    'SetUpCuttingStation': 17,
    'SlideDishwasherRack': 2,
    'StackBowlsCabinet': 8,
    'SteamInMicrowave': 17,
    'StirVegetables': 16,
    'StoreLeftoversInBowl': 15,
    'TurnOffStove': 3,
    'TurnOnElectricKettle': 2,
    'TurnOnMicrowave': 2,
    'TurnOnSinkFaucet': 2,
    'WaffleReheat': 13,
    'WashFruitColander': 20,
    'WashLettuce': 8,
    'WeighIngredients': 8,
}

# TARGET_EVAL_EPISODES: the 1500-episode eval manifest (50 tasks x 30 episodes), task ->
# episode indices. 30 PER TASK to match the leaderboard denominator (it was 20; the first 20 of
# every task are unchanged, so the 1000-episode runs recorded at 20 remain a valid prefix and
# --resume only has to run the 10 new episodes per task).
#
# Regenerated from robocasa_target_episode_full.json, which enumerates every successfully
# extracted target-split episode (24637 across the 50 tasks, 140-543 per task). Indices are the
# SOURCE LeRobot indices and are SPARSE -- only extracted episodes are present -- so they are not
# contiguous 0..N-1 (CloseBlenderLid starts 4, 6, 7, 10, ...). Taking the first 30 of the sorted
# list is what "the first 30 episodes" means here.
TARGET_EVAL_EPISODES: dict[str, list[int]] = {
    'ArrangeBreadBasket': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'ArrangeTea': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'BreadSelection': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'CategorizeCondiments': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'CloseBlenderLid': [
        4, 6, 7, 10, 11, 17, 24, 26, 27, 28, 29, 38, 42, 43, 48, 55, 58, 60, 67, 72, 74, 77, 78, 79, 81, 83, 86,
        89, 91, 92],
    'CloseFridge': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'CloseToasterOvenDoor': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'CoffeeSetupMug': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'CuttingToolSelection': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'DeliverStraw': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'GarnishPancake': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'GatherTableware': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'GetToastedBread': [
        0, 1, 2, 3, 4, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37,
        38, 39],
    'HeatKebabSandwich': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'KettleBoiling': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'LoadDishwasher': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'MakeIceLemonade': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'NavigateKitchen': [
        1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34,
        37, 38],
    'OpenCabinet': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'OpenDrawer': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'OpenStandMixerHead': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PackIdenticalLunches': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PanTransfer': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PickPlaceCounterToCabinet': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PickPlaceCounterToStove': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PickPlaceDrawerToCounter': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PickPlaceSinkToCounter': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PickPlaceToasterToCounter': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PortionHotDogs': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PreSoakPan': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'PrepareCoffee': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'RecycleBottlesByType': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'RinseSinkBasin': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'ScrubCuttingBoard': [
        0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        29, 30],
    'SearingMeat': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'SeparateFreezerRack': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'SetUpCuttingStation': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'SlideDishwasherRack': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'StackBowlsCabinet': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'SteamInMicrowave': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'StirVegetables': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28,
        29, 30],
    'StoreLeftoversInBowl': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'TurnOffStove': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'TurnOnElectricKettle': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'TurnOnMicrowave': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'TurnOnSinkFaucet': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'WaffleReheat': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'WashFruitColander': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'WashLettuce': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
    'WeighIngredients': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29],
}

# task -> split (atomic_seen / composite_seen / composite_unseen), same source.
TARGET_TASK_SPLIT: dict[str, str] = {
    'ArrangeBreadBasket': 'composite_unseen',
    'ArrangeTea': 'composite_unseen',
    'BreadSelection': 'composite_unseen',
    'CategorizeCondiments': 'composite_unseen',
    'CloseBlenderLid': 'atomic_seen',
    'CloseFridge': 'atomic_seen',
    'CloseToasterOvenDoor': 'atomic_seen',
    'CoffeeSetupMug': 'atomic_seen',
    'CuttingToolSelection': 'composite_unseen',
    'DeliverStraw': 'composite_seen',
    'GarnishPancake': 'composite_unseen',
    'GatherTableware': 'composite_unseen',
    'GetToastedBread': 'composite_seen',
    'HeatKebabSandwich': 'composite_unseen',
    'KettleBoiling': 'composite_seen',
    'LoadDishwasher': 'composite_seen',
    'MakeIceLemonade': 'composite_unseen',
    'NavigateKitchen': 'atomic_seen',
    'OpenCabinet': 'atomic_seen',
    'OpenDrawer': 'atomic_seen',
    'OpenStandMixerHead': 'atomic_seen',
    'PackIdenticalLunches': 'composite_seen',
    'PanTransfer': 'composite_unseen',
    'PickPlaceCounterToCabinet': 'atomic_seen',
    'PickPlaceCounterToStove': 'atomic_seen',
    'PickPlaceDrawerToCounter': 'atomic_seen',
    'PickPlaceSinkToCounter': 'atomic_seen',
    'PickPlaceToasterToCounter': 'atomic_seen',
    'PortionHotDogs': 'composite_unseen',
    'PreSoakPan': 'composite_seen',
    'PrepareCoffee': 'composite_seen',
    'RecycleBottlesByType': 'composite_unseen',
    'RinseSinkBasin': 'composite_seen',
    'ScrubCuttingBoard': 'composite_seen',
    'SearingMeat': 'composite_seen',
    'SeparateFreezerRack': 'composite_unseen',
    'SetUpCuttingStation': 'composite_seen',
    'SlideDishwasherRack': 'atomic_seen',
    'StackBowlsCabinet': 'composite_seen',
    'SteamInMicrowave': 'composite_seen',
    'StirVegetables': 'composite_seen',
    'StoreLeftoversInBowl': 'composite_seen',
    'TurnOffStove': 'atomic_seen',
    'TurnOnElectricKettle': 'atomic_seen',
    'TurnOnMicrowave': 'atomic_seen',
    'TurnOnSinkFaucet': 'atomic_seen',
    'WaffleReheat': 'composite_unseen',
    'WashFruitColander': 'composite_unseen',
    'WashLettuce': 'composite_seen',
    'WeighIngredients': 'composite_unseen',
}


# ---------------------------------------------------------------------------
# Per-task System2 TURN BUDGET.
#
# A flat --max-turns is wrong in both directions: per-task max_subgoal_turns ranges 2
# (TurnOnSinkFaucet) .. 35 (PackIdenticalLunches). A flat 20 STARVED the 12 long composites (they
# could never finish) and over-budgeted the other 36 by up to 3x, which inflates wall-clock and lets
# a stuck episode grind. So:
#
#     max_turns(task) = TARGET_MAX_SUBGOAL_TURNS[task] + TURN_HEADROOM + TURN_DEF_OFFSET
#
# TURN_DEF_OFFSET exists because the table and this loop COUNT DIFFERENTLY. The table counts
# "execution turns with a real <subgoal>", EXCLUDING the task_begin unroll and the task_finish turn;
# our ``turn`` counter is 0-indexed over EVERY System2 exec call, which includes both. Ignoring this
# would silently eat 2 of the headroom.
TURN_HEADROOM = 5      # slack over the observed worst case
TURN_DEF_OFFSET = 2    # +1 task_begin unroll, +1 possible task_finish turn


def _turn_bonus() -> dict[str, int]:
    """Per-task turn-budget bonus, from SYS2_TURN_BONUS="Task:2,Other:3". Empty unless set.

    A turn bonus cannot be a sys2_rules rule: max_turns is resolved ONCE before the turn loop, so
    nothing a rule returns mid-episode can change it. This is the experimental hook for "give task X
    +2 turns" -- inert when the variable is unset, and the bonus is written into max_turns_reason so
    every episode.json records whether it was applied.
    """
    out: dict[str, int] = {}
    for part in (os.environ.get("SYS2_TURN_BONUS") or "").split(","):
        part = part.strip()
        if not part:
            continue
        task, _, delta = part.partition(":")
        try:
            out[task.strip()] = int(delta)
        except ValueError:
            print(f"WARNING: ignoring bad SYS2_TURN_BONUS entry {part!r}", flush=True)
    return out


def max_turns_for(task_name: str, flat_max: int, *, dynamic: bool = True) -> tuple[int, str]:
    """Return (max_turns, why) for a task.

    Falls back to the flat cap for any task absent from the table (e.g. a non-target task), so an
    unknown task is never silently given a tiny budget.
    """
    bonus = _turn_bonus().get(task_name, 0)
    if not dynamic:
        return flat_max + bonus, "flat" + (f" +{bonus} bonus" if bonus else "")
    n = TARGET_MAX_SUBGOAL_TURNS.get(task_name)
    if n is None:
        return flat_max + bonus, "flat (task not in budget table)" + (f" +{bonus} bonus" if bonus else "")
    return (n + TURN_HEADROOM + TURN_DEF_OFFSET + bonus,
            f"max_subgoal_turns {n} + {TURN_HEADROOM} headroom + {TURN_DEF_OFFSET} def-offset"
            + (f" +{bonus} bonus" if bonus else ""))


def _episode_done(method_dir: Path, lerobot_dir: Path, ep_index: int) -> bool:
    """True if this episode already has a COMPLETE result under ``method_dir``.

    Complete means episode.json exists, carries a ``termination``, and has no ``error`` -- so a
    crashed or half-written episode is re-run rather than silently kept. Used by --resume to make a
    killed sweep restartable without redoing finished work.
    """
    task = _task_name_from_lerobot_dir(lerobot_dir)
    split = _split_from_lerobot_dir(lerobot_dir)
    flat = f"{task}/{split}/episode_{ep_index:06d}".replace("/", "__")
    f = method_dir / flat / "episode.json"
    if not f.exists():
        return False
    try:
        doc = json.loads(f.read_text())
    except Exception:  # noqa: BLE001 - torn write: treat as not done
        return False
    return bool(doc.get("termination")) and not doc.get("error")


def _parse_episodes(spec: str) -> list[int]:
    """'0' | '0,1' | '0-4' | '0,3-5' -> [0] | [0,1] | [0,1,2,3,4] | [0,3,4,5]."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def _task_name_from_lerobot_dir(ld: Path) -> str:
    """Pull the RoboCasa task name out of .../<split>/<category>/<Task>/<date>/lerobot."""
    parts = [p for p in Path(ld).parts if p]
    for i, p in enumerate(parts):
        if p == "lerobot" and i >= 2:
            return parts[i - 2]
    return Path(ld).name


def _split_from_lerobot_dir(ld: Path) -> str:
    """'target' / 'pretrain' from .../v1.0/<split>/... (falls back to 'unknown')."""
    for p in Path(ld).parts:
        if p in ("target", "pretrain"):
            return p
    return "unknown"


def _timing_summary(t: dict[str, list[float]]) -> dict:
    """Collapse per-call timing samples into {n, total_s, mean_s, p50_s, max_s} per label.

    Every expensive call in the loop is timed separately (System2 vLLM request, System1 policy
    request, observation assembly, MuJoCo render, MuJoCo step, env reset, video encode) so a slow
    run can be attributed to the right component instead of guessed at.
    """
    out = {}
    for k, v in t.items():
        if not v:
            continue
        s = sorted(v)
        out[k] = {
            "n": len(v), "total_s": round(sum(v), 3), "mean_s": round(sum(v) / len(v), 4),
            "p50_s": round(s[len(s) // 2], 4), "max_s": round(s[-1], 4),
        }
    return out


# Privileged "Current gripper status" tail length. MUST match the System2 training producer:
# RoboAnnotator/robo_annotator/system2/v2_rows.py::_gripper_status uses _GRIP_TAIL = 5.
GRIP_TAIL = 5


def _gripper_status(grip_cmds: list[float]) -> str | None:
    """The privileged 'Current gripper status' line: open / close / unsure.

    EXACT parity with the System2 training producer
    (``RoboAnnotator/robo_annotator/system2/v2_rows.py::_gripper_status``): look at the binary
    ``gripper_close`` ACTION channel (+1 close / -1 open) over the LAST ``GRIP_TAIL`` frames of the
    clip just executed and require unanimity — all > 0 -> "close", all < 0 -> "open", any
    disagreement -> "unsure" (i.e. the gripper was still transitioning across the tail).

    A single-step delta is NOT equivalent: a command that flipped 3 steps before the end is still
    mid-transition over the tail and must read "unsure", which a last-vs-previous comparison would
    wrongly report as settled.

    Returns None when there is no executed step yet (turn 0), so the caller can omit/choose.
    """
    if not grip_cmds:
        return None
    tail = [float(g) for g in grip_cmds[-GRIP_TAIL:]]
    if all(g > 0 for g in tail):
        return "close"
    if all(g < 0 for g in tail):
        return "open"
    return "unsure"


def do_plan_cold(s2_client, instruction: str, plan_dir: Path, args) -> dict:
    """COLD plan step: one tiled opening still -> <plan> milestone checklist.

    Factored out (and swappable) so a plan VARIANT can be evaluated without duplicating the
    execution loop: set ``args.plan_fn`` to another callable with this signature and
    ``eval_episode`` uses it instead (see ``combine_memory_eval.py``, which first narrates a demo
    video into a recipe and then issues a with-memory plan). Contract:

        (s2_client, instruction, plan_dir, args) -> {"plan": str, "doc": dict, "variant": str}

    ``plan`` is the checklist handed to the exec loop, ``doc`` is what lands in
    ``episode.json["plan"]``, and the callee owns everything it writes under ``plan_dir``.
    """
    plan_dir.mkdir(parents=True, exist_ok=True)
    # The planner READS this image, so it stays at model resolution; a small copy is written
    # alongside for the GUI.
    scene_full = plan_dir / "scene_full.png"
    img0 = S2C.write_image(args.scene0, scene_full)
    S2C.write_image(S2C.downscale([args.scene0])[0], plan_dir / "scene.png")
    p = s2_client.plan_cold(instruction, scene_full)
    plan = (p.get("plan") or "").strip()
    _write_json(plan_dir / "plan.json", {
        "mode": "plan_cold",
        "s2_system_prompt": S2C.SYS_PLAN_COLD,
        "s2_user_prompt": S2C.user_plan_cold(instruction),
        "s2_response_raw": p["raw"], "thought": p.get("thought"), "plan": plan,
        "media": {"image": img0}, "latency_s": p.get("latency_s"), "usage": p.get("usage"),
    })
    return {
        "plan": plan,
        "variant": "cold",
        "doc": {"thought": p.get("thought"), "plan": plan, "latency_s": p.get("latency_s"),
                "dir": plan_dir.name},
    }


def run_s1_segment(
    env, s1_client, *, subgoal_text: str, task_goal: str, est_length: int,
    base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state, resize: int,
    replan_steps: int, budget: int, stop_cfg: StopConfig, norm_stats,
    last_cmd_grip_init: float, zero_arm_in_base: bool, act_override: dict | None = None,
    force_steps: int = 0,
) -> dict:
    """Roll System1 on ONE System2 subgoal until the stop rule fires or the budget runs out.

    Mirrors episode_eval's rollout but is driven by a System2 subgoal + estimated_step instead of
    an annotated span, and records everything the GUI needs. ``executed_step`` starts at 0 for
    every call (per-turn reset) and the anchor is whatever the caller snapshotted at segment start.

    ``force_steps`` > 0 SUPPRESSES BOTH STOP RULES for the first ``force_steps`` executed steps, so
    the segment runs at least that long. It exists for genuine WAIT subgoals, where the stop rule is
    structurally wrong rather than merely mistuned: a waiting arm commands no motion, so action
    quiescence is satisfied on the very first window and the segment ends while the physical process
    (a toaster) is still running. The DENSE ``_check_success()`` break is deliberately NOT
    suppressed -- if the task completes mid-wait the episode still ends there, which is the whole
    point of waiting. ``stop_reason`` is reported as "forced_steps" when the force window is what
    kept the segment alive.
    """
    q01s = q99s = q01a = q99a = None
    if norm_stats:
        if "state" in norm_stats:
            q01s = np.asarray(norm_stats["state"]["q01"], np.float32)
            q99s = np.asarray(norm_stats["state"]["q99"], np.float32)
        if "actions" in norm_stats:
            q01a = np.asarray(norm_stats["actions"]["q01"], np.float32)
            q99a = np.asarray(norm_stats["actions"]["q99"], np.float32)

    clean_frames: list[np.ndarray] = []      # tiled 256x768 RGB, one per executed step
    step_records: list[dict] = []
    motion_norms: list[float] = []           # per-step commanded-motion norm (for clip compaction)
    grip_cmds: list[float] = []               # per-step commanded gripper (+1 close / -1 open)
    grip_widths: list[float] = []              # per-step OBSERVED gripper width (pad distance, m)
    prompts_seen: list[str] = []             # every distinct S1 prompt string the server tokenized

    timings: dict[str, list[float]] = {
        "s1_infer": [], "s1_obs_build": [], "env_render": [], "env_step": [],
        "env_success_check": [],
    }
    tracker = StopTracker(cfg=stop_cfg)
    tracker.set_est_length(est_length)   # progreg relaxes its threshold by est_length
    last_cmd_grip = float(last_cmd_grip_init)
    prev_cmd_grip: float | None = None
    executed = 0
    success_step: int | None = None      # step at which _check_success() first went true
    stop_reason = "budget"
    stop_debug: dict | None = None
    import collections

    action_plan: collections.deque = collections.deque()

    while executed < budget:
        replanned = False
        prog = None
        query = None
        if not action_plan:
            replanned = True
            gripper_flag = "Close" if last_cmd_grip > 0 else "Open"
            _t = time.perf_counter()
            element = SE._obs_dict(
                env, base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                anchor_imgs=anchor_imgs, anchor_state=anchor_state,
                subgoal_text=subgoal_text, task_goal=task_goal, est_length=est_length,
                executed_step=executed, gripper_flag=gripper_flag, resize=resize)
            t_obs_build = time.perf_counter() - _t
            _t = time.perf_counter()
            result = s1_client.infer(element)
            t_infer = time.perf_counter() - _t
            timings["s1_infer"].append(t_infer)
            timings["s1_obs_build"].append(t_obs_build)
            chunk = np.asarray(result["actions"])                 # (H,12) LeRobot order
            chunk_sim = np.stack([lerobot_action_to_sim(a) for a in chunk], axis=0)
            # ACTION OVERRIDE (sys2_rules.action_overrides). Applied to the WHOLE chunk here, not
            # at step time, so the executed actions, the recorded chunk, the motion norms and the
            # lookahead stop all reason about the same values -- otherwise the stop criterion would
            # judge a chunk that was never executed.
            if act_override and act_override.get("grip") is not None:
                chunk_sim[:, SIM_GRIP_IDX] = float(act_override["grip"])
            prog = SE._read_progress(result)
            if prog:
                prog["at_step"] = executed
            action_plan.extend(chunk_sim[:replan_steps])
            # The REAL prompt the server tokenized (true discretized state ints) when available.
            real_prompt = result.get("prompt_text") if isinstance(result, dict) else None
            if real_prompt and (not prompts_seen or prompts_seen[-1] != real_prompt):
                prompts_seen.append(real_prompt)
            # Record the WHOLE predicted action chunk (not just the steps we execute) plus its
            # per-step motion delta, so the GUI can show what System1 planned vs what actually ran:
            # only the first `replan_steps` of the H-step chunk are executed before the next replan.
            chunk_motion = []
            _pg = last_cmd_grip
            for a in chunk_sim:
                chunk_motion.append(round(float(action_eef_base_norm(a, prev_grip=_pg)), 5))
                _pg = float(a[SIM_GRIP_IDX])
            query = dict(prompt=real_prompt, gripper_flag=gripper_flag,
                         executed_step=int(executed), replan_steps=int(replan_steps),
                         horizon=int(HORIZON), s1_infer_s=round(t_infer, 4),
                         chunk_progress=(prog.get("progress_chunk") if prog else None),
                         progress_now=(prog.get("progress_now") if prog else None),
                         chunk_raw12=np.round(chunk_sim, 4).tolist(),
                         chunk_motion=chunk_motion)

        # LOOKAHEAD stop: if the chunk we just received commands no motion for the next `window`
        # steps and progress is at threshold (and the gripper has settled, if it flipped), the
        # subgoal is done -- executing those quiescent actions would only burn sim steps + renders.
        if replanned and executed >= force_steps and tracker.should_stop_lookahead(chunk_sim):
            stop_reason = "stop_rule_lookahead"
            stop_debug = tracker.debug()
            break

        action_sim = action_plan.popleft()
        prev_cmd_grip = last_cmd_grip
        last_cmd_grip = float(action_sim[SIM_GRIP_IDX])

        _t = time.perf_counter()
        obs = env._get_observations(force_update=True)
        t_render = time.perf_counter() - _t
        timings["env_render"].append(t_render)
        fld = dict(
            frame_step=int(executed), phase="act", subgoal=subgoal_text, task_goal=task_goal,
            est_length=int(est_length), budget=int(budget),
            action_raw12=np.round(np.asarray(action_sim, float), 4).tolist(),
            cur_raw16=np.round(raw_state_from_obs(obs), 4).tolist(),
            action_eef_pos_norm=round(float(np.linalg.norm(action_sim[0:3])), 4),
            action_eef_rot_norm=round(float(np.linalg.norm(action_sim[3:6])), 4),
            action_base_norm=round(float(np.linalg.norm(action_sim[7:11])), 4),
            gripper_flag=("Close" if last_cmd_grip > 0 else "Open"),
            progress=SE._progress_str(prog) if prog else "-", progress_raw=prog,
            replanned=bool(replanned),
        )
        if query is not None:
            fld["query"] = query
        grip_cmds.append(float(action_sim[SIM_GRIP_IDX]))
        # Gripper STATE (finger pad distance) and its per-step delta -- tracked separately from the
        # arm/base motion norm, and plotted as its own GUI curve.
        gw = gripper_width(fld["cur_raw16"])
        fld["grip_width"] = round(gw, 5) if gw is not None else None
        fld["grip_width_delta"] = (round(abs(gw - grip_widths[-1]), 5)
                                   if (gw is not None and grip_widths) else 0.0)
        if gw is not None:
            grip_widths.append(gw)
        mnorm = action_eef_base_norm(action_sim)
        fld["motion_norm"] = round(float(mnorm), 5)
        fld["t_env_render_s"] = round(t_render, 4)
        motion_norms.append(float(mnorm))
        clean_frames.append(SE._stacked_from_obs(obs))
        step_records.append(fld)

        a_step = action_sim
        if zero_arm_in_base and action_sim[SIM_CTRL_IDX] > 0.0:
            a_step = action_sim.copy()
            a_step[0:6] = 0.0
        _t = time.perf_counter()
        env.step(a_step)
        t_step = time.perf_counter() - _t
        timings["env_step"].append(t_step)
        fld["t_env_step_s"] = round(t_step, 4)
        executed += 1

        # DENSE success check, every step — this is how the RoboCasa benchmark scores a rollout
        # (see robocasa/utils/eval_utils.py::run_random_rollouts: `if env._check_success(): break`).
        # Sampling only once per segment would keep driving after the task is already solved and
        # could even undo it (e.g. push a closed door back open) before anyone looked.
        _t = time.perf_counter()
        env_ok = bool(env._check_success())
        timings["env_success_check"].append(time.perf_counter() - _t)
        fld["sim_check_success"] = env_ok
        if env_ok:
            success_step = executed - 1
            stop_reason = "env_success"
            break

        tracker.update(action_sim, prog, raw16=fld["cur_raw16"])
        fld["stop_signals"] = tracker.debug()
        # The tracker is still UPDATED inside the force window (above), so its history is continuous
        # and the recorded stop_signals stay truthful -- only the decision to break is withheld.
        if executed < force_steps:
            fld["forced"] = True
            stop_reason = "forced_steps"
            continue
        if tracker.should_stop():
            stop_reason = "stop_rule"
            stop_debug = tracker.debug()
            break

    return {
        "n_steps": executed, "stop_reason": stop_reason, "success_step": success_step,
        "stop_signals": stop_debug or tracker.debug(),
        "progress_thresh": tracker.progress_threshold(),
        "progress_done": bool(tracker.progress_done()), "quiescent": bool(tracker.quiescent()),
        "last_cmd_grip": last_cmd_grip, "prev_cmd_grip": prev_cmd_grip,
        "grip_cmds": grip_cmds,
        "s1_prompts": prompts_seen,
        "timings": _timing_summary(timings),
        "_clean_frames": clean_frames, "_step_records": step_records, "_motion": motion_norms,
    }


def eval_episode(episode_dir: Path, s1_client, s2_client: S2C.Sys2Client, args,
                 out_root: Path, method: str, norm_stats=None) -> dict:
    """Full closed-loop episode. Writes every artifact under <out_root>/<method>/<flat_ep>/.

    ``episode_dir`` is a raw LeRobot dataset dir and ``args`` carries the episode index. NO
    annotation files are read: the task goal comes from the dataset's own ``ep_meta['lang']`` and
    success from the env's ``_check_success()``. Annotations were training data for System1 and
    System2; at rollout time System2 supplies the subgoals and System1 the actions.
    """
    t0 = time.time()
    ld = Path(episode_dir)
    ep_index = int(args.episode_index)
    _t = time.perf_counter()
    env = make_camera_env(str(ld))
    t_env_make = time.perf_counter() - _t
    ep_meta = LU.get_episode_meta(ld, ep_index)
    task_name = _task_name_from_lerobot_dir(ld)
    instruction = (ep_meta.get("lang") or "").strip()
    episode_id = f"{task_name}/{_split_from_lerobot_dir(ld)}/episode_{ep_index:06d}"
    flat = episode_id.replace("/", "__")
    ep_out = out_root / method / flat
    ep_out.mkdir(parents=True, exist_ok=True)

    stop_cfg = StopConfig(progress_thresh=args.stop_progress, eps=args.stop_eps,
                          window=args.stop_window)
    doc: dict = {
        "episode_id": episode_id, "task_name": task_name, "instruction": instruction,
        "method": method, "eval_kind": "combine",
        "lerobot_dir": str(ld), "episode_index": ep_index,
        "rule_config": _rule_config(args),
        "config": {
            "reset_mode": "reset_to(states[0]) from the raw LeRobot episode (no annotations)",
            "s1_port": args.s1_port, "s2_port": args.s2_port, "s2_model": args.s2_model,
            "s1_ckpt": args.s1_dir, "s2_ckpt": args.s2_dir,
            "horizon_mult": args.horizon_mult, "max_steps_cap": args.max_steps_cap,
            "replan_steps": args.replan_steps, "max_turns_flat": args.max_turns,
            "stop": {"progress": args.stop_progress, "eps": args.stop_eps, "window": args.stop_window},
            "clip": {"fps": S2C.CLIP_FPS, "crf": S2C.CLIP_CRF, "tile": list(S2C.TILE_HW),
                     "sim_fps": S2C.SIM_FPS, "static_eps": args.static_eps},
            "video_policy_source": S2C.policy_source(),
            "privileged_task_status": True,   # fed from env._check_success(); GT bit by design
            # Overwritten by the plan step with the variant it actually ran ("cold" / "memory").
            "plan_variant": "cold",
        },
        "turns": [],
    }

    # Per-EPISODE rule state (skip counters) and the running intervention log. Both are
    # per-episode by construction: a fresh dict here means one episode's skips can never
    # leak into the next.
    rule_state: dict = {}
    ep_rule_log: list[dict] = []
    ep_timings: dict[str, list[float]] = {
        "env_make": [], "env_reset": [], "s2_plan": [], "s2_exec": [], "video_encode": [],
    }
    ep_timings["env_make"].append(t_env_make)
    try:
        if not instruction:
            raise ValueError(f"episode {ep_index} has no ep_meta['lang'] task goal")
        states = LU.get_episode_states(ld, ep_index)
        # Reset to the FIRST frame of the recorded episode (states[0]) — the episode's own start
        # scene. The recorded actions are never read: System1 generates every action.
        init = dict(states=states[0],
                    model=LU.get_episode_model_xml(ld, ep_index),
                    ep_meta=json.dumps(ep_meta))
        _t = time.perf_counter()
        reset_to(env, init)
        ep_timings["env_reset"].append(time.perf_counter() - _t)
        obs0 = env._get_observations(force_update=True)
        base_pos_ref, base_yaw_ref = base_reference(obs0)
        doc["n_recorded_frames"] = int(len(states))

        # ---------------- PLAN ----------------
        # Cold by default; ``args.plan_fn`` swaps in a variant (e.g. the memory/recipe planner)
        # without touching the execution loop below. ``args.scene0`` is the tiled opening still the
        # planner reads.
        args.scene0 = SE._stacked_from_obs(obs0)
        plan_dir = ep_out / "plan"
        _t = time.perf_counter()
        res = getattr(args, "plan_fn", None) or do_plan_cold
        res = res(s2_client, instruction, plan_dir, args)
        ep_timings["s2_plan"].append(time.perf_counter() - _t)
        plan = (res.get("plan") or "").strip()
        doc["plan"] = res.get("doc") or {}
        doc["config"]["plan_variant"] = res.get("variant", "cold")
        if not plan:
            raise ValueError("System2 returned no <plan>")
        # PLAN-MODE rules: the one place a rule can replace the checklist itself. Runs once, here;
        # during the exec loop the plan is then maintained as usual (System2's plan_update marks
        # progress), so no rule has to reconstruct the marks. episode.json["plan"]["s2_plan_before_
        # rules"] keeps what System2 actually emitted, so the override is always auditable.
        # Every plan rule is gated on one task, so this is the TASK tier -- a general-only arm does
        # not get it.
        if args.task_rules:
            pr = SR.apply_plan_rules(task_name, plan=plan)
            # Apply the revision whenever the plan actually changed -- NOT only when the rule also
            # reported an intervention. Gating the assignment on ``interventions`` silently discarded
            # the override of any rule that forgot to log one, which is a void experiment that still
            # looks like a clean run.
            if pr["plan"] != plan:
                doc["plan"]["s2_plan_before_rules"] = plan
                plan = pr["plan"]
                doc["plan"]["plan_after_rules"] = plan
            if pr["interventions"]:
                ep_rule_log.append({"turn": "plan", "interventions": pr["interventions"]})
                for _iv in pr["interventions"]:
                    print(f"  RULE [{_iv['kind']}] {_iv['rule']}: plan-mode override", flush=True)
            elif pr["plan"] != doc["plan"].get("s2_plan_before_rules", plan):
                print("  WARNING: a plan-mode rule changed the plan without logging an "
                      "intervention -- applied anyway, but fix the rule", flush=True)

        # ---------------- EXEC loop ----------------
        last_cmd_grip = 0.0
        # Commanded gripper values from the segment just executed; the last GRIP_TAIL of these
        # decide the privileged "Current gripper status" (see _gripper_status).
        seg_grip_cmds: list[float] = []
        turn = 0
        term = "max_turns"
        env_success = bool(env._check_success())
        # The clip System2 watches on turn N is the one PRODUCED by turn N-1's System1 segment, so
        # it lives in the PREVIOUS turn's dir. Carried forward explicitly instead of being rebuilt
        # from `tdir` (which points at the current, still-empty turn).
        prev_clip_path: Path | None = None
        prev_clip_frames = 0
        clip_info = None
        clip_stats = None
        n_clip_frames = 0
        max_turns, mt_why = max_turns_for(task_name, args.max_turns,
                                          dynamic=not args.flat_max_turns)
        doc["max_turns"] = max_turns
        doc["max_turns_reason"] = mt_why
        print(f"  turn budget: {max_turns} ({mt_why})", flush=True)

        while turn < max_turns:
            tdir = ep_out / f"turn{turn:02d}"
            tdir.mkdir(parents=True, exist_ok=True)
            task_status = "finished" if env_success else "ongoing"
            # Turn 0 has no executed step yet: fall back to the gripper actually commanded by the
            # reset state rather than assuming "open".
            grip_status = _gripper_status(seg_grip_cmds) or ("close" if last_cmd_grip > 0 else "open")

            # -- System2: what next? --
            # HELD TURN. A rule borrowed the PREVIOUS turn (a re-grasp, an extra carry), so the
            # subgoal System2 proposed then has not been executed yet. There is nothing for it to
            # judge and nothing new to decide, so it is NOT queried: the held subgoal runs against the
            # unchanged plan, and the next real query sees the clip of THAT segment.
            #
            # Querying anyway was a defect, not merely a waste. The answer had to be discarded by
            # tx_resume, but its <plan_update> was still applied -- so the checklist advanced past a
            # step System1 had not finished, and the fine step System2 proposed was SWALLOWED:
            # PackIdenticalLunches ep0 proposed "search for the counter" (M14.1) on the resume turn,
            # it was thrown away, and the next turn marked M14.1 [x] though it never ran.
            # UNCONDITIONAL: pending_resume returns None unless a rule actually borrowed the
            # previous turn, so the gate was redundant when the whole layer was one switch -- and it
            # became a BUG once a borrow-a-turn rule moved to the general tier. regrasp_recovery holds
            # System2's subgoal and resumes it next turn; with this gated on --task-rules, a general-
            # only arm would borrow the turn and then never resume, silently dropping the held subgoal
            # and advancing the checklist past a step System1 never ran (the PackIdenticalLunches ep0
            # failure described above, which is exactly what the hold exists to prevent).
            held = SR.pending_resume(rule_state)
            _t = time.perf_counter()
            if held:
                s2 = {"held_by_rule": held["rule"], "subgoal": held["subgoal"],
                      "subgoal_detail": held.get("subgoal_detail") or held["subgoal"],
                      "estimated_step": held.get("est"), "judge": None,
                      "thought": (f"System2 NOT queried: {held['rule']} borrowed turn {turn - 1}, so "
                                  "the subgoal it proposed there runs now against the same plan."),
                      "plan_update": None}
                s2_media = {"held_by_rule": held["rule"]}
                s2_user = None
                plan_in = plan          # unchanged: no plan_update is applied on a held turn
                clip_stats = None
            elif turn == 0:
                cur = env._get_observations(force_update=True)
                tile0 = SE._stacked_from_obs(cur)
                S2C.write_image(tile0, tdir / "s2_input_scene_full.png")     # what the model reads
                sc = S2C.write_image(S2C.downscale([tile0])[0], tdir / "s2_input_scene.png")
                s2 = s2_client.exec_first(instruction, plan, tdir / "s2_input_scene_full.png")
                s2_media = {"image": sc}
                s2_user = S2C.user_exec_first(instruction, plan)
                # The plan actually HANDED TO System2 for this call. Recorded because it cannot be
                # reconstructed from the previous turn's plan_after once a rule revises the plan and
                # re-queries within the same turn (repeat_cap's milestone close, coffee_skip_failed):
                # the GUI used to derive "plan handed in" from turn-1 and so showed the pre-rule plan.
                plan_in = plan
                clip_stats = None
            else:
                if prev_clip_path is None or not prev_clip_path.exists():
                    raise RuntimeError(
                        f"turn {turn} has no clip from the previous segment "
                        f"(expected {prev_clip_path}) — System1 produced no frames")
                s2 = s2_client.exec_turn(instruction, plan, prev_clip_path,
                                         prev_clip_frames, task_status, grip_status)
                s2_media = {"video": clip_info, "clip_stats": clip_stats,
                            "clip_path": str(prev_clip_path), "clip_from_turn": turn - 1,
                            "model_res": list(S2C.TILE_HW), "display_res": list(S2C.DISPLAY_HW)}
                s2_user = S2C.user_exec_turn(instruction, plan, task_status, grip_status)
                plan_in = plan
            t_s2 = time.perf_counter() - _t
            ep_timings["s2_exec"].append(t_s2)

            # No plan_update on a held turn -- there was no System2 call to produce one, and the plan
            # must stay exactly as it was so the held subgoal is still the current step.
            if not held:
                plan = S2C.apply_plan_update(plan, s2.get("plan_update"))
            # PLAN-EXHAUSTED cutoff, re-evaluated against the LATEST plan every turn: while System2
            # keeps sitting on the last fine step the run grows; the moment it appends a new step the
            # run RESETS, because there is fresh work planned.
            judge = s2.get("judge")
            subgoal = (s2.get("subgoal") or "").strip()
            sg_detail = (s2.get("subgoal_detail") or "").strip()
            est = s2.get("estimated_step")

            # ---- THE RULE LAYER, IN TWO TIERS -----------------------------------------------
            # MANDATORY (repeat_cap) runs on EVERY run, with or without --task-rules: it is the
            # loop's termination policy, not a revision of System2's output. Without it a stuck
            # subgoal is re-issued until max_turns with nothing advancing, so a "no rules" arm
            # would measure the harness's inability to escape a repeat rather than the model. The
            # no-rules baseline is therefore "repeat_cap only".
            # OPTIONAL (general + per-task + the sys2_rules_exp*.py patch layer) is what
            # --task-rules adds, and it is off by default. Tier selection lives in apply_rules, so
            # this call is UNCONDITIONAL and passes the flag through.
            # The revised PLAN is assigned back to `plan`, which is what gets fed to every later
            # exec_turn -- so a rule's plan edit becomes System2's context for the rest of the
            # episode, exactly like a model-authored <plan_update>.
            # System2's OWN output, kept verbatim for the record. The rules below rebind
            # `subgoal`/`est`/`plan` to the EFFECTIVE values handed to System1; these three keep
            # what the model actually said, so turn.json's "s2" block never misreports the planner
            # (its <estimated_step> stays whatever it emitted, even when a rule overrode the budget
            # in flight). The raw response string is never touched.
            s2_subgoal, s2_sg_detail, s2_est, s2_plan = subgoal, sg_detail, est, plan
            rule_ivs: list[dict] = []
            skip_s1 = False
            tx_label = None
            force_steps = 0
            rr_stop = False
            # Reset per turn: a HELD turn skips the rule call below, and without this rr would still
            # be the PREVIOUS turn's result -- a stale requery_s2 would fire a spurious second
            # System2 call. (Latent before the mandatory tier existed, because the requery guard also
            # tested args.task_rules; now that the guard is unconditional it would be reachable.)
            rr: dict = {}
            s2_requery = None          # the SECOND System2 response, when a rule asked for one
            # A held turn is ALREADY a rule decision: the subgoal is the one System2 proposed last
            # turn and a rule deferred. Re-running the rule layer over it would let the same rule that
            # borrowed the turn look at the same plan and borrow again (its own counter guards that,
            # but the est rules would also re-fire on a subgoal whose est was resolved a turn ago).
            # So the rules are skipped and the held values are used verbatim.
            if not held:
                # Rules that act on System2's judgement (rather than merely its wording) read the
                # current value from the shared per-episode state. Assign even when None so a stale
                # subgoal_failed can never leak from the previous turn.
                rule_state["judge"] = judge
                rr = SR.apply_rules(task_name, plan=plan, subgoal=subgoal,
                                    subgoal_detail=sg_detail, est=est, state=rule_state,
                                    general=bool(args.general_rules), task_tier=bool(args.task_rules))
                plan, subgoal, sg_detail, est = rr["plan"], rr["subgoal"], rr["subgoal_detail"], rr["est"]
                skip_s1 = rr["skip_s1"]
                tx_label = rr.get("tx_label")
                # force_steps: run EXACTLY this many steps, ignoring the stop rule and the
                # --max-steps-cap ceiling. For a genuine wait ("wait for the bread to pop up") the
                # stop rule is structurally wrong -- a waiting arm is quiescent by definition, so
                # quiescence fires immediately and the segment ends while the toaster is still
                # running. Only a rule can know a subgoal is a wait, hence the channel.
                force_steps = int(rr.get("force_steps") or 0)
                rr_stop = bool(rr.get("stop_episode"))
                rule_ivs = rr["interventions"]
                if rule_ivs:
                    ep_rule_log.append({"turn": turn, "interventions": rule_ivs})
                    for _iv in rule_ivs:
                        print(f"  RULE [{_iv['kind']}] {_iv['rule']}: "
                              f"{_iv['before']} -> {_iv['after']}", flush=True)

            # -- RULE RE-QUERY: the plan moved on, so ask System2 again THIS turn ---------------
            # repeat_cap closed a stuck milestone because there was no next fine step to advance into.
            # Re-running the exhausted subgoal would waste the turn, so System2 is asked again with the
            # revised checklist and ITS new subgoal is what System1 executes. One extra S2 call (~2-4s)
            # and at most ONE per turn: the rule resets its repeat counter when it closes a milestone,
            # so the re-applied rules cannot ask again and walk the whole plan.
            if rr.get("requery_s2"):
                _t2 = time.perf_counter()
                if turn == 0:
                    s2_requery = s2_client.exec_first(instruction, plan,
                                                      tdir / "s2_input_scene_full.png")
                    s2_user = S2C.user_exec_first(instruction, plan)
                else:
                    s2_requery = s2_client.exec_turn(instruction, plan, prev_clip_path,
                                                     prev_clip_frames, task_status, grip_status)
                    s2_user = S2C.user_exec_turn(instruction, plan, task_status, grip_status)
                ep_timings["s2_exec"].append(time.perf_counter() - _t2)
                plan_in = plan          # the REVISED checklist is what this call was given
                s2_prev, s2 = s2, s2_requery
                judge = s2.get("judge")
                subgoal = (s2.get("subgoal") or "").strip()
                sg_detail = (s2.get("subgoal_detail") or "").strip()
                est = s2.get("estimated_step")
                plan = S2C.apply_plan_update(plan, s2.get("plan_update"))
                s2_subgoal, s2_sg_detail, s2_est, s2_plan = subgoal, sg_detail, est, plan
                rule_state["judge"] = judge
                rr = SR.apply_rules(task_name, plan=plan, subgoal=subgoal,
                                    subgoal_detail=sg_detail, est=est, state=rule_state,
                                    general=bool(args.general_rules), task_tier=bool(args.task_rules))
                plan, subgoal, sg_detail, est = (rr["plan"], rr["subgoal"],
                                                 rr["subgoal_detail"], rr["est"])
                skip_s1, tx_label = rr["skip_s1"], rr.get("tx_label")
                force_steps = int(rr.get("force_steps") or 0)
                rr_stop = bool(rr.get("stop_episode"))
                rule_ivs = rule_ivs + rr["interventions"]
                if rr["interventions"]:
                    ep_rule_log.append({"turn": turn, "requery": True,
                                        "interventions": rr["interventions"]})
                print(f"  RULE re-queried System2 after the milestone close -> {subgoal!r}", flush=True)

            turn_rec: dict = {
                "turn": turn, "dir": tdir.name,
                "s2": {
                    "system_prompt": (S2C.SYS_EXEC),
                    "user_prompt": s2_user,
                    # The checklist this System2 call actually received (post-rule on a re-query).
                    "plan_in": plan_in,
                    "response_raw": s2.get("raw"),
                    "thought": s2.get("thought"), "judge": judge, "judge_raw": s2.get("judge_raw"),
                    # PRE-rule: what System2 itself emitted (see s2_* capture above).
                    "plan_update": s2.get("plan_update"), "estimated_step": s2_est,
                    # Set when a rule closed a milestone and System2 was asked again the same turn:
                    # this is the FIRST (superseded) response; the block around it is the second.
                    "superseded_by_requery": ({"subgoal": s2_prev.get("subgoal"),
                                               "response_raw": s2_prev.get("raw")}
                                              if s2_requery is not None else None),
                    "subgoal": s2_subgoal, "subgoal_detail": s2_sg_detail,
                    # Names the rule that borrowed the PREVIOUS turn when System2 was not queried at
                    # all this turn -- so "no user_prompt / no judge" reads as deliberate rather than
                    # as a missing record.
                    "held_by_rule": s2.get("held_by_rule"),
                    "latency_s": s2.get("latency_s"), "usage": s2.get("usage"),
                    "nframes_requested": s2.get("nframes"),
                    "media": s2_media,
                    "privileged": {"task_status": task_status, "gripper_status": grip_status},
                    "t_total_s": round(t_s2, 3),   # request + media prep (latency_s = request only)
                },
                "plan_after": plan,
                # Full audit trail: what System2 actually said, and every override applied to it.
                # Empty list == no rule fired, so an unrevised turn is unambiguous.
                # New runs use the run-level rule_config.json as the source of truth. Keep this
                # per-turn bit for old readers, but make it mean what it says: at least one tier ran.
                "rules": {"enabled": _rules_present(args),
                          "tier": _rule_tier(args),
                          "interventions": rule_ivs,
                          # set when a rule INJECTED this turn rather than System2 asking for it
                          "tx_label": tx_label,
                          # What System1 was ACTUALLY given, after any override. Equal to the "s2"
                          # block above when no rule fired, so the two are always comparable.
                          "effective": {"subgoal": subgoal, "subgoal_detail": sg_detail,
                                        "estimated_step": est},
                          "s2_plan_before_rules": (s2_plan if s2_plan != plan else None)},
            }

            if judge == "task_finish":
                term = "task_finish"
                turn_rec["s1"] = None
                _write_json(tdir / "turn.json", turn_rec)
                doc["turns"].append({k: turn_rec[k] for k in ("turn", "dir", "plan_after")}
                                    | {"judge": judge, "subgoal": None, "n_steps": 0})
                turn += 1
                break
            # -- RULE STOP: the repeat cap gave up on a loop it cannot break --------------------
            # A rule asked to end the episode. Today only repeat_cap does, when the same subgoal has
            # been re-issued past the cap AND there is no next fine step and no later milestone to
            # advance into, MAX_CAP_DECLINES times over. Such an episode is effectively dead --
            # measured over two 1500-episode sweeps it succeeded 1.5% / 2.3% of the time, against ~65%
            # for episodes that never trip the cap -- so the remaining turn budget is better not spent.
            # Recorded as its OWN termination ("max_cap"), never as max_turns, so the two are always
            # distinguishable in the results.
            if rr_stop:
                term = "max_cap"
                turn_rec["s1"] = None
                _write_json(tdir / "turn.json", turn_rec)
                doc["turns"].append({k: turn_rec[k] for k in ("turn", "dir", "plan_after")}
                                    | {"judge": judge, "subgoal": subgoal, "n_steps": 0})
                turn += 1
                break
            if not subgoal:
                term = "no_subgoal"
                turn_rec["s1"] = None
                turn_rec["error"] = "System2 emitted no <subgoal> and did not judge task_finish"
                _write_json(tdir / "turn.json", turn_rec)
                doc["turns"].append({"turn": turn, "dir": tdir.name, "judge": judge,
                                     "subgoal": None, "n_steps": 0, "error": turn_rec["error"]})
                turn += 1
                break

            # -- RULE SKIP: this turn runs NO System1 segment ---------------------------------
            # The rule already marked the step done in `plan`, so next turn System2 sees it
            # completed and moves on. prev_clip_path is deliberately left untouched: no segment
            # ran, so there is no new video and System2 re-reads the previous one. The env is not
            # stepped, so success state is unchanged.
            if skip_s1:
                turn_rec["s1"] = None
                turn_rec["s1_skipped_by_rule"] = True
                _write_json(tdir / "turn.json", turn_rec)
                doc["turns"].append({"turn": turn, "dir": tdir.name, "judge": judge,
                                     "subgoal": subgoal, "n_steps": 0,
                                     "s1_skipped_by_rule": True,
                                     "rules": turn_rec["rules"], "plan_after": plan})
                print(f"  turn {turn}: System1 SKIPPED by rule (subgoal={subgoal!r})", flush=True)
                turn += 1
                continue

            # -- System1: execute that subgoal --
            # anchor = FIRST frame of THIS segment (== last frame of the previous one).
            anchor_obs = env._get_observations(force_update=True)
            anchor_imgs = images_from_obs(anchor_obs)
            anchor_state = raw_state_from_obs(anchor_obs)
            anchor_tile = SE._stacked_from_obs(anchor_obs)
            anchor_info = S2C.write_image(S2C.downscale([anchor_tile])[0], tdir / "s1_anchor.png")

            # Per-step action override for this segment (gripper pinning); {} when none applies.
            act_override = SR.action_overrides(task_name, plan, subgoal,
                                       task_tier=bool(args.task_rules))
            if act_override:
                rule_ivs.append({"rule": "action_override", "kind": "action_override",
                                 "detail": act_override.get("why", ""),
                                 "before": "System1 gripper command",
                                 "after": f"pinned to {act_override.get('grip')}"})
                ep_rule_log.append({"turn": turn, "interventions": rule_ivs[-1:]})
                print(f"  RULE [action_override] {act_override.get('why','')}", flush=True)
                turn_rec["rules"]["interventions"] = rule_ivs
            est_eff = int(est) if isinstance(est, int) and est > 0 else args.default_est_length
            budget = int(min(args.max_steps_cap, max(1, round(est_eff * args.horizon_mult))))
            # force_steps DELIBERATELY BYPASSES --max-steps-cap. The cap is a global guard against
            # one subgoal eating the whole episode; a rule that names an exact step count has
            # already made that judgement for this subgoal, and clamping it to the cap would
            # silently deliver a different number than the rule asked for (with a 400 cap, a
            # 1200-step wait would run 400). Recorded in turn.json as forced_steps so a segment
            # longer than the cap is never mysterious.
            if force_steps > 0:
                budget = force_steps
            s1_text = sg_detail if (args.prompt_source == "subgoal_detail" and sg_detail) else subgoal

            roll = run_s1_segment(
                env, s1_client, subgoal_text=s1_text, task_goal=instruction,
                est_length=est_eff, base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                anchor_imgs=anchor_imgs, anchor_state=anchor_state, resize=args.resize_size,
                replan_steps=args.replan_steps, budget=budget, stop_cfg=stop_cfg,
                norm_stats=norm_stats, last_cmd_grip_init=last_cmd_grip,
                zero_arm_in_base=not args.no_zero_arm_in_base, act_override=act_override,
                force_steps=force_steps)

            frames = roll.pop("_clean_frames")
            steps = roll.pop("_step_records")
            # Expose the gripper WIDTH left by this segment to the rules, so a rule can react to the
            # physical outcome rather than only to System2's text. Read on the NEXT turn, when the
            # env is still in the pose this segment ended in -- e.g. a near-zero width after a grasp
            # means the fingers closed on nothing. Taken from the recorded steps rather than a fresh
            # observation, so it costs no extra render.
            _gw = [x.get("grip_width") for x in steps if x.get("grip_width") is not None]
            if _gw:
                rule_state["grip_width"] = float(_gw[-1])
                rule_state["grip_width_min"] = float(min(_gw))
            motion = roll.pop("_motion")
            seg_grip_cmds = roll.get("grip_cmds") or []
            last_cmd_grip = roll["last_cmd_grip"]
            env_success = bool(env._check_success())

            # RAW rollout video (every executed step, 20 fps) — what actually happened.
            _t = time.perf_counter()
            # SAVED AT FULL CAPTURED RESOLUTION (TILE_HW, 256x768 = 3 cameras of 256x256 hstacked),
            # i.e. NO downscale. This file is for HUMAN inspection in the GUI, and 128x384 was too
            # coarse for what it is most used for: telling a held object from a dropped one. The model
            # never reads it -- System2's input is the separate 4-fps clip below, already at TILE_HW.
            # Cost measured on the v2 sweep: downscaled files averaged 56 KB, ~1.1 GB per 1500-episode
            # run; at 4x the pixels expect ~3 GB. SYS2_RAW_VIDEO_DOWNSCALE=1 restores the small copies.
            _raw = S2C.downscale(frames) if os.environ.get("SYS2_RAW_VIDEO_DOWNSCALE") else frames
            raw_info = (S2C.write_clip(_raw, tdir / "s1_rollout_raw.mp4",
                                       fps=S2C.SIM_FPS) if frames else None)
            # CONDENSED 4-fps clip = exactly what System2 sees next turn. Static frames are
            # dropped for motion subgoals and KEPT for wait/hold subgoals.
            clip_frames, clip_stats = S2C.build_clip_frames(
                frames, motion, subgoal, eps=args.static_eps)
            # The clip System2 consumes MUST stay at the training resolution (256x768). We write it
            # to a sidecar name, and a separate DOWNSCALED copy under the display name the GUI
            # serves -- so shrinking artifacts can never silently change model input.
            model_clip_path = tdir / "s2_input_clip_full.mp4"
            clip_path = tdir / "s2_input_clip.mp4"
            model_info = (S2C.write_clip(clip_frames, model_clip_path) if clip_frames else None)
            clip_info = (S2C.write_clip(S2C.downscale(clip_frames), clip_path)
                         if clip_frames else None)
            if clip_info and model_info:
                clip_info["model_clip"] = model_info      # provenance: what the model actually saw
            n_clip_frames = len(clip_frames)
            # Hand the FULL-RES clip to the NEXT turn's System2 call.
            prev_clip_path = model_clip_path if model_info else None
            prev_clip_frames = n_clip_frames
            t_video = time.perf_counter() - _t
            ep_timings["video_encode"].append(t_video)
            # Per-frame audit trail so the condensing is VERIFIABLE frame by frame in the GUI:
            # for every executed step, its motion norm, whether it passed the static filter, and
            # whether it survived into the clip System2 actually saw.
            kept = set(clip_stats.get("kept_indices") or [])
            clip_stats["per_frame"] = [
                {"i": i, "motion": round(float(m), 5),
                 "moving": bool(float(m) >= args.static_eps),
                 "in_clip": bool(i in kept)}
                for i, m in enumerate(motion)
            ]
            # Also dump the condensed frames as PNGs (small: <=32 per turn) so the exact images
            # fed to System2 can be eyeballed without decoding the mp4.
            # NO per-frame PNG dump. The condensed frames are already in s2_input_clip.mp4
            # (~43 KB) which the GUI plays and can step through; writing them again as PNGs cost
            # ~2 MB per episode (~50 GB across the 25k-episode benchmark) for zero extra information.
            # clip_stats.per_frame below still records exactly which steps were kept and why.

            _write_json(tdir / "s1_steps.json", {
                "subgoal": subgoal, "subgoal_detail": sg_detail, "s1_prompt_text": s1_text,
                "est_length": est_eff, "budget": budget,
                "n_steps": roll["n_steps"], "stop_reason": roll["stop_reason"],
                "success_step": roll.get("success_step"),
                "progress_done": roll["progress_done"], "quiescent": roll["quiescent"],
                "s1_prompts": roll["s1_prompts"],
                "anchor_image": anchor_info,
                "steps": steps,
            })
            turn_rec["s1"] = {
                "prompt_text": s1_text, "est_length": est_eff, "budget": budget,
                # Non-zero only when a rule forced the segment length; makes a segment that exceeds
                # --max-steps-cap self-explanatory in the record and in the GUI.
                "forced_steps": force_steps or None,
                "n_steps": roll["n_steps"], "stop_reason": roll["stop_reason"],
                "success_step": roll.get("success_step"),
                "progress_done": roll["progress_done"], "quiescent": roll["quiescent"],
                "s1_prompts": roll["s1_prompts"],
                "anchor_image": anchor_info,
                "video_raw": raw_info, "video_clip": clip_info, "clip_stats": clip_stats,
                "timings": roll.get("timings"),
            }
            turn_rec["env_success_after"] = env_success
            # Per-turn wall-clock attribution: which component consumed the turn.
            _s1t = roll.get("timings") or {}
            turn_rec["timings"] = {
                "s2_call_s": round(t_s2, 3),
                "s1_infer_total_s": (_s1t.get("s1_infer") or {}).get("total_s"),
                "s1_infer_mean_s": (_s1t.get("s1_infer") or {}).get("mean_s"),
                "s1_infer_calls": (_s1t.get("s1_infer") or {}).get("n"),
                "env_step_total_s": (_s1t.get("env_step") or {}).get("total_s"),
                "env_render_total_s": (_s1t.get("env_render") or {}).get("total_s"),
                "obs_build_total_s": (_s1t.get("s1_obs_build") or {}).get("total_s"),
                "video_encode_s": round(t_video, 3),
            }
            _write_json(tdir / "turn.json", turn_rec)

            doc["turns"].append({
                "turn": turn, "dir": tdir.name, "judge": judge, "subgoal": subgoal,
                "estimated_step": est, "budget": budget, "n_steps": roll["n_steps"],
                "stop_reason": roll["stop_reason"], "success_step": roll.get("success_step"),
                "env_success_after": env_success,
                "clip_frames": n_clip_frames,
                "clip_static_dropped": clip_stats.get("frac_static_dropped") if clip_stats else None,
                "wait_subgoal": clip_stats.get("wait_subgoal") if clip_stats else None,
                "plan_after": plan,
                "timings": turn_rec["timings"],
            })
            _write_json(ep_out / "episode.json", doc)   # checkpoint after every turn
            turn += 1

            if env_success and args.stop_on_env_success:
                term = "env_success"
                break
        doc["n_turns"] = turn
        doc["termination"] = term
        # Episode-level rule summary: the per-turn detail lives in each turn.json, this is the
        # roll-up the report script and the GUI read. `rule_interventions: []` with
        # `task_rules: true` means the rules were ON but nothing matched this episode.
        doc["task_rules"] = bool(args.task_rules)
        doc["general_rules"] = bool(args.general_rules)
        # The mandatory tier (repeat_cap) runs regardless; task_rules above is the
        # OPTIONAL tier only. Recorded explicitly so an arm is self-describing.
        doc["rule_tier"] = _rule_tier(args)
        doc["rule_interventions"] = ep_rule_log
        doc["n_rule_interventions"] = sum(len(t["interventions"]) for t in ep_rule_log)
        doc["episode_success"] = bool(env._check_success())
        doc["final_plan"] = plan
        doc["seconds"] = round(time.time() - t0, 2)
        # Episode-level wall-clock attribution across every timed component.
        _agg: dict[str, list[float]] = {k: list(v) for k, v in ep_timings.items()}
        for t in doc["turns"]:
            tt = t.get("timings") or {}
            for src, dst in (("s1_infer_total_s", "s1_infer"), ("env_step_total_s", "env_step"),
                             ("env_render_total_s", "env_render"),
                             ("obs_build_total_s", "s1_obs_build")):
                if tt.get(src) is not None:
                    _agg.setdefault(dst, []).append(float(tt[src]))
        doc["timings"] = _timing_summary(_agg)
        doc["timings"]["_note"] = ("per-component wall clock; s1_infer/env_* are summed per turn, "
                                  "s2_exec/s2_plan are per call")
        _write_json(ep_out / "episode.json", doc)
        print(f"  {task_name}: success={doc['episode_success']} turns={turn} term={term}"
              f" ({doc['seconds']}s)", flush=True)
        return doc

    except Exception as e:
        doc["task_rules"] = bool(args.task_rules)
        doc["general_rules"] = bool(args.general_rules)
        # The mandatory tier (repeat_cap) runs regardless; task_rules above is the
        # OPTIONAL tier only. Recorded explicitly so an arm is self-describing.
        doc["rule_tier"] = _rule_tier(args)
        doc["rule_interventions"] = ep_rule_log
        doc["error"] = f"{type(e).__name__}: {e}"
        doc["traceback"] = traceback.format_exc()
        doc["seconds"] = round(time.time() - t0, 2)
        _write_json(ep_out / "episode.json", doc)
        print(f"  ERROR {task_name}: {doc['error']}", flush=True)
        return doc
    finally:
        try:
            env.close()
        except Exception:
            pass


def _rule_tier(args) -> str:
    """Which rule tiers this run applied -- recorded so an arm is self-describing.

    Three arms, matching the CLI: "mandatory" (repeat_cap only -- this is the historical '-base'
    arm), "mandatory+general", "mandatory+general+task". A leading "none" appears only when the
    mandatory tier was explicitly disabled with SYS2_RULES_NO_MANDATORY=1.
    """
    tiers = ["mandatory"] if SR.mandatory_on() else ["none"]
    if getattr(args, "general_rules", False):
        tiers.append("general")
    if getattr(args, "task_rules", False):
        tiers.append("task")
    return "+".join(tiers)


def _rule_config(args) -> dict:
    """Stable, serializable configuration shared by result files and the GUI."""
    cfg = SR.rule_config(general=bool(getattr(args, "general_rules", False)),
                         task_tier=bool(getattr(args, "task_rules", False)))
    return {**cfg, "tier": _rule_tier(args)}


def _rules_present(args) -> bool:
    cfg = _rule_config(args)
    return bool(cfg["mandatory_rules"] or cfg["general_rules"] or cfg["task_rules"])


def _write_run_rule_config(out_root: Path, method: str, args) -> dict:
    """Persist the method's rule tiers and refuse to mix incompatible arms in one directory."""
    cfg = _rule_config(args)
    path = out_root / method / "rule_config.json"
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except Exception as e:
            raise RuntimeError(f"invalid existing rule config {path}: {e}") from e
        if old != cfg:
            raise RuntimeError(
                f"rule configuration mismatch for method {method!r}: existing {old}, requested "
                f"{cfg}. Use a distinct --method/RUN_LABEL for each evaluation arm.")
    _write_json(path, cfg)
    return cfg


def build_argparser(description: str | None = None) -> argparse.ArgumentParser:
    """The full CLI. Exposed so a plan VARIANT script can extend it instead of copying it
    (``combine_memory_eval.py`` adds its own flags on top of this parser)."""
    ap = argparse.ArgumentParser(description=description or __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lerobot-dir", required=True,
                    help="raw LeRobot dataset dir, e.g. "
                         "$ROBOCASA_LEROBOT_ROOT/v1.0/target/atomic/CloseFridge/20250816/lerobot")
    ap.add_argument("--episodes", default="0",
                    help="episode indices: comma list and/or ranges, e.g. '0', '0,1', '0-4'")
    ap.add_argument("--method", default=None,
                    help="run label; default is derived from --s1-dir/--s2-dir, e.g. "
                         "s1-progreg270k_s2-qwen35-4b-full-ep3-11416")
    ap.add_argument("--s1-dir", default=None,
                    help="System1 checkpoint dir being served (recorded + used for the run name)")
    ap.add_argument("--s2-dir", default=None,
                    help="System2 checkpoint dir being served (recorded + used for the run name)")
    ap.add_argument("--out-root", type=Path, default=SYS1_RESULTS_DIR / "combine",
                    help="where rollouts are written (default: $SYS1_RESULTS_DIR/combine)")
    # System1 (JAX policy server)
    ap.add_argument("--s1-host", default="127.0.0.1")
    ap.add_argument("--s1-port", type=int, default=8060)
    ap.add_argument("--norm-stats", type=Path, default=None,
                    help="ckpt assets/robocasa_system1/norm_stats.json (for normalized traces)")
    # System2 (vLLM server)
    ap.add_argument("--s2-host", default="127.0.0.1")
    ap.add_argument("--s2-port", type=int, default=8100)
    ap.add_argument("--s2-model", default="system2-full")
    ap.add_argument("--s2-max-tokens", type=int, default=512)
    ap.add_argument("--s2-file-uri", action="store_true",
                    help="pass media as file:// URIs instead of inlined base64 (server must share the FS)")
    # loop control
    ap.add_argument("--max-turns", type=int, default=20,
                    help="fallback turn cap; by default the cap is PER TASK from "
                         "robocasa_target_maxturns.json (max_subgoal_turns + headroom + offset)")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes already finished under --out-root/<method> (episode.json "
                         "present with a termination and no error). Makes a killed sweep restartable.")
    ap.add_argument("--flat-max-turns", action="store_true",
                    help="ignore the per-task table and use --max-turns for every task")
    ap.add_argument("--horizon-mult", type=float, default=2.0,
                    help="segment budget = estimated_step * this, capped by --max-steps-cap")
    ap.add_argument("--max-steps-cap", type=int, default=400,
                    help="hard ceiling on ONE subgoal segment: budget = min(this, est_length * "
                         "horizon_mult). BACK TO 400 (it was raised to 800 for one round of runs). "
                         "800 existed because the cap silently truncated long WAIT subgoals -- "
                         "GetToastedBread asks est 500-600, so est*2 was clipped and episodes were cut "
                         "mid-wait. That is now handled properly by sys2_rules' wait rule, which "
                         "returns ``force_steps`` and DELIBERATELY BYPASSES this cap, so the wait gets "
                         "its full length whatever the ceiling is -- and the ceiling can go back to "
                         "guarding against a runaway segment. Measured on the 800 run: only 112 of "
                         "13263 segments (0.8%%) had a budget above 400 and just 28 actually ran past "
                         "it, half of them the GetToastedBread waits that now bypass the cap. Runs at "
                         "400 ARE step-for-step comparable with the 1000-episode no-rules baseline and "
                         "are NOT comparable with the -estbump sweep, which used 800.")
    ap.add_argument("--general-rules", action="store_true",
                    help="add the GENERAL (task-agnostic) rule tier on top of the mandatory one. "
                         "Implied by --task-rules, so the three arms are: no flag = mandatory only "
                         "(repeat_cap, the '-base' arm); --general-rules = mandatory + general; "
                         "--task-rules = mandatory + general + per-task. The general tier is "
                         f"currently {SR.TASKS_WITH_RULES[1]}.")
    ap.add_argument("--task-rules", action="store_true",
                    help="add the OPTIONAL rule tiers (general + per-task + any sys2_rules_exp*.py "
                         "patch file) on top of the MANDATORY tier. The mandatory tier -- repeat_cap "
                         "-- runs on EVERY run whether or not this flag is given, because it is the "
                         "loop's termination policy rather than a revision of System2's output: "
                         "without it a stuck subgoal is re-issued until max_turns with nothing "
                         "advancing. So the no-rules baseline is 'repeat_cap only', and this flag "
                         "chooses between the two arms. Scope, derived from the live registry: "
                         f"{'; '.join(SR.TASKS_WITH_RULES)}. Every intervention is recorded under "
                         "turn.json:rules (with the tier) and episode.json:rule_interventions. To "
                         "reproduce the historical zero-rules baseline that predates the mandatory "
                         "tier, set SYS2_RULES_NO_MANDATORY=1 as well.")
    ap.add_argument("--default-est-length", type=int, default=50,
                    help="fallback when System2 omits/garbles <estimated_step>")
    ap.add_argument("--replan-steps", type=int, default=16)
    ap.add_argument("--resize-size", type=int, default=224)
    ap.add_argument("--prompt-source", choices=["subgoal", "subgoal_detail"], default="subgoal",
                    help="which System2 text becomes the System1 instruction")
    ap.add_argument("--stop-progress", type=float, default=0.95)
    ap.add_argument("--stop-eps", type=float, default=0.03,
                    help="arm/base action-norm below this counts as not moving "
                         "(eef_pos+eef_rot+base only; the gripper is tracked separately)")
    ap.add_argument("--stop-window", type=int, default=5)
    ap.add_argument("--static-eps", type=float, default=S2C.STATIC_EPS,
                    help="motion-norm below this counts as a STATIC frame when condensing clips "
                         "(10x stricter than the stop rule's --stop-eps on purpose; see sys2_client)")
    ap.add_argument("--stop-on-env-success", action="store_true", default=True)
    ap.add_argument("--no-zero-arm-in-base", action="store_true")
    return ap


def run_sweep(args, *, method_suffix: str = "") -> None:
    # The task tier sits on top of the general tier, so --task-rules implies --general-rules.
    # Resolved ONCE here rather than at each use, so the rule call, the recorded tier and the log
    # line can never disagree about which arm ran.
    args.general_rules = bool(getattr(args, "general_rules", False) or args.task_rules)
    """Connect to both servers, roll every requested episode, write the index part.

    Shared by ``main()`` and by plan-variant scripts: a variant only has to set ``args.plan_fn``
    (and pass its own ``method_suffix``) to get the identical loop, resume logic and output layout.
    """
    ep_indices = _parse_episodes(args.episodes)
    # The servers are addressed by PORT, so the checkpoints they serve are not otherwise recorded
    # anywhere in the output. Pass --s1-dir/--s2-dir to bake that provenance into the run.
    if not args.method:
        args.method = _short_method_name(args.s1_dir, args.s2_dir, method_suffix)
        print(f"derived --method: {args.method}", flush=True)
    out_root = Path(args.out_root)
    run_rule_config = _write_run_rule_config(out_root, args.method, args)
    print(f"rule config: {run_rule_config['tier']}  ({out_root / args.method / 'rule_config.json'})",
          flush=True)
    norm_stats = None
    if args.norm_stats and args.norm_stats.exists():
        raw = json.loads(args.norm_stats.read_text())
        norm_stats = raw.get("norm_stats", raw)

    s1 = _wcp.WebsocketClientPolicy(host=args.s1_host, port=args.s1_port)
    s2 = S2C.Sys2Client(args.s2_host, args.s2_port, args.s2_model,
                        max_tokens=args.s2_max_tokens, inline_media=not args.s2_file_uri)
    print(f"System2 server health: {s2.health()}  (model={args.s2_model})", flush=True)

    results = []
    for i, ep_idx in enumerate(ep_indices):
        if args.resume and _episode_done(out_root / args.method, Path(args.lerobot_dir), ep_idx):
            print(f"=== [{i+1}/{len(ep_indices)}] episode {ep_idx}: SKIP (already done) ===",
                  flush=True)
            continue
        print(f"=== [{i+1}/{len(ep_indices)}] {args.lerobot_dir} episode {ep_idx} ===", flush=True)
        args.episode_index = ep_idx
        results.append(eval_episode(Path(args.lerobot_dir), s1, s2, args, out_root,
                                    args.method, norm_stats))

    ok = sum(1 for r in results if r.get("episode_success"))
    idx = {
        "eval_kind": "combine", "method": args.method, "n_episodes": len(results),
        "rule_config": run_rule_config,
        "n_success": ok,
        "success_rate": (ok / len(results)) if results else None,
        "episodes": [{k: r.get(k) for k in
                      ("episode_id", "task_name", "episode_success", "n_turns", "termination",
                       "seconds", "error")} for r in results],
    }
    # CONCURRENCY: the fleet runs 8 workers against ONE method dir, so a shared index.json would be
    # clobbered (last writer wins, every other worker's episodes lost). Each invocation writes its
    # own shard under index_parts/ keyed by task+pid; scripts/merge_combine_index.py (and the GUI's
    # fallback) union them into index.json.
    part = f"{_task_name_from_lerobot_dir(Path(args.lerobot_dir))}-{os.getpid()}.json"
    _write_json(out_root / args.method / "index_parts" / part, idx)
    # The part file above is the durable record; index.json is a DERIVED convenience that any later
    # merge (or scripts/extract_combine_results.py) can rebuild from the parts. So a merge failure
    # must never fail the run -- the rollout is already safely on disk by this point, and letting
    # this raise would report a completed episode as FAILED.
    try:
        merge_combine_index(out_root / args.method)
    except Exception as e:  # noqa: BLE001 - derived artifact; parts/ still holds the truth
        print(f"WARNING: index.json merge failed ({e}); parts/ intact, rebuild with "
              f"scripts/extract_combine_results.py", flush=True)
    print(f"\nWROTE {out_root / args.method / 'index_parts' / part}  {ok}/{len(results)} success",
          flush=True)


def main():
    run_sweep(build_argparser().parse_args())


if __name__ == "__main__":
    main()
