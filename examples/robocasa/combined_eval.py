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


def _short_method_name(s1_dir: str | None, s2_dir: str | None) -> str:
    """Derive a compact run name that names BOTH systems, e.g.

        s1-progreg270k_s2-qwen35-4b-full-ep3-11416

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

    return f"{s1_short(s1_dir)}_{s2_short(s2_dir)}"



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

# TARGET_EVAL_EPISODES: the 1000-episode mid-scale eval manifest (50 tasks x 20 episodes),
# from robocasa_target_episode.json (_evallogs/eps_target20/all.txt). task -> episode indices.
TARGET_EVAL_EPISODES: dict[str, list[int]] = {
    'ArrangeBreadBasket': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'ArrangeTea': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'BreadSelection': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'CategorizeCondiments': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'CloseBlenderLid': [4, 6, 7, 10, 11, 17, 24, 26, 27, 28, 29, 38, 42, 43, 48, 55, 58, 60, 67, 72],
    'CloseFridge': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'CloseToasterOvenDoor': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'CoffeeSetupMug': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'CuttingToolSelection': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'DeliverStraw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'GarnishPancake': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'GatherTableware': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'GetToastedBread': [0, 1, 2, 3, 4, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 27, 28],
    'HeatKebabSandwich': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'KettleBoiling': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'LoadDishwasher': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'MakeIceLemonade': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'NavigateKitchen': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25],
    'OpenCabinet': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'OpenDrawer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'OpenStandMixerHead': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PackIdenticalLunches': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PanTransfer': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PickPlaceCounterToCabinet': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PickPlaceCounterToStove': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PickPlaceDrawerToCounter': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PickPlaceSinkToCounter': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PickPlaceToasterToCounter': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PortionHotDogs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PreSoakPan': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'PrepareCoffee': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'RecycleBottlesByType': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'RinseSinkBasin': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'ScrubCuttingBoard': [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'SearingMeat': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'SeparateFreezerRack': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'SetUpCuttingStation': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'SlideDishwasherRack': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'StackBowlsCabinet': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'SteamInMicrowave': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'StirVegetables': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'StoreLeftoversInBowl': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'TurnOffStove': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'TurnOnElectricKettle': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'TurnOnMicrowave': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'TurnOnSinkFaucet': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'WaffleReheat': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'WashFruitColander': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'WashLettuce': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'WeighIngredients': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
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


def max_turns_for(task_name: str, flat_max: int, *, dynamic: bool = True) -> tuple[int, str]:
    """Return (max_turns, why) for a task.

    Falls back to the flat cap for any task absent from the table (e.g. a non-target task), so an
    unknown task is never silently given a tiny budget.
    """
    if not dynamic:
        return flat_max, "flat"
    n = TARGET_MAX_SUBGOAL_TURNS.get(task_name)
    if n is None:
        return flat_max, "flat (task not in budget table)"
    return (n + TURN_HEADROOM + TURN_DEF_OFFSET,
            f"max_subgoal_turns {n} + {TURN_HEADROOM} headroom + {TURN_DEF_OFFSET} def-offset")


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


def run_s1_segment(
    env, s1_client, *, subgoal_text: str, task_goal: str, est_length: int,
    base_pos_ref, base_yaw_ref, anchor_imgs, anchor_state, resize: int,
    replan_steps: int, budget: int, stop_cfg: StopConfig, norm_stats,
    last_cmd_grip_init: float, zero_arm_in_base: bool,
) -> dict:
    """Roll System1 on ONE System2 subgoal until the stop rule fires or the budget runs out.

    Mirrors episode_eval's rollout but is driven by a System2 subgoal + estimated_step instead of
    an annotated span, and records everything the GUI needs. ``executed_step`` starts at 0 for
    every call (per-turn reset) and the anchor is whatever the caller snapshotted at segment start.
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
        if replanned and tracker.should_stop_lookahead(chunk_sim):
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
            "plan_variant": "cold",
        },
        "turns": [],
    }

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

        # ---------------- PLAN (cold) ----------------
        scene0 = SE._stacked_from_obs(obs0)
        plan_dir = ep_out / "plan"
        # The planner READS this image, so it stays at model resolution; a small copy is written
        # alongside for the GUI.
        img0 = S2C.write_image(scene0, plan_dir / "scene_full.png")
        S2C.write_image(S2C.downscale([scene0])[0], plan_dir / "scene.png")
        _t = time.perf_counter()
        p = s2_client.plan_cold(instruction, plan_dir / "scene_full.png")
        ep_timings["s2_plan"].append(time.perf_counter() - _t)
        plan = (p.get("plan") or "").strip()
        _write_json(plan_dir / "plan.json", {
            "mode": "plan_cold",
            "s2_system_prompt": S2C.SYS_PLAN_COLD,
            "s2_user_prompt": S2C.user_plan_cold(instruction),
            "s2_response_raw": p["raw"], "thought": p.get("thought"), "plan": plan,
            "media": {"image": img0}, "latency_s": p.get("latency_s"), "usage": p.get("usage"),
        })
        doc["plan"] = {"thought": p.get("thought"), "plan": plan, "latency_s": p.get("latency_s"),
                       "dir": "plan"}
        if not plan:
            raise ValueError("System2 returned no <plan>")

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
            _t = time.perf_counter()
            if turn == 0:
                cur = env._get_observations(force_update=True)
                tile0 = SE._stacked_from_obs(cur)
                S2C.write_image(tile0, tdir / "s2_input_scene_full.png")     # what the model reads
                sc = S2C.write_image(S2C.downscale([tile0])[0], tdir / "s2_input_scene.png")
                s2 = s2_client.exec_first(instruction, plan, tdir / "s2_input_scene_full.png")
                s2_media = {"image": sc}
                s2_user = S2C.user_exec_first(instruction, plan)
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
            t_s2 = time.perf_counter() - _t
            ep_timings["s2_exec"].append(t_s2)

            plan = S2C.apply_plan_update(plan, s2.get("plan_update"))
            # PLAN-EXHAUSTED cutoff, re-evaluated against the LATEST plan every turn: while System2
            # keeps sitting on the last fine step the run grows; the moment it appends a new step the
            # run RESETS, because there is fresh work planned.
            judge = s2.get("judge")
            subgoal = (s2.get("subgoal") or "").strip()
            sg_detail = (s2.get("subgoal_detail") or "").strip()
            est = s2.get("estimated_step")

            turn_rec: dict = {
                "turn": turn, "dir": tdir.name,
                "s2": {
                    "system_prompt": (S2C.SYS_EXEC),
                    "user_prompt": s2_user,
                    "response_raw": s2.get("raw"),
                    "thought": s2.get("thought"), "judge": judge, "judge_raw": s2.get("judge_raw"),
                    "plan_update": s2.get("plan_update"), "estimated_step": est,
                    "subgoal": subgoal, "subgoal_detail": sg_detail,
                    "latency_s": s2.get("latency_s"), "usage": s2.get("usage"),
                    "nframes_requested": s2.get("nframes"),
                    "media": s2_media,
                    "privileged": {"task_status": task_status, "gripper_status": grip_status},
                    "t_total_s": round(t_s2, 3),   # request + media prep (latency_s = request only)
                },
                "plan_after": plan,
            }

            if judge == "task_finish":
                term = "task_finish"
                turn_rec["s1"] = None
                _write_json(tdir / "turn.json", turn_rec)
                doc["turns"].append({k: turn_rec[k] for k in ("turn", "dir", "plan_after")}
                                    | {"judge": judge, "subgoal": None, "n_steps": 0})
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

            # -- System1: execute that subgoal --
            # anchor = FIRST frame of THIS segment (== last frame of the previous one).
            anchor_obs = env._get_observations(force_update=True)
            anchor_imgs = images_from_obs(anchor_obs)
            anchor_state = raw_state_from_obs(anchor_obs)
            anchor_tile = SE._stacked_from_obs(anchor_obs)
            anchor_info = S2C.write_image(S2C.downscale([anchor_tile])[0], tdir / "s1_anchor.png")

            est_eff = int(est) if isinstance(est, int) and est > 0 else args.default_est_length
            budget = int(min(args.max_steps_cap, max(1, round(est_eff * args.horizon_mult))))
            s1_text = sg_detail if (args.prompt_source == "subgoal_detail" and sg_detail) else subgoal

            roll = run_s1_segment(
                env, s1_client, subgoal_text=s1_text, task_goal=instruction,
                est_length=est_eff, base_pos_ref=base_pos_ref, base_yaw_ref=base_yaw_ref,
                anchor_imgs=anchor_imgs, anchor_state=anchor_state, resize=args.resize_size,
                replan_steps=args.replan_steps, budget=budget, stop_cfg=stop_cfg,
                norm_stats=norm_stats, last_cmd_grip_init=last_cmd_grip,
                zero_arm_in_base=not args.no_zero_arm_in_base)

            frames = roll.pop("_clean_frames")
            steps = roll.pop("_step_records")
            motion = roll.pop("_motion")
            seg_grip_cmds = roll.get("grip_cmds") or []
            last_cmd_grip = roll["last_cmd_grip"]
            env_success = bool(env._check_success())

            # RAW rollout video (every executed step, 20 fps) — what actually happened.
            _t = time.perf_counter()
            # SAVED at display resolution (128x384). The model never reads this file; it exists so
            # the GUI player can scrub one frame per executed step.
            raw_info = (S2C.write_clip(S2C.downscale(frames), tdir / "s1_rollout_raw.mp4",
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
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
    ap.add_argument("--max-steps-cap", type=int, default=400)
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
    args = ap.parse_args()

    ep_indices = _parse_episodes(args.episodes)
    # The servers are addressed by PORT, so the checkpoints they serve are not otherwise recorded
    # anywhere in the output. Pass --s1-dir/--s2-dir to bake that provenance into the run.
    if not args.method:
        args.method = _short_method_name(args.s1_dir, args.s2_dir)
        print(f"derived --method: {args.method}", flush=True)
    norm_stats = None
    if args.norm_stats and args.norm_stats.exists():
        raw = json.loads(args.norm_stats.read_text())
        norm_stats = raw.get("norm_stats", raw)

    s1 = _wcp.WebsocketClientPolicy(host=args.s1_host, port=args.s1_port)
    s2 = S2C.Sys2Client(args.s2_host, args.s2_port, args.s2_model,
                        max_tokens=args.s2_max_tokens, inline_media=not args.s2_file_uri)
    print(f"System2 server health: {s2.health()}  (model={args.s2_model})", flush=True)

    out_root = Path(args.out_root)
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


if __name__ == "__main__":
    main()
