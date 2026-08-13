"""Build ``robocasa_target_episode_eval30.json`` -- the 30-episode-per-task eval manifest WITH the
real (ground-truth) rollout length of every episode.

WHY. A flat-policy baseline (pi0.5 with no System2) has no planner to tell it when a subgoal is done,
so the only fair stopping rule is a STEP BUDGET. The natural budget is the length of the human
demonstration for that exact episode -- "solve it in no more steps than the demo took". This file
supplies that number per episode so a baseline can be run at parity and its success compared with the
hierarchical System2+System1 runs on the same 1500 episodes.

CONTENTS, per task: the 30 source LeRobot episode indices (identical to
``combined_eval.TARGET_EVAL_EPISODES`` -- this file does not invent a new split), each episode's demo
length in frames, and, when a reference run is given, what the hierarchical system actually did on
that episode (success / steps executed / turns). The reference columns are what makes the comparison
readable: demo length is the budget, ``ref_steps`` is what the hierarchical system spent inside it.

LENGTH DEFINITION. ``length`` = the ``length`` field of the episode's record in the dataset's
``meta/episodes.jsonl``, i.e. the number of recorded frames in the demonstration. At the sim's 20 Hz
control rate one frame == one env step, so it is directly usable as a step budget. It is
cross-checked against ``n_recorded_frames`` recorded independently by combined_eval.py during the
reference run; any disagreement is reported rather than silently averaged.

Usage:
    python scripts/build_eval30_manifest.py                       # writes beside the sibling manifests
    python scripts/build_eval30_manifest.py --out /tmp/x.json \
        --reference-method s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-v2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
from combined_eval import TARGET_EVAL_EPISODES

_REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[2])
_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO_ROOT / "data", Path.home() / "data") if c.is_dir()), _REPO_ROOT / "data"))
DATASET = Path(os.environ.get("ROBOCASA_LEROBOT_ROOT") or _DATA / "robocasa_dataset")
RESULTS = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results")


def lerobot_dir(task: str) -> Path | None:
    hits = sorted(glob.glob(str(DATASET / "v1.0" / "target" / "*" / task / "*" / "lerobot")))
    return Path(hits[0]) if hits else None


def demo_lengths(ld: Path) -> dict[int, int]:
    """episode_index -> demo length in frames, from the dataset's own metadata."""
    out: dict[int, int] = {}
    f = ld / "meta" / "episodes.jsonl"
    if not f.exists():
        return out
    for raw in f.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:  # a torn line must not lose the whole file
            continue
        if r.get("episode_index") is not None and r.get("length") is not None:
            out[int(r["episode_index"])] = int(r["length"])
    return out


def reference_run(method: str) -> dict[tuple[str, int], dict]:
    """(task, episode) -> {success, steps, turns, termination, n_recorded_frames} for a finished run."""
    ref: dict[tuple[str, int], dict] = {}
    root = RESULTS / "combine" / method
    for f in glob.glob(str(root / "*" / "episode.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        turns = d.get("turns") or []
        ref[(d.get("task_name"), int(d.get("episode_index", -1)))] = {
            "success": bool(d.get("episode_success")),
            "steps": sum(int(t.get("n_steps") or 0) for t in turns),
            "turns": len(turns),
            "termination": d.get("termination"),
            "n_recorded_frames": d.get("n_recorded_frames"),
        }
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "robocasa_target_episode_eval30.json")
    ap.add_argument("--reference-method", default="s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-v2",
                    help="finished combine run whose per-episode outcome is embedded for comparison "
                         "('' to omit)")
    a = ap.parse_args()

    ref = reference_run(a.reference_method) if a.reference_method else {}
    tasks: dict[str, dict] = {}
    missing_len: list[str] = []
    mismatch: list[str] = []
    grand = 0

    for task in sorted(TARGET_EVAL_EPISODES):
        eps = list(TARGET_EVAL_EPISODES[task])
        ld = lerobot_dir(task)
        lens = demo_lengths(ld) if ld else {}
        per, tot = [], 0
        for e in eps:
            n = lens.get(e)
            if n is None:
                missing_len.append(f"{task}:{e}")
            else:
                tot += n
            rec: dict = {"episode_index": e, "length": n}
            r = ref.get((task, e))
            if r:
                # combined_eval records the demo length independently; disagreement means the two
                # sources describe different episodes, which would invalidate the budget.
                if r.get("n_recorded_frames") is not None and n is not None \
                        and int(r["n_recorded_frames"]) != n:
                    mismatch.append(f"{task}:{e} meta={n} run={r['n_recorded_frames']}")
                rec.update(ref_success=r["success"], ref_steps=r["steps"],
                           ref_turns=r["turns"], ref_termination=r["termination"])
            per.append(rec)
        grand += tot
        # LONGEST SUCCESSFUL EPISODE, measured in System1's total EXECUTED env steps (the sum of
        # n_steps over the episode's turns -- not the demo length, and not wall-clock). This is the
        # per-task step budget for a flat baseline: "the most steps this task ever needed in a run
        # that actually solved it". Taking the max over SUCCESSES only, so an episode that thrashed to
        # the turn cap without solving anything cannot inflate the budget.
        done = [r for r in per if r.get("ref_success")]
        longest = max(done, key=lambda r: r["ref_steps"]) if done else None
        tasks[task] = {
            # MINIMAL by request: the 30 episode indices to run, and the step cap. Everything else
            # (demo lengths, per-episode reference outcomes, splits, dataset paths) was dropped --
            # re-run this script from an older revision if the detail is wanted again.
            "episodes": eps,
            "max_steps": longest["ref_steps"] if longest else None,
        }

    n_eps = sum(len(v["episodes"]) for v in tasks.values())

    doc = {
        "description": "The 30-episode-per-task RoboCasa target eval set, with a per-task STEP CAP "
                       "for running a flat-policy (pi0.5, no System2) baseline on the same episodes.",
        "definition": "max_steps = the largest TOTAL EXECUTED System1 env-step count among the "
                      "reference run's SUCCESSFUL episodes of that task (summed over the episode's "
                      "turns). Successes only, so an episode that thrashed to the turn cap cannot "
                      "inflate the cap. null = the reference run never solved that task, so there "
                      "is no evidence of how many steps a solution needs.",
        "episode_index_note": "Indices are SOURCE LeRobot indices and are SPARSE (only successfully "
                              "extracted episodes appear), so they are not contiguous 0..29 for "
                              "every task -- e.g. CloseBlenderLid starts 4, 6, 7, 10.",
        "reference_run": a.reference_method or None,
        "source": "examples/robocasa/combined_eval.py:TARGET_EVAL_EPISODES + the reference run's "
                  "per-episode turn steps; regenerate with scripts/build_eval30_manifest.py",
        "total_tasks": len(tasks),
        "total_episodes": n_eps,
        "tasks_without_success": sorted(t for t, v in tasks.items() if v["max_steps"] is None),
        "tasks": tasks,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1))

    nb = doc["tasks_without_success"]
    print(f"wrote {a.out}")
    print(f"  tasks {len(tasks)}  episodes {n_eps}")
    print(f"  max_steps set for {len(tasks) - len(nb)}/{len(tasks)} tasks "
          f"(reference run: {a.reference_method or 'none'}, {len(ref)} episodes matched)")
    if nb:
        print(f"  null (no reference success) for {len(nb)}: {', '.join(nb)}")
    if missing_len:
        print(f"  NOTE: {len(missing_len)} episode(s) had no demo length in the dataset metadata "
              "(demo lengths are no longer written to the file, so this is informational only)")
    if mismatch:
        print(f"  WARNING: {len(mismatch)} demo-length MISMATCH vs the run's n_recorded_frames: "
              f"{', '.join(mismatch[:8])}{' ...' if len(mismatch) > 8 else ''}")


if __name__ == "__main__":
    main()
