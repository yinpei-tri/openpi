"""Regenerate ``examples/robocasa/robocasa_horizons.py`` from the RoboCasa dataset registry.

The horizon is RoboCasa's own per-task step budget for a rollout, and it is the denominator of the
REFINED success rate reported by ``scripts/extract_combine_results.py`` (a success counts only if the
env's success check fired within ``horizon`` cumulative env steps). Keeping a generated copy in-tree
means the openpi venv -- which has neither robosuite nor robocasa -- can score runs, and that the
numbers behind a published table are auditable here rather than in a sibling checkout.

Needs the robocasa env, because importing the registry imports robosuite:

    ~/micromamba/envs/robocasa/bin/python scripts/build_robocasa_horizons.py
    ~/micromamba/envs/robocasa/bin/python scripts/build_robocasa_horizons.py --check   # CI-style

``--check`` regenerates in memory and exits non-zero if the in-tree file is stale, without writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parents[1] / "examples" / "robocasa" / "robocasa_horizons.py"


def eval_tasks() -> list[str]:
    """The 50 tasks of combined_eval.TARGET_EVAL_EPISODES, read without importing it.

    combined_eval pulls in jax/lerobot/robosuite, none of which this script needs, so the manifest is
    sliced out of the source text instead.
    """
    src = (OUT.parent / "combined_eval.py").read_text()
    i = src.index("TARGET_EVAL_EPISODES")
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
    return sorted(eval(src[j:k + 1]))  # a dict literal from our own source


def render() -> str:
    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
    from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS

    # Stamp the robocasa revision the numbers came from, so a stale copy is diagnosable.
    repo = Path(__import__("robocasa").__file__).parents[1]
    rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    rows = []
    for t in eval_tasks():
        cfg = ATOMIC_TASK_DATASETS.get(t) or COMPOSITE_TASK_DATASETS.get(t)
        if cfg is None or "horizon" not in cfg:
            print(f"  WARNING: no horizon in the registry for {t} -- omitted", file=sys.stderr)
            continue
        rows.append((t, cfg["horizon"], "atomic" if t in ATOMIC_TASK_DATASETS else "composite"))
    body = "".join(f'    "{t}": {h},{" " * max(1, 34 - len(t) - len(str(h)))}# {kind}\n'
                   for t, h, kind in rows)
    return f'''"""RoboCasa's OFFICIAL per-task rollout horizon, in env steps -- the benchmark's own step budget.

GENERATED, do not hand-edit. Source of truth is ``robocasa.utils.dataset_registry``
(``ATOMIC_TASK_DATASETS`` / ``COMPOSITE_TASK_DATASETS``, field ``horizon``, read by
``dataset_registry_utils.get_task_horizon``) at robocasa revision {rev}. Copied in-tree so scoring
does not need robosuite/robocasa importable -- the openpi venv has neither, and a number the eval
tables depend on should be auditable in this repo rather than fetched from a sibling checkout.

WHY IT MATTERS. RoboCasa scores a rollout by stepping the env at most ``horizon`` steps while polling
``env._check_success()`` densely. Our hierarchical eval budgets per SUBGOAL (est_length *
horizon_mult, capped by --max-steps-cap) and per TURN (max_turns) and never counts total env steps,
so an episode can legitimately execute more than ``horizon`` steps -- which makes our raw success
rate incomparable with any number measured under the official protocol. ``scripts/extract_combine_
results.py`` therefore also reports a REFINED rate: a success counts only if the env's success check
first fired at or before ``horizon`` cumulative steps.

To regenerate (needs the robocasa env, e.g. ~/micromamba/envs/robocasa/bin/python):
    python scripts/build_robocasa_horizons.py
"""

from __future__ import annotations

# task -> official horizon in env steps. {len(rows)} tasks: the 50 in combined_eval.TARGET_EVAL_EPISODES.
TASK_HORIZON: dict[str, int] = {{
{body}}}
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the in-tree file is stale")
    a = ap.parse_args()
    new = render()
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != new:
            print(f"STALE: {OUT} differs from the registry -- rerun without --check", file=sys.stderr)
            raise SystemExit(1)
        print(f"up to date: {OUT}")
        return
    OUT.write_text(new)
    n = new.count('\n    "')
    print(f"wrote {OUT} ({n} tasks)")


if __name__ == "__main__":
    main()
