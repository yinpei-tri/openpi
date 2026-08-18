"""Build cumulative step caps for RoboCasa composite tasks missing registry horizons.

This does not pretend inferred caps are official. Registered tasks keep RoboCasa's exact ``horizon``.
For an environment-only task, ``num_subtasks`` in RoboCasa's task attributes is used as the atomic
task-count proxy. By default, the cap is the *largest registered horizon* among the official
composite-unseen tasks with exactly the same count. This keeps inferred limits anchored to the
benchmark split being extended instead of to unrelated pretraining tasks.

The executable limits JSON is accompanied by an audit JSON containing every donor, duration, scaled
estimate, formula parameter, task-set definition, and RoboCasa revision. This makes the proxy budget
reviewable and regenerable instead of a hidden hand-tuned number.

Requires the RoboCasa Python environment. Examples:

    python scripts/build_procedural_robocasa_limits.py
    python scripts/build_procedural_robocasa_limits.py --task-set all_composite_unseen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "examples" / "robocasa" / "procedural_unseen_task_limits.json"


def _git_revision(repo: Path) -> str:
    return (subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-set", choices=["extra_unseen", "all_composite_unseen"], default="extra_unseen",
        help="extra_unseen = the 49 beyond the leaderboard's unseen16; all = all 65 composite tasks "
             "absent from pretrain300")
    parser.add_argument(
        "--tasks",
        default=None,
        help="optional comma-separated subset of the selected task set",
    )
    parser.add_argument(
        "--donor-task-set",
        choices=["official_composite_unseen", "all_registered_composite"],
        default="official_composite_unseen",
        help="registered tasks eligible to donate a same-subtask-count horizon",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import robocasa
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
    from robocasa.utils.dataset_registry import COMPOSITE_TASK_DATASETS
    from robocasa.utils.dataset_registry import PRETRAINING_TASKS
    from robocasa.utils.dataset_registry import TARGET_TASKS

    robocasa_root = Path(robocasa.__file__).parents[1]
    attributes_path = robocasa_root / "docs" / "composite_tasks" / "task_attributes.json"
    attributes = {row["name"]: row for row in json.loads(attributes_path.read_text())["tasks"]}

    envs = set(ALL_KITCHEN_ENVIRONMENTS)
    composite365 = {
        name for name in envs
        if ".composite." in REGISTERED_KITCHEN_ENVS[name].__module__
    }
    pretrain = set(PRETRAINING_TASKS["pretrain300"])
    official_unseen = set(TARGET_TASKS["composite_unseen"])
    tasks = composite365 - pretrain
    if args.task_set == "extra_unseen":
        tasks -= official_unseen
    if args.tasks:
        requested = {task.strip() for task in args.tasks.split(",") if task.strip()}
        unknown = sorted(requested - tasks)
        if unknown:
            raise SystemExit(f"--tasks contains tasks outside {args.task_set}: {unknown}")
        tasks = requested

    donor_tasks = (
        official_unseen
        if args.donor_task_set == "official_composite_unseen"
        else set(COMPOSITE_TASK_DATASETS)
    )

    limits: dict[str, int] = {}
    audit_tasks: dict[str, dict] = {}
    for task in sorted(tasks):
        cfg = COMPOSITE_TASK_DATASETS.get(task)
        if cfg and cfg.get("horizon") is not None:
            cap = int(cfg["horizon"])
            limits[task] = cap
            audit_tasks[task] = {
                "max_official_steps": cap,
                "source": "robocasa_dataset_registry",
                "derived": False,
            }
            continue

        attr = attributes[task]
        donor_pool = []
        for donor in sorted(donor_tasks):
            donor_cfg = COMPOSITE_TASK_DATASETS.get(donor, {})
            if donor not in attributes or donor_cfg.get("horizon") is None:
                continue
            donor_attr = attributes[donor]
            if donor_attr["num_subtasks"] != attr["num_subtasks"]:
                continue
            donor_horizon = int(donor_cfg["horizon"])
            donor_pool.append({
                "task": donor,
                "activity": donor_attr["activity"],
                "moma_required": donor_attr["moma_required"],
                "num_subtasks": donor_attr["num_subtasks"],
                "registered_horizon": donor_horizon,
            })
        if not donor_pool:
            raise RuntimeError(f"no registered same-subtask-count donors for {task}")
        cap = max(row["registered_horizon"] for row in donor_pool)
        donors = sorted(
            (row for row in donor_pool if row["registered_horizon"] == cap),
            key=lambda row: row["task"],
        )
        limits[task] = cap
        audit_tasks[task] = {
            "max_official_steps": cap,
            "source": (
                "longest_official_composite_unseen_horizon_with_same_num_subtasks"
                if args.donor_task_set == "official_composite_unseen"
                else "longest_registered_horizon_with_same_num_subtasks"
            ),
            "derived": True,
            "activity": attr["activity"],
            "moma_required": attr["moma_required"],
            "num_subtasks": attr["num_subtasks"],
            "eligible_registered_donor_count": len(donor_pool),
            "longest_donors": donors,
        }

    audit = {
        "schema": 1,
        "task_set": ("custom_subset" if args.tasks else args.task_set),
        "parent_task_set": args.task_set,
        "definition": (
            f"selected {len(tasks)}-task subset of {args.task_set}"
            if args.tasks
            else (
                "RoboCasa365 composite tasks absent from pretrain300, excluding the official "
                "composite_unseen16" if args.task_set == "extra_unseen"
                else "all RoboCasa365 composite tasks absent from pretrain300"
            )
        ),
        "n_tasks": len(limits),
        "n_registered": sum(not row["derived"] for row in audit_tasks.values()),
        "n_derived": sum(row["derived"] for row in audit_tasks.values()),
        "robocasa_revision": _git_revision(robocasa_root),
        "parameters": {
            "atomic_task_count_proxy": "task_attributes.num_subtasks",
            "selection": "maximum registered composite horizon at identical num_subtasks",
            "donor_task_set": args.donor_task_set,
        },
        "tasks": audit_tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(limits, indent=2, sort_keys=True) + "\n")
    audit_path = args.output.with_name(args.output.stem + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"WROTE {args.output} ({len(limits)} tasks)")
    print(f"WROTE {audit_path} ({audit['n_registered']} registered, {audit['n_derived']} derived)")


if __name__ == "__main__":
    main()
