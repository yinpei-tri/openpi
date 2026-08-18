#!/usr/bin/env python3
"""Generate a frozen RoboCasa composite-unseen reset bank.

By default the bank includes every RoboCasa365 composite environment absent from ``pretrain300``;
``--tasks`` creates an explicitly selected subset. Each task gets target-split resets. Generation
delegates to ``examples/robocasa/seeded_combined_eval.py`` so every reset has
the exact model XML, simulator state, episode metadata, task goal, individual initial camera PNGs,
and an annotated three-camera preview.

The operation is resumable. Complete bundles are checksum-verified and reused; they are never
regenerated unless the low-level command is explicitly run with ``--overwrite-reset-bundles``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/home/ec2-user/data/new_tasks")
DEFAULT_ROBOCASA_PY = Path("/home/ec2-user/micromamba/envs/robocasa/bin/python")
RESET_FILES = {
    "reset.json", "ep_meta.json", "task_goal.txt", "initial_state.npz", "model.xml.gz",
    "initial_agentview_left.png", "initial_agentview_right.png", "initial_eye_in_hand.png",
    "initial_views.png",
}
_PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed-base", type=int, default=1_000_000)
    parser.add_argument("--episodes", default="0-19")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--robocasa-python", type=Path, default=DEFAULT_ROBOCASA_PY)
    parser.add_argument(
        "--tasks", default=None,
        help="optional comma-separated subset for smoke testing; default is all 65 unseen tasks")
    parser.add_argument(
        "--reuse-existing-limits",
        action="store_true",
        help="reuse MAX_OFFICIAL_STEPS.json and its audit instead of regenerating them",
    )
    return parser.parse_args()


def _git_revision(repo: Path) -> str:
    return (subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "unknown")


def _parse_episode_slots(spec: str) -> list[int]:
    slots = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            slots.extend(range(int(lo), int(hi) + 1))
        else:
            slots.append(int(part))
    return sorted(set(slots))


def _task_sets() -> tuple[list[str], set[str]]:
    from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
    from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
    from robocasa.utils.dataset_registry import PRETRAINING_TASKS
    from robocasa.utils.dataset_registry import TARGET_TASKS

    composite = {
        name for name in ALL_KITCHEN_ENVIRONMENTS
        if ".composite." in REGISTERED_KITCHEN_ENVS[name].__module__
    }
    unseen = sorted(composite - set(PRETRAINING_TASKS["pretrain300"]))
    official = set(TARGET_TASKS["composite_unseen"])
    if len(unseen) != 65 or len(official) != 16:
        raise RuntimeError(
            f"RoboCasa task-set drift: expected 65 unseen / 16 official, got "
            f"{len(unseen)} / {len(official)}")
    return unseen, official


def _run_task(task: str, task_index: int, args: argparse.Namespace, gpus: list[str]) -> dict:
    gpu = gpus[task_index % len(gpus)]
    log_dir = args.root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task}.log"
    cmd = [
        str(args.robocasa_python),
        str(REPO / "examples" / "robocasa" / "seeded_combined_eval.py"),
        "--task", task,
        "--episodes", args.episodes,
        "--seed-base", str(args.seed_base),
        "--scene-split", "target",
        "--reset-only",
        "--reset-bundle-root", str(args.root),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        # RoboSuite validates this against the physical IDs listed in CUDA_VISIBLE_DEVICES rather
        # than treating the visible device as re-indexed to zero.
        "MUJOCO_EGL_DEVICE_ID": gpu,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "NUMBA_DISABLE_JIT": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": ":".join(filter(None, [
            str(REPO / "packages" / "openpi-client" / "src"),
            "/home/ec2-user/robocasa",
            env.get("PYTHONPATH", ""),
        ])),
    })
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                              check=False)
    with _PRINT_LOCK:
        print(f"[{task_index + 1:02d}] GPU {gpu} {task}: "
              f"{'OK' if proc.returncode == 0 else f'FAILED({proc.returncode})'}", flush=True)
    return {"task": task, "gpu": gpu, "returncode": proc.returncode, "log": str(log_path)}


def _validate_task(root: Path, task: str, seed_base: int, slots: list[int]) -> dict:
    manifest_path = root / task / "manifest.json"
    errors = []
    if not manifest_path.is_file():
        return {"task": task, "complete": False, "n_resets": 0,
                "errors": [f"missing {manifest_path}"]}
    manifest = json.loads(manifest_path.read_text())
    by_seed = {int(row["seed"]): row for row in manifest.get("resets", [])}
    expected = {seed_base + slot for slot in slots}
    for seed in sorted(expected):
        bundle = root / task / f"seed_{seed:010d}"
        missing = sorted(name for name in RESET_FILES if not (bundle / name).is_file())
        if missing:
            errors.append(f"seed {seed}: missing {missing}")
        if seed not in by_seed:
            errors.append(f"seed {seed}: absent from manifest")
    return {
        "task": task,
        "complete": not errors and expected.issubset(by_seed),
        "n_resets": len(expected & set(by_seed)),
        "errors": errors,
        "manifest": str(manifest_path),
    }


def _write_readme(root: Path, benchmark: dict) -> None:
    content = f"""# RoboCasa frozen composite-unseen reset bank v1

This directory is an evaluation input, not a demonstration dataset. It contains no recorded
actions. It freezes the exact initial condition for {benchmark['n_tasks']} tasks ×
{benchmark['episodes_per_task']} episodes ({benchmark['n_resets_expected']} resets total).

- Definition: {benchmark['definition']}.
- Composition: {benchmark['n_official_composite_unseen']} official composite-unseen tasks plus
  {benchmark['n_additional_unseen']} additional unseen tasks.
- Scene/object split: target.
- Seed lookup: `actual_seed = {benchmark['seed_base']} + episode_slot`, slots
  `{benchmark['episode_slots'][0]}..{benchmark['episode_slots'][-1]}`.
- Authoritative initial condition: `model.xml.gz` + `initial_state.npz` + `ep_meta.json`.
- Human inspection: `task_goal.txt`, three individual initial camera PNGs, and
  `initial_views.png` (annotated contact sheet).
- Rollout budget reference: `MAX_OFFICIAL_STEPS.json`; pass it directly as
  `--task-limits-json {root / 'MAX_OFFICIAL_STEPS.json'}`. Values without a registered RoboCasa
  horizon are conservative estimates, fully documented in `MAX_OFFICIAL_STEPS.audit.json`.

Use `examples/robocasa/seeded_combined_eval.py --load-reset-root {root}` for every experimental
arm. Do not regenerate bundles between model/rule comparisons. The numeric seed is provenance;
the frozen files and their SHA-256 values in `reset.json` define the benchmark.

See `BENCHMARK.json` for the task list, completion state, revisions, and per-task manifests.
"""
    (root / "README.md").write_text(content)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU")
    args.workers = min(args.workers, len(gpus))
    slots = _parse_episode_slots(args.episodes)
    if not slots:
        raise SystemExit("--episodes selected no seed slots")

    all_unseen, official = _task_sets()
    if args.tasks:
        requested = [task.strip() for task in args.tasks.split(",") if task.strip()]
        unknown = sorted(set(requested) - set(all_unseen))
        if unknown:
            raise SystemExit(f"--tasks contains tasks outside the unseen65 set: {unknown}")
        tasks = sorted(set(requested))
    else:
        tasks = all_unseen

    args.root.mkdir(parents=True, exist_ok=True)
    caps_path = args.root / "MAX_OFFICIAL_STEPS.json"
    cap_audit_path = caps_path.with_name(caps_path.stem + ".audit.json")
    if args.reuse_existing_limits:
        missing = [path for path in (caps_path, cap_audit_path) if not path.is_file()]
        if missing:
            raise SystemExit(f"--reuse-existing-limits requested but files are missing: {missing}")
        print(f"Reusing {caps_path} and {cap_audit_path}", flush=True)
    else:
        cap_env = os.environ.copy()
        cap_env["NUMBA_DISABLE_JIT"] = "1"
        cap_env["PYTHONPATH"] = ":".join(filter(None, [
            "/home/ec2-user/robocasa", cap_env.get("PYTHONPATH", ""),
        ]))
        subprocess.run(
            [
                str(args.robocasa_python),
                str(REPO / "scripts" / "build_procedural_robocasa_limits.py"),
                "--task-set", "all_composite_unseen",
                "--tasks", ",".join(tasks),
                "--donor-task-set", "official_composite_unseen",
                "--output", str(caps_path),
            ],
            cwd=REPO, env=cap_env, check=True,
        )
    print(f"Generating {len(tasks)} tasks × {len(slots)} resets under {args.root} "
          f"with {args.workers} workers", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_task, task, i, args, gpus): task
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            results.append(future.result())

    validations = [_validate_task(args.root, task, args.seed_base, slots) for task in tasks]
    failed_runs = sorted(row["task"] for row in results if row["returncode"] != 0)
    incomplete = sorted(row["task"] for row in validations if not row["complete"])
    caps = json.loads(caps_path.read_text())
    cap_audit = json.loads(caps_path.with_name(caps_path.stem + ".audit.json").read_text())
    benchmark = {
        "schema": 1,
        "name": f"robocasa365-composite-unseen{len(tasks)}-newseed-v1",
        "created_or_updated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": (
            f"selected {len(tasks)}-task composite-unseen benchmark from RoboCasa365 "
            "environments absent from pretrain300"
            if args.tasks
            else "all RoboCasa365 composite environments absent from pretrain300"
        ),
        "root": str(args.root.resolve()),
        "scene_split": "target",
        "seed_base": args.seed_base,
        "episode_slots": slots,
        "episodes_per_task": len(slots),
        "n_tasks": len(tasks),
        "n_resets_expected": len(tasks) * len(slots),
        "n_resets_complete": sum(row["n_resets"] for row in validations),
        "n_official_composite_unseen": sum(task in official for task in tasks),
        "n_additional_unseen": sum(task not in official for task in tasks),
        "complete": not failed_runs and not incomplete,
        "failed_runs": failed_runs,
        "incomplete_tasks": incomplete,
        "openpi_revision": _git_revision(REPO),
        "robocasa_revision": _git_revision(Path(__import__("robocasa").__file__).parents[1]),
        "max_official_steps_reference": str(caps_path),
        "max_official_steps_audit": str(caps_path.with_name(
            caps_path.stem + ".audit.json")),
        "tasks": [
            {
                **validation,
                "split_group": ("official_composite_unseen16" if task in official
                                else "additional_unseen49"),
                "max_official_steps_reference": caps.get(task),
                "max_official_steps_source": cap_audit["tasks"][task]["source"],
                "max_official_steps_derived": cap_audit["tasks"][task]["derived"],
            }
            for task, validation in zip(tasks, validations, strict=True)
        ],
    }
    (args.root / "BENCHMARK.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    _write_readme(args.root, benchmark)
    print(f"WROTE {args.root / 'BENCHMARK.json'}", flush=True)
    print(f"COMPLETE {benchmark['n_resets_complete']}/{benchmark['n_resets_expected']} resets",
          flush=True)
    if not benchmark["complete"]:
        raise SystemExit(f"reset bank incomplete; failed={failed_runs}, incomplete={incomplete}")


if __name__ == "__main__":
    main()
