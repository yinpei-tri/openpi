"""Re-score eval results under RoboCasa's OFFICIAL per-task step horizon -- the "refined" number.

WHY THIS EXISTS. RoboCasa scores a rollout by stepping the env at most ``horizon`` steps (per task,
from its dataset registry) while polling ``env._check_success()`` densely. Historical and default
50-task hierarchical runs budget per SUBGOAL (``est_length * horizon_mult``, capped by
``--max-steps-cap``) and per TURN (``max_turns``), not by cumulative steps, so an episode can execute
far past the horizon. New extra-task runs can enforce ``max_official_steps`` online; this gate remains
the shared scorer for both schemas and for old results.
Worst case measured: a GetToastedBread wait subgoal forced 1200 steps against a 500-step toaster and a
3000-step horizon, and 8 wins landed outside the budget.

WHAT THE GATE IS. A success counts only if the env's success check first fired at or before
``horizon`` CUMULATIVE env steps. It is a pure post-hoc filter: it can only demote a success, never
create one, because truncating a rollout cannot produce a success the longer rollout did not have. So
``rate_refined <= rate`` always, over the same denominator.

WHAT IT IS NOT. The cut is applied to rollouts GENERATED without the horizon. A policy that knew it
had 450 steps might act differently, so this answers "what would our runs have scored under the
official protocol", not "what would this method score if tuned for it".

READ-ONLY. Nothing here writes into an eval tree; the episode records are never modified. The CLI
(scripts/build_refined_results.py) writes separate JSON files.

Usage:
    from horizon_gate import gate_episode, gate_method, TASK_HORIZON

    g = gate_episode(json.load(open("…/episode.json")))
    g["refined"]      # bool: success within the official horizon
    g["overshoot"]    # steps past the horizon, or None

    rec = gate_method(Path("…/combine/<method>"))
    rec["overall"]["rate_refined"], rec["demoted"]      # aggregate + the episodes that lost a win
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

try:
    from robocasa_horizons import TASK_HORIZON
except ImportError:  # importable from anywhere, e.g. scripts/ with a different sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from robocasa_horizons import TASK_HORIZON

__all__ = ["TASK_HORIZON", "gate_episode", "gate_method", "mark_refined", "success_cum_step"]

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen", "other")


def success_cum_step(doc: dict) -> int | None:
    """Cumulative env step at which the env's success check first fired, or None if it never did.

    ``success_step`` on a turn record is the 0-based index WITHIN that turn's own segment
    (combined_eval.run_s1_segment: ``success_step = executed - 1``), so the cumulative index is the
    steps of every earlier turn plus ``success_step + 1``. The loop breaks on the FIRST turn that
    carries one, which is also the only one that can: the segment stops on env success.
    """
    cum = 0
    for t in doc.get("turns") or []:
        ss = t.get("success_step")
        if ss is not None:
            return cum + ss + 1
        cum += t.get("n_steps") or 0
    return None


def total_steps(doc: dict) -> int:
    """Env steps the episode actually executed, summed over turns."""
    return sum((t.get("n_steps") or 0) for t in (doc.get("turns") or []))


def gate_episode(doc: dict, *, task: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    """Gate ONE episode.json document against its task's official horizon.

    ``unknown`` marks a success that could not be scored (no horizon for the task, or no
    ``success_step`` recorded). Such an episode is NOT counted as a refined success -- but it is
    reported, so a refined rate is never quietly computed over episodes nobody could score.
    """
    task = task or doc.get("task_name") or ""
    # Procedural/extra tasks may carry a configured official-step cap in the episode itself even when
    # they are absent from the generated 50-task table. Prefer an explicit function argument, then the
    # episode's audited cap, then the vendored benchmark table.
    h = (horizon if horizon is not None
         else doc.get("max_official_steps")
         if doc.get("max_official_steps") is not None
         else TASK_HORIZON.get(task))
    ok = bool(doc.get("episode_success"))
    cum = success_cum_step(doc)
    unknown = bool(ok and (h is None or cum is None))
    refined = bool(ok and h is not None and cum is not None and cum <= h)
    return {
        "task": task,
        "horizon": h,
        "success": ok,
        "refined": refined,
        "unknown": unknown,
        "step_at_success": cum,
        "total_steps": total_steps(doc),
        "overshoot": (cum - h) if (ok and cum is not None and h is not None and cum > h) else None,
        "termination": doc.get("termination"),
    }


def mark_refined(method_dir: Path, eps: list[dict], *, task_split: dict[str, str] | None = None) -> None:
    """Annotate index.json records IN MEMORY with ``refined_success`` / ``refined_unknown``.

    Only SUCCESSFUL episodes are opened -- the horizon cannot promote a failure -- so this reads a few
    hundred episode.json files per method rather than the whole sweep. ``eps`` are the light records
    from a method's index.json; they are mutated in place and nothing is written to disk.
    """
    for e in eps:
        if not e.get("episode_success"):
            continue
        ep_dir = Path(method_dir) / str(e.get("episode_id") or "").replace("/", "__")
        try:
            doc = json.loads((ep_dir / "episode.json").read_text())
        except Exception:
            e["refined_unknown"] = True
            continue
        g = gate_episode(doc, task=e.get("task_name"))
        if g["refined"]:
            e["refined_success"] = True
        if g["unknown"]:
            e["refined_unknown"] = True
        e["step_at_success"] = g["step_at_success"]
        if g["overshoot"] is not None:
            e["overshoot"] = g["overshoot"]


def _agg(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    s = sum(1 for r in rows if r["success"])
    sr = sum(1 for r in rows if r["refined"])
    return {
        "n": n,
        "n_success": s,
        "rate": (s / n) if n else None,
        "n_success_refined": sr,
        "rate_refined": (sr / n) if n else None,
        "n_refined_unknown": sum(1 for r in rows if r["unknown"]),
        "n_demoted": s - sr - sum(1 for r in rows if r["unknown"]),
    }


def gate_method(method_dir: Path, *, task_split: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Gate a whole method directory (``…/combine/<method>``). Returns None if it has no episodes.

    Reads every episode.json, so this is the AUDIT path -- it reports each demoted episode with its
    overshoot. ``extract_combine_results.py`` uses ``mark_refined`` instead, which opens only the
    successes.
    """
    method_dir = Path(method_dir)
    rows: list[dict] = []
    for f in sorted(method_dir.glob("*__episode_*/episode.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        if not doc.get("termination"):      # written at plan time; no termination = never finished
            continue
        g = gate_episode(doc)
        g["episode_id"] = doc.get("episode_id") or f.parent.name
        g["dir"] = f.parent.name
        rows.append(g)
    if not rows:
        return None

    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
        by_split[(task_split or {}).get(r["task"], "other")].append(r)
    demoted = [r for r in rows if r["success"] and not r["refined"] and not r["unknown"]]
    return {
        "method": method_dir.name,
        "gate": "robocasa_official_task_horizon",
        "overall": _agg(rows),
        "per_split": {k: _agg(by_split[k]) for k in SPLITS if by_split.get(k)},
        "per_task": {
            t: {**_agg(v), "horizon": v[0]["horizon"],
                "overshoots": sorted((r["overshoot"] for r in v if r["overshoot"]), reverse=True)}
            for t, v in sorted(by_task.items())
        },
        # Every win the gate removed, with how far past the budget it landed -- the audit trail for
        # any refined number quoted anywhere.
        "demoted": sorted(
            ({"episode_id": r["episode_id"], "task": r["task"], "horizon": r["horizon"],
              "step_at_success": r["step_at_success"], "overshoot": r["overshoot"]}
             for r in demoted),
            key=lambda r: (-r["overshoot"], r["task"]),
        ),
        "unknown": [r["episode_id"] for r in rows if r["unknown"]],
    }
