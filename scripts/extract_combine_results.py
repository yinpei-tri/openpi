"""Extract COMBINED System2+System1 eval results into small per-method summary JSONs.

The GUI's /stats page should not have to aggregate raw rollout output. Each method's
``index.json`` already avoids reading the per-episode docs, but it still carries EVERY episode
(~217 KB at 1000 episodes, ~5 MB at the full 25,307-episode benchmark) and /stats re-derives the
per-split and per-task tables from it on every cache miss. This script precomputes those tables
once, per method, into::

    $SYS1_RESULTS_DIR/combine_results/<method>.json   # summary + per_split + per_task
    $SYS1_RESULTS_DIR/combine_results/SUMMARY.json    # one row per method, for the table

so /stats can render from a handful of KB with no aggregation at all.

The output is DURABLE in the same sense as the episode_results/ files: it survives deleting the big
per-episode video dirs, so a finished sweep can be pruned on disk and still be reportable.

Run after a sweep (or any time; it is idempotent):

    uv run python scripts/extract_combine_results.py
    uv run python scripts/extract_combine_results.py --combine-root eval_results/combine
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import sys

# The task -> split map lives with the eval (inlined there, no sidecar JSON).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
try:
    from combined_eval import TARGET_TASK_SPLIT
except Exception:
    TARGET_TASK_SPLIT = {}
# REFINED rate: the same episodes re-scored under RoboCasa's official per-task step horizon. One
# implementation, in horizon_gate, shared with scripts/build_refined_results.py so the audit JSONs and
# the numbers the GUI renders can never disagree. Absent (older checkout, missing horizon table) the
# refined fields are simply all-zero and the GUI's toggle has nothing to show.
try:
    from horizon_gate import mark_refined
except Exception as _e:
    print(f"  NOTE: refined (official-horizon) scoring unavailable: {_e}")
    mark_refined = None

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen", "other")

# Derived, not hardcoded: env contract first, else the first existing "data" dir near the repo.
_REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[1].parent)
_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO_ROOT / "data", Path.home() / "data") if c.is_dir()), _REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results").expanduser()


def _stat_block(eps: list[dict]) -> dict:
    """success/n/rate + timing + turn stats for a group of episodes.

    ``n_success_refined`` / ``rate_refined`` re-score the same episodes under RoboCasa's official
    per-task step horizon (see horizon_gate.mark_refined). Same denominator ``n`` -- the horizon can only demote a
    success, never create one, so the refined rate is always <= rate.
    """
    n = len(eps)
    s = sum(1 for e in eps if e.get("episode_success"))
    secs = [e["seconds"] for e in eps if isinstance(e.get("seconds"), int | float)]
    turns = [e["n_turns"] for e in eps if isinstance(e.get("n_turns"), int)]
    sr = sum(1 for e in eps if e.get("refined_success"))
    unk = sum(1 for e in eps if e.get("refined_unknown"))
    return {
        "n": n,
        "n_success": s,
        "rate": (s / n) if n else None,
        "n_success_refined": sr,
        "rate_refined": (sr / n) if n else None,
        # Successes whose step-to-success could not be resolved (episode.json pruned, or no
        # success_step recorded). NOT counted as refined successes, and surfaced rather than folded
        # in, so a refined rate is never quietly computed over episodes nobody could score.
        "n_refined_unknown": unk,
        "avg_seconds": round(sum(secs) / len(secs), 2) if secs else None,
        "total_seconds": round(sum(secs), 1) if secs else None,
        "avg_turns": round(sum(turns) / len(turns), 2) if turns else None,
    }




def extract_method(method_dir: Path) -> dict | None:
    """Build one method's summary from its index.json (no per-episode docs read)."""
    idx = method_dir / "index.json"
    if not idx.exists():
        return None
    try:
        doc = json.loads(idx.read_text())
    except Exception as e:
        print(f"  WARNING: unreadable {idx}: {e}")
        return None
    eps = doc.get("episodes") or []
    if not eps:
        return None

    # Adds refined_success / refined_unknown to each record in memory (reads only the successful
    # episodes' episode.json). The eval tree itself is never written to.
    if mark_refined is not None:
        mark_refined(method_dir, eps)

    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for e in eps:
        task = e.get("task_name") or ""
        by_split[TARGET_TASK_SPLIT.get(task, "other")].append(e)
        by_task[task].append(e)

    # Episodes that errored are counted in `n` but never succeed; surface them explicitly so a
    # rate is never quietly computed over a denominator that includes crashes.
    n_error = sum(1 for e in eps if e.get("error"))
    terms = collections.Counter(e.get("termination") or "error" for e in eps)

    # GOAL SOURCE, from the run guard. Without this the aggregated tables cannot tell a terse-goal
    # arm from a full-goal one -- only the individual episode.json files carried it.
    goal = None
    plan = None
    try:
        run_cfg = json.loads((method_dir / "rule_config.json").read_text()) or {}
        goal = run_cfg.get("goal")
        plan = run_cfg.get("plan")
    except Exception:
        goal = plan = None
    out = {
        "method": method_dir.name,
        "eval_kind": "combine",
        "goal": goal,
        "plan": plan,
        "overall": _stat_block(eps),
        "n_error": n_error,
        "terminations": dict(sorted(terms.items(), key=lambda kv: -kv[1])),
        "per_split": {k: _stat_block(by_split[k]) for k in SPLITS if by_split.get(k)},
        "per_task": {
            t: {**_stat_block(v), "split": TARGET_TASK_SPLIT.get(t, "other")}
            for t, v in sorted(by_task.items())
        },
    }
    mean = out["overall"]["avg_seconds"]
    # Extrapolate the measured mean to the full benchmark (50 tasks x ~506 episodes).
    out["bench"] = {
        "episodes": 25307,
        "eta_serial_h": round(mean * 25307 / 3600, 1) if mean else None,
        "eta_8gpu_h": round(mean * 25307 / 3600 / 8, 1) if mean else None,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combine-root", type=Path, default=RESULTS_DIR / "combine")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "combine_results")
    a = ap.parse_args()

    if not a.combine_root.is_dir():
        raise SystemExit(f"no such combine root: {a.combine_root}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for md in sorted(p for p in a.combine_root.iterdir() if p.is_dir()):
        rec = extract_method(md)
        if rec is None:
            continue
        (a.out_dir / f"{md.name}.json").write_text(json.dumps(rec, indent=1))
        o = rec["overall"]
        summary.append({
            "method": md.name, **o, "n_error": rec["n_error"],
            "terminations": rec["terminations"],
            "per_split_rate": {k: v["rate"] for k, v in rec["per_split"].items()},
            "goal": rec.get("goal"),
            "plan": rec.get("plan"),
            "bench": rec["bench"],
        })
        print(f"  {md.name}: {o['n_success']}/{o['n']} = "
              f"{(100 * o['rate']):.1f}%  {o['avg_seconds']}s/ep"
              + (f"  refined {o['n_success_refined']}/{o['n']} = "
                 f"{(100 * o['rate_refined']):.1f}%" if o.get("rate_refined") is not None else "")
              + (f"  ({rec['n_error']} errored)" if rec["n_error"] else ""))

    (a.out_dir / "SUMMARY.json").write_text(json.dumps(
        {"eval_kind": "combine", "n_methods": len(summary), "methods": summary}, indent=1))
    print(f"wrote {len(summary)} method file(s) + SUMMARY.json to {a.out_dir}")


if __name__ == "__main__":
    main()
