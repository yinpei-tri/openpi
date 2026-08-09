"""Extract COMBINED System2+System1 eval results into small per-method summary JSONs.

The GUI's /stats page should not have to aggregate raw rollout output. Each method's
``index.json`` already avoids reading the per-episode docs, but it still carries EVERY episode
(~217 KB at 1000 episodes, ~5 MB at the full 25,307-episode benchmark) and /stats re-derives the
per-split and per-task tables from it on every cache miss. This script precomputes those tables
once, per method, into::

    eval_results/combine_results/<method>.json     # summary + per_split + per_task
    eval_results/combine_results/SUMMARY.json      # one row per method, for the table

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
from pathlib import Path
import sys

# The task -> split map lives with the eval (inlined there, no sidecar JSON).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "robocasa"))
try:
    from combined_eval import TARGET_TASK_SPLIT
except Exception:
    TARGET_TASK_SPLIT = {}

SPLITS = ("atomic_seen", "composite_seen", "composite_unseen", "other")


def _stat_block(eps: list[dict]) -> dict:
    """success/n/rate + timing + turn stats for a group of episodes."""
    n = len(eps)
    s = sum(1 for e in eps if e.get("episode_success"))
    secs = [e["seconds"] for e in eps if isinstance(e.get("seconds"), int | float)]
    turns = [e["n_turns"] for e in eps if isinstance(e.get("n_turns"), int)]
    return {
        "n": n,
        "n_success": s,
        "rate": (s / n) if n else None,
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

    out = {
        "method": method_dir.name,
        "eval_kind": "combine",
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
    ap.add_argument("--combine-root", type=Path, default=Path("eval_results/combine"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/combine_results"))
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
            "bench": rec["bench"],
        })
        print(f"  {md.name}: {o['n_success']}/{o['n']} = "
              f"{(100 * o['rate']):.1f}%  {o['avg_seconds']}s/ep"
              + (f"  ({rec['n_error']} errored)" if rec["n_error"] else ""))

    (a.out_dir / "SUMMARY.json").write_text(json.dumps(
        {"eval_kind": "combine", "n_methods": len(summary), "methods": summary}, indent=1))
    print(f"wrote {len(summary)} method file(s) + SUMMARY.json to {a.out_dir}")


if __name__ == "__main__":
    main()
