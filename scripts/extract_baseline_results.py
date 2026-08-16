"""Summarise a FLAT-POLICY baseline rollout tree into the same per-method JSON the GUI already reads.

WHAT THIS IS FOR. ``sys1_eval_results/baseline/<run>/<task>/episode_NNNNNN/{episode.json,rollout.mp4}``
is a Xiaomi-Robotics-1 RoboCasa365 eval: one flat policy, no System2, no turns. It carries no
``index.json`` and no turn records, so neither ``extract_combine_results.py`` (which aggregates a
combine tree's turns) nor the GUI's live fallback can read it. This script emits
``sys1_eval_results/combine_results/<run>.json`` in the EXACT schema
``subtask_eval_gui._combine_stats`` consumes, so the baseline lands in /stats §0 and §0b with the
ranking, the per-split table and the refined toggle all working, with no special-casing in the GUI
beyond one visibility checkbox.

HOW THE GUI KNOWS IT IS A BASELINE. By the ``kind: "baseline_flat_policy"`` field, NOT by the method
name -- the name is just the run directory, so runs can be renamed freely (``xiaomi-robo1``,
``xiaomi-robo1-unseenshort``, ...) without breaking the toggle. ``subtask_eval_gui`` passes ``kind``
through in ``_combine_stats`` and ``isBaselineMethod`` reads it; a ``baseline-*`` name still works as a
fallback for anything produced before the flag existed.

REFINED SCORING. Their ``horizon`` field was verified equal to ``horizon_gate.TASK_HORIZON`` for all
50 + 16 tasks, so the two evals are scored against the same official per-task step budget. Two
sources are used, in order:

  1. ``official_horizon_migration.success_after_official_horizon`` when present -- these runs were
     generated under a LONGER ``source_horizon`` and re-scored, so the flag is authoritative;
  2. otherwise ``steps <= horizon``.

Measured on the delivered trees, EVERY episode satisfies ``steps <= horizon`` and every migrated
success has ``success_after_official_horizon: false`` -- i.e. refined == raw here, because unlike our
hierarchical runs these rollouts were GENERATED with the official horizon as the cap and cannot
overshoot it. That is the point of comparison: their raw rate is already horizon-compliant, so it
belongs next to our REFINED rate, not our raw one. ``n_refined_unknown`` counts anything that cannot
be decided (a success beyond the horizon with no migration flag); it is 0 today and surfaced rather
than folded in, so the number can never be quietly computed over episodes nobody could score.

TERMINATIONS ARE SYNTHESISED, and labelled so no one mistakes them for recorded ones: a flat rollout
has no termination field, it simply runs to the horizon. ``env_success`` = solved,
``horizon_exhausted`` = did not. ``avg_turns`` is null because there are no turns.

Usage:
    python scripts/extract_baseline_results.py                 # all runs under baseline/
    python scripts/extract_baseline_results.py --run xr1-robocasa365-target1500-allvideos-20260813
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "examples" / "robocasa"))
try:
    from combined_eval import TARGET_TASK_SPLIT
except Exception:
    TARGET_TASK_SPLIT = {}
try:
    from horizon_gate import TASK_HORIZON
except Exception:
    TASK_HORIZON = {}

_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO.parent / "data", Path.home() / "data") if c.is_dir()), _REPO.parent / "data"))
RESULTS = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results")
BASELINE_ROOT = RESULTS / "baseline"
OUT_DIR = RESULTS / "combine_results"


def _norm(d: dict) -> dict:
    """One shape for both baseline schemas, because they disagree on the two fields that matter.

    XIAOMI (``xiaomi-robo1*``):  success / steps / horizon, plus an ``official_horizon_migration``
        block. ``success`` is the RAW outcome; the rollouts were generated under the official horizon
        so raw == refined in practice.

    PI05 (``pi05``, our own flat pi0.5 baseline, written in combined_eval's schema):
        ``raw_episode_success`` is the raw outcome and ``episode_success`` is ALREADY GATED at
        ``scoring_cap`` (== horizon_gate.TASK_HORIZON, verified equal for all 50 tasks). Reading
        ``episode_success`` as the raw rate understates it -- measured on the delivered tree, 64 of
        1500 episodes have raw=True / scored=False, i.e. they solved the task past the horizon. Steps
        live in ``n_steps`` and the horizon in ``scoring_cap``; there is no ``horizon`` key at all.
    """
    if "raw_episode_success" in d or "episode_success" in d:      # pi05 / combined_eval schema
        raw = bool(d.get("raw_episode_success", d.get("episode_success")))
        return {"raw": raw, "scored": d.get("episode_success"),
                "steps": d.get("n_steps"), "horizon": d.get("scoring_cap"),
                "success_step": d.get("success_step")}
    # ABOT (``abot_m05``) is the xiaomi shape with different horizon keys: no ``horizon`` at all, the
    # official budget in ``official_max_steps`` / ``robocasa_task_horizon`` (both verified equal to
    # TASK_HORIZON for all 50 tasks). Without this fallback every abot success scored as
    # refined_unknown, because horizon came back None.
    hz = next((d[k] for k in ("horizon", "scoring_cap", "official_max_steps",
                              "robocasa_task_horizon") if d.get(k) is not None), None)
    return {"raw": bool(d.get("success")), "scored": None,
            "steps": d.get("steps"), "horizon": hz, "success_step": None}


def _refined(d: dict) -> tuple[bool, bool]:
    """(refined_success, refined_unknown) for one baseline episode record.

    Computed from the STEP AT WHICH SUCCESS FIRED wherever that is recorded, so the definition is the
    same one horizon_gate applies to our hierarchical runs rather than a field we trust blindly. The
    producer's own gated flag is used only as a fallback.
    """
    n = _norm(d)
    if not n["raw"]:
        return False, False
    hz = n["horizon"]
    # 1. success_step vs horizon -- the direct measurement (pi05 records it).
    if n["success_step"] is not None and hz is not None:
        return int(n["success_step"]) <= int(hz), False
    # 2. the producer's pre-gated flag (pi05 without success_step).
    if n["scored"] is not None:
        return bool(n["scored"]), False
    # 3. xiaomi's migration block, then total steps vs horizon.
    mig = d.get("official_horizon_migration") or {}
    if "success_after_official_horizon" in mig:
        return not mig["success_after_official_horizon"], False
    if n["steps"] is None or hz is None:
        return False, True          # a success nobody can place against the horizon
    return int(n["steps"]) <= int(hz), False


def _block(eps: list[dict]) -> dict:
    n = len(eps)
    ns = sum(1 for e in eps if _norm(e)["raw"])
    sr = sum(1 for e in eps if e["_refined"])
    unk = sum(1 for e in eps if e["_refined_unknown"])
    secs = [e.get("seconds") or 0.0 for e in eps]
    return {
        "n": n,
        "n_success": ns,
        "rate": (ns / n) if n else None,
        "n_success_refined": sr,
        "rate_refined": (sr / n) if n else None,
        "n_refined_unknown": unk,
        "avg_seconds": round(sum(secs) / n, 2) if n else None,
        "total_seconds": round(sum(secs), 1),
        # No turns exist in a flat rollout. None (not 0) so the GUI renders a dash rather than
        # implying the policy solved everything in zero turns.
        "avg_turns": None,
    }


def extract(run_dir: Path) -> dict | None:
    eps: list[dict] = []
    # TWO LAYOUTS, matched by SHAPE not by episode-dir NAME. xiaomi nests <Task>/<episode dir>/; pi05
    # is flat, one dir per episode named <Task>__target__episode_NNNNNN/ (combined_eval's convention).
    # The nested pattern used to be `episode_*`, which silently found ZERO episodes in the newtask
    # baseline -- it names its episode dirs `seed_0001000009` after the reset seed, so the whole 680-
    # episode run extracted as "no episodes, skipped". Any per-episode dir name works now.
    files = (sorted(glob.glob(str(run_dir / "*" / "*" / "episode.json")))
             or sorted(glob.glob(str(run_dir / "*" / "episode.json"))))
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if d.get("error"):
            continue
        d["_refined"], d["_refined_unknown"] = _refined(d)
        d["_dir"] = str(Path(f).parent.relative_to(run_dir))
        eps.append(d)
    if not eps:
        return None
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for e in eps:
        by_task[e["task_name"]].append(e)
        by_split[TARGET_TASK_SPLIT.get(e["task_name"], "other")].append(e)
    # pi05 records a real termination; xiaomi has none, so it gets a synthesised one. Flagged in the
    # output either way so nobody reads a synthesised label as recorded.
    synth = not any(e.get("termination") for e in eps)
    term = collections.Counter(
        e.get("termination") or ("env_success" if _norm(e)["raw"] else "horizon_exhausted")
        for e in eps)
    # Horizon agreement is the precondition for comparing refined numbers at all, so it is recorded
    # in the output rather than only checked once by hand.
    hz_mismatch = sorted({(t, _norm(v[0])["horizon"], TASK_HORIZON.get(t))
                          for t, v in by_task.items()
                          if TASK_HORIZON.get(t) is not None
                          and _norm(v[0])["horizon"] != TASK_HORIZON[t]})
    return {
        # The run directory name, verbatim. See the docstring: the GUI identifies a baseline by the
        # `kind` field below, so the name carries no meaning and can be changed at will.
        "method": run_dir.name,
        "eval_kind": eps[0].get("eval_kind"),
        "kind": "baseline_flat_policy",
        "model_path": eps[0].get("model_path"),
        "instruction_mode": eps[0].get("instruction_mode"),
        "source": str(run_dir),
        "n_error": 0,
        "overall": _block(eps),
        "per_split": {k: _block(v) for k, v in sorted(by_split.items())},
        "per_task": {t: {**_block(v), "split": TARGET_TASK_SPLIT.get(t, "other"),
                         "horizon": _norm(v[0])["horizon"]}
                     for t, v in sorted(by_task.items())},
        "terminations": dict(term),
        "terminations_synthesised": synth,
        "horizon_mismatch_vs_ours": hz_mismatch,
        "bench": {},
        "src": "extracted-baseline",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--run", default=None, help="only this run directory (default: all)")
    a = ap.parse_args()
    if not a.baseline_root.is_dir():
        print(f"no baseline tree at {a.baseline_root}")
        return
    runs = ([a.baseline_root / a.run] if a.run
            else sorted(p for p in a.baseline_root.iterdir() if p.is_dir()))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for r in runs:
        rec = extract(r)
        if rec is None:
            print(f"  {r.name}: no episodes, skipped")
            continue
        (a.out_dir / f"{rec['method']}.json").write_text(json.dumps(rec, indent=1))
        o = rec["overall"]
        print(f"  {rec['method']}: {o['n_success']}/{o['n']} = {100*o['rate']:.1f}%  "
              f"refined {o['n_success_refined']}/{o['n']} = {100*o['rate_refined']:.1f}%"
              + (f"  ({o['n_refined_unknown']} unscorable)" if o["n_refined_unknown"] else "")
              + (f"  HORIZON MISMATCH {rec['horizon_mismatch_vs_ours']}"
                 if rec["horizon_mismatch_vs_ours"] else ""))
        wrote += 1
    print(f"wrote {wrote} baseline file(s) to {a.out_dir}")


if __name__ == "__main__":
    main()
