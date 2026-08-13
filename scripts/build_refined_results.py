"""Write the REFINED (official-horizon) re-scoring of every COMBINED eval method as JSON.

The refined rate re-scores our runs under RoboCasa's own per-task step budget: a success counts only
if the env's success check fired within ``horizon`` cumulative env steps. See
``examples/robocasa/horizon_gate.py`` for the definition and its caveats.

READ-ONLY with respect to the eval trees -- no episode record is touched. Output::

    $SYS1_RESULTS_DIR/combine_refined/<method>.json   # aggregates + per-task + EVERY demoted episode
    $SYS1_RESULTS_DIR/combine_refined/SUMMARY.json    # one row per method, raw vs refined

The per-method file is the audit trail: it names each win the gate removed and how many steps past
the budget it landed, so any refined number quoted in a table or a paper can be traced to episodes.

The GUI does NOT read these files -- ``extract_combine_results.py`` folds the same numbers into
``combine_results/<method>.json`` (which /stats already loads) via the same library, so the two can
never disagree. These files exist for auditing and for offline analysis.

    uv run python scripts/build_refined_results.py
    uv run python scripts/build_refined_results.py --methods s1-progact270k_s2-qwen35-4b-full-ep3-11416-base
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_RC = Path(__file__).resolve().parents[1] / "examples" / "robocasa"
sys.path.insert(0, str(_RC))
from horizon_gate import gate_method  # noqa: E402 -- needs the sys.path line above

try:
    from combined_eval import TARGET_TASK_SPLIT
except Exception:
    TARGET_TASK_SPLIT = {}

_REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[1].parent)
_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO_ROOT / "data", Path.home() / "data") if c.is_dir()), _REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combine-root", type=Path, default=RESULTS_DIR / "combine")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "combine_refined")
    ap.add_argument("--methods", nargs="*", default=None, help="method dir names; default all")
    a = ap.parse_args()

    if not a.combine_root.is_dir():
        raise SystemExit(f"no such combine root: {a.combine_root}")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    dirs = [p for p in sorted(a.combine_root.iterdir()) if p.is_dir()
            and (a.methods is None or p.name in a.methods)]
    summary = []
    for md in dirs:
        rec = gate_method(md, task_split=TARGET_TASK_SPLIT)
        if rec is None:
            continue
        (a.out_dir / f"{md.name}.json").write_text(json.dumps(rec, indent=1))
        o = rec["overall"]
        summary.append({"method": md.name, **o,
                        "n_demoted_by_task": {t: v["n_demoted"] for t, v in rec["per_task"].items()
                                              if v["n_demoted"]}})
        unk = f"  ({o['n_refined_unknown']} unscorable)" if o["n_refined_unknown"] else ""
        print(f"  {md.name}: raw {o['n_success']}/{o['n']} = {100 * (o['rate'] or 0):.1f}%  ->  "
              f"refined {o['n_success_refined']}/{o['n']} = {100 * (o['rate_refined'] or 0):.1f}%  "
              f"(-{o['n_demoted']}){unk}")

    (a.out_dir / "SUMMARY.json").write_text(json.dumps(
        {"eval_kind": "combine", "gate": "robocasa_official_task_horizon",
         "n_methods": len(summary), "methods": summary}, indent=1))
    print(f"wrote {len(summary)} method file(s) + SUMMARY.json to {a.out_dir}")


if __name__ == "__main__":
    main()
