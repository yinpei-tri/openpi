"""Build a /stats row from PUBLISHED leaderboard percentages, for a model we cannot run ourselves.

WHY THIS EXISTS. https://robocasa.ai/leaderboard.html reports per-task success RATES for models whose
weights or rollouts we do not have. Those numbers are directly comparable with ours -- same 50 tasks,
same 30 episodes per task, same official protocol -- so they belong in the table. But there is NO
EPISODE DATA behind them: no videos, no steps, no terminations, nothing /combine could open.

So the row is SYNTHETIC and says so in every place a reader might look:
  * ``synthetic: true`` and ``kind: "baseline_published"`` in the JSON;
  * ``source_url`` recording where the numbers came from;
  * ``avg_seconds`` / ``avg_turns`` / ``terminations`` left empty rather than invented, so the table
    shows a dash instead of a fabricated latency;
  * no ``combine/<method>/`` tree, so the /combine browser simply does not list it.

COUNTS FROM PERCENTAGES. Every published rate is a multiple of 1/30 (30 episodes per task), so
``round(pct * 30 / 100)`` recovers the integer success count exactly -- 96.67% -> 29, 3.33% -> 1. The
script ASSERTS that the recovered per-split sums equal the published aggregates, so a typo in a
transcribed percentage fails loudly instead of quietly shifting a rate.

REFINED == RAW, deliberately, and this is the one judgement call. The leaderboard protocol steps each
episode at most ``horizon`` times, so a success past the horizon cannot exist in it by construction --
the same reason refined == raw held for the xiaomi and abot_m05 rollouts, which we verified from their
episode records. Here there are no records to verify, so it is an assumption; it is flagged as
``refined_assumed_equal_raw`` in the JSON. The alternative (null refined) would drop the row out of the
default view entirely, which hides a comparable result for a worse reason.

Usage:
    python scripts/make_leaderboard_method.py                     # writes abot-m0.6
    python scripts/make_leaderboard_method.py --method-name foo --dry-run
"""

from __future__ import annotations

import argparse
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

_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO.parent / "data", Path.home() / "data") if c.is_dir()), _REPO.parent / "data"))
OUT_DIR = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results") / "combine_results"

EPISODES_PER_TASK = 30
SOURCE_URL = "https://robocasa.ai/leaderboard.html"

# Published per-task success rates, transcribed verbatim. AGGREGATE lines are kept as the check.
PUBLISHED: dict[str, dict[str, float]] = {
    "atomic_seen": {
        "CloseBlenderLid": 30.00, "CloseFridge": 96.67, "CloseToasterOvenDoor": 90.00,
        "CoffeeSetupMug": 56.67, "NavigateKitchen": 66.67, "OpenCabinet": 96.67,
        "OpenDrawer": 76.67, "OpenStandMixerHead": 96.67, "PickPlaceCounterToCabinet": 76.67,
        "PickPlaceCounterToStove": 76.67, "PickPlaceDrawerToCounter": 80.00,
        "PickPlaceSinkToCounter": 90.00, "PickPlaceToasterToCounter": 100.00,
        "SlideDishwasherRack": 83.33, "TurnOffStove": 33.33, "TurnOnElectricKettle": 96.67,
        "TurnOnMicrowave": 86.67, "TurnOnSinkFaucet": 96.67,
    },
    "composite_seen": {
        "DeliverStraw": 16.67, "GetToastedBread": 50.00, "KettleBoiling": 46.67,
        "LoadDishwasher": 80.00, "PackIdenticalLunches": 16.67, "PreSoakPan": 93.33,
        "PrepareCoffee": 33.33, "RinseSinkBasin": 63.33, "ScrubCuttingBoard": 56.67,
        "SearingMeat": 13.33, "SetUpCuttingStation": 50.00, "StackBowlsCabinet": 56.67,
        "SteamInMicrowave": 66.67, "StirVegetables": 30.00, "StoreLeftoversInBowl": 56.67,
        "WashLettuce": 43.33,
    },
    "composite_unseen": {
        "ArrangeBreadBasket": 0.00, "ArrangeTea": 0.00, "BreadSelection": 23.33,
        "CategorizeCondiments": 13.33, "CuttingToolSelection": 10.00, "GarnishPancake": 36.67,
        "GatherTableware": 0.00, "HeatKebabSandwich": 3.33, "MakeIceLemonade": 0.00,
        "PanTransfer": 0.00, "PortionHotDogs": 0.00, "RecycleBottlesByType": 10.00,
        "SeparateFreezerRack": 3.33, "WaffleReheat": 0.00, "WashFruitColander": 13.33,
        "WeighIngredients": 13.33,
    },
}
# The published AGGREGATE per split: (n_success, n). The script fails if the recovered counts disagree.
AGGREGATES = {"atomic_seen": (429, 540), "composite_seen": (232, 480), "composite_unseen": (38, 480)}


def _block(ns: int, n: int) -> dict:
    return {
        "n": n, "n_success": ns, "rate": (ns / n) if n else None,
        # See the module docstring: assumed equal to raw because the published protocol is
        # horizon-bounded by construction. Flagged at the top level of the record.
        "n_success_refined": ns, "rate_refined": (ns / n) if n else None,
        "n_refined_unknown": 0,
        # Not published. Left null rather than invented -- the table renders a dash.
        "avg_seconds": None, "total_seconds": None, "avg_turns": None,
    }


def build(method: str) -> dict:
    per_task: dict[str, dict] = {}
    per_split: dict[str, dict] = {}
    bad_names: list[str] = []
    for split, rates in PUBLISHED.items():
        s_ns = 0
        for task, pct in sorted(rates.items()):
            exact = pct * EPISODES_PER_TASK / 100.0
            ns = round(exact)
            # Every published rate must be a multiple of 1/30; anything else means a mis-transcription.
            if abs(exact - ns) > 0.02:
                raise SystemExit(f"{task}: {pct}% is not a multiple of 1/{EPISODES_PER_TASK} "
                                 f"({exact:.4f}) -- check the transcription")
            if TARGET_TASK_SPLIT and TARGET_TASK_SPLIT.get(task) != split:
                bad_names.append(f"{task}: published under {split}, "
                                 f"ours says {TARGET_TASK_SPLIT.get(task)}")
            per_task[task] = {**_block(ns, EPISODES_PER_TASK), "split": split}
            s_ns += ns
        n = EPISODES_PER_TASK * len(rates)
        want_ns, want_n = AGGREGATES[split]
        if (s_ns, n) != (want_ns, want_n):
            raise SystemExit(f"{split}: recovered {s_ns}/{n} from the per-task rates but the page "
                             f"says {want_ns}/{want_n} -- transcription error")
        per_split[split] = _block(s_ns, n)
    if bad_names:
        raise SystemExit("task/split disagreement with TARGET_TASK_SPLIT:\n  " + "\n  ".join(bad_names))
    tot_ns = sum(v["n_success"] for v in per_split.values())
    tot_n = sum(v["n"] for v in per_split.values())
    return {
        "method": method,
        "eval_kind": "combine",
        # Governs the /stats baseline checkbox (isBaselineMethod matches any kind starting "baseline").
        "kind": "baseline_published",
        "synthetic": True,
        "source_url": SOURCE_URL,
        "note": ("PUBLISHED NUMBERS, NOT A LOCAL RUN. Per-task success rates transcribed from the "
                 "RoboCasa leaderboard and converted to counts (30 episodes/task). There is no "
                 "episode data, so /combine cannot open this method and no latency or termination "
                 "figures are reported."),
        "refined_assumed_equal_raw": True,
        "n_error": 0,
        "overall": _block(tot_ns, tot_n),
        "per_split": per_split,
        "per_task": per_task,
        "terminations": {},
        "bench": {},
        "src": "published-leaderboard",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method-name", default="abot-m0.6")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rec = build(a.method_name)
    o = rec["overall"]
    print(f"  {rec['method']}: {o['n_success']}/{o['n']} = {100*o['rate']:.1f}%  "
          f"(refined assumed equal)   {len(rec['per_task'])} tasks")
    for s, v in rec["per_split"].items():
        print(f"     {s:18s} {v['n_success']:4d}/{v['n']:<4d} {100*v['rate']:5.2f}%")
    if a.dry_run:
        print("  (dry run, nothing written)")
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p = a.out_dir / f"{rec['method']}.json"
    p.write_text(json.dumps(rec, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
