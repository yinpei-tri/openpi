"""Report what the hardcoded per-task rules (examples/robocasa/sys2_rules.py) actually did.

These rules override a learned planner, so a result from a rules-on run is only interpretable
next to an audit of the overrides: which fired, how often, and -- when a baseline is given --
whether the episodes they touched actually got better.

Reads ``episode.json`` (``rule_interventions`` / ``rule_tier``), written by combined_eval for every
rule configuration. Never infers an intervention from a name or a diff.

    python scripts/report_rule_interventions.py --method debug-...-improve
    python scripts/report_rule_interventions.py --method debug-...-improve \
        --baseline s1-progact270k_s2-qwen35-4b-full-ep3-11416
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

_REPO_ROOT = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[1].parent)
_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO_ROOT / "data", Path.home() / "data") if c.is_dir()), _REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results").expanduser()


def _episodes(method_dir: Path) -> dict[str, dict]:
    """episode_id -> episode.json, for COMPLETE episodes only (termination set, no error)."""
    out: dict[str, dict] = {}
    if not method_dir.is_dir():
        return out
    for d in method_dir.iterdir():
        if not d.is_dir() or d.name == "index_parts":
            continue
        f = d / "episode.json"
        if not f.exists():
            continue
        try:
            doc = json.loads(f.read_text())
        except Exception:  # torn write
            continue
        if doc.get("termination") and not doc.get("error"):
            out[doc.get("episode_id") or d.name] = doc
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--baseline", default=None,
                    help="method to compare against, episode by episode")
    ap.add_argument("--combine-root", type=Path, default=RESULTS_DIR / "combine")
    a = ap.parse_args()

    eps = _episodes(a.combine_root / a.method)
    if not eps:
        raise SystemExit(f"no complete episodes under {a.combine_root / a.method}")

    on = sum(1 for d in eps.values()
             if d.get("rule_tier") not in (None, "none") or d.get("task_rules"))
    tiers = collections.Counter(d.get("rule_tier") or
                                ("legacy-task" if d.get("task_rules") else "legacy-none")
                                for d in eps.values())
    print(f"method   : {a.method}")
    print(f"episodes : {len(eps)} complete   ({on} ran with at least one rule tier)")
    print("tiers    : " + ", ".join(f"{k}={v}" for k, v in sorted(tiers.items())))
    if on == 0:
        print("\nWARNING: no episode recorded an active rule tier.")

    # ---- what fired ----
    by_rule = collections.Counter()
    by_kind = collections.Counter()
    by_task = collections.defaultdict(collections.Counter)
    eps_touched = set()
    for eid, d in eps.items():
        task = d.get("task_name") or "?"
        for t in d.get("rule_interventions") or []:
            for iv in t.get("interventions", []):
                # Skip NO-OP records: a rule that "overrode" est 75 with 75 changed nothing, and
                # counting it would inflate the intervention total. The rules log unconditionally
                # (cheap and honest at the source); the accounting is done here.
                if iv.get("before") == iv.get("after"):
                    by_kind[iv["kind"] + "(no-op)"] += 1
                    continue
                by_rule[iv["rule"]] += 1
                by_kind[iv["kind"]] += 1
                by_task[task][iv["rule"]] += 1
                eps_touched.add(eid)
    print(f"\n== INTERVENTIONS ==  {sum(by_rule.values())} across {len(eps_touched)} episode(s)")
    for r, n in by_rule.most_common():
        print(f"  {r:22s} {n:4d}")
    print("  by kind: " + ", ".join(f"{k}={v}" for k, v in by_kind.most_common()))
    print("\n  by task:")
    for task in sorted(by_task):
        inner = ", ".join(f"{r}x{n}" for r, n in by_task[task].most_common())
        print(f"    {task:28s} {inner}")

    # Tasks that have rules but where nothing fired are worth surfacing: usually the rule's
    # trigger never appeared, which is information, not success.
    silent = sorted({d.get("task_name") for d in eps.values()} - set(by_task))
    if silent:
        print(f"\n  no interventions on: {', '.join(silent)}")

    # ---- outcome, per episode, vs baseline ----
    if not a.baseline:
        s = sum(1 for d in eps.values() if d.get("episode_success"))
        print(f"\n== OUTCOME ==\n  {s}/{len(eps)} success ({100*s/len(eps):.1f}%)")
        print("  (pass --baseline to compare episode by episode)")
        return

    base = _episodes(a.combine_root / a.baseline)
    shared = [e for e in eps if e in base]
    print(f"\n== VS BASELINE {a.baseline} ==")
    print(f"  episodes in both: {len(shared)}")
    if not shared:
        return

    fixed, broke, same_ok, same_bad = [], [], [], []
    for eid in sorted(shared):
        b = bool(base[eid].get("episode_success"))
        r = bool(eps[eid].get("episode_success"))
        (fixed if (r and not b) else broke if (b and not r)
         else same_ok if b else same_bad).append(eid)
    kb, kr = sum(1 for e in shared if base[e].get("episode_success")), \
             sum(1 for e in shared if eps[e].get("episode_success"))
    print(f"  baseline : {kb}/{len(shared)} = {100*kb/len(shared):.1f}%")
    print(f"  with rules: {kr}/{len(shared)} = {100*kr/len(shared):.1f}%   "
          f"({'+' if kr>=kb else ''}{kr-kb})")
    print(f"\n  FIXED  (fail -> success): {len(fixed)}")
    for e in fixed:
        print(f"      {e}  turns {base[e].get('n_turns')} -> {eps[e].get('n_turns')}")
    print(f"  BROKE  (success -> fail): {len(broke)}")
    for e in broke:
        print(f"      {e}  {base[e].get('termination')} -> {eps[e].get('termination')}")
    print(f"  unchanged: {len(same_ok)} still pass, {len(same_bad)} still fail")

    # Per-task breakdown -- an aggregate can hide one rule helping while another hurts.
    print("\n  per task:")
    pt = collections.defaultdict(lambda: [0, 0, 0])
    for eid in shared:
        t = eps[eid].get("task_name") or "?"
        pt[t][0] += 1
        pt[t][1] += bool(base[eid].get("episode_success"))
        pt[t][2] += bool(eps[eid].get("episode_success"))
    for t in sorted(pt):
        n, b, r = pt[t]
        print(f"    {t:28s} {b}/{n} -> {r}/{n}   ({'+' if r>=b else ''}{r-b})")

    print("\n  NOTE: these subsets are small; a +/-1 change is noise. System1's sampler is also\n"
          "  nondeterministic (see scripts/compare_replication.py), so an unchanged episode can\n"
          "  still differ in turn count, and a flip is not by itself evidence a rule worked.")


if __name__ == "__main__":
    main()
