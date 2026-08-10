"""Compare a replication run against its baseline, episode by episode.

Used to validate an environment/repo migration: re-run a stratified subset of a finished sweep
and check the results still agree.

WHAT AGREEMENT TO EXPECT. The loop is NOT deterministic, so per-episode identity is the wrong
bar:

  * System2 decodes greedily (temperature 0, seed 0) -- reproducible for a given input.
  * System1 is a flow-matching sampler whose noise comes from an rng that ADVANCES on every
    infer() call (openpi policy.py: ``self._rng, sample_rng = jax.random.split(self._rng)``,
    seeded from ``jax.random.key(0)`` at server start). A server that has served a different
    number of prior calls is at a different rng state, so the same episode re-runs with
    different noise -> different actions -> possibly a different outcome.

Because the fleet shards episodes across 8 servers, an episode almost never sees the same rng
state twice. So we judge the migration on AGGREGATE agreement (does the success rate land within
sampling error?) and on the ABSENCE OF STRUCTURAL BREAKAGE (crashes, empty plans, no-subgoal
terminations, wildly different turn counts) -- not on per-episode flips, which are expected.

A useful reference point: for n=50 split 25/25, binomial noise alone is about +/-7 pp at 1 sigma
on each arm, so a few flips in each direction is the healthy outcome; a one-sided collapse
(e.g. every baseline success now failing) is the red flag.

Usage:
    python scripts/compare_replication.py \
        --baseline-index <combine>/<method>/index.json \
        --repro-index    <combine>/<repro-method>/index.json \
        --selection      repro50.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
from pathlib import Path


def _load(index_path: Path) -> dict[str, dict]:
    doc = json.loads(index_path.read_text())
    return {e["episode_id"]: e for e in doc.get("episodes", [])}


def _wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval -- valid for small n, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-index", required=True, type=Path)
    ap.add_argument("--repro-index", required=True, type=Path)
    ap.add_argument("--selection", type=Path, default=None,
                    help="repro50.json: the episodes that were selected, with baseline outcomes")
    args = ap.parse_args()

    base = _load(args.baseline_index)
    repro = _load(args.repro_index)
    sel = json.loads(args.selection.read_text()) if args.selection else None
    ids = [s["episode_id"] for s in sel] if sel else sorted(repro)
    split_of = {s["episode_id"]: s.get("split", "?") for s in (sel or [])}

    rows = []
    for eid in ids:
        b, r = base.get(eid), repro.get(eid)
        if r is None:
            rows.append((eid, b, None))
            continue
        rows.append((eid, b, r))

    ran = [(e, b, r) for e, b, r in rows if r is not None]
    print(f"episodes selected: {len(rows)}   completed in replication: {len(ran)}")
    if not ran:
        print("nothing to compare yet")
        return

    # ---- structural health: these SHOULD be zero regardless of stochasticity ----
    errs = [(e, r.get("error")) for e, _, r in ran if r.get("error")]
    nosub = [e for e, _, r in ran if r.get("termination") == "no_subgoal"]
    print("\n== STRUCTURAL (must be clean; independent of sampling noise) ==")
    print(f"  crashes / errors      : {len(errs)}")
    for e, msg in errs[:5]:
        print(f"      {e}: {str(msg)[:90]}")
    nosub_b = sum(1 for _, b, _ in ran if b and b.get("termination") == "no_subgoal")
    print(f"  no_subgoal terminations: {len(nosub)}   (baseline had {nosub_b} on these episodes)")

    # ---- aggregate agreement ----
    def rate(g, key):
        n = len(g)
        k = sum(1 for x in g if x[key])
        return k, n, (k / n if n else 0.0)

    bl = [{"s": bool(b and b.get("episode_success"))} for _, b, _ in ran]
    rp = [{"s": bool(r.get("episode_success"))} for _, _, r in ran]
    kb, nb, pb = rate(bl, "s")
    kr, nr, pr = rate(rp, "s")
    lo, hi = _wilson(kb, nb)
    print("\n== AGGREGATE ==")
    print(f"  baseline    : {kb}/{nb} = {100*pb:.1f}%")
    print(f"  replication : {kr}/{nr} = {100*pr:.1f}%")
    print(f"  baseline 95% Wilson CI: [{100*lo:.1f}%, {100*hi:.1f}%]  "
          f"-> replication {'INSIDE' if lo <= pr <= hi else 'OUTSIDE'}")

    # ---- per-episode flips ----
    same = sum(1 for (_, b, r) in ran
               if bool(b and b.get("episode_success")) == bool(r.get("episode_success")))
    s2f = [e for e, b, r in ran if b and b.get("episode_success") and not r.get("episode_success")]
    f2s = [e for e, b, r in ran if b and not b.get("episode_success") and r.get("episode_success")]
    print(f"\n== PER-EPISODE ==\n  agree: {same}/{len(ran)} ({100*same/len(ran):.0f}%)")
    print(f"  success -> failure: {len(s2f)}")
    for e in s2f[:8]:
        print(f"      {e}")
    print(f"  failure -> success: {len(f2s)}")
    for e in f2s[:8]:
        print(f"      {e}")
    print("  (flips in BOTH directions are expected: System1 noise differs per rng state;\n"
          "   a one-sided collapse would indicate real breakage)")

    # ---- by split and by baseline arm ----
    print("\n== BY SPLIT ==")
    for sp in ("atomic_seen", "composite_seen", "composite_unseen"):
        g = [(e, b, r) for e, b, r in ran if split_of.get(e) == sp]
        if not g:
            continue
        kb2 = sum(1 for _, b, _ in g if b and b.get("episode_success"))
        kr2 = sum(1 for _, _, r in g if r.get("episode_success"))
        print(f"  {sp:<18} baseline {kb2}/{len(g)}  ->  replication {kr2}/{len(g)}")
    print("\n== BY BASELINE ARM ==")
    for arm, want in (("was SUCCESS", True), ("was FAILURE", False)):
        g = [(e, b, r) for e, b, r in ran if b and bool(b.get("episode_success")) is want]
        if not g:
            continue
        kr2 = sum(1 for _, _, r in g if r.get("episode_success"))
        print(f"  {arm:<12} n={len(g):2d}  now success: {kr2}/{len(g)} ({100*kr2/len(g):.0f}%)")

    # ---- turn / timing sanity: distributions should overlap ----
    import statistics
    bt = [b["n_turns"] for _, b, _ in ran if b and isinstance(b.get("n_turns"), int)]
    rt = [r["n_turns"] for _, _, r in ran if isinstance(r.get("n_turns"), int)]
    print("\n== TURNS / TIME ==")
    print(f"  turns  baseline median={statistics.median(bt):.0f} mean={statistics.mean(bt):.1f}   "
          f"replication median={statistics.median(rt):.0f} mean={statistics.mean(rt):.1f}")
    bs = [b["seconds"] for _, b, _ in ran if b and isinstance(b.get("seconds"), (int, float))]
    rs = [r["seconds"] for _, _, r in ran if isinstance(r.get("seconds"), (int, float))]
    print(f"  s/ep   baseline median={statistics.median(bs):.0f}   "
          f"replication median={statistics.median(rs):.0f}")
    tb = collections.Counter(b.get("termination") for _, b, _ in ran if b)
    tr = collections.Counter(r.get("termination") for _, _, r in ran)
    print(f"  terminations baseline    : {dict(tb)}")
    print(f"  terminations replication : {dict(tr)}")

    # ---- verdict ----
    ok = (not errs) and (lo <= pr <= hi) and len(nosub) <= max(1, nosub_b)
    print("\n== VERDICT ==")
    print("  MIGRATION LOOKS HEALTHY" if ok else "  NEEDS A LOOK")
    if not ok:
        if errs:
            print("    - episodes crashed (structural, not noise)")
        if not (lo <= pr <= hi):
            print("    - success rate outside the baseline's 95% CI")
        if len(nosub) > max(1, nosub_b):
            print("    - more no_subgoal terminations than baseline (System2 output/plumbing)")


if __name__ == "__main__":
    main()
