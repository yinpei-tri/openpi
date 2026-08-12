#!/usr/bin/env python3
"""Score a rule experiment against reference runs on the SAME episodes.

One n=20 run cannot resolve a 1-2 episode effect, and the System1 sampler is stateful (policy.py
splits a per-call key off jax.random.key(0), so flow-matching noise differs run to run and
replication is never bit-exact -- measured ~+-3 episodes at n=20). So an experiment is only readable
as a PAIRED comparison on identical episode ids, run at least twice.

    uv run scripts/compare_rule_experiment.py --task ScrubCuttingBoard \\
        --exp debug-scrub-c debug-scrub-d

Reference methods default to the 1000-episode baseline and the verified-rules run. Output: the 2x2
agreement table the experiments are graduated on, plus turns and stop reasons -- a rule that trades
successes for turns on a turn-bound task shows up there and nowhere else.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics

BASELINE = "s1-progact270k_s2-qwen35-4b-full-ep3-11416"
VERIFIED = "s1-progact270k_s2-qwen35-4b-full-ep3-11416-estbump"


def load(root: str, method: str, task: str | None) -> dict[str, dict]:
    """episode_id -> record, for one results method dir."""
    out = {}
    for f in glob.glob(os.path.join(root, method, "*", "episode.json")):
        name = os.path.basename(os.path.dirname(f))
        if task and not name.startswith(task + "__"):
            continue
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001 - a hard-stopped run leaves truncated json; skip it
            continue
        # NB: the success field is "episode_success". A "success" lookup silently reads None for
        # every episode and reports 0/20.
        out[name] = {
            "ok": bool(d.get("episode_success")),
            "turns": d.get("n_turns"),
            "stop": d.get("max_turns_reason") or d.get("termination") or "?",
            "task": name.split("__")[0],
        }
    return out


def rate(recs: dict[str, dict], ids=None) -> str:
    sel = [r for k, r in recs.items() if ids is None or k in ids]
    n = len(sel)
    return f"{sum(r['ok'] for r in sel):>2d}/{n:<2d}" + (f" {100*sum(r['ok'] for r in sel)/n:5.1f}%" if n else "      ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", help="restrict to one task (recommended: experiments are task-gated)")
    p.add_argument("--exp", nargs="+", required=True, help="experiment method dir(s), e.g. two repeats")
    p.add_argument("--baseline", default=BASELINE)
    p.add_argument("--verified", default=VERIFIED)
    p.add_argument("--root", default=os.path.expanduser(
        os.environ.get("SYS1_RESULTS_DIR", "/shared/data/sys1_eval_results") + "/combine"))
    a = p.parse_args()

    runs = {m: load(a.root, m, a.task) for m in [a.baseline, a.verified, *a.exp]}
    # Compare only where every run has a result: an experiment that lost episodes to a crash would
    # otherwise be scored on a different denominator than its references.
    common = set.intersection(*(set(v) for v in runs.values())) if all(runs.values()) else set()
    print(f"task={a.task or 'ALL'}  episodes present in every run: {len(common)}")
    if not common:
        print("  no shared episodes -- check the run label(s) and that the run finished")
        return

    print(f"\n  {'tag':<9s} {'run':<44s} {'shared':<11s} {'all':<11s} turns")
    for m, recs in runs.items():
        t = [recs[k]["turns"] for k in common if isinstance(recs[k]["turns"], (int, float))]
        tag = "baseline" if m == a.baseline else ("verified" if m == a.verified else "EXP")
        turns = f"{statistics.mean(t):4.1f}" if t else "   -"
        print(f"  {tag:<9s} {m[:43]:<44s} {rate(recs, common):<11s} {rate(recs):<11s} {turns}")

    for exp in a.exp:
        for ref_name, ref in (("baseline", a.baseline), ("verified", a.verified)):
            b, e = runs[ref], runs[exp]
            both = [k for k in common if b[k]["ok"] and e[k]["ok"]]
            only_ref = [k for k in common if b[k]["ok"] and not e[k]["ok"]]
            only_exp = [k for k in common if not b[k]["ok"] and e[k]["ok"]]
            neither = [k for k in common if not b[k]["ok"] and not e[k]["ok"]]
            print(f"\n  {exp}  vs  {ref_name}")
            print(f"    both succeed      {len(both):>3d}")
            print(f"    {ref_name+' only':<17s} {len(only_ref):>3d}  {[k.split('__')[-1] for k in sorted(only_ref)][:6]}")
            print(f"    exp only          {len(only_exp):>3d}  {[k.split('__')[-1] for k in sorted(only_exp)][:6]}")
            print(f"    neither           {len(neither):>3d}")
            print(f"    net               {len(only_exp)-len(only_ref):+d}")

    print("\n  stop reasons (experiment runs)")
    for exp in a.exp:
        c = collections.Counter(runs[exp][k]["stop"] for k in common if not runs[exp][k]["ok"])
        print(f"    {exp}: {dict(c)}")


if __name__ == "__main__":
    main()
