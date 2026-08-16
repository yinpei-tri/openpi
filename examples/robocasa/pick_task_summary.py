"""Choose ONE episode per task to be that task's canonical recipe, and log the choice.

WHY. ``build_task_summaries.py`` writes one recipe per (task, episode). Five episodes of the same
task produce five different recipes, and the spread is large -- measured on the 16 composite_unseen
tasks, per-task truth recall ranges like 20-80% (PanTransfer) and 14-60% (WeighIngredients) inside a
single method. So the episode you happen to narrate matters about as much as which model narrates it,
and a library needs one chosen representative rather than an arbitrary first episode.

WHAT IT WRITES, one file per method directory (so a method is self-contained):

    <out-root>/<planner-tag>/TASK_SUMMARY.json
        {task: {chosen_episode, recipe, rule, candidates:[...], picks:{rule: episode}, ...}}
    <out-root>/<planner-tag>/TASK_SUMMARY.txt      the same thing, readable

SELECTION RULES. Every rule here is computed WITHOUT the ground-truth instruction, because a recipe
library that was selected using the true task description cannot then be used to evaluate anything --
the selection would carry the answer. Measured against an oracle that does peek (truth-noun recall,
16 tasks):

    rule                 qwen3vl   qwen35     uses
    ---------------- best-of-5 -----------------
    mean of 5 (none)      36.5%    32.5%      --
    consensus             41.2%    36.1%      recipes only: medoid by keyword Jaccard
    narration-coverage    46.4%    36.8%      how much of its OWN narrations the recipe retained
    longest               33.0%    28.7%      word count
    most-nouns            32.6%    34.0%      distinct content words
    ORACLE                50.9%    44.9%      the true instruction (NOT usable)

``narration-coverage`` is the default: it is the best or equal-best on both planners and captures
about 69% of the available gain on qwen3vl. It is still a recipe score -- it asks whether the recipe
kept the content it was built from, not whether the narration was any good.

GEMINI IS PICKED AS THE WORST OF THE FIVE (``--gemini-rule oracle-worst``), by request. Its units
carry no narrations (one whole-video call, no chunking), so the default rule cannot be computed for it.
Taking the weakest candidate is a deliberate conservative floor, and it is the one place using the
ground truth to select is sound: worst-case selection can only understate the method, never flatter
it, so it cannot manufacture a favourable comparison the way best-of-N can. ``--gemini-rule random``
restores the previous behaviour (seeded, recorded).

EVERY RULE'S PICK IS RECORDED, not just the winner, so a different rule can be adopted later without
re-running anything. ``oracle_episode`` is recorded too, clearly separated, for auditing how much the
truth-free rule left on the table -- never as the chosen value.

Usage:
    python examples/robocasa/pick_task_summary.py
    python examples/robocasa/pick_task_summary.py --rule consensus
    python examples/robocasa/pick_task_summary.py --methods qwen3vl-4b-full-ep3-17124
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import random
import re

DEFAULT_ROOT = Path(os.environ.get("TASK_SUMMARY_ROOT") or "~/data/unseen_task_summary").expanduser()

# Function words plus the verbs every recipe uses regardless of task ("grasp", "place", "retract"):
# they are noise for telling two recipes of the same task apart.
_STOP = frozenset(["a", "an", "the", "and", "then", "to", "on", "in", "of", "it", "its", "from", "with", "into", "onto", "place", "put", "pick", "up", "next", "first", "finally", "robot", "arm", "gripper", "task", "each", "other", "alongside", "their", "there", "this", "that", "for", "by", "at", "is", "are", "be", "as", "while", "so", "must", "can", "will", "one", "two", "both", "all", "them", "they", "need", "needed", "because", "ensure", "complete", "process", "repeat", "locate", "grasp", "reach", "carry", "transport", "lower", "release", "retract", "move", "take", "begin", "start", "target", "designated", "item", "items", "object", "objects", "area", "surface"])


def kw(s: str | None) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", (s or "").lower()) if w not in _STOP and len(w) > 3}


def load_units(method_dir: Path) -> dict[str, dict[int, dict]]:
    """task -> episode -> {recipe, narrations, truth}. Handles both layouts."""
    out: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(method_dir / "*__episode_*" / "memory.json"))):
        d = json.loads(Path(f).read_text())
        out[d["task"]][int(d["episode"])] = {
            "recipe": d.get("recipe") or "",
            "narrations": " ".join(d.get("narrations") or []),
            "truth": d.get("instruction_original") or "",
            "dir": Path(f).parent.name,
        }
    # gemini layout: recipe.json, no narrations
    for f in sorted(glob.glob(str(method_dir / "*__episode_*" / "recipe.json"))):
        if (Path(f).parent / "memory.json").exists():
            continue
        d = json.loads(Path(f).read_text())
        m = re.search(r"episode_(\d+)", f)
        out[d.get("task_name") or "?"][int(m.group(1))] = {
            "recipe": d.get("task_recipe") or "",
            "narrations": "",
            "truth": d.get("true_instruction") or "",
            "inferred_goal": d.get("task_goal"),
            "dir": Path(f).parent.name,
        }
    return out


def rule_picks(eps: list[int], u: dict[int, dict], rng: random.Random) -> dict[str, int]:
    """Every rule's chosen episode. Only `oracle` looks at the truth."""
    kws = {e: kw(u[e]["recipe"]) for e in eps}
    picks = {}
    picks["consensus"] = max(eps, key=lambda e: sum(
        len(kws[e] & kws[o]) / max(1, len(kws[e] | kws[o])) for o in eps if o != e))
    if any(u[e]["narrations"] for e in eps):
        picks["narration-coverage"] = max(eps, key=lambda e: (
            len(kws[e] & kw(u[e]["narrations"])) / max(1, len(kw(u[e]["narrations"])))))
    picks["longest"] = max(eps, key=lambda e: len(u[e]["recipe"].split()))
    picks["most-nouns"] = max(eps, key=lambda e: len(kws[e]))
    picks["random"] = rng.choice(eps)
    # The LEAST typical candidate -- the truth-free counterpart of `oracle-worst`.
    picks["consensus-worst"] = min(eps, key=lambda e: sum(
        len(kws[e] & kws[o]) / max(1, len(kws[e] | kws[o])) for o in eps if o != e))
    if any(u[e]["truth"] for e in eps):
        def _tr(e: int) -> float:
            """Recall of the true instruction's content words -- the only truth-using score here."""
            t = kw(u[e]["truth"])
            return len(t & kws[e]) / max(1, len(t))

        picks["oracle"] = max(eps, key=_tr)
        # DELIBERATE WORST CASE. Unlike `oracle`, selecting the worst with the ground truth is safe to
        # use downstream: it can only make a method look worse than it is, never better, so it cannot
        # manufacture a favourable result the way best-of-N selection can.
        picks["oracle-worst"] = min(eps, key=_tr)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--methods", default=None, help="comma list of planner dirs (default: all)")
    ap.add_argument("--rule", default="narration-coverage",
                    help="preferred rule; falls back to consensus when a method has no narrations, "
                         "and to random for a method with neither")
    ap.add_argument("--gemini-rule", default="oracle-worst",
                    help="rule for gemini_* dirs, which carry no narrations so the default rule "
                         "cannot be computed for them. 'oracle-worst' takes the WEAKEST of the five "
                         "on purpose (a conservative floor: worst-case selection can only understate "
                         "a method, never flatter it). 'random' or any other rule name also works.")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random rule, recorded in the file")
    a = ap.parse_args()

    dirs = ([a.root / m for m in a.methods.split(",")] if a.methods
            else sorted(p for p in a.root.iterdir() if p.is_dir()))
    for md in dirs:
        units = load_units(md)
        if not units:
            print(f"  {md.name}: no units, skipped")
            continue
        rng = random.Random(a.seed)
        is_gem = "gemini" in md.name.lower()
        rows, used = {}, collections.Counter()
        for task in sorted(units):
            u = units[task]
            eps = sorted(u)
            picks = rule_picks(eps, u, rng)
            want = a.gemini_rule if is_gem else a.rule
            rule = (want if want in picks
                    else "consensus" if "consensus" in picks else "random")
            chosen = picks[rule]
            used[rule] += 1
            rows[task] = {
                "chosen_episode": chosen,
                "rule": rule,
                "recipe": u[chosen]["recipe"],
                "dir": u[chosen]["dir"],
                "n_candidates": len(eps),
                # Every rule's pick, so switching rules later needs no re-run.
                "picks": picks,
                # Recorded for AUDIT ONLY -- it used the true instruction and must never be the
                # chosen value, or a library selected with the answer would be used to score it.
                "oracle_episode": picks.get("oracle"),
                "candidates": [{"episode": e, "recipe": u[e]["recipe"],
                                "true_instruction": u[e]["truth"]} for e in eps],
            }
        doc = {"planner_tag": md.name, "rule_default": a.rule, "seed": a.seed,
               "rules_used": dict(used), "n_tasks": len(rows),
               "note": ("chosen_episode is selected WITHOUT the ground-truth instruction; "
                        "oracle_episode is audit-only"),
               "tasks": rows}
        (md / "TASK_SUMMARY.json").write_text(json.dumps(doc, indent=1))
        lines = [f"# {md.name}   rule={a.rule}  ({len(rows)} tasks)", ""]
        for t, r in rows.items():
            agree = "" if r["oracle_episode"] is None else (
                "  [= oracle]" if r["oracle_episode"] == r["chosen_episode"]
                else f"  [oracle would pick ep{r['oracle_episode']}]")
            lines += [f"## {t}   ep{r['chosen_episode']}  (rule: {r['rule']}){agree}",
                      f"   {r['recipe']}", ""]
        (md / "TASK_SUMMARY.txt").write_text("\n".join(lines))
        agree = sum(1 for r in rows.values()
                    if r["oracle_episode"] is not None and r["oracle_episode"] == r["chosen_episode"])
        print(f"  {md.name:30s} {len(rows)} tasks  rules={dict(used)}  "
              f"agrees with oracle on {agree}/{len(rows)}")
    print(f"wrote TASK_SUMMARY.json + .txt into each method dir under {a.root}")


if __name__ == "__main__":
    main()
