"""Assert ``sys2_memory``'s prompts are BYTE-IDENTICAL to the System2 training shards.

The memory modes are only useful if the model sees the prompts it was supervised on: a reworded
system prompt or a dropped pair of quotes turns a trained capability into an off-distribution guess,
and nothing else in the pipeline would complain. This checks the real thing -- it reads records out
of the training shards, re-renders each user turn from the record's OWN fields, and compares.

Not a pytest (the shards live outside the repo and the robocasa env is a separate interpreter). Run
it after touching any prompt string in ``sys2_memory.py``:

    "$ROBOCASA_PY" examples/robocasa/sys2_memory_prompt_check.py

Exits non-zero on any mismatch, printing both strings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sys2_memory as MEM  # noqa: E402

# The trained target dataset. Overridable for a re-built shard set.
DEFAULT_SHARD_DIR = Path("/shared/data/sys2_data/system2_target_0804/shards")
# sub_mode strings, exactly as the producer writes them -- note the U+00B7 MIDDLE DOT separator.
NARRATE, RECIPE, WARM_PLAN = "summary_v2", "summary_v2 · recipe", "plan_exec_v2 · with memory"


def collect(shard_dir: Path, per_mode: int = 6) -> dict[str, list[dict]]:
    """Pull up to ``per_mode`` records of each memory sub_mode out of the shards."""
    want: dict[str, list[dict]] = {NARRATE: [], RECIPE: [], WARM_PLAN: []}
    for shard in sorted(shard_dir.glob("shard_*.tar")):
        with tarfile.open(shard) as t:
            for m in t:
                if not m.name.endswith(".json"):
                    continue
                d = json.loads(t.extractfile(m).read())
                sm = (d.get("meta") or {}).get("sub_mode")
                if sm in want and len(want[sm]) < per_mode:
                    want[sm].append(d)
                if all(len(v) >= per_mode for v in want.values()):
                    return want
    return want


def user_text(rec: dict) -> str:
    """The user turn as ONE string, with media parts back as <image>/<video> placeholders."""
    c = rec["messages"][1]["content"]
    if isinstance(c, str):
        return c
    return "".join(
        p.get("text", "<image>" if p.get("type") == "image" else "<video>") for p in c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    a = ap.parse_args()
    if not a.shard_dir.is_dir():
        print(f"FATAL: no shard dir {a.shard_dir}", file=sys.stderr)
        return 2

    recs = collect(a.shard_dir)
    missing = [k for k, v in recs.items() if not v]
    if missing:
        print(f"FATAL: no records found for {missing} in {a.shard_dir}", file=sys.stderr)
        return 2

    fails = 0

    def cmp(label: str, got: str, mine: str) -> None:
        nonlocal fails
        if got != mine:
            fails += 1
            print(f"\nMISMATCH {label}\n--- shard ---\n{got!r}\n--- ours ---\n{mine!r}")

    # ---- system prompts ----
    for sm, mine in ((NARRATE, MEM.SYS_NARRATE), (RECIPE, MEM.SYS_RECIPE),
                     (WARM_PLAN, MEM.SYS_PLAN_MEM)):
        before = fails
        for d in recs[sm]:
            cmp(f"system/{sm}", d["messages"][0]["content"], mine)
            if fails > before:
                break
        if fails == before:
            print(f"OK system  {sm:32s} ({len(recs[sm])} records)")

    # ---- user turns: re-render from each record's own fields ----
    before = fails
    for d in recs[NARRATE]:
        u = user_text(d)
        goal = re.search(r"The goal of the task is: (.*?)\n\n", u, re.S).group(1)
        if "This is the first clip" in u:
            prior: list[str] = []
        else:
            block = re.search(r"So far:\n(.*?)\n\nHere is the", u, re.S).group(1)
            prior = [re.sub(r"^\d+\.\s*", "", ln) for ln in block.split("\n")]
        cmp("user/narrate", u, MEM.user_narrate(goal, prior, len(prior)))
        if fails > before:
            break
    if fails == before:
        print(f"OK user    {NARRATE + ' narrate':32s} ({len(recs[NARRATE])} records)")

    before = fails
    for d in recs[RECIPE]:
        u = user_text(d)
        goal = re.search(r"The goal of the task was: (.*?)\n\n", u, re.S).group(1)
        block = re.search(r"narrated across the run:\n(.*?)\n\nThat's the whole task", u, re.S).group(1)
        narr = [re.sub(r"^\d+\.\s*", "", ln) for ln in block.split("\n")]
        cmp("user/recipe", u, MEM.user_recipe(goal, narr))
        if fails > before:
            break
    if fails == before:
        print(f"OK user    {RECIPE:32s} ({len(recs[RECIPE])} records)")

    before = fails
    for d in recs[WARM_PLAN]:
        u = user_text(d)
        goal = re.search(r"The goal is: (.*?)\n\n", u, re.S).group(1)
        rec = re.search(r'to help you plan:\n"(.*?)"\n\nHere is the scene', u, re.S).group(1)
        cmp("user/warm-plan", u, MEM.user_plan_mem(goal, rec))
        if fails > before:
            break
    if fails == before:
        print(f"OK user    {WARM_PLAN:32s} ({len(recs[WARM_PLAN])} records)")

    # ---- media contract: narrate=1 video, recipe=NONE, warm plan=1 image ----
    for sm, want_v, want_i in ((NARRATE, 1, 0), (RECIPE, 0, 0), (WARM_PLAN, 0, 1)):
        d = recs[sm][0]
        c = d["messages"][1]["content"]
        if isinstance(c, list):
            nv = sum(1 for p in c if p.get("type") == "video")
            ni = sum(1 for p in c if p.get("type") == "image")
        else:
            nv, ni = c.count("<video>"), c.count("<image>")
        ok = (nv, ni) == (want_v, want_i)
        fails += 0 if ok else 1
        print(f"{'OK' if ok else 'MISMATCH'} media   {sm:32s} video={nv} image={ni} "
              f"(want video={want_v} image={want_i})")

    # ---- chunk windows: fixed 80-frame windows, verified against the shards' own clip spans ----
    by_ep: dict[str, dict[str, tuple]] = {}
    for shard in sorted(a.shard_dir.glob("shard_*.tar"))[:2]:
        with tarfile.open(shard) as t:
            for m in t:
                if not m.name.endswith(".json"):
                    continue
                d = json.loads(t.extractfile(m).read())
                mt = d.get("meta") or {}
                if mt.get("sub_mode") != NARRATE:
                    continue
                c = mt.get("clip") or {}
                by_ep.setdefault(mt["episode_id"], {})[mt["file"]] = (c.get("start"), c.get("end"))
    checked = bad = 0
    for ep, files in by_ep.items():
        ws = [files[k] for k in sorted(files)]
        if len(ws) < 2:
            continue
        checked += 1
        # The last window's end is num_frames-1, which is all chunk_windows needs.
        mine = set(MEM.chunk_windows(ws[-1][1] + 1))
        stray = [w for w in ws if tuple(w) not in mine]
        if stray:
            bad += 1
            fails += 1
            print(f"MISMATCH windows {ep}: shard={ws} ours={sorted(mine)}")
    print(f"{'OK' if not bad else 'MISMATCH'} windows  {checked} episodes checked, {bad} mismatched")

    print(f"\n{'ALL PROMPTS MATCH THE TRAINING SHARDS' if not fails else f'{fails} MISMATCH(ES)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
