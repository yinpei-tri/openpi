"""Generate System2 MEMORY narrations + task recipes for a set of tasks, AHEAD of any rollout.

The narrate -> recipe pass in ``sys2_memory.py`` reads the DEMO video straight out of the LeRobot
dataset, so it never needed System1 -- it was only ever *called* from the memory eval. This script
calls it on its own, one episode per task, so a recipe library can be built once and reused:

    NARRATE chunk i : goal + narrations 1..i-1 + tiled 4s clip -> <narration>  (one sentence)
    RECIPE          : goal + ALL narrations, no media          -> <summary>

Chunking, clip encoding and every prompt string come from sys2_memory (byte-identical to the
training producer); nothing about them is re-chosen here. The warm-plan call is NOT made -- that
belongs to the eval that consumes the recipe.

ONE DIRECTORY PER PLANNER. The narration is a property of the VLM that wrote it, so qwen35 and
qwen3vl recipes must never share a directory: the output path carries the S2 checkpoint tag, derived
from --s2-dir (or given with --planner-tag). Re-running a planner over the same task overwrites only
that planner's copy.

    <out-root>/<planner-tag>/<Task>__episode_XXXXXX/
        chunk00/{clip_full.mp4, clip.mp4, narration.json}   <- input media + prompt + raw response
        ...
        recipe.json                                        <- the text-only aggregation call
        memory.json                                        <- recipe + narrations + provenance
    <out-root>/<planner-tag>/INDEX.json                    <- one row per task, for a quick read

Usage (needs the robocasa env for LeRobot video decoding, and a System2 vLLM server):
    MUJOCO_GL=egl ~/micromamba/envs/robocasa/bin/python examples/robocasa/build_task_summaries.py \
        --split composite_unseen --limit 3 \
        --s2-port 8100 --s2-dir "$CKPT_DIR/system2-full-0804-qwen35-.../checkpoint-11416"

    # all 16 composite-unseen tasks, one episode each
    ... --split composite_unseen
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from combined_eval import ROBOCASA_DATASET
from combined_eval import TARGET_EVAL_EPISODES
from combined_eval import TARGET_TASK_SPLIT
from combined_eval import _load_goal_map
import robocasa.utils.lerobot_utils as LU
import sys2_client as S2C
import sys2_memory as MEM

DEFAULT_OUT = Path(os.environ.get("TASK_SUMMARY_ROOT") or "~/data/unseen_task_summary").expanduser()


def planner_tag(s2_dir: str | None, explicit: str | None = None) -> str:
    """Short, stable name for the VLM that wrote these narrations.

    Mirrors the s2 half of combined_eval._short_method_name so a recipe directory and an eval method
    name refer to the same checkpoint in the same words: qwen35-4b-full-ep3-11416.
    """
    if explicit:
        return explicit
    if not s2_dir:
        return "unknown-s2"
    p = Path(s2_dir)
    step = p.name.split("-")[-1] if p.name.startswith("checkpoint") else ""
    run = p.parent.name if step else p.name
    m = re.search(r"(qwen[\w.]*?-\d+b)", run, re.IGNORECASE)
    fam = m.group(1).lower().replace(".", "") if m else run[:24]
    bits = [fam]
    for kw in ("full", "lora"):
        if f"-{kw}" in run:
            bits.append(kw)
            break
    m = re.search(r"-(ep\d+)", run)
    if m:
        bits.append(m.group(1))
    if step:
        bits.append(step)
    return "-".join(bits)


def _goal_source_line(m: dict) -> str:
    """Where the goal in these prompts came from -- the one thing a reader must not have to guess."""
    return (f"--goal-json {m.get('goal_json')}" if m.get("goal_override")
            else "dataset ep_meta['lang']")


def dump_raw_text(dest: Path) -> int:
    """Write the prompts and the RAW model responses as plain .txt beside the JSON records.

    All of this already lives in narration.json / recipe.json, but reviewing a narration pass means
    reading a prompt next to its video, and unpacking JSON in a terminal to do that is friction that
    stops people looking. These files are a VIEW, never a source: rewritten from the JSON on every
    run, with the JSON authoritative.

        chunkNN/prompt_system.txt / prompt_user.txt / response_raw.txt
        chunkNN/clip_full.mp4     the raw visual input (already written by build_memory)
        recipe_prompt_system.txt / recipe_prompt_user.txt / recipe_response_raw.txt
        NARRATION.txt             goal, every chunk's window + sentence, then the recipe
    """
    n = 0
    for j in sorted(dest.glob("chunk*/narration.json")):
        d = json.loads(j.read_text())
        for key, name in (("s2_system_prompt", "prompt_system.txt"),
                          ("s2_user_prompt", "prompt_user.txt"),
                          ("s2_response_raw", "response_raw.txt")):
            (j.parent / name).write_text((d.get(key) or "") + "\n")
            n += 1
    rj = dest / "recipe.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        for key, name in (("s2_system_prompt", "recipe_prompt_system.txt"),
                          ("s2_user_prompt", "recipe_prompt_user.txt"),
                          ("s2_response_raw", "recipe_response_raw.txt")):
            (dest / name).write_text((d.get(key) or "") + "\n")
            n += 1
    mj = dest / "memory.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        lines = [f"task        : {m.get('task')}  episode {m.get('episode')}",
                 f"planner     : {m.get('planner_tag')}",
                 f"goal given  : {m.get('instruction')}",
                 f"goal source : {_goal_source_line(m)}",
                 f"dataset goal: {m.get('instruction_original')}",
                 f"source      : {m.get('n_source_frames')} frames, "
                 f"{m.get('n_chunks_narrated')} chunks of {m.get('chunk_source_frames')} "
                 f"({m.get('n_chunks_skipped')} skipped)", "",
                 "NARRATIONS (one sentence per 4s chunk)"]
        for c in m.get("chunks") or []:
            w = c.get("window") or [0, 0]
            lines.append(f"  {c.get('chunk'):2d} [{w[0]:5d}-{w[1]:5d}]  {c.get('narration')}")
        lines += ["", "RECIPE (text-only aggregation of the narrations above)", "",
                  m.get("recipe") or "", ""]
        (dest / "NARRATION.txt").write_text("\n".join(lines))
        n += 1
    return n


def lerobot_dir_for(task: str, root: Path) -> Path | None:
    hits = sorted(root.glob(f"v1.0/target/*/{task}/*/lerobot"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="composite_unseen",
                    help="task split to cover (TARGET_TASK_SPLIT), or 'all'")
    ap.add_argument("--tasks", default=None, help="explicit comma list; overrides --split")
    ap.add_argument("--episode", type=int, default=None,
                    help="ONE episode index per task (overrides --episodes)")
    ap.add_argument("--episodes", default="first",
                    help="which episodes per task: 'first' (the task's first id in "
                         "TARGET_EVAL_EPISODES), 'all' (every id in it -- 30 per task, 480 units for "
                         "composite_unseen), or an explicit spec like '0-9' / '0,5,10'. A recipe built "
                         "from an episode the eval also runs is ORACLE for that episode; say so when "
                         "reporting.")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process only units I, I+N, I+2N... of the (task, episode) list -- one shard "
                         "per System2 server. Each shard MUST also get its own --index-name.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks (a quick check)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="truncate the narration pass (debugging; it changes the recipe and is "
                         "recorded in memory.json)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lerobot-root", type=Path, default=ROBOCASA_DATASET)
    ap.add_argument("--s2-host", default="127.0.0.1")
    ap.add_argument("--s2-port", type=int, default=8100)
    ap.add_argument("--s2-model", default="system2-full")
    ap.add_argument("--s2-dir", default=None, help="S2 checkpoint dir, for the planner tag + record")
    ap.add_argument("--planner-tag", default=None, help="override the derived planner tag")
    ap.add_argument("--goal-json", default=None,
                    help="JSON map {task: goal} replacing the dataset's ep_meta['lang'] in the "
                         "narration and recipe prompts -- the same file the eval takes, so a recipe "
                         "is built from the goal the planner will actually be given. Recorded in "
                         "memory.json (goal_override / instruction_original) and in the reuse guard.")
    ap.add_argument("--index-name", default="INDEX.json",
                    help="index filename. Parallel shards MUST each use their own (they all write "
                         "into one planner directory, and a shared name means last-writer-wins and "
                         "the other shards' rows are lost); then --reindex rebuilds INDEX.json.")
    ap.add_argument("--reindex", action="store_true",
                    help="do not call the model: rebuild INDEX.json from every memory.json already "
                         "in the planner directory. Use after parallel shards finish.")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo tasks that already have a memory.json (default: skip them)")
    a = ap.parse_args()

    if a.tasks:
        tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    elif a.split == "all":
        tasks = sorted(TARGET_TASK_SPLIT)
    else:
        tasks = sorted(t for t, s in TARGET_TASK_SPLIT.items() if s == a.split)
    if a.limit:
        tasks = tasks[:a.limit]
    if not tasks:
        raise SystemExit(f"no tasks for split {a.split!r}")

    gmap = _load_goal_map(a.goal_json) if a.goal_json else {}
    tag = planner_tag(a.s2_dir, a.planner_tag)
    out_root = a.out_root.expanduser() / tag
    out_root.mkdir(parents=True, exist_ok=True)
    s2 = S2C.Sys2Client(host=a.s2_host, port=a.s2_port, model=a.s2_model)
    print(f"planner tag : {tag}")
    print(f"out         : {out_root}")
    print(f"tasks       : {len(tasks)} ({a.split if not a.tasks else 'explicit'})")

    if a.reindex:
        rows = []
        for mj in sorted(out_root.glob("*__episode_*/memory.json")):
            m = json.loads(mj.read_text())
            rows.append({"task": m.get("task"), "episode": m.get("episode"), "dir": mj.parent.name,
                         "instruction": m.get("instruction"), "goal_override": m.get("goal_override"),
                         "recipe": m.get("recipe"), "n_chunks": m.get("n_chunks_narrated"),
                         "n_frames": m.get("n_source_frames"), "seconds": m.get("seconds")})
        idx = out_root / a.index_name
        idx.write_text(json.dumps({"planner_tag": tag, "split": a.split,
                                   "goal_json": str(a.goal_json) if a.goal_json else None,
                                   "n_tasks": len(rows), "tasks": rows}, indent=1, default=str))
        print(f"reindexed {len(rows)} recipes -> {idx}")
        return

    def episodes_for(task: str) -> list[int]:
        avail = TARGET_EVAL_EPISODES.get(task) or [0]
        if a.episode is not None:
            return [a.episode]
        if a.episodes == "first":
            return [avail[0]]
        if a.episodes.startswith("first:"):
            # The FIRST N ids the eval uses for this task -- not range(N): some tasks' manifest ids
            # are sparse source indices, so "the first 5 episodes" and "episodes 0-4" are not the
            # same set and only the former lines up with the eval.
            return list(avail[:int(a.episodes.split(":", 1)[1])])
        if a.episodes == "all":
            return list(avail)
        out: list[int] = []
        for part in a.episodes.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                out += list(range(int(lo), int(hi) + 1))
            elif part:
                out.append(int(part))
        return out

    units = [(t, e) for t in tasks for e in episodes_for(t)]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        units = units[si::sn]
        print(f"shard       : {si}/{sn} -> {len(units)} units")
    print(f"units       : {len(units)} (task, episode) pairs")

    rows = []
    for i, (task, ep) in enumerate(units, 1):
        ld = lerobot_dir_for(task, a.lerobot_root)
        if ld is None:
            print(f"[{i}/{len(units)}] {task}: NO lerobot dir under {a.lerobot_root} -- skipped")
            rows.append({"task": task, "error": "no lerobot dir"})
            continue
        dest = out_root / f"{task}__episode_{ep:06d}"
        # A recipe is a function of the GOAL as much as of the video, so a cached one built from a
        # different goal source is not reusable -- that would quietly mix short-goal and full-goal
        # recipes in one library (the same defect the eval's run guard was fixed for).
        cached = dest / "memory.json"
        if cached.exists() and not a.overwrite:
            try:
                prev = json.loads(cached.read_text())
            except Exception:
                prev = {}
            if bool(prev.get("goal_override")) != bool(a.goal_json):
                print(f"[{i}/{len(units)}] {task}: existing recipe used "
                      f"{'--goal-json' if prev.get('goal_override') else 'the dataset goal'}, "
                      f"this run uses {'--goal-json' if a.goal_json else 'the dataset goal'} -- "
                      "re-run with --overwrite to replace it. SKIPPED")
                rows.append({"task": task, "episode": ep, "error": "goal source mismatch"})
                continue
        if (dest / "memory.json").exists() and not a.overwrite:
            print(f"[{i}/{len(units)}] {task}: already done ({dest.name}) -- skipped")
            rows.append(json.loads((dest / "memory.json").read_text()) | {"task": task, "episode": ep,
                                                                          "reused": True})
            continue
        try:
            meta = LU.get_episode_meta(ld, ep)
            instruction_original = (meta.get("lang") or "").strip()
            if not instruction_original:
                raise ValueError(f"episode {ep} has no ep_meta['lang']")
            goal_override = gmap.get(task) if gmap else None
            if gmap and goal_override is None:
                raise ValueError(f"--goal-json has no entry for {task}")
            instruction = goal_override or instruction_original
            t0 = time.perf_counter()
            mem = MEM.build_memory(s2, lerobot_dir=ld, ep_index=ep, instruction=instruction,
                                   out_dir=dest, max_chunks=a.max_chunks)
            secs = round(time.perf_counter() - t0, 1)
            # Provenance the recipe cannot be read without: which planner, which episode, which goal.
            mem_doc = json.loads((dest / "memory.json").read_text())
            mem_doc.update({"task": task, "episode": ep, "lerobot_dir": str(ld),
                            "instruction": instruction,
                            "instruction_original": instruction_original,
                            "goal_override": goal_override,
                            "goal_json": str(a.goal_json) if a.goal_json else None,
                            "planner_tag": tag,
                            "s2": {"dir": a.s2_dir, "model": a.s2_model, "port": a.s2_port},
                            "seconds": secs})
            (dest / "memory.json").write_text(json.dumps(mem_doc, indent=1, default=str))
            n_txt = dump_raw_text(dest)
            print(f"[{i}/{len(units)}] {task} ep{ep}: {mem['n_chunks_narrated']} chunks "
                  f"({mem['n_source_frames']} frames), {secs}s")
            print(f"      recipe: {mem['recipe'][:150]}")
            print(f"      raw prompts/responses: {n_txt} .txt files + {mem['n_chunks_narrated']} clips")
            rows.append({"task": task, "episode": ep, "dir": dest.name,
                         "instruction": instruction, "goal_override": goal_override,
                         "recipe": mem["recipe"],
                         "n_chunks": mem["n_chunks_narrated"], "n_frames": mem["n_source_frames"],
                         "seconds": secs})
        except Exception as e:  # one bad task must not lose the rest of the sweep
            print(f"[{i}/{len(units)}] {task} ep{ep}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({"task": task, "episode": ep, "error": f"{type(e).__name__}: {e}"})

    idx = out_root / "INDEX.json"
    idx.write_text(json.dumps({"planner_tag": tag, "split": a.split,
                               "goal_json": str(a.goal_json) if a.goal_json else None,
                               "s2": {"dir": a.s2_dir, "model": a.s2_model},
                               "n_tasks": len(rows), "tasks": rows}, indent=1, default=str))
    ok = sum(1 for r in rows if r.get("recipe"))
    print(f"\n{ok}/{len(rows)} recipes written -> {out_root}\nindex: {idx}")


if __name__ == "__main__":
    main()
