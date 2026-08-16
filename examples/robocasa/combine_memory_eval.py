"""COMBINED System2 + System1 eval with a MEMORY-BASED plan (the ``-memory`` variant).

Identical to ``combined_eval.py`` except for the PLAN step. Instead of one look at the opening
scene, System2 first watches a video of the task chunk by chunk, narrates each chunk, aggregates
the narrations into a reusable recipe, and only then plans:

    NARRATE  chunk i   goal + narrations so far + tiled 4s clip   -> <narration>   (x N chunks)
    RECIPE             goal + ALL narrations, NO media            -> <summary>
    PLAN (warm)        goal + recipe + tiled opening still        -> <thought> + <plan>
      |
      +-> then the EXACT execution loop of combined_eval.py (unchanged): EXEC turn -> S1 segment
          -> clip -> EXEC turn -> ... -> <judge>task_finish

All four System2 modes here are ones the checkpoint was TRAINED on (``summary_v2``,
``summary_v2 · recipe``, ``plan_exec_v2 · with memory``, ``plan_exec_v2 · execution``); the prompts
live in ``sys2_memory.py`` and are byte-identical to the training shards. Nothing about the
execution half changes, so a ``-memory`` run is directly comparable to its cold twin: same S1, same
S2, same stop rule, same turn budget -- only the plan differs.

PRIVILEGED / ORACLE INPUT -- read this before comparing numbers
--------------------------------------------------------------
``--memory-source demo`` (the default) narrates the RECORDED HUMAN DEMO OF THE EPISODE BEING
EVALUATED. That is ground truth for this exact scene: the recipe is distilled from a successful
solution of the very layout the robot is about to face. It measures the CEILING of memory-based
planning, not transfer. Every episode.json records ``config.memory.privileged: true`` and the
episode index the memory came from, so this can never be mistaken for a clean number.

``--memory-episode N`` narrates a DIFFERENT episode of the same task instead (held-out demo). That
is the honest transfer setting -- the recipe comes from prior experience with the task kind, and the
eval scene is unseen. ``config.memory.privileged`` is then false. Use it once the oracle variant
shows the plan is worth conditioning on.

Run (robocasa micromamba env; needs a System2 vLLM server AND a System1 policy server) --
one CloseFridge episode:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$ROBOCASA_PY" examples/robocasa/combine_memory_eval.py \
        --lerobot-dir "$ROBOCASA_LEROBOT_ROOT/v1.0/target/atomic/CloseFridge/20250816/lerobot" \
        --episodes 0 \
        --s1-dir "$S1_CKPT" --s2-dir "$S2_CKPT" \
        --s1-port 8060 --s2-port 8100 \
        --norm-stats "$S1_CKPT/assets/robocasa_system1/norm_stats.json" \
        --out-root "$SYS1_RESULTS_DIR/combine"

Results land in ``<out-root>/<derived-method>-memory/`` (e.g.
``s1-progact270k_s2-qwen35-4b-full-ep3-11416-memory``) with the same layout the /combine GUI
already reads, plus one extra dir per episode:

    plan/
      memory/chunk00/{clip_full.mp4, clip.mp4, narration.json}   <- what S2 watched + said
      memory/recipe.json                                          <- the aggregation call
      memory/memory.json                                          <- recipe + provenance
      plan.json                                                   <- the warm plan call
      scene_full.png / scene.png
"""

from __future__ import annotations

# This variant is defined as "combined_eval's loop with a different plan step", so it deliberately
# reuses that module's helpers (_write_json, _task_name_from_lerobot_dir) rather than duplicating
# them -- keeping the two files' output layout identical is the whole point. Hence SLF001 is waived
# here; it is a design decision, not an oversight.
# ruff: noqa: SLF001
import argparse
import hashlib
import json
from pathlib import Path
import time

# Sibling imports (this file runs as a script; examples/robocasa is sys.path[0]).
import combined_eval as CE
import sys2_client as S2C
import sys2_memory as MEM


def load_recipe_library(path: Path) -> dict[str, dict]:
    """task -> {recipe, source_episode, planner_tag}. Accepts two shapes.

    * ``TASK_SUMMARY.json`` as written by pick_task_summary.py:
      ``{"planner_tag": ..., "tasks": {"<Task>": {"recipe": ..., "chosen_episode": N, ...}}}``
    * a bare map ``{"<Task>": "<recipe text>"}``

    A LIBRARY RECIPE REPLACES THE NARRATE->RECIPE PASS ENTIRELY. That is the point: the recipe is
    built once per task (from whichever episode the library chose, by whatever model wrote it) and
    reused for every episode of that task, so the plan is conditioned on task-level prior experience
    rather than on a fresh look at the episode under eval.
    """
    doc = json.loads(Path(path).read_text())
    tag = doc.get("planner_tag")
    rows = doc.get("tasks") if isinstance(doc.get("tasks"), dict) else doc
    out: dict[str, dict] = {}
    for task, v in rows.items():
        if isinstance(v, str):
            out[task] = {"recipe": v.strip(), "source_episode": None, "planner_tag": tag}
        elif isinstance(v, dict) and (v.get("recipe") or "").strip():
            out[task] = {"recipe": v["recipe"].strip(),
                         "source_episode": v.get("chosen_episode"),
                         "rule": v.get("rule"), "planner_tag": tag}
    if not out:
        raise SystemExit(f"--recipe-json {path} has no usable recipes")
    return out


def memory_run_config(args) -> dict:
    """Stable plan provenance stored in the run guard before any episode is evaluated.

    The recipe text is model input, just like the goal. Hash its CONTENTS so editing a library in
    place, switching libraries, or switching between a library and per-episode demo memory cannot
    silently append a different plan arm to the same method directory under ``--resume``.
    """
    if args.recipe_json:
        p = Path(args.recipe_json).expanduser()
        raw = p.read_bytes()
        lib = load_recipe_library(p)
        tags = sorted({v.get("planner_tag") for v in lib.values() if v.get("planner_tag")})
        return {
            "mode": "memory-library",
            "recipe_json": str(p),
            "sha256": hashlib.sha256(raw).hexdigest()[:16],
            "n_tasks": len(lib),
            "planner_tags": tags,
            "strict": bool(args.recipe_json_strict),
        }
    return {
        "mode": "memory-demo",
        "memory_source": args.memory_source,
        # None means each evaluated episode supplies its own demo: the fully privileged oracle arm.
        "memory_episode": args.memory_episode,
        "max_chunks": args.max_chunks,
    }


def make_plan_fn(args):
    """Build the ``plan_fn`` that ``combined_eval.eval_episode`` calls in place of the cold plan.

    Closes over ``args`` for the memory knobs. The returned callable owns everything it writes
    under ``plan_dir`` and returns the ``{"plan", "doc", "variant"}`` contract documented on
    ``combined_eval.do_plan_cold``.
    """

    lib = load_recipe_library(Path(args.recipe_json)) if args.recipe_json else {}

    def plan_fn(s2_client, instruction: str, plan_dir: Path, a) -> dict:
        lerobot_dir = Path(a.lerobot_dir)
        task_name = CE._task_name_from_lerobot_dir(lerobot_dir)
        if lib:
            return _plan_from_library(s2_client, instruction, plan_dir, a, task_name, lib)
        # Which episode's video becomes the memory. Default: the episode under eval (oracle).
        mem_ep = a.memory_episode if a.memory_episode is not None else int(a.episode_index)
        privileged = (mem_ep == int(a.episode_index))

        t0 = time.perf_counter()
        mem = MEM.build_memory(
            s2_client, lerobot_dir=lerobot_dir, ep_index=mem_ep, instruction=instruction,
            out_dir=plan_dir / "memory", max_chunks=a.max_chunks)
        t_mem = time.perf_counter() - t0
        print(f"  memory: {mem['n_chunks_narrated']} chunks narrated from episode {mem_ep} "
              f"({t_mem:.1f}s) -> recipe {len(mem['recipe'])} chars", flush=True)

        # ---- warm plan: recipe + the opening tiled still ----
        plan_dir.mkdir(parents=True, exist_ok=True)
        scene_full = plan_dir / "scene_full.png"
        img0 = S2C.write_image(a.scene0, scene_full)
        S2C.write_image(S2C.downscale([a.scene0])[0], plan_dir / "scene.png")
        goal = mem["goal_phrase"]
        t1 = time.perf_counter()
        p = MEM.plan_with_memory(s2_client, goal, mem["recipe"], scene_full)
        t_plan = time.perf_counter() - t1
        plan = (p.get("plan") or "").strip()

        CE._write_json(plan_dir / "plan.json", {
            "mode": "plan_with_memory",
            "s2_system_prompt": MEM.SYS_PLAN_MEM,
            "s2_user_prompt": p["user_prompt"],
            "s2_response_raw": p["raw"], "thought": p.get("thought"), "plan": plan,
            "media": {"image": img0},
            "latency_s": p.get("latency_s"), "usage": p.get("usage"),
            # The recipe is repeated here so plan.json alone explains the prompt it produced.
            "memory": {"recipe": mem["recipe"], "narrations": mem["narrations"],
                       "source_episode": mem_ep, "privileged": privileged,
                       "dir": "memory"},
        })
        return {
            "plan": plan,
            "variant": "memory",
            "doc": {
                "thought": p.get("thought"), "plan": plan, "latency_s": p.get("latency_s"),
                "dir": plan_dir.name,
                "memory": {
                    "source": a.memory_source, "source_episode": mem_ep,
                    "privileged": privileged,
                    "recipe": mem["recipe"], "narrations": mem["narrations"],
                    "n_chunks_narrated": mem["n_chunks_narrated"],
                    "n_chunks_skipped": mem["n_chunks_skipped"],
                    "n_source_frames": mem["n_source_frames"],
                    "chunk_source_frames": mem["chunk_source_frames"],
                    "dir": "memory",
                    "seconds": {"memory": round(t_mem, 2), "plan": round(t_plan, 2)},
                },
            },
        }

    return plan_fn


def _plan_from_library(s2_client, instruction: str, plan_dir: Path, a, task_name: str,
                       lib: dict[str, dict]) -> dict:
    """Warm plan from a PRE-BUILT per-task recipe: no narration, no per-episode recipe call.

    One System2 call total (the warm plan), against the same ``SYS_PLAN_MEM`` prompt the per-episode
    memory path uses -- so the only difference from that path is WHERE the recipe came from.
    ``--recipe-json-strict`` decides whether a task missing from the library is a hard error or falls
    back to the cold plan; strict by default, because silently mixing cold and warm plans in one
    sweep would make the arm uninterpretable.
    """
    ent = lib.get(task_name)
    if ent is None:
        if a.recipe_json_strict:
            raise SystemExit(f"--recipe-json has no recipe for {task_name} "
                             "(use --no-recipe-json-strict to fall back to the cold plan)")
        print(f"  recipe library MISS for {task_name}: falling back to the cold plan", flush=True)
        return CE.do_plan_cold(s2_client, instruction, plan_dir, a)

    plan_dir.mkdir(parents=True, exist_ok=True)
    scene_full = plan_dir / "scene_full.png"
    img0 = S2C.write_image(a.scene0, scene_full)
    S2C.write_image(S2C.downscale([a.scene0])[0], plan_dir / "scene.png")
    recipe = ent["recipe"]
    t1 = time.perf_counter()
    p = MEM.plan_with_memory(s2_client, instruction, recipe, scene_full)
    t_plan = time.perf_counter() - t1
    plan = (p.get("plan") or "").strip()

    # PRIVILEGED only if the recipe happens to come from the very episode under eval. A library
    # recipe is normally held-out for 4 of 5 episodes of its task and for every other episode, so
    # this is recorded per episode rather than assumed either way.
    src_ep = ent.get("source_episode")
    privileged = (src_ep is not None and int(src_ep) == int(a.episode_index))
    prov = {"source": "recipe_library", "recipe_json": str(a.recipe_json),
            "planner_tag": ent.get("planner_tag"), "selection_rule": ent.get("rule"),
            "source_episode": src_ep, "privileged": privileged,
            "recipe": recipe, "narrations": [], "n_chunks_narrated": 0,
            "seconds": {"memory": 0.0, "plan": round(t_plan, 2)}}
    CE._write_json(plan_dir / "plan.json", {
        "mode": "plan_with_memory (library recipe)",
        "s2_system_prompt": MEM.SYS_PLAN_MEM,
        "s2_user_prompt": p["user_prompt"],
        "s2_response_raw": p["raw"], "thought": p.get("thought"), "plan": plan,
        "media": {"image": img0},
        "latency_s": p.get("latency_s"), "usage": p.get("usage"),
        "memory": prov,
    })
    print(f"  recipe library: {task_name} <- {ent.get('planner_tag')} "
          f"ep{src_ep} ({len(recipe)} chars), warm plan {t_plan:.1f}s", flush=True)
    return {"plan": plan, "variant": "memory-library",
            "doc": {"thought": p.get("thought"), "plan": plan, "latency_s": p.get("latency_s"),
                    "dir": plan_dir.name, "memory": prov}}


def main():
    ap = CE.build_argparser(__doc__)
    g = ap.add_argument_group("memory plan variant")
    g.add_argument("--memory-source", choices=["demo"], default="demo",
                   help="what System2 narrates. 'demo' = the recorded LeRobot demo video "
                        "(the only source implemented; --memory-episode picks WHICH demo)")
    g.add_argument("--memory-episode", type=int, default=None,
                   help="episode index whose demo video becomes the memory. Default: the episode "
                        "being evaluated -- which is PRIVILEGED (the model sees a successful "
                        "solution of this exact scene). Pass a different index for the honest "
                        "held-out-demo setting.")
    g.add_argument("--recipe-json", default=None,
                   help="PRE-BUILT recipe library: TASK_SUMMARY.json (pick_task_summary.py) or a bare "
                        "{task: recipe} map. When given, the narrate->recipe pass is SKIPPED and the "
                        "task's library recipe conditions the warm plan for EVERY episode of that "
                        "task, so the plan reflects task-level prior experience instead of a fresh "
                        "look at the episode under eval.")
    g.add_argument("--recipe-json-strict", action=argparse.BooleanOptionalAction, default=True,
                   help="with --recipe-json, fail on a task the library does not cover rather than "
                        "silently falling back to the cold plan (which would mix two plan modes)")
    g.add_argument("--max-chunks", type=int, default=None,
                   help="narrate only the first N 4s chunks (debugging; it changes the recipe and "
                        "is recorded in memory.json)")
    args = ap.parse_args()
    args.run_plan_config = memory_run_config(args)
    args.plan_fn = make_plan_fn(args)
    CE.run_sweep(args, method_suffix="-memory")


if __name__ == "__main__":
    main()
