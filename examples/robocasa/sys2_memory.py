"""System2 MEMORY mode: narrate a demo video chunk by chunk -> reusable recipe -> warm plan.

This is the 4th System2 mode (``summary_v2``), the one ``sys2_client`` deliberately left out because
the cold-plan loop does not need it. It turns a video into a *recipe* the planner can be conditioned
on:

    NARRATE  chunk i : goal + narrations 1..i-1 + tiled 4s clip  -> <narration>   (one sentence)
    RECIPE           : goal + ALL narrations, NO media           -> <summary>     (reusable recipe)
    PLAN (warm)      : goal + recipe + tiled opening still       -> <thought> + <plan>

Every prompt string below was lifted VERBATIM from the training producer -- do not reword:
  * ``SYS_NARRATE``  == RoboAnnotator/robo_annotator/server.py::SUMV2_SYS       (server.py:1146)
  * ``SYS_RECIPE``   == RoboAnnotator/robo_annotator/server.py::SUMV2_FIN_SYS   (server.py:1155)
  * ``SYS_PLAN_MEM`` == RoboAnnotator/robo_annotator/server.py::PLANV2_SYS_MEM  (server.py:1248)
  * user templates   == RoboAnnotator/robo_annotator/system2/v2_rows.py:426-430 (narrate),
                        :445-447 (recipe), :477-480 (warm plan)
The shard builder lifts those same closures by ``ast.literal_eval`` (``v2_rows.prompts()``) and
``scripts/check_v2_rows_match.py`` asserts byte-identity, so the strings here are the training
targets, not a paraphrase of them.

Three details are easy to get wrong and all three are load-bearing:

  1. CHUNK BOUNDARIES ARE FIXED WINDOWS, NOT ANNOTATED SPANS. ``CHUNK = 80`` source frames (4 s at
     20 Hz) from frame 0: [0,79], [80,159], ... with the LAST window truncated to the episode end
     (``refine_summary.py::_chunk_windows``). Execution turns use ground-truth step spans; summary
     mode never did. Feeding subgoal spans here would be off-distribution in both length and
     content.
  2. THE RECIPE CALL CARRIES NO MEDIA. It is text-only -- goal + the numbered narration list.
  3. THE RECIPE IS INTERPOLATED INSIDE LITERAL DOUBLE QUOTES in the warm-plan user turn
     (``f'"{memory}"'``). The model was supervised on the quotes.

``goal`` is the instruction with its FIRST LETTER LOWERCASED (``v2_rows._goal_line``), because the
templates embed it mid-sentence ("The goal of the task is: close the fridge door.").

Clip encoding matches the shard producer: 3 cameras (agentview_left | agentview_right | eye_in_hand)
hstacked with no gap -> 256x768, resampled to a uniform 4 fps by ``np.linspace`` over the window
with the endpoints included, x264 crf 26. Windows too short to yield 2 frames are DROPPED, never
padded -- padding would invent content and lie about the duration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sys2_client as S2C

# ---------------------------------------------------------------------------
# Chunking / clip constants. Mirrored from the producer, NOT re-chosen here.
#   CHUNK            RoboAnnotator/producers/refine_summary.py:47 (== gen_repurpose.py:47)
#   SRC_FPS/CLIP_FPS/CLIP_CRF/CLIP_MIN_FRAMES  producers/build_system2_shards.py:66-73
#   CAM_ORDER        producers/build_system2_shards.py:82
CHUNK = 80              # source frames per narration window = 4 s at 20 Hz
CLIP_MIN_FRAMES = 2     # x264 needs >= 2; shorter windows are dropped, never padded
CAM_ORDER = ("scene_left", "scene_right", "wrist")   # == agentview_left | agentview_right | wrist
# The LeRobot video key for each tile position, in the same order as CAM_ORDER.
LEROBOT_VIDEO_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


# ---------------------------------------------------------------------------
# Prompts -- VERBATIM from the training producer. Do not reword.

SYS_NARRATE = (
    "You are watching a single Franka Panda arm on a mobile base perform a kitchen task through "
    "three synchronized cameras (left-shoulder, right-shoulder, wrist) that are tiled side by "
    "side, a few seconds at a time. "
    "Given the goal and the running narration so far, narrate THIS clip in one faithful sentence: "
    "what the robot does and how it advances the task. Be concrete — name the objects/fixtures and "
    "the motion — but describe ONLY what is visible and NEW in this clip (no unseen intent, no "
    "repeating earlier clips, no frame numbers). Respond with <narration>...</narration>."
)

SYS_RECIPE = (
    "You have watched a single Franka Panda arm on a mobile base perform a kitchen task through "
    "three cameras (left-shoulder, right-shoulder, wrist) that are tiled side by side, clip by "
    "clip. Now turn what you saw "
    "into a short, REUSABLE recipe for a task of this KIND: the general strategy plus the few "
    "non-obvious details that decide success (fixture structure, counts, access/reposition, "
    "recovery), explaining WHY each is needed. Keep it GENERAL — no run-specific sides or "
    "positions (say 'the target container', not 'the bowl on the left'). Respond with "
    "<summary>...</summary>."
)

SYS_PLAN_MEM = (
    "You are the high-level planner for a single Franka Panda arm on a mobile base in a home "
    "kitchen, seen from three synchronized cameras (left-shoulder, right-shoulder, wrist) that "
    "are tiled side by side into one image. Given the task goal, the opening scene, and a short "
    "recipe recalled from past experience with this kind of task, perceive the scene then reason "
    "toward the plan by COMBINING what you see with the recipe — follow it where it fits, adapt "
    "or reorder where the scene differs; the scene is the final authority.\n"
    "Respond with <thought>...</thought> — FIRST-PERSON narrative ('I see...', 'I'll need to...') "
    "covering what you see, what the goal needs, why this order (and where something is unknown, "
    "say you'll discover it) — then <plan>...</plan>, a checklist of milestones only (one "
    "`- [ ] Mk: ...` line each, no fine steps). Always close every tag you open."
)

# server.py:1260 -- the plan turn's camera line; one <image> for the tiled 3-view still.
PLAN_CAM_LINE = "Here is the scene right now:\n<image>"


def goal_phrase(instruction: str) -> str:
    """The instruction with its first letter lowercased (``v2_rows._goal_line``).

    The templates embed the goal mid-sentence, so training always saw it lowercased:
    "The goal of the task is: close the fridge door." Passing the raw capitalised instruction is a
    small but real off-distribution change.
    """
    instr = (instruction or "").strip()
    return (instr[0].lower() + instr[1:]) if instr else "complete the task."


def user_narrate(goal: str, prior_narrations: list[str], chunk_index: int) -> str:
    """NARRATE turn i -- goal + the running narration list + one 4 s <video>.

    ``chunk_index`` only picks "clip" vs "next clip" (it is the index among the chunks actually
    narrated, matching the producer's enumerate over the surviving chunk files).
    """
    prior = ("This is the first clip, so there's nothing narrated yet."
             if not prior_narrations else
             "So far:\n" + "\n".join(f"{j+1}. {t}" for j, t in enumerate(prior_narrations)))
    clipword = "clip" if chunk_index == 0 else "next clip"
    return (f"The goal of the task is: {goal}\n\n{prior}\n\n"
            f"Here is the {clipword} (about 4 seconds):\n<video>\n\n"
            "What happens in the video?")


def user_recipe(goal: str, narrations: list[str]) -> str:
    """RECIPE turn -- goal + every narration, NO media (text only)."""
    lst = "\n".join(f"{j+1}. {t}" for j, t in enumerate(narrations))
    return (f"The goal of the task was: {goal}\n\nHere is everything you "
            f"narrated across the run:\n{lst}\n\nThat's the whole task — "
            "write the reusable recipe.")


def user_plan_mem(goal: str, recipe: str) -> str:
    """PLAN (warm) turn -- goal + the recalled recipe IN QUOTES + one tiled <image>.

    With an empty recipe the memory block vanishes entirely (the producer's ``if memory else ""``),
    which leaves a prompt that is the cold one with a different closing line -- so an empty recipe
    is a caller error, not something to silently paper over. ``build_memory`` raises instead.
    """
    # The recipe is wrapped in LITERAL DOUBLE QUOTES -- the model was supervised on the quotes
    # (v2_rows.py:478 `\"{memory}\"`), so they are part of the prompt, not formatting.
    mem_block = ("Here is a recipe recalled from earlier experience with this kind of task, to "
                 f'help you plan:\n"{recipe}"\n\n' if recipe else "")
    return (f"The goal is: {goal}\n\n{mem_block}{PLAN_CAM_LINE}\n\n"
            "Combine what you see with the recalled recipe, then lay out your milestone plan.")


# ---------------------------------------------------------------------------
# Demo-video decoding -> fixed 4 s windows -> tiled clips


def chunk_windows(total_frames: int, *, chunk: int = CHUNK) -> list[tuple[int, int]]:
    """Fixed windows [0,79], [80,159], ... with the last truncated to ``total_frames - 1``.

    Byte-for-byte the producer's ``refine_summary.py::_chunk_windows``. NOT annotated spans.
    """
    out: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        out.append((start, min(start + chunk - 1, total_frames - 1)))
        start += chunk
    return out


def sampled_frame_count(span_len: int, *, src_fps: float = S2C.SIM_FPS,
                        clip_fps: float = S2C.CLIP_FPS) -> int:
    """Frames STORED for a window: uniform ``clip_fps``, no cap (the loader caps at 32).

    ``build_system2_shards.py::sampled_frame_count``. Returns 0 when the window is too short to be
    representable, which means "drop this window" -- never pad.
    """
    if span_len <= 0:
        return 0
    n = round(span_len / src_fps * clip_fps)
    n = min(n, span_len)
    return n if n >= CLIP_MIN_FRAMES else 0


def window_frame_indices(start: int, end: int) -> list[int]:
    """Absolute source-frame indices sampled from [start, end], endpoints INCLUDED.

    ``np.linspace(...).round()``, not a stride: the first frame is the state the window starts
    from and the last is what it ends at (``_encode_clip_uncached``).
    """
    span = list(range(int(start), int(end) + 1))
    n = sampled_frame_count(len(span))
    if n < CLIP_MIN_FRAMES:
        return []
    idx = np.linspace(0, len(span) - 1, n).round().astype(int)
    return [span[i] for i in idx]


def read_demo_tiles(lerobot_dir: Path, ep_index: int, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode the 3 recorded camera mp4s and return {frame_index: tiled 256x768 uint8}.

    The recorded videos are stored TOP-DOWN, i.e. already in the same orientation as
    ``subtask_env.images_from_obs`` (verified against a live ``reset_to`` render: per-camera MSE ~7
    as-is vs ~7000 vertically flipped -- the residual is h264 quantisation). So no flip is applied
    and the tile matches what the exec loop feeds System2 for its own rollout clips.

    Only frames in ``wanted`` are kept, so a 500-frame episode costs ~100 tiles, not 500.
    """
    import imageio.v2 as imageio

    ld = Path(lerobot_dir)
    chunk_id = ep_index // 1000          # info.json: chunks_size = 1000
    per_cam: dict[str, dict[int, np.ndarray]] = {}
    for key in LEROBOT_VIDEO_KEYS:
        path = ld / "videos" / f"chunk-{chunk_id:03d}" / key / f"episode_{ep_index:06d}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"missing recorded demo video: {path}")
        keep: dict[int, np.ndarray] = {}
        rdr = imageio.get_reader(str(path))
        try:
            for i, frame in enumerate(rdr):
                if i in wanted:
                    keep[i] = np.asarray(frame, dtype=np.uint8)
                if len(keep) == len(wanted):
                    break
        finally:
            rdr.close()
        per_cam[key] = keep

    tiles: dict[int, np.ndarray] = {}
    for i in sorted(wanted):
        panels = [per_cam[k].get(i) for k in LEROBOT_VIDEO_KEYS]
        if any(p is None for p in panels):
            continue          # a frame missing from one camera: skip it rather than tile a hole
        tiles[i] = S2C.tile_frame(*panels)
    return tiles


def demo_frame_count(lerobot_dir: Path, ep_index: int) -> int:
    """Frames in the recorded demo, read from meta/episodes.jsonl (no decode needed)."""
    f = Path(lerobot_dir) / "meta" / "episodes.jsonl"
    with f.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if int(d.get("episode_index", -1)) == int(ep_index):
                return int(d["length"])
    raise KeyError(f"episode {ep_index} not found in {f}")


# ---------------------------------------------------------------------------
# The memory pass


def parse_narration(raw: str) -> str | None:
    return S2C.tag_text(raw, "narration")


def parse_recipe(raw: str) -> str | None:
    return S2C.tag_text(raw, "summary")


def parse_plan_mem(raw: str) -> dict:
    """Same tags as the cold plan: <thought> + <plan>."""
    return {"thought": S2C.tag_text(raw, "thought"), "plan": S2C.tag_text(raw, "plan"), "raw": raw}


def build_memory(
    s2: S2C.Sys2Client, *, lerobot_dir: Path, ep_index: int, instruction: str, out_dir: Path,
    max_chunks: int | None = None, save_display_copies: bool = True,
) -> dict:
    """Narrate the demo chunk by chunk, then aggregate into one recipe.

    Returns a dict with the recipe, every narration, every prompt/response, and the per-chunk media
    provenance -- all of it also written under ``out_dir`` so a run is auditable without re-calling
    the model:

        memory/
          chunk00/{clip_full.mp4, clip.mp4, narration.json}
          ...
          recipe.json
          memory.json          <- the summary this function returns

    ``max_chunks`` truncates the narration pass (debugging only; it changes the recipe, so it is
    recorded in the output).
    """
    goal = goal_phrase(instruction)
    total = demo_frame_count(lerobot_dir, ep_index)
    windows = chunk_windows(total)
    if max_chunks is not None:
        windows = windows[:max_chunks]

    # One decode pass for every frame any window needs.
    per_window_idx = [window_frame_indices(s, e) for (s, e) in windows]
    wanted = {i for idx in per_window_idx for i in idx}
    if not wanted:
        raise ValueError(f"episode {ep_index} ({total} frames) yields no usable narration window")
    tiles = read_demo_tiles(lerobot_dir, ep_index, wanted)

    narrations: list[str] = []
    chunks: list[dict] = []
    skipped: list[dict] = []
    for (start, end), idx in zip(windows, per_window_idx, strict=True):
        w = len(chunks) + len(skipped)
        if not idx:
            # Too short to be representable at 4 fps -- the producer drops these.
            skipped.append({"window": [start, end], "reason": "span too short for 4 fps (<2 frames)"})
            continue
        frames = [tiles[i] for i in idx if i in tiles]
        if len(frames) < CLIP_MIN_FRAMES:
            skipped.append({"window": [start, end], "reason": "missing decoded frames"})
            continue

        cdir = out_dir / f"chunk{w:02d}"
        # The model reads the FULL-RES clip; the small copy is for the GUI only.
        model_clip = cdir / "clip_full.mp4"
        info = S2C.write_clip(frames, model_clip)
        disp = (S2C.write_clip(S2C.downscale(frames), cdir / "clip.mp4")
                if save_display_copies else None)

        # ``chunk_index`` is the index among NARRATED chunks (it only selects "clip"/"next clip"),
        # matching the producer's enumerate over the surviving chunk files.
        user = user_narrate(goal, narrations, len(narrations))
        r = s2.chat(SYS_NARRATE, user, video=model_clip, n_video_frames=len(frames))
        narr = parse_narration(r["text"])

        rec = {
            "chunk": w, "window": [int(start), int(end)],
            "n_source_frames": int(end - start + 1),
            "n_clip_frames": len(frames), "frame_indices": [int(i) for i in idx],
            "nframes_requested": r["nframes"],
            "s2_system_prompt": SYS_NARRATE, "s2_user_prompt": user,
            "s2_response_raw": r["text"], "narration": narr,
            "media": {"video": info, "display_video": disp,
                      "model_res": list(S2C.TILE_HW), "display_res": list(S2C.DISPLAY_HW),
                      "source": "recorded LeRobot demo video"},
            "latency_s": r["latency_s"], "usage": r["usage"],
        }
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "narration.json").write_text(json.dumps(rec, indent=1, default=str))
        chunks.append(rec)
        if narr:
            narrations.append(narr)
        else:
            # A chunk that produced no <narration> contributes nothing to the running context;
            # say so loudly rather than letting the recipe quietly lose a window.
            print(f"  [memory] WARNING chunk {w} ({start}-{end}) returned no <narration>", flush=True)

    if not narrations:
        raise ValueError("no chunk produced a <narration>; cannot build a recipe")

    # ---- aggregation: ONE final model call, text only (no video, no image) ----
    ruser = user_recipe(goal, narrations)
    rr = s2.chat(SYS_RECIPE, ruser)
    recipe = (parse_recipe(rr["text"]) or "").strip()
    rdoc = {
        "s2_system_prompt": SYS_RECIPE, "s2_user_prompt": ruser,
        "s2_response_raw": rr["text"], "recipe": recipe,
        "narrations": narrations, "media": None,      # text-only by design
        "latency_s": rr["latency_s"], "usage": rr["usage"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "recipe.json").write_text(json.dumps(rdoc, indent=1, default=str))
    if not recipe:
        raise ValueError("recipe call returned no <summary>")

    summary = {
        "goal_phrase": goal,
        "recipe": recipe,
        "narrations": narrations,
        # Per-chunk index for the GUI: enough to play each clip and read what it produced, without
        # re-opening every chunk's narration.json.
        "chunks": [{"chunk": c["chunk"], "window": c["window"],
                    "n_clip_frames": c["n_clip_frames"], "narration": c["narration"],
                    "latency_s": c["latency_s"]} for c in chunks],
        "n_chunks_narrated": len(chunks),
        "n_chunks_skipped": len(skipped),
        "skipped": skipped,
        "n_source_frames": int(total),
        "chunk_source_frames": CHUNK,
        "windows": [[int(a), int(b)] for a, b in windows],
        "max_chunks": max_chunks,
        "clip": {"fps": S2C.CLIP_FPS, "crf": S2C.CLIP_CRF, "tile": list(S2C.TILE_HW),
                 "src_fps": S2C.SIM_FPS, "sampling": "np.linspace over the window, endpoints included"},
        "video_policy_source": S2C.policy_source(),
        "latency_s": {
            "narrate_total": round(sum(c["latency_s"] or 0 for c in chunks), 2),
            "recipe": rr["latency_s"],
        },
        "dir": out_dir.name,
    }
    (out_dir / "memory.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def plan_with_memory(s2: S2C.Sys2Client, goal: str, recipe: str, scene_png: Path) -> dict:
    """PLAN (warm) -- the recalled recipe + the opening tiled still -> <thought> + <plan>."""
    user = user_plan_mem(goal, recipe)
    r = s2.chat(SYS_PLAN_MEM, user, image=scene_png)
    out = parse_plan_mem(r["text"])
    out.update(latency_s=r["latency_s"], usage=r["usage"], user_prompt=user)
    return out


if __name__ == "__main__":   # self-check: no server, no sim
    import argparse

    ap = argparse.ArgumentParser(description="Self-check the memory-mode prompts and chunking.")
    ap.add_argument("--lerobot-dir", default=None, help="optional: also chunk a real episode")
    ap.add_argument("--episode", type=int, default=0)
    a = ap.parse_args()

    print("windows(258) =", chunk_windows(258))
    print("windows(429) =", chunk_windows(429))
    print("stored frames: 80 ->", sampled_frame_count(80), " 18 ->", sampled_frame_count(18),
          " 7 ->", sampled_frame_count(7), "(0 = dropped)")
    print("indices for [240,257] =", window_frame_indices(240, 257))
    print("request nframes for a 16-frame clip =",
          S2C.smart_nframes(16, S2C.CLIP_FPS, S2C.DEFAULT_VIDEO_POLICY))

    g = goal_phrase("Close the fridge door.")
    print("\n--- narrate, chunk 0 ---\n" + user_narrate(g, [], 0))
    print("\n--- narrate, chunk 2 ---\n" + user_narrate(g, ["The arm reaches toward the door.",
                                                            "The arm pushes the door shut."], 2))
    print("\n--- recipe ---\n" + user_recipe(g, ["The arm reaches toward the door.",
                                                 "The arm pushes the door shut."]))
    print("\n--- warm plan ---\n" + user_plan_mem(g, "Approach the open door, then push it shut."))

    if a.lerobot_dir:
        n = demo_frame_count(Path(a.lerobot_dir), a.episode)
        ws = chunk_windows(n)
        print(f"\nepisode {a.episode}: {n} frames -> {len(ws)} windows")
        for i, (s, e) in enumerate(ws):
            idx = window_frame_indices(s, e)
            print(f"  chunk{i:02d} [{s},{e}] -> {len(idx)} clip frames"
                  f"{' (DROPPED)' if not idx else ''}")
