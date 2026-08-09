"""System2 (Qwen3.5-4B planner) client for the COMBINED System2+System1 eval.

Talks to a vLLM OpenAI-protocol server serving the trained System2 checkpoint and reproduces the
EXACT prompt shapes the model was trained on. Every prompt string here was lifted verbatim from
the training/eval shards (``system2_target_0804``); do not "improve" the wording — the model was
supervised on these exact templates and paraphrasing them is an off-distribution change.

Three modes are used by the closed loop (a 4th, summary/recipe, is only needed for the
"with memory" plan variant and is not used by the cold-plan loop):

  PLAN (cold)        1 tiled image  -> <thought> + <plan>            (milestone checklist)
  EXEC turn 1        checklist + 1 tiled image                       -> ... <subgoal>
  EXEC turn n        checklist + tiled video of the last S1 segment  -> ... <subgoal>

The EXEC user turn also carries two PRIVILEGED environment signals the model was trained to
trust — ``Current task status`` (ongoing/finished) and ``Current gripper status``
(open/close/unsure). The system prompt explicitly tells the model to judge ``task_finish`` only
when the status reads ``finished``, so the caller MUST feed these from the live env (see
combined_eval.py) or the loop will never terminate.

Media contract (verified against the shards, NOT guessed):
  * 3 cameras (scene_left, scene_right, wrist) hstacked -> 256x768 per frame ("tiled").
  * clips encoded at a uniform 4.0 fps, x264 crf 26, >=2 frames (README + meta.json).
  * source sim runs at 20 fps, so a span is decimated 20->4 fps (every 5th frame).
  * per-request frame count reproduces qwen-vl-utils smart_nframes via the named video policy;
    ``sys2_train_eval/sys2/data/video_policy.py`` is the single source of truth and is imported
    when reachable, with a vendored fallback so this file works without that repo on sys.path.
"""

from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# Video policy. Prefer the real module in sys2_train_eval so training and eval cannot drift;
# fall back to a vendored copy of the same constants/formula when that repo isn't importable.
SYS2_REPO = Path(os.environ.get("SYS2_REPO", "/home/ec2-user/sys2_train_eval"))
try:  # pragma: no cover - import path depends on the box
    import sys as _sys

    if str(SYS2_REPO) not in _sys.path:
        _sys.path.insert(0, str(SYS2_REPO))
    from sys2.data.video_policy import DEFAULT_VIDEO_POLICY  # type: ignore
    from sys2.data.video_policy import get_video_policy  # type: ignore
    from sys2.data.video_policy import smart_nframes  # type: ignore

    _POLICY_SOURCE = "sys2_train_eval"
except Exception:
    DEFAULT_VIDEO_POLICY = "system2_4fps_32"
    _VENDORED = {"system2_4fps_32": dict(fps=4.0, min_frames=2, max_frames=32, frame_factor=2)}
    _POLICY_SOURCE = "vendored"

    class _P:  # minimal stand-in with the fields we use
        def __init__(self, name, fps, min_frames, max_frames, frame_factor):
            self.name, self.fps = name, fps
            self.min_frames, self.max_frames, self.frame_factor = min_frames, max_frames, frame_factor

    def get_video_policy(name: str) -> _P:
        return _P(name, **_VENDORED[name])

    def smart_nframes(total_frames: int, source_fps: float, policy) -> int:
        """Vendored qwen-vl-utils 0.0.14 smart_nframes (same formula as the real module)."""
        p = get_video_policy(policy) if isinstance(policy, str) else policy
        if total_frames < p.frame_factor:
            raise ValueError(f"video has {total_frames} frames < frame_factor={p.frame_factor}")
        ceil_f = lambda v: math.ceil(v / p.frame_factor) * p.frame_factor  # noqa: E731
        floor_f = lambda v: math.floor(v / p.frame_factor) * p.frame_factor  # noqa: E731
        n = total_frames / source_fps * p.fps
        n = min(min(max(n, ceil_f(p.min_frames)), floor_f(min(p.max_frames, total_frames))), total_frames)
        return floor_f(n)


# Sim/source frame rate of the RoboCasa rollout (matches meta.json "src_fps": 20.0).
SIM_FPS = 20.0
CLIP_FPS = 4.0        # uniform clip fps baked into every training clip
CLIP_CRF = 26         # x264 crf used by the shard producer
TILE_HW = (256, 768)  # one tiled frame: 3 cameras of 256x256 hstacked -- MODEL INPUT resolution
# Saved-artifact resolution for the GUI. HALF the model tile: the browser renders these small, and
# at benchmark scale (50 tasks x ~506 episodes) full-res copies cost ~4x the disk for no benefit.
# NEVER feed a downscaled clip to System2 -- it was trained on 256x768 (see write_clip/model_clip).
DISPLAY_HW = (128, 384)

TAGS = ("thought", "judge", "plan_update", "estimated_step", "subgoal", "subgoal_detail")
JUDGES = ("task_begin", "subgoal_complete", "subgoal_incomplete", "subgoal_failed", "task_finish")


# ---------------------------------------------------------------------------
# Prompts — VERBATIM from system2_target_0804 shards. Do not reword.

SYS_PLAN_COLD = (
    "You are the high-level planner for a single Franka Panda arm on a mobile base in a home "
    "kitchen, seen from three synchronized cameras (left-shoulder, right-shoulder, wrist) that "
    "are tiled side by side into one image. Given the task goal and the opening scene, perceive "
    "the scene (name the objects and fixtures you can see and their state), then reason from the "
    "goal to an ordered list of milestones.\n"
    "Respond with <thought>...</thought> — FIRST-PERSON narrative ('I see...', 'I'll need to...') "
    "covering what you see, what the goal needs, why this order (and where something is unknown, "
    "say you'll discover it) — then <plan>...</plan>, a checklist of milestones only (one "
    "`- [ ] Mk: ...` line each, no fine steps). Always close every tag you open."
)

SYS_EXEC = (
    "You are the robot's high-level planner controlling a single Franka Panda arm with a "
    "parallel-jaw gripper on a mobile base in a home kitchen, seen from three synchronized "
    "cameras (left-shoulder, right-shoulder, wrist) that are tiled side by side. Each turn you "
    "watch the video of the step just executed (the first turn has no video — just the opening "
    "scene) and issue the next instruction. Reason only from what you can SEE; when a step goes "
    "wrong, say so and recover. A privileged 'Current task status' (ongoing/finished) from the "
    "environment tells you whether the goal is actually satisfied — judge task_finish only when "
    "it reads finished. A privileged 'Current gripper status' (open/close/unsure) reports whether "
    "the gripper is commanded open or closed at the end of the clip (unsure = mid-transition).\n"
    "The plan is a two-level checklist handed back each turn: milestones `- [ ] Mk: ...` with "
    "stable ids, fine steps `* [ ] Mk.n: ...` under the one in progress. Marks: `[ ]` todo, "
    "`[~]` doing, `[x]` done.\n"
    "Respond with these tags in order, each with its OPENING and CLOSING tag:\n"
    "<thought>...</thought> — FIRST-PERSON reasoning ('I see...', 'I've just...', 'I'll now...'): "
    "what the clip achieved, what the scene now affords or lacks, and WHY the next step follows "
    "from the goal; motion detail is secondary. Concise.\n"
    "<judge>...</judge> — one of: task_begin, subgoal_complete, subgoal_incomplete (step only "
    "PARTWAY done — keep it `[~]` and CONTINUE the same step), subgoal_failed (attempt fell short "
    "— name the cause honestly, keep `[~]`, issue the RECOVERY), task_finish.\n"
    "<plan_update>...</plan_update> — only the changed milestone block(s), ids/marks exact — also "
    "where you reveal now-visible detail.\n"
    "<estimated_step>...</estimated_step> — integer control-steps the next instruction should take "
    "(scaled to motion size).\n"
    "<subgoal>...</subgoal> — terse imperative; then <subgoal_detail>...</subgoal_detail> — "
    "natural elaboration.\n"
    "Always close every tag you open. On task_finish the task is done, so emit only "
    "<thought>...</thought>, <judge>...</judge>, <plan_update>...</plan_update> — no "
    "<estimated_step>, <subgoal>, or <subgoal_detail>."
)


def user_plan_cold(goal: str) -> str:
    """PLAN (cold / no memory) user turn — one <image> of the opening scene."""
    return (
        f"The goal is: {goal}\n\n"
        "Here is the scene right now:\n<image>\n\n"
        "How would you go about this? Perceive the scene, then lay out your milestone plan."
    )


def user_exec_first(goal: str, plan: str) -> str:
    """EXEC turn 1 — milestone plan + opening <image>; no step has run yet."""
    return (
        f"The goal is: {goal}\n\n"
        f"You've just been handed this milestone plan:\n{plan.strip()}\n\n"
        "Here is the scene right now:\n<image>\n\n"
        "No step has run yet. Unroll the current milestone into fine steps and issue the first fine step."
    )


def user_exec_turn(goal: str, plan: str, task_status: str, gripper_status: str) -> str:
    """EXEC turn n — plan state + <video> of the segment just executed + privileged signals."""
    return (
        f"The goal is: {goal}\n\n"
        f"Here's where the plan stands:\n{plan.strip()}\n\n"
        "The controller just carried out the current step — here's the video of what it did:\n<video>\n\n"
        f"Current task status: {task_status}\n"
        f"Current gripper status: {gripper_status}\n\n"
        "Please reason and update the plan about what to do next."
    )


# ---------------------------------------------------------------------------
# Response parsing


def tag_text(s: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", s or "", re.DOTALL)
    return m.group(1).strip() if m else None


def parse_exec(raw: str) -> dict:
    """Pull the ordered tags out of an EXEC response.

    ``estimated_step`` is coerced to int (the model occasionally writes '50 steps'); a missing or
    unparseable value returns None so the caller can apply its own default rather than silently
    running with a bogus budget.
    """
    out = {t: tag_text(raw, t) for t in TAGS}
    judge = (out.get("judge") or "").strip()
    out["judge"] = judge if judge in JUDGES else None
    out["judge_raw"] = judge
    est = out.get("estimated_step")
    if est is not None:
        m = re.search(r"-?\d+", est)
        out["estimated_step"] = int(m.group()) if m else None
    out["raw"] = raw
    return out


def parse_plan(raw: str) -> dict:
    return {"thought": tag_text(raw, "thought"), "plan": tag_text(raw, "plan"), "raw": raw}


# ---------------------------------------------------------------------------
# Plan checklist bookkeeping


def apply_plan_update(plan: str, plan_update: str | None) -> str:
    """Merge a <plan_update> block into the running checklist.

    The model returns ONLY the changed milestone block(s) (a `- [x] Mk: ...` line plus any `* [ ]
    Mk.n: ...` children). We replace the block for each milestone id present in the update and
    leave every other milestone untouched, preserving order. Unknown ids are appended so a
    revealed milestone is never dropped.
    """
    if not plan_update or not plan_update.strip():
        return plan

    def split_blocks(text: str) -> list[tuple[str, list[str]]]:
        """-> [(milestone_id, [lines...])] keeping each milestone with its child lines."""
        blocks: list[tuple[str, list[str]]] = []
        for line in (text or "").splitlines():
            if not line.strip():
                continue
            m = re.match(r"\s*-\s*\[.\]\s*(M\d+)\s*:", line)
            if m:
                blocks.append((m.group(1), [line]))
            elif blocks:
                blocks[-1][1].append(line)
        return blocks

    cur = split_blocks(plan)
    upd = dict(split_blocks(plan_update))
    seen: set[str] = set()
    out: list[str] = []
    for mid, lines in cur:
        if mid in upd:
            out.extend(upd[mid])
            seen.add(mid)
        else:
            out.extend(lines)
    for mid, lines in upd.items():
        if mid not in seen:
            out.extend(lines)
    return "\n".join(out)


def plan_is_all_done(plan: str) -> bool:
    """True when every milestone line is marked `[x]` (a soft completion hint, not a terminator)."""
    ms = re.findall(r"^\s*-\s*\[(.)\]\s*M\d+\s*:", plan or "", re.MULTILINE)
    return bool(ms) and all(m == "x" for m in ms)


# ---------------------------------------------------------------------------
# Media encoding


def tile_frame(scene_left: np.ndarray, scene_right: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """hstack the 3 views into one 256x768 tile (gap_px=0, order matches the training producer)."""
    return np.concatenate([scene_left, scene_right, wrist], axis=1)


# Subgoals whose WHOLE POINT is to stay still ("wait for the toaster to pop up", "hold the
# colander under the water"). Dropping static frames there would delete the very content the clip
# has to show and read to System2 as "nothing happened", so compaction is SKIPPED (plain 20->4 fps
# decimation only). Verified against the training shards: 11/2515 subgoals are wait-like.
WAIT_RE = re.compile(r"\b(wait|hold|hold\s+down|pause|stay|settle|keep\s+still|remain)\b", re.IGNORECASE)


def is_wait_subgoal(subgoal: str | None) -> bool:
    """True when the subgoal is stationary-by-design, so static frames must be KEPT."""
    return bool(WAIT_RE.search(subgoal or ""))


def decimate_only(
    frames: list[np.ndarray], *, policy: str = DEFAULT_VIDEO_POLICY
) -> tuple[list[np.ndarray], dict]:
    """20->4 fps decimation with NO static-frame removal (the wait/hold path).

    Keeps the stillness intact so "wait for the bread to pop up" still looks like waiting, while
    matching the training clip cadence and the 32-frame policy ceiling.
    """
    n_raw = len(frames)
    if n_raw == 0:
        return [], {"n_raw": 0, "n_final": 0, "mode": "decimate_only"}
    stride = max(1, int(round(SIM_FPS / CLIP_FPS)))
    idx = list(range(0, n_raw, stride))
    if (n_raw - 1) not in idx:
        idx.append(n_raw - 1)
    pol = get_video_policy(policy)
    if len(idx) > pol.max_frames:
        sel = np.linspace(0, len(idx) - 1, pol.max_frames).round().astype(int)
        idx = [idx[i] for i in sorted(set(sel.tolist()))]
    if len(idx) < 2:
        idx = sorted({0, n_raw - 1}) if n_raw >= 2 else [0, 0]
    return [frames[i] for i in idx], {
        "n_raw": n_raw, "n_moving": None, "n_kept_after_static_drop": n_raw,
        "n_final": len(idx), "eps": None, "stride": stride, "kept_indices": idx,
        "frac_static_dropped": 0.0, "mode": "decimate_only",
    }


# Static-frame threshold for CLIP CONDENSING. Deliberately 10x STRICTER than the stop rule's
# quiescence eps (0.02): that one asks "has motion effectively ceased?" (a decision), while this
# one asks "is this frame visually redundant?" (a filter). At 0.02 a slow-but-real approach would
# be discarded as static; 0.002 drops only near-zero motion, so genuine slow movement survives.
STATIC_EPS = 0.002


def build_clip_frames(
    frames: list[np.ndarray], motion: list[float], subgoal: str | None, *,
    eps: float = STATIC_EPS, policy: str = DEFAULT_VIDEO_POLICY,
) -> tuple[list[np.ndarray], dict]:
    """Pick the right frame-selection path for this segment's subgoal.

    wait/hold subgoals -> decimate only (stillness is the signal);
    everything else    -> drop near-static frames, then decimate.
    """
    if is_wait_subgoal(subgoal):
        out, st = decimate_only(frames, policy=policy)
        st["wait_subgoal"] = True
        return out, st
    out, st = compact_static_frames(frames, motion, eps=eps, policy=policy)
    st["wait_subgoal"] = False
    st["mode"] = "compact_static"
    return out, st


def compact_static_frames(
    frames: list[np.ndarray],
    motion: list[float],
    *,
    eps: float = STATIC_EPS,
    policy: str = DEFAULT_VIDEO_POLICY,
) -> tuple[list[np.ndarray], dict]:
    """Drop near-static frames from an OVERLONG System1 segment, then decimate to clip fps.

    Why this exists: System1's progress signal is noisy, so a segment can run far past the point
    where the arm stopped meaningfully moving (up to the step budget). Training clips are GT spans
    decimated 20->4 fps, median ~15 frames; a 400-step rollout would yield ~80 frames that are
    mostly a motionless arm. Feeding that to System2 is off-distribution in BOTH length and
    content (it would read as "nothing happened").

    ``motion[i]`` is the per-step commanded-motion norm from
    ``stop_criterion.action_eef_base_norm`` — the SAME quantity and ``eps`` the quiescence half of
    the stop rule uses, so "static" means exactly what it means there. Order of operations:

      1. keep frames whose motion >= eps (genuine movement),
      2. ALWAYS keep the first and last frame (the before/after the model reasons over),
      3. decimate the survivors 20->4 fps (every 5th) to match the training cadence,
      4. cap at the policy max (32) by uniform subsampling, keeping first+last.

    Returns (frames, stats) where stats records every stage so the GUI can show what was removed.
    """
    n_raw = len(frames)
    if n_raw == 0:
        return [], {"n_raw": 0, "n_moving": 0, "n_kept": 0, "n_final": 0, "eps": eps}
    if len(motion) < n_raw:  # pad (first step has no gripper delta -> caller may pass fewer)
        motion = list(motion) + [0.0] * (n_raw - len(motion))

    keep = [i for i in range(n_raw) if float(motion[i]) >= eps]
    keep_set = set(keep) | {0, n_raw - 1}
    kept = sorted(keep_set)

    stride = max(1, int(round(SIM_FPS / CLIP_FPS)))     # 20 -> 4 fps  => 5
    dec = [idx for k, idx in enumerate(kept) if k % stride == 0]
    for edge in (kept[0], kept[-1]):                    # never lose the endpoints to striding
        if edge not in dec:
            dec.append(edge)
    dec = sorted(set(dec))

    pol = get_video_policy(policy)
    if len(dec) > pol.max_frames:
        sel = np.linspace(0, len(dec) - 1, pol.max_frames).round().astype(int)
        dec = [dec[i] for i in sorted(set(sel.tolist()))]
    if len(dec) < 2:                                    # a clip needs >= frame_factor frames
        dec = sorted({0, n_raw - 1}) if n_raw >= 2 else [0, 0]

    stats = {
        "n_raw": n_raw,
        "n_moving": len(keep),
        "n_kept_after_static_drop": len(kept),
        "n_final": len(dec),
        "eps": eps,
        "stride": stride,
        "kept_indices": dec,
        "frac_static_dropped": round(1.0 - (len(kept) / n_raw), 4),
    }
    return [frames[i] for i in dec], stats


def downscale(frames: list[np.ndarray], hw: tuple[int, int] = DISPLAY_HW) -> list[np.ndarray]:
    """Nearest-neighbour decimate tiled frames to ``hw`` (for SAVED artifacts only).

    Cheap on purpose: 256x768 -> 128x384 is an exact 2x stride, so this is a view-slice, not a
    resample. Used for the GUI copies; the model always receives the full-resolution tile.
    """
    out = []
    for f in frames:
        a = np.asarray(f, dtype=np.uint8)
        sh = max(1, a.shape[0] // hw[0])
        sw = max(1, a.shape[1] // hw[1])
        out.append(a[::sh, ::sw])
    return out


def write_clip(frames: list[np.ndarray], path: Path, *, fps: float = CLIP_FPS, crf: int = CLIP_CRF) -> dict:
    """Encode tiled frames to x264 mp4 at the training clip settings.

    Uses imageio/ffmpeg when available and falls back to a raw ffmpeg pipe. Even dimensions are
    required by yuv420p; the 256x768 tile already satisfies that.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frames, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(f"expected (N,H,W,3) tiled frames, got {arr.shape}")
    try:
        import imageio.v2 as imageio

        # pixelformat= (not an output_params -pix_fmt) so imageio doesn't emit a duplicate flag.
        w = imageio.get_writer(
            str(path), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
            output_params=["-crf", str(crf)],
        )
        for f in arr:
            w.append_data(f)
        w.close()
    except Exception:
        n, h, wd = arr.shape[0], arr.shape[1], arr.shape[2]
        cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wd}x{h}",
               "-r", str(fps), "-i", "-", "-an", "-vcodec", "libx264", "-crf", str(crf),
               "-pix_fmt", "yuv420p", str(path)]
        p = subprocess.run(cmd, input=arr.tobytes(), capture_output=True, check=False)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {p.stderr[-500:]!r}") from None
        del n
    return {"path": str(path), "n_frames": int(arr.shape[0]), "fps": fps, "crf": crf,
            "shape": [int(arr.shape[1]), int(arr.shape[2])]}


def write_thumb(frame: np.ndarray, path: Path, *, height: int = 96) -> dict:
    """Write a DOWNSCALED preview of a tiled frame (for GUI contact sheets).

    The condensed-clip frames were previously stored as full 256x768 PNGs (~16 KB each, ~12 MB per
    episode) purely so the GUI could show a thumbnail strip — a ~30x duplication of the 408 KB of
    mp4 that already holds the identical frames. At 96px tall they cost ~2 KB each and render
    identically in the strip; the mp4 remains the full-resolution source of truth.
    """
    a = np.asarray(frame, dtype=np.uint8)
    h, w = a.shape[:2]
    if h > height:
        step = max(1, int(round(h / height)))          # cheap nearest-neighbour decimation
        a = a[::step, ::step]
    return write_image(a, path)


def write_image(frame: np.ndarray, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), np.asarray(frame, dtype=np.uint8))
    except Exception:
        from PIL import Image

        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(str(path))
    return {"path": str(path), "shape": list(np.asarray(frame).shape[:2])}


# ---------------------------------------------------------------------------
# vLLM client


class Sys2Client:
    """Minimal OpenAI-protocol client for the System2 vLLM server.

    Mirrors ``sys2/eval/eval_vllm.py``: greedy decoding (temperature 0, seed 0),
    ``enable_thinking=False``, and the per-request frame-count handling that avoids vLLM's
    double-sampling bug. Media is passed as a ``file://`` URI (the server is launched with
    ``--allowed-local-media-path``) or inlined as base64 when ``inline_media=True``, which is the
    safer default when the server may not share this filesystem.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8100, model: str = "system2-full",
                 *, max_tokens: int = 512, timeout: float = 300.0, retries: int = 3,
                 video_policy: str = DEFAULT_VIDEO_POLICY, inline_media: bool = True):
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.video_policy = video_policy
        self.inline_media = inline_media

    # -- media -> OpenAI content parts ------------------------------------
    def _media_url(self, path: Path, kind: str) -> str:
        if not self.inline_media:
            return Path(path).resolve().as_uri()
        b = Path(path).read_bytes()
        mime = "video/mp4" if kind == "video" else "image/png"
        return f"data:{mime};base64,{base64.b64encode(b).decode()}"

    def _content(self, text: str, *, image: Path | None = None, video: Path | None = None) -> list[dict]:
        """Split the SWIFT-style <image>/<video> placeholders into interleaved content parts."""
        parts: list[dict] = []
        for piece in re.split(r"(<image>|<video>)", text):
            if not piece:
                continue
            if piece == "<image>" and image is not None:
                parts.append({"type": "image_url", "image_url": {"url": self._media_url(image, "image")}})
            elif piece == "<video>" and video is not None:
                parts.append({"type": "video_url", "video_url": {"url": self._media_url(video, "video")}})
            elif piece not in ("<image>", "<video>"):
                parts.append({"type": "text", "text": piece})
        return parts

    # -- request ----------------------------------------------------------
    def chat(self, system: str, user_text: str, *, image: Path | None = None,
             video: Path | None = None, n_video_frames: int | None = None) -> dict:
        """One greedy completion. Returns {'text', 'latency_s', 'nframes', 'usage'}."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": self._content(user_text, image=image, video=video)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        nframes = None
        if video is not None and n_video_frames:
            # Reproduce the training-aligned frame count. vLLM has two independent samplers
            # (VideoMediaIO and the HF Qwen processor) — never enable both, or you get the
            # `index 32 is out of bounds ... size 32` double-sampling failure. Below the Qwen
            # processor's min_frames=4 there is a separate timestamp bug, so for that edge we
            # sample once in VideoMediaIO and disable HF sampling.
            nframes = smart_nframes(int(n_video_frames), CLIP_FPS, self.video_policy)
            if nframes < 4:
                payload["media_io_kwargs"] = {"video": {"num_frames": nframes, "frame_recovery": True}}
                payload["mm_processor_kwargs"] = {"do_sample_frames": False}
            else:
                # MUST be exactly {do_sample_frames, num_frames} -- passing fps/min_frames/
                # max_frames instead makes the Qwen processor build a timestamp list that
                # disagrees with the sampled frame count and vLLM 500s with
                # "timestamps and tokens_per_frame must have the same length".
                payload["media_io_kwargs"] = {"video": {"num_frames": -1, "frame_recovery": True}}
                payload["mm_processor_kwargs"] = {"do_sample_frames": True, "num_frames": nframes}

        body = json.dumps(payload).encode()
        last_err: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                req = urllib.request.Request(self.url, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                return {"text": d["choices"][0]["message"]["content"],
                        "latency_s": round(time.time() - t0, 2),
                        "nframes": nframes, "usage": d.get("usage", {})}
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.retries - 1:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"System2 request failed after {self.retries} tries: {last_err}")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url.replace("/v1/chat/completions", "/health"),
                                        timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    # -- high-level mode calls -------------------------------------------
    def plan_cold(self, goal: str, scene_png: Path) -> dict:
        r = self.chat(SYS_PLAN_COLD, user_plan_cold(goal), image=scene_png)
        out = parse_plan(r["text"])
        out.update(latency_s=r["latency_s"], usage=r["usage"])
        return out

    def exec_first(self, goal: str, plan: str, scene_png: Path) -> dict:
        r = self.chat(SYS_EXEC, user_exec_first(goal, plan), image=scene_png)
        out = parse_exec(r["text"])
        out.update(latency_s=r["latency_s"], usage=r["usage"], nframes=None)
        return out

    def exec_turn(self, goal: str, plan: str, clip_mp4: Path, n_frames: int,
                  task_status: str, gripper_status: str) -> dict:
        r = self.chat(SYS_EXEC, user_exec_turn(goal, plan, task_status, gripper_status),
                      video=clip_mp4, n_video_frames=n_frames)
        out = parse_exec(r["text"])
        out.update(latency_s=r["latency_s"], usage=r["usage"], nframes=r["nframes"])
        return out


def policy_source() -> str:
    """Which video-policy implementation is in use (for run provenance)."""
    return _POLICY_SOURCE


if __name__ == "__main__":  # tiny self-check, no server needed
    import argparse

    ap = argparse.ArgumentParser(description="Self-check the System2 prompt/parse/compaction logic.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--ping", action="store_true", help="also probe the server /health")
    a = ap.parse_args()

    print(f"video policy source: {policy_source()}  policy={DEFAULT_VIDEO_POLICY}")
    print("smart_nframes(11 frames @4fps) =", smart_nframes(11, CLIP_FPS, DEFAULT_VIDEO_POLICY))
    print("smart_nframes(80 frames @4fps) =", smart_nframes(80, CLIP_FPS, DEFAULT_VIDEO_POLICY))

    demo = ("<thought>I see the fridge.</thought>\n<judge>subgoal_incomplete</judge>\n"
            "<plan_update>\n- [~] M1: close the right fridge door\n</plan_update>\n"
            "<estimated_step>50</estimated_step>\n<subgoal>continue to reach</subgoal>\n"
            "<subgoal_detail>keep moving right</subgoal_detail>")
    p = parse_exec(demo)
    print("parsed judge/est/subgoal:", p["judge"], p["estimated_step"], repr(p["subgoal"]))

    base = "- [ ] M1: close the right fridge door\n- [ ] M2: close the left fridge door"
    merged = apply_plan_update(base, "- [~] M1: close the right fridge door\n  * [~] M1.1: reach")
    print("merged plan:\n" + merged)

    # compaction: 100 steps where only the first 20 actually move
    frames = [np.zeros((*TILE_HW, 3), np.uint8) for _ in range(100)]
    motion = [0.5] * 20 + [0.001] * 80
    _, st = compact_static_frames(frames, motion)
    print("compaction stats:", {k: st[k] for k in ("n_raw", "n_moving", "n_final", "frac_static_dropped")})

    if a.ping:
        c = Sys2Client(a.host, a.port)
        print("server health:", c.health())
