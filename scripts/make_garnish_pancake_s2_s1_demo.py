#!/usr/bin/env python3
"""Render the GarnishPancake ep0 System-2/System-1 planning demo."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 960
FPS = 20

BG = (247, 249, 252)
WHITE = (255, 255, 255)
NAVY = (19, 44, 75)
TEXT = (31, 45, 61)
MUTED = (105, 119, 135)
LINE = (202, 211, 221)
BLUE = (37, 99, 235)
LIGHT_BLUE = (232, 240, 255)
GREEN = (25, 146, 91)
LIGHT_GREEN = (230, 247, 238)
AMBER = (205, 123, 19)
LIGHT_AMBER = (255, 247, 224)
RED = (205, 61, 66)

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(path, size)


F_TITLE = font(30, bold=True)
F_SECTION = font(24, bold=True)
F_BODY = font(21)
F_BODY_B = font(21, bold=True)
F_SMALL = font(18)
F_SMALL_B = font(18, bold=True)
F_MONO = font(19, mono=True)
F_SUBGOAL = font(23, bold=True)
F_SUBGOAL_LARGE = font(26, bold=True)


def rr(draw: ImageDraw.ImageDraw, box, radius=14, fill=WHITE, outline=LINE, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text_width(draw: ImageDraw.ImageDraw, s: str, fnt) -> float:
    return draw.textlength(s, font=fnt)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    lines: list[str] = []
    for para in str(text).splitlines() or [""]:
        if not para:
            lines.append("")
            continue
        prefix = ""
        stripped = para
        m = re.match(r"^(\s*(?:[-*]|[✓▶○])\s+)", para)
        if m:
            prefix = m.group(1)
            stripped = para[len(prefix):]
        words = stripped.split()
        current = prefix
        continuation = " " * max(0, len(prefix))
        for word in words:
            trial = (current + " " + word).strip() if not current.isspace() else current + word
            if text_width(draw, trial, fnt) <= width or current.strip() == "":
                current = trial
            else:
                lines.append(current.rstrip())
                current = continuation + word
        if current.strip():
            lines.append(current.rstrip())
    return lines


def fit_crop(tile: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w = tile.shape[:2]
    scale = max(out_w / w, out_h / h)
    resized = cv2.resize(tile, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    y = max(0, (resized.shape[0] - out_h) // 2)
    x = max(0, (resized.shape[1] - out_w) // 2)
    return resized[y:y + out_h, x:x + out_w]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def progress_series(data: dict, n_steps: int) -> list[float]:
    """Reconstruct the online S1 curve from every predicted progress chunk."""
    values: list[float | None] = [None] * n_steps
    for step in data.get("steps", []):
        raw = step.get("progress_raw") or {}
        chunk = raw.get("progress_chunk") or []
        if not chunk:
            continue
        start = int(raw.get("at_step", step.get("frame_step", 0)))
        for offset, value in enumerate(chunk):
            idx = start + offset
            if 0 <= idx < n_steps:
                try:
                    values[idx] = min(1.0, max(0.0, float(value)))
                except (TypeError, ValueError):
                    pass
    # Fill any tail not covered by the last prediction, preserving the latest value.
    last = 0.0
    for i, value in enumerate(values):
        if value is None:
            values[i] = last
        else:
            last = value
    return [float(v) for v in values]


def compact_response(s2: dict) -> str:
    thought = (s2.get("thought") or "").strip()
    judge = (s2.get("judge") or "").strip()
    update = (s2.get("plan_update") or "").strip()
    estimate = s2.get("estimated_step")
    parts = ["THOUGHT", thought, "", f"JUDGE  {judge}"]
    if update:
        parts += ["", "PLAN UPDATE", update]
    if estimate is not None:
        parts += ["", f"ESTIMATED STEPS  {estimate}"]
    subgoal = (s2.get("subgoal") or "").strip()
    if subgoal:
        parts += ["", "SUBGOAL", subgoal]
    return "\n".join(parts)


def compact_plan_response(plan_meta: dict) -> str:
    return "\n".join([
        "THOUGHT",
        (plan_meta.get("thought") or "").strip(),
        "",
        "MILESTONE PLAN",
        (plan_meta.get("plan") or "").strip(),
    ])


def format_raw_response(raw: str) -> str:
    """Keep the recorded tagged response, but lay each field out as a block."""
    raw = re.sub(
        r"\s*<subgoal_detail>.*?</subgoal_detail>", "", raw or "",
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    blocks: list[str] = []
    pattern = re.compile(r"<([a-zA-Z0-9_]+)>(.*?)</\1>", re.DOTALL)
    for match in pattern.finditer(raw):
        tag, body = match.group(1), match.group(2).strip()
        blocks.append(f"<{tag}>")
        for line in body.splitlines() or [""]:
            blocks.append("  " + line.rstrip())
        blocks.append(f"</{tag}>")
    return "\n".join(blocks) if blocks else raw


def plan_rows(plan: str) -> list[tuple[str, str]]:
    """Show all milestones, but expand only the active milestone."""
    rows: list[tuple[str, str]] = []
    active_milestone = None
    for line in plan.splitlines():
        s = line.strip()
        m = re.match(r"- \[([x~ ])\]\s+(M\d+):\s*(.*)", s)
        if m:
            state, mid, body = m.groups()
            if state == "~":
                active_milestone = mid
            rows.append((state, f"{mid}  {body}"))
            continue
        m = re.match(r"\* \[([x~ ])\]\s+(M\d+\.\d+):\s*(.*)", s)
        if m and active_milestone and m.group(2).startswith(active_milestone + "."):
            state, sid, body = m.groups()
            rows.append(("sub_" + state, f"{sid}  {body}"))
    return rows


def draw_badge(draw, xy, label, fill, fg, fnt=F_SMALL_B):
    x, y = xy
    tw = text_width(draw, label, fnt)
    draw.rounded_rectangle((x, y, x + tw + 24, y + 34), radius=17, fill=fill)
    draw.text((x + 12, y + 5), label, font=fnt, fill=fg)
    return x + tw + 34


def draw_task_header(draw: ImageDraw.ImageDraw, task_name: str, task_goal: str):
    rr(draw, (25, 15, 1895, 90), outline=NAVY, width=3)
    draw.text((48, 30), "TASK", font=F_SMALL_B, fill=BLUE)
    draw.text((108, 24), task_name, font=F_TITLE, fill=NAVY)
    draw.line((420, 28, 420, 77), fill=LINE, width=2)
    draw.text((446, 30), "TASK GOAL", font=F_SMALL_B, fill=BLUE)
    goal_lines = wrap_text(draw, task_goal, F_BODY_B, 1290)
    for j, line in enumerate(goal_lines[:2]):
        draw.text((565, 25 + 25 * j), line, font=F_BODY_B, fill=TEXT)


def draw_plan(draw: ImageDraw.ImageDraw, plan: str, reveal: float = 1.0):
    x0, y0, x1, y1 = 25, 105, 730, 365
    rr(draw, (x0, y0, x1, y1), outline=NAVY, width=3)
    draw.text((48, 125), "MILESTONE PLAN", font=F_SECTION, fill=NAVY)
    draw.line((48, 160, 706, 160), fill=LINE, width=2)

    rows = plan_rows(plan)
    keep = int(math.ceil(len(rows) * max(0.0, min(1.0, reveal))))
    y = 171
    for state, body in rows[:keep]:
        sub = state.startswith("sub_")
        base_state = state[-1] if sub else state
        if base_state == "x":
            symbol, color = "✓", GREEN
        elif base_state == "~":
            symbol, color = "▶", AMBER
        else:
            symbol, color = "○", MUTED
        indent = 34 if sub else 0
        fnt = F_SMALL if sub else F_SMALL_B
        maxw = 600 - indent
        lines = wrap_text(draw, body, fnt, maxw)
        row_h = max(22, 20 * len(lines) + 2)
        if y + row_h > y1 - 14:
            draw.text((660, y1 - 35), "…", font=F_BODY_B, fill=MUTED)
            break
        if base_state == "~":
            draw.rounded_rectangle((47 + indent, y - 3, 705, y + row_h - 2), radius=8, fill=LIGHT_AMBER)
        draw.text((55 + indent, y), symbol, font=fnt, fill=color)
        tx = 82 + indent
        for j, line in enumerate(lines):
            draw.text((tx, y + 20 * j), line, font=fnt, fill=TEXT if base_state != " " else MUTED)
        y += row_h


def visible_response_lines(draw, content: str, reveal: float, max_lines: int = 12, width: int = 650):
    n = max(0, min(len(content), int(math.ceil(len(content) * reveal))))
    shown = content[:n]
    lines: list[str] = []
    for raw in shown.splitlines():
        lines.extend(wrap_text(draw, raw, F_MONO, width))
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


def visible_raw_lines(draw, content: str, reveal: float, max_lines: int = 23, width: int = 650):
    n = max(0, min(len(content), int(math.ceil(len(content) * reveal))))
    shown = content[:n]
    lines: list[tuple[str, bool]] = []
    for raw in shown.splitlines():
        leading = len(raw) - len(raw.lstrip(" "))
        body = raw.strip()
        is_tag = body.startswith("<")
        indent = " " * min(4, leading)
        available = width - int(text_width(draw, indent, F_MONO))
        wrapped = wrap_text(draw, body, F_MONO, available) if body else [""]
        for line in wrapped:
            lines.append((indent + line, is_tag))
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


def draw_s2(draw: ImageDraw.ImageDraw, content: str, reveal: float, streaming: bool):
    x0, y0, x1, y1 = 25, 385, 730, 935
    rr(draw, (x0, y0, x1, y1), outline=NAVY, width=3)
    draw.text((48, 405), "S2 RESPONSE", font=F_SECTION, fill=NAVY)
    state = "● STREAMING" if streaming else "● COMPLETE"
    draw.text((535, 411), state, font=F_SMALL_B, fill=BLUE if streaming else GREEN)
    draw.line((48, 446, 706, 446), fill=LINE, width=2)

    lines = visible_raw_lines(draw, content, reveal, max_lines=25, width=650)
    y = 457
    for line, is_tag in lines:
        draw.text((49, y), line, font=F_MONO, fill=BLUE if is_tag else TEXT)
        y += 19


def draw_camera(
    draw, canvas: Image.Image, frame_bgr: np.ndarray, mode: str,
    turn_idx: int = 0, step: int = 0, n_steps: int = 0, subgoal: str = "",
    phase: int = 1, phase_value: float = 0.0,
):
    # Native frame is left shoulder | right shoulder | wrist.
    left = frame_bgr[:, 0:256]
    right = frame_bgr[:, 256:512]
    wrist = frame_bgr[:, 512:768]
    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
    left = cv2.resize(left, (300, 300), interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(right, (300, 300), interpolation=cv2.INTER_CUBIC)
    wrist = cv2.resize(wrist, (300, 300), interpolation=cv2.INTER_CUBIC)
    # Keep both camera tiles at their native 256 x 256 resolution.
    rr(draw, (755, 105, 1895, 590), outline=NAVY, width=3)
    draw.text((780, 125), "S1 EXECUTION VIDEO", font=F_SECTION, fill=NAVY)
    mode_label = "MODE · PLAN" if mode == "plan" else "MODE · EXECUTION"
    mode_fill, mode_fg = (LIGHT_BLUE, BLUE) if mode == "plan" else (LIGHT_GREEN, GREEN)
    draw_badge(draw, (1660, 123), mode_label, mode_fill, mode_fg)
    draw.line((780, 166, 1870, 166), fill=LINE, width=2)
    if subgoal:
        prefix = "Subgoal:"
        draw.text((780, 176), prefix, font=F_SUBGOAL_LARGE, fill=BLUE)
        px = 780 + text_width(draw, prefix + " ", F_SUBGOAL_LARGE)
        subgoal_lines = wrap_text(draw, subgoal, F_SUBGOAL_LARGE, 1855 - px)
        for j, line in enumerate(subgoal_lines[:2]):
            draw.text((px if j == 0 else 780, 176 + 31 * j), line, font=F_SUBGOAL_LARGE, fill=NAVY)
    else:
        draw.text((780, 176), "S1 idle, waiting for S2 subgoal", font=F_SMALL, fill=MUTED)

    camera_x = [855, 1175, 1495]
    camera_names = ["LEFT SHOULDER", "RIGHT SHOULDER", "WRIST"]
    camera_frames = [left, right, wrist]
    for x, name, image_rgb in zip(camera_x, camera_names, camera_frames):
        tw = text_width(draw, name, F_SMALL_B)
        canvas.paste(Image.fromarray(image_rgb), (x, 240))
        draw.rectangle((x - 1, 239, x + 301, 541), outline=NAVY, width=3)
        draw.text((x + (300 - tw) / 2, 548), name, font=F_SMALL_B, fill=NAVY)


def draw_progress(draw: ImageDraw.ImageDraw, series: list[float], upto: int, mode: str):
    x0, y0, x1, y1 = 755, 610, 1895, 935
    rr(draw, (x0, y0, x1, y1), outline=NAVY, width=3)
    draw.text((780, 630), "S1 EXECUTION PROGRESS", font=F_SECTION, fill=NAVY)
    draw.line((780, 670, 1870, 670), fill=LINE, width=2)

    gx0, gy0, gx1, gy1 = 840, 710, 1850, 880
    for tick in range(5):
        value = tick / 4
        y = round(gy1 - value * (gy1 - gy0))
        draw.line((gx0, y, gx1, y), fill=(224, 230, 237), width=2)
        draw.text((785, y - 10), f"{value:.2f}", font=F_SMALL, fill=MUTED)
    threshold_y = round(gy1 - 0.95 * (gy1 - gy0))
    for x in range(gx0, gx1, 18):
        draw.line((x, threshold_y, min(x + 9, gx1), threshold_y), fill=GREEN, width=2)
    draw.line((gx0, gy0, gx0, gy1), fill=NAVY, width=2)
    draw.line((gx0, gy1, gx1, gy1), fill=NAVY, width=2)

    if mode == "plan":
        msg = "Waiting for the milestone plan before S1 begins execution"
        tw = text_width(draw, msg, F_BODY_B)
        draw.text(((gx0 + gx1 - tw) / 2, 785), msg, font=F_BODY_B, fill=MUTED)

    n = max(1, len(series))
    upto = min(max(0, upto), len(series) - 1) if series else 0
    points = []
    if series:
        for i in range(upto + 1):
            x = gx0 + round((gx1 - gx0) * i / max(1, n - 1))
            y = gy1 - round((gy1 - gy0) * series[i])
            points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=BLUE, width=5, joint="curve")
    if points:
        x, y = points[-1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLUE, outline=WHITE, width=2)
        value = series[upto]
    else:
        value = 0.0
    if mode != "plan" and series:
        step_label = f"STEP {upto + 1:03d}"
        sw = text_width(draw, step_label, F_BODY_B) + 28
        value_label = f"PROGRESS {value:.2f}"
        vw = text_width(draw, value_label, F_BODY_B) + 28
        draw.rounded_rectangle((gx1 - sw - vw - 12, 628, gx1 - sw - 12, 664), radius=18, fill=LIGHT_BLUE)
        draw.text((gx1 - sw - vw + 2, 633), value_label, font=F_BODY_B, fill=BLUE)
        draw.rounded_rectangle((gx1 - sw, 628, gx1, 664), radius=18, fill=LIGHT_GREEN)
        draw.text((gx1 - sw + 14, 633), step_label, font=F_BODY_B, fill=GREEN)
        for tick in range(5):
            idx = round((n - 1) * tick / 4)
            x = gx0 + round((gx1 - gx0) * tick / 4)
            label = str(idx)
            draw.text((x - text_width(draw, label, F_SMALL) / 2, gy1 + 10), label, font=F_SMALL, fill=MUTED)
        draw.text((1170, 906), "execution step within current subgoal", font=F_SMALL, fill=MUTED)


def render_frame(
    frame_bgr, plan, response_text, reveal, streaming, mode, progress=None,
    upto=0, turn_idx=0, step=0, n_steps=0, subgoal="", plan_reveal=1.0,
    success=False, failure=False, phase=1, phase_value=0.0, task_name="", task_goal="",
):
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_task_header(draw, task_name, task_goal)
    draw_plan(draw, plan, plan_reveal)
    draw_s2(draw, response_text, reveal, streaming)
    draw_camera(
        draw, canvas, frame_bgr, mode, turn_idx, step, n_steps, subgoal,
        phase, phase_value,
    )
    draw_progress(draw, progress or [], upto, mode)
    if success:
        draw.rounded_rectangle((1565, 174, 1855, 212), radius=19, fill=GREEN, outline=WHITE, width=3)
        draw.text((1584, 181), "✓  ENVIRONMENT SUCCESS", font=F_SMALL_B, fill=WHITE)
    elif failure:
        draw.rounded_rectangle((1565, 174, 1855, 212), radius=19, fill=RED, outline=WHITE, width=3)
        draw.text((1632, 181), "×  TASK FAILURE", font=F_SMALL_B, fill=WHITE)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-speed", type=float, default=1.5)
    parser.add_argument("--stream-seconds", type=float, default=1.6)
    parser.add_argument("--milestone-update-delay", type=float, default=1.0)
    parser.add_argument("--pre-execution-hold", type=float, default=3.0)
    parser.add_argument("--post-execution-hold", type=float, default=2.0)
    parser.add_argument("--initial-plan-seconds", type=float, default=2.8)
    args = parser.parse_args()

    turn_dirs = sorted(p for p in args.episode.glob("turn[0-9][0-9]") if (p / "turn.json").exists())
    if not turn_dirs:
        raise SystemExit(f"No turns found in {args.episode}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output: {args.output}")

    last_render = None
    try:
        episode_meta = load_json(args.episode / "episode.json")
        render_context = {
            "task_name": episode_meta.get("task_name") or "GarnishPancake",
            "task_goal": episode_meta.get("instruction") or "",
        }
        episode_success = bool(episode_meta.get("episode_success"))
        # The one and only plan-mode phase: S2 observes the initial scene and
        # produces milestone structure. The page layout stays fixed afterwards.
        plan_meta = load_json(args.episode / "plan" / "plan.json")
        initial_frame = cv2.imread(str(args.episode / "plan" / "scene_full.png"))
        if initial_frame is None:
            raise RuntimeError("Cannot read initial planning scene")
        milestone_plan = plan_meta.get("plan") or ""
        initial_response = format_raw_response(plan_meta.get("s2_response_raw") or "")
        initial_frames = max(1, round(args.initial_plan_seconds * FPS))
        for i in range(initial_frames):
            reveal = (i + 1) / initial_frames
            last_render = render_frame(
                initial_frame, "", initial_response, reveal, True,
                "plan", plan_reveal=0.0, phase=1, phase_value=reveal,
                **render_context,
            )
            writer.write(last_render)
        update_delay_frames = max(1, round(args.milestone_update_delay * FPS))
        for _ in range(update_delay_frames):
            last_render = render_frame(
                initial_frame, "", initial_response, 1.0, False,
                "plan", plan_reveal=0.0, phase=2, **render_context,
            )
            writer.write(last_render)
        initial_hold_frames = max(1, round(args.pre_execution_hold * FPS))
        for _ in range(initial_hold_frames):
            last_render = render_frame(
                initial_frame, milestone_plan, initial_response, 1.0, False,
                "plan", plan_reveal=1.0, phase=2, **render_context,
            )
            writer.write(last_render)
        previous_plan = milestone_plan
        last_scene_frame = initial_frame

        for turn_idx, turn_dir in enumerate(turn_dirs):
            meta = load_json(turn_dir / "turn.json")
            s2 = meta["s2"]
            plan = meta.get("plan_after") or previous_plan
            has_s1_execution = (
                (turn_dir / "s1_steps.json").exists()
                and (turn_dir / "s1_rollout_raw.mp4").exists()
            )
            if not has_s1_execution:
                response_text = format_raw_response(s2.get("response_raw") or "")
                stream_frames = max(1, round(args.stream_seconds * FPS))
                for i in range(stream_frames):
                    reveal = (i + 1) / stream_frames
                    last_render = render_frame(
                        last_scene_frame, previous_plan, response_text, reveal, True,
                        "execution", turn_idx=turn_idx, subgoal="", phase=1,
                        phase_value=reveal, **render_context,
                    )
                    writer.write(last_render)
                update_delay_frames = max(1, round(args.milestone_update_delay * FPS))
                for _ in range(update_delay_frames):
                    last_render = render_frame(
                        last_scene_frame, previous_plan, response_text, 1.0, False,
                        "execution", turn_idx=turn_idx, subgoal="", phase=2,
                        **render_context,
                    )
                    writer.write(last_render)
                hold_frames = max(1, round(args.pre_execution_hold * FPS))
                for _ in range(hold_frames):
                    last_render = render_frame(
                        last_scene_frame, plan, response_text, 1.0, False,
                        "execution", turn_idx=turn_idx, subgoal="", phase=2,
                        **render_context,
                    )
                    writer.write(last_render)
                is_terminal = turn_idx == len(turn_dirs) - 1
                last_render = render_frame(
                    last_scene_frame, plan, response_text, 1.0, False,
                    "execution", turn_idx=turn_idx, subgoal="", phase=2,
                    success=episode_success and is_terminal,
                    failure=(not episode_success) and is_terminal,
                    **render_context,
                )
                writer.write(last_render)
                previous_plan = plan
                continue
            steps_data = load_json(turn_dir / "s1_steps.json")
            cap = cv2.VideoCapture(str(turn_dir / "s1_rollout_raw.mp4"))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, first = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {turn_dir / 's1_rollout_raw.mp4'}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            progress = progress_series(steps_data, n_frames)

            stream_frames = max(1, round(args.stream_seconds * FPS))
            response_text = format_raw_response(s2.get("response_raw") or "")
            for i in range(stream_frames):
                reveal = (i + 1) / stream_frames
                last_render = render_frame(
                    first, previous_plan, response_text, reveal, True, "execution",
                    turn_idx=turn_idx, n_steps=n_frames, subgoal="",
                    phase=1, phase_value=reveal, **render_context,
                )
                writer.write(last_render)
            update_delay_frames = max(1, round(args.milestone_update_delay * FPS))
            for _ in range(update_delay_frames):
                last_render = render_frame(
                    first, previous_plan, response_text, 1.0, False, "execution",
                    turn_idx=turn_idx, n_steps=n_frames, subgoal="", phase=2,
                    **render_context,
                )
                writer.write(last_render)
            hold_frames = max(1, round(args.pre_execution_hold * FPS))
            for _ in range(hold_frames):
                last_render = render_frame(
                    first, plan, response_text, 1.0, False, "execution",
                    turn_idx=turn_idx, n_steps=n_frames, subgoal="",
                    phase=2, **render_context,
                )
                writer.write(last_render)

            source_idx = 0
            output_idx = 0
            last_frame = first
            last_emitted_idx = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                last_frame = frame
                target_idx = int(output_idx * args.execution_speed)
                if source_idx == target_idx:
                    terminal_frame = turn_idx == len(turn_dirs) - 1 and source_idx >= n_frames - 2
                    success = episode_success and terminal_frame
                    failure = (not episode_success) and terminal_frame
                    last_render = render_frame(
                        frame, plan, response_text, 1.0, False, "execution",
                        progress=progress, upto=source_idx, turn_idx=turn_idx,
                        step=source_idx, n_steps=n_frames,
                        subgoal=s2.get("subgoal") or "", success=success,
                        failure=failure, phase=3, **render_context,
                    )
                    writer.write(last_render)
                    last_emitted_idx = source_idx
                    output_idx += 1
                source_idx += 1
            if n_frames and last_emitted_idx != n_frames - 1:
                terminal_frame = turn_idx == len(turn_dirs) - 1
                success = episode_success and terminal_frame
                failure = (not episode_success) and terminal_frame
                last_render = render_frame(
                    last_frame, plan, response_text, 1.0, False, "execution",
                    progress=progress, upto=n_frames - 1, turn_idx=turn_idx,
                    step=n_frames - 1, n_steps=n_frames,
                    subgoal=s2.get("subgoal") or "", success=success,
                    failure=failure, phase=3,
                    **render_context,
                )
                writer.write(last_render)
            cap.release()
            last_scene_frame = last_frame
            if turn_idx < len(turn_dirs) - 1:
                wait_frames = max(1, round(args.post_execution_hold * FPS))
                for _ in range(wait_frames):
                    last_render = render_frame(
                        last_frame, plan, response_text, 1.0, False, "execution",
                        progress=progress, upto=n_frames - 1, turn_idx=turn_idx,
                        step=n_frames - 1, n_steps=n_frames, subgoal="", phase=1,
                        **render_context,
                    )
                    writer.write(last_render)
            previous_plan = plan

        if last_render is not None:
            for _ in range(round(1.8 * FPS)):
                writer.write(last_render)
    finally:
        writer.release()

    print(args.output)


if __name__ == "__main__":
    main()
