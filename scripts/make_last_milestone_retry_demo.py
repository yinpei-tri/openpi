#!/usr/bin/env python3
"""Render a presentation video for a final-milestone retry rollout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_W, OUT_H = 720, 1440
VIEW_SIZE = 620
VIEW_X, VIEW_Y = 50, 20
PANEL = (25, 675, 695, 1415)
FONT_REGULAR = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=face) <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def active_block(plan: str) -> tuple[str, list[tuple[str, str]]]:
    """Return the active milestone and its fine-step mark/text pairs."""
    lines = plan.splitlines()
    milestone = ""
    steps: list[tuple[str, str]] = []
    start = None
    for i, line in enumerate(lines):
        match = re.match(r"- \[([ x~])\] (M\d+):\s*(.*)", line)
        if match and match.group(1) == "~":
            milestone = f"{match.group(2)}: {match.group(3)}"
            start = i + 1
            break
    if start is None:
        # A task_finish update may collapse the last block. Use the last milestone.
        for line in reversed(lines):
            match = re.match(r"- \[([ x~])\] (M\d+):\s*(.*)", line)
            if match:
                milestone = f"{match.group(2)}: {match.group(3)}"
                break
        return milestone, steps
    for line in lines[start:]:
        if line.startswith("-"):
            break
        match = re.match(r"\s*\* \[([ x~])\] (M\d+\.\d+):\s*(.*)", line)
        if match:
            steps.append((match.group(1), f"{match.group(2)}: {match.group(3)}"))
    return milestone, steps


def cameras(frame_bgr: np.ndarray) -> tuple[Image.Image, Image.Image]:
    if frame_bgr.ndim != 3 or frame_bgr.shape[1] % 3:
        raise ValueError(f"expected a three-camera tile, got {frame_bgr.shape}")
    tile_w = frame_bgr.shape[1] // 3
    right = frame_bgr[:, tile_w:2 * tile_w]
    wrist = frame_bgr[:, 2 * tile_w:3 * tile_w]
    result = []
    for view in (right, wrist):
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (VIEW_SIZE, VIEW_SIZE), interpolation=cv2.INTER_LANCZOS4)
        result.append(Image.fromarray(rgb))
    return result[0], result[1]


def pill(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str,
    *, fill: tuple[int, int, int], text_fill: tuple[int, int, int] = (255, 255, 255),
) -> int:
    face = font(22, bold=True)
    x, y = xy
    w = int(draw.textlength(label, font=face)) + 40
    draw.rounded_rectangle((x, y, x + w, y + 44), radius=22, fill=fill)
    box = draw.textbbox((0, 0), label, font=face)
    draw.text((x + 20, y + 22 - (box[3] - box[1]) / 2 - box[1]), label, font=face, fill=text_fill)
    return w


def render(
    frame_bgr: np.ndarray,
    *,
    judge: str,
    milestone: str,
    steps: list[tuple[str, str]],
    subgoal: str,
    phase: str = "normal",
    success: bool = False,
) -> np.ndarray:
    right, _wrist = cameras(frame_bgr)
    canvas = Image.new("RGB", (OUT_W, OUT_H), (255, 255, 255))
    canvas.paste(right, (VIEW_X, VIEW_Y))
    draw = ImageDraw.Draw(canvas)

    border = (186, 191, 198)
    draw.rounded_rectangle(
        (VIEW_X - 3, VIEW_Y - 3, VIEW_X + VIEW_SIZE + 2, VIEW_Y + VIEW_SIZE + 2),
        radius=10, outline=border, width=3,
    )
    tag_face = font(22, bold=True)
    label = "RIGHT SHOULDER"
    tag_w = int(draw.textlength(label, font=tag_face)) + 30
    draw.rounded_rectangle((VIEW_X + 14, VIEW_Y + 14, VIEW_X + 14 + tag_w, VIEW_Y + 53), radius=10, fill=(255, 255, 255))
    draw.text((VIEW_X + 29, VIEW_Y + 20), label, font=tag_face, fill=(39, 45, 54))

    px1, py1, px2, py2 = PANEL
    retry_phase = phase in {"retry_reset", "retry_trigger", "retry"}
    panel_fill = (255, 251, 242) if phase == "false_finish" or retry_phase else (250, 251, 253)
    panel_outline = (234, 177, 38) if retry_phase else border
    draw.rounded_rectangle(PANEL, radius=24, fill=panel_fill, outline=panel_outline, width=4)

    title = "LAST-MILESTONE RETRY"
    draw.text((px1 + 28, py1 + 19), title, font=font(25, bold=True), fill=(54, 63, 74))

    judge_label = judge or "rule replay"
    if phase == "false_finish":
        judge_label = "task_finish"
        judge_color = (201, 52, 58)
    elif success:
        judge_label = "env_success"
        judge_color = (43, 143, 84)
    elif phase in {"retry_reset", "retry_trigger"}:
        judge_label = "task_finish  →  suppressed"
        judge_color = (226, 151, 25)
    else:
        judge_color = (74, 105, 158)
    draw.text((px1 + 28, py1 + 62), "S2 JUDGE", font=font(17, bold=True), fill=(103, 110, 120))
    pill(draw, (px1 + 28, py1 + 87), judge_label, fill=judge_color)

    if phase == "false_finish":
        callout = "S2 stops, but the environment is still ongoing."
        draw.text((px1 + 28, py1 + 146), callout, font=font(21, bold=True), fill=(177, 43, 48))
    elif phase == "retry_reset":
        callout = "Retry the last milestone instead of stopping"
        draw.text((px1 + 28, py1 + 146), callout, font=font(21, bold=True), fill=(178, 111, 7))

    milestone_y = py1 + (194 if phase in {"false_finish", "retry_reset"} else 151)
    draw.text((px1 + 28, milestone_y), "CURRENT MILESTONE", font=font(17, bold=True), fill=(103, 110, 120))
    milestone_face = font(25, bold=True)
    y = milestone_y + 29
    for line in wrap(draw, milestone, milestone_face, px2 - px1 - 56):
        draw.text((px1 + 28, y), line, font=milestone_face, fill=(26, 31, 38))
        y += 32

    y += 10
    draw.line((px1 + 28, y, px2 - 28, y), fill=(211, 214, 219), width=2)
    y += 16
    step_face = font(19)
    for mark, text in steps:
        if mark == "x":
            symbol, color, fill = "✓", (41, 134, 78), (232, 246, 237)
        elif mark == "~":
            symbol, color, fill = "▶", (177, 111, 7), (255, 238, 195)
        else:
            symbol, color, fill = "○", (121, 128, 138), (245, 246, 248)
        lines = wrap(draw, text, step_face, px2 - px1 - 104)
        h = max(42, len(lines) * 25 + 14)
        draw.rounded_rectangle((px1 + 28, y, px2 - 28, y + h), radius=11, fill=fill)
        draw.text((px1 + 41, y + 8), symbol, font=font(21, bold=True), fill=color)
        ty = y + 7
        for line in lines:
            draw.text((px1 + 74, ty), line, font=step_face, fill=(34, 39, 46))
            ty += 25
        y += h + 8

    sub_y = py2 - 175
    box_fill = (255, 221, 112) if retry_phase else (235, 240, 248)
    box_outline = (222, 157, 23) if retry_phase else (148, 162, 184)
    draw.rounded_rectangle((px1 + 28, sub_y, px2 - 28, py2 - 22), radius=16, fill=box_fill, outline=box_outline, width=3)
    label = "RETRY ACTION" if retry_phase else "ACTIVE SUBGOAL"
    draw.text((px1 + 45, sub_y + 13), label, font=font(16, bold=True), fill=(91, 88, 73))
    sg_face = font(30, bold=retry_phase)
    lines = wrap(draw, subgoal, sg_face, px2 - px1 - 90)
    line_h = 36
    total_h = len(lines) * line_h
    sy = sub_y + 48 + max(0, (68 - total_h) // 2)
    for line in lines[:3]:
        draw.text((px1 + 45, sy), line, font=sg_face, fill=(24, 28, 33))
        sy += line_h

    if success:
        draw.rounded_rectangle((px1 + 28, py1 + 87, px2 - 28, py1 + 133), radius=23, fill=(43, 143, 84))
        msg = "ENVIRONMENT CONFIRMS TASK SUCCESS"
        face = font(21, bold=True)
        tw = draw.textlength(msg, font=face)
        draw.text(((px1 + px2 - tw) / 2, py1 + 97), msg, font=face, fill=(255, 255, 255))

    return np.asarray(canvas, dtype=np.uint8)


def read_last_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last = frame
    cap.release()
    if last is None:
        raise RuntimeError(f"video has no frames: {path}")
    return last


def append_hold(writer: imageio.Writer, frame: np.ndarray, seconds: float, fps: float) -> int:
    count = round(seconds * fps)
    for _ in range(count):
        writer.append_data(frame)
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--pre-retry-stride", type=int, default=2)
    ap.add_argument("--retry-stride", type=int, default=2)
    ap.add_argument("--retry-frame-repeat", type=int, default=1)
    ap.add_argument("--subgoal-pause", type=float, default=0.0)
    args = ap.parse_args()

    episode = json.loads((args.episode_dir / "episode.json").read_text())
    if not episode.get("episode_success"):
        raise ValueError("episode must end in environment success")
    turns = {int(t["turn"]): t for t in episode["turns"]}
    retry_turns = [
        int(event["turn"]) for event in episode.get("rule_interventions", [])
        if any(i.get("rule") == "final_milestone_retry" for i in event.get("interventions", []))
    ]
    if len(retry_turns) != 1:
        raise ValueError(f"expected exactly one final-milestone retry, got {retry_turns}")
    retry_turn = retry_turns[0]
    retry_json = json.loads((args.episode_dir / f"turn{retry_turn:02d}" / "turn.json").read_text())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", quality=None,
        pixelformat="yuv420p", macro_block_size=2,
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    n_frames = 0
    last_frame: np.ndarray | None = None
    try:
        for turn in sorted(turns):
            info = turns[turn]
            plan = info.get("plan_after") or ""
            milestone, steps = active_block(plan)
            subgoal = str(info.get("subgoal") or "")
            judge = str(info.get("judge") or "rule replay")
            video = args.episode_dir / f"turn{turn:02d}" / "s1_rollout_raw.mp4"

            if turn == retry_turn:
                assert last_frame is not None
                completed_steps = [
                    ("x", "M4.1: lift and carry the small brown bowl to the large black bowl in the cabinet"),
                    ("x", "M4.2: lower the small brown bowl onto the large black bowl and release"),
                    ("x", "M4.3: retract the arm from the cabinet"),
                ]
                false_finish = render(
                    last_frame, judge="task_finish", milestone="M4: place the small brown bowl on top of the large black bowl in the cabinet",
                    steps=completed_steps, subgoal="stop execution", phase="false_finish",
                )
                n_frames += append_hold(writer, false_finish, 1.8, args.fps)
                reset_milestone, reset_steps = active_block(retry_json["plan_after"])
                retry_reset = render(
                    last_frame, judge="task_finish", milestone=reset_milestone, steps=reset_steps,
                    subgoal=subgoal, phase="retry_reset",
                )
                n_frames += append_hold(writer, retry_reset, 2.0, args.fps)

            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {video}")
            stride = args.retry_stride if turn >= retry_turn else args.pre_retry_stride
            i = 0
            while True:
                ok, raw = cap.read()
                if not ok:
                    break
                last_frame = raw
                if i % stride == 0:
                    phase = "retry_trigger" if turn == retry_turn else ("retry" if turn > retry_turn else "normal")
                    out = render(
                        raw, judge=judge, milestone=milestone, steps=steps,
                        subgoal=subgoal, phase=phase, success=False,
                    )
                    if turn > retry_turn and i == 0:
                        n_frames += append_hold(writer, out, args.subgoal_pause, args.fps)
                    repeat = args.retry_frame_repeat if turn >= retry_turn else 1
                    for _ in range(repeat):
                        writer.append_data(out)
                        n_frames += 1
                i += 1
            cap.release()

        assert last_frame is not None
        final_info = turns[max(turns)]
        milestone, steps = active_block(final_info["plan_after"])
        success = render(
            last_frame, judge="env_success", milestone=milestone, steps=steps,
            subgoal=str(final_info["subgoal"]), phase="retry", success=True,
        )
        n_frames += append_hold(writer, success, 1.4, args.fps)
    finally:
        writer.close()

    print(json.dumps({
        "output": str(args.output), "frames": n_frames, "fps": args.fps,
        "duration_s": round(n_frames / args.fps, 2), "size": [OUT_W, OUT_H],
        "retry_turn": retry_turn,
    }))


if __name__ == "__main__":
    main()
