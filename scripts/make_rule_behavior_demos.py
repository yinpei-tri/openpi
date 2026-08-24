#!/usr/bin/env python3
"""Create short portrait clips demonstrating regrasp and toaster-wait rules."""

from __future__ import annotations

import argparse
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
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=face) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def crop_camera(frame_bgr: np.ndarray, camera: int, size: int = VIEW_SIZE) -> Image.Image:
    if frame_bgr.ndim != 3 or frame_bgr.shape[1] % 3:
        raise ValueError(f"expected a three-camera tile, got {frame_bgr.shape}")
    tile_w = frame_bgr.shape[1] // 3
    view = frame_bgr[:, camera * tile_w:(camera + 1) * tile_w]
    view = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    view = cv2.resize(view, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(view)


def draw_pill(
    draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
    fill: tuple[int, int, int],
) -> None:
    face = font(21, bold=True)
    w = int(draw.textlength(text, font=face)) + 38
    draw.rounded_rectangle((x, y, x + w, y + 43), radius=21, fill=fill)
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x + 19, y + 21 - (box[3] - box[1]) / 2 - box[1]), text, font=face, fill="white")


def render(
    raw: np.ndarray,
    *,
    camera: int,
    camera_label: str,
    title: str,
    status: str,
    status_color: tuple[int, int, int],
    detail: str,
    sequence: list[str],
    active: int,
    subgoal: str,
    progress: float | None = None,
    secondary_camera: int | None = None,
    secondary_label: str = "",
) -> np.ndarray:
    canvas = Image.new("RGB", (OUT_W, OUT_H), "white")
    draw = ImageDraw.Draw(canvas)
    border = (185, 190, 198)
    if secondary_camera is None:
        views = [(crop_camera(raw, camera), VIEW_X, VIEW_Y, VIEW_SIZE, camera_label)]
        panel = PANEL
        label_size = 22
    else:
        dual_size = 325
        views = [
            (crop_camera(raw, camera, dual_size), 25, 20, dual_size, camera_label),
            (crop_camera(raw, secondary_camera, dual_size), 370, 20, dual_size, secondary_label),
        ]
        panel = (25, 380, 695, 1415)
        label_size = 16
    for view, vx, vy, size, label in views:
        canvas.paste(view, (vx, vy))
        draw.rounded_rectangle(
            (vx - 3, vy - 3, vx + size + 2, vy + size + 2),
            radius=10, outline=border, width=3,
        )
        label_face = font(label_size, bold=True)
        label_w = int(draw.textlength(label, font=label_face)) + 24
        tag_h = 33 if secondary_camera is not None else 39
        draw.rounded_rectangle(
            (vx + 10, vy + 10, vx + 10 + label_w, vy + 10 + tag_h),
            radius=9, fill="white",
        )
        draw.text((vx + 22, vy + 15), label, font=label_face, fill=(39, 45, 54))

    px1, py1, px2, py2 = panel
    warm = status not in {"TRANSFER ATTEMPT", "START TOASTER", "S2 SUBGOAL", "CARRY ATTEMPT"}
    fill = (255, 251, 241) if warm else (250, 251, 253)
    outline = (229, 172, 30) if warm else border
    draw.rounded_rectangle(PANEL, radius=24, fill=fill, outline=outline, width=4)
    draw.text((px1 + 28, py1 + 20), title, font=font(27, bold=True), fill=(47, 55, 66))
    draw_pill(draw, px1 + 28, py1 + 63, status, status_color)

    detail_face = font(20, bold=warm)
    y = py1 + 121
    for line in wrap(draw, detail, detail_face, px2 - px1 - 56)[:2]:
        draw.text((px1 + 28, y), line, font=detail_face, fill=(154, 74, 26) if warm else (76, 84, 95))
        y += 27

    y = py1 + 188
    draw.text((px1 + 28, y), "SUBGOAL LIST", font=font(17, bold=True), fill=(103, 110, 120))
    y += 29
    step_face = font(19)
    for index, text in enumerate(sequence):
        first_carry = next((i for i, item in enumerate(sequence) if "carry" in item.lower()), -1)
        failed_carry = title == "REGRASP RECOVERY" and index == first_carry and (
            active > first_carry or status == "NO GRASP DETECTED"
        )
        if failed_carry:
            symbol, color, row_fill = "!", (194, 54, 59), (253, 232, 233)
        elif index < active:
            symbol, color, row_fill = "✓", (41, 134, 78), (232, 246, 237)
        elif index == active:
            symbol, color, row_fill = "▶", (177, 111, 7), (255, 238, 195)
        else:
            symbol, color, row_fill = "○", (121, 128, 138), (245, 246, 248)
        lines = wrap(draw, text, step_face, px2 - px1 - 104)
        h = max(42, len(lines) * 25 + 14)
        draw.rounded_rectangle((px1 + 28, y, px2 - 28, y + h), radius=11, fill=row_fill)
        draw.text((px1 + 41, y + 8), symbol, font=font(21, bold=True), fill=color)
        ty = y + 7
        for line in lines[:2]:
            draw.text((px1 + 74, ty), line, font=step_face, fill=(34, 39, 46))
            ty += 25
        y += h + 8

    if progress is not None:
        bar_y = py2 - 222
        draw.text((px1 + 28, bar_y), "FORCED WAIT WINDOW", font=font(16, bold=True), fill=(103, 110, 120))
        draw.rounded_rectangle((px1 + 28, bar_y + 28, px2 - 28, bar_y + 48), radius=10, fill=(226, 228, 232))
        progress = min(1.0, max(0.0, progress))
        progress_x = px1 + 28 + int((px2 - px1 - 56) * progress)
        if progress_x > px1 + 28:
            draw.rounded_rectangle((px1 + 28, bar_y + 28, progress_x, bar_y + 48), radius=10, fill=(226, 151, 25))

    box_y = py2 - 158
    draw.rounded_rectangle(
        (px1 + 28, box_y, px2 - 28, py2 - 22), radius=16,
        fill=(255, 221, 112) if warm else (235, 240, 248),
        outline=(222, 157, 23) if warm else (148, 162, 184), width=3,
    )
    draw.text((px1 + 45, box_y + 13), "ACTIVE SUBGOAL", font=font(16, bold=True), fill=(91, 88, 73))
    sg_face = font(29, bold=warm)
    lines = wrap(draw, subgoal, sg_face, px2 - px1 - 90)
    line_h = 35
    sy = box_y + 48 + max(0, (68 - len(lines) * line_h) // 2)
    for line in lines[:3]:
        draw.text((px1 + 45, sy), line, font=sg_face, fill=(24, 28, 33))
        sy += line_h
    return np.asarray(canvas, dtype=np.uint8)


def video_writer(path: Path, fps: float) -> imageio.Writer:
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
        macro_block_size=2, output_params=["-crf", "18", "-movflags", "+faststart"],
    )


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def hold(writer: imageio.Writer, frame: np.ndarray, seconds: float, fps: float) -> int:
    count = round(seconds * fps)
    for _ in range(count):
        writer.append_data(frame)
    return count


def make_regrasp(episode: Path, output: Path, fps: float) -> dict[str, object]:
    sequence = [
        "reach to the kettle",
        "grasp the kettle",
        "carry the kettle to the tray",
        "reach and grasp the kettle again",
        "carry the kettle to the tray",
    ]
    configs = {
        0: ("S2 SUBGOAL", (75, 105, 158), "", 0, "reach to the kettle"),
        1: ("S2 SUBGOAL", (75, 105, 158), "", 1, "grasp the kettle"),
        2: ("CARRY ATTEMPT", (75, 105, 158), "", 2, "carry the kettle to the tray"),
        3: ("REGRASP INSERTED", (211, 126, 18), "Gripper closed on nothing; reacquire the kettle.", 3, "reach and grasp the kettle again"),
        4: ("HELD CARRY REPLAYED", (43, 143, 84), "", 4, "carry the kettle to the tray"),
    }
    writer = video_writer(output, fps)
    count = 0
    previous: np.ndarray | None = None
    try:
        for turn in (0, 1, 2, 3, 4):
            status, color, detail, active, subgoal = configs[turn]
            frames = read_frames(episode / f"turn{turn:02d}" / "s1_rollout_raw.mp4")
            if turn == 3 and previous is not None:
                trigger = render(
                    previous, camera=1, camera_label="RIGHT SHOULDER", title="REGRASP RECOVERY",
                    status="NO GRASP DETECTED", status_color=(194, 54, 59),
                    detail="The gripper is closed on nothing.",
                    sequence=sequence, active=2, subgoal="carry the kettle to the tray",
                )
                count += hold(writer, trigger, 1.8, fps)
            if turn == 4 and previous is not None:
                replay = render(
                    previous, camera=1, camera_label="RIGHT SHOULDER", title="REGRASP RECOVERY",
                    status=status, status_color=color, detail=detail,
                    sequence=sequence, active=active, subgoal=subgoal,
                )
                count += hold(writer, replay, 1.0, fps)
            for index, raw in enumerate(frames):
                previous = raw
                if index % 2:
                    continue
                writer.append_data(render(
                    raw, camera=1, camera_label="RIGHT SHOULDER", title="REGRASP RECOVERY",
                    status=status, status_color=color, detail=detail,
                    sequence=sequence, active=active, subgoal=subgoal,
                ))
                count += 1
    finally:
        writer.close()
    return {"output": str(output), "frames": count, "fps": fps, "duration_s": round(count / fps, 2)}


def make_regrasp_ep2_dual(episode: Path, output: Path, fps: float) -> dict[str, object]:
    sequence = [
        "reach and grasp the kettle",
        "lift and carry the kettle to the tray",
        "reach and grasp the kettle again",
        "lift and carry the kettle to the tray",
        "lower the kettle onto the tray and release",
    ]
    configs = {
        0: ("S2 SUBGOAL", (75, 105, 158), "", 0, sequence[0]),
        1: ("CARRY ATTEMPT", (75, 105, 158), "", 1, sequence[1]),
        2: ("REGRASP INSERTED", (211, 126, 18), "Gripper closed on nothing; reacquire the kettle.", 2, sequence[2]),
        3: ("HELD CARRY REPLAYED", (43, 143, 84), "", 3, sequence[3]),
        4: ("S2 PLANNING RESUMES", (43, 143, 84), "", 4, sequence[4]),
    }

    def frame_for(
        raw: np.ndarray, status: str, color: tuple[int, int, int], detail: str,
        active: int, subgoal: str,
    ) -> np.ndarray:
        return render(
            raw, camera=1, camera_label="RIGHT SHOULDER", secondary_camera=2,
            secondary_label="WRIST", title="REGRASP RECOVERY", status=status,
            status_color=color, detail=detail, sequence=sequence, active=active,
            subgoal=subgoal,
        )

    writer = video_writer(output, fps)
    count = 0
    previous: np.ndarray | None = None
    try:
        for turn in (0, 1, 2, 3, 4):
            status, color, detail, active, subgoal = configs[turn]
            frames = read_frames(episode / f"turn{turn:02d}" / "s1_rollout_raw.mp4")
            if turn == 2 and previous is not None:
                trigger = frame_for(
                    previous, "NO GRASP DETECTED", (194, 54, 59),
                    "The gripper is closed on nothing.", 1, sequence[1],
                )
                count += hold(writer, trigger, 1.8, fps)
            if turn == 3 and previous is not None:
                replay = frame_for(previous, status, color, detail, active, subgoal)
                count += hold(writer, replay, 1.0, fps)
            for index, raw in enumerate(frames):
                previous = raw
                if index % 2:
                    continue
                writer.append_data(frame_for(raw, status, color, detail, active, subgoal))
                count += 1
    finally:
        writer.close()
    return {"output": str(output), "frames": count, "fps": fps, "duration_s": round(count / fps, 2)}


def make_toaster(episode: Path, output: Path, fps: float) -> dict[str, object]:
    sequence = [
        "push the toaster lever down",
        "move over the toaster slot and wait",
        "continue to move over the toaster slot and wait",
        "reach and grasp the bread",
    ]
    configs = {
        1: ("START TOASTER", (75, 105, 158), "The toaster begins heating the bread.", 0, "push the toaster lever down", 2),
        2: ("FORCED WAIT • 950 STEPS", (211, 126, 18), "Do not stop just because the robot arm becomes still.", 1, "move over the toaster slot and wait", 5),
        3: ("WAIT CONTINUES", (211, 126, 18), "The bread remains in the slot; keep waiting.", 2, "continue to move over the toaster slot and wait", 10),
        4: ("BREAD READY", (43, 143, 84), "The waiting window has completed; retrieval can begin.", 3, "reach and grasp the bread", 2),
    }
    writer = video_writer(output, fps)
    count = 0
    try:
        trigger_frames = read_frames(episode / "turn02" / "s1_rollout_raw.mp4")
        trigger_raw = trigger_frames[min(100, len(trigger_frames) - 1)]
        trigger = render(
            trigger_raw, camera=2, camera_label="WRIST • TOASTER SLOT", title="TOASTER WAIT RULE",
            status="WAIT RULE TRIGGERED", status_color=(194, 54, 59),
            detail="The planned pickup has no safe wait; insert one now.",
            sequence=sequence, active=1, subgoal="move over the toaster slot and wait", progress=0,
        )
        count += hold(writer, trigger, 1.8, fps)
        for turn in (2, 3, 4):
            status, color, detail, active, subgoal, stride = configs[turn]
            frames = trigger_frames if turn == 2 else read_frames(episode / f"turn{turn:02d}" / "s1_rollout_raw.mp4")
            for index, raw in enumerate(frames):
                if index % stride:
                    continue
                progress = None
                if turn == 2:
                    progress = index / max(1, len(frames) - 1)
                elif turn == 3:
                    progress = 1.0
                writer.append_data(render(
                    raw, camera=2, camera_label="WRIST • TOASTER SLOT", title="TOASTER WAIT RULE",
                    status=status, status_color=color, detail=detail,
                    sequence=sequence, active=active, subgoal=subgoal, progress=progress,
                ))
                count += 1
    finally:
        writer.close()
    return {"output": str(output), "frames": count, "fps": fps, "duration_s": round(count / fps, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()
    regrasp_episode = Path("/home/ec2-user/data/sys1_eval_results/combine/ablation-regrasp-recovery/ArrangeTea__target__episode_000018")
    regrasp_ep2 = Path("/home/ec2-user/data/sys1_eval_results/combine/ablation-regrasp-recovery/ArrangeTea__target__episode_000002")
    toaster_episode = Path("/home/ec2-user/data/sys1_eval_results/combine/ablation-task-specific-bundle/GetToastedBread__target__episode_000027")
    results = [
        make_regrasp(regrasp_episode, args.output_dir / "regrasp_recovery_ArrangeTea_ep18.mp4", args.fps),
        make_regrasp_ep2_dual(regrasp_ep2, args.output_dir / "regrasp_recovery_ArrangeTea_ep2.mp4", args.fps),
        make_toaster(toaster_episode, args.output_dir / "gettoastedbread_wait_GetToastedBread_ep27.mp4", args.fps),
    ]
    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
