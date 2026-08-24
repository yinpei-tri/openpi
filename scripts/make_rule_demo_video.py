#!/usr/bin/env python3
"""Create a presentation clip from a combined-eval episode.

The saved rollout video is a 3-camera horizontal tile in the order
left-shoulder | right-shoulder | wrist.  This renderer keeps the latter two,
places them side by side, and overlays only the active subgoal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SIDE_BY_SIDE = {
    "size": (1280, 720), "view_size": 590,
    "positions": ((30, 15), (660, 15)), "bar": (30, 625, 1250, 700),
}
STACKED = {
    "size": (720, 1280), "view_size": 570,
    "positions": ((75, 15), (75, 600)), "bar": (30, 1185, 690, 1265),
}
FONT_REGULAR = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise FileNotFoundError(f"missing presentation font: {path}")
    return ImageFont.truetype(str(path), size=size)


def _fit_font(
    draw: ImageDraw.ImageDraw, text: str, *, bold: bool, bar: tuple[int, int, int, int],
) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    for size in range(38, 21, -1):
        font = _font(path, size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= bar[2] - bar[0] - 80:
            return font
    return _font(path, 21)


def _render(
    frame_bgr: np.ndarray, subgoal: str, highlight: str, layout: str, theme: str,
) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[1] % 3:
        raise ValueError(f"expected a three-camera tile, got {frame_bgr.shape}")
    tile_w = frame_bgr.shape[1] // 3
    # OpenCV decodes BGR. Select scene_right (middle) and wrist (right), then convert to RGB.
    right = cv2.cvtColor(frame_bgr[:, tile_w:2 * tile_w], cv2.COLOR_BGR2RGB)
    wrist = cv2.cvtColor(frame_bgr[:, 2 * tile_w:3 * tile_w], cv2.COLOR_BGR2RGB)
    spec = STACKED if layout == "stacked" else SIDE_BY_SIDE
    out_w, out_h = spec["size"]
    view_size = spec["view_size"]
    positions = spec["positions"]
    bar = spec["bar"]
    right = cv2.resize(right, (view_size, view_size), interpolation=cv2.INTER_LANCZOS4)
    wrist = cv2.resize(wrist, (view_size, view_size), interpolation=cv2.INTER_LANCZOS4)

    white = theme == "white"
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255) if white else (14, 16, 20))
    canvas.paste(Image.fromarray(right), positions[0])
    canvas.paste(Image.fromarray(wrist), positions[1])
    draw = ImageDraw.Draw(canvas)
    # A restrained frame keeps the two views visually separated without adding labels or metadata.
    for x, y in positions:
        draw.rounded_rectangle(
            (x - 2, y - 2, x + view_size + 1, y + view_size + 1),
            radius=8, outline=(180, 184, 190) if white else (88, 94, 105), width=2,
        )

    is_highlight = subgoal.strip().lower().rstrip(".") == highlight.strip().lower().rstrip(".")
    fill = (255, 205, 55) if is_highlight else ((255, 255, 255) if white else (31, 35, 43))
    outline = (236, 174, 20) if is_highlight else ((180, 184, 190) if white else (76, 82, 94))
    text_fill = (18, 20, 24) if is_highlight or white else (246, 247, 249)
    draw.rounded_rectangle(bar, radius=18, fill=fill, outline=outline, width=3)
    font = _fit_font(draw, subgoal, bold=is_highlight, bar=bar)
    box = draw.textbbox((0, 0), subgoal, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    tx = (out_w - tw) // 2
    ty = bar[1] + ((bar[3] - bar[1] - th) // 2) - box[1]
    draw.text((tx, ty), subgoal, font=font, fill=text_fill)
    return np.asarray(canvas, dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--highlight-subgoal", required=True)
    ap.add_argument("--layout", choices=("side-by-side", "stacked"), default="side-by-side")
    ap.add_argument("--theme", choices=("dark", "white"), default="dark")
    ap.add_argument("--fps", type=float, default=20.0)
    args = ap.parse_args()

    episode = json.loads((args.episode_dir / "episode.json").read_text())
    turns = {int(t["turn"]): t for t in episode.get("turns", [])}
    if not episode.get("episode_success"):
        raise ValueError("the requested episode did not finish successfully")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.output), fps=args.fps, codec="libx264", quality=None,
        pixelformat="yuv420p", output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    n_frames = 0
    highlighted_frames = 0
    try:
        for turn_dir in sorted(args.episode_dir.glob("turn[0-9][0-9]")):
            turn = int(turn_dir.name.removeprefix("turn"))
            subgoal = str(turns.get(turn, {}).get("subgoal") or "").strip()
            if not subgoal:
                continue
            video = turn_dir / "s1_rollout_raw.mp4"
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {video}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.append_data(
                    _render(frame, subgoal, args.highlight_subgoal, args.layout, args.theme)
                )
                n_frames += 1
                if subgoal.lower().rstrip(".") == args.highlight_subgoal.lower().rstrip("."):
                    highlighted_frames += 1
            cap.release()
    finally:
        writer.close()

    if n_frames == 0:
        raise RuntimeError("no rollout frames were rendered")
    if highlighted_frames == 0:
        raise RuntimeError(f"highlight subgoal never appeared: {args.highlight_subgoal!r}")
    print(json.dumps({
        "output": str(args.output), "frames": n_frames, "fps": args.fps,
        "duration_s": round(n_frames / args.fps, 3),
        "layout": args.layout,
        "theme": args.theme,
        "highlighted_frames": highlighted_frames,
    }))


if __name__ == "__main__":
    main()
