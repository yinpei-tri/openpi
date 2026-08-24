#!/usr/bin/env python3
"""Build presentation videos for one-shot last-milestone-retry rescues.

The source is one successful episode from the retry-only evaluation. All turns before the single
``final_milestone_retry`` intervention are played at 6x; the triggering turn and the remainder of
the successful rollout are played at 4x. Right-shoulder and wrist views are stacked vertically, and
the exact language prompt executed by System1 is overlaid on the image.

The paired mandatory-only episode is validated as a failure, but its pixels are not spliced into
the movie: the before/after boundary therefore remains one continuous simulator trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


CAMERAS = ("scene_left", "scene_right", "wrist")
DEFAULT_EPISODES = (
    "TurnOnMicrowave:15",
    "StackBowlsCabinet:16",
    "BreadSelection:4",
)


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _retry_intervention(value) -> bool:
    if isinstance(value, dict):
        return value.get("rule") == "final_milestone_retry" or any(
            _retry_intervention(v) for v in value.values())
    if isinstance(value, list):
        return any(_retry_intervention(v) for v in value)
    return False


def _parse_episode(value: str) -> tuple[str, int]:
    try:
        task, episode = value.rsplit(":", 1)
        return task.strip(), int(episode)
    except Exception as exc:
        raise argparse.ArgumentTypeError("episode must be TASK:INDEX, e.g. TurnOnMicrowave:15") from exc


def _wrap_pixels(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return ["(no subgoal text recorded)"]
    lines: list[str] = []
    current = words.pop(0)
    for word in words:
        proposed = f"{current} {word}"
        if draw.textbbox((0, 0), proposed, font=font)[2] <= width:
            current = proposed
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class Renderer:
    def __init__(self, *, size: int, fps: int):
        self.size = int(size)
        self.fps = int(fps)
        self.canvas_hw = (self.size * 2, self.size)
        self.phase_font = _font(max(16, round(size * 0.037)), bold=True)
        self.subgoal_font = _font(max(18, round(size * 0.043)))
        self.label_font = _font(max(12, round(size * 0.025)), bold=True)
        self.success_font = _font(max(27, round(size * 0.066)), bold=True)

    def annotated(self, top: np.ndarray, bottom: np.ndarray, *, phase: str, subgoal: str,
                  retry: bool, success: bool = False) -> np.ndarray:
        canvas = Image.new("RGB", (self.size, self.size * 2))
        for y, frame in ((0, top), (self.size, bottom)):
            view = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
                (self.size, self.size), Image.Resampling.LANCZOS)
            canvas.paste(view, (0, y))

        # The only large annotation is the phase and current System1 language prompt.
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        lines = _wrap_pixels(draw, subgoal, self.subgoal_font, self.size - 28)[:2]
        banner_h = 42 + len(lines) * (self.subgoal_font.size + 3)
        draw.rectangle((0, 0, self.size, banner_h), fill=(0, 0, 0, 174))
        accent = (97, 255, 169) if retry else (116, 211, 255)
        draw.text((14, 9), phase, font=self.phase_font, fill=accent)
        y = 38
        for line in lines:
            draw.text((14, y), line, font=self.subgoal_font, fill="white")
            y += self.subgoal_font.size + 3
        draw.text((self.size - 125, self.size - 28), "RIGHT SHOULDER", font=self.label_font,
                  fill="white", stroke_width=2, stroke_fill="black")
        draw.text((self.size - 58, self.size * 2 - 28), "WRIST", font=self.label_font,
                  fill="white", stroke_width=2, stroke_fill="black")
        if success:
            text = "SUCCESS"
            box = draw.textbbox((0, 0), text, font=self.success_font, stroke_width=2)
            tw = box[2] - box[0]
            x, sy = (self.size - tw) // 2, self.size - self.success_font.size
            draw.rounded_rectangle((x - 18, sy - 12, x + tw + 18, sy + self.success_font.size + 14),
                                   radius=12, fill=(0, 75, 41, 220))
            draw.text((x, sy), text, font=self.success_font, fill=(112, 255, 184),
                      stroke_width=2, stroke_fill=(0, 40, 22))
        return np.asarray(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))

    def success_overlay(self, frame: np.ndarray) -> np.ndarray:
        canvas = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text = "SUCCESS"
        box = draw.textbbox((0, 0), text, font=self.success_font, stroke_width=2)
        tw = box[2] - box[0]
        x, y = (self.size - tw) // 2, self.size - self.success_font.size
        draw.rounded_rectangle((x - 18, y - 12, x + tw + 18, y + self.success_font.size + 14),
                               radius=12, fill=(0, 75, 41, 220))
        draw.text((x, y), text, font=self.success_font, fill=(112, 255, 184),
                  stroke_width=2, stroke_fill=(0, 40, 22))
        return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB"))


def _turns(episode_dir: Path, cameras: tuple[str, str]) -> tuple[list[dict], int]:
    turns: list[dict] = []
    retry_turns: list[int] = []
    for turn_dir in sorted(episode_dir.glob("turn[0-9][0-9]")):
        path = turn_dir / "turn.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        s1 = doc.get("s1") or {}
        raw = s1.get("video_raw") or {}
        retry = _retry_intervention((doc.get("rules") or {}).get("interventions") or [])
        if retry:
            retry_turns.append(int(doc["turn"]))
        if not raw:
            continue
        tiled = raw.get("layout") != "separate_views"
        videos: dict[str, Path] = {}
        for camera in cameras:
            if tiled:
                video_path = Path(raw.get("path") or turn_dir / "s1_rollout_raw.mp4")
            else:
                info = (raw.get("views") or {}).get(camera) or {}
                video_path = Path(info.get("path") or "")
            if not video_path.is_file():
                fallback = turn_dir / video_path.name
                if fallback.is_file():
                    video_path = fallback
                else:
                    raise FileNotFoundError(f"missing {camera} raw video for {turn_dir}: {video_path}")
            videos[camera] = video_path
        turns.append({
            "turn": int(doc["turn"]),
            "videos": videos,
            "tiled": tiled,
            "subgoal": s1.get("prompt_text") or (doc.get("s2") or {}).get("subgoal") or "",
            "n_steps": int(s1.get("n_steps") or 0),
        })
    retry_turns = sorted(set(retry_turns))
    if len(retry_turns) != 1:
        raise ValueError(f"{episode_dir.name}: expected exactly one retry turn, got {retry_turns}")
    return turns, retry_turns[0]


def _camera_frame(frame: np.ndarray, camera: str, *, tiled: bool) -> np.ndarray:
    a = np.asarray(frame, dtype=np.uint8)
    if not tiled:
        return a
    if a.ndim != 3 or a.shape[1] % 3:
        raise ValueError(f"expected a three-camera tile, got {a.shape}")
    index = CAMERAS.index(camera)
    width = a.shape[1] // 3
    return np.ascontiguousarray(a[:, index * width:(index + 1) * width])


def _append_video(writer, renderer: Renderer, turn: dict, *, cameras: tuple[str, str], speed: int,
                  retry: bool) -> tuple[int, np.ndarray | None]:
    top_camera, bottom_camera = cameras
    top_reader = imageio.get_reader(str(turn["videos"][top_camera]))
    same_video = turn["videos"][top_camera] == turn["videos"][bottom_camera]
    bottom_reader = None if same_video else imageio.get_reader(str(turn["videos"][bottom_camera]))
    count = 0
    last_pair = None
    last_i = -1
    last_written = -1
    last_output = None
    try:
        frames = ((frame, frame) for frame in top_reader) if same_video else zip(top_reader, bottom_reader)
        for i, (top_frame, bottom_frame) in enumerate(frames):
            last_pair, last_i = (top_frame, bottom_frame), i
            if i % speed:
                continue
            last_output = renderer.annotated(
                _camera_frame(top_frame, top_camera, tiled=turn["tiled"]),
                _camera_frame(bottom_frame, bottom_camera, tiled=turn["tiled"]),
                phase=(f"RETRY · {speed}× · T{turn['turn']}" if retry else
                       f"INITIAL · {speed}× · T{turn['turn']}"),
                subgoal=turn["subgoal"], retry=retry)
            writer.append_data(last_output)
            count += 1
            last_written = i
        if last_pair is not None and last_i != last_written:
            last_output = renderer.annotated(
                _camera_frame(last_pair[0], top_camera, tiled=turn["tiled"]),
                _camera_frame(last_pair[1], bottom_camera, tiled=turn["tiled"]),
                phase=(f"RETRY · {speed}× · T{turn['turn']}" if retry else
                       f"INITIAL · {speed}× · T{turn['turn']}"),
                subgoal=turn["subgoal"], retry=retry)
            writer.append_data(last_output)
            count += 1
    finally:
        top_reader.close()
        if bottom_reader is not None:
            bottom_reader.close()
    return count, last_output


def build_one(args, task: str, episode: int) -> dict:
    flat = f"{task}__target__episode_{episode:06d}"
    retry_dir = args.retry_root / flat
    base_dir = args.base_root / flat
    retry_doc = json.loads((retry_dir / "episode.json").read_text())
    base_doc = json.loads((base_dir / "episode.json").read_text())
    if not retry_doc.get("episode_success"):
        raise ValueError(f"{flat}: retry episode is not successful")
    if base_doc.get("episode_success"):
        raise ValueError(f"{flat}: paired mandatory-only episode already succeeded")
    cameras = (args.top_camera, args.bottom_camera)
    turns, retry_turn = _turns(retry_dir, cameras)
    if not turns or not any(t["turn"] >= retry_turn for t in turns):
        raise ValueError(f"{flat}: no executable retry video")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"last_milestone_retry_{task}_ep{episode:06d}_{args.top_camera}_{args.bottom_camera}.mp4")
    renderer = Renderer(size=args.size, fps=args.fps)
    writer = imageio.get_writer(
        str(output), fps=args.fps, codec="libx264", quality=None, pixelformat="yuv420p",
        output_params=["-crf", str(args.crf), "-movflags", "+faststart"],
    )
    counts = {"before_retry": 0, "retry": 0, "success_overlay": 0}
    last_frame = None
    try:
        for turn in turns:
            if turn["turn"] >= retry_turn:
                break
            count, last_frame = _append_video(
                writer, renderer, turn, cameras=cameras, speed=args.pre_speed, retry=False)
            counts["before_retry"] += count

        for turn in turns:
            if turn["turn"] < retry_turn:
                continue
            count, last_frame = _append_video(
                writer, renderer, turn, cameras=cameras, speed=args.retry_speed, retry=True)
            counts["retry"] += count

        if last_frame is not None:
            success = renderer.success_overlay(last_frame)
            for _ in range(round(args.success_seconds * args.fps)):
                writer.append_data(success)
                counts["success_overlay"] += 1
    finally:
        writer.close()

    return {
        "task": task,
        "episode": episode,
        "episode_id": retry_doc.get("episode_id"),
        "instruction": retry_doc.get("instruction"),
        "cameras": [args.top_camera, args.bottom_camera],
        "retry_turn": retry_turn,
        "base_termination": base_doc.get("termination"),
        "retry_termination": retry_doc.get("termination"),
        "pre_retry_speed": args.pre_speed,
        "retry_speed": args.retry_speed,
        "output_fps": args.fps,
        "output_shape": [renderer.canvas_hw[0], renderer.canvas_hw[1]],
        "frames": counts,
        "duration_seconds": round(sum(counts.values()) / args.fps, 2),
        "output": str(output),
        "bytes": output.stat().st_size,
    }


def build_argparser() -> argparse.ArgumentParser:
    data_dir = Path(os.environ.get("DATA_DIR") or "/home/ec2-user/data")
    combine = data_dir / "sys1_eval_results" / "combine"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-root", type=Path,
                        default=combine / "ablation-last-milestone-retry-only")
    parser.add_argument("--base-root", type=Path,
                        default=combine / "s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode", action="append", type=_parse_episode,
                        help="TASK:INDEX; repeat for multiple videos. Defaults to one per split.")
    parser.add_argument("--top-camera", choices=CAMERAS, default="scene_right")
    parser.add_argument("--bottom-camera", choices=CAMERAS, default="wrist")
    parser.add_argument("--pre-speed", type=int, default=6)
    parser.add_argument("--retry-speed", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--success-seconds", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    episodes = args.episode or [_parse_episode(x) for x in DEFAULT_EPISODES]
    if args.pre_speed < 1 or args.retry_speed < 1 or args.fps < 1:
        raise SystemExit("speeds and fps must be positive integers")
    records = []
    for task, episode in episodes:
        record = build_one(args, task, episode)
        records.append(record)
        print(f"WROTE {record['output']} ({record['duration_seconds']}s, {record['bytes']} bytes)",
              flush=True)
    manifest = {
        "schema": 1,
        "description": "One-shot last-milestone-retry rescue presentation videos",
        "retry_root": str(args.retry_root),
        "base_root": str(args.base_root),
        "videos": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
