#!/usr/bin/env python3
"""Compose unit clips into a wide banner whose columns scroll right to left.

WHAT IT MAKES. A long-and-short frame (default 3840x800) holding a conveyor of COLUMNS. Each column
is one unit clip -- a milestone panel plus its right-shoulder view, as produced by
``make_plan_rollout_video.py`` -- with a visible gap between columns so neighbouring panels never
read as one block. The belt holds still for a moment so the first clips can be watched in place, then
slides left at a constant speed while fresh columns keep entering from the right.

HOW IT STAYS CHEAP. A slot's clip is opened only when that column first becomes visible and closed
when it leaves the left edge, so decode cost tracks what is ON SCREEN (a handful of clips) rather
than the size of the pool. Two useful consequences:

  * a column starts from its clip's FIRST frame at the moment it appears, so every unit that scrolls
    in begins at the beginning rather than joining a loop mid-motion;
  * a pool smaller than the number of columns simply cycles, and the same episode appearing twice
    gets independent playback because each slot owns its own capture.

The unit aspect is measured from the first clip rather than assumed, so re-rendering units at another
size needs no change here.

Usage:
    python scripts/make_gallery_montage.py --units-dir ~/data/video_materials/gallery \\
        --output ~/data/video_materials/gallery/montage_banner.mp4

    # taller two-row belt, slower
    python scripts/make_gallery_montage.py --units-dir <dir> --output out.mp4 \\
        --height 1400 --rows 2 --scroll-speed 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import cv2
import imageio.v2 as imageio
import numpy as np

# Must match the unit renderer's theme, or a light strip sits in a black surround.
BACKDROPS = {"dark": {"bg": (14, 16, 20), "edge": (72, 80, 92)},
             "light": {"bg": (255, 255, 255), "edge": (206, 212, 220)}}


class Slot:
    """One column-row cell: a clip, opened lazily and looped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.cap: cv2.VideoCapture | None = None
        self.last: np.ndarray | None = None

    def open(self) -> None:
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open unit clip {self.path}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.last = None

    def next_rgb(self, size: tuple[int, int]) -> np.ndarray:
        self.open()
        ok, frame = self.cap.read()
        if not ok:                                    # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"unit clip has no frames: {self.path}")
        self.last = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size,
                               interpolation=cv2.INTER_AREA)
        return self.last


def unit_aspect(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if not w or not h:
        raise SystemExit(f"no dimensions for {path}")
    return w / h


def interleave_outcomes(pool: list[Path], success: dict[str, bool]) -> list[Path]:
    """Order the pool so failures are spread EVENLY through the belt, not clumped.

    With 12 successes and 6 failures the ratio only reads as 2:1 if the failures are distributed --
    six red pills arriving together looks like a broken run, and six at the very end looks like a
    postscript. Bresenham-style placement: walk the output positions and emit a failure whenever the
    running quota crosses an integer, which puts one failure roughly every third column.

    Each stream is task-spread first, so the interleave inherits that and rarely needs repair.
    """
    ok = spread_tasks([p for p in pool if success.get(p.name, True)])
    bad = spread_tasks([p for p in pool if not success.get(p.name, True)])
    if not bad or not ok:
        return ok + bad
    out: list[Path] = []
    i = j = 0
    n = len(pool)
    for k in range(n):
        want_bad = int((k + 1) * len(bad) / n) > int(k * len(bad) / n)
        if want_bad and j < len(bad):
            out.append(bad[j])
            j += 1
        elif i < len(ok):
            out.append(ok[i])
            i += 1
        else:
            out.append(bad[j])
            j += 1
    assert len(out) == n, (len(out), n)
    # Repair any same-task neighbours the merge introduced, by swapping with the next safe element.
    for k in range(1, len(out)):
        if out[k].name.split("__")[0] != out[k - 1].name.split("__")[0]:
            continue
        for m in range(k + 1, len(out)):
            a_ok = out[m].name.split("__")[0] != out[k - 1].name.split("__")[0]
            b_ok = (k + 1 >= len(out)
                    or out[m].name.split("__")[0] != out[k + 1].name.split("__")[0])
            if a_ok and b_ok:
                out[k], out[m] = out[m], out[k]
                break
    return out


def spread_tasks(pool: list[Path]) -> list[Path]:
    """Reorder so no two ADJACENT columns show the same task.

    The pool holds several episodes of some tasks, and a plain shuffle happily put two CoffeeSetupMug
    columns next to each other -- two near-identical panels side by side read as a rendering bug
    rather than as two episodes. Greedy: at each step take the task with the most clips left that is
    not the task just placed. Falls back to placing a repeat only if nothing else remains, so the
    function never drops or duplicates a clip.
    """
    from collections import defaultdict
    by_task: dict[str, list[Path]] = defaultdict(list)
    for p_ in pool:
        by_task[p_.name.split("__")[0]].append(p_)
    out: list[Path] = []
    last = None
    while any(by_task.values()):
        cands = [t for t, v in by_task.items() if v and t != last]
        if not cands:                       # only the just-placed task is left
            cands = [t for t, v in by_task.items() if v]
        t = max(cands, key=lambda k: len(by_task[k]))
        out.append(by_task[t].pop(0))
        last = t
    assert len(out) == len(pool), (len(out), len(pool))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units-dir", type=Path, default=None, help="directory of unit .mp4 files")
    ap.add_argument("--unit", type=Path, action="append", default=None,
                    help="explicit unit clip; repeatable. Overrides --units-dir ordering")
    ap.add_argument("--output", type=Path, required=True)
    # LONG AND SHORT by default: one row of 2:1 units reads as a banner, which is what a scrolling
    # belt is for. --rows 2 stacks it if a taller frame is wanted.
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--col-gap", type=int, default=90, help="space BETWEEN columns")
    ap.add_argument("--row-gap", type=int, default=40)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--scroll-speed", type=float, default=150.0, help="pixels per second, right->left")
    ap.add_argument("--static-seconds", type=float, default=2.5,
                    help="hold the belt still at the start so the first clips can be watched in place")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds; default = long enough for every column to cross the frame once")
    ap.add_argument("--shuffle", type=int, default=None, metavar="SEED",
                    help="shuffle the pool with this seed (default: sorted by filename)")
    ap.add_argument("--bg", choices=sorted(BACKDROPS), default="dark",
                    help="backdrop behind the belt; use the same one the units were rendered with")
    # ALIGNED BY DEFAULT: every row shares the same column boundaries, so units line up in a clean
    # grid and the eye can compare panels straight down a column. A non-zero value staggers each row
    # leftwards by that fraction of the column pitch, which reads as a looser flow but makes the
    # vertical edges ragged.
    ap.add_argument("--row-offset-frac", type=float, default=0.0,
                    help="stagger each row by this fraction of the column pitch (0 = aligned, default)")
    ap.add_argument("--no-row-rotate", dest="row_rotate", action="store_false",
                    help="deal the pool straight down each column; see the row-rotation comment "
                         "-- with an interleaved pool this stacks every failure on one row")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="MANIFEST.json from the unit renderer; enables even interleaving of the "
                         "failure clips through the belt instead of letting them clump")
    ap.add_argument("--no-spread-tasks", action="store_true",
                    help="allow two episodes of the same task in adjacent columns")
    ap.add_argument("--loops", type=float, default=1.0,
                    help="how many times the pool passes the screen (scales the default duration)")
    a = ap.parse_args()

    if a.unit:
        pool = list(a.unit)
    elif a.units_dir:
        pool = sorted(p for p in a.units_dir.glob("*.mp4") if not p.name.startswith("montage"))
    else:
        raise SystemExit("give --units-dir or --unit")
    if not pool:
        raise SystemExit("no unit clips found")
    if a.width % 2 or a.height % 2:
        raise SystemExit("width and height must be even for yuv420p")
    if a.shuffle is not None:
        random.Random(a.shuffle).shuffle(pool)
    manifest_path = a.manifest or ((a.units_dir / "MANIFEST.json") if a.units_dir else None)
    success: dict[str, bool] = {}
    if manifest_path and manifest_path.exists():
        success = {r["file"]: bool(r["success"])
                   for r in json.loads(manifest_path.read_text()).get("units", [])}
    if not a.no_spread_tasks:
        pool = interleave_outcomes(pool, success) if success else spread_tasks(pool)

    theme = BACKDROPS[a.bg]
    aspect = unit_aspect(pool[0])
    unit_h = (a.height - 2 * a.margin - (a.rows - 1) * a.row_gap) // a.rows
    unit_w = round(unit_h * aspect)
    pitch = unit_w + a.col_gap
    if unit_h <= 0 or unit_w <= 0:
        raise SystemExit("frame too small for the requested rows/margins")

    # One pass = every column travels from its start position off the left edge.
    n_cols_pool = -(-len(pool) // a.rows)
    span = n_cols_pool * pitch * a.loops
    duration = a.duration if a.duration else a.static_seconds + span / a.scroll_speed
    n_frames = round(duration * a.fps)

    # Columns are laid out on an infinite belt; a slot is (column, row) and cycles through the pool,
    # so a pool shorter than the belt repeats instead of leaving gaps.
    max_offset = a.scroll_speed * max(0.0, duration - a.static_seconds)
    n_cols = int(max_offset // pitch) + -(-a.width // pitch) + a.rows + 3
    slots: dict[tuple[int, int], Slot] = {}

    # The realised grid for the first pass over the pool, so the outcome spread and any same-task
    # horizontal neighbour can be checked from the printed summary rather than by eye.
    grid_rows = []
    for r in range(a.rows):
        row = []
        for c in range(n_cols_pool):
            rr = (r + c) % a.rows if a.row_rotate else r
            pth = pool[(c * a.rows + rr) % len(pool)]
            row.append(("FAIL" if not success.get(pth.name, True) else "ok", pth.name.split("__")[0]))
        grid_rows.append(row)
    h_dupes = [f"row{i}:{x[1]}" for i, row in enumerate(grid_rows)
               for x, y in zip(row, row[1:]) if x[1] == y[1]]
    fail_per_row = [sum(1 for x in row if x[0] == "FAIL") for row in grid_rows]

    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(a.output), fps=a.fps, codec="libx264", quality=None, pixelformat="yuv420p",
        macro_block_size=1, output_params=["-crf", "20", "-movflags", "+faststart"])
    live_peak = 0
    try:
        for i in range(n_frames):
            t = i / a.fps
            offset = 0.0 if t < a.static_seconds else a.scroll_speed * (t - a.static_seconds)
            canvas = np.full((a.height, a.width, 3), theme["bg"], dtype=np.uint8)
            visible: set[tuple[int, int]] = set()
            for c in range(n_cols):
                for r in range(a.rows):
                    # NEGATIVE stagger: each lower row starts further LEFT, so every row is full at
                    # t=0. Offsetting rightwards instead left a ragged wedge of empty backdrop down
                    # the left edge for the whole static intro.
                    x = round(a.margin + c * pitch - offset - r * a.row_offset_frac * pitch)
                    if x >= a.width or x + unit_w <= 0:
                        continue
                    # ROW ROTATION. Dealing the pool straight down each column puts pool item
                    # c*rows+r at row r, and with failures interleaved every `rows`-th item that
                    # lands EVERY failure on the same row -- a bottom row of nothing but red pills,
                    # which reads as a broken run rather than a 2:1 mix. Rotating the row by the
                    # column index cycles the failure through the rows instead. Same clips in the
                    # same columns; only which row shows which one changes.
                    rr = (r + c) % a.rows if a.row_rotate else r
                    idx = (c * a.rows + rr) % len(pool)
                    key = (c, r)
                    visible.add(key)
                    slot = slots.get(key)
                    if slot is None:
                        slot = slots[key] = Slot(pool[idx])
                    img = slot.next_rgb((unit_w, unit_h))
                    y = a.margin + r * (unit_h + a.row_gap)
                    # Clip against the frame edges so a column can slide partly off-screen.
                    x0, x1 = max(0, x), min(a.width, x + unit_w)
                    canvas[y:y + unit_h, x0:x1] = img[:, x0 - x:x1 - x]
                    cv2.rectangle(canvas, (x, y), (x + unit_w - 1, y + unit_h - 1), theme["edge"], 2)
            # Release anything that has left the frame; decode cost stays proportional to what is
            # on screen rather than to the size of the pool.
            for key in [k for k in slots if k not in visible]:
                slots[key].close()
                del slots[key]
            live_peak = max(live_peak, len(visible))
            writer.append_data(canvas)
    finally:
        writer.close()
        for s in slots.values():
            s.close()

    print(json.dumps({
        "output": str(a.output), "size": [a.width, a.height], "fps": a.fps,
        "frames": n_frames, "duration_s": round(duration, 1),
        "units": len(pool), "rows": a.rows, "columns_drawn": n_cols,
        "unit_size": [unit_w, unit_h], "col_gap": a.col_gap,
        "scroll_speed_px_s": a.scroll_speed, "static_seconds": a.static_seconds,
        "bg": a.bg, "row_offset_frac": a.row_offset_frac,
        "n_success": sum(1 for p in pool if success.get(p.name, True)) if success else None,
        "n_failed": sum(1 for p in pool if not success.get(p.name, True)) if success else None,
        "failures_per_row": fail_per_row, "same_task_horizontal_neighbours": h_dupes,
        "peak_clips_decoding": live_peak,
    }))


if __name__ == "__main__":
    main()
