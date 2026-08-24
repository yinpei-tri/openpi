#!/usr/bin/env python3
"""Render a combined-eval episode as a 2:1 clip: the live milestone plan on the left, camera on the right.

WHAT IT SHOWS. One frame per recorded rollout frame, at the episode's real 20 fps, with:

  LEFT   the accumulated milestone checklist exactly as System2 wrote it for that turn -- every
         milestone always visible, and the fine steps unrolled under the ACTIVE milestone only. That
         behaviour is not imposed by this renderer: it is what ``turn["plan_after"]`` already contains,
         because System2 expands the milestone it is working on and collapses the ones it has closed.
         Verified across episodes -- a completed milestone comes back as a single ``- [x] M2: ...``
         line with no children.
  RIGHT  the right-shoulder camera, which is the MIDDLE tile of the saved rollout video. The saved
         file is a three-camera horizontal strip (left-shoulder | right-shoulder | wrist) at 256px
         per tile, so the middle third is taken and upscaled.

THE ACTIVE-SUBGOAL BOX is drawn around the line System1 is actually executing, found as the deepest
``[~]`` in the checklist -- the active fine step if the active milestone has children, otherwise the
active milestone line itself. It is cross-checked against ``turn["subgoal"]`` (ignoring a "continue
to " prefix, which System2 adds on repeat turns without changing which step is meant); a mismatch is
counted and reported rather than silently drawn on the wrong line.

WHY THE LEFT PANEL IS CACHED PER TURN. The checklist only changes when System2 replans, i.e. once per
turn, but a turn is 100-250 frames. Rendering the text once per turn instead of once per frame is the
difference between ~1 minute and ~20 minutes for a long episode.

MARK GLYPHS come from DejaVu (checked at startup, since a missing glyph renders as a blank box and
would silently produce an unreadable video):  [x] done, [~] active, [ ] not started.

Usage:
    # one episode
    python scripts/make_plan_rollout_video.py --episode-dir <combine>/<method>/<Task>__target__episode_000005 \\
        --out-dir ~/data/video_materials/gallery

    # several, and a still frame to eyeball the layout before committing to a render
    python scripts/make_plan_rollout_video.py --episode-dir A --episode-dir B --out-dir <dir>
    python scripts/make_plan_rollout_video.py --episode-dir A --probe-frame /tmp/probe.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

FONT_REGULAR = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")

# Two palettes, both tuned so the three mark states are distinguishable WITHOUT colour alone (the
# glyph and the weight differ too), which keeps the checklist readable in a greyscale still or for a
# colour-blind viewer. The light theme is not the dark one inverted: the active amber has to darken
# considerably to hold contrast on white, and the "done" green likewise, or the checked items turn
# into pale mush.
THEMES = {
    "dark": {
        "BG": (14, 16, 20), "PANEL": (20, 23, 29), "RULE": (52, 58, 68),
        "TXT": (232, 236, 242), "DIM": (128, 136, 148),
        "DONE": (92, 176, 120), "ACTIVE": (255, 197, 61), "BOX_FILL": (58, 46, 12),
        "CAM_EDGE": (88, 94, 105),
        "DONE_MS": (168, 178, 190), "DONE_FINE": (150, 160, 172),
        "OK_BG": (22, 66, 40), "OK_EDGE": (92, 176, 120), "OK_TXT": (198, 240, 214),
        "BAD_BG": (66, 26, 26), "BAD_EDGE": (214, 96, 96), "BAD_TXT": (246, 186, 186),
        "RUN_BG": (24, 27, 34), "RUN_EDGE": (78, 86, 98),
    },
    "light": {
        "BG": (255, 255, 255), "PANEL": (247, 248, 250), "RULE": (214, 219, 226),
        "TXT": (22, 26, 32), "DIM": (122, 130, 141),
        "DONE": (26, 122, 70), "ACTIVE": (176, 104, 4), "BOX_FILL": (255, 244, 208),
        "CAM_EDGE": (196, 202, 211),
        "DONE_MS": (96, 106, 118), "DONE_FINE": (118, 128, 140),
        "OK_BG": (223, 247, 231), "OK_EDGE": (38, 140, 82), "OK_TXT": (14, 82, 46),
        "BAD_BG": (253, 228, 228), "BAD_EDGE": (196, 62, 62), "BAD_TXT": (128, 22, 22),
        "RUN_BG": (238, 240, 244), "RUN_EDGE": (200, 206, 214),
    },
}

# Set once from the CLI by set_theme(). Kept as module globals, and rebound rather than threaded
# through render_panel / draw_status / render_episode: the palette is a single process-wide choice for
# a one-shot renderer, and adding a palette argument to every signature and call site would be more
# code to read for no extra capability.
BG = PANEL = RULE = TXT = DIM = DONE = ACTIVE = BOX_FILL = CAM_EDGE = (0, 0, 0)
DONE_MS = DONE_FINE = OK_BG = OK_EDGE = OK_TXT = BAD_BG = BAD_EDGE = BAD_TXT = (0, 0, 0)
RUN_BG = RUN_EDGE = (0, 0, 0)


def set_theme(name: str) -> None:
    if name not in THEMES:
        raise SystemExit(f"unknown theme {name!r}; pick one of {sorted(THEMES)}")
    globals().update(THEMES[name])

MARK_DONE, MARK_ACTIVE, MARK_TODO = "✓", "▸", "▫"   # ✓  ▸  ▫

_MS_RE = re.compile(r"^\s*-\s*\[(.)\]\s*(M\d+)\s*:\s*(.*)$")
_FINE_RE = re.compile(r"^\s*\*\s*\[(.)\]\s*(M\d+\.\d+)\s*:\s*(.*)$")
# Prefixes/suffixes the planner and the retry rules add to a subgoal WITHOUT changing which checklist
# step is meant: "continue to X" on a repeat turn, "X again" after a failed attempt. Stripped before
# the box/subgoal cross-check, which otherwise reports a mismatch on a correctly-placed box.
_CONT_RE = re.compile(r"^\s*continue\s+to\s+", re.IGNORECASE)
_AGAIN_RE = re.compile(r"\s+again\s*$", re.IGNORECASE)


def _norm_subgoal(text: str) -> str:
    return _AGAIN_RE.sub("", _CONT_RE.sub("", text or "")).strip().lower().rstrip(".")


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    if not path.exists():
        raise FileNotFoundError(f"missing font: {path}")
    return ImageFont.truetype(str(path), size=size)


def _check_glyphs() -> None:
    """A missing glyph renders as blank, so verify rather than trust the font."""
    f = _font(FONT_REGULAR, 24)
    missing = [g for g in (MARK_DONE, MARK_ACTIVE, MARK_TODO) if f.getbbox(g)[2] == 0]
    if missing:
        raise SystemExit(f"font {FONT_REGULAR} lacks glyph(s) {missing!r} -- pick different markers")


def parse_plan(text: str) -> list[dict]:
    """Checklist text -> [{mark, mid, text, fine:[{mark, fid, text}]}]. Unparsable lines are dropped."""
    out: list[dict] = []
    for line in (text or "").splitlines():
        m = _MS_RE.match(line)
        if m:
            out.append({"mark": m.group(1).strip(), "mid": m.group(2), "text": m.group(3).strip(),
                        "fine": []})
            continue
        f = _FINE_RE.match(line)
        if f and out:
            out[-1]["fine"].append({"mark": f.group(1).strip(), "fid": f.group(2),
                                    "text": f.group(3).strip()})
    return out


def active_line(blocks: list[dict]) -> tuple[int, int] | None:
    """(milestone index, fine index or -1) of the deepest active `[~]`, or None."""
    for i, b in enumerate(blocks):
        if b["mark"] == "~":
            for j, f in enumerate(b["fine"]):
                if f["mark"] == "~":
                    return (i, j)
            return (i, -1)
    return None


def _entries(blocks: list[dict]) -> list[tuple[tuple[int, int], str]]:
    out: list[tuple[tuple[int, int], str]] = []
    for i, b in enumerate(blocks):
        out.append(((i, -1), b["text"]))
        out.extend(((i, j), f["text"]) for j, f in enumerate(b["fine"]))
    return out


def resolve_active(blocks: list[dict], subgoal: str) -> tuple[tuple[int, int] | None, str]:
    """Which line the box goes on: the SUBGOAL's line first, the `[~]` mark second.

    The box is meant to say "this is what System1 is executing right now", so the subgoal actually
    sent to System1 is the authority and the mark is only a fallback. They usually agree, but not
    always: System2 sometimes ticks a fine step `[x]` while still re-issuing it, which leaves no
    `[~]` at the fine level and would otherwise put the box on the parent milestone rather than on
    the step being executed (seen on CoffeeSetupMug ep6 turn 5). Matching the text first fixes that.

    Returns (key, how) with `how` in {"subgoal", "mark", "none"} so the caller can report how often
    the two disagreed instead of hiding it.
    """
    want = _norm_subgoal(subgoal)
    if want:
        hits = [k for k, text in _entries(blocks) if text.strip().lower().rstrip(".") == want]
        if len(hits) == 1:
            return hits[0], "subgoal"
    fallback = active_line(blocks)
    return fallback, ("mark" if fallback else "none")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


PAD = 22
def _indents(base: int) -> tuple[int, int, int]:
    """(glyph->text gap, fine-step indent, fine glyph->text gap), PROPORTIONAL to the font size.

    These were fixed pixel values, which is fine at 21px and cramped at 33: the mark glyph ended up
    touching the "M2:" label once the font ceiling was raised. Scaling them keeps the same visual
    rhythm at every size.
    """
    return max(18, int(base * 0.82)), max(20, int(base * 0.95)), max(18, int(base * 0.72))


def _layout(
    d: ImageDraw.ImageDraw, base: int, w: int, instruction: str, blocks: list[dict], *, hud: bool,
) -> tuple[list[dict], float]:
    """Wrapped layout for one checklist at one font size, plus the total height it needs.

    WRAPS rather than letting a line run off the panel. Raising the font ceiling to 40 made the
    height-only fit insufficient: "M1: close the right fridge door" simply overflowed the panel edge
    and was silently clipped mid-word. Every entry is wrapped to the width actually available at its
    indent, and the wrapped line count feeds back into the height, so fit_base() converges on a size
    that fits in BOTH directions.
    """
    f_instr = _font(FONT_REGULAR, base - 3)
    f_ms = _font(FONT_BOLD, base)
    f_fine = _font(FONT_REGULAR, base - 1)
    inner = w - 2 * PAD
    glyph_ind, fine_ind, fine_text_ind = _indents(base)
    line_h, gap = int(base * 1.42), int(base * 0.34)

    head = int((base + 3) * 1.5)
    instr_lines = _wrap(d, instruction, f_instr, inner)
    y = PAD + head + len(instr_lines) * int((base - 3) * 1.45) + 26

    entries: list[dict] = []
    for i, b in enumerate(blocks):
        txt = f"{b['mid']}: {b['text']}"
        lines = _wrap(d, txt, f_ms, inner - glyph_ind)
        entries.append({"kind": "ms", "key": (i, -1), "mark": b["mark"], "lines": lines,
                        "font": f_ms, "y": y, "h": len(lines) * line_h})
        y += len(lines) * line_h + gap
        for j, fs in enumerate(b["fine"]):
            fl = _wrap(d, fs["text"], f_fine, inner - fine_ind - fine_text_ind - 12)
            entries.append({"kind": "fine", "key": (i, j), "mark": fs["mark"], "lines": fl,
                            "font": f_fine, "y": y, "h": len(fl) * line_h})
            y += len(fl) * line_h + gap
    need = y - PAD + (int(base * 1.9) if hud else 0)
    return entries, need


# The ceiling is what a SHORT plan gets to use. It was 21, which left a three-milestone checklist
# floating in a half-empty panel at the same size a ten-milestone one used; 40 lets a short plan fill
# the panel and stay readable in a slide or a thumbnail. The floor is the readability limit -- below
# ~13px the fine steps stop being legible at 1280 wide, and a plan that long wants a taller frame
# rather than smaller type.
FONT_MAX, FONT_MIN = 40, 13


def fit_base(
    size: tuple[int, int], instruction: str, all_blocks: list[list[dict]], *, hud: bool,
    font_max: int = FONT_MAX, font_min: int = FONT_MIN,
) -> int:
    """ONE font size for the whole episode, sized to the LONGEST checklist any turn will show.

    Fitting per turn instead would resize the text mid-clip: the checklist grows when System2 unrolls
    a milestone with four fine steps and shrinks when it closes one, so the panel would visibly
    breathe from turn to turn. A single size keeps every line in the same place all the way through,
    which is what makes the box look like it is moving down a stable list.
    """
    w, h = size
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    for base in range(font_max, font_min - 1, -1):
        if all(_layout(probe, base, w, instruction, b, hud=hud)[1] <= h - 2 * PAD
               for b in all_blocks):
            return base
    return font_min


# How the loop stopped, in words a viewer can read. These are combined_eval's `termination` values;
# anything unmapped is shown verbatim rather than hidden, so a new reason is visible not silent.
_TERM_WORDS = {
    "env_success": "env success",
    "task_finish": "planner declared done",
    "max_cap": "step cap reached",
    "horizon_exhausted": "out of steps",
    "max_turns": "turn limit reached",
    "no_subgoal": "planner issued no subgoal",
}


def draw_status(canvas: Image.Image, cam_box: tuple[int, int, int, int], state: str, text: str,
                base: int) -> None:
    """Outcome pill, inset into the TOP-LEFT of the camera panel.

    Drawn on the CANVAS, not into the cached per-turn panel, because the outcome changes mid-turn --
    the env success check fires at a specific frame inside the last turn, not at a turn boundary.
    Showing the verdict from frame 0 would be a spoiler and would assert something untrue of the first
    99% of the clip, so it reads "in progress" until the exact frame success fires, and a failure is
    only declared at the end, when it is actually known.

    Over the video rather than under the checklist: at the larger font sizes the checklist's own
    bottom line and this pill were colliding in the left panel, and the camera's top-left corner is
    both always free and where a viewer already looks for a status badge.
    """
    fill, edge, txt = {
        "ok": (OK_BG, OK_EDGE, OK_TXT),
        "fail": (BAD_BG, BAD_EDGE, BAD_TXT),
    }.get(state, (RUN_BG, RUN_EDGE, DIM))
    d = ImageDraw.Draw(canvas)
    path = FONT_BOLD if state != "run" else FONT_REGULAR
    pad_x, pad_y, inset = 13, 7, 12
    # MUST FIT THE CAMERA WIDTH. At the raised font ceiling "TASK FAILED . planner issued no subgoal"
    # was wider than the camera panel and ran off its right edge. Shrink first; if even the floor is
    # too wide, drop the reason rather than clip the text -- a truncated word is worse than a shorter
    # label, and the reason is still in the JSON the renderer prints.
    avail = (cam_box[2] - cam_box[0]) - 2 * inset - 2 * pad_x
    f = _font(path, max(13, base - 5))
    while d.textlength(text, font=f) > avail and f.size > 13:
        f = _font(path, f.size - 1)
    if d.textlength(text, font=f) > avail and "·" in text:
        text = text.split("·")[0].strip()
        f = _font(path, max(13, base - 5))
        while d.textlength(text, font=f) > avail and f.size > 13:
            f = _font(path, f.size - 1)
    tw = d.textlength(text, font=f)
    top, bot = f.getbbox("Hg")[1], f.getbbox("Hg")[3]
    x0, y0 = cam_box[0] + inset, cam_box[1] + inset
    x1, y1 = x0 + tw + 2 * pad_x, y0 + (bot - top) + 2 * pad_y
    d.rounded_rectangle((x0, y0, x1, y1), radius=(y1 - y0) // 2, fill=fill, outline=edge, width=2)
    d.text((x0 + pad_x, y0 + pad_y - top), text, font=f, fill=txt)


def render_panel(
    size: tuple[int, int], task: str, instruction: str, blocks: list[dict],
    active: tuple[int, int] | None, hud: str | None, base: int,
) -> Image.Image:
    """The left panel for ONE turn, at the episode-wide font size from fit_base()."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL)
    d = ImageDraw.Draw(img)
    entries, _ = _layout(d, base, w, instruction, blocks, hud=bool(hud))
    glyph_ind, fine_ind, fine_text_ind = _indents(base)
    line_h = int(base * 1.42)

    y = PAD
    d.text((PAD, y), task, font=_font(FONT_BOLD, base + 3), fill=TXT)
    y += int((base + 3) * 1.5)
    f_instr = _font(FONT_REGULAR, base - 3)
    for ln in _wrap(d, instruction, f_instr, w - 2 * PAD):
        d.text((PAD, y), ln, font=f_instr, fill=DIM)
        y += int((base - 3) * 1.45)
    y += 12
    d.line((PAD, y, w - PAD, y), fill=RULE, width=1)

    for e in entries:
        ms = e["kind"] == "ms"
        act = e["mark"] == "~"
        glyph = MARK_DONE if e["mark"] == "x" else (MARK_ACTIVE if act else MARK_TODO)
        gcol = DONE if e["mark"] == "x" else (ACTIVE if act else DIM)
        # done: dimmed but still legible (milestones a touch brighter than their fine steps);
        # active: full white; not started: muted.
        done_col = DONE_MS if ms else DONE_FINE
        tcol = done_col if e["mark"] == "x" else (TXT if act else DIM)
        gx = PAD if ms else PAD + fine_ind
        tx = gx + (glyph_ind if ms else fine_text_ind)
        # The box spans every WRAPPED line of the entry, so a two-line subgoal is fully enclosed.
        if active == e["key"]:
            d.rounded_rectangle((gx - 6, e["y"] - 4, w - PAD + 2, e["y"] + e["h"] - 2),
                                radius=7, fill=BOX_FILL, outline=ACTIVE, width=2)
        d.text((gx, e["y"]), glyph, font=e["font"], fill=gcol)
        for n, ln in enumerate(e["lines"]):
            d.text((tx, e["y"] + n * line_h), ln, font=e["font"], fill=tcol)
    if hud:
        d.text((PAD, h - PAD - int(base * 1.4)), hud, font=_font(FONT_REGULAR, base - 4), fill=DIM)
    return img


def render_episode(
    ep_dir: Path, out_path: Path, *, size: tuple[int, int], fps: float, stride: int,
    hud: bool, probe: Path | None, font_max: int = FONT_MAX, font_min: int = FONT_MIN,
    hold: float = 1.5,
) -> dict:
    ep = json.loads((ep_dir / "episode.json").read_text())
    turns = {int(t["turn"]): t for t in ep.get("turns", [])}
    task = ep.get("task_name", "?")
    instruction = (ep.get("instruction") or "").strip()

    out_w, out_h = size
    margin = 20
    cam = out_h - 2 * margin                  # square camera panel
    panel_w = out_w - cam - 3 * margin
    cam_x = out_w - margin - cam

    writer = None if probe else imageio.get_writer(
        str(out_path), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
        macro_block_size=1, output_params=["-crf", "18", "-movflags", "+faststart"])

    # Fall back to the previous turn's checklist when a turn has no plan_after (System2 declined to
    # replan), and to the cold plan for turn 0 -- never render an empty panel.
    cold = ep.get("plan")
    if isinstance(cold, dict):
        cold = cold.get("plan") or ""
    last_plan = str(cold or "")

    # PASS 1: every turn's checklist, so the font size can be fixed for the whole episode before a
    # single frame is written. Cheap -- it only parses JSON, it does not touch the videos.
    plan_seq: dict[int, list[dict]] = {}
    scan = last_plan
    for tdir in sorted(ep_dir.glob("turn[0-9][0-9]")):
        ti = int(tdir.name.removeprefix("turn"))
        scan = str(turns.get(ti, {}).get("plan_after") or "").strip() or scan
        b = parse_plan(scan)
        if b:
            plan_seq[ti] = b
    if not plan_seq:
        raise RuntimeError(f"no parsable checklist in any turn of {ep_dir}")
    base = fit_base((panel_w, cam), instruction, list(plan_seq.values()), hud=hud,
                    font_max=font_max, font_min=font_min)

    # frames == n_steps exactly (verified per turn on this tree), so an env-step index IS a frame
    # index and the success moment can be placed to the frame rather than to the turn.
    total_src = sum((turns.get(int(t.name.removeprefix("turn")), {}).get("n_steps") or 0)
                    for t in sorted(ep_dir.glob("turn[0-9][0-9]")))
    succeeded = bool(ep.get("episode_success"))
    term = str(ep.get("termination") or "")
    term_words = _TERM_WORDS.get(term, term or "no reason recorded")
    # Failure is only knowable at the end, so it is announced over the last 2 seconds of OUTPUT.
    tail_src = int(2.0 * fps * stride)

    n_frames = 0
    cum_steps = 0
    gframe = 0
    last_canvas: Image.Image | None = None
    success_frame: int | None = None
    mismatches: list[str] = []
    try:
        for tdir in sorted(ep_dir.glob("turn[0-9][0-9]")):
            ti = int(tdir.name.removeprefix("turn"))
            t = turns.get(ti, {})
            subgoal = str(t.get("subgoal") or "").strip()
            blocks = plan_seq.get(ti)
            if not blocks:
                continue
            act, how = resolve_active(blocks, subgoal)
            # Report only what could NOT be reconciled: the box fell back to the `[~]` mark and that
            # mark's text is not the subgoal, so the highlighted line is a guess.
            if act and subgoal and how == "mark":
                shown = (blocks[act[0]]["fine"][act[1]]["text"] if act[1] >= 0
                         else blocks[act[0]]["text"])
                if _norm_subgoal(subgoal) != shown.strip().lower().rstrip("."):
                    mismatches.append(f"turn{ti}: box={shown!r} subgoal={subgoal!r}")

            video = tdir / "s1_rollout_raw.mp4"
            if not video.exists():
                continue
            if success_frame is None and t.get("success_step") is not None:
                success_frame = gframe + int(t["success_step"])
            hud_txt = (f"turn {ti + 1}/{len(turns)}   ·   {cum_steps + (t.get('n_steps') or 0)} env steps"
                       if hud else None)
            panel = render_panel((panel_w, cam), task, instruction, blocks, act, hud_txt, base)
            cum_steps += t.get("n_steps") or 0

            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {video}")
            k = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if k % stride:
                    k += 1
                    gframe += 1
                    continue
                k += 1
                gframe += 1
                if frame_bgr.ndim != 3 or frame_bgr.shape[1] % 3:
                    raise ValueError(f"expected a 3-camera tile, got {frame_bgr.shape}")
                tw = frame_bgr.shape[1] // 3
                # MIDDLE tile = right-shoulder (strip order: left-shoulder | right-shoulder | wrist).
                right = cv2.cvtColor(frame_bgr[:, tw:2 * tw], cv2.COLOR_BGR2RGB)
                right = cv2.resize(right, (cam, cam), interpolation=cv2.INTER_LANCZOS4)

                canvas = Image.new("RGB", (out_w, out_h), BG)
                canvas.paste(panel, (margin, margin))
                canvas.paste(Image.fromarray(right), (cam_x, margin))
                ImageDraw.Draw(canvas).rounded_rectangle(
                    (cam_x - 2, margin - 2, cam_x + cam + 1, margin + cam + 1),
                    radius=8, outline=CAM_EDGE, width=2)
                if success_frame is not None and gframe >= success_frame:
                    state, label = "ok", "TASK SUCCESS"
                elif not succeeded and gframe >= total_src - tail_src:
                    state, label = "fail", f"TASK FAILED  ·  {term_words}"
                else:
                    state, label = "run", "in progress"
                draw_status(canvas, (cam_x, margin, cam_x + cam, margin + cam),
                            state, label, base)
                last_canvas = canvas
                arr = np.asarray(canvas, dtype=np.uint8)
                if probe:
                    capture.release()
                    Image.fromarray(arr).save(probe)
                    return {"probe": str(probe), "task": task, "turn": ti}
                writer.append_data(arr)
                n_frames += 1
            capture.release()
        # HOLD THE VERDICT. The env success check fires on the LAST env step of the last turn, so
        # without this the green pill is on screen for a single frame -- 1/20 s, invisible. The clip
        # therefore ends by freezing its final frame with the outcome stamped on it. Nothing is
        # fabricated: it is the real last frame, repeated, and the pill states what the episode's
        # recorded outcome was.
        if writer is not None and last_canvas is not None and hold > 0:
            final = last_canvas.copy()
            draw_status(final, (cam_x, margin, cam_x + cam, margin + cam),
                        "ok" if succeeded else "fail",
                        "TASK SUCCESS" if succeeded else f"TASK FAILED  ·  {term_words}", base)
            arr = np.asarray(final, dtype=np.uint8)
            for _ in range(int(hold * fps)):
                writer.append_data(arr)
                n_frames += 1
    finally:
        if writer is not None:
            writer.close()

    if not n_frames:
        raise RuntimeError(f"no frames rendered for {ep_dir}")
    return {
        "output": str(out_path), "task": task, "episode": ep.get("episode_index"),
        "success": ep.get("episode_success"), "termination": term,
        "success_frame": success_frame, "turns": ep.get("n_turns"),
        "frames": n_frames, "fps": fps, "duration_s": round(n_frames / fps, 1),
        "size": [out_w, out_h], "stride": stride, "speed_x": stride, "font_base": base, "hold_s": hold,
        # Surfaced, not swallowed: a box drawn on a line that is not the executed subgoal is a bug in
        # either the plan parse or the data, and the viewer cannot tell by watching.
        "box_subgoal_mismatches": mismatches,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dir", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("~/data/video_materials/gallery").expanduser())
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=640, help="default gives the 2:1 frame")
    ap.add_argument("--fps", type=float, default=20.0, help="20 = the rollout's real time")
    # 2x by default: keeping every 2nd frame and playing at the source 20 fps halves the runtime
    # while the OUTPUT still carries 20 distinct frames per second, so it stays smooth and stays
    # inside what any player handles. Raising fps instead would need 40 fps playback for the same
    # speed-up. --speed 1 restores real time.
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed-up; 2 = twice real time (default), 1 = real time")
    ap.add_argument("--stride", type=int, default=None,
                    help="keep 1 frame in N; overrides --speed if given")
    ap.add_argument("--theme", choices=sorted(THEMES), default="dark",
                    help="dark (default) or light, for a white-background deck or paper figure")
    ap.add_argument("--hold", type=float, default=1.5,
                    help="seconds to freeze the final frame with the outcome stamped on it; the env "
                         "success check fires on the last step, so without a hold the verdict is "
                         "visible for one frame. 0 disables.")
    ap.add_argument("--max-font", type=int, default=FONT_MAX,
                    help=f"largest checklist font a short plan may use (default {FONT_MAX})")
    ap.add_argument("--min-font", type=int, default=FONT_MIN)
    ap.add_argument("--no-hud", action="store_true", help="drop the turn / env-step line")
    ap.add_argument("--probe-frame", type=Path, default=None,
                    help="write the first frame as a PNG and exit (layout check, no encode)")
    a = ap.parse_args()

    set_theme(a.theme)
    _check_glyphs()
    if a.width % 2 or a.height % 2:
        raise SystemExit("width and height must be even for yuv420p")
    if a.speed <= 0:
        raise SystemExit("--speed must be positive")
    stride = a.stride if a.stride else max(1, round(a.speed))
    if a.stride is None and abs(a.speed - stride) > 0.01:
        print(f"note: --speed {a.speed} rounded to stride {stride} "
              f"({stride}x), since frames can only be dropped whole", file=sys.stderr)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in a.episode_dir:
        if not (ep / "episode.json").exists():
            raise SystemExit(f"not an episode dir: {ep}")
        out = a.out_dir / f"{ep.name}.mp4"
        r = render_episode(ep, out, size=(a.width, a.height), fps=a.fps, stride=stride,
                           hud=not a.no_hud, probe=a.probe_frame,
                           font_max=a.max_font, font_min=a.min_font, hold=a.hold)
        results.append(r)
        print(json.dumps(r), flush=True)
        if a.probe_frame:
            break
    # MANIFEST. The outcome cannot be recovered from an .mp4, and the montage needs it to interleave
    # successes and failures at a chosen ratio, so record it next to the clips. Written even for a
    # single-episode run, and merged with any existing manifest so incremental renders accumulate.
    if not a.probe_frame:
        mf = a.out_dir / "MANIFEST.json"
        old_rows = {}
        if mf.exists():
            try:
                old_rows = {r["file"]: r for r in json.loads(mf.read_text()).get("units", [])}
            except Exception:
                old_rows = {}
        for r in results:
            f = Path(r["output"]).name
            old_rows[f] = {"file": f, "task": r["task"], "episode": r["episode"],
                           "success": r["success"], "termination": r["termination"],
                           "duration_s": r["duration_s"], "font_base": r["font_base"]}
        rows = [old_rows[k] for k in sorted(old_rows)]
        mf.write_text(json.dumps({"n": len(rows),
                                  "n_success": sum(1 for r in rows if r["success"]),
                                  "units": rows}, indent=1))
        print(f"manifest: {mf}  ({sum(1 for r in rows if r['success'])}/{len(rows)} success)")

    bad = [m for r in results for m in r.get("box_subgoal_mismatches") or []]
    if bad:
        print(f"WARNING: {len(bad)} active-box/subgoal mismatch(es):", file=sys.stderr)
        for m in bad[:10]:
            print("   ", m, file=sys.stderr)


if __name__ == "__main__":
    main()
