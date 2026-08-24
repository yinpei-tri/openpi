#!/usr/bin/env python3
"""Create a tall infographic explaining one-shot video learning for unseen-short tasks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 3600
MARGIN = 80
FONT_REGULAR = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf")
SUMMARY_ROOT = Path("/home/ec2-user/data/unseen_task_summary/qwen3vl-4b-full-ep3-17124")
EPISODE_DIR = SUMMARY_ROOT / "ArrangeTea__episode_000003"
OUTPUT = Path("/home/ec2-user/data/video_materials/one_shot_video_learning_unseenshort_ArrangeTea.png")

INK = (29, 36, 48)
MUTED = (97, 107, 121)
LINE = (210, 216, 225)
NAVY = (52, 78, 128)
BLUE = (66, 111, 184)
BLUE_BG = (236, 243, 255)
GOLD = (215, 146, 25)
GOLD_BG = (255, 248, 229)
GREEN = (42, 142, 84)
GREEN_BG = (233, 248, 239)
RED = (188, 57, 63)
RED_BG = (253, 238, 239)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    paragraphs = text.splitlines() or [""]
    output: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            output.append("")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or draw.textlength(candidate, font=face) <= width:
                current = candidate
            else:
                output.append(current)
                current = word
        if current:
            output.append(current)
    return output


def text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    face: ImageFont.FreeTypeFont,
    width: int,
    fill: tuple[int, int, int] = INK,
    line_gap: int = 10,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines = wrap(draw, text, face, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    box = draw.textbbox((0, 0), "Ag", font=face)
    line_h = box[3] - box[1] + line_gap
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += line_h
    return y


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = (255, 255, 255),
    outline: tuple[int, int, int] = LINE,
    width: int = 3,
    radius: int = 28,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    text_fill: tuple[int, int, int] = (255, 255, 255),
    size: int = 25,
) -> int:
    face = font(size, bold=True)
    x, y = xy
    w = int(draw.textlength(text, font=face)) + 42
    h = size + 25
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill)
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x + 21, y + h / 2 - (box[3] - box[1]) / 2 - box[1]), text, font=face, fill=text_fill)
    return w


def step_title(
    draw: ImageDraw.ImageDraw,
    number: int,
    y: int,
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
) -> int:
    cx = MARGIN + 42
    cy = y + 42
    draw.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), fill=color)
    nface = font(38, bold=True)
    nbox = draw.textbbox((0, 0), str(number), font=nface)
    draw.text((cx - (nbox[2] - nbox[0]) / 2, cy - (nbox[3] - nbox[1]) / 2 - nbox[1]), str(number), font=nface, fill="white")
    draw.text((MARGIN + 105, y + 4), title, font=font(43, bold=True), fill=INK)
    draw.text((MARGIN + 105, y + 60), subtitle, font=font(25), fill=MUTED)
    return y + 105


def down_arrow(draw: ImageDraw.ImageDraw, y1: int, y2: int, color: tuple[int, int, int] = NAVY) -> None:
    x = WIDTH // 2
    draw.line((x, y1, x, y2 - 18), fill=color, width=8)
    draw.polygon(((x - 20, y2 - 30), (x + 20, y2 - 30), (x, y2)), fill=color)


def video_frame(chunk: int, camera: int) -> Image.Image:
    path = EPISODE_DIR / f"chunk{chunk:02d}" / "clip_full.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {path}")
    tile_w = frame.shape[1] // 3
    crop = frame[:, camera * tile_w:(camera + 1) * tile_w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop)


def fit_crop(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    target_w, target_h = box
    ratio = max(target_w / image.width, target_h / image.height)
    resized = image.resize((round(image.width * ratio), round(image.height * ratio)), Image.Resampling.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def main() -> None:
    memory = json.loads((EPISODE_DIR / "memory.json").read_text())
    task_summary = json.loads((SUMMARY_ROOT / "TASK_SUMMARY.json").read_text())["tasks"]["ArrangeTea"]
    detailed_goal = memory["instruction_original"]
    short_goal = memory["goal_phrase"]
    recipe = task_summary["recipe"]
    narrations = memory["narrations"]

    canvas = Image.new("RGB", (WIDTH, HEIGHT), (247, 249, 252))
    draw = ImageDraw.Draw(canvas)

    # Header.
    draw.rectangle((0, 0, WIDTH, 245), fill=(35, 48, 72))
    draw.text((MARGIN, 45), "ONE-SHOT VIDEO LEARNING", font=font(64, bold=True), fill="white")
    draw.text((MARGIN, 128), "Recovering task knowledge for the unseen-short setting", font=font(33), fill=(210, 222, 243))
    chip(draw, (1010, 177), "RUNNING EXAMPLE  •  ARRANGE TEA", fill=(73, 111, 174), size=20)

    # Step 1: the hidden detailed goal.
    y = 290
    card(draw, (MARGIN, y, WIDTH - MARGIN, 650), fill=(255, 255, 255), outline=(207, 211, 219))
    content_y = step_title(
        draw, 1, y + 30, "Original task goal", "Dataset annotation — shown here only for comparison", RED,
    )
    chip(draw, (MARGIN + 42, content_y + 8), "NOT PROVIDED IN UNSEEN-SHORT EVALUATION", fill=RED, size=20)
    quote_y = content_y + 78
    draw.rounded_rectangle((MARGIN + 42, quote_y, WIDTH - MARGIN - 42, 635), radius=20, fill=RED_BG)
    draw.text((MARGIN + 73, quote_y + 20), "“", font=font(62, bold=True), fill=RED)
    text_block(
        draw, (MARGIN + 123, quote_y + 25), detailed_goal, face=font(27, bold=True),
        width=WIDTH - 2 * MARGIN - 220, line_gap=9,
    )
    down_arrow(draw, 650, 705)

    # Step 2: short goal.
    y = 720
    card(draw, (MARGIN, y, WIDTH - MARGIN, 1015), fill=BLUE_BG, outline=(150, 180, 228), width=4)
    content_y = step_title(
        draw, 2, y + 28, "Replace it with a short goal", "The same compact phrase is used across all episodes of the task", BLUE,
    )
    draw.rounded_rectangle((MARGIN + 42, content_y + 14, WIDTH - MARGIN - 42, 946), radius=22, fill="white", outline=(162, 187, 226), width=3)
    short_face = font(50, bold=True)
    short_text = f'“{short_goal}”'
    tw = draw.textlength(short_text, font=short_face)
    draw.text(((WIDTH - tw) / 2, content_y + 55), short_text, font=short_face, fill=(45, 82, 145))
    draw.text((MARGIN + 66, 968), "Source: composite_unseen_short_goal.json", font=font(22), fill=MUTED)
    down_arrow(draw, 1015, 1070)

    # Step 3: one-shot summary mode.
    y = 1085
    card(draw, (MARGIN, y, WIDTH - MARGIN, 2140), fill=(255, 255, 255), outline=(160, 183, 218), width=4)
    content_y = step_title(
        draw, 3, y + 28, "Summarize one successful demonstration", "Qwen3-VL watches the video using only the short goal", NAVY,
    )
    chip(draw, (MARGIN + 42, content_y + 5), "1 VIDEO", fill=NAVY, size=19)
    chip(draw, (MARGIN + 175, content_y + 5), "935 FRAMES", fill=(93, 112, 146), size=19)
    chip(draw, (MARGIN + 355, content_y + 5), "12 × 4-SECOND CHUNKS", fill=(93, 112, 146), size=19)

    # Film strip.
    film_y = content_y + 72
    frame_specs = [
        (0, 1, "reach kettle"), (2, 2, "place kettle"), (4, 2, "grasp cup"),
        (6, 2, "place cup"), (8, 0, "close door"), (11, 1, "close door"),
    ]
    gap = 14
    film_x = MARGIN + 42
    film_w = WIDTH - 2 * MARGIN - 84
    thumb_w = (film_w - gap * 5) // 6
    thumb_h = 210
    for index, (chunk, camera, label) in enumerate(frame_specs):
        x = film_x + index * (thumb_w + gap)
        im = fit_crop(video_frame(chunk, camera), (thumb_w, thumb_h))
        canvas.paste(im, (x, film_y))
        draw.rounded_rectangle((x, film_y, x + thumb_w, film_y + thumb_h), radius=12, outline=(164, 170, 179), width=3)
        label_face = font(19, bold=True)
        label_w = int(draw.textlength(label, font=label_face)) + 24
        draw.rounded_rectangle((x + 8, film_y + thumb_h - 38, x + 8 + label_w, film_y + thumb_h - 8), radius=8, fill=(255, 255, 255))
        draw.text((x + 20, film_y + thumb_h - 34), label, font=label_face, fill=INK)

    # Inputs -> summary model.
    input_y = film_y + thumb_h + 36
    left_box = (MARGIN + 42, input_y, 610, input_y + 135)
    right_box = (650, input_y, WIDTH - MARGIN - 42, input_y + 135)
    card(draw, left_box, fill=BLUE_BG, outline=(152, 181, 226), width=3, radius=18)
    draw.text((left_box[0] + 24, input_y + 17), "SHORT GOAL", font=font(19, bold=True), fill=BLUE)
    draw.text((left_box[0] + 24, input_y + 55), f'“{short_goal}”', font=font(31, bold=True), fill=INK)
    card(draw, right_box, fill=(243, 245, 248), outline=(194, 199, 208), width=3, radius=18)
    draw.text((right_box[0] + 24, input_y + 17), "ONE SUCCESSFUL VIDEO", font=font(19, bold=True), fill=MUTED)
    draw.text((right_box[0] + 24, input_y + 55), "temporal visual evidence", font=font(31, bold=True), fill=INK)
    model_y = input_y + 165
    draw.line((WIDTH // 2, input_y + 136, WIDTH // 2, model_y - 14), fill=NAVY, width=6)
    draw.polygon(((WIDTH // 2 - 16, model_y - 24), (WIDTH // 2 + 16, model_y - 24), (WIDTH // 2, model_y)), fill=NAVY)
    model_box = (390, model_y, 1210, model_y + 105)
    card(draw, model_box, fill=(43, 59, 86), outline=(43, 59, 86), width=3, radius=22)
    model_text = "Qwen3-VL  •  SUMMARY MODE"
    model_face = font(34, bold=True)
    mtw = draw.textlength(model_text, font=model_face)
    draw.text(((WIDTH - mtw) / 2, model_y + 30), model_text, font=model_face, fill="white")

    # Narration examples.
    narration_y = model_y + 140
    draw.text((MARGIN + 42, narration_y), "Chunk-level narrations", font=font(25, bold=True), fill=INK)
    examples = [narrations[0], narrations[4], narrations[11]]
    box_gap = 18
    box_w = (film_w - box_gap * 2) // 3
    for index, narration in enumerate(examples):
        x = film_x + index * (box_w + box_gap)
        box = (x, narration_y + 42, x + box_w, narration_y + 225)
        card(draw, box, fill=(246, 248, 251), outline=(209, 214, 222), width=2, radius=16)
        draw.text((x + 19, narration_y + 57), f"CHUNK {index * 4 + (3 if index == 2 else 0):02d}", font=font(17, bold=True), fill=NAVY)
        text_block(draw, (x + 19, narration_y + 90), narration, face=font(20), width=box_w - 38, fill=INK, line_gap=7, max_lines=5)
    draw.text((MARGIN + 42, 2092), "Narrate each chunk  →  aggregate the narrations into reusable task knowledge", font=font(24, bold=True), fill=NAVY)
    down_arrow(draw, 2140, 2195, GOLD)

    # Step 4: generated recipe.
    y = 2210
    card(draw, (MARGIN, y, WIDTH - MARGIN, 2785), fill=GOLD_BG, outline=(231, 180, 78), width=4)
    content_y = step_title(
        draw, 4, y + 28, "Generated task recipe", "A text memory distilled from one demonstration", GOLD,
    )
    chip(draw, (MARGIN + 42, content_y + 5), "SELECTED ONE-SHOT RECIPE", fill=GOLD, size=20)
    recipe_y = content_y + 74
    draw.rounded_rectangle((MARGIN + 42, recipe_y, WIDTH - MARGIN - 42, 2722), radius=22, fill="white", outline=(232, 190, 105), width=3)
    draw.text((MARGIN + 68, recipe_y + 18), "TASK RECIPE", font=font(21, bold=True), fill=GOLD)
    text_block(
        draw, (MARGIN + 68, recipe_y + 62), recipe, face=font(25),
        width=WIDTH - 2 * MARGIN - 136, fill=INK, line_gap=10,
    )
    chip(draw, (MARGIN + 42, 2730), "CACHED ONCE", fill=(164, 112, 20), size=18)
    chip(draw, (MARGIN + 240, 2730), "REUSED FOR EVERY ARRANGE TEA EPISODE", fill=(164, 112, 20), size=18)
    down_arrow(draw, 2785, 2840, GREEN)

    # Step 5: deployment.
    y = 2855
    card(draw, (MARGIN, y, WIDTH - MARGIN, 3550), fill=GREEN_BG, outline=(121, 194, 151), width=4)
    content_y = step_title(
        draw, 5, y + 28, "Plan a new episode with memory", "The detailed dataset instruction is never restored", GREEN,
    )

    # Input equation.
    equation_y = content_y + 18
    eq_specs = [
        ("CURRENT SCENE", "visual observation", (242, 245, 249), MUTED),
        ("SHORT GOAL", f'“{short_goal}”', BLUE_BG, BLUE),
        ("TASK RECIPE", "recalled text memory", GOLD_BG, GOLD),
    ]
    eq_gap = 48
    eq_x = MARGIN + 42
    eq_w = 390
    for index, (label, value, fill, color) in enumerate(eq_specs):
        x = eq_x + index * (eq_w + eq_gap)
        card(draw, (x, equation_y, x + eq_w, equation_y + 118), fill=fill, outline=color, width=3, radius=18)
        draw.text((x + 22, equation_y + 16), label, font=font(18, bold=True), fill=color)
        draw.text((x + 22, equation_y + 55), value, font=font(25, bold=True), fill=INK)
        if index < 2:
            plus_x = x + eq_w + eq_gap // 2
            draw.text((plus_x - 12, equation_y + 39), "+", font=font(39, bold=True), fill=MUTED)

    arrow_y = equation_y + 145
    down_arrow(draw, arrow_y, arrow_y + 54, GREEN)
    draw.text((MARGIN + 42, arrow_y + 70), "S2 MILESTONE PLAN", font=font(22, bold=True), fill=GREEN)
    plan = [
        "M1  grasp the kettle",
        "M2  place the kettle on the tray",
        "M3  grasp the mug",
        "M4  place the mug on the tray",
        "M5  close the left cabinet door",
        "M6  close the right cabinet door",
    ]
    plan_y = arrow_y + 112
    plan_w = (WIDTH - 2 * MARGIN - 102) // 2
    for index, item in enumerate(plan):
        col, row = index % 2, index // 2
        x = MARGIN + 42 + col * (plan_w + 18)
        py = plan_y + row * 69
        draw.rounded_rectangle((x, py, x + plan_w, py + 54), radius=13, fill="white", outline=(161, 207, 178), width=2)
        draw.text((x + 17, py + 12), "✓", font=font(23, bold=True), fill=GREEN)
        draw.text((x + 53, py + 12), item, font=font(22, bold=True), fill=INK)

    badge_y = 3475
    draw.rounded_rectangle((MARGIN + 42, badge_y, WIDTH - MARGIN - 42, badge_y + 55), radius=27, fill=(39, 132, 78))
    message = "SHORT GOAL + ONE VIDEO → REUSABLE RECIPE → BETTER PLANNING ON NEW EPISODES"
    message_face = font(24, bold=True)
    message_w = draw.textlength(message, font=message_face)
    draw.text(((WIDTH - message_w) / 2, badge_y + 13), message, font=message_face, fill="white")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(json.dumps({"output": str(OUTPUT), "size": [WIDTH, HEIGHT], "bytes": OUTPUT.stat().st_size}))


if __name__ == "__main__":
    main()
