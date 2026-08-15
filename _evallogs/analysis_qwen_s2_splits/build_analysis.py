from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont
from scipy.stats import binomtest, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "robocasa"))
from combined_eval import TARGET_TASK_SPLIT  # noqa: E402
from horizon_gate import gate_episode, total_steps  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = Path("/home/ec2-user/data/sys1_eval_results/combine")
METHODS = {
    ("Qwen3-VL", "base"): "s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-base",
    ("Qwen3-VL", "inst"): "s1-progact270k_s2-qwen3vl-4b-full-ep3-17124-inst",
    ("Qwen3.5", "base"): "s1-progact270k_s2-qwen35-4b-full-ep3-11416-base",
    ("Qwen3.5", "inst"): "s1-progact270k_s2-qwen35-4b-full-ep3-11416-inst",
}
SPLITS = ("atomic_seen", "composite_seen", "composite_unseen")
SPLIT_LABEL = {
    "atomic_seen": "Atomic-seen",
    "composite_seen": "Composite-seen",
    "composite_unseen": "Composite-unseen",
}
COLORS = {"Qwen3-VL": "#3978c5", "Qwen3.5": "#e07a35"}
ARM_ALPHA = {"base": 145, "inst": 255}
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rgba(hex_color: str, alpha: int = 255):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def safe_mean(xs):
    return statistics.mean(xs) if xs else math.nan


def percentile(xs, q):
    if not xs:
        return math.nan
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def norm_subgoal(s):
    s = (s or "").strip().lower().rstrip(".")
    s = re.sub(r"^continue\s+to\s+", "", s)
    s = re.sub(r"\s+again$", "", s)
    return re.sub(r"\s+", " ", s)


def action_class(s):
    s = norm_subgoal(s)
    if re.search(r"\bwait\b", s):
        return "wait"
    if re.match(r"^(retract|withdraw|move (the )?arm away)", s):
        return "retract"
    if re.search(r"\b(base|navigate|drive|go to|search|locate)\b", s):
        return "navigate/base"
    if re.search(r"\bgrasp|pick up|grab\b", s):
        return "grasp/pick"
    if re.match(r"^(reach|move (the )?(arm|gripper)|extend)", s):
        return "reach"
    if re.search(r"\b(place|release|put|drop|lower)\b", s):
        return "place/release"
    if re.search(r"\b(carry|lift|transport)\b", s):
        return "carry/lift"
    if re.search(r"\b(push|pull|press|turn|open|close|slide|tilt|rotate)\b", s):
        return "actuate"
    return "other"


def est_bucket(x):
    if x is None:
        return None
    for edge in (50, 75, 100, 125, 150, 175, 200, 250, 300):
        if x <= edge:
            return f"≤{edge}" if edge == 50 else str(edge)
    return ">300"


EST_BUCKETS = ("≤50", "75", "100", "125", "150", "175", "200", "250", "300", ">300")
ACTION_CLASSES = (
    "reach",
    "grasp/pick",
    "carry/lift",
    "place/release",
    "actuate",
    "navigate/base",
    "retract",
    "wait",
    "other",
)


def load_data():
    episodes = []
    turns = []
    rules = []
    for (model, arm), method in METHODS.items():
        method_dir = ROOT / method
        for ep_file in sorted(method_dir.glob("*__episode_*/episode.json")):
            try:
                doc = json.loads(ep_file.read_text())
            except Exception:
                continue
            if not doc.get("termination") or doc.get("error"):
                continue
            split = TARGET_TASK_SPLIT.get(doc.get("task_name"), "other")
            gated = gate_episode(doc)
            erow = {
                "model": model,
                "arm": arm,
                "method": method,
                "split": split,
                "task": doc.get("task_name"),
                "episode_id": doc.get("episode_id"),
                "success": int(bool(doc.get("episode_success"))),
                "refined": int(bool(gated["refined"])),
                "steps": total_steps(doc),
                "n_turns": doc.get("n_turns") or len(doc.get("turns") or []),
                "termination": doc.get("termination"),
            }
            episodes.append(erow)
            for tr in doc.get("rule_interventions") or []:
                for iv in tr.get("interventions", []):
                    if iv.get("before") == iv.get("after"):
                        continue
                    rules.append(
                        {
                            **{k: erow[k] for k in ("model", "arm", "split", "task", "episode_id")},
                            "rule": iv.get("rule"),
                            "kind": iv.get("kind"),
                        }
                    )
            ep_dir = ep_file.parent
            for summary in doc.get("turns") or []:
                effective_est = summary.get("estimated_step")
                raw_est = effective_est
                raw_subgoal = summary.get("subgoal")
                if arm == "inst":
                    turn_file = ep_dir / str(summary.get("dir") or f"turn{summary.get('turn', 0):02d}") / "turn.json"
                    try:
                        td = json.loads(turn_file.read_text())
                        s2 = td.get("s2") or {}
                        if isinstance(s2.get("estimated_step"), int):
                            raw_est = s2["estimated_step"]
                        raw_subgoal = s2.get("subgoal") or raw_subgoal
                        eff = (td.get("rules") or {}).get("effective") or {}
                        if isinstance(eff.get("estimated_step"), int):
                            effective_est = eff["estimated_step"]
                    except Exception:
                        pass
                turns.append(
                    {
                        **{k: erow[k] for k in ("model", "arm", "split", "task", "episode_id")},
                        "turn": summary.get("turn"),
                        "judge": summary.get("judge"),
                        "subgoal": summary.get("subgoal"),
                        "raw_subgoal": raw_subgoal,
                        "action": action_class(raw_subgoal),
                        "continue": int((raw_subgoal or "").strip().lower().startswith("continue to ")),
                        "raw_est": raw_est,
                        "effective_est": effective_est,
                        "raw_est_bucket": est_bucket(raw_est),
                        "effective_est_bucket": est_bucket(effective_est),
                        "n_steps": summary.get("n_steps") or 0,
                        "budget": summary.get("budget"),
                        "stop_reason": summary.get("stop_reason"),
                    }
                )
    return episodes, turns, rules


def write_csv(name, rows):
    path = OUT / name
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def group(rows, **want):
    return [r for r in rows if all(r.get(k) == v for k, v in want.items())]


def compute_tables(episodes, turns, rules):
    split_summary = []
    behavior = []
    est_summary = []
    task_summary = []
    rule_summary = []
    for split in SPLITS:
        for model in COLORS:
            for arm in ("base", "inst"):
                es = group(episodes, split=split, model=model, arm=arm)
                ts = group(turns, split=split, model=model, arm=arm)
                n = len(es)
                split_summary.append(
                    {
                        "split": split,
                        "model": model,
                        "arm": arm,
                        "n": n,
                        "success": sum(r["success"] for r in es),
                        "success_rate": sum(r["success"] for r in es) / n,
                        "refined": sum(r["refined"] for r in es),
                        "refined_rate": sum(r["refined"] for r in es) / n,
                        "avg_steps": round(safe_mean([r["steps"] for r in es]), 2),
                        "avg_turns": round(safe_mean([r["n_turns"] for r in es]), 3),
                    }
                )
                raw = [r["raw_est"] for r in ts if isinstance(r["raw_est"], int)]
                eff = [r["effective_est"] for r in ts if isinstance(r["effective_est"], int)]
                est_summary.append(
                    {
                        "split": split,
                        "model": model,
                        "arm": arm,
                        "n_turns": len(ts),
                        "n_raw_est": len(raw),
                        "raw_mean": round(safe_mean(raw), 3),
                        "raw_median": percentile(raw, 0.5),
                        "raw_p25": percentile(raw, 0.25),
                        "raw_p75": percentile(raw, 0.75),
                        "effective_mean": round(safe_mean(eff), 3),
                        "effective_median": percentile(eff, 0.5),
                        "pct_raw_le50": round(sum(x <= 50 for x in raw) / len(raw), 5),
                        "pct_raw_ge100": round(sum(x >= 100 for x in raw) / len(raw), 5),
                    }
                )
                adjacent_n = adjacent_same = 0
                by_ep = collections.defaultdict(list)
                for r in ts:
                    by_ep[r["episode_id"]].append(r)
                for ers in by_ep.values():
                    ers.sort(key=lambda r: r["turn"])
                    for a, b in zip(ers, ers[1:]):
                        adjacent_n += 1
                        adjacent_same += norm_subgoal(a["raw_subgoal"]) == norm_subgoal(b["raw_subgoal"])
                judges = collections.Counter(r["judge"] for r in ts)
                terms = collections.Counter(r["termination"] for r in es)
                behavior.append(
                    {
                        "split": split,
                        "model": model,
                        "arm": arm,
                        "turns": len(ts),
                        "continue_rate": sum(r["continue"] for r in ts) / len(ts),
                        "adjacent_repeat_rate": adjacent_same / adjacent_n,
                        "incomplete_rate": judges["subgoal_incomplete"] / len(ts),
                        "failed_rate": judges["subgoal_failed"] / len(ts),
                        "budget_stop_rate": sum(r["stop_reason"] == "budget" for r in ts) / len(ts),
                        "max_cap_rate": terms["max_cap"] / n,
                        "max_turns_rate": terms["max_turns"] / n,
                        "task_finish_rate": terms["task_finish"] / n,
                    }
                )
        tasks = sorted({r["task"] for r in episodes if r["split"] == split})
        for task in tasks:
            row = {"split": split, "task": task}
            for model in COLORS:
                for arm in ("base", "inst"):
                    es = group(episodes, split=split, task=task, model=model, arm=arm)
                    row[f"{model}_{arm}_success"] = sum(r["success"] for r in es)
                    row[f"{model}_{arm}_refined"] = sum(r["refined"] for r in es)
                row[f"{model}_raw_gain"] = row[f"{model}_inst_success"] - row[f"{model}_base_success"]
                row[f"{model}_refined_gain"] = row[f"{model}_inst_refined"] - row[f"{model}_base_refined"]
            task_summary.append(row)
    for split in SPLITS:
        for model in COLORS:
            rs = group(rules, split=split, model=model, arm="inst")
            counts = collections.Counter(r["rule"] for r in rs)
            eps = collections.defaultdict(set)
            for r in rs:
                eps[r["rule"]].add(r["episode_id"])
            for rule, n in sorted(counts.items()):
                rule_summary.append(
                    {
                        "split": split,
                        "model": model,
                        "rule": rule,
                        "firings": n,
                        "episodes": len(eps[rule]),
                        "firings_per_100_episodes": round(n / 5.4 if split == "atomic_seen" else n / 4.8, 3),
                    }
                )
    return split_summary, behavior, est_summary, task_summary, rule_summary


def compute_diagnostics(episodes, turns):
    paired = []
    action_summary = []
    rule_effect = []
    calibration = []

    episode_index = {
        (r["model"], r["arm"], r["episode_id"]): r
        for r in episodes
    }
    comparisons = []
    for split in SPLITS:
        for model in COLORS:
            comparisons.append((split, f"{model}: rules", (model, "base"), (model, "inst")))
        comparisons.extend(
            [
                (split, "Base: Qwen3-VL vs Qwen3.5", ("Qwen3-VL", "base"), ("Qwen3.5", "base")),
                (split, "Inst: Qwen3-VL vs Qwen3.5", ("Qwen3-VL", "inst"), ("Qwen3.5", "inst")),
            ]
        )
    for split, comparison, left, right in comparisons:
        left_rows = group(episodes, split=split, model=left[0], arm=left[1])
        wins_left = wins_right = both_success = both_fail = 0
        for lr in left_rows:
            rr = episode_index[(right[0], right[1], lr["episode_id"])]
            pair = (lr["success"], rr["success"])
            wins_left += pair == (1, 0)
            wins_right += pair == (0, 1)
            both_success += pair == (1, 1)
            both_fail += pair == (0, 0)
        discordant = wins_left + wins_right
        pvalue = binomtest(min(wins_left, wins_right), discordant, 0.5).pvalue if discordant else 1.0
        paired.append(
            {
                "split": split,
                "comparison": comparison,
                "left": f"{left[0]} {left[1]}",
                "right": f"{right[0]} {right[1]}",
                "left_only_success": wins_left,
                "right_only_success": wins_right,
                "net_right_minus_left": wins_right - wins_left,
                "both_success": both_success,
                "both_fail": both_fail,
                "exact_p": round(pvalue, 7),
            }
        )

    for split in SPLITS:
        for model in COLORS:
            for action in ACTION_CLASSES:
                ts = group(turns, split=split, model=model, arm="base", action=action)
                vals = [r["raw_est"] for r in ts if isinstance(r["raw_est"], int)]
                if not vals:
                    continue
                action_summary.append(
                    {
                        "split": split,
                        "model": model,
                        "action": action,
                        "n": len(vals),
                        "mean_est": round(safe_mean(vals), 2),
                        "median_est": round(percentile(vals, 0.5), 2),
                        "mean_executed_steps": round(safe_mean([r["n_steps"] for r in ts]), 2),
                        "budget_hit_rate": round(sum(r["stop_reason"] == "budget" for r in ts) / len(ts), 5),
                    }
                )

            ts = group(turns, split=split, model=model, arm="inst")
            valid = [r for r in ts if isinstance(r["raw_est"], int) and isinstance(r["effective_est"], int)]
            deltas = [r["effective_est"] - r["raw_est"] for r in valid]
            rule_effect.append(
                {
                    "split": split,
                    "model": model,
                    "turns": len(ts),
                    "estimate_changed": sum(x != 0 for x in deltas),
                    "estimate_changed_rate": round(sum(x != 0 for x in deltas) / len(valid), 5),
                    "estimate_increased": sum(x > 0 for x in deltas),
                    "estimate_decreased": sum(x < 0 for x in deltas),
                    "mean_estimate_delta_all_turns": round(safe_mean(deltas), 3),
                    "mean_abs_delta_changed": round(safe_mean([abs(x) for x in deltas if x]), 3),
                    "subgoal_changed": sum(norm_subgoal(r["raw_subgoal"]) != norm_subgoal(r["subgoal"]) for r in ts),
                    "subgoal_changed_rate": round(sum(norm_subgoal(r["raw_subgoal"]) != norm_subgoal(r["subgoal"]) for r in ts) / len(ts), 5),
                }
            )

            base_ts = group(turns, split=split, model=model, arm="base")
            for bucket in EST_BUCKETS:
                bs = [r for r in base_ts if r["raw_est_bucket"] == bucket]
                if not bs:
                    continue
                hits = sum(r["stop_reason"] == "budget" for r in bs)
                calibration.append(
                    {
                        "split": split,
                        "model": model,
                        "bucket": bucket,
                        "n": len(bs),
                        "budget_hits": hits,
                        "budget_hit_rate": round(hits / len(bs), 5),
                    }
                )
    return paired, action_summary, rule_effect, calibration


def base_canvas(title, subtitle, width=1800, height=700):
    im = Image.new("RGBA", (width, height), "white")
    d = ImageDraw.Draw(im, "RGBA")
    d.text((35, 22), title, fill="#1f2937", font=font(30, True))
    d.text((35, 62), subtitle, fill="#5f6b7a", font=font(17))
    return im, d


def axes(d, box, ymin, ymax, yticks=5, percent=False):
    x0, y0, x1, y1 = box
    d.line((x0, y1, x1, y1), fill="#64748b", width=2)
    d.line((x0, y0, x0, y1), fill="#64748b", width=2)
    for i in range(yticks + 1):
        v = ymin + (ymax - ymin) * i / yticks
        y = y1 - (y1 - y0) * i / yticks
        d.line((x0, y, x1, y), fill="#e5e7eb", width=1)
        lab = f"{100*v:.0f}%" if percent else f"{v:.0f}"
        d.text((x0 - 10, y), lab, fill="#6b7280", font=font(13), anchor="rm")


def signed_axes(d, box, limit, yticks=4):
    """Draw a symmetric percentage-point axis and return its value-to-y mapper."""
    x0, y0, x1, y1 = box

    def y_for(value):
        return y1 - (value + limit) / (2 * limit) * (y1 - y0)

    d.line((x0, y0, x0, y1), fill="#64748b", width=2)
    for i in range(yticks + 1):
        value = -limit + 2 * limit * i / yticks
        y = y_for(value)
        is_zero = abs(value) < 1e-9
        d.line((x0, y, x1, y), fill="#64748b" if is_zero else "#e5e7eb", width=2 if is_zero else 1)
        d.text((x0 - 10, y), f"{100 * value:+.0f}", fill="#4b5563", font=font(13), anchor="rm")
    return y_for


def panel_boxes(width, top, bottom, n=3, left=80, right=35, gap=55):
    w = (width - left - right - gap * (n - 1)) / n
    return [(left + i * (w + gap), top, left + i * (w + gap) + w, bottom) for i in range(n)]


def legend(d, items, x, y):
    for label, color, alpha in items:
        d.rounded_rectangle((x, y, x + 24, y + 16), 3, fill=rgba(color, alpha), outline=rgba(color, 255))
        d.text((x + 32, y + 8), label, fill="#374151", font=font(15), anchor="lm")
        x += 32 + d.textlength(label, font=font(15)) + 28


def plot_success(split_summary):
    im, d = base_canvas(
        "Success by split: base vs instruction rules",
        "Solid bar = raw success; dark marker = refined success under the official task horizon",
    )
    legend(d, [("Qwen3-VL base", COLORS["Qwen3-VL"], 130), ("Qwen3-VL inst", COLORS["Qwen3-VL"], 255), ("Qwen3.5 base", COLORS["Qwen3.5"], 130), ("Qwen3.5 inst", COLORS["Qwen3.5"], 255)], 700, 30)
    boxes = panel_boxes(im.width, 130, 620)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 0.9, percent=True)
        d.text(((x0 + x1) / 2, y0 - 30), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        rows = [r for r in split_summary if r["split"] == split]
        order = [("Qwen3-VL", "base"), ("Qwen3-VL", "inst"), ("Qwen3.5", "base"), ("Qwen3.5", "inst")]
        bw = (x1 - x0) / 7
        for i, key in enumerate(order):
            r = next(x for x in rows if (x["model"], x["arm"]) == key)
            cx = x0 + (i + 1.4) * (x1 - x0) / 5
            top = y1 - r["success_rate"] / 0.9 * (y1 - y0)
            color = rgba(COLORS[key[0]], ARM_ALPHA[key[1]])
            d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=color, outline=rgba(COLORS[key[0]]), width=2)
            ry = y1 - r["refined_rate"] / 0.9 * (y1 - y0)
            d.line((cx - bw / 2, ry, cx + bw / 2, ry), fill="#111827", width=4)
            d.text((cx, y1 + 12), f"{key[0].replace('Qwen','Q')}\n{key[1]}", fill="#374151", font=font(12), anchor="ma", align="center")
            d.text((cx, top - 8), f"{100*r['success_rate']:.1f}", fill="#111827", font=font(13, True), anchor="mb")
    im.convert("RGB").save(OUT / "01_success_by_split.png", quality=95)


def grouped_distribution_plot(turns, arm, raw_field, filename, title, subtitle):
    im, d = base_canvas(title, subtitle, width=1900, height=720)
    legend(d, [("Qwen3-VL higher", COLORS["Qwen3-VL"], 255), ("Qwen3.5 higher", COLORS["Qwen3.5"], 255)], 1250, 32)
    boxes = panel_boxes(im.width, 135, 625, left=85, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        y_for = signed_axes(d, box, 0.08, yticks=4)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(EST_BUCKETS)
        for bi, bucket in enumerate(EST_BUCKETS):
            rates = {}
            for model in COLORS:
                ts = group(turns, split=split, model=model, arm=arm)
                vals = [r[raw_field] for r in ts if r[raw_field]]
                rates[model] = sum(v == bucket for v in vals) / len(vals)
            diff = rates["Qwen3-VL"] - rates["Qwen3.5"]
            cx = x0 + (bi + 0.5) * slot
            bw = slot * 0.56
            zero_y, value_y = y_for(0), y_for(diff)
            color = COLORS["Qwen3-VL"] if diff >= 0 else COLORS["Qwen3.5"]
            d.rectangle((cx - bw / 2, min(zero_y, value_y), cx + bw / 2, max(zero_y, value_y)), fill=rgba(color), outline=rgba(color))
            if abs(diff) >= 0.002:
                d.text((cx, value_y + (-5 if diff >= 0 else 5)), f"{100 * diff:+.1f}", fill=rgba(color), font=font(10, True), anchor="mb" if diff >= 0 else "ma")
            d.text((x0 + (bi + 0.5) * slot, y1 + 8), bucket, fill="#4b5563", font=font(11), anchor="ma")
        d.text(((x0 + x1) / 2, y1 + 42), "S2 estimated length bucket · difference in percentage points", fill="#4b5563", font=font(13), anchor="mm")
    im.convert("RGB").save(OUT / filename, quality=95)


def grouped_distribution_paired_plot(turns, arm, raw_field, filename):
    im, d = base_canvas(
        "Base arm: raw S2 estimated-length distribution · paired values",
        "Percent of turns in each estimate bucket; distributions are normalized separately inside each split.",
        width=1900,
        height=720,
    )
    legend(d, [(m, COLORS[m], 255) for m in COLORS], 1320, 32)
    boxes = panel_boxes(im.width, 135, 625, left=85, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 0.7, percent=True, yticks=7)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(EST_BUCKETS)
        for bi, bucket in enumerate(EST_BUCKETS):
            for mi, model in enumerate(COLORS):
                ts = group(turns, split=split, model=model, arm=arm)
                vals = [r[raw_field] for r in ts if r[raw_field]]
                rate = sum(v == bucket for v in vals) / len(vals)
                cx = x0 + (bi + 0.5) * slot + (mi - 0.5) * slot * 0.3
                bw = slot * 0.26
                top = y1 - rate / 0.7 * (y1 - y0)
                d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=rgba(COLORS[model]))
                if rate >= 0.008:
                    d.text((cx, top - 4), f"{100 * rate:.1f}", fill=rgba(COLORS[model]), font=font(9, True), anchor="mb")
            d.text((x0 + (bi + 0.5) * slot, y1 + 8), bucket, fill="#4b5563", font=font(11), anchor="ma")
        d.text(((x0 + x1) / 2, y1 + 42), "S2 estimated length bucket", fill="#4b5563", font=font(14), anchor="mm")
    im.convert("RGB").save(OUT / filename, quality=95)


def plot_inst_shift(turns):
    im, d = base_canvas(
        "Instruction arm: raw S2 estimate vs rule-effective estimate",
        "Each panel is normalized within model and split. Blue/orange = S2 prediction; gray outline = estimate sent to S1 after rules.",
        width=1900,
        height=1120,
    )
    boxes = []
    left, top, pw, ph, gx, gy = 90, 150, 545, 390, 55, 115
    for row in range(2):
        for col in range(3):
            boxes.append((left + col * (pw + gx), top + row * (ph + gy), left + col * (pw + gx) + pw, top + row * (ph + gy) + ph))
    for row, model in enumerate(COLORS):
        for col, split in enumerate(SPLITS):
            box = boxes[row * 3 + col]
            x0, y0, x1, y1 = box
            axes(d, box, 0, 0.6, percent=True, yticks=6)
            d.text(((x0 + x1) / 2, y0 - 27), f"{model} · {SPLIT_LABEL[split]}", fill="#111827", font=font(18, True), anchor="mm")
            ts = group(turns, split=split, model=model, arm="inst")
            raw = [r["raw_est_bucket"] for r in ts if r["raw_est_bucket"]]
            eff = [r["effective_est_bucket"] for r in ts if r["effective_est_bucket"]]
            slot = (x1 - x0) / len(EST_BUCKETS)
            points_raw, points_eff = [], []
            for bi, bucket in enumerate(EST_BUCKETS):
                cx = x0 + (bi + 0.5) * slot
                pr = sum(v == bucket for v in raw) / len(raw)
                pe = sum(v == bucket for v in eff) / len(eff)
                points_raw.append((cx, y1 - pr / 0.6 * (y1 - y0)))
                points_eff.append((cx, y1 - pe / 0.6 * (y1 - y0)))
                d.text((cx, y1 + 8), bucket, fill="#4b5563", font=font(10), anchor="ma")
            d.line(points_raw, fill=rgba(COLORS[model]), width=4)
            d.line(points_eff, fill="#111827", width=3)
            for p in points_raw:
                d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=rgba(COLORS[model]))
            for p in points_eff:
                d.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill="white", outline="#111827", width=2)
    legend(d, [("raw S2 prediction", COLORS["Qwen3-VL"], 255), ("effective after rules", "#111827", 255)], 1230, 32)
    im.convert("RGB").save(OUT / "03_inst_raw_vs_effective_est.png", quality=95)


def plot_action_est(turns):
    im, d = base_canvas(
        "Base arm: mean predicted estimate by action family",
        "Action families are inferred from the raw S2 subgoal text; every split is shown separately.",
        width=2000,
        height=760,
    )
    legend(d, [(m, COLORS[m], 255) for m in COLORS], 1430, 32)
    boxes = panel_boxes(im.width, 140, 645, left=90, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 150, yticks=6)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(ACTION_CLASSES)
        for ai, action in enumerate(ACTION_CLASSES):
            for mi, model in enumerate(COLORS):
                ts = group(turns, split=split, model=model, arm="base", action=action)
                vals = [r["raw_est"] for r in ts if isinstance(r["raw_est"], int)]
                if not vals:
                    continue
                mean = statistics.mean(vals)
                cx = x0 + (ai + 0.5) * slot + (mi - 0.5) * slot * 0.28
                bw = slot * 0.24
                top = y1 - min(mean, 150) / 150 * (y1 - y0)
                d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=rgba(COLORS[model]), outline=rgba(COLORS[model]))
            label = action.replace("/", "/\n")
            d.text((x0 + (ai + 0.5) * slot, y1 + 7), label, fill="#4b5563", font=font(10), anchor="ma", align="center")
    im.convert("RGB").save(OUT / "04_est_by_action_and_split.png", quality=95)


def plot_judges(turns):
    judges = ("task_begin", "subgoal_complete", "subgoal_incomplete", "subgoal_failed", "task_finish")
    labels = ("begin", "complete", "incomplete", "failed", "finish")
    im, d = base_canvas(
        "Base arm: S2 judge distribution by split",
        "Rates are fractions of execution turns. Composite Qwen3.5 almost never emits subgoal_failed.",
        width=1800,
        height=720,
    )
    legend(d, [(m, COLORS[m], 255) for m in COLORS], 1290, 32)
    boxes = panel_boxes(im.width, 135, 620, left=85, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 0.8, percent=True, yticks=4)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(judges)
        for ji, judge in enumerate(judges):
            for mi, model in enumerate(COLORS):
                ts = group(turns, split=split, model=model, arm="base")
                p = sum(r["judge"] == judge for r in ts) / len(ts)
                cx = x0 + (ji + 0.5) * slot + (mi - 0.5) * slot * 0.3
                bw = slot * 0.26
                top = y1 - p / 0.8 * (y1 - y0)
                d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=rgba(COLORS[model]))
            d.text((x0 + (ji + 0.5) * slot, y1 + 8), labels[ji], fill="#4b5563", font=font(11), anchor="ma")
    im.convert("RGB").save(OUT / "05_judge_distribution_by_split.png", quality=95)


def plot_behavior(behavior):
    metrics = (
        ("adjacent_repeat_rate", "adjacent repeat"),
        ("incomplete_rate", "judge incomplete"),
        ("failed_rate", "judge failed"),
        ("max_cap_rate", "max-cap deaths"),
        ("max_turns_rate", "max-turn deaths"),
    )
    im, d = base_canvas(
        "Base arm: Qwen3-VL minus Qwen3.5 repeat/judge/termination rates",
        "Signed percentage-point difference. Positive (blue) = Qwen3-VL higher; negative (orange) = Qwen3.5 higher.",
        width=1850,
        height=740,
    )
    legend(d, [("Qwen3-VL higher", COLORS["Qwen3-VL"], 255), ("Qwen3.5 higher", COLORS["Qwen3.5"], 255)], 1260, 32)
    boxes = panel_boxes(im.width, 140, 630, left=90, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        y_for = signed_axes(d, box, 0.2, yticks=4)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(metrics)
        for qi, (key, label) in enumerate(metrics):
            q3 = next(x for x in behavior if x["split"] == split and x["model"] == "Qwen3-VL" and x["arm"] == "base")
            q35 = next(x for x in behavior if x["split"] == split and x["model"] == "Qwen3.5" and x["arm"] == "base")
            diff = q3[key] - q35[key]
            cx = x0 + (qi + 0.5) * slot
            bw = slot * 0.54
            zero_y, value_y = y_for(0), y_for(diff)
            color = COLORS["Qwen3-VL"] if diff >= 0 else COLORS["Qwen3.5"]
            d.rectangle((cx - bw / 2, min(zero_y, value_y), cx + bw / 2, max(zero_y, value_y)), fill=rgba(color))
            d.text((cx, value_y + (-6 if diff >= 0 else 6)), f"{100 * diff:+.1f}", fill=rgba(color), font=font(12, True), anchor="mb" if diff >= 0 else "ma")
            d.text((x0 + (qi + 0.5) * slot, y1 + 8), label.replace(" ", "\n"), fill="#4b5563", font=font(10), anchor="ma", align="center")
    im.convert("RGB").save(OUT / "06_loop_behavior_by_split.png", quality=95)


def plot_behavior_paired(behavior):
    metrics = (
        ("adjacent_repeat_rate", "adjacent repeat"),
        ("incomplete_rate", "judge incomplete"),
        ("failed_rate", "judge failed"),
        ("max_cap_rate", "max-cap deaths"),
        ("max_turns_rate", "max-turn deaths"),
    )
    im, d = base_canvas(
        "Base arm: repeat/judge/termination rates · paired values",
        "Adjacent-repeat and judge rates are turn-level; max-cap and max-turn rates are episode-level.",
        width=1850,
        height=740,
    )
    legend(d, [(m, COLORS[m], 255) for m in COLORS], 1330, 32)
    boxes = panel_boxes(im.width, 140, 630, left=90, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 0.5, percent=True, yticks=5)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(metrics)
        for qi, (key, label) in enumerate(metrics):
            for mi, model in enumerate(COLORS):
                row = next(x for x in behavior if x["split"] == split and x["model"] == model and x["arm"] == "base")
                rate = row[key]
                cx = x0 + (qi + 0.5) * slot + (mi - 0.5) * slot * 0.3
                bw = slot * 0.26
                top = y1 - rate / 0.5 * (y1 - y0)
                d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=rgba(COLORS[model]))
                d.text((cx, top - 4), f"{100 * rate:.1f}", fill=rgba(COLORS[model]), font=font(9, True), anchor="mb")
            d.text((x0 + (qi + 0.5) * slot, y1 + 8), label.replace(" ", "\n"), fill="#4b5563", font=font(10), anchor="ma", align="center")
    im.convert("RGB").save(OUT / "06b_loop_behavior_by_split_paired.png", quality=95)


def plot_task_gains(task_summary):
    for split in SPLITS:
        rows = [r for r in task_summary if r["split"] == split]
        rows.sort(key=lambda r: max(abs(r["Qwen3-VL_raw_gain"]), abs(r["Qwen3.5_raw_gain"])), reverse=True)
        width, row_h = 1500, 38
        height = 145 + row_h * len(rows)
        im, d = base_canvas(
            f"Rule gain by task · {SPLIT_LABEL[split]}",
            "Raw success delta (instruction-rules minus base), 30 episodes per task. Positive is right of zero.",
            width=width,
            height=height,
        )
        legend(d, [(m, COLORS[m], 255) for m in COLORS], 1040, 32)
        xzero, scale, y = 900, 32, 120
        d.line((xzero, 105, xzero, height - 20), fill="#374151", width=2)
        for tick in range(-12, 13, 4):
            x = xzero + tick * scale
            d.line((x, 105, x, height - 20), fill="#e5e7eb", width=1)
            d.text((x, 100), str(tick), fill="#6b7280", font=font(12), anchor="mb")
        for r in rows:
            d.text((25, y + 9), r["task"], fill="#1f2937", font=font(14), anchor="lm")
            for mi, model in enumerate(COLORS):
                val = r[f"{model}_raw_gain"]
                yy = y + mi * 15
                x = xzero + val * scale
                d.line((xzero, yy, x, yy), fill=rgba(COLORS[model]), width=10)
                d.ellipse((x - 5, yy - 5, x + 5, yy + 5), fill=rgba(COLORS[model]))
                d.text((x + (8 if val >= 0 else -8), yy), f"{val:+d}", fill=rgba(COLORS[model]), font=font(12, True), anchor="lm" if val >= 0 else "rm")
            y += row_h
        idx = SPLITS.index(split) + 7
        im.convert("RGB").save(OUT / f"{idx:02d}_task_rule_gain_{split}.png", quality=95)


def plot_est_calibration(calibration):
    im, d = base_canvas(
        "Base arm: estimate bucket vs segment budget-hit rate",
        "A budget hit is a segment ending because its allocated control-step budget was exhausted; lower is better calibrated.",
        width=1900,
        height=720,
    )
    legend(d, [(m, COLORS[m], 255) for m in COLORS], 1360, 32)
    boxes = panel_boxes(im.width, 135, 620, left=85, gap=65)
    for split, box in zip(SPLITS, boxes):
        x0, y0, x1, y1 = box
        axes(d, box, 0, 0.4, percent=True, yticks=4)
        d.text(((x0 + x1) / 2, y0 - 28), SPLIT_LABEL[split], fill="#111827", font=font(20, True), anchor="mm")
        slot = (x1 - x0) / len(EST_BUCKETS)
        for model in COLORS:
            pts = []
            for bi, bucket in enumerate(EST_BUCKETS):
                row = next((r for r in calibration if r["split"] == split and r["model"] == model and r["bucket"] == bucket), None)
                # Very long estimates can have only a handful of observations. Do not
                # draw those as zero-rate evidence; retain them in calibration.csv.
                if row and row["n"] >= 20:
                    p = row["budget_hit_rate"]
                    pts.append((x0 + (bi + 0.5) * slot, y1 - min(p, 0.4) / 0.4 * (y1 - y0)))
                d.text((x0 + (bi + 0.5) * slot, y1 + 8), bucket, fill="#4b5563", font=font(10), anchor="ma")
            if len(pts) > 1:
                d.line(pts, fill=rgba(COLORS[model]), width=4)
            for p in pts:
                d.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=rgba(COLORS[model]))
    im.convert("RGB").save(OUT / "10_estimate_budget_hit_calibration.png", quality=95)


def write_report(episodes, turns, split_summary, behavior, est_summary, task_summary, rule_summary, paired, action_summary, rule_effect):
    lines = ["# Qwen3-VL vs Qwen3.5 split-first analysis", ""]
    lines.append("All results use the same 1,500-episode manifest and are always separated into atomic-seen, composite-seen, and composite-unseen.")
    lines += ["", "## Split summary", "", "| Split | Model | Arm | Raw | Refined | Avg steps | Avg turns |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for model in COLORS:
            for arm in ("base", "inst"):
                r = next(x for x in split_summary if x["split"] == split and x["model"] == model and x["arm"] == arm)
                lines.append(f"| {SPLIT_LABEL[split]} | {model} | {arm} | {r['success']}/{r['n']} ({100*r['success_rate']:.1f}%) | {r['refined']}/{r['n']} ({100*r['refined_rate']:.1f}%) | {r['avg_steps']:.0f} | {r['avg_turns']:.2f} |")
    lines += ["", "## Estimate summary", "", "| Split | Model | Arm | Raw mean | Median | P25–P75 | ≤50 | ≥100 | Effective mean |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for model in COLORS:
            for arm in ("base", "inst"):
                r = next(x for x in est_summary if x["split"] == split and x["model"] == model and x["arm"] == arm)
                lines.append(f"| {SPLIT_LABEL[split]} | {model} | {arm} | {r['raw_mean']:.1f} | {r['raw_median']:.0f} | {r['raw_p25']:.0f}–{r['raw_p75']:.0f} | {100*r['pct_raw_le50']:.1f}% | {100*r['pct_raw_ge100']:.1f}% | {r['effective_mean']:.1f} |")
    lines += ["", "## Behavior summary (base)", "", "| Split | Model | Continue | Adjacent repeat | Incomplete | Failed | Max-cap death | Max-turn death |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        for model in COLORS:
            r = next(x for x in behavior if x["split"] == split and x["model"] == model and x["arm"] == "base")
            lines.append(f"| {SPLIT_LABEL[split]} | {model} | {100*r['continue_rate']:.1f}% | {100*r['adjacent_repeat_rate']:.1f}% | {100*r['incomplete_rate']:.1f}% | {100*r['failed_rate']:.2f}% | {100*r['max_cap_rate']:.1f}% | {100*r['max_turns_rate']:.1f}% |")
    lines += ["", "## Paired episode transitions", "", "Right-only and left-only successes compare the exact same task episode/seed. The exact p-value tests the discordant pairs.", "", "| Split | Comparison | Left only | Right only | Net right | Exact p |", "|---|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        for r in [x for x in paired if x["split"] == split]:
            lines.append(f"| {SPLIT_LABEL[split]} | {r['comparison']} | {r['left_only_success']} | {r['right_only_success']} | {r['net_right_minus_left']:+d} | {r['exact_p']:.4g} |")
    lines += ["", "## Rule impact on planner output", "", "This separates the raw S2 output from the estimate/subgoal actually sent to S1.", "", "| Split | Model | Estimate changed | Mean Δ estimate | Subgoal changed |", "|---|---|---:|---:|---:|"]
    for split in SPLITS:
        for model in COLORS:
            r = next(x for x in rule_effect if x["split"] == split and x["model"] == model)
            lines.append(f"| {SPLIT_LABEL[split]} | {model} | {r['estimate_changed']}/{r['turns']} ({100*r['estimate_changed_rate']:.1f}%) | {r['mean_estimate_delta_all_turns']:+.1f} | {r['subgoal_changed']}/{r['turns']} ({100*r['subgoal_changed_rate']:.1f}%) |")

    lines += ["", "## Base estimate by action family", "", "Mean raw S2 estimate; parenthesized value is the number of turns.", "", "| Split | Model | Reach | Grasp | Carry | Place | Actuate | Navigate | Retract | Wait |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    action_keys = (("reach", "Reach"), ("grasp/pick", "Grasp"), ("carry/lift", "Carry"), ("place/release", "Place"), ("actuate", "Actuate"), ("navigate/base", "Navigate"), ("retract", "Retract"), ("wait", "Wait"))
    for split in SPLITS:
        for model in COLORS:
            cells = []
            for action, _ in action_keys:
                r = next((x for x in action_summary if x["split"] == split and x["model"] == model and x["action"] == action), None)
                cells.append(f"{r['mean_est']:.1f} ({r['n']})" if r else "—")
            lines.append(f"| {SPLIT_LABEL[split]} | {model} | " + " | ".join(cells) + " |")
    lines += ["", "## Figures", ""]
    for name in (
        "01_success_by_split.png",
        "02_base_s2_estimate_distribution.png",
        "02b_base_s2_estimate_distribution_paired.png",
        "03_inst_raw_vs_effective_est.png",
        "04_est_by_action_and_split.png",
        "05_judge_distribution_by_split.png",
        "06_loop_behavior_by_split.png",
        "06b_loop_behavior_by_split_paired.png",
        "07_task_rule_gain_atomic_seen.png",
        "08_task_rule_gain_composite_seen.png",
        "09_task_rule_gain_composite_unseen.png",
        "10_estimate_budget_hit_calibration.png",
    ):
        lines.append(f"- [{name}]({name})")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    episodes, turns, rules = load_data()
    write_csv("episodes.csv", episodes)
    write_csv("turns.csv", turns)
    write_csv("rule_firings.csv", rules)
    tables = compute_tables(episodes, turns, rules)
    split_summary, behavior, est_summary, task_summary, rule_summary = tables
    paired, action_summary, rule_effect, calibration = compute_diagnostics(episodes, turns)
    write_csv("split_summary.csv", split_summary)
    write_csv("behavior_summary.csv", behavior)
    write_csv("estimate_summary.csv", est_summary)
    write_csv("task_summary.csv", task_summary)
    write_csv("rule_summary.csv", rule_summary)
    write_csv("paired_summary.csv", paired)
    write_csv("action_summary.csv", action_summary)
    write_csv("rule_effect_summary.csv", rule_effect)
    write_csv("estimate_calibration.csv", calibration)
    plot_success(split_summary)
    grouped_distribution_plot(
        turns,
        "base",
        "raw_est_bucket",
        "02_base_s2_estimate_distribution.png",
        "Base arm: Qwen3-VL minus Qwen3.5 estimated-length distribution",
        "Signed percentage-point difference per bucket. Positive (blue) = Qwen3-VL higher; negative (orange) = Qwen3.5 higher.",
    )
    grouped_distribution_paired_plot(
        turns, "base", "raw_est_bucket", "02b_base_s2_estimate_distribution_paired.png"
    )
    plot_inst_shift(turns)
    plot_action_est(turns)
    plot_judges(turns)
    plot_behavior(behavior)
    plot_behavior_paired(behavior)
    plot_task_gains(task_summary)
    plot_est_calibration(calibration)
    write_report(episodes, turns, split_summary, behavior, est_summary, task_summary, rule_summary, paired, action_summary, rule_effect)
    print(f"loaded {len(episodes)} episodes, {len(turns)} turns, {len(rules)} effective interventions")
    print(f"wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
