"""Eval #3 — Gemini subtask-success judge.

For each subtask rollout (subtask_eval.py output: <root>/<method>/<episode>/child<NN>_<prim>/
clean.mp4, a 3-view side-by-side stack), ask Gemini to decide whether the policy COMPLETED the
subgoal. We show Gemini BOTH:
  * the GROUND-TRUTH oracle clip (same subtask, recorded human demo replayed by the oracle run),
  * the POLICY rollout clip,
so it judges the rollout by comparison to the reference rather than in a vacuum (per the user).
Each clean.mp4 already concatenates the three camera views (scene_left|scene_right|wrist).

Writes <root>/<method>/<episode>/child<NN>_<prim>/gemini.json = {success, reason, confidence}.
Reuses RoboAnnotator's video-capable Gemini backend (robo_annotator.vlm_backend._gemini_generate).

Run (env with google-genai + RoboAnnotator importable):
    ROBOANNOTATOR=/home/yinpeidai/RoboAnnotator \
    python scripts/gemini_judge_subtasks.py \
        --rollout-root subtask_rollouts --method reg --oracle-method oracle \
        [--gemini-api-key ... | uses Vertex ADC by default]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# The Gemini video backend lives in RoboAnnotator; make it importable.
_ROBOANNO = os.environ.get("ROBOANNOTATOR", "/home/yinpeidai/RoboAnnotator")
if _ROBOANNO not in sys.path:
    sys.path.insert(0, _ROBOANNO)

from robo_annotator.vlm_backend import _gemini_generate  # noqa: E402

# gemini-3.x pro (matches the model that produced the subgoal annotations) — required
# because RoboAnnotator's shared _gemini_config pins thinking_level=HIGH, which only
# gemini-3 models accept (gemini-2.5-pro returns 400 INVALID_ARGUMENT). gemini-3.x is
# GLOBAL/Vertex-only, so the default backend is Vertex ADC (no --gemini-api-key).
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_PROJECT = "gen-lang-client-0648980768"
DEFAULT_LOCATION = "global"

# Structured verdict schema (google-genai response_schema).
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "confidence": {"type": "number"},        # 0..1
        "reason": {"type": "string"},
    },
    "required": ["success", "confidence", "reason"],
}


def _judge_prompt(subgoal: str, task_goal: str, primitive: str) -> str:
    return (
        "You are judging whether a robot completed a single sub-task in a kitchen manipulation "
        "environment. Each video is a side-by-side stack of THREE camera views "
        "(left scene | right scene | wrist).\n\n"
        f"Whole-task goal: {task_goal}\n"
        f"Sub-task to judge: \"{subgoal}\" (primitive: {primitive})\n\n"
        "The FIRST video is the GROUND-TRUTH reference (a successful human demonstration of this "
        "exact sub-task). The SECOND video is the ROBOT POLICY's attempt from the same start state.\n\n"
        "Decide whether the POLICY video (the second one) SUCCESSFULLY accomplished the sub-task, "
        "using the reference video only to understand what success looks like. Judge the end state / "
        "effect on objects, not stylistic differences. Return JSON: success (bool), confidence (0..1), "
        "and a one-sentence reason."
    )


def _find_oracle_clip(rollout_root: Path, oracle_method: str, episode: str, child_dir: str) -> Path | None:
    p = rollout_root / oracle_method / episode / child_dir / "clean.mp4"
    return p if p.is_file() else None


def judge_episode(rollout_root: Path, method: str, oracle_method: str, ep_dir: Path, args) -> list[dict]:
    ep_doc = json.loads((ep_dir / "episode.json").read_text())
    episode = ep_dir.name
    results = []
    for sg in ep_doc.get("subgoals", []):
        child_dir = Path(sg["out_dir"]).name
        sub_dir = ep_dir / child_dir
        rollout_clip = sub_dir / "clean.mp4"
        if not rollout_clip.is_file():
            print(f"  [skip] no rollout clip: {sub_dir}")
            continue
        out_f = sub_dir / "gemini.json"
        if out_f.is_file() and not args.overwrite:
            results.append({**json.loads(out_f.read_text()), "child_dir": child_dir, "cached": True})
            continue

        oracle_clip = _find_oracle_clip(rollout_root, oracle_method, episode, child_dir)
        prompt = _judge_prompt(sg.get("subgoal", ""), ep_doc.get("instruction", ""), sg.get("primitive", ""))
        # Ordered parts: prompt, GT (oracle) clip FIRST, rollout clip SECOND (matches the prompt).
        parts = [prompt]
        if oracle_clip is not None:
            parts.append(oracle_clip)
        else:
            parts[0] = prompt + "\n\n(No reference video available; judge the single policy video below.)"
        parts.append(rollout_clip)

        try:
            verdict = _gemini_generate(
                parts, VERDICT_SCHEMA, args.model,
                project=args.project, location=args.location,
                api_key=args.gemini_api_key, max_tokens=2048)
        except Exception as e:
            traceback.print_exc()
            verdict = {"success": None, "confidence": 0.0, "reason": f"gemini error: {e!r}"}
        verdict["child_dir"] = child_dir
        verdict["had_reference"] = oracle_clip is not None
        out_f.write_text(json.dumps(verdict, indent=2))
        results.append(verdict)
        print(f"  {child_dir}: success={verdict.get('success')} conf={verdict.get('confidence')} "
              f"ref={oracle_clip is not None}")
    # Aggregate into episode.json (gemini_success per subtask + task-level rate).
    ok = [r for r in results if r.get("success") is True]
    ep_doc["gemini"] = dict(
        n_judged=len(results), n_success=len(ok),
        success_rate=(len(ok) / len(results) if results else None),
        per_subtask={r["child_dir"]: {"success": r.get("success"), "confidence": r.get("confidence")}
                     for r in results})
    (ep_dir / "episode.json").write_text(json.dumps(ep_doc, indent=1))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-root", type=Path, required=True)
    ap.add_argument("--method", required=True, help="policy rollout method dir to judge (e.g. reg)")
    ap.add_argument("--oracle-method", default="oracle", help="GT reference method dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    ap.add_argument("--gemini-api-key", default=None,
                    help="AI Studio key (else Vertex ADC via --project/--location)")
    ap.add_argument("--overwrite", action="store_true", help="re-judge even if gemini.json exists")
    args = ap.parse_args()

    method_root = args.rollout_root / args.method
    ep_dirs = sorted(p.parent for p in method_root.rglob("episode.json"))
    if not ep_dirs:
        raise SystemExit(f"No episodes with episode.json under {method_root}")
    print(f"judging {len(ep_dirs)} episodes under {method_root} (oracle={args.oracle_method})")
    for ep_dir in ep_dirs:
        print(f"=== {ep_dir.name} ===")
        judge_episode(args.rollout_root, args.method, args.oracle_method, ep_dir, args)
    print("done.")


if __name__ == "__main__":
    main()
