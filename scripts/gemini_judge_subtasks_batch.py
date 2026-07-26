"""Eval #3 (BATCH) — Gemini subtask-success judge via VERTEX GCS BATCH.

Same judgement as scripts/gemini_judge_subtasks.py (Gemini sees the GT oracle clip FIRST +
the policy rollout clip SECOND, decides if the sub-task succeeded), but scaled through the
Vertex batch API (50% price, async) instead of one sync call per subtask. Reuses the proven
batch primitives in robo_annotator.vlm_backend (gemini_batch_jsonl_line / gemini_submit_vertex_batch
/ parse_vertex_batch_output) — the SAME path producers/run_subgoal_batch_gcs.py uses.

Pipeline:
  STAGE A  build   — one judge request per (method, episode, subtask): prompt + oracle clip +
                     rollout clip + verdict schema. Keyed "<method>||<episode>||<child_dir>".
  STAGE B  upload  — bulk `gcloud storage cp -I` every unique clip to gs://BUCKET/<run>/clips/.
  STAGE C  submit  — chunked JSONL (clips as gs:// fileData refs) -> GCS; submit Vertex batch job(s).
  STAGE D  poll    — poll to SUCCEEDED, parse output BY KEY, write gemini.json per subtask +
                     aggregate a `gemini` block into each episode.json (parity with the sync judge).
  STAGE E  cleanup — delete the run's GCS prefix (unless --keep-clips).

Sync DEBUG mode (--debug-sync N): judge the FIRST N subtasks via the Standard API and PRINT the
full prompt + verdict, so you can eyeball that prompts/outputs are reasonable BEFORE paying for
the batch. No GCS, no batch job.

Run (robocasa env; ROBOANNOTATOR on path; Vertex ADC creds):
    PY=/home/yinpei.dai/micromamba/envs/robocasa/bin/python
    ROBOANNOTATOR=/home/yinpei.dai/RoboAnnotator $PY scripts/gemini_judge_subtasks_batch.py \
        --rollout-root subtask_rollouts_big --oracle-method oracle \
        --methods v1_progcls,v2_progact,... \
        --bucket gs://roboannotator-clips-gen-lang-0648980768 \
        --model gemini-3.6-flash [--debug-sync 3] [--keep-clips]
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROBOANNO = os.environ.get("ROBOANNOTATOR", "/home/yinpei.dai/RoboAnnotator")
if _ROBOANNO not in sys.path:
    sys.path.insert(0, _ROBOANNO)

from robo_annotator import vlm_backend as VB  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_PROJECT = "gen-lang-client-0648980768"
DEFAULT_LOCATION = "global"   # gemini-3.x previews are global-only

# Ordered fields: the model FIRST describes each video and REASONS about completion (a "think
# before you answer" analysis), and ONLY THEN commits to a verdict. JSON property order is
# preserved in the response, so `reason` (the analysis) comes BEFORE `verdict` — the verdict is a
# conclusion of the reasoning, not a snap answer. `completion` records the degree of completion so
# partial attempts are distinguished from clean successes and complete failures.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "oracle_description": {"type": "string"},    # what the robot does in the GT reference video
        "rollout_description": {"type": "string"},    # what the robot does in the policy rollout video
        "reason": {"type": "string"},                 # analysis FIRST: did the policy do it correctly,
                                                       # partially, or not at all — reason before deciding
        "completion": {"type": "string", "enum": ["complete", "partial", "none"]},
        "verdict": {"type": "string", "enum": ["success", "failure", "uncertain"]},
        "confidence": {"type": "number"},
    },
    "required": ["oracle_description", "rollout_description", "reason",
                 "completion", "verdict", "confidence"],
}


def _judge_prompt(subgoal: str, task_goal: str) -> str:
    return (
        "You are judging whether a robot completed a single sub-task in a kitchen manipulation "
        "environment.\n\n"
        "CAMERA VIEWS: each video frame is THREE views of the SAME scene at the SAME instant, placed "
        "side by side (each pane 256x256, so the frame is 768 wide). Left pane = left shoulder "
        "camera; middle pane = right shoulder camera; right pane = the robot's wrist (gripper) "
        "camera. These are three camera ANGLES of one scene, NOT three different scenes: do not "
        "double-count objects, and do not infer an object's location from which pane it appears in. "
        "Use the wrist view for close-up gripper/contact detail and the two shoulder views for the "
        "overall scene.\n\n"
        f"Whole-task goal: {task_goal}\n"
        f"Sub-task to judge: \"{subgoal}\"\n\n"
        "The FIRST video is the GROUND-TRUTH reference (a successful human demonstration of this "
        "exact sub-task). The SECOND video is the ROBOT POLICY's attempt from the same start state. "
        "Judge the end state / effect on objects, not stylistic differences or speed.\n\n"
        "Think it through in THIS ORDER, and return JSON with these fields in this order:\n"
        "  1. oracle_description  — describe what the robot does in the FIRST (reference) video and "
        "the end state it reaches.\n"
        "  2. rollout_description — describe what the robot does in the SECOND (policy) video and the "
        "end state it reaches.\n"
        "  3. reason — REASON about whether the policy did the sub-task CORRECTLY: did it fully "
        "achieve the same effect as the reference, only PARTIALLY finish it (e.g. moved toward / "
        "touched but did not complete the state change), or NOT accomplish it at all? Compare the "
        "two end states explicitly.\n"
        "  4. completion — \"complete\" (fully achieved), \"partial\" (attempted, some progress, but "
        "not fully done), or \"none\" (no meaningful progress).\n"
        "  5. verdict — \"success\" ONLY if completion is clearly complete; \"failure\" if it "
        "partially finished or clearly did not accomplish it; \"uncertain\" if you genuinely cannot "
        "tell (ambiguous/occluded view) or cannot see/decode one of the videos.\n"
        "  6. confidence — 0..1.\n\n"
        "Base everything ONLY on what you actually observe in the two videos — do NOT guess from the "
        "text description. If you cannot see a video, say so in its description and return uncertain."
    )


# key <-> its subtask dir + parts, so we can finalize by key after the batch returns.
class _Item:
    __slots__ = ("key", "method", "episode", "child_dir", "sub_dir", "parts",
                 "oracle_clip", "rollout_clip", "had_reference", "eef_disp",
                 "subgoal", "skip_reason")


# Movement gate: skip Gemini for subtasks where the gripper barely moves over the whole span.
# For these (e.g. a "grasp" where the arm is already at the object and only the fingers twitch),
# the two clips look nearly identical and Gemini cannot reliably tell success from failure — a
# wasted, low-confidence query. We measure the ROLLOUT's net end-effector displacement (straight-
# line start→end of eef_pos_world in steps.npz). Net displacement (not path length) separates the
# low-signal cases cleanly: e.g. CloseBlenderLid grasp ~0.025 m vs a faucet turn ~0.052 m / a
# drawer pull ~0.44 m. Threshold is conservative so real (small-but-real) motions still get judged.
MOVE_GATE_EEF_M = 0.04


def _rollout_eef_displacement(sub_dir: Path) -> float | None:
    """Net straight-line eef travel over the rollout span (meters), from steps.npz. None if
    unavailable."""
    npz = sub_dir / "steps.npz"
    if not npz.is_file():
        return None
    try:
        import numpy as _np
        z = _np.load(npz)
        ep = z["eef_pos_world"].astype(float)
        if ep.shape[0] < 2:
            return 0.0
        return float(_np.linalg.norm(ep[-1] - ep[0]))
    except Exception:
        return None


def _collect_items(rollout_root: Path, methods: list[str], oracle_method: str,
                   only_episodes: set[str] | None = None,
                   move_gate: float = MOVE_GATE_EEF_M) -> tuple[list[_Item], list[_Item]]:
    """Return (items_to_judge, skipped_items). One judge item per (method, episode, subtask) that
    has a rollout clip. Subtasks whose rollout eef displacement < ``move_gate`` are SKIPPED (not
    sent to Gemini) and returned separately so the caller can record a 'skipped_low_movement'
    verdict. ``only_episodes``: restrict to episode-flat names containing any token."""
    items: list[_Item] = []
    skipped: list[_Item] = []
    for method in methods:
        mroot = rollout_root / method
        if not mroot.is_dir():
            print(f"  [skip method] no dir: {mroot}")
            continue
        for ep_json in sorted(mroot.rglob("episode.json")):
            ep_dir = ep_json.parent
            episode = ep_dir.name
            if only_episodes and not any(tok in episode for tok in only_episodes):
                continue
            try:
                ep_doc = json.loads(ep_json.read_text())
            except Exception as e:  # noqa: BLE001
                print(f"  [skip ep] {ep_dir}: {e}")
                continue
            for sg in ep_doc.get("subgoals", []):
                child_dir = Path(sg["out_dir"]).name
                sub_dir = ep_dir / child_dir
                rollout_clip = sub_dir / "clean.mp4"
                if not rollout_clip.is_file():
                    continue
                oracle_clip = rollout_root / oracle_method / episode / child_dir / "clean.mp4"
                had_ref = oracle_clip.is_file()
                prompt = _judge_prompt(sg.get("subgoal", ""), ep_doc.get("instruction", ""))
                parts = [prompt]
                if had_ref:
                    parts.append(oracle_clip)
                else:
                    parts[0] = prompt + "\n\n(No reference video available; judge the single policy video below.)"
                parts.append(rollout_clip)
                it = _Item()
                it.key = f"{method}||{episode}||{child_dir}"
                it.method, it.episode, it.child_dir, it.sub_dir = method, episode, child_dir, sub_dir
                it.parts, it.oracle_clip, it.rollout_clip, it.had_reference = parts, oracle_clip, rollout_clip, had_ref
                it.subgoal = sg.get("subgoal", "")
                it.eef_disp = _rollout_eef_displacement(sub_dir)
                it.skip_reason = None
                # Skip "retract the arm ..." subtasks: pure repositioning-away moves with no object/
                # fixture state change, so there is nothing for a video judge to verify.
                if "retract" in it.subgoal.lower():
                    it.skip_reason = "retract"
                elif it.eef_disp is not None and it.eef_disp < move_gate:
                    it.skip_reason = "low_movement"
                (skipped if it.skip_reason else items).append(it)
    return items, skipped


# Structured, SELF-CONTAINED artifact layout per subtask, all UNDER <sub_dir>/judge/ (never touches
# the eval-results data). Everything needed to reproduce/inspect one judge query lives here:
#   judge/inputs/prompt.txt              -- the prompt string
#   judge/inputs/schema.json             -- the verdict response schema
#   judge/inputs/oracle.mp4              -- COPY of the reference clip actually sent (if any)
#   judge/inputs/rollout.mp4             -- COPY of the policy clip actually sent
#   judge/inputs/request.json            -- manifest: prompt + local video filenames + gs uris
#   judge/response.txt                   -- raw model text
#   judge/response_meta.json             -- model / usage / per-query cost / finish_reason
#   judge/gemini.json                    -- the parsed verdict (what /subtask + /stats read)
JUDGE_DIRNAME = "judge"


def _write_request_artifacts(it: _Item, model: str, clip_uri: dict | None = None):
    """STAGE A: persist the judge INPUTS (no network) so judge/ is SELF-CONTAINED and inspectable.
    Copies the oracle + rollout clips INTO judge/inputs/ (so the folder has its own videos, not just
    paths elsewhere). clip_uri (batch only) maps each clip Path -> its gs:// uri; None for sync."""
    import shutil
    art = it.sub_dir / JUDGE_DIRNAME / "inputs"
    art.mkdir(parents=True, exist_ok=True)
    (art / "prompt.txt").write_text(it.parts[0])
    (art / "schema.json").write_text(json.dumps(VERDICT_SCHEMA, indent=2))
    # copy the actual clips in, under stable local names
    local_names = {}
    if it.had_reference and it.oracle_clip.is_file():
        shutil.copyfile(it.oracle_clip, art / "oracle.mp4")
        local_names[it.oracle_clip] = "oracle.mp4"
    if it.rollout_clip.is_file():
        shutil.copyfile(it.rollout_clip, art / "rollout.mp4")
        local_names[it.rollout_clip] = "rollout.mp4"
    parts_manifest = []
    for p in it.parts:
        if isinstance(p, Path):
            parts_manifest.append({"type": "video",
                                   "role": ("oracle_reference" if p == it.oracle_clip else "policy_rollout"),
                                   "local_file": local_names.get(p),
                                   "source_path": str(p),
                                   "gs_uri": (clip_uri or {}).get(p)})
        else:
            parts_manifest.append({"type": "text", "text": p})
    (art / "request.json").write_text(json.dumps({
        "key": it.key, "model": model, "method": it.method,
        "episode": it.episode, "child_dir": it.child_dir,
        "had_reference": it.had_reference,
        "rollout_eef_displacement_m": it.eef_disp,
        "parts": parts_manifest,
    }, indent=2))
    return art


def _finalize_item(it: _Item, parsed: dict, *, model: str, raw: str | None = None,
                   usage: dict | None = None, finish_reason: str = "", backend: str = "vertex-batch"):
    """Write ALL judge output UNDER <sub_dir>/judge/ — never touches the eval-results data
    (no root gemini.json, no episode.json mutation). Writes judge/gemini.json (parsed verdict),
    judge/response.txt (raw), judge/response_meta.json (usage/cost/finish)."""
    jdir = it.sub_dir / JUDGE_DIRNAME
    jdir.mkdir(parents=True, exist_ok=True)
    verdict = dict(parsed)
    verdict["child_dir"] = it.child_dir
    verdict["had_reference"] = it.had_reference
    (jdir / "gemini.json").write_text(json.dumps(verdict, indent=2))
    if raw is not None:
        (jdir / "response.txt").write_text(raw)
    # Always write response_meta.json with a per-query cost. `backend` sets the price tier:
    # "vertex-batch" -> 50% batch price; "vertex-sync" -> full price. Even if usage tokens are
    # missing we still write the block (cost fields zero) so the file is always present.
    u = usage or {}
    norm = {"prompt_token_count": u.get("promptTokenCount", u.get("prompt_token_count")),
            "candidates_token_count": u.get("candidatesTokenCount", u.get("candidates_token_count")),
            "total_token_count": u.get("totalTokenCount", u.get("total_token_count"))}
    is_batch = (backend == "vertex-batch")
    meta = {"model": model, "backend": backend, "key": it.key,
            "usage": norm, "cost": VB._cost_meta(model, norm, batch=is_batch),
            "finish_reason": finish_reason or "STOP"}
    (jdir / "response_meta.json").write_text(json.dumps(meta, indent=2))


def _write_method_summary(rollout_root: Path, method: str):
    """Roll up a method's per-subtask verdicts into a STANDALONE summary file, WITHOUT
    touching any eval data. Written to <rollout_root>/<method>/gemini_summary.json.
    Three-way verdict counts (success/failure/uncertain); success_rate is over the
    DECIDED subtasks (success / (success+failure)), with uncertain reported separately.
    Reads verdicts from each subtask's judge/gemini.json."""
    mroot = rollout_root / method
    per_episode = {}
    tot = {"success": 0, "failure": 0, "uncertain": 0}
    for ep_json in sorted(mroot.rglob("episode.json")):
        ep_dir = ep_json.parent
        try:
            ep_doc = json.loads(ep_json.read_text())
        except Exception:
            continue
        per = {}
        cnt = {"success": 0, "failure": 0, "uncertain": 0}
        for sg in ep_doc.get("subgoals", []):
            cd = Path(sg["out_dir"]).name
            gf = ep_dir / cd / JUDGE_DIRNAME / "gemini.json"
            if not gf.is_file():
                continue
            g = json.loads(gf.read_text())
            v = g.get("verdict")
            per[cd] = {"verdict": v, "confidence": g.get("confidence")}
            if v in cnt:
                cnt[v] += 1
        n = sum(cnt.values())
        if n:
            decided = cnt["success"] + cnt["failure"]
            per_episode[ep_dir.name] = dict(
                n_judged=n, **{f"n_{k}": cnt[k] for k in cnt},
                success_rate=(cnt["success"] / decided if decided else None),
                per_subtask=per)
            for k in tot:
                tot[k] += cnt[k]
    n_tot = sum(tot.values())
    decided_tot = tot["success"] + tot["failure"]
    (mroot / "gemini_summary.json").write_text(json.dumps(dict(
        method=method, n_judged=n_tot,
        **{f"n_{k}": tot[k] for k in tot},
        success_rate=(tot["success"] / decided_tot if decided_tot else None),
        success_rate_incl_uncertain=(tot["success"] / n_tot if n_tot else None),
        per_episode=per_episode), indent=1))


# Verdict written (without a Gemini call) for subtasks the gate skips: "retract"-type moves (no
# state change to verify) and low-movement subtasks (clips near-identical → judge unreliable).
def _write_skipped(it: _Item):
    """Record a 'skipped' verdict for a gated subtask — self-contained under judge/, no API call,
    no cost. Copies the clips in too so the folder is inspectable."""
    _write_request_artifacts(it, f"(none: skipped {it.skip_reason})")
    if it.skip_reason == "retract":
        reason = ("Skipped Gemini: this is a 'retract the arm' sub-task — a pure repositioning-away "
                  "move with no object/fixture state change, so there is nothing for a video judge "
                  "to verify.")
    else:
        reason = (f"Skipped Gemini: the gripper barely moved during this sub-task "
                  f"(net eef displacement {it.eef_disp:.4f} m < {MOVE_GATE_EEF_M} m gate), so the "
                  f"reference and rollout clips are nearly identical and a video judge cannot "
                  f"reliably distinguish success from failure.")
    verdict = {
        "oracle_description": "", "rollout_description": "", "reason": reason,
        "completion": "none", "verdict": "uncertain", "confidence": 0.0,
        "child_dir": it.child_dir, "had_reference": it.had_reference,
        "skipped": True, "skip_reason": it.skip_reason, "eef_displacement_m": it.eef_disp,
    }
    jdir = it.sub_dir / JUDGE_DIRNAME
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "gemini.json").write_text(json.dumps(verdict, indent=2))
    (jdir / "response_meta.json").write_text(json.dumps(
        {"model": None, "backend": f"skipped_{it.skip_reason}", "key": it.key,
         "usage": None, "cost": {"pricing_known": True, "total_cost_usd": 0.0},
         "eef_displacement_m": it.eef_disp}, indent=2))


# ----------------------------------------------------------------------------- SYNC DEBUG
def _sync_judge_one(it: _Item, args):
    """One direct Vertex sync call for a single item; returns (verdict_dict, raw_text, usage_dict).
    Calls the model directly (not VB._gemini_generate) so we capture usage_metadata for cost."""
    import google.genai as g
    from google.genai import types
    client = g.Client(vertexai=True, project=args.project, location=args.location)
    resp = client.models.generate_content(
        model=args.model, contents=VB._parts_to_contents(it.parts, types),
        config=VB._gemini_config(types, VERDICT_SCHEMA, 2048))
    raw = resp.text or "{}"
    um = getattr(resp, "usage_metadata", None)
    usage = None
    if um is not None:
        usage = {"prompt_token_count": getattr(um, "prompt_token_count", None),
                 "candidates_token_count": getattr(um, "candidates_token_count", None),
                 "total_token_count": getattr(um, "total_token_count", None)}
    return json.loads(raw), raw, usage


def _debug_sync(items: list[_Item], skipped: list[_Item], n: int, args):
    print(f"\n===== SYNC DEBUG: {min(n, len(items))} judged + {len(skipped)} gated (retract / low movement) =====\n")
    for it in skipped[:n]:
        disp = f"{it.eef_disp:.4f} m" if it.eef_disp is not None else "n/a"
        print(f"--- [GATED:{it.skip_reason}] {it.key}  (eef_disp={disp}) -> skipped, no query")
        if args.write:
            _write_skipped(it)
    for it in items[:n]:
        print(f"--- {it.key}  (eef_disp={it.eef_disp}) ---")
        if args.write:
            art = _write_request_artifacts(it, args.model)   # judge/inputs/{prompt,schema,videos,request}
            print(f"  wrote inputs -> {art}")
        try:
            verdict, raw, usage = _sync_judge_one(it, args)
            cost = VB._cost_meta(args.model, {"prompt_token_count": (usage or {}).get("prompt_token_count"),
                                              "candidates_token_count": (usage or {}).get("candidates_token_count")},
                                 batch=False)
            print(f"  VERDICT: completion={verdict.get('completion')} verdict={verdict.get('verdict')} "
                  f"conf={verdict.get('confidence')}  cost=${cost.get('total_cost_usd')}")
            print(f"    reason: {verdict.get('reason','')[:120]}")
            if args.write:
                _finalize_item(it, verdict, model=args.model, raw=raw,
                               usage=usage, finish_reason="STOP", backend="vertex-sync")
                print(f"  wrote verdict -> {it.sub_dir / JUDGE_DIRNAME / 'gemini.json'}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e!r}")
        print()


# ----------------------------------------------------------------------------- BATCH
def _gcs_bulk_upload(local_to_uri: dict, stage_dir: Path) -> None:
    """Upload many local files via one `gcloud storage cp -I` (parallel, in-process).
    Copied from producers/run_subgoal_batch_gcs.py: -I flattens into one dest prefix, so
    we stage uniquely-named symlinks and upload those."""
    items = list(local_to_uri.items())
    if not items:
        return
    prefixes = {uri.rsplit("/", 1)[0] for uri in local_to_uri.values()}
    if len(prefixes) != 1:
        raise ValueError(f"_gcs_bulk_upload expects ONE dest prefix, got {len(prefixes)}")
    dest_prefix = prefixes.pop() + "/"
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_dir / "_manifest.txt"
    with manifest.open("w") as mf:
        for lp, uri in items:
            name = uri.rsplit("/", 1)[1]
            link = stage_dir / name
            try:
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(Path(lp).resolve())
            except OSError:
                continue
            mf.write(str(link) + "\n")
    with manifest.open() as mf:
        subprocess.run(["gcloud", "storage", "cp", "-I", dest_prefix],
                       stdin=mf, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def _run_batch(items: list[_Item], args):
    bucket = args.bucket.rstrip("/")
    run_id = f"judge-{int(time.time())}"
    base = f"{bucket}/{run_id}"
    print(f"[batch] bucket={bucket} run={run_id} items={len(items)} model={args.model}")
    t0 = time.time()

    # STAGE B: map each clip to a unique GCS URI (basename unique across items via index).
    # Both oracle + rollout clips upload; identical oracle clips across methods dedupe by path.
    clip_uri_global = {}   # local Path -> gs uri (dedup by local path)
    per_item_clips = []    # list of {Path -> uri} for the item's clips
    for i, it in enumerate(items):
        m = {}
        for role, p in (("o", it.oracle_clip if it.had_reference else None), ("r", it.rollout_clip)):
            if p is None:
                continue
            p = Path(p)
            if p not in clip_uri_global:
                clip_uri_global[p] = f"{base}/clips/{i:05d}_{role}_{p.name}"
            m[p] = clip_uri_global[p]
        per_item_clips.append(m)
    print(f"[batch] uploading {len(clip_uri_global)} unique clips to GCS ...")
    stage_dir = Path(f"/tmp/{run_id}_clipstage")
    _gcs_bulk_upload(clip_uri_global, stage_dir)
    print(f"[batch] upload done in {time.time()-t0:.0f}s")

    # STAGE C: chunked JSONL -> GCS; submit concurrent Vertex batch jobs.
    client = VB.gemini_vertex_client(args.project, args.location)
    chunk = max(1, args.chunk_size)
    idx_chunks = [list(range(i, min(i + chunk, len(items)))) for i in range(0, len(items), chunk)]
    print(f"[batch] {len(items)} items -> {len(idx_chunks)} chunk job(s) of <= {chunk}")
    jobs = []  # (job_name, [i...], out_prefix)
    for k, idxs in enumerate(idx_chunks):
        lines = [VB.gemini_batch_jsonl_line(items[i].parts, VERDICT_SCHEMA,
                                            per_item_clips[i], key=items[i].key) for i in idxs]
        local_jsonl = Path(f"/tmp/{run_id}_chunk{k:03d}.jsonl")
        local_jsonl.write_text("\n".join(lines) + "\n")
        in_uri = f"{base}/chunk{k:03d}/input.jsonl"
        out_prefix = f"{base}/chunk{k:03d}/output/"
        subprocess.run(["gcloud", "storage", "cp", str(local_jsonl), in_uri],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        job = VB.gemini_submit_vertex_batch(client, args.model, input_gcs_uri=in_uri,
                                            output_gcs_uri=out_prefix, display_name=f"{run_id}-c{k:03d}")
        jobs.append((job.name, idxs, out_prefix))
        print(f"  chunk {k}: {len(idxs)} items -> {job.name} ({getattr(job.state,'name',job.state)})")

    # STAGE D: poll all jobs; finalize each chunk as it succeeds.
    ok = 0
    failed = []  # (i, reason)
    in_tok = out_tok = 0
    done_jobs = set()
    touched_methods = set()   # methods to summarize at the end (standalone, no eval mutation)
    t_poll = time.time()
    while len(done_jobs) < len(jobs):
        for jn, idxs, op in jobs:
            if jn in done_jobs:
                continue
            st = getattr(client.batches.get(name=jn).state, "name", "?")
            if st not in VB._BATCH_DONE:
                continue
            done_jobs.add(jn)
            if st != "JOB_STATE_SUCCEEDED":
                for i in idxs:
                    failed.append((i, f"chunk job {st}"))
                print(f"  [{time.time()-t_poll:5.0f}s] {jn} {st} -> {len(idxs)} items failed")
                continue
            listing = subprocess.run(["gcloud", "storage", "ls", op + "**"],
                                     capture_output=True, text=True).stdout.split()
            by_key = {}
            for u in (x for x in listing if x.endswith(".jsonl")):
                txt = subprocess.run(["gcloud", "storage", "cat", u],
                                     capture_output=True, text=True).stdout
                by_key.update(VB.parse_vertex_batch_output(txt))
            c_ok = c_fail = 0
            for i in idxs:
                row = by_key.get(items[i].key)
                if not row or "__error__" in row:
                    failed.append((i, (row or {}).get("__error__", "no row for key")))
                    c_fail += 1
                    continue
                try:
                    u = row.get("usage") or {}
                    _finalize_item(items[i], row["parsed"], model=args.model, raw=row.get("raw"),
                                   usage=u, finish_reason=row.get("finish_reason", ""))
                    in_tok += u.get("promptTokenCount", u.get("prompt_token_count", 0)) or 0
                    out_tok += u.get("candidatesTokenCount", u.get("candidates_token_count", 0)) or 0
                    touched_methods.add(items[i].method)
                    ok += 1
                    c_ok += 1
                except Exception as e:  # noqa: BLE001
                    failed.append((i, f"{type(e).__name__}: {e}"))
                    c_fail += 1
            print(f"  [{time.time()-t_poll:5.0f}s] {jn} SUCCEEDED -> {c_ok} ok, {c_fail} fail "
                  f"({len(done_jobs)}/{len(jobs)} chunks)")
        if len(done_jobs) < len(jobs):
            time.sleep(args.poll_s)

    # aggregate per-episode gemini blocks
    for method in sorted(touched_methods):
        _write_method_summary(Path(args.rollout_root), method)

    print(f"[batch] FINAL: {ok}/{len(items)} judged, {len(failed)} failed")
    if failed:
        for i, r in failed[:40]:
            print(f"    FAIL {items[i].key}  <- {str(r)[:80]}")
        if args.failures_out:
            Path(args.failures_out).write_text("\n".join(items[i].key for i, _ in failed) + "\n")
            print(f"[batch] wrote {len(failed)} failed keys -> {args.failures_out}")

    # cost (batch 50%)
    base_cost = VB._cost_meta(args.model, {"prompt_token_count": in_tok // max(1, ok),
                                           "candidates_token_count": out_tok // max(1, ok)})
    if base_cost.get("pricing_known"):
        ic = in_tok / 1e6 * base_cost["input_usd_per_mtok"] * 0.5
        oc = out_tok / 1e6 * base_cost["output_usd_per_mtok"] * 0.5
        print(f"[batch] COST (batch 50%): ${ic+oc:.4f} (in {in_tok:,} + out {out_tok:,} tok)")
    else:
        print(f"[batch] pricing unknown for {args.model}; tokens in={in_tok:,} out={out_tok:,}")

    import shutil as _shutil
    _shutil.rmtree(stage_dir, ignore_errors=True)
    if args.keep_clips:
        print(f"[batch] --keep-clips: LEAVING {base}/")
    else:
        subprocess.run(["gcloud", "storage", "rm", "-r", base + "/"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[batch] deleted run prefix {base}/")
    print(f"[batch] done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-root", type=Path, required=True)
    ap.add_argument("--methods", required=True, help="comma-separated policy method dirs to judge")
    ap.add_argument("--oracle-method", default="oracle")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--location", default=DEFAULT_LOCATION)
    ap.add_argument("--bucket", default="gs://roboannotator-clips-gen-lang-0648980768")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--keep-clips", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip subtasks that already have gemini.json (idempotent restart)")
    ap.add_argument("--failures-out", default="")
    ap.add_argument("--move-gate", type=float, default=MOVE_GATE_EEF_M,
                    help="skip Gemini for subtasks whose rollout net eef displacement (m) is below "
                         "this — the clips are near-identical and the video judge is unreliable. "
                         "Set 0 to disable the gate.")
    ap.add_argument("--only-episodes", default="",
                    help="comma-separated tokens; restrict judging to episodes whose flat name "
                         "CONTAINS any token (e.g. a task name 'DeliverStraw' or a full episode dir). "
                         "Handy for a targeted sample.")
    ap.add_argument("--gemini-api-key", default=None, help="only for --debug-sync (AI Studio key)")
    ap.add_argument("--write", action="store_true",
                    help="in --debug-sync, ALSO persist judge/inputs + judge/gemini.json to disk "
                         "(under each subtask's judge/ subdir) so you can inspect the artifacts")
    ap.add_argument("--debug-sync", type=int, default=0,
                    help="judge the first N subtasks via the sync API and PRINT prompt+verdict "
                         "(no GCS/batch). Use to sanity-check before paying for the batch.")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    only_eps = {t.strip() for t in args.only_episodes.split(",") if t.strip()} or None
    items, skipped = _collect_items(args.rollout_root, methods, args.oracle_method,
                                    only_episodes=only_eps, move_gate=args.move_gate)
    if args.skip_existing:
        items = [it for it in items if not (it.sub_dir / JUDGE_DIRNAME / "gemini.json").is_file()]
        skipped = [it for it in skipped if not (it.sub_dir / JUDGE_DIRNAME / "gemini.json").is_file()]
    print(f"[judge] {len(items)} subtasks to judge + {len(skipped)} gated (low movement, "
          f"<{args.move_gate} m eef) across methods {methods}")
    if not items and not skipped:
        raise SystemExit("nothing to judge")

    if args.debug_sync > 0:
        _debug_sync(items, skipped, args.debug_sync, args)
        return
    # Record the gated subtasks (no API call), then batch-judge the rest.
    for it in skipped:
        _write_skipped(it)
    if items:
        _run_batch(items, args)
    else:
        print("[judge] all subtasks were gated; nothing sent to Gemini.")


if __name__ == "__main__":
    main()
