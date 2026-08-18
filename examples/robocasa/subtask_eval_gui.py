"""
Standalone Flask GUI to browse subtask-eval rollouts faithfully.

Reads the structured rollout tree written by subtask_eval.py:
  <rollout-root>/<method>/<episode_flat>/child<NN>_<primitive>/{clean.mp4, anchor_*.jpg,
                                                                steps.npz + steps_meta.json}
Per-step logs ship as a compact steps.npz (fp16 arrays) + a small steps_meta.json sidecar
(subtask_eval._write_steps_npz). This GUI reconstructs the full per-step doc server-side
(numpy is available here), so /api/steps returns the same JSON shape the old steps.json had;
legacy steps.json is still read if present.

Full-page layout (borrows /system1_training_sample + /system2_prompt): a top two-lane track
(milestones + subgoals, current subtask highlighted), then a grid showing the clean rollout
video (with play + frame-by-frame prev/next / arrow keys), the language prompt, anchor images,
raw + normalized anchor/current state, the full predicted action chunk, progress, and the
executed-vs-oracle action — all synced to the current frame.

Run both the rollout browser and live human-interactive evaluator:
    bash examples/robocasa/run_sys1_eval_gui.sh \
        --port 8092 --s1-port 8060 --s2-port 8100
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, send_file

app = Flask(__name__)

# Derived path roots, defined BEFORE first use (MSE_DIRS below needs them). Nothing is hardcoded
# to one machine: env contract first, else the first existing "data" dir near this repo. Keeps both
# the shared-filesystem layout and the classic "repo + sibling data/" layout working with no flags.
_OPENPI_REPO = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(os.environ.get("REPO_ROOT") or _OPENPI_REPO.parent).expanduser()
_DATA = Path(os.environ.get("DATA_DIR") or next(
    (c for c in (_REPO_ROOT / "data", _OPENPI_REPO.parent / "data", Path.home() / "data")
     if c.is_dir()), _REPO_ROOT / "data")).expanduser()
_RESULTS = Path(os.environ.get("SYS1_RESULTS_DIR") or _DATA / "sys1_eval_results").expanduser()

ROOT: Path = Path(".")                        # default / legacy subtask rollout root
# Named rollout trees, both written in the SAME layout by subtask_eval.py (#3) and
# episode_eval.py (#2), so ONE viewer serves both — selected by the <root> path segment.
ROOTS: dict[str, Path] = {}
VAL_MSE_DIR: Path = Path("eval_out/val_mse")  # eval #1 curves (scripts/eval_val_mse.py output)
# Train/val MSE sweep result dirs (per-ckpt JSON from scripts/run_val_sweep.py). Each holds
# <exp>__<step>.json = {exp_name, steps:[{step, action_mse, progress_acc, progress_mae,
# progress_mode}]}. The /val_mse page overlays both splits with per-split + per-method toggles.
MSE_DIRS: dict[str, Path] = {
    "val": _RESULTS / "valmse_results",
    "train": _RESULTS / "trainmse_results",
}


def _root(name: str | None) -> Path:
    """Resolve a named rollout root (falls back to the legacy single ROOT)."""
    if name and name in ROOTS:
        return ROOTS[name]
    return ROOT


def _methods(root: Path):
    return sorted([d.name for d in root.iterdir() if d.is_dir() and (d / "index.json").exists()]) \
        if root.exists() else []


def _episodes(root: Path, method):
    md = root / method
    return sorted([d.name for d in md.iterdir() if d.is_dir() and (d / "episode.json").exists()]) \
        if md.is_dir() else []


def _safe(root: Path, method, episode=None, sub=None) -> Path:
    p = root / method
    if episode:
        p = p / episode
    if sub:
        p = p / sub
    p = p.resolve()
    # Path-traversal guard: resolved path must stay under root (compare with a trailing sep so a
    # sibling like "<root>_evil" cannot pass a naive prefix check).
    base = str(root.resolve())
    if p != root.resolve() and not str(p).startswith(base + "/"):
        abort(403)
    return p


# ---- root-namespaced API (root = "subtask" | "episode"; falls back to legacy ROOT) ----
@app.route("/api/<root>/methods")
def api_methods_r(root):
    rp = _root(root)
    out = []
    for m in _methods(rp):
        idx = json.loads((rp / m / "index.json").read_text())
        out.append(dict(method=m, n_episodes=idx.get("n_episodes"),
                        horizon_mult=idx.get("horizon_mult"), settle_steps=idx.get("settle_steps"),
                        eval_kind=idx.get("eval_kind")))
    return jsonify(out)


@app.route("/api/<root>/episodes/<method>")
def api_episodes_r(root, method):
    rp = _root(root)
    out = []
    for ep in _episodes(rp, method):
        out.append(_adapt_milestone_doc(json.loads((_safe(rp, method, ep) / "episode.json").read_text())))
    return jsonify(out)


@app.route("/api/<root>/episode_list/<method>")
def api_episode_list_r(root, method):
    """LIGHTWEIGHT episode list for the dropdown — reads ONLY index.json (1 file) instead of every
    episode.json (~1000 files / 10 MB). Returns just what the type/task/episode cascade needs;
    the full per-episode doc (subgoals) is fetched lazily on click via /api/<root>/episode/..."""
    rp = _root(root)
    idx_f = rp / method / "index.json"
    by_id: dict[str, dict] = {}
    # 1) index.json entries (fast; carries n_subgoals). May be PARTIAL if a re-run overwrote it with
    #    only its shard — so we ALWAYS also union the directory scan below.
    if idx_f.is_file():
        try:
            idx = json.loads(idx_f.read_text())
            for e in idx.get("episodes", []):
                eid = e.get("episode_id") or ""
                if not eid:
                    continue   # errored/incomplete episode (no episode_id) — skip so the GUI never
                               # fetches /episode/<method>/ with an empty id (404). The dir scan
                               # below still surfaces any episode that produced an on-disk dir.
                parts = eid.split("/")
                task = parts[2] if len(parts) > 2 else eid   # RoboCasa/<Atomic|Composite>/<Task>/...
                by_id[eid] = dict(episode_id=eid, task_name=task,
                                  n_subgoals=e.get("n_subgoals"), error=e.get("error"))
        except Exception:
            pass
    # 2) directory scan (still NO episode.json reads) — add any episode dir the index missed, so a
    #    truncated/partial index (mid re-run) never hides episodes that exist on disk.
    for ep in _episodes(rp, method):
        eid = ep.replace("__", "/")
        if eid in by_id:
            continue
        parts = eid.split("/")
        by_id[eid] = dict(episode_id=eid, task_name=(parts[2] if len(parts) > 2 else eid),
                          n_subgoals=None, error=None)
    return jsonify(sorted(by_id.values(), key=lambda e: e["episode_id"]))


def _adapt_milestone_doc(doc: dict) -> dict:
    """MILESTONE episode.json stores `milestones[]` (goal_primitive / milestone_sim_check / ref), but
    the viewer JS expects the fine-step/episode `subgoals[]` shape (child_index / primitive / subgoal
    / span / out_dir / is_terminal). Synthesize `subgoals` from `milestones` so ONE viewer serves all
    three roots unchanged; keep the milestone verdict on each so the panel can show it."""
    if "subgoals" in doc or "milestones" not in doc:
        return doc
    subs = []
    ms = doc.get("milestones", [])
    ep_ss = doc.get("sim_success_final")   # episode-level env _check_success (top-level in the doc)
    for i, m in enumerate(ms):
        term = m.get("is_terminal", False) or (i == len(ms) - 1)
        subs.append(dict(
            child_index=m.get("milestone_index", i),
            milestone_index=m.get("milestone_index", i),
            is_terminal=term,
            span=m.get("span", [0, 0]),
            primitive=m.get("goal_primitive", "other"),
            subgoal=m.get("milestone_subgoal", ""),
            subgoal_detail="; ".join(m.get("child_subgoals", []) or []),
            milestone_subgoal=m.get("milestone_subgoal", ""),
            out_dir=m.get("out_dir", ""),
            n_children=m.get("n_children"), n_advanced=m.get("n_advanced"),
            milestone_sim_check=m.get("milestone_sim_check"),
            # surface the episode-level env _check_success on the TERMINAL milestone so the panel's
            # 'sim_check_success (episode)' shows SUCCESS/FAIL there (not everywhere -> only terminal).
            sim_success_final=(ep_ss if term else None),
        ))
    doc = dict(doc)
    doc["subgoals"] = subs
    return doc


@app.route("/api/<root>/episode/<method>/<episode>")
def api_episode_one_r(root, method, episode):
    """Full per-episode doc (subgoals + all fields) for ONE episode — fetched lazily when the user
    selects an episode, so the page load doesn't read 1000 episode.json up front."""
    f = _safe(_root(root), method, episode) / "episode.json"
    if not f.is_file():
        abort(404)
    return jsonify(_adapt_milestone_doc(json.loads(f.read_text())))


@app.route("/api/task_splits")
def api_task_splits():
    """task_name -> target-split label (atomic_seen / composite_seen / composite_unseen), so the
    viewer can group the task dropdown by type."""
    return jsonify(_target_split_map())


# ---- legacy non-namespaced routes (default to the finestep root) ----
@app.route("/api/methods")
def api_methods():
    return api_methods_r("finestep")


@app.route("/api/episodes/<method>")
def api_episodes(method):
    return api_episodes_r("finestep", method)


# NOTE: no non-namespaced /api/episode_list or /api/episode/<m>/<e> aliases — they COLLIDE with the
# namespaced /api/<root>/episode_list/<method> (a URL like /api/episode/episode_list/<m> matches BOTH
# and Werkzeug picks the wrong one -> 404). The JS always uses the namespaced /api/<root>/... forms.


def _reconstruct_full(sub_dir: Path) -> dict | None:
    """Rebuild the FULL per-step doc from steps.npz + steps_meta.json (the inverse of
    subtask_eval._write_steps_npz), so /api/steps returns the shape the GUI's JS expects
    ({**meta, "steps": [...]}). Falls back to a legacy monolithic steps.json. None if absent."""
    meta_f = sub_dir / "steps_meta.json"
    npz_f = sub_dir / "steps.npz"
    if not (meta_f.is_file() and npz_f.is_file()):
        legacy = sub_dir / "steps.json"
        return json.loads(legacy.read_text()) if legacy.is_file() else None
    meta = json.loads(meta_f.read_text())
    z = np.load(npz_f)
    n = int(z["frame_step"].shape[0])
    rp_by_step = {int(r["step"]): r for r in meta.get("replan", [])}
    qstep = [int(x) for x in z["q_step"]] if "q_step" in z else []
    q_pos = {s: k for k, s in enumerate(qstep)}
    steps = []
    for i in range(n):
        nm = z["action_norms"][i]
        fs = int(z["frame_step"][i])
        s = dict(
            frame_step=fs,
            phase="act" if int(z["phase"][i]) else "settle",
            replanned=bool(z["replanned"][i]),
            sim_check_success=bool(z["sim_check_success"][i]),
            cur_lean_norm=z["cur_lean_norm"][i].astype(float).round(4).tolist(),
            cur_lean=z["cur_lean"][i].astype(float).round(4).tolist(),
            cur_raw16=z["cur_raw16"][i].astype(float).round(4).tolist(),
            action_raw12=z["action_raw12"][i].astype(float).round(4).tolist(),
            oracle_action_raw12=z["oracle_action_raw12"][i].astype(float).round(4).tolist(),
            eef_pos_world=z["eef_pos_world"][i].astype(float).round(4).tolist(),
            gripper_width=round(float(z["gripper_width"][i]), 4),
            action_mse_vs_oracle=round(float(z["action_mse_vs_oracle"][i]), 6),
            action_eef_pos_norm=round(float(nm[0]), 4),
            action_eef_rot_norm=round(float(nm[1]), 4),
            action_base_norm=round(float(nm[2]), 4),
        )
        ps = float(z["progress_scalar"][i])
        s["progress"] = "-" if np.isnan(ps) else f"{ps:.3f}"
        rp = rp_by_step.get(fs)
        if rp is not None and "q_chunk_norm" in z:
            k = q_pos[fs]
            s["progress_raw"] = rp.get("progress_raw")
            prog = z["q_chunk_progress"][k].astype(float) if "q_chunk_progress" in z else None
            prog = None if (prog is None or np.all(np.isnan(prog))) else prog.round(4).tolist()
            # ORACLE pads past-episode-end rows with NaN so the GUI shows them blank. Bare NaN is
            # invalid JSON, so convert any NaN cell to null (JS reads it as blank). round4 first.
            def _rows_json(arr):
                a = np.asarray(arr, float).round(4)
                return [[None if np.isnan(v) else float(v) for v in row] for row in a]
            s["query"] = dict(
                prompt=rp.get("prompt"), gripper_flag=rp.get("gripper_flag"),
                executed_step=rp.get("executed_step"), replan_steps=rp.get("replan_steps"),
                horizon=rp.get("horizon"),
                chunk_lean11_norm=_rows_json(z["q_chunk_norm"][k]),
                chunk_lean11=_rows_json(z["q_chunk_lean"][k]),
                chunk_progress=prog)
        steps.append(s)
    return {**meta, "steps": steps}


@app.route("/api/<root>/gemini/<method>/<episode>/<sub>")
def api_gemini_r(root, method, episode, sub):
    """Serve a subtask's Gemini verdict if the judge has run. Prefers the new self-contained
    location judge/gemini.json (three-way verdict/completion/reason + skip info); falls back to the
    legacy root gemini.json (older success-bool schema)."""
    sub_dir = _safe(_root(root), method, episode, sub)
    for f in (sub_dir / "judge" / "gemini.json", sub_dir / "gemini.json"):
        if f.is_file():
            try:
                return jsonify(json.loads(f.read_text()))
            except Exception:
                return jsonify(None)
    return jsonify(None)


@app.route("/api/<root>/steps/<method>/<episode>/<sub>")
def api_steps_r(root, method, episode, sub):
    doc = _reconstruct_full(_safe(_root(root), method, episode, sub))
    if doc is None:
        abort(404)
    return jsonify(doc)


@app.route("/api/<root>/media/<method>/<episode>/<sub>/<path:fname>")
def api_media_r(root, method, episode, sub, fname):
    f = _safe(_root(root), method, episode, sub) / fname
    if not f.exists():
        abort(404)
    mt = "video/mp4" if fname.endswith(".mp4") else ("image/jpeg" if fname.endswith(".jpg") else None)
    return send_file(f, mimetype=mt)


# legacy aliases (default finestep root)
@app.route("/api/steps/<method>/<episode>/<sub>")
def api_steps(method, episode, sub):
    return api_steps_r("finestep", method, episode, sub)


@app.route("/api/media/<method>/<episode>/<sub>/<path:fname>")
def api_media(method, episode, sub, fname):
    return api_media_r("finestep", method, episode, sub, fname)


_ROOT_LABEL = {"finestep": "Fine-step", "milestone": "Milestone", "episode": "Episode"}


def _viewer_page(root_name: str) -> str:
    """The rollout viewer page bound to a rollout root ('finestep' | 'milestone' | 'episode').
    Injects window.RN so the shared gui.js hits /api/<root>/... and a small nav bar."""
    nav = ('<nav style="font-size:13px">'
           '<a href="/" style="color:#b0431c;margin-right:8px">home</a>'
           '<a href="/finestep" style="color:#b0431c;margin-right:8px">finestep</a>'
           '<a href="/milestone" style="color:#b0431c;margin-right:8px">milestone</a>'
           '<a href="/episode" style="color:#b0431c;margin-right:8px">episode</a>'
           '<a href="/val_mse" style="color:#b0431c;margin-right:8px">val_mse</a>'
           '<a href="/stats" style="color:#b0431c">stats</a></nav>')
    inject = f"<script>window.RN={root_name!r};</script>"
    # Put the RN global BEFORE gui.js loads, and drop the nav into the top bar.
    html = INDEX_HTML.replace("<script src=\"/gui.js\">", inject + "<script src=\"/gui.js\">")
    html = html.replace("<div id=\"top\">", f"<div id=\"top\">{nav}", 1)
    label = _ROOT_LABEL.get(root_name, "Fine-step")
    html = html.replace("<title>Subtask Eval Viewer</title>", f"<title>{label} Eval Viewer</title>")
    html = html.replace("<h1><b>Subtask</b> Eval</h1>", f"<h1><b>{label}</b> Eval</h1>")
    return html


@app.route("/")
def index():
    return HOME_HTML


@app.route("/finestep")
def finestep_page():
    return _viewer_page("finestep")


@app.route("/milestone")
def milestone_page():
    return _viewer_page("milestone")


@app.route("/episode")
def episode_page():
    return _viewer_page("episode")


# ---------------------------------------------------------------------------------
# Eval #1: train/val action-MSE (+ progress) curves per ckpt/step. Reads the per-ckpt JSON the
# sweep (scripts/run_val_sweep.py) writes into MSE_DIRS["val"] / ["train"]. The page overlays
# both splits with per-split + per-method toggles so you can compare seen vs unseen + ablations.
# ---------------------------------------------------------------------------------
def _load_mse_split(d: Path) -> dict:
    """Aggregate a split's per-ckpt JSONs -> {exp_name: [step-rows sorted]}."""
    runs: dict[str, list] = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            if f.name == "SUMMARY.json":
                continue
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            exp = doc.get("exp_name")
            for s in doc.get("steps", []):
                runs.setdefault(exp, []).append(s)
    for exp in runs:
        runs[exp].sort(key=lambda s: s.get("step") or 0)
    return runs


@app.route("/api/val_mse")
def api_val_mse():
    """Return {split: {exp_name: [{step, action_mse, progress_acc, progress_mae, progress_mode}]}}
    for both 'val' and 'train' (whichever result dirs exist / are populated)."""
    return jsonify({split: _load_mse_split(d) for split, d in MSE_DIRS.items()})


@app.route("/val_mse")
def val_mse_page():
    return VAL_MSE_HTML


# ---------------------------------------------------------------------------------
# Eval #4: STATISTICS across all methods (val MSE + episode success + subtask Gemini).
# ---------------------------------------------------------------------------------
# Seen (first-5, trained) vs unseen (last-5, untrained) episode sets, keyed by the episode_id
# suffix ("RoboCasa/<Cat>/<Task>/pretrain/episode_NNNNNN"). Built from the eps-list files the
# big-eval used, so the split is derived without re-tagging the existing rollout output.
_SEEN_UNSEEN_CACHE: dict[str, set[str]] | None = None


def _seen_unseen_sets() -> dict[str, set[str]]:
    global _SEEN_UNSEEN_CACHE
    if _SEEN_UNSEEN_CACHE is not None:
        return _SEEN_UNSEEN_CACHE
    out = {"seen": set(), "unseen": set()}
    base = Path("_evallogs/eps_lists")
    for split, fn in (("seen", "eps_seen.txt"), ("unseen", "eps_unseen.txt")):
        f = base / fn
        if f.is_file():
            for ln in f.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                # keep from "RoboCasa/" onward so it matches episode_id exactly
                i = ln.find("RoboCasa/")
                out[split].add(ln[i:] if i >= 0 else ln)
    _SEEN_UNSEEN_CACHE = out
    return out


# task_name -> target-split label (atomic_seen / composite_seen / composite_unseen), parsed once
# from robocasa's dataset_registry TARGET_TASKS. Used to group the per-task stats table by split.
_TARGET_SPLIT_CACHE: dict[str, str] | None = None


def _target_split_map() -> dict[str, str]:
    global _TARGET_SPLIT_CACHE
    if _TARGET_SPLIT_CACHE is not None:
        return _TARGET_SPLIT_CACHE
    out: dict[str, str] = {}
    # ROBOCASA_REPO lets a different box point at its own robocasa checkout (default unchanged).
    # Default to the shared checkout; the old default was another host's home dir, so the split
    # lookup silently failed and every task showed up as type "other".
    # Prefer $ROBOCASA_REPO; else the robocasa checkout beside this repo. The old default was
    # another host's home dir, so the split lookup silently failed and every task read "other".
    reg = (Path(os.environ.get("ROBOCASA_REPO") or _REPO_ROOT / "robocasa")
           / "robocasa/utils/dataset_registry.py")
    try:
        import re
        src = reg.read_text()
        for key in ("atomic_seen", "composite_seen", "composite_unseen"):
            m = re.search(key + r"=\[(.*?)\]", src, re.S)
            if m:
                for t in re.findall(r'"([^"]+)"', m.group(1)):
                    out[t] = key
    except Exception:
        pass
    _TARGET_SPLIT_CACHE = out
    return out


# DURABLE per-checkpoint episode results (compact summaries extracted by
# scripts/extract_episode_results.py). These SURVIVE deleting the big per-episode video dirs under
# episode/<m>/, and are the PREFERRED source for /stats #2. Falls back to episode/<m>/index.json.
EPISODE_RESULTS_DIR = Path("eval_results/episode_results")


def _episode_index_docs() -> dict:
    """method -> episode summary doc (episodes[]). Prefer the durable episode_results/<m>.json;
    fall back to episode/<m>/index.json for any method not yet extracted."""
    docs = {}
    if EPISODE_RESULTS_DIR.is_dir():
        for f in sorted(EPISODE_RESULTS_DIR.glob("*.json")):
            try:
                docs[f.stem] = json.loads(f.read_text())
            except Exception:
                pass
    ep_root = _root("episode")
    if ep_root.exists():
        for m in _methods(ep_root):
            if m in docs:
                continue   # durable copy already has it
            try:
                docs[m] = json.loads((ep_root / m / "index.json").read_text())
            except Exception:
                pass
    return docs


def _episode_stats(root: Path) -> dict:
    """Per-method episode-success stats. Breakdown by the RoboCasa TARGET SPLIT: overall +
    atomic_seen / composite_seen / composite_unseen (task_name via _target_split_map). Reads the
    DURABLE episode_results/<m>.json (survives video cleanup), falling back to episode/<m>/index.json."""
    tsplit = _target_split_map()   # task_name -> atomic_seen / composite_seen / composite_unseen
    keys = ["all", "atomic_seen", "composite_seen", "composite_unseen"]
    out = {}
    for m, idx in _episode_index_docs().items():
        succ = {k: [0, 0] for k in keys}  # [n_success, n_total]
        by_task: dict[str, list] = {}     # task_name -> [n_success, n_total, split]
        for ep in idx.get("episodes", []):
            if ep.get("episode_success") is None:
                continue
            eid = ep.get("episode_id", "")
            task = ep.get("task_name") or (eid.split("/")[2] if "/" in eid else eid)
            cat = "atomic" if "/Atomic/" in eid else ("composite" if "/Composite/" in eid else None)
            spl = tsplit.get(task)   # atomic_seen / composite_seen / composite_unseen (or None)
            ok = 1 if ep["episode_success"] else 0
            def _bump(k):
                succ[k][0] += ok; succ[k][1] += 1
            _bump("all")
            if spl in succ:
                _bump(spl)
            b = by_task.setdefault(task, [0, 0, spl or cat or "?"])
            b[0] += ok; b[1] += 1
        row = {k: (v[0] / v[1] if v[1] else None) for k, v in succ.items()}
        row["n"] = succ["all"][1]
        row["counts"] = {k: succ[k][1] for k in keys}        # n episodes per cell (total)
        row["succ_counts"] = {k: succ[k][0] for k in keys}   # n SUCCESSFUL episodes per cell
        row["per_task"] = {t: {"rate": (b[0] / b[1] if b[1] else None), "s": b[0], "n": b[1], "split": b[2]}
                           for t, b in sorted(by_task.items())}
        row["avg_seconds_per_episode"] = idx.get("avg_seconds_per_episode")
        row["total_seconds"] = idx.get("total_seconds")
        out[m] = row
    return out


def _prim_from_child(cd: str) -> str:
    """child dir 'child03_close' -> primitive 'close' (the batch judge summary keys by child_dir)."""
    p = cd.split("_", 1)
    return p[1] if len(p) > 1 else "other"


def _subtask_stats(root: Path) -> dict:
    """Per-method subtask-success stats from the Gemini BATCH judge. Reads the per-method
    <method>/gemini_summary.json (ONE file — the rollup the batch judge writes) rather than
    thousands of episode.json. THREE-WAY verdicts (success / failure / uncertain); success_rate is
    over DECIDED spans (success/(success+failure)), uncertain reported separately. Per-primitive
    breakdown is derived from the per_subtask child_dir keys. Falls back to scanning judge/gemini.json
    if the summary is missing."""
    out = {}
    for m in _methods(root):
        summ_f = root / m / "gemini_summary.json"
        tot = {"success": 0, "failure": 0, "uncertain": 0}
        by_prim: dict[str, dict] = {}   # primitive -> {success, failure, uncertain}
        def _bump(prim, v):
            if v not in tot:
                return
            tot[v] += 1
            b = by_prim.setdefault(prim, {"success": 0, "failure": 0, "uncertain": 0})
            b[v] += 1
        source = "gemini"
        if summ_f.is_file():
            try:
                summ = json.loads(summ_f.read_text())
            except Exception:
                summ = {}
            for _ep, e in (summ.get("per_episode") or {}).items():
                for cd, sub in (e.get("per_subtask") or {}).items():
                    _bump(_prim_from_child(cd), sub.get("verdict"))
        else:
            # No Gemini summary (e.g. ORACLE — never judged; it's the sim-grounded upper-bound
            # reference). Use the per-span SIM CHECK verdict instead: success/failure are decided,
            # 'unknown' maps to uncertain (no high-precision rule applies). Read from episode.json.
            source = "sim_check"
            for ep in _episodes(root, m):
                try:
                    doc = json.loads((_safe(root, m, ep) / "episode.json").read_text())
                except Exception:
                    continue
                for sg in doc.get("subgoals", []):
                    sc = sg.get("subtask_sim_check") or {}
                    v = sc.get("verdict")
                    v = "uncertain" if v in (None, "unknown", "error") else v
                    _bump(sg.get("primitive", "other"), v)
        decided = tot["success"] + tot["failure"]
        n = decided + tot["uncertain"]
        def _rate(b):
            d = b["success"] + b["failure"]
            return (b["success"] / d) if d else None
        out[m] = dict(
            overall=(tot["success"] / decided if decided else None),
            n=n, n_decided=decided, source=source,
            n_success=tot["success"], n_failure=tot["failure"], n_uncertain=tot["uncertain"],
            per_primitive={p: _rate(b) for p, b in sorted(by_prim.items())},
            per_primitive_counts={p: dict(b) for p, b in sorted(by_prim.items())})
    return out


# /api/stats reads thousands of episode.json (8 methods x 500 eps) — ~3s/call. The data only
# changes when a rollout/judge writes new files, so cache the computed result and invalidate on a
# cheap fingerprint: the newest index.json mtime + method-dir count across the episode+subtask roots
# (index.json is rewritten whenever a method's episodes change). ?refresh=1 forces a recompute.
_STATS_CACHE: dict = {"fp": None, "data": None}


def _stats_fingerprint() -> tuple:
    fp = []
    # durable episode summaries (preferred source for #2) — invalidate when re-extracted
    if EPISODE_RESULTS_DIR.is_dir():
        for f in EPISODE_RESULTS_DIR.glob("*.json"):
            try:
                fp.append((str(f), f.stat().st_mtime))
            except OSError:
                pass
    for root in (_root("episode"), _root("finestep"), _root("milestone")):
        if not root.exists():
            continue
        for idx in root.glob("*/index.json"):
            try:
                fp.append((str(idx), idx.stat().st_mtime))
            except OSError:
                pass
        # finestep Gemini verdicts land in <method>/gemini_summary.json — invalidate #3 when re-judged
        for gs in root.glob("*/gemini_summary.json"):
            try:
                fp.append((str(gs), gs.stat().st_mtime))
            except OSError:
                pass
    if VAL_MSE_DIR.is_dir():
        for f in VAL_MSE_DIR.glob("*.json"):
            try:
                fp.append((str(f), f.stat().st_mtime))
            except OSError:
                pass
    # COMBINED roots. These were missing, and the omission was silent in the worst way: sections #0
    # and #0b are built from them, so a newly finished sweep (or a fresh
    # scripts/extract_combine_results.py) did not change the fingerprint and /stats kept serving a
    # cached payload WITHOUT that method -- it simply was not in the table, with no error, until the
    # GUI was restarted or ?refresh=1 was passed by hand.
    #   * combine/*/index.json  -> a sweep writing new episodes
    #   * combine_results/*.json -> a re-extraction (the preferred source for #0/#0b)
    if COMBINE_ROOT.is_dir():
        for idx in COMBINE_ROOT.glob("*/index.json"):
            try:
                fp.append((str(idx), idx.stat().st_mtime))
            except OSError:
                pass
    if COMBINE_RESULTS_DIR.is_dir():
        for f in COMBINE_RESULTS_DIR.glob("*.json"):
            try:
                fp.append((str(f), f.stat().st_mtime))
            except OSError:
                pass
    return tuple(sorted(fp))


def _compute_stats() -> dict:
    val = []
    if VAL_MSE_DIR.is_dir():
        for f in sorted(VAL_MSE_DIR.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                steps = [s for s in d.get("steps", []) if s.get("action_mse") is not None]
                last = max(steps, key=lambda s: s.get("step") or 0) if steps else None
                val.append(dict(exp_name=d.get("exp_name", f.stem),
                                final_step=(last.get("step") if last else None),
                                action_mse=(last.get("action_mse") if last else None),
                                flow_loss=(last.get("flow_loss") if last else None)))
            except Exception:
                continue
    return dict(val_mse=val,
                episode=_episode_stats(_root("episode")),
                subtask=_subtask_stats(_root("finestep")),
                combine=_combine_stats())


@app.route("/api/stats")
def api_stats():
    from flask import request
    fp = _stats_fingerprint()
    if request.args.get("refresh") == "1" or _STATS_CACHE["fp"] != fp or _STATS_CACHE["data"] is None:
        _STATS_CACHE["data"] = _compute_stats()
        _STATS_CACHE["fp"] = fp
    return jsonify(_STATS_CACHE["data"])


@app.route("/stats")
def stats_page():
    return STATS_HTML.replace("<script>", _ERR_JS + "<script>", 1)


VAL_MSE_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Train/Val MSE</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body{font-family:system-ui,sans-serif;margin:0;color:#1a1a1a;background:#fafafb}
  header{padding:10px 16px;border-bottom:1px solid #ddd;background:#fff;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header h1{font-size:16px;margin:0}header h1 b{color:#b0431c}
  nav a{margin-right:10px;color:#b0431c;text-decoration:none;font-size:13px}
  #wrap{padding:16px;max-width:1200px;margin:0 auto}
  label{font-size:13px}select{font-size:13px;padding:2px 6px}
  #chart-box{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin-top:12px}
  .toggles{display:flex;gap:24px;flex-wrap:wrap;margin-top:10px}
  .tgroup{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 12px}
  .tgroup h3{font-size:11px;margin:0 0 6px;text-transform:uppercase;letter-spacing:.05em;color:#666}
  .tgroup label{display:flex;align-items:center;gap:6px;font-family:ui-monospace,monospace;font-size:12px;margin:2px 0;cursor:pointer}
  .sw{width:11px;height:11px;border-radius:2px;display:inline-block}
  .btnrow{margin-top:4px}.btnrow button{font-size:11px;padding:1px 7px;margin-right:4px;cursor:pointer}
  table{border-collapse:collapse;font-size:12px;margin-top:16px;background:#fff}
  th,td{border:1px solid #ddd;padding:3px 8px;text-align:right}th{background:#f0f0f0}
  td.exp{text-align:left;font-family:ui-monospace,monospace}
  .note{font-size:11px;color:#888;margin-left:auto}
</style></head><body>
<header>
  <h1>Train / Val <b>MSE</b></h1>
  <nav><a href="/">home</a><a class="hi" href="/combine">combine</a><a href="/finestep">finestep</a><a href="/milestone">milestone</a><a href="/episode">episode</a><a href="/val_mse">val_mse</a><a href="/baseline">baseline</a><a href="/stats">stats</a></nav>
  <label>metric <select id="metric">
    <option value="action_mse">action_mse</option>
    <option value="progress_mae">progress_mae</option>
    <option value="progress_acc">progress_acc (classes only)</option>
  </select></label>
  <span class="note">solid = val (unseen) · dashed = train (seen)</span>
</header>
<div id="wrap">
  <div class="toggles">
    <div class="tgroup"><h3>split</h3><div id="split-toggles"></div></div>
    <div class="tgroup"><h3>method</h3><div id="method-toggles"></div>
      <div class="btnrow"><button id="all-on">all</button><button id="all-off">none</button></div>
    </div>
  </div>
  <div id="chart-box"><canvas id="chart" height="110"></canvas></div>
  <div id="table"></div>
</div>
<script>
// stable color per method (exp base name, tag-stripped); train reuses the val color but dashed.
const COLORS=["#b0431c","#1c6bb0","#2e8b3d","#8b2eb0","#b0902e","#2eb0a3","#b02e5a","#555",
              "#d17c1c","#1c9bb0","#6b8b2e","#b02e8b"];
const short=n=>(n||"").replace("m0717-50k-bs512-","").replace("_granfine_verbsimp","");
let DATA={}, CH=null;
const ON={split:{val:true,train:true}, method:{}};   // method: filled after load
let methodColor={};

async function load(){
  DATA=await (await fetch('/api/val_mse')).json();     // {split:{exp:[rows]}}
  // union of all method (exp) names across splits, stable-sorted
  const methods=[...new Set(Object.values(DATA).flatMap(m=>Object.keys(m)))].sort();
  methods.forEach((m,i)=>{ methodColor[m]=COLORS[i%COLORS.length]; if(!(m in ON.method))ON.method[m]=true; });
  buildToggles(methods);
  draw();
}

function buildToggles(methods){
  const sp=document.getElementById('split-toggles'); sp.innerHTML="";
  ["val","train"].forEach(s=>{
    const has=DATA[s] && Object.keys(DATA[s]).length;
    sp.insertAdjacentHTML('beforeend',
      `<label><input type=checkbox data-split="${s}" ${ON.split[s]?'checked':''} ${has?'':'disabled'}>`+
      `${s}${has?'':' (none yet)'}</label>`);
  });
  sp.querySelectorAll('input').forEach(cb=>cb.onchange=()=>{ON.split[cb.dataset.split]=cb.checked;draw();});
  const mt=document.getElementById('method-toggles'); mt.innerHTML="";
  methods.forEach(m=>{
    mt.insertAdjacentHTML('beforeend',
      `<label><input type=checkbox data-m="${m}" ${ON.method[m]?'checked':''}>`+
      `<span class=sw style="background:${methodColor[m]}"></span>${short(m)}</label>`);
  });
  mt.querySelectorAll('input').forEach(cb=>cb.onchange=()=>{ON.method[cb.dataset.m]=cb.checked;draw();});
}
document.getElementById('all-on').onclick=()=>{Object.keys(ON.method).forEach(m=>ON.method[m]=true);
  document.querySelectorAll('#method-toggles input').forEach(cb=>cb.checked=true);draw();};
document.getElementById('all-off').onclick=()=>{Object.keys(ON.method).forEach(m=>ON.method[m]=false);
  document.querySelectorAll('#method-toggles input').forEach(cb=>cb.checked=false);draw();};

function draw(){
  const metric=document.getElementById('metric').value;
  const ds=[];
  ["val","train"].forEach(split=>{
    if(!ON.split[split] || !DATA[split])return;
    Object.entries(DATA[split]).forEach(([exp,rows])=>{
      if(!ON.method[exp])return;
      const pts=rows.filter(s=>s[metric]!=null).map(s=>({x:s.step,y:s[metric]}));
      if(!pts.length)return;
      ds.push({label:`${short(exp)} [${split}]`,data:pts,borderColor:methodColor[exp],
               backgroundColor:methodColor[exp],borderDash:split==="train"?[6,4]:[],
               tension:0.15,pointRadius:2});
    });
  });
  if(CH)CH.destroy();
  CH=new Chart(document.getElementById('chart'),{type:'line',data:{datasets:ds},
    options:{responsive:true,parsing:false,interaction:{mode:'nearest'},
      scales:{x:{type:'linear',title:{display:true,text:'step'}},y:{title:{display:true,text:metric}}},
      plugins:{legend:{labels:{font:{size:10},boxWidth:18}}}}});
  drawTable(metric);
}

// table sort state: key = 'method'|'val'|'train'|'gap', dir = 1 asc / -1 desc.
const SORT={key:'method', dir:1};
function drawTable(metric){
  // final-step value per method, val vs train side by side + gap
  const methods=[...new Set(Object.values(DATA).flatMap(m=>Object.keys(m)))].filter(m=>ON.method[m]);
  const last=(split,exp)=>{const r=(DATA[split]||{})[exp]||[];const s=r.filter(x=>x[metric]!=null).slice(-1)[0];return s?s[metric]:null;};
  const recs=methods.map(m=>{const v=last('val',m),t=last('train',m);
    return {m, name:short(m), val:v, train:t, gap:(v!=null&&t!=null)?v-t:null};});
  // sort: nulls always last regardless of direction; 'method' sorts by name.
  const cmp=(a,b)=>{
    if(SORT.key==='method') return a.name<b.name?-1:a.name>b.name?1:0;
    const x=a[SORT.key], y=b[SORT.key];
    if(x==null&&y==null)return 0; if(x==null)return 1; if(y==null)return -1;
    return x-y;
  };
  recs.sort((a,b)=>SORT.dir*cmp(a,b));
  const arrow=k=>SORT.key===k?(SORT.dir>0?' ▲':' ▼'):'';
  const th=(k,lbl,cls)=>`<th class="${cls||''} sorth" data-k="${k}" style="cursor:pointer">${lbl}${arrow(k)}</th>`;
  const f4=x=>x!=null?(+x).toFixed(4):'–';
  let h="<table><tr>"+th('method','method','exp')+th('val','val')+th('train','train')+th('gap','gap (val−train)')+"</tr>";
  recs.forEach(r=>{ h+=`<tr><td class='exp'>${r.name}</td><td>${f4(r.val)}</td><td>${f4(r.train)}</td><td>${f4(r.gap)}</td></tr>`; });
  const box=document.getElementById('table'); box.innerHTML=h+"</table>";
  box.querySelectorAll('.sorth').forEach(el=>el.onclick=()=>{
    const k=el.dataset.k;
    if(SORT.key===k) SORT.dir=-SORT.dir; else {SORT.key=k; SORT.dir=1;}
    drawTable(document.getElementById('metric').value);
  });
}
document.getElementById('metric').onchange=draw;
// Swallow a failed poll: a dropped /api/stats request is expected (tunnel, or this server being
// restarted mid-sweep) and the next tick recovers. Unguarded, each failure became an unhandled
// rejection and a scary banner once a minute.
const safeLoad=()=>Promise.resolve().then(load).catch(e=>console.warn('stats refresh failed:', e));
safeLoad();
setInterval(safeLoad, 60000);   // refresh while the sweep is still writing results
</script></body></html>"""


STATS_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Eval Stats</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body{font-family:system-ui,sans-serif;margin:0;color:#1a1a1a;background:#fafafb}
  header{padding:10px 16px;border-bottom:1px solid #ddd;background:#fff;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
  header h1{font-size:16px;margin:0}header h1 b{color:#b0431c}
  nav a{margin-right:10px;color:#b0431c;text-decoration:none;font-size:13px}
  #wrap{padding:16px;max-width:1200px;margin:0 auto}
  h2{font-size:14px;margin:18px 0 6px}
  table{border-collapse:collapse;font-size:12px;background:#fff;margin-bottom:8px}
  th,td{border:1px solid #ddd;padding:3px 8px;text-align:right}th{background:#f0f0f0}
  td.exp,th.exp{text-align:left;font-family:ui-monospace,monospace}
  .box{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px;margin-top:10px}
  /* per-task matrix: wrap the long s1_/s2_ method names instead of stretching the table */
  table.cmbtask th{white-space:normal;max-width:150px;line-height:1.25;font-size:11px;vertical-align:bottom}
  table.cmbtask td.exp{white-space:nowrap}
  /* #0 summary: the method name is the widest thing in the row, so give it room and keep it on one
     line -- dropping the terminations column freed the space. */
  table.cmbmain td.exp{white-space:nowrap;min-width:340px}
  /* the combined S2+S1 rollout browser is the main drill-down from these tables */
  nav a.hi{background:#b0431c;color:#fff;border-radius:4px;padding:1px 8px;font-weight:700}
  nav a.hi:hover{background:#8f3616}
</style></head><body>
<header>
  <h1>Eval <b>Statistics</b></h1>
  <nav><a href="/">home</a><a class="hi" href="/combine">combine</a><a href="/finestep">finestep</a><a href="/milestone">milestone</a><a href="/episode">episode</a><a href="/val_mse">val_mse</a><a href="/baseline">baseline</a><a href="/stats">stats</a></nav>
</header>
<div id="wrap">
  <div id="loading" style="padding:10px 0;color:#b0431c;font-size:14px">⏳ loading stats… (first load scans the eval results; ~a few seconds)</div>
  <details id="tagdoc" style="margin:6px 0 14px;border:1px solid #e3e3e8;border-radius:8px;background:#fff;padding:6px 12px">
    <summary style="cursor:pointer;font-size:15px;font-weight:700;color:#b0431c">▸ Ablation tag legend — what each method-name token means (click to expand)</summary>
    <div id="tagdoc-body" style="margin-top:10px;font-size:13px;line-height:1.55"></div>
  </details>
  <h2>#0 COMBINED System2+System1 <span style="font-size:11px;color:#888;font-weight:400">— closed-loop hierarchical rollouts (/combine) · ranked best → worst · rate and #success/#episodes, overall and per split</span></h2>
  <div id="cmbdebug" style="font-size:11px;margin:0 0 4px"></div>
  <div id="cmbtab" style="overflow-x:auto"></div>
  <div id="cmbsplit" style="overflow-x:auto"></div>
  <h2>#0b Per-task COMBINED success <span style="font-size:11px;color:#888;font-weight:400">— task × method · #successful / #episodes (hover for % and s/ep) · OVERALL row = the three splits combined, then per-split totals and their tasks</span></h2>
  <div id="cmbcmp" style="font-size:11px;margin:0 0 5px"></div>
  <div id="cmbtask" style="overflow-x:auto"></div>
  <h2>#2 Episode success rate <span style="font-size:11px;color:#888;font-weight:400">— % (n episodes)</span></h2>
  <div id="epfilter" style="margin:2px 0 8px;font-size:12px;display:flex;gap:6px;align-items:center;flex-wrap:wrap"></div>
  <div id="eptab"></div>
  <h2>#2b Per-task episode success <span style="font-size:11px;color:#888;font-weight:400">— task × method, grouped by split · #successful / #episodes (hover for %)</span></h2><div id="tasktab" style="overflow-x:auto"></div>
  <h2>#3 Subtask Gemini success rate <span style="font-size:11px;color:#888;font-weight:400">— three-way verdict; rate = success / (success+failure), uncertain excluded; skipped (retract/low-movement) not counted</span></h2><div id="subtab" style="overflow-x:auto"></div>
  <div class="box"><canvas id="chart" height="90"></canvas></div>
</div>
<script>
const COLORS=["#b0431c","#1c6bb0","#2e8b3d","#8b2eb0","#b0902e","#2eb0a3","#b02e5a","#555"];
// #0b compare mode: highlight a per-task gap of at least this many EPISODES. 4 because at n=30 per
// task the binomial SD is ~2.7 episodes, so anything under ~4 is inside ordinary sampling noise --
// a highlighted cell is a gap worth looking at, not proof of one.
const CMP_HL=4;
// Method DISPLAY name: fix up truncated/short output-dir labels to the complete ablation name.
// (v12's rollout dir is 'v12_progact_noanchorstate' but the ckpt is '..._noanchorstate_noanchor';
//  shown complete here until the dir is renamed post-run.)
const _MNAME={'v12_progact_noanchorstate':'v12_progact_noanchorstate_noanchor'};
const mName=m=>_MNAME[m]||m;
// method row color by progress-head family: progact* light blue, progreg* light red, else default.
const mColor=m=>{const t=m.toLowerCase();
  if(t.includes('progact'))return '#e6f0fb'; if(t.includes('progreg'))return '#fdeaea'; return '';};
// Ablation-tag legend: what each method-name token means. Prompt examples are the REAL assembled
// System1 prompt (task goal + subgoal + offline-RL conditioning + discretized state + gripper).
// Long combined method names ("s1-progact270k_s2-qwen35-4b-full-ep3-11416") blow out the per-task
// table's column widths. Split on the s1_/s2_ boundary and stack the halves so the header wraps to
// two short lines; the full name stays in the title tooltip.
// Shorten a System1 tag for display: -noexec is on for every current ckpt so it carries no
// information, and -noanchor-noanchorstate (either order) is reported as just -noanchor.
function s1Short(t){
  return (t||'')
    .replace(/-noexec/g,'')
    .replace(/-noanchorstate-noanchor|-noanchor-noanchorstate/g,'-noanchor')
    .replace(/-noanchorstate/g,'-noanchor');
}
// COLOUR BY FAMILY, so a table of near-identical names can be read at a glance. Two independent
// axes, because a run is a PAIR of checkpoints and either half can be the thing that differs:
//   System1 progress head -- progact (teal) vs progreg (violet)
//   System2 planner       -- qwen35 (blue) vs qwen3vl (rose)
// Anything unrecognised stays neutral grey rather than being given a colour it might share with a
// family it is not in.
const s1Color=t=>{const x=(t||'').toLowerCase();
  return x.includes('progact')?'#0f766e':x.includes('progreg')?'#6d28d9':'#333';};
const s2Color=t=>{const x=(t||'').toLowerCase();
  return x.includes('qwen3vl')?'#be123c':x.includes('qwen35')?'#1d4ed8':'#666';};
// FULL method name, COLOURED by family. The name is not shortened -- the checkpoint tail is what
// makes a run identifiable, and collapsing it once already cost a round of confusion -- so only the
// colour carries the family, on two independent axes (see s1Color / s2Color above):
//   progact270k-qwen3vl-4b-full-ep3-17124-base
//   ^^^^^^^^^^^ teal        ^^^^^^^^^^^^^^^^^^ rose        ^^^^^ grey (the rule ARM, not a checkpoint)
// The arm suffix is split off by matching a known checkpoint prefix; anything unrecognised is left
// whole and coloured as one piece, so a new checkpoint is never silently mis-split.
const _S2_CKPTS=['qwen35-4b-full-ep3-11416','qwen3vl-4b-full-ep3-17124'];
const _s2Parts=t=>{const s=t||'';
  for(const k of _S2_CKPTS){
    if(s===k)return [k,''];
    if(s.startsWith(k+'-'))return [k, s.slice(k.length)];
  }
  return [s,''];};
// ``br`` puts the System2 half on its own line, which is what the narrow table headers want; inline
// otherwise (pickers, legends). Full name also in the tooltip.
// PROVENANCE SUFFIX on baseline rows, so the table distinguishes a number we produced from a number
// we copied. `kind` comes from the record (set by scripts/extract_baseline_results.py /
// make_leaderboard_method.py) rather than from the name, so renaming a run cannot silently drop it.
//   baseline_flat_policy -> "(replicate)"  we re-ran the model ourselves on this manifest
//   baseline_published   -> "(published)"  transcribed from the leaderboard; no episode data exists
// window._recOf is set by the #0 renderer, which is the only place the records are in scope.
function mSuffix(m){
  const k=((window._recOf||{})[m]||{}).kind||'';
  if(k==='baseline_flat_policy')return ' <span style="color:#888;font-weight:400">(replicate)</span>';
  if(k==='baseline_published')return ' <span style="color:#b0431c;font-weight:400">(published)</span>';
  return '';
}
function mLabel(m, br){
  const x=/^s1-(.+?)_s2-(.+)$/.exec(m||'');
  if(!x)return `<span title="${m}">${mName(m)}${mSuffix(m)}</span>`;
  const [ck,arm]=_s2Parts(x[2]);
  return `<span title="${m}"><b style="color:${s1Color(x[1])}">${s1Short(x[1])}</b>`
    +(br?'<br>':`<span style="color:#bbb">-</span>`)
    +`<span style="font-weight:400;color:${s2Color(ck)}">${ck}</span>`
    +(arm?`<span style="font-weight:400;color:#888">${arm}</span>`:'')
    +mSuffix(m)
    +`</span>`;
}
// SHORT form, used by #0b and its compare controls only: the per-task matrix has 50 rows and one
// column per method, so the full checkpoint tail costs width that the task names need. #0 keeps the
// full name -- that table is the record, this one is the working view.
//   progact270k / qwen3vl (inst)
// The family map is EXPLICIT: any checkpoint outside it keeps its full name, so a new step is never
// rendered as if it were the established one.
const _S2_FAMILY={'qwen35-4b-full-ep3-11416':'qwen35','qwen3vl-4b-full-ep3-17124':'qwen3vl'};
function mLabelShort(m, br){
  const x=/^s1-(.+?)_s2-(.+)$/.exec(m||'');
  if(!x)return `<span title="${m}">${mName(m)}</span>`;
  const [ck,arm]=_s2Parts(x[2]);
  const fam=_S2_FAMILY[ck]||ck;
  const armTxt=arm?`(${arm.replace(/^-/,'')})`:'';
  return `<span title="${m}"><b style="color:${s1Color(x[1])}">${s1Short(x[1])}</b>`
    +(br?'<br>':`<span style="color:#bbb"> / </span>`)
    +`<span style="font-weight:400;color:${s2Color(ck)}">${fam}</span>`
    +(armTxt?` <span style="font-weight:400;color:#888">${armTxt}</span>`:'')
    +`</span>`;
}
const mShort=m=>mLabelShort(m,false);      // #0b compare picker / legend
const mName2=m=>mLabel(m,true);            // #0: full name
function renderTagDoc(){
  const box=document.getElementById('tagdoc-body'); if(!box)return;
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const FULL=[
    'Task: boil water; Current Subgoal: grasp the kettle',
    'Quality: Success; Estimated Length: 50; Executed Step: 7',
    'Initial State: 48 131 0 255 …; Current State: 48 131 0 255 …; Current Gripper: Open;',
    'Action:',
  ].join('\n');
  const pbox=t=>`<pre style="background:#f7f7fa;border:1px solid #ececef;border-radius:5px;padding:6px 8px;margin:4px 0;white-space:pre-wrap;font-size:11.5px">${esc(t)}</pre>`;
  // progress predictor (exactly one per model)
  let h='<b>Progress predictor</b> (the <code>prog*</code> prefix — exactly one per model, how the '
       +'model represents sub-task progress):<ul style="margin:4px 0">'
    +'<li><code>progcls</code> — progress as a 10-way <b>classification</b> head (decile buckets).</li>'
    +'<li><code>progreg</code> — progress as a <b>continuous regression</b> head (scalar 0→1).</li>'
    +'<li><code>progact</code> — progress as a <b>12th action dim</b> (predicted per step, no separate head).</li>'
    +'<li><code>prognone</code> — no progress signal at all.</li></ul>';
  h+='<div style="margin:8px 0 2px"><b>Full prompt</b> (no ablations — the default input the model sees):</div>'+pbox(FULL);
  // language-prompt ablations, each with the concrete line it removes
  h+='<b>Language-prompt ablations</b> (each token DROPS part of the prompt above; default = all kept):';
  const rows=[
    ['noexec','drops only <b>Executed Step</b> from the conditioning line',
      'Quality: Success; Estimated Length: 50\n   (no "Executed Step: 7")'],
    ['noestl','drops only <b>Estimated Length</b> from the conditioning line',
      'Quality: Success; Executed Step: 7'],
    ['nocond','drops the <b>entire conditioning line</b> (Quality / Estimated Length / Executed Step)',
      'Task: boil water; Current Subgoal: grasp the kettle\n   (conditioning line gone) → Initial State: …'],
    ['notask','drops the <b>whole-task goal</b>, keeping only the current subgoal',
      'Current Subgoal: grasp the kettle\n   (no "Task: boil water")'],
    ['nostate','drops the <b>entire discretized state block</b> (both Initial + Current State)',
      'Quality: …; Executed Step: 7\n   (no state ints) → Current Gripper: Open;'],
    ['noanchorstate','drops only the <b>Initial (anchor) State</b>; the current State stays (shown as "State:")',
      'State: 48 131 0 255 …; Current Gripper: Open;\n   (no "Initial State: …" half)'],
    ['nogrip','drops the <b>Current Gripper</b> flag line','… Current State: 48 131 0 255 …;\n   (no "Current Gripper: Open;")'],
  ];
  h+='<table style="margin:6px 0"><tr><th class="exp">token</th><th class="exp">effect</th><th class="exp">prompt becomes</th></tr>';
  rows.forEach(([t,e,ex])=>{h+=`<tr><td class="exp"><code>${t}</code></td><td class="exp">${e}</td><td class="exp">${pbox(ex)}</td></tr>`;});
  h+='</table>';
  // non-prompt (vision/state-tensor) ablations
  h+='<b>Vision / state-tensor ablations</b> (change the model INPUT tensors, not the text prompt):<ul style="margin:4px 0">'
    +'<li><code>noanchor</code> — the model does <b>not</b> receive the anchor (sub-task start) camera images; only the 3 current views. Prompt text unchanged.</li>'
    +'<li><em>(<code>noanchorstate</code> above also drops the anchor half of the proprioceptive state tensor, 28-d→14-d.)</em></li></ul>';
  h+='<div style="color:#888;font-size:12px;margin-top:6px">granularity/verbosity (<code>granfine</code>, <code>verbsimp</code>) describe the subgoal text source; all current runs use fine+simple.</div>';
  box.innerHTML=h;
}
// Hidden-by-default methods: debug-* only (throwaway runs, small n, often unfinished), which would
// otherwise sit beside a finished 1000-episode cold run and invite a false comparison. They stay
// LOADED and one checkbox away -- only the default view is filtered.
//
// *-memory used to be hidden here too, on the argument that the memory VARIANT (narrate -> recipe ->
// warm plan) is a different PIPELINE rather than a different checkpoint, so a row-by-row comparison
// with the cold runs misleads. It is shown again by request: the variant is a first-class result, and
// the row is labelled "[memory]" in the /combine picker so the distinction is still visible. The
// caveat stands when reading the table -- a memory row and a cold row differ by pipeline, not by
// checkpoint.
const isDebugMethod=m=>/^debug-/i.test(String(m||''));
// The memory VARIANT (narrate -> recipe -> warm plan). Its own toggle in #0, default ON: it is a real
// result, but it is a different PIPELINE (not a different checkpoint) and is usually run on a single
// split, so its overall is not comparable with a cold run's -- hence the ability to drop it.
const isMemoryMethod=m=>/-memory$/i.test(String(m||''));
// TERSE-GOAL FAMILY. All the short-goal arms: ...-unseenshort (cold plan), ...-unseenshort-memory
// (warm plan from the qwen3vl recipe library) and ...-unseenshort-gemini (warm plan from the gemini
// library), plus the flat baseline xiaomi-robo1-unseenshort.
//
// CONTAINS, not a suffix test -- anchoring on the end would catch only the first of those. It also
// means ...-unseenshort-memory is classified here rather than as a *-memory run, which is the useful
// split now: every arm in this family shares the DEGRADED GOAL and differs only in where its plan
// comes from. 480 episodes over the 16 composite_unseen tasks, so its overall is not comparable with
// a 50-task overall -- hence the two boxes below (one to include it, one to see nothing else).
const isUnseenShortMethod=m=>/unseenshort/i.test(String(m||''));
// NEW-TASK runs: the held-out set outside the 50-task manifest -- every task reports split "other".
// Flagged server-side (`newtask`), with the name as a fallback. The set GROWS (24 tasks, then 34 as of
// 2026-08-16), so no size is hardcoded anywhere below; every count is read off the data. Their refined
// fields ARE shown: the gate knows these tasks' horizons and demotes nothing on them, so refined
// equals raw, and printing n/a would hide a real result behind a technicality.
const isNewTaskMethod=(m,rec)=>((rec&&rec.newtask)||/newtask/i.test(String(m||'')));
// HUMAN-RECIPE runs: the composite_unseen split re-run with a HAND-WRITTEN recipe supplied to the
// planner, so its 16-task result is not truth-free the way every other arm's is -- a human read the
// task and wrote the plan. Both the ordinary and official-step retry variants stay behind this
// toggle. The retry variant also matches isRetryMethod below, so visibility is deliberately the AND
// of "human-recipe" and "other runs"; the ordinary variant needs only "human-recipe". Their donor
// mappings live in donorOf.
const isHumanRecipeMethod=m=>/-human-recipe(?:-retry)?$/.test(String(m||''));
// SPLIT DISPLAY NAMES. "other" is the split TARGET_TASK_SPLIT assigns to anything off the 50-task
// manifest, and in this tree it is owned exclusively by the newtask runs -- verified: every task in it
// is reported by a newtask method and by nothing else (34 distinct tasks as of 2026-08-16, up from 24;
// the column header carries the live count). So it is labelled "newtask", because "other" tells a
// reader nothing about what the column holds.
const SPL_NAME={atomic_seen:'atomic-seen',composite_seen:'composite-seen',
                composite_unseen:'composite-unseen',other:'newtask'};
const splName=x=>SPL_NAME[x]||String(x||'').replace('_','-');
// The -v2 RULE ARM. Its own toggle in #0, default ON: these are real 1500-episode results, but they
// ran the SUPERSEDED rule layer (est_bump global + sink_faucet_est, before the three-tier split), so
// when comparing a -base control against a -inst arm head to head they are noise in the middle of the
// ranking. One click removes them. Deliberately matches on the SUFFIX only, so a future "-v2-foo"
// variant is not silently swept up with them.
const isV2Method=m=>/-v2$/i.test(String(m||''));
// EXPERIMENTAL OFFICIAL-STEP RETRY ARM. It is a complete 30/task sweep, but it changes the rollout
// stopping policy by replaying the last milestone until the official horizon is exhausted. Keep it
// behind "other runs" while that policy is being evaluated, rather than placing its raw score in the
// default ranking beside the established -inst arm.
const isRetryMethod=m=>/-retry$/i.test(String(m||''));
// FLAT-POLICY BASELINE (Xiaomi-Robotics-1 RoboCasa365): one policy, no System2, no turns. Summarised
// into combine_results/ by scripts/extract_baseline_results.py, which is also what stamps the
// `baseline-` prefix this keys on. Its own toggle in #0, default ON: it is the external reference the
// hierarchical runs are measured against, but it is a DIFFERENT SYSTEM, not another arm of ours, so
// one click removes it when comparing our arms to each other.
//
// COMPARE IT AGAINST THE REFINED COLUMN, not the raw one. These rollouts were GENERATED with
// RoboCasa's official per-task horizon as the cap, so they cannot overshoot it and their raw rate is
// already horizon-compliant (measured: refined == raw on every episode delivered). Our hierarchical
// runs budget per subgoal and per turn, so only their refined rate is scored the same way.
// Keyed on the RECORD's `kind`, with the historic `baseline-` name prefix kept as a fallback for
// files written before that field existed. Pass C[m] at every call site.
const isBaselineMethod=(m,rec)=>((rec&&/^baseline/.test(String(rec.kind||'')))
                                 ||/^baseline-/i.test(String(m||'')));
const pct=v=>v==null?'–':(100*v).toFixed(1)+'%';
const f4=v=>v==null?'–':(+v).toFixed(4);
async function load(){
  const _ld=document.getElementById('loading');
  let d;
  try{
    d=await (await fetch('/api/stats')).json();
  }catch(e){
    if(_ld){_ld.textContent='⚠ failed to load stats: '+e; _ld.style.color='#c0392b';}
    return;
  }
  if(_ld)_ld.remove();   // data in hand -> drop the loading banner
  renderTagDoc();
  let h;
  // #2 episode table: overall + category (atomic/composite) + split (seen/unseen) + 2x2 cross,
  // each cell "% (n)"; last column = avg wall-time per episode. Sortable: click a header.
  const cell=(v,c,cnt)=>{const n=cnt&&cnt[c]!=null?cnt[c]:null; return `<td title="${n!=null?n+' episodes':''}">${pct(v[c])}${n!=null?` <span style="color:#aaa">(${n})</span>`:''}</td>`;};
  const ecols=[['all','overall'],['atomic_seen','atomic-seen'],
               ['composite_seen','composite-seen'],['composite_unseen','composite-unseen']];
  // sortable columns: key -> how to extract the sort value from a method row
  const sortKeys={overall:v=>v.all, 'atomic-seen':v=>v.atomic_seen, 'composite-seen':v=>v.composite_seen,
                  'composite-unseen':v=>v.composite_unseen, 'avg s/ep':v=>v.avg_seconds_per_episode};
  window._epSort=window._epSort||{key:'overall',dir:-1};   // default: overall descending
  // Method-type toggles: a method's name carries tag tokens (progact/progreg/progcls + deviations
  // noexec/nocond/nostate/notask/noanchor/noanchorstate). Show only methods whose name contains
  // AT LEAST ONE checked token; if none are checked, show all. Filters BOTH #2 and #2b.
  // debug-* runs are hidden here too (same convention as #0); the token chips below are built from
  // the visible set so a debug-only token never offers a chip that filters to nothing.
  const ALL_METHODS=Object.keys(d.episode).filter(m=>window._showDebug||!isDebugMethod(m));
  const TOKENS=['progact','progreg','progcls','noexec','nocond','nostate','noanchorstate','noanchor','notask','nogrip'];
  // only offer tokens that actually appear in the loaded methods
  const availTokens=TOKENS.filter(tk=>ALL_METHODS.some(m=>m.toLowerCase().includes(tk)));
  window._epTokens=window._epTokens||{};   // token -> checked
  const methodOn=(m)=>{const ml=m.toLowerCase();
    // debug gate first: it must apply even with no token checked, otherwise the "nothing checked ->
    // show all" shortcut below would let debug-* runs back into #2/#2b.
    if(isDebugMethod(m)&&!window._showDebug)return false;
    const on=availTokens.filter(tk=>window._epTokens[tk]);
    if(!on.length)return true;                       // nothing checked -> show all
    return on.some(tk=>ml.includes(tk));};           // OR across checked tokens
  function renderEpFilter(){
    const bar=document.getElementById('epfilter'); if(!bar)return;
    const chip=(tk)=>{const c=!!window._epTokens[tk];
      const col=tk.startsWith('progact')?'#e6f0fb':tk.startsWith('progreg')?'#fdeaea':'#eee';
      return `<label style="cursor:pointer;padding:2px 8px;border-radius:10px;border:1px solid ${c?'#b0431c':'#ccc'};background:${c?col:'#fff'};font-weight:${c?'700':'400'}">`+
             `<input type="checkbox" data-tk="${tk}" ${c?'checked':''} style="margin-right:4px">${tk}</label>`;};
    bar.innerHTML='<span style="color:#888">show:</span>'+availTokens.map(chip).join('')+
      `<button id="epclear" style="margin-left:6px;font-size:11px">all</button>`;
    bar.querySelectorAll('input[data-tk]').forEach(cb=>cb.onchange=()=>{
      window._epTokens[cb.dataset.tk]=cb.checked; renderEpTable(); renderTaskTab();});
    const clr=document.getElementById('epclear'); if(clr)clr.onclick=()=>{
      window._epTokens={}; renderEpFilter(); renderEpTable(); renderTaskTab();};
  }
  function renderEpTable(){
    const {key,dir}=window._epSort;
    const getv=sortKeys[key]||sortKeys.overall;
    const rows=Object.entries(d.episode).filter(([m])=>methodOn(m)).sort((a,b)=>{
      const va=getv(a[1]), vb=getv(b[1]);
      if(va==null&&vb==null)return 0; if(va==null)return 1; if(vb==null)return -1;
      return dir*(vb-va);});
    const arrow=c=>window._epSort.key===c?(dir<0?' ▼':' ▲'):'';
    const hdr=(label)=>`<th class="sortable" data-k="${label}" style="cursor:pointer" title="click to sort">${label}${arrow(label)}</th>`;
    let hh="<table><tr><th class='exp'>method</th><th>n</th>"+ecols.map(c=>hdr(c[1])).join("")+hdr('avg s/ep')+"</tr>";
    // color the method cell by progress-head family: progact* = light blue, progreg* = light red
    // (progcls / other = default). Keyed on the token after the vN_ prefix.
    rows.forEach(([m,v])=>{
      const bg=mColor(m); const st=bg?` style="background:${bg}"`:'';
      hh+=`<tr><td class='exp'${st}>${mName(m)}</td><td>${v.n}</td>`+ecols.map(c=>cell(v,c[0],v.counts)).join("")+
         `<td>${v.avg_seconds_per_episode!=null?(+v.avg_seconds_per_episode).toFixed(1):'–'}</td></tr>`;});
    document.getElementById('eptab').innerHTML=hh+"</table>";
    document.querySelectorAll('#eptab th.sortable').forEach(th=>th.onclick=()=>{
      const k=th.dataset.k;
      if(window._epSort.key===k)window._epSort.dir*=-1; else window._epSort={key:k,dir:-1};
      renderEpTable();});
  }
  // #2b per-task matrix: rows = tasks (grouped by split), cols = methods (filtered), cell = s/n.
  // Wrapped in a function so the type-toggles re-render it alongside #2.
  const taskInfo={};
  Object.keys(d.episode).forEach(m=>{const pt=d.episode[m].per_task||{}; for(const t in pt){taskInfo[t]=taskInfo[t]||pt[t].split;}});
  const splitOrder={atomic_seen:0,composite_seen:1,composite_unseen:2};
  const tasks=Object.keys(taskInfo).sort((a,b)=>{
    const sa=splitOrder[taskInfo[a]]??9, sb=splitOrder[taskInfo[b]]??9;
    return sa!==sb?sa-sb:a.localeCompare(b);});
  const cntCell=(s,n,rate,bold)=>{
    if(n==null||n===0)return '<td style="color:#ccc">–</td>';
    const w=bold?'font-weight:700;':'';
    const r=rate!=null?rate:(s/n);
    return `<td title="${(100*r).toFixed(1)}%" style="${w}background:rgba(46,139,61,${(0.10+0.5*r).toFixed(2)})">${s}<span style="color:#888">/${n}</span></td>`;};
  const splitLabel={atomic_seen:'atomic-seen',composite_seen:'composite-seen',composite_unseen:'composite-unseen'};
  function renderTaskTab(){
    const emethods=Object.keys(d.episode).filter(methodOn);  // respect the type-toggles
    const aggRow=(label,aggKey,bg)=>`<tr><td class='exp' style="background:${bg};font-weight:700">${label}</td>`+
        emethods.map(m=>cntCell((d.episode[m].succ_counts||{})[aggKey], (d.episode[m].counts||{})[aggKey],
                                d.episode[m][aggKey], true)).join("")+"</tr>";
    let th="<table><tr><th class='exp'>task</th>"+emethods.map(m=>`<th>${mName(m).replace(/_/g,' ')}</th>`).join("")+"</tr>";
    th+=aggRow('TOTAL (all tasks)','all','#dfe6f5');
    for(const sp of ['atomic_seen','composite_seen','composite_unseen']){
      th+=aggRow(splitLabel[sp]+' — overall', sp, '#eef');
      tasks.filter(t=>taskInfo[t]===sp).forEach(t=>{
        th+=`<tr><td class='exp'>${t}</td>`+emethods.map(m=>{const e=(d.episode[m].per_task||{})[t];
          return cntCell(e?e.s:null, e?e.n:null, e?e.rate:null, false);}).join("")+"</tr>";});
    }
    document.getElementById('tasktab').innerHTML=th+"</table>";
  }
  renderEpFilter(); renderEpTable(); renderTaskTab();
  // #0 COMBINED: per-method success + timing + extrapolated cost for the full 50-task benchmark.
  (function(){
    const C=d.combine||{};
    // DEBUG RUNS: by convention any method named "debug-*" is a throwaway (a few episodes, often a
    // half-finished sweep), so its rate is noise and it would otherwise sit in the table next to
    // 1000-episode runs inviting a false comparison. They stay LOADED and one click away, but are
    // hidden by default. Re-rendering on toggle keeps #0 and #0b consistent, since both derive
    // their method list from `ms` below.
    // Two INDEPENDENT toggles, because the two kinds of run are hidden for different reasons:
    //   debug-*   throwaway runs (small n, often unfinished) -- default OFF, their rate is noise.
    //   *-memory  the memory VARIANT (narrate -> recipe -> warm plan). A different PIPELINE, not a
    //             different checkpoint, and it is typically run on one split only, so its "overall"
    //             is not comparable with a cold run's overall. Default ON (it is a real result), but
    //             one click removes it when comparing cold runs head to head.
    const ALL=Object.keys(C);
    window._recOf=C;          // so mSuffix() can read each method's `kind`
    const DBG=ALL.filter(isDebugMethod);
    // The *-memory toggle is GONE: every terse-goal arm is now identified by isUnseenShortMethod
    // (including ...-unseenshort-memory), so a box keyed on the -memory suffix only ever hid part of
    // that family. isMemoryMethod survives for the /combine picker label.
    const US=ALL.filter(isUnseenShortMethod);
    // MANIFEST SIZE, the third toggle. The eval set grew from 50x20=1000 episodes to 50x30=1500 (the
    // leaderboard denominator), and runs of both sizes sit in this table. Their overalls are not the
    // same measurement: the 1500 set CONTAINS the 1000 set as its first 20 episodes per task, so an
    // old run's rate is computed on a subset and ranking the two together silently compares a rate on
    // 1000 episodes with a rate on 1500. Default view is the 1500-episode runs only.
    //
    // Detected from the DATA, not from the method name: no 20-per-task run can have a task with more
    // than 20 episodes, so max-per-task > 20 identifies the 1500 manifest even for a sweep that is
    // only part-way through. debug-* runs are exempt -- they are tiny by design and already behind
    // their own opt-in box, so this filter would hide them a second time for the wrong reason.
    const maxTaskN=m=>{const pt=(C[m]||{}).per_task||{}; let x=0;
      for(const t in pt){const n=pt[t].n||0; if(n>x)x=n;} return x;};
    const isEval30=m=>maxTaskN(m)>20;
    const NT=ALL.filter(m=>isNewTaskMethod(m,C[m]));
    // "OTHER RUNS" = the superseded -v2 rule arm, experimental -retry arms, and the older
    // 20-episodes-per-task sweeps, merged into one box because they are the same kind of thing: real
    // results on a rule set, stopping policy, or denominator that is not currently the default
    // comparison. newtask is excluded (its 20 episodes/task would otherwise read as a 1000-episode
    // run) and so are debug runs, which have their own box.
    const isOtherRun=m=>!isDebugMethod(m)&&!isNewTaskMethod(m,C[m])
                        &&(isV2Method(m)||isRetryMethod(m)||!isEval30(m));
    const OTH=ALL.filter(isOtherRun);
    // debug-* excluded so the count matches what ticking the box reveals: a debug human-recipe run
    // stays behind the debug box, and counting it here would promise a row that never appears.
    const HR=ALL.filter(m=>isHumanRecipeMethod(m)&&!isDebugMethod(m));
    const BL=ALL.filter(m=>isBaselineMethod(m,C[m]));
    if(window._showDebug===undefined)window._showDebug=false;    // default OFF
    // Merged -v2 + 20/task box, and the newtask box. Both default OFF: the default view is the arms
    // currently being compared on the 1500-episode manifest.
    if(window._showOther===undefined)
      window._showOther=(localStorage.getItem('cmbShowOther')==='1');
    if(window._showNT===undefined)
      window._showNT=(localStorage.getItem('cmbShowNT')==='1');
    // Default ON (a real result), but REMEMBERED like `refined` rather than reset on every reload:
    // hiding the superseded arm is a comparison mode you stay in for a whole session, unlike the
    // debug/memory boxes which you flick on to check one thing.
    // Default ON and REMEMBERED: whether the external baseline belongs in the table is
    // a comparison mode you stay in, not something you flick on to check one number.
    if(window._showBaseline===undefined)
      window._showBaseline=(localStorage.getItem('cmbBaseline')!=='0');
    // REFINED: render the SAME episodes re-scored under RoboCasa's official per-task step horizon --
    // a success counts only if the env's success check fired within `horizon` cumulative env steps
    // (examples/robocasa/horizon_gate.py; precomputed per method by extract_combine_results.py).
    // PRESENTATION ONLY: it gates which episodes count as successes and touches no eval data. Both #0
    // and #0b switch together, including the ranking, so a screenshot is never half refined. Default
    // ON by default, because it is the only scoring comparable with a published RoboCasa number; the
    // choice is remembered, so unticking it sticks ('0' is stored explicitly).
    if(window._refined===undefined)
      window._refined=(localStorage.getItem('cmbRefined')!=='0');
    if(window._showUS===undefined)
      window._showUS=(localStorage.getItem('cmbShowUS')==='1');   // default OFF
    // Default OFF, remembered: including a human-written-recipe arm is a deliberate comparison mode,
    // not something you flick on to check one number.
    if(window._showHR===undefined)
      window._showHR=(localStorage.getItem('cmbShowHR')==='1');   // default OFF
    if(window._onlyUS===undefined)
      window._onlyUS=(localStorage.getItem('cmbOnlyUS')==='1');   // default OFF
    // FAMILY FILTERS. Two INDEPENDENT axes -- the System1 progress head and the System2 planner --
    // because a run is a pair of checkpoints and either half can be the thing you want to hold fixed.
    // Exclusive WITHIN an axis (progact-only and progreg-only cannot both be on; the second click
    // releases the first), and ANDed ACROSS axes, so "progreg-only + qwen3vl-only" is one cell of the
    // 2x2. Read off the method NAME, which is where the pair is encoded.
    //
    // A run with no family -- every external baseline (xr1 / abot / pi05) -- is hidden while any family
    // filter is on. That is the point of the filter (compare like with like) but it is easy to forget,
    // so the button title says it.
    const s1Fam=m=>{const x=String(m||'').toLowerCase();
      return x.includes('progact')?'progact':x.includes('progreg')?'progreg':'';};
    const s2Fam=m=>{const x=String(m||'').toLowerCase();
      return x.includes('qwen3vl')?'qwen3vl':x.includes('qwen35')?'qwen35':'';};
    if(window._famS1===undefined)window._famS1=(localStorage.getItem('cmbFamS1')||'');
    if(window._famS2===undefined)window._famS2=(localStorage.getItem('cmbFamS2')||'');
    const famOK=m=>(!window._famS1||s1Fam(m)===window._famS1)
                  &&(!window._famS2||s2Fam(m)===window._famS2);
    // ONLY-MODE short-circuits every other visibility box: the point is to see the terse-goal arms
    // and nothing else, so a stale debug/v2/baseline tick cannot silently drop one of them. The family
    // filters still apply -- they narrow WHICH arms, not which kind.
    const ms=(window._onlyUS ? ALL.filter(isUnseenShortMethod)
      : ALL.filter(m=>(window._showDebug||!isDebugMethod(m))
                        && (window._showBaseline||!isBaselineMethod(m,C[m]))
                        && (window._showUS||!isUnseenShortMethod(m))
                        && (window._showHR||!isHumanRecipeMethod(m))
                        && (window._showNT||!isNewTaskMethod(m,C[m]))
                        && (window._showOther||!isOtherRun(m)))).filter(famOK);
    const dbgBar=document.getElementById('cmbdebug');
    if(dbgBar){
      const boxes=[];
      if(DBG.length)boxes.push(`<label style="cursor:pointer;color:#888;margin-right:12px">`
        +`<input type="checkbox" id="cmbdbgcb" ${window._showDebug?'checked':''} `
        +`style="margin-right:4px">show ${DBG.length} debug-* run${DBG.length>1?'s':''}</label>`);
      if(US.length){
        // The first box is greyed and struck through while only-mode is on, because only-mode makes it
        // inert -- a live unticked box would misdescribe what is on screen.
        const ovr=window._onlyUS;
        boxes.push(`<label style="cursor:pointer;color:${ovr?'#ccc':'#888'};margin-right:12px`
          +`${ovr?';text-decoration:line-through':''}" `
          +`title="The terse-goal family: the 16 composite_unseen tasks x 30 run with one-line goals `
          +`(...-unseenshort cold plan, -memory and -gemini warm plans from a recipe library, plus the `
          +`flat baseline). 480 episodes over the hardest split only, so its overall is NOT comparable `
          +`with a 50-task overall.${ovr?' (overridden by only-mode)':''}">`
          +`<input type="checkbox" id="cmbuscb" ${window._showUS?'checked':''} `
          +`style="margin-right:4px">show ${US.length} unseen-short run${US.length>1?'s':''} `
          +`<span style="color:#aaa">(terse goals, 16 tasks)</span></label>`);
        boxes.push(`<label style="cursor:pointer;color:${window._onlyUS?'#b45309':'#888'};`
          +`margin-right:12px" `
          +`title="Show ONLY the terse-goal arms and hide everything else, so they sit on one `
          +`denominator and can be read against each other. Overrides every other visibility box.">`
          +`<input type="checkbox" id="cmbusonlycb" ${window._onlyUS?'checked':''} `
          +`style="margin-right:4px">unseen-short ONLY</label>`);
      }
      if(HR.length)boxes.push(`<label style="cursor:pointer;color:#888;margin-right:12px" `
        +`title="The composite_unseen split re-run with a HAND-WRITTEN recipe given to the planner. `
        +`Every other arm infers the plan itself, so this one is not truth-free and is not a like-for-`
        +`like entry in the ranking. It also borrows its atomic-seen and composite-seen splits from the `
        +`matching -inst run, so its overall is composed rather than measured end to end.">`
        +`<input type="checkbox" id="cmbhrcb" ${window._showHR?'checked':''} `
        +`style="margin-right:4px">include ${HR.length} human-recipe run${HR.length>1?'s':''} `
        +`<span style="color:#aaa">(hand-written recipe)</span></label>`);
      if(OTH.length)boxes.push(`<label style="cursor:pointer;color:#888;margin-right:12px" `
        +`title="The superseded -v2 rule arm, experimental -retry arms, and older 20-episodes-per-task `
        +`sweeps. Real results, but on a rule set, stopping policy, or denominator that is not the `
        +`default comparison.">`
        +`<input type="checkbox" id="cmbothcb" ${window._showOther?'checked':''} `
        +`style="margin-right:4px">show ${OTH.length} other run${OTH.length>1?'s':''} `
        +`<span style="color:#aaa">(-v2, -retry, and 20/task sweeps)</span></label>`);
      if(NT.length)boxes.push(`<label style="cursor:pointer;color:#888;margin-right:12px" `
        +`title="The 24-task NEW-TASK held-out set. Outside the 50-task manifest, so it has no overall `
        +`column and no refined scoring -- the horizon gate is defined against the manifest.">`
        +`<input type="checkbox" id="cmbntcb" ${window._showNT?'checked':''} `
        +`style="margin-right:4px">show ${NT.length} newtask run${NT.length>1?'s':''} `
        +`<span style="color:#aaa">(24 held-out tasks)</span></label>`);
      if(BL.length)boxes.push(`<label style="cursor:pointer;color:#888;margin-right:12px" `
        +`title="Flat-policy reference (Xiaomi-Robotics-1 RoboCasa365): one policy, no System2, no `
        +`turns, so the turns column is blank. Generated WITH RoboCasa's official per-task step `
        +`horizon as the cap, so its raw rate is already horizon-compliant -- compare it against our `
        +`REFINED column, not our raw one.">`
        +`<input type="checkbox" id="cmbblcb" ${window._showBaseline?'checked':''} `
        +`style="margin-right:4px">show ${BL.length} baseline run${BL.length>1?'s':''} `
        +`<span style="color:#aaa">(flat policy, no System2)</span></label>`);
      // Family buttons: counts are of what the button would SHOW, so a 0 never hides silently.
      const famBtn=(axis,val,label)=>{
        const cur=axis==='s1'?window._famS1:window._famS2;
        const on=cur===val;
        const n=ALL.filter(m=>(axis==='s1'?s1Fam(m):s2Fam(m))===val).length;
        return `<button class="fambtn" data-axis="${axis}" data-val="${val}" `
          +`title="Show only ${label} runs. Exclusive with the other button on this axis; combines `
          +`with the planner/head axis. Runs with no ${axis==='s1'?'progress head':'planner'} in their `
          +`name (the external baselines) are hidden while this is on." `
          +`style="font-size:11px;margin-right:4px;cursor:pointer;`
          +`${on?'background:#b45309;color:#fff;border:1px solid #92400e':'background:#f3f4f6'}">`
          +`${label}-only (${n})</button>`;};
      boxes.push(`<span style="margin-right:10px">`
        +famBtn('s1','progact','progact')+famBtn('s1','progreg','progreg')
        +`<span style="color:#ddd">|</span> `
        +famBtn('s2','qwen35','qwen35')+famBtn('s2','qwen3vl','qwen3vl')
        +((window._famS1||window._famS2)
          ?`<button id="famclear" style="font-size:11px;margin-left:4px;cursor:pointer">clear</button>`
          :'')+`</span>`);
      boxes.push(`<label style="cursor:pointer;color:${window._refined?'#b45309':'#888'}" `
        +`title="A success counts only if the env's success check fired within RoboCasa's official `
        +`per-task step horizon (450-4350 env steps, robocasa dataset_registry). Our loop budgets per `
        +`subgoal and per turn, never over total env steps, so an episode can run past it. `
        +`Presentation only -- gates which episodes count, changes no data.">`
        +`<input type="checkbox" id="cmbrefcb" ${window._refined?'checked':''} `
        +`style="margin-right:4px">refined `
        +`<span style="color:#aaa">(official RoboCasa step horizon)</span></label>`);
      dbgBar.innerHTML=boxes.join('');
    }
    const cb=document.getElementById('cmbdbgcb');
    if(cb)cb.onchange=()=>{window._showDebug=cb.checked; load();};
    const uscb=document.getElementById('cmbuscb');
    if(uscb)uscb.onchange=()=>{window._showUS=uscb.checked;
      localStorage.setItem('cmbShowUS', window._showUS?'1':'0'); load();};
    const usonly=document.getElementById('cmbusonlycb');
    if(usonly)usonly.onchange=()=>{window._onlyUS=usonly.checked;
      localStorage.setItem('cmbOnlyUS', window._onlyUS?'1':'0'); load();};
    const hrcb=document.getElementById('cmbhrcb');
    if(hrcb)hrcb.onchange=()=>{window._showHR=hrcb.checked;
      localStorage.setItem('cmbShowHR', window._showHR?'1':'0'); load();};
    const ocb=document.getElementById('cmbothcb');
    if(ocb)ocb.onchange=()=>{window._showOther=ocb.checked;
      localStorage.setItem('cmbShowOther', window._showOther?'1':'0'); load();};
    const ntcb=document.getElementById('cmbntcb');
    if(ntcb)ntcb.onchange=()=>{window._showNT=ntcb.checked;
      localStorage.setItem('cmbShowNT', window._showNT?'1':'0'); load();};
    const blcb=document.getElementById('cmbblcb');
    if(blcb)blcb.onchange=()=>{window._showBaseline=blcb.checked;
      localStorage.setItem('cmbBaseline', window._showBaseline?'1':'0'); load();};
    document.querySelectorAll('.fambtn').forEach(b=>{b.onclick=()=>{
      const axis=b.dataset.axis, val=b.dataset.val;
      const key=axis==='s1'?'_famS1':'_famS2', store=axis==='s1'?'cmbFamS1':'cmbFamS2';
      window[key]=(window[key]===val)?'':val;      // clicking the active one releases it
      localStorage.setItem(store, window[key]); load();};});
    const fclr=document.getElementById('famclear');
    if(fclr)fclr.onclick=()=>{window._famS1=''; window._famS2='';
      localStorage.setItem('cmbFamS1',''); localStorage.setItem('cmbFamS2',''); load();};
    const rcb=document.getElementById('cmbrefcb');
    if(rcb)rcb.onchange=()=>{window._refined=rcb.checked;
      localStorage.setItem('cmbRefined', window._refined?'1':'0'); load();};
    // Guard on EITHER kind being hidden: with only *-memory runs present and its box unticked,
    // DBG.length alone would be 0 and the page would claim there are no runs at all.
    if(!ms.length){document.getElementById('cmbtab').innerHTML=
      '<span style="color:#888">'+((DBG.length||US.length||HR.length||OTH.length||NT.length||BL.length)
        ? (window._onlyUS
            ? 'No unseen-short runs to show \u2014 untick \u201cunseen-short ONLY\u201d.'
            : 'Only hidden runs present (debug-* / other / baseline / unseen-short / newtask / '
              +'human-recipe) \u2014 tick a box above to see them.')
        : 'No combined runs yet — see /combine.')+'</span>';return;}
    // ONE table: overall + per-split, RANKED best -> worst by overall rate.
    //
    // Previously this was two tables (overall/timing, then per-split rates), which meant reading a
    // method's headline number in one and its split breakdown in another, with the rows in
    // alphabetical order so the best run was wherever its name happened to sort. Merged and ranked.
    //
    // DROPPED COLUMNS: errors / total / ETA serial / ETA 8 GPU. The ETAs extrapolated a measured
    // mean to a 25,307-episode benchmark that is not the denominator anyone uses now (the manifest
    // is 1500), and `total` was wall-clock for the sweep, which is an operational number rather than
    // a result. `errors` is preserved as a tooltip on the rate cell together with the termination
    // breakdown, so nothing is actually lost -- it is just no longer taking a numeric column.
    //
    // EVERY cell carries BOTH the percentage and the #success/#episodes it came from: a bare rate is
    // unreadable when the denominators differ across methods (1500 vs 320 vs 20) and across splits
    // (540 vs 480), which is exactly the situation here.
    const sp=['atomic_seen','composite_seen','composite_unseen','other'];
    // Only show a split column some method actually has episodes for ('other' is normally empty).
    const spShown=sp.filter(x=>ms.some(m=>((C[m].per_split||{})[x]||{}).n));
    // FULL-MANIFEST COVERAGE. A run evaluated on only one slice -- composite-unseen (16 tasks) or the
    // newtask set (34 tasks) -- has no comparable "overall": averaging the hardest split alone against
    // a 50-task average compares two different measurements. Such rows get an EMPTY overall cell and
    // are ranked BELOW every complete run, so the leaderboard reads top-down as "best full eval" and
    // the partial runs sit underneath rather than being interleaved by a number that is not the same
    // quantity. Their per-split cells still show, which is where their result actually lives.
    const MANIFEST=['atomic_seen','composite_seen','composite_unseen'];
    const fullCov=m=>{const P=C[m].per_split||{};
      return MANIFEST.every(x=>P[x]&&P[x].n);};
    // BORROWED SPLITS -- deliberately scoped to the two human-recipe method patterns. The ordinary
    // `<x>-inst-human-recipe` arm borrows from `<x>-inst`; the official-step retry arm
    // `<x>-inst-human-recipe-retry` borrows from `<x>-inst-retry`. Each recipe arm re-runs only
    // composite-unseen, so filling atomic-seen and composite-seen from its exact cold-plan twin gives
    // a comparable 1500-episode overall without mixing rollout-limit policies.
    //
    // NOT a general "longest full-coverage prefix" rule. That was tried and it silently paired
    // `xiaomi-robo1-unseenshort` with `xiaomi-robo1` -- the prefix matched and the size guard passed,
    // so a TERSE-GOAL arm would have been handed full-goal atomic and composite-seen cells and ranked
    // on them. Borrowing is only sound when the modification is confined to the split being re-run,
    // and that is a fact about the arm, not about its name. So it is enumerated, not inferred: adding
    // another borrowing arm means adding it here on purpose.
    const donorOf=m=>{
      const retry=/^(.*-inst)-human-recipe-retry$/.exec(m||'');
      const ordinary=/^(.*-inst)-human-recipe$/.exec(m||'');
      if(!retry&&!ordinary)return null;
      const dn=retry ? `${retry[1]}-retry` : ordinary[1];
      if(!C[dn]||!fullCov(dn))return null;
      // Still guarded on size: if this arm's own split is smaller than the donor's (a sweep still in
      // flight), composing would put a mongrel denominator in the overall column.
      const P=C[m].per_split||{}, Q=C[dn].per_split||{};
      const own=MANIFEST.filter(y=>P[y]&&P[y].n);
      if(!own.length||!own.every(y=>Q[y]&&P[y].n===Q[y].n))return null;
      return dn;};
    // Per split: the block to render and, when it is not this arm's own, where it came from.
    const blocksOf=m=>{const P=C[m].per_split||{}, dn=donorOf(m), out={};
      MANIFEST.forEach(x=>{
        if(P[x]&&P[x].n){out[x]={b:P[x],from:null};}
        else if(dn){const Q=C[dn].per_split||{}; if(Q[x]&&Q[x].n)out[x]={b:Q[x],from:dn};}});
      return out;};
    // Coverage AFTER borrowing -- what decides whether the row gets an overall and where it ranks.
    const fullCovEff=m=>{const B=blocksOf(m); return MANIFEST.every(x=>B[x]);};
    const borrowedIn=m=>MANIFEST.filter(x=>(blocksOf(m)[x]||{}).from);
    // SUC reads a stat block through the refined toggle, and is the ONLY place the two scorings are
    // chosen between -- #0, #0b and the ranking all go through it, so they cannot disagree.
    // Returns null for an empty block, and {na:true} when refined was asked for but the block has no
    // refined fields (a method extracted before the gate existed, or one whose raw episodes were
    // pruned so the step-to-success is unrecoverable). Denominator `n` is identical either way: the
    // horizon can only demote a success, never create one.
    const SUC=b=>{
      if(!b||!b.n)return null;
      if(!window._refined){const s=b.n_success??b.s??0;
        return {n:b.n,s:s,r:(b.rate!=null?b.rate:s/b.n)};}
      if(b.n_success_refined==null)return {n:b.n,na:true};
      return {n:b.n,s:b.n_success_refined,
              r:(b.rate_refined!=null?b.rate_refined:b.n_success_refined/b.n),
              unk:b.n_refined_unknown||0,drop:(b.n_success??0)-b.n_success_refined};};
    // overall = the three splits combined, summed from per_split so it is always consistent with the
    // cells beside it (a partial sweep can fill the splits unevenly).
    const overallOf=m=>{const P=C[m].per_split||{}; let on=0,os=0,na=false;
      for(const x of sp){const b=SUC(P[x]); if(!b)continue; on+=b.n;
        if(b.na){na=true;}else{os+=b.s;}}
      return on?(na?{n:on,na:true}:{n:on,n_success:os,rate:os/on}):null;};
    // Rank on the top-level rate, falling back to the split-derived one. Under the toggle this ranks
    // by the refined rate, so the order matches the numbers on screen.
    const rateIn=b=>{const u=SUC(b); return (u&&!u.na&&u.r!=null)?u.r:null;};
    // Composed overall: sum the three blocks after borrowing. For a run with no borrowing this is the
    // same number as before (its own three splits).
    const ovEff=m=>{if(!fullCovEff(m))return null;
      const B=blocksOf(m); let n=0,ns=0,na=false;
      MANIFEST.forEach(x=>{const u=SUC(B[x].b); if(!u)return; n+=u.n; if(u.na)na=true; else ns+=u.s;});
      return n?(na?{n:n,na:true}:{n:n,n_success:ns,rate:ns/n}):null;};
    const ovRate=m=>{const o=ovEff(m); return (o&&!o.na&&o.rate!=null)?o.rate:null;};
    const rateOf=m=>{const r=ovRate(m); if(r!=null)return r;
      const t=SUC(C[m]); if(t&&!t.na&&t.r!=null)return t.r;
      const o=overallOf(m); return (o&&o.rate!=null)?o.rate:-1;};
    // Complete runs first, each group ranked by rate.
    // ROW GROUPS. Rows with a comparable 1500-episode overall come first; everything measured on a
    // different question is separated below it under its own heading, because those numbers are not on
    // the same axis and interleaving them by rate invites reading down the column as a ranking.
    //   0 comparable  full manifest coverage, or composed from a donor
    //   1 terse-goal  the unseen-short family: 16 tasks, one-line goals
    //   2 newtask     the held-out set (size read from the data -- it grows)
    //   3 partial     anything else without full coverage (a sweep still in flight)
    const grpOf=m=>fullCovEff(m)?0:isUnseenShortMethod(m)?1:isNewTaskMethod(m,C[m])?2:3;
    // Size of the newtask set as the DATA reports it, not as a literal: it went 24 -> 34 and the old
    // hardcoded label then described a set that no longer existed.
    const NT_TASKS=new Set();
    ALL.forEach(m=>{if(isNewTaskMethod(m,C[m]))Object.keys(C[m].per_task||{}).forEach(t=>NT_TASKS.add(t));});
    const GRP_LABEL={1:'terse-goal (unseen-short) — 16 composite-unseen tasks, one-line goals',
                     2:'newtask — the '+(NT_TASKS.size?NT_TASKS.size+'-task ':'')
                       +'held-out set, outside the 50-task manifest',
                     3:'partial coverage — no comparable overall'};
    const ranked=ms.slice().sort((a,b)=>(grpOf(a)-grpOf(b))||(rateOf(b)-rateOf(a)));
    // COLUMN BEST. One highlight per column, on the leading cell(s) only -- previously the overall
    // column was tinted on every row, which marked nothing. Computed over the rows CURRENTLY VISIBLE
    // (`ranked` is rebuilt by load() on every checkbox change) and under the CURRENT scoring, so
    // ticking a box or flipping `refined` moves the highlight to whatever now leads. Ties are all
    // highlighted rather than picking one arbitrarily.
    // Best-in-column is computed over the COMPARABLE group only. A terse-goal or newtask row cannot win
    // a column it is not competing in -- its composite-unseen number answers a different question than
    // the full-goal one directly above it.
    const cmpRows=ranked.filter(m=>grpOf(m)===0);
    const bestOf={overall:null};
    cmpRows.forEach(m=>{const r=ovRate(m);
      if(r!=null&&(bestOf.overall==null||r>bestOf.overall))bestOf.overall=r;});
    spShown.forEach(x=>{let mx=null;
      cmpRows.forEach(m=>{const e=blocksOf(m)[x];
        const r=rateIn(e&&!e.from?e.b:((C[m].per_split||{})[x]));
        if(r!=null&&(mx==null||r>mx))mx=r;});
      bestOf[x]=mx;});
    // Rates are exact fractions here, but compare with a tolerance so 24/30 vs 0.8 cannot miss.
    const isBest=(r,key)=>r!=null&&bestOf[key]!=null&&Math.abs(r-bestOf[key])<1e-9;
    const HL=' style="background:#eef2fb;box-shadow:inset 0 0 0 2px #7fa8f0"';
    // Named cellSp, NOT cell: #0b below declares its own `cell` in this same block scope, and a
    // duplicate `const` is a SyntaxError that kills the whole script -- the page then hangs forever
    // on "loading stats..." because load() never runs.
    const cellSp=(b,key,from)=>{
      const u=SUC(b);
      if(!u)return '<td style="color:#ccc">–</td>';
      if(u.na)return `<td style="color:#ccc" title="not scorable under the horizon gate — `
        +`re-run scripts/extract_combine_results.py, or the raw episodes were pruned">n/a</td>`;
      const drop=u.drop?` · refined: ${u.drop} win${u.drop>1?'s':''} outside the horizon`:'';
      // A BORROWED cell is greyed and italic and never counts as the column best -- it is another
      // arm's measurement shown here for completeness, so highlighting it would credit this row for a
      // result it did not produce.
      const top=!from&&isBest(u.r,key);
      const st=from?' style="color:#999;font-style:italic;background:#fbfbfc"':(top?HL:'');
      return `<td title="${from?'borrowed from '+from+' — NOT measured in this arm. ':''}`
        +`${b.avg_seconds!=null?b.avg_seconds+'s/ep':''}${drop}`
        +`${top?' · best in this column':''}"${st}>`
        +`<b>${pct(u.r)}</b><br><span style="color:#888;font-size:11px">${u.s}/${u.n}</span>`
        +`${from?'<br><span style="color:#aaa;font-size:10px">borrowed</span>':''}</td>`;};
    let h="<table class=cmbmain><tr><th>#</th><th class='exp'>method</th><th>overall</th>"
      +spShown.map(x=>`<th title="${x==='other'
          ?'the 24-task NEW-TASK held-out set, outside the 50-task manifest. The horizon gate demotes '
           +'nothing on these tasks, so refined equals raw and the two scorings show the same number.'
          :x}">${splName(x)}</th>`).join('')
      +"<th>avg s/ep</th><th>avg turns</th></tr>";
    const NCOL=5+spShown.length;
    let lastGrp=null;
    ranked.forEach((m,i)=>{const v=C[m], P=v.per_split||{};
      const g=grpOf(m);
      if(g!==lastGrp&&GRP_LABEL[g]){
        h+=`<tr><td colspan="${NCOL}" style="background:#f2f2f5;border-top:2px solid #ccc;`
          +`color:#555;font-size:11px;font-weight:700;letter-spacing:.03em;padding:5px 8px">`
          +`${GRP_LABEL[g]}</td></tr>`;}
      lastGrp=g;
      const tt=Object.entries(v.terminations||{}).sort((a,b2)=>b2[1]-a[1])
                 .map(([k,n])=>`${k}: ${n}`).join(' · ');
      // Prefer the top-level record for the overall cell (it is what the extractor computed over
      // every episode); fall back to the split sum when a method predates per_split.
      const t0=SUC(v);
      const B=blocksOf(m), bor=borrowedIn(m);
      // Overall comes from the composed blocks, so an arm that re-ran one split still gets a
      // comparable 1500-episode figure. Runs with no full-coverage donor still get nothing.
      const ov=ovEff(m);
      // Under the toggle the name says "(refined)" too, so a screenshot of this table cannot be
      // mistaken for the raw one.
      // No "(refined)" suffix on the name: it doubled the width of every row label, and the toggle
      // above the table already says which scoring is on.
      const nm=mName2(m);
      const dropTip=(window._refined&&t0&&!t0.na&&t0.drop)
        ? ` · horizon gate removed ${t0.drop} win${t0.drop>1?'s':''}`
        : (window._refined&&t0&&t0.na?' · not scorable under the horizon gate':'');
      // A COMPOSED overall is not this arm's own measurement end to end, so it is marked and it is
      // barred from winning the column -- otherwise a row could top the table on borrowed cells.
      const ovTop=!bor.length&&isBest(ovRate(m),'overall');
      const bTip=bor.length
        ? ` · COMPOSED: ${bor.map(splName).join(' + ')} borrowed from ${donorOf(m)}; only `
          +`${MANIFEST.filter(x=>B[x]&&!B[x].from).map(x=>(B[x].b.n||0)).reduce((a,c)=>a+c,0)} `
          +`of ${ov?ov.n:'?'} episodes were run in this arm`
        : '';
      h+=`<tr><td style="color:#888">${i+1}</td><td class='exp'>${nm}</td>`
        +`<td title="errors: ${v.n_error??0} · terminations — ${tt||'n/a'}${dropTip}${bTip}`
        +`${ovTop?' · best overall':''}"${ovTop?HL:''}>`
        +(!ov?`<span style="color:#ccc" title="evaluated on part of the manifest only `
              +`(${Object.keys(v.per_split||{}).map(splName).join(', ')}) — `
              +`no comparable overall">—</span>`
          :ov.na?'<span style="color:#ccc">n/a</span>'
          :`<b>${pct(ov.rate)}</b>`
           +(bor.length?'<span style="color:#b0431c" title="composed overall">&#8853;</span>':'')
           +`<br><span style="color:#888;font-size:11px">${ov.n_success}/${ov.n}</span>`)
        +`</td>`
        +spShown.map(x=>{const e=(x==='other')?(P[x]?{b:P[x],from:null}:null):(B[x]||null);
                         return e?cellSp(e.b,x,e.from):cellSp(P[x],x,null);}).join('')
        +`<td>${v.avg_seconds??'–'}</td><td>${v.avg_turns??'–'}</td></tr>`;});
    // No explanatory paragraph under the table, by request. Everything it said is either visible in
    // the table itself (rate + #success/#episodes per cell), in a hover title (errors, terminations,
    // s/ep, the refined drop), or on the toggle labels above.
    h+="</table>";
    document.getElementById('cmbtab').innerHTML=h;
    document.getElementById('cmbsplit').innerHTML='';
    // #0b per-task matrix: rows = tasks grouped by split, cols = methods.
    const order={atomic_seen:0,composite_seen:1,composite_unseen:2,other:3};
    const info={};
    ms.forEach(m=>{const pt=C[m].per_task||{};
      for(const t in pt) info[t]=info[t]||pt[t].split||'other';});
    const tasks=Object.keys(info).sort((a,b)=>
      (order[info[a]]??9)-(order[info[b]]??9) || a.localeCompare(b));
    // Same SUC as #0, so #0b follows the refined toggle in lockstep. A per-task block with no refined
    // fields shows n/a rather than a green 0, which would read as "this task never succeeded".
    const cell=(b)=>{
      const u=SUC(b);
      if(!u)return '<td style="color:#ccc">–</td>';
      if(u.na)return '<td style="color:#ccc" title="not scorable under the horizon gate">n/a</td>';
      const drop=u.drop?` · refined: -${u.drop} outside the ${b.horizon??'official'}-step horizon`:'';
      return `<td title="${(100*u.r).toFixed(1)}% · ${b.avg_seconds??'?'}s/ep${drop}" `
        +`style="background:rgba(46,139,61,${(0.10+0.5*u.r).toFixed(2)})">`
        +`${u.s}<span style="color:#888">/${u.n}</span></td>`;};
    // ---- COMPARE MODE ------------------------------------------------------------------------
    // Pick exactly two methods and the table collapses to those two columns plus a per-task Δ, so the
    // question "which tasks does A win or lose on" is answered by reading one column instead of
    // subtracting 50 pairs by eye. Δ is in EPISODES (A - B), not percentage points, because that is
    // the unit the cells are in and the unit the noise band is quoted in.
    if(window._cmpMode===undefined)window._cmpMode=false;
    if(!window._cmpSel)window._cmpSel=[];
    const cmpOn=!!(window._cmpRun && window._cmpRun.length===2
                   && window._cmpRun.every(m=>C[m]));
    const cols=cmpOn?window._cmpRun:ranked;
    const cbar=document.getElementById('cmbcmp');
    if(cbar){
      let cb='';
      if(!window._cmpMode && !cmpOn){
        cb=`<button id="cmpbtn" style="font-size:11px">compare…</button>`
          +`<span style="color:#888;margin-left:6px">pick two methods and diff them task by task</span>`;
      }else if(!cmpOn){
        cb=`<span style="color:#888">pick <b>two</b> methods:</span> `
          +ranked.map(m=>`<label style="margin-right:8px;cursor:pointer;white-space:nowrap">`
            +`<input type="checkbox" class="cmpck" data-m="${m}" `
            +`${window._cmpSel.includes(m)?'checked':''}> ${mShort(m)}</label>`).join('')
          +`<button id="cmpgo" style="font-size:11px;margin-left:4px" `
          +`${window._cmpSel.length===2?'':'disabled'}>start</button>`
          +`<button id="cmpoff" style="font-size:11px;margin-left:4px">cancel</button>`;
      }else{
        cb=`<b>comparing</b> A=${mShort(window._cmpRun[0])} &nbsp; B=${mShort(window._cmpRun[1])}`
          +`<span style="color:#888"> &nbsp;Δ = A − B in episodes; ▲ A better, ▼ B better; </span>`
          +`<b style="background:#fde68a">|Δ| ≥ ${CMP_HL}</b><span style="color:#888"> highlighted</span>`
          +`<button id="cmpoff" style="font-size:11px;margin-left:8px">exit compare</button>`;
      }
      cbar.innerHTML=cb;
      const gb=document.getElementById('cmpbtn');
      if(gb)gb.onclick=()=>{window._cmpMode=true; load();};
      document.querySelectorAll('.cmpck').forEach(el=>{el.onchange=()=>{
        const m=el.dataset.m;
        let s=window._cmpSel.filter(x=>x!==m);
        if(el.checked)s.push(m);
        window._cmpSel=s.slice(-2);        // keep the two most recent picks
        load();};});
      const go=document.getElementById('cmpgo');
      if(go)go.onclick=()=>{window._cmpRun=window._cmpSel.slice(0,2); window._cmpMode=false; load();};
      const off=document.getElementById('cmpoff');
      if(off)off.onclick=()=>{window._cmpRun=null; window._cmpMode=false; window._cmpSel=[]; load();};
    }
    // One Δ cell: counts from SUC, so it follows the refined toggle like every other cell here.
    const dcell=(ba,bb)=>{
      const A=SUC(ba), B=SUC(bb);
      if(!A||!B||A.na||B.na)return '<td style="color:#ccc">–</td>';
      const d=A.s-B.s;
      const arrow=d>0?'▲':(d<0?'▼':'=');
      const col=d>0?'#166534':(d<0?'#b91c1c':'#999');
      const hl=Math.abs(d)>=CMP_HL?';background:#fde68a':'';
      const warn=A.n!==B.n?` (different denominators: ${A.n} vs ${B.n})`:'';
      return `<td title="A ${A.s}/${A.n} vs B ${B.s}/${B.n}${warn}" `
        +`style="text-align:center;font-weight:700;color:${col}${hl}">`
        +`${arrow}${d===0?'':(d>0?'+'+d:d)}${warn?'<span style="color:#b45309">*</span>':''}</td>`;};
    // Method COLUMNS follow #0's ranking (left = best overall), not alphabetical order, so a task's
    // row reads in the same left-to-right order as the summary table above it. `ranked` is the exact
    // array #0 rendered, so the two tables can never disagree. In compare mode the columns are the
    // two chosen methods instead, in the order they were picked (A then B).
    let h3="<table class=cmbtask><tr><th class='exp'>task</th>"
      +cols.map(m=>`<th>${mLabelShort(m,true)}</th>`).join('')
      +(cmpOn?"<th>Δ<br><span style='font-weight:400;color:#888'>A−B</span></th>":"")+"</tr>";
    // OVERALL row first: the three splits combined, summed from per_split so it always agrees with
    // the split-total rows below it (and with a partial sweep's uneven splits).
    // The OVERALL row is summed from per_split, so it needs its own block builder rather than a
    // lookup -- shared by the cells and by the Δ column so both read the same numbers.
    const ovBlk=m=>{const P=C[m].per_split||{};
      let n=0,s=0,sr=0,unk=0,sec=0,sn=0,anyRef=false;
      for(const x of ['atomic_seen','composite_seen','composite_unseen','other']){
        const b=P[x]; if(!b||!b.n)continue;
        n+=b.n; s+=(b.n_success??b.s??0);
        if(b.n_success_refined!=null){anyRef=true; sr+=b.n_success_refined; unk+=(b.n_refined_unknown||0);}
        if(typeof b.avg_seconds==='number'){sec+=b.avg_seconds*b.n; sn+=b.n;}}
      if(!n)return null;
      const o={n:n,n_success:s,rate:s/n,avg_seconds:sn?+(sec/sn).toFixed(2):null};
      if(anyRef){o.n_success_refined=sr; o.rate_refined=sr/n; o.n_refined_unknown=unk;}
      return o;};
    h3+=`<tr><td class='exp' style="background:#cdd8ee;font-weight:700">OVERALL</td>`
      +cols.map(m=>cell(ovBlk(m))).join('')
      +(cmpOn?dcell(ovBlk(cols[0]),ovBlk(cols[1])):'')+`</tr>`;
    // split-total rows so a whole split reads at a glance
    ['atomic_seen','composite_seen','composite_unseen','other'].forEach(sp=>{
      const rows=tasks.filter(t=>info[t]===sp);
      if(!rows.length)return;
      const sb=m=>(C[m].per_split||{})[sp];
      h3+=`<tr><td class='exp' style="background:#dfe6f5;font-weight:700">${splName(sp)}</td>`
        +cols.map(m=>cell(sb(m))).join('')
        +(cmpOn?dcell(sb(cols[0]),sb(cols[1])):'')+`</tr>`;
      rows.forEach(t=>{const tb=m=>(C[m].per_task||{})[t];
        h3+=`<tr><td class='exp' style="padding-left:14px">${t}</td>`
        +cols.map(m=>cell(tb(m))).join('')
        +(cmpOn?dcell(tb(cols[0]),tb(cols[1])):'')+`</tr>`;});
    });
    document.getElementById('cmbtask').innerHTML=h3+"</table>";
  })();
  // #3 subtask table: THREE-WAY Gemini verdict (success/failure/uncertain). "success rate" is over
  // DECIDED spans (success/(success+failure)); uncertain shown separately. Per-primitive = decided-rate.
  const prims=[...new Set(Object.values(d.subtask).flatMap(v=>Object.keys(v.per_primitive||{})))].sort();
  h="<table><tr><th class='exp'>method</th><th>decided</th><th>✓ succ</th><th>✗ fail</th><th>? unc</th>"
    +"<th>success rate<br><span style='font-weight:400;color:#888'>succ/decided</span></th>"
    +prims.map(p=>`<th>${p}</th>`).join("")+"</tr>";
  Object.entries(d.subtask).forEach(([m,v])=>{
    h+=`<tr><td class='exp'>${mName(m)}</td><td>${v.n_decided??'-'}</td>`
      +`<td>${v.n_success??'-'}</td><td>${v.n_failure??'-'}</td>`
      +`<td style='color:#888'>${v.n_uncertain??'-'}</td><td><b>${pct(v.overall)}</b></td>`
      +prims.map(p=>{const c=v.per_primitive_counts?.[p];
        const tip=c?`✓${c.success} ✗${c.failure} ?${c.uncertain}`:'';
        return `<td title="${tip}">${pct(v.per_primitive?.[p])}</td>`;}).join("")+"</tr>";});
  document.getElementById('subtab').innerHTML=h+"</table>";
  // grouped bar: episode-all vs subtask-overall per method
  const methods=[...new Set([...Object.keys(d.episode),...Object.keys(d.subtask)])];
  const ds=[
    {label:'episode success (all)',data:methods.map(m=>d.episode[m]?100*(d.episode[m].all||0):null),backgroundColor:COLORS[0]},
    {label:'subtask gemini (decided success rate)',data:methods.map(m=>d.subtask[m]&&d.subtask[m].overall!=null?100*d.subtask[m].overall:null),backgroundColor:COLORS[1]},
  ];
  new Chart(document.getElementById('chart'),{type:'bar',data:{labels:methods,datasets:ds},
    options:{responsive:true,scales:{y:{min:0,max:100,title:{display:true,text:'%'}}}}});
}
// A failed /api/stats (tunnel hiccup, or this server restarting mid-sweep) must not surface as an
// 'unhandled promise rejection' banner -- it is a dropped request, not a script bug. The toggles
// call load() again on every click, so the page recovers on its own.
Promise.resolve().then(load).catch(e=>console.warn('stats load failed:', e));
</script></body></html>"""


# Landing page ("/"): a guide/navigation to the four eval pages.
HOME_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>RoboCasa System1 Eval</title>
<style>
  body{font-family:system-ui,sans-serif;margin:0;color:#1a1a1a;background:#fafafb}
  header{padding:16px 28px;border-bottom:1px solid #ddd;background:#fff}
  header h1{font-size:20px;margin:0}header h1 b{color:#b0431c}
  header p{margin:6px 0 0;color:#666;font-size:13px}
  #wrap{max-width:820px;margin:24px auto;padding:0 20px;display:grid;gap:14px}
  a.card{display:block;text-decoration:none;color:inherit;background:#fff;border:1px solid #e3e3e8;
         border-radius:10px;padding:16px 18px;transition:box-shadow .12s,border-color .12s}
  a.card:hover{border-color:#b0431c;box-shadow:0 2px 10px rgba(176,67,28,.10)}
  a.card h2{margin:0 0 4px;font-size:16px;color:#b0431c}
  a.card .path{font-family:ui-monospace,monospace;font-size:12px;color:#999;font-weight:400;margin-left:6px}
  a.card p{margin:0;font-size:13px;color:#444;line-height:1.5}
  /* the combined S2+S1 view is the headline entry point */
  a.card.hero{border-color:#b0431c;border-left-width:4px;background:linear-gradient(#fff,#fffaf8)}
</style></head><body>
<header>
  <h1>RoboCasa <b>System2 + System1</b> Eval</h1>
  <p>Browse rollouts and aggregate metrics for the Qwen3.5 System2 planner and the
     subgoal-conditioned pi0.5 System1 policies.</p>
</header>
<div id="wrap">
  <a class="card hero" href="/combine"><h2>Combined System2 + System1<span class="path">/combine</span></h2>
    <p>CLOSED-LOOP hierarchical rollouts: System2 plans the milestones, then each turn issues a
       <i>subgoal</i> + <i>estimated_step</i> that System1 executes until its progress/quiescence
       stop rule fires; the condensed 4-fps clip of that segment goes back to System2 to pick the
       next subgoal. Every intermediate result is inspectable per turn — both System2 prompts, its
       raw response and parsed tags, the System1 prompt + anchor image, the raw 20-fps rollout and
       the condensed clip actually fed back (with a per-frame keep/drop audit), plus per-component
       wall-clock timings.</p></a>
  <a class="card hero" href="/human-interactive"><h2>Human-interactive System2 + System1<span class="path">/human-interactive</span></h2>
    <p>Run an official target episode live against the deployed System2 and System1 servers. Review
       the raw planner output and rule-adjusted decision, then edit the checklist, judge, subgoal,
       detail, or estimated length before execution. Live action/progress curves are shown while
       System1 acts, and one-turn rollback restores the exact turn-start state by reset-and-replay.
       Every model decision, rule intervention, human edit, execution, and abandoned retry is saved
       with its target split, task, and episode provenance.</p></a>
  <a class="card" href="/episode"><h2>Episode Eval<span class="path">/episode</span></h2>
    <p>Whole-episode OPEN-LOOP rollouts: reset once to the first subgoal, then roll the policy
       continuously through the subgoal list. Per-subgoal video, prompt, anchor, action chunk, and
       the env <i>_check_success</i> task verdict. Seen (first-5) vs unseen (last-5) episodes.</p></a>
  <a class="card" href="/milestone"><h2>Milestone Eval<span class="path">/milestone</span></h2>
    <p>Per-MILESTONE CLOSED-LOOP rollouts: hard-reset to each milestone start, roll through its child
       subgoals, then an ORACLE-REFERENCED sim-check at the settled end (grasp / pick / place / open
       / close / turn / navigate) — the reliable success/failure label for offline RL.</p></a>
  <a class="card" href="/finestep"><h2>Fine-step Eval<span class="path">/finestep</span></h2>
    <p>Per-child-subgoal CLOSED-LOOP rollouts: hard-reset to each subgoal start independently. Same
       rich per-frame panels, plus the oracle GT reference and the Gemini success verdict per span.</p></a>
  <a class="card" href="/baseline"><h2>Flat-policy baseline<span class="path">/baseline</span></h2>
    <p>Xiaomi-Robotics-1 RoboCasa365 rollouts: one policy, no System2. Each episode is just the task
       goal and one video of the whole attempt, with its step count against the official horizon.</p></a>
  <a class="card" href="/stats"><h2>Statistics<span class="path">/stats</span></h2>
    <p>Aggregate success across methods: episode success rate (all / atomic / composite / seen /
       unseen), subtask Gemini success by primitive, and final-step validation MSE.</p></a>
  <a class="card" href="/val_mse"><h2>Train / Val MSE<span class="path">/val_mse</span></h2>
    <p>Action-MSE + progress-metric curves across checkpoints and steps, overlaying the train (seen)
       and val (unseen) splits with per-method toggles.</p></a>
</div>
</body></html>"""


INDEX_HTML = r"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Subtask Eval Viewer</title>
<style>
  *{box-sizing:border-box}html,body{height:100%;margin:0}
  body{font-family:system-ui,sans-serif;color:#1a1a1a;background:#fafafb;display:flex;flex-direction:column;overflow:hidden}
  #top{display:flex;gap:12px;align-items:center;padding:6px 12px;border-bottom:1px solid #ddd;background:#fff;flex:0 0 auto;flex-wrap:wrap}
  #top h1{font-size:14px;margin:0 4px 0 0}#top h1 b{color:#b0431c}
  .dssel{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.04em;display:flex;align-items:center;gap:5px}
  .dssel select{font-family:ui-monospace,monospace;font-size:12px;text-transform:none;letter-spacing:normal;padding:2px 5px;border:1px solid #bbb;border-radius:4px}
  .nav{display:flex;align-items:center;gap:5px}
  .nav button{font-size:13px;padding:3px 10px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
  .nav button:disabled{opacity:.4;cursor:default}.nav button:not(:disabled):hover{background:#f0f3fa}
  .pos{font-family:ui-monospace,monospace;font-size:12px;color:#555}
  .chip{font-size:11px;font-weight:700;padding:2px 9px;border-radius:11px;border:1px solid}
  .chip.k{color:#555;background:#f0f0f4;border-color:#ddd;font-weight:400;font-family:ui-monospace,monospace}
  .dataset{font-size:10px;color:#999;font-family:ui-monospace,monospace;margin-left:auto}

  /* two-lane track */
  #track{flex:0 0 auto;padding:5px 12px 8px;background:#fff;border-bottom:1px solid #eee}
  .pb-row{display:flex;align-items:center;gap:8px;margin-top:4px}
  .pb-lab{flex:0 0 70px;font-size:10px;color:#999;text-align:right;text-transform:uppercase;letter-spacing:.04em}
  .pb-lane{position:relative;flex:1;height:22px;background:#f1f1f4;border-radius:4px;border:1px solid #e3e3e8}
  .pb-seg{position:absolute;top:1px;bottom:1px;box-sizing:border-box;border-radius:3px;cursor:pointer;padding:0 4px;
          font-size:10px;line-height:18px;text-align:left;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-family:ui-monospace,monospace}
  .pb-seg:hover{filter:brightness(.96)}
  .pb-ms{background:#cfe0ff;border:1px solid #7fa8f0;color:#1c3a6b}
  .pb-ms.cur{background:#2a6df4;border-color:#1c4e9c;color:#fff;font-weight:700;z-index:2}
  .pb-fs{background:#e7dbff;border:1px solid #b79ee8;color:#4a2f7a}
  .pb-fs.cur{background:#7a3ff0;border-color:#5b2ac0;color:#fff;font-weight:700;z-index:2}

  #grid{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:minmax(400px,1.1fr) 1fr 1fr;gap:8px;padding:8px}
  .col{min-height:0;display:flex;flex-direction:column;gap:7px;overflow:hidden}
  .card{background:#fff;border:1px solid #e5e5ea;border-radius:7px;padding:7px 9px;min-height:0}
  .card.grow{flex:1 1 auto;overflow:auto}
  .card h3{margin:0 0 4px;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#999}
  video{width:100%;background:#000;border:1px solid #333;border-radius:4px}
  .half{width:66%!important}  /* anchor + video at 2/3 width */
  .anchor.half{width:66%}
  .vidnav{display:flex;align-items:center;gap:6px;margin-top:4px;flex-wrap:wrap}
  .vidnav button{font-size:12px;padding:2px 9px;border-radius:5px;border:1px solid #bbb;background:#fff;cursor:pointer}
  .vidnav button:hover{background:#f0f3fa}
  #slider{flex:1;min-width:120px}
  .fchips{font-size:10px;font-family:ui-monospace,monospace;color:#555;margin-top:3px;display:flex;flex-wrap:wrap;gap:4px}
  .fchips .c{background:#f0f0f4;border:1px solid #e0e0e6;border-radius:9px;padding:1px 7px}
  .fchips .c.hot{background:#fdeee6;border-color:#e8b48f;color:#8a3a10}
  .fchips .c.rep{background:#ffe9d6;border-color:#e8a060;color:#7a3a00}
  .fchips .c.ok{background:#e9f3ec;border-color:#b7dcc4;color:#2a6b45}
  /* fixed-height curve canvases (like /system1_prompt) so they never balloon to fill the column */
  #curveP{width:100%;height:90px;display:block;margin-top:4px;background:#fbfbfd;border:1px solid #ececef;border-radius:5px}
  #curveA{width:100%;height:110px;display:block;margin-top:4px;background:#fbfbfd;border:1px solid #ececef;border-radius:5px}
  .clegend{font-size:9.5px;font-family:ui-monospace,monospace;display:flex;gap:10px;margin-top:3px;flex-wrap:wrap}
  .anchor{display:flex;gap:4px}.anchor figure{margin:0;flex:1;min-width:0}
  .anchor img{width:100%;border:1px solid #ccc;border-radius:3px;background:#000;display:block}
  .anchor figcaption{font-size:9px;color:#888;font-family:ui-monospace,monospace;text-align:center}
  /* plain separated fields (no black box) so it renders fast + reads cleanly */
  .prompt-box{font-family:ui-monospace,monospace;font-size:12px;line-height:1.5}
  .prompt-box .pf{display:flex;gap:8px;padding:1px 0;border-bottom:1px solid #f2f2f5}
  .prompt-box .pfk{color:#b0431c;flex:0 0 120px;text-align:right;font-weight:600}
  .prompt-box .pfv{color:#222;flex:1;min-width:0;word-break:break-word}
  .prompt-box .pfv.hl{color:#1c6b3a;font-weight:700}
  .prompt-box .pfv.ints{color:#2d5bd7;font-size:10.5px;word-break:break-all}
  /* TASK SUCCESS panel rows (#success): same key/value layout as .prompt-box .pf, but the panel
     is NOT a .prompt-box, so its .pf/.pfk/.pfv need their own rules — otherwise the key and value
     render with no gap and collide ("subgoals reached2 of 2..."). */
  #success{font-family:ui-monospace,monospace;font-size:12px;line-height:1.6}
  #success .pf{display:flex;gap:12px;align-items:baseline;padding:2px 0;border-bottom:1px solid #f2f2f5}
  #success .pfk{color:#666;flex:0 0 190px;text-align:right}
  #success .pfv{color:#222;flex:1;min-width:0;word-break:break-word}
  #success .pfv.hl{font-weight:700}
  /* full-width stacked row (e.g. the gemini reason): column layout, so the label must NOT keep the
     190px flex-basis (in a column flex that basis becomes HEIGHT -> a huge blank gap). */
  #success .pf.stack{flex-direction:column;align-items:flex-start;gap:2px}
  #success .pf.stack .pfk{flex:0 0 auto;text-align:left}
  #success .pf.stack .pfv{white-space:normal;font-weight:400}
  .lab{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.04em;margin:5px 0 2px}
  .state-line{font-family:ui-monospace,monospace;font-size:10.5px;white-space:pre;overflow-x:auto;
              background:#f7f7f9;border:1px solid #ececef;border-radius:5px;padding:5px 7px}
  .kv{font-family:ui-monospace,monospace;font-size:11px;line-height:1.5}
  .kv .key{color:#999;display:inline-block;min-width:96px}.kv .val{color:#222}
  table.act{border-collapse:collapse;font-family:ui-monospace,monospace;font-size:10px;width:100%}
  table.act th,table.act td{border:1px solid #e8e8ec;padding:1px 4px;text-align:right}
  table.act th{background:#f6f6f8;color:#666;position:sticky;top:0}
  table.act tr.exec td{background:#eef7ee}
  table.act tr.cur td{background:#ffe9c7;font-weight:700}  /* row executing at the current frame */
  table.act td.step{color:#b0431c}
  table.act.cmp td.rl{text-align:left;color:#666;font-weight:700;background:#f6f6f8}
  table.act.cmp tr:nth-child(2) td{color:#1c4e9c}  /* executed row */
  table.act.cmp tr:nth-child(3) td{color:#8a5a00}  /* oracle row */
  table.act th.progcol{background:#eef3ff;color:#2d5bd7}
  table.act td.progcol{color:#2d5bd7;background:#f4f7ff;font-weight:600;border-left:2px solid #2d5bd7}
  .toggle{font-size:9px;font-weight:700;padding:1px 7px;border-radius:5px;border:1px solid #b0431c;
          color:#b0431c;background:#fff;cursor:pointer;margin-left:6px;text-transform:none;letter-spacing:normal}
  .toggle:hover{background:#fdeee6}
  #err{padding:24px;color:#a12020}
</style></head><body>
  <div id="top">
    <h1><b>Subtask</b> Eval</h1>
    <label class="dssel">method<select id="method"></select></label>
    <label class="dssel">type<select id="tasktype"></select></label>
    <label class="dssel">task<select id="task"></select></label>
    <label class="dssel">episode<select id="episode"></select></label>
    <div class="nav">
      <button id="prevSub">‹ subtask</button>
      <span class="pos" id="subpos">— / —</span>
      <button id="nextSub">subtask ›</button>
    </div>
    <span class="chip k" id="primchip"></span>
    <span class="chip k" id="mschip"></span>
    <span class="dataset" id="dsinfo"></span>
  </div>
  <div id="track"></div>
  <div id="grid"><div id="err">Loading…</div></div>
<script src="/gui.js"></script>
</body></html>"""


@app.route("/gui.js")
def gui_js():
    return GUI_JS, 200, {"Content-Type": "application/javascript"}


GUI_JS = r"""
const RN=window.RN||'finestep';   // rollout root: 'finestep' | 'milestone' | 'episode'
const $=s=>document.querySelector(s);
// Gemini verdict for a subgoal (#3): episode.json.gemini.per_subtask[<child_dir>].success.
// Map a subtask's Gemini verdict (from the prefetched cache) to true/false/null for track coloring:
// success->true, failure->false, uncertain/skipped/absent->null.
// Page-appropriate sim_check verdict for a span (drives the thin green/red line under the track).
// The line spans the whole UNIT of that page, colored by the unit's sim_check:
//   /episode   -> EVERY span gets the whole-episode env _check_success (one line over the episode)
//   /milestone -> every span gets ITS MILESTONE's verdict (line spans the whole milestone), read
//                 from the milestone-end child of the same milestone_index
//   /finestep  -> the span's OWN sim_check (line only on the fine-step span itself)
// Returns true (success) / false (failure) / null (None). No Gemini.
function _vOf(sc){ if(!sc)return null; if(sc.verdict==='success')return true; if(sc.verdict==='failure')return false; return null; }
function subVerdict(ep,s){
  if(RN==='episode'){
    const es = (ep.episode_success!=null)?ep.episode_success:(ep.sim_success_final!=null?ep.sim_success_final:null);
    return es===true?true:(es===false?false:null);
  }
  if(RN==='milestone'){
    // color the whole milestone by its verdict: find the milestone-end child of this milestone.
    const end = (ep.subgoals||[]).find(x=>x.milestone_index===s.milestone_index && (x.is_milestone_end || x.is_terminal));
    return _vOf((end||s).milestone_sim_check);
  }
  // finestep: the span's own sim_check.
  return _vOf(s.milestone_sim_check || s.subtask_sim_check);
}
const S={method:null,eps:[],epi:0,subi:0,steps:null,fps:20,norm:true};
const esc=s=>(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
// Text that drove the policy prompt: subgoal_detail for verbrich runs (episode.prompt_source==
// 'subgoal_detail', e.g. v17/v18), else the terse subgoal. Falls back to subgoal if detail is empty.
function sgText(ep,s){return (ep&&ep.prompt_source==='subgoal_detail'&&s.subgoal_detail)?s.subgoal_detail:s.subgoal;}
const fmt=(a,p=3)=>(a||[]).map(v=>(v>=0?'+':'')+Number(v).toFixed(p)).join(' ');

function promptHTML(p){return esc(p).replace(/(Task:|Current Subgoal:|Quality:|Estimated Length:|Executed Step:|Initial State:|Current State:|Current Gripper:|Action:)/g,'<span class="k">$1</span>');}
// Render the policy prompt as PLAIN labeled fields = the TRUE model input at the GOVERNING replan
// (`gs`). Because the model is only queried every replan_steps, this prompt is CONSTANT for the
// whole 5-step window and only changes at the next replan — matching what the policy actually saw.
// (Executed Step / gripper / state-ints are all taken at the governing replan frame, not the
// currently-displayed video frame.)
function renderPromptFields(gs){
  if(!gs){$('#prompt').innerHTML='<div class="pf"><span class="pfv">(warmup — no query yet)</span></div>';return;}
  const q=gs.query||{}; const P=q.prompt||"";
  // P is now the REAL assembled prompt string the server tokenized (result["prompt_text"]),
  // with the actual discretized state ints — parse every field straight out of it.
  const grab=(re)=>{const m=P.match(re); return m?m[1].trim():"—";};
  const task=grab(/Task:\s*([^\n;]*)/), subgoal=grab(/Current Subgoal:\s*([^\n]*?)(?:\n|$)/);
  const quality=grab(/Quality:\s*([^\n;]*)/), estlen=grab(/Estimated Length:\s*([^\n;]*)/);
  const execstep=grab(/Executed Step:\s*([^\n;]*)/), grip=grab(/Current Gripper:\s*([^\n;]*)/);
  const initSt=grab(/Initial State:\s*([^;]*)/), curSt=grab(/Current State:\s*([^;]*)/);
  const row=(k,v,cls)=>`<div class="pf"><span class="pfk">${k}</span><span class="pfv ${cls||''}">${esc(String(v))}</span></div>`;
  $('#prompt').innerHTML=
    row('Task', task)+
    row('Current Subgoal', subgoal, 'hl')+
    row('Quality', quality)+
    row('Estimated Length', estlen)+
    row('Executed Step', execstep, 'hl')+
    row('Current Gripper', grip)+
    row('Initial State', initSt, 'ints')+
    row('Current State', curSt, 'ints')+
    row('Action', '→ predicted chunk (right)');
}
// discretize a NORMALIZED value array into 256 bins over [-1,1] (matches PaligemmaTokenizer:
// np.digitize(x, linspace(-1,1,257)[:-1]) - 1). Returns ints in [0,255].
function discretize(vals){
  if(!vals)return null;
  return vals.map(v=>{
    // bins = linspace(-1,1,256)[:-1] ... digitize counts how many edges <= v, minus 1
    let b=Math.floor((v+1)/2*256); if(b<0)b=0; if(b>255)b=255; return b;
  });
}
// Fill the prompt's Initial/Current State placeholders with the REAL discretized ints
// (tokenizer renders anchor=Initial first, then current=Current), like /system1_training_sample.
function fillStateInts(prompt, curNorm, ancNorm){
  let out=prompt;
  const ci=discretize(curNorm), ai=discretize(ancNorm);
  if(ai) out=out.replace(/(Initial State:\s*)([^;]*)/, (m,p1)=>p1+ai.join(' '));
  if(ci) out=out.replace(/(Current State:\s*)([^;]*)/, (m,p1)=>p1+ci.join(' '));
  return out;
}

const STATE_GROUPS=[["base_pos",0,3],["base_quat",3,7],["eef_pos_rel",7,10],["eef_quat",10,14],["grip_qpos",14,16]];
const LEAN_GROUPS=[["eef_pos",0,3],["eef_rot6d",3,9],["grip_w",9,10],["base_xy_rel",10,12],["yaw_sincos",12,14]];
const SIM_LABELS=["eef_dx","eef_dy","eef_dz","d_roll","d_pitch","d_yaw","grip","base_vx","base_vy","base_vz","yaw_v","ctrl"];
// lean-11 order (the raw model action layout): base_vx base_vy yaw_v ctrl eef_dx..dz d_roll..d_yaw grip
const LEAN_LABELS=["base_vx","base_vy","yaw_v","ctrl","eef_dx","eef_dy","eef_dz","d_roll","d_pitch","d_yaw","grip"];
// robosuite-native sim-12 -> lean-11 (drop torso base_vz). sim order: eef_pos[0:3] eef_rot[3:6]
// grip[6] base[7:11] ctrl[11]; lean: base_vx,base_vy(=base[0,1]=sim7,8), yaw_v(=base[3]=sim10),
// ctrl(sim11), eef_dx..dz(sim0..2), d_roll..d_yaw(sim3..5), grip(sim6).
function sim12ToLean11(a){return [a[7],a[8],a[10],a[11],a[0],a[1],a[2],a[3],a[4],a[5],a[6]];}
function stateBlock(vals,groups){if(!vals)return '';return groups.map(([n,s,e])=>`  ${n.padEnd(12)} ${vals.slice(s,e).map(x=>Number(x).toFixed(3).padStart(8)).join(' ')}`).join("\n");}

// scalar progress in [0,1] from a step's progress_raw. classes -> argmax CLASS normalized
// (argmax/(K-1)): a faithful staircase of the classifier's discrete decile prediction, NOT the
// smoothed expected value. continuous/action -> the value directly.
function progScalar(pr){
  if(!pr)return null;
  if(pr.progress_kind==='classes'){
    const k=pr.progress_num_classes||10;
    return (pr.progress_argmax!=null)?pr.progress_argmax/(k-1):pr.progress_expected_frac;
  }
  if(pr.progress_kind==='continuous')return pr.progress_now;
  if(pr.progress_kind==='action')return pr.progress_now;
  return null;
}
// Human-readable progress label for the current step (shown in the success/prompt panels).
function progLabel(pr){
  if(!pr)return '—';
  if(pr.progress_kind==='classes')return `class ${pr.progress_argmax}/${(pr.progress_num_classes||10)-1} (conf ${(pr.progress_conf||0).toFixed(2)})`;
  if(pr.progress_kind==='continuous')return `${(pr.progress_now||0).toFixed(3)}`;
  if(pr.progress_kind==='action')return `${(pr.progress_now||0).toFixed(3)} → ${(pr.progress_end||0).toFixed(3)}`;
  return '—';
}
// Build per-step series for the current subtask. Also precompute, for EACH frame, the index of
// the GOVERNING query — the most recent replan step at/before it. The model is only queried on
// replan steps (every replan_steps), so its prompt + predicted chunk stay CONSTANT until the next
// replan. Rendering from govQ makes the prompt/chunk correct at any frame, including when scrubbing
// backward (S._lastQuery mutated during playback would otherwise be wrong).
function buildSeries(){
  const st=S.steps.steps; let lastP=null; let govIdx=-1; let govFS=null; let govProg=null;
  S.series={prog:[],eef_pos:[],eef_rot:[],base:[],phase:[]};
  S.govQ=[];       // per frame: index of the frame whose .query governs it (-1 during warmup)
  S.progDense=[];  // per frame: the DENSE model progress at THIS step (see below), or null
  st.forEach((s,i)=>{
    if(s.query){govIdx=i; govFS=s.frame_step; govProg=(s.query&&s.query.chunk_progress)||null;}
    // DENSE per-step progress (progact): the model predicts progress for ALL H horizon steps at
    // each replan, but we only EXECUTE the first `replan_steps` of them. Frame `fs` executes chunk
    // offset (fs − governing-replan-fs), so index the governing replan's full progress chunk at
    // that offset instead of repeating chunk[0] across the whole window (the old stair-step curve).
    // progcls/progreg have no per-step chunk (single head value) → carry-forward the replan value.
    let dense=null;
    if(govProg){
      const off=s.frame_step-govFS;
      if(off>=0&&off<govProg.length)dense=govProg[off];
    }
    const raw=progScalar(s.progress_raw);
    const p=(dense!=null)?dense:raw;   // dense (progact) else the replan's scalar (cls/reg)
    if(p!=null)lastP=p;
    S.progDense.push(dense);           // null on warmup / non-progact frames
    S.series.prog.push(lastP);
    S.series.eef_pos.push(s.action_eef_pos_norm);
    S.series.eef_rot.push(s.action_eef_rot_norm);
    S.series.base.push(s.action_base_norm);
    S.series.phase.push(s.phase);
    S.govQ.push(govIdx);
  });
}
// Draw one canvas: `lines`=[{arr,color}], y-range [0,ymax]. Shared grid/settle/cursor.
function _drawCurve(cvId, lines, ymax, cur, yTicks){
  const cv=document.getElementById(cvId); if(!cv||!S.series)return;
  // match the canvas BITMAP to its CSS box (fixed height) so it never stretches/distorts
  const W=cv.clientWidth||600, H=cv.clientHeight||90; cv.width=W; cv.height=H;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const n=S.series.prog.length; if(n<2)return;
  const pad={l:32,r:8,t:6,b:12}; const gw=W-pad.l-pad.r, gh=H-pad.t-pad.b;
  const X=i=>pad.l+gw*i/(n-1);
  const Y=v=>pad.t+gh*(1-(v==null?0:v)/ymax);
  ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.fillStyle='#aaa';ctx.font='9px monospace';
  (yTicks||[0,ymax]).forEach(t=>{const y=Y(t);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(String(t),2,y+3);});
  const firstAct=S.series.phase.findIndex(p=>p==='act');
  if(firstAct>0){const x=X(firstAct);ctx.strokeStyle='#bbb';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();ctx.setLineDash([]);}
  lines.forEach(({arr,color})=>{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let started=false;
    arr.forEach((v,i)=>{if(v==null)return;const x=X(i),y=Y(v);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);});ctx.stroke();});
  if(cur!=null){const x=X(cur);ctx.strokeStyle='#333';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,pad.t);ctx.lineTo(x,H-pad.b);ctx.stroke();}
}
function drawCurves(cur){
  if(!S.series)return;
  // progress curve (0..1) — its own field/canvas
  _drawCurve('curveP',[{arr:S.series.prog,color:'#2d5bd7'}],1.0,cur,[0,0.5,1]);
  // action Δ magnitude curve — its own field/canvas, auto-scaled
  const nm=Math.max(0.05,...S.series.eef_pos,...S.series.eef_rot,...S.series.base);
  _drawCurve('curveA',[{arr:S.series.eef_pos,color:'#e5484d'},{arr:S.series.eef_rot,color:'#f59e0b'},
                       {arr:S.series.base,color:'#16a34a'}],nm,cur,[0,+(nm/2).toFixed(2),+nm.toFixed(2)]);
}

// task_name -> split label (atomic_seen/composite_seen/composite_unseen), fetched once.
const TYPE_LABEL={atomic_seen:'atomic-seen',composite_seen:'composite-seen',composite_unseen:'composite-unseen'};
const TYPE_ORDER=['atomic_seen','composite_seen','composite_unseen'];
async function loadMethods(){
  const ms=await (await fetch(`/api/${RN}/methods`)).json();
  $('#method').innerHTML=ms.map(m=>`<option value="${m.method}">${m.method}</option>`).join('');
  window._ms=ms;
  try{ S.splits=await (await fetch('/api/task_splits')).json(); }catch(e){ S.splits={}; }
  if(ms.length){S.method=ms[0].method; await loadEpisodes();}
}
function taskSplit(task){ return (S.splits&&S.splits[task]) || 'other'; }
async function loadEpisodes(){
  // Remember current type/task/episode so switching METHOD keeps you in place.
  const prevType=$('#tasktype').value||null, prevTask=$('#task').value||null;
  const prevEpName=(S.eps&&S.eps[S.epi])?S.eps[S.epi].episode_id.split('/').pop():null;
  S.method=$('#method').value;
  // LAZY: fetch only the lightweight list (episode_id/task_name/n_subgoals) from index.json — the
  // full per-episode doc (subgoals) is loaded on demand in selectEpisode(). Avoids reading ~1000
  // episode.json (~10 MB) just to fill the dropdowns.
  S.eps=await (await fetch(`/api/${RN}/episode_list/`+S.method)).json();
  S._full={};   // cache: episode index -> full doc, populated lazily on selection
  // TYPE dropdown: the splits present among this method's tasks, in canonical order.
  const present=new Set(S.eps.map(e=>taskSplit(e.task_name)));
  const types=TYPE_ORDER.filter(t=>present.has(t)).concat([...present].filter(t=>!TYPE_ORDER.includes(t)));
  $('#tasktype').innerHTML=types.map(t=>`<option value="${t}">${TYPE_LABEL[t]||t}</option>`).join('');
  if(prevType&&types.includes(prevType))$('#tasktype').value=prevType;
  const m=window._ms.find(x=>x.method===S.method)||{};
  $('#dsinfo').textContent=`${S.method} · ${m.n_episodes} eps · budget ${m.horizon_mult}x · settle ${m.settle_steps}`;
  fillTasks(prevTask,prevEpName);
}
// Populate the TASK dropdown with tasks of the selected type, then the episode dropdown.
function fillTasks(keepTask,keepEpName){
  const type=$('#tasktype').value;
  const tasks=[...new Set(S.eps.filter(e=>taskSplit(e.task_name)===type).map(e=>e.task_name))].sort();
  $('#task').innerHTML=tasks.map(t=>`<option value="${t}">${t}</option>`).join('');
  if(keepTask&&tasks.includes(keepTask))$('#task').value=keepTask;
  fillEpisodes(keepEpName);
}
// Populate the episode dropdown for the selected task. Keep the previously-selected episode
// (by name) if it exists under the new method/task; otherwise fall back to the first one.
function fillEpisodes(keepEpName){
  const task=$('#task').value;
  const opts=S.eps.map((e,i)=>({e,i})).filter(x=>x.e.task_name===task);
  $('#episode').innerHTML=opts.map(x=>`<option value="${x.i}">${x.e.episode_id.split('/').pop()}</option>`).join('');
  if(!opts.length)return;
  let pick=opts[0];
  if(keepEpName){
    const hit=opts.find(x=>x.e.episode_id.split('/').pop()===keepEpName);
    if(hit)pick=hit;
  }
  $('#episode').value=pick.i;
  selectEpisode(pick.i);
}
// Prefetch every subtask's Gemini verdict for this episode into S._verdicts (keyed by child_dir),
// so subVerdict()/the track/the panel all read one cache instead of re-fetching. Verdicts now live
// per-subtask in judge/gemini.json (three-way verdict/completion/confidence/reason + skip info).
// Gemini judge removed — verdicts now come from the per-span sim_check stored in the episode doc.
async function loadVerdicts(ep){ S._verdicts={}; }
// LAZY-LOAD the full episode doc (subgoals + fields) on selection, replacing the lightweight list
// entry in S.eps[i] so the track/subtask panels get the real data. Cached in S._full so
// re-selecting the same episode is instant (no refetch).
async function selectEpisode(i){
  S.epi=i; S.subi=0;
  let ep=S.eps[i];
  if(!ep.subgoals){
    if(!S._full[i]){
      const epf=ep.episode_id.replaceAll('/','__');
      S._full[i]=await (await fetch(`/api/${RN}/episode/${S.method}/${epf}`)).json();
    }
    // merge full doc into S.eps[i] (keep list fields) so downstream code reads S.eps[S.epi].subgoals
    S.eps[i]=Object.assign({}, ep, S._full[i]);
    ep=S.eps[i];
  }
  await loadVerdicts(ep); drawTrack(); await selectSub(0);
}

function drawTrack(){
  const ep=S.eps[S.epi]; const subs=ep.subgoals;
  const T=Math.max(...subs.map(s=>s.span[1]))+1;
  // milestones: union child spans by milestone_index
  const ms={}; subs.forEach(s=>{const k=s.milestone_index; if(!(k in ms))ms[k]={a:s.span[0],b:s.span[1],text:s.milestone_subgoal||('milestone '+k)}; ms[k].a=Math.min(ms[k].a,s.span[0]); ms[k].b=Math.max(ms[k].b,s.span[1]);});
  const curMs=subs[S.subi].milestone_index;
  const pct=(x)=>100*x/T;
  const msSegs=Object.entries(ms).map(([k,v])=>`<div class="pb-seg pb-ms ${+k===curMs?'cur':''}" style="left:${pct(v.a)}%;width:${pct(v.b-v.a+1)}%" title="${esc(v.text)}" data-ms="${k}">${esc(v.text)}</div>`).join("");
  const fsSegs=subs.map((s,i)=>{
    const v=subVerdict(ep,s);   // sim_check success: true/false/null (page-appropriate)
    const vc=v===true?'#2e8b3d':(v===false?'#b0431c':'');   // green / red / default (None)
    const adv=(s.advanced===true)?' ✓':(s.advanced===false?' ⧗':'');  // self-stop fired / timeout
    const style=`left:${pct(s.span[0])}%;width:${pct(s.span[1]-s.span[0]+1)}%`+(vc?`;box-shadow:inset 0 -3px 0 ${vc}`:'');
    const vt=v===null?'':' | sim_check:'+(v?'SUCCESS':'FAIL');
    const st=sgText(ep,s);
    return `<div class="pb-seg pb-fs ${i===S.subi?'cur':''}" style="${style}" title="[${s.primitive}] ${esc(st)}${vt}" data-sub="${i}">${esc(st)}${adv}</div>`;
  }).join("");
  $('#track').innerHTML=`<div class="pb-row"><div class="pb-lab">milestones</div><div class="pb-lane">${msSegs}</div></div>
    <div class="pb-row"><div class="pb-lab">subgoals</div><div class="pb-lane">${fsSegs}</div></div>`;
  document.querySelectorAll('.pb-fs').forEach(el=>el.onclick=()=>selectSub(+el.dataset.sub));
}

async function selectSub(i){
  const ep=S.eps[S.epi]; if(i<0||i>=ep.subgoals.length)return;
  S.subi=i; const sg=ep.subgoals[i];
  S.steps=await (await fetch(`/api/${RN}/steps/${S.method}/${ep.episode_id.replaceAll('/','__')}/${sg.out_dir.split('/').pop()}`)).json();
  S.fps=S.steps.fps||20;
  $('#subpos').textContent=`subtask ${i} / ${ep.subgoals.length-1}`;
  $('#prevSub').disabled=i<=0; $('#nextSub').disabled=i>=ep.subgoals.length-1;
  // primchip: primitive + subgoal + the reset mode for this page.
  const reset=(RN==='episode')?'  · reset: first-only':(RN==='milestone')?'  · reset: per-milestone (GT)':'  · reset: per-subgoal (GT)';
  $('#primchip').textContent=`[${sg.primitive}] ${sgText(ep,sg)}${reset}`;
  let ms=`milestone: ${sg.milestone_subgoal||sg.milestone_index}`;
  if(ep.episode_success!=null)ms+=`  ·  EPISODE: ${ep.episode_success?'SUCCESS':'fail'} (advanced ${ep.n_advanced}/${ep.n_subgoals})`;
  else if(ep.sim_success_final!=null)ms+=`  ·  EPISODE _check_success: ${ep.sim_success_final?'SUCCESS':'fail'}`;
  $('#mschip').textContent=ms;
  buildSeries();
  drawTrack();
  renderStatic();   // rebuilds #grid (fresh #vid/#play/#slider) and re-wires them via wireVideo()
  const base=`/api/${RN}/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${sg.out_dir.split('/').pop()}/`;
  resetVideoForClip(base+S.steps.clean_video);  // pause, cancel stale follow-loop, swap src + load
  $('#slider').max=S.steps.steps.length-1; $('#slider').value=0; showFrame(0);
}

// TASK SUCCESS panel — TWO signals only (no Gemini):
//   (1) sim_check — page-appropriate: /episode = env _check_success (only decisive at the terminal
//       span, else None); /milestone + /finestep = the milestone criterion (decisive only at each
//       milestone's ENDING span, with a rollout-vs-oracle explanation of WHY on failure).
//   (2) self-stopped — did the policy self-terminate (progress+quiescence) or hit the budget cap.
async function renderSuccess(){
  const el=$('#success'); if(!el)return;
  const ep=S.eps[S.epi], sg=ep.subgoals[S.subi];
  const rows=[];
  const row=(k,v,cls)=>`<div class="pf"><span class="pfk">${k}</span><span class="pfv ${cls||''}">${v}</span></div>`;
  const lab=v=>v==='success'?'<b style="color:#2e8b3d">SUCCESS</b>':v==='failure'?'<b style="color:#b0431c">FAIL</b>':'<span style="color:#999">None</span>';

  if(RN==='episode'){
    // whole-episode env _check_success (decisive only at the terminal span).
    const es = (sg.is_terminal||sg.sim_success_final!=null) ? (ep.episode_success??sg.sim_success_final) : null;
    rows.push(row('sim_check (episode) <span style="font-weight:400;color:#888">env _check_success @ end</span>',
                  lab(es===true?'success':es===false?'failure':null), 'hl'));
    if(ep.n_advanced!=null)rows.push(row('subgoals reached', `${ep.n_advanced} of ${ep.n_subgoals} self-advanced`));
  } else {
    // milestone / finestep: the span's own sim_check verdict + the WHY (rollout vs oracle numbers).
    const sc = sg.milestone_sim_check || sg.subtask_sim_check || {};
    const lbl = sc.verdict==='reference' ? '<span style="color:#1c6bb0">REFERENCE (oracle)</span>' : lab(sc.verdict);
    const tag = (RN==='milestone') ? 'sim_check (milestone)' : 'sim_check (finestep = milestone criterion)';
    rows.push(row(tag+(sc.rule?` <span style="font-weight:400;color:#888">${sc.rule}</span>`:''), lbl, 'hl'));
    // detail: the rollout-vs-oracle comparison + why it failed (only meaningful for milestone check).
    if(sc.detail)rows.push(`<div class="pf stack"><span class="pfk">why</span><span class="pfv">${esc(sc.detail)}</span></div>`);
  }
  // (2) self-stopped — same for all pages.
  const adv = (sg.advanced===true||sg.stopped===true) ? '<b style="color:#2e8b3d">yes → self-stopped (progress+quiescence)</b>'
            : (sg.advanced===false||sg.stopped===false) ? '<b style="color:#b0431c">no → timed out (budget cap)</b>' : '—';
  rows.push(row('self-stopped', adv));
  // current-step progress readout (argmax class for progcls) — id'd so showFrame can refresh it
  const cur=S._curStep||(S.steps&&S.steps.steps?S.steps.steps[0]:null);
  rows.push(`<div class="pf"><span class="pfk">progress @ frame</span><span class="pfv hl" id="succ-prog">${cur?progLabel(cur.progress_raw):'—'}</span></div>`);
  el.innerHTML=rows.join('');
}

function renderStatic(){
  const d=S.steps, ep=S.eps[S.epi];
  const base=`/api/${RN}/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${d.clean_video.replace('clean.mp4','')}`;
  const bdir=`/api/${RN}/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${ep.subgoals[S.subi].out_dir.split('/').pop()}/`;
  const anc=Object.entries(d.anchor_images||{}).map(([k,f])=>`<figure><img src="${bdir}${f}"><figcaption>${k}</figcaption></figure>`).join("");
  $('#grid').innerHTML=`
    <div class="col">
      <div class="card">
        <h3>ANCHOR — subgoal-start views</h3>
        <div class="anchor half">${anc}</div>
      </div>
      <div class="card">
        <h3>meta / spans</h3>
        <div class="kv">
          <div><span class="key">task_goal</span><span class="val">${esc(d.task_goal)}</span></div>
          <div><span class="key">subgoal</span><span class="val">${esc(d.subgoal)}</span></div>
          <div><span class="key">detail</span><span class="val">${esc(d.subgoal_detail||'—')}</span></div>
          <div><span class="key">milestone</span><span class="val">${esc(d.milestone_subgoal||'—')}</span></div>
          <div><span class="key">span</span><span class="val">[${(d.span||[]).join(', ')}]${d.summary?` (len ${d.summary.span_len}) · est_len ${d.summary.est_length}`:''}</span></div>
          <div><span class="key">budget</span><span class="val">${d.budget!=null?d.budget+' steps':'—'}${d.settle_steps!=null?' (settle '+d.settle_steps+')':''}</span></div>
          ${d.summary?`<div><span class="key">1st-chunk mse</span><span class="val">${(d.summary.first_chunk_action_mse??0).toFixed(4)} · mean-step ${(d.summary.mean_step_action_mse??0).toFixed(4)}</span></div>`:''}
          ${d.summary?`<div><span class="key">sim_success</span><span class="val">final ${d.summary.sim_success_final} · any ${d.summary.sim_success_any}</span></div>`:''}
          <div><span class="key">base_pos_ref</span><span class="val">[${(d.base_pos_ref||[]).map(x=>x.toFixed(3)).join(', ')}] yaw ${(d.base_yaw_ref??0).toFixed(4)}</span></div>
        </div>
      </div>
      <div class="card grow">
        <h3>STATE <button class="toggle" id="tg-state"></button></h3>
        <div class="lab">raw 16-d — current frame</div><div class="state-line" id="st-raw"></div>
        <div class="lab"><span id="lean-lab">lean 14-d</span> — current frame</div><div class="state-line" id="st-lean"></div>
        <div class="lab">raw 16-d — ANCHOR (subgoal-start)</div><div class="state-line">${stateBlock(d.anchor_state_raw16,STATE_GROUPS)}</div>
        <div class="lab"><span class="lean-lab2">lean 14-d</span> — ANCHOR</div><div class="state-line" id="st-anchor-lean"></div>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h3>rollout — clean video (subtask ${d.child_index}: ${esc(d.subgoal)})</h3>
        <video id="vid" class="half" muted preload="metadata"></video>
        <div class="vidnav">
          <button id="play">▶ play</button>
          <button id="bb">‹ frame</button><button id="ff">frame ›</button>
          <input type="range" id="slider" min="0" max="0" value="0">
          <span class="pos" id="fpos"></span>
        </div>
        <div class="fchips" id="fchips"></div>
      </div>
      <div class="card"><h3>LANGUAGE PROMPT (policy input @ current step — same within a chunk, changes on replan)</h3><div class="prompt-box" id="prompt"></div></div>
      <div class="card grow">
        <h3>PROGRESS (predicted, executed rollout)</h3>
        <canvas id="curveP"></canvas>
        <h3 style="margin-top:8px">ACTION Δ MAGNITUDE (executed rollout)</h3>
        <canvas id="curveA"></canvas>
        <div class="clegend">
          <span style="color:#e5484d">■ |eef_pos|</span>
          <span style="color:#f59e0b">■ |eef_rot|</span>
          <span style="color:#16a34a">■ |base|</span>
          <span style="color:#999">┊ settle→act</span>
          <span style="color:#333">│ current frame</span>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card"><h3>TASK SUCCESS</h3><div id="success"></div></div>
      <div class="card">
        <h3>EXECUTED vs ORACLE @ current step</h3>
        <div id="execcmp"></div>
      </div>
      <div class="card grow">
        <h3>PREDICTED ACTION CHUNK @ current query (full horizon) <button class="toggle" id="tg-chunk"></button></h3>
        <div id="chunk"></div>
      </div>
    </div>`;
  renderSuccess();
  wireVideo();
}

// Wire the <video> + its controls. renderStatic() rebuilds #grid's innerHTML on EVERY subtask, so
// #vid / #play / #slider are FRESH DOM nodes each time and their listeners must be (re)attached here
// (the old nodes are discarded, so listeners don't truly stack). The real cross-clip bug was a stale
// requestVideoFrameCallback loop still bound to the PREVIOUS #vid: it kept S._rvfc set, so the new
// clip's follow-loop never started and the ▶ chunk pointer + progress bar stopped moving. Fix: cancel
// any pending rVFC before wiring (here) and again when loading a new clip (resetVideoForClip).
function wireVideo(){
  const v=$('#vid');
  if(S._rvfc!=null){                            // kill a follow-loop bound to the old #vid node
    if('cancelVideoFrameCallback' in HTMLVideoElement.prototype){try{v.cancelVideoFrameCallback(S._rvfc);}catch(e){}}
    else{cancelAnimationFrame(S._rvfc);}
    S._rvfc=null;
  }
  // Follow the video PER PRESENTED FRAME while playing. requestVideoFrameCallback fires once for
  // every painted frame, so the panel + ▶ chunk pointer advance ONE step at a time in lockstep
  // (the old `timeupdate` event was throttled to ~4Hz, making the pointer jump 0,5,14,24,…).
  // During manual step/slider the slider index stays authoritative (rVFC only runs while playing).
  const hasRVFC='requestVideoFrameCallback' in HTMLVideoElement.prototype;
  const follow=(now,meta)=>{
    if(!S.steps||v.paused){S._rvfc=null;return;}
    const t=(meta&&meta.mediaTime!=null)?meta.mediaTime:v.currentTime;
    const i=Math.min(S.steps.steps.length-1,Math.round(t*S.fps));
    if(i!==(+$('#slider').value)){$('#slider').value=i;showFrame(i);}
    S._rvfc=v.requestVideoFrameCallback(follow);
  };
  const raf=()=>{if(!S.steps||v.paused){S._rvfc=null;return;}
    const i=Math.min(S.steps.steps.length-1,Math.round(v.currentTime*S.fps));
    if(i!==(+$('#slider').value)){$('#slider').value=i;showFrame(i);}
    S._rvfc=requestAnimationFrame(raf);};
  v.addEventListener('play',()=>{ if(S._rvfc)return;
    S._rvfc=hasRVFC?v.requestVideoFrameCallback(follow):requestAnimationFrame(raf); });
  // keep the play button label in sync no matter HOW play/pause happened (button, end, new clip)
  v.addEventListener('pause',()=>{$('#play').textContent='▶ play';});
  v.addEventListener('play', ()=>{$('#play').textContent='❚❚ pause';});
  v.addEventListener('ended',()=>{$('#play').textContent='▶ play';});
  $('#slider').oninput=e=>{v.pause();gotoFrame(+e.target.value);};
  $('#play').onclick=()=>{ if(v.paused){v.play().catch(()=>{});}else{v.pause();} };
  $('#ff').onclick=()=>stepFrame(1); $('#bb').onclick=()=>stepFrame(-1);
  const tog=()=>{S.norm=!S.norm; renderNormable();};
  if($('#tg-state'))$('#tg-state').onclick=tog;
  if($('#tg-chunk'))$('#tg-chunk').onclick=tog;
}

// Load a new clip into the shared <video>: pause + cancel any running follow-loop from the PREVIOUS
// clip, swap src, and force a reload so .play() works on the fresh media. Called by selectSub.
function resetVideoForClip(srcUrl){
  const v=$('#vid');
  v.pause();
  if(S._rvfc!=null){
    if('cancelVideoFrameCallback' in HTMLVideoElement.prototype){try{v.cancelVideoFrameCallback(S._rvfc);}catch(e){}}
    else{cancelAnimationFrame(S._rvfc);}
    S._rvfc=null;
  }
  $('#play').textContent='▶ play';
  // preload only metadata: these clips are full-res GOP=1 and can be 1-15 MB; eagerly buffering the
  // whole file on load() is what made big composite milestones slow. metadata + Range streaming lets
  // the browser fetch just the header, then seek/play on demand.
  v.preload='metadata';
  v.src=srcUrl;
  v.load();            // ensure the new source is actually loaded (some browsers keep the old buffer)
}
// Seek the video AND update the panel to frame i (slider index is the source of truth).
function gotoFrame(i){
  i=Math.max(0,Math.min(S.steps.steps.length-1,i));
  $('#slider').value=i;
  // seek to the MIDDLE of frame i's interval so we land inside it, not on the prior keyframe edge
  $('#vid').currentTime=(i+0.5)/S.fps;
  showFrame(i);
}
function stepFrame(d){$('#vid').pause();gotoFrame((+$('#slider').value)+d);}

function showFrame(i){
  const s=S.steps.steps[i]; if(!s)return;
  $('#fpos').textContent=`frame ${i} (step ${s.frame_step})`;
  // Dense per-step progress (progact): the model's progress prediction for the exact chunk offset
  // this frame executes (built in buildSeries). '*' marks it as the dense value; falls back to the
  // replan-only string s.progress (progcls/progreg have no per-step chunk).
  const dense=(S.progDense&&S.progDense[i]!=null)?S.progDense[i]:null;
  const progChip=(dense!=null)?`progress ${dense.toFixed(3)}*`:`progress ${s.progress}`;
  $('#fchips').innerHTML=[
    `<span class="c hot">|eef_pos| ${s.action_eef_pos_norm.toFixed(3)}</span>`,
    `<span class="c hot">|eef_rot| ${s.action_eef_rot_norm.toFixed(3)}</span>`,
    `<span class="c hot">|base| ${s.action_base_norm.toFixed(3)}</span>`,
    `<span class="c" title="${dense!=null?'dense per-step model progress (chunk offset for this frame)':'progress'}">${progChip}</span>`,
    s.sim_check_success?`<span class="c ok">sim_check_success</span>`:'',
    `<span class="c">gripper_w ${s.gripper_width.toFixed(3)}</span>`,
    s.replanned?`<span class="c rep">REPLAN</span>`:'',
  ].join('');
  S._curStep=s;
  const sp=$('#succ-prog'); if(sp)sp.textContent=(dense!=null?`${dense.toFixed(3)}* (dense per-step)`:progLabel(s.progress_raw));
  // Governing replan for THIS frame: the model was queried there and its prompt + chunk hold
  // until the next replan. Deriving from govQ (not a mutated _lastQuery) is correct when scrubbing
  // in any direction. gs = the governing step record; its .query is what the model actually saw.
  const gi=(S.govQ&&S.govQ[i]>=0)?S.govQ[i]:-1;
  const gs=gi>=0?S.steps.steps[gi]:null;
  S._govStep=gs; S._lastQuery=gs?gs.query:null;
  renderPromptFields(gs);   // prompt = the TRUE model input at the governing replan (constant for the window)
  renderNormable();
  // executed vs oracle as a labeled TABLE (one column per action dim, lean-11 order),
  // with a Δ row so it's obvious which number is which and where they diverge.
  const exL=sim12ToLean11(s.action_raw12), orL=sim12ToLean11(s.oracle_action_raw12);
  let h="<table class='act cmp'><tr><th></th>"+LEAN_LABELS.map(l=>`<th>${l}</th>`).join("")+"</tr>";
  h+="<tr><td class='rl'>executed</td>"+exL.map(x=>`<td>${x.toFixed(3)}</td>`).join("")+"</tr>";
  h+="<tr><td class='rl'>oracle</td>"+orL.map(x=>`<td>${x.toFixed(3)}</td>`).join("")+"</tr>";
  h+="<tr><td class='rl'>Δ</td>"+exL.map((x,j)=>{const dv=x-orL[j];const hot=Math.abs(dv)>0.15?' style=\"color:#c22;font-weight:700\"':'';return `<td${hot}>${dv>=0?'+':''}${dv.toFixed(3)}</td>`;}).join("")+"</tr>";
  h+="</table>";
  $('#execcmp').innerHTML=h+
    `<div class="kv" style="margin-top:5px"><span class="key">mse vs oracle</span><span class="val">${s.action_mse_vs_oracle.toFixed(4)}</span></div>`+
    `<div class="kv"><span class="key">eef_pos_world</span><span class="val">${fmt(s.eef_pos_world,3)}</span></div>`;
  drawCurves(i);
}

// Render the STATE block + the PREDICTED ACTION CHUNK in the currently-selected mode
// (normalized ⇄ raw), toggled together like /system1_training_sample's lean toggle.
function renderNormable(){
  const s=S._curStep, d=S.steps; if(!s)return;
  const norm=S.norm;
  const tgTxt=norm?'NORMALIZED — click for raw':'RAW — click to normalize';
  if($('#tg-state'))$('#tg-state').textContent=tgTxt;
  if($('#tg-chunk'))$('#tg-chunk').textContent=tgTxt;
  document.querySelectorAll('#lean-lab, .lean-lab2').forEach(e=>e.textContent=norm?'lean 14-d (normalized)':'lean 14-d (raw)');
  // state (floats) + the discretized 256-bin ints the tokenizer emits (from the normalized lean)
  const intsLine=(normArr)=>{const b=discretize(normArr);return b?('\n  256-bin ints: '+b.join(' ')):'';};
  $('#st-raw').textContent=stateBlock(s.cur_raw16,STATE_GROUPS);
  $('#st-lean').textContent=((norm&&s.cur_lean_norm)?stateBlock(s.cur_lean_norm,LEAN_GROUPS):stateBlock(s.cur_lean,LEAN_GROUPS))
                            +(s.cur_lean_norm?intsLine(s.cur_lean_norm):'');
  $('#st-anchor-lean').textContent=((norm&&d.anchor_state_lean14_norm)?stateBlock(d.anchor_state_lean14_norm,LEAN_GROUPS):stateBlock(d.anchor_state_lean14,LEAN_GROUPS))
                            +(d.anchor_state_lean14_norm?intsLine(d.anchor_state_lean14_norm):'');
  // full predicted chunk (from the last query), in LEAN-11 order for BOTH modes:
  //   normalized = chunk_lean11_norm (the model's DIRECT output, quantile [-1,1])
  //   unnormalized = chunk_lean11     (the REAL action that drives the sim, after Unnormalize)
  const q=S._lastQuery;
  if(q){
    // A run with NO norm_stats (e.g. the ORACLE method) still stores a chunk_lean11_norm, but it's
    // ALL ZEROS (nothing to normalize with). Treat that as "no norm available" and fall back to the
    // RAW chunk_lean11 so the panel isn't a table of zeros — the real GT actions show instead.
    const hasNorm=Array.isArray(q.chunk_lean11_norm)&&q.chunk_lean11_norm.some(r=>r.some(v=>v!=null&&v!==0));
    const useNorm=norm&&hasNorm;
    const rows=useNorm?q.chunk_lean11_norm:q.chunk_lean11;
    const normUnavail=norm&&!hasNorm;   // user wants norm but this run has none
    // progact ONLY: the 12th action dim = per-step progress. Stored as UNNORMALIZED [0,1]
    // (SplitProgressAction already did progress=(prog_norm+1)/2). In the NORMALIZED view show
    // the model's DIRECT output 2*p-1 ∈ [-1,1]; in the UNNORMALIZED view show p ∈ [0,1].
    const prog=q.chunk_progress;
    const progCell=(p)=>useNorm?(2*p-1):p;
    const labs=prog?LEAN_LABELS.concat(['progress*']):LEAN_LABELS;
    // which chunk row is executing at the CURRENT frame = (current step − governing replan step)
    const curOff=(S._govStep!=null)?(s.frame_step - S._govStep.frame_step):-1;
    const modeLbl=useNorm?'NORMALIZED — model direct output (quantile → [-1,1])'
                          :(normUnavail?'RAW (no norm_stats for this method — showing real actions → env.step)'
                                       :'UNNORMALIZED — real action → env.step');
    let h=`<div class="lab">${modeLbl} · lean-11 order${prog?' + progress* (12th action dim, '+(useNorm?'[-1,1] model output':'[0,1] unnorm')+')':''} · horizon ${q.horizon} · green = ${q.replan_steps} executed · ▶ = current frame</div>`;
    h+="<table class='act'><tr><th>t</th>"+labs.map(l=>`<th${l==='progress*'?' class="progcol"':''}>${l}</th>`).join("")+"</tr>";
    // a null cell = no GT action past the episode end (oracle) -> render blank, not "NaN"/crash.
    const fmtCell=x=>(x==null||Number.isNaN(x))?'':(+x).toFixed(3);
    rows.forEach((r,t)=>{
      // fully-empty row (all null) = past the recorded end: show a single "—" spanning the row.
      const empty=r.every(x=>x==null||Number.isNaN(x));
      let cells;
      if(empty){cells=`<td colspan="${labs.length}" style="color:#bbb;text-align:center">— (past episode end)</td>`;}
      else{cells=r.map(x=>`<td>${fmtCell(x)}</td>`).join("");
        if(prog)cells+=`<td class="progcol">${prog[t]==null?'':progCell(prog[t]).toFixed(3)}</td>`;}
      const cls=(t===curOff?'cur':(t<q.replan_steps?'exec':''));
      h+=`<tr class="${cls}"><td class="step">${t===curOff?'▶':''}${t}</td>`+cells+"</tr>";
    });
    $('#chunk').innerHTML=h+"</table>";
  } else $('#chunk').innerHTML='(warmup — no query yet)';
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  // Shift+Left/Right jumps SUBTASK (same as Up/Down); plain Left/Right steps one video FRAME.
  if(e.key==='ArrowRight'&&e.shiftKey){e.preventDefault();selectSub(S.subi+1);return;}
  if(e.key==='ArrowLeft'&&e.shiftKey){e.preventDefault();selectSub(S.subi-1);return;}
  if(e.key==='ArrowRight'){e.preventDefault();stepFrame(1);}
  if(e.key==='ArrowLeft'){e.preventDefault();stepFrame(-1);}
  if(e.key==='ArrowDown'){e.preventDefault();selectSub(S.subi+1);}
  if(e.key==='ArrowUp'){e.preventDefault();selectSub(S.subi-1);}
});
$('#method').onchange=loadEpisodes;
$('#tasktype').onchange=()=>fillTasks();   // switch type -> first task of that type
$('#task').onchange=()=>fillEpisodes();    // switch task -> first episode of that task
$('#episode').onchange=e=>selectEpisode(+e.target.value);
$('#prevSub').onclick=()=>selectSub(S.subi-1);
$('#nextSub').onclick=()=>selectSub(S.subi+1);
window.addEventListener('resize',()=>{if(S.series)drawCurves(+($('#slider')?.value||0));});
loadMethods();
"""


# ---------------------------------------------------------------------------
# /combine — COMBINED System2+System1 closed-loop rollouts (combined_eval.py output).
#
# Every intermediate artifact is exposed so a run can be audited turn by turn: the System2
# system+user prompt, its raw response and parsed tags, the exact media it saw, the System1 prompt
# + anchor image, the raw 20-fps rollout video, the condensed 4-fps clip that was actually fed
# back, and a per-frame table showing which frames the condenser kept or dropped and why.
# Result trees default under the shared results dir (05b-shared-bashrc.sh exports
# SYS1_RESULTS_DIR); a non-interactive shell falls back to the /shared layout.
# Results roots are DERIVED, never hardcoded to one machine: prefer the env contract, else the
# first existing "data" dir near this repo (<repo_root>/data, sibling, ~/data). Keeps both the
# shared-filesystem layout and the classic repo+sibling-data layout working with no flags.
COMBINE_ROOT: Path = _RESULTS / "combine"
# Flat-policy baseline rollout tree (see /baseline). Populated by an aws s3 sync; summarised for
# /stats by scripts/extract_baseline_results.py.
BASELINE_ROOT = _RESULTS / "baseline"
# Precomputed per-method summaries (scripts/extract_combine_results.py). /stats reads these
# instead of aggregating raw rollout output.
COMBINE_RESULTS_DIR: Path = _RESULTS / "combine_results"



# Benchmark scale: 50 target tasks x ~506 episodes each (from the LeRobot meta/info.json totals).
# Used only to extrapolate a wall-clock estimate from the episodes actually run.
COMBINE_BENCH_TASKS = 50
COMBINE_BENCH_EPISODES = 25307


def _is_snapshot_dir(name: str) -> bool:
    """A snapshot or container dir sitting beside real methods, not a method. See _combine_stats step 2.

    `<method>.premerge-backup-YYYYMMDD` is written by the merge tooling next to the tree it backs up,
    and `_merge_backups/` / `_stale_results/` are containers -- all three would otherwise be scanned
    for an index.json and listed as runs.
    """
    return (name.startswith("_") or ".premerge-backup" in name or ".backup" in name
            or name.endswith((".bak", ".old")))


def _is_newtask(rec: dict) -> bool:
    """True for a run over the held-out NEW-TASK set rather than the 50-task manifest.

    Detected from the DATA (every per_task entry has split "other"), with the method name as a
    fallback, so a future newtask run is classified without being renamed. The set GROWS -- it was
    24 tasks and is 34 as of 2026-08-16 -- so nothing here or in the UI hardcodes its size; the
    count shown in #0 is taken from each run's own per_task.
    """
    pt = rec.get("per_task") or {}
    if pt and all((t.get("split") or "other") == "other" for t in pt.values()):
        return True
    return "newtask" in str(rec.get("method", "")).lower()


def _combine_stats() -> dict:
    """Per-method COMBINED-eval summary, read from PRECOMPUTED per-method JSONs.

    ``scripts/extract_combine_results.py`` aggregates each method's index.json once into
    ``eval_results/combine_results/<method>.json`` (~10 KB vs the 217 KB index at 1000 episodes,
    and ~5 MB at the full 25,307-episode benchmark). /stats then renders with NO aggregation.

    Falls back to aggregating index.json in-process when a method has no extracted file yet -- so a
    sweep still in flight is visible immediately, and the page never silently shows nothing.
    """
    out: dict = {}
    # 1) precomputed files (preferred)
    if COMBINE_RESULTS_DIR.is_dir():
        for f in sorted(COMBINE_RESULTS_DIR.glob("*.json")):
            if f.name == "SUMMARY.json":
                continue
            try:
                rec = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            o = rec.get("overall") or {}
            out[rec.get("method", f.stem)] = {
                "n": o.get("n"), "n_success": o.get("n_success"), "rate": o.get("rate"),
                # REFINED = the same episodes re-scored under RoboCasa's official per-task step
                # horizon (horizon_gate.py). per_split / per_task below pass through whole, so they
                # already carry their own refined fields. None for a method extracted before this
                # existed, or one whose raw tree was pruned -- #0's toggle renders that as "n/a"
                # rather than as 0%.
                "n_success_refined": o.get("n_success_refined"),
                "rate_refined": o.get("rate_refined"),
                "n_refined_unknown": o.get("n_refined_unknown"),
                "avg_seconds": o.get("avg_seconds"), "total_seconds": o.get("total_seconds"),
                "avg_turns": o.get("avg_turns"), "n_error": rec.get("n_error"),
                "terminations": rec.get("terminations") or {},
                "per_split": rec.get("per_split") or {},
                "per_task": rec.get("per_task") or {},
                "bench": rec.get("bench") or {},
                # What KIND of eval this is. Set to "baseline_flat_policy" by
                # scripts/extract_baseline_results.py; absent for our own combine runs. #0 keys its
                # baseline toggle on this rather than on the method name, so a baseline run can be
                # named anything.
                "kind": rec.get("kind"),
                # NEWTASK runs: the 24-task held-out set, which lives outside TARGET_TASK_SPLIT (every
                # task reports split "other"). Used to hide these rows behind their own checkbox and to
                # deny them an "overall" (they cover none of the manifest). Their REFINED figure is
                # left INTACT and displayed: the gate does know these tasks' horizons and demotes
                # nothing on them, so refined equals raw -- rendering n/a instead would hide a real
                # result behind a technicality.
                "newtask": _is_newtask(rec),
                "src": "extracted",
            }
    # 2) live fallback for methods not extracted yet (a sweep still running)
    if COMBINE_ROOT.is_dir():
        splits = _target_split_map()
        for md in sorted(p for p in COMBINE_ROOT.iterdir() if p.is_dir()):
            # `<method>.premerge-backup-YYYYMMDD` snapshots carry a real index.json, so without this
            # they surface here as an extra method (the extractors skip them, which is exactly why
            # they reach this fallback). Matched by shape so future snapshots are skipped too.
            if md.name in out or _is_snapshot_dir(md.name):
                continue
            f = md / "index.json"
            if not f.exists():
                continue
            try:
                doc = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            eps = doc.get("episodes") or []
            if not eps:
                continue
            def blk(group):
                n = len(group)
                s_ = sum(1 for e in group if e.get("episode_success"))
                secs = [e["seconds"] for e in group if isinstance(e.get("seconds"), (int, float))]
                return {"n": n, "n_success": s_, "rate": (s_ / n) if n else None,
                        "avg_seconds": round(sum(secs) / len(secs), 2) if secs else None}
            per_split: dict = {}
            per_task: dict = {}
            for e in eps:
                t = e.get("task_name") or ""
                per_split.setdefault(splits.get(t, "other"), []).append(e)
                per_task.setdefault(t, []).append(e)
            terms: dict = {}
            for e in eps:
                k = e.get("termination") or "error"
                terms[k] = terms.get(k, 0) + 1
            o = blk(eps)
            turns = [e["n_turns"] for e in eps if isinstance(e.get("n_turns"), (int, float))]
            out[md.name] = {
                **o, "total_seconds": doc.get("total_seconds"),
                # index.json already carries n_turns per episode, so an in-flight sweep can show this
                # instead of "-" until extract_combine_results.py runs.
                "avg_turns": round(sum(turns) / len(turns), 2) if turns else None,
                "n_error": sum(1 for e in eps if e.get("error")),
                "terminations": terms,
                "per_split": {k: blk(v) for k, v in sorted(per_split.items())},
                "per_task": {k: {**blk(v), "split": splits.get(k, "other")}
                             for k, v in sorted(per_task.items())},
                "bench": {}, "src": "live (not extracted yet)",
            }
    return out


def _combine_methods() -> list[str]:
    r = COMBINE_ROOT
    return sorted([d.name for d in r.iterdir() if d.is_dir()]) if r.exists() else []


@app.route("/api/combine/methods")
def api_combine_methods():
    out = []
    for m in _combine_methods():
        idx = COMBINE_ROOT / m / "index.json"
        cfg_file = COMBINE_ROOT / m / "rule_config.json"
        doc = {}
        rule_config = None
        if idx.exists():
            try:
                doc = json.loads(idx.read_text())
            except Exception:  # noqa: BLE001
                doc = {}
        if cfg_file.exists():
            try:
                rule_config = json.loads(cfg_file.read_text())
            except Exception:  # noqa: BLE001 - surface the method even if metadata is torn
                rule_config = None
        if rule_config is None:
            rule_config = doc.get("rule_config")
        out.append({"method": m, "n_episodes": doc.get("n_episodes"),
                    "n_success": doc.get("n_success"), "success_rate": doc.get("success_rate"),
                    "rule_config": rule_config})
    return jsonify(out)


@app.route("/api/combine/episodes/<method>")
def api_combine_episodes(method):
    """LIGHTWEIGHT episode list for the dropdowns (task, task type, episode number).

    Reads the per-method ``index.json`` (one small aggregate file) rather than every per-episode
    ``episode.json`` -- at benchmark scale (25k episodes) the latter is thousands of multi-KB reads
    just to fill a dropdown. The FULL doc is fetched lazily by
    ``/api/combine/episode/<method>/<episode>`` only when an episode is actually selected.

    Each row is enriched with ``task_split`` (atomic_seen / composite_seen / composite_unseen, from
    robocasa's TARGET_TASKS registry) and ``episode_index``, so the GUI can group and label without
    parsing ids in JS.
    """
    md = COMBINE_ROOT / method
    splits = _target_split_map()

    def enrich(rec: dict) -> dict:
        ep_id = rec.get("episode_id") or ""
        task = rec.get("task_name") or (ep_id.split("/")[0] if ep_id else "")
        m = re.search(r"episode_(\d+)", ep_id) or re.search(r"episode_(\d+)", rec.get("dir") or "")
        rec["task_name"] = task
        rec["episode_index"] = int(m.group(1)) if m else None
        rec["task_split"] = splits.get(task, "other")
        return rec

    idx = md / "index.json"
    if idx.exists():
        try:
            doc = json.loads(idx.read_text())
            out = []
            for e in doc.get("episodes", []):
                ep_id = e.get("episode_id") or ""
                flat = ep_id.replace("/", "__")
                if flat and not (md / flat).is_dir():
                    continue          # listed but no longer on disk
                out.append(enrich({"dir": flat, **e}))
            if out:
                return jsonify(out)
        except Exception:  # noqa: BLE001 - fall through to the directory scan
            pass

    eps = []
    if md.exists():
        for d in sorted(p for p in md.iterdir() if p.is_dir()):
            f = d / "episode.json"
            if not f.exists():
                continue
            try:
                doc = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            eps.append(enrich({
                "dir": d.name, "episode_id": doc.get("episode_id"),
                "task_name": doc.get("task_name"),
                "episode_success": doc.get("episode_success"),
                "n_turns": doc.get("n_turns"), "termination": doc.get("termination"),
                "error": doc.get("error")}))
    return jsonify(eps)


@app.route("/api/combine/episode/<method>/<episode>")
def api_combine_episode(method, episode):
    f = COMBINE_ROOT / method / episode / "episode.json"
    if not f.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(f.read_text()))


@app.route("/api/combine/turn/<method>/<episode>/<turn>")
def api_combine_turn(method, episode, turn):
    """Full per-turn doc: S2 prompts/response + S1 segment + clip stats (incl. per-frame table)."""
    base = COMBINE_ROOT / method / episode / turn
    out = {}
    # memory/*.json exist only for the -memory plan variant (combine_memory_eval.py): memory.json
    # carries the per-chunk narration index, recipe.json the aggregation call's prompt+response.
    # Absent = cold run, and the keys are simply missing (the memory panel then never renders).
    for name, key in (("turn.json", "turn"), ("s1_steps.json", "s1_steps"), ("plan.json", "plan"),
                      ("memory/memory.json", "memory"), ("memory/recipe.json", "recipe")):
        f = base / name
        if f.exists():
            try:
                out[key] = json.loads(f.read_text())
            except Exception as e:  # noqa: BLE001
                out[key] = {"error": str(e)}
    if not out:
        return jsonify({"error": "not found"}), 404
    return jsonify(out)


@app.route("/api/combine/media/<method>/<episode>/<turn>/<path:fname>")
def api_combine_media(method, episode, turn, fname):
    """Serve any artifact under a turn dir (mp4 / png), including clip_frames/*.png."""
    base = (COMBINE_ROOT / method / episode / turn).resolve()
    p = (base / fname).resolve()
    if not str(p).startswith(str(base)) or not p.exists():   # no path traversal
        return jsonify({"error": "not found"}), 404
    return send_file(str(p))

# =================================================================================================
# FLAT-POLICY BASELINE browser (/baseline)
#
# A baseline unit is minimal by nature: ``baseline/<run>/<Task>/episode_NNNNNN/`` holds one
# ``episode.json`` (goal, success, steps, horizon) and one ``rollout.mp4`` of the WHOLE episode.
# There are no turns, no subgoals and no per-turn media, so /combine's turn-oriented UI has nothing
# to show for it -- hence a separate page rather than a mode of that one. This is a viewer only; the
# aggregate numbers live in /stats via scripts/extract_baseline_results.py.
# =================================================================================================


# TWO BASELINE LAYOUTS, same as scripts/extract_baseline_results.py:
#   xiaomi-robo1*  <Task>/episode_NNNNNN/episode.json      (nested)
#   pi05           <Task>__target__episode_NNNNNN/episode.json   (flat, combined_eval convention)
def _baseline_episode_files(run: Path) -> list[Path]:
    return (sorted(run.glob("*/episode_*/episode.json"))
            or sorted(run.glob("*/episode.json")))


def _baseline_runs() -> list[str]:
    if not BASELINE_ROOT.is_dir():
        return []
    return sorted(d.name for d in BASELINE_ROOT.iterdir()
                  if d.is_dir() and _baseline_episode_files(d))


@app.route("/api/baseline/methods")
def api_baseline_methods():
    out = []
    for name in _baseline_runs():
        n = ns = 0
        for f in _baseline_episode_files(BASELINE_ROOT / name):
            try:
                d = json.loads(f.read_text())
            except Exception:  # noqa: BLE001 - a torn file must not hide the run
                continue
            n += 1
            # RAW outcome. pi05's `episode_success` is already horizon-gated, so its raw rate lives in
            # `raw_episode_success`; xiaomi has a plain `success`. Counting the wrong one understates
            # pi05 by 64 episodes.
            ns += bool(d.get("raw_episode_success", d.get("success", d.get("episode_success"))))
        out.append({"method": name, "n_episodes": n, "n_success": ns})
    return jsonify(out)


@app.route("/api/baseline/episodes/<method>")
def api_baseline_episodes(method):
    """One row per episode: the goal, the outcome, and where the video is."""
    base = (BASELINE_ROOT / method).resolve()
    if not str(base).startswith(str(BASELINE_ROOT.resolve())) or not base.is_dir():
        return jsonify({"error": "not found"}), 404
    rows = []
    for f in _baseline_episode_files(base):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        task = d.get("task_name") or f.parent.parent.name
        vids = [x for x in f.parent.iterdir() if x.suffix == ".mp4"]
        raw = bool(d.get("raw_episode_success", d.get("success", d.get("episode_success"))))
        scored = d.get("episode_success") if "raw_episode_success" in d else None
        rows.append({
            "task": task,
            "task_split": _target_split_map().get(task, "other"),
            "episode": int(d.get("episode_index", 0)),
            # Relative to the run dir, so one media route serves both layouts.
            "dir": str(f.parent.relative_to(base)),
            # The GOAL as the policy received it, and the dataset's own wording when they differ
            # (the -unseenshort runs were given a one-line goal instead of the full instruction).
            "instruction": d.get("instruction"),
            "source_instruction": d.get("source_instruction"),
            "instruction_mode": d.get("instruction_mode"),
            "success": raw,
            # Present only where the producer also reports a horizon-gated outcome (pi05). When it
            # differs from `success`, the episode solved the task PAST the official horizon.
            "success_scored": scored,
            "success_step": d.get("success_step"),
            "termination": d.get("termination"),
            # CAREFUL: pi05 uses "steps" for the steps.npz FILENAME and "n_steps" for the count,
            # while xiaomi uses "steps" for the count. Take whichever is actually a number.
            "steps": (d["steps"] if isinstance(d.get("steps"), (int, float))
                      else d.get("n_steps")),
            # Horizon key differs per producer: xiaomi `horizon`, pi05 `scoring_cap`,
            # abot `official_max_steps` / `robocasa_task_horizon`.
            "horizon": next((d[k] for k in ("horizon", "scoring_cap", "official_max_steps",
                                            "robocasa_task_horizon") if d.get(k) is not None), None),
            "seconds": d.get("seconds"),
            # BASENAME ONLY. abot records an ABSOLUTE path from the machine that produced it
            # (/tmp/ABot_eval_results/...), which is meaningless here; the file sits beside
            # episode.json. `has_video` is reported because abot ships videos for only the first 10
            # episodes of each task (500 of 1500), so the player must be able to say so rather than
            # silently fail.
            "video": os.path.basename(d.get("video") or "") or "rollout.mp4",
            "has_video": bool(vids),
        })
    rows.sort(key=lambda r: (r["task"], r["episode"]))
    return jsonify(rows)


@app.route("/api/baseline/media/<method>/<path:relpath>")
def api_baseline_media(method, relpath):
    base = (BASELINE_ROOT / method).resolve()
    p = (base / relpath).resolve()
    if not str(p).startswith(str(base)) or not p.exists():   # no path traversal
        return jsonify({"error": "not found"}), 404
    return send_file(str(p))


BASELINE_HTML = """<!doctype html><meta charset=utf-8>
<title>flat-policy baseline rollouts</title>
<style>
 *{box-sizing:border-box}
 body{font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;color:#1a1a1a;
      background:#fafafb;height:100vh;display:flex;flex-direction:column;overflow:hidden}
 a{color:#0a58ca}
 select,button{font:13px inherit;padding:3px 8px}
 #top{flex:0 0 auto;background:#fff;border-bottom:1px solid #ddd;padding:7px 12px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 h1{font-size:14px;margin:0 6px 0 0}h1 b{color:#b0431c}
 nav{margin-left:auto;display:flex;gap:10px}
 .nav button{border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer;padding:3px 11px}
 .nav button:disabled{opacity:.4;cursor:default}
 .pos{font-family:ui-monospace,monospace;font-size:12px;color:#555;min-width:110px}
 #body{flex:1 1 auto;overflow:auto;padding:12px;display:flex;gap:14px;align-items:flex-start}
 #vidwrap{flex:0 0 auto;display:flex;flex-direction:column;gap:6px}
 video{background:#000;border-radius:8px;max-height:68vh}
 /* video scrubber, same interaction model as /combine's .player: a range the timeupdate handler
    drives, and which seeks (paused) on drag. */
 .player{display:flex;align-items:center;gap:7px}
 .player input[type=range]{flex:1;min-width:220px}
 .player button{border:1px solid #bbb;background:#fff;border-radius:5px;cursor:pointer;padding:2px 9px}
 .player .cnt{font-family:ui-monospace,monospace;font-size:11px;color:#555;min-width:150px;
              text-align:right}
 .card{background:#fff;border:1px solid #e2e2e6;border-radius:9px;padding:11px 13px;flex:1 1 320px;
       min-width:300px}
 .k{color:#666;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:2px}
 .v{margin-bottom:10px;font-size:13px}
 .goal{font-size:15px;font-weight:600;color:#1a1a1a}
 .mono{font-family:ui-monospace,monospace;font-size:12px}
 .ok{color:#137333;font-weight:700}.bad{color:#b3261e;font-weight:700}
 .pill{display:inline-block;border:1px solid #ddd;border-radius:999px;padding:1px 8px;font-size:11px;
       color:#555;background:#fafafa;margin-right:5px}
</style>
<div id=top>
 <div class=row>
  <h1>flat-policy <b>baseline</b> rollouts</h1>
  <select id=method></select>
  <select id=tasktype></select>
  <select id=task></select>
  <label>episode <select id=episode></select></label>
  <span class="nav"><button id=prev>&#8592; prev</button><button id=next>next &#8594;</button></span>
  <span class=pos id=pos></span>
  <nav><a href="/">home</a><a href="/combine">combine</a><a href="/stats">stats</a></nav>
 </div>
</div>
<div id=body>
 <div id=vidwrap>
  <video id=vid playsinline muted></video>
  <div class=player>
   <button id=play>&#9654;</button>
   <input type=range id=srange min=0 max=1000 value=0 step=1>
   <span class=cnt id=cnt>-- / --</span>
  </div>
 </div>
 <div class=card id=meta></div>
</div>
<script>
const S={method:null,all:[],eps:[],i:0};
const $=q=>document.querySelector(q);
const esc=t=>String(t==null?'':t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const mmss=t=>{if(!isFinite(t))return'--:--';const m=Math.floor(t/60),x=Math.floor(t%60);
  return `${m}:${String(x).padStart(2,'0')}`;};

async function boot(){
  const ms=await (await fetch('/api/baseline/methods')).json();
  if(!ms.length){$('#body').innerHTML='<div class=card>No baseline runs under the baseline root.</div>';return;}
  $('#method').innerHTML=ms.map(m=>`<option value="${m.method}">${m.method} (${m.n_success}/${m.n_episodes})</option>`).join('');
  $('#method').onchange=()=>loadMethod($('#method').value);
  wirePlayer();
  await loadMethod(ms[0].method);
}
async function loadMethod(m){
  S.method=m;
  S.all=await (await fetch(`/api/baseline/episodes/${m}`)).json();
  // ONE split and ONE task at a time, exactly like /combine. An "all" option would make the episode
  // labels ambiguous -- ep0 exists in every task -- and /combine's convention is that the episode
  // dropdown is always scoped to a single task, so `\u2713 ep0` identifies a row on its own.
  const order=['atomic_seen','composite_seen','composite_unseen','other'];
  const present=new Set(S.all.map(e=>e.task_split||'other'));
  const splits=order.filter(t=>present.has(t));
  const keepSp=$('#tasktype').value;
  $('#tasktype').innerHTML=splits.map(t=>{
    const n=S.all.filter(e=>(e.task_split||'other')===t);
    return `<option value="${t}">${t} (${n.filter(e=>e.success).length}/${n.length})</option>`;}).join('');
  if(splits.includes(keepSp))$('#tasktype').value=keepSp;
  $('#tasktype').onchange=fillTasks; fillTasks();
}
function fillTasks(){
  const sp=$('#tasktype').value;
  const pool=S.all.filter(e=>(e.task_split||'other')===sp);
  const tasks=[...new Set(pool.map(e=>e.task))].sort();
  const keep=$('#task').value;
  $('#task').innerHTML=tasks.map(t=>{const n=pool.filter(e=>e.task===t);
      return `<option value="${t}">${t} (${n.filter(e=>e.success).length}/${n.length})</option>`;}).join('');
  if(tasks.includes(keep))$('#task').value=keep;
  $('#task').onchange=applyFilter; applyFilter();
}
function applyFilter(){
  const sp=$('#tasktype').value, tk=$('#task').value;
  S.eps=S.all.filter(e=>(e.task_split||'other')===sp&&e.task===tk)
             .sort((a,b)=>(a.episode??0)-(b.episode??0));
  S.i=0; fillEpisodes(); show();
}
// SUCCESS MARK IN THE DROPDOWN, the same label /combine uses: "<mark> ep<N>". The list is always
// scoped to one task by the selectors above, so the episode number alone identifies the row.
function fillEpisodes(){
  $('#episode').innerHTML=S.eps.map((e,i)=>{
    const ep=e.episode!=null?`ep${e.episode}`:'ep?';
    return `<option value="${i}">${e.success?'\u2713':'\u2717'} ${ep}</option>`;}).join('');
  $('#episode').value=String(S.i);
}
$('#episode').onchange=e=>{S.i=+e.target.value; show();};

function wirePlayer(){
  const v=$('#vid'), rng=$('#srange');
  v.ontimeupdate=()=>{
    if(!isFinite(v.duration)||v.duration<=0)return;
    const f=v.currentTime/v.duration;
    if(!rng.dragging)rng.value=Math.round(f*1000);
    renderCnt();
  };
  v.onloadedmetadata=()=>{rng.value=0;renderCnt();};
  v.onplay=()=>{$('#play').innerHTML='&#10073;&#10073;';};
  v.onpause=()=>{$('#play').innerHTML='&#9654;';};
  v.onended=()=>{$('#play').innerHTML='&#9654;';};
  $('#play').onclick=()=>{if(v.paused)v.play().catch(()=>{});else v.pause();};
  rng.oninput=e=>{
    if(!isFinite(v.duration))return;
    v.pause(); v.currentTime=(+e.target.value/1000)*v.duration; renderCnt();
  };
  // Drag guard: while the thumb is held, timeupdate must not fight the user for the value.
  rng.onpointerdown=()=>{rng.dragging=true;};
  rng.onpointerup=()=>{rng.dragging=false;};
}
// The counter shows BOTH clocks: video time, and the env step the playhead implies. A flat rollout
// ran `steps` env steps into a `horizon`-step budget, so "step 412 / 1223" is the number that makes
// the video comparable with the /stats row.
function renderCnt(){
  const v=$('#vid'), e=S.eps[S.i];
  if(!e){$('#cnt').textContent='-- / --';return;}
  const d=isFinite(v.duration)?v.duration:0;
  const f=d>0?v.currentTime/d:0;
  const step=e.steps!=null?Math.round(f*e.steps):null;
  $('#cnt').textContent=`${mmss(v.currentTime)} / ${mmss(d)}`
    +(step!=null?`   step ${step} / ${e.steps}`:'');
}
function show(){
  const e=S.eps[S.i];
  if(!e){$('#meta').innerHTML='<div class=v>no episodes match</div>';$('#vid').removeAttribute('src');
         $('#pos').textContent='0 / 0';$('#cnt').textContent='-- / --';return;}
  $('#pos').textContent=`${S.i+1} / ${S.eps.length}`;
  $('#prev').disabled=S.i<=0; $('#next').disabled=S.i>=S.eps.length-1;
  if($('#episode').value!==String(S.i))$('#episode').value=String(S.i);
  // abot ships video for only 10 episodes per task, so an absent file is normal, not an error.
  if(e.has_video===false){$('#vid').removeAttribute('src');$('#cnt').textContent='no video for this episode';}
  else $('#vid').src=`/api/baseline/media/${S.method}/${e.dir}/${e.video}`;
  const pct=(e.steps!=null&&e.horizon)?Math.round(100*e.steps/e.horizon):null;
  const orig=e.source_instruction&&e.source_instruction!==e.instruction
    ? `<div class=k>dataset instruction</div><div class="v mono">${esc(e.source_instruction)}</div>`:'';
  $('#meta').innerHTML=
    `<div class=k>task goal given to the policy</div><div class="v goal">${esc(e.instruction)}</div>`
   +orig
   +`<div class=v>${e.instruction_mode?`<span class=pill>${esc(e.instruction_mode)}</span>`:''}`
   +`<span class=pill>${esc(e.task_split)}</span><span class=pill>${esc(e.task)}</span>`
   +`<span class=pill>episode ${e.episode}</span></div>`
   +`<div class=k>outcome</div><div class=v>`
   +(e.success?'<span class=ok>&#10003; SUCCESS</span>':'<span class=bad>&#10007; FAILURE</span>')
   // A run that reports a horizon-gated outcome too (pi05): when the two disagree the task WAS solved
   // but only after the official budget, which is the distinction /stats' refined column is built on.
   +((e.success&&e.success_scored===false)
      ? ' <span class=pill style="border-color:#dcae4a;color:#6b4a12;background:#fff8e8">past horizon'
        +' \u2014 not counted as refined</span>':'')
   +(e.success_step!=null?` <span class=pill>success at step ${e.success_step}</span>`:'')
   +`</div>`
   +(e.termination?`<div class=k>termination</div><div class="v mono">${esc(e.termination)}</div>`:'')
   +`<div class=k>env steps</div><div class="v mono">${e.steps} / horizon ${e.horizon}`
   +(pct!=null?`  (${pct}% of budget)`:'')+`</div>`
   +`<div class=k>wall clock</div><div class="v mono">${e.seconds}s</div>`
   +`<div class=k>dir</div><div class="v mono">${esc(e.dir)}</div>`;
}
$('#prev').onclick=()=>{if(S.i>0){S.i--;show();}};
$('#next').onclick=()=>{if(S.i<S.eps.length-1){S.i++;show();}};
document.addEventListener('keydown',ev=>{
  if(ev.target.tagName==='SELECT'||ev.target.tagName==='INPUT')return;
  if(ev.key==='ArrowLeft'){$('#prev').click();ev.preventDefault();}
  if(ev.key==='ArrowRight'){$('#next').click();ev.preventDefault();}
  if(ev.key===' '){$('#play').click();ev.preventDefault();}
});
boot();
</script>
"""


@app.route("/baseline")
def baseline_page():
    return BASELINE_HTML.replace("<script>", _ERR_JS + "<script>", 1)


COMBINE_HTML = """<!doctype html><meta charset=utf-8>
<title>System2+System1 combined eval</title>
<style>
 *{box-sizing:border-box}
 body{font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;color:#1a1a1a;
      background:#fafafb;height:100vh;display:flex;flex-direction:column;overflow:hidden}
 a{color:#0a58ca}
 select,button{font:13px inherit;padding:3px 8px}
 /* ---- sticky top: selectors + turn nav + track ---- */
 #top{flex:0 0 auto;background:#fff;border-bottom:1px solid #ddd;padding:7px 12px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 h1{font-size:14px;margin:0 6px 0 0}h1 b{color:#b0431c}
 .nav button{border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer;padding:3px 11px}
 .nav button:disabled{opacity:.4;cursor:default}
 .nav button:not(:disabled):hover{background:#f0f3fa}
 .pos{font-family:ui-monospace,monospace;font-size:12px;color:#555;min-width:96px}
 /* Method name + episode summary sit together at the right end of the header bar. margin-left:auto
    lives on .epmeth (the FIRST of the pair) so the two stay adjacent instead of being pushed apart. */
 .epmeth{font-size:12px;color:#b0431c;font-weight:700;margin-left:auto;
         font-family:ui-monospace,monospace}
 .epsum{font-size:12px;color:#666;font-family:ui-monospace,monospace}
 /* ---- track: one segment per turn, width ~ steps executed ---- */
 #track{margin-top:6px}
 .lane{display:flex;align-items:center;gap:8px;margin-top:3px}
 .lab{flex:0 0 74px;font-size:10px;color:#999;text-align:right;text-transform:uppercase;letter-spacing:.04em}
 .bar{position:relative;flex:1;height:24px;background:#f1f1f4;border:1px solid #e3e3e8;border-radius:4px}
 .seg{position:absolute;top:1px;bottom:1px;border-radius:3px;cursor:pointer;padding:0 5px;font-size:10px;
      line-height:20px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;
      font-family:ui-monospace,monospace;box-sizing:border-box;border:1px solid}
 .seg:hover{filter:brightness(.95)}
 .seg.cur{font-weight:700;z-index:3;box-shadow:0 0 0 2px #1a1a1a inset}
 .s-plan{background:#e9e2f8;border-color:#a98fd8;color:#3f2a6b}
 /* the memory pass (narrate -> recipe) exists only in the -memory plan variant */
 .s-memory{background:#dcefe4;border-color:#7fb99a;color:#1f4a35}
 .s-begin{background:#e8e8ec;border-color:#bbb;color:#444}
 .s-complete{background:#cfe0ff;border-color:#7fa8f0;color:#1c3a6b}
 .s-incomplete{background:#fff1cf;border-color:#dcae4a;color:#6b4a12}
 .s-failed{background:#fbdcdc;border-color:#d07070;color:#7a1f1f}
 .s-finish{background:#d3f0d8;border-color:#5fa96b;color:#1d4b26}
 /* ---- body: 2 columns ---- */
 #body{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:2fr 4fr;gap:9px;padding:9px}
 #body.single{grid-template-columns:minmax(0,900px)}
 .col{min-height:0;display:flex;flex-direction:column;gap:8px;overflow-y:auto;overflow-x:hidden}
 .card{background:#fff;border:1px solid #e3e3e8;border-radius:7px;padding:8px 10px}
 .card h3{margin:0 0 6px;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#8a8a92;
          border-bottom:1px solid #f0f0f3;padding-bottom:4px}
 .card h3 span{float:right;text-transform:none;letter-spacing:0;color:#b0431c;font-weight:700}
 pre{white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid #eee;border-radius:4px;
     padding:7px;margin:3px 0;font:12px/1.45 ui-monospace,Menlo,monospace}
 .k{color:#999;font-size:11px}
 .kv{margin:2px 0}
 .tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;border:1px solid #bbb;margin-right:4px}
 .j-task_finish{background:#d3f0d8;border-color:#5fa96b}
 .j-subgoal_complete{background:#cfe0ff;border-color:#7fa8f0}
 .j-subgoal_incomplete{background:#fff1cf;border-color:#dcae4a}
 .j-subgoal_failed{background:#fbdcdc;border-color:#d07070}
 .j-task_begin{background:#e8e8ec}
 .sg{font-size:15px;font-weight:700;color:#0a4a9c;margin:3px 0}
 /* visual inputs at 2/3 width: the tiled 3-cam frames are very wide, so shrinking them frees
    horizontal space for the text/number panels beside them */
 video,img.tile{width:66%;max-width:100%;border:1px solid #ccc;border-radius:4px;background:#000;display:block}
 video.full,img.tile.full{width:100%}
 table{border-collapse:collapse;font-size:11px;width:100%} td,th{border:1px solid #e6e6ea;padding:1px 5px}
 th{background:#f5f5f8;position:sticky;top:0;z-index:1}
 .drop{color:#c0c0c0} .keep{background:rgba(46,139,61,.13);font-weight:600}
 /* condensed-clip thumbnails: the strip is capped at the same 2/3 width as every other visual
    input so it lines up with the videos above it instead of running the full column. */
 .frames{display:flex;gap:3px;flex-wrap:wrap;width:66%;max-width:100%}
 .frames img{height:46px;border:1px solid #ccc}
 .muted{color:#888;font-size:12px}
 details>summary{cursor:pointer;font-size:11px;color:#666;padding:2px 0}
 .scroll{max-height:230px;overflow:auto}
 .pill{display:inline-block;background:#f0f0f4;border:1px solid #ddd;border-radius:4px;padding:1px 6px;
       font-family:ui-monospace,monospace;font-size:11px;margin-right:4px}
 /* maintained task plan (aggregated from System2 <plan_update>s) — fixed height, scrollable so the
    full history of milestones AND fine steps stays reachable without pushing the panels down */
 .plangrid{display:flex;flex-direction:column;gap:1px;padding:5px 7px;background:#fcfcfd;
           border:1px solid #e6e6ea;border-radius:5px;height:190px;overflow:auto}
 .pl{font-family:ui-monospace,monospace;font-size:11.5px;padding:1px 5px;border-radius:3px;
     border-left:3px solid transparent;white-space:pre}
 .pl-child{margin-left:16px;font-size:11px;opacity:.92}
 .pl-todo{color:#666;border-left-color:#ddd}
 .pl-doing{color:#7a4a10;background:#fff8e8;border-left-color:#dcae4a;font-weight:600}
 .pl-done{color:#2b6b36;background:#f0f9f1;border-left-color:#5fa96b}
 .pl-new{box-shadow:0 0 0 1px #b0431c inset}
 /* step scrubber: drag to walk the segment; prompt + action-chunk pointer follow */
 .scrub{display:flex;align-items:center;gap:8px;margin:5px 0 2px}
 .scrub input[type=range]{flex:1}
 .scrub .cnt{font-family:ui-monospace,monospace;font-size:11px;color:#555;min-width:96px}
 .chunkrow.at{background:#ffe9d6;box-shadow:0 0 0 1px #b0431c inset;font-weight:700}
 .chunkrow.exec{background:rgba(46,139,61,.10)}
 .ptr{color:#b0431c;font-weight:700}
 /* video player + synchronized curves, same interaction model as /episode */
 .player{display:flex;align-items:center;gap:7px;margin:4px 0}
 .player input[type=range]{flex:1;min-width:120px}
 .player button{border:1px solid #bbb;background:#fff;border-radius:5px;cursor:pointer;padding:2px 8px}
 .fld{margin-top:5px}
 .fld .cap{font-size:10px;color:#999;text-transform:uppercase;letter-spacing:.04em;display:flex;
           justify-content:space-between}
 canvas.curve{width:100%;height:82px;display:block;border:1px solid #eee;border-radius:4px;background:#fff}
 .lg{font-size:10px;color:#666} .lg i{display:inline-block;width:9px;height:3px;margin:0 3px 2px 6px}
 .tcost{font-size:11px;color:#555;font-family:ui-monospace,monospace;margin-top:3px}
 /* MEMORY step: one row per narrated clip -- the 4s video System2 watched beside the single
    sentence it produced, so the narration can be checked against the pixels that caused it. */
 .memrow{display:grid;grid-template-columns:300px 1fr;gap:10px;align-items:start;
   padding:5px 0;border-top:1px solid #f0f0f0}
 .memrow:first-child{border-top:0}
 .memrow video{width:300px;border-radius:3px;display:block;background:#000}
 .memrow .narr{margin-top:2px}
 /* the recipe is the artifact that crosses from the memory step into the plan step */
 pre.recipe{background:#f4fbf7;border-color:#cfe8dc}
 /* two minipage columns inside the System1 output card: video+curves stacked on the left, the
    predicted action chunk beside them on the right so the pointer is visible while the video plays */
 .mini{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-items:start}
 .minicol{min-width:0}
 .minicol .scroll{max-height:none}
 /* the action chunk is short (one H-step chunk), so show it in full — no inner scroll */
 #chunkbody .scroll{max-height:none;overflow:visible}
 .stopwhy{margin-top:7px;padding:5px 8px;border:1px solid;border-radius:5px;font-size:12px}
 .stopwhy code{font-family:ui-monospace,monospace;background:rgba(0,0,0,.05);padding:0 3px;border-radius:2px}
</style>
<div id=top>
  <div class=row>
    <h1>RoboCasa <b>S2+S1</b></h1>
    <label>method <select id=method></select></label>
    <label>type <select id=tasktype></select></label>
    <label>task <select id=task></select></label>
    <label>episode <select id=episode></select></label>
    <span class=nav>
      <button id=prev>&lsaquo; prev</button>
      <span class=pos id=pos></span>
      <button id=next>next &rsaquo;</button>
    </span>
    <span class=epmeth id=epmethod title="the run these rollouts came from (the method dropdown above)"></span>
    <span class=epsum id=epsum></span>
  </div>
  <div id=track></div>
</div>
<div id=body>
  <div class=col id=left></div>
  <div class=col id=right></div>
</div>
<script>
const $=s=>document.querySelector(s);
const S={method:null,ep:null,doc:null,ti:0,turns:[],ruleConfig:null}; // ti=0 is plan
const esc=s=>(s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const jtag=j=>j?`<span class="tag j-${esc(j)}">${esc(j)}</span>`:'';
const nn=v=>(v==null?'&ndash;':v);

// Judge -> track segment class. The synthetic "memory"/"plan" turns get their own colours.
function segClass(t){
  if(t.kind==='memory')return 's-memory';
  if(t.kind==='plan')return 's-plan';
  const j=t.judge||'';
  if(j==='task_finish')return 's-finish';
  if(j==='subgoal_complete')return 's-complete';
  if(j==='subgoal_incomplete')return 's-incomplete';
  if(j==='subgoal_failed')return 's-failed';
  return 's-begin';
}

async function loadMethods(){
  const ms=await (await fetch('/api/combine/methods')).json();
  window._combineRuleConfigs=Object.fromEntries(ms.map(m=>[m.method,m.rule_config||null]));
  // On THIS page debug-* runs stay selectable -- inspecting a debug rollout turn by turn is the
  // whole point of /combine -- they just must not be the DEFAULT selection, which would silently
  // open a 2-episode throwaway instead of a real sweep. So: keep every option, sort debug last,
  // and default to the first non-debug method.
  // debug-* sorts last and is never the default selection; *-memory is a first-class run and sorts
  // with the rest (it keeps its "[memory]" tag below so the different pipeline stays visible).
  const isDbg=m=>/^debug-/i.test(String(m||''));
  const ordered=ms.slice().sort((a,b)=>(isDbg(a.method)?1:0)-(isDbg(b.method)?1:0));
  // TAG PRECEDENCE: [unseenshort] wins over [memory]. Every terse-goal arm carries "unseenshort" in
  // its name, including ...-unseenshort-memory, and what matters when picking one to inspect is that
  // the GOAL was degraded to one line -- the plan source is the axis those arms differ on, not a
  // separate pipeline. [memory] is kept for a memory run that is NOT part of that family.
  const tagOf=m=>/unseenshort/i.test(m)?'[unseenshort] '
                :/-memory$/i.test(m)?'[memory] '
                :isDbg(m)?'[debug] ':'';
  $('#method').innerHTML=ordered.map(m=>`<option value="${m.method}">${tagOf(m.method)}${m.method} (${m.n_success??'?'}/${m.n_episodes??'?'})</option>`).join('');
  if(!ms.length){$('#left').innerHTML='<div class=card><span class=muted>No combined runs under eval_results/combine yet.</span></div>';return;}
  S.method=(ordered.find(m=>!isDbg(m.method))||ordered[0]).method;
  S.ruleConfig=window._combineRuleConfigs[S.method]||null;
  $('#method').value=S.method;   // explicit: option 0 may now be a different method than S.method
  await loadEpisodes();
}
const SPLIT_LABEL={atomic_seen:'atomic-seen',composite_seen:'composite-seen',
                   composite_unseen:'composite-unseen',other:'other'};
const SPLIT_ORDER=['atomic_seen','composite_seen','composite_unseen','other'];
// Only the LIGHTWEIGHT list is fetched here (from the method's index.json). The full per-episode
// doc — every turn, prompt and clip_stats — is loaded lazily in selectEpisode().
async function loadEpisodes(){
  const eps=await (await fetch(`/api/combine/episodes/${S.method}`)).json();
  window._eps=eps;
  const present=new Set(eps.map(e=>e.task_split||'other'));
  const types=SPLIT_ORDER.filter(t=>present.has(t));
  const keep=$('#tasktype').value;
  $('#tasktype').innerHTML=types.map(t=>{
    const n=eps.filter(e=>(e.task_split||'other')===t).length;
    return `<option value="${t}">${SPLIT_LABEL[t]||t} (${n})</option>`;}).join('');
  $('#tasktype').value=(keep&&types.includes(keep))?keep:(types[0]||'');
  fillTasks();
}
// TYPE -> TASK -> EPISODE cascade. Each level filters the next and tries to preserve the current
// selection, so switching type/task doesn't lose your place when the same task/episode still exists.
function fillTasks(){
  const t=$('#tasktype').value, eps=window._eps||[];
  const tasks=[...new Set(eps.filter(e=>(e.task_split||'other')===t).map(e=>e.task_name))].sort();
  const keep=$('#task').value;
  $('#task').innerHTML=tasks.map(n=>{
    const rows=eps.filter(e=>e.task_name===n);
    const ok=rows.filter(e=>e.episode_success).length;
    return `<option value="${n}">${n} (${ok}/${rows.length})</option>`;}).join('');
  // Set the value EXPLICITLY: don't rely on innerHTML implicitly selecting option 0, otherwise a
  // stale value from the previous type can survive and leave the episode list empty.
  $('#task').value=(keep&&tasks.includes(keep))?keep:(tasks[0]||'');
  fillEpisodes();
}
function fillEpisodes(){
  const task=$('#task').value, eps=window._eps||[];
  // Keep the ORIGINAL index so selectEpisode() still addresses window._eps directly.
  const rows=eps.map((e,i)=>({e,i})).filter(({e})=>e.task_name===task)
                .sort((a,b)=>(a.e.episode_index??0)-(b.e.episode_index??0));
  const keep=+$('#episode').value;
  $('#episode').innerHTML=rows.map(({e,i})=>{
    const ep=e.episode_index!=null?`ep${e.episode_index}`:'ep?';
    return `<option value="${i}">${e.episode_success?'\u2713':'\u2717'} ${ep}</option>`;}).join('');
  const first=rows.length?String(rows[0].i):'';
  $('#episode').value=rows.some(r=>r.i===keep)?String(keep):first;
  const sel=+$('#episode').value;
  if(rows.length)selectEpisode(Number.isNaN(sel)?rows[0].i:sel);
  else{$('#left').innerHTML='<div class=card><span class=muted>no episodes</span></div>';
       $('#right').innerHTML='';$('#track').innerHTML='';}
}
async function selectEpisode(i){
  const e=window._eps[i]; S.ep=e.dir;
  S.doc=await (await fetch(`/api/combine/episode/${S.method}/${S.ep}`)).json();
  // The method-level file is authoritative. Per-episode metadata is a fallback for copied legacy
  // episodes whose rule_config.json was not copied with them.
  S.ruleConfig=(window._combineRuleConfigs||{})[S.method]||S.doc.rule_config||null;
  const d=S.doc;
  // Turn list = a synthetic MEMORY step (only for the -memory variant, where System2 narrated a
  // video before planning) + a synthetic PLAN turn (System2 only) + every execution turn.
  const hasMem=!!(d.plan||{}).memory;
  S.turns=(hasMem?[{kind:'memory'}]:[]).concat([{kind:'plan'}],
    (d.turns||[]).map(t=>({kind:'exec',...t})));
  // Method name in front of the episode summary: with several sweeps in the dropdown it is otherwise
  // easy to read a rollout and forget WHICH run it belongs to (the dropdown scrolls out of view once
  // the turn track and video are open). Taken from S.method, i.e. the run actually loaded.
  $('#epmethod').textContent=S.method?`${S.method} ·`:'';
  const rt=S.ruleConfig&&S.ruleConfig.tier?` · rules=${S.ruleConfig.tier}`:'';
  $('#epsum').textContent=`${d.task_name} · ${d.episode_success?'SUCCESS':'FAIL'} · ${d.n_turns} turns · ${d.termination}${rt} · ${d.seconds}s`;
  drawTrack();
  selectTurn(0);
}
// Track: segment width proportional to steps executed (plan/finish turns get a fixed slice), so
// the bar reads as a timeline of where the episode actually spent its control steps.
function drawTrack(){
  const ts=S.turns;
  const w=ts.map(t=>Math.max(t.kind==='exec'?(t.n_steps||0):26,26));
  const tot=w.reduce((a,b)=>a+b,0)||1;
  const mem=(S.doc.plan||{}).memory||{};
  let acc=0;
  const segs=ts.map((t,i)=>{
    const L=100*acc/tot, W=100*w[i]/tot; acc+=w[i];
    const lab=t.kind==='memory'?'MEMORY':t.kind==='plan'?'PLAN'
      :`t${t.turn} ${(t.judge||'?').replace('subgoal_','sg_')}`;
    const tip=t.kind==='memory'
        ?`memory pass (System2 only): narrated ${mem.n_chunks_narrated??'?'} clips of the demo of `
         +`episode ${mem.source_episode??'?'} into a reusable recipe`
      :t.kind==='plan'?'plan mode (System2 only)'
      :`turn ${t.turn}: ${t.judge||'?'}\\n${t.subgoal||'(no subgoal)'}\\nsteps=${t.n_steps??0} est=${t.estimated_step??'-'} stop=${t.stop_reason||'-'}`;
    return `<div class="seg ${segClass(t)}${i===S.ti?' cur':''}" data-i="${i}" title="${esc(tip)}"
              style="left:${L}%;width:${W}%">${esc(lab)}</div>`;
  }).join('');
  $('#track').innerHTML=`<div class=lane><div class=lab>turns</div><div class=bar>${segs}</div></div>`;
  document.querySelectorAll('#track .seg').forEach(s=>s.onclick=()=>selectTurn(+s.dataset.i));
}
function setNav(){
  $('#prev').disabled=S.ti<=0;
  $('#next').disabled=S.ti>=S.turns.length-1;
  const t=S.turns[S.ti];
  $('#pos').textContent=`${S.ti+1}/${S.turns.length}  `
    +(t.kind==='exec'?'turn '+t.turn:t.kind);
  document.querySelectorAll('#track .seg').forEach(s=>s.classList.toggle('cur',+s.dataset.i===S.ti));
}

// ---- MAINTAINED TASK PLAN ----
// The live two-level checklist AS OF the selected turn: System2 returns only the changed milestone
// block each turn (<plan_update>), which combined_eval merges into the running plan and saves as
// plan_after. We show that merged state, marking lines the CURRENT turn changed so you can watch
// the plan evolve turn by turn. Marks: [ ] todo, [~] doing, [x] done.
function planAsOf(i){
  const t=S.turns[i];
  if(!t)return '';
  if(t.kind==='memory')return '';   // the memory pass runs BEFORE any plan exists
  if(t.kind==='plan')return (S.doc.plan||{}).plan||'';
  return t.plan_after||planBefore(t);
}
function planCard(){
  const cur=planAsOf(S.ti), before=S.ti>0?planAsOf(S.ti-1):'';
  if(!cur)return '';
  const changed=new Set(cur.split('\\n').filter(l=>l.trim()&&!before.split('\\n').includes(l)));
  const rows=cur.split('\\n').filter(l=>l.trim()).map(l=>{
    const m=l.match(/\\[(.)\\]/), mark=m?m[1]:' ';
    const isChild=/^\\s*\\*/.test(l);
    const cls=mark==='x'?'pl-done':(mark==='~'?'pl-doing':'pl-todo');
    const ch=changed.has(l)?' pl-new':'';
    return `<div class="pl ${cls}${ch}${isChild?' pl-child':''}">${esc(l.trim())}</div>`;
  }).join('');
  const t=S.turns[S.ti];
  const label=t.kind==='plan'?'initial plan':`merged through turn ${t.turn}`;
  return `<div class=card><h3>Maintained task plan<span>${label}</span></h3>
    <div class=muted style="margin-bottom:3px">aggregated from every System2 &lt;plan_update&gt;;
      outlined = changed this turn. Scroll for full history.</div>
    <div class=plangrid>${rows}</div></div>`;
}

// ---- MEMORY step (the -memory plan variant only): the chunk-by-chunk narration pass that
// produced the recipe. Its own timeline step, BEFORE plan -- this is the stage that distinguishes
// narrate->summarize->plan->execute from plain plan->execute. Absent for cold runs. ----
async function renderMemory(){
  const d=S.doc;
  const j=await (await fetch(`/api/combine/turn/${S.method}/${S.ep}/plan`)).json();
  // memory.json = the per-chunk index (+ aggregate timings); recipe.json = the aggregation call's
  // own prompt/response. Merged field by field, NOT Object.assign: both carry `latency_s`, as a
  // {narrate_total, recipe} dict in the former and a bare float in the latter.
  const mem=j.memory||{}, agg=j.recipe||{}, P=j.plan||{};
  const M=f=>`/api/combine/media/${S.method}/${S.ep}/plan/${f}`;
  const pm=(d.plan||{}).memory||{};
  const srcEp=pm.source_episode, priv=pm.privileged;
  // `chunks` is the per-chunk index. Runs made before it was added carry only the flat narration
  // list, so fall back to that (windows/clips unknown -> the row shows the sentence alone).
  const chunks=mem.chunks||(mem.narrations||pm.narrations||[]).map((t,i)=>
    ({chunk:i, window:null, n_clip_frames:null, narration:t, latency_s:null}));
  const recipe=mem.recipe||pm.recipe||'';

  // Each chunk row = the 4s clip System2 watched + the one sentence it produced. clip.mp4 is the
  // display copy at 128x384; the model read clip_full.mp4 at the trained 256x768.
  const rows=chunks.map(c=>{
    const cd=`memory/chunk${String(c.chunk).padStart(2,'0')}`;
    return `<div class=memrow>
      <video src="${M(cd+'/clip.mp4')}" muted loop autoplay playsinline controls></video>
      <div>
        <div class=muted>clip ${c.chunk+1}/${chunks.length}${c.window?` &middot; source frames ${c.window[0]}&ndash;${c.window[1]}`:''}
          ${c.n_clip_frames?` &middot; ${c.n_clip_frames} frames @4fps`:''}
          ${c.latency_s!=null?` &middot; vLLM ${nn(c.latency_s)}s`:''}</div>
        <div class=narr>${esc(c.narration)}</div>
      </div></div>`;
  }).join('');
  const skipped=(mem.skipped||[]).length
    ? `<div class=muted>skipped ${mem.n_chunks_skipped} window(s): `
      +esc((mem.skipped||[]).map(s=>`[${s.window}] ${s.reason}`).join('; '))+'</div>' : '';

  // Prompts for the FIRST and LAST narration turn: the running "So far:" list is what makes this a
  // sequential pass rather than N independent captions, and it is only visible by comparing them.
  const first=chunks.length?await (await fetch(
    `/api/combine/media/${S.method}/${S.ep}/plan/memory/chunk00/narration.json`)).json():null;
  const lastI=chunks.length?chunks[chunks.length-1].chunk:null;
  const last=chunks.length>1?await (await fetch(
    `/api/combine/media/${S.method}/${S.ep}/plan/memory/chunk${String(lastI).padStart(2,'0')}/narration.json`)).json():null;

  $('#body').classList.add('single'); $('#right').innerHTML='';
  $('#left').innerHTML=
   `<div class=card><h3>Memory pass &mdash; narrate &rarr; summarize
      <span>step 1 of 3 &middot; System2 only, no robot yet</span></h3>
      <div class=muted style="margin-bottom:5px">This run is the <b>-memory</b> variant: before
      planning, System2 watched a video <b>clip by clip</b> and narrated each one, then turned the
      whole narration into a reusable recipe. Fixed 4-second windows from frame 0
      (${mem.chunk_source_frames||80} source frames each) over ${nn(mem.n_source_frames)} recorded
      frames &mdash; the <code>summary_v2</code> training layout, NOT annotated spans. Each turn is
      conditioned on the narrations before it (&ldquo;So far: 1. &hellip; 2. &hellip;&rdquo;), so
      this is a sequential read of the video, not ${chunks.length} independent captions.</div>
      <div class=kv><span class=k>goal</span> <b>${esc(d.instruction)}</b></div>
      <div class=kv><span class=k>video watched</span> recorded demo of
        <b>episode ${nn(srcEp)}</b> (3 cams tiled 256&times;768)</div>
      ${priv?`<div class=kv style="color:#a33"><span class=k>privileged</span>
        <b>this memory is the demo of the VERY episode under eval &mdash; oracle input, an upper
        bound, not a transfer number</b></div>`:
       `<div class=kv><span class=k>held-out</span> memory came from a different episode
         (${nn(srcEp)}) than the one under eval &mdash; no ground truth for this scene</div>`}
      ${skipped}
    </div>
    <div class=card><h3>Per-clip narrations<span>${chunks.length} clips &rarr; ${chunks.length} &times; &lt;narration&gt;</span></h3>
      ${rows||'<span class=muted>no narrations recorded</span>'}
      ${first?`<details><summary>System2 prompt &mdash; first clip (no history yet)</summary>
        <pre>${esc(first.s2_system_prompt)}</pre><pre>${esc(first.s2_user_prompt)}</pre></details>`:''}
      ${last?`<details><summary>System2 prompt &mdash; last clip (carries the running narration)</summary>
        <pre>${esc(last.s2_user_prompt)}</pre></details>`:''}
    </div>
    <div class=card><h3>Aggregation &rarr; recipe<span>1 text-only call &middot; vLLM ${nn((mem.latency_s||{}).recipe)}s</span></h3>
      <div class=muted style="margin-bottom:4px">The narrations are handed back as a numbered list
      with <b>no video and no image</b>; System2 compresses them into a general recipe for tasks of
      this KIND (the <code>summary_v2 &middot; recipe</code> mode). This string is what conditions
      the plan in the next step.</div>
      <div class=k>&lt;summary&gt; &mdash; the recipe handed to the planner</div>
      <pre class=recipe>${esc(recipe)}</pre>
      ${agg.s2_user_prompt?`<details><summary>aggregation prompt (text only &mdash; no media)</summary>
        <pre>${esc(agg.s2_system_prompt)}</pre><pre>${esc(agg.s2_user_prompt)}</pre></details>`:''}
      ${agg.s2_response_raw?`<details><summary>raw response</summary>
        <pre>${esc(agg.s2_response_raw)}</pre></details>`:''}
      <div class=tcost>time cost &mdash; ${chunks.length} narration calls
        ${nn((mem.latency_s||{}).narrate_total)}s + aggregation ${nn((mem.latency_s||{}).recipe)}s
        ${(pm.seconds||{}).memory!=null?` &middot; memory pass total ${nn(pm.seconds.memory)}s
        (incl. video decode + clip encode)`:''}</div>
    </div>`;
}

// ---- PLAN turn (turn 0): System2 only, single wide column. ----
async function renderPlan(){
  const d=S.doc;
  const j=await (await fetch(`/api/combine/turn/${S.method}/${S.ep}/plan`)).json();
  const P=j.plan||{};
  const M=f=>`/api/combine/media/${S.method}/${S.ep}/plan/${f}`;
  $('#body').classList.add('single'); $('#right').innerHTML='';
  const warm=(P.mode==='plan_with_memory');
  const recipe=(P.memory||{}).recipe;
  $('#left').innerHTML=
   planCard()+
   `<div class=card><h3>Plan mode &mdash; System2 input<span>${warm?'step 2 of 3 &middot; warm / with memory':'cold / no memory'}</span></h3>
      <div class=kv><span class=k>goal</span> <b>${esc(d.instruction)}</b></div>
      ${warm&&recipe?`<div class=kv><span class=k>recalled recipe</span> from the memory pass
        (previous step) &mdash; quoted verbatim inside the user prompt below</div>
        <pre class=recipe>${esc(recipe)}</pre>`:''}
      <div class=kv><span class=k>opening scene (3 cams tiled 256&times;768)</span></div>
      <img class=tile src="${M('scene.png')}">
      <details><summary>system prompt</summary><pre>${esc(P.s2_system_prompt)}</pre></details>
      <details open><summary>user prompt</summary><pre>${esc(P.s2_user_prompt)}</pre></details>
    </div>
    <div class=card><h3>Plan mode &mdash; System2 output<span>vLLM ${P.latency_s ?? '-'}s</span></h3>
      <div class=k>&lt;thought&gt;</div><pre>${esc(P.thought)}</pre>
      <div class=k>&lt;plan&gt; (milestone checklist handed to execution mode)</div>
      <pre>${esc(P.plan)}</pre>
      <details><summary>raw response</summary><pre>${esc(P.s2_response_raw)}</pre></details>
      <div class=tcost>time cost &mdash; vLLM request ${nn(P.latency_s)}s</div>
    </div>`;
}


// ---- EXECUTION turn: System2 left, System1 right. Terminal turns have no System1. ----
async function renderExec(t){
  const dir=`turn${String(t.turn).padStart(2,'0')}`;
  const j=await (await fetch(`/api/combine/turn/${S.method}/${S.ep}/${dir}`)).json();
  const T=j.turn||{}, s2=T.s2||{}, s1=T.s1;
  const M=f=>`/api/combine/media/${S.method}/${S.ep}/${dir}/${f}`;
  const priv=s2.privileged||{};
  const cm=s2.media||{};
  // System2 watches the clip PRODUCED BY THE PREVIOUS turn, so serve it from that turn's dir.
  const clipFrom=(cm.clip_from_turn!=null)?`turn${String(cm.clip_from_turn).padStart(2,'0')}`:null;
  const clipUrl=clipFrom?`/api/combine/media/${S.method}/${S.ep}/${clipFrom}/s2_input_clip.mp4`:null;
  let inMedia;
  if(t.turn===0) inMedia=`<div class=k>opening scene (no step has run yet)</div><img class=tile src="${M('s2_input_scene.png')}">`;
  else if(clipUrl){
    const ci=cm.video||{};
    inMedia=`<div class=k>clip of the segment System1 just ran &mdash; from ${clipFrom}
      (${ci.n_frames??'?'} frames @${ci.fps??4}fps${s2.nframes_requested!=null?`, ${s2.nframes_requested} sampled by the model`:''})</div>
      <video controls loop src="${clipUrl}"></video>`;
  }
  else inMedia='<span class=muted>no input media recorded</span>';

  $('#body').classList.toggle('single', !s1);
  $('#left').innerHTML=
   planCard()+
   `<div class=card><h3>System2 input &mdash; execution<span>turn ${t.turn}</span></h3>
      <div class=kv><span class=k>goal</span> ${esc(S.doc.instruction)}</div>
      <div class=kv><span class=k>privileged from env</span>
        <span class=pill>task status: ${esc(priv.task_status)}</span>
        <span class=pill>gripper: ${esc(priv.gripper_status)}</span></div>
      ${inMedia}
      <div class=kv style="margin-top:6px"><span class=k>plan handed in</span>
        ${t.turn===0&&(S.doc.plan||{}).plan_after_rules
          ?'<span class=pill title="a plan-mode rule replaced the System2 checklist">rule-forced</span>':''}</div>
      <pre>${esc(planBefore(t, s2))}</pre>
      ${t.turn===0&&(S.doc.plan||{}).s2_plan_before_rules
        ?`<details><summary>what System2 actually proposed (overridden)</summary>
           <pre>${esc((S.doc.plan||{}).s2_plan_before_rules)}</pre></details>`:''}
      <details><summary>system prompt</summary><pre>${esc(s2.system_prompt)}</pre></details>
      <details><summary>full user prompt</summary><pre>${esc(s2.user_prompt)}</pre></details>
    </div>
    <div class=card><h3>System2 output<span>vLLM ${s2.latency_s ?? '-'}s</span></h3>
      <div>${jtag(s2.judge)}<span class=k>estimated_step</span> <b>${nn(s2.estimated_step)}</b></div>
      ${s2.subgoal?`<div class=sg>&rarr; ${esc(s2.subgoal)}</div>
        <div class=muted>${esc(s2.subgoal_detail)}</div>`
        :'<div class=muted style="margin:4px 0">no subgoal issued (terminal turn)</div>'}
      <div class=k>&lt;thought&gt;</div><pre>${esc(s2.thought)}</pre>
      <div class=k>&lt;plan_update&gt;</div><pre>${esc(s2.plan_update)}</pre>
      <details><summary>plan after this turn</summary><pre>${esc(T.plan_after)}</pre></details>
      <details><summary>raw response</summary><pre>${esc(s2.response_raw)}</pre></details>
      <div class=tcost>time cost &mdash; request ${nn(s2.latency_s)}s
        &middot; +media prep ${nn(s2.t_total_s)}s
        &middot; ${s2.nframes_requested!=null?s2.nframes_requested+' frames sampled':'image input'}</div>
    </div>`;

  if(!s1){
    $('#right').innerHTML='';
    return;
  }
  const st=s1.clip_stats||{};
  // The condensed frames are no longer dumped as PNGs (they were a ~2 MB/episode duplicate of the
  // clip mp4). The strip is gone; the clip video above IS the frame-by-frame record.
  const frames='';
  let pf='';
  if(st.per_frame&&st.per_frame.length){
    pf=`<details><summary>per-frame condensing audit (${st.per_frame.length} steps &rarr; ${st.n_final} clip frames, eps=${st.eps??'n/a'})</summary>`+
       `<div class=scroll><table><tr><th>step</th><th>motion</th><th>&ge;eps</th><th>in clip</th></tr>`+
       st.per_frame.map(r=>`<tr class="${r.in_clip?'keep':(r.moving?'':'drop')}"><td>${r.i}</td><td>${r.motion}</td><td>${r.moving?'yes':'no'}</td><td>${r.in_clip?'KEPT':''}</td></tr>`).join('')+
       `</table></div></details>`;
  }
  const x=T.timings||{};
  const steps=(j.s1_steps?.steps||[]);
  window._steps=steps;
  S._s1=s1; S._turn=T;      // the stop-reason panel reads these
  // Build the plot series once per turn (progress + the three action-delta channels), mirroring
  // /episode: the video, slider, curves, prompt and action-chunk pointer all key off ONE step index.
  S.series={
    // Progress is only returned on REPLAN steps (the policy is queried once per chunk), so hold the
    // last reading forward — otherwise the curve is a row of isolated dots and the readout reads n/a
    // on the steps in between. Held values are the model's most recent estimate, not interpolation.
    // PER-STEP progress. For progact ('action' head) the policy predicts progress for EVERY step of
    // the chunk, so index chunk[offset] where offset = steps consumed since the replan — the same
    // value the stop rule now uses. That makes the curve genuinely continuous instead of a staircase
    // of chunk[0] readings. progreg ('continuous') returns one scalar per replan, so it is held
    // forward; progIsFresh marks which points are model readings vs held.
    prog: (()=>{
      let chunk=null, off=0, last=null;
      return steps.map(v=>{
        const q=v.query, p=v.progress_raw||{};
        if(v.replanned&&q&&q.chunk_progress&&q.chunk_progress.length){chunk=q.chunk_progress;off=0;}
        else if(v.replanned){chunk=null;off=0;}
        else off++;
        if(chunk&&off<chunk.length){last=+chunk[off];}
        else if(p.progress_now!=null){last=+p.progress_now;}
        return last;});})(),
    // A point is "fresh" when it comes from a per-step chunk entry or a replan scalar (progreg).
    progIsFresh: (()=>{
      let chunk=null, off=0;
      return steps.map(v=>{
        const q=v.query;
        if(v.replanned&&q&&q.chunk_progress&&q.chunk_progress.length){chunk=q.chunk_progress;off=0;return true;}
        if(v.replanned){chunk=null;off=0;return (v.progress_raw||{}).progress_now!=null;}
        off++;
        return !!(chunk&&off<chunk.length);});})(),
    motion: steps.map(v=>v.motion_norm??0),
    // Gripper STATE delta (finger pad distance, m) -- its own series, NOT folded into |da|, which
    // is eef_pos+eef_rot+base only. A grasp shows up here as a ~0.005 spike settling to ~0.0006.
    gripd: steps.map(v=>v.grip_width_delta??0),
    gripw: steps.map(v=>v.grip_width??null),
    eef_pos: steps.map(v=>v.action_eef_pos_norm??0),
    eef_rot: steps.map(v=>v.action_eef_rot_norm??0),
    base: steps.map(v=>v.action_base_norm??0),
    replan: steps.map(v=>!!v.replanned),
  };
  S.fps=(s1.video_raw&&s1.video_raw.fps)||20;
  $('#right').innerHTML=
   `<div class=card><h3>System1 input<span id=promptpos>${esc(s1.stop_reason)}</span>${ruleBadge(T)}</h3>
      ${ruleWhy(T)}
      <div class=sg>${esc(s1.prompt_text)}</div>
      <div class=kv>
        <span class=pill>est_length ${s1.est_length}</span>
        <span class=pill>budget ${s1.budget}</span>
        <span class=pill>steps ${s1.n_steps}</span>
        <span class=pill>progress_done ${s1.progress_done}</span>
        <span class=pill>quiescent ${s1.quiescent}</span></div>
      <div id=promptbody></div>
    </div>
    <div class=card><h3>System1 output &mdash; play the rollout<span id=scrubpos></span></h3>
      <div class=mini>
        <div class=minicol>
          <div class=k>raw rollout, every executed step
            (${(s1.video_raw&&s1.video_raw.n_frames)??s1.n_steps} frames @${S.fps}fps)</div>
          <video id=vid class=full preload=metadata src="${M('s1_rollout_raw.mp4')}"></video>
          <div class=player>
            <button id=play>&#9654; play</button>
            <button id=bb>&lsaquo;</button>
            <input type=range id=srange min=0 max="${Math.max(0,steps.length-1)}" value=0>
            <button id=ff>&rsaquo;</button>
            <span class=cnt id=scnt></span>
          </div>
          <div id=scrubbody></div>
          <div class=fld><div class=cap><span>progress</span><span class=lg id=proglab></span></div>
            <canvas class=curve id=curveP></canvas></div>
          <div class=fld><div class=cap><span>action &Delta; magnitude &mdash; eef + base only</span>
            <span class=lg><i style="background:#e5484d"></i>eef_pos<i style="background:#f59e0b"></i>eef_rot<i style="background:#16a34a"></i>base<i style="background:#6b7280"></i>|&Delta;a|</span></div>
            <canvas class=curve id=curveA></canvas></div>
          <div class=fld><div class=cap><span>gripper state &Delta; (pad distance)</span>
            <span class=lg id=griplab></span></div>
            <canvas class=curve id=curveG></canvas></div>
          <div class=muted>dashed grey = replans &middot; <span style="color:#2b8a3e">green</span> = step where
        _check_success() fired (checked every step) &middot; <span style="color:#b0431c">orange</span> = current step</div>
        </div>
        <div class=minicol>
          <div id=chunkbody></div>
        </div>
      </div>
      <div class=tcost>time cost &mdash; policy ${nn(x.s1_infer_total_s)}s over ${x.s1_infer_calls??0} calls
        (mean ${nn(x.s1_infer_mean_s)}s) &middot; env step ${nn(x.env_step_total_s)}s
        &middot; render ${nn(x.env_render_total_s)}s &middot; obs build ${nn(x.obs_build_total_s)}s
        &middot; video encode ${nn(x.video_encode_s)}s</div>
    </div>

    <div class=card><h3>Segment endpoints &mdash; anchor in, condensed clip out</h3>
      <div class=mini>
        <div class=minicol>
          <div class=k>ANCHOR &mdash; first frame of this segment; the subgoal-start views
            System1 is conditioned on</div>
          <img class="tile full" src="${M('s1_anchor.png')}">
        </div>
        <div class=minicol>
          <div class=k>CONDENSED CLIP &mdash; what System2 receives on the NEXT turn
            (${(s1.video_clip&&s1.video_clip.n_frames)??st.n_final} frames @${(s1.video_clip&&s1.video_clip.fps)??4}fps)${st.wait_subgoal?' &middot; WAIT subgoal: static frames KEPT':''}</div>
          <video class=full controls loop src="${M('s2_input_clip.mp4')}"></video>
          <div class=muted style="margin-top:3px">mode=${esc(st.mode)} · raw=${st.n_raw} · moving=${nn(st.n_moving)}
            · dropped=${st.frac_static_dropped!=null?(100*st.frac_static_dropped).toFixed(1)+'%':'-'} · stride=${st.stride}</div>
        </div>
      </div>
      ${frames}${pf}
    </div>
    <div class=card><h3>Per-step trace<span>${steps.length} steps</span></h3>
      <div class=scroll><table><tr><th>step</th><th>progress</th><th>|&Delta;a|</th><th>grip</th><th>grip_w</th><th>eef_pos</th><th>eef_rot</th><th>base</th><th>rp</th><th>S1 s</th><th>step s</th></tr>`+
      steps.map((v,i)=>`<tr class=trow data-i="${i}" style="cursor:pointer"><td>${v.frame_step}</td><td>${esc(v.progress)}</td><td>${v.motion_norm}</td><td>${esc(v.gripper_flag)}</td><td>${v.grip_width==null?'':v.grip_width.toFixed(4)}</td><td>${v.action_eef_pos_norm}</td><td>${v.action_eef_rot_norm}</td><td>${v.action_base_norm}</td><td>${v.replanned?'*':''}</td><td>${(v.query&&v.query.s1_infer_s)??''}</td><td>${v.t_env_step_s??''}</td></tr>`).join('')+
      `</table></div>
    </div>
    `;
  wireVideo();
}
// ---- STEP SCRUBBER ----
// Drag through a segment's executed steps. System1 re-plans every `replan_steps`, so BOTH the
// language prompt (its state ints + Executed Step change per replan) and the predicted action
// chunk are rolling quantities. For any step we find the replan that produced it, show that exact
// prompt, and point at WHICH action inside the chunk is being executed at this step.
function renderScrub(i){
  const steps=window._steps||[];
  if(!steps.length){$('#scrubbody').innerHTML='<span class=muted>no steps</span>';return;}
  i=Math.max(0,Math.min(i,steps.length-1));
  window._si=i;
  const st=steps[i];
  // the most recent replan at or before this step = the chunk currently being consumed
  let ri=i; while(ri>0 && !steps[ri].replanned) ri--;
  const rq=(steps[ri]||{}).query||{};
  const off=i-ri;                                  // index inside that chunk
  const ch=rq.chunk_raw12||[], mo=rq.chunk_motion||[], ex=rq.replan_steps||0;
  const rng=$('#srange'); if(rng&&+rng.value!==i)rng.value=i;
  const sp=document.getElementById('scrubpos');
  if(sp)sp.textContent=`step ${st.frame_step} / ${steps.length-1}`;
  const sc=document.getElementById('scnt');
  if(sc)sc.textContent=`${i+1}/${steps.length} · replan@${steps[ri].frame_step} +${off}`;
  drawCurves(i);

  const prog=st.progress_raw||{};
  const head=`<div class=kv>
      <span class=pill>progress ${(()=>{
        // The policy only returns progress on REPLAN steps, so st.progress is the literal "-" in
        // between. Use the hold-forward series (same values the curve draws) and mark held readings.
        // Read the SERIES (per-step for progact, held-forward for progreg) so the pill always
        // agrees with the curve and the stop rule.
        const v=(S.series&&S.series.prog)?S.series.prog[i]:null;
        const fresh=!!(S.series&&S.series.progIsFresh&&S.series.progIsFresh[i]);
        return v==null?'n/a':v.toFixed(4)+(fresh?'':' (held)');
      })()}</span>
      <span class=pill>|&Delta;a| ${st.motion_norm}</span>
      <span class=pill>grip ${esc(st.gripper_flag)}</span>
      ${st.grip_width==null?''
        // Finger-pad distance |q[14]-q[15]| in metres, and its per-step delta. The FLAG above is only
        // Open/Close; the width is what distinguishes closed ON an object (~0.062) from closed on
        // NOTHING (~0.001) -- i.e. a missed grasp -- against ~0.0799 fully open. The delta is shown
        // because a grasp appears as a ~0.005 spike settling to ~0.0006, which is how you see the
        // fingers still closing while a segment is being cut short.
        :`<span class=pill title="finger-pad distance: ~0.0799 open, ~0.062 closed on an object, ~0.001 closed on nothing">grip_w ${st.grip_width.toFixed(4)}</span>`
         +(st.grip_width_delta==null?''
           :`<span class=pill title="per-step change in finger-pad distance">&Delta;grip ${st.grip_width_delta.toFixed(4)}</span>`)}
      <span class=pill>eef_pos ${st.action_eef_pos_norm}</span>
      <span class=pill>eef_rot ${st.action_eef_rot_norm}</span>
      <span class=pill>base ${st.action_base_norm}</span>
      ${st.replanned?'<span class=pill style="background:#ffe9d6;border-color:#b0431c">REPLAN</span>':''}
    </div>`;
  const prompt=rq.prompt?`<div class=k>language prompt in force at this step (from the replan at step ${steps[ri].frame_step})</div>
      <pre>${esc(rq.prompt)}</pre>`
    :'<div class=muted>no prompt recorded for this replan</div>';
  // WHY this segment ended — only meaningful on the LAST step, so it is shown there (and greyed
  // out earlier). Three outcomes, matching what the loop records:
  //   stop_rule   -> progress >= thresh AND the arm stopped moving (quiescence) -> subgoal done
  //   budget      -> timeout: the step budget (estimated_step * horizon_mult, capped) ran out
  //   env_success -> the simulator's own _check_success() went true, so the TASK is finished
  const atEnd=(i===steps.length-1);
  const sr=S._s1||{};
  const why=(()=>{
    // _check_success() is evaluated EVERY step and breaks immediately (RoboCasa benchmark
    // semantics), so stop_reason==='env_success' means the task was solved AT success_step.
    if(sr.stop_reason==='env_success')return ['task finished',
      `the simulator's _check_success() returned true at step ${sr.success_step} \u2014 the rollout stopped there`,'#d3f0d8','#5fa96b'];
    if(sr.stop_reason==='stop_rule')return ['progress reached &amp; arm stopped',
      `progress ${sr.progress_done?'\u2265 threshold':'below threshold'} AND commanded motion quiescent over the last steps \u2014 subgoal judged complete`,'#cfe0ff','#7fa8f0'];
    if(sr.stop_reason==='budget'){
      // The budget is min(max_steps_cap, est_length * horizon_mult). Saying it "= est_length x
      // horizon_mult" was WRONG whenever the CAP bound instead: a wait with est_length 600 showed
      // "400 = est_length 600 x horizon_mult", which does not multiply out and hid the real
      // limiter. Detect which term bound and say so.
      const prod=(sr.est_length!=null&&S.doc&&S.doc.config&&S.doc.config.horizon_mult!=null)
                  ? sr.est_length*S.doc.config.horizon_mult : null;
      const capped=(prod!=null&&sr.budget!=null&&sr.budget<prod);
      return ['timeout', capped
        ? `hit the step budget ${sr.budget} \u2014 this is the <b>--max-steps-cap</b>, not est_length \u00d7 horizon_mult `
          +`(that would be ${prod} = ${sr.est_length} \u00d7 ${S.doc.config.horizon_mult}). The segment was cut by the CAP, `
          +`so raising est_length alone will not lengthen it.`
        : `hit the step budget (${sr.budget}${prod!=null?` = est_length ${sr.est_length} \u00d7 ${S.doc.config.horizon_mult}`:''}) without the stop rule firing`,
        '#fff1cf','#dcae4a'];
    }
    return [esc(sr.stop_reason||'unknown'),'',' #eee','#bbb'];
  })();
  const stopPanel=`<div class=stopwhy style="opacity:${atEnd?1:.45};background:${why[2]};border-color:${why[3]}">
      <b>segment ended: ${why[0]}</b>${atEnd?'':' <span class=muted>(fires at the last step)</span>'}
      ${why[1]?`<div class=muted style="margin-top:2px">${why[1]}</div>`:''}
      <div class=muted style="margin-top:2px">stop_reason=<code>${esc(sr.stop_reason)}</code>
        &middot; progress_done=${sr.progress_done} &middot; quiescent=${sr.quiescent}
        &middot; steps ${sr.n_steps}/${sr.budget}</div>
    </div>`;
  const act=`<div class=k>executed action (robosuite 12-d)</div>
      <pre>${(st.action_raw12||[]).map(v=>(+v).toFixed(3)).join('  ')}</pre>`;
  let chunk='';
  if(ch.length){
    chunk=`<div class=k>predicted action chunk &mdash; replan@${steps[ri].frame_step}, ${ch.length} steps,
        first ${ex} executed (shaded); <span class=ptr>&#9654;</span> = running at this step (+${off})</div>
      <div class=scroll id=chunkscroll><table>
        <tr><th></th><th>i</th><th>|&Delta;a|</th><th>eef_pos xyz</th><th>eef_rot rpy</th><th>grip</th><th>base</th></tr>`+
      ch.map((a,k)=>`<tr class="chunkrow${k===off?' at':''}${k<ex?' exec':''}" ${k===off?'id=atrow':''}>
          <td>${k===off?'<span class=ptr>&#9654;</span>':''}</td><td>${k}</td><td>${mo[k]??''}</td>
          <td>${a.slice(0,3).map(v=>v.toFixed(3)).join(' ')}</td>
          <td>${a.slice(3,6).map(v=>v.toFixed(3)).join(' ')}</td>
          <td>${a[6].toFixed(2)}</td>
          <td>${a.slice(7,11).map(v=>v.toFixed(3)).join(' ')}</td></tr>`).join('')+
      `</table></div>`;
  } else {
    chunk='<div class=muted>action chunk not recorded in this run (re-run with the current combined_eval.py to capture it)</div>';
  }
  // Left minipage keeps the video+curves; the per-step readout and prompt go under them, while the
  // action chunk renders in the RIGHT minipage so the moving pointer stays visible during playback.
  // readout line lives under the video; the rolling language prompt lives in the System1 input card
  $('#scrubbody').innerHTML=head;
  const pb=document.getElementById('promptbody');
  if(pb)pb.innerHTML=prompt;
  const pp=document.getElementById('promptpos');
  if(pp)pp.textContent=`step ${st.frame_step} · replan@${steps[ri].frame_step}`;
  // executed action sits directly UNDER the chunk table: it's the row the pointer marks, so
  // reading "predicted chunk -> what actually got sent" top-to-bottom keeps them adjacent.
  const cb=document.getElementById('chunkbody');
  if(cb)cb.innerHTML=chunk+act+stopPanel;
  const at=document.getElementById('atrow');
  if(at&&at.scrollIntoView)at.scrollIntoView({block:'nearest'});
}
// ---- CURVES (same canvas approach as /episode) ----
// One canvas per field; a vertical cursor marks the current step and dashed lines mark replans.
function _curve(cvId, lines, ymax, cur, yTicks){
  const cv=document.getElementById(cvId); if(!cv||!S.series)return;
  const W=cv.clientWidth||600, H=cv.clientHeight||82; cv.width=W; cv.height=H;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const n=S.series.motion.length; if(n<2)return;
  const pad={l:34,r:8,t:6,b:12}, gw=W-pad.l-pad.r, gh=H-pad.t-pad.b;
  const X=i=>pad.l+gw*i/(n-1), Y=v=>pad.t+gh*(1-(v==null?0:v)/ymax);
  ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.fillStyle='#aaa';ctx.font='9px monospace';
  (yTicks||[0,ymax]).forEach(t=>{const y=Y(t);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.fillText(String(t),2,y+3);});
  ctx.strokeStyle='#d8d8e0';ctx.setLineDash([3,3]);
  S.series.replan.forEach((r,i)=>{if(!r||i===0)return;const xx=X(i);ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,H-pad.b);ctx.stroke();});
  ctx.setLineDash([]);
  lines.forEach(({arr,color,w})=>{ctx.strokeStyle=color;ctx.lineWidth=w||1.5;ctx.beginPath();let go=false;
    arr.forEach((v,i)=>{if(v==null)return;const xx=X(i),yy=Y(v);if(!go){ctx.moveTo(xx,yy);go=true;}else ctx.lineTo(xx,yy);});ctx.stroke();});
  // green marker at the step where _check_success() first fired (dense per-step benchmark check)
  const ss=(S._s1||{}).success_step;
  if(ss!=null&&ss<n){const xs=X(ss);ctx.strokeStyle='#2b8a3e';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(xs,pad.t);ctx.lineTo(xs,H-pad.b);ctx.stroke();}
  if(cur!=null){const xx=X(cur);ctx.strokeStyle='#b0431c';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,H-pad.b);ctx.stroke();}
}
function drawCurves(cur){
  if(!S.series)return;
  _curve('curveP',[{arr:S.series.prog,color:'#2d5bd7',w:1.8}],1.0,cur,[0,0.5,1]);
  const nm=Math.max(0.05,...S.series.eef_pos,...S.series.eef_rot,...S.series.base,...S.series.motion);
  _curve('curveA',[{arr:S.series.eef_pos,color:'#e5484d'},{arr:S.series.eef_rot,color:'#f59e0b'},
                   {arr:S.series.base,color:'#16a34a'},{arr:S.series.motion,color:'#6b7280',w:1}],
         nm,cur,[0,+(nm/2).toFixed(2),+nm.toFixed(2)]);
  // gripper STATE delta on its own axis; the dashed line is grip_eps (0.001), the settle bar.
  const gm=Math.max(0.002,...S.series.gripd);
  _curve('curveG',[{arr:S.series.gripd,color:'#8b5cf6',w:1.6}],gm,cur,[0,0.001,+gm.toFixed(4)]);
  const gl=document.getElementById('griplab');
  if(gl){const w=S.series.gripw[cur];
    gl.textContent=`width ${w!=null?w.toFixed(4):'n/a'} m · |Δ| ${(S.series.gripd[cur]??0).toFixed(5)} (settle < 0.001)`;}
  const p=S.series.prog[cur];
  const fresh=S.series.progIsFresh&&S.series.progIsFresh[cur];
  const pl=document.getElementById('proglab');
  if(pl)pl.textContent=(p!=null?p.toFixed(4):'n/a')+(fresh?' (fresh)':' (held from last replan)')+'  ·  stop ≥ 0.95';
}

// ---- VIDEO PLAYER (mirrors /episode: rVFC follow-loop, slider is the source of truth) ----
function wireVideo(){
  const v=$('#vid'), rng=$('#srange');
  if(!v||!rng)return;
  if(S._rvfc!=null){
    if('cancelVideoFrameCallback' in HTMLVideoElement.prototype){try{v.cancelVideoFrameCallback(S._rvfc);}catch(e){}}
    else{cancelAnimationFrame(S._rvfc);}
    S._rvfc=null;
  }
  const nSteps=()=>(window._steps||[]).length;
  const hasRVFC='requestVideoFrameCallback' in HTMLVideoElement.prototype;
  // Follow per PRESENTED frame so the panel + chunk pointer advance one step at a time.
  const follow=(now,meta)=>{
    if(v.paused){S._rvfc=null;return;}
    const t=(meta&&meta.mediaTime!=null)?meta.mediaTime:v.currentTime;
    const i=Math.min(nSteps()-1,Math.round(t*S.fps));
    if(i!==(+rng.value)){rng.value=i;renderScrub(i);}
    S._rvfc=v.requestVideoFrameCallback(follow);
  };
  const raf=()=>{if(v.paused){S._rvfc=null;return;}
    const i=Math.min(nSteps()-1,Math.round(v.currentTime*S.fps));
    if(i!==(+rng.value)){rng.value=i;renderScrub(i);}
    S._rvfc=requestAnimationFrame(raf);};
  v.addEventListener('play',()=>{if(S._rvfc)return;
    S._rvfc=hasRVFC?v.requestVideoFrameCallback(follow):requestAnimationFrame(raf);});
  // The card is rebuilt on every turn/episode change while the <video> can still emit one last
  // pause/ended, so #play may already be gone -- guard instead of throwing TypeError on null.
  const setPlayLabel=h=>{const b=$('#play'); if(b)b.innerHTML=h;};
  v.addEventListener('play', ()=>setPlayLabel('&#10074;&#10074; pause'));
  v.addEventListener('pause',()=>setPlayLabel('&#9654; play'));
  v.addEventListener('ended',()=>setPlayLabel('&#9654; play'));
  $('#play').onclick=()=>{if(v.paused)v.play().catch(()=>{});else v.pause();};
  rng.oninput=e=>{v.pause();gotoStep(+e.target.value);};
  $('#ff').onclick=()=>{v.pause();gotoStep((+rng.value)+1);};
  $('#bb').onclick=()=>{v.pause();gotoStep((+rng.value)-1);};
  document.querySelectorAll('#right .trow').forEach(r=>r.onclick=()=>{v.pause();gotoStep(+r.dataset.i);});
  renderScrub(0);
}
// Seek the video AND update panel+curves to step i (slider index is authoritative).
function gotoStep(i){
  const n=(window._steps||[]).length; if(!n)return;
  i=Math.max(0,Math.min(i,n-1));
  const rng=$('#srange'); if(rng)rng.value=i;
  const v=$('#vid'); if(v)v.currentTime=(i+0.5)/S.fps;   // land inside frame i, not on its edge
  renderScrub(i);
}


// The plan a turn was HANDED (i.e. the previous turn's plan_after, or the initial plan for turn 0).
// WHY the instruction System1 got differs from what System2 asked for. Collapsed by default -- the
// prompt itself is what you normally read -- and revealed by the badge in the card's top-right corner.
// The state is global and persisted, so paging through turns keeps it open once you have opened it.
// Every override is recorded per turn (rule / kind / detail / before / after) and ``detail`` carries
// the rule's own reasoning, so it is shown verbatim. A DECLINED rule is included: one that saw a
// problem and chose NOT to act (repeat_cap with no next step) is otherwise invisible.
window.SHOW_RULE_WHY = (localStorage.getItem('showRuleWhy')==='1');
function toggleRuleWhy(){
  window.SHOW_RULE_WHY=!window.SHOW_RULE_WHY;
  localStorage.setItem('showRuleWhy', window.SHOW_RULE_WHY?'1':'0');
  selectTurn(S.ti);          // re-render the current turn in place
}
function _ruleIvs(T){
  const R=T.rules||{}, C=S.ruleConfig;
  // New runs are governed by method/rule_config.json, not the historical task_rules bit. This is
  // what makes mandatory-only and general-only interventions visible. Old runs fall back to the
  // per-turn enabled field because they have no method-level configuration.
  if(C){
    if(!(C.mandatory_rules||C.general_rules||C.task_rules))return [];
  }else if(!R.enabled)return [];
  return (R.interventions||[]).filter(i=>i&&i.rule);
}
function configuredRuleTier(rule){
  const A=(S.ruleConfig||{}).active_rules||{};
  if((A.mandatory||[]).includes(rule))return 'mandatory';
  if((A.general||[]).includes(rule))return 'general';
  if((A.task||[]).includes(rule)||(A.task_plan||[]).includes(rule)
      ||(A.task_action||[]).includes(rule))return 'task';
  if(rule==='est_resolve')return 'resolver';
  return 'task';
}
function ruleBadge(T){
  const ivs=_ruleIvs(T); const R=T.rules||{};
  const eff=(R.effective||{}).subgoal, s2=(T.s2||{}).subgoal;
  const changed = eff!=null && s2!=null && eff!==s2;
  if(!ivs.length && !changed) return '';
  const col = changed?'#0a7d33':'#6b7280';
  const lbl = (changed?'overridden':'rule') + (ivs.length>1?(' x'+ivs.length):'');
  // Its OWN element, NOT inside #promptpos: the rollout scrubber does
  //     document.getElementById('promptpos').textContent = 'step N .. replan@M'
  // on every frame, which silently wiped the badge the moment the video rendered.
  return `<span class=pill id=rulebadge title="click to show the rules that applied"
    style="float:right;cursor:pointer;margin-right:8px;border-color:${col};color:${col}"
    onclick="toggleRuleWhy()">${lbl} ${window.SHOW_RULE_WHY?'&#9652;':'&#9662;'}</span>`;
}
function ruleWhy(T){
  if(!window.SHOW_RULE_WHY) return '';
  const ivs=_ruleIvs(T); const R=T.rules||{};
  const eff=(R.effective||{}).subgoal, s2=(T.s2||{}).subgoal;
  const changed = eff!=null && s2!=null && eff!==s2;
  if(!ivs.length && !changed) return '';
  const col=k=>/declin|exempt/.test(k||'')?'#6b7280':(/^tx_/.test(k||'')?'#b0431c':'#0a7d33');
  // Keep it to ONE short line per rule: the reason strings carry a long justification after ": " or
  // " -- ", which belongs on hover, not on screen.
  const brief=d=>{const t=String(d||'').split(' -- ')[0].split(': ')[0];
                  return t.length>84?t.slice(0,84)+'\u2026':t;};
  const rows=ivs.map(i=>`<div style="margin-top:2px;line-height:1.35">
      <span class=pill style="border-color:${col(i.kind)};color:${col(i.kind)}">${esc(i.kind)}</span>
      <span class=k>${esc(i.rule)}</span>
      <span class=muted style="font-size:11px">${configuredRuleTier(i.rule)}</span>
      <span title="${esc(String(i.detail||''))}">&mdash; ${esc(brief(i.detail))}</span></div>`).join('');
  return `<div class=kv style="display:block;background:#faf7f2;border:1px solid #e6ddd0;
      border-radius:6px;padding:5px 8px;margin-bottom:6px;font-size:12px">
    ${changed?`<div class=muted>System2 asked: <b>${esc(s2)}</b></div>`:''}
    ${R.tx_label?`<div class=muted>rule-injected turn &middot; ${esc(R.tx_label)}</div>`:''}
    ${rows}</div>`;
}

function planBefore(t, s2){
  // A PLAN-MODE rule can replace System2's checklist before the exec loop starts, so on turn 0 the
  // plan HANDED IN is the forced one -- doc.plan.plan stays the faithful record of what System2
  // itself proposed, and showing that here misreported what System1/System2 actually received.
  // Deriving this from the PREVIOUS turn's plan_after is wrong whenever a rule revised the checklist
  // and re-queried System2 WITHIN this turn -- repeat_cap's milestone close and coffee_skip_failed both
  // do. The call that produced this turn's response saw the REVISED plan. Observed on
  // ArrangeBreadBasket ep0 t4: the panel showed "M1 [~] / M1.2 [~]" while the prompt actually sent
  // said "M1 [x] / M1.2 [x]".
  //
  // Three sources, best first:
  //   1. s2.plan_in -- recorded explicitly by combined_eval (new runs).
  //   2. the checklist parsed out of s2.user_prompt -- the prompt IS the ground truth of what System2
  //      received, and it is present in every run ever recorded, so this repairs old results too.
  //   3. the turn-1 derivation, for a turn whose prompt was not captured.
  if(s2&&s2.plan_in)return s2.plan_in;
  const up=(s2&&s2.user_prompt)||'';
  // COMBINE_HTML is a NON-RAW triple-quoted string, so every backslash escape here must be DOUBLED.
  // A single backslash-n is consumed by Python and reaches the browser as a real newline, which
  // splits this regex literal across lines and kills the whole script with "Invalid regular
  // expression: missing /" -- i.e. a blank page. Backslash-s survives only because Python leaves
  // unknown escapes alone. This applies to comments too: a broken comment line becomes bare code.
  const m=up.match(/Here's where the plan stands:\\s*\\n([\\s\\S]*?)(?:\\n\\s*\\n|$)/);
  if(m&&m[1].trim())return m[1].replace(/\s+$/,'');
  if(t.turn===0){const p=S.doc.plan||{};return p.plan_after_rules||p.plan||'';}
  const prev=(S.doc.turns||[]).find(x=>x.turn===t.turn-1);
  return prev?(prev.plan_after||''):'';
}

async function selectTurn(i){
  S.ti=Math.max(0,Math.min(i,S.turns.length-1));
  setNav();
  const t=S.turns[S.ti];
  $('#left').innerHTML='<div class=card><span class=muted>loading…</span></div>';
  $('#right').innerHTML='';
  if(t.kind==='memory')await renderMemory();
  else if(t.kind==='plan')await renderPlan();
  else await renderExec(t);
  $('#left').scrollTop=0; $('#right').scrollTop=0;
}

$('#method').onchange=async e=>{S.method=e.target.value;
  S.ruleConfig=(window._combineRuleConfigs||{})[S.method]||null; await loadEpisodes();};
$('#episode').onchange=e=>selectEpisode(+e.target.value);
$('#tasktype').onchange=()=>fillTasks();
$('#task').onchange=()=>fillEpisodes();
$('#prev').onclick=()=>selectTurn(S.ti-1);
$('#next').onclick=()=>selectTurn(S.ti+1);
// Arrow keys / j,k walk the turns (ignored while typing in a control).
window.addEventListener('keydown',e=>{
  if(/^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName))return;
  if(e.key==='ArrowRight'||e.key==='j'){selectTurn(S.ti+1);e.preventDefault();}
  if(e.key==='ArrowLeft' ||e.key==='k'){selectTurn(S.ti-1);e.preventDefault();}
});
// keep the canvases crisp when the window resizes
window.addEventListener('resize',()=>{const r=document.getElementById('srange');if(S.series&&r)drawCurves(+r.value||0);});
loadMethods();
</script>
"""


# A blank page is the failure mode of a JS-heavy page: an exception during init leaves nothing
# rendered, which looks exactly like "no data". There is no JS engine on this box to reproduce it
# offline, so the page reports its own errors -- visibly in a banner AND back to this server, where
# they land in the GUI log next to the request that served the page.
_ERR_JS = """<script>
(function(){
  function post(o){try{fetch('/api/client_error',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(o)});}catch(e){}}
  function note(msg){
    var d=document.getElementById('__note');
    if(!d){d=document.createElement('div'); d.id='__note';
      d.style.cssText='position:fixed;right:8px;bottom:8px;z-index:9999;background:#f1f5f9;'
        +'color:#475569;border:1px solid #cbd5e1;border-radius:4px;padding:4px 8px;'
        +'font:11px/1.3 ui-monospace,monospace';
      document.body.appendChild(d);}
    d.textContent=msg;
    clearTimeout(window.__noteT);
    window.__noteT=setTimeout(function(){if(d&&d.parentNode)d.parentNode.removeChild(d);},4000);
  }
  function banner(msg){
    var d=document.getElementById('__err')||document.createElement('div');
    d.id='__err';
    d.style.cssText='background:#fee2e2;color:#7f1d1d;border:1px solid #fca5a5;padding:6px 9px;'
      +'font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;margin:0 0 6px';
    d.textContent='page script error — '+msg;
    if(document.body&&!document.getElementById('__err'))document.body.insertBefore(d,document.body.firstChild);
  }
  window.addEventListener('error',function(e){
    var m=(e.message||'error')+' @ '+(e.filename||'')+':'+(e.lineno||'?')+':'+(e.colno||'?');
    banner(m); post({page:location.pathname,msg:e.message,file:e.filename,line:e.lineno,
                     col:e.colno,stack:(e.error&&e.error.stack)||null});});
  // A DROPPED REQUEST IS NOT A SCRIPT BUG. /stats polls every 60s and the viewer fetches on every
  // click; over a tunnel (or while this server restarts) any of those can fail, and the page's
  // awaits are not individually guarded, so each one surfaces here. Shouting "page script error" at
  // that trained the reader to ignore the banner -- which is the one thing it must not do. Network
  // failures get a quiet, self-dismissing note and are NOT posted; everything else stays loud.
  var NETRE=/failed to fetch|networkerror|load failed|network request failed|aborted/i;
  window.addEventListener('unhandledrejection',function(e){
    var r=e.reason||{}; var msg=r.message||String(e.reason);
    if(NETRE.test(msg)){note('lost contact with the eval server — retrying'); return;}
    var m='unhandled promise rejection: '+msg;
    banner(m); post({page:location.pathname,msg:m,stack:r.stack||null});});
})();
</script>"""


@app.route("/api/client_error", methods=["POST"])
def api_client_error():
    from flask import request
    try:
        d = request.get_json(force=True, silent=True) or {}
    except Exception:  # noqa: BLE001
        d = {}
    print(f"CLIENT JS ERROR on {d.get('page')}: {d.get('msg')} "
          f"({d.get('file')}:{d.get('line')}:{d.get('col')})\n{d.get('stack') or ''}", flush=True)
    return jsonify({"ok": True})


@app.route("/combine")
def combine_page():
    return COMBINE_HTML.replace("<script>", _ERR_JS + "<script>", 1)


# The live evaluator is a separate blueprint/controller so the existing read-only rollout browser
# and batch output contract stay untouched. Simulator/model imports inside it are lazy: browsing
# /combine still works in a lightweight environment, while /human-interactive clearly reports that
# it needs the RoboCasa launcher if those dependencies are absent.
try:
    from human_interactive_gui import configure_human_interactive
    from human_interactive_gui import register_human_interactive
except ImportError:  # package-style import in tests
    from examples.robocasa.human_interactive_gui import configure_human_interactive
    from examples.robocasa.human_interactive_gui import register_human_interactive

register_human_interactive(app)


def main():
    global ROOT, ROOTS, VAL_MSE_DIR, COMBINE_ROOT
    p = argparse.ArgumentParser()
    # --finestep-root is the primary; --rollout-root kept as a back-compat alias for the same tree.
    p.add_argument("--finestep-root", "--rollout-root", dest="finestep_root", type=Path, default=None,
                   help="FINE-STEP rollout tree (subtask_eval.py output) -> /finestep")
    p.add_argument("--milestone-root", type=Path, default=None,
                   help="MILESTONE rollout tree (milestone_eval.py output) -> /milestone")
    p.add_argument("--episode-root", type=Path, default=None,
                   help="EPISODE rollout tree (episode_eval.py output) -> /episode")
    p.add_argument("--val-mse-dir", type=Path, default=_RESULTS / "valmse_results",
                   help="dir of scripts/eval_val_mse.py output JSONs (the /val_mse curves)")
    p.add_argument("--combine-root", type=Path, default=None,
                   help="COMBINED System2+System1 tree (combined_eval.py output) -> /combine")
    p.add_argument("--hitl", action=argparse.BooleanOptionalAction, default=True,
                   help="enable the live /human-interactive System2+System1 controller")
    p.add_argument("--hitl-data-root", type=Path,
                   default=_DATA / "robocasa_dataset" / "v1.0" / "target",
                   help="official target LeRobot root used by the interactive episode picker")
    p.add_argument("--hitl-results-root", type=Path,
                   default=_RESULTS / "human_interactive",
                   help="durable human-interactive sessions and branch artifacts")
    p.add_argument("--s1-host", "--hitl-s1-host", dest="hitl_s1_host",
                   default=os.environ.get("HITL_S1_HOST", "127.0.0.1"),
                   help="System1 websocket server host used by /human-interactive")
    p.add_argument("--s1-port", "--hitl-s1-port", dest="hitl_s1_port", type=int,
                   default=int(os.environ.get("HITL_S1_PORT", "8060")))
    p.add_argument("--hitl-s1-checkpoint", default=os.environ.get("HITL_S1_CHECKPOINT"),
                   help="served System1 checkpoint path, recorded as provenance")
    p.add_argument("--hitl-norm-stats", type=Path,
                   default=(Path(os.environ["HITL_NORM_STATS"])
                            if os.environ.get("HITL_NORM_STATS") else None))
    p.add_argument("--s2-host", "--hitl-s2-host", dest="hitl_s2_host",
                   default=os.environ.get("HITL_S2_HOST", "127.0.0.1"),
                   help="System2 OpenAI-compatible server host used by /human-interactive")
    p.add_argument("--s2-port", "--hitl-s2-port", dest="hitl_s2_port", type=int,
                   default=int(os.environ.get("HITL_S2_PORT", "8100")))
    p.add_argument("--s2-model", "--hitl-s2-model", dest="hitl_s2_model",
                   default=os.environ.get("HITL_S2_MODEL", "system2-full"))
    p.add_argument("--hitl-s2-checkpoint", default=os.environ.get("HITL_S2_CHECKPOINT"),
                   help="served System2 checkpoint path, recorded as provenance")
    p.add_argument("--hitl-s2-max-tokens", type=int, default=512)
    p.add_argument("--hitl-s2-file-uri", action="store_true")
    p.add_argument("--hitl-general-rules", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hitl-task-rules", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--hitl-last-milestone-retry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="retry the final milestone on a false finish (default: follow effective general rules)",
    )
    p.add_argument("--hitl-prompt-source", choices=("subgoal", "subgoal_detail"),
                   default="subgoal")
    p.add_argument("--hitl-horizon-mult", type=float, default=2.0)
    p.add_argument("--hitl-max-steps-cap", type=int, default=400)
    p.add_argument("--hitl-default-est-length", type=int, default=50)
    p.add_argument("--hitl-replan-steps", type=int, default=16)
    p.add_argument("--hitl-resize-size", type=int, default=224)
    p.add_argument("--hitl-stop-progress", type=float, default=0.95)
    p.add_argument("--hitl-stop-eps", type=float, default=0.03)
    p.add_argument("--hitl-stop-window", type=int, default=5)
    p.add_argument("--hitl-static-eps", type=float, default=0.003)
    p.add_argument("--hitl-zero-arm-in-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    args = p.parse_args()
    # Default every root under the shared results dir so `subtask_eval_gui.py --port N` just works.
    if args.finestep_root is None:
        args.finestep_root = _RESULTS / "finestep"
    if args.milestone_root is None:
        args.milestone_root = _RESULTS / "milestone"
    if args.episode_root is None:
        args.episode_root = _RESULTS / "episode"
    ROOT = args.finestep_root.resolve()          # legacy default root = finestep
    ROOTS = {"finestep": ROOT}
    if args.milestone_root is not None:
        ROOTS["milestone"] = args.milestone_root.resolve()
    if args.episode_root is not None:
        ROOTS["episode"] = args.episode_root.resolve()
    VAL_MSE_DIR = args.val_mse_dir.resolve()
    if args.combine_root is not None:
        COMBINE_ROOT = args.combine_root.resolve()
    hitl_cfg = None
    if args.hitl:
        # The task tier includes the general tier, matching combined_eval.run_sweep.
        hitl_general_rules = bool(args.hitl_general_rules or args.hitl_task_rules)
        hitl_last_milestone_retry = (
            hitl_general_rules
            if args.hitl_last_milestone_retry is None
            else bool(args.hitl_last_milestone_retry)
        )
        hitl_cfg = {
            "dataset_root": args.hitl_data_root.resolve(),
            "results_root": args.hitl_results_root.resolve(),
            "s1_host": args.hitl_s1_host, "s1_port": args.hitl_s1_port,
            "s1_checkpoint": args.hitl_s1_checkpoint, "norm_stats": args.hitl_norm_stats,
            "s2_host": args.hitl_s2_host, "s2_port": args.hitl_s2_port,
            "s2_model": args.hitl_s2_model, "s2_checkpoint": args.hitl_s2_checkpoint,
            "s2_max_tokens": args.hitl_s2_max_tokens,
            "s2_file_uri": args.hitl_s2_file_uri,
            "general_rules": hitl_general_rules,
            "task_rules": bool(args.hitl_task_rules),
            "last_milestone_retry": hitl_last_milestone_retry,
            "prompt_source": args.hitl_prompt_source,
            "horizon_mult": args.hitl_horizon_mult,
            "max_steps_cap": args.hitl_max_steps_cap,
            "default_est_length": args.hitl_default_est_length,
            "replan_steps": args.hitl_replan_steps,
            "resize_size": args.hitl_resize_size,
            "stop_progress": args.hitl_stop_progress,
            "stop_eps": args.hitl_stop_eps,
            "stop_window": args.hitl_stop_window,
            "static_eps": args.hitl_static_eps,
            "zero_arm_in_base": args.hitl_zero_arm_in_base,
        }
    configure_human_interactive(hitl_cfg)
    print(f"Serving finestep rollouts from {ROOT}  ->  http://{args.host}:{args.port}/finestep")
    print(f"  combined S2+S1 rollouts from {COMBINE_ROOT}  ->  /combine")
    if "milestone" in ROOTS:
        print(f"  milestone rollouts from {ROOTS['milestone']}  ->  /milestone")
    if "episode" in ROOTS:
        print(f"  episode rollouts from {ROOTS['episode']}  ->  /episode")
    print(f"  val_mse curves from {VAL_MSE_DIR}  ->  /val_mse")
    if hitl_cfg:
        print("  human-interactive target sessions -> /human-interactive")
        print(f"    episodes: {hitl_cfg['dataset_root']}")
        print(f"    results:  {hitl_cfg['results_root']}")
        print(f"    S1 {args.hitl_s1_host}:{args.hitl_s1_port} | "
              f"S2 {args.hitl_s2_host}:{args.hitl_s2_port} ({args.hitl_s2_model})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
