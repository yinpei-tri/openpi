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

Run:
    python examples/robocasa/subtask_eval_gui.py --rollout-root subtask_rollouts --port 8092
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, send_file

app = Flask(__name__)
ROOT: Path = Path(".")                        # default / legacy subtask rollout root
# Named rollout trees, both written in the SAME layout by subtask_eval.py (#3) and
# episode_eval.py (#2), so ONE viewer serves both — selected by the <root> path segment.
ROOTS: dict[str, Path] = {}
VAL_MSE_DIR: Path = Path("eval_out/val_mse")  # eval #1 curves (scripts/eval_val_mse.py output)
# Train/val MSE sweep result dirs (per-ckpt JSON from scripts/run_val_sweep.py). Each holds
# <exp>__<step>.json = {exp_name, steps:[{step, action_mse, progress_acc, progress_mae,
# progress_mode}]}. The /val_mse page overlays both splits with per-split + per-method toggles.
MSE_DIRS: dict[str, Path] = {
    "val": Path("_evallogs/valmse_results"),
    "train": Path("_evallogs/trainmse_results"),
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
        out.append(json.loads((_safe(rp, method, ep) / "episode.json").read_text()))
    return jsonify(out)


@app.route("/api/task_splits")
def api_task_splits():
    """task_name -> target-split label (atomic_seen / composite_seen / composite_unseen), so the
    viewer can group the task dropdown by type."""
    return jsonify(_target_split_map())


# ---- legacy routes (the existing "/" subtask page uses these; keep them working) ----
@app.route("/api/methods")
def api_methods():
    return api_methods_r("subtask")


@app.route("/api/episodes/<method>")
def api_episodes(method):
    return api_episodes_r("subtask", method)


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


# legacy aliases (subtask root)
@app.route("/api/steps/<method>/<episode>/<sub>")
def api_steps(method, episode, sub):
    return api_steps_r("subtask", method, episode, sub)


@app.route("/api/media/<method>/<episode>/<sub>/<path:fname>")
def api_media(method, episode, sub, fname):
    return api_media_r("subtask", method, episode, sub, fname)


def _viewer_page(root_name: str) -> str:
    """The rollout viewer page bound to a rollout root ('subtask' or 'episode'). Injects
    window.RN so the shared gui.js hits /api/<root>/... and a small nav bar."""
    nav = ('<nav style="font-size:13px">'
           '<a href="/" style="color:#b0431c;margin-right:8px">home</a>'
           '<a href="/subtask" style="color:#b0431c;margin-right:8px">subtask</a>'
           '<a href="/episode" style="color:#b0431c;margin-right:8px">episode</a>'
           '<a href="/val_mse" style="color:#b0431c;margin-right:8px">val_mse</a>'
           '<a href="/stats" style="color:#b0431c">stats</a></nav>')
    inject = f"<script>window.RN={root_name!r};</script>"
    # Put the RN global BEFORE gui.js loads, and drop the nav into the top bar.
    html = INDEX_HTML.replace("<script src=\"/gui.js\">", inject + "<script src=\"/gui.js\">")
    html = html.replace("<div id=\"top\">", f"<div id=\"top\">{nav}", 1)
    # Per-root title + heading: /episode -> "Episode Eval", /subtask -> "Subtask Eval".
    label = "Episode" if root_name == "episode" else "Subtask"
    html = html.replace("<title>Subtask Eval Viewer</title>", f"<title>{label} Eval Viewer</title>")
    html = html.replace("<h1><b>Subtask</b> Eval</h1>", f"<h1><b>{label}</b> Eval</h1>")
    return html


@app.route("/")
def index():
    return HOME_HTML


@app.route("/subtask")
def subtask_page():
    return _viewer_page("subtask")


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
    reg = Path("/home/yinpei.dai/robocasa/robocasa/utils/dataset_registry.py")
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


def _episode_stats(root: Path) -> dict:
    """Per-method episode-success stats (from episode_eval.py output). Breakdown by the RoboCasa
    TARGET SPLIT: overall + atomic_seen / composite_seen / composite_unseen (from the task_name via
    _target_split_map). Also per-task rates and per-method timing (from index.json)."""
    tsplit = _target_split_map()   # task_name -> atomic_seen / composite_seen / composite_unseen
    keys = ["all", "atomic_seen", "composite_seen", "composite_unseen"]
    out = {}
    for m in _methods(root):
        succ = {k: [0, 0] for k in keys}  # [n_success, n_total]
        by_task: dict[str, list] = {}     # task_name -> [n_success, n_total, split]
        for ep in _episodes(root, m):
            try:
                doc = json.loads((_safe(root, m, ep) / "episode.json").read_text())
            except Exception:
                continue
            if doc.get("episode_success") is None:
                continue
            eid = doc.get("episode_id", "")
            task = doc.get("task_name") or (eid.split("/")[2] if "/" in eid else eid)
            cat = "atomic" if "/Atomic/" in eid else ("composite" if "/Composite/" in eid else None)
            spl = tsplit.get(task)   # atomic_seen / composite_seen / composite_unseen (or None)
            ok = 1 if doc["episode_success"] else 0
            def _bump(k):
                succ[k][0] += ok; succ[k][1] += 1
            _bump("all")
            if spl in succ:
                _bump(spl)
            # per-task tally (the requested per-task breakdown)
            b = by_task.setdefault(task, [0, 0, spl or cat or "?"])
            b[0] += ok; b[1] += 1
        row = {k: (v[0] / v[1] if v[1] else None) for k, v in succ.items()}
        row["n"] = succ["all"][1]
        row["counts"] = {k: succ[k][1] for k in keys}        # n episodes per cell (total)
        row["succ_counts"] = {k: succ[k][0] for k in keys}   # n SUCCESSFUL episodes per cell
        row["per_task"] = {t: {"rate": (b[0] / b[1] if b[1] else None), "s": b[0], "n": b[1], "split": b[2]}
                           for t, b in sorted(by_task.items())}
        # per-method timing from the method index.json (avg s/episode + per-task table)
        try:
            idx = json.loads((root / m / "index.json").read_text())
            row["avg_seconds_per_episode"] = idx.get("avg_seconds_per_episode")
            row["total_seconds"] = idx.get("total_seconds")
        except Exception:
            pass
        out[m] = row
    return out


def _subtask_stats(root: Path) -> dict:
    """Per-method Gemini subtask-success stats (from subtask_eval + gemini_judge): overall +
    per primitive."""
    out = {}
    for m in _methods(root):
        overall = [0, 0]
        by_prim: dict[str, list[int]] = {}
        for ep in _episodes(root, m):
            try:
                doc = json.loads((_safe(root, m, ep) / "episode.json").read_text())
            except Exception:
                continue
            g = doc.get("gemini")
            if not g:
                continue
            per = g.get("per_subtask", {})
            for sg in doc.get("subgoals", []):
                cd = Path(sg["out_dir"]).name
                v = per.get(cd, {}).get("success")
                if v is None:
                    continue
                prim = sg.get("primitive", "other")
                ok = 1 if v else 0
                overall[0] += ok; overall[1] += 1
                b = by_prim.setdefault(prim, [0, 0]); b[0] += ok; b[1] += 1
        out[m] = dict(
            overall=(overall[0] / overall[1] if overall[1] else None), n=overall[1],
            per_primitive={p: (v[0] / v[1] if v[1] else None) for p, v in sorted(by_prim.items())})
    return out


# /api/stats reads thousands of episode.json (8 methods x 500 eps) — ~3s/call. The data only
# changes when a rollout/judge writes new files, so cache the computed result and invalidate on a
# cheap fingerprint: the newest index.json mtime + method-dir count across the episode+subtask roots
# (index.json is rewritten whenever a method's episodes change). ?refresh=1 forces a recompute.
_STATS_CACHE: dict = {"fp": None, "data": None}


def _stats_fingerprint() -> tuple:
    fp = []
    for root in (_root("episode"), _root("subtask")):
        if not root.exists():
            continue
        for idx in root.glob("*/index.json"):
            try:
                fp.append((str(idx), idx.stat().st_mtime))
            except OSError:
                pass
    if VAL_MSE_DIR.is_dir():
        for f in VAL_MSE_DIR.glob("*.json"):
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
                subtask=_subtask_stats(_root("subtask")))


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
    return STATS_HTML


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
  <nav><a href="/">home</a><a href="/subtask">subtask</a><a href="/episode">episode</a><a href="/val_mse">val_mse</a><a href="/stats">stats</a></nav>
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
load();
setInterval(load, 60000);   // refresh while the sweep is still writing results
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
</style></head><body>
<header>
  <h1>Eval <b>Statistics</b></h1>
  <nav><a href="/">home</a><a href="/subtask">subtask</a><a href="/episode">episode</a><a href="/val_mse">val_mse</a><a href="/stats">stats</a></nav>
</header>
<div id="wrap">
  <h2>#1 Validation MSE (final step)</h2><div id="valtab"></div>
  <h2>#2 Episode success rate <span style="font-size:11px;color:#888;font-weight:400">— % (n episodes)</span></h2><div id="eptab"></div>
  <h2>#2b Per-task episode success <span style="font-size:11px;color:#888;font-weight:400">— task × method, grouped by split · #successful / #episodes (hover for %)</span></h2><div id="tasktab" style="overflow-x:auto"></div>
  <h2>#3 Subtask Gemini success rate</h2><div id="subtab"></div>
  <div class="box"><canvas id="chart" height="90"></canvas></div>
</div>
<script>
const COLORS=["#b0431c","#1c6bb0","#2e8b3d","#8b2eb0","#b0902e","#2eb0a3","#b02e5a","#555"];
const pct=v=>v==null?'–':(100*v).toFixed(1)+'%';
const f4=v=>v==null?'–':(+v).toFixed(4);
async function load(){
  const d=await (await fetch('/api/stats')).json();
  // #1 val table
  let h="<table><tr><th class='exp'>exp_name</th><th>step</th><th>action_mse</th><th>flow_loss</th></tr>";
  d.val_mse.forEach(r=>h+=`<tr><td class='exp'>${r.exp_name}</td><td>${r.final_step??'–'}</td><td>${f4(r.action_mse)}</td><td>${f4(r.flow_loss)}</td></tr>`);
  document.getElementById('valtab').innerHTML=h+"</table>";
  // #2 episode table: overall + category (atomic/composite) + split (seen/unseen) + 2x2 cross,
  // each cell "% (n)"; last column = avg wall-time per episode. Sortable: click a header.
  const cell=(v,c,cnt)=>{const n=cnt&&cnt[c]!=null?cnt[c]:null; return `<td title="${n!=null?n+' episodes':''}">${pct(v[c])}${n!=null?` <span style="color:#aaa">(${n})</span>`:''}</td>`;};
  const ecols=[['all','overall'],['atomic_seen','atomic-seen'],
               ['composite_seen','composite-seen'],['composite_unseen','composite-unseen']];
  // sortable columns: key -> how to extract the sort value from a method row
  const sortKeys={overall:v=>v.all, 'atomic-seen':v=>v.atomic_seen, 'composite-seen':v=>v.composite_seen,
                  'composite-unseen':v=>v.composite_unseen, 'avg s/ep':v=>v.avg_seconds_per_episode};
  window._epSort=window._epSort||{key:'overall',dir:-1};   // default: overall descending
  function renderEpTable(){
    const {key,dir}=window._epSort;
    const getv=sortKeys[key]||sortKeys.overall;
    const rows=Object.entries(d.episode).sort((a,b)=>{
      const va=getv(a[1]), vb=getv(b[1]);
      if(va==null&&vb==null)return 0; if(va==null)return 1; if(vb==null)return -1;
      return dir*(vb-va);});
    const arrow=c=>window._epSort.key===c?(dir<0?' ▼':' ▲'):'';
    const hdr=(label)=>`<th class="sortable" data-k="${label}" style="cursor:pointer" title="click to sort">${label}${arrow(label)}</th>`;
    let hh="<table><tr><th class='exp'>method</th><th>n</th>"+ecols.map(c=>hdr(c[1])).join("")+hdr('avg s/ep')+"</tr>";
    // color the method cell by progress-head family: progact* = light blue, progreg* = light red
    // (progcls / other = default). Keyed on the token after the vN_ prefix.
    const mColor=(m)=>{const t=m.toLowerCase();
      if(t.includes('progact'))return '#e6f0fb';   // light blue
      if(t.includes('progreg'))return '#fdeaea';   // light red
      return '';};
    rows.forEach(([m,v])=>{
      const bg=mColor(m); const st=bg?` style="background:${bg}"`:'';
      hh+=`<tr><td class='exp'${st}>${m}</td><td>${v.n}</td>`+ecols.map(c=>cell(v,c[0],v.counts)).join("")+
         `<td>${v.avg_seconds_per_episode!=null?(+v.avg_seconds_per_episode).toFixed(1):'–'}</td></tr>`;});
    document.getElementById('eptab').innerHTML=hh+"</table>";
    document.querySelectorAll('#eptab th.sortable').forEach(th=>th.onclick=()=>{
      const k=th.dataset.k;
      if(window._epSort.key===k)window._epSort.dir*=-1; else window._epSort={key:k,dir:-1};
      renderEpTable();});
  }
  renderEpTable();
  // #2b per-task matrix: rows = tasks (grouped by split), cols = methods, cell = success % (n).
  const emethods=Object.keys(d.episode);
  // union of tasks + each task's split, from the per_task maps
  const taskInfo={};
  emethods.forEach(m=>{const pt=d.episode[m].per_task||{}; for(const t in pt){taskInfo[t]=taskInfo[t]||pt[t].split;}});
  const splitOrder={atomic_seen:0,composite_seen:1,composite_unseen:2};
  const tasks=Object.keys(taskInfo).sort((a,b)=>{
    const sa=splitOrder[taskInfo[a]]??9, sb=splitOrder[taskInfo[b]]??9;
    return sa!==sb?sa-sb:a.localeCompare(b);});
  // a cell showing SUCCESSFUL episode count / total (e.g. "7/10"), green-shaded by rate; the % is
  // in the tooltip. rate is used only for shading.
  const cntCell=(s,n,rate,bold)=>{
    if(n==null||n===0)return '<td style="color:#ccc">–</td>';
    const w=bold?'font-weight:700;':'';
    const r=rate!=null?rate:(s/n);
    return `<td title="${(100*r).toFixed(1)}%" style="${w}background:rgba(46,139,61,${(0.10+0.5*r).toFixed(2)})">${s}<span style="color:#888">/${n}</span></td>`;};
  // aggregate row (bold): success/total from succ_counts/counts; label spans the task col.
  const aggRow=(label,aggKey,bg)=>`<tr><td class='exp' style="background:${bg};font-weight:700">${label}</td>`+
      emethods.map(m=>cntCell((d.episode[m].succ_counts||{})[aggKey], (d.episode[m].counts||{})[aggKey],
                              d.episode[m][aggKey], true)).join("")+"</tr>";
  let th="<table><tr><th class='exp'>task</th>"+emethods.map(m=>`<th>${m.replace('_',' ')}</th>`).join("")+"</tr>";
  // top: TOTAL over all tasks
  th+=aggRow('TOTAL (all tasks)','all','#dfe6f5');
  // then each split: a section 'overall' row, then its tasks (no split column)
  const splitLabel={atomic_seen:'atomic-seen',composite_seen:'composite-seen',composite_unseen:'composite-unseen'};
  for(const sp of ['atomic_seen','composite_seen','composite_unseen']){
    th+=aggRow(splitLabel[sp]+' — overall', sp, '#eef');
    tasks.filter(t=>taskInfo[t]===sp).forEach(t=>{
      th+=`<tr><td class='exp'>${t}</td>`+emethods.map(m=>{const e=(d.episode[m].per_task||{})[t];
        return cntCell(e?e.s:null, e?e.n:null, e?e.rate:null, false);}).join("")+"</tr>";});
  }
  document.getElementById('tasktab').innerHTML=th+"</table>";
  // #3 subtask table (overall + primitives)
  const prims=[...new Set(Object.values(d.subtask).flatMap(v=>Object.keys(v.per_primitive||{})))].sort();
  h="<table><tr><th class='exp'>method</th><th>n</th><th>overall</th>"+prims.map(p=>`<th>${p}</th>`).join("")+"</tr>";
  Object.entries(d.subtask).forEach(([m,v])=>{h+=`<tr><td class='exp'>${m}</td><td>${v.n}</td><td>${pct(v.overall)}</td>`+
    prims.map(p=>`<td>${pct(v.per_primitive?.[p])}</td>`).join("")+"</tr>";});
  document.getElementById('subtab').innerHTML=h+"</table>";
  // grouped bar: episode-all vs subtask-overall per method
  const methods=[...new Set([...Object.keys(d.episode),...Object.keys(d.subtask)])];
  const ds=[
    {label:'episode success (all)',data:methods.map(m=>d.episode[m]?100*(d.episode[m].all||0):null),backgroundColor:COLORS[0]},
    {label:'subtask gemini (overall)',data:methods.map(m=>d.subtask[m]?100*(d.subtask[m].overall||0):null),backgroundColor:COLORS[1]},
  ];
  new Chart(document.getElementById('chart'),{type:'bar',data:{labels:methods,datasets:ds},
    options:{responsive:true,scales:{y:{min:0,max:100,title:{display:true,text:'%'}}}}});
}
load();
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
</style></head><body>
<header>
  <h1>RoboCasa <b>System1</b> Eval</h1>
  <p>Browse rollouts and aggregate metrics for the subgoal-conditioned pi0.5 System1 policies.</p>
</header>
<div id="wrap">
  <a class="card" href="/episode"><h2>Episode Eval<span class="path">/episode</span></h2>
    <p>Whole-episode OPEN-LOOP rollouts: reset once to the first subgoal, then roll the policy
       continuously through the subgoal list. Per-subgoal video, prompt, anchor, action chunk, and
       the env <i>_check_success</i> task verdict. Seen (first-5) vs unseen (last-5) episodes.</p></a>
  <a class="card" href="/subtask"><h2>Subtask Eval<span class="path">/subtask</span></h2>
    <p>Per-subtask CLOSED-LOOP rollouts: hard-reset to each subgoal start independently. Same rich
       per-frame panels, plus the oracle GT reference and the Gemini success verdict per subtask.</p></a>
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
const RN=window.RN||'subtask';   // rollout root: 'subtask' (#3) or 'episode' (#2)
const $=s=>document.querySelector(s);
// Gemini verdict for a subgoal (#3): episode.json.gemini.per_subtask[<child_dir>].success.
// Map a subtask's Gemini verdict (from the prefetched cache) to true/false/null for track coloring:
// success->true, failure->false, uncertain/skipped/absent->null.
function subVerdict(ep,s){
  const cd=(s.out_dir||'').split('/').pop();
  const g=S._verdicts&&S._verdicts[cd];
  if(!g)return null;
  const v=g.verdict;
  if(v==='success')return true;
  if(v==='failure')return false;
  if(v===undefined&&typeof g.success==='boolean')return g.success;  // legacy schema
  return null;   // uncertain / skipped
}
const S={method:null,eps:[],epi:0,subi:0,steps:null,fps:20,norm:true};
const esc=s=>(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
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
function stateBlock(vals,groups){return groups.map(([n,s,e])=>`  ${n.padEnd(12)} ${vals.slice(s,e).map(x=>Number(x).toFixed(3).padStart(8)).join(' ')}`).join("\n");}

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
  S.eps=await (await fetch(`/api/${RN}/episodes/`+S.method)).json();
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
async function loadVerdicts(ep){
  S._verdicts={};
  const epf=ep.episode_id.replaceAll('/','__');
  await Promise.all((ep.subgoals||[]).map(async sg=>{
    const cd=sg.out_dir.split('/').pop();
    try{ const g=await (await fetch(`/api/${RN}/gemini/${S.method}/${epf}/${cd}`)).json();
         if(g) S._verdicts[cd]=g; }catch(e){}
  }));
}
async function selectEpisode(i){ S.epi=i; S.subi=0; await loadVerdicts(S.eps[i]); drawTrack(); await selectSub(0); }

function drawTrack(){
  const ep=S.eps[S.epi]; const subs=ep.subgoals;
  const T=Math.max(...subs.map(s=>s.span[1]))+1;
  // milestones: union child spans by milestone_index
  const ms={}; subs.forEach(s=>{const k=s.milestone_index; if(!(k in ms))ms[k]={a:s.span[0],b:s.span[1],text:s.milestone_subgoal||('milestone '+k)}; ms[k].a=Math.min(ms[k].a,s.span[0]); ms[k].b=Math.max(ms[k].b,s.span[1]);});
  const curMs=subs[S.subi].milestone_index;
  const pct=(x)=>100*x/T;
  const msSegs=Object.entries(ms).map(([k,v])=>`<div class="pb-seg pb-ms ${+k===curMs?'cur':''}" style="left:${pct(v.a)}%;width:${pct(v.b-v.a+1)}%" title="${esc(v.text)}" data-ms="${k}">${esc(v.text)}</div>`).join("");
  const fsSegs=subs.map((s,i)=>{
    const v=subVerdict(ep,s);   // gemini success: true/false/null
    const vc=v===true?'#2e8b3d':(v===false?'#b0431c':'');   // green / red / default
    const adv=(s.advanced===true)?' ✓':(s.advanced===false?' ⧗':'');  // #2: stop fired / timeout
    const style=`left:${pct(s.span[0])}%;width:${pct(s.span[1]-s.span[0]+1)}%`+(vc?`;box-shadow:inset 0 -3px 0 ${vc}`:'');
    return `<div class="pb-seg pb-fs ${i===S.subi?'cur':''}" style="${style}" title="[${s.primitive}] ${esc(s.subgoal)}${v===null?'':' | gemini:'+(v?'PASS':'FAIL')}" data-sub="${i}">${esc(s.subgoal)}${adv}</div>`;
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
  // primchip: primitive + subgoal, plus (#3) the Gemini verdict, plus (#2) episode success banner.
  const cd0=(sg.out_dir||'').split('/').pop();
  const g0=S._verdicts&&S._verdicts[cd0];
  const vtxt = g0 ? ('  · gemini: '+ (g0.skipped?'SKIPPED':(g0.verdict||(g0.success===true?'success':g0.success===false?'failure':'—')).toUpperCase())) : '';
  const reset=(RN==='episode')?'  · reset: first-only':'  · reset: hard (GT)';
  $('#primchip').textContent=`[${sg.primitive}] ${sg.subgoal}${vtxt}${reset}`;
  let ms=`milestone: ${sg.milestone_subgoal||sg.milestone_index}`;
  if(ep.episode_success!=null)ms+=`  ·  EPISODE: ${ep.episode_success?'SUCCESS':'fail'} (advanced ${ep.n_advanced}/${ep.n_subgoals})`;
  else if(S._verdicts){
    // episode gemini task rate = success / (success+failure) over judged (non-skipped) subtasks.
    let s=0,d=0; for(const k in S._verdicts){const g=S._verdicts[k]; if(g.skipped)continue;
      const v=g.verdict||(g.success===true?'success':g.success===false?'failure':null);
      if(v==='success'){s++;d++;} else if(v==='failure'){d++;}}
    if(d)ms+=`  ·  gemini task rate: ${(100*s/d).toFixed(0)}% (${s}/${d})`;
  }
  $('#mschip').textContent=ms;
  buildSeries();
  drawTrack();
  renderStatic();   // rebuilds #grid (fresh #vid/#play/#slider) and re-wires them via wireVideo()
  const base=`/api/${RN}/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${sg.out_dir.split('/').pop()}/`;
  resetVideoForClip(base+S.steps.clean_video);  // pause, cancel stale follow-loop, swap src + load
  $('#slider').max=S.steps.steps.length-1; $('#slider').value=0; showFrame(0);
}

// TASK SUCCESS panel: episode success (#2) OR subtask Gemini verdict + reasoning (#3),
// plus the model's self-stop info and the current progress readout.
async function renderSuccess(){
  const el=$('#success'); if(!el)return;
  const ep=S.eps[S.epi], sg=ep.subgoals[S.subi];
  const cd=(sg.out_dir||'').split('/').pop();
  const rows=[];
  const row=(k,v,cls)=>`<div class="pf"><span class="pfk">${k}</span><span class="pfv ${cls||''}">${v}</span></div>`;
  const yn=b=>b===true?'<b style="color:#2e8b3d">SUCCESS</b>':(b===false?'<b style="color:#b0431c">FAIL</b>':'—');
  if(RN==='episode' || ep.episode_success!=null){
    // Whole-episode ground truth: the RoboCasa env's own _check_success() at the end of the
    // open-loop rollout. THIS is the task-success verdict.
    rows.push(row('TASK SUCCESS &nbsp;<span style="font-weight:400;color:#888">(env _check_success @ end)</span>',
                  yn(ep.episode_success), 'hl'));
    // How far the policy got: how many subgoals it self-terminated ("advanced past") before
    // either finishing or hitting a budget cap. NOT a per-subgoal success signal.
    rows.push(row('subgoals reached', `${ep.n_advanced} of ${ep.n_subgoals} self-advanced`));
    // This subgoal: did the policy DECIDE it was done and move on, or did it run out of budget?
    // (advanced = self-stopped & moved to next; NOT "this subgoal succeeded".)
    const adv = (sg.advanced===true) ? '<b style="color:#2e8b3d">advanced → moved to next</b>'
              : (sg.advanced===false) ? '<b style="color:#b0431c">timed out (budget cap)</b>' : '—';
    rows.push(row('this subgoal', adv));
    if(sg.stop_reason&&sg.stop_reason.timeout)rows.push(row('why it stopped', 'budget cap reached — policy never self-stopped'));
    else if(sg.advanced)rows.push(row('why it stopped', 'progress ≥ threshold AND motion quiescent'));
  } else {
    // subtask (#3): env auxiliary + Gemini three-way verdict + completion + confidence + reasoning
    // (from the prefetched judge/gemini.json cache).
    rows.push(row('sim _check_success (aux)', yn(sg.sim_success_final)));
    rows.push(row('self-stopped', sg.stopped==null?'—':(sg.stopped?'yes (progress+quiescence)':'no (budget cap)')));
    const g=S._verdicts&&S._verdicts[cd];
    // verdict label: color success/failure/uncertain; SKIPPED (low-movement gate) shown distinctly.
    const vlabel=(g)=>{
      if(g.skipped)return '<b style="color:#999">SKIPPED</b> <span style="color:#999;font-weight:400">(low movement — not judged)</span>';
      const v=g.verdict||(g.success===true?'success':g.success===false?'failure':null);
      if(v==='success')return '<b style="color:#2e8b3d">SUCCESS</b>';
      if(v==='failure')return '<b style="color:#b0431c">FAILURE</b>';
      if(v==='uncertain')return '<b style="color:#b08a1c">UNCERTAIN</b>';
      return '—';
    };
    // Gemini reason FIRST (the analysis), then the verdict/completion/confidence it concludes.
    if(g&&g.reason)rows.push(`<div class="pf stack"><span class="pfk">gemini reason</span><span class="pfv">${esc(g.reason)}</span></div>`);
    rows.push(row('GEMINI verdict', g?vlabel(g):'(not judged)', 'hl'));
    if(g&&!g.skipped&&g.completion!=null)rows.push(row('completion', g.completion));
    if(g&&g.confidence!=null)rows.push(row('confidence', (+g.confidence).toFixed(2)));
  }
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
          <div><span class="key">span</span><span class="val">[${d.span.join(', ')}] (len ${d.summary.span_len}) · est_len ${d.summary.est_length}</span></div>
          <div><span class="key">budget</span><span class="val">${d.budget} steps (settle ${d.settle_steps})</span></div>
          <div><span class="key">1st-chunk mse</span><span class="val">${(d.summary.first_chunk_action_mse??0).toFixed(4)} · mean-step ${(d.summary.mean_step_action_mse??0).toFixed(4)}</span></div>
          <div><span class="key">sim_success</span><span class="val">final ${d.summary.sim_success_final} · any ${d.summary.sim_success_any}</span></div>
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
        <video id="vid" class="half" muted></video>
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


def main():
    global ROOT, ROOTS, VAL_MSE_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-root", type=Path, required=True,
                   help="SUBTASK rollout tree (subtask_eval.py output) -> /subtask")
    p.add_argument("--episode-root", type=Path, default=None,
                   help="EPISODE rollout tree (episode_eval.py output) -> /episode")
    p.add_argument("--val-mse-dir", type=Path, default=Path("eval_out/val_mse"),
                   help="dir of scripts/eval_val_mse.py output JSONs (the /val_mse curves)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    args = p.parse_args()
    ROOT = args.rollout_root.resolve()
    ROOTS = {"subtask": ROOT}
    if args.episode_root is not None:
        ROOTS["episode"] = args.episode_root.resolve()
    VAL_MSE_DIR = args.val_mse_dir.resolve()
    print(f"Serving subtask rollouts from {ROOT}  ->  http://{args.host}:{args.port}/subtask")
    if "episode" in ROOTS:
        print(f"  episode rollouts from {ROOTS['episode']}  ->  /episode")
    print(f"  val_mse curves from {VAL_MSE_DIR}  ->  /val_mse")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
