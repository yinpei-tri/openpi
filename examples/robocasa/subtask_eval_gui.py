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
ROOT: Path = Path(".")


def _methods():
    return sorted([d.name for d in ROOT.iterdir() if d.is_dir() and (d / "index.json").exists()]) \
        if ROOT.exists() else []


def _episodes(method):
    md = ROOT / method
    return sorted([d.name for d in md.iterdir() if d.is_dir() and (d / "episode.json").exists()]) \
        if md.is_dir() else []


def _safe(method, episode=None, sub=None) -> Path:
    p = ROOT / method
    if episode:
        p = p / episode
    if sub:
        p = p / sub
    p = p.resolve()
    if not str(p).startswith(str(ROOT.resolve())):
        abort(403)
    return p


@app.route("/api/methods")
def api_methods():
    out = []
    for m in _methods():
        idx = json.loads((ROOT / m / "index.json").read_text())
        out.append(dict(method=m, n_episodes=idx.get("n_episodes"),
                        horizon_mult=idx.get("horizon_mult"), settle_steps=idx.get("settle_steps")))
    return jsonify(out)


@app.route("/api/episodes/<method>")
def api_episodes(method):
    out = []
    for ep in _episodes(method):
        doc = json.loads((_safe(method, ep) / "episode.json").read_text())
        out.append(doc)
    return jsonify(out)


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
            s["query"] = dict(
                prompt=rp.get("prompt"), gripper_flag=rp.get("gripper_flag"),
                executed_step=rp.get("executed_step"), replan_steps=rp.get("replan_steps"),
                horizon=rp.get("horizon"),
                chunk_lean11_norm=z["q_chunk_norm"][k].astype(float).round(4).tolist(),
                chunk_lean11=z["q_chunk_lean"][k].astype(float).round(4).tolist(),
                chunk_progress=prog)
        steps.append(s)
    return {**meta, "steps": steps}


@app.route("/api/steps/<method>/<episode>/<sub>")
def api_steps(method, episode, sub):
    doc = _reconstruct_full(_safe(method, episode, sub))
    if doc is None:
        abort(404)
    return jsonify(doc)


@app.route("/api/media/<method>/<episode>/<sub>/<path:fname>")
def api_media(method, episode, sub, fname):
    f = _safe(method, episode, sub) / fname
    if not f.exists():
        abort(404)
    mt = "video/mp4" if fname.endswith(".mp4") else ("image/jpeg" if fname.endswith(".jpg") else None)
    return send_file(f, mimetype=mt)


@app.route("/")
def index():
    return INDEX_HTML


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
    <label class="dssel">task/episode<select id="episode"></select></label>
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
const $=s=>document.querySelector(s);
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
  const grab=(re)=>{const m=P.match(re); return m?m[1].trim():"—";};
  const task=grab(/Task:\s*([^\n;]*)/), subgoal=grab(/Current Subgoal:\s*([^\n;]*)/);
  const quality=grab(/Quality:\s*([^\n;]*)/), estlen=grab(/Estimated Length:\s*([^\n;]*)/);
  // discretized 256-bin ints of the state AT THE GOVERNING REPLAN (anchor=Initial first, then Current)
  const curInts=discretize(gs.cur_lean_norm), ancInts=discretize(S.steps.anchor_state_lean14_norm);
  const row=(k,v,cls)=>`<div class="pf"><span class="pfk">${k}</span><span class="pfv ${cls||''}">${esc(String(v))}</span></div>`;
  $('#prompt').innerHTML=
    row('Task', task)+
    row('Current Subgoal', subgoal, 'hl')+
    row('Quality', quality)+
    row('Estimated Length', estlen)+
    row('Executed Step', gs.frame_step, 'hl')+
    row('Current Gripper', gs.gripper_flag)+
    row('Initial State', ancInts?ancInts.join(' '):'—', 'ints')+
    row('Current State', curInts?curInts.join(' '):'—', 'ints')+
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

// scalar progress in [0,1] from a step's progress_raw (classes -> E[frac]; continuous/action -> value)
function progScalar(pr){
  if(!pr)return null;
  if(pr.progress_kind==='classes')return pr.progress_expected_frac;
  if(pr.progress_kind==='continuous')return pr.progress_now;
  if(pr.progress_kind==='action')return pr.progress_now;
  return null;
}
// Build per-step series for the current subtask. Also precompute, for EACH frame, the index of
// the GOVERNING query — the most recent replan step at/before it. The model is only queried on
// replan steps (every replan_steps), so its prompt + predicted chunk stay CONSTANT until the next
// replan. Rendering from govQ makes the prompt/chunk correct at any frame, including when scrubbing
// backward (S._lastQuery mutated during playback would otherwise be wrong).
function buildSeries(){
  const st=S.steps.steps; let lastP=null; let govIdx=-1;
  S.series={prog:[],eef_pos:[],eef_rot:[],base:[],phase:[]};
  S.govQ=[];  // per frame: index of the frame whose .query governs it (-1 during warmup)
  st.forEach((s,i)=>{
    const p=progScalar(s.progress_raw); if(p!=null)lastP=p;
    S.series.prog.push(lastP);
    S.series.eef_pos.push(s.action_eef_pos_norm);
    S.series.eef_rot.push(s.action_eef_rot_norm);
    S.series.base.push(s.action_base_norm);
    S.series.phase.push(s.phase);
    if(s.query)govIdx=i;      // this frame issued a fresh model query
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

async function loadMethods(){
  const ms=await (await fetch('/api/methods')).json();
  $('#method').innerHTML=ms.map(m=>`<option value="${m.method}">${m.method}</option>`).join('');
  window._ms=ms; if(ms.length){S.method=ms[0].method; await loadEpisodes();}
}
async function loadEpisodes(){
  S.method=$('#method').value;
  S.eps=await (await fetch('/api/episodes/'+S.method)).json();
  $('#episode').innerHTML=S.eps.map((e,i)=>`<option value="${i}">${e.task_name} :: ${e.episode_id.split('/').pop()}</option>`).join('');
  const m=window._ms.find(x=>x.method===S.method)||{};
  $('#dsinfo').textContent=`${S.method} · ${m.n_episodes} eps · budget ${m.horizon_mult}x · settle ${m.settle_steps}`;
  S.epi=0; S.subi=0; await selectEpisode(0);
}
async function selectEpisode(i){ S.epi=i; S.subi=0; drawTrack(); await selectSub(0); }

function drawTrack(){
  const ep=S.eps[S.epi]; const subs=ep.subgoals;
  const T=Math.max(...subs.map(s=>s.span[1]))+1;
  // milestones: union child spans by milestone_index
  const ms={}; subs.forEach(s=>{const k=s.milestone_index; if(!(k in ms))ms[k]={a:s.span[0],b:s.span[1],text:s.milestone_subgoal||('milestone '+k)}; ms[k].a=Math.min(ms[k].a,s.span[0]); ms[k].b=Math.max(ms[k].b,s.span[1]);});
  const curMs=subs[S.subi].milestone_index;
  const pct=(x)=>100*x/T;
  const msSegs=Object.entries(ms).map(([k,v])=>`<div class="pb-seg pb-ms ${+k===curMs?'cur':''}" style="left:${pct(v.a)}%;width:${pct(v.b-v.a+1)}%" title="${esc(v.text)}" data-ms="${k}">${esc(v.text)}</div>`).join("");
  const fsSegs=subs.map((s,i)=>`<div class="pb-seg pb-fs ${i===S.subi?'cur':''}" style="left:${pct(s.span[0])}%;width:${pct(s.span[1]-s.span[0]+1)}%" title="[${s.primitive}] ${esc(s.subgoal)}" data-sub="${i}">${esc(s.subgoal)}</div>`).join("");
  $('#track').innerHTML=`<div class="pb-row"><div class="pb-lab">milestones</div><div class="pb-lane">${msSegs}</div></div>
    <div class="pb-row"><div class="pb-lab">subgoals</div><div class="pb-lane">${fsSegs}</div></div>`;
  document.querySelectorAll('.pb-fs').forEach(el=>el.onclick=()=>selectSub(+el.dataset.sub));
}

async function selectSub(i){
  const ep=S.eps[S.epi]; if(i<0||i>=ep.subgoals.length)return;
  S.subi=i; const sg=ep.subgoals[i];
  S.steps=await (await fetch(`/api/steps/${S.method}/${ep.episode_id.replaceAll('/','__')}/${sg.out_dir.split('/').pop()}`)).json();
  S.fps=S.steps.fps||20;
  $('#subpos').textContent=`subtask ${i} / ${ep.subgoals.length-1}`;
  $('#prevSub').disabled=i<=0; $('#nextSub').disabled=i>=ep.subgoals.length-1;
  $('#primchip').textContent=`[${sg.primitive}] ${sg.subgoal}`;
  $('#mschip').textContent=`milestone: ${sg.milestone_subgoal||sg.milestone_index}`;
  buildSeries();
  drawTrack();
  renderStatic();
  const base=`/api/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${sg.out_dir.split('/').pop()}/`;
  $('#vid').src=base+S.steps.clean_video;
  $('#slider').max=S.steps.steps.length-1; $('#slider').value=0; showFrame(0);
}

function renderStatic(){
  const d=S.steps, ep=S.eps[S.epi];
  const base=`/api/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${d.clean_video.replace('clean.mp4','')}`;
  const bdir=`/api/media/${S.method}/${ep.episode_id.replaceAll('/','__')}/${ep.subgoals[S.subi].out_dir.split('/').pop()}/`;
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
      <div class="card grow">
        <h3>PREDICTED ACTION CHUNK @ current query (full horizon) <button class="toggle" id="tg-chunk"></button></h3>
        <div id="chunk"></div>
      </div>
      <div class="card">
        <h3>EXECUTED vs ORACLE @ current step</h3>
        <div id="execcmp"></div>
      </div>
    </div>`;
  wireVideo();
}

function wireVideo(){
  const v=$('#vid');
  // Only follow the video clock while it is actively PLAYING. During manual step/slider the
  // slider index is authoritative (video currentTime seeks snap to sparse mp4 keyframes, which
  // otherwise bounces the panel back to a keyframe — the "stuck at frame 40" bug).
  v.addEventListener('timeupdate',()=>{if(!S.steps||v.paused)return;
    const i=Math.min(S.steps.steps.length-1,Math.round(v.currentTime*S.fps));$('#slider').value=i;showFrame(i);});
  $('#slider').oninput=e=>{v.pause();gotoFrame(+e.target.value);};
  $('#play').onclick=()=>{if(v.paused){v.play();$('#play').textContent='❚❚ pause';}else{v.pause();$('#play').textContent='▶ play';}};
  $('#ff').onclick=()=>stepFrame(1); $('#bb').onclick=()=>stepFrame(-1);
  const tog=()=>{S.norm=!S.norm; renderNormable();};
  if($('#tg-state'))$('#tg-state').onclick=tog;
  if($('#tg-chunk'))$('#tg-chunk').onclick=tog;
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
  $('#fchips').innerHTML=[
    `<span class="c hot">|eef_pos| ${s.action_eef_pos_norm.toFixed(3)}</span>`,
    `<span class="c hot">|eef_rot| ${s.action_eef_rot_norm.toFixed(3)}</span>`,
    `<span class="c hot">|base| ${s.action_base_norm.toFixed(3)}</span>`,
    `<span class="c">progress ${s.progress}</span>`,
    s.sim_check_success?`<span class="c ok">sim_check_success</span>`:'',
    `<span class="c">gripper_w ${s.gripper_width.toFixed(3)}</span>`,
    s.replanned?`<span class="c rep">REPLAN</span>`:'',
  ].join('');
  S._curStep=s;
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
    const useNorm=norm&&q.chunk_lean11_norm;
    const rows=useNorm?q.chunk_lean11_norm:q.chunk_lean11;
    // progact ONLY: the 12th action dim = per-step progress. Stored as UNNORMALIZED [0,1]
    // (SplitProgressAction already did progress=(prog_norm+1)/2). In the NORMALIZED view show
    // the model's DIRECT output 2*p-1 ∈ [-1,1]; in the UNNORMALIZED view show p ∈ [0,1].
    const prog=q.chunk_progress;
    const progCell=(p)=>useNorm?(2*p-1):p;
    const labs=prog?LEAN_LABELS.concat(['progress*']):LEAN_LABELS;
    // which chunk row is executing at the CURRENT frame = (current step − governing replan step)
    const curOff=(S._govStep!=null)?(s.frame_step - S._govStep.frame_step):-1;
    let h=`<div class="lab">${useNorm?'NORMALIZED — model direct output (quantile → [-1,1])':'UNNORMALIZED — real action → env.step'} · lean-11 order${prog?' + progress* (12th action dim, '+(useNorm?'[-1,1] model output':'[0,1] unnorm')+')':''} · horizon ${q.horizon} · green = ${q.replan_steps} executed · ▶ = current frame</div>`;
    h+="<table class='act'><tr><th>t</th>"+labs.map(l=>`<th${l==='progress*'?' class="progcol"':''}>${l}</th>`).join("")+"</tr>";
    rows.forEach((r,t)=>{
      let cells=r.map(x=>`<td>${x.toFixed(3)}</td>`).join("");
      if(prog)cells+=`<td class="progcol">${progCell(prog[t]??0).toFixed(3)}</td>`;
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
$('#episode').onchange=e=>selectEpisode(+e.target.value);
$('#prevSub').onclick=()=>selectSub(S.subi-1);
$('#nextSub').onclick=()=>selectSub(S.subi+1);
window.addEventListener('resize',()=>{if(S.series)drawCurves(+($('#slider')?.value||0));});
loadMethods();
"""


def main():
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-root", type=Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8092)
    args = p.parse_args()
    ROOT = args.rollout_root.resolve()
    print(f"Serving rollouts from {ROOT}  ->  http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
