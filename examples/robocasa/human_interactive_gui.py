"""Flask blueprint and browser UI for live human-interactive System2 + System1 evaluation."""

# Module-global configuration is deliberate: Flask registers one blueprint and this process owns
# exactly one non-reentrant simulator. En dashes are intentional UI typography.
# ruff: noqa: PLW0603, RUF001

from __future__ import annotations

import io
import json
from pathlib import Path
import threading
from typing import Any

from flask import Blueprint
from flask import Response
from flask import abort
from flask import jsonify
from flask import request
from flask import send_file

bp = Blueprint("human_interactive", __name__)
_LOCK = threading.RLock()
_CONFIG: dict[str, Any] | None = None
_MANAGER = None


def register_human_interactive(app) -> None:
    app.register_blueprint(bp)


def configure_human_interactive(config: dict[str, Any] | None) -> None:
    """Configure lazily; simulator imports happen only when a HITL endpoint is actually used."""
    global _CONFIG, _MANAGER
    with _LOCK:
        _CONFIG = dict(config) if config is not None else None
        _MANAGER = None


def _controller_types():
    try:
        from human_interactive_controller import InteractiveConfig
        from human_interactive_controller import InteractiveManager
        from human_interactive_controller import official_target_catalog
    except ImportError:
        from examples.robocasa.human_interactive_controller import InteractiveConfig
        from examples.robocasa.human_interactive_controller import InteractiveManager
        from examples.robocasa.human_interactive_controller import official_target_catalog
    return InteractiveConfig, InteractiveManager, official_target_catalog


def _config():
    if _CONFIG is None:
        raise RuntimeError("human-interactive evaluation is disabled; launch the GUI without --no-hitl")
    interactive_config, _, _ = _controller_types()
    return interactive_config(**_CONFIG)


def _manager():
    global _MANAGER
    with _LOCK:
        if _MANAGER is None:
            _, interactive_manager, _ = _controller_types()
            _MANAGER = interactive_manager(_config())
        return _MANAGER


def _error(exc: BaseException, status: int = 409):
    return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), status


@bp.route("/api/hitl/config")
def hitl_config():
    try:
        cfg = _config()
        return jsonify({"ok": True, "enabled": True, "config": cfg.public()})
    except Exception as exc:
        return _error(exc, 503)


@bp.route("/api/hitl/episodes")
def hitl_episodes():
    try:
        _, _, catalog = _controller_types()
        return jsonify(catalog(_config()))
    except Exception as exc:
        return _error(exc, 503)


@bp.route("/api/hitl/sessions")
def hitl_sessions():
    try:
        root = Path(_config().results_root)
        sessions = []
        for manifest in sorted(root.glob("*/manifest.json"), reverse=True):
            try:
                doc = json.loads(manifest.read_text())
            except Exception:
                continue
            sessions.append(
                {
                    "session_id": doc.get("session_id"),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "status": doc.get("status"),
                    "episode": doc.get("episode"),
                    "session_dir": str(manifest.parent),
                }
            )
        return jsonify(sessions)
    except Exception as exc:
        return _error(exc, 503)


@bp.route("/api/hitl/session", methods=["POST"])
def hitl_start_session():
    body = request.get_json(force=True, silent=True) or {}
    try:
        state = _manager().start(
            str(body["task_name"]),
            int(body["episode_index"]),
            operator=(str(body.get("operator") or "").strip() or None),
        )
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/state")
def hitl_state():
    try:
        return jsonify({"ok": True, "state": _manager().snapshot()})
    except Exception as exc:
        return _error(exc, 404)


@bp.route("/api/hitl/plan/propose", methods=["POST"])
def hitl_plan_propose():
    try:
        return jsonify({"ok": True, "state": _manager().call_session("propose_plan")})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/plan/accept", methods=["POST"])
def hitl_plan_accept():
    body = request.get_json(force=True, silent=True) or {}
    try:
        state = _manager().call_session(
            "accept_plan",
            str(body.get("plan") or ""),
            author=(str(body.get("author") or "").strip() or None),
            rationale=(str(body.get("rationale") or "").strip() or None),
        )
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/turn/propose", methods=["POST"])
def hitl_turn_propose():
    body = request.get_json(force=True, silent=True) or {}
    try:
        if body.get("plan_override") is not None:
            state = _manager().call_session(
                "requery_with_plan",
                str(body.get("plan_override") or ""),
                author=(str(body.get("author") or "").strip() or None),
                rationale=(str(body.get("rationale") or "").strip() or None),
            )
        else:
            state = _manager().call_session("propose_turn")
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/turn/intervene", methods=["POST"])
def hitl_turn_intervene():
    body = request.get_json(force=True, silent=True) or {}
    try:
        state = _manager().call_session(
            "begin_intervention",
            author=(str(body.get("author") or "").strip() or None),
            rationale=(str(body.get("rationale") or "").strip() or None),
        )
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/turn/execute", methods=["POST"])
def hitl_turn_execute():
    body = request.get_json(force=True, silent=True) or {}
    try:
        state = _manager().call_session(
            "execute_turn",
            body.get("final") or {},
            author=(str(body.get("author") or "").strip() or None),
            rationale=(str(body.get("rationale") or "").strip() or None),
            override_rule_stop=bool(body.get("override_rule_stop")),
            execute_despite_skip=bool(body.get("execute_despite_skip")),
            disable_action_override=bool(body.get("disable_action_override")),
            disable_force_steps=bool(body.get("disable_force_steps")),
        )
        return jsonify({"ok": True, "state": state})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/revert", methods=["POST"])
def hitl_revert():
    try:
        return jsonify({"ok": True, "state": _manager().call_session("revert_one_turn")})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/stop", methods=["POST"])
def hitl_stop():
    try:
        return jsonify({"ok": True, "state": _manager().stop_execution()})
    except Exception as exc:
        return _error(exc)


@bp.route("/api/hitl/frame.jpg")
def hitl_frame():
    try:
        frame = _manager().require().latest_frame
        if frame is None:
            abort(404)
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=88)
        buf.seek(0)
        response = send_file(buf, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:
        if getattr(exc, "code", None) == 404:
            raise
        return _error(exc, 404)


@bp.route("/api/hitl/frame/next.jpg")
def hitl_frame_next():
    """Return one queued post-action frame, or 204 while retaining the browser's last image."""
    try:
        after = int(request.args.get("after", "-1"))
        item = _manager().require().next_frame(after)
        if item is None:
            return Response(status=204)
        sequence, frame = item
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=84)
        buf.seek(0)
        response = send_file(buf, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Sequence"] = str(sequence)
        return response
    except Exception as exc:
        return _error(exc, 404)


@bp.route("/api/hitl/media/<path:relative>")
def hitl_media(relative: str):
    try:
        base = _manager().require().store.session_dir.resolve()
        path = (base / relative).resolve()
        if path != base and base not in path.parents:
            abort(403)
        if not path.is_file():
            abort(404)
        return send_file(path, conditional=True)
    except Exception as exc:
        if getattr(exc, "code", None) in (403, 404):
            raise
        return _error(exc, 404)


@bp.route("/human-interactive")
@bp.route("/hitl")
def hitl_page():
    return Response(HITL_HTML, mimetype="text/html")


HITL_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human-interactive S2 + S1</title>
<style>
*{box-sizing:border-box}body{margin:0;font:13px/1.35 system-ui,sans-serif;color:#172033;background:#f5f7fa;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#fff;border-bottom:1px solid #dce2ea;padding:7px 14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;flex:0 0 auto}
header h1{font-size:18px;margin:0;color:#8f3518}nav a{color:#536174;text-decoration:none;margin-right:10px}nav a.hi{color:#b0431c;font-weight:700}
.selectors{display:flex;gap:8px;align-items:end;flex:1;flex-wrap:wrap}.field{display:flex;flex-direction:column;gap:3px}.field label{font-size:11px;color:#6c7788;text-transform:uppercase;letter-spacing:.04em}
select,input,textarea,button{font:inherit}select,input,textarea{border:1px solid #bfc8d5;border-radius:5px;background:#fff;padding:5px 7px}button{border:1px solid #aeb9c8;background:#fff;border-radius:6px;padding:6px 10px;cursor:pointer}button.primary{background:#b0431c;border-color:#b0431c;color:#fff;font-weight:700}button.danger{color:#a32424;border-color:#d9a0a0}button:disabled{opacity:.42;cursor:default}
#statusbar{padding:5px 14px;background:#fff;border-bottom:1px solid #dce2ea;display:flex;gap:6px;align-items:center;flex-wrap:wrap;flex:0 0 auto}.chip{border-radius:12px;padding:2px 8px;background:#e8edf3;color:#344155;font:11px ui-monospace,monospace}.chip.target{background:#fff0e8;color:#983914}.chip.ok{background:#e4f5e8;color:#18622b}.chip.bad{background:#fee9e7;color:#971f1b}
#error{display:none;margin:9px 18px 0;padding:8px 10px;background:#fee9e7;color:#7e1f1b;border:1px solid #f4aaa4;border-radius:6px;white-space:pre-wrap}
main{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:minmax(680px,.9fr) minmax(760px,1.1fr);grid-template-rows:minmax(0,1fr) auto;gap:8px;padding:8px 10px;overflow:hidden}.card{background:#fff;border:1px solid #dce2ea;border-radius:8px;padding:8px;min-width:0}.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#5d6878;margin:0 0 6px}.scene-grid{display:grid;grid-template-columns:minmax(0,768px) minmax(120px,1fr);gap:8px;align-items:start}.scene{width:100%;max-width:768px;max-height:256px;aspect-ratio:3/1;object-fit:contain;background:#111;border-radius:5px;display:block}.scene-meta{display:flex;flex-direction:column;gap:5px}.meta-box{border:1px solid #dce2ea;border-radius:5px;background:#f8fafc;padding:5px 6px}.meta-box b{display:block;font-size:9px;text-transform:uppercase;color:#778396;letter-spacing:.04em}.meta-box span{font:11px ui-monospace,monospace;color:#344155}.live-metrics{display:flex;flex-wrap:wrap;gap:4px;padding:5px;border:1px solid #dce2ea;border-radius:6px;background:#f8fafc}.metric-bubble{display:inline-flex;gap:4px;align-items:baseline;padding:3px 6px;border-radius:999px;background:#e7edf5;color:#344155;font:9.5px ui-monospace,monospace;white-space:nowrap}.metric-bubble b{font:700 8px system-ui,sans-serif;text-transform:uppercase;letter-spacing:.02em;color:#687588}.metric-bubble.progress{background:#e1edfb;color:#215f9f}.metric-bubble.motion{background:#f8eadd;color:#8c4618}.metric-bubble.grip{background:#e1f2e8;color:#216744}.metric-bubble.step{background:#ece7f7;color:#57428a}.video{width:100%;max-height:180px;background:#111;border-radius:5px;margin-top:5px}.last-rollout{font-size:11px;color:#667386}.last-rollout summary{cursor:pointer}.goal{padding:5px 7px;background:#f7f8fa;border-radius:5px;margin-bottom:6px}.editgrid{display:grid;grid-template-columns:105px 1fr;gap:6px}.editgrid label{color:#657184;padding-top:5px}.editgrid textarea{min-height:46px;resize:vertical}.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:6px}.row.push{justify-content:space-between}.muted{color:#778396;font-size:11px}.modelgrid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:8px}.layer{border:1px solid #e0e5ec;border-radius:6px;padding:7px;min-height:70px}.layer h3{font-size:11px;margin:0 0 4px;text-transform:uppercase;color:#697588}.layer pre{font:11px/1.35 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;margin:0;max-height:170px;overflow:auto}.iv{font:11px ui-monospace,monospace;color:#7a351e;border-top:1px solid #eee;padding-top:3px;margin-top:3px}
.decision-columns{display:grid;grid-template-columns:minmax(390px,1.08fr) minmax(320px,.92fr);gap:8px;min-height:0;flex:1}.plan-editor{border:1px solid #d9e0e9;border-radius:7px;background:#f7f9fc;padding:7px;margin:0;display:flex;flex-direction:column;min-height:0;overflow:hidden}.turn-editor{border:1px solid #e0e5ec;border-radius:7px;padding:7px;min-width:0;min-height:0;display:flex;flex-direction:column;overflow:hidden}.turn-title{font-size:11px;color:#596678;text-transform:uppercase;letter-spacing:.04em;font-weight:700;margin-bottom:5px}.turn-fields{min-height:0;overflow:auto;padding-right:2px}.plan-toolbar{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:5px}.plan-toolbar b{font-size:11px;color:#596678;text-transform:uppercase;letter-spacing:.04em}.plan-list{display:flex;flex-direction:column;gap:5px;flex:1;min-height:0;overflow:auto;padding-right:2px}.plan-empty{border:1px dashed #bdc7d4;border-radius:6px;padding:15px;text-align:center;color:#778396}.plan-item{border:1px solid #cfd8e4;border-left:4px solid #aab5c4;border-radius:7px;background:#fff;padding:5px}.plan-item.active{border-left-color:#d8782d;background:#fffaf4}.plan-item.done{border-left-color:#31915c;background:#f7fcf9}.plan-main{display:grid;grid-template-columns:auto auto minmax(110px,1fr) auto;gap:5px;align-items:center}.plan-id{font:700 11px ui-monospace,monospace;color:#435167;background:#edf1f6;border-radius:5px;padding:4px 6px}.plan-sentence{width:100%;min-width:70px;padding:4px 6px}.status-buttons{display:flex;border:1px solid #bdc7d4;border-radius:5px;overflow:hidden;white-space:nowrap}.status-buttons button{border:0;border-right:1px solid #d4dbe4;border-radius:0;padding:3px 5px;font-size:10px;color:#657184;background:#fff}.status-buttons button:last-child{border-right:0}.status-buttons button.selected[data-mark=" "]{background:#e8edf3;color:#263448;font-weight:700}.status-buttons button.selected[data-mark="~"]{background:#f9dfc9;color:#8b3d0d;font-weight:700}.status-buttons button.selected[data-mark="x"]{background:#dcefe3;color:#176239;font-weight:700}.item-actions{display:flex;gap:2px}.item-actions button,.fine-add{padding:2px 5px;font-size:10px}.fine-list{display:flex;flex-direction:column;gap:4px;margin:5px 0 0 22px;padding-left:7px;border-left:2px solid #e0e5ec}.fine-item{display:grid;grid-template-columns:auto auto minmax(90px,1fr) auto;gap:4px;align-items:center}.fine-item .plan-id{font-size:10px;padding:3px 5px}.fine-item .status-buttons button{padding:2px 4px}.fine-tools{display:flex;align-items:center;justify-content:space-between;margin:4px 0 0 22px}.plan-help{font-size:10px;color:#788495}.mini-danger{color:#a32424}.plan-error{display:none;color:#9a2c22;font-size:11px;margin-top:4px}
.decision-source{display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin:0 0 7px}.decision-source button{font-size:11px;padding:4px 8px}.decision-source button.selected{background:#344d70;border-color:#344d70;color:#fff;font-weight:700}.source-label{font-size:11px;color:#667386;margin-left:3px}
#s2Reference{grid-column:1/-1;padding:6px 9px;max-height:52vh;overflow:auto}#s2Reference>summary{cursor:pointer;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#5d6878;font-weight:700}.s2-detail-body{padding-top:8px}.s2-meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.ref-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.ref-panel{border:1px solid #dce2ea;border-radius:6px;background:#fafbfd;padding:8px;min-width:0}.ref-panel h3{font-size:12px;color:#596678;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em}.ref-panel details{border-top:1px solid #e2e7ed;padding-top:5px;margin-top:5px}.ref-panel summary{cursor:pointer;color:#46566d;font-weight:600;font-size:12px}.ref-panel pre{font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;max-height:260px;overflow:auto;background:#fff;border:1px solid #e0e5ec;border-radius:5px;padding:7px;margin:5px 0 0}.s2-media{width:66%;max-height:220px;object-fit:contain;background:#111;border-radius:5px;margin-top:6px}.ref-note{font-size:11px;color:#758195}
.timing-strip{margin-left:auto;display:flex;gap:3px;align-items:center;flex-wrap:nowrap;overflow-x:auto;max-width:58vw}.timing-item{font:10px ui-monospace,monospace;color:#586679;border-left:1px solid #d8dee7;padding-left:5px;white-space:nowrap}.timing-item b{color:#8b3b1d}.chip.working{background:#fff0d9;color:#8a4b0f}
#liveCard,#decisionCard{min-height:0;overflow:hidden;display:flex;flex-direction:column}.telemetry-stack{display:flex;flex-direction:column;gap:3px;margin-top:5px}.telemetry-row .cap{display:flex;justify-content:space-between;font-size:9px;color:#778396;text-transform:uppercase;letter-spacing:.04em}.telemetry-row canvas{width:100%;height:48px;display:block;border:1px solid #e1e5eb;border-radius:4px;background:#fff}.motion-legend{display:flex;gap:8px;text-transform:none;letter-spacing:0}.motion-legend i{display:inline-block;width:8px;height:2px;vertical-align:3px;margin-right:3px}.action-label{display:flex;justify-content:space-between;margin-top:5px}.action-window{border:1px solid #e1e5eb;border-radius:5px;overflow:auto;height:125px;background:#fff}.action-window table{border-collapse:collapse;width:100%;font:9.5px ui-monospace,monospace}.action-window td,.action-window th{border:1px solid #e7eaf0;padding:2px 4px;white-space:nowrap}.action-window th{position:sticky;top:0;background:#f3f5f8;z-index:1}.chunkrow.exec{background:#edf8f0}.chunkrow.at{background:#ffe7d5;box-shadow:0 0 0 1px #b0431c inset;font-weight:700}.ptr{color:#b0431c}.action-foot{font:10px ui-monospace,monospace;color:#667386;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.s1-reference{display:grid;grid-template-columns:1.5fr 1fr;gap:6px;margin-top:5px;min-height:82px}.s1-ref-panel{border:1px solid #dce2ea;border-radius:5px;background:#f8fafc;padding:5px;min-width:0}.s1-ref-panel b{display:block;font-size:9px;text-transform:uppercase;color:#778396;margin-bottom:3px}.s1-prompt{margin:0;max-height:72px;overflow:auto;white-space:pre-wrap;word-break:break-word;font:9.5px/1.3 ui-monospace,monospace}.anchor-image{width:100%;max-height:72px;object-fit:contain;background:#111;border-radius:3px}.timeline{display:flex;gap:3px;overflow:auto}.node{padding:2px 6px;border-radius:4px;background:#edf1f6;font:10px ui-monospace,monospace;white-space:nowrap}.node:last-child{background:#f6dccc;color:#7f2e14}.checks{margin-top:5px}.checks label{font-size:11px;color:#566274;margin-right:8px}.checks input{vertical-align:-2px}.override-help{font-size:10px;color:#667386;margin-top:4px}.override-help summary{cursor:pointer}.rule-hint{display:block;width:100%;text-align:left;border-color:#e3a74e;background:#fff5dc;color:#7b460b;font-weight:700;margin-bottom:6px}.s2-stream-card{border:1px solid #b9c9dc;border-radius:6px;background:#101722;color:#dbe8f7;margin-bottom:7px;min-height:120px;max-height:190px;display:flex;flex-direction:column}.s2-stream-head{display:flex;justify-content:space-between;padding:5px 7px;border-bottom:1px solid #2a394c;color:#9fb1c7;font-size:10px;text-transform:uppercase}.s2-stream-card pre{margin:0;padding:7px;white-space:pre-wrap;word-break:break-word;overflow:auto;min-height:0;font:11px/1.4 ui-monospace,monospace;flex:1}.streaming-dot{color:#58d68d}.decision-actions{margin-top:auto;padding-top:7px;border-top:1px solid #e0e5ec;display:flex;gap:5px;align-items:center;flex-wrap:wrap}.decision-actions button{padding:6px 8px}.hidden{display:none!important}@keyframes fieldShine{0%{box-shadow:0 0 0 1px #ffc44d,0 0 18px 6px rgba(255,196,77,.9);background:#fff8cf}55%{box-shadow:0 0 14px 3px rgba(255,196,77,.45)}100%{box-shadow:none;background:inherit}}.shine-update{animation:fieldShine 1.15s ease-out}.plan-editor.shine-update{animation-duration:1.4s}
@media(max-width:1500px){body{overflow:auto;height:auto}main{grid-template-columns:1fr;grid-template-rows:auto;overflow:visible}.decision-columns{grid-template-columns:1fr 1fr;height:720px}.plan-list{min-height:500px}#s2Reference{grid-column:1}.timing-strip{max-width:100%;flex-wrap:wrap}.scene-grid{grid-template-columns:minmax(0,768px) minmax(140px,1fr)}}
#liveCard{overflow:auto}.scene-grid{display:flex;flex-direction:column;width:min(100%,768px);gap:5px}.scene-meta{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:5px;height:64px;min-height:64px;max-height:64px}.scene-meta .live-metrics{display:flex;align-content:center;align-items:center;flex-wrap:wrap;gap:3px;height:64px;overflow:hidden;padding:4px}.scene-meta .metric-bubble{padding:2px 5px;font-size:9px}.scene-controls{display:flex;align-items:center;gap:5px;height:64px;white-space:nowrap}.scene-controls .row{margin-top:0;flex-wrap:nowrap}.scene-controls button{padding:4px 7px;font-size:11px}.last-rollout{position:relative}.last-rollout .video{position:absolute;z-index:6;right:0;top:20px;width:360px;max-width:70vw;border:5px solid #fff;box-shadow:0 4px 16px #26344855}.s1-reference{grid-template-columns:1fr;margin-top:6px}.detail-anchor{border:1px solid #dce2ea;border-radius:6px;background:#fafbfd;padding:8px;margin-bottom:8px}.detail-anchor h3{font-size:12px;color:#596678;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em}.detail-anchor .anchor-image{width:min(100%,768px);max-height:256px;object-fit:contain;background:#111;border-radius:5px;display:block}.s2-stream-card{height:360px;min-height:270px;max-height:48vh}.preset-bar{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:6px}.preset-bar button{padding:5px 7px;font-size:11px}.preset-bar button.selected,.decision-actions button.selected{background:#344d70;border-color:#344d70;color:#fff;font-weight:700}.preset-hint{padding:6px 8px;border:1px solid #e3a74e;border-radius:5px;background:#fff5dc;color:#7b460b;font-size:11px;margin-bottom:6px}
</style></head><body>
<header><h1>Human-interactive S2 + S1</h1><nav><a href="/">home</a><a href="/combine">combine</a><a class="hi" href="/human-interactive">interactive</a></nav>
<div class="selectors"><div class="field"><label>target split</label><select id="split"><option>atomic_seen</option><option>composite_seen</option><option>composite_unseen</option></select></div><div class="field"><label>task</label><select id="task"></select></div><div class="field"><label>official episode</label><select id="episode"></select></div><div class="field"><label>operator</label><input id="operator" placeholder="name"></div><button class="primary" id="start">Start target episode</button></div></header>
<div id="statusbar"><span class="chip" id="stage">not loaded</span><span class="chip target" id="target">official target</span><span class="chip" id="turn">turn –</span><span class="chip" id="steps">steps –</span><span class="chip" id="success">success –</span><span class="chip" id="latencyNow">idle</span><span class="muted" id="saved"></span><span class="timing-strip" id="timingStrip"></span></div><div id="error"></div>
<main><section class="card" id="liveCard"><h2>Live System1 execution</h2><div class="scene-grid"><img class="scene" id="frame"><div class="scene-meta"><div class="live-metrics" id="s1Metrics"><span class="metric-bubble">Waiting for executed action</span></div><div class="scene-controls"><details class="last-rollout"><summary>Last video</summary><video class="video hidden" id="video" controls></video></details><div class="row"><button id="stop" class="danger" disabled>Stop</button><button id="revert" disabled>↶ Revert</button></div></div></div></div>
<div class="s1-reference"><div class="s1-ref-panel"><b>Actual System1 language prompt · updates each replan</b><pre class="s1-prompt" id="s1Prompt">Waiting for the first System1 query.</pre></div></div><div class="telemetry-stack"><div class="telemetry-row"><div class="cap"><span>Progress</span><span id="progressValue">–</span></div><canvas id="curveProgress"></canvas></div><div class="telemetry-row"><div class="cap"><span>Commanded motion</span><span class="motion-legend"><span><i style="background:#d8792d"></i>EEF pos</span><span><i style="background:#7a5cc7"></i>EEF rot</span><span><i style="background:#199bb2"></i>base</span><span><i style="background:#263d64"></i>|Δa|</span></span><span id="motionValue">–</span></div><canvas id="curveMotion"></canvas></div><div class="telemetry-row"><div class="cap"><span>Gripper width</span><span id="gripperValue">–</span></div><canvas id="curveGripper"></canvas></div></div><div class="action-label"><b class="plan-help">Latest predicted System1 action chunk</b><span class="plan-help">green = executed window · orange = current action</span></div><div class="action-window" id="actionChunk"><div class="plan-empty">Waiting for the first System1 prediction.</div></div><div class="action-foot" id="actionFoot">No executed action yet.</div><div class="muted" id="liveText">Waiting for a target episode.</div></section>
<section class="card" id="decisionCard"><h2>Human-final decision</h2><div class="goal"><b>Task goal:</b> <span id="goal">–</span></div><button class="rule-hint hidden" id="ruleHint" type="button"></button><div class="decision-columns"><div class="plan-editor" id="planEditor"><div class="plan-toolbar"><div><b>Plan checklist</b><div class="plan-help">Edit sentences only; IDs and syntax are automatic.</div></div><button id="addMilestone" type="button">＋ Milestone</button></div><div class="decision-source"><button id="useS2" type="button">S2 prediction</button><button id="useRules" type="button">Rule-adjusted</button><span class="source-label" id="editorSource">Waiting for S2.</span></div><div class="plan-list" id="planCards"><div class="plan-empty">Plan appears after System2 planning.</div></div><div class="plan-error" id="planError"></div></div>
<div class="turn-editor"><div class="turn-title">Judge &amp; subgoal</div><div class="s2-stream-card"><div class="s2-stream-head"><span>Live System2 response</span><span id="s2StreamState">idle</span></div><pre id="s2Stream">System2 output will stream here token by token.</pre></div><div class="preset-bar"><button id="presetContinue" type="button">Continue last subgoal</button><button id="presetRedo" type="button">Redo last subgoal</button><button id="presetCurrent" class="selected" type="button">Use current subgoal</button><button id="presetNext" type="button">Step next subgoal</button></div><div class="preset-hint hidden" id="presetHint"></div><div class="turn-fields"><div class="editgrid"><label>Judge</label><select id="judge"><option value="">none</option><option>task_begin</option><option>subgoal_complete</option><option>subgoal_incomplete</option><option>subgoal_failed</option><option>task_finish</option></select><label>Subgoal</label><input id="subgoal"><label class="hidden">Subgoal detail</label><textarea class="hidden" id="detail"></textarea><label>Estimated length</label><input id="est" type="number" min="1" step="1"><label>Edit rationale</label><textarea id="rationale" placeholder="Why did you intervene?"></textarea></div></div><div class="decision-actions"><button class="primary" id="commit" disabled>Execute</button><button id="intervene" type="button" disabled>Intervene</button><button id="ask" disabled>Ask S2 for next subgoal</button><button id="replan" disabled>Regenerate plan</button><span class="muted" id="control"></span></div></div></div></section>
<details class="card" id="s2Reference"><summary>More details</summary><div class="s2-detail-body"><div class="modelgrid"><div class="layer"><h3>Raw System2 prediction</h3><pre id="raw">–</pre></div><div class="layer"><h3>Rule-adjusted prediction</h3><pre id="ruled">–</pre><div id="interventions"></div></div></div><div class="s2-meta" id="s2Meta"><span class="ref-note">No System2 prediction yet.</span></div><div class="detail-anchor"><h3>Turn anchor image</h3><img class="anchor-image hidden" id="s1Anchor"><span class="muted" id="s1AnchorEmpty">Available when execution begins.</span></div><div class="ref-grid"><div class="ref-panel"><h3>Exact input to System2</h3><details><summary>System prompt</summary><pre id="s2System">–</pre></details><details open><summary>User prompt</summary><pre id="s2User">–</pre></details><div id="s2MediaWrap" class="hidden"><div class="ref-note">Visual input</div><img class="s2-media hidden" id="s2Image"><video class="s2-media hidden" id="s2Video" controls></video></div></div><div class="ref-panel"><h3>Exact output from System2</h3><details open><summary>Raw response</summary><pre id="s2Raw">–</pre></details><details open><summary>Parsed prediction used to populate the editor</summary><pre id="s2Parsed">–</pre></details></div></div></div></details></main>
<script>
const $=s=>document.querySelector(s);let catalog=[],state=null,lastProposal=null,lastNode=null,pollTimer=null,planItems=[],operationTimer=null,operationStarted=0,operationLabel='',frameTimer=null,frameBusy=false,frameSequence=-1,frameSession=null,frameObjectUrl=null,lastChunkKey='',lastStreamId=-1,lastStreamText='',streamFields={},needsS2Advance=false,currentPreset='current';const clientTimings={page_s:null,catalog_s:null};
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function api(url,opt={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opt});const d=await r.json();if(!r.ok||d.ok===false)throw Error(d.error||r.statusText);return d}
function err(e){$('#error').style.display='block';$('#error').textContent=e.message||e}function clearErr(){$('#error').style.display='none'}
function seconds(x,digits=2){return Number.isFinite(+x)?`${(+x).toFixed(digits)} s`:'–'}
function beginOperation(label){clearInterval(operationTimer);operationLabel=label;operationStarted=performance.now();const tick=()=>{$('#latencyNow').textContent=`${operationLabel} · ${((performance.now()-operationStarted)/1000).toFixed(1)} s`;$('#latencyNow').className='chip working'};tick();operationTimer=setInterval(tick,100)}
function endOperation(){clearInterval(operationTimer);operationTimer=null;$('#latencyNow').textContent='idle';$('#latencyNow').className='chip'}
function numberStats(values){const a=(values||[]).filter(v=>v!==null&&v!==undefined&&v!=='').map(Number).filter(Number.isFinite);return a.length?{n:a.length,total:a.reduce((x,y)=>x+y,0),mean:a.reduce((x,y)=>x+y,0)/a.length,last:a.at(-1)}:{n:0,total:null,mean:null,last:null}}
function flash(el){if(!el)return;el.classList.remove('shine-update');void el.offsetWidth;el.classList.add('shine-update')}
function timingItem(label,value,title){return `<span class="timing-item" title="${esc(title)}"><b>${esc(label)}</b> ${esc(value)}</span>`}
function renderTimings(s){const startup=s?.timings?.startup||{},ref=s?.proposal||s?.last_s2||{},model=ref.model||{},live=s?.live||{},s1=numberStats(live.s1_infer_seconds),step=numberStats(live.step_seconds),render=numberStats(live.env_render_seconds),stream=numberStats(live.stream_render_seconds);const items=[timingItem('page',seconds(clientTimings.page_s,3),'HTML response time'),timingItem('catalog',seconds(clientTimings.catalog_s),'Cold load includes RoboCasa/MuJoCo imports'),timingItem('reset',seconds(startup.episode_load_reset_s),'Episode state/XML, environment construction and reset'),timingItem('S2',seconds(ref.query_seconds??model.latency_s),`Model request ${seconds(model.latency_s)} including total prompt/media/rules overhead`),timingItem('S1',s1.n?seconds(s1.mean):'–',s1.n?`${s1.n} calls; latest ${seconds(s1.last)}; total ${seconds(s1.total)}`:'Mean System1 policy latency'),timingItem('step avg',step.n?seconds(step.mean,3):'–',step.n?`${step.n} env steps; regular render ${seconds(render.mean,3)}; stream render ${seconds(stream.mean,3)}`:'Average env.step time'),timingItem('turn',seconds(live.turn_wall_seconds??live.rollout_wall_seconds),`phase ${live.phase||'idle'}; save/encode ${seconds(live.postprocess_seconds)}`)];$('#timingStrip').innerHTML=items.join('')}
function planMark(mark){return String(mark).toLowerCase()==='x'?'x':mark==='~'?'~':' '}
function cleanPlanSentence(line){return line.trim().replace(/^[-*\d.)\s]+/,'').replace(/^\[[ xX~]\]\s*/,'').replace(/^M\d+(?:\.\d+)?\s*:\s*/,'').trim()}
function parsePlan(text){const blocks=[];let current=null;for(const raw of String(text||'').split(/\r?\n/)){if(!raw.trim())continue;let m=raw.match(/^\s*-\s*\[(.)\]\s*(M\d+)\s*:\s*(.*)$/);if(m){current={mark:planMark(m[1]),text:m[3].trim(),fine:[]};blocks.push(current);continue}m=raw.match(/^\s*\*\s*\[(.)\]\s*M\d+\.(\d+)\s*:\s*(.*)$/);if(m&&current){current.fine.push({mark:planMark(m[1]),text:m[3].trim()});continue}const sentence=cleanPlanSentence(raw);if(!sentence)continue;if(current){const target=current.fine.at(-1)||current;target.text=(target.text+' '+sentence).trim()}else{current={mark:' ',text:sentence,fine:[]};blocks.push(current)}}return blocks}
function renumberPlan(){planItems.forEach((m,i)=>{m.mid=`M${i+1}`;m.fine.forEach((f,j)=>f.fid=`M${i+1}.${j+1}`)})}
function statusButtons(mark,kind,mi,fi=''){return `<span class="status-buttons" aria-label="status"><button type="button" data-action="status" data-kind="${kind}" data-mi="${mi}" data-fi="${fi}" data-mark=" " class="${mark===' '?'selected':''}" title="To do">To do</button><button type="button" data-action="status" data-kind="${kind}" data-mi="${mi}" data-fi="${fi}" data-mark="~" class="${mark==='~'?'selected':''}" title="In progress">Active</button><button type="button" data-action="status" data-kind="${kind}" data-mi="${mi}" data-fi="${fi}" data-mark="x" class="${mark==='x'?'selected':''}" title="Done">Done</button></span>`}
function renderPlanEditor(focusKey=''){renumberPlan();const root=$('#planCards');if(!planItems.length){root.innerHTML='<div class="plan-empty">No milestones. Click “Add milestone” to create one.</div>';return}root.innerHTML=planItems.map((m,mi)=>`<div class="plan-item ${m.mark==='x'?'done':m.mark==='~'?'active':''}"><div class="plan-main"><span class="plan-id">${m.mid}</span>${statusButtons(m.mark,'milestone',mi)}<input class="plan-sentence" data-action="sentence" data-kind="milestone" data-mi="${mi}" value="${esc(m.text)}" placeholder="Describe this milestone"><span class="item-actions"><button type="button" data-action="up" data-kind="milestone" data-mi="${mi}" title="Move up">↑</button><button type="button" data-action="down" data-kind="milestone" data-mi="${mi}" title="Move down">↓</button><button type="button" class="mini-danger" data-action="delete" data-kind="milestone" data-mi="${mi}" title="Delete milestone">✕</button></span></div><div class="fine-list">${m.fine.map((f,fi)=>`<div class="fine-item"><span class="plan-id">${f.fid}</span>${statusButtons(f.mark,'fine',mi,fi)}<input class="plan-sentence" data-action="sentence" data-kind="fine" data-mi="${mi}" data-fi="${fi}" value="${esc(f.text)}" placeholder="Describe this fine step"><span class="item-actions"><button type="button" data-action="up" data-kind="fine" data-mi="${mi}" data-fi="${fi}" title="Move up">↑</button><button type="button" data-action="down" data-kind="fine" data-mi="${mi}" data-fi="${fi}" title="Move down">↓</button><button type="button" class="mini-danger" data-action="delete" data-kind="fine" data-mi="${mi}" data-fi="${fi}" title="Delete fine step">✕</button></span></div>`).join('')}</div><div class="fine-tools"><span class="plan-help">Optional execution steps within ${m.mid}</span><button type="button" class="fine-add" data-action="add-fine" data-mi="${mi}">＋ Add fine step</button></div></div>`).join('');if(focusKey){const input=root.querySelector(focusKey);if(input){input.focus();input.select()}}}
function loadPlanEditor(text,shine=false){const before=JSON.stringify(planItems),next=parsePlan(text);planItems=next;renderPlanEditor();$('#planError').style.display='none';if(shine&&JSON.stringify(next)!==before)flash($('#planEditor'))}
function planText(){renumberPlan();const missing=[];const lines=[];planItems.forEach((m,mi)=>{m.text=m.text.trim();if(!m.text)missing.push(`M${mi+1}`);lines.push(`- [${m.mark}] M${mi+1}: ${m.text}`);m.fine.forEach((f,fi)=>{f.text=f.text.trim();if(!f.text)missing.push(`M${mi+1}.${fi+1}`);lines.push(`  * [${f.mark}] M${mi+1}.${fi+1}: ${f.text}`)})});if(!lines.length)throw Error('Add at least one plan milestone.');if(missing.length)throw Error(`Add a sentence for ${missing.join(', ')}.`);return lines.join('\n')}
function moveItem(items,index,delta){const to=index+delta;if(to<0||to>=items.length)return;[items[index],items[to]]=[items[to],items[index]]}
function editPlan(e){const t=e.target,a=t.dataset.action;if(!a)return;const mi=+t.dataset.mi,fi=t.dataset.fi===''||t.dataset.fi===undefined?null:+t.dataset.fi,kind=t.dataset.kind;if(a==='sentence'){const item=kind==='fine'?planItems[mi]?.fine[fi]:planItems[mi];if(item)item.text=t.value;return}e.preventDefault();if(a==='status'){const item=kind==='fine'?planItems[mi]?.fine[fi]:planItems[mi];if(item)item.mark=t.dataset.mark;renderPlanEditor();return}if(a==='add-fine'){planItems[mi].fine.push({mark:' ',text:''});renderPlanEditor(`[data-kind="fine"][data-mi="${mi}"][data-fi="${planItems[mi].fine.length-1}"][data-action="sentence"]`);return}const items=kind==='fine'?planItems[mi].fine:planItems;if(a==='delete'){items.splice(kind==='fine'?fi:mi,1)}else if(a==='up'){moveItem(items,kind==='fine'?fi:mi,-1)}else if(a==='down'){moveItem(items,kind==='fine'?fi:mi,1)}renderPlanEditor()}
function fillTasks(){const split=$('#split').value,ts=catalog.filter(x=>x.task_split===split&&x.available);$('#task').innerHTML=ts.map(x=>`<option>${esc(x.task_name)}</option>`).join('');fillEpisodes()}
function fillEpisodes(){const t=catalog.find(x=>x.task_name===$('#task').value);$('#episode').innerHTML=(t?.episodes||[]).map(e=>`<option value="${e.episode_index}">#${e.manifest_rank+1} · episode_${String(e.episode_index).padStart(6,'0')}</option>`).join('')}
function setBusy(v,msg){$('#start').disabled=v;if(msg)$('#stage').textContent=msg}
async function start(){clearErr();setBusy(true,'loading reset + planning…');beginOperation('Loading episode + initial S2 plan');state=null;beginPoll();try{const d=await api('/api/hitl/session',{method:'POST',body:JSON.stringify({task_name:$('#task').value,episode_index:+$('#episode').value,operator:$('#operator').value})});render(d.state);beginPoll()}catch(e){err(e)}finally{setBusy(false);endOperation()}}
function predictionValues(p){if(!p)return {};if(p.prediction)return p.prediction;const m=p.model||{};return p.kind==='plan'?{plan:m.plan}:{plan:p.effective?.plan??m.plan_in,judge:m.judge,subgoal:m.subgoal,subgoal_detail:m.subgoal_detail,estimated_step:m.estimated_step}}
function proposalText(p){if(!p)return '–';const m=p.model||{};return JSON.stringify({thought:m.thought,...predictionValues(p),plan_update:m.plan_update},null,2)}
function effectiveText(p){return p?JSON.stringify(p.effective||{},null,2):'–'}
function setEditorValue(id,value,shine=true){const el=$(id),next=value??'';if(String(el.value)!==String(next)){el.value=next;if(shine)flash(el)}}
function loadDecision(values,source){const e=values||{};loadPlanEditor(e.plan??state?.plan??'',true);setEditorValue('#judge',e.judge);setEditorValue('#subgoal',e.subgoal);setEditorValue('#detail',e.subgoal_detail);setEditorValue('#est',e.estimated_step);$('#editorSource').textContent=source==='s2'?'Editor loaded from raw S2 prediction.':source==='rules'?'Editor loaded from rule-adjusted prediction.':'Free intervention; the checklist stays unchanged unless you edit it here.';$('#useS2').classList.toggle('selected',source==='s2');$('#useRules').classList.toggle('selected',source==='rules')}
function selectPreset(name){currentPreset=name;for(const [id,n] of [['#presetContinue','continue'],['#presetRedo','redo'],['#presetCurrent','current'],['#presetNext','next'],['#intervene','intervene']])$(id).classList.toggle('selected',n===name)}
function presetHint(message=''){const el=$('#presetHint');el.textContent=message;el.classList.toggle('hidden',!message)}
function loadEditor(p){if(!p)return;needsS2Advance=false;if(p.human_intervention){selectPreset('intervene');presetHint('Free intervention mode: type any subgoal and estimated length, then Execute. The plan changes only if you edit the checklist directly.');loadDecision(predictionValues(p),'intervention')}else{selectPreset('current');presetHint();loadDecision(predictionValues(p),'s2')}$('#rationale').value=''}
function chooseDecisionSource(source){const p=state?.proposal;if(!p)return;needsS2Advance=false;selectPreset('current');presetHint();loadDecision(source==='rules'?(p.effective||{}):predictionValues(p),source)}
function normStep(text){return String(text||'').toLowerCase().replace(/^(continue to|continue|redo)\s+/,'').replace(/\s+again$/,'').replace(/[^a-z0-9]+/g,' ').trim()}
function locatePlanStep(text,activeOnly=false){const needle=normStep(text),hits=[];planItems.forEach((m,mi)=>{m.fine.forEach((f,fi)=>{if(!activeOnly||f.mark==='~')hits.push({mi,fi,item:f,milestone:m,score:normStep(f.text)===needle?2:(needle&&(normStep(f.text).includes(needle)||needle.includes(normStep(f.text)))?1:0)})});if(!m.fine.length&&(!activeOnly||m.mark==='~'))hits.push({mi,fi:null,item:m,milestone:m,score:normStep(m.text)===needle?2:0})});return hits.sort((a,b)=>b.score-a.score).find(x=>x.score>0)||hits.find(x=>x.item.mark==='~')||null}
function activatePreviousStep(lastSubgoal){const hit=locatePlanStep(lastSubgoal);if(!hit||hit.score===0)return false;planItems.forEach((m,mi)=>{if(mi!==hit.mi&&m.mark==='~')m.mark=' ';m.fine.forEach((f,fi)=>{if(f.mark==='~'&&(mi!==hit.mi||fi!==hit.fi))f.mark=' '})});hit.item.mark='~';hit.milestone.mark='~';renderPlanEditor();flash($('#planEditor'));return true}
function applyPreset(name){const p=state?.proposal;if(!p)return;needsS2Advance=false;presetHint();loadDecision(predictionValues(p),'s2');selectPreset(name);const previous=state.previous_turn||{},last=previous.subgoal||'';if(name==='current')return;if(name==='continue'||name==='redo'){if(!last||!activatePreviousStep(last)){presetHint('No previously executed subgoal could be found in this plan.');selectPreset('current');return}const base=last.replace(/^continue(?: to)?\s+/i,'').replace(/\s+again$/i,'').trim(),next=name==='continue'?`continue to ${base}`:`${base} again`;setEditorValue('#judge',name==='continue'?'subgoal_incomplete':'subgoal_failed');setEditorValue('#subgoal',next);setEditorValue('#detail',next,false);if(previous.estimated_step)setEditorValue('#est',previous.estimated_step);presetHint(name==='continue'?'The previous fine step is active again and the newer step is back to todo.':'The previous fine step is active for a fresh recovery attempt.');return}const raw=predictionValues(p),hit=locatePlanStep(raw.subgoal,true)||locatePlanStep(raw.subgoal);if(!hit){presetHint('Could not identify the current fine step in the checklist.');selectPreset('current');return}hit.item.mark='x';let next=null;if(hit.fi!==null){for(let fi=hit.fi+1;fi<hit.milestone.fine.length;fi++){if(hit.milestone.fine[fi].mark!=='x'){next=hit.milestone.fine[fi];break}}}setEditorValue('#judge','subgoal_complete');if(next){next.mark='~';hit.milestone.mark='~';setEditorValue('#subgoal',next.text);setEditorValue('#detail',next.text,false);presetHint(`Advanced to the next planned fine step: ${next.text}`)}else{hit.milestone.fine.forEach(f=>f.mark='x');hit.milestone.mark='x';setEditorValue('#subgoal','');setEditorValue('#detail','',false);needsS2Advance=true;presetHint('This was the last fine step in the milestone. The milestone is marked done; click “Ask S2 for next subgoal” to unroll the next milestone.')}renderPlanEditor();flash($('#planEditor'));$('#ask').disabled=!needsS2Advance&&state.stage!=='ready'}
function partialTag(text,tag){const open=`<${tag}>`,i=String(text||'').lastIndexOf(open);if(i<0)return null;const tail=String(text).slice(i+open.length),j=tail.indexOf(`</${tag}>`);return (j<0?tail:tail.slice(0,j)).trim()}
function applyPlanUpdateText(plan,update){const blocks=text=>{const out=[];for(const line of String(text||'').split(/\r?\n/)){const m=line.match(/^\s*-\s*\[.\]\s*(M\d+)\s*:/);if(m)out.push([m[1],[line]]);else if(out.length&&line.trim())out.at(-1)[1].push(line)}return out},cur=blocks(plan),upd=new Map(blocks(update));if(!upd.size)return plan;const seen=new Set(),lines=[];for(const [id,part] of cur){if(upd.has(id)){lines.push(...upd.get(id));seen.add(id)}else lines.push(...part)}for(const [id,part] of upd)if(!seen.has(id))lines.push(...part);return lines.join('\n')}
function renderS2Stream(stream,s){const st=stream||{},id=Number(st.request_id??-1),text=st.text||'',pre=$('#s2Stream'),label=$('#s2StreamState');label.textContent=st.status==='streaming'?`● streaming · ${st.kind||''}`:st.status||'idle';label.className=st.status==='streaming'?'streaming-dot':'';if(id!==lastStreamId){lastStreamId=id;lastStreamText='';streamFields={}}if(text!==lastStreamText){lastStreamText=text;pre.textContent=text||'Waiting for the first token…';pre.scrollTop=pre.scrollHeight}else if(!text&&st.status==='idle')pre.textContent='System2 output will stream here token by token.';if(st.status!=='streaming'||!text)return;const update=(name,value,id)=>{if(value===null||value===''||streamFields[name]===value)return;streamFields[name]=value;setEditorValue(id,value,true)},kind=st.kind||'';if(kind==='cold_plan'){const p=partialTag(text,'plan');if(p&&/^\s*-\s*\[./m.test(p)&&streamFields.plan!==p){streamFields.plan=p;loadPlanEditor(p,true)}}else{const judge=partialTag(text,'judge'),valid=['task_begin','subgoal_complete','subgoal_incomplete','subgoal_failed','task_finish'];if(valid.includes(judge))update('judge',judge,'#judge');const est=partialTag(text,'estimated_step'),n=est?.match(/-?\d+/)?.[0];if(n)update('estimated_step',n,'#est');update('subgoal',partialTag(text,'subgoal'),'#subgoal');update('subgoal_detail',partialTag(text,'subgoal_detail'),'#detail');const pu=partialTag(text,'plan_update');if(pu&&/^\s*-\s*\[./m.test(pu)&&streamFields.plan_update!==pu){streamFields.plan_update=pu;loadPlanEditor(applyPlanUpdateText(s.plan||'',pu),true)}}}
function mediaURL(path){return '/api/hitl/media/'+String(path).split('/').map(encodeURIComponent).join('/')}
async function pumpFrame(){if(frameBusy||!state?.session_id)return;if(state.stage!=='executing'&&frameSequence>=Number(state.frame_version||0))return;frameBusy=true;try{const r=await fetch(`/api/hitl/frame/next.jpg?after=${frameSequence}`,{cache:'no-store'});if(r.status===204)return;if(!r.ok)return;const seq=Number(r.headers.get('X-Frame-Sequence'));const blob=await r.blob(),url=URL.createObjectURL(blob),img=$('#frame'),old=frameObjectUrl;frameObjectUrl=url;if(Number.isFinite(seq))frameSequence=seq;img.onload=()=>{if(old)URL.revokeObjectURL(old)};img.src=url}catch(e){console.warn('frame stream',e)}finally{frameBusy=false}}
function f3(v){return Number.isFinite(+v)?(+v).toFixed(3):'–'}
function renderActionChunk(l){const q=l.last_chunk||{},rows=q.chunk_raw12||[],off=Number.isFinite(+l.chunk_offset)?+l.chunk_offset:-1,key=`${l.chunk_replan_step}:${off}:${rows.length}`;if(key===lastChunkKey)return;lastChunkKey=key;const root=$('#actionChunk');if(!rows.length){root.innerHTML='<div class="plan-empty">Waiting for the first System1 prediction.</div>';$('#actionFoot').textContent='No executed action yet.';return}const motions=q.chunk_motion||[],eligible=+q.replan_steps||0;root.innerHTML=`<table><tr><th></th><th>i</th><th>|Δa|</th><th>eef position xyz</th><th>eef rotation rpy</th><th>grip</th><th>base</th></tr>${rows.map((a,i)=>`<tr class="chunkrow${i<eligible?' exec':''}${i===off?' at':''}" ${i===off?'id="liveChunkRow"':''}><td class="ptr">${i===off?'▶':''}</td><td>${i}</td><td>${f3(motions[i])}</td><td>${(a||[]).slice(0,3).map(f3).join(' ')}</td><td>${(a||[]).slice(3,6).map(f3).join(' ')}</td><td>${f3((a||[])[6])}</td><td>${(a||[]).slice(7,11).map(f3).join(' ')}</td></tr>`).join('')}</table>`;const at=$('#liveChunkRow');if(at)at.scrollIntoView({block:'nearest'});const applied=l.last_step?.action_applied_raw12||[];$('#actionFoot').textContent=`replan @ step ${l.chunk_replan_step??'–'} · pointer +${off<0?'–':off} · executed action: ${applied.map(f3).join(' ')||'–'}`}
function renderS2Reference(p){const m=p?.model||{};if(!p){$('#s2Meta').innerHTML='<span class="ref-note">No System2 prediction yet.</span>';$('#s2System').textContent='–';$('#s2User').textContent='–';$('#s2Raw').textContent='–';$('#s2Parsed').textContent='–';$('#s2MediaWrap').classList.add('hidden');return}const meta=[p.kind==='plan'?'cold plan':`execution turn ${p.turn??m.turn??'–'}`,m.held_by_rule?`S2 skipped · held by ${m.held_by_rule}`:'S2 queried',m.latency_s!=null?`${Number(m.latency_s).toFixed(2)} s`:null,m.usage?`usage ${JSON.stringify(m.usage)}`:null].filter(Boolean);$('#s2Meta').innerHTML=meta.map((x,i)=>`<span class="chip ${i===1&&m.held_by_rule?'bad':''}">${esc(x)}</span>`).join('');$('#s2System').textContent=m.system_prompt||'–';$('#s2User').textContent=m.user_prompt||(m.held_by_rule?'System2 was not queried for this rule-held continuation.':'–');$('#s2Raw').textContent=m.response_raw||(m.held_by_rule?'No raw response: the rule resumed a held System2 turn.':'–');$('#s2Parsed').textContent=JSON.stringify(m.parsed_prediction||predictionValues(p),null,2);const wrap=$('#s2MediaWrap'),img=$('#s2Image'),vid=$('#s2Video');img.classList.add('hidden');vid.classList.add('hidden');vid.pause();vid.removeAttribute('src');img.removeAttribute('src');if(m.media){wrap.classList.remove('hidden');const url=mediaURL(m.media);if(m.media_kind==='video'||/\.mp4$/i.test(m.media)){vid.src=url;vid.classList.remove('hidden')}else{img.src=url;img.classList.remove('hidden')}}else wrap.classList.add('hidden')}
function render(s){state=s;if(frameSession!==s.session_id){frameSession=s.session_id;frameSequence=-1;lastChunkKey='';lastStreamId=-1;lastStreamText='';streamFields={}}$('#stage').textContent=s.stage;$('#turn').textContent=`turn ${s.turn_index}`;$('#steps').textContent=`steps ${s.episode_steps}`;$('#success').textContent=`success ${s.env_success?'YES':'no'}`;$('#success').className='chip '+(s.env_success?'ok':'');$('#target').textContent=`target · ${s.episode.task_split} · ${s.episode.task_name} · ep${s.episode.episode_index}`;$('#saved').textContent=s.session_dir;$('#goal').textContent=s.instruction;renderS2Stream(s.s2_stream,s);
 const p=s.proposal,freePlay=!!p?.human_intervention,detail=(freePlay?s.last_s2:p)||s.last_s2,ivs=freePlay?[]:(p?.rules?.interventions||[]);$('#raw').textContent=proposalText(detail);$('#ruled').textContent=effectiveText(detail);$('#interventions').innerHTML=(detail?.rules?.interventions||[]).map(i=>`<div class="iv">${esc(i.rule)} · ${esc(i.kind)}: ${esc(i.before)} → ${esc(i.after)}</div>`).join('');const rh=$('#ruleHint');if(ivs.length){rh.textContent=`⚡ ${ivs.length} rule adjustment${ivs.length===1?'':'s'} matched — click to preview the rule-adjusted decision`;rh.title=ivs.map(i=>`${i.rule}: ${i.kind}`).join('\n');rh.classList.remove('hidden')}else rh.classList.add('hidden');$('#control').textContent=freePlay?'human intervention · System2 and rules bypassed':'';
 const pid=p?.attempt_id||null;if(pid!==lastProposal){lastProposal=pid;if(p)loadEditor(p);else if(s.plan)loadPlanEditor(s.plan)}
 const review=s.stage==='plan_review'||s.stage==='turn_review',turnReview=s.stage==='turn_review';$('#commit').disabled=!review||needsS2Advance;$('#commit').textContent=s.stage==='plan_review'?'Accept plan':'Execute';$('#ask').disabled=!(s.stage==='ready'||(turnReview&&needsS2Advance));$('#intervene').disabled=!(s.stage==='ready'||turnReview);$('#replan').disabled=!['ready_for_plan','plan_review'].includes(s.stage);$('#revert').disabled=!s.can_revert||['executing','reverting'].includes(s.stage);$('#stop').disabled=s.stage!=='executing';for(const id of ['#presetContinue','#presetRedo','#presetCurrent','#presetNext'])$(id).disabled=!turnReview||freePlay;$('#presetContinue').disabled=!turnReview||freePlay||!s.previous_turn?.subgoal;$('#presetRedo').disabled=!turnReview||freePlay||!s.previous_turn?.subgoal;
 $('#useS2').disabled=!p||freePlay;$('#useRules').disabled=!p||freePlay;renderS2Reference(detail);renderTimings(s);if(!operationTimer&&s.stage==='executing'&&s.live?.started_at){$('#latencyNow').textContent=`${s.live.phase||'System1 execution'} · ${Math.max(0,Date.now()/1000-s.live.started_at).toFixed(1)} s`;$('#latencyNow').className='chip working'}else if(!operationTimer&&s.stage!=='executing'){endOperation()}
 const live=s.live||{};$('#liveText').textContent=s.stage==='executing'?`Executing ${live.executed}/${live.budget} steps`:(s.stage==='reverting'?`Replaying ${live.replay_progress||''}`:`${s.stage}; branch depth ${s.lineage.length}`);$('#s1Prompt').textContent=live.s1_language_prompt||live.last_chunk?.prompt||'Waiting for the first System1 query.';const anchor=live.anchor_media,ai=$('#s1Anchor'),ae=$('#s1AnchorEmpty');if(anchor){if(ai.dataset.path!==anchor){ai.dataset.path=anchor;ai.src=mediaURL(anchor)}ai.classList.remove('hidden');ae.classList.add('hidden')}else{ai.classList.add('hidden');ae.classList.remove('hidden')}
 if(s.active_node_id!==lastNode){lastNode=s.active_node_id;const via=s.lineage.at(-1)?.via_attempt_id,v=$('#video');if(via){v.src=`/api/hitl/media/attempts/${via}/execution/s1_rollout_raw.mp4`;v.classList.remove('hidden')}else{v.classList.add('hidden');v.removeAttribute('src')}}
 if(s.error)err({message:s.error});drawAll(s.live||{})}
async function poll(){try{const d=await api('/api/hitl/state');render(d.state)}catch(e){if(state)console.warn(e)}const querying=state?.stage?.startsWith('querying_s2');pollTimer=setTimeout(poll,querying||!state?150:state?.stage==='executing'?200:1000)}function beginPoll(){clearTimeout(pollTimer);poll()}
async function commit(){clearErr();$('#planError').style.display='none';try{const plan=planText();if(state.stage==='plan_review'){const d=await api('/api/hitl/plan/accept',{method:'POST',body:JSON.stringify({plan,author:$('#operator').value,rationale:$('#rationale').value})});render(d.state)}else{const final={plan,judge:$('#judge').value,subgoal:$('#subgoal').value,subgoal_detail:$('#detail').value,estimated_step:$('#est').value};const d=await api('/api/hitl/turn/execute',{method:'POST',body:JSON.stringify({final,author:$('#operator').value,rationale:$('#rationale').value,override_rule_stop:true,execute_despite_skip:true})});render(d.state)}}catch(e){$('#planError').textContent=e.message||e;$('#planError').style.display='block';err(e)}}
async function ask(){clearErr();$('#ask').disabled=true;$('#stage').textContent='querying System2…';beginOperation('Waiting for S2 turn prediction');const advancing=needsS2Advance;try{const body=advancing?{plan_override:planText(),author:$('#operator').value,rationale:$('#rationale').value}:{};const d=await api('/api/hitl/turn/propose',{method:'POST',body:JSON.stringify(body)});needsS2Advance=false;render(d.state)}catch(e){needsS2Advance=advancing;err(e)}finally{endOperation()}}
async function intervene(){clearErr();$('#intervene').disabled=true;try{const d=await api('/api/hitl/turn/intervene',{method:'POST',body:JSON.stringify({author:$('#operator').value,rationale:$('#rationale').value})});render(d.state);$('#subgoal').focus()}catch(e){err(e)}}
async function replan(){clearErr();beginOperation('Waiting for S2 plan');try{const d=await api('/api/hitl/plan/propose',{method:'POST',body:'{}'});render(d.state)}catch(e){err(e)}finally{endOperation()}}
async function revert(){if(!confirm('Revert one complete turn and restore its beginning state? The old attempt remains saved as a superseded branch.'))return;clearErr();try{const d=await api('/api/hitl/revert',{method:'POST',body:'{}'});render(d.state)}catch(e){err(e)}}
async function stop(){try{await api('/api/hitl/stop',{method:'POST',body:'{}'})}catch(e){err(e)}}
function canvas(id){const c=$(id),r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const x=c.getContext('2d');x.scale(d,d);return [x,r.width,r.height]}
function plotOne(id,values,color,fixedRange=null){const [x,w,h]=canvas(id),vals=(values||[]).map(v=>v==null?null:+v),finite=vals.filter(Number.isFinite);x.clearRect(0,0,w,h);if(!finite.length){x.fillStyle='#8993a2';x.font='10px system-ui';x.fillText('Waiting for executed steps',10,17);return}let lo=fixedRange?fixedRange[0]:Math.min(...finite),hi=fixedRange?fixedRange[1]:Math.max(...finite);if(hi===lo){const p=Math.max(.001,Math.abs(hi)*.1);hi+=p;lo-=p}const pad={l:38,r:7,t:5,b:8},gw=w-pad.l-pad.r,gh=h-pad.t-pad.b,X=i=>pad.l+gw*i/Math.max(1,vals.length-1),Y=v=>pad.t+gh*(hi-v)/(hi-lo);x.strokeStyle='#e4e8ee';x.lineWidth=1;x.font='9px ui-monospace';x.fillStyle='#7d8795';[lo,(lo+hi)/2,hi].forEach(v=>{const y=Y(v);x.beginPath();x.moveTo(pad.l,y);x.lineTo(w-pad.r,y);x.stroke();x.fillText(v.toFixed(3),1,y+3)});x.strokeStyle=color;x.lineWidth=1.7;x.beginPath();let begun=false;vals.forEach((v,i)=>{if(!Number.isFinite(v))return;begun?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v));begun=true});x.stroke();const i=vals.length-1,v=vals[i];if(Number.isFinite(v)){x.strokeStyle='#b0431c';x.lineWidth=1;x.beginPath();x.moveTo(X(i),pad.t);x.lineTo(X(i),h-pad.b);x.stroke();x.fillStyle=color;x.beginPath();x.arc(X(i),Y(v),2.8,0,Math.PI*2);x.fill()}}
function plotMany(id,series){const [x,w,h]=canvas(id),sets=series.map(s=>({...s,values:(s.values||[]).map(v=>v==null?null:+v)})),finite=sets.flatMap(s=>s.values.filter(Number.isFinite));x.clearRect(0,0,w,h);if(!finite.length){x.fillStyle='#8993a2';x.font='10px system-ui';x.fillText('Waiting for executed steps',10,17);return}let lo=Math.min(0,...finite),hi=Math.max(...finite);if(hi===lo)hi=lo+.001;const n=Math.max(...sets.map(s=>s.values.length)),pad={l:38,r:7,t:5,b:8},gw=w-pad.l-pad.r,gh=h-pad.t-pad.b,X=i=>pad.l+gw*i/Math.max(1,n-1),Y=v=>pad.t+gh*(hi-v)/(hi-lo);x.strokeStyle='#e4e8ee';x.lineWidth=1;x.font='9px ui-monospace';x.fillStyle='#7d8795';[lo,(lo+hi)/2,hi].forEach(v=>{const y=Y(v);x.beginPath();x.moveTo(pad.l,y);x.lineTo(w-pad.r,y);x.stroke();x.fillText(v.toFixed(3),1,y+3)});for(const s of sets){x.strokeStyle=s.color;x.lineWidth=s.width||1.5;x.beginPath();let begun=false;s.values.forEach((v,i)=>{if(!Number.isFinite(v))return;begun?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v));begun=true});x.stroke()}x.strokeStyle='#b0431c';x.lineWidth=1;x.beginPath();x.moveTo(X(n-1),pad.t);x.lineTo(X(n-1),h-pad.b);x.stroke()}
function latest(values){const a=(values||[]).filter(v=>Number.isFinite(+v));return a.length?f3(a.at(-1)):'–'}
function metricNumber(v,digits){return Number.isFinite(+v)?(+v).toFixed(digits):'–'}
function renderS1Metrics(l){const st=l.last_step||{},stage=state?.stage||'waiting',turnStep=Number.isFinite(+l.executed)?+l.executed:0,budget=Number.isFinite(+l.budget)&&+l.budget>0?+l.budget:null,episodeStep=Number.isFinite(+state?.episode_steps)?+state.episode_steps:0,totalStep=stage==='executing'?episodeStep+turnStep:episodeStep,values=[['execution',stage,'step'],['progress',metricNumber((l.progress||[]).at(-1),4),'progress'],['|Δa|',metricNumber(st.motion_norm??(l.motion||[]).at(-1),5),'motion'],['grip',st.gripper_flag||'–','grip'],['grip_w',metricNumber(st.grip_width??(l.gripper_width||[]).at(-1),4),'grip'],['Δgrip',metricNumber(st.grip_width_delta,4),'grip'],['eef_pos',metricNumber(st.action_eef_pos_norm,4),'motion'],['eef_rot',metricNumber(st.action_eef_rot_norm,4),'motion'],['base',metricNumber(st.action_base_norm,4),'motion'],['subgoal step',budget?`${turnStep}/${budget}`:String(turnStep),'step'],['total step',String(totalStep),'step']];$('#s1Metrics').innerHTML=values.map(([label,value,tone])=>`<span class="metric-bubble ${tone}"><b>${esc(label)}</b>${esc(value)}</span>`).join('')}
function drawAll(l){plotOne('#curveProgress',l.progress||[],'#2f73bf',[0,1]);plotMany('#curveMotion',[{values:l.motion_eef_pos,color:'#d8792d'},{values:l.motion_eef_rot,color:'#7a5cc7'},{values:l.motion_base,color:'#199bb2'},{values:l.motion,color:'#263d64',width:2}]);plotOne('#curveGripper',l.gripper_width||[],'#16845b');$('#progressValue').textContent=latest(l.progress);$('#motionValue').textContent=`|Δa| ${latest(l.motion)}`;$('#gripperValue').textContent=latest(l.gripper_width);renderS1Metrics(l);renderActionChunk(l)}
$('#split').onchange=fillTasks;$('#task').onchange=fillEpisodes;$('#start').onclick=start;$('#commit').onclick=commit;$('#intervene').onclick=intervene;$('#ask').onclick=ask;$('#replan').onclick=replan;$('#revert').onclick=revert;$('#stop').onclick=stop;$('#useS2').onclick=()=>chooseDecisionSource('s2');$('#useRules').onclick=()=>chooseDecisionSource('rules');$('#ruleHint').onclick=()=>chooseDecisionSource('rules');$('#presetContinue').onclick=()=>applyPreset('continue');$('#presetRedo').onclick=()=>applyPreset('redo');$('#presetCurrent').onclick=()=>applyPreset('current');$('#presetNext').onclick=()=>applyPreset('next');$('#planCards').addEventListener('click',editPlan);$('#planCards').addEventListener('input',editPlan);$('#addMilestone').onclick=()=>{planItems.push({mark:' ',text:'',fine:[]});renderPlanEditor(`[data-kind="milestone"][data-mi="${planItems.length-1}"][data-action="sentence"]`)};window.onresize=()=>state&&drawAll(state.live||{});frameTimer=setInterval(pumpFrame,50);
(async()=>{const nav=performance.getEntriesByType('navigation')[0];if(nav)clientTimings.page_s=Math.max(0,nav.responseEnd-nav.requestStart)/1000;const started=performance.now();beginOperation('Loading RoboCasa task catalog');renderTimings(null);try{catalog=await (await fetch('/api/hitl/episodes')).json();clientTimings.catalog_s=(performance.now()-started)/1000;if(!Array.isArray(catalog))throw Error(catalog.error||'catalog unavailable');fillTasks();renderTimings(state)}catch(e){err(e)}finally{endOperation()}})();
</script></body></html>"""
