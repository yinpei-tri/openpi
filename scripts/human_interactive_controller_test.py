from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import threading
import urllib.request

from flask import Flask
import numpy as np

from examples.robocasa.human_interactive_controller import InteractiveConfig
from examples.robocasa.human_interactive_controller import InteractiveSession
from examples.robocasa.human_interactive_controller import _field_edits
from examples.robocasa.human_interactive_controller import _restore_simple_attrs
from examples.robocasa.human_interactive_controller import _snapshot_simple_attrs
from examples.robocasa.human_interactive_controller import _task_finish_is_terminal
from examples.robocasa.human_interactive_gui import configure_human_interactive
from examples.robocasa.human_interactive_gui import HITL_HTML
from examples.robocasa.human_interactive_gui import register_human_interactive
from examples.robocasa.sys2_client import Sys2Client


class _LatchObject:
    def __init__(self):
        self.turned_on = False
        self.counters = {("left", "right"): 0}
        self.vector = np.asarray([1.0, 2.0])
        self.simulator_reference = object()


def test_plan_editor_uses_structured_cards_instead_of_raw_checklist_text() -> None:
    assert 'id="planCards"' in HITL_HTML
    assert 'id="addMilestone"' in HITL_HTML
    assert 'data-action="add-fine"' in HITL_HTML
    assert "function planText()" in HITL_HTML
    assert '<textarea id="plan"' not in HITL_HTML


def test_s2_prediction_is_editor_default_and_exact_io_stays_visible() -> None:
    assert 'id="useS2"' in HITL_HTML
    assert "loadDecision(predictionValues(p),'s2')" in HITL_HTML
    assert 'id="s2Reference"' in HITL_HTML
    assert 'id="s2System"' in HITL_HTML
    assert 'id="s2User"' in HITL_HTML
    assert 'id="s2Raw"' in HITL_HTML
    assert 'id="s2Parsed"' in HITL_HTML


def test_loading_and_model_timings_are_exposed_in_the_gui() -> None:
    assert 'id="latencyNow"' in HITL_HTML
    assert 'id="timingStrip"' in HITL_HTML
    assert "step avg" in HITL_HTML
    assert 'id="actionChunk"' in HITL_HTML
    assert 'id="curveProgress"' in HITL_HTML
    assert 'id="curveMotion"' in HITL_HTML
    assert 'id="curveGripper"' in HITL_HTML
    assert 'id="s1Metrics"' in HITL_HTML
    assert "function renderS1Metrics" in HITL_HTML
    assert "['execution',stage" in HITL_HTML
    assert "['subgoal step'" in HITL_HTML
    assert "['total step'" in HITL_HTML
    assert "height:64px;min-height:64px;max-height:64px" in HITL_HTML
    assert 'id="sceneExecution"' not in HITL_HTML
    assert 'id="sceneFrame"' not in HITL_HTML
    assert 'id="sceneReplan"' not in HITL_HTML
    assert 'id="timeline"' not in HITL_HTML
    assert 'id="ruleHint"' in HITL_HTML
    assert 'id="s2Stream"' in HITL_HTML
    assert 'id="presetContinue"' in HITL_HTML
    assert 'id="presetRedo"' in HITL_HTML
    assert 'id="presetCurrent"' in HITL_HTML
    assert 'id="presetNext"' in HITL_HTML
    assert 'id="intervene"' in HITL_HTML
    assert ".s2-stream-card{height:360px" in HITL_HTML
    assert "/api/hitl/turn/intervene" in HITL_HTML
    assert "Free intervention mode" in HITL_HTML
    assert 'class="hidden" id="detail"' in HITL_HTML
    assert 'id="ruleStop"' not in HITL_HTML
    assert 'id="ruleSkip"' not in HITL_HTML
    assert 'id="actOverride"' not in HITL_HTML
    assert 'id="forceOverride"' not in HITL_HTML
    assert 'id="commit" disabled>Execute</button>' in HITL_HTML
    assert "?'Accept plan':'Execute'" in HITL_HTML
    assert "needsS2Advance" in HITL_HTML
    assert 'id="s1Prompt"' in HITL_HTML
    live_order = [
        HITL_HTML.index('id="frame"'),
        HITL_HTML.index('id="s1Metrics"'),
        HITL_HTML.index('id="s1Prompt"'),
        HITL_HTML.index('id="curveProgress"'),
        HITL_HTML.index('id="actionChunk"'),
    ]
    assert live_order == sorted(live_order)
    assert ".scene-grid{display:flex;flex-direction:column" in HITL_HTML
    assert 'id="s1Anchor"' in HITL_HTML
    assert '<summary>More details</summary>' in HITL_HTML
    assert HITL_HTML.index('id="s2Reference"') < HITL_HTML.index('id="s1Anchor"')
    assert "function plotMany" in HITL_HTML
    assert "shine-update" in HITL_HTML
    assert 'id="timingCard"' not in HITL_HTML
    live = InteractiveSession._empty_live()
    assert live["s1_infer_seconds"] == []
    assert live["env_render_seconds"] == []
    assert live["motion_eef_pos"] == []
    assert live["motion_eef_rot"] == []
    assert live["motion_base"] == []
    assert live["phase"] is None


def test_sys2_client_reports_real_accumulated_sse_text(monkeypatch) -> None:
    events = [
        b'data: {"choices":[{"delta":{"content":"<subgoal>turn"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" on</subgoal>"}}]}\n',
        b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
        b'data: [DONE]\n',
    ]
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(events)

    def fake_urlopen(req, timeout):
        captured["payload"] = json.loads(req.data)
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    updates = []
    result = Sys2Client(retries=1).chat("system", "user", on_text=updates.append)
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    assert updates == ["", "<subgoal>turn", "<subgoal>turn on</subgoal>"]
    assert result["text"] == "<subgoal>turn on</subgoal>"
    assert result["usage"] == {"completion_tokens": 2}


def test_live_frame_queue_is_ordered_and_empty_means_keep_last_frame() -> None:
    session = InteractiveSession.__new__(InteractiveSession)
    session.lock = threading.RLock()
    session.latest_frame = None
    session.frame_queue = deque(maxlen=2)
    session.frame_sequence = 0
    session._queue_frame(np.full((2, 3, 3), 1, np.uint8))
    session._queue_frame(np.full((2, 3, 3), 2, np.uint8))
    first = session.next_frame(-1)
    second = session.next_frame(first[0])
    assert first[0] == 1 and np.all(first[1] == 1)
    assert second[0] == 2 and np.all(second[1] == 2)
    assert session.next_frame(second[0]) is None


def test_python_latch_snapshot_excludes_object_graphs_and_restores_values() -> None:
    obj = _LatchObject()
    saved = _snapshot_simple_attrs(obj)
    assert "simulator_reference" not in saved
    obj.turned_on = True
    obj.counters[("left", "right")] = 19
    obj.vector[:] = 0
    _restore_simple_attrs(obj, saved)
    assert obj.turned_on is False
    assert obj.counters == {("left", "right"): 0}
    assert obj.vector.tolist() == [1.0, 2.0]


def test_field_edits_are_explicit_and_ordered() -> None:
    before = {"plan": "p", "subgoal": "old", "estimated_step": 50}
    after = {"plan": "p", "subgoal": "new", "estimated_step": 75}
    assert _field_edits(before, after) == [
        ("subgoal", "old", "new"),
        ("estimated_step", 50, 75),
    ]


def test_free_intervention_is_a_saved_human_attempt_without_a_plan_update() -> None:
    class _Store:
        def __init__(self):
            self.created = None
            self.event = None

        def create_attempt(self, node_id, *, model, rules):
            self.created = (node_id, model, rules)
            return "attempt-human"

        def record_event(self, event_type, payload, **where):
            self.event = (event_type, payload, where)

    class _Rules:
        @staticmethod
        def rule_config(*, general, task_tier, last_milestone_retry):
            return {
                "general": general,
                "task_tier": task_tier,
                "last_milestone_retry": last_milestone_retry,
            }

    session = InteractiveSession.__new__(InteractiveSession)
    session.lock = threading.RLock()
    session.stage = "ready"
    session.proposal = None
    session.current_node_id = "node-1"
    session.nodes = {
        "node-1": {
            "node_id": "node-1",
            "turn_index": 3,
            "plan": "- [~] M1: original plan",
            "rule_state": {"judge": "subgoal_failed", "x": 1},
        }
    }
    session.operator = "tester"
    session.config = InteractiveConfig(dataset_root=Path("/target"), results_root=Path("/results"))
    session.SR = _Rules()
    session.store = _Store()
    session.snapshot = lambda: {"proposal": session.proposal, "stage": session.stage}

    state = session.begin_intervention(rationale="try a custom recovery")
    proposal = state["proposal"]
    assert state["stage"] == "turn_review"
    assert proposal["human_intervention"] is True
    assert proposal["prediction"]["subgoal"] == ""
    assert proposal["prediction"]["plan"] == "- [~] M1: original plan"
    assert proposal["rule_state_after"] == {"judge": None, "x": 1}
    assert session.store.created[1]["kind"] == "human_intervention"
    assert session.store.created[2]["bypassed"] is True
    assert session.store.created[2]["rule_config"]["last_milestone_retry"] is True
    assert session.store.event[0] == "human_intervention_started"


def test_interactive_rule_call_receives_task_status_and_retry_toggle() -> None:
    captured = {}

    class _Rules:
        @staticmethod
        def apply_rules(task_name, **kwargs):
            captured["task_name"] = task_name
            captured.update(kwargs)
            return {"suppress_task_finish": True}

    session = InteractiveSession.__new__(InteractiveSession)
    session.task_name = "PickPlaceSinkToCounter"
    session.config = InteractiveConfig(
        dataset_root=Path("/target"),
        results_root=Path("/results"),
        general_rules=True,
        task_rules=False,
        last_milestone_retry=True,
    )
    session.SR = _Rules()
    state = {"judge": "subgoal_failed"}
    raw = {
        "plan": "- [~] M1: move the mug\n  - [~] M1.1: carry the mug to the counter",
        "judge": "task_finish",
        "subgoal": "",
        "subgoal_detail": "",
        "estimated_step": None,
    }

    result = session._apply_execution_rules(raw, state, task_status="ongoing")

    assert result["suppress_task_finish"] is True
    assert state["judge"] == "task_finish"
    assert captured["task_name"] == "PickPlaceSinkToCounter"
    assert captured["task_status"] == "ongoing"
    assert captured["last_milestone_retry"] is True
    assert captured["general"] is True
    assert captured["task_tier"] is False


def test_suppressed_task_finish_executes_but_genuine_finish_is_terminal() -> None:
    assert _task_finish_is_terminal("task_finish", {"suppress_task_finish": False}) is True
    assert _task_finish_is_terminal("task_finish", {"suppress_task_finish": True}) is False
    assert _task_finish_is_terminal("subgoal_complete", {"suppress_task_finish": True}) is False


def test_blueprint_mount_and_public_config(tmp_path: Path) -> None:
    app = Flask(__name__)
    register_human_interactive(app)
    configure_human_interactive({"dataset_root": tmp_path / "target", "results_root": tmp_path / "results"})
    client = app.test_client()
    assert client.get("/human-interactive").status_code == 200
    config = client.get("/api/hitl/config")
    assert config.status_code == 200
    body = config.get_json()
    assert body["config"]["dataset_root"] == str(tmp_path / "target")
    assert body["config"]["general_rules"] is True
    assert body["config"]["last_milestone_retry"] is True


def test_config_serializes_paths() -> None:
    cfg = InteractiveConfig(dataset_root=Path("/target"), results_root=Path("/results"))
    assert cfg.public()["dataset_root"] == "/target"
