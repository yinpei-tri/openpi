from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from examples.robocasa.human_interactive_store import HumanInteractiveStore
from examples.robocasa.human_interactive_store import TargetEpisode
from examples.robocasa.human_interactive_store import sha256_bytes


def _target_episode() -> TargetEpisode:
    return TargetEpisode(
        episode_id="ArrangeTea/target/episode_000007",
        task_name="ArrangeTea",
        episode_index=7,
        task_split="composite_unseen",
        lerobot_dir="/data/target/kitchen/ArrangeTea/demo/lerobot",
        instruction="Put the kettle and mug on the tray, then close the cabinet.",
        initial_state_sha256=sha256_bytes(b"initial-state"),
        model_xml_sha256=sha256_bytes(b"model-xml"),
    )


def _assert_target_label(doc: dict) -> None:
    episode = doc["episode"]
    assert episode["dataset_split"] == "target"
    assert episode["scene_split"] == "target"
    assert episode["object_instance_split"] == "target"
    assert episode["task_split"] == "composite_unseen"
    assert episode["task_name"] == "ArrangeTea"
    assert episode["episode_index"] == 7
    assert episode["episode_id"] == "ArrangeTea/target/episode_000007"


def test_rejects_non_target_episode() -> None:
    values = _target_episode().as_dict()
    values["dataset_split"] = "train"
    with pytest.raises(ValueError, match="target split"):
        TargetEpisode(**values)


def test_session_records_are_target_labelled_and_branchable(tmp_path: Path) -> None:
    store = HumanInteractiveStore.create(
        _target_episode(),
        results_root=tmp_path,
        runtime={"s1_checkpoint": "progact-270k", "s2_model": "Qwen3-VL-4B"},
        operator="tester",
        session_id="a" * 32,
    )
    assert store.session_dir.name.startswith("target__composite_unseen__ArrangeTea__episode_000007__")
    manifest = json.loads((store.session_dir / "manifest.json").read_text())
    _assert_target_label(manifest)
    assert manifest["data_policy"]["training_export_default"] == "blocked"

    node_id = store.create_node(
        turn_index=0,
        action_count=0,
        checkpoint={"state": "nodes/node/state.npz", "replay_action_count": 0},
    )
    store.update_node_checkpoint(
        node_id,
        {"state": f"nodes/{node_id}/checkpoint/state.npz", "replay_action_count": 0},
    )
    attempt_id = store.create_attempt(
        node_id,
        model={"subgoal": "grasp the mug", "estimated_length": 75},
        rules={"subgoal": "grasp the mug", "estimated_length": 75, "hits": []},
    )
    store.record_edit(
        attempt_id,
        field="estimated_length",
        before=75,
        after=100,
        author="tester",
        rationale="The cabinet is far away.",
    )
    store.commit_attempt(
        attempt_id,
        human={"edited": True, "author": "tester"},
        final={"subgoal": "grasp the mug", "estimated_length": 100},
    )
    store.finish_attempt(attempt_id, {"steps": 91, "success": False})
    store.label_artifact(
        attempt_id,
        "execution/actions.npz",
        artifact_type="action_trace",
        metadata={"steps": 91},
        node_id=node_id,
    )
    store.write_attempt_record(
        attempt_id,
        "execution/steps.json",
        record_type="s1_step_trace",
        payload={"steps": []},
        node_id=node_id,
    )
    store.label_node_artifact(
        node_id,
        "checkpoint/action_prefix.npz",
        artifact_type="replay_action_prefix",
        metadata={"steps": 0},
    )
    store.label_session_artifact(
        "reset/model.xml",
        artifact_type="initial_model_xml",
        metadata={"sha256": _target_episode().model_xml_sha256},
    )
    store.supersede_attempt(attempt_id)
    store.set_session_status("active", note="test")

    retry_id = store.create_attempt(
        node_id,
        model={"subgoal": "grasp the mug", "estimated_length": 75},
        rules={"subgoal": "grasp the mug", "estimated_length": 75, "hits": []},
    )
    assert retry_id != attempt_id
    summary = store.summary()
    assert summary["n_nodes"] == 1
    assert summary["attempts_by_status"] == {"proposed": 1, "superseded": 1}

    session_dir = store.session_dir
    store.close()

    # A process restart must preserve the active episode identity and branch history.
    reopened = HumanInteractiveStore(session_dir)
    assert reopened.summary()["episode"] == _target_episode().as_dict()
    reopened.close()

    for path in session_dir.rglob("*.json"):
        doc = json.loads(path.read_text())
        if path.name == "manifest.json":
            _assert_target_label(doc)
        else:
            _assert_target_label(doc)
    for line in (session_dir / "events.jsonl").read_text().splitlines():
        _assert_target_label(json.loads(line))

    db = sqlite3.connect(session_dir / "session.sqlite")
    try:
        for table in ("events", "nodes", "attempts", "edits"):
            rows = db.execute(f"SELECT episode_json FROM {table}").fetchall()
            assert rows, table
            for (episode_json,) in rows:
                _assert_target_label({"episode": json.loads(episode_json)})
        attempts = db.execute("SELECT branch_index,status FROM attempts ORDER BY branch_index").fetchall()
        assert attempts == [(0, "superseded"), (1, "proposed")]
    finally:
        db.close()


def test_artifact_path_cannot_escape_session(tmp_path: Path) -> None:
    store = HumanInteractiveStore.create(_target_episode(), results_root=tmp_path)
    try:
        with pytest.raises(ValueError, match="escapes"):
            store.artifact_path("attempt-x", "../../outside.npz")
        with pytest.raises(ValueError, match="escapes"):
            store.node_artifact_path("node-x", "../../outside.npz")
        with pytest.raises(ValueError, match="escapes"):
            store.session_artifact_path("../../outside.npz")
    finally:
        store.close()
