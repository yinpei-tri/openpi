"""Durable, provenance-first storage for human-in-the-loop S1+S2 sessions.

The live simulator is intentionally kept out of this module.  This file owns the part that must
remain correct even if the GUI or simulator process crashes: target-split provenance, append-only
events, branch nodes, model/rule/human decision layers, and atomic JSON sidecars for large binary
artifacts.

Every serialized record embeds the full episode provenance.  Joining through a session table would
be smaller, but it is unsafe for the intended use: individual JSON rows and attempt directories will
later be copied into S2 training datasets, where a join back to the original SQLite file may no
longer be possible.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any
import uuid

SCHEMA_VERSION = 1
TARGET_CATEGORIES = frozenset({"atomic_seen", "composite_seen", "composite_unseen"})
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "HUMAN_INTERACTIVE_RESULTS_DIR",
        "/home/ec2-user/data/sys1_eval_results/human_interactive",
    )
).expanduser()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _write_json(path: Path, value: Any) -> None:
    """Atomic JSON replacement; readers see either the old complete file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _safe_token(value: str, *, max_len: int = 80) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return (token or "unknown")[:max_len]


@dataclasses.dataclass(frozen=True)
class TargetEpisode:
    """Identity and reset provenance that accompanies every saved HITL record."""

    episode_id: str
    task_name: str
    episode_index: int
    task_split: str
    lerobot_dir: str
    instruction: str
    initial_state_sha256: str
    model_xml_sha256: str
    dataset_name: str = "robocasa-v1.0"
    dataset_split: str = "target"
    scene_split: str = "target"
    object_instance_split: str = "target"

    def __post_init__(self) -> None:
        if self.dataset_split != "target" or self.scene_split != "target":
            raise ValueError("human-interactive sessions currently require the target split")
        if self.object_instance_split != "target":
            raise ValueError("object_instance_split must be 'target'")
        if self.task_split not in TARGET_CATEGORIES:
            raise ValueError(f"task_split must be one of {sorted(TARGET_CATEGORIES)}, got {self.task_split!r}")
        if not self.task_name or int(self.episode_index) < 0:
            raise ValueError("task_name and a non-negative episode_index are required")
        expected_suffix = f"episode_{int(self.episode_index):06d}"
        if expected_suffix not in self.episode_id:
            raise ValueError(f"episode_id {self.episode_id!r} does not identify {expected_suffix}")
        for field in ("initial_state_sha256", "model_xml_sha256"):
            value = getattr(self, field)
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a full lowercase SHA-256 digest")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class HumanInteractiveStore:
    """One append-only interactive session plus its filesystem artifact tree."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.db_path = self.session_dir / "session.sqlite"
        self._lock = threading.RLock()
        manifest = json.loads((self.session_dir / "manifest.json").read_text())
        self.session_id = manifest["session_id"]
        self.episode = TargetEpisode(**manifest["episode"])
        self._db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")

    @classmethod
    def create(
        cls,
        episode: TargetEpisode,
        *,
        results_root: Path = DEFAULT_RESULTS_ROOT,
        runtime: dict[str, Any] | None = None,
        operator: str | None = None,
        session_id: str | None = None,
    ) -> HumanInteractiveStore:
        session_id = session_id or uuid.uuid4().hex
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        name = "__".join(
            (
                "target",
                episode.task_split,
                _safe_token(episode.task_name),
                f"episode_{episode.episode_index:06d}",
                stamp,
                session_id[:8],
            )
        )
        session_dir = Path(results_root).expanduser().resolve() / name
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "nodes").mkdir()
        (session_dir / "attempts").mkdir()
        db = sqlite3.connect(session_dir / "session.sqlite")
        try:
            db.executescript(
                """
                PRAGMA foreign_keys=ON;
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE session_meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                CREATE TABLE events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    node_id TEXT,
                    attempt_id TEXT,
                    episode_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY,
                    parent_node_id TEXT,
                    via_attempt_id TEXT,
                    turn_index INTEGER NOT NULL,
                    action_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    episode_json TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL
                );
                CREATE TABLE attempts (
                    attempt_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    branch_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    episode_json TEXT NOT NULL,
                    model_json TEXT NOT NULL,
                    rules_json TEXT NOT NULL,
                    human_json TEXT,
                    final_json TEXT,
                    result_json TEXT,
                    FOREIGN KEY(node_id) REFERENCES nodes(node_id)
                );
                CREATE TABLE edits (
                    edit_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    author TEXT,
                    field TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    rationale TEXT,
                    episode_json TEXT NOT NULL,
                    FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
                );
                CREATE INDEX events_attempt_idx ON events(attempt_id, seq);
                CREATE INDEX attempts_node_idx ON attempts(node_id, branch_index);
                """
            )
            meta = {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "created_at": _utc_now(),
                "episode": episode.as_dict(),
                "runtime": runtime or {},
                "operator": operator,
                # Target episodes can be debugged here, but exporting them for training makes the
                # official target benchmark contaminated.  Export requires an explicit override.
                "data_policy": {
                    "source_is_official_target": True,
                    "training_export_default": "blocked",
                    "training_export_requires_explicit_target_opt_in": True,
                },
                "active_node_id": None,
                "status": "created",
            }
            for key, value in meta.items():
                db.execute("INSERT INTO session_meta(key,value_json) VALUES (?,?)", (key, _json(value)))
            db.commit()
            _write_json(session_dir / "manifest.json", meta)
        finally:
            db.close()
        store = cls(session_dir)
        store.record_event("session_created", {"runtime": runtime or {}, "operator": operator})
        return store

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()

    def _envelope(
        self,
        record_type: str,
        payload: dict[str, Any],
        *,
        node_id: str | None = None,
        attempt_id: str | None = None,
        record_id: str | None = None,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "record_id": record_id or uuid.uuid4().hex,
            "record_type": record_type,
            "created_at": created_at or _utc_now(),
            "episode": self.episode.as_dict(),
            "node_id": node_id,
            "attempt_id": attempt_id,
            "payload": payload,
        }

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        node_id: str | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any]:
        event = self._envelope(event_type, payload, node_id=node_id, attempt_id=attempt_id)
        with self._lock:
            self._db.execute(
                """INSERT INTO events
                   (event_id,created_at,event_type,node_id,attempt_id,episode_json,payload_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    event["record_id"],
                    event["created_at"],
                    event_type,
                    node_id,
                    attempt_id,
                    _json(event["episode"]),
                    _json(payload),
                ),
            )
            self._db.commit()
            # Human-readable append-only mirror. SQLite remains authoritative.
            with (self.session_dir / "events.jsonl").open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return event

    def _update_manifest(self, **updates: Any) -> None:
        with self._lock:
            manifest_path = self.session_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(updates)
            manifest["updated_at"] = _utc_now()
            for key, value in updates.items():
                self._db.execute(
                    "INSERT OR REPLACE INTO session_meta(key,value_json) VALUES (?,?)",
                    (key, _json(value)),
                )
            self._db.commit()
            _write_json(manifest_path, manifest)

    def create_node(
        self,
        *,
        turn_index: int,
        action_count: int,
        checkpoint: dict[str, Any],
        parent_node_id: str | None = None,
        via_attempt_id: str | None = None,
        node_id: str | None = None,
    ) -> str:
        node_id = node_id or f"node-{uuid.uuid4().hex[:12]}"
        created = _utc_now()
        node_dir = self.session_dir / "nodes" / node_id
        node_dir.mkdir(parents=True, exist_ok=False)
        doc = self._envelope(
            "turn_checkpoint",
            {
                "turn_index": int(turn_index),
                "action_count": int(action_count),
                "parent_node_id": parent_node_id,
                "via_attempt_id": via_attempt_id,
                "checkpoint": checkpoint,
            },
            node_id=node_id,
            record_id=node_id,
            created_at=created,
        )
        _write_json(node_dir / "node.json", doc)
        with self._lock:
            self._db.execute(
                """INSERT INTO nodes
                   (node_id,parent_node_id,via_attempt_id,turn_index,action_count,status,
                    created_at,episode_json,checkpoint_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    node_id,
                    parent_node_id,
                    via_attempt_id,
                    int(turn_index),
                    int(action_count),
                    "active",
                    created,
                    _json(self.episode.as_dict()),
                    _json(checkpoint),
                ),
            )
            self._db.commit()
        self._update_manifest(active_node_id=node_id, status="active")
        self.record_event(
            "checkpoint_created",
            {"turn_index": turn_index, "action_count": action_count},
            node_id=node_id,
        )
        return node_id

    def update_node_checkpoint(self, node_id: str, checkpoint: dict[str, Any]) -> None:
        """Replace a node's checkpoint descriptor after its binary artifacts are durable."""
        path = self.session_dir / "nodes" / node_id / "node.json"
        if not path.is_file():
            raise KeyError(f"unknown node {node_id}")
        doc = json.loads(path.read_text())
        doc["payload"]["checkpoint"] = checkpoint
        doc["updated_at"] = _utc_now()
        _write_json(path, doc)
        with self._lock:
            cur = self._db.execute(
                "UPDATE nodes SET checkpoint_json=? WHERE node_id=?",
                (_json(checkpoint), node_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"unknown node {node_id}")
            self._db.commit()
        self.record_event("checkpoint_updated", checkpoint, node_id=node_id)

    def create_attempt(
        self,
        node_id: str,
        *,
        model: dict[str, Any],
        rules: dict[str, Any],
        attempt_id: str | None = None,
    ) -> str:
        attempt_id = attempt_id or f"attempt-{uuid.uuid4().hex[:12]}"
        created = _utc_now()
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(MAX(branch_index),-1)+1 AS n FROM attempts WHERE node_id=?",
                (node_id,),
            ).fetchone()
            branch_index = int(row["n"])
            self._db.execute(
                """INSERT INTO attempts
                   (attempt_id,node_id,branch_index,status,created_at,updated_at,episode_json,
                    model_json,rules_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    attempt_id,
                    node_id,
                    branch_index,
                    "proposed",
                    created,
                    created,
                    _json(self.episode.as_dict()),
                    _json(model),
                    _json(rules),
                ),
            )
            self._db.commit()
        attempt_dir = self.session_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        doc = self._envelope(
            "decision_attempt",
            {
                "branch_index": branch_index,
                "status": "proposed",
                "model": model,
                "rules": rules,
                "human": None,
                "final": None,
                "result": None,
            },
            node_id=node_id,
            attempt_id=attempt_id,
            record_id=attempt_id,
            created_at=created,
        )
        _write_json(attempt_dir / "attempt.json", doc)
        self.record_event(
            "decision_proposed",
            {"branch_index": branch_index, "model": model, "rules": rules},
            node_id=node_id,
            attempt_id=attempt_id,
        )
        return attempt_id

    def _attempt_doc(self, attempt_id: str) -> tuple[Path, dict[str, Any]]:
        path = self.session_dir / "attempts" / attempt_id / "attempt.json"
        if not path.is_file():
            raise KeyError(f"unknown attempt {attempt_id}")
        return path, json.loads(path.read_text())

    def record_edit(
        self,
        attempt_id: str,
        *,
        field: str,
        before: Any,
        after: Any,
        author: str | None = None,
        rationale: str | None = None,
    ) -> str:
        edit_id = f"edit-{uuid.uuid4().hex[:12]}"
        created = _utc_now()
        path, attempt = self._attempt_doc(attempt_id)
        node_id = attempt["node_id"]
        edit = self._envelope(
            "human_edit",
            {
                "field": field,
                "before": before,
                "after": after,
                "author": author,
                "rationale": rationale,
            },
            node_id=node_id,
            attempt_id=attempt_id,
            record_id=edit_id,
            created_at=created,
        )
        edit_dir = path.parent / "edits"
        _write_json(edit_dir / f"{edit_id}.json", edit)
        with self._lock:
            self._db.execute(
                """INSERT INTO edits
                   (edit_id,attempt_id,created_at,author,field,before_json,after_json,rationale,
                    episode_json) VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    edit_id,
                    attempt_id,
                    created,
                    author,
                    field,
                    _json(before),
                    _json(after),
                    rationale,
                    _json(self.episode.as_dict()),
                ),
            )
            self._db.commit()
        self.record_event(
            "human_edit",
            edit["payload"],
            node_id=node_id,
            attempt_id=attempt_id,
        )
        return edit_id

    def commit_attempt(
        self,
        attempt_id: str,
        *,
        human: dict[str, Any],
        final: dict[str, Any],
    ) -> None:
        path, doc = self._attempt_doc(attempt_id)
        doc["payload"]["human"] = human
        doc["payload"]["final"] = final
        doc["payload"]["status"] = "committed"
        doc["updated_at"] = _utc_now()
        _write_json(path, doc)
        with self._lock:
            self._db.execute(
                """UPDATE attempts SET status='committed',updated_at=?,human_json=?,final_json=?
                   WHERE attempt_id=?""",
                (doc["updated_at"], _json(human), _json(final), attempt_id),
            )
            self._db.commit()
        self.record_event(
            "decision_committed",
            {"human": human, "final": final},
            node_id=doc["node_id"],
            attempt_id=attempt_id,
        )

    def finish_attempt(self, attempt_id: str, result: dict[str, Any]) -> None:
        path, doc = self._attempt_doc(attempt_id)
        doc["payload"]["result"] = result
        doc["payload"]["status"] = "executed"
        doc["updated_at"] = _utc_now()
        _write_json(path, doc)
        with self._lock:
            self._db.execute(
                "UPDATE attempts SET status='executed',updated_at=?,result_json=? WHERE attempt_id=?",
                (doc["updated_at"], _json(result), attempt_id),
            )
            self._db.commit()
        self.record_event(
            "attempt_executed",
            result,
            node_id=doc["node_id"],
            attempt_id=attempt_id,
        )

    def supersede_attempt(self, attempt_id: str, *, reason: str = "one_turn_revert") -> None:
        path, doc = self._attempt_doc(attempt_id)
        doc["payload"]["status"] = "superseded"
        doc["payload"]["superseded_reason"] = reason
        doc["updated_at"] = _utc_now()
        _write_json(path, doc)
        with self._lock:
            self._db.execute(
                "UPDATE attempts SET status='superseded',updated_at=? WHERE attempt_id=?",
                (doc["updated_at"], attempt_id),
            )
            self._db.commit()
        self.record_event(
            "attempt_superseded",
            {"reason": reason},
            node_id=doc["node_id"],
            attempt_id=attempt_id,
        )

    def set_active_node(self, node_id: str, *, reason: str) -> None:
        with self._lock:
            row = self._db.execute("SELECT node_id FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown node {node_id}")
            self._db.execute("UPDATE nodes SET status='inactive' WHERE status='active'")
            self._db.execute("UPDATE nodes SET status='active' WHERE node_id=?", (node_id,))
            self._db.commit()
        self._update_manifest(active_node_id=node_id)
        self.record_event("active_node_changed", {"reason": reason}, node_id=node_id)

    def set_session_status(self, status: str, **details: Any) -> None:
        """Persist a coarse lifecycle state without discarding the append-only event history."""
        self._update_manifest(status=status)
        self.record_event("session_status", {"status": status, **details})

    def artifact_path(self, attempt_id: str, relative: str) -> Path:
        """Resolve an attempt artifact path without allowing directory traversal."""
        base = (self.session_dir / "attempts" / attempt_id).resolve()
        path = (base / relative).resolve()
        if path != base and base not in path.parents:
            raise ValueError("artifact path escapes the attempt directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def node_artifact_path(self, node_id: str, relative: str) -> Path:
        """Resolve a checkpoint artifact path without allowing directory traversal."""
        base = (self.session_dir / "nodes" / node_id).resolve()
        path = (base / relative).resolve()
        if path != base and base not in path.parents:
            raise ValueError("artifact path escapes the node directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def session_artifact_path(self, relative: str) -> Path:
        """Resolve a session-level reset artifact without allowing directory traversal."""
        base = self.session_dir.resolve()
        path = (base / relative).resolve()
        if path != base and base not in path.parents:
            raise ValueError("artifact path escapes the session directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _label_artifact_path(
        self,
        artifact: Path,
        *,
        artifact_type: str,
        metadata: dict[str, Any],
        node_id: str | None,
        attempt_id: str | None,
    ) -> Path:
        sidecar = artifact.with_suffix(artifact.suffix + ".json")
        doc = self._envelope(
            "artifact",
            {
                "artifact_type": artifact_type,
                "artifact_path": str(artifact.relative_to(self.session_dir)),
                "metadata": metadata,
            },
            node_id=node_id,
            attempt_id=attempt_id,
        )
        _write_json(sidecar, doc)
        return sidecar

    def label_artifact(
        self,
        attempt_id: str,
        relative: str,
        *,
        artifact_type: str,
        metadata: dict[str, Any],
        node_id: str | None = None,
    ) -> Path:
        """Write a target-labelled JSON sidecar for an NPZ/video/image artifact."""
        artifact = self.artifact_path(attempt_id, relative)
        return self._label_artifact_path(
            artifact,
            artifact_type=artifact_type,
            metadata=metadata,
            node_id=node_id,
            attempt_id=attempt_id,
        )

    def write_attempt_record(
        self,
        attempt_id: str,
        relative: str,
        *,
        record_type: str,
        payload: dict[str, Any],
        node_id: str | None = None,
    ) -> Path:
        """Write a structured attempt JSON whose content itself carries target provenance."""
        path = self.artifact_path(attempt_id, relative)
        if path.suffix.lower() != ".json":
            raise ValueError("structured attempt records must use a .json suffix")
        doc = self._envelope(
            record_type,
            payload,
            node_id=node_id,
            attempt_id=attempt_id,
        )
        _write_json(path, doc)
        return path

    def label_node_artifact(
        self,
        node_id: str,
        relative: str,
        *,
        artifact_type: str,
        metadata: dict[str, Any],
    ) -> Path:
        artifact = self.node_artifact_path(node_id, relative)
        return self._label_artifact_path(
            artifact,
            artifact_type=artifact_type,
            metadata=metadata,
            node_id=node_id,
            attempt_id=None,
        )

    def label_session_artifact(
        self,
        relative: str,
        *,
        artifact_type: str,
        metadata: dict[str, Any],
    ) -> Path:
        artifact = self.session_artifact_path(relative)
        return self._label_artifact_path(
            artifact,
            artifact_type=artifact_type,
            metadata=metadata,
            node_id=None,
            attempt_id=None,
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = {
                row["status"]: row["n"]
                for row in self._db.execute("SELECT status,COUNT(*) AS n FROM attempts GROUP BY status").fetchall()
            }
            nodes = self._db.execute("SELECT COUNT(*) AS n FROM nodes").fetchone()["n"]
            events = self._db.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        manifest = json.loads((self.session_dir / "manifest.json").read_text())
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "episode": self.episode.as_dict(),
            "status": manifest.get("status"),
            "active_node_id": manifest.get("active_node_id"),
            "n_nodes": int(nodes),
            "n_events": int(events),
            "attempts_by_status": counts,
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
