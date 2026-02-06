from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionStatus(Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class ActionNode:
    action_id: str
    task_name: str
    parent_id: str | None
    status: ActionStatus
    inputs: dict | None = None
    outputs: Any = None
    error: str | None = None
    output_path: str | None = None
    has_report: bool = False
    cache_enabled: bool = False
    cache_hit: bool = False
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None


def _safe_json(obj: Any) -> Any:
    """Convert *obj* to a JSON-safe structure.

    Tries ``json.dumps`` first; on failure recurses into dicts/lists and
    falls back to ``repr()`` for non-serializable leaves (FlyteFile,
    DataFrame, etc.).
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError, OverflowError):
        return repr(obj)


class ActionTracker:
    """Thread-safe event collector consumed by the TUI via polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nodes: dict[str, ActionNode] = {}
        self._root_ids: list[str] = []
        self._children: dict[str, list[str]] = {}
        self._version: int = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def record_start(
        self,
        *,
        action_id: str,
        task_name: str,
        parent_id: str | None = None,
        inputs: dict | None = None,
        output_path: str | None = None,
        has_report: bool = False,
        cache_enabled: bool = False,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            node = ActionNode(
                action_id=action_id,
                task_name=task_name,
                parent_id=parent_id,
                status=ActionStatus.RUNNING,
                inputs=_safe_json(inputs) if inputs else None,
                output_path=output_path,
                has_report=has_report,
                cache_enabled=cache_enabled,
                cache_hit=cache_hit,
            )
            self._nodes[action_id] = node
            if parent_id is None:
                if action_id not in self._root_ids:
                    self._root_ids.append(action_id)
            else:
                self._children.setdefault(parent_id, []).append(action_id)
            self._version += 1

    def record_complete(self, *, action_id: str, outputs: Any = None) -> None:
        with self._lock:
            node = self._nodes.get(action_id)
            if node is None:
                return
            node.status = ActionStatus.SUCCEEDED
            node.outputs = _safe_json(outputs)
            node.end_time = time.monotonic()
            self._version += 1

    def record_failure(self, *, action_id: str, error: str) -> None:
        with self._lock:
            node = self._nodes.get(action_id)
            if node is None:
                return
            node.status = ActionStatus.FAILED
            node.error = error
            node.end_time = time.monotonic()
            self._version += 1

    def snapshot(self) -> tuple[list[str], dict[str, list[str]], dict[str, ActionNode]]:
        """Return (root_ids, children_map, all_nodes) — a consistent snapshot."""
        with self._lock:
            return (
                list(self._root_ids),
                {k: list(v) for k, v in self._children.items()},
                dict(self._nodes),
            )

    def get_action(self, action_id: str) -> ActionNode | None:
        with self._lock:
            return self._nodes.get(action_id)
