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
    short_name: str | None = None
    inputs: dict | None = None
    outputs: Any = None
    error: str | None = None
    output_path: str | None = None
    has_report: bool = False
    cache_enabled: bool = False
    cache_hit: bool = False
    context: dict | None = None
    group: str | None = None
    log_links: list[tuple[str, str]] | None = None
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
        short_name: str | None = None,
        inputs: dict | None = None,
        output_path: str | None = None,
        has_report: bool = False,
        cache_enabled: bool = False,
        cache_hit: bool = False,
        context: dict | None = None,
        group: str | None = None,
        log_links: list[tuple[str, str]] | None = None,
    ) -> None:
        with self._lock:
            node = ActionNode(
                action_id=action_id,
                task_name=task_name,
                parent_id=parent_id,
                status=ActionStatus.RUNNING,
                short_name=short_name,
                inputs=_safe_json(inputs) if inputs else None,
                output_path=output_path,
                has_report=has_report,
                cache_enabled=cache_enabled,
                cache_hit=cache_hit,
                context=_safe_json(context) if context else None,
                group=group,
                log_links=log_links,
            )
            self._nodes[action_id] = node

            if group and parent_id is not None:
                group_key = f"__group__{parent_id}__{group}"
                if group_key not in self._nodes:
                    group_node = ActionNode(
                        action_id=group_key,
                        task_name=group,
                        parent_id=parent_id,
                        status=ActionStatus.RUNNING,
                    )
                    self._nodes[group_key] = group_node
                    self._children.setdefault(parent_id, []).append(group_key)
                self._children.setdefault(group_key, []).append(action_id)
                node.parent_id = group_key
            elif parent_id is None:
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
            node.outputs = outputs
            node.end_time = time.monotonic()
            self._update_group_status(action_id)
            self._version += 1

    def record_log_links(self, *, action_id: str, log_links: list[tuple[str, str]]) -> None:
        with self._lock:
            node = self._nodes.get(action_id)
            if node is None:
                return
            node.log_links = log_links
            self._version += 1

    def record_failure(self, *, action_id: str, error: str) -> None:
        from flyte._internal.runtime.convert import Error

        with self._lock:
            node = self._nodes.get(action_id)
            if node is None:
                return
            node.status = ActionStatus.FAILED
            if isinstance(error, Error):
                node.error = error.err
            else:
                node.error = error
            node.end_time = time.monotonic()
            self._update_group_status(action_id)
            self._version += 1

    def _update_group_status(self, action_id: str) -> None:
        """If action belongs to a group, recompute the group node's status.

        Must be called while ``self._lock`` is held.
        """
        node = self._nodes.get(action_id)
        if node is None or node.group is None:
            return
        if node.parent_id is None or not node.parent_id.startswith("__group__"):
            return
        group_node = self._nodes.get(node.parent_id)
        if group_node is None:
            return
        children = self._children.get(node.parent_id, [])
        child_nodes = [self._nodes[c] for c in children if c in self._nodes]
        if any(c.status == ActionStatus.FAILED for c in child_nodes):
            group_node.status = ActionStatus.FAILED
        elif any(c.status == ActionStatus.RUNNING for c in child_nodes):
            group_node.status = ActionStatus.RUNNING
        elif all(c.status == ActionStatus.SUCCEEDED for c in child_nodes):
            group_node.status = ActionStatus.SUCCEEDED
            end_times = [c.end_time for c in child_nodes if c.end_time is not None]
            if end_times:
                group_node.end_time = max(end_times)

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
