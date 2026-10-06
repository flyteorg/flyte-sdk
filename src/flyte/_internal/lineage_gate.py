"""
Cheap checks for whether a deploy or run declares any lineage, so code that does not use lineage never imports the
lineage machinery (`flyte.artifacts._lineage`, `_refresh`, `_refs`, `_handle`) nor pays for it.

Nothing here imports those modules. A value can only be an artifact handle once `flyte.artifacts._handle` has been
imported (by whoever built the handle), so handle checks consult `sys.modules` instead of importing it.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, List, Mapping, Optional

_HANDLE_MODULE = "flyte.artifacts._handle"


def is_handle(value: Any) -> bool:
    """`flyte.artifacts._handle.is_handle`, without importing that module when no handle can exist yet."""
    mod = sys.modules.get(_HANDLE_MODULE)
    return mod is not None and bool(mod.is_handle(value))


def _env_of(task: Any) -> Any:
    ref = getattr(task, "parent_env", None)
    return ref() if callable(ref) else None


def task_declares_lineage(task: Any, extra_labels: Optional[Mapping[str, str]] = None) -> bool:
    """
    True when serializing `task` may write lineage tags: deploy-wide labels, labels on the task or its environment,
    a tuple of produced handles, or a non-empty `consumes_artifacts`. `produces_artifacts=True` alone is the
    pre-lineage flag and does not count.
    """
    if extra_labels:
        return True
    if getattr(task, "labels", None):
        return True
    env = _env_of(task)
    if env is not None and getattr(env, "labels", None):
        return True
    produced = getattr(task, "produces_artifacts", None)
    if isinstance(produced, tuple) and any(h is not None for h in produced):
        return True
    return bool(getattr(task, "consumes_artifacts", None))


def env_declares_lineage(env: Any) -> bool:
    """True when `env` (a task or app environment, not its `depends_on`) declares any lineage."""
    if getattr(env, "labels", None):
        return True
    if getattr(env, "lineage", False) is True or getattr(env, "consumes_artifacts", None):
        return True
    for p in getattr(env, "parameters", None) or ():
        value = getattr(p, "value", None)
        # A Parameter bound to an Artifact handle carries it on its ArtifactValue.
        if getattr(value, "handle", None) is not None or is_handle(value):
            return True
    tasks = getattr(env, "tasks", None)
    if isinstance(tasks, Mapping):
        return any(task_declares_lineage(t) for t in tasks.values())
    return False


def _walk(envs: Iterable[Any]) -> List[Any]:
    """`envs` and, transitively, their `depends_on` environments (each once), as a deploy plans them."""
    out: dict = {}
    stack = list(envs)
    while stack:
        env = stack.pop()
        if id(env) in out:
            continue
        out[id(env)] = env
        stack.extend(getattr(env, "depends_on", None) or [])
    return list(out.values())


def any_lineage(envs: Iterable[Any], labels: Optional[Mapping[str, str]] = None) -> bool:
    """
    True when a deploy of `envs` (and the environments they depend on) with deploy-wide `labels` declares any
    lineage: labels anywhere, an app with `lineage=True`, `consumes_artifacts` or a handle-bound parameter, or a
    task that produces handles (which is also how refresh policies enter a deploy) or consumes artifacts.
    """
    if labels:
        return True
    return any(env_declares_lineage(e) for e in _walk(envs))
