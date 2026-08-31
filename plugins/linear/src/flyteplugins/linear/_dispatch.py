"""Idempotent run launching for event-driven workflows.

When a webhook receiver launches a Flyte run in reaction to an external event,
the same event may be delivered more than once (Linear retries on non-2xx
responses, and operators re-trigger manually). This module makes that safe:

1. Every event-driven run carries a `dedupe` label derived from the event.
   Before launching, we query for a live or already-succeeded run with that
   label and refuse to launch a duplicate.
2. Failed / aborted / timed-out runs do *not* block: re-triggering after a
   failure is a retry, which is what an operator wants.
3. The run name is allocated to be free before launch, since the control plane
   treats a launch under an existing name as a silent no-op.
"""

from __future__ import annotations

import re
from typing import Any

DUPE_LABEL_KEY = "dedupe"

#: Terminal phases that unblock a key. A run in any live phase, or one that
#: SUCCEEDED, means the work is in flight or done — a second launch would be a
#: duplicate.
_RETRIABLE_PHASES = ("FAILED", "ABORTED", "TIMED_OUT")

#: The control plane caps run names at 30 characters.
RUN_NAME_MAX = 30

_MAX_NAME_ATTEMPTS = 32


class DuplicateRun(Exception):
    """Raised when this dedupe key already has a live or succeeded run."""

    def __init__(self, run_name: str, url: str = ""):
        self.run_name = run_name
        self.url = url
        super().__init__(f"run {run_name!r} already covers this key: {url or '(no url)'}")


def _ensure_flyte_initialized() -> None:
    """Initialize the SDK against the surrounding cluster when needed.

    Webhook handlers run in an app process, not a task, so the SDK is not
    initialized automatically. `init_in_cluster` uses the app's own identity,
    so launched runs are attributed to the app rather than a person.
    """
    import flyte
    from flyte._initialize import _get_init_config

    if _get_init_config() is None:
        flyte.init_in_cluster()


def run_name_for(key: str, prefix: str = "ln") -> str:
    """Turn a dedupe key into a legal Flyte run name base.

    Run names must be lowercase alphanumeric and are capped at 30 characters.
    The returned name is a *base*: `launch_task` suffixes it when the base is
    occupied by a run that no longer blocks (e.g. an aborted predecessor).
    """
    slug = re.sub(r"[^a-z0-9]", "", f"{prefix}{key}".lower())
    return slug[:RUN_NAME_MAX]


def blocking_run(key: str) -> Any:
    """Return the run that blocks this key, or None.

    A key is blocked while any run carrying its label is live or succeeded.
    """
    import flyte.remote as remote

    _ensure_flyte_initialized()
    for run in remote.Run.listall(with_labels={DUPE_LABEL_KEY: key}, limit=200):
        if not _is_retriable(str(run.phase)):
            return run
    return None


def _is_retriable(phase: str) -> bool:
    phase = phase.upper()
    return any(p in phase for p in _RETRIABLE_PHASES)


def _unique_name(base: str, attempt: int) -> str:
    slug = re.sub(r"[^a-z0-9]", "", base.lower())[:RUN_NAME_MAX]
    if attempt == 0:
        return slug
    suffix = str(attempt)
    return slug[: RUN_NAME_MAX - len(suffix)] + suffix


def _run_exists(name: str) -> bool:
    import flyte.remote as remote

    try:
        return remote.Run.get(name=name) is not None
    except Exception:
        return False


def _allocate_name(base: str) -> str:
    """Find a free run name at or near `base`."""
    for attempt in range(_MAX_NAME_ATTEMPTS):
        name = _unique_name(base, attempt)
        if not _run_exists(name):
            return name
    raise RuntimeError(f"could not allocate a run name for base {base!r}")


def launch_task(
    task: Any,
    *,
    key: str,
    run_name_base: str | None = None,
    prefix: str = "ln",
    copy_style: str = "",
    **inputs: Any,
) -> Any:
    """Launch `task` idempotently for `key`, or raise `DuplicateRun`.

    Args:
        task: The task to launch — either a `flyte.remote.Task` looked up by
            name, or a local `TaskEnvironment` task object.
        key: Stable dedupe key for the triggering event
            (`LinearEvent.dedupe_key()`).
        run_name_base: Optional explicit run-name base; defaults to
            `run_name_for(key, prefix)`.
        prefix: Prefix used when deriving the run name from the key.
        copy_style: Pass `"all"` when `task` is a local task object so the
            whole module tree is bundled. Leave empty when launching a
            `remote.Task` by name.
        **inputs: Keyword inputs forwarded to the task.

    Returns:
        The launched run handle.

    Raises:
        DuplicateRun: when a live or succeeded run already carries this key.
    """
    import flyte

    _ensure_flyte_initialized()
    dup = blocking_run(key)
    if dup is not None:
        raise DuplicateRun(dup.name, dup.url)

    base = run_name_base or run_name_for(key, prefix)
    name = _allocate_name(base)
    context = flyte.with_runcontext(
        name=name,
        labels={DUPE_LABEL_KEY: key},
        **({"copy_style": copy_style} if copy_style else {}),
    )
    try:
        return context.run(task, **inputs)
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message or "alreadyexists" in message:
            dup = blocking_run(key)
            if dup is not None:
                raise DuplicateRun(dup.name, dup.url) from exc
        raise
