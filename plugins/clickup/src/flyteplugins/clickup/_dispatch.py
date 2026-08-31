"""Idempotent run launching for event-driven workflows.

When a webhook receiver launches a Flyte run in reaction to an external event,
the same event may be delivered more than once (ClickUp retries on non-2xx
responses, and operators re-trigger manually). This module makes that safe.

Idempotency is keyed entirely on a run **label**. Every event-driven run
carries `dedupe=<key>`, and a launch is refused when a run already carrying
that key is live or has succeeded. Failed / aborted / timed-out runs do not
block: re-triggering after a failure is a retry, which is what an operator
wants.

Run *names* are deliberately not part of this. A name is an allocation detail,
not an identity — probing names for freeness races against concurrent launches
and caps how many runs one key can ever have. Names are left to the control
plane, which generates a fresh one per launch.

The label check is a read followed by a launch, so two *simultaneous*
deliveries of one event can both observe no blocker and both launch. Redeliveries
are seconds to minutes apart and dedupe reliably; closing the concurrent case
needs a compare-and-set the control plane does not currently expose.
"""

from __future__ import annotations

from typing import Any

from flyte.syncify import syncify

#: Run label that carries the dedupe key.
DUPE_LABEL_KEY = "dedupe"

#: Terminal phases that unblock a key. A run in any live phase, or one that
#: SUCCEEDED, means the work is in flight or done — a second launch would be a
#: duplicate.
_RETRIABLE_PHASES = ("FAILED", "ABORTED", "TIMED_OUT")

#: How many runs carrying a key to scan when looking for a blocker.
_LOOKBACK_LIMIT = 200


class DuplicateRun(Exception):
    """Raised when this dedupe key already has a live or succeeded run."""

    def __init__(self, run_name: str, url: str = ""):
        self.run_name = run_name
        self.url = url
        super().__init__(f"run {run_name!r} already covers this key: {url or '(no url)'}")


async def _ensure_flyte_initialized() -> None:
    """Initialize the SDK against the surrounding cluster when needed.

    Webhook handlers run in an app process, not a task, so the SDK is
    not initialized automatically. `init_in_cluster` uses the app's own identity, so launched
    runs are attributed to the app rather than a person.
    """
    import flyte
    from flyte._initialize import _get_init_config

    if _get_init_config() is None:
        await flyte.init_in_cluster.aio()


def _is_retriable(phase: str) -> bool:
    phase = phase.upper()
    return any(p in phase for p in _RETRIABLE_PHASES)


@syncify
async def blocking_run(key: str) -> Any:
    """Return the run that blocks this key, or None.

    A key is blocked while any run carrying its label is live or succeeded.

    Call `blocking_run(key)` from sync code, or `await blocking_run.aio(key)`
    from an async handler.
    """
    import flyte.remote as remote

    await _ensure_flyte_initialized()
    async for run in remote.Run.listall.aio(with_labels={DUPE_LABEL_KEY: key}, limit=_LOOKBACK_LIMIT):
        if not _is_retriable(str(run.phase)):
            return run
    return None


@syncify
async def launch_task(
    task: Any,
    *,
    key: str,
    copy_style: str = "",
    **inputs: Any,
) -> Any:
    """Launch `task` idempotently for `key`, or raise `DuplicateRun`.

    **Use `await launch_task.aio(...)` inside an async handler.** The synchronous
    form blocks the calling thread until the launch completes; on an app's event
    loop that stalls every other in-flight request, and ClickUp times webhook
    deliveries out in seconds.

    Args:
        task: The task to launch — either a `flyte.remote.Task` looked up by
            name, or a local `TaskEnvironment` task object.
        key: Stable dedupe key for the triggering event. `ClickUpEvent.dedupe_key()`
            supplies a sensible default, but any string works — pass your own to
            choose a different idempotency scope.
        copy_style: Pass `"all"` when `task` is a local task object so the
            whole module tree is bundled. Leave empty when launching a
            `remote.Task` by name.
        **inputs: Keyword inputs forwarded to the task.

    Returns:
        The launched run handle. Its name is assigned by the control plane.

    Raises:
        DuplicateRun: when a live or succeeded run already carries this key.
    """
    import flyte

    await _ensure_flyte_initialized()
    dup = await blocking_run.aio(key)
    if dup is not None:
        raise DuplicateRun(dup.name, dup.url)

    context = flyte.with_runcontext(
        labels={DUPE_LABEL_KEY: key},
        **({"copy_style": copy_style} if copy_style else {}),
    )
    return await context.run.aio(task, **inputs)
