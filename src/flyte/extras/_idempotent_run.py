"""Idempotent run launching for event-driven workflows.

When something outside Flyte launches a run in reaction to an event — a webhook
receiver, a poller, an operator clicking retry — the same event often arrives
more than once. Webhook senders retry on any non-2xx response, pollers overlap
their windows, and people re-trigger by hand. `idempotent_run` makes that safe.

Idempotency is keyed entirely on a run **label**. Every run launched this way
carries `dedupe=<key>`, and a launch is refused when a run already carrying that
key is live or has succeeded. Failed, aborted, and timed-out runs do not block:
re-triggering after a failure is a retry, which is what an operator wants.

Run *names* are deliberately not part of this. A name is an allocation detail,
not an identity — probing names for freeness races against concurrent launches
and caps how many runs one key can ever have. Names are left to the control
plane, which generates a fresh one per launch.

```python
from flyte.extras import DuplicateRun, idempotent_run

try:
    run = await idempotent_run.aio(task, key=event_id, x=1)
except DuplicateRun as exc:
    ...  # this event already has a run; exc.url points at it
```

The label check is a read followed by a launch, so two *simultaneous* deliveries
of one event can both observe no blocker and both launch. Redeliveries are
seconds to minutes apart and dedupe reliably; closing the concurrent case needs
a compare-and-set the control plane does not currently expose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyte._code_bundle import CopyFiles

#: Run label that carries the dedupe key.
DUPE_LABEL_KEY = "dedupe"

#: Phases that leave a key free. A run in any live phase, or one that SUCCEEDED,
#: means the work is in flight or done — a second launch would be a duplicate.
_RETRIABLE_PHASES = ("FAILED", "ABORTED", "TIMED_OUT")

#: How many runs carrying a key to scan when looking for a blocker.
_LOOKBACK_LIMIT = 200


class DuplicateRun(Exception):
    """Raised when a dedupe key already has a live or succeeded run.

    Attributes:
        run_name: Name of the run that already covers the key.
        url: Link to that run, when the backend supplied one.
    """

    def __init__(self, run_name: str, url: str = ""):
        self.run_name = run_name
        self.url = url
        super().__init__(f"run {run_name!r} already covers this key: {url or '(no url)'}")


async def _ensure_initialized() -> None:
    """Initialize against the surrounding cluster when nothing else has.

    App processes are not tasks, so the SDK is not initialized for them
    automatically. `init_in_cluster` uses the app's own identity, so runs
    launched from an app are attributed to the app rather than to a person.
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

    Args:
        key: The dedupe key to look up.

    Returns:
        The blocking `flyte.remote.Run`, or None when the key is free.
    """
    import flyte.remote as remote

    await _ensure_initialized()
    async for run in remote.Run.listall.aio(with_labels={DUPE_LABEL_KEY: key}, limit=_LOOKBACK_LIMIT):
        if not _is_retriable(str(run.phase)):
            return run
    return None


@syncify
async def idempotent_run(
    task: Any,
    *,
    key: str,
    copy_style: CopyFiles | None = None,
    **inputs: Any,
) -> Any:
    """Launch `task` once for `key`, or raise `DuplicateRun`.

    **Use `await idempotent_run.aio(...)` inside an async handler.** The
    synchronous form blocks the calling thread until the launch completes; on an
    app's event loop that stalls every other in-flight request, and webhook
    senders time deliveries out in seconds.

    Args:
        task: The task to launch — either a `flyte.remote.Task` looked up by
            name, or a local `TaskEnvironment` task object.
        key: Stable dedupe key for the triggering event. Any string works: the
            key *is* the idempotency scope, so choose it to match what "the same
            event" means for your workflow.
        copy_style: Pass `"all"` when `task` is a local task object so the whole
            module tree is bundled. Leave as None when launching a `remote.Task`
            by name, which needs no bundle.
        **inputs: Keyword inputs forwarded to the task.

    Returns:
        The launched run handle. Its name is assigned by the control plane.

    Raises:
        DuplicateRun: when a live or succeeded run already carries this key.
    """
    import flyte

    await _ensure_initialized()
    duplicate = await blocking_run.aio(key)
    if duplicate is not None:
        raise DuplicateRun(duplicate.name, duplicate.url)

    labels = {DUPE_LABEL_KEY: key}
    context = (
        flyte.with_runcontext(labels=labels, copy_style=copy_style)
        if copy_style is not None
        else flyte.with_runcontext(labels=labels)
    )
    return await context.run.aio(task, **inputs)
