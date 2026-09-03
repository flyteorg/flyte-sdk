"""Launch-once semantics for event-driven workflows.

When something outside Flyte launches a run in reaction to an event — a webhook
receiver, a poller, an operator clicking retry — the same event often arrives
more than once. Webhook senders retry on any non-2xx response, pollers overlap
their windows, and people re-trigger by hand. `run_once` makes that safe: the
same event may be delivered any number of times and still produce one run.

Either way you get the run that covers the event, paired with a `created` flag
saying whether this call was the one that launched it. A handler that wants to
answer "already handled" and link to the existing run can; one that does not
care can ignore the flag and use `result.run` unconditionally.

Deduplication is keyed entirely on a run **label**. Every run launched this way
carries `dedupe=<key>`, and a launch is refused when a run already carrying that
key is live or has succeeded. Failed, aborted, and timed-out runs do not block:
re-triggering after a failure is a retry, which is what an operator wants.

Run *names* are deliberately not part of this. A name is an allocation detail,
not an identity — probing names for freeness races against concurrent launches
and caps how many runs one key can ever have. Names are left to the control
plane, which generates a fresh one per launch.

```python
from flyte.extras.webhooks import run_once

result = await run_once.aio(task, key=event_id, x=1)
if not result.created:
    ...  # an earlier delivery already launched result.run
```

The label check is a read followed by a launch, so two *simultaneous* deliveries
of one event can both observe no blocker and both launch. Redeliveries are
seconds to minutes apart and dedupe reliably; closing the concurrent case needs
a compare-and-set the control plane does not currently expose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

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


class RunOnceResult(NamedTuple):
    """The run covering a dedupe key, and whether this call created it.

    Unpacks as a plain tuple, so `run, created = await run_once.aio(...)` works.

    Attributes:
        run: The run carrying the key — freshly launched by this call, or the
            one an earlier delivery already launched.
        created: True when this call launched `run`; False when it found one.
    """

    run: Any
    created: bool


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
async def run_once(
    task: Any,
    *,
    key: str,
    copy_style: CopyFiles | None = None,
    runcontext_kwargs: dict[str, Any] | None = None,
    **inputs: Any,
) -> RunOnceResult:
    """Launch `task` once for `key`, returning the run that covers it.

    **Use `await run_once.aio(...)` inside an async handler.** The
    synchronous form blocks the calling thread until the launch completes; on an
    app's event loop that stalls every other in-flight request, and webhook
    senders time deliveries out in seconds.

    Args:
        task: The task to launch — either a `flyte.remote.Task` looked up by
            name, or a local `TaskEnvironment` task object.
        key: Stable dedupe key for the triggering event. Any string works: the
            key *is* the dedupe scope, so choose it to match what "the same
            event" means for your workflow.
        copy_style: Pass `"all"` when `task` is a local task object so the whole
            module tree is bundled. Leave as None when launching a `remote.Task`
            by name, which needs no bundle.
        runcontext_kwargs: Forwarded to `flyte.with_runcontext`, for anything
            else the run needs — `env_vars`, `queue`, `interruptible`,
            `service_account`, `notifications`, and so on:

            ```python
            await run_once.aio(
                task,
                key=event.dedupe_key(),
                runcontext_kwargs={"queue": "webhooks", "labels": {"team": "platform"}},
            )
            ```

            Labels merge with the `dedupe` label rather than replacing it, so
            extra labels are fine — but setting `dedupe` yourself is not, since
            it is what makes the launch unique.
        **inputs: Keyword inputs forwarded to the task.

    Returns:
        A `RunOnceResult` pairing the run that covers `key` with a `created`
        flag: True when this call launched it, False when a live or succeeded
        run already carried the key and is returned instead. Names of launched
        runs are assigned by the control plane.

    Raises:
        ValueError: when `runcontext_kwargs` sets `dedupe` to something other
            than `key`, or passes `copy_style` alongside the argument.
    """
    import flyte

    await _ensure_initialized()
    duplicate = await blocking_run.aio(key)
    if duplicate is not None:
        return RunOnceResult(run=duplicate, created=False)

    context_kwargs: dict[str, Any] = dict(runcontext_kwargs or {})
    if copy_style is not None:
        if "copy_style" in context_kwargs:
            raise ValueError("pass copy_style either directly or in runcontext_kwargs, not both")
        context_kwargs["copy_style"] = copy_style

    # Extra labels are welcome; the dedupe label is not the caller's to set,
    # because overwriting it is indistinguishable from turning idempotency off.
    labels: dict[str, str] = dict(context_kwargs.pop("labels", None) or {})
    if labels.get(DUPE_LABEL_KEY, key) != key:
        raise ValueError(
            f"runcontext_kwargs set the {DUPE_LABEL_KEY!r} label to {labels[DUPE_LABEL_KEY]!r}, "
            f"which would break idempotency — pass key={labels[DUPE_LABEL_KEY]!r} instead"
        )
    labels[DUPE_LABEL_KEY] = key

    context = flyte.with_runcontext(labels=labels, **context_kwargs)
    return RunOnceResult(run=await context.run.aio(task, **inputs), created=True)
