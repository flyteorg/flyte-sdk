"""Sessions: child actions that serve traffic at an address while they run.

Private. Used by the sandbox plugin to start interactive sandbox sessions; not exported from `flyte`.

A session is the address-shaped sibling of a condition. Both are child actions of the running task whose
result arrives inline on `ActionUpdate` rather than through outputs:

- a condition's signalled value arrives with its terminal update;
- a session's `ActionAddress` arrives while it runs, as soon as the worker serving it is ready.

```python
sess = await new_session.aio("sandbox", task_spec=spec)
address = await sess.wait.aio()  # ready: where to reach it
...
await sess.abort.aio()  # done with it
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import rich.repr

from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyteidl2.common import address_pb2
    from flyteidl2.task import task_definition_pb2

DEFAULT_READY_TIMEOUT_SECONDS = 30.0


@rich.repr.auto
@dataclass
class _Session:
    """
    A session started with `new_session`. Sessions never retry: a retry would replace the session with a fresh,
    empty one at a new address.

    Args:
        name: Name of the session, used for the action name.
        task_spec: The task the session runs. The serving worker reads its type and template.
        ready_timeout: Seconds `wait` allows for the session to become addressable before aborting it.
        queue: Optional queue to run the session in.
    """

    name: str
    task_spec: "task_definition_pb2.TaskSpec"
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    queue: Optional[str] = None
    #: Where the session is served, set by `wait`.
    address: Optional["address_pb2.ActionAddress"] = field(default=None, init=False)
    #: The session's action name, set by the controller when it registers the session. Unique per session,
    #: so several sessions may share a `name`.
    action_name: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive")

    @syncify
    async def wait(self) -> "address_pb2.ActionAddress":
        """
        Wait for the session to become addressable.

        Returns:
            The session's address. Resolve it with `url` when set, else the pod's `ip`, `port` and `path`.

        Raises:
            flyte.errors.ActionAbortedError: If the session is aborted before it is ready.
            flyte.errors.RuntimeSystemError: If it fails before it is ready, or is not ready within `ready_timeout`
                (it is aborted first).
        """
        from flyte._internal.controllers import get_controller

        self.address = await get_controller().wait_for_registered_action(self)
        return self.address

    @syncify
    async def abort(self, reason: str = "session closed") -> None:
        """Abort the session. Safe to call more than once."""
        from flyte._internal.controllers import get_controller

        await get_controller().abort_registered_action(self, reason)


@syncify
async def new_session(
    name: str,
    /,
    task_spec: "task_definition_pb2.TaskSpec",
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    queue: Optional[str] = None,
) -> _Session:
    """
    Start a session as a child action of the running task. Returns once it is enqueued; call `wait` for its
    address.

    Args:
        name: Name of the session.
        task_spec: The task the session runs.
        ready_timeout: Seconds `wait` allows for the session to become addressable.
        queue: Optional queue to run the session in.

    Raises:
        RuntimeError: Outside a task.
    """
    from flyte._context import internal_ctx

    session = _Session(name=name, task_spec=task_spec, ready_timeout=ready_timeout, queue=queue)
    if not internal_ctx().is_task_context():
        raise RuntimeError("Sessions can only be started within a task.")
    from flyte._internal.controllers import get_controller

    await get_controller().register_action(session)
    return session
