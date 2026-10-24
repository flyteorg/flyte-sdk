"""Sessions: child actions that serve traffic at an address while they run.

Warning: Experimental feature: alpha — APIs may change without notice.

A session is the address-shaped sibling of a condition. Both are child actions of the running task whose
result arrives inline on `ActionUpdate` rather than through outputs:

- a condition's signalled value arrives with its terminal update;
- a session's address arrives while it runs, as soon as the worker serving it is ready.

```python
from flyte.extend import new_session

sess = await new_session.aio("sandbox", task=task)
address = await sess.wait.aio()  # ready: a SessionAddress, where to reach it
...
await sess.abort.aio()  # done with it
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import rich.repr

from flyte._task import TaskTemplate
from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyteidl2.common import address_pb2
    from flyteidl2.task import task_definition_pb2

    from flyte.models import TaskContext

DEFAULT_READY_TIMEOUT_SECONDS = 30.0


@rich.repr.auto
@dataclass(frozen=True)
class SessionAddress:
    """
    Where a session can be reached, as its worker reported it. Use `url` when it is set; otherwise the session is
    served on a pod, at `pod_ip`:`pod_port` and `path`.

    Warning: Experimental feature: alpha — APIs may change without notice.

    Args:
        url: A URL that reaches the session, when its worker publishes one.
        path: The path the session is served at, on the pod or under `url`.
        pod_ip: IP of the pod serving the session.
        pod_port: Port on that pod.
        pod_name: Name of that pod.
        pod_namespace: Namespace of that pod.
        cluster: The cluster the session runs in.
    """

    url: Optional[str] = None
    path: str = ""
    pod_ip: Optional[str] = None
    pod_port: Optional[int] = None
    pod_name: Optional[str] = None
    pod_namespace: Optional[str] = None
    cluster: Optional[str] = None

    @classmethod
    def _from_proto(cls, address: "address_pb2.ActionAddress") -> SessionAddress:
        pod = address.pod if address.HasField("pod") else None
        return cls(
            url=address.url or None,
            path=address.path,
            pod_ip=(pod.ip or None) if pod else None,
            pod_port=(pod.port or None) if pod else None,
            pod_name=(pod.name or None) if pod else None,
            pod_namespace=(pod.namespace or None) if pod else None,
            cluster=address.cluster or None,
        )


@rich.repr.auto
@dataclass
class _Session:
    """
    A session started with `new_session`. Sessions never retry: a retry would replace the session with a fresh,
    empty one at a new address.

    Args:
        name: Name of the session, used for the action name.
        task: The task the session runs. The serving worker reads its type and template.
        ready_timeout: Seconds `wait` allows for the session to become addressable before aborting it.
        queue: Optional queue to run the session in.
    """

    name: str
    task: TaskTemplate
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS
    queue: Optional[str] = None
    #: Where the session is served, set by `wait`.
    address: Optional[SessionAddress] = field(default=None, init=False)
    #: The session's action name, set by the controller when it registers the session. Unique per session,
    #: so several sessions may share a `name`.
    action_name: Optional[str] = field(default=None, init=False)

    def __post_init__(self):
        if self.ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive")

    @syncify
    async def wait(self) -> SessionAddress:
        """
        Wait for the session to become addressable.

        Returns:
            The session's address.

        Raises:
            flyte.errors.ActionAbortedError: If the session is aborted before it is ready.
            flyte.errors.RuntimeSystemError: If it fails before it is ready, or is not ready within `ready_timeout`
                (it is aborted first).
        """
        from flyte._internal.controllers import get_controller

        self.address = SessionAddress._from_proto(await get_controller().wait_for_registered_action(self))
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
    task: TaskTemplate,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECONDS,
    queue: Optional[str] = None,
) -> _Session:
    """
    Start a session as a child action of the running task. Returns once it is enqueued; call `wait` for its
    address.

    Warning: Experimental feature: alpha — APIs may change without notice.

    Args:
        name: Name of the session.
        task: The task the session runs.
        ready_timeout: Seconds `wait` allows for the session to become addressable.
        queue: Optional queue to run the session in.

    Raises:
        RuntimeError: Outside a task.
    """
    from flyte._context import internal_ctx

    session = _Session(name=name, task=task, ready_timeout=ready_timeout, queue=queue)
    if not internal_ctx().is_task_context():
        raise RuntimeError("Sessions can only be started within a task.")
    from flyte._internal.controllers import get_controller

    await get_controller().register_action(session)
    return session


def _session_task_spec(task: TaskTemplate, tctx: "TaskContext") -> "task_definition_pb2.TaskSpec":
    """
    The wire spec of a session's task, serialized as a child task of the running task is. Sessions never retry: a
    retry would start a fresh, empty session at a new address while the caller still holds the old one, so the
    task's retries are cleared.
    """
    from flyte._internal.runtime.task_serde import translate_task_to_wire
    from flyte.models import SerializationContext

    action = tctx.action
    code_bundle = tctx.code_bundle
    serialization_context = SerializationContext(
        project=action.project,
        domain=action.domain,
        org=action.org,
        code_bundle=code_bundle,
        version=tctx.version,
        image_cache=tctx.compiled_image_cache,
        root_dir=Path(code_bundle.destination).absolute() if code_bundle else Path.cwd(),
    )
    spec = translate_task_to_wire(task, serialization_context, task_context=tctx)
    spec.task_template.metadata.ClearField("retries")
    return spec
