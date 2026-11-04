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
class SessionRoutingInfo:
    """
    Where in the platform a session is served: the pod its worker serves it from, and the cluster.

    Warning: Experimental feature: alpha — APIs may change without notice.

    Args:
        ip: IP of the pod serving the session.
        port: Port on that pod.
        name: Name of that pod.
        namespace: Namespace of that pod.
        cluster: The cluster the session runs in.
    """

    ip: Optional[str] = None
    port: Optional[int] = None
    name: Optional[str] = None
    namespace: Optional[str] = None
    cluster: Optional[str] = None


@rich.repr.auto
@dataclass(frozen=True)
class SessionAddress:
    """
    Where a session can be reached, as its worker reported it. Use `url` when it is set; otherwise the session is
    served at `routing_info`'s `ip`:`port` and `path`.

    Warning: Experimental feature: alpha — APIs may change without notice.

    Args:
        url: A URL that reaches the session, when its worker publishes one.
        path: The path the session is served at, under `url` or on the pod.
        routing_info: Where in the platform the session is served, when its worker reports it.
    """

    url: Optional[str] = None
    path: str = ""
    routing_info: Optional[SessionRoutingInfo] = None

    @classmethod
    def _from_proto(cls, address: "address_pb2.ActionAddress") -> SessionAddress:
        routing_info = None
        if address.HasField("pod") or address.cluster:
            pod = address.pod
            routing_info = SessionRoutingInfo(
                ip=pod.ip or None,
                port=pod.port or None,
                name=pod.name or None,
                namespace=pod.namespace or None,
                cluster=address.cluster or None,
            )
        return cls(url=address.url or None, path=address.path, routing_info=routing_info)


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
    #: Why the session ended, once `wait` has seen it end or `abort` has stopped it: every later `wait` raises it.
    _ended: Optional[Exception] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.ready_timeout <= 0:
            raise ValueError("ready_timeout must be positive")

    @syncify
    async def wait(self) -> SessionAddress:
        """
        Wait for the session to become addressable. Safe to call again: while the session runs, each call returns its
        address at once; once it has ended, each call raises why.

        Returns:
            The session's address.

        Raises:
            flyte.errors.ActionAbortedError: If the session was aborted.
            flyte.errors.RuntimeSystemError: If it ended, failing or not, or was not ready within `ready_timeout` (it
                is aborted then).
        """
        from flyte._internal.controllers import get_controller

        if self._ended is not None:
            raise self._ended
        try:
            address = await get_controller().wait_for_registered_action(self)
        except Exception as e:
            # The controller stops tracking a session once it has ended, so how it ended is kept here.
            self._ended = e
            raise
        self.address = SessionAddress._from_proto(address)
        return self.address

    @syncify
    async def abort(self, reason: str = "session closed") -> None:
        """Abort the session. Safe to call more than once. A later `wait` raises `flyte.errors.ActionAbortedError`."""
        import flyte.errors
        from flyte._internal.controllers import get_controller

        await get_controller().abort_registered_action(self, reason)
        if self._ended is None:
            self._ended = flyte.errors.ActionAbortedError(
                f"Session '{self.name}' ({self.action_name}) was aborted: {reason}"
            )


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
