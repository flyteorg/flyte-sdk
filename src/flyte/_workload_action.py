"""
The action a request is made for, for pods that authenticate with a pod workload identity.

A reusable worker pod runs many actions, possibly concurrently and for different runs, under one identity that names
the pod (`pod/<org>/<project>/<domain>/<cluster>/<namespace>/<pod-name>`). On its own that identity cannot tell the
control plane which of the pod's actions a request acts for, so every request sent with a pod workload token also names
the action, as `<run-name>/<action-name>`, in the `x-union-workload-action` header. The auth interceptor adds it (see
`WorkloadTokenAuthenticator.workload_action_headers`), resolving the action in this order:

1. The request's own Actions service metadata (`x-actions-run`, `x-actions-parent-action`): the controller sends it
   with every call it makes for a child action, naming the parent action that makes the call.
2. `acting_for(...)`: set by code that makes a call for an action it holds but that has no such metadata, such as the
   controller's queue and state service calls. These run on the controller's own thread, where the task context of
   the action is not available.
3. The task context of the running task: calls made from the task's own code path.

When none of them names an action, the header is left out.

This module only depends on the standard library, so the controller can import it cheaply.
"""

from __future__ import annotations

import contextlib
import contextvars
import typing

WORKLOAD_ACTION_HEADER = "x-union-workload-action"
# The prefix of a workload identity that names a pod rather than a single action attempt.
POD_WORKLOAD_PREFIX = "pod/"

_RUN_HEADER = "x-actions-run"
_PARENT_ACTION_HEADER = "x-actions-parent-action"

_acting_for: contextvars.ContextVar[typing.Optional[str]] = contextvars.ContextVar(
    "flyte_workload_action", default=None
)


def is_pod_workload(workload: str) -> bool:
    """Whether a `_U_WORKLOAD` value names a pod (which may serve many actions) rather than one attempt or app."""
    return workload.startswith(POD_WORKLOAD_PREFIX)


def format_action(run_name: typing.Optional[str], action_name: typing.Optional[str]) -> typing.Optional[str]:
    """`<run-name>/<action-name>`, or None unless both are known."""
    if not run_name or not action_name:
        return None
    return f"{run_name}/{action_name}"


@contextlib.contextmanager
def acting_for(run_name: typing.Optional[str], action_name: typing.Optional[str]) -> typing.Iterator[None]:
    """
    Within this block, requests are made for the action `action_name` of run `run_name`.

    Context variables follow the asyncio task, so this covers the calls awaited (or streams iterated) inside the block,
    and tasks created inside it.
    """
    token = _acting_for.set(format_action(run_name, action_name))
    try:
        yield
    finally:
        _acting_for.reset(token)


def _from_task_context() -> typing.Optional[str]:
    try:
        from flyte._context import internal_ctx

        tctx = internal_ctx().data.task_context
    except Exception:
        return None
    if tctx is None:
        return None
    # Within a @trace, `action` is the trace's; the pod runs the task action.
    action = tctx.task_action or tctx.action
    if action is None:
        return None
    return format_action(action.run_name, action.name)


def current_workload_action(request_headers: typing.Optional[typing.Mapping[str, str]] = None) -> typing.Optional[str]:
    """The `<run-name>/<action-name>` a request with these headers is made for, or None when no action is known."""
    if request_headers is not None:
        explicit = format_action(request_headers.get(_RUN_HEADER), request_headers.get(_PARENT_ACTION_HEADER))
        if explicit is not None:
            return explicit
    held = _acting_for.get()
    if held is not None:
        return held
    return _from_task_context()
