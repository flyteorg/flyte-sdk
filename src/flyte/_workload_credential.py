"""
The bearer credential a task pod presents when it authenticates with its workload identity.

The pod sends one credential, `authorization: Bearer <credential>`, shaped as a compact JWS:
`<header>.<payload>.<signature>`, each segment base64url-encoded without padding.

- header: `{"typ":"union-workload","alg":"none"}`.
- payload: `{"token": <projected service-account token>, "workload": <_U_WORKLOAD>}`, plus, for a pod workload with a
  known action, `"action": {"run": <run name>, "name": <action name>}`.
- signature: empty, so the credential ends with ".". It is unsigned for now; the JWS shape leaves room to sign it later.

A reusable worker pod runs many actions, possibly concurrently and for different runs, under one identity that names the
pod (`pod/<org>/<project>/<domain>/<cluster>/<namespace>/<pod-name>`). On its own that identity cannot tell the control
plane which of the pod's actions a request acts for, so the credential of such a pod also names the action, resolved in
this order:

1. The request's own Actions service metadata (`x-actions-run`, `x-actions-parent-action`): the controller sends it
   with every call it makes for a child action (enqueue, abort, watch), naming the parent action that makes the call.
   These run on the controller's own thread, where the task context of the action is not available.
2. The task context of the running task: calls made from the task's own code path, such as data uploads.

When neither names an action, the `action` key is left out. Attempt and app workloads never carry one.

This module only depends on the standard library, so it is cheap to import.
"""

from __future__ import annotations

import base64
import json
import typing

# (run name, action name)
Action = typing.Tuple[str, str]

CREDENTIAL_HEADER = {"typ": "union-workload", "alg": "none"}
# The prefix of a workload identity that names a pod rather than a single action attempt.
POD_WORKLOAD_PREFIX = "pod/"

_RUN_HEADER = "x-actions-run"
_PARENT_ACTION_HEADER = "x-actions-parent-action"


def is_pod_workload(workload: str) -> bool:
    """Whether a `_U_WORKLOAD` value names a pod (which may serve many actions) rather than one attempt or app."""
    return workload.startswith(POD_WORKLOAD_PREFIX)


def _action(run_name: typing.Optional[str], action_name: typing.Optional[str]) -> typing.Optional[Action]:
    """`(run name, action name)`, or None unless both are known."""
    if not run_name or not action_name:
        return None
    return run_name, action_name


def _from_task_context() -> typing.Optional[Action]:
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
    return _action(action.run_name, action.name)


def current_workload_action(
    request_headers: typing.Optional[typing.Mapping[str, str]] = None,
) -> typing.Optional[Action]:
    """The `(run name, action name)` a request with these headers is made for, or None when no action is known."""
    if request_headers is not None:
        explicit = _action(request_headers.get(_RUN_HEADER), request_headers.get(_PARENT_ACTION_HEADER))
        if explicit is not None:
            return explicit
    return _from_task_context()


def _segment(obj: dict) -> str:
    raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def encode_workload_credential(token: str, workload: str, action: typing.Optional[Action] = None) -> str:
    """The compact, unsigned credential for this token and workload, naming `action` (run name, action name) if set."""
    payload: dict = {"token": token, "workload": workload}
    if action is not None:
        payload["action"] = {"run": action[0], "name": action[1]}
    return f"{_segment(CREDENTIAL_HEADER)}.{_segment(payload)}."


def workload_credential(
    token: str, workload: str, request_headers: typing.Optional[typing.Mapping[str, str]] = None
) -> str:
    """
    The credential for a request with these headers: names the action the request is made for when the workload is a
    pod and the action is known.
    """
    action = current_workload_action(request_headers) if is_pod_workload(workload) else None
    return encode_workload_credential(token, workload, action)
