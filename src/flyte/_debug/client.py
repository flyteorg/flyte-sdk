"""Client-side helpers for VSCode debug mode.

These utilities are used by `flyte.run` and the `flyte run` CLI to poll a
remote run for the VS Code Debugger URL and print it to the user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from flyte.remote._action import ActionDetails
    from flyte.remote._run import Run


def _extract_vscode_uri(details: ActionDetails) -> str | None:
    """Return the VS Code Debugger URI from action attempts, or `None`.

    Only returns the URI when the attempt also has a cluster event whose
    `message` equals `"Vscode server is ready"`, indicating that the
    code-server is actually accepting connections.
    """
    from flyteidl2.core.execution_pb2 import TaskLog

    for attempt in details.pb2.attempts:
        for log in attempt.log_info:
            if log.ready and log.link_type == TaskLog.LinkType.IDE:
                return log.uri
    return None


def _build_full_url(endpoint: str, uri: str) -> str:
    """Combine the tenant endpoint with a relative TaskLog URI."""
    parsed = urlparse(endpoint)
    if parsed.scheme == "dns":
        domain = parsed.path.lstrip("/")
    else:
        domain = parsed.netloc or parsed.path
    domain = domain.split(":")[0]
    return f"https://{domain}{uri}"


async def watch_for_vscode_url(run: Run) -> str | None:
    """Poll a remote run until the VS Code Debugger URL appears.

    Watches the run's action details for a `TaskLog` entry named
    *"VS Code Debugger"* and combines its URI with the configured endpoint.

    Returns the full clickable URL, or `None` if the URL is not found within
    *timeout* seconds or the run reaches a terminal state first.
    """
    from flyte._initialize import get_client

    client = get_client()
    endpoint = client.endpoint

    async for action_details in run.action.watch():
        uri = _extract_vscode_uri(action_details)
        if uri:
            return _build_full_url(endpoint, uri)
        if action_details.done():
            return None
    return None
