"""The tool bridge — let the harness runtime call Flyte tasks as durable actions.

The harness runs the model loop in its own subprocess and only ever calls tools
that its Cordis composition provides. The one such tool every composition ships
is local bash, scoped to the workspace directory the adapter chooses. So the
bridge meets it there:

1. each `HarnessTool` is published into `<workspace>/.flyte_tools/<name>` as an
   executable shim (stdlib-only Python, run under this process's own
   interpreter, so it needs nothing installed in the harness runtime);
2. this process listens on a Unix domain socket next to the run;
3. the model runs `.flyte_tools/get_weather '{"city": "Paris"}'` via bash, the
   shim forwards the JSON arguments over the socket, and the bridge awaits
   `task.aio(...)` — a durable Flyte child action — before writing the result
   back for the shim to print.

The shim speaks one newline-delimited JSON request/response per connection, so
concurrent tool calls are just concurrent connections.

The socket lives in a private temp directory rather than the workspace: the
model can read and write the workspace, and a socket it can reach directly is
one more thing to reason about. The workspace only ever holds the shims.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import sys
import tempfile
import typing
from pathlib import Path

from flyte._logging import logger
from flyteplugins.agents.core import ReportTimeline, abbrev

if typing.TYPE_CHECKING:
    from ._tools import HarnessTool

# Directory (relative to the harness workspace) holding the published shims.
TOOLS_DIRNAME = ".flyte_tools"

# Written verbatim into each shim; ``SOCKET_PATH`` / ``TOOL_NAME`` are filled in
# per tool. Stdlib only, so any Python 3 interpreter can run it.
_SHIM = '''\
#!{interpreter}
"""Flyte tool shim for {tool_name!r} — forwards to the Flyte task over a socket."""
import json
import socket
import sys

SOCKET_PATH = {socket_path!r}
TOOL_NAME = {tool_name!r}


def main() -> int:
    raw = (sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()).strip() or "{{}}"
    try:
        args = json.loads(raw)
    except ValueError as exc:
        print(f"error: arguments must be a JSON object ({{exc}}): {{raw}}", file=sys.stderr)
        return 2
    if not isinstance(args, dict):
        print("error: arguments must be a JSON object, e.g. '{{\\"city\\": \\"Paris\\"}}'", file=sys.stderr)
        return 2

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
    except OSError as exc:
        print(f"error: the Flyte tool bridge is not reachable: {{exc}}", file=sys.stderr)
        return 3
    try:
        stream = sock.makefile("rwb")
        stream.write((json.dumps({{"tool": TOOL_NAME, "args": args}}) + "\\n").encode("utf-8"))
        stream.flush()
        line = stream.readline()
    finally:
        sock.close()

    if not line:
        print("error: the Flyte tool bridge closed without responding", file=sys.stderr)
        return 3
    response = json.loads(line.decode("utf-8"))
    if response.get("ok"):
        print(response.get("result", ""))
        return 0
    print(f"error: {{response.get('error') or 'tool call failed'}}", file=sys.stderr)
    return 1


sys.exit(main())
'''


class ToolBridge:
    """Publish `HarnessTool`s into a workspace and serve their callbacks.

    Use as an async context manager around the harness run; on exit the socket
    and its temp directory are removed (the published shims are left in the
    workspace, which the adapter owns and cleans up separately).
    """

    def __init__(self, tools: typing.Sequence["HarnessTool"], *, timeline: ReportTimeline | None = None) -> None:
        self._tools = {t.name: t for t in tools}
        self._timeline = timeline
        self._server: asyncio.AbstractServer | None = None
        self._sockdir: str | None = None
        self._socket_path: str | None = None
        self._tools_dir: Path | None = None

    @property
    def socket_path(self) -> str | None:
        return self._socket_path

    @property
    def tools_dir(self) -> Path | None:
        return self._tools_dir

    async def start(self, workspace: Path) -> None:
        """Start the callback server and publish a shim per tool into `workspace`."""
        if not self._tools:
            return
        self._sockdir = tempfile.mkdtemp(prefix="flyte-dsh-")
        self._socket_path = os.path.join(self._sockdir, "tools.sock")
        self._server = await asyncio.start_unix_server(self._handle, path=self._socket_path)
        self._publish(workspace)

    async def stop(self) -> None:
        """Close the server and remove the socket directory. Never raises."""
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # pragma: no cover - already torn down
                logger.debug("DeepSeek tool bridge: server close raced with shutdown", exc_info=True)
            self._server = None
        if self._sockdir:
            shutil.rmtree(self._sockdir, ignore_errors=True)
            self._sockdir = None
            self._socket_path = None

    async def __aenter__(self) -> "ToolBridge":  # pragma: no cover - exercised via run_agent
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:  # pragma: no cover - exercised via run_agent
        await self.stop()

    def _publish(self, workspace: Path) -> None:
        """Write one executable shim per tool into `<workspace>/.flyte_tools/`."""
        tools_dir = workspace / TOOLS_DIRNAME
        tools_dir.mkdir(parents=True, exist_ok=True)
        for name in self._tools:
            shim = tools_dir / name
            shim.write_text(
                _SHIM.format(interpreter=sys.executable, socket_path=self._socket_path, tool_name=name),
                encoding="utf-8",
            )
            shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self._tools_dir = tools_dir

    def instructions(self) -> str:
        """The tool manual handed to the model, or `""` when there are no tools.

        The harness has no tool-declaration channel, so the contract is stated in
        prose: what exists, how to call it, and what comes back.
        """
        if not self._tools:
            return ""
        lines = [
            "## Flyte tools",
            "",
            (
                "The following tools are available as executable commands in "
                f"`{TOOLS_DIRNAME}/` under your working directory. Run one with bash, passing its"
            ),
            "arguments as a single JSON object argument:",
            "",
            f"    {TOOLS_DIRNAME}/<tool_name> '<json_arguments>'",
            "",
            "The tool prints its result to stdout and exits 0; on failure it prints the reason",
            "to stderr and exits non-zero. Prefer these tools over writing your own code for the",
            "same job — each one runs as a tracked, retried step of this workflow.",
            "",
            "Available tools:",
            "",
        ]
        for name, harness_tool in self._tools.items():
            lines.append(f"- `{harness_tool.usage()}`")
            lines.append(f"  example: `{TOOLS_DIRNAME}/{name} '{_example_args(harness_tool)}'`")
        return "\n".join(lines)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one shim callback: decode, dispatch to the task, write the result."""
        try:
            line = await reader.readline()
            if not line:
                return
            response = await self._dispatch(json.loads(line.decode("utf-8")))
            writer.write((json.dumps(response, default=str) + "\n").encode("utf-8"))
            await writer.drain()
        except Exception as exc:  # pragma: no cover - transport-level failure
            logger.warning("DeepSeek tool bridge: failed to serve a tool call: %s", exc)
        finally:
            writer.close()

    async def _dispatch(self, request: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Run the requested tool, returning the shim's `{ok, result|error}` reply."""
        name = request.get("tool")
        args = request.get("args") or {}
        harness_tool = self._tools.get(name)
        if harness_tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}; available: {', '.join(self._tools)}"}

        if self._timeline is not None:
            self._timeline.row(icon="🛠️", label=name, meta="tool", detail=abbrev(args, 160))
        try:
            result = await harness_tool.invoke(args)
        except Exception as exc:
            # A failed tool is reported to the model, not raised: the agent gets to
            # see the error and decide what to do, exactly as with a native tool.
            if self._timeline is not None:
                self._timeline.row(icon="❌", label=name, meta="tool error", detail=abbrev(exc, 160), error="error")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if self._timeline is not None:
            self._timeline.row(icon="🔧", label=name, meta="tool result", detail=abbrev(result, 160))
        return {"ok": True, "result": result}


def _example_args(harness_tool: "HarnessTool") -> str:
    """A filled-in example argument object, so the model sees the exact shape."""
    placeholders = {
        "string": "...",
        "integer": 0,
        "number": 0,
        "boolean": True,
        "array": [],
        "object": {},
    }
    properties = harness_tool.schema.get("properties") or {}
    example = {}
    for key, spec in properties.items():
        declared = spec.get("type") if isinstance(spec, dict) else None
        if isinstance(spec, dict) and spec.get("enum"):
            example[key] = spec["enum"][0]
        else:
            example[key] = placeholders.get(declared if isinstance(declared, str) else "", "...")
    return json.dumps(example)
