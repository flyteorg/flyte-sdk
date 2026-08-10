import asyncio
from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator, Iterator

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.app import app_definition_pb2, app_logs_payload_pb2, replica_definition_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.logs.dataplane import payload_pb2
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from flyte._initialize import ensure_client, get_client
from flyte._logging import logger
from flyte._tools import ipython_check, ipywidgets_check
from flyte.errors import LogsNotYetAvailableError
from flyte.syncify import syncify

style_map = {
    payload_pb2.LogLineOriginator.SYSTEM: "bold magenta",
    payload_pb2.LogLineOriginator.USER: "cyan",
    payload_pb2.LogLineOriginator.UNKNOWN: "light red",
}


def _format_line(logline: payload_pb2.LogLine, show_ts: bool, filter_system: bool) -> Text | None:
    """
    Format a log line for display with optional timestamp and system filtering.

    Args:
        logline: The log line protobuf to format.
        show_ts: Whether to include timestamps.
        filter_system: Whether to filter out system log lines.

    Returns:
        A formatted Text object or None if the line should be filtered out.
    """
    if filter_system:
        if logline.originator == payload_pb2.LogLineOriginator.SYSTEM:
            return None
    style = style_map.get(logline.originator, "")
    if "[flyte]" in logline.message and "flyte.errors" not in logline.message:
        if filter_system:
            return None
        style = "dim"
    ts = ""
    if show_ts:
        ts = f"[{logline.timestamp.ToDatetime().isoformat()}]"
    return Text(f"{ts} {logline.message}", style=style)


class AsyncLogViewer:
    """
    A class to view logs asynchronously in the console or terminal or jupyter notebook.
    """

    def __init__(
        self,
        log_source: AsyncIterator,
        max_lines: int = 30,
        name: str = "Logs",
        show_ts: bool = False,
        filter_system: bool = False,
        panel: bool = False,
    ):
        self.console = Console()
        self.log_source = log_source
        self.max_lines = max_lines
        self.lines: deque = deque(maxlen=max_lines + 1)
        self.name = name
        self.show_ts = show_ts
        self.total_lines = 0
        self.filter_flyte = filter_system
        self.panel = panel

    def _render(self) -> Panel | Text:
        """
        Render the current log lines as a Panel or Text object for display.
        """
        log_text = Text()
        for line in self.lines:
            log_text.append(line)
        if self.panel:
            return Panel(log_text, title=self.name, border_style="yellow")
        return log_text

    async def run(self):
        """
        Run the log viewer, streaming and displaying log lines until completion.
        """
        with Live(self._render(), refresh_per_second=20, console=self.console) as live:
            try:
                async for logline in self.log_source:
                    formatted = _format_line(logline, show_ts=self.show_ts, filter_system=self.filter_flyte)
                    if formatted:
                        self.lines.append(formatted)
                    self.total_lines += 1
                    live.update(self._render())
            except asyncio.CancelledError:
                pass
            except KeyboardInterrupt:
                pass
            except StopAsyncIteration:
                self.console.print("[dim]Log stream ended.[/dim]")
            except LogsNotYetAvailableError as e:
                self.console.print(f"[red]Error:[/red] {e}")
                live.update("")
        self.console.print(f"Scrolled {self.total_lines} lines of logs.")


@dataclass
class Logs:
    @syncify
    @classmethod
    async def tail(
        cls,
        action_id: identifier_pb2.ActionIdentifier,
        attempt: int = 1,
        retry: int = 5,
    ) -> AsyncGenerator[payload_pb2.LogLine, None]:
        """
        Tail the logs for a given action ID and attempt.

        Args:
            action_id: The action ID to tail logs for.
            attempt: The attempt number (default is 0).
        """
        ensure_client()
        client = get_client()
        retries = 0
        while True:
            try:
                resp = client.dataproxy_service.tail_logs(
                    dataproxy_service_pb2.TailLogsRequest(
                        action_id=action_id,
                        attempt=attempt,
                    )
                )
                async for log_set in resp:
                    if log_set.logs:
                        for log in log_set.logs:
                            for line in log.lines:
                                yield line
                return
            except asyncio.CancelledError:
                return
            except KeyboardInterrupt:
                return
            except StopAsyncIteration:
                return
            except ConnectError as e:
                if e.code == Code.NOT_FOUND:
                    retries += 1
                    logger.debug(f"Log stream not found (attempt {retries}/{retry})")
                    if retries >= retry:
                        raise LogsNotYetAvailableError(
                            f"Log stream not available for action {action_id.name} in run {action_id.run.name}."
                        )
                    await asyncio.sleep(2)
                else:
                    raise

    @classmethod
    async def create_viewer(
        cls,
        action_id: identifier_pb2.ActionIdentifier,
        attempt: int = 1,
        max_lines: int = 30,
        show_ts: bool = False,
        raw: bool = False,
        filter_system: bool = False,
        panel: bool = False,
    ):
        """
        Create a log viewer for a given action ID and attempt.

        Args:
            action_id: Action ID to view logs for.
            attempt: Attempt number (default is 1).
            max_lines: Maximum number of lines to show if using the viewer. The logger will scroll
                and keep only max_lines in view.
            show_ts: Whether to show timestamps in the logs.
            raw: if True, return the raw log lines instead of a viewer.
            filter_system: Whether to filter log lines based on system logs.
            panel: Whether to use a panel for the log viewer. only applicable if raw is False.
        """
        if attempt < 1:
            raise ValueError("Attempt number must be greater than 0.")

        if ipython_check():
            if not ipywidgets_check():
                logger.warning("IPython widgets is not available, defaulting to console output.")
                raw = True

        if raw:
            console = Console()
            async for line in cls.tail.aio(action_id=action_id, attempt=attempt):
                line_text = _format_line(line, show_ts=show_ts, filter_system=filter_system)
                if line_text:
                    console.print(line_text, end="")
            return
        viewer = AsyncLogViewer(
            log_source=cls.tail.aio(action_id=action_id, attempt=attempt),
            max_lines=max_lines,
            show_ts=show_ts,
            name=f"{action_id.run.name}:{action_id.name} ({attempt})",
            filter_system=filter_system,
            panel=panel,
        )
        await viewer.run()


def _normalize_log_line(line: payload_pb2.LogLine, replica: str = "") -> payload_pb2.LogLine:
    """Prefix with the replica name and guarantee a trailing newline.

    The log renderers (AsyncLogViewer and the raw printer) rely on each message
    carrying its own newline, but app log lines arrive without one.
    """
    message = f"[{replica}] {line.message}" if replica else line.message
    if not message.endswith("\n"):
        message += "\n"
    if message == line.message:
        return line
    normalized = payload_pb2.LogLine()
    normalized.CopyFrom(line)
    normalized.message = message
    return normalized


def _iter_app_log_lines(resp: app_logs_payload_pb2.TailLogsResponse) -> Iterator[payload_pb2.LogLine]:
    """
    Normalize an app TailLogsResponse into a flat stream of LogLine protos.

    The response oneof carries either a replica list (informational only), plain
    log lines, or per-replica batches. The backend sends the same content both
    as structured LogLine protos and as raw strings (which lack timestamps and
    originators), so structured lines are preferred and the raw strings only
    used when no structured lines are present. Batched lines are prefixed with
    their replica name so interleaved multi-replica streams stay readable.
    """

    def _lines(structured, raw_lines, replica: str) -> Iterator[payload_pb2.LogLine]:
        if structured:
            for line in structured:
                yield _normalize_log_line(line, replica)
        else:
            for raw in raw_lines:
                yield _normalize_log_line(payload_pb2.LogLine(message=raw), replica)

    which = resp.WhichOneof("resp")
    if which == "replicas":
        logger.debug(f"App log stream replicas: {[r.name for r in resp.replicas.replicas]}")
        return
    if which == "log_lines":
        yield from _lines(resp.log_lines.structured_lines, resp.log_lines.lines, "")
        return
    if which == "batches":
        for log_lines in resp.batches.logs:
            yield from _lines(log_lines.structured_lines, log_lines.lines, log_lines.replica_id.name)


class _ReplayFilter:
    """Drops log lines already seen in a previous connection of the same tail.

    Every new TailLogs connection re-delivers the persisted backlog, so after a
    reconnect the stream starts with lines the viewer has already shown. Memory
    is constant: only the newest timestamp seen is kept, plus the messages
    sharing that exact timestamp (to tell replays apart from new lines within
    the same second). Lines without a timestamp cannot be distinguished from
    replays and are passed through.
    """

    def __init__(self):
        self._last_ts = (0, 0)
        self._seen_at_last_ts: set[str] = set()

    def is_new(self, line: payload_pb2.LogLine) -> bool:
        ts = (line.timestamp.seconds, line.timestamp.nanos)
        if ts == (0, 0):
            return True
        if ts < self._last_ts:
            return False
        if ts == self._last_ts:
            if line.message in self._seen_at_last_ts:
                return False
            self._seen_at_last_ts.add(line.message)
            return True
        self._last_ts = ts
        self._seen_at_last_ts = {line.message}
        return True


@dataclass
class AppLogs:
    @syncify
    @classmethod
    async def tail(
        cls,
        app_id: app_definition_pb2.Identifier,
        replica_name: str | None = None,
        retry: int = 5,
        follow: bool = True,
        idle_reconnects: int = 3,
    ) -> AsyncGenerator[payload_pb2.LogLine, None]:
        """
        Tail the logs for a given app, optionally scoped to a single replica.

        The server binds each stream to the replicas that exist at connect time
        and closes it whenever the replica set churns (a revision rollout, a
        scale-down). With ``follow=True`` the tail reconnects across those
        closes so e.g. a new revision's startup logs still appear, deduplicating
        the persisted backlog the server re-delivers on every connection. When
        ``idle_reconnects`` consecutive reconnects produce nothing new — the
        signature of an app with no live replicas (scaled to zero or
        deactivated) — the tail ends instead of re-reading the backlog forever.
        Re-running the tail always fetches the persisted logs again.

        Args:
            app_id: The app identifier to tail logs for.
            replica_name: Optional replica name to restrict the stream to.
            retry: Number of NOT_FOUND retries before giving up (the stream can
                lag a few seconds behind app activation).
            follow: If True, reconnect when the server closes or drops the
                stream. If False, return on the first stream close.
            idle_reconnects: With follow, the number of consecutive reconnects
                yielding no new lines tolerated before the tail ends.
        """
        ensure_client()
        client = get_client()
        if replica_name:
            request = app_logs_payload_pb2.TailLogsRequest(
                replica_id=replica_definition_pb2.ReplicaIdentifier(app_id=app_id, name=replica_name)
            )
        else:
            request = app_logs_payload_pb2.TailLogsRequest(app_id=app_id)
        retries = 0
        streamed_any = False
        replay_filter = _ReplayFilter()
        idle = 0
        while True:
            got_new = False
            try:
                resp = client.app_logs_service.tail_logs(request)
                async for log_set in resp:
                    for line in _iter_app_log_lines(log_set):
                        if not replay_filter.is_new(line):
                            continue
                        streamed_any = True
                        got_new = True
                        yield line
                if not follow:
                    return
            except asyncio.CancelledError:
                return
            except KeyboardInterrupt:
                return
            except StopAsyncIteration:
                return
            except ConnectError as e:
                if e.code == Code.NOT_FOUND:
                    if streamed_any:
                        # The stream existed and is now gone — the app was
                        # deleted or deactivated; treat as end of stream.
                        logger.debug(f"App log stream for {app_id.name} is gone, stopping tail.")
                        return
                    retries += 1
                    logger.debug(f"App log stream not found (attempt {retries}/{retry})")
                    if retries >= retry:
                        raise LogsNotYetAvailableError(f"Log stream not available for app {app_id.name}.")
                    await asyncio.sleep(2)
                    continue
                if not (follow and streamed_any):
                    raise
                # A mid-stream drop while following: reconnect like a close.
                logger.debug(f"App log stream for {app_id.name} disconnected ({e}), reconnecting...")
            if got_new:
                idle = 0
            else:
                idle += 1
                if idle >= idle_reconnects:
                    logger.debug(
                        f"App log stream for {app_id.name} produced no new logs after {idle} "
                        "reconnects (no live replicas?), stopping tail."
                    )
                    return
            await asyncio.sleep(2)

    @classmethod
    async def create_viewer(
        cls,
        app_id: app_definition_pb2.Identifier,
        max_lines: int = 30,
        show_ts: bool = False,
        raw: bool = False,
        filter_system: bool = False,
        panel: bool = False,
        replica_name: str | None = None,
    ):
        """
        Create a log viewer for a given app.

        Args:
            app_id: App identifier to view logs for.
            max_lines: Maximum number of lines to show if using the viewer. The logger will scroll
                and keep only max_lines in view.
            show_ts: Whether to show timestamps in the logs.
            raw: if True, print the raw log lines instead of using a viewer.
            filter_system: Whether to filter log lines based on system logs.
            panel: Whether to use a panel for the log viewer. only applicable if raw is False.
            replica_name: Optional replica name to restrict the stream to.
        """
        if ipython_check():
            if not ipywidgets_check():
                logger.warning("IPython widgets is not available, defaulting to console output.")
                raw = True

        if raw:
            console = Console()
            async for line in cls.tail.aio(app_id=app_id, replica_name=replica_name):
                line_text = _format_line(line, show_ts=show_ts, filter_system=filter_system)
                if line_text:
                    console.print(line_text, end="")
            return
        viewer = AsyncLogViewer(
            log_source=cls.tail.aio(app_id=app_id, replica_name=replica_name),
            max_lines=max_lines,
            show_ts=show_ts,
            name=f"app:{app_id.name}",
            filter_system=filter_system,
            panel=panel,
        )
        await viewer.run()
