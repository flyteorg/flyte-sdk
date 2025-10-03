import asyncio
from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator

import grpc
from flyteidl2.common import identifier_pb2
from flyteidl2.logs.dataplane import payload_pb2
from flyteidl2.workflow import run_logs_service_pb2
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
        log_text = Text()
        for line in self.lines:
            log_text.append(line)
        if self.panel:
            return Panel(log_text, title=self.name, border_style="yellow")
        return log_text

    async def run(self):
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
        :param action_id: The action ID to tail logs for.
        :param attempt: The attempt number (default is 0).
        """
        ensure_client()
        retries = 0
        while True:
            try:
                resp = get_client().logs_service.TailLogs(
                    run_logs_service_pb2.TailLogsRequest(action_id=action_id, attempt=attempt)
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
            except grpc.aio.AioRpcError as e:
                retries += 1
                if retries >= retry:
                    if e.code() == grpc.StatusCode.NOT_FOUND:
                        raise LogsNotYetAvailableError(
                            f"Log stream not available for action {action_id.name} in run {action_id.run.name}."
                        )
                else:
                    await asyncio.sleep(2)

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
        :param action_id: Action ID to view logs for.
        :param attempt: Attempt number (default is 1).
        :param max_lines: Maximum number of lines to show if using the viewer. The logger will scroll
           and keep only max_lines in view.
        :param show_ts: Whether to show timestamps in the logs.
        :param raw: if True, return the raw log lines instead of a viewer.
        :param filter_system: Whether to filter log lines based on system logs.
        :param panel: Whether to use a panel for the log viewer. only applicable if raw is False.
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
