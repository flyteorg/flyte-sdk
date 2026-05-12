from __future__ import annotations

import contextvars
import re
import sys
from contextlib import contextmanager
from typing import Generator, Literal

OutputMode = Literal["rich", "plain"]

# Default to None = auto-detect on first use
_output_mode_var: contextvars.ContextVar[OutputMode | None] = contextvars.ContextVar("status_output_mode", default=None)
_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar("status_depth", default=0)


def set_output_mode(mode: OutputMode) -> None:
    _output_mode_var.set(mode)


def get_output_mode() -> OutputMode:
    """Return the current output mode, auto-detecting on first call."""
    mode = _output_mode_var.get()
    if mode is not None:
        return mode
    mode = _auto_detect_mode()
    _output_mode_var.set(mode)
    return mode


def _auto_detect_mode() -> OutputMode:
    """Detect whether we should use rich or plain output."""
    from flyte._tools import ipython_check

    if ipython_check():
        return "rich"
    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        return "rich"
    return "plain"


_URL_RE = re.compile(r"(https?://\S+)")


def _linkify(message: str) -> str:
    """Wrap bare URLs in Rich [link] markup so they are clickable in terminals that support OSC 8."""
    return _URL_RE.sub(r"[link=\1]\1[/link]", message)


class StatusProxy:
    """Always-visible status output channel, independent of log levels.
    Writes to stderr to keep stdout clean for structured output."""

    def step(self, message: str) -> None:
        """Progress step: 'Building image...'"""
        self._emit("step", message)

    def success(self, message: str) -> None:
        """Success: 'Image built: ...'"""
        self._emit("success", message)

    def info(self, message: str) -> None:
        """Informational: 'Skipping build, image already exists'"""
        self._emit("info", message)

    def warn(self, message: str) -> None:
        """Warning that doesn't use logging system."""
        self._emit("warn", message)

    @contextmanager
    def group(self, message: str) -> Generator[None, None, None]:
        """Nest subsequent status messages under a parent step.

        Usage::

            with status.group("Building images..."):
                # messages here are indented one level deeper
                status.step("Building image for env1")
                status.success("Built image for env1")
        """
        self.step(message)
        token = _depth_var.set(_depth_var.get() + 1)
        try:
            yield
        finally:
            _depth_var.reset(token)

    def _emit(self, level: str, message: str) -> None:
        if get_output_mode() == "rich":
            self._emit_rich(level, message)
        else:
            self._emit_plain(level, message)

    def _emit_rich(self, level: str, message: str) -> None:
        from rich.console import Console

        console = Console(stderr=True)
        icons = {
            "step": "[bold blue]>[/bold blue]",
            "success": "[bold green]✓[/bold green]",
            "info": "[dim]i[/dim]",
            "warn": "[bold yellow]⚠[/bold yellow]",
        }
        depth = _depth_var.get()
        indent = "  " * (depth + 1)
        icon = icons.get(level, "")
        message = _linkify(message)
        console.print(f"{indent}{icon} {message}", highlight=False)

    def _emit_plain(self, level: str, message: str) -> None:
        prefixes = {"step": ">>", "success": "OK", "info": "--", "warn": "!!"}
        prefix = prefixes.get(level, "--")
        depth = _depth_var.get()
        indent = "  " * depth
        print(f"[flyte] {indent}{prefix} {message}", file=sys.stderr)


status = StatusProxy()
