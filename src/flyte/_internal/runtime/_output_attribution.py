"""Attribute raw stdout and stderr lines to their run and action in reusable containers.

A reusable container runs many task invocations concurrently in one Python process, all writing
to the same stdout and stderr. Log records get a `[run][action]` prefix from the logging
formatter, but `print()` and other raw writes do not, so a log view filtered to one action
drops them. This module wraps `sys.stdout` and `sys.stderr` so that each complete line written
inside a task context starts with `[run_name][action_name]`.

The task context comes from a context variable, so concurrent asyncio tasks, and threads that
copy the context (as sync tasks do), each get their own prefix. A thread started with a plain
`threading.Thread` does not copy the context, and its lines pass through unchanged, as do lines
written outside any task.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, TextIO, Tuple

_Key = Tuple[str, str]

_MAX_PENDING = 8192


def _current_key() -> Optional[_Key]:
    try:
        from flyte._context import internal_ctx

        tctx = internal_ctx().data.task_context
    except Exception:
        return None
    if tctx is None:
        return None
    action = tctx.action
    if not action.run_name or not action.name:
        return None
    return action.run_name, action.name


def _prefix(key: _Key) -> str:
    return f"[{key[0]}][{key[1]}]"


def _already_attributed(line: str, prefix: str) -> bool:
    # Lines from the logging formatter already carry the prefix, and the JSON formatter puts the
    # run and action into fields of its own.
    if line.startswith(prefix):
        return True
    stripped = line.lstrip()
    return stripped.startswith("{") and '"action_name"' in stripped


class AttributedStream:
    """A text stream that prefixes each line written inside a task context with `[run][action]`.

    A partial line is held per run and action until its newline arrives, so concurrent writers
    never split each other's lines; `flush()` does not write it out early for the same reason.
    A partial line longer than `_MAX_PENDING` characters (a progress bar redrawn with carriage
    returns, say) is written out as a line of its own. Everything else is delegated to the wrapped
    stream.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.RLock()
        # Text written by a context that has not yet ended in a newline.
        self._pending: dict[_Key, str] = {}

    @property
    def wrapped(self) -> TextIO:
        return self._stream

    def write(self, s: str) -> int:
        key = _current_key()
        if key is None or not s:
            return self._stream.write(s)
        with self._lock:
            text = self._pending.pop(key, "") + s
            *lines, rest = text.split("\n")
            if len(rest) > _MAX_PENDING:
                lines.append(rest)
                rest = ""
            if lines:
                self._stream.write("".join(self._format(key, line) + "\n" for line in lines))
            if rest:
                self._pending[key] = rest
        return len(s)

    def writelines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def flush_context(self, key: _Key) -> None:
        """Write out what a context left without a trailing newline, for when it ends."""
        with self._lock:
            rest = self._pending.pop(key, "")
            if rest:
                self._stream.write(self._format(key, rest) + "\n")
        self._stream.flush()

    def _format(self, key: _Key, line: str) -> str:
        prefix = _prefix(key)
        if not line or _already_attributed(line, prefix):
            return line
        return f"{prefix} {line}"

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return self._stream.encoding

    @property
    def errors(self) -> Optional[str]:
        return self._stream.errors

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def install() -> None:
    """Wrap `sys.stdout` and `sys.stderr`. Safe to call more than once."""
    if not isinstance(sys.stdout, AttributedStream):
        sys.stdout = AttributedStream(sys.stdout)  # type: ignore[assignment]
    if not isinstance(sys.stderr, AttributedStream):
        sys.stderr = AttributedStream(sys.stderr)  # type: ignore[assignment]


@contextmanager
def attributed_output(run_name: str, action_name: str) -> Iterator[None]:
    """Attribute raw output for one task invocation, and write out its last partial line at the end."""
    install()
    try:
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, AttributedStream):
                stream.flush_context((run_name, action_name))
