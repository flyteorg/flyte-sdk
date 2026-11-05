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

The wrappers are put in place when the first invocation starts and taken out again when the last
one in progress ends, so a process with no task running has its original streams back.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional, TextIO, Tuple

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
            pending = self._pending.pop(key, "")
            lines: List[str] = []
            if pending and _already_attributed(s, _prefix(key)):
                # A log record arriving while a print(..., end="") line is still open starts a line of
                # its own, so end the open line first instead of joining the record onto it.
                lines.append(pending)
                pending = ""
            *complete, rest = (pending + s).split("\n")
            lines.extend(complete)
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

    def flush_ended(self, open_keys: Iterable[_Key]) -> None:
        """Write out the partial lines of every context except those in `open_keys`.

        A context can still write after its invocation ended, from a thread or an asyncio task it
        started, so this covers every context whose invocation is over and not only the one ending now.
        """
        keep = set(open_keys)
        with self._lock:
            ended = [k for k in self._pending if k not in keep]
            out = "".join(self._format(k, self._pending.pop(k)) + "\n" for k in ended)
            if out:
                self._stream.write(out)
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


_state_lock = threading.Lock()
# Invocations in progress, counted per run and action.
_active: dict[_Key, int] = {}
# The wrappers this module put in place, with the `sys` attribute each one replaced.
_installed: List[Tuple[str, AttributedStream]] = []


def _start(key: _Key) -> None:
    with _state_lock:
        _active[key] = _active.get(key, 0) + 1
        for name in ("stdout", "stderr"):
            current = getattr(sys, name)
            if not isinstance(current, AttributedStream):
                wrapper = AttributedStream(current)
                setattr(sys, name, wrapper)
                _installed.append((name, wrapper))


def _end(key: _Key) -> None:
    with _state_lock:
        remaining = _active.pop(key, 1) - 1
        if remaining:
            _active[key] = remaining
        # Flush the wrappers this module installed even if something has replaced `sys.stdout` or
        # `sys.stderr` since, so a trailing partial line is never left behind in one of them.
        streams = [wrapper for _, wrapper in _installed]
        for current in (sys.stdout, sys.stderr):
            if isinstance(current, AttributedStream) and not any(current is s for s in streams):
                streams.append(current)
        for stream in streams:
            stream.flush_ended(_active)
        if not _active:
            # No invocation is left to attribute, so put back the streams that were there before.
            # A stream replaced by something else since is left as it is.
            for name, wrapper in reversed(_installed):
                if getattr(sys, name) is wrapper:
                    setattr(sys, name, wrapper.wrapped)
            _installed.clear()


@contextmanager
def attributed_output(run_name: str, action_name: str) -> Iterator[None]:
    """Attribute raw output for one task invocation, and write out its last partial line at the end.

    Concurrent invocations share one wrapper per stream, which stays in place until the last of them ends.
    """
    key = (run_name, action_name)
    _start(key)
    try:
        yield
    finally:
        _end(key)
