"""Capture a local run's console output and publish it as the run's report.

Two things worth knowing:

1. **Capture is run-level, not per-action.** A local run executes every action in one process
   against one shared ``sys.stdout``; concurrent actions interleave, so a line cannot be
   attributed to a specific action without cooperation from user code.

2. **It is published as a report, not into the log pane.** The log pane reads live Kubernetes
   pod logs, and there is no pod. Reports are the one artifact the console fetches through a
   signed download URL (``ARTIFACT_TYPE_REPORT`` resolves from ``attempt.outputs.report_uri``),
   so this needs no credentials and no backend change.

Both a stdout/stderr tee (for ``print``) and a logging handler (for ``logger`` calls) are
installed: ``logging.StreamHandler`` binds its stream at construction, so handlers created
before the tee would otherwise write straight past it.

The interpreter and package manifest are *not* repeated here -- they live on each task
(``container.env`` and ``task_template.custom``), which is where a remote run describes what it
ran in.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, TextIO

from flyte._logging import logger

_REPORT_FILE_NAME = "local_run_report.html"

# Cap the retained output so a chatty run cannot exhaust memory; the tail is what matters when
# diagnosing a failure.
_MAX_CAPTURE_BYTES = 8 * 1024 * 1024
_TRUNCATION_NOTICE = "... [earlier output truncated] ...\n"

# An unwritable destination does not fail fast, so without this every run pays that latency.
_UPLOAD_TIMEOUT_SEC = 15.0


class _LogBuffer:
    """Thread-safe, byte-bounded buffer retaining the most recent output."""

    def __init__(self, max_bytes: int = _MAX_CAPTURE_BYTES) -> None:
        self._chunks: list[str] = []
        self._size = 0
        self._max_bytes = max_bytes
        self._truncated = False
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._chunks.append(text)
            self._size += len(text)
            while self._size > self._max_bytes and len(self._chunks) > 1:
                self._size -= len(self._chunks.pop(0))
                self._truncated = True

    def render(self) -> str:
        with self._lock:
            body = "".join(self._chunks)
            return (_TRUNCATION_NOTICE + body) if self._truncated else body


class _Tee:
    """Write-through stream wrapper that also records into a buffer."""

    def __init__(self, stream: TextIO, buffer: _LogBuffer) -> None:
        self._stream = stream
        self._buffer = buffer

    def write(self, text: str) -> int:
        self._buffer.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _BufferHandler(logging.Handler):
    """Routes log records into the buffer regardless of when handlers bound their streams."""

    def __init__(self, buffer: _LogBuffer) -> None:
        super().__init__()
        self._buffer = buffer
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buffer.write(self.format(record) + "\n")
        except Exception:  # pragma: no cover - logging must never break the run
            pass


class LogCapture:
    """Captures console output for the duration of a local run.

    Usage::

        with LogCapture() as capture:
            ...
        text = capture.render()
    """

    def __init__(self) -> None:
        self._buffer = _LogBuffer()
        self._saved: tuple[TextIO, TextIO] | None = None
        self._handler: _BufferHandler | None = None
        self._loggers: list[logging.Logger] = []

    def __enter__(self) -> "LogCapture":
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._buffer)  # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, self._buffer)  # type: ignore[assignment]

        self._handler = _BufferHandler(self._buffer)
        # The user-facing logger and the root logger together cover task code, whether it uses
        # flyte's logger or plain `logging`.
        for name in ("flyte.user", ""):
            lg = logging.getLogger(name)
            lg.addHandler(self._handler)
            self._loggers.append(lg)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        handler = self._handler
        if handler is not None:
            for lg in self._loggers:
                try:
                    lg.removeHandler(handler)
                except Exception:
                    pass
        self._loggers.clear()
        self._handler = None
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved

    def render(self) -> str:
        return self._buffer.render()


def render_html(text: str) -> str:
    """Wrap captured output in a minimal HTML document for the report viewer."""
    import html

    return (
        "<!doctype html><meta charset='utf-8'><title>Local run</title>"
        "<style>body{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:1rem}"
        "pre{white-space:pre-wrap;word-break:break-word;margin:0}</style>"
        "<pre>" + html.escape(text) + "</pre>"
    )


async def publish_report(*, text: str, timeout: float = _UPLOAD_TIMEOUT_SEC) -> str | None:
    """Upload captured output as an HTML report and return its URI, or None on failure.

    Goes through the data proxy's signed URL -- the same path the code bundle uses -- so no local
    cloud credentials are needed. Never raises and never hangs: losing the report must not fail,
    or noticeably delay, an otherwise successful run.
    """
    if not text:
        return None
    import asyncio
    import tempfile
    from pathlib import Path

    try:
        # The plain coroutine, not the syncified `flyte.remote.upload_file`: that wrapper's
        # `.aio()` hands work to syncify's shared loop, which under `flyte run` is the loop
        # running the user's task, so the upload would queue behind it and time out.
        from flyte._initialize import get_init_config
        from flyte.remote._data import _upload_single_file

        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / _REPORT_FILE_NAME
            local.write_text(render_html(text), encoding="utf-8")
            _, uri = await asyncio.wait_for(
                # Without the MIME type the console downloads the report instead of showing it.
                _upload_single_file(get_init_config(), local, content_type="text/html"),
                timeout=timeout,
            )
    except asyncio.TimeoutError:
        logger.warning(f"Timed out after {timeout:.0f}s publishing the local run report")
        return None
    except Exception as e:
        logger.warning(f"Failed to publish the local run report: {type(e).__name__}: {e}")
        return None
    logger.debug(f"Published local run report to {uri}")
    return uri
