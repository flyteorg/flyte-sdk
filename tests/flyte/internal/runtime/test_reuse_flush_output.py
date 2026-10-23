"""Tests that a reusable container's worker flushes task output after every invocation.

The a0 entrypoint flushes stdout and stderr only right before its process exits. A reusable
container's worker process never gets there, and with its output going to a pipe Python buffers
writes in blocks, so a task's print() output could stay in the buffer and never reach the logs.
"""

import asyncio
import io
import logging
import sys
from types import SimpleNamespace

import pytest

from flyte._internal.runtime import rusty


class _PipeLikeStream(io.TextIOWrapper):
    """A text stream buffered like stdout on a pipe: nothing reaches `raw` until it is flushed."""

    def __init__(self):
        self.raw_bytes = io.BytesIO()
        super().__init__(io.BufferedWriter(self.raw_bytes, buffer_size=1 << 20), encoding="utf-8")

    def written(self) -> str:
        return self.raw_bytes.getvalue().decode("utf-8")


class _UnflushableStream(io.StringIO):
    def flush(self):
        raise ValueError("I/O operation on closed file.")


@pytest.fixture(autouse=True)
def flyte_log():
    """
    Route the flyte logger to a private buffer for the duration of each test.

    rusty logs a line when an invocation fails. A handler that another test left on the flyte logger
    (the rich handler, for one, writes to whatever sys.stdout is when it emits) would put that line into
    the stream under test, so each test swaps in its own handler and puts the original ones back after.
    """
    flyte_logger = logging.getLogger("flyte")
    saved = flyte_logger.handlers[:]
    buffer = io.StringIO()
    flyte_logger.handlers = [logging.StreamHandler(buffer)]
    try:
        yield buffer
    finally:
        flyte_logger.handlers = saved


def _install_pipe_streams(monkeypatch):
    # Installed from the test body, not a fixture: pytest's own capture swaps sys.stdout back in
    # between fixture setup and the test call.
    out, err = _PipeLikeStream(), _PipeLikeStream()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    return out, err


def _run(monkeypatch, body):
    async def fake_extract_download_run_upload(task, **kwargs):
        body()

    monkeypatch.setattr(rusty, "extract_download_run_upload", fake_extract_download_run_upload)
    return asyncio.run(
        rusty.run_task(
            task=SimpleNamespace(name="t"),
            controller=None,
            org="org",
            project="p",
            domain="d",
            run_name="r",
            name="a1",
            raw_data_path="s3://bucket/raw",
            output_path="s3://bucket/a1/0",
            run_base_dir="s3://bucket",
            version="v1",
        )
    )


def test_stream_is_buffered_until_flushed(monkeypatch):
    # Guards the premise of the tests below: without a flush the output really is held back.
    out, _ = _install_pipe_streams(monkeypatch)
    print("held")
    assert out.written() == ""


def test_print_output_is_flushed_after_successful_invocation(monkeypatch):
    out, err = _install_pipe_streams(monkeypatch)

    def body():
        print("hello from the task")
        print("warning from the task", file=sys.stderr)

    _run(monkeypatch, body)

    assert out.written() == "hello from the task\n"
    assert err.written() == "warning from the task\n"


def test_print_output_is_flushed_when_invocation_fails(monkeypatch, flyte_log):
    out, _ = _install_pipe_streams(monkeypatch)

    def body():
        print("output before the failure")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run(monkeypatch, body)

    assert out.written() == "output before the failure\n"
    # The failure itself goes to the flyte logger, not into the task's stdout.
    assert "Task failed: boom" in flyte_log.getvalue()


def test_each_invocation_flushes_its_own_output(monkeypatch):
    # The worker process stays up between invocations, so output must not wait for a later one.
    out, _ = _install_pipe_streams(monkeypatch)

    _run(monkeypatch, lambda: print("first"))
    assert out.written() == "first\n"

    _run(monkeypatch, lambda: print("second"))
    assert out.written() == "first\nsecond\n"


def test_unflushable_stream_does_not_fail_the_invocation(monkeypatch):
    # User code may close or swap out sys.stdout; that must not turn a finished task into a failure.
    monkeypatch.setattr(sys, "stdout", _UnflushableStream())
    monkeypatch.setattr(sys, "stderr", None)

    _run(monkeypatch, lambda: print("ignored"))
