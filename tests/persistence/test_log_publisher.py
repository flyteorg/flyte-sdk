import asyncio
import logging
import sys

import pytest

from flyte._persistence._log_publisher import (
    LogCapture,
    _LogBuffer,
    publish_report,
    render_html,
)


class TestLogBuffer:
    def test_accumulates_in_order(self):
        b = _LogBuffer()
        b.write("a")
        b.write("b")
        assert b.render() == "ab"

    def test_empty_writes_ignored(self):
        b = _LogBuffer()
        b.write("")
        assert b.render() == ""

    def test_retains_tail_when_over_budget(self):
        b = _LogBuffer(max_bytes=10)
        b.write("x" * 8)
        b.write("y" * 8)
        out = b.render()
        # The most recent output is what matters for diagnosing a failure.
        assert "yyyyyyyy" in out
        assert "truncated" in out

    def test_single_oversized_chunk_is_kept(self):
        # Never drop the only chunk, or a single huge line would render as nothing.
        b = _LogBuffer(max_bytes=4)
        b.write("z" * 100)
        assert "z" * 100 in b.render()


class TestLogCapture:
    def test_captures_stdout_and_stderr(self):
        with LogCapture() as cap:
            print("to-stdout")
            print("to-stderr", file=sys.stderr)
        out = cap.render()
        assert "to-stdout" in out
        assert "to-stderr" in out

    def test_writes_through_to_original_stream(self, capsys):
        with LogCapture():
            print("still-visible")
        # The user must still see their output locally.
        assert "still-visible" in capsys.readouterr().out

    def test_restores_streams_on_exit(self):
        before_out, before_err = sys.stdout, sys.stderr
        with LogCapture():
            assert sys.stdout is not before_out
        assert sys.stdout is before_out
        assert sys.stderr is before_err

    def test_restores_streams_on_exception(self):
        before = sys.stdout
        try:
            with LogCapture():
                raise ValueError("boom")
        except ValueError:
            pass
        assert sys.stdout is before

    def test_captures_logging_calls(self):
        # logging.StreamHandler binds its stream at construction, so a tee alone would miss
        # handlers created before the capture started.
        with LogCapture() as cap:
            logging.getLogger("flyte.user").warning("via-logger")
        assert "via-logger" in cap.render()

    def test_removes_handler_on_exit(self):
        lg = logging.getLogger("flyte.user")
        before = len(lg.handlers)
        with LogCapture():
            assert len(lg.handlers) == before + 1
        assert len(lg.handlers) == before

    def test_isatty_does_not_raise(self):
        with LogCapture():
            # Rich/textual probe this; it must never blow up.
            assert isinstance(sys.stdout.isatty(), bool)


class TestRenderHtml:
    def test_escapes_content(self):
        out = render_html("<script>alert(1)</script>")
        assert "&lt;script&gt;" in out
        assert "<script>alert(1)</script>" not in out

    def test_wraps_in_pre(self):
        assert "<pre>" in render_html("hello")


class TestPublishLogs:
    """Logs go through the data proxy's signed URL, not a direct bucket write.

    That is what lets publishing work without the developer holding write credentials for the
    backend's bucket -- the same reason the code bundle uses that path.
    """

    @staticmethod
    def _patch(monkeypatch, fn):
        import flyte.remote._data as data_mod

        async def _call(cfg, fp, **kw):
            return await fn(fp, **kw)

        monkeypatch.setattr(data_mod, "_upload_single_file", _call)
        monkeypatch.setattr("flyte._initialize.get_init_config", object)

    @pytest.mark.asyncio
    async def test_uploads_via_signed_url_and_returns_uri(self, monkeypatch):
        captured = {}

        async def fake_upload(local, **kwargs):
            captured["name"] = local.name
            captured["body"] = local.read_text()
            captured["content_type"] = kwargs.get("content_type")
            return "md5", "s3://backend-chosen/uploads/local_run_report.html"

        self._patch(monkeypatch, fake_upload)
        uri = await publish_report(text="hello")

        assert uri == "s3://backend-chosen/uploads/local_run_report.html"
        assert captured["name"] == "local_run_report.html"
        # Without this the object is served as a generic blob and the browser downloads it
        # instead of rendering it in the report view.
        assert captured["content_type"] == "text/html"
        assert "hello" in captured["body"]

    @pytest.mark.asyncio
    async def test_upload_failure_returns_none(self, monkeypatch):
        async def boom(local, **kwargs):
            raise OSError("upload rejected")

        self._patch(monkeypatch, boom)
        # Losing logs must not fail an otherwise-successful run.
        assert await publish_report(text="hi") is None

    @pytest.mark.asyncio
    async def test_slow_upload_is_bounded(self, monkeypatch):
        async def slow(local, **kwargs):
            await asyncio.sleep(60)

        self._patch(monkeypatch, slow)
        # An unreachable backend must not add minutes to the end of every published run.
        assert await publish_report(text="hi", timeout=0.2) is None
