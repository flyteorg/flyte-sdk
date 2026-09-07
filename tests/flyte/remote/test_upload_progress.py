"""Tests for upload progress reporting and the Content-Type carried on signed-URL PUTs."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from flyte.remote import _progress
from flyte.remote._data import _upload_with_retry, hash_file


class RecordingHandler:
    """Minimal `UploadProgressHandler` that records the calls it receives."""

    def __init__(self):
        self.events: list[tuple] = []
        self.totals: dict[str, int] = {}
        self.advanced: dict[str, int] = {}

    def start(self, key, *, name, phase, total):
        self.events.append(("start", key, name, phase, total))
        self.totals[key] = total
        self.advanced[key] = 0

    def advance(self, key, size):
        self.advanced[key] = self.advanced.get(key, 0) + size

    def finish(self, key, *, failed=False):
        self.events.append(("finish", key, failed))


@pytest.fixture
def payload(tmp_path: Path) -> Path:
    f = tmp_path / "model.bin"
    f.write_bytes(b"x" * (3 * _progress.CHUNK_SIZE + 17))
    return f


def test_no_handler_means_no_reporting(payload: Path):
    """The default path must not touch the (absent) handler."""
    assert _progress.current_handler() is None
    _progress.report_start("k", name="n", phase="hashing", total=1)
    _progress.report_advance("k", 1)
    _progress.report_finish("k")


def test_report_uploads_restores_previous_handler():
    first, second = RecordingHandler(), RecordingHandler()
    with _progress.report_uploads(first):
        assert _progress.current_handler() is first
        with _progress.report_uploads(second):
            assert _progress.current_handler() is second
        assert _progress.current_handler() is first
    assert _progress.current_handler() is None


def test_a_broken_handler_does_not_break_the_upload():
    class Boom:
        def start(self, *a, **kw):
            raise RuntimeError("nope")

        def advance(self, *a, **kw):
            raise RuntimeError("nope")

        def finish(self, *a, **kw):
            raise RuntimeError("nope")

    with _progress.report_uploads(Boom()):
        _progress.report_start("k", name="n", phase="hashing", total=1)
        _progress.report_advance("k", 1)
        _progress.report_finish("k")


def test_hash_file_reports_progress(payload: Path):
    handler = RecordingHandler()
    with _progress.report_uploads(handler):
        _, digest, size = hash_file(payload)

    key = _progress.hash_key(payload)
    assert handler.totals[key] == payload.stat().st_size
    assert handler.advanced[key] == size
    assert ("finish", key, False) in handler.events

    import hashlib

    assert digest == hashlib.md5(payload.read_bytes()).hexdigest()


def test_hash_file_result_is_memoized(payload: Path):
    hash_file(payload)  # prime the lru_cache
    handler = RecordingHandler()
    with _progress.report_uploads(handler):
        hash_file(payload)
    assert handler.events == []


@pytest.mark.asyncio
async def test_upload_reports_every_byte(payload: Path):
    """The stream wrapper must hand httpx the whole file and count all of it."""
    sent = bytearray()

    async def fake_put(url, headers=None, content=None):
        async for chunk in content:
            sent.extend(chunk)
        return httpx.Response(200)

    handler = RecordingHandler()
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = fake_put
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with _progress.report_uploads(handler):
            resp = await _upload_with_retry(payload, "https://signed.url/upload", {}, verify=True)

    assert resp.status_code == 200
    assert bytes(sent) == payload.read_bytes()
    key = _progress.upload_key(payload)
    assert handler.totals[key] == payload.stat().st_size
    assert handler.advanced[key] == payload.stat().st_size
    assert ("finish", key, False) in handler.events


@pytest.mark.asyncio
async def test_upload_without_handler_streams_the_file_object(payload: Path):
    """Without a handler the body is the aiofiles handle, exactly as before."""
    seen = {}

    async def fake_put(url, headers=None, content=None):
        seen["content"] = content
        return httpx.Response(200)

    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = fake_put
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        await _upload_with_retry(payload, "https://signed.url/upload", {}, verify=True)

    assert hasattr(seen["content"], "read")


@pytest.mark.asyncio
async def test_retry_restarts_the_bar_instead_of_stacking_one(payload: Path):
    """Each attempt re-reads the file, so each attempt must restart the same key."""
    attempts = {"n": 0}

    async def fake_put(url, headers=None, content=None):
        attempts["n"] += 1
        async for _ in content:
            pass
        return httpx.Response(200 if attempts["n"] > 1 else 500)

    handler = RecordingHandler()
    with patch("flyte.remote._data.httpx.AsyncClient") as mock_cls:
        client = AsyncMock()
        client.put.side_effect = fake_put
        ctx = AsyncMock()
        ctx.__aenter__.return_value = client
        ctx.__aexit__.return_value = False
        mock_cls.return_value = ctx

        with _progress.report_uploads(handler):
            await _upload_with_retry(
                payload, "https://signed.url/upload", {}, verify=True, min_backoff_sec=0.0, max_backoff_sec=0.0
            )

    key = _progress.upload_key(payload)
    starts = [e for e in handler.events if e[0] == "start" and e[1] == key]
    assert len(starts) == 2, "every attempt restarts the same bar"


async def _upload_capturing_headers(tmp_path: Path, *, content_type, signed_headers):
    """Run _upload_single_file with the control plane stubbed, returning the PUT headers."""
    from unittest.mock import MagicMock

    from flyte.remote._data import _upload_single_file

    fp = tmp_path / "card.html"
    fp.write_text("<h1>card</h1>")

    resp = MagicMock()
    resp.signed_url = "https://signed.url/upload"
    resp.native_url = "s3://bucket/card.html"
    resp.headers = signed_headers

    client = MagicMock()
    client.dataproxy_service.create_upload_location = AsyncMock(return_value=resp)

    cfg = MagicMock(project="p", domain="d", org="o")
    captured = {}

    async def fake_retry(*, fp, signed_url, extra_headers, **kwargs):
        captured.update(extra_headers)
        return httpx.Response(200)

    with (
        patch("flyte._initialize._get_init_config", return_value=cfg),
        patch("flyte.remote._data.get_client", return_value=client),
        patch("flyte.remote._data._upload_with_retry", side_effect=fake_retry),
    ):
        await _upload_single_file(cfg, fp, content_type=content_type)
    return captured


@pytest.mark.asyncio
async def test_content_type_is_sent_on_the_put(tmp_path: Path):
    headers = await _upload_capturing_headers(tmp_path, content_type="text/html", signed_headers={})
    assert headers["Content-Type"] == "text/html"


@pytest.mark.asyncio
async def test_content_type_never_overrides_the_signed_one(tmp_path: Path):
    """The signing service's headers are part of the signature; ours must not win."""
    headers = await _upload_capturing_headers(
        tmp_path, content_type="text/html", signed_headers={"content-type": "application/octet-stream"}
    )
    assert headers["content-type"] == "application/octet-stream"
    assert "Content-Type" not in headers


@pytest.mark.asyncio
async def test_no_content_type_by_default(tmp_path: Path):
    headers = await _upload_capturing_headers(tmp_path, content_type=None, signed_headers={})
    assert not any(h.lower() == "content-type" for h in headers)


class _PutRecorder(BaseHTTPRequestHandler):
    received: ClassVar[dict] = {}

    def do_PUT(self):
        length = int(self.headers.get("Content-Length", 0))
        _PutRecorder.received = {
            "body": self.rfile.read(length),
            "headers": {k.lower(): v for k, v in self.headers.items()},
        }
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.mark.asyncio
async def test_progress_stream_sends_a_well_formed_request(payload: Path):
    """
    Round-trip against a real server: a streamed body with an explicit Content-Length
    must not flip httpx into chunked encoding, and the Content-Type must survive.
    """
    server = HTTPServer(("127.0.0.1", 0), _PutRecorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/upload"
        headers = {"Content-Length": str(payload.stat().st_size), "Content-Type": "text/html"}
        with _progress.report_uploads(RecordingHandler()):
            resp = await _upload_with_retry(payload, url, headers, verify=True)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert resp.status_code == 200
    assert _PutRecorder.received["body"] == payload.read_bytes()
    assert _PutRecorder.received["headers"]["content-type"] == "text/html"
    assert "transfer-encoding" not in _PutRecorder.received["headers"]
