"""Tests for flyte.remote._data.upload_location: the destination an upload would get, without uploading."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from flyteidl2.dataproxy import dataproxy_service_pb2

import flyte.remote._data as data
from flyte._upload_queue import upload_queue


@pytest.mark.asyncio
async def test_upload_location_asks_like_an_upload(monkeypatch):
    create = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(native_url="s3://bucket/p/d/X/f.tar.gz")
    )
    monkeypatch.setattr(data, "ensure_client", lambda: None)
    monkeypatch.setattr(data, "get_init_config", lambda: SimpleNamespace(org="o", project="p", domain="d"))
    monkeypatch.setattr(
        data, "get_client", lambda: SimpleNamespace(dataproxy_service=SimpleNamespace(create_upload_location=create))
    )

    with upload_queue("q1"):
        assert await data.upload_location("00ff", "f.tar.gz") == "s3://bucket/p/d/X/f.tar.gz"

    req = create.await_args[0][0]
    assert (req.org, req.project, req.domain, req.filename) == ("o", "p", "d", "f.tar.gz")
    assert req.content_md5 == bytes.fromhex("00ff")
    assert req.queue == "q1"
