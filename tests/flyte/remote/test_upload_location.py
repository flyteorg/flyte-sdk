"""Tests for flyte.remote._data.upload_storage: the storage an upload would land in, without uploading."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from flyteidl2.dataproxy import dataproxy_service_pb2

import flyte.remote._data as data
from flyte._upload_queue import upload_queue


@pytest.mark.asyncio
async def test_upload_storage_asks_like_an_upload(monkeypatch):
    create = AsyncMock(
        return_value=dataproxy_service_pb2.CreateUploadLocationResponse(
            native_url="s3://bucket/p/d/X/f.tar.gz", signed_url="https://bucket.s3.amazonaws.com/p/d/X/f.tar.gz?sig"
        )
    )
    cfg = SimpleNamespace(org="o", project="p", domain="d")
    monkeypatch.setattr(data, "ensure_client", lambda: None)
    monkeypatch.setattr(data, "get_init_config", lambda: cfg)
    monkeypatch.setattr(
        data, "get_client", lambda: SimpleNamespace(dataproxy_service=SimpleNamespace(create_upload_location=create))
    )
    md5 = "0123456789abcdef0123456789abcdef"

    with upload_queue("q1"):
        assert await data.upload_storage(md5, "f.tar.gz") == "s3://bucket@bucket.s3.amazonaws.com"
        expected = data._upload_location_request(cfg, bytes.fromhex(md5), "f.tar.gz")

    sent = create.await_args[0][0]
    assert sent.SerializeToString(deterministic=True) == expected.SerializeToString(deterministic=True)
    assert sent.queue == "q1" and sent.content_md5 == bytes.fromhex(md5)
