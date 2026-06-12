from __future__ import annotations

import pytest
from flyteidl2.task import common_pb2 as run_definition_pb2

from flyte._internal.runtime import io


def _outputs() -> io.Outputs:
    return io.Outputs(proto_outputs=run_definition_pb2.Outputs())


@pytest.mark.asyncio
async def test_upload_outputs_deletes_stale_error_on_restarted_attempt(monkeypatch):
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "1")
    output_path = "s3://bucket/outputs"
    stale_error_uri = io.error_path(output_path)
    outputs_uri = io.outputs_path(output_path)

    calls: list[tuple[str, str]] = []

    async def fake_exists(path: str, **_kwargs) -> bool:
        return path == stale_error_uri

    class FakeFS:
        def rm_file(self, path: str):
            calls.append(("delete", path))

    async def fake_put_stream(data_iterable, to_path):
        calls.append(("put", to_path))

    monkeypatch.setattr(io.storage, "exists", fake_exists)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: FakeFS())
    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)

    await io.upload_outputs(_outputs(), output_path)

    assert calls == [("delete", stale_error_uri), ("put", outputs_uri)]


@pytest.mark.asyncio
async def test_upload_outputs_attempt_zero_skips_stale_error_delete(monkeypatch):
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "0")
    output_path = "s3://bucket/outputs"
    call_count = {"exists": 0, "fs": 0, "put": 0}

    async def fake_exists(path: str, **_kwargs) -> bool:
        call_count["exists"] += 1
        return False

    def fake_get_underlying_filesystem(**_kwargs):
        call_count["fs"] += 1
        return object()

    async def fake_put_stream(data_iterable, to_path):
        call_count["put"] += 1

    monkeypatch.setattr(io.storage, "exists", fake_exists)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", fake_get_underlying_filesystem)
    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)

    await io.upload_outputs(_outputs(), output_path)

    assert call_count["exists"] == 0
    assert call_count["fs"] == 0
    assert call_count["put"] == 1


@pytest.mark.asyncio
async def test_upload_outputs_delete_failure_is_soft_failed(monkeypatch):
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "2")
    output_path = "s3://bucket/outputs"
    stale_error_uri = io.error_path(output_path)
    outputs_uri = io.outputs_path(output_path)
    warnings: list[str] = []
    puts: list[str] = []

    async def fake_exists(path: str, **_kwargs) -> bool:
        return path == stale_error_uri

    class FailingDeleteFS:
        def rm_file(self, _path: str):
            raise RuntimeError("delete failed")

    async def fake_put_stream(data_iterable, to_path):
        puts.append(to_path)

    monkeypatch.setattr(io.storage, "exists", fake_exists)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: FailingDeleteFS())
    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)
    monkeypatch.setattr(io.logger, "warning", warnings.append)

    await io.upload_outputs(_outputs(), output_path)

    assert puts == [outputs_uri]
    assert len(warnings) == 1
    assert stale_error_uri in warnings[0]
