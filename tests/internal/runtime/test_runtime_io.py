import pytest
from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2 as run_definition_pb2

import flyte.errors
import flyte.models
import flyte.types as types
from flyte._internal.runtime import io


async def create_inputs(size):
    return io.Inputs(
        run_definition_pb2.Inputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="a",
                    value=await types.TypeEngine.to_literal(
                        "x" * size, python_type=str, expected=types.TypeEngine.to_literal_type(str)
                    ),
                )
            ],
        )
    )


async def create_outputs(size):
    return io.Outputs(
        proto_outputs=run_definition_pb2.Outputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="a",
                    value=await types.TypeEngine.to_literal(
                        "x" * size, python_type=str, expected=types.TypeEngine.to_literal_type(str)
                    ),
                )
            ],
        )
    )


async def create_blob_inputs(uri: str, count: int = 5) -> io.Inputs:
    return io.Inputs(
        run_definition_pb2.Inputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name=f"a{x}",
                    value=literals_pb2.Literal(
                        scalar=literals_pb2.Scalar(
                            blob=literals_pb2.Blob(
                                uri=uri,
                            ),
                        ),
                    ),
                )
                for x in range(count)
            ],
        )
    )


@pytest.mark.asyncio
async def test_upload_inputs(monkeypatch):
    called = {}

    async def fake_put_stream(data_iterable, to_path):
        called["data"] = data_iterable
        called["path"] = to_path

    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)
    inputs = await create_inputs(10)
    await io.upload_inputs(inputs, "some/path")
    assert called["data"] == inputs.proto_inputs.SerializeToString()
    assert called["path"] == "some/path"


@pytest.mark.asyncio
async def test_upload_outputs_within_limit(monkeypatch):
    called = {}

    async def fake_put_stream(data_iterable, to_path):
        called["data"] = data_iterable
        called["path"] = to_path

    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)
    outputs = await create_outputs(5)

    await io.upload_outputs(outputs, "out/path", max_bytes=100)
    assert called["data"] == outputs.proto_outputs.SerializeToString()
    assert called["path"].endswith("outputs.pb")


@pytest.mark.asyncio
async def test_upload_outputs_exceeds_limit(monkeypatch):
    monkeypatch.setattr(io.storage, "put_stream", lambda *a, **kw: None)
    outputs = await create_outputs(50)

    with pytest.raises(flyte.errors.InlineIOMaxBytesBreached) as excinfo:
        await io.upload_outputs(outputs, "out/path", max_bytes=10)
    assert "exceeds max_bytes limit" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_inputs_within_limit(monkeypatch):
    inputs = await create_inputs(10)
    serialized = inputs.proto_inputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    loaded = await io.load_inputs("some/path", max_bytes=100)
    assert loaded.proto_inputs == inputs.proto_inputs


@pytest.mark.asyncio
async def test_load_inputs_exceeds_limit(monkeypatch):
    inputs = await create_inputs(20)
    serialized = inputs.proto_inputs.SerializeToString()

    async def fake_get_stream(path):
        # Simulate chunking
        yield serialized[:10]
        yield serialized[10:]

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    with pytest.raises(flyte.errors.InlineIOMaxBytesBreached) as excinfo:
        await io.load_inputs("some/path", max_bytes=15)
    assert "exceeds max_bytes limit" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_outputs_within_limit(monkeypatch):
    outputs = await create_outputs(10)
    serialized = outputs.proto_outputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    loaded = await io.load_outputs("out/path", max_bytes=100)
    assert loaded.proto_outputs == outputs.proto_outputs


@pytest.mark.asyncio
async def test_load_outputs_exceeds_limit(monkeypatch):
    outputs = await create_outputs(20)
    serialized = outputs.proto_outputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized[:10]
        yield serialized[10:]

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    with pytest.raises(flyte.errors.InlineIOMaxBytesBreached) as excinfo:
        await io.load_outputs("out/path", max_bytes=15)
    assert "exceeds max_bytes limit" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_inputs_path_rewrite(monkeypatch):
    inputs = await create_blob_inputs("s3://old_prefix/some/path")
    serialized = inputs.proto_inputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    path_rewrite_config = flyte.models.PathRewrite(old_prefix="s3://old_prefix", new_prefix="/tmp/new_prefix")
    loaded = await io.load_inputs("some/path", max_bytes=1000, path_rewrite_config=path_rewrite_config)
    for lit in loaded.proto_inputs.literals:
        assert lit.value.scalar.blob.uri.startswith("/tmp/new_prefix")
        assert lit.value.scalar.blob.uri == "/tmp/new_prefix/some/path"


@pytest.mark.asyncio
async def test_load_inputs_path_rewrite_no_match(monkeypatch):
    inputs = await create_blob_inputs("s3://old_prefix/some/path")
    serialized = inputs.proto_inputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    path_rewrite_config = flyte.models.PathRewrite(old_prefix="s3://old_prefix1", new_prefix="/tmp/new_prefix")
    loaded = await io.load_inputs("some/path", max_bytes=1000, path_rewrite_config=path_rewrite_config)
    for lit in loaded.proto_inputs.literals:
        assert lit.value.scalar.blob.uri.startswith("s3://old_prefix")
        assert lit.value.scalar.blob.uri == "s3://old_prefix/some/path"


@pytest.mark.asyncio
async def test_load_inputs_uses_the_prefetched_download(monkeypatch):
    """The runtime starts the inputs download early (prefetch_inputs); load_inputs
    takes that result instead of downloading a second time."""
    import threading

    inputs = await create_inputs(10)
    serialized = inputs.proto_inputs.SerializeToString()
    calls = []

    async def fake_get_stream(path):
        calls.append(threading.current_thread().name)
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    io.prefetch_inputs("prefetch/ok")
    io.prefetch_inputs("prefetch/ok")  # started once
    loaded = await io.load_inputs("prefetch/ok")
    assert loaded.proto_inputs == inputs.proto_inputs
    assert calls == ["flyte-inputs-prefetch"]
    assert "prefetch/ok" not in io._prefetched_inputs


@pytest.mark.asyncio
async def test_load_inputs_downloads_again_when_the_prefetch_failed(monkeypatch):
    inputs = await create_inputs(10)
    serialized = inputs.proto_inputs.SerializeToString()
    attempts = []

    async def flaky_get_stream(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise OSError("transient")
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", flaky_get_stream)
    io.prefetch_inputs("prefetch/flaky")
    loaded = await io.load_inputs("prefetch/flaky")
    assert loaded.proto_inputs == inputs.proto_inputs
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_prefetched_inputs_still_honour_max_bytes(monkeypatch):
    inputs = await create_inputs(20)
    serialized = inputs.proto_inputs.SerializeToString()

    async def fake_get_stream(path):
        yield serialized

    monkeypatch.setattr(io.storage, "get_stream", fake_get_stream)
    io.prefetch_inputs("prefetch/big")
    with pytest.raises(flyte.errors.InlineIOMaxBytesBreached):
        await io.load_inputs("prefetch/big", max_bytes=15)
