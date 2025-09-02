import pytest

import flyte.errors
import flyte.types as types
from flyte._internal.runtime import io
from flyteidl2.workflow import run_definition_pb2


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
