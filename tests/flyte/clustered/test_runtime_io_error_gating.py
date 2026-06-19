from __future__ import annotations

import pytest
from flyteidl2.core import execution_pb2
from flyteidl2.task import common_pb2 as run_definition_pb2

from flyte._internal.runtime import io


def _error() -> execution_pb2.ExecutionError:
    return execution_pb2.ExecutionError(code="UserError", message="boom", kind=execution_pb2.ExecutionError.USER)


def _outputs() -> io.Outputs:
    return io.Outputs(proto_outputs=run_definition_pb2.Outputs())


def _isolate_clustered_env(monkeypatch):
    """Start from a clean slate so each test sets only the env it cares about."""
    for var in ("JOBSET_RESTART_ATTEMPT", "JOBSET_MAX_RESTARTS", "TORCHELASTIC_RUN_ID", "RANK"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def captured_puts(monkeypatch):
    puts: list[str] = []

    async def fake_put_stream(data_iterable, to_path):
        puts.append(to_path)
        return to_path

    monkeypatch.setattr(io.storage, "put_stream", fake_put_stream)
    return puts


def test_is_clustered_worker(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    assert io._is_clustered_worker() is False
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-123")
    assert io._is_clustered_worker() is True


@pytest.mark.asyncio
async def test_upload_error_skips_write_on_transient_restart(monkeypatch, captured_puts):
    """attempt 0 of a 2-restart budget → not terminal → no error.pb written."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "0")
    monkeypatch.setenv("JOBSET_MAX_RESTARTS", "2")
    output_prefix = "s3://bucket/outputs"

    uri = await io.upload_error(_error(), output_prefix)

    assert uri == io.error_path(output_prefix)
    assert captured_puts == []


@pytest.mark.asyncio
async def test_upload_error_writes_on_terminal_attempt(monkeypatch, captured_puts):
    """attempt == max → terminal → error.pb written."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "2")
    monkeypatch.setenv("JOBSET_MAX_RESTARTS", "2")
    output_prefix = "s3://bucket/outputs"

    await io.upload_error(_error(), output_prefix)

    assert captured_puts == [io.error_path(output_prefix)]


@pytest.mark.asyncio
async def test_upload_error_writes_for_non_clustered_task(monkeypatch, captured_puts):
    """No JOBSET_RESTART_ATTEMPT → regular task → always write."""
    _isolate_clustered_env(monkeypatch)
    output_prefix = "s3://bucket/outputs"

    await io.upload_error(_error(), output_prefix)

    assert captured_puts == [io.error_path(output_prefix)]


@pytest.mark.asyncio
async def test_upload_error_writes_when_budget_env_missing(monkeypatch, captured_puts):
    """attempt set but JOBSET_MAX_RESTARTS not injected (old operator) → safe fallback: write."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "0")
    output_prefix = "s3://bucket/outputs"

    await io.upload_error(_error(), output_prefix)

    assert captured_puts == [io.error_path(output_prefix)]


@pytest.mark.asyncio
async def test_upload_error_max_restarts_zero_writes_immediately(monkeypatch, captured_puts):
    """max_restarts=0 → attempt 0 is already terminal → write."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "0")
    monkeypatch.setenv("JOBSET_MAX_RESTARTS", "0")
    output_prefix = "s3://bucket/outputs"

    await io.upload_error(_error(), output_prefix)

    assert captured_puts == [io.error_path(output_prefix)]


@pytest.mark.asyncio
async def test_upload_error_nonzero_rank_skips_write(monkeypatch, captured_puts):
    """A non-rank-0 clustered worker never writes error.pb, regardless of restart budget."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-123")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "2")
    monkeypatch.setenv("JOBSET_MAX_RESTARTS", "2")
    output_prefix = "s3://bucket/outputs"

    uri = await io.upload_error(_error(), output_prefix)

    assert uri == io.error_path(output_prefix)
    assert captured_puts == []


@pytest.mark.asyncio
async def test_upload_outputs_does_not_delete_on_restarted_attempt(monkeypatch, captured_puts):
    """upload_outputs no longer touches error.pb: no exists() / filesystem delete calls."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "2")
    output_path = "s3://bucket/outputs"

    def _fail(*_args, **_kwargs):
        raise AssertionError("upload_outputs must not inspect or delete error.pb")

    monkeypatch.setattr(io.storage, "exists", _fail)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.upload_outputs(_outputs(), output_path)

    assert captured_puts == [io.outputs_path(output_path)]
