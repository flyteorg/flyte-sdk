"""Unit tests for the stale error.pb cleanup that runs at the start of every clustered rank-0 worker."""

from __future__ import annotations

from unittest import mock

import pytest
from fsspec.asyn import AsyncFileSystem

from flyte._internal.runtime import io

OUTPUT_PATH = "s3://bucket/outputs"
ERROR_URI = io.error_path(OUTPUT_PATH)


def _isolate_clustered_env(monkeypatch):
    for var in (
        "JOBSET_RESTART_ATTEMPT",
        "JOBSET_MAX_RESTARTS",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "RANK",
    ):
        monkeypatch.delenv(var, raising=False)


def _as_rank0_clustered(monkeypatch, restart_attempt: str | None = "1"):
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-123")
    monkeypatch.setenv("RANK", "0")
    if restart_attempt is not None:
        monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", restart_attempt)


def _fail(*_args, **_kwargs):
    raise AssertionError("storage must not be touched")


async def _exists_only_error(path: str, **_kwargs) -> bool:
    return path == ERROR_URI


async def _exists_false(_path: str, **_kwargs) -> bool:
    return False


@pytest.fixture
def warnings(monkeypatch) -> list[str]:
    captured: list[str] = []
    monkeypatch.setattr(io.logger, "warning", captured.append)
    return captured


@pytest.mark.asyncio
async def test_clear_stale_error_deletes_via_async_rm_file(monkeypatch, warnings):
    """Remote stores (obstore FsspecStore) are AsyncFileSystems: the async _rm_file must be awaited."""
    _as_rank0_clustered(monkeypatch, "1")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    fs = mock.MagicMock(spec=AsyncFileSystem)
    monkeypatch.setattr(io.storage, "exists", _exists_only_error)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: fs)

    await io.clear_stale_clustered_error(OUTPUT_PATH)

    fs._rm_file.assert_awaited_once_with(ERROR_URI)
    fs.rm_file.assert_not_called()
    assert len(warnings) == 1
    assert "Removed stale" in warnings[0]
    assert ERROR_URI in warnings[0]
    assert "JOBSET_RESTART_ATTEMPT=1" in warnings[0]
    assert "TORCHELASTIC_RESTART_COUNT=0" in warnings[0]


@pytest.mark.asyncio
async def test_clear_stale_error_deletes_via_sync_rm_file_on_local_fs(monkeypatch, tmp_path, warnings):
    """Real local filesystem end to end: storage.exists + LocalFileSystem.rm_file (the sync branch)."""
    _as_rank0_clustered(monkeypatch, "2")
    stale = tmp_path / io._ERROR_FILE_NAME
    stale.write_bytes(b"stale")

    await io.clear_stale_clustered_error(str(tmp_path))

    assert not stale.exists()
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_delete_path_uses_sync_rm_file_for_sync_filesystems(monkeypatch):
    class SyncFS:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def rm_file(self, path: str) -> None:
            self.deleted.append(path)

    fs = SyncFS()
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: fs)

    await io._delete_path(ERROR_URI)

    assert fs.deleted == [ERROR_URI]


@pytest.mark.asyncio
async def test_clear_stale_error_absent_file_is_noop(monkeypatch, warnings):
    _as_rank0_clustered(monkeypatch, "1")
    monkeypatch.setattr(io.storage, "exists", _exists_false)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.clear_stale_clustered_error(OUTPUT_PATH)

    assert warnings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_attempt", ["0", None])
async def test_clear_stale_error_runs_on_first_start_too(monkeypatch, restart_attempt):
    """The cleanup is deliberately not keyed on JOBSET_RESTART_ATTEMPT: that counter is exactly what
    over-counts under free restarts, and a first start is a cheap no-op after one existence check."""
    _as_rank0_clustered(monkeypatch, restart_attempt)
    seen: list[str] = []

    async def exists_spy(path: str, **_kwargs) -> bool:
        seen.append(path)
        return False

    monkeypatch.setattr(io.storage, "exists", exists_spy)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.clear_stale_clustered_error(OUTPUT_PATH)

    assert seen == [ERROR_URI]


@pytest.mark.asyncio
async def test_clear_stale_error_skips_nonzero_rank(monkeypatch):
    """Only rank-0 owns error.pb; a late-starting rank must never delete what rank-0 just wrote."""
    _as_rank0_clustered(monkeypatch, "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setattr(io.storage, "exists", _fail)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.clear_stale_clustered_error(OUTPUT_PATH)


@pytest.mark.asyncio
async def test_clear_stale_error_skips_non_clustered_task(monkeypatch):
    """RANK / JOBSET_RESTART_ATTEMPT without the torchrun marker is not a clustered worker."""
    _isolate_clustered_env(monkeypatch)
    monkeypatch.setenv("JOBSET_RESTART_ATTEMPT", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(io.storage, "exists", _fail)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.clear_stale_clustered_error(OUTPUT_PATH)


@pytest.mark.asyncio
async def test_clear_stale_error_delete_failure_is_soft(monkeypatch, warnings):
    _as_rank0_clustered(monkeypatch, "1")
    fs = mock.MagicMock(spec=AsyncFileSystem)
    fs._rm_file.side_effect = RuntimeError("delete failed")
    monkeypatch.setattr(io.storage, "exists", _exists_only_error)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: fs)

    await io.clear_stale_clustered_error(OUTPUT_PATH)  # must not raise

    assert len(warnings) == 1
    assert "Could not remove" in warnings[0]
    assert "delete failed" in warnings[0]


@pytest.mark.asyncio
async def test_clear_stale_error_vanished_before_delete_is_debug(monkeypatch, warnings):
    _as_rank0_clustered(monkeypatch, "1")
    debugs: list[str] = []
    monkeypatch.setattr(io.logger, "debug", debugs.append)
    fs = mock.MagicMock(spec=AsyncFileSystem)
    fs._rm_file.side_effect = FileNotFoundError(ERROR_URI)
    monkeypatch.setattr(io.storage, "exists", _exists_only_error)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", lambda **_kwargs: fs)

    await io.clear_stale_clustered_error(OUTPUT_PATH)

    assert warnings == []
    assert len(debugs) == 1
    assert ERROR_URI in debugs[0]


@pytest.mark.asyncio
async def test_clear_stale_error_exists_failure_is_soft(monkeypatch, warnings):
    _as_rank0_clustered(monkeypatch, "1")

    async def exists_boom(_path: str, **_kwargs) -> bool:
        raise PermissionError("denied")

    monkeypatch.setattr(io.storage, "exists", exists_boom)
    monkeypatch.setattr(io.storage, "get_underlying_filesystem", _fail)

    await io.clear_stale_clustered_error(OUTPUT_PATH)  # must not raise

    assert len(warnings) == 1
    assert "denied" in warnings[0]
