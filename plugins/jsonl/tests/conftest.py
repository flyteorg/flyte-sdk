import os
from unittest.mock import patch

import pytest
import pytest_asyncio
from flyte._cache.local_cache import LocalTaskCache
from flyte._context import RawDataPath, internal_ctx
from flyte._persistence._db import LocalDB


@pytest.fixture
def ctx_with_test_raw_data_path():
    """Pytest fixture to set a RawDataPath in the internal_ctx."""
    raw_data_path = RawDataPath.from_local_folder()
    ctx = internal_ctx()
    new_context = ctx.new_raw_data_path(raw_data_path=raw_data_path)
    with new_context as ctx:
        yield ctx


@pytest_asyncio.fixture(autouse=True)
async def isolate_local_cache(tmp_path):
    """Isolate LocalTaskCache for each test."""
    with patch.object(LocalDB, "_get_db_path", staticmethod(lambda: str(tmp_path / "test_cache.db"))):
        LocalDB._initialized = False
        LocalDB._conn = None
        LocalDB._conn_sync = None
        yield
        await LocalTaskCache.close()


@pytest.fixture(autouse=True)
def patch_os_exit(monkeypatch):
    def mock_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", mock_exit)
