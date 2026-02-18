import os
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from flyte._cache.local_cache import LocalTaskCache
from flyte._context import RawDataPath, internal_ctx
from flyte.models import SerializationContext


@pytest.fixture
def ctx_with_test_raw_data_path():
    """Pytest fixture to set a RawDataPath in the internal_ctx."""
    raw_data_path = RawDataPath.from_local_folder()
    ctx = internal_ctx()
    new_context = ctx.new_raw_data_path(raw_data_path=raw_data_path)
    with new_context as ctx:
        yield ctx


@pytest.fixture
def dummy_serialization_context():
    yield SerializationContext(
        code_bundle=None,
        version="abc123",
        input_path="s3://bucket/test/run/inputs.pb",
        output_path="s3://bucket/outputs/0/jfkljfa/0",
        root_dir=Path.cwd(),
    )


@pytest_asyncio.fixture(autouse=True)
async def isolate_local_cache(tmp_path):
    """
    Global fixture to isolate LocalTaskCache for each test.
    Uses temporary directory to avoid polluting local development cache.
    """
    with patch.object(LocalTaskCache, "_get_cache_path", return_value=str(tmp_path / "test_cache.db")):
        LocalTaskCache._initialized = False
        yield
        await LocalTaskCache.close()


@pytest.fixture(autouse=True)
def patch_os_exit(monkeypatch):
    def mock_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", mock_exit)
