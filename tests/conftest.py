import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from flyte._cache.local_cache import LocalTaskCache
from flyte._context import Context, ContextData, internal_ctx
from flyte.models import ActionID, RawDataPath, SerializationContext, TaskContext
from flyte.report import Report


@pytest.fixture
def ctx_with_test_raw_data_path():
    """Pytest fixture to set a RawDataPath in the internal_ctx."""
    raw_data_path = RawDataPath.from_local_folder()
    ctx = internal_ctx()
    new_context = ctx.new_raw_data_path(raw_data_path=raw_data_path)
    with new_context as ctx:
        yield ctx


@pytest.fixture
def ctx_with_preserve_original_types():
    """Pytest fixture to set up a TaskContext with preserve_original_types=True."""
    raw_data_path = RawDataPath.from_local_folder()
    action = ActionID.create_random()
    task_context = TaskContext(
        action=action,
        version="test",
        raw_data_path=raw_data_path,
        output_path="/tmp/test/output",
        run_base_dir="/tmp/test",
        report=Report(name=action.name),
        mode="local",
        preserve_original_types=True,
    )
    ctx = Context(data=ContextData(task_context=task_context, raw_data_path=raw_data_path))
    with ctx:
        yield ctx


@pytest.fixture
def ctx_with_test_local_s3_stack_raw_data_path():
    """Pytest fixture to set a RawDataPath in the internal_ctx."""
    raw_data_path = RawDataPath(path="s3://bucket/tests/default_upload/")
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


@pytest.fixture
def local_dummy_file():
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write("Hello File")
        yield path
    finally:
        os.remove(path)


@pytest.fixture
def local_dummy_directory():
    temp_dir = tempfile.TemporaryDirectory()
    try:
        with open(os.path.join(temp_dir.name, "file"), "w") as tmp:
            tmp.write("Hello Dir")
        yield temp_dir.name
    finally:
        temp_dir.cleanup()


@pytest.fixture(autouse=True)
def patch_os_exit(monkeypatch):
    def mock_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", mock_exit)
