from pathlib import Path

import pytest

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
