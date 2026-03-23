import pytest
from flyte._context import RawDataPath, internal_ctx


@pytest.fixture
def ctx_with_test_raw_data_path():
    raw_data_path = RawDataPath.from_local_folder()
    ctx = internal_ctx()
    new_context = ctx.new_raw_data_path(raw_data_path=raw_data_path)
    with new_context as c:
        yield c
