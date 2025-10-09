# from flytekit import dynamic, kwtypes, task, workflow

import os
import tempfile

import pandas as pd
import pytest
from mashumaro.jsonschema.models import JSONSchema

import flyte
from flyte.io._file import File, FileTransformer
from flyte.io._hashing_io import HashlibAccumulator
from flyte.storage import S3
from flyte.types import TypeEngine

TEST_CONTENT = "correct test content"
TEST_SHA256 = "88a884456e029050823d8a0474b8c96986fcc3996a2a2a018b918181633cbe56"


@pytest.mark.asyncio
async def test_file_is_schemable():
    f = File(path="s3://bucket/file.txt")

    pydantic_schema = f.model_json_schema()
    assert JSONSchema.from_dict(pydantic_schema)


@pytest.mark.asyncio
async def test_transformer_serde():
    f = File(path="s3://bucket/file.txt")
    lt = TypeEngine.to_literal_type(File)
    lv = await FileTransformer().to_literal(f, File, lt)
    assert not lv.hash
    pv = await FileTransformer().to_python_value(lv, File)
    assert pv == f


@pytest.mark.asyncio
async def test_transformer_serde_set_hash():
    f = File.from_existing_remote("s3://bucket/file.txt", file_cache_key="abc")
    lt = TypeEngine.to_literal_type(File)
    lv = await FileTransformer().to_literal(f, File, lt)
    assert lv.hash == "abc"
    pv = await FileTransformer().to_python_value(lv, File)
    assert pv == f
    assert pv.hash == "abc"
    assert pv.path == f.path


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_async_file_read(ctx_with_test_raw_data_path):
    # Create a temporary file to simulate the remote file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "data.csv")
    with open(file_path, "w") as f:  # noqa: ASYNC230
        f.write("col1,col2\n1,2\n3,4")

    flyte.init(storage=S3.for_sandbox())
    # Simulate an async file read
    uploaded_file = await File.from_local(file_path)
    print(uploaded_file)
    lv = await FileTransformer().to_literal(uploaded_file, File, TypeEngine.to_literal_type(File))
    pv = await FileTransformer().to_python_value(lv, File)
    async with pv.open() as fh:
        content = await fh.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content

    pv2 = File.from_existing_remote(uploaded_file.path)
    async with pv2.open() as fh:
        content = await fh.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content


def test_sync_file_read():
    # Create a temporary file to simulate the remote file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "data.csv")
    with open(file_path, "w") as f:
        f.write("col1,col2\n1,2\n3,4")

    # Simulate an async file read
    csv_file = File[pd.DataFrame](path=file_path)
    with csv_file.open_sync() as f:
        content = f.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content


@pytest.mark.asyncio
async def test_task_write_file_streaming(ctx_with_test_raw_data_path):
    flyte.init()

    # Simulate writing a file by streaming it directly to blob storage
    async def my_task() -> File[pd.DataFrame]:
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        file = File.new_remote()
        with file.open_sync("wb") as fh:
            df.to_csv(fh, index=False)
        return file

    file = await my_task()
    assert file.hash is None
    async with file.open() as f:
        content = await f.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_task_write_file_streaming_locals3(ctx_with_test_local_s3_stack_raw_data_path):
    flyte.init(storage=S3.for_sandbox())

    # Simulate writing a file by streaming it directly to blob storage
    async def my_task() -> File[pd.DataFrame]:
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        a = HashlibAccumulator.from_hash_name("sha256")
        file = File.new_remote(hash_method=a)
        with file.open_sync("wb") as fh:
            df.to_csv(fh, index=False)
            fh.close()  # context manager should also close but this should still work
        return file

    file = await my_task()
    assert file.hash == "9b0a34a69b639520f1a18e54f85544d9d379f81727eb44b8814e1c4707e1760d"
    async with file.open() as f:
        content = await f.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_task_write_file_local_then_upload(ctx_with_test_raw_data_path):
    flyte.init(storage=S3.for_sandbox())

    # Simulate writing a file locally first, then uploading it
    async def my_task() -> File[pd.DataFrame]:
        temp_dir = tempfile.mkdtemp()
        local_path = os.path.join(temp_dir, "data.csv")
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(local_path, index=False)
        uploaded_file = await File.from_local(local_path, remote_destination="s3://bucket/data.csv")
        return uploaded_file

    file = await my_task()
    assert file.path == "s3://bucket/data.csv"
    pv2 = File.from_existing_remote(file.path)
    async with pv2.open() as fh:
        content = await fh.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content


@pytest.mark.asyncio
async def test_from_local_with_local_files():
    flyte.init()

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "source.txt")
        remote_path = os.path.join(temp_dir, "destination.txt")

        test_content = "correct test content"
        with open(local_path, "w") as f:  # noqa: ASYNC230
            f.write(test_content)

        result = await File.from_local(local_path, remote_path)

        assert result.path == remote_path
        async with result.open() as f:
            content = await f.read()
        content = content.decode("utf-8")
        assert content == test_content


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_from_local_to_s3(ctx_with_test_local_s3_stack_raw_data_path):
    flyte.init(storage=S3.for_sandbox())

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "source.txt")

        with open(local_path, "w") as f:  # noqa: ASYNC230
            f.write(TEST_CONTENT)

        a = HashlibAccumulator.from_hash_name("sha256")
        result = await File.from_local(local_path, hash_method=a)
        assert result.path.startswith("s3://bucket/tests/default_upload/")
        assert result.hash == TEST_SHA256

        async with result.open() as f:
            content = await f.read()
        content = content.decode("utf-8")
        assert content == TEST_CONTENT


@pytest.mark.asyncio
async def test_transformer_serde_with_hash():
    """
    Test that the FileTransformer correctly serializes and deserializes File objects with hash values.
    """
    f = File.from_existing_remote("s3://bucket/file.txt", file_cache_key="abc123")
    lt = TypeEngine.to_literal_type(File)
    lv = await FileTransformer().to_literal(f, File, lt)

    # Hash should be preserved in the literal
    assert lv.hash == "abc123"

    # Convert back to Python
    pv = await FileTransformer().to_python_value(lv, File)
    assert pv.path == f.path
    assert pv.hash == "abc123"


@pytest.mark.asyncio
async def test_multiple_files_with_hashes():
    """
    Test handling multiple File objects with different hash scenarios.
    """
    # Create multiple files with different hash scenarios
    file_with_hash = File.from_existing_remote("s3://bucket/file1.txt", file_cache_key="hash1")
    file_without_hash = File.from_existing_remote("s3://bucket/file2.txt")

    files = [file_with_hash, file_without_hash]

    # Convert to literals
    transformer = FileTransformer()
    lt = TypeEngine.to_literal_type(File)

    literals = []
    for f in files:
        lv = await transformer.to_literal(f, File, lt)
        literals.append(lv)

    # First file should have hash, second should not
    assert literals[0].hash == "hash1"
    assert not literals[1].hash

    # Convert back to Python objects
    recovered_files = []
    for lv in literals:
        pv = await transformer.to_python_value(lv, File)
        recovered_files.append(pv)

    # Verify all properties are preserved
    assert recovered_files[0].path == file_with_hash.path
    assert recovered_files[0].hash == "hash1"
    assert recovered_files[1].path == file_without_hash.path
    assert recovered_files[1].hash is None


@pytest.mark.asyncio
async def test_fdsjkl():
    await flyte.init.aio(storage=S3.for_sandbox())

    small_file = File.from_existing_remote(
        "s3://bucket/tests/default_upload/38d779853cea2083f740ab048e9185fd/one_hundred_bytes"
    )
    xx = await small_file.download("/Users/ytong/temp/my_small_file")
    print(xx)
