# from flytekit import dynamic, kwtypes, task, workflow

import os
import tempfile

import pandas as pd
import pytest

import flyte
from flyte.io._file import File, FileTransformer
from flyte.storage import S3
from flyte.types import TypeEngine


@pytest.mark.asyncio
async def test_transformer_serde():
    f = File(path="s3://bucket/file.txt")
    lt = TypeEngine.to_literal_type(File)
    lv = await FileTransformer().to_literal(f, File, lt)
    pv = await FileTransformer().to_python_value(lv, File)
    assert pv == f


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
        content = fh.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content

    pv2 = File.from_existing_remote(uploaded_file.path)
    async with pv2.open() as fh:
        content = fh.read()
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


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_task_write_file_streaming(ctx_with_test_raw_data_path):
    flyte.init(storage=S3.for_sandbox())

    # Simulate writing a file by streaming it directly to blob storage
    async def my_task() -> File[pd.DataFrame]:
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        file = File.new_remote()
        async with file.open("wb") as fh:
            df.to_csv(fh, index=False)
        return file

    file = await my_task()
    async with file.open() as f:
        content = f.read()
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
        uploaded_file = await File.from_local(local_path, remote_destination="s3://my-s3-bucket/data.csv")
        return uploaded_file

    file = await my_task()
    assert file.path == "s3://my-s3-bucket/data.csv"
    pv2 = File.from_existing_remote(file.path)
    async with pv2.open() as fh:
        content = fh.read()
    content = content.decode("utf-8")
    assert "col1,col2" in content
