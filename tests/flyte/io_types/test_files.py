from __future__ import annotations

import filecmp
import os
import tempfile
from typing import Optional

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
    content = str(content, "utf-8")
    assert "col1,col2" in content

    pv2 = File.from_existing_remote(uploaded_file.path)
    async with pv2.open() as fh:
        content = memoryview(await fh.read())
    content = str(content, "utf-8")
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
    async def my_task(
        file_name: Optional[str] = None,
    ) -> File[pd.DataFrame]:
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        file = File.new_remote(file_name=file_name)
        with file.open_sync("wb") as fh:
            df.to_csv(fh, index=False)
        return file

    file_names = [None, "data.csv"]
    for file_name in file_names:
        file = await my_task(file_name=file_name)
        _, ext = os.path.splitext(os.path.basename(file.path))
        if file_name is None:
            assert len(ext) == 0
        else:
            assert len(ext) > 0

        assert file.hash is None
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
    content = str(content, "utf-8")
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
        content = str(content, "utf-8")
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


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_file_with_name(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Test downloading a file from S3 to a local path with a specified file name.
    """
    await flyte.init.aio(storage=S3.for_sandbox())

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to S3
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Download to a specific local path with a custom name
    download_target = tmp_path / "downloaded" / "my_custom_filename"
    download_target.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_file.download(str(download_target))

    print(f"Downloaded file to {downloaded_path}", flush=True)

    assert downloaded_path.endswith("my_custom_filename")
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)
    assert downloaded_path == str(download_target)


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_file_with_folder_name(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Test downloading a file to a directory path.
    When a directory path is provided (either existing directory or path ending with os.sep),
    the file should be downloaded with its original remote filename.
    """
    await flyte.init.aio(storage=S3.for_sandbox())

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to S3
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Test 1: Download to an existing directory
    download_dir = tmp_path / "downloaded_existing"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_file.download(str(download_dir))

    print(f"Downloaded file to {downloaded_path}", flush=True)

    # Verify the file was downloaded into the directory with the remote filename
    remote_filename = uploaded_file.path.split("/")[-1]
    expected_path = download_dir / remote_filename
    assert downloaded_path == str(expected_path)
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)

    # Test 2: Download to a non-existent path ending with os.sep
    download_dir2_str = str(tmp_path / "downloaded_new") + os.sep  # Ends with separator
    downloaded_path2 = await uploaded_file.download(download_dir2_str)

    print(f"Downloaded file to {downloaded_path2}", flush=True)

    # Verify the file was downloaded into the directory with the remote filename
    expected_path2 = tmp_path / "downloaded_new" / remote_filename
    assert downloaded_path2 == str(expected_path2)
    assert filecmp.cmp(local_file, downloaded_path2, shallow=False)


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_file_with_no_local_target(tmp_path, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Test downloading a file from S3 without specifying a target path.
    The file should be downloaded to a temporary location.
    """
    await flyte.init.aio(storage=S3.for_sandbox())

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to S3
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Download without specifying a target path
    downloaded_path = await uploaded_file.download()

    print(f"Downloaded file to {downloaded_path}", flush=True)

    # Verify the files match
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)
    assert downloaded_path is not None
    assert os.path.isfile(downloaded_path)
    suffix = uploaded_file.path.split("/")[-1]
    assert downloaded_path.endswith(suffix)


@pytest.mark.asyncio
async def test_download_file_with_name_local(tmp_path, ctx_with_test_raw_data_path):
    """
    Test downloading a file from local storage to a local path with a specified file name.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to "remote" (which is actually local)
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Download to a specific local path with a custom name
    download_target = tmp_path / "downloaded" / "my_custom_filename"
    download_target.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_file.download(str(download_target))

    print(f"Downloaded file to {downloaded_path}", flush=True)

    assert downloaded_path.endswith("my_custom_filename")
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)
    assert downloaded_path == str(download_target)


@pytest.mark.asyncio
async def test_download_file_with_folder_name_local(tmp_path, ctx_with_test_raw_data_path):
    """
    Test downloading a file to a directory path using local storage.
    When a directory path is provided (either existing directory or path ending with os.sep),
    the file should be downloaded with its original remote filename.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to "remote" (which is actually local)
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Test 1: Download to an existing directory
    download_dir = tmp_path / "downloaded_existing"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_file.download(str(download_dir))

    print(f"Downloaded file to {downloaded_path}", flush=True)

    # Verify the file was downloaded into the directory with the remote filename
    remote_filename = uploaded_file.path.split(os.sep)[-1]
    expected_path = download_dir / remote_filename
    assert downloaded_path == str(expected_path)
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)

    # Test 2: Download to a non-existent path ending with os.sep
    download_dir2_str = str(tmp_path / "downloaded_new") + os.sep  # Ends with separator
    downloaded_path2 = await uploaded_file.download(download_dir2_str)

    print(f"Downloaded file to {downloaded_path2}", flush=True)

    # Verify the file was downloaded into the directory with the remote filename
    expected_path2 = tmp_path / "downloaded_new" / remote_filename
    assert downloaded_path2 == str(expected_path2)
    assert filecmp.cmp(local_file, downloaded_path2, shallow=False)


@pytest.mark.asyncio
async def test_download_file_with_no_local_target_local(tmp_path, ctx_with_test_raw_data_path):
    """
    Test downloading a file from local storage without specifying a target path.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Create a local file with random content
    local_file = tmp_path / "one_hundred_bytes"
    local_file.write_bytes(os.urandom(100))

    # Upload to "remote" (which is actually local)
    uploaded_file = await File.from_local(str(local_file))
    print(f"Uploaded temp file {local_file} to {uploaded_file.path}", flush=True)

    # Download without specifying a target path
    downloaded_path = await uploaded_file.download()

    print(f"Downloaded file to {downloaded_path}", flush=True)

    # Verify the files match
    assert filecmp.cmp(local_file, downloaded_path, shallow=False)
    assert downloaded_path is not None
    assert os.path.isfile(downloaded_path)
    suffix = uploaded_file.path.split(os.sep)[-1]
    assert downloaded_path.endswith(suffix)


# Tests for lazy_uploader functionality


@pytest.mark.asyncio
async def test_file_from_local_creates_lazy_uploader_without_raw_data_context():
    """Test that File.from_local creates a lazy_uploader when there's no raw_data context."""
    flyte.init()

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "test_file.txt")
        with open(local_path, "w") as f:  # noqa: ASYNC230
            f.write("test content for lazy uploader")

        # When creating a File from local without raw_data context, it should have lazy_uploader
        file = await File.from_local(local_path)

        # The file should have a lazy_uploader set
        assert file.lazy_uploader is not None
        # The path should be the local path
        assert file.path == local_path


@pytest.mark.asyncio
async def test_file_from_local_sync_creates_lazy_uploader_without_raw_data_context():
    """Test that File.from_local_sync creates a lazy_uploader when there's no raw_data context."""
    flyte.init()

    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, "test_file_sync.txt")
        with open(local_path, "w") as f:
            f.write("test content for lazy uploader sync")

        # When creating a File from local without raw_data context, it should have lazy_uploader
        file = File.from_local_sync(local_path)

        # The file should have a lazy_uploader set
        assert file.lazy_uploader is not None
        # The path should be the local path
        assert file.path == local_path


@pytest.mark.asyncio
async def test_lazy_uploader_returns_local_path_in_local_mode():
    """Test that lazy_uploader returns local path when in local mode."""
    from flyte._run import _run_mode_var

    flyte.init()
    _run_mode_var.set("local")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "test_local_mode.txt")
            with open(local_path, "w") as f:  # noqa: ASYNC230
                f.write("content for local mode test")

            file = await File.from_local(local_path)
            assert file.lazy_uploader is not None

            # When we call lazy_uploader in local mode, it should return the local path
            hash_val, uri = await file.lazy_uploader()
            assert uri == local_path
            assert hash_val is None
    finally:
        _run_mode_var.set(None)


@pytest.mark.asyncio
async def test_file_transformer_uses_lazy_uploader_in_local_mode():
    """Test that FileTransformer.to_literal uses lazy_uploader when in local mode."""
    from flyte._run import _run_mode_var

    flyte.init()
    _run_mode_var.set("local")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = os.path.join(temp_dir, "transformer_test.txt")
            with open(local_path, "w") as f:  # noqa: ASYNC230
                f.write("content for transformer test")

            file = await File.from_local(local_path)

            lt = TypeEngine.to_literal_type(File)
            lv = await FileTransformer().to_literal(file, File, lt)

            # The literal should contain the local path
            assert lv.scalar.blob.uri == local_path
    finally:
        _run_mode_var.set(None)


@pytest.mark.asyncio
async def test_file_without_lazy_uploader_uses_existing_path():
    """Test that File without lazy_uploader uses the existing path in to_literal."""
    flyte.init()

    # Create a File pointing to a remote path (no lazy_uploader)
    remote_file = File.from_existing_remote("s3://bucket/remote_file.txt")
    assert remote_file.lazy_uploader is None

    lt = TypeEngine.to_literal_type(File)
    lv = await FileTransformer().to_literal(remote_file, File, lt)

    # The literal should contain the original remote path
    assert lv.scalar.blob.uri == "s3://bucket/remote_file.txt"
