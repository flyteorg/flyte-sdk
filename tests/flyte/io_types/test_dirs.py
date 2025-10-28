from __future__ import annotations

import os
import os.path
import pathlib
import shutil
import tempfile

import pytest

import flyte
import flyte.storage as storage
from flyte.io._dir import Dir, DirTransformer
from flyte.io._file import File
from flyte.types import TypeEngine


@pytest.fixture
def tmp_dir_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Root level file
        with open(os.path.join(tmpdir, "root.txt"), "w") as f:
            f.write("root level file")

        # Nested folder 1
        nested1 = os.path.join(tmpdir, "nested1")
        os.makedirs(nested1)
        with open(os.path.join(nested1, "file1.txt"), "w") as f:
            f.write("file in nested1")

        # Nested folder 2 inside nested1
        nested2 = os.path.join(nested1, "nested2")
        os.makedirs(nested2)
        with open(os.path.join(nested2, "file2.txt"), "w") as f:
            f.write("file in nested2")

        # Parallel folder to nested1
        sibling_folder = os.path.join(tmpdir, "sibling")
        os.makedirs(sibling_folder)
        with open(os.path.join(sibling_folder, "sibling_file.txt"), "w") as f:
            f.write("file in sibling folder")

        yield tmpdir


@pytest.mark.asyncio
async def test_transformer_serde():
    f = Dir(path="s3://bucket/files")
    lt = TypeEngine.to_literal_type(Dir)
    lv = await DirTransformer().to_literal(f, Dir, lt)
    pv = await DirTransformer().to_python_value(lv, Dir)
    assert pv == f

    f = Dir(path="s3://bucket/files/")
    lt = TypeEngine.to_literal_type(Dir)
    lv = await DirTransformer().to_literal(f, Dir, lt)
    pv = await DirTransformer().to_python_value(lv, Dir)
    assert pv == f


def test_walk_sync_local(tmp_dir_structure):
    dir_obj = Dir(path=tmp_dir_structure)
    files = list(dir_obj.walk_sync())
    assert len(files) == 4
    assert isinstance(files[0], File)
    assert files[0].path.endswith("root.txt")
    assert os.path.exists(files[0].path)


@pytest.mark.asyncio
async def test_walk_async_local(tmp_dir_structure):
    dir_obj = Dir(path=tmp_dir_structure)
    files = [f async for f in dir_obj.walk()]
    assert len(files) == 4
    f = files[0]
    assert isinstance(f, File)
    assert f.path.endswith("root.txt")
    assert os.path.exists(f.path)


@pytest.mark.asyncio
async def test_download_async_local(tmp_dir_structure):
    dest = tempfile.mkdtemp()
    dir_obj = Dir(path=tmp_dir_structure)
    output = await dir_obj.download(dest)
    assert os.path.exists(os.path.join(output, "root.txt"))
    shutil.rmtree(dest)


@pytest.mark.asyncio
async def test_get_file_local(tmp_dir_structure):
    dir_obj = Dir(path=tmp_dir_structure)
    file = await dir_obj.get_file("root.txt")
    assert isinstance(file, File)
    assert os.path.exists(file.path)


@pytest.mark.asyncio
async def test_dir_local_remote(tmp_dir_structure, ctx_with_test_raw_data_path):
    upload_location = await storage.put(tmp_dir_structure, recursive=True)
    d = Dir(path=upload_location)
    files = [f async for f in d.walk()]
    assert len(files) == 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dir_walk_s3(tmp_dir_structure, ctx_with_test_local_s3_stack_raw_data_path):
    flyte.init(storage=storage.S3(endpoint="http://localhost:4566", secret_access_key="minio", access_key_id="minio"))

    upload_location = await storage.put(tmp_dir_structure, recursive=True)
    d = Dir(path=upload_location)
    files = [f async for f in d.walk()]
    assert len(files) == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        await d.download(tmpdir)
        # With the new behavior, contents go directly into the specified path
        root_file = pathlib.Path(tmpdir) / "root.txt"
        assert root_file.exists()
        assert root_file.is_file()


@pytest.mark.asyncio
async def test_transformer_serde_with_hash():
    """
    Test that the DirTransformer correctly serializes and deserializes Dir objects with hash values.
    """
    d = Dir.from_existing_remote("s3://bucket/data/", dir_cache_key="abc123")
    lt = TypeEngine.to_literal_type(Dir)
    lv = await DirTransformer().to_literal(d, Dir, lt)

    # Hash should be preserved in the literal
    assert lv.hash == "abc123"

    # Convert back to Python
    pv = await DirTransformer().to_python_value(lv, Dir)
    assert pv.path == d.path
    assert pv.hash == "abc123"


@pytest.mark.asyncio
async def test_from_existing_remote_with_hash():
    """
    Test Dir.from_existing_remote with hash functionality.
    """
    # Without hash
    d1 = Dir.from_existing_remote("s3://bucket/data/")
    assert d1.hash is None
    assert d1.path == "s3://bucket/data/"

    # With hash
    d2 = Dir.from_existing_remote("s3://bucket/data/", dir_cache_key="dir_hash_123")
    assert d2.hash == "dir_hash_123"
    assert d2.path == "s3://bucket/data/"


@pytest.mark.asyncio
async def test_from_local_with_precomputed_hash(tmp_dir_structure, ctx_with_test_raw_data_path):
    """
    Test Dir.from_local with PrecomputedValue hash method.
    """
    flyte.init()

    # Test with PrecomputedValue
    d = await Dir.from_local(tmp_dir_structure, dir_cache_key="directory_hash_abc123")

    assert d.hash == "directory_hash_abc123"
    assert d.path is not None
    assert d.name is not None


@pytest.mark.asyncio
async def test_multiple_dirs_with_hashes():
    """
    Test handling multiple Dir objects with different hash scenarios.
    """
    # Create multiple dirs with different hash scenarios
    dir_with_hash = Dir.from_existing_remote("s3://bucket/dir1/", dir_cache_key="hash1")
    dir_without_hash = Dir.from_existing_remote("s3://bucket/dir2/")

    dirs = [dir_with_hash, dir_without_hash]

    # Convert to literals
    transformer = DirTransformer()
    lt = TypeEngine.to_literal_type(Dir)

    literals = []
    for d in dirs:
        lv = await transformer.to_literal(d, Dir, lt)
        literals.append(lv)

    # First dir should have hash, second should not
    assert literals[0].hash == "hash1"
    assert not literals[1].hash

    # Convert back to Python objects
    recovered_dirs = []
    for lv in literals:
        pv = await transformer.to_python_value(lv, Dir)
        recovered_dirs.append(pv)

    # Verify all properties are preserved
    assert recovered_dirs[0].path == dir_with_hash.path
    assert recovered_dirs[0].hash == "hash1"
    assert recovered_dirs[1].path == dir_without_hash.path
    assert recovered_dirs[1].hash is None


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_dir_with_name(tmp_path, tmp_dir_structure, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Test downloading a directory from S3 to a local path with a specified directory name.
    """
    from flyte.storage import S3

    await flyte.init.aio(storage=S3.for_sandbox())

    # Upload to S3
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Download to a specific local path with a custom name
    download_target = tmp_path / "downloaded" / "my_custom_dirname"
    download_target.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_dir.download(str(download_target))

    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory was downloaded to the custom name
    assert downloaded_path == str(download_target)
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_dir_with_folder_name(tmp_path, tmp_dir_structure, ctx_with_test_local_s3_stack_raw_data_path):
    """
    Test downloading a directory to a directory path.
    When a directory path is provided, the directory contents should be downloaded directly into that path.
    """
    from flyte.storage import S3

    await flyte.init.aio(storage=S3.for_sandbox())

    # Upload to S3
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Test 1: Download to an existing directory
    download_dir = tmp_path / "downloaded_existing"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_dir.download(str(download_dir))

    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory contents were downloaded directly into the target directory
    assert downloaded_path == str(download_dir)
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))

    # Test 2: Download to a non-existent path ending with os.sep
    download_dir2_str = str(tmp_path / "downloaded_new") + os.sep  # Ends with separator
    downloaded_path2 = await uploaded_dir.download(download_dir2_str)

    print(f"Downloaded directory to {downloaded_path2}", flush=True)

    # Verify the directory contents were downloaded directly into the target directory
    expected_path2 = tmp_path / "downloaded_new"
    assert downloaded_path2 == str(expected_path2) + os.sep  # Trailing separator is preserved
    assert os.path.isdir(str(expected_path2))
    assert os.path.exists(os.path.join(str(expected_path2), "root.txt"))
    assert os.path.isdir(os.path.join(expected_path2, "sibling"))


@pytest.mark.sandbox
@pytest.mark.asyncio
async def test_download_dir_with_no_local_target(
    tmp_path, tmp_dir_structure, ctx_with_test_local_s3_stack_raw_data_path
):
    """
    Test downloading a directory from S3 without specifying a target path.
    The directory should be downloaded to a temporary location.
    """
    from flyte.storage import S3

    await flyte.init.aio(storage=S3.for_sandbox())

    # Upload to S3
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Download without specifying a target path
    downloaded_path = await uploaded_dir.download()

    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory was downloaded
    assert downloaded_path is not None
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))
    assert os.path.isdir(os.path.join(downloaded_path, "sibling"))
    suffix = uploaded_dir.path.split("/")[-1]
    assert downloaded_path.endswith(suffix)


@pytest.mark.asyncio
async def test_download_dir_with_name_local(tmp_path, tmp_dir_structure, ctx_with_test_raw_data_path):
    """
    Test downloading a directory from local storage to a local path with a specified directory name.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Upload to "remote" (which is actually local)
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Download to a specific local path with a custom name
    download_target = tmp_path / "downloaded" / "my_custom_dirname"
    download_target.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_dir.download(str(download_target))

    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory was downloaded to the custom name
    assert downloaded_path == str(download_target)
    assert os.path.isdir(os.path.join(download_target, "sibling"))
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))


@pytest.mark.asyncio
async def test_download_dir_with_folder_name_local(tmp_path, tmp_dir_structure, ctx_with_test_raw_data_path):
    """
    Test downloading a directory to a directory path using local storage.
    When a directory path is provided, the directory contents should be downloaded directly into that path.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Upload to "remote" (which is actually local)
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Test 1: Download to an existing directory
    download_dir = tmp_path / "downloaded_existing"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = await uploaded_dir.download(str(download_dir))

    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory contents were downloaded directly into the target directory
    assert downloaded_path == str(download_dir)
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))

    # Test 2: Download to a non-existent path ending with os.sep
    download_dir2_str = str(tmp_path / "downloaded_new") + os.sep  # Ends with separator
    downloaded_path2 = await uploaded_dir.download(download_dir2_str)

    print(f"Downloaded directory to {downloaded_path2}", flush=True)

    # Verify the directory contents were downloaded directly into the target directory
    expected_path2 = tmp_path / "downloaded_new"
    assert downloaded_path2 == str(expected_path2) + os.sep  # Trailing separator is preserved
    assert os.path.isdir(str(expected_path2))
    assert os.path.exists(os.path.join(str(expected_path2), "root.txt"))
    assert os.path.isdir(os.path.join(expected_path2, "sibling"))


@pytest.mark.asyncio
async def test_download_dir_with_no_local_target_local(tmp_path, tmp_dir_structure, ctx_with_test_raw_data_path):
    """
    Test downloading a directory from local storage without specifying a target path.
    This test is separate because the local filesystem doesn't use obstore.
    """
    flyte.init()

    # Upload to "remote" (which is actually local)
    uploaded_dir = await Dir.from_local(tmp_dir_structure)
    print(f"Uploaded directory {tmp_dir_structure} to {uploaded_dir.path}", flush=True)

    # Download without specifying a target path
    downloaded_path = await uploaded_dir.download()
    print(f"Downloaded directory to {downloaded_path}", flush=True)

    # Verify the directory was downloaded
    assert downloaded_path is not None
    assert os.path.isdir(downloaded_path)
    assert os.path.exists(os.path.join(downloaded_path, "root.txt"))
    assert os.path.isdir(os.path.join(downloaded_path, "sibling"))
    suffix = uploaded_dir.path.split(os.sep)[-1]
    assert downloaded_path.endswith(suffix)
