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
        # grab the name of the folder from the upload_location using Path even though it's s3.
        folder_name = pathlib.Path(upload_location).name
        root_file = pathlib.Path(tmpdir) / folder_name / "root.txt"
        assert root_file.exists()
        assert root_file.is_file()
