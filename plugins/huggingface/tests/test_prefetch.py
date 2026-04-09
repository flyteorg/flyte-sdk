import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flyteplugins.huggingface._prefetch import (
    HuggingFaceDatasetInfo,
    _download_dataset_to_local,
    _stream_dataset_to_remote,
    _validate_input_name,
    store_hf_dataset_task,
)

# ============================================================================
# Input validation tests
# ============================================================================


def test_validate_input_name_valid():
    _validate_input_name("my-dataset_v1")
    _validate_input_name("IMDB")
    _validate_input_name("v1.0")
    _validate_input_name(None)


def test_validate_input_name_invalid():
    with pytest.raises(ValueError, match="must only contain"):
        _validate_input_name("my dataset")
    with pytest.raises(ValueError, match="must only contain"):
        _validate_input_name("data/set")
    with pytest.raises(ValueError, match="must only contain"):
        _validate_input_name("../etc")


# ============================================================================
# HuggingFaceDatasetInfo tests
# ============================================================================


def test_dataset_info_serialization():
    info = HuggingFaceDatasetInfo(repo="stanfordnlp/imdb", split="train", revision="v1.0")
    dumped = info.model_dump_json()
    restored = HuggingFaceDatasetInfo.model_validate_json(dumped)
    assert restored.repo == "stanfordnlp/imdb"
    assert restored.split == "train"
    assert restored.revision == "v1.0"
    assert restored.name is None


def test_dataset_info_defaults():
    info = HuggingFaceDatasetInfo(repo="squad")
    assert info.name is None
    assert info.split is None
    assert info.revision is None


# ============================================================================
# Streaming tests
# ============================================================================


def _make_mock_hub(parquet_entries, split="train"):
    """Helper: create a mock huggingface_hub module with given parquet entries."""
    mock_hfs = MagicMock()

    def ls_side_effect(path, revision=None, detail=True):
        if path.endswith(split):
            return parquet_entries
        return []

    mock_hfs.ls.side_effect = ls_side_effect

    # For each parquet file, create a readable BytesIO
    def open_side_effect(name, mode="rb", revision=None):
        buf = BytesIO(b"fake-parquet-content")
        buf.name = name
        return buf

    mock_hfs.open.side_effect = open_side_effect

    mock_hub = MagicMock()
    mock_hub.HfFileSystem.return_value = mock_hfs
    return mock_hub


def test_stream_dataset_no_parquet_files():
    mock_hfs = MagicMock()
    mock_hfs.ls.return_value = []

    mock_hub = MagicMock()
    mock_hub.HfFileSystem.return_value = mock_hfs

    mock_fs = MagicMock()

    with (
        patch.dict("sys.modules", {"huggingface_hub": mock_hub}),
        patch("flyte.storage.get_underlying_filesystem", return_value=mock_fs),
    ):
        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            _stream_dataset_to_remote("fake/dataset", None, "train", None, None, "s3://bucket/output")


def test_stream_dataset_single_split():
    entries = [
        {"type": "file", "name": "datasets/org/ds/default/train/0000.parquet"},
        {"type": "file", "name": "datasets/org/ds/default/train/0001.parquet"},
    ]
    mock_hub = _make_mock_hub(entries, split="train")
    mock_fs = MagicMock()
    mock_fs.open.return_value.__enter__ = MagicMock(return_value=BytesIO())
    mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.dict("sys.modules", {"huggingface_hub": mock_hub}),
        patch("flyte.storage.get_underlying_filesystem", return_value=mock_fs),
    ):
        result = _stream_dataset_to_remote("org/ds", None, "train", None, None, "s3://bucket/out")
        assert result == "s3://bucket/out"
        # With a single split, files go directly into the root
        write_calls = [str(c) for c in mock_fs.open.call_args_list]
        assert any("0000.parquet" in c for c in write_calls)
        assert any("0001.parquet" in c for c in write_calls)


def test_stream_dataset_multi_split_preserves_split_dirs():
    """When split=None, parquet files from different splits should go into separate subdirs."""
    mock_hfs = MagicMock()

    def ls_side_effect(path, revision=None, detail=True):
        if path == "datasets/org/ds/default":
            return [
                {"type": "directory", "name": "datasets/org/ds/default/train"},
                {"type": "directory", "name": "datasets/org/ds/default/test"},
            ]
        elif path.endswith("/train"):
            return [{"type": "file", "name": "datasets/org/ds/default/train/0000.parquet"}]
        elif path.endswith("/test"):
            return [{"type": "file", "name": "datasets/org/ds/default/test/0000.parquet"}]
        return []

    mock_hfs.ls.side_effect = ls_side_effect
    mock_hfs.open.side_effect = lambda name, mode="rb", revision=None: BytesIO(b"data")

    mock_hub = MagicMock()
    mock_hub.HfFileSystem.return_value = mock_hfs

    mock_fs = MagicMock()
    mock_fs.open.return_value.__enter__ = MagicMock(return_value=BytesIO())
    mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

    with (
        patch.dict("sys.modules", {"huggingface_hub": mock_hub}),
        patch("flyte.storage.get_underlying_filesystem", return_value=mock_fs),
    ):
        result = _stream_dataset_to_remote("org/ds", None, None, None, None, "s3://bucket/out")
        assert result == "s3://bucket/out"
        # Verify files go into split subdirs (train/0000.parquet, test/0000.parquet)
        open_paths = [str(c) for c in mock_fs.open.call_args_list]
        assert any("train/0000.parquet" in p for p in open_paths)
        assert any("test/0000.parquet" in p for p in open_paths)
        # mkdirs called for each split subdir
        mkdirs_calls = [str(c) for c in mock_fs.mkdirs.call_args_list]
        assert any("train" in c for c in mkdirs_calls)
        assert any("test" in c for c in mkdirs_calls)


# ============================================================================
# Download fallback tests
# ============================================================================


def test_download_dataset_to_local_no_files():
    mock_hub = MagicMock()
    mock_hub.list_repo_files.return_value = ["other/file.txt"]

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as flat_dir:
            with pytest.raises(FileNotFoundError, match="No parquet files found"):
                _download_dataset_to_local("fake/ds", None, "train", None, None, local_dir, flat_dir)


def test_download_dataset_flattens_parquet_files():
    mock_hub = MagicMock()
    mock_hub.list_repo_files.return_value = [
        "default/train/0000.parquet",
        "default/train/0001.parquet",
    ]

    def fake_download(repo_id, filename, repo_type, revision, local_dir, token):
        dest = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Write a minimal parquet file
        table = pa.table({"col": [1]})
        pq.write_table(table, dest)

    mock_hub.hf_hub_download.side_effect = fake_download

    with patch.dict("sys.modules", {"huggingface_hub": mock_hub}):
        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as flat_dir:
            result = _download_dataset_to_local("org/ds", None, "train", None, None, local_dir, flat_dir)
            assert result == flat_dir
            files = os.listdir(flat_dir)
            assert "0000.parquet" in files
            assert "0001.parquet" in files


# ============================================================================
# store_hf_dataset_task tests
# ============================================================================


def test_store_hf_dataset_task_warns_no_token():
    info = HuggingFaceDatasetInfo(repo="org/ds", split="train")

    mock_hub = MagicMock()
    mock_hfs = MagicMock()
    mock_hfs.ls.return_value = [{"type": "file", "name": "datasets/org/ds/default/train/0000.parquet"}]
    mock_hfs.open.return_value = BytesIO(b"data")
    mock_hub.HfFileSystem.return_value = mock_hfs

    mock_fs = MagicMock()
    mock_fs.open.return_value.__enter__ = MagicMock(return_value=BytesIO())
    mock_fs.open.return_value.__exit__ = MagicMock(return_value=False)

    mock_ctx = MagicMock()
    mock_ctx.raw_data_path.get_random_remote_path.return_value = "/tmp/test-output"

    with (
        patch.dict("sys.modules", {"huggingface_hub": mock_hub}),
        patch.dict(os.environ, {}, clear=True),
        patch("flyte.storage.get_underlying_filesystem", return_value=mock_fs),
        patch("flyte.ctx", return_value=mock_ctx),
        patch("flyte.io.Dir.from_existing_remote") as mock_dir,
    ):
        mock_dir.return_value = MagicMock(path="/tmp/test-output")
        result = store_hf_dataset_task(info.model_dump_json())
        assert result.path == "/tmp/test-output"
