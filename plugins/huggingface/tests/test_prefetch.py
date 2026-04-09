from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.huggingface._prefetch import (
    HuggingFaceDatasetInfo,
    _stream_dataset_to_remote,
    _validate_input_name,
)


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
