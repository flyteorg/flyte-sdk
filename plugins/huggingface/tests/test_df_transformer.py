import typing
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import flyte
import pytest
from flyte.io import DataFrame
from flyte.io._dataframe.dataframe import PARQUET, DataFrameTransformerEngine
from flyte.types import TypeEngine
from flyteidl2.core import literals_pb2

from flyteplugins.huggingface import HFSource, from_hf
from flyteplugins.huggingface.dataset._io import (
    HFParquetError,
    _resolve_hf_config,
    join_uri_path,
    list_parquet_files,
)
from flyteplugins.huggingface.dataset._transformers import (
    HFToHuggingFaceDatasetDecodingHandler,
    HFToHuggingFaceIterableDatasetDecodingHandler,
    HuggingFaceDatasetToParquetEncodingHandler,
    HuggingFaceIterableDatasetToParquetEncodingHandler,
    ParquetToHuggingFaceDatasetDecodingHandler,
    ParquetToHuggingFaceIterableDatasetDecodingHandler,
    _ROWS_PER_SHARD,
)

datasets = pytest.importorskip("datasets")

TEST_DATA = {
    "text": ["hello world", "flyte rocks", "hugging face datasets"],
    "label": [0, 1, 0],
}


@pytest.fixture
def sample_dataset():
    return datasets.Dataset.from_dict(TEST_DATA)


# ============================================================================
# HF source references
# ============================================================================


def test_from_hf_returns_dataframe_reference():
    ref = from_hf("stanfordnlp/imdb", split="train")

    assert isinstance(ref, DataFrame)
    assert ref.uri == "hf://stanfordnlp/imdb?split=train"
    assert ref.format == PARQUET
    assert ref.hash is not None


def test_hfsource_to_hf_uri_with_all_fields():
    source = HFSource(
        repo="stanfordnlp/imdb",
        name="plain_text",
        split="train",
        revision="refs/convert/parquet",
        cache_root="s3://bucket/flyte-hf-cache",
    )
    assert (
        source.to_hf_uri() == "hf://stanfordnlp/imdb?name=plain_text&split=train"
        "&cache_root=s3://bucket/flyte-hf-cache&revision=refs/convert/parquet"
    )


def test_hfsource_to_hf_uri_omits_name_and_split():
    source = HFSource(repo="stanfordnlp/imdb")
    assert source.to_hf_uri() == "hf://stanfordnlp/imdb"


def test_hfsource_uri_roundtrip():
    for source in [
        HFSource(repo="stanfordnlp/imdb"),
        HFSource(repo="stanfordnlp/imdb", split="train"),
        HFSource(
            repo="stanfordnlp/imdb",
            name="plain_text",
            split="test",
            cache_root="s3://bucket/cache",
            revision="refs/convert/parquet",
        ),
        HFSource(repo="glue", name="mrpc", split="train"),
    ]:
        assert HFSource.from_hf_uri(source.to_hf_uri()) == source


# ============================================================================
# Config resolution
# ============================================================================


def test_resolve_hf_config_prefers_explicit_name():
    hfs = MagicMock()
    hfs.ls.side_effect = lambda path, revision=None, detail=True: {
        "datasets/glue/mrpc": [
            {"type": "directory", "name": "datasets/glue/mrpc/train"},
        ]
    }[path]

    name, path, entries = _resolve_hf_config(
        hfs, HFSource(repo="glue", name="mrpc"), "refs/convert/parquet"
    )

    assert name == "mrpc"
    assert path == "datasets/glue/mrpc"
    assert entries == [{"type": "directory", "name": "datasets/glue/mrpc/train"}]


def test_resolve_hf_config_uses_default_when_available():
    hfs = MagicMock()

    def ls(path, revision=None, detail=True):
        if path == "datasets/org/ds/default":
            return [{"type": "directory", "name": "datasets/org/ds/default/train"}]
        raise FileNotFoundError(path)

    hfs.ls.side_effect = ls

    name, path, entries = _resolve_hf_config(
        hfs, HFSource(repo="org/ds"), "refs/convert/parquet"
    )

    assert name == "default"
    assert path == "datasets/org/ds/default"
    assert entries == [{"type": "directory", "name": "datasets/org/ds/default/train"}]


def test_resolve_hf_config_falls_back_to_only_available_config():
    hfs = MagicMock()

    def ls(path, revision=None, detail=True):
        if path == "datasets/stanfordnlp/imdb/default":
            raise FileNotFoundError(path)
        if path == "datasets/stanfordnlp/imdb":
            return [
                {"type": "directory", "name": "datasets/stanfordnlp/imdb/plain_text"}
            ]
        if path == "datasets/stanfordnlp/imdb/plain_text":
            return [
                {
                    "type": "directory",
                    "name": "datasets/stanfordnlp/imdb/plain_text/train",
                }
            ]
        raise FileNotFoundError(path)

    hfs.ls.side_effect = ls

    name, path, entries = _resolve_hf_config(
        hfs, HFSource(repo="stanfordnlp/imdb"), "refs/convert/parquet"
    )

    assert name == "plain_text"
    assert path == "datasets/stanfordnlp/imdb/plain_text"
    assert entries == [
        {"type": "directory", "name": "datasets/stanfordnlp/imdb/plain_text/train"}
    ]


def test_resolve_hf_config_requires_name_when_multiple_configs_exist():
    hfs = MagicMock()

    def ls(path, revision=None, detail=True):
        if path == "datasets/glue/default":
            raise FileNotFoundError(path)
        if path == "datasets/glue":
            return [
                {"type": "directory", "name": "datasets/glue/mrpc"},
                {"type": "directory", "name": "datasets/glue/sst2"},
            ]
        raise FileNotFoundError(path)

    hfs.ls.side_effect = ls

    with pytest.raises(HFParquetError, match="multiple parquet configs"):
        _resolve_hf_config(hfs, HFSource(repo="glue"), "refs/convert/parquet")


# ============================================================================
# Registration and handler properties
# ============================================================================


def test_registered_dataset_handlers():
    assert isinstance(
        DataFrameTransformerEngine.get_encoder(datasets.Dataset, "file", PARQUET),
        HuggingFaceDatasetToParquetEncodingHandler,
    )
    assert isinstance(
        DataFrameTransformerEngine.get_decoder(datasets.Dataset, "file", PARQUET),
        ParquetToHuggingFaceDatasetDecodingHandler,
    )
    assert isinstance(
        DataFrameTransformerEngine.get_decoder(datasets.Dataset, "hf", PARQUET),
        HFToHuggingFaceDatasetDecodingHandler,
    )


def test_registered_iterable_handlers():
    assert isinstance(
        DataFrameTransformerEngine.get_encoder(
            datasets.IterableDataset, "file", PARQUET
        ),
        HuggingFaceIterableDatasetToParquetEncodingHandler,
    )
    assert isinstance(
        DataFrameTransformerEngine.get_decoder(
            datasets.IterableDataset, "file", PARQUET
        ),
        ParquetToHuggingFaceIterableDatasetDecodingHandler,
    )
    assert isinstance(
        DataFrameTransformerEngine.get_decoder(datasets.IterableDataset, "hf", PARQUET),
        HFToHuggingFaceIterableDatasetDecodingHandler,
    )


# ============================================================================
# datasets.Dataset roundtrip
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_roundtrip(ctx_with_test_raw_data_path, sample_dataset):
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    lit = await fdt.to_literal(sample_dataset, python_type=datasets.Dataset, expected=lt)
    assert (
        lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    )

    restored = await fdt.to_python_value(lit, expected_python_type=datasets.Dataset)
    assert isinstance(restored, datasets.Dataset)
    assert len(restored) == len(sample_dataset)
    assert restored.column_names == sample_dataset.column_names
    assert restored["text"] == sample_dataset["text"]


@pytest.mark.asyncio
async def test_dataset_column_projection(ctx_with_test_raw_data_path, sample_dataset):
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)
    lit = await fdt.to_literal(
        sample_dataset, python_type=datasets.Dataset, expected=lt
    )

    columns = OrderedDict(text=str)
    restored = await fdt.to_python_value(
        lit,
        expected_python_type=typing.Annotated[datasets.Dataset, columns],
    )

    assert isinstance(restored, datasets.Dataset)
    assert restored.column_names == ["text"]


@pytest.mark.asyncio
async def test_dataset_passthrough_task(sample_dataset):
    flyte.init()
    env = flyte.TaskEnvironment(name="hf-dataset-test")

    @env.task
    async def select_two(ds: datasets.Dataset) -> datasets.Dataset:
        return ds.select(range(2))

    run = flyte.with_runcontext("local").run(select_two, sample_dataset)
    result = run.outputs()[0]

    assert isinstance(result, datasets.Dataset)
    assert len(result) == 2


# ============================================================================
# hf:// source literals
# ============================================================================


@pytest.mark.asyncio
async def test_encode_from_hf_reference_as_dataset_emits_hf_uri(
    ctx_with_test_raw_data_path,
):
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)

    ref = from_hf(
        "stanfordnlp/imdb",
        name="plain_text",
        split="train",
        cache_root="s3://bucket/cache",
    )
    lit = await fdt.to_literal(ref, python_type=datasets.Dataset, expected=lt)

    assert (
        lit.scalar.structured_dataset.uri
        == "hf://stanfordnlp/imdb?name=plain_text&split=train&cache_root=s3://bucket/cache"
    )
    assert (
        lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    )


@pytest.mark.asyncio
async def test_encode_from_hf_reference_as_iterable_dataset_emits_hf_uri(
    ctx_with_test_raw_data_path,
):
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.IterableDataset)

    ref = from_hf("stanfordnlp/imdb", split="train")
    lit = await fdt.to_literal(ref, python_type=datasets.IterableDataset, expected=lt)

    assert lit.scalar.structured_dataset.uri == "hf://stanfordnlp/imdb?split=train"


@pytest.mark.asyncio
async def test_decode_hf_uri_as_dataset_uses_cache_materialization(
    ctx_with_test_raw_data_path, sample_dataset
):
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.Dataset)
    cached_lit = await fdt.to_literal(
        sample_dataset, python_type=datasets.Dataset, expected=lt
    )
    cached_uri = cached_lit.scalar.structured_dataset.uri

    hf_lit = literals_pb2.Literal(
        scalar=literals_pb2.Scalar(
            structured_dataset=literals_pb2.StructuredDataset(
                uri="hf://stanfordnlp/imdb?split=train",
                metadata=cached_lit.scalar.structured_dataset.metadata,
            )
        )
    )

    with patch(
        "flyteplugins.huggingface.dataset._transformers.ensure_hf_cached",
        new=AsyncMock(return_value=cached_uri),
    ) as ensure_cached:
        restored = await fdt.to_python_value(
            hf_lit, expected_python_type=datasets.Dataset
        )

    ensure_cached.assert_awaited_once_with(
        HFSource(repo="stanfordnlp/imdb", split="train")
    )
    assert isinstance(restored, datasets.Dataset)
    assert len(restored) == len(sample_dataset)


@pytest.mark.asyncio
async def test_decode_hf_uri_as_iterable_dataset_uses_cache_materialization(
    ctx_with_test_raw_data_path, sample_dataset
):
    fdt = DataFrameTransformerEngine()
    cached_lit = await fdt.to_literal(
        sample_dataset,
        python_type=datasets.Dataset,
        expected=TypeEngine.to_literal_type(datasets.Dataset),
    )
    cached_uri = cached_lit.scalar.structured_dataset.uri

    hf_lit = literals_pb2.Literal(
        scalar=literals_pb2.Scalar(
            structured_dataset=literals_pb2.StructuredDataset(
                uri="hf://stanfordnlp/imdb?name=plain_text&split=test",
                metadata=cached_lit.scalar.structured_dataset.metadata,
            )
        )
    )

    with patch(
        "flyteplugins.huggingface.dataset._transformers.ensure_hf_cached",
        new=AsyncMock(return_value=cached_uri),
    ) as ensure_cached:
        restored = await fdt.to_python_value(
            hf_lit, expected_python_type=datasets.IterableDataset
        )

    ensure_cached.assert_awaited_once_with(
        HFSource(repo="stanfordnlp/imdb", name="plain_text", split="test")
    )
    assert isinstance(restored, datasets.IterableDataset)
    assert len(list(restored)) == len(sample_dataset)


# ============================================================================
# datasets.IterableDataset roundtrip
# ============================================================================


@pytest.mark.asyncio
async def test_iterable_dataset_roundtrip(ctx_with_test_raw_data_path, sample_dataset):
    iterable_ds = sample_dataset.to_iterable_dataset()
    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.IterableDataset)

    lit = await fdt.to_literal(
        iterable_ds, python_type=datasets.IterableDataset, expected=lt
    )
    restored = await fdt.to_python_value(
        lit, expected_python_type=datasets.IterableDataset
    )

    assert isinstance(restored, datasets.IterableDataset)
    rows = list(restored)
    assert len(rows) == len(sample_dataset)
    assert set(rows[0].keys()) == set(sample_dataset.column_names)


@pytest.mark.asyncio
async def test_iterable_dataset_large_sharding(ctx_with_test_raw_data_path):
    big_ds = datasets.Dataset.from_dict({"x": list(range(_ROWS_PER_SHARD + 1))})
    iterable_ds = big_ds.to_iterable_dataset()

    fdt = DataFrameTransformerEngine()
    lt = TypeEngine.to_literal_type(datasets.IterableDataset)

    lit = await fdt.to_literal(
        iterable_ds, python_type=datasets.IterableDataset, expected=lt
    )
    uri = lit.scalar.structured_dataset.uri

    import flyte.storage as storage

    fs = storage.get_underlying_filesystem(path=uri)
    files = await list_parquet_files(uri, fs)

    assert len(files) == 2

    restored = await fdt.to_python_value(
        lit, expected_python_type=datasets.IterableDataset
    )
    assert len(list(restored)) == _ROWS_PER_SHARD + 1


@pytest.mark.asyncio
async def test_iterable_dataset_passthrough_task(sample_dataset):
    flyte.init()
    env = flyte.TaskEnvironment(name="hf-iterable-test")

    @env.task
    async def passthrough(ds: datasets.IterableDataset) -> datasets.IterableDataset:
        return ds

    iterable_ds = sample_dataset.to_iterable_dataset()
    run = flyte.with_runcontext("local").run(passthrough, iterable_ds)
    result = run.outputs()[0]

    assert isinstance(result, datasets.IterableDataset)
    assert len(list(result)) == len(sample_dataset)


# ============================================================================
# Helpers
# ============================================================================


def test_join_uri_path_preserves_uri_scheme():
    assert join_uri_path("s3://bucket/root/", "/a/", "b") == "s3://bucket/root/a/b"
    assert (
        join_uri_path("hf://stanfordnlp/imdb", "train") == "hf://stanfordnlp/imdb/train"
    )
    assert (
        join_uri_path("datasets", "stanfordnlp/imdb", "plain_text")
        == "datasets/stanfordnlp/imdb/plain_text"
    )
