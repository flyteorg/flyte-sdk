import os
import tempfile
import typing
from collections import OrderedDict
from pathlib import Path

import mock
import pytest
from flyteidl.core import literals_pb2, types_pb2
from fsspec.utils import get_protocol

import flyte
from flyte._context import Context, RawDataPath, internal_ctx
from flyte._utils.lazy_module import is_imported
from flyte.io._dataframe.dataframe import (
    PARQUET,
    DataFrame,
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
    extract_cols_and_format,
)
from flyte.models import SerializationContext
from flyte.types import TypeEngine

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

my_cols = OrderedDict(w=typing.Dict[str, typing.Dict[str, int]], x=typing.List[typing.List[int]], y=int, z=str)

fields = [("some_int", pa.int32()), ("some_string", pa.string())]
arrow_schema = pa.schema(fields)

serialization_context = SerializationContext(
    version="123",
)
df = pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})


def test_protocol():
    assert get_protocol("s3://my-s3-bucket/file") == "s3"
    assert get_protocol("/file") == "file"


def generate_pandas() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Tom", "Joseph"], "age": [20, 22]})


flyte.init()


@pytest.fixture
def local_tmp_pqt_file():
    df = generate_pandas()

    # Create a temporary parquet file
    with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix=".parquet") as pqt_file:
        pqt_path = pqt_file.name
        df.to_parquet(pqt_path)

    yield pqt_path

    # Cleanup
    Path(pqt_path).unlink(missing_ok=True)


def test_types_pandas():
    pt = pd.DataFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None
    assert lt.structured_dataset_type.format == ""
    assert lt.structured_dataset_type.columns == []

    pt = typing.Annotated[pd.DataFrame, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type.format == "csv"


def test_annotate_extraction():
    xyz = typing.Annotated[pd.DataFrame, "myformat"]
    a, b, c, d = extract_cols_and_format(xyz)
    assert a is pd.DataFrame
    assert b is None
    assert c == "myformat"
    assert d is None

    a, b, c, d = extract_cols_and_format(pd.DataFrame)
    assert a is pd.DataFrame
    assert b is None
    assert c == ""
    assert d is None


def test_types_annotated():
    pt = typing.Annotated[pd.DataFrame, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4
    assert (
        lt.structured_dataset_type.columns[0].literal_type.map_value_type.map_value_type.simple
        == types_pb2.SimpleType.INTEGER
    )
    assert (
        lt.structured_dataset_type.columns[1].literal_type.collection_type.collection_type.simple
        == types_pb2.SimpleType.INTEGER
    )
    assert lt.structured_dataset_type.columns[2].literal_type.simple == types_pb2.SimpleType.INTEGER
    assert lt.structured_dataset_type.columns[3].literal_type.simple == types_pb2.SimpleType.STRING

    pt = typing.Annotated[pd.DataFrame, PARQUET, arrow_schema]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type.external_schema_type == "arrow"
    assert "some_string" in str(lt.structured_dataset_type.external_schema_bytes)

    pt = typing.Annotated[pd.DataFrame, OrderedDict(a=None)]
    with pytest.raises(AssertionError, match="type None is currently not supported by DataFrame"):
        TypeEngine.to_literal_type(pt)


def test_types_sd():
    pt = DataFrame
    lt = TypeEngine.to_literal_type(pt)
    assert lt.structured_dataset_type is not None

    pt = typing.Annotated[DataFrame, my_cols]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4

    pt = typing.Annotated[DataFrame, my_cols, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 4
    assert lt.structured_dataset_type.format == "csv"

    pt = typing.Annotated[DataFrame, {}, "csv"]
    lt = TypeEngine.to_literal_type(pt)
    assert len(lt.structured_dataset_type.columns) == 0
    assert lt.structured_dataset_type.format == "csv"


class MyDF(pd.DataFrame): ...


def test_retrieving():
    assert DataFrameTransformerEngine.get_encoder(pd.DataFrame, "file", PARQUET) is not None
    # Asking for a generic means you're okay with any one registered for that
    # type assuming there's just one.
    assert DataFrameTransformerEngine.get_encoder(pd.DataFrame, "file", "") is DataFrameTransformerEngine.get_encoder(
        pd.DataFrame, "file", PARQUET
    )

    class TempEncoder(DataFrameEncoder):
        def __init__(self, protocol):
            super().__init__(MyDF, protocol)

        def encode(self): ...

    DataFrameTransformerEngine.register(TempEncoder("gs"), default_for_type=False)
    with pytest.raises(ValueError):
        DataFrameTransformerEngine.register(TempEncoder("gs://"), default_for_type=False)

    with pytest.raises(ValueError, match="Use None instead"):
        e = TempEncoder("")
        e._protocol = ""
        DataFrameTransformerEngine.register(e)

    class TempEncoder:
        pass

    with pytest.raises(TypeError, match="We don't support this type of handler"):
        DataFrameTransformerEngine.register(TempEncoder, default_for_type=False)


@pytest.mark.asyncio
async def test_to_literal(ctx_with_test_raw_data_path):
    lt = TypeEngine.to_literal_type(pd.DataFrame)
    df = generate_pandas()

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(df, python_type=pd.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == PARQUET

    sd_with_literal_and_df = DataFrame.create_from(val=df)
    sd_with_literal_and_df._literal_sd = lit

    with pytest.raises(ValueError, match="Shouldn't have specified both literal"):
        await fdt.to_literal(sd_with_literal_and_df, python_type=DataFrame, expected=lt)

    sd_with_nothing = DataFrame()
    with pytest.raises(ValueError, match="If dataframe is not specified"):
        await fdt.to_literal(sd_with_nothing, python_type=DataFrame, expected=lt)

    sd_with_uri = DataFrame.from_existing_remote(remote_path="s3://some/extant/df.parquet")

    lt = TypeEngine.to_literal_type(typing.Annotated[DataFrame, {}, "new-df-format"])
    lit = await fdt.to_literal(sd_with_uri, python_type=DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.uri == "s3://some/extant/df.parquet"
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == "new-df-format"


@pytest.mark.asyncio
async def test_to_literal_through_df_with_format(ctx_with_test_raw_data_path):
    lt = TypeEngine.to_literal_type(typing.Annotated[pd.DataFrame, "csv"])
    df = generate_pandas()
    fdf = DataFrame.create_from(val=df)

    fdt = DataFrameTransformerEngine()

    lit = await fdt.to_literal(fdf, python_type=pd.DataFrame, expected=lt)
    assert lit.scalar.structured_dataset.metadata.structured_dataset_type.format == "csv"

    # go backwards to get a python value
    py_val = await fdt.to_python_value(lit, expected_python_type=DataFrame)
    restored_df = await py_val.open(pd.DataFrame).all()

    restored_df_2 = await fdt.to_python_value(lit, expected_python_type=pd.DataFrame)
    assert restored_df.equals(restored_df_2)
    assert restored_df.equals(df)

