import os
import sys
import tempfile
import typing
from collections import OrderedDict
from pathlib import Path

import mock
import pytest
from flyteidl2.core import literals_pb2, types_pb2
from fsspec.utils import get_protocol

import flyte
from flyte._context import Context, RawDataPath, internal_ctx
from flyte._utils.lazy_module import is_imported
from flyte.io import (
    PARQUET,
    DataFrame,
)
from flyte.io.extend import (
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
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


@pytest.mark.asyncio
async def test_aio_running(ctx_with_test_raw_data_path):
    from flyte._internal.runtime import task_serde

    env = flyte.TaskEnvironment(name="test-sd-formats")

    @env.task
    async def t1(a: pd.DataFrame) -> pd.DataFrame:
        print(a)
        return generate_pandas()

    await flyte.init.aio()
    result = await flyte.run.aio(t1, a=generate_pandas())
    assert isinstance(result.outputs(), pd.DataFrame)
    result = flyte.run(t1, a=generate_pandas())
    assert isinstance(result.outputs(), pd.DataFrame)

    # this should be an empty string format
    flyte_interface = task_serde.transform_native_to_typed_interface(t1.native_interface)
    assert flyte_interface.outputs.variables["o0"].type.structured_dataset_type.format == ""
    assert flyte_interface.inputs.variables["a"].type.structured_dataset_type.format == ""


@pytest.mark.asyncio
async def test_setting_of_unset_formats():
    custom = typing.Annotated[DataFrame, "parquet"]
    example = custom.from_df(val=df, uri="/path")
    # It's okay that the annotation is not used here yet.
    assert example.format == ""

    env = flyte.TaskEnvironment(name="hello_world-test-sd")

    @env.task
    async def t2(path: str) -> DataFrame:
        sd = DataFrame.from_df(val=df, uri=path)
        return sd

    @env.task
    async def wf(path: str) -> DataFrame:
        return await t2(path=path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fname = os.path.join(tmp_dir, "somewhere")
        res = await wf(path=fname)
        # This should be empty because we're not doing flyte.run, which means the encoder/decoder isn't getting
        # called at all.
        assert res.format == ""
        res = flyte.run(wf, path=fname)
        # Now that it's passed through an encoder however, it should be set.
        assert res.outputs().format == "parquet"


class MyDF(pd.DataFrame): ...


@pytest.mark.asyncio
async def test_fill_in_literal_type():
    class TempEncoder(DataFrameEncoder):
        def __init__(self, fmt: str):
            super().__init__(MyDF, "tmpfs://", supported_format=fmt)

        async def encode(
            self,
            structured_dataset: DataFrame,
            structured_dataset_type: types_pb2.StructuredDatasetType,
        ) -> literals_pb2.StructuredDataset:
            return literals_pb2.StructuredDataset(uri="")

    default_encoder = TempEncoder("myavro")
    DataFrameTransformerEngine.register(default_encoder, default_for_type=True)
    lt = TypeEngine.to_literal_type(MyDF)
    assert lt.structured_dataset_type.format == ""

    fdt = DataFrameTransformerEngine()
    sd = DataFrame.from_df(val=MyDF())
    literal = await fdt.to_literal(sd, MyDF, lt)
    # Test that the literal type is filled in even though the encode function
    # above doesn't do it.
    assert literal.scalar.structured_dataset.metadata.structured_dataset_type.format == "myavro"

    # Test that looking up encoders/decoders falls back to the ""
    # encoder/decoder
    empty_format_temp_encoder = TempEncoder("")
    DataFrameTransformerEngine.register(empty_format_temp_encoder, default_for_type=False)

    res = DataFrameTransformerEngine.get_encoder(MyDF, "tmpfs", "rando")
    assert res is empty_format_temp_encoder


def test_slash_register():
    class TempEncoder(DataFrameEncoder):
        def __init__(self, fmt: str):
            super().__init__(MyDF, None, supported_format=fmt)

        async def encode(
            self,
            ctx: Context,
            structured_dataset: DataFrame,
            structured_dataset_type: types_pb2.StructuredDatasetType,
        ) -> literals_pb2.StructuredDataset:
            return literals_pb2.StructuredDataset(uri="")

    # Check that registering with a / triggers the file protocol instead.
    DataFrameTransformerEngine.register(TempEncoder("/"))
    res = DataFrameTransformerEngine.get_encoder(MyDF, "file", "/")
    # Test that the one we got was registered under fsspec
    assert res is DataFrameTransformerEngine.ENCODERS[MyDF].get("fsspec")["/"]
    assert res is not None


@pytest.mark.asyncio
async def test_sd():
    sd = DataFrame.from_df(val="hi")
    sd.uri = "my uri"
    assert sd.format == ""

    with pytest.raises(ValueError, match="No dataframe type set"):
        await sd.all()

    with pytest.raises(ValueError, match=r"No dataframe type set\."):
        await sd.iter()

    class MockPandasDecodingHandlers(DataFrameDecoder):
        async def decode(
            self,
            flyte_value: literals_pb2.StructuredDataset,
            current_task_metadata: literals_pb2.StructuredDatasetMetadata,
        ) -> typing.Union[typing.Generator[pd.DataFrame, None, None]]:
            yield pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})

    DataFrameTransformerEngine.register(MockPandasDecodingHandlers(pd.DataFrame, "tmpfs"), default_for_type=False)
    sd = DataFrame()
    sd._literal_sd = literals_pb2.StructuredDataset(
        uri="tmpfs://somewhere",
        metadata=literals_pb2.StructuredDatasetMetadata(
            structured_dataset_type=types_pb2.StructuredDatasetType(format="")
        ),
    )
    assert isinstance(await sd.open(pd.DataFrame).iter(), typing.AsyncGenerator)

    if sys.version_info < (3, 14):
        with pytest.raises(TypeError, match="object async_generator can't be used"):
            await sd.open(pd.DataFrame).all()
    else:
        with pytest.raises(TypeError, match="'async_generator' object can't be awaited"):
            await sd.open(pd.DataFrame).all()

    class MockPandasDecodingHandlers(DataFrameDecoder):
        async def decode(
            self,
            flyte_value: literals_pb2.StructuredDataset,
            current_task_metadata: literals_pb2.StructuredDatasetMetadata,
        ) -> pd.DataFrame:
            return pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})

    DataFrameTransformerEngine.register(
        MockPandasDecodingHandlers(pd.DataFrame, "tmpfs"), default_for_type=False, override=True
    )
    sd = DataFrame()
    sd._literal_sd = literals_pb2.StructuredDataset(
        uri="tmpfs://somewhere",
        metadata=literals_pb2.StructuredDatasetMetadata(
            structured_dataset_type=types_pb2.StructuredDatasetType(format="")
        ),
    )

    with pytest.raises(ValueError):
        await sd.open(pd.DataFrame).iter()


@pytest.mark.asyncio
async def test_to_python_value_with_incoming_columns(ctx_with_test_raw_data_path):
    # make a literal with a type that has two columns
    original_type = typing.Annotated[pd.DataFrame, OrderedDict(name=str, age=int)]
    lt = TypeEngine.to_literal_type(original_type)
    df = generate_pandas()
    fdt = DataFrameTransformerEngine()
    lit = await fdt.to_literal(df, python_type=original_type, expected=lt)
    assert len(lit.scalar.structured_dataset.metadata.structured_dataset_type.columns) == 2

    # declare a new type that only has one column
    # get the dataframe, make sure it has the column that was asked for.
    subset_sd_type = typing.Annotated[DataFrame, OrderedDict(age=int)]
    sd = await fdt.to_python_value(lit, subset_sd_type)
    assert sd.metadata.structured_dataset_type.columns[0].name == "age"
    sub_df = await sd.open(pd.DataFrame).all()
    assert sub_df.shape[1] == 1

    # check when columns are not specified, should pull both and add column
    # information.
    sd = await fdt.to_python_value(lit, DataFrame)
    assert len(sd.metadata.structured_dataset_type.columns) == 2

    # should also work if subset type is just an annotated pd.DataFrame
    subset_pd_type = typing.Annotated[pd.DataFrame, OrderedDict(age=int)]
    sub_df = await fdt.to_python_value(lit, subset_pd_type)
    assert sub_df.shape[1] == 1


@pytest.mark.asyncio
async def test_to_python_value_without_incoming_columns(ctx_with_test_raw_data_path):
    # make a literal with a type with no columns
    lt = TypeEngine.to_literal_type(pd.DataFrame)
    df = generate_pandas()
    fdt = DataFrameTransformerEngine()
    lit = await fdt.to_literal(df, python_type=pd.DataFrame, expected=lt)
    assert len(lit.scalar.structured_dataset.metadata.structured_dataset_type.columns) == 0

    # declare a new type that only has one column
    # get the dataframe, make sure it has the column that was asked for.
    subset_sd_type = typing.Annotated[DataFrame, OrderedDict(age=int)]
    sd = await fdt.to_python_value(lit, subset_sd_type)
    assert sd.metadata.structured_dataset_type.columns[0].name == "age"
    sub_df = await sd.open(pd.DataFrame).all()
    assert sub_df.shape[1] == 1

    # check when columns are not specified, should pull both and add column information.
    # todo: see the todos in the open_as, and iter_as functions in DataFrameTransformerEngine
    # we have to recreate the literal because the test case above filled in
    # the metadata
    lit = await fdt.to_literal(df, python_type=pd.DataFrame, expected=lt)
    sd = await fdt.to_python_value(lit, DataFrame)
    assert sd.metadata.structured_dataset_type.columns == []
    sub_df = await sd.open(pd.DataFrame).all()
    assert sub_df.shape[1] == 2

    # should also work if subset type is just an annotated pd.DataFrame
    lit = await fdt.to_literal(df, python_type=pd.DataFrame, expected=lt)
    subset_pd_type = typing.Annotated[pd.DataFrame, OrderedDict(age=int)]
    sub_df = await fdt.to_python_value(lit, subset_pd_type)
    assert sub_df.shape[1] == 1


@pytest.mark.asyncio
async def test_format_correct(ctx_with_test_raw_data_path):
    class TempEncoder(DataFrameEncoder):
        def __init__(self):
            super().__init__(pd.DataFrame, "/", "avro")

        async def encode(
            self,
            dataframe: DataFrame,
            structured_dataset_type: types_pb2.StructuredDatasetType,
        ) -> literals_pb2.StructuredDataset:
            return literals_pb2.StructuredDataset(
                uri="/tmp/avro",
                metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
            )

    df = pd.DataFrame({"name": ["Tom", "Joseph"], "age": [20, 22]})

    annotated_sd_type = typing.Annotated[DataFrame, "avro", OrderedDict(name=str, age=int)]
    df_literal_type = TypeEngine.to_literal_type(annotated_sd_type)
    assert df_literal_type.structured_dataset_type is not None
    assert len(df_literal_type.structured_dataset_type.columns) == 2
    assert df_literal_type.structured_dataset_type.columns[0].name == "name"
    assert df_literal_type.structured_dataset_type.columns[0].literal_type.simple is not None
    assert df_literal_type.structured_dataset_type.columns[1].name == "age"
    assert df_literal_type.structured_dataset_type.columns[1].literal_type.simple is not None
    assert df_literal_type.structured_dataset_type.format == "avro"

    sd = annotated_sd_type.from_df(val=df)
    with pytest.raises(ValueError, match="Failed to find a handler"):
        await TypeEngine.to_literal(sd, python_type=annotated_sd_type, expected=df_literal_type)

    DataFrameTransformerEngine.register(TempEncoder(), default_for_type=False)
    sd2 = annotated_sd_type.from_df(val=df)
    sd_literal = await TypeEngine.to_literal(sd2, python_type=annotated_sd_type, expected=df_literal_type)
    assert sd_literal.scalar.structured_dataset.metadata.structured_dataset_type.format == "avro"

    env = flyte.TaskEnvironment("test")

    @env.task
    async def t1() -> typing.Annotated[DataFrame, "avro"]:
        return DataFrame.from_df(val=df)

    # pr: this test doesn't work right now, because calling the task just calls the function.
    # res = await t1()
    # assert res.format == "avro"


def test_protocol_detection(ctx_with_test_raw_data_path):
    # We don't register defaults to the transformer engine
    assert pd.DataFrame not in DataFrameTransformerEngine.DEFAULT_PROTOCOLS
    e = DataFrameTransformerEngine()
    ctx = internal_ctx()
    protocol = e._protocol_from_type_or_prefix(pd.DataFrame)
    assert protocol == "file"

    with tempfile.TemporaryDirectory():
        ctx2 = ctx.new_raw_data_path(raw_data_path=RawDataPath(path="s3://bucket"))
        with ctx2:
            protocol = e._protocol_from_type_or_prefix(pd.DataFrame)
            assert protocol == "s3"

            protocol = e._protocol_from_type_or_prefix(pd.DataFrame, "bq://foo")
            assert protocol == "bq"


def test_register_renderers():
    class DummyRenderer:
        def to_html(self, input: str) -> str:
            return "hello " + input

    renderers = DataFrameTransformerEngine.Renderers
    DataFrameTransformerEngine.register_renderer(str, DummyRenderer())
    assert renderers[str].to_html("flyte") == "hello flyte"
    assert pd.DataFrame in renderers
    assert pa.Table in renderers

    with pytest.raises(NotImplementedError, match="Could not find a renderer for <class 'int'> in"):
        eng = DataFrameTransformerEngine()
        eng.to_html(3, int)


def test_list_of_annotated():
    WineDataset = typing.Annotated[
        DataFrame,
        OrderedDict(
            alcohol=float,
            malic_acid=float,
        ),
    ]

    env = flyte.TaskEnvironment(name="test-env")

    @env.task
    async def no_op(data: WineDataset) -> typing.List[WineDataset]:
        return [data]


class PrivatePandasToBQEncodingHandlers(DataFrameEncoder):
    def __init__(self):
        super().__init__(pd.DataFrame, "bq", supported_format="")

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        return literals_pb2.StructuredDataset(
            uri=typing.cast(str, dataframe.uri),
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


@pytest.mark.asyncio
async def test_reregister_encoder(ctx_with_test_raw_data_path):
    # Test that lazy import can run after a user has already registered a custom handler.
    # The default handlers don't have override=True (and should not) but the
    # call should not fail.
    import google.cloud.bigquery

    dir(google.cloud.bigquery)
    assert is_imported("google.cloud.bigquery")

    DataFrameTransformerEngine.register(
        PrivatePandasToBQEncodingHandlers(), default_format_for_type=False, override=True
    )
    TypeEngine.lazy_import_transformers()

    sd = DataFrame.from_df(val=pd.DataFrame({"a": [1, 2], "b": [3, 4]}), uri="bq://blah")

    df_literal_type = TypeEngine.to_literal_type(pd.DataFrame)

    await TypeEngine.to_literal(sd, python_type=pd.DataFrame, expected=df_literal_type)


@pytest.mark.asyncio
async def test_default_args_task(ctx_with_test_raw_data_path, dummy_serialization_context):
    from flyte._internal.runtime import task_serde

    env = flyte.TaskEnvironment(name="test-sd-default-args")
    default_val = pd.DataFrame({"name": ["Aegon"], "age": [27]})
    input_val = generate_pandas()

    @env.task
    async def t1(a: pd.DataFrame = default_val) -> pd.DataFrame:
        return a

    @env.task
    async def wf_no_input() -> pd.DataFrame:
        return await t1()

    @env.task
    async def wf_with_input() -> pd.DataFrame:
        return await t1(a=input_val)

    wf_no_input_spec = task_serde.translate_task_to_wire(wf_no_input, dummy_serialization_context)

    wf_with_input_spec = task_serde.translate_task_to_wire(wf_with_input, dummy_serialization_context)

    assert wf_with_input_spec
    assert wf_no_input_spec

    # uncomment the asserts below after adding compilation feature
    # assert wf_no_input_spec.template.nodes[0].inputs[
    #     0
    # ].binding.value._structured_dataset.metadata == literals_pb2.StructuredDatasetMetadata(
    #     structured_dataset_type=types_pb2.StructuredDatasetType(
    #         format="parquet",
    #     ),
    # )
    # assert wf_with_input_spec.template.nodes[0].inputs[
    #     0
    # ].binding.value._structured_dataset.metadata == StructuredDatasetMetadata(
    #     structured_dataset_type=StructuredDatasetType(
    #         format="parquet",
    #     ),
    # )
    #
    # assert wf_no_input_spec.template.interface.outputs["o0"].type == LiteralType(
    #     structured_dataset_type=StructuredDatasetType()
    # )
    # assert wf_with_input_spec.template.interface.outputs["o0"].type == LiteralType(
    #     structured_dataset_type=StructuredDatasetType()
    # )
    #
    # pd.testing.assert_frame_equal(wf_no_input(), default_val)
    # pd.testing.assert_frame_equal(wf_with_input(), input_val)


@pytest.mark.asyncio
async def test_read_sd_from_local_uri(local_tmp_pqt_file, ctx_with_test_raw_data_path):
    env = flyte.TaskEnvironment(name="test-sd-local-uri")

    @env.task
    async def read_sd_from_uri(uri: str) -> pd.DataFrame:
        sd = DataFrame.from_existing_remote(uri, format="parquet")
        df = await sd.open(pd.DataFrame).all()

        return df

    @env.task
    async def read_sd_from_local_uri(uri: str) -> pd.DataFrame:
        df = await read_sd_from_uri(uri=uri)

        return df

    df = generate_pandas()

    # Read sd from local uri
    df_local = await read_sd_from_local_uri(uri=local_tmp_pqt_file)
    pd.testing.assert_frame_equal(df, df_local)


@pytest.mark.asyncio
@mock.patch("flyte.storage._remote_fs.RemoteFSPathResolver")
@mock.patch("flyte.io.extend.DataFrameTransformerEngine.get_encoder")
async def test_modify_literal_uris_call(mock_get_encoder, mock_resolver, ctx_with_test_raw_data_path):
    sd = DataFrame.from_df(val=pd.DataFrame({"a": [1, 2], "b": [3, 4]}), uri="bq://blah")

    def mock_resolve_remote_path(flyte_uri: str) -> typing.Optional[str]:
        if flyte_uri == "bq://blah":
            return "bq://blah/blah/blah"
        return ""

    mock_resolver.resolve_remote_path.side_effect = mock_resolve_remote_path
    mock_resolver.protocol = "bq"

    dummy_encoder = mock.AsyncMock()
    dummy_encoder.supported_format = "parquet"
    sd_model = literals_pb2.StructuredDataset(
        uri="bq://blah",
        metadata=literals_pb2.StructuredDatasetMetadata(
            structured_dataset_type=types_pb2.StructuredDatasetType(format="parquet")
        ),
    )
    dummy_encoder.encode.return_value = sd_model

    mock_get_encoder.return_value = dummy_encoder

    sdte = DataFrameTransformerEngine()
    lit = await sdte.encode(
        sd,
        df_type=pd.DataFrame,
        protocol="bq",
        format="parquet",
        structured_literal_type=types_pb2.StructuredDatasetType(),
    )
    assert lit.scalar.structured_dataset.uri == "bq://blah/blah/blah"


def test_schema():
    # from flytekit.types.file import FlyteFile
    # from flytekit.types.directory import FlyteDirectory
    import json

    from pydantic import BaseModel, Field

    class BM(BaseModel):
        # ff: FlyteFile = Field(default=None)
        # fd: FlyteDirectory = Field(default=None)
        sd: DataFrame = Field(default=None)

    ss = BM.model_json_schema()
    assert json.dumps(ss, indent=2)


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
