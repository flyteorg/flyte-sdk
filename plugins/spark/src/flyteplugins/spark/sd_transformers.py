import typing

from flyte._context import internal_ctx
from flyte._utils.lazy_module import lazy_module
from flyte.io import PARQUET, DataFrame, DataFrameDecoder, DataFrameEncoder, DataFrameTransformerEngine
from flyteidl.core import literals_pb2, types_pb2
from typing_extensions import Any

pd = lazy_module("pandas")
pyspark = lazy_module("pyspark")
ps_dataframe = lazy_module("pyspark.sql.dataframe")
PSDataFrame = ps_dataframe.DataFrame


class SparkDataFrameRenderer:
    """
    Render a Spark dataframe schema as an HTML table.
    """

    def to_html(self, python_value: Any) -> str:
        """Convert an object(markdown, pandas.dataframe) to HTML and return HTML as a unicode string.
        Returns: An HTML document as a string.
        """
        assert isinstance(python_value, PSDataFrame)
        return pd.DataFrame(python_value.schema, columns=["StructField"]).to_html()


class SparkToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(PSDataFrame, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        path = typing.cast(str, dataframe.uri)
        # ctx = internal_ctx()
        # if ctx and not path:
        #     path = ctx.file_access.join(
        #         ctx.file_access.raw_output_prefix,
        #         ctx.file_access.get_random_string(),
        #     )

        df = typing.cast(PSDataFrame, dataframe)
        ss = pyspark.sql.SparkSession.builder.getOrCreate()
        # Avoid generating SUCCESS files
        ss.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        df.write.mode("overwrite").parquet(path=path)

        return literals_pb2.StructuredDataset(
            uri=path,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type),
        )


class ParquetToSparkDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(PSDataFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> PSDataFrame:
        ctx = internal_ctx()
        user_ctx = ctx.data.task_context.data["user_space_params"]

        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
            return user_ctx.spark_session.read.parquet(flyte_value.uri).select(*columns)
        return user_ctx.spark_session.read.parquet(flyte_value.uri)


DataFrameTransformerEngine.register(SparkToParquetEncodingHandler())
DataFrameTransformerEngine.register(ParquetToSparkDecodingHandler())
DataFrameTransformerEngine.register_renderer(PSDataFrame, SparkDataFrameRenderer())

classic_ps_dataframe = lazy_module("pyspark.sql.classic.dataframe")
ClassicDataFrame = classic_ps_dataframe.DataFrame


class ClassicSparkToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(ClassicDataFrame, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        path = typing.cast(str, dataframe.uri)

        # if not path:
        #     path = ctx.file_access.join(
        #         ctx.file_access.raw_output_prefix,
        #         ctx.file_access.get_random_string(),
        #     )

        df = typing.cast(ClassicDataFrame, dataframe)
        ss = pyspark.sql.SparkSession.builder.getOrCreate()
        ss.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        df.write.mode("overwrite").parquet(path=path)

        return literals_pb2.StructuredDataset(
            uri=path, metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type)
        )


class ParquetToClassicSparkDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(ClassicDataFrame, None, PARQUET)

    def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> ClassicDataFrame:
        ctx = internal_ctx()
        user_ctx = ctx.data.task_context.data["user_space_params"]
        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
            return user_ctx.spark_session.read.parquet(flyte_value.uri).select(*columns)
        return user_ctx.spark_session.read.parquet(flyte_value.uri)


# Register the handlers
DataFrameTransformerEngine.register(ClassicSparkToParquetEncodingHandler())
DataFrameTransformerEngine.register(ParquetToClassicSparkDecodingHandler())
DataFrameTransformerEngine.register_renderer(ClassicDataFrame, SparkDataFrameRenderer())
