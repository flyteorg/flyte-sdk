import os
import typing

import flyte
import pyspark
from flyte._context import internal_ctx
from flyte.io import PARQUET, DataFrame, DataFrameDecoder, DataFrameEncoder, DataFrameTransformerEngine
from flyteidl2.core import literals_pb2, types_pb2
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame as PSDataFrame
from typing_extensions import cast


class SparkToParquetEncodingHandler(DataFrameEncoder):
    def __init__(self):
        super().__init__(PSDataFrame, None, PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        uri = typing.cast(str, dataframe.uri)
        ctx = internal_ctx()
        if ctx and not uri:
            uri = ctx.raw_data.get_random_remote_path()

        df = typing.cast(PSDataFrame, dataframe.val)
        ss = pyspark.sql.SparkSession.builder.getOrCreate()
        path = os.path.join(uri, f"{0:05}")

        print("[debug]: ctx:", ctx)
        print("[debug] uri:", uri)
        print("[debug] path:", path)

        # Avoid generating SUCCESS files
        ss.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        df.write.mode("overwrite").parquet(path=path)

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=uri,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class ParquetToSparkDecodingHandler(DataFrameDecoder):
    def __init__(self):
        super().__init__(PSDataFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> PSDataFrame:
        print("[debug]: flyte_value:", flyte_value)
        ctx = flyte.ctx()

        uri = flyte_value.uri
        columns = None
        path = os.path.join(uri, f"{0:05}")
        spark = ctx["spark_session"]
        spark = cast(SparkSession, spark)

        print("[debug]: current_task_metadata:", current_task_metadata)
        print("[debug]: uri:", uri)
        print("[debug]: path:", path)

        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
            return spark.read.parquet(path).select(*columns)
        return spark.read.parquet(path)


DataFrameTransformerEngine.register(SparkToParquetEncodingHandler(), default_format_for_type=True)
DataFrameTransformerEngine.register(ParquetToSparkDecodingHandler(), default_format_for_type=True)
