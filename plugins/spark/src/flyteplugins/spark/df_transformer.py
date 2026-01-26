import functools

import flyte
import pyspark
from flyte.io import PARQUET, DataFrame
from flyte.io.extend import DataFrameDecoder, DataFrameEncoder, DataFrameTransformerEngine
from flyteidl2.core import literals_pb2, types_pb2
from typing_extensions import cast


class SparkToParquetEncoder(DataFrameEncoder):
    def __init__(self):
        super().__init__(python_type=pyspark.sql.DataFrame, supported_format=PARQUET)

    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        path = dataframe.uri
        ctx = flyte.ctx()
        if ctx and not path:
            path = ctx.raw_data_path.get_random_remote_path()

        ss = pyspark.sql.SparkSession.builder.getOrCreate()

        # Avoid generating SUCCESS files
        ss.conf.set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        cast(pyspark.sql.DataFrame, dataframe._raw_df).write.mode("overwrite").parquet(path=path)

        structured_dataset_type.format = PARQUET
        return literals_pb2.StructuredDataset(
            uri=path,
            metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_dataset_type),
        )


class ParquetToSparkDecoder(DataFrameDecoder):
    def __init__(self):
        super().__init__(pyspark.sql.DataFrame, None, PARQUET)

    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> pyspark.sql.DataFrame:
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        path = flyte_value.uri

        if current_task_metadata.structured_dataset_type and current_task_metadata.structured_dataset_type.columns:
            columns = [c.name for c in current_task_metadata.structured_dataset_type.columns]
            return spark.read.parquet(path).select(*columns)
        return spark.read.parquet(path)


@functools.lru_cache(maxsize=None)
def register_spark_df_transformers():
    """Register Spark DataFrame encoders and decoders with the DataFrameTransformerEngine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).
    """
    DataFrameTransformerEngine.register(SparkToParquetEncoder(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToSparkDecoder(), default_format_for_type=True)


# Also register at module import time for backwards compatibility
register_spark_df_transformers()
