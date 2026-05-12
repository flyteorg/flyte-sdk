from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flyte.types import TypeEngine

pytest.importorskip("pyspark")
pandera_typing_pyspark_sql = pytest.importorskip("pandera.typing.pyspark_sql")

try:
    import pandera.pyspark as pandera_pyspark
except ImportError:
    pandera_pyspark = None

import pyspark.sql.types as T
from flyteplugins.spark.df_transformer import register_spark_df_transformers
from pyspark.sql import SparkSession

from flyteplugins.pandera.transformers.pyspark_sql import (
    PanderaPySparkSqlDataFrameTransformer,
    register_pandera_pyspark_sql_type_transformers,
)


def test_register_pyspark_sql_transformer():
    register_pandera_pyspark_sql_type_transformers()
    t = TypeEngine.get_transformer(pandera_typing_pyspark_sql.DataFrame)
    assert isinstance(t, PanderaPySparkSqlDataFrameTransformer)


@pytest.mark.skipif(pandera_pyspark is None, reason="pandera pyspark DataFrameModel not importable in this env")
@pytest.mark.asyncio
async def test_pandera_pyspark_sql_roundtrip(ctx_with_test_raw_data_path):
    class RowSchema(pandera_pyspark.DataFrameModel):
        value: int = pandera_pyspark.Field()

    register_spark_df_transformers()
    transformer = PanderaPySparkSqlDataFrameTransformer()
    try:
        spark = SparkSession.builder.master("local[1]").appName("flyteplugins-pandera-pyspark").getOrCreate()
    except Exception as exc:
        pytest.skip(f"SparkSession not available: {exc}")
    try:
        data = [(1,), (2,), (3,)]
        sdf = spark.createDataFrame(data, schema=T.StructType([T.StructField("value", T.IntegerType(), False)]))
        df_type = pandera_typing_pyspark_sql.DataFrame[RowSchema]
        lt = TypeEngine.to_literal_type(df_type)

        with patch("flyte.report.get_tab") as get_tab:
            fake_tab = MagicMock()
            get_tab.return_value = fake_tab
            lit = await transformer.to_literal(sdf, df_type, lt)
            restored = await transformer.to_python_value(lit, df_type)

        rows = restored.select("value").orderBy("value").collect()
        assert [r.value for r in rows] == [1, 2, 3]
        assert get_tab.call_count >= 1
        assert fake_tab.replace.call_count >= 1
    finally:
        spark.stop()
