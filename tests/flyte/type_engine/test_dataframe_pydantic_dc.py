from __future__ import annotations

import sys
import typing
from collections import OrderedDict

import pytest
from pydantic import BaseModel
from typing_extensions import Annotated

from flyte.io._dataframe import DataFrame
from flyte.types._type_engine import (
    DataclassTransformer,
    PydanticTransformer,
)

T = typing.TypeVar("T")


@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
@pytest.mark.asyncio
async def test_structured_dataset_in_dataclass(ctx_with_test_raw_data_path):
    import pandas as pd
    from pandas._testing import assert_frame_equal
    from pydantic.dataclasses import dataclass as pyd_dataclass

    df = pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})
    People = Annotated[DataFrame, "parquet", OrderedDict(Name=str, Age=int)]

    @pyd_dataclass
    class InnerDatasetStruct:
        a: DataFrame
        b: typing.List[Annotated[DataFrame, "parquet"]]
        c: typing.Dict[str, Annotated[DataFrame, OrderedDict(Name=str, Age=int)]]

    @pyd_dataclass
    class DatasetStruct:
        a: People
        b: InnerDatasetStruct

    sd = DataFrame.from_df(val=df)
    o = DatasetStruct(a=sd, b=InnerDatasetStruct(a=sd, b=[sd], c={"hello": sd}))

    tf = DataclassTransformer()
    lt = tf.get_literal_type(DatasetStruct)
    lv = await tf.to_literal(o, DatasetStruct, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=DatasetStruct)

    return_df = ot.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.b[0].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.c["hello"].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    assert "parquet" == ot.a.format
    assert "parquet" == ot.b.a.format
    assert "parquet" == ot.b.b[0].format
    assert "parquet" == ot.b.c["hello"].format


@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
@pytest.mark.asyncio
async def test_structured_dataset_in_pydantic(ctx_with_test_raw_data_path):
    import pandas as pd
    from pandas._testing import assert_frame_equal

    df = pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})
    People = Annotated[DataFrame, "parquet", OrderedDict(Name=str, Age=int)]

    class InnerDatasetStruct(BaseModel):
        a: DataFrame
        b: typing.List[Annotated[DataFrame, "parquet"]]
        c: typing.Dict[str, Annotated[DataFrame, OrderedDict(Name=str, Age=int)]]

    class DatasetStruct(BaseModel):
        a: People
        b: InnerDatasetStruct

    sd = DataFrame.from_df(val=df)
    o = DatasetStruct(a=sd, b=InnerDatasetStruct(a=sd, b=[sd], c={"hello": sd}))

    tf = PydanticTransformer()
    lt = tf.get_literal_type(DatasetStruct)
    lv = await tf.to_literal(o, DatasetStruct, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=DatasetStruct)

    return_df = ot.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.b[0].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.c["hello"].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    assert "parquet" == ot.a.format
    assert "parquet" == ot.b.a.format
    assert "parquet" == ot.b.b[0].format
    assert "parquet" == ot.b.c["hello"].format
