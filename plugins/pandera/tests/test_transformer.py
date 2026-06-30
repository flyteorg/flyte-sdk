from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from flyte.types import TypeEngine

# Import pandera typing *before* flyteplugins.pandera so `pandera.typing.pandas.DataFrame` is the
# same class object Flyte's TypeEngine keys on after plugin entry points load (see plugin README).
try:
    import pandera
    import pandera.typing.pandas as pandera_typing_pandas
except ImportError:
    pandera = None
    pandera_typing_pandas = None

from flyteplugins.pandera import ValidationConfig
from flyteplugins.pandera.renderers.pandas import PanderaPandasReportRenderer
from flyteplugins.pandera.transformers.pandas import (
    PanderaPandasDataFrameTransformer,
    register_pandera_pandas_type_transformers,
)

pytestmark = pytest.mark.skipif(pandera is None or pandera_typing_pandas is None, reason="pandera is not installed")


if pandera is not None:

    class InputSchema(pandera.DataFrameModel):
        value: int


@pytest.mark.asyncio
async def test_pandera_pandas_roundtrip(ctx_with_test_raw_data_path):
    transformer = PanderaPandasDataFrameTransformer()
    data = pd.DataFrame({"value": [1, 2, 3]})
    df_type = pandera_typing_pandas.DataFrame[InputSchema]
    lt = TypeEngine.to_literal_type(df_type)

    with patch("flyte.report.get_tab") as get_tab:
        fake_tab = MagicMock()
        get_tab.return_value = fake_tab
        lit = await transformer.to_literal(data, df_type, lt)
        restored = await transformer.to_python_value(lit, df_type)

    assert isinstance(restored, pd.DataFrame)
    assert restored["value"].tolist() == [1, 2, 3]
    assert get_tab.call_count >= 1
    assert fake_tab.replace.call_count >= 1


@pytest.mark.asyncio
async def test_pandera_warn_mode(ctx_with_test_raw_data_path):
    transformer = PanderaPandasDataFrameTransformer()
    invalid = pd.DataFrame({"value": ["a", "b"]})
    df_type = pandera_typing_pandas.DataFrame[InputSchema]
    configured_type = typing.Annotated[df_type, ValidationConfig(on_error="warn")]
    lt = TypeEngine.to_literal_type(configured_type)

    with patch("flyte.report.get_tab") as get_tab:
        get_tab.return_value = MagicMock()
        lit = await transformer.to_literal(invalid, configured_type, lt)
        restored = await transformer.to_python_value(lit, configured_type)

    assert isinstance(restored, pd.DataFrame)


def test_register_transformer():
    register_pandera_pandas_type_transformers()
    t = TypeEngine.get_transformer(pandera_typing_pandas.DataFrame)
    assert isinstance(t, PanderaPandasDataFrameTransformer)


def test_reshape_long_failure_cases_duplicate_pivot_column():
    """Pandera repeats the model name in ``column`` for each missing field under ``column_in_dataframe``."""
    long = pd.DataFrame(
        {
            "schema_context": ["DataFrameSchema", "DataFrameSchema"],
            "check": ["column_in_dataframe", "column_in_dataframe"],
            "index": [pd.NA, pd.NA],
            "column": ["EmployeeSchemaWithStatus", "EmployeeSchemaWithStatus"],
            "failure_case": ["name", "status"],
        }
    )
    out = PanderaPandasReportRenderer._reshape_long_failure_cases(long)
    assert list(out.columns) == ["check", "index", "failure_case"]
    assert out.iloc[0]["check"] == "column_in_dataframe"
    assert isinstance(out.iloc[0]["failure_case"], dict)
    assert "EmployeeSchemaWithStatus" in out.iloc[0]["failure_case"]
