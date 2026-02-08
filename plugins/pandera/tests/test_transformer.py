"""Tests for the Pandera plugin transformer."""

import typing

import pandas as pd
import pandera as pa
import pytest
from flyte.types import TypeEngine
from flyteidl2.core import literals_pb2
from pandera.typing import DataFrame, Series

# Import to ensure registration
import flyteplugins.pandera.transformer  # noqa: F401
from flyteplugins.pandera.config import ValidationConfig
from flyteplugins.pandera.transformer import PanderaTransformer

# ============================================================================
# Test schemas
# ============================================================================


class UserSchema(pa.DataFrameModel):
    name: Series[str]
    age: Series[int] = pa.Field(ge=0, le=120)
    email: Series[str]


class StrictSchema(pa.DataFrameModel):
    value: Series[float] = pa.Field(ge=0.0, le=1.0)
    label: Series[str]


# ============================================================================
# Sample data
# ============================================================================

VALID_USER_DATA = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
}

INVALID_USER_DATA = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, -5, 200],  # -5 and 200 are out of range
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
}

VALID_STRICT_DATA = {
    "value": [0.1, 0.5, 0.9],
    "label": ["low", "mid", "high"],
}


# ============================================================================
# Registration tests
# ============================================================================


def test_pandera_transformer_registered():
    """Test that the Pandera transformer is registered with the TypeEngine."""
    transformer = TypeEngine.get_transformer(DataFrame[UserSchema])
    assert isinstance(transformer, PanderaTransformer)


def test_pandera_transformer_name():
    """Test the transformer has the correct name."""
    transformer = PanderaTransformer()
    assert transformer.name == "Pandera Transformer"


# ============================================================================
# Literal type tests
# ============================================================================


def test_get_literal_type_basic():
    """Test that get_literal_type works for basic pandera DataFrame."""
    transformer = PanderaTransformer()
    lt = transformer.get_literal_type(DataFrame[UserSchema])
    assert lt.structured_dataset_type is not None


def test_get_literal_type_unannotated():
    """Test get_literal_type for an unannotated pandera DataFrame."""
    transformer = PanderaTransformer()
    lt = transformer.get_literal_type(DataFrame)
    assert lt.structured_dataset_type is not None


# ============================================================================
# Schema extraction tests
# ============================================================================


def test_get_pandera_schema_with_model():
    """Test schema extraction from pandera DataFrameModel."""
    schema, config = PanderaTransformer._get_pandera_schema(DataFrame[UserSchema])
    assert schema is not None
    assert "name" in schema.columns
    assert "age" in schema.columns
    assert config.on_error == "raise"


def test_get_pandera_schema_without_model():
    """Test schema extraction for unannotated pandera DataFrame."""
    schema, config = PanderaTransformer._get_pandera_schema(DataFrame)
    assert isinstance(schema, pa.DataFrameSchema)
    assert config.on_error == "raise"


def test_get_pandera_schema_with_config():
    """Test schema extraction with custom ValidationConfig."""
    annotated_type = typing.Annotated[DataFrame[UserSchema], ValidationConfig(on_error="warn")]
    schema, config = PanderaTransformer._get_pandera_schema(annotated_type)
    assert schema is not None
    assert config.on_error == "warn"


# ============================================================================
# Validation tests (to_literal)
# ============================================================================


@pytest.mark.asyncio
async def test_to_literal_valid_data(ctx_with_test_raw_data_path):
    """Test to_literal with valid data passes validation."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(VALID_USER_DATA)
    lt = transformer.get_literal_type(DataFrame[UserSchema])

    lit = await transformer.to_literal(df, DataFrame[UserSchema], lt)
    assert lit.scalar.structured_dataset.uri is not None


@pytest.mark.asyncio
async def test_to_literal_invalid_data_raises(ctx_with_test_raw_data_path):
    """Test to_literal with invalid data raises SchemaErrors."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(INVALID_USER_DATA)
    lt = transformer.get_literal_type(DataFrame[UserSchema])

    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        await transformer.to_literal(df, DataFrame[UserSchema], lt)


@pytest.mark.asyncio
async def test_to_literal_invalid_data_warn(ctx_with_test_raw_data_path):
    """Test to_literal with invalid data and on_error=warn logs warning."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(INVALID_USER_DATA)
    annotated_type = typing.Annotated[DataFrame[UserSchema], ValidationConfig(on_error="warn")]
    lt = transformer.get_literal_type(annotated_type)

    # Should not raise, but should warn
    lit = await transformer.to_literal(df, annotated_type, lt)
    assert lit.scalar.structured_dataset.uri is not None


@pytest.mark.asyncio
async def test_to_literal_non_dataframe_raises(ctx_with_test_raw_data_path):
    """Test to_literal raises when given a non-DataFrame value."""
    transformer = PanderaTransformer()
    lt = transformer.get_literal_type(DataFrame[UserSchema])

    with pytest.raises(AssertionError, match="Only pandas DataFrame"):
        await transformer.to_literal("not a dataframe", DataFrame[UserSchema], lt)


# ============================================================================
# Deserialization tests (to_python_value)
# ============================================================================


@pytest.mark.asyncio
async def test_roundtrip_valid_data(ctx_with_test_raw_data_path):
    """Test roundtrip encoding/decoding with valid data."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(VALID_USER_DATA)
    lt = transformer.get_literal_type(DataFrame[UserSchema])

    # Encode
    lit = await transformer.to_literal(df, DataFrame[UserSchema], lt)

    # Clear validation memo to force re-validation
    PanderaTransformer._VALIDATION_MEMO.clear()

    # Decode
    restored_df = await transformer.to_python_value(lit, DataFrame[UserSchema])
    assert isinstance(restored_df, pd.DataFrame)
    assert restored_df.shape == df.shape
    assert list(restored_df.columns) == list(df.columns)


@pytest.mark.asyncio
async def test_roundtrip_strict_schema(ctx_with_test_raw_data_path):
    """Test roundtrip with a different schema."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(VALID_STRICT_DATA)
    lt = transformer.get_literal_type(DataFrame[StrictSchema])

    lit = await transformer.to_literal(df, DataFrame[StrictSchema], lt)

    PanderaTransformer._VALIDATION_MEMO.clear()

    restored_df = await transformer.to_python_value(lit, DataFrame[StrictSchema])
    assert isinstance(restored_df, pd.DataFrame)
    assert list(restored_df["value"]) == VALID_STRICT_DATA["value"]
    assert list(restored_df["label"]) == VALID_STRICT_DATA["label"]


@pytest.mark.asyncio
async def test_to_python_value_invalid_literal_raises():
    """Test to_python_value raises on invalid literal."""
    transformer = PanderaTransformer()

    with pytest.raises(AssertionError, match="Can only convert a literal structured dataset"):
        await transformer.to_python_value(literals_pb2.Literal(), DataFrame[UserSchema])


# ============================================================================
# Validation memo (caching) tests
# ============================================================================


@pytest.mark.asyncio
async def test_validation_memo_skips_revalidation(ctx_with_test_raw_data_path):
    """Test that validation memo prevents duplicate validation."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(VALID_USER_DATA)
    lt = transformer.get_literal_type(DataFrame[UserSchema])

    # Encode (adds to memo)
    lit = await transformer.to_literal(df, DataFrame[UserSchema], lt)

    # Decode should skip validation (in memo)
    restored_df = await transformer.to_python_value(lit, DataFrame[UserSchema])
    assert isinstance(restored_df, pd.DataFrame)


# ============================================================================
# Assert type tests
# ============================================================================


def test_assert_type_valid():
    """Test assert_type with valid DataFrame."""
    transformer = PanderaTransformer()
    df = pd.DataFrame(VALID_USER_DATA)
    # Should not raise
    transformer.assert_type(pd.DataFrame, df)


def test_assert_type_invalid():
    """Test assert_type with invalid type."""
    transformer = PanderaTransformer()
    with pytest.raises(TypeError):
        transformer.assert_type(pd.DataFrame, "not a dataframe")
