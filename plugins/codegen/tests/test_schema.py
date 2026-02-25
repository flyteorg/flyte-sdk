"""Tests for flyteplugins.codegen.data.schema."""

from unittest.mock import MagicMock

import pandas as pd
import pandera.pandas as pa

from flyteplugins.codegen.core.types import _ConstraintParameters, _ConstraintParse
from flyteplugins.codegen.data.schema import (
    apply_parsed_constraint,
    extract_token_usage,
    infer_conservative_schema,
    schema_to_script,
)

# ---------------------------------------------------------------------------
# infer_conservative_schema
# ---------------------------------------------------------------------------


class TestInferConservativeSchema:
    def test_numeric_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1.1, 2.2, 3.3]})
        schema = infer_conservative_schema(df)
        assert "x" in schema.columns
        assert "y" in schema.columns
        # Should have no value-based checks
        assert schema.columns["x"].checks is None or schema.columns["x"].checks == []
        assert schema.columns["y"].checks is None or schema.columns["y"].checks == []

    def test_nullable_detection(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        schema = infer_conservative_schema(df)
        # Column 'a' has nulls, should be nullable
        assert schema.columns["a"].nullable is True

    def test_string_columns(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})
        schema = infer_conservative_schema(df)
        assert "name" in schema.columns

    def test_schema_is_not_strict(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        schema = infer_conservative_schema(df)
        assert schema.strict is False

    def test_nullable_extension_dtypes_normalized(self):
        """Nullable extension dtypes like pd.Int64Dtype should be normalized."""
        df = pd.DataFrame({"x": pd.array([1, 2, 3], dtype=pd.Int64Dtype())})
        schema = infer_conservative_schema(df)
        assert "x" in schema.columns

    def test_boolean_dtype(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        schema = infer_conservative_schema(df)
        assert "flag" in schema.columns


# ---------------------------------------------------------------------------
# schema_to_script
# ---------------------------------------------------------------------------


class TestSchemaToScript:
    def test_produces_string(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        schema = infer_conservative_schema(df)
        result = schema_to_script(schema)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# apply_parsed_constraint
# ---------------------------------------------------------------------------


class TestApplyParsedConstraint:
    def _make_schema(self):
        return pa.DataFrameSchema(
            columns={
                "price": pa.Column(float, nullable=True),
                "name": pa.Column(str, nullable=True),
                "qty": pa.Column(int, nullable=True),
            },
            strict=False,
        )

    def test_greater_than(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="price",
            check_type="greater_than",
            parameters=_ConstraintParameters(value=0),
            explanation="price must be positive",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert result.columns["price"].checks is not None
        assert len(result.columns["price"].checks) > 0

    def test_less_than(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="price",
            check_type="less_than",
            parameters=_ConstraintParameters(value=1000),
            explanation="price must be < 1000",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert len(result.columns["price"].checks) > 0

    def test_between(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="qty",
            check_type="between",
            parameters=_ConstraintParameters(min=0, max=100),
            explanation="qty between 0 and 100",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert len(result.columns["qty"].checks) > 0

    def test_regex(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="name",
            check_type="regex",
            parameters=_ConstraintParameters(pattern=r"^[A-Z]"),
            explanation="name starts with uppercase",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert len(result.columns["name"].checks) > 0

    def test_isin(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="name",
            check_type="isin",
            parameters=_ConstraintParameters(values=["Alice", "Bob"]),
            explanation="name must be Alice or Bob",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert len(result.columns["name"].checks) > 0

    def test_not_null(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="price",
            check_type="not_null",
            parameters=_ConstraintParameters(),
            explanation="price cannot be null",
        )
        result = apply_parsed_constraint(schema, parsed)
        assert result.columns["price"].nullable is False

    def test_none_type_returns_unchanged(self):
        schema = self._make_schema()
        parsed = _ConstraintParse(
            column_name="price",
            check_type="none",
            parameters=_ConstraintParameters(),
            explanation="not applicable",
        )
        result = apply_parsed_constraint(schema, parsed)
        # Should be unchanged
        assert result.columns["price"].checks is None or result.columns["price"].checks == []


# ---------------------------------------------------------------------------
# extract_token_usage
# ---------------------------------------------------------------------------


class TestExtractTokenUsage:
    def test_with_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 200
        response = MagicMock()
        response.usage = usage

        in_tok, out_tok = extract_token_usage(response)
        assert in_tok == 100
        assert out_tok == 200

    def test_without_usage(self):
        response = MagicMock()
        response.usage = None
        # getattr on None should still return 0
        in_tok, out_tok = extract_token_usage(response)
        assert in_tok == 0
        assert out_tok == 0

    def test_exception_returns_zeros(self):
        """If anything goes wrong, return (0, 0)."""
        response = MagicMock()
        response.usage = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))

        # Force an exception
        del response.usage
        type(response).usage = property(lambda self: (_ for _ in ()).throw(RuntimeError("fail")))

        in_tok, out_tok = extract_token_usage(response)
        assert in_tok == 0
        assert out_tok == 0
