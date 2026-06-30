"""Tests for flyteplugins.codegen.data.extraction."""

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from flyteplugins.codegen.data.extraction import (
    extract_dataframe_context,
    extract_file_context,
    is_dataframe,
)

# ---------------------------------------------------------------------------
# is_dataframe
# ---------------------------------------------------------------------------


class TestIsDataframe:
    def test_true_for_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert is_dataframe(df) is True

    def test_false_for_dict(self):
        assert is_dataframe({"a": 1}) is False

    def test_false_for_list(self):
        assert is_dataframe([1, 2, 3]) is False

    def test_false_for_string(self):
        assert is_dataframe("hello") is False

    def test_false_for_none(self):
        assert is_dataframe(None) is False


# ---------------------------------------------------------------------------
# extract_dataframe_context
# ---------------------------------------------------------------------------


class TestExtractDataframeContext:
    @pytest.mark.asyncio
    async def test_basic_context(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
                "score": [88.5, 92.0, 75.3, 98.1, 81.0],
            }
        )
        result = await extract_dataframe_context(df, "test_data")
        assert "## Data: test_data" in result
        assert "5 rows x 3 columns" in result
        assert "score" in result

    @pytest.mark.asyncio
    async def test_numeric_stats_included(self):
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = await extract_dataframe_context(df, "numbers")
        assert "Numeric columns:" in result
        assert "value" in result

    @pytest.mark.asyncio
    async def test_categorical_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red", "green", "blue"]})
        result = await extract_dataframe_context(df, "colors")
        assert "Categorical columns:" in result

    @pytest.mark.asyncio
    async def test_with_schema(self):

        from flyteplugins.codegen.data.schema import infer_conservative_schema

        df = pd.DataFrame({"x": [1, 2, 3]})
        schema = infer_conservative_schema(df)
        result = await extract_dataframe_context(df, "data", schema=schema)
        assert "Pandera Schema" in result

    @pytest.mark.asyncio
    async def test_duplicate_detection(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = await extract_dataframe_context(df, "dupes")
        assert "duplicate" in result.lower()

    @pytest.mark.asyncio
    async def test_unique_id_detection(self):
        df = pd.DataFrame({"uid": [1, 2, 3], "val": ["a", "b", "c"]})
        result = await extract_dataframe_context(df, "with_id")
        assert "unique identifier" in result.lower()


# ---------------------------------------------------------------------------
# extract_file_context
# ---------------------------------------------------------------------------


class TestExtractFileContext:
    @pytest.mark.asyncio
    async def test_reads_text_file(self, tmp_path):
        # Create a temp CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        mock_file = MagicMock()
        mock_file.download = AsyncMock(return_value=str(csv_file))

        result = await extract_file_context(mock_file, "test_csv")
        assert "## Data: test_csv" in result
        assert "Text file" in result
        assert "a,b" in result

    @pytest.mark.asyncio
    async def test_binary_file_fallback(self, tmp_path):
        # Create a binary file
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")

        mock_file = MagicMock()
        mock_file.download = AsyncMock(return_value=str(bin_file))

        # The text reading won't fail for small binary, but let's test the path
        result = await extract_file_context(mock_file, "binary_data")
        assert "## Data: binary_data" in result
