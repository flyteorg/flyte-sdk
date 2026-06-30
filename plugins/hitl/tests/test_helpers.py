"""Tests for HITL helper functions."""

from unittest.mock import MagicMock, patch

import pytest

from flyteplugins.hitl._helpers import (
    _convert_value,
    _get_request_path,
    _get_response_path,
)


class TestConvertValue:
    """Tests for _convert_value() helper function."""

    def test_convert_value_int(self):
        """Test converting string to int."""
        assert _convert_value("42", "int") == 42
        assert _convert_value("-10", "int") == -10

    def test_convert_value_float(self):
        """Test converting string to float."""
        assert _convert_value("3.14", "float") == 3.14
        assert _convert_value("42", "float") == 42.0

    def test_convert_value_bool(self):
        """Test converting string to bool."""
        assert _convert_value("true", "bool") is True
        assert _convert_value("True", "bool") is True
        assert _convert_value("1", "bool") is True
        assert _convert_value("yes", "bool") is True
        assert _convert_value("false", "bool") is False
        assert _convert_value("0", "bool") is False
        assert _convert_value("anything_else", "bool") is False

    def test_convert_value_int_invalid_raises_error(self):
        """Test that invalid int conversion raises ValueError."""
        with pytest.raises(ValueError):
            _convert_value("not_a_number", "int")

    def test_convert_value_float_invalid_raises_error(self):
        """Test that invalid float conversion raises ValueError."""
        with pytest.raises(ValueError):
            _convert_value("not_a_number", "float")


class TestGetRequestPath:
    """Tests for _get_request_path() helper function."""

    @patch("flyte._context.internal_ctx")
    def test_get_request_path_with_raw_data(self, mock_internal_ctx):
        """Test request path when raw_data is available."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = True
        mock_ctx.raw_data.path = "s3://bucket/data"
        mock_internal_ctx.return_value = mock_ctx

        result = _get_request_path("test-request-id")

        assert result == "s3://bucket/data/hitl-requests/test-request-id/request.json"

    @patch("flyte._context.internal_ctx")
    @patch.dict("os.environ", {"RAW_DATA_PATH": "gs://my-bucket/raw"})
    def test_get_request_path_with_env_var(self, mock_internal_ctx):
        """Test request path when RAW_DATA_PATH env var is set."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = False
        mock_internal_ctx.return_value = mock_ctx

        result = _get_request_path("test-request-id")

        assert result == "gs://my-bucket/raw/hitl-requests/test-request-id/request.json"

    @patch("flyte._context.internal_ctx")
    @patch.dict("os.environ", {}, clear=True)
    def test_get_request_path_fallback_to_local(self, mock_internal_ctx):
        """Test request path falls back to local path when no raw_data or env var."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = False
        mock_internal_ctx.return_value = mock_ctx

        result = _get_request_path("test-request-id")

        assert result == "/tmp/flyte/hitl/hitl-requests/test-request-id/request.json"


class TestGetResponsePath:
    """Tests for _get_response_path() helper function."""

    @patch("flyte._context.internal_ctx")
    def test_get_response_path_with_raw_data(self, mock_internal_ctx):
        """Test response path when raw_data is available."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = True
        mock_ctx.raw_data.path = "s3://bucket/data"
        mock_internal_ctx.return_value = mock_ctx

        result = _get_response_path("test-request-id")

        assert result == "s3://bucket/data/hitl-requests/test-request-id/response.json"

    @patch("flyte._context.internal_ctx")
    @patch.dict("os.environ", {"RAW_DATA_PATH": "gs://my-bucket/raw"})
    def test_get_response_path_with_env_var(self, mock_internal_ctx):
        """Test response path when RAW_DATA_PATH env var is set."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = False
        mock_internal_ctx.return_value = mock_ctx

        result = _get_response_path("test-request-id")

        assert result == "gs://my-bucket/raw/hitl-requests/test-request-id/response.json"

    @patch("flyte._context.internal_ctx")
    @patch.dict("os.environ", {}, clear=True)
    def test_get_response_path_fallback_to_local(self, mock_internal_ctx):
        """Test response path falls back to local path when no raw_data or env var."""
        mock_ctx = MagicMock()
        mock_ctx.has_raw_data = False
        mock_internal_ctx.return_value = mock_ctx

        result = _get_response_path("test-request-id")

        assert result == "/tmp/flyte/hitl/hitl-requests/test-request-id/response.json"
