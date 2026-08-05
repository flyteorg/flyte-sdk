"""Tests for ConnectError handling in _upload_single_file."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte.errors import InitializationError, RuntimeSystemError
from flyte.remote._data import _upload_single_file


def _make_mock_client(side_effect):
    client = MagicMock()
    client.dataproxy_service.create_upload_location = AsyncMock(side_effect=side_effect)
    return client


def _make_cfg():
    cfg = MagicMock()
    cfg.project = "test-project"
    cfg.domain = "development"
    cfg.org = "test-org"
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code, expected_error, expected_code",
    [
        (Code.NOT_FOUND, RuntimeSystemError, "NotFound"),
        (Code.PERMISSION_DENIED, RuntimeSystemError, "PermissionDenied"),
        (Code.UNAVAILABLE, InitializationError, "EndpointUnavailable"),
        (Code.INTERNAL, RuntimeSystemError, None),  # code == e.code.value
    ],
)
async def test_upload_connect_error_translation(code, expected_error, expected_code, tmp_path):
    """Each ConnectError code from create_upload_location maps to a typed domain error."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    cfg = _make_cfg()
    mock_client = _make_mock_client(ConnectError(code, "server says no"))

    with (
        patch("flyte._initialize._get_init_config", return_value=cfg),
        patch("flyte.remote._data.get_client", return_value=mock_client),
    ):
        with pytest.raises(expected_error) as exc_info:
            await _upload_single_file(cfg, test_file, basedir="")

        if expected_code:
            assert exc_info.value.code == expected_code


@pytest.mark.asyncio
async def test_upload_generic_exception_includes_cause_message(tmp_path):
    """Non-ConnectError exceptions surface the underlying cause's message to the user.

    Regression test for FLYTE-SDK-2H: the prior wrapper dropped the cause message,
    leaving users with only "Failed to get signed url for ..." and no hint that the
    real problem was e.g. missing 'org' config rejected by server validation.
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    cfg = _make_cfg()
    underlying = RuntimeError(
        "SelectCluster failed for operation=1: validation error:\n"
        " - project_id.organization: value length must be at least 1 characters [string.min_len]"
    )
    mock_client = _make_mock_client(underlying)

    with (
        patch("flyte._initialize._get_init_config", return_value=cfg),
        patch("flyte.remote._data.get_client", return_value=mock_client),
    ):
        with pytest.raises(RuntimeSystemError) as exc_info:
            await _upload_single_file(cfg, test_file, basedir="")

        assert "project_id.organization" in str(exc_info.value)
        assert exc_info.value.__cause__ is underlying
