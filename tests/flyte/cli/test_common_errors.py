"""Tests for ConnectError → click.ClickException translation in CLI invoke."""

from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from flyte.cli._common import CommandBase


@pytest.mark.parametrize(
    "code, expected_substring",
    [
        (Code.UNAUTHENTICATED, "Authentication failed"),
        (Code.NOT_FOUND, "NOT FOUND"),
        (Code.ALREADY_EXISTS, "already exists"),
        (Code.INTERNAL, "Internal server error"),
        (Code.UNAVAILABLE, "unavailable"),
        (Code.PERMISSION_DENIED, "Permission denied"),
        (Code.INVALID_ARGUMENT, "Invalid argument"),
        (Code.RESOURCE_EXHAUSTED, "RPC error invoking command"),  # fallback
    ],
)
def test_connect_error_translated_to_click_exception(code, expected_substring):
    """Each ConnectError code maps to a specific click.ClickException message."""

    @click.command(cls=CommandBase)
    @click.pass_context
    def failing_command(ctx, **kwargs):
        raise ConnectError(code, "test error")

    runner = CliRunner()
    with patch("os._exit"):  # Prevent os._exit(0) on success path
        result = runner.invoke(failing_command)
    assert result.exit_code != 0
    assert expected_substring in result.output
