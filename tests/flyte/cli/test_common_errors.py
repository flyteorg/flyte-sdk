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


def _invoke_raising(exc: Exception):
    @click.command(cls=CommandBase)
    @click.pass_context
    def failing_command(ctx, **kwargs):
        raise exc

    with patch("os._exit"):
        return CliRunner().invoke(failing_command)


_URL = "https://www.union.ai/contact"


def test_org_limit_refusal_prints_the_limit_and_upgrade_path(org_limit_detail):
    """A plugin command's RPC (e.g. creating a cluster) refused at an org limit prints the readable message."""
    detail = org_limit_detail(limit="max_clusters", ceiling=2, current=2, requested=1, tier_default=2, upgrade_url=_URL)
    result = _invoke_raising(ConnectError(Code.FAILED_PRECONDITION, "org would exceed max_clusters", details=[detail]))

    assert result.exit_code != 0
    assert "Org limit max_clusters (2) reached: 2 in use, 1 more requested." in result.output
    assert _URL in result.output
    assert "RPC error" not in result.output


def test_concurrency_limit_refusal_says_to_wait(org_limit_detail):
    detail = org_limit_detail(limit="max_concurrent_runs", ceiling=1000, current=1000, requested=1, upgrade_url=_URL)
    result = _invoke_raising(ConnectError(Code.RESOURCE_EXHAUSTED, "at max_concurrent_runs", details=[detail]))

    assert "Wait for running work to finish" in result.output


def test_entitlement_refusal_prints_entitlement_not_permission_denied(org_entitlement_detail):
    detail = org_entitlement_detail(entitlement="private_control_plane_endpoint", upgrade_url=_URL)
    result = _invoke_raising(ConnectError(Code.PERMISSION_DENIED, "not entitled", details=[detail]))

    assert result.exit_code != 0
    assert "This org isn't entitled to private_control_plane_endpoint." in result.output
    assert "Permission denied" not in result.output


def test_sdk_org_errors_print_without_the_kind_and_code_prefix():
    import flyte.errors

    limit = flyte.errors.OrgLimit(limit="max_concurrent_runs", ceiling=1, current=1, requested=1)
    for exc in (
        flyte.errors.OrgLimitExceededError("Org limit reached."),
        flyte.errors.ResourceExhaustedError("Org limit reached.", org_limit=limit),
        flyte.errors.OrgEntitlementRequiredError("Org limit reached.", entitlement="byok"),
    ):
        result = _invoke_raising(exc)
        assert "│ Org limit reached." in result.output, result.output
        assert "user failure" not in result.output

    result = _invoke_raising(flyte.errors.ResourceExhaustedError("slow down"))
    assert "user failure, SlowDownError. slow down" in result.output
