import sys
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main

# flyte.cli/__init__.py rebinds the `main` attribute to the click group, so
# patch("flyte.cli.main.CLIConfig") resolves to the group, not the module.
_main_module = sys.modules["flyte.cli.main"]


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


@patch("flyte.remote.Secret.delete")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_delete_secret_no_scope_flags_does_not_inherit_config(mock_cli_config, mock_secret_delete, runner: CliRunner):
    """Without --project/--domain, CLI must pass empty strings so the config default does not leak in."""
    result = runner.invoke(main, ["delete", "secret", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="", domain="")
    mock_secret_delete.assert_called_once_with(name="my_secret", cluster_pool=None)


@patch("flyte.remote.Secret.delete")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_delete_secret_domain_only(mock_cli_config, mock_secret_delete, runner: CliRunner):
    """--domain alone must not inherit project from the config."""
    result = runner.invoke(main, ["delete", "secret", "--domain", "development", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="", domain="development")
    mock_secret_delete.assert_called_once_with(name="my_secret", cluster_pool=None)


@patch("flyte.remote.Secret.delete")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_delete_secret_project_and_domain(mock_cli_config, mock_secret_delete, runner: CliRunner):
    result = runner.invoke(
        main,
        ["delete", "secret", "--project", "my-proj", "--domain", "development", "my_secret"],
    )
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="my-proj", domain="development")
    mock_secret_delete.assert_called_once_with(name="my_secret", cluster_pool=None)


def test_delete_secret_missing_name(runner: CliRunner):
    result = runner.invoke(main, ["delete", "secret"])
    assert result.exit_code != 0


@patch("flyte.remote.Secret.delete")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_delete_secret_with_cluster_pool(mock_cli_config, mock_secret_delete, runner: CliRunner):
    result = runner.invoke(main, ["delete", "secret", "--cluster-pool", "pool-a", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="", domain="")
    mock_secret_delete.assert_called_once_with(name="my_secret", cluster_pool="pool-a")


def test_delete_secret_cluster_pool_rejects_project(runner: CliRunner):
    result = runner.invoke(
        main,
        ["delete", "secret", "--cluster-pool", "pool-a", "--project", "p", "my_secret"],
    )
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "project" in result.stderr


def test_delete_secret_cluster_pool_rejects_domain(runner: CliRunner):
    result = runner.invoke(
        main,
        ["delete", "secret", "--cluster-pool", "pool-a", "--domain", "d", "my_secret"],
    )
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "domain" in result.stderr
