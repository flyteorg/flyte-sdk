import sys
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main

_main_module = sys.modules["flyte.cli.main"]


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Secret.listall", return_value=iter([]))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_secrets_default_passes_no_cluster_pool(mock_cli_config, mock_listall, mock_format, runner: CliRunner):
    result = runner.invoke(main, ["get", "secret"])
    assert result.exit_code == 0, result.stderr
    mock_listall.assert_called_once_with(cluster_pool=None)


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Secret.get", return_value=Mock())
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_secret_by_name_passes_no_cluster_pool(mock_cli_config, mock_get, mock_format, runner: CliRunner):
    result = runner.invoke(main, ["get", "secret", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_get.assert_called_once_with("my_secret", cluster_pool=None)


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Secret.listall", return_value=iter([]))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_secrets_with_cluster_pool(mock_cli_config, mock_listall, mock_format, runner: CliRunner):
    result = runner.invoke(main, ["get", "secret", "--cluster-pool", "pool-a"])
    assert result.exit_code == 0, result.stderr
    mock_listall.assert_called_once_with(cluster_pool="pool-a")


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Secret.get", return_value=Mock())
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_secret_by_name_with_cluster_pool(mock_cli_config, mock_get, mock_format, runner: CliRunner):
    result = runner.invoke(main, ["get", "secret", "my_secret", "--cluster-pool", "pool-a"])
    assert result.exit_code == 0, result.stderr
    mock_get.assert_called_once_with("my_secret", cluster_pool="pool-a")


def test_get_secret_cluster_pool_rejects_project(runner: CliRunner):
    result = runner.invoke(main, ["get", "secret", "--cluster-pool", "pool-a", "--project", "p"])
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "project" in result.stderr


def test_get_secret_cluster_pool_rejects_domain(runner: CliRunner):
    result = runner.invoke(main, ["get", "secret", "--cluster-pool", "pool-a", "--domain", "d"])
    assert result.exit_code == 2
    assert "Illegal usage" in result.stderr
    assert "cluster_pool" in result.stderr
    assert "domain" in result.stderr
