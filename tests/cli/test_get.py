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


@patch("flyte.cli._get.remote.Run.get")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_code_with_dest_downloads_and_extracts(mock_cli_config, mock_run_get, runner: CliRunner, tmp_path):
    run = Mock()
    run.download_code.return_value = tmp_path / "code"
    mock_run_get.return_value = run

    result = runner.invoke(main, ["get", "code", "my_run", "--dest", str(tmp_path / "code")])

    assert result.exit_code == 0, result.stderr
    mock_run_get.assert_called_once_with(name="my_run")
    run.download_code.assert_called_once_with(dest=tmp_path / "code", extract=True, attempt=None)


@patch("flyte.cli._get.remote.Action.get")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_code_for_an_action_keeps_the_archive(mock_cli_config, mock_action_get, runner: CliRunner, tmp_path):
    action = Mock()
    action.download_code.return_value = tmp_path / "code" / "fast123.tar.gz"
    mock_action_get.return_value = action

    result = runner.invoke(
        main,
        ["get", "code", "my_run", "my_action", "--dest", str(tmp_path / "code"), "--no-extract", "--attempt", "2"],
    )

    assert result.exit_code == 0, result.stderr
    mock_action_get.assert_called_once_with(run_name="my_run", name="my_action")
    action.download_code.assert_called_once_with(dest=tmp_path / "code", extract=False, attempt=2)


@patch("flyte.cli._get.remote.Run.get")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_code_without_dest_lists_the_bundle(mock_cli_config, mock_run_get, runner: CliRunner, tmp_path):
    """With no --dest the bundle is fetched to a scratch dir, listed, and discarded."""
    import tarfile

    source = tmp_path / "main.py"
    source.write_text("print('hi')\n")
    archive = tmp_path / "fast123.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="workflows/main.py")

    run = Mock()
    run.download_code.return_value = archive
    mock_run_get.return_value = run

    result = runner.invoke(main, ["get", "code", "my_run"])

    assert result.exit_code == 0, result.stderr
    assert "workflows/main.py" in result.output
    # Fetched without extracting, into a directory the command owns.
    kwargs = run.download_code.call_args.kwargs
    assert kwargs["extract"] is False
    assert kwargs["dest"] != tmp_path


def test_get_code_no_extract_requires_dest(runner: CliRunner):
    result = runner.invoke(main, ["get", "code", "my_run", "--no-extract"])
    assert result.exit_code != 0
    assert "--no-extract" in result.stderr
