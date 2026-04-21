from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


@patch("flyte.remote.Secret.delete")
@patch("flyte.cli.main.CLIConfig", return_value=Mock())
def test_delete_secret_no_scope_flags_does_not_inherit_config(mock_cli_config, mock_secret_delete, runner: CliRunner):
    """Without --project/--domain, CLI must pass empty strings so the config default does not leak in."""
    result = runner.invoke(main, ["delete", "secret", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="", domain="")
    mock_secret_delete.assert_called_once_with(name="my_secret")


@patch("flyte.remote.Secret.delete")
@patch("flyte.cli.main.CLIConfig", return_value=Mock())
def test_delete_secret_domain_only(mock_cli_config, mock_secret_delete, runner: CliRunner):
    """--domain alone must not inherit project from the config."""
    result = runner.invoke(main, ["delete", "secret", "--domain", "development", "my_secret"])
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="", domain="development")
    mock_secret_delete.assert_called_once_with(name="my_secret")


@patch("flyte.remote.Secret.delete")
@patch("flyte.cli.main.CLIConfig", return_value=Mock())
def test_delete_secret_project_and_domain(mock_cli_config, mock_secret_delete, runner: CliRunner):
    result = runner.invoke(
        main,
        ["delete", "secret", "--project", "my-proj", "--domain", "development", "my_secret"],
    )
    assert result.exit_code == 0, result.stderr
    mock_cli_config.return_value.init.assert_called_once_with(project="my-proj", domain="development")
    mock_secret_delete.assert_called_once_with(name="my_secret")


def test_delete_secret_missing_name(runner: CliRunner):
    result = runner.invoke(main, ["delete", "secret"])
    assert result.exit_code != 0
