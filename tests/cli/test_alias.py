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


@patch("flyte.remote.TaskAlias.set")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_update_alias_reports_the_move(mock_cli_config, mock_set, runner: CliRunner):
    mock_set.return_value = (Mock(version="v2"), "v1")
    result = runner.invoke(
        main, ["update", "alias", "prod", "my_env.my_task", "--to", "v2", "--project", "p", "--domain", "d"]
    )
    assert result.exit_code == 0, result.output
    mock_cli_config.return_value.init.assert_called_once_with("p", "d")
    mock_set.assert_called_once_with(task_name="my_env.my_task", alias="prod", version="v2", project="p", domain="d")
    assert "v1" in result.output
    assert "v2" in result.output


@patch("flyte.remote.TaskAlias.set")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_update_alias_reports_creation(mock_cli_config, mock_set, runner: CliRunner):
    mock_set.return_value = (Mock(version="v1"), "")
    result = runner.invoke(main, ["update", "alias", "prod", "my_env.my_task", "--to", "v1"])
    assert result.exit_code == 0, result.output
    mock_set.assert_called_once_with(task_name="my_env.my_task", alias="prod", version="v1", project=None, domain=None)
    assert "created" in result.output


def test_update_alias_requires_to(runner: CliRunner):
    result = runner.invoke(main, ["update", "alias", "prod", "my_env.my_task"])
    assert result.exit_code != 0
    assert "--to" in result.output


@patch("flyte.cli._common.format", return_value="")
@patch("flyte.remote.TaskAlias.listall")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_alias_lists_without_name(mock_cli_config, mock_listall, _format, runner: CliRunner):
    result = runner.invoke(main, ["get", "alias", "my_env.my_task"])
    assert result.exit_code == 0, result.output
    mock_listall.assert_called_once_with(task_name="my_env.my_task", project=None, domain=None, limit=100)


@patch("flyte.cli._common.format", return_value="")
@patch("flyte.remote.TaskAlias.get")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_alias_by_name(mock_cli_config, mock_get, _format, runner: CliRunner):
    result = runner.invoke(main, ["get", "alias", "my_env.my_task", "prod"])
    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with(task_name="my_env.my_task", alias="prod", project=None, domain=None)


@patch("flyte.cli._common.format", return_value="")
@patch("flyte.remote.TaskAlias.history")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_get_alias_history(mock_cli_config, mock_history, _format, runner: CliRunner):
    result = runner.invoke(main, ["get", "alias", "my_env.my_task", "prod", "--history", "--limit", "5"])
    assert result.exit_code == 0, result.output
    mock_history.assert_called_once_with(task_name="my_env.my_task", alias="prod", project=None, domain=None, limit=5)


def test_get_alias_history_requires_name(runner: CliRunner):
    result = runner.invoke(main, ["get", "alias", "my_env.my_task", "--history"])
    assert result.exit_code != 0
    assert "--history requires an alias name" in result.output


@patch("flyte.remote.TaskAlias.delete")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_delete_alias(mock_cli_config, mock_delete, runner: CliRunner):
    result = runner.invoke(main, ["delete", "alias", "prod", "my_env.my_task"])
    assert result.exit_code == 0, result.output
    mock_delete.assert_called_once_with(task_name="my_env.my_task", alias="prod", project=None, domain=None)


def test_alias_is_not_a_top_level_group(runner: CliRunner):
    result = runner.invoke(main, ["alias", "--help"])
    assert result.exit_code != 0
