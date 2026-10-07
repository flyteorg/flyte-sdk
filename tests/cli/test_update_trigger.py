import re
import sys
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# flyte.cli/__init__.py rebinds the `main` attribute to the click group, so
# patch("flyte.cli.main.CLIConfig") resolves to the group, not the module.
_main_module = sys.modules["flyte.cli.main"]


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


@patch("flyte.remote.Trigger.update")
@patch("flyte.remote.Trigger.promote", return_value=(Mock(), "v1.4.0"))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_update_trigger_to_version(mock_cli_config, mock_promote, mock_update, runner: CliRunner):
    result = runner.invoke(main, ["update", "trigger", "prod", "event_driven.ingest", "--to-version", "v1.6.0"])

    assert result.exit_code == 0, result.output
    mock_promote.assert_called_once_with("prod", "event_driven.ingest", "v1.6.0")
    mock_update.assert_not_called()
    assert "v1.4.0 -> v1.6.0" in _ANSI_RE.sub("", result.output)


@patch("flyte.remote.Trigger.update")
@patch("flyte.remote.Trigger.promote", return_value=(Mock(), "v1.6.0"))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_update_trigger_to_current_version(mock_cli_config, mock_promote, mock_update, runner: CliRunner):
    result = runner.invoke(main, ["update", "trigger", "prod", "event_driven.ingest", "--to-version", "v1.6.0"])

    assert result.exit_code == 0, result.output
    assert "pinned to v1.6.0" in _ANSI_RE.sub("", result.output)


@patch("flyte.remote.Trigger.update")
@patch("flyte.remote.Trigger.promote")
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_update_trigger_activate_only(mock_cli_config, mock_promote, mock_update, runner: CliRunner):
    result = runner.invoke(main, ["update", "trigger", "prod", "event_driven.ingest", "--activate"])

    assert result.exit_code == 0, result.output
    mock_update.assert_called_once_with("prod", "event_driven.ingest", True)
    mock_promote.assert_not_called()


def test_update_trigger_requires_an_action(runner: CliRunner):
    result = runner.invoke(main, ["update", "trigger", "prod", "event_driven.ingest"])

    assert result.exit_code != 0
    assert "--to-version" in result.output
