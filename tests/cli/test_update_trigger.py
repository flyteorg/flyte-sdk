from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(
    ("flag", "active", "state"), [("--activate", True, "active"), ("--deactivate", False, "inactive")]
)
@patch("flyte.remote.Trigger.update")
@patch("flyte.cli._common.CLIConfig", return_value=Mock())
def test_update_trigger_reports_new_state(mock_cli_config, mock_update, runner, flag, active, state):
    result = runner.invoke(main, ["update", "trigger", "my-trigger", "my_task", flag])
    assert result.exit_code == 0, result.output
    mock_update.assert_called_once_with("my-trigger", "my_task", active)
    assert f"Trigger my-trigger for task my_task is now {state}." in result.output
    assert "set to deactivate" not in result.output
