"""Tests for `flyte start app` and `flyte stop app` CLI commands."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


class TestStartApp:
    @patch("flyte.remote.App.get")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_activate_called_with_wait(self, mock_cli_config, mock_app_get, runner):
        mock_app = Mock()
        mock_app.activate = Mock(return_value=mock_app)
        mock_app_get.return_value = mock_app

        result = runner.invoke(main, ["start", "app", "my-app"])

        assert result.exit_code == 0, result.output
        mock_app_get.assert_called_once_with(name="my-app", project=None, domain=None)
        mock_app.activate.assert_called_once_with(wait=True)

    @patch("flyte.remote.App.get")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_project_and_domain(self, mock_cli_config, mock_app_get, runner):
        mock_app = Mock()
        mock_app.activate = Mock(return_value=mock_app)
        mock_app_get.return_value = mock_app

        result = runner.invoke(
            main,
            ["start", "app", "my-app", "-p", "my-project", "-d", "development"],
        )

        assert result.exit_code == 0, result.output
        mock_app_get.assert_called_once_with(name="my-app", project="my-project", domain="development")
        mock_app.activate.assert_called_once_with(wait=True)


class TestStopApp:
    @patch("flyte.remote.App.get")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_deactivate_called_with_wait(self, mock_cli_config, mock_app_get, runner):
        mock_app = Mock()
        mock_app.deactivate = Mock(return_value=mock_app)
        mock_app_get.return_value = mock_app

        result = runner.invoke(main, ["stop", "app", "my-app"])

        assert result.exit_code == 0, result.output
        mock_app_get.assert_called_once_with(name="my-app", project=None, domain=None)
        mock_app.deactivate.assert_called_once_with(wait=True)

    @patch("flyte.remote.App.get")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_project_and_domain(self, mock_cli_config, mock_app_get, runner):
        mock_app = Mock()
        mock_app.deactivate = Mock(return_value=mock_app)
        mock_app_get.return_value = mock_app

        result = runner.invoke(
            main,
            ["stop", "app", "my-app", "-p", "my-project", "-d", "development"],
        )

        assert result.exit_code == 0, result.output
        mock_app_get.assert_called_once_with(name="my-app", project="my-project", domain="development")
        mock_app.deactivate.assert_called_once_with(wait=True)
