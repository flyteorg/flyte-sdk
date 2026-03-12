"""Tests for CLI create project, update project, and get project --state commands."""

from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from flyteidl2.project import project_service_pb2

from flyte.cli.main import main
from flyte.remote._project import Project


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


def _make_project_pb(id="test-proj", name="test-proj", state=project_service_pb2.PROJECT_STATE_ACTIVE):
    return project_service_pb2.Project(id=id, name=name, state=state)


# ---------------------------------------------------------------------------
# create project
# ---------------------------------------------------------------------------


class TestCreateProject:
    @patch("flyte.remote.Project.create")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_create_minimal(self, mock_cli_config, mock_create, runner):
        mock_create.return_value = Project(_make_project_pb())
        result = runner.invoke(main, ["create", "project", "--id", "my-project-id", "--name", "My Project"])
        assert result.exit_code == 0, result.output
        mock_create.assert_called_once_with(id="my-project-id", name="My Project", description="", labels=None)

    @patch("flyte.remote.Project.create")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_create_with_description(self, mock_cli_config, mock_create, runner):
        mock_create.return_value = Project(_make_project_pb())
        result = runner.invoke(
            main,
            ["create", "project", "--id", "my-project-id", "--name", "My Project", "--description", "A cool project"],
        )
        assert result.exit_code == 0, result.output
        mock_create.assert_called_once_with(
            id="my-project-id", name="My Project", description="A cool project", labels=None
        )

    @patch("flyte.remote.Project.create")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_create_with_labels(self, mock_cli_config, mock_create, runner):
        mock_create.return_value = Project(_make_project_pb())
        result = runner.invoke(
            main,
            [
                "create",
                "project",
                "--id",
                "my-project-id",
                "--name",
                "My Project",
                "--label",
                "team=ml",
                "--label",
                "env=prod",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_create.assert_called_once_with(
            id="my-project-id", name="My Project", description="", labels={"team": "ml", "env": "prod"}
        )

    def test_create_missing_id(self, runner):
        result = runner.invoke(main, ["create", "project", "--name", "My Project"])
        assert result.exit_code != 0

    def test_create_missing_name(self, runner):
        result = runner.invoke(main, ["create", "project", "--id", "my-project-id"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# update project
# ---------------------------------------------------------------------------


class TestUpdateProject:
    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_archive(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(main, ["update", "project", "my-project", "--archive"])
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(id="my-project", name=None, description=None, labels=None, state="archived")

    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_unarchive(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(main, ["update", "project", "my-project", "--unarchive"])
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(id="my-project", name=None, description=None, labels=None, state="active")

    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_description(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(main, ["update", "project", "my-project", "--description", "New desc"])
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(id="my-project", name=None, description="New desc", labels=None, state=None)

    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_name(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(main, ["update", "project", "my-project", "--name", "New Display Name"])
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(
            id="my-project", name="New Display Name", description=None, labels=None, state=None
        )

    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_labels(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(
            main, ["update", "project", "my-project", "--label", "team=ml", "--label", "env=staging"]
        )
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(
            id="my-project", name=None, description=None, labels={"team": "ml", "env": "staging"}, state=None
        )

    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_no_options_fails(self, mock_cli_config, runner):
        result = runner.invoke(main, ["update", "project", "my-project"])
        assert result.exit_code != 0
        assert "At least one of" in result.output

    @patch("flyte.remote.Project.update")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_update_multiple_options(self, mock_cli_config, mock_update, runner):
        mock_update.return_value = Project(_make_project_pb())
        result = runner.invoke(
            main, ["update", "project", "my-project", "--archive", "--description", "Archived project"]
        )
        assert result.exit_code == 0, result.output
        mock_update.assert_called_once_with(
            id="my-project", name=None, description="Archived project", labels=None, state="archived"
        )

    def test_update_missing_id(self, runner):
        result = runner.invoke(main, ["update", "project"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# get project --state
# ---------------------------------------------------------------------------


class TestGetProjectArchived:
    @patch("flyte.remote.Project.listall")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_default_lists_active(self, mock_cli_config, mock_listall, runner):
        mock_listall.return_value = iter([])
        result = runner.invoke(main, ["get", "project"])
        assert result.exit_code == 0, result.output
        mock_listall.assert_called_once_with(archived=False)

    @patch("flyte.remote.Project.listall")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_archived_flag(self, mock_cli_config, mock_listall, runner):
        mock_listall.return_value = iter([])
        result = runner.invoke(main, ["get", "project", "--archived"])
        assert result.exit_code == 0, result.output
        mock_listall.assert_called_once_with(archived=True)

    @patch("flyte.remote.Project.get")
    @patch("flyte.cli._common.CLIConfig", return_value=Mock())
    def test_get_by_name_ignores_archived(self, mock_cli_config, mock_get, runner):
        """When a name is given, --archived is not used—Project.get is called instead."""
        mock_get.return_value = Project(_make_project_pb(id="my-proj"))
        result = runner.invoke(main, ["get", "project", "my-proj"])
        assert result.exit_code == 0, result.output
        mock_get.assert_called_once_with("my-proj")
