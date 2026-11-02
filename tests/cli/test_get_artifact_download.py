"""`flyte get artifact --download`: which version it selects and where it writes."""

import sys
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from flyte.cli.main import main

_main_module = sys.modules["flyte.cli.main"]


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


def _artifact(local: Path) -> Mock:
    art = Mock()
    art.name = "scored-rollouts"
    art.version = "v1"
    art.download = Mock(return_value=local)
    return art


@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_download_a_pinned_version(mock_cli_config, runner: CliRunner, tmp_path: Path):
    art = _artifact(tmp_path / "data")
    with patch("flyte.cli._get.remote.Artifact.get", return_value=art) as mock_get:
        result = runner.invoke(main, ["get", "artifact", "scored-rollouts", "v1", "--download", str(tmp_path / "data")])

    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with("scored-rollouts", version="v1", project=None, domain=None, partitions=None)
    art.download.assert_called_once_with(dest=tmp_path / "data")
    assert "Downloaded" in result.output


@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_download_the_latest_version_of_one_partition(mock_cli_config, runner: CliRunner, tmp_path: Path):
    art = _artifact(tmp_path / "data")
    with patch("flyte.cli._get.remote.Artifact.get", return_value=art) as mock_get:
        result = runner.invoke(
            main,
            ["get", "artifact", "raw_events", "--partition", "date=2026-08-01", "--download", str(tmp_path / "data")],
        )

    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with(
        "raw_events", version="latest", project=None, domain=None, partitions={"date": date(2026, 8, 1)}
    )


def test_download_needs_a_name(runner: CliRunner):
    result = runner.invoke(main, ["get", "artifact", "--download", "./data"])
    assert result.exit_code != 0
    assert "needs an artifact name" in result.output


def test_download_needs_one_version(runner: CliRunner):
    result = runner.invoke(
        main, ["get", "artifact", "raw_events", "--partition", "region=us,eu", "--download", "./data"]
    )
    assert result.exit_code != 0
    assert "needs one version" in result.output
