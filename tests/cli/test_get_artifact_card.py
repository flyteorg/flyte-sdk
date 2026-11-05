"""`flyte get artifact --download-card`: which version it selects and where it writes."""

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
    art.name = "trained-model"
    art.version = "v1"
    art.download_card = Mock(return_value=local)
    return art


@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_download_the_card_of_a_pinned_version(mock_cli_config, runner: CliRunner, tmp_path: Path):
    art = _artifact(tmp_path / "card.html")
    with patch("flyte.cli._get.remote.Artifact.get", return_value=art) as mock_get:
        result = runner.invoke(main, ["get", "artifact", "trained-model", "v1", "--download-card", str(tmp_path)])

    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with("trained-model", version="v1", project=None, domain=None, partitions=None)
    art.download_card.assert_called_once_with(dest=tmp_path)
    assert "Downloaded the card" in result.output


@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_download_the_card_of_the_latest_version_of_one_partition(mock_cli_config, runner: CliRunner, tmp_path: Path):
    art = _artifact(tmp_path / "card.html")
    with patch("flyte.cli._get.remote.Artifact.get", return_value=art) as mock_get:
        result = runner.invoke(
            main,
            ["get", "artifact", "raw_events", "--partition", "date=2026-08-01", "--download-card", str(tmp_path)],
        )

    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with(
        "raw_events", version="latest", project=None, domain=None, partitions={"date": date(2026, 8, 1)}
    )


def test_download_card_needs_a_name(runner: CliRunner):
    result = runner.invoke(main, ["get", "artifact", "--download-card", "./cards"])
    assert result.exit_code != 0
    assert "needs an artifact name" in result.output


def test_download_card_needs_one_version(runner: CliRunner):
    result = runner.invoke(
        main, ["get", "artifact", "raw_events", "--partition", "region=us,eu", "--download-card", "./cards"]
    )
    assert result.exit_code != 0
    assert "needs one version" in result.output
