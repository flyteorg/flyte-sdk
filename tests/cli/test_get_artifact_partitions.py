"""`flyte get artifact --partition`: parsing, and which remote call each form makes."""

import sys
from datetime import date, datetime, timezone
from unittest.mock import Mock, patch

import click
import pytest
from click.testing import CliRunner

from flyte.cli._get import partition_callback
from flyte.cli.main import main

_main_module = sys.modules["flyte.cli.main"]


@pytest.fixture(scope="function")
def runner():
    return CliRunner()


class TestPartitionCallback:
    def test_date_and_string(self):
        assert partition_callback(None, "p", ["date=2026-08-01", "region=us"]) == {
            "date": date(2026, 8, 1),
            "region": "us",
        }

    def test_hour(self):
        assert partition_callback(None, "p", ["hour=2026-08-01T13"]) == {
            "hour": datetime(2026, 8, 1, 13, tzinfo=timezone.utc)
        }

    def test_range(self):
        assert partition_callback(None, "p", ["date=2026-08-01..2026-08-31"]) == {
            "date": (date(2026, 8, 1), date(2026, 8, 31))
        }

    def test_list(self):
        assert partition_callback(None, "p", ["region=us, eu"]) == {"region": ["us", "eu"]}

    def test_empty_is_none(self):
        assert partition_callback(None, "p", []) is None

    def test_bad_forms(self):
        with pytest.raises(click.BadParameter):
            partition_callback(None, "p", ["region"])
        with pytest.raises(click.BadParameter):
            partition_callback(None, "p", ["=us"])


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Artifact.get", return_value=Mock())
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_one_partition_gets_the_latest_version(mock_cli_config, mock_get, mock_format, runner: CliRunner):
    result = runner.invoke(
        main, ["get", "artifact", "raw_events", "--partition", "date=2026-08-01", "--partition", "region=us"]
    )
    assert result.exit_code == 0, result.output
    mock_get.assert_called_once_with("raw_events", project=None, domain=None, date=date(2026, 8, 1), region="us")


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Artifact.listall", return_value=iter([]))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_range_lists_latest_per_partition(mock_cli_config, mock_listall, mock_format, runner: CliRunner):
    result = runner.invoke(
        main,
        ["get", "artifact", "raw_events", "--partition", "date=2026-08-01..2026-08-31", "--latest-per-partition"],
    )
    assert result.exit_code == 0, result.output
    kwargs = mock_listall.call_args.kwargs
    assert kwargs["name"] == "raw_events"
    assert kwargs["latest_per_partition"] is True
    assert kwargs["partitions"] == {"date": (date(2026, 8, 1), date(2026, 8, 31))}


@patch("flyte.cli._get.common.format", return_value="")
@patch("flyte.cli._get.remote.Artifact.listall", return_value=iter([]))
@patch.object(_main_module, "CLIConfig", return_value=Mock())
def test_value_list_lists_versions(mock_cli_config, mock_listall, mock_format, runner: CliRunner):
    result = runner.invoke(main, ["get", "artifact", "raw_events", "--partition", "region=us,eu"])
    assert result.exit_code == 0, result.output
    kwargs = mock_listall.call_args.kwargs
    assert kwargs["partitions"] == {"region": ["us", "eu"]}
    assert kwargs["latest_per_partition"] is False


def test_partition_needs_a_name(runner: CliRunner):
    result = runner.invoke(main, ["get", "artifact", "--partition", "region=us"])
    assert result.exit_code != 0
    assert "needs an artifact name" in result.output


def test_version_and_partition_conflict(runner: CliRunner):
    result = runner.invoke(main, ["get", "artifact", "raw_events", "v1", "--partition", "region=us"])
    assert result.exit_code != 0
    assert "not both" in result.output


class TestListsOnTheTimeKey:
    def test_a_list_of_dates_is_rejected_with_the_fix(self):
        with pytest.raises(click.BadParameter, match="use a range \\(lo\\.\\.hi\\) or call once per value"):
            partition_callback(None, "p", ["date=2026-09-15,2026-09-16"])

    def test_a_list_of_strings_still_parses(self):
        assert partition_callback(None, "p", ["region=us,eu"]) == {"region": ["us", "eu"]}

    def test_a_single_time_value_stays_a_time_value(self):
        assert partition_callback(None, "p", ["date=2026-09-15"]) == {"date": date(2026, 9, 15)}
