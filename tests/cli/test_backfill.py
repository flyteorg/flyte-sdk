import re
from datetime import datetime, timedelta, timezone

import mock
import pytest
from click.testing import CliRunner

from flyte.cli._backfill import _parse_time, backfill
from flyte.cli.main import main


def _plain(output: str) -> str:
    """rich-click styles its output; strip ANSI before matching."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def test_backfill_registered_on_main():
    assert "backfill" in main.commands


def test_backfill_takes_a_trigger_name_and_the_expected_options():
    opts = {o for p in backfill.params for o in p.opts}
    assert {"--from", "--to", "--force", "--suffix", "--dry-run", "--max-runs", "--queue"} <= opts
    assert any(p.name == "trigger_name" for p in backfill.params)


def test_from_is_required():
    result = CliRunner().invoke(backfill, ["nightly_eval"])
    assert result.exit_code != 0
    assert "--from" in _plain(result.output)


def test_suffix_requires_force():
    result = CliRunner().invoke(backfill, ["nightly_eval", "--from", "7d", "--suffix", "x"])
    assert result.exit_code != 0
    assert "--suffix requires --force" in _plain(result.output)


def test_window_must_be_ordered():
    result = CliRunner().invoke(backfill, ["nightly_eval", "--from", "2026-05-10", "--to", "2026-05-01"])
    assert result.exit_code != 0
    assert "earlier than" in _plain(result.output)


class TestParseTime:
    def test_iso_timestamp(self):
        assert _parse_time("2026-05-01T02:00", what="--from") == datetime(2026, 5, 1, 2, 0, tzinfo=timezone.utc)

    def test_plain_date(self):
        assert _parse_time("2026-05-01", what="--from") == datetime(2026, 5, 1, tzinfo=timezone.utc)

    def test_explicit_offset_is_preserved(self):
        parsed = _parse_time("2026-05-01T02:00+02:00", what="--from")
        assert parsed.utcoffset() == timedelta(hours=2)

    @pytest.mark.parametrize(
        ("value", "delta"), [("30d", timedelta(days=30)), ("12h", timedelta(hours=12)), ("45m", timedelta(minutes=45))]
    )
    def test_relative_ages(self, value, delta):
        before = datetime.now(timezone.utc) - delta
        parsed = _parse_time(value, what="--from")
        assert abs((parsed - before).total_seconds()) < 5

    def test_now(self):
        parsed = _parse_time("now", what="--to")
        assert abs((parsed - datetime.now(timezone.utc)).total_seconds()) < 5

    def test_garbage_is_rejected(self):
        with pytest.raises(Exception, match="ISO timestamp"):
            _parse_time("last tuesday", what="--from")

    def test_empty_is_none(self):
        assert _parse_time(None, what="--to") is None


def test_dry_run_never_launches_anything():
    """--dry-run must print the plan and stop short of creating runs."""
    from tests.backfill.test_plan import FakeTriggerDetails

    trigger = mock.MagicMock()
    trigger.get.aio = mock.AsyncMock(return_value=FakeTriggerDetails())

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.remote.Trigger", trigger),
        mock.patch("flyte.backfill._execute.probe_existing", mock.AsyncMock(return_value=set())),
        mock.patch("flyte.backfill._driver.launch_backfill") as launch,
        mock.patch("flyte._initialize.get_init_config") as init,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        init.return_value = mock.MagicMock(org="acme", project="ml-platform", domain="production")
        result = CliRunner().invoke(
            backfill,
            [
                "nightly_eval",
                "--task-name",
                "evals.weekly.run",
                "--from",
                "2026-05-01",
                "--to",
                "2026-05-05",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    launch.assert_not_called()
    assert "Dry run" in _plain(result.output)
