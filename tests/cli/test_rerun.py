"""Tests for the `flyte rerun <run>` CLI command."""

from unittest import mock

from click.testing import CliRunner
from mock.mock import AsyncMock

from flyte.cli._rerun import _parse_kv, rerun
from flyte.cli.main import main


def test_rerun_registered_on_main():
    assert "rerun" in main.commands


def test_rerun_takes_run_name_and_no_recover_flag():
    # Takes the run name as a positional argument.
    assert any(p.name == "run_name" for p in rerun.params)
    # --recover is omitted until the backend ships (see TODO in _rerun.py).
    opts = {o for p in rerun.params for o in p.opts}
    assert "--recover" not in opts


def test_parse_kv():
    assert _parse_kv((), "--env") is None
    assert _parse_kv(("A=1", "B=2"), "--env") == {"A": "1", "B": "2"}


def test_rerun_delegates_to_runner_rerun():
    """`flyte rerun <run> --name n -e K=V` builds the run context and calls runner.rerun(run)."""
    runner_obj = mock.MagicMock()
    runner_obj.rerun = mock.MagicMock(return_value=mock.MagicMock())
    runner_obj.rerun.aio = AsyncMock(return_value=mock.MagicMock(name="new", url="http://x"))

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--name", "n", "-e", "K=V"])

    assert result.exit_code == 0, result.output
    # env parsed, name forwarded; recover is not wired on the CLI yet.
    _, kwargs = wrc.call_args
    assert "recover" not in kwargs
    assert kwargs["name"] == "n"
    assert kwargs["env_vars"] == {"K": "V"}
    assert kwargs["mode"] == "remote"
    runner_obj.rerun.aio.assert_awaited_once_with("my-run")
