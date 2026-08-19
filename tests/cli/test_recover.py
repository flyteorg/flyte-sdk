"""Tests for the `flyte recover <run>` CLI command (SDK-16)."""

from unittest import mock

from click.testing import CliRunner
from mock.mock import AsyncMock

from flyte.cli._recover import _parse_kv, recover
from flyte.cli.main import main


def test_recover_registered_on_main():
    assert "recover" in main.commands


def test_rerun_command_removed_from_main():
    """`flyte rerun` was folded into `flyte recover`; `flyte.rerun` stays as a Python API."""
    assert "rerun" not in main.commands


def test_recover_options():
    opts = {o for p in recover.params for o in p.opts}
    assert "--force-replay-action" in opts
    assert "--allow-missing-outputs" in opts
    # Recovery is never gated behind a flag on this verb — the verb *is* the opt-in.
    assert "--recover" not in opts
    assert "--force-rerun-action" not in opts
    # Takes the run name as a positional argument.
    assert any(p.name == "run_name" for p in recover.params)


def test_parse_kv():
    assert _parse_kv((), "--env") is None
    assert _parse_kv(("A=1", "B=2"), "--env") == {"A": "1", "B": "2"}


def _mock_runner():
    runner_obj = mock.MagicMock()
    runner_obj.rerun.aio = AsyncMock(return_value=mock.MagicMock(name="new", url="http://x"))
    return runner_obj


def test_recover_delegates_to_runner_rerun_with_recover_set():
    """`flyte recover <run> --name n -e K=V` builds the run context and recovers the run."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(recover, ["my-run", "--name", "n", "-e", "K=V"])

    assert result.exit_code == 0, result.output
    kwargs = wrc.call_args.kwargs
    # The verb always recovers — no flag needed.
    assert kwargs["recover"] is True
    assert kwargs["mode"] == "remote"
    assert kwargs["name"] == "n"
    assert kwargs["env_vars"] == {"K": "V"}
    assert kwargs["recover_force_rerun_actions"] is None
    assert kwargs["allow_missing_source_outputs"] is False
    runner_obj.rerun.aio.assert_awaited_once_with("my-run")


def test_recover_force_replay_action_passed_through():
    """--force-replay-action needs no gating flag and maps to recover_force_rerun_actions."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(recover, ["my-run", "--force-replay-action", "a3", "--force-replay-action", "a7"])

    assert result.exit_code == 0, result.output
    assert wrc.call_args.kwargs["recover"] is True
    assert wrc.call_args.kwargs["recover_force_rerun_actions"] == ("a3", "a7")


def test_recover_labels_and_allow_missing_outputs():
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(recover, ["my-run", "--label", "team=ml", "--allow-missing-outputs"])

    assert result.exit_code == 0, result.output
    assert wrc.call_args.kwargs["labels"] == {"team": "ml"}
    assert wrc.call_args.kwargs["allow_missing_source_outputs"] is True


def test_recover_reports_failure_without_traceback():
    runner_obj = mock.MagicMock()
    runner_obj.rerun.aio = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(recover, ["my-run"])

    assert result.exit_code == 0, result.output
    assert "Recovery failed" in result.output
