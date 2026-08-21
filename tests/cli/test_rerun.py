"""Tests for the `flyte rerun <run>` CLI command."""

import re
from unittest import mock

from click.testing import CliRunner
from mock.mock import AsyncMock

from flyte.cli._rerun import _parse_kv, rerun
from flyte.cli.main import main


def test_rerun_registered_on_main():
    assert "rerun" in main.commands


def test_recover_is_not_a_separate_command():
    """Recovery is a flag on rerun, not its own verb."""
    assert "recover" not in main.commands


def test_rerun_options():
    opts = {o for p in rerun.params for o in p.opts}
    assert "--recover" in opts
    assert "--force-rerun-action" in opts
    assert "--allow-missing-outputs" in opts
    # Takes the run name as a positional argument.
    assert any(p.name == "run_name" for p in rerun.params)


def test_parse_kv():
    assert _parse_kv((), "--env") is None
    assert _parse_kv(("A=1", "B=2"), "--env") == {"A": "1", "B": "2"}


def _mock_runner():
    runner_obj = mock.MagicMock()
    runner_obj.rerun.aio = AsyncMock(return_value=mock.MagicMock(name="new", url="http://x"))
    return runner_obj


def test_rerun_delegates_to_runner_rerun():
    """`flyte rerun <run> --name n -e K=V` builds the run context and calls runner.rerun(run)."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--name", "n", "-e", "K=V"])

    assert result.exit_code == 0, result.output
    # Recover options live on rerun(), not on the run context.
    kwargs = wrc.call_args.kwargs
    assert kwargs["name"] == "n"
    assert kwargs["env_vars"] == {"K": "V"}
    assert kwargs["mode"] == "remote"
    assert "recover" not in kwargs
    assert "recover_force_rerun_actions" not in kwargs
    runner_obj.rerun.aio.assert_awaited_once_with(
        "my-run",
        action_name="a0",
        recover=False,
        force_rerun_actions=None,
        allow_missing_source_outputs=False,
    )


def test_rerun_recover_flag_passed_to_rerun():
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--recover"])

    assert result.exit_code == 0, result.output
    assert runner_obj.rerun.aio.call_args.kwargs["recover"] is True


def test_rerun_force_rerun_action_requires_recover():
    with mock.patch("flyte.cli._common.initialize_config"):
        result = CliRunner().invoke(rerun, ["my-run", "--force-rerun-action", "a1"])
    assert result.exit_code != 0
    # rich-click may style the flag names with ANSI codes (e.g. on CI); strip before matching.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--force-rerun-action requires --recover" in plain


def test_rerun_force_rerun_action_passed_through():
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(
            rerun, ["my-run", "--recover", "--force-rerun-action", "a3", "--force-rerun-action", "a7"]
        )

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["recover"] is True
    assert kwargs["force_rerun_actions"] == ("a3", "a7")


def test_rerun_allow_missing_outputs_goes_to_rerun_not_run_context():
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--label", "team=ml", "--allow-missing-outputs"])

    assert result.exit_code == 0, result.output
    # Run-context options stay on with_runcontext; this one rides on rerun() now.
    assert wrc.call_args.kwargs["labels"] == {"team": "ml"}
    assert "allow_missing_source_outputs" not in wrc.call_args.kwargs
    assert runner_obj.rerun.aio.call_args.kwargs["allow_missing_source_outputs"] is True


def test_rerun_has_action_name_option():
    opts = {o for p in rerun.params for o in p.opts}
    assert "--action-name" in opts


def test_rerun_defaults_to_root_action():
    """Without --action-name the whole run is re-run, i.e. the root action a0."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run"])

    assert result.exit_code == 0, result.output
    assert runner_obj.rerun.aio.call_args.kwargs["action_name"] == "a0"


def test_rerun_action_name_passed_through():
    """--action-name selects which action supplies the task + inputs for the new run."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--action-name", "a3"])

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["action_name"] == "a3"
    # Re-running one action is always a plain re-execution.
    assert kwargs["recover"] is False


def test_rerun_action_name_is_mutually_exclusive_with_recover():
    with mock.patch("flyte.cli._common.initialize_config"):
        result = CliRunner().invoke(rerun, ["my-run", "--action-name", "a3", "--recover"])
    assert result.exit_code != 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--action-name cannot be combined with --recover" in plain


def test_rerun_action_name_composes_with_other_options():
    """--action-name is orthogonal to naming/env/labels on the new run."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj) as wrc,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--action-name", "a3", "--name", "just-a3", "-e", "K=V"])

    assert result.exit_code == 0, result.output
    assert runner_obj.rerun.aio.call_args.kwargs["action_name"] == "a3"
    assert wrc.call_args.kwargs["name"] == "just-a3"
    assert wrc.call_args.kwargs["env_vars"] == {"K": "V"}
