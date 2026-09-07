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


def _interface(**types):
    """A TypedInterface with one input per keyword, e.g. _interface(n="INTEGER", s="STRING")."""
    from flyteidl2.core import interface_pb2, types_pb2

    return interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(
                    key=name,
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=getattr(types_pb2.SimpleType, t))),
                )
                for name, t in types.items()
            ]
        )
    )


def _invoke_with_interface(args, interface, runner_obj=None):
    """Invoke `flyte rerun` with the source task's interface stubbed out."""
    runner_obj = runner_obj or _mock_runner()
    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
        mock.patch("flyte.cli._rerun._fetch_source_interface", AsyncMock(return_value=interface)) as fetch,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, args)
    return result, runner_obj, fetch


def test_rerun_builds_an_option_per_input_of_the_source_task():
    """Inputs become options the way `flyte run` builds them, typed by the source interface."""
    result, runner_obj, fetch = _invoke_with_interface(
        ["my-run", "--n", "10", "--s", "hello"], _interface(n="INTEGER", s="STRING")
    )

    assert result.exit_code == 0, result.output
    fetch.assert_awaited_once_with("my-run", "a0")
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    # Converted to native types by click, not forwarded as strings.
    assert kwargs["n"] == 10
    assert kwargs["s"] == "hello"


def test_rerun_omitted_inputs_are_not_sent_at_all():
    """An input left out keeps the source run's value, so it must not be sent as a default."""
    result, runner_obj, _ = _invoke_with_interface(["my-run", "--n", "10"], _interface(n="INTEGER", s="STRING"))

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["n"] == 10
    assert "s" not in kwargs


def test_rerun_bool_input_can_be_set_to_either_value():
    """`--flag/--no-flag`: on rerun "not passed" means "keep the prior value", so False has to be
    expressible on its own."""
    result, runner_obj, _ = _invoke_with_interface(["my-run", "--flag"], _interface(flag="BOOLEAN"))
    assert result.exit_code == 0, result.output
    assert runner_obj.rerun.aio.call_args.kwargs["flag"] is True

    result, runner_obj, _ = _invoke_with_interface(["my-run", "--no-flag"], _interface(flag="BOOLEAN"))
    assert result.exit_code == 0, result.output
    assert runner_obj.rerun.aio.call_args.kwargs["flag"] is False

    result, runner_obj, _ = _invoke_with_interface(["my-run"], _interface(flag="BOOLEAN"))
    assert result.exit_code == 0, result.output
    assert "flag" not in runner_obj.rerun.aio.call_args.kwargs


def test_rerun_inputs_compose_with_recover():
    """--recover and new inputs are supported together."""
    result, runner_obj, _ = _invoke_with_interface(
        ["my-run", "--recover", "--n", "10", "--force-rerun-action", "a3"], _interface(n="INTEGER")
    )

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["recover"] is True
    assert kwargs["n"] == 10
    assert kwargs["force_rerun_actions"] == ("a3",)


def test_rerun_without_inputs_skips_the_interface_fetch():
    """Nothing left over on the command line means no input was passed, so no extra round trip."""
    result, runner_obj, fetch = _invoke_with_interface(["my-run", "--name", "n"], _interface(n="INTEGER"))

    assert result.exit_code == 0, result.output
    fetch.assert_not_awaited()
    runner_obj.rerun.aio.assert_awaited_once_with(
        "my-run",
        action_name="a0",
        recover=False,
        force_rerun_actions=None,
        allow_missing_source_outputs=False,
    )


def test_rerun_inputs_are_read_off_the_selected_action():
    """With --action-name, the options come from that action's interface."""
    result, runner_obj, fetch = _invoke_with_interface(
        ["my-run", "--action-name", "a3", "--n", "10"], _interface(n="INTEGER")
    )

    assert result.exit_code == 0, result.output
    fetch.assert_awaited_once_with("my-run", "a3")
    assert runner_obj.rerun.aio.call_args.kwargs["n"] == 10


def test_rerun_own_options_are_not_shadowed_by_task_inputs():
    """A task input named like one of rerun's own options stays rerun's: `--name` names the new
    run. That input simply keeps the prior run's value."""
    result, runner_obj, _ = _invoke_with_interface(
        ["my-run", "--name", "retry-1", "--n", "10"], _interface(name="STRING", n="INTEGER")
    )

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["n"] == 10
    assert "name" not in kwargs


def test_rerun_rejects_an_input_the_source_task_does_not_have():
    result, _runner_obj, _ = _invoke_with_interface(["my-run", "--nope", "1"], _interface(n="INTEGER"))

    assert result.exit_code != 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "No such option" in plain


def test_rerun_help_for_a_run_lists_its_inputs():
    """`flyte rerun <run> --help` is how the available inputs are discovered."""
    result, _runner_obj, _ = _invoke_with_interface(["my-run", "--help"], _interface(n="INTEGER"))

    assert result.exit_code == 0, result.output
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--n" in plain


def test_rerun_help_without_a_run_needs_no_platform_call():
    with mock.patch("flyte.cli._rerun._fetch_source_interface", AsyncMock()) as fetch:
        result = CliRunner().invoke(rerun, ["--help"])
    assert result.exit_code == 0, result.output
    fetch.assert_not_awaited()


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
