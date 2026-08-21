"""Tests for the `flyte rerun <run>` CLI command."""

import re
from unittest import mock

import pytest
import rich_click as click
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
    assert "--input" in opts
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
        inputs=None,
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


def test_rerun_input_is_converted_and_passed_through():
    """`--input k=v` values are converted against the source task's interface, not sent raw."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
        mock.patch("flyte.cli._rerun._resolve_inputs", AsyncMock(return_value={"n": 10})) as resolve,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--input", "n=10"])

    assert result.exit_code == 0, result.output
    resolve.assert_awaited_once_with("my-run", "a0", {"n": "10"})
    assert runner_obj.rerun.aio.call_args.kwargs["inputs"] == {"n": 10}


def test_rerun_input_composes_with_recover():
    """--recover and new inputs are supported together."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
        mock.patch("flyte.cli._rerun._resolve_inputs", AsyncMock(return_value={"n": 10})),
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--recover", "--input", "n=10", "--force-rerun-action", "a3"])

    assert result.exit_code == 0, result.output
    kwargs = runner_obj.rerun.aio.call_args.kwargs
    assert kwargs["recover"] is True
    assert kwargs["inputs"] == {"n": 10}
    assert kwargs["force_rerun_actions"] == ("a3",)


def test_rerun_input_reads_interface_of_the_selected_action():
    """With --action-name, inputs are converted against that action's interface."""
    runner_obj = _mock_runner()

    with (
        mock.patch("flyte.cli._common.initialize_config") as init_cfg,
        mock.patch("flyte.with_runcontext", return_value=runner_obj),
        mock.patch("flyte.cli._rerun._resolve_inputs", AsyncMock(return_value={"n": 10})) as resolve,
    ):
        init_cfg.return_value = mock.MagicMock(output_format="table")
        result = CliRunner().invoke(rerun, ["my-run", "--action-name", "a3", "--input", "n=10"])

    assert result.exit_code == 0, result.output
    resolve.assert_awaited_once_with("my-run", "a3", {"n": "10"})


def test_rerun_input_requires_key_value():
    with mock.patch("flyte.cli._common.initialize_config"):
        result = CliRunner().invoke(rerun, ["my-run", "--input", "justakey"])
    assert result.exit_code != 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "expected KEY=VALUE" in plain


@pytest.mark.asyncio
async def test_resolve_inputs_converts_against_the_remote_interface():
    """Values arrive as strings from the shell and are parsed with the source task's types."""
    from flyteidl2.core import identifier_pb2, interface_pb2, tasks_pb2, types_pb2
    from flyteidl2.task import task_definition_pb2
    from flyteidl2.workflow import run_definition_pb2

    from flyte.cli._rerun import _resolve_inputs

    iface = interface_pb2.TypedInterface(
        inputs=interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(
                    key="n",
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)),
                ),
                interface_pb2.VariableEntry(
                    key="s",
                    value=interface_pb2.Variable(type=types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)),
                ),
            ]
        )
    )
    action = mock.MagicMock()
    action.pb2 = run_definition_pb2.ActionDetails(
        task=task_definition_pb2.TaskSpec(
            task_template=tasks_pb2.TaskTemplate(
                id=identifier_pb2.Identifier(name="test.task1", version="v1"), interface=iface
            )
        )
    )
    run_details = mock.MagicMock(action_details=action)

    with mock.patch("flyte.remote._run.RunDetails") as RD:
        RD.get.aio = AsyncMock(return_value=run_details)
        converted = await _resolve_inputs("my-run", "a0", {"n": "10", "s": "hello"})
        assert converted == {"n": 10, "s": "hello"}

        with pytest.raises(click.BadParameter, match="Unknown input 'nope'"):
            await _resolve_inputs("my-run", "a0", {"nope": "1"})


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
