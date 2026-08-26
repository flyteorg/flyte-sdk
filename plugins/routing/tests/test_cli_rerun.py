"""Resolving a profile for `rerun`, which needs it before Click parses.

`rerun` reads the source run's interface during `parse_args` so it can turn `--some-input v` into
a typed option. That read goes to the control plane holding the run, so an invoke-time hook is too
late -- the profile has to be settled first.
"""

from __future__ import annotations

import pathlib

import flyte.config as config
import pytest
import rich_click as click
from click.testing import CliRunner
from flyte.cli._common import CLIConfig
from flyte.cli._plugins import _apply_hook_to_command

from flyteplugins.routing._tags import make_run_name
from flyteplugins.routing.cli import peek_argument, resolve_profile_at_parse

_CONFIG = """
task:
  project: shared
  domain: development
profiles:
  us-east:
    task: {project: east-proj}
  gpu-pool:
    task: {project: gpu-proj}
"""


@pytest.fixture
def cfg_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "config.yaml"
    p.write_text(_CONFIG)
    return p


@pytest.fixture
def root(cfg_file: pathlib.Path):
    """A `rerun`-shaped command that, like the real one, needs config while parsing."""
    seen: dict = {}

    class FakeRerun(click.Command):
        def parse_args(self, ctx, args):
            # Mirrors RerunCommand: config is consulted here, not at invoke.
            seen["profile_at_parse"] = ctx.obj.config.profile if ctx.obj else None
            seen["project_at_parse"] = ctx.obj.config.task.project if ctx.obj else None
            return super().parse_args(ctx, args)

    @click.group()
    def rootgrp():
        pass

    @rootgrp.command("rerun", cls=FakeRerun)
    @click.argument("run_name", required=True)
    @click.option("-p", "--project", default=None)
    @click.option("--name", default=None)
    @click.option("--recover", is_flag=True, default=False)
    @click.pass_obj
    def rerun(obj, run_name, project, name, recover):
        seen["profile_at_invoke"] = obj.config.profile
        seen["run_name"] = run_name
        seen["recover"] = recover

    rootgrp.seen = seen  # type: ignore[attr-defined]
    return rootgrp


def _invoke(root, args, cfg_file):
    obj = CLIConfig(config=config.auto(cfg_file), ctx=None)  # type: ignore[arg-type]
    return CliRunner().invoke(root, args, obj=obj)


def test_profile_is_resolved_before_parsing(root, cfg_file: pathlib.Path) -> None:
    """The whole point: an invoke-time hook would leave parse_args on the wrong profile."""
    name = make_run_name("gpu-pool")
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))

    result = _invoke(root, ["rerun", name], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile_at_parse"] == "gpu-pool"
    assert root.seen["project_at_parse"] == "gpu-proj"
    assert root.seen["profile_at_invoke"] == "gpu-pool"


def test_recover_resolves_the_same_way(root, cfg_file: pathlib.Path) -> None:
    """`--recover` is a flag on rerun, so it needs no separate handling -- assert that it does."""
    name = make_run_name("gpu-pool")
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))

    result = _invoke(root, ["rerun", name, "--recover"], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile_at_parse"] == "gpu-pool"
    assert root.seen["recover"] is True


def test_options_before_the_run_name_do_not_confuse_the_peek(root, cfg_file: pathlib.Path) -> None:
    name = make_run_name("us-east")
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))

    result = _invoke(root, ["rerun", "--project", "someproject", name], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile_at_parse"] == "us-east"
    assert root.seen["run_name"] == name


def test_a_name_we_did_not_mint_leaves_the_ambient_profile(root, cfg_file: pathlib.Path) -> None:
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))
    result = _invoke(root, ["rerun", "somecontrolplanename"], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile_at_parse"] is None


def test_resolution_failure_does_not_break_the_command(root, cfg_file: pathlib.Path, monkeypatch) -> None:
    """A plugin fault must degrade to the ambient profile, not replace the command's own error."""
    import flyteplugins.routing.cli as cli_mod

    def boom(ctx, run_name):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(cli_mod, "_switch_profile", boom)
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))

    result = _invoke(root, ["rerun", "somerun"], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile_at_parse"] is None


def test_help_still_renders(root, cfg_file: pathlib.Path) -> None:
    _apply_hook_to_command(root, "rerun", resolve_profile_at_parse("run_name"))
    result = _invoke(root, ["rerun", "--help"], cfg_file)
    assert result.exit_code == 0
    assert "run_name" in result.output.lower() or "usage" in result.output.lower()


class TestPeekArgument:
    """Reading a positional out of raw argv, before Click has parsed it."""

    def _cmd(self) -> click.Command:
        @click.command()
        @click.argument("run_name")
        @click.argument("second", required=False)
        @click.option("-p", "--project")
        @click.option("--name")
        @click.option("--recover", is_flag=True)
        @click.option("-v", "--verbose", count=True)
        def c(**kw):
            pass

        return c

    @pytest.mark.parametrize(
        "args,expected",
        [
            (["my-run"], "my-run"),
            (["--project", "p", "my-run"], "my-run"),
            (["-p", "p", "my-run"], "my-run"),
            (["--project=p", "my-run"], "my-run"),
            (["--recover", "my-run"], "my-run"),
            (["-v", "my-run"], "my-run"),
            (["-vvv", "my-run"], "my-run"),
            (["my-run", "--project", "p"], "my-run"),
            (["my-run", "extra"], "my-run"),
            (["--name", "newname", "my-run"], "my-run"),
            (["--", "my-run"], "my-run"),
            ([], None),
            (["--project", "p"], None),
        ],
    )
    def test_finds_the_first_positional(self, args, expected) -> None:
        assert peek_argument(self._cmd(), args, "run_name") == expected

    def test_finds_a_later_positional(self) -> None:
        assert peek_argument(self._cmd(), ["--project", "p", "my-run", "other"], "second") == "other"

    def test_unknown_parameter_returns_none(self) -> None:
        assert peek_argument(self._cmd(), ["my-run"], "nope") is None
