"""The CLI hooks point run-addressed commands at the profile holding the run."""

from __future__ import annotations

import pathlib

import flyte.config as config
import pytest
import rich_click as click
from click.testing import CliRunner
from flyte.cli._common import CLIConfig
from flyte.cli._plugins import _apply_hook_to_subcommand

from flyteplugins.routing._tags import make_run_name
from flyteplugins.routing.cli import resolve_profile_for

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
    """A `get run`-shaped command tree, recording the profile the command actually ran under."""
    seen: dict = {}

    @click.group()
    def rootgrp():
        pass

    @click.group("get")
    def get():
        pass

    @get.command("run")
    @click.argument("name", required=False)
    @click.pass_obj
    def get_run(obj, name):
        seen["profile"] = obj.config.profile
        seen["project"] = obj.config.task.project
        seen["name"] = name

    rootgrp.add_command(get)
    rootgrp.seen = seen  # type: ignore[attr-defined]
    rootgrp.cfg_file = cfg_file  # type: ignore[attr-defined]
    return rootgrp


def _invoke(root, args, cfg_file):
    ctx_obj = CLIConfig(config=config.auto(cfg_file), ctx=None)  # type: ignore[arg-type]
    return CliRunner().invoke(root, args, obj=ctx_obj)


def test_tagged_name_switches_the_profile(root, cfg_file: pathlib.Path) -> None:
    name = make_run_name("gpu-pool")
    result = _invoke(root, ["get", "run", name], cfg_file)
    assert result.exit_code == 0, result.output

    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    result = _invoke(root, ["get", "run", name], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile"] == "gpu-pool"
    assert root.seen["project"] == "gpu-proj"


def test_hook_does_not_run_the_command_at_registration(root, cfg_file: pathlib.Path) -> None:
    """Regression against the SDK bug this hook style depends on: registering a subcommand hook
    used to invoke the command's callback immediately."""
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    assert root.seen == {}


def test_no_run_name_is_left_alone(root, cfg_file: pathlib.Path) -> None:
    """`flyte get run` with no argument is a listing -- there is nothing to resolve."""
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    result = _invoke(root, ["get", "run"], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile"] is None
    assert root.seen["project"] == "shared"


def test_command_still_receives_its_arguments(root, cfg_file: pathlib.Path) -> None:
    name = make_run_name("us-east")
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    _invoke(root, ["get", "run", name], cfg_file)
    assert root.seen["name"] == name


def test_a_name_we_did_not_mint_leaves_the_ambient_profile(root, cfg_file: pathlib.Path) -> None:
    """No tag, no search: the command runs against the default profile and reports not-found
    itself, rather than the plugin pre-empting it."""
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    result = _invoke(root, ["get", "run", "somecontrolplanename"], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile"] is None
    assert root.seen["project"] == "shared"


def test_a_colliding_tag_leaves_the_ambient_profile(root, cfg_file: pathlib.Path, monkeypatch) -> None:
    """With no search to disambiguate, the default beats a confident guess."""
    import flyteplugins.routing._resolve as resolve_mod

    monkeypatch.setattr(resolve_mod, "profiles_for_tag", lambda tag, profiles: ["us-east", "gpu-pool"])
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    result = _invoke(root, ["get", "run", make_run_name("us-east")], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile"] is None


def test_resolution_makes_no_network_calls(root, cfg_file: pathlib.Path, monkeypatch) -> None:
    """The read hooks must stay pure string work -- no client, no round trip per profile."""
    import flyte

    monkeypatch.setattr(flyte, "use_profile", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no client")))
    _apply_hook_to_subcommand(root, "get", "run", resolve_profile_for("name"))
    result = _invoke(root, ["get", "run", make_run_name("gpu-pool")], cfg_file)
    assert result.exit_code == 0, result.output
    assert root.seen["profile"] == "gpu-pool"


@pytest.fixture
def abort_root(cfg_file: pathlib.Path):
    """An `abort run`-shaped command: keyed on `run_name`, and it writes."""
    seen: dict = {}

    @click.group()
    def rootgrp():
        pass

    @click.group("abort")
    def abort():
        pass

    @abort.command("run")
    @click.argument("run_name", required=True)
    @click.pass_obj
    def abort_run(obj, run_name):
        seen["profile"] = obj.config.profile
        seen["project"] = obj.config.task.project

    rootgrp.add_command(abort)
    rootgrp.seen = seen  # type: ignore[attr-defined]
    return rootgrp


def test_write_commands_resolve_too(abort_root, cfg_file: pathlib.Path) -> None:
    """Aborting against the wrong control plane simply fails, so writes need resolution as much
    as reads do."""
    name = make_run_name("gpu-pool")
    _apply_hook_to_subcommand(abort_root, "abort", "run", resolve_profile_for("run_name"))

    result = _invoke(abort_root, ["abort", "run", name], cfg_file)
    assert result.exit_code == 0, result.output
    assert abort_root.seen["profile"] == "gpu-pool"
    assert abort_root.seen["project"] == "gpu-proj"
