"""`--profile` on the CLI, and `flyte create config --profile`."""

from __future__ import annotations

import pathlib

import pytest
import rich_click as click
import yaml
from click.testing import CliRunner

import flyte.config as config
from flyte.cli._create import config as create_config
from flyte.cli.main import main


@pytest.fixture(autouse=True)
def reset_active_profile():
    yield
    config.set_active_profile(None)


def test_profile_option_exists() -> None:
    assert "profile" in {p.name for p in main.params}


def test_profile_option_has_no_short_flag_colliding_with_project() -> None:
    """`-p` is `--project` throughout this CLI; profile must not shadow it."""
    (profile_opt,) = [p for p in main.params if p.name == "profile"]
    assert "-p" not in profile_opt.opts


def test_unknown_profile_is_a_usage_error(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("admin:\n  endpoint: dns:///a.example.com\n")
    result = CliRunner().invoke(main, ["--config", str(cfg), "--profile", "nope", "whoami"])
    assert result.exit_code != 0
    assert "nope" in result.output


def test_create_config_writes_top_level_by_default(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "config.yaml"
    result = CliRunner().invoke(
        create_config, ["--output", str(out), "--endpoint", "dns:///a.example.com", "--project", "p"]
    )
    assert result.exit_code == 0, result.output
    doc = yaml.safe_load(out.read_text())
    assert doc["admin"]["endpoint"] == "dns:///a.example.com"
    assert "profiles" not in doc


def test_create_config_profile_merges_into_existing_file(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "config.yaml"
    runner = CliRunner()

    r = runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///default.example.com", "--project", "shared"],
    )
    assert r.exit_code == 0, r.output

    r = runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///prod.example.com", "--profile", "prod"],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///gpu.example.com", "--profile", "gpu"],
    )
    assert r.exit_code == 0, r.output

    doc = yaml.safe_load(out.read_text())
    # Top-level defaults survived both profile writes.
    assert doc["admin"]["endpoint"] == "dns:///default.example.com"
    assert doc["task"]["project"] == "shared"
    # And both profiles are present -- the second did not clobber the first.
    assert set(doc["profiles"]) == {"prod", "gpu"}
    assert doc["profiles"]["prod"]["admin"]["endpoint"] == "dns:///prod.example.com"
    assert doc["profiles"]["gpu"]["admin"]["endpoint"] == "dns:///gpu.example.com"


def test_create_config_profile_is_readable_back(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "config.yaml"
    runner = CliRunner()
    runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///default.example.com", "--project", "shared", "--domain", "dev"],
    )
    runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///prod.example.com", "--domain", "production", "--profile", "prod"],
    )

    cfg = config.auto(out, profile="prod")
    assert cfg.platform.endpoint == "dns:///prod.example.com"
    assert cfg.task.domain == "production"
    assert cfg.task.project == "shared"  # inherited


def test_create_config_profile_does_not_prompt_to_overwrite(tmp_path: pathlib.Path) -> None:
    """A profile write merges, so the overwrite confirm would be misleading. Empty stdin would
    hang or abort if a prompt were shown."""
    out = tmp_path / "config.yaml"
    runner = CliRunner()
    runner.invoke(create_config, ["--output", str(out), "--endpoint", "dns:///a.example.com"])
    r = runner.invoke(
        create_config,
        ["--output", str(out), "--endpoint", "dns:///b.example.com", "--profile", "p2"],
        input="",
    )
    assert r.exit_code == 0, r.output
    assert "Overwrite" not in r.output


def test_create_config_profile_rejects_non_mapping_file(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "config.yaml"
    out.write_text("- not\n- a mapping\n")
    r = CliRunner().invoke(
        create_config, ["--output", str(out), "--endpoint", "dns:///a.example.com", "--profile", "p"]
    )
    assert r.exit_code != 0
    assert "mapping" in r.output


def test_profile_reaches_cli_state(tmp_path: pathlib.Path) -> None:
    """`--profile` must land in the CLIConfig the subcommands read, not just be parsed."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "task:\n  project: shared\n  domain: development\nprofiles:\n  prod:\n    task:\n      domain: production\n"
    )
    seen = {}

    @click.command("probe")
    @click.pass_obj
    def probe(obj):
        seen["profile"] = obj.profile
        seen["config_profile"] = obj.config.profile
        seen["domain"] = obj.config.task.domain

    main.add_command(probe)
    try:
        result = CliRunner().invoke(main, ["--config", str(cfg), "--profile", "prod", "probe"])
        assert result.exit_code == 0, result.output
        assert seen == {"profile": "prod", "config_profile": "prod", "domain": "production"}
    finally:
        main.commands.pop("probe", None)


def test_profile_sets_the_active_profile_for_lazy_readers(tmp_path: pathlib.Path) -> None:
    """Config reads that do not go through the CLIConfig (e.g. a lazy `ImageConfig.auto()`) must
    still see the selected profile rather than silently falling back to the top level."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("image:\n  builder: local\nprofiles:\n  prod:\n    image:\n      builder: remote\n")
    seen = {}

    @click.command("probe")
    @click.pass_obj
    def probe(obj):
        from flyte.config._config import ImageConfig

        seen["active"] = config.get_active_profile()
        seen["builder"] = ImageConfig.auto(str(cfg)).builder

    main.add_command(probe)
    try:
        result = CliRunner().invoke(main, ["--config", str(cfg), "--profile", "prod", "probe"])
        assert result.exit_code == 0, result.output
        assert seen == {"active": "prod", "builder": "remote"}
    finally:
        main.commands.pop("probe", None)


def test_get_profiles_lists_and_marks_the_active_one(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "task:\n  project: shared\nprofiles:\n  prod:\n    task:\n      domain: production\n"
        "  gpu:\n    task:\n      project: gpu-proj\n"
    )
    result = CliRunner().invoke(main, ["--config", str(cfg), "--profile", "prod", "get", "profiles"])
    assert result.exit_code == 0, result.output
    assert "prod" in result.output
    assert "gpu" in result.output


def test_get_profiles_on_a_file_without_any(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("admin:\n  endpoint: dns:///a.example.com\n")
    result = CliRunner().invoke(main, ["--config", str(cfg), "get", "profiles"])
    assert result.exit_code == 0, result.output
    assert "No profiles declared" in result.output
