"""Errors for top-level `flyte` options passed after a subcommand."""

import re

import pytest
from click.testing import CliRunner

from flyte.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


def _flat(output: str) -> str:
    """The error text without rich's panel borders and line wrapping."""
    return " ".join(re.sub(r"[│╭╮╰╯─]", " ", output).split())


def test_image_builder_after_subcommand_says_where_it_goes(runner):
    result = runner.invoke(main, ["get", "project", "--image-builder", "remote"], prog_name="flyte")
    assert result.exit_code == 2
    assert "--image-builder is an option of 'flyte' itself" in _flat(result.output)
    assert "flyte --image-builder {local|local-podman|remote} get project" in _flat(result.output)
    # The old message suggested an unrelated subcommand option.
    assert "Did you mean" not in _flat(result.output)


def test_image_builder_with_equals_after_subcommand(runner):
    result = runner.invoke(main, ["deploy", "--image-builder=remote", "x.py"], prog_name="flyte")
    assert result.exit_code == 2
    assert "goes before the subcommand" in _flat(result.output)
    assert "Did you mean --image?" not in _flat(result.output)


def test_top_level_flag_after_subcommand(runner):
    result = runner.invoke(main, ["get", "project", "--insecure"], prog_name="flyte")
    assert result.exit_code == 2
    assert "flyte --insecure get project" in _flat(result.output)


def test_unknown_option_keeps_click_message(runner):
    result = runner.invoke(main, ["get", "project", "--bogus"], prog_name="flyte")
    assert result.exit_code == 2
    assert "No such option: --bogus" in _flat(result.output)
    assert "goes before the subcommand" not in _flat(result.output)
