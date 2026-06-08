import pathlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from flyte._internal.imagebuild.image_builder import ImageCache
from flyte.cli._build import BuildEnvRecursiveCommand, build
from flyte.cli.main import main

TEST_CODE_PATH = pathlib.Path(__file__).parent
HELLO_WORLD_PY = TEST_CODE_PATH / "run_testdata" / "hello_world.py"


@pytest.fixture
def runner():
    return CliRunner()


def test_build_exposes_copy_style_and_root_dir(runner):
    """flyte build must accept --copy-style and --root-dir so CI can split
    image builds from deploys without hash drift."""
    result = runner.invoke(build, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--copy-style" in result.output
    assert "--root-dir" in result.output


def test_build_rejects_invalid_copy_style(runner):
    """--copy-style is a Click.Choice constrained to the CopyFiles values;
    unknown values must fail loudly (not silently build with the wrong hash)."""
    result = runner.invoke(
        build,
        ["--copy-style", "bogus", str(HELLO_WORLD_PY), "hello_world"],
    )
    assert result.exit_code != 0
    assert "bogus" in result.output.lower() or "invalid" in result.output.lower()


def test_build_exposes_all_flag(runner):
    """`flyte build --all` mirrors `flyte deploy --all` so CI can build every
    environment in a module without naming one."""
    result = runner.invoke(build, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--all" in result.output
    assert "--recursive" in result.output


def test_build_all_resolves_to_recursive_command():
    """With --all set, EnvFiles.get_command must return the recursive command
    instead of listing the module's environments as sub-commands."""
    import click

    ctx = click.Context(build)
    ctx.params = {
        "all": True,
        "recursive": False,
        "copy_style": "loaded_modules",
        "root_dir": None,
        "ignore_load_errors": False,
    }
    resolved = build.get_command(ctx, str(HELLO_WORLD_PY))
    assert isinstance(resolved, BuildEnvRecursiveCommand)


def test_build_all_builds_every_environment(runner):
    """`flyte build --all <file>` builds images for ALL environments in the
    module in a single pass and prints the resulting image(s)."""
    fake_cache = ImageCache(image_lookup={"hello_world": "ghcr.io/org/img:abc123"})

    with (
        patch("flyte.build_images", return_value=fake_cache) as mock_build,
        patch("flyte.cli._common.CLIConfig.init"),
    ):
        result = main.main(
            args=["build", "--all", str(HELLO_WORLD_PY)],
            standalone_mode=False,
        )
    # main.main returns the command's return value (None) and raises on error;
    # assert the build path actually invoked build_images with all envs.
    assert result is None
    assert mock_build.called
    # The env from hello_world.py must have been passed positionally.
    called_env_names = [getattr(e, "name", None) for e in mock_build.call_args.args]
    assert "hello_world" in called_env_names


def test_no_progress_uses_static_status(runner):
    """The global --no-progress flag must disable the animated spinner and use
    a static, CI-friendly status instead."""
    from flyte.cli import _common

    fake_cache = ImageCache(image_lookup={"hello_world": "ghcr.io/org/img:abc123"})

    seen = {}
    real_cli_status = _common.cli_status

    def spy_cli_status(output_format, message, spinner="dots", no_progress=False):
        seen["no_progress"] = no_progress
        return real_cli_status(output_format, message, spinner=spinner, no_progress=no_progress)

    with (
        patch("flyte.build_images", return_value=fake_cache),
        patch("flyte.cli._common.CLIConfig.init"),
        patch("flyte.cli._build.common.cli_status", side_effect=spy_cli_status),
    ):
        main.main(
            args=["--no-progress", "build", "--all", str(HELLO_WORLD_PY)],
            standalone_mode=False,
        )

    assert seen.get("no_progress") is True


def test_cli_status_static_when_no_progress():
    """cli_status returns a non-animated context manager when no_progress=True."""
    from flyte.cli._common import _StaticStatus, cli_status

    cm = cli_status("table", "Building...", no_progress=True)
    assert isinstance(cm, _StaticStatus)
