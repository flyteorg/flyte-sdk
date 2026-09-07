import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from flyte import Secret
from flyte._image import PixiScript
from flyte._internal.imagebuild.docker_builder import (
    PIXI_VERSION,
    PixiProjectHandler,
    _get_secret_commands,
    _process_layer,
)
from flyte._internal.imagebuild.utils import pixi_project_to_primitive_layers, pixi_script_to_project

PIXI_SCRIPT = """\
# /// script
# requires-python = ">=3.12"
# dependencies = ["flyte"]
#
# [tool.pixi.workspace]
# channels = ["conda-forge"]
#
# [tool.pixi.dependencies]
# numpy = "*"
# ///
import flyte
"""


def write_script(directory: str, content: str = PIXI_SCRIPT, name: str = "script.py") -> Path:
    script = Path(directory) / name
    script.write_text(content)
    return script.absolute()


@pytest.mark.asyncio
async def test_pixi_script_lowers_to_a_pixi_project():
    """A pixi script installs through the same dockerfile lines as a pixi project."""
    with tempfile.TemporaryDirectory() as tmp_context, tempfile.TemporaryDirectory() as tmp_user:
        script = write_script(tmp_user)
        project = pixi_script_to_project(PixiScript(script=script, platforms=("linux-64",)))

        # The generated manifest carries the script's metadata over faithfully.
        manifest = toml.loads(project.manifest.read_text())
        assert manifest["workspace"] == {"channels": ["conda-forge"], "platforms": ["linux-64"]}
        assert manifest["dependencies"] == {"numpy": "*", "python": ">=3.12"}
        assert manifest["pypi-dependencies"] == {"flyte": "*"}

        result = await PixiProjectHandler.handle(
            layer=project,
            context_path=Path(tmp_context),
            dockerfile="FROM python:3.12\n",
            docker_ignore_patterns=[],
        )

        assert f"COPY --from=ghcr.io/prefix-dev/pixi:{PIXI_VERSION} /usr/local/bin/pixi /usr/local/bin/pixi" in result
        assert " /opt/pixi-project/pixi.toml" in result
        assert "pixi install --manifest-path /opt/pixi-project/pixi.toml" in result
        # No sidecar lock next to the script, so the environment is resolved at build time.
        assert "--locked" not in result
        # The pixi environment becomes the image's runtime environment.
        assert "VIRTUAL_ENV=/opt/pixi-project/.pixi/envs/default" in result


@pytest.mark.asyncio
async def test_pixi_script_with_sidecar_lock():
    """`pixi lock --script` writes <script>.pixi.lock, which makes the install reproducible."""
    with tempfile.TemporaryDirectory() as tmp_context, tempfile.TemporaryDirectory() as tmp_user:
        script = write_script(tmp_user)
        (Path(tmp_user) / "script.py.pixi.lock").write_text("version: 7")

        project = pixi_script_to_project(PixiScript(script=script, platforms=("linux-64",)))
        assert project.pixi_lock is not None
        assert project.pixi_lock.read_text() == "version: 7"

        result = await PixiProjectHandler.handle(
            layer=project,
            context_path=Path(tmp_context),
            dockerfile="FROM python:3.12\n",
            docker_ignore_patterns=[],
        )
        assert " /opt/pixi-project/pixi.lock" in result
        assert "--environment default --locked" in result


@pytest.mark.asyncio
async def test_pixi_script_processed_as_a_layer():
    """_process_layer dispatches PixiScript without the caller lowering it first."""
    with tempfile.TemporaryDirectory() as tmp_context, tempfile.TemporaryDirectory() as tmp_user:
        result = await _process_layer(
            PixiScript(script=write_script(tmp_user), platforms=("linux-64",)),
            Path(tmp_context),
            "FROM python:3.12\n",
        )
        assert "pixi install --manifest-path /opt/pixi-project/pixi.toml" in result


@pytest.mark.asyncio
async def test_pixi_script_environment_and_extra_args():
    """A non-default pixi environment and extra install args reach the pixi command."""
    with tempfile.TemporaryDirectory() as tmp_context, tempfile.TemporaryDirectory() as tmp_user:
        script = write_script(
            tmp_user,
            content=(
                "# /// script\n"
                '# dependencies = ["flyte"]\n'
                "#\n"
                "# [tool.pixi.environments]\n"
                '# gpu = ["cuda"]\n'
                "#\n"
                "# [tool.pixi.feature.cuda.dependencies]\n"
                '# cuda-version = "12.*"\n'
                "# ///\n"
            ),
        )
        layer = PixiScript(
            script=script,
            platforms=("linux-64",),
            environment="gpu",
            extra_args="--no-progress",
        )
        # Tables flyte does not model still reach pixi.
        manifest = toml.loads(pixi_script_to_project(layer).manifest.read_text())
        assert manifest["environments"] == {"gpu": ["cuda"]}
        assert manifest["feature"] == {"cuda": {"dependencies": {"cuda-version": "12.*"}}}

        result = await _process_layer(layer, Path(tmp_context), "FROM python:3.12\n")
        assert "--environment gpu --no-progress" in result
        assert "VIRTUAL_ENV=/opt/pixi-project/.pixi/envs/gpu" in result


def test_pixi_script_build_secrets():
    """Secrets on a pixi script layer reach `docker buildx build --secret`."""
    with tempfile.TemporaryDirectory() as tmp_user:
        layer = PixiScript(
            script=write_script(tmp_user),
            platforms=("linux-64",),
            secret_mounts=(Secret(key="pixi_token"),),
        )
        with patch.dict("os.environ", {"PIXI_TOKEN": "shhh"}):
            commands = _get_secret_commands([layer])
        assert commands[0] == "--secret"
        assert "env=PIXI_TOKEN" in commands[1]


def test_pixi_script_remote_builder_lowering():
    """The remote builder IDL has no pixi layer, so a script becomes primitive layers."""
    with tempfile.TemporaryDirectory() as tmp_user:
        layers = pixi_project_to_primitive_layers(
            pixi_script_to_project(PixiScript(script=write_script(tmp_user), platforms=("linux-64",)))
        )
        assert [type(layer).__name__ for layer in layers] == [
            "AptPackages",
            "Commands",
            "CopyConfig",
            "Commands",
            "Env",
        ]
        # The generated manifest is what gets copied in and installed.
        assert layers[2].dst == "/opt/pixi-project/pixi.toml"
        assert "pixi install --manifest-path /opt/pixi-project/pixi.toml" in layers[3].commands[0]


def test_pixi_script_hash_tracks_metadata_not_code():
    """Editing the script's code reuses the image; editing its dependencies rebuilds it."""
    import hashlib

    def digest(script: Path) -> str:
        hasher = hashlib.md5()
        PixiScript(script=script, platforms=("linux-64",)).update_hash(hasher)
        return hasher.hexdigest()

    with tempfile.TemporaryDirectory() as tmp_user:
        script = write_script(tmp_user)
        original = digest(script)

        script.write_text(PIXI_SCRIPT + "\nprint('a new line of code')\n")
        assert digest(script) == original

        script.write_text(PIXI_SCRIPT.replace('numpy = "*"', 'numpy = "<2"'))
        assert digest(script) != original
