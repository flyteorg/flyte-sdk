import asyncio
import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path
from string import Template
from typing import ClassVar, Optional, Protocol, cast

import aiofiles
import click

from flyte import Secret
from flyte._image import (
    AptPackages,
    Commands,
    CopyConfig,
    DockerIgnore,
    Env,
    Image,
    Layer,
    PipOption,
    PipPackages,
    PoetryProject,
    PythonWheels,
    Requirements,
    UVProject,
    UVScript,
    WorkDir,
    _DockerLines,
    _ensure_tuple,
)
from flyte._internal.imagebuild.image_builder import (
    DockerAPIImageChecker,
    ImageBuilder,
    ImageChecker,
    LocalDockerCommandImageChecker,
    LocalPodmanCommandImageChecker,
)
from flyte._internal.imagebuild.utils import copy_files_to_context, get_and_list_dockerignore
from flyte._logging import logger

_F_IMG_ID = "_F_IMG_ID"
FLYTE_DOCKER_BUILDER_CACHE_FROM = "FLYTE_DOCKER_BUILDER_CACHE_FROM"
FLYTE_DOCKER_BUILDER_CACHE_TO = "FLYTE_DOCKER_BUILDER_CACHE_TO"

UV_LOCK_WITHOUT_PROJECT_INSTALL_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   --mount=type=bind,target=uv.lock,src=$UV_LOCK_PATH,rw \
   --mount=type=bind,target=pyproject.toml,src=$PYPROJECT_PATH \
   $SECRET_MOUNT \
   uv sync --active --inexact $PIP_INSTALL_ARGS
""")

UV_LOCK_INSTALL_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   --mount=type=bind,target=/root/.flyte/$PYPROJECT_PATH,src=$PYPROJECT_PATH,rw \
   $SECRET_MOUNT \
   uv sync --active --inexact --no-editable $PIP_INSTALL_ARGS --project /root/.flyte/$PYPROJECT_PATH
""")

POETRY_LOCK_WITHOUT_PROJECT_INSTALL_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   uv pip install poetry

ENV POETRY_CACHE_DIR=/tmp/poetry_cache \
   POETRY_VIRTUALENVS_IN_PROJECT=true

RUN --mount=type=cache,sharing=locked,mode=0777,target=/tmp/poetry_cache,id=poetry \
   --mount=type=bind,target=poetry.lock,src=$POETRY_LOCK_PATH \
   --mount=type=bind,target=pyproject.toml,src=$PYPROJECT_PATH \
   $SECRET_MOUNT \
   poetry install $POETRY_INSTALL_ARGS
""")

POETRY_LOCK_INSTALL_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   uv pip install poetry

ENV POETRY_CACHE_DIR=/tmp/poetry_cache \
   POETRY_VIRTUALENVS_IN_PROJECT=true

RUN --mount=type=cache,sharing=locked,mode=0777,target=/tmp/poetry_cache,id=poetry \
   --mount=type=bind,target=/root/.flyte/$PYPROJECT_PATH,src=$PYPROJECT_PATH,rw \
   $SECRET_MOUNT \
   poetry install $POETRY_INSTALL_ARGS -C /root/.flyte/$PYPROJECT_PATH
""")

UV_PACKAGE_INSTALL_COMMAND_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   $REQUIREMENTS_MOUNT \
   $SECRET_MOUNT \
   uv pip install --python $$UV_PYTHON $PIP_INSTALL_ARGS
""")

UV_WHEEL_INSTALL_COMMAND_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=wheel \
   --mount=source=/dist,target=/dist,type=bind \
   $SECRET_MOUNT \
   uv pip install --python $$UV_PYTHON $PIP_INSTALL_ARGS
""")

APT_INSTALL_COMMAND_TEMPLATE = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/var/cache/apt,id=apt \
   $SECRET_MOUNT \
   apt-get update && apt-get install -y --no-install-recommends \
   $APT_PACKAGES
""")

UV_PYTHON_INSTALL_COMMAND = Template("""\
RUN --mount=type=cache,sharing=locked,mode=0777,target=/root/.cache/uv,id=uv \
   $SECRET_MOUNT \
   uv pip install $PIP_INSTALL_ARGS
""")

# uv pip install --python /root/env/bin/python
# new template
DOCKER_FILE_UV_BASE_TEMPLATE = Template("""\
# syntax=docker/dockerfile:1.10
FROM ghcr.io/astral-sh/uv:0.8.13 AS uv
FROM $BASE_IMAGE


USER root


# Copy in uv so that later commands don't have to mount it in
COPY --from=uv /uv /usr/bin/uv


# Configure default envs
ENV UV_COMPILE_BYTECODE=1 \
   UV_LINK_MODE=copy \
   VIRTUALENV=/opt/venv \
   UV_PYTHON=/opt/venv/bin/python \
   PATH="/opt/venv/bin:$$PATH"


# Create a virtualenv with the user specified python version
RUN uv venv $$VIRTUALENV --python=$PYTHON_VERSION


# Adds nvidia just in case it exists
ENV PATH="$$PATH:/usr/local/nvidia/bin:/usr/local/cuda/bin" \
   LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
""")

# This gets added on to the end of the dockerfile
DOCKER_FILE_BASE_FOOTER = Template("""\
ENV _F_IMG_ID=$F_IMG_ID
WORKDIR /root
SHELL ["/bin/bash", "-c"]
""")


class Handler(Protocol):
    @staticmethod
    async def handle(layer: Layer, context_path: Path, dockerfile: str) -> str: ...


class PipAndRequirementsHandler:
    @staticmethod
    async def handle(layer: PipPackages, context_path: Path, dockerfile: str) -> str:
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)

        # Set pip_install_args based on the layer type - either a requirements file or a list of packages
        if isinstance(layer, Requirements):
            if not layer.file.exists():
                raise FileNotFoundError(f"Requirements file {layer.file} does not exist")
            if not layer.file.is_file():
                raise ValueError(f"Requirements file {layer.file} is not a file")

            # Copy the requirements file to the context path
            requirements_path = copy_files_to_context(layer.file, context_path)
            rel_path = str(requirements_path.relative_to(context_path))
            pip_install_args = layer.get_pip_install_args()
            pip_install_args.extend(["--requirement", "requirements.txt"])
            mount = f"--mount=type=bind,target=requirements.txt,src={rel_path}"
        else:
            mount = ""
            requirements = list(layer.packages) if layer.packages else []
            reqs = " ".join(requirements)
            pip_install_args = layer.get_pip_install_args()
            pip_install_args.append(reqs)

        delta = UV_PACKAGE_INSTALL_COMMAND_TEMPLATE.substitute(
            SECRET_MOUNT=secret_mounts,
            REQUIREMENTS_MOUNT=mount,
            PIP_INSTALL_ARGS=" ".join(pip_install_args),
        )

        dockerfile += delta

        return dockerfile


class PythonWheelHandler:
    @staticmethod
    async def handle(layer: PythonWheels, context_path: Path, dockerfile: str) -> str:
        shutil.copytree(layer.wheel_dir, context_path / "dist", dirs_exist_ok=True)
        pip_install_args = layer.get_pip_install_args()
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)

        # First install: Install the wheel without dependencies using --no-deps
        pip_install_args_no_deps = [
            *pip_install_args,
            *[
                "--find-links",
                "/dist",
                "--no-deps",
                "--no-index",
                "--reinstall",
                layer.package_name,
            ],
        ]

        delta1 = UV_WHEEL_INSTALL_COMMAND_TEMPLATE.substitute(
            PIP_INSTALL_ARGS=" ".join(pip_install_args_no_deps), SECRET_MOUNT=secret_mounts
        )
        dockerfile += delta1

        # Second install: Install dependencies from PyPI
        pip_install_args_deps = [*pip_install_args, layer.package_name]
        delta2 = UV_WHEEL_INSTALL_COMMAND_TEMPLATE.substitute(
            PIP_INSTALL_ARGS=" ".join(pip_install_args_deps), SECRET_MOUNT=secret_mounts
        )
        dockerfile += delta2

        return dockerfile


class _DockerLinesHandler:
    @staticmethod
    async def handle(layer: _DockerLines, context_path: Path, dockerfile: str) -> str:
        # Add the lines to the dockerfile
        for line in layer.lines:
            dockerfile += f"\n{line}\n"

        return dockerfile


class EnvHandler:
    @staticmethod
    async def handle(layer: Env, context_path: Path, dockerfile: str) -> str:
        # Add the env vars to the dockerfile
        for key, value in layer.env_vars:
            dockerfile += f"\nENV {key}={value}\n"

        return dockerfile


class AptPackagesHandler:
    @staticmethod
    async def handle(layer: AptPackages, _: Path, dockerfile: str) -> str:
        packages = layer.packages
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)
        delta = APT_INSTALL_COMMAND_TEMPLATE.substitute(APT_PACKAGES=" ".join(packages), SECRET_MOUNT=secret_mounts)
        dockerfile += delta

        return dockerfile


class UVProjectHandler:
    @staticmethod
    async def handle(
        layer: UVProject, context_path: Path, dockerfile: str, docker_ignore_patterns: list[str] = []
    ) -> str:
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)
        if layer.project_install_mode == "dependencies_only":
            pip_install_args = " ".join(layer.get_pip_install_args())
            if "--no-install-project" not in pip_install_args:
                pip_install_args += " --no-install-project"
            if "--no-sources" not in pip_install_args:
                pip_install_args += " --no-sources"
            # Only Copy pyproject.yaml and uv.lock.
            pyproject_dst = copy_files_to_context(layer.pyproject, context_path)
            uvlock_dst = copy_files_to_context(layer.uvlock, context_path)
            delta = UV_LOCK_WITHOUT_PROJECT_INSTALL_TEMPLATE.substitute(
                UV_LOCK_PATH=uvlock_dst.relative_to(context_path),
                PYPROJECT_PATH=pyproject_dst.relative_to(context_path),
                PIP_INSTALL_ARGS=pip_install_args,
                SECRET_MOUNT=secret_mounts,
            )
        else:
            # Copy the entire project.
            pyproject_dst = copy_files_to_context(layer.pyproject.parent, context_path, docker_ignore_patterns)

            # Make sure pyproject.toml and uv.lock files are not removed by docker ignore
            uv_lock_context_path = pyproject_dst / "uv.lock"
            pyproject_context_path = pyproject_dst / "pyproject.toml"
            if not uv_lock_context_path.exists():
                shutil.copy(layer.uvlock, pyproject_dst)
            if not pyproject_context_path.exists():
                shutil.copy(layer.pyproject, pyproject_dst)

            delta = UV_LOCK_INSTALL_TEMPLATE.substitute(
                PYPROJECT_PATH=pyproject_dst.relative_to(context_path),
                PIP_INSTALL_ARGS=" ".join(layer.get_pip_install_args()),
                SECRET_MOUNT=secret_mounts,
            )

        dockerfile += delta
        return dockerfile


class PoetryProjectHandler:
    @staticmethod
    async def handel(
        layer: PoetryProject, context_path: Path, dockerfile: str, docker_ignore_patterns: list[str] = []
    ) -> str:
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)
        extra_args = layer.extra_args or ""
        if layer.project_install_mode == "dependencies_only":
            # Only Copy pyproject.yaml and poetry.lock.
            pyproject_dst = copy_files_to_context(layer.pyproject, context_path)
            poetry_lock_dst = copy_files_to_context(layer.poetry_lock, context_path)
            if "--no-root" not in extra_args:
                extra_args += " --no-root"
            delta = POETRY_LOCK_WITHOUT_PROJECT_INSTALL_TEMPLATE.substitute(
                POETRY_LOCK_PATH=poetry_lock_dst.relative_to(context_path),
                PYPROJECT_PATH=pyproject_dst.relative_to(context_path),
                POETRY_INSTALL_ARGS=extra_args,
                SECRET_MOUNT=secret_mounts,
            )
        else:
            # Copy the entire project.
            pyproject_dst = copy_files_to_context(layer.pyproject.parent, context_path, docker_ignore_patterns)

            # Make sure pyproject.toml and poetry.lock files are not removed by docker ignore
            poetry_lock_context_path = pyproject_dst / "poetry.lock"
            pyproject_context_path = pyproject_dst / "pyproject.toml"
            if not poetry_lock_context_path.exists():
                shutil.copy(layer.poetry_lock, pyproject_dst)
            if not pyproject_context_path.exists():
                shutil.copy(layer.pyproject, pyproject_dst)

            delta = POETRY_LOCK_INSTALL_TEMPLATE.substitute(
                PYPROJECT_PATH=pyproject_dst.relative_to(context_path),
                POETRY_INSTALL_ARGS=extra_args,
                SECRET_MOUNT=secret_mounts,
            )
        dockerfile += delta
        return dockerfile


class DockerIgnoreHandler:
    @staticmethod
    async def handle(layer: DockerIgnore, context_path: Path, _: str):
        shutil.copy(layer.path, context_path)


class CopyConfigHandler:
    @staticmethod
    async def handle(
        layer: CopyConfig, context_path: Path, dockerfile: str, docker_ignore_patterns: list[str] = []
    ) -> str:
        # Copy the source config file or directory to the context path
        if layer.src.is_absolute() or ".." in str(layer.src):
            dst_path = context_path / str(layer.src.absolute()).replace("/", "./_flyte_abs_context/", 1)
        else:
            dst_path = context_path / layer.src

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path = layer.src.absolute()

        if layer.src.is_file():
            # Copy the file
            shutil.copy(abs_path, dst_path)
        elif layer.src.is_dir():
            # Copy the entire directory
            shutil.copytree(
                abs_path, dst_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*docker_ignore_patterns)
            )
        else:
            logger.error(f"Source path not exists: {layer.src}")
            return dockerfile

        # Add a copy command to the dockerfile
        dockerfile += f"\nCOPY {dst_path.relative_to(context_path)} {layer.dst}\n"
        return dockerfile


class CommandsHandler:
    @staticmethod
    async def handle(layer: Commands, _: Path, dockerfile: str) -> str:
        # Append raw commands to the dockerfile
        secret_mounts = _get_secret_mounts_layer(layer.secret_mounts)
        for command in layer.commands:
            dockerfile += f"\nRUN {secret_mounts} {command}\n"

        return dockerfile


class WorkDirHandler:
    @staticmethod
    async def handle(layer: WorkDir, _: Path, dockerfile: str) -> str:
        # cd to the workdir
        dockerfile += f"\nWORKDIR {layer.workdir}\n"

        return dockerfile


def _get_secret_commands(layers: typing.Tuple[Layer, ...]) -> typing.List[str]:
    commands = []

    def _get_secret_command(secret: str | Secret) -> typing.List[str]:
        if isinstance(secret, str):
            secret = Secret(key=secret)
        secret_id = hash(secret)
        secret_env_key = "_".join([k.upper() for k in filter(None, (secret.group, secret.key))])
        if os.getenv(secret_env_key):
            return ["--secret", f"id={secret_id},env={secret_env_key}"]
        secret_file_name = "_".join(list(filter(None, (secret.group, secret.key))))
        secret_file_path = f"/etc/secrets/{secret_file_name}"
        if not os.path.exists(secret_file_path):
            raise FileNotFoundError(f"Secret not found in Env Var {secret_env_key} or file {secret_file_path}")
        return ["--secret", f"id={secret_id},src={secret_file_path}"]

    for layer in layers:
        if isinstance(layer, (PipOption, AptPackages, Commands)):
            if layer.secret_mounts:
                for secret_mount in layer.secret_mounts:
                    commands.extend(_get_secret_command(secret_mount))
    return commands


def _get_secret_mounts_layer(secrets: typing.Tuple[str | Secret, ...] | None) -> str:
    if secrets is None:
        return ""
    secret_mounts_layer = ""
    for s in secrets:
        secret = Secret(key=s) if isinstance(s, str) else s
        secret_id = hash(secret)
        if secret.mount:
            secret_mounts_layer += f"--mount=type=secret,id={secret_id},target={secret.mount}"
        elif secret.as_env_var:
            secret_mounts_layer += f"--mount=type=secret,id={secret_id},env={secret.as_env_var}"
        else:
            secret_default_env_key = "_".join(list(filter(None, (secret.group, secret.key))))
            secret_mounts_layer += f"--mount=type=secret,id={secret_id},env={secret_default_env_key}"

    return secret_mounts_layer


async def _process_layer(
    layer: Layer, context_path: Path, dockerfile: str, docker_ignore_patterns: list[str] = []
) -> str:
    match layer:
        case PythonWheels():
            # Handle Python wheels
            dockerfile = await PythonWheelHandler.handle(layer, context_path, dockerfile)

        case UVScript():
            # Handle UV script
            from flyte._utils import parse_uv_script_file

            header = parse_uv_script_file(layer.script)
            if header.dependencies:
                pip = PipPackages(
                    packages=_ensure_tuple(header.dependencies) if header.dependencies else None,
                    secret_mounts=layer.secret_mounts,
                    index_url=layer.index_url,
                    extra_args=layer.extra_args,
                    pre=layer.pre,
                    extra_index_urls=layer.extra_index_urls,
                )
                dockerfile = await PipAndRequirementsHandler.handle(pip, context_path, dockerfile)

        case Requirements() | PipPackages():
            # Handle pip packages and requirements
            dockerfile = await PipAndRequirementsHandler.handle(layer, context_path, dockerfile)

        case AptPackages():
            # Handle apt packages
            dockerfile = await AptPackagesHandler.handle(layer, context_path, dockerfile)

        case UVProject():
            # Handle UV project
            dockerfile = await UVProjectHandler.handle(layer, context_path, dockerfile, docker_ignore_patterns)

        case PoetryProject():
            # Handle Poetry project
            dockerfile = await PoetryProjectHandler.handel(layer, context_path, dockerfile, docker_ignore_patterns)

        case PoetryProject():
            # Handle Poetry project
            dockerfile = await PoetryProjectHandler.handel(layer, context_path, dockerfile, docker_ignore_patterns)

        case CopyConfig():
            # Handle local files and folders
            dockerfile = await CopyConfigHandler.handle(layer, context_path, dockerfile, docker_ignore_patterns)

        case Commands():
            # Handle commands
            dockerfile = await CommandsHandler.handle(layer, context_path, dockerfile)

        case DockerIgnore():
            # Handle dockerignore
            await DockerIgnoreHandler.handle(layer, context_path, dockerfile)

        case WorkDir():
            # Handle workdir
            dockerfile = await WorkDirHandler.handle(layer, context_path, dockerfile)

        case Env():
            # Handle environment variables
            dockerfile = await EnvHandler.handle(layer, context_path, dockerfile)

        case _DockerLines():
            # Only for internal use
            dockerfile = await _DockerLinesHandler.handle(layer, context_path, dockerfile)

        case _:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")

    return dockerfile


class DockerImageBuilder(ImageBuilder):
    """Image builder using Docker and buildkit."""

    builder_type: ClassVar = "docker"
    _builder_name: ClassVar = "flytex"

    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        # Can get a public token for docker.io but ghcr requires a pat, so harder to get the manifest anonymously
        return [LocalDockerCommandImageChecker, LocalPodmanCommandImageChecker, DockerAPIImageChecker]

    async def build_image(self, image: Image, dry_run: bool = False) -> str:
        if image.dockerfile:
            # If a dockerfile is provided, use it directly
            return await self._build_from_dockerfile(image, push=True)

        if len(image._layers) == 0:
            logger.warning("No layers to build, returning the image URI as is.")
            return image.uri

        return await self._build_image(
            image,
            push=True,
            dry_run=dry_run,
        )

    async def _build_from_dockerfile(self, image: Image, push: bool) -> str:
        """
        Build the image from a provided Dockerfile.
        """
        assert image.dockerfile  # for mypy
        await DockerImageBuilder._ensure_buildx_builder()

        command = [
            "docker",
            "buildx",
            "build",
            "--builder",
            DockerImageBuilder._builder_name,
            "-f",
            str(image.dockerfile),
            "--tag",
            f"{image.uri}",
            "--platform",
            ",".join(image.platform),
            str(image.dockerfile.parent.absolute()),  # Use the parent directory of the Dockerfile as the context
        ]

        if image.registry and push:
            command.append("--push")
        else:
            command.append("--load")

        command.extend(_get_secret_commands(layers=image._layers))

        concat_command = " ".join(command)
        logger.debug(f"Build command: {concat_command}")
        click.secho(f"Run command: {concat_command} ", fg="blue")

        await asyncio.to_thread(subprocess.run, command, cwd=str(cast(Path, image.dockerfile).cwd()), check=True)

        return image.uri

    @staticmethod
    async def _ensure_buildx_builder():
        """Ensure there is a docker buildx builder called flyte"""
        # Check if buildx is available
        try:
            await asyncio.to_thread(
                subprocess.run, ["docker", "buildx", "version"], check=True, stdout=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            raise RuntimeError("Docker buildx is not available. Make sure BuildKit is installed and enabled.")

        # List builders
        result = await asyncio.to_thread(
            subprocess.run, ["docker", "buildx", "ls"], capture_output=True, text=True, check=True
        )
        builders = result.stdout

        # Check if there's any usable builder
        if DockerImageBuilder._builder_name not in builders:
            # No default builder found, create one
            logger.info("No buildx builder found, creating one...")
            await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "buildx",
                    "create",
                    "--name",
                    DockerImageBuilder._builder_name,
                    "--platform",
                    "linux/amd64,linux/arm64",
                ],
                check=True,
            )
        else:
            logger.info("Buildx builder already exists.")

    async def _build_image(self, image: Image, *, push: bool = True, dry_run: bool = False) -> str:
        """
        if default image (only base image and locked), raise an error, don't have a dockerfile
        if dockerfile, just build
        in the main case, get the default Dockerfile template
          - start from the base image
          - use python to create a default venv and export variables


          Then for the layers
          - for each layer
            - find the appropriate layer handler
            - call layer handler with the context dir and the dockerfile
              - handler can choose to do something (copy files from local) to the context and update the dockerfile
                contents, returning the new string
        """
        # For testing, set `push=False` to just build the image locally and not push to
        # registry.

        await DockerImageBuilder._ensure_buildx_builder()

        with tempfile.TemporaryDirectory() as tmp_dir:
            logger.warning(f"Temporary directory: {tmp_dir}")
            tmp_path = Path(tmp_dir)

            dockerfile = DOCKER_FILE_UV_BASE_TEMPLATE.substitute(
                BASE_IMAGE=image.base_image,
                PYTHON_VERSION=f"{image.python_version[0]}.{image.python_version[1]}",
            )

            # Get .dockerignore file patterns first
            docker_ignore_patterns = get_and_list_dockerignore(image)

            for layer in image._layers:
                dockerfile = await _process_layer(layer, tmp_path, dockerfile, docker_ignore_patterns)

            dockerfile += DOCKER_FILE_BASE_FOOTER.substitute(F_IMG_ID=image.uri)

            dockerfile_path = tmp_path / "Dockerfile"
            async with aiofiles.open(dockerfile_path, mode="w") as f:
                await f.write(dockerfile)

            command = [
                "docker",
                "buildx",
                "build",
                "--builder",
                DockerImageBuilder._builder_name,
                "--tag",
                f"{image.uri}",
                "--platform",
                ",".join(image.platform),
            ]

            cache_from = os.getenv(FLYTE_DOCKER_BUILDER_CACHE_FROM)
            cache_to = os.getenv(FLYTE_DOCKER_BUILDER_CACHE_TO)
            if cache_from and cache_to:
                command[3:3] = [
                    f"--cache-from={cache_from}",
                    f"--cache-to={cache_to}",
                ]

            if image.registry and push:
                command.append("--push")
            else:
                command.append("--load")

            command.extend(_get_secret_commands(layers=image._layers))
            command.append(tmp_dir)

            concat_command = " ".join(command)
            logger.debug(f"Build command: {concat_command}")
            if dry_run:
                click.secho("Dry run for docker builder...")
                click.secho(f"Context path: {tmp_path}")
                click.secho(f"Dockerfile: {dockerfile}")
                click.secho(f"Command: {concat_command}")
                return image.uri
            else:
                click.secho(f"Run command: {concat_command} ", fg="blue")

            try:
                await asyncio.to_thread(subprocess.run, command, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build image: {e}")
                raise RuntimeError(f"Failed to build image: {e}")

            return image.uri
