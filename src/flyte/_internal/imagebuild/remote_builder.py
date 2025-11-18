import gzip
import os
import shutil
import tarfile
import tempfile
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, cast
from uuid import uuid4

import aiofiles

import flyte
import flyte.errors
from flyte import Image, remote
from flyte._code_bundle._utils import tar_strip_file_attributes
from flyte._image import (
    _BASE_REGISTRY,
    AptPackages,
    Architecture,
    Commands,
    CopyConfig,
    DockerIgnore,
    Env,
    PipOption,
    PipPackages,
    PoetryProject,
    PythonWheels,
    Requirements,
    UVProject,
    UVScript,
    WorkDir,
)
from flyte._internal.imagebuild.image_builder import ImageBuilder, ImageChecker
from flyte._internal.imagebuild.utils import copy_files_to_context, get_and_list_dockerignore
from flyte._internal.runtime.task_serde import get_security_context
from flyte._logging import logger
from flyte._secret import Secret
from flyte.remote import ActionOutputs, Run

if TYPE_CHECKING:
    from flyteidl2.imagebuilder import definition_pb2 as image_definition_pb2

IMAGE_TASK_NAME = "build-image"
IMAGE_TASK_PROJECT = "system"
IMAGE_TASK_DOMAIN = "production"


class RemoteImageChecker(ImageChecker):
    _images_client = None

    @classmethod
    async def image_exists(
        cls, repository: str, tag: str, arch: Tuple[Architecture, ...] = ("linux/amd64",)
    ) -> Optional[str]:
        try:
            import flyte.remote as remote

            remote.Task.get(
                name=IMAGE_TASK_NAME,
                project=IMAGE_TASK_PROJECT,
                domain=IMAGE_TASK_DOMAIN,
                auto_version="latest",
            )
        except Exception as e:
            msg = "remote image builder is not enabled. Please contact Union support to enable it."
            raise flyte.errors.ImageBuildError(msg) from e

        image_name = f"{repository.split('/')[-1]}:{tag}"

        try:
            from flyteidl2.common.identifier_pb2 import ProjectIdentifier
            from flyteidl2.imagebuilder import definition_pb2 as image_definition__pb2
            from flyteidl2.imagebuilder import payload_pb2 as image_payload__pb2
            from flyteidl2.imagebuilder import service_pb2_grpc as image_service_pb2_grpc

            from flyte._initialize import _get_init_config

            cfg = _get_init_config()
            if cfg is None:
                raise ValueError("Init config should not be None")
            image_id = image_definition__pb2.ImageIdentifier(name=image_name)
            req = image_payload__pb2.GetImageRequest(
                id=image_id,
                organization=cfg.org,
                project_id=ProjectIdentifier(organization=cfg.org, domain=cfg.domain, name=cfg.project),
            )
            if cls._images_client is None:
                if cfg.client is None:
                    raise ValueError("remote client should not be None")
                cls._images_client = image_service_pb2_grpc.ImageServiceStub(cfg.client._channel)
            resp = await cls._images_client.GetImage(req)
            logger.warning(f"[blue]Image {resp.image.fqin} found. Skip building.[/blue]")
            return resp.image.fqin
        except Exception:
            logger.warning(f"[blue]Image {image_name} was not found or has expired.[/blue]", extra={"highlight": False})
            return None


class RemoteImageBuilder(ImageBuilder):
    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """Return the image checker."""
        return [RemoteImageChecker]

    async def build_image(self, image: Image, dry_run: bool = False) -> str:
        from flyteidl2.workflow import run_definition_pb2

        image_name = f"{image.name}:{image._final_tag}"
        spec, context = await _validate_configuration(image)

        start = datetime.now(timezone.utc)
        entity = await remote.Task.get(
            name=IMAGE_TASK_NAME,
            project=IMAGE_TASK_PROJECT,
            domain=IMAGE_TASK_DOMAIN,
            auto_version="latest",
        ).override.aio(secrets=_get_build_secrets_from_image(image))

        logger.warning("[bold blue]🐳 Submitting a new build...[/bold blue]")
        if image.registry and image.registry != _BASE_REGISTRY:
            target_image = f"{image.registry}/{image_name}"
        else:
            # Use the default system registry in the backend.
            target_image = image_name

        from flyte._initialize import get_init_config

        cfg = get_init_config()
        run = cast(
            Run,
            await flyte.with_runcontext(
                project=cfg.project, domain=cfg.domain, cache_lookup_scope="project-domain"
            ).run.aio(entity, spec=spec, context=context, target_image=target_image),
        )
        logger.warning(f"⏳ Waiting for build to finish at: [bold cyan link={run.url}]{run.url}[/bold cyan link]")

        await run.wait.aio(quiet=True)
        run_details = await run.details.aio()

        elapsed = str(datetime.now(timezone.utc) - start).split(".")[0]

        if run_details.action_details.raw_phase == run_definition_pb2.PHASE_SUCCEEDED:
            logger.warning(f"[bold green]✅ Build completed in {elapsed}![/bold green]")
        else:
            raise flyte.errors.ImageBuildError(f"❌ Build failed in {elapsed} at {run.url}")

        outputs = await run_details.outputs()
        return _get_fully_qualified_image_name(outputs)


async def _validate_configuration(image: Image) -> Tuple[str, Optional[str]]:
    """Validate the configuration and prepare the spec and context files."""  # Prepare the spec file
    tmp_path = Path(tempfile.gettempdir()) / str(uuid4())
    os.makedirs(tmp_path, exist_ok=True)

    context_path = tmp_path / "build.uc-image-builder"
    context_path.mkdir(exist_ok=True)

    image_idl = _get_layers_proto(image, context_path)

    spec_path = tmp_path / "spec.pb"
    with spec_path.open("wb") as f:
        f.write(image_idl.SerializeToString())

    _, spec_url = await remote.upload_file.aio(spec_path)

    if any(context_path.iterdir()):
        # If there are files in the context directory, upload it
        tar_path = tmp_path / "context.tar"
        with tarfile.open(tar_path, "w", dereference=False) as tar:
            files: typing.List[str] = os.listdir(context_path)
            for ws_file in files:
                tar.add(
                    os.path.join(context_path, ws_file),
                    recursive=True,
                    arcname=ws_file,
                    filter=tar_strip_file_attributes,
                )
        context_dst = Path(f"{tar_path!s}.gz")
        with gzip.GzipFile(filename=context_dst, mode="wb", mtime=0) as gzipped:
            async with aiofiles.open(tar_path, "rb") as tar_file:
                content = await tar_file.read()
                gzipped.write(content)

        context_size = tar_path.stat().st_size
        if context_size > 5 * 1024 * 1024:
            logger.warning(
                f"[yellow]Context size is {context_size / (1024 * 1024):.2f} MB, which is larger than 5 MB. "
                "Upload and build speed will be impacted.[/yellow]",
            )
        _, context_url = await remote.upload_file.aio(context_dst)
    else:
        context_url = ""

    return spec_url, context_url


def _get_layers_proto(image: Image, context_path: Path) -> "image_definition_pb2.ImageSpec":
    from flyteidl2.imagebuilder import definition_pb2 as image_definition_pb2

    if image.dockerfile is not None:
        raise flyte.errors.ImageBuildError(
            "Custom Dockerfile is not supported with remote image builder.You can use local image builder instead."
        )

    layers = []
    for layer in image._layers:
        secret_mounts = None
        pip_options = image_definition_pb2.PipOptions()

        if isinstance(layer, PipOption):
            pip_options = image_definition_pb2.PipOptions(
                index_url=layer.index_url,
                extra_index_urls=layer.extra_index_urls,
                pre=layer.pre,
                extra_args=layer.extra_args,
            )

        if hasattr(layer, "secret_mounts"):
            sc = get_security_context(layer.secret_mounts)
            secret_mounts = sc.secrets if sc else None

        if isinstance(layer, AptPackages):
            apt_layer = image_definition_pb2.Layer(
                apt_packages=image_definition_pb2.AptPackages(
                    packages=layer.packages,
                    secret_mounts=secret_mounts,
                ),
            )
            layers.append(apt_layer)
        elif isinstance(layer, PythonWheels):
            dst_path = copy_files_to_context(layer.wheel_dir, context_path)
            wheel_layer = image_definition_pb2.Layer(
                python_wheels=image_definition_pb2.PythonWheels(
                    dir=str(dst_path.relative_to(context_path)),
                    options=pip_options,
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(wheel_layer)

        elif isinstance(layer, Requirements):
            dst_path = copy_files_to_context(layer.file, context_path)
            requirements_layer = image_definition_pb2.Layer(
                requirements=image_definition_pb2.Requirements(
                    file=str(dst_path.relative_to(context_path)),
                    options=pip_options,
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(requirements_layer)
        elif isinstance(layer, PipPackages) or isinstance(layer, UVScript):
            if isinstance(layer, UVScript):
                from flyte._utils import parse_uv_script_file

                header = parse_uv_script_file(layer.script)
                if not header.dependencies:
                    continue
                packages: typing.Iterable[str] = header.dependencies
            else:
                packages = layer.packages or []
            pip_layer = image_definition_pb2.Layer(
                pip_packages=image_definition_pb2.PipPackages(
                    packages=packages,
                    options=pip_options,
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(pip_layer)
        elif isinstance(layer, UVProject):
            for line in layer.pyproject.read_text().splitlines():
                if "tool.uv.index" in line:
                    raise ValueError("External sources are not supported in pyproject.toml")

            if layer.project_install_mode == "dependencies_only":
                # Copy pyproject itself
                pyproject_dst = copy_files_to_context(layer.pyproject, context_path)
                if pip_options.extra_args:
                    if "--no-install-project" not in pip_options.extra_args:
                        pip_options.extra_args += " --no-install-project"
                else:
                    pip_options.extra_args = " --no-install-project"
                if "--no-sources" not in pip_options.extra_args:
                    pip_options.extra_args += " --no-sources"
            else:
                # Copy the entire project
                docker_ignore_patterns = get_and_list_dockerignore(image)
                pyproject_dst = copy_files_to_context(layer.pyproject.parent, context_path, docker_ignore_patterns)

            uv_layer = image_definition_pb2.Layer(
                uv_project=image_definition_pb2.UVProject(
                    pyproject=str(pyproject_dst.relative_to(context_path)),
                    uvlock=str(copy_files_to_context(layer.uvlock, context_path).relative_to(context_path)),
                    options=pip_options,
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(uv_layer)
        elif isinstance(layer, PoetryProject):
            for line in layer.pyproject.read_text().splitlines():
                if "tool.poetry.source" in line:
                    raise ValueError("External sources are not supported in pyproject.toml")
            extra_args = layer.extra_args or ""
            if layer.project_install_mode == "dependencies_only":
                # Copy pyproject itself
                if "--no-root" not in extra_args:
                    extra_args += " --no-root"
                pyproject_dst = copy_files_to_context(layer.pyproject, context_path)
            else:
                # Copy the entire project
                pyproject_dst = copy_files_to_context(layer.pyproject.parent, context_path)

            poetry_layer = image_definition_pb2.Layer(
                poetry_project=image_definition_pb2.PoetryProject(
                    pyproject=str(pyproject_dst.relative_to(context_path)),
                    poetry_lock=str(copy_files_to_context(layer.poetry_lock, context_path).relative_to(context_path)),
                    extra_args=extra_args,
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(poetry_layer)
        elif isinstance(layer, Commands):
            commands_layer = image_definition_pb2.Layer(
                commands=image_definition_pb2.Commands(
                    cmd=list(layer.commands),
                    secret_mounts=secret_mounts,
                )
            )
            layers.append(commands_layer)
        elif isinstance(layer, DockerIgnore):
            shutil.copy(layer.path, context_path)
        elif isinstance(layer, CopyConfig):
            dst_path = copy_files_to_context(layer.src, context_path)

            copy_layer = image_definition_pb2.Layer(
                copy_config=image_definition_pb2.CopyConfig(
                    src=str(dst_path.relative_to(context_path)),
                    dst=str(layer.dst),
                )
            )
            layers.append(copy_layer)
        elif isinstance(layer, Env):
            env_layer = image_definition_pb2.Layer(
                env=image_definition_pb2.Env(
                    env_variables=dict(layer.env_vars),
                )
            )
            layers.append(env_layer)
        elif isinstance(layer, WorkDir):
            workdir_layer = image_definition_pb2.Layer(
                workdir=image_definition_pb2.WorkDir(workdir=layer.workdir),
            )
            layers.append(workdir_layer)

    return image_definition_pb2.ImageSpec(
        base_image=image.base_image,
        python_version=f"{image.python_version[0]}.{image.python_version[1]}",
        layers=layers,
    )


def _get_fully_qualified_image_name(outputs: ActionOutputs) -> str:
    return outputs.pb2.literals[0].value.scalar.primitive.string_value


def _get_build_secrets_from_image(image: Image) -> Optional[typing.List[Secret]]:
    secrets = []
    DEFAULT_SECRET_DIR = Path("/etc/flyte/secrets")
    for layer in image._layers:
        if isinstance(layer, (PipOption, Commands, AptPackages)) and layer.secret_mounts is not None:
            for secret_mount in layer.secret_mounts:
                # Mount all the image secrets to a default directory that will be passed to the BuildKit server.
                if isinstance(secret_mount, Secret):
                    secrets.append(Secret(key=secret_mount.key, group=secret_mount.group, mount=DEFAULT_SECRET_DIR))
                elif isinstance(secret_mount, str):
                    secrets.append(Secret(key=secret_mount, mount=DEFAULT_SECRET_DIR))
                else:
                    raise ValueError(f"Unsupported secret_mount type: {type(secret_mount)}")

    image_registry_secret = image._image_registry_secret
    if image_registry_secret:
        secrets.append(
            Secret(key=image_registry_secret.key, group=image_registry_secret.group, mount=DEFAULT_SECRET_DIR)
        )
    return secrets
