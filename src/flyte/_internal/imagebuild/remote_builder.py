import os
import shutil
import tempfile
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, cast
from uuid import uuid4

import click

import flyte
import flyte.errors
from flyte import Image, remote
from flyte._image import (
    AptPackages,
    Architecture,
    Commands,
    CopyConfig,
    DockerIgnore,
    Env,
    PipPackages,
    PythonWheels,
    Requirements,
    UVProject,
    UVScript,
    WorkDir,
)
from flyte._internal.imagebuild.image_builder import ImageBuilder, ImageChecker
from flyte._internal.imagebuild.utils import copy_files_to_context
from flyte._logging import logger
from flyte._secret import secrets_from_request
from flyte.remote import ActionOutputs, Run

if TYPE_CHECKING:
    from flyte._protos.imagebuilder import definition_pb2 as image_definition_pb2

IMAGE_TASK_NAME = "build-image"
OPTIMIZE_TASK_NAME = "optimize_task"
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
            from flyte._initialize import _get_init_config
            from flyte._protos.imagebuilder import definition_pb2 as image_definition__pb2
            from flyte._protos.imagebuilder import payload_pb2 as image_payload__pb2
            from flyte._protos.imagebuilder import service_pb2_grpc as image_service_pb2_grpc

            cfg = _get_init_config()
            if cfg is None:
                raise ValueError("Init config should not be None")
            image_id = image_definition__pb2.ImageIdentifier(name=image_name)
            req = image_payload__pb2.GetImageRequest(id=image_id, organization=cfg.org)
            if cls._images_client is None:
                if cfg.client is None:
                    raise ValueError("remote client should not be None")
                cls._images_client = image_service_pb2_grpc.ImageServiceStub(cfg.client._channel)
            resp = await cls._images_client.GetImage(req)
            logger.warning(click.style(f"Image {resp.image.fqin} found. Skip building.", fg="blue"))
            return resp.image.fqin
        except Exception:
            logger.warning(click.style(f"Image {image_name} was not found or has expired.", fg="blue"))
            return None


class RemoteImageBuilder(ImageBuilder):
    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """Return the image checker."""
        return [RemoteImageChecker]

    async def build_image(self, image: Image, dry_run: bool = False) -> str:
        from flyte._protos.workflow import run_definition_pb2

        image_name = f"{image.name}:{image._final_tag}"
        spec, context = await _validate_configuration(image)

        start = datetime.now(timezone.utc)
        entity = remote.Task.get(
            name=IMAGE_TASK_NAME,
            project=IMAGE_TASK_PROJECT,
            domain=IMAGE_TASK_DOMAIN,
            auto_version="latest",
        )
        run = cast(
            Run,
            await flyte.with_runcontext(project=IMAGE_TASK_PROJECT, domain=IMAGE_TASK_DOMAIN).run.aio(
                entity, spec=spec, context=context, target_image=image_name
            ),
        )
        logger.warning(click.style("🐳 Submitting a new build...", fg="blue", bold=True))

        logger.warning(click.style("⏳ Waiting for build to finish at: " + click.style(run.url, fg="cyan"), bold=True))
        await run.wait.aio(quiet=True)
        run_details = await run.details.aio()

        elapsed = str(datetime.now(timezone.utc) - start).split(".")[0]

        if run_details.action_details.raw_phase == run_definition_pb2.PHASE_SUCCEEDED:
            logger.warning(click.style(f"✅ Build completed in {elapsed}!", bold=True, fg="green"))
            try:
                entity = remote.Task.get(
                    name=OPTIMIZE_TASK_NAME,
                    project=IMAGE_TASK_PROJECT,
                    domain=IMAGE_TASK_DOMAIN,
                    auto_version="latest",
                )
                await flyte.with_runcontext(project=IMAGE_TASK_PROJECT, domain=IMAGE_TASK_DOMAIN).run.aio(
                    entity, spec=spec, context=context, target_image=image_name
                )
            except Exception as e:
                # Ignore the error if optimize is not enabled in the backend.
                logger.warning(f"Failed to run optimize task with error: {e}")
        else:
            raise flyte.errors.ImageBuildError(f"❌ Build failed in {elapsed} at {click.style(run.url, fg='cyan')}")

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
        archive = Path(shutil.make_archive(str(tmp_path / "context"), "xztar", context_path))
        st = archive.stat()
        if st.st_size > 5 * 1024 * 1024:
            logger.warning(
                click.style(
                    f"Context size is {st.st_size / (1024 * 1024):.2f} MB, which is larger than 5 MB. "
                    "Upload and build speed will be impacted.",
                    fg="yellow",
                )
            )
        _, context_url = await remote.upload_file.aio(archive)
    else:
        context_url = ""

    return spec_url, context_url


def _get_layers_proto(image: Image, context_path: Path) -> "image_definition_pb2.ImageSpec":
    from flyte._protos.imagebuilder import definition_pb2 as image_definition_pb2

    layers = []
    for layer in image._layers:
        if isinstance(layer, AptPackages):
            apt_layer = image_definition_pb2.Layer(
                apt_packages=image_definition_pb2.AptPackages(
                    packages=layer.packages,
                    secret_mounts=secrets_from_request(layer.secret_mounts) if layer.secret_mounts else None,
                ),
            )
            layers.append(apt_layer)
        elif isinstance(layer, PythonWheels):
            dst_path = copy_files_to_context(layer.wheel_dir, context_path)
            wheel_layer = image_definition_pb2.Layer(
                python_wheels=image_definition_pb2.PythonWheels(
                    dir=str(dst_path.relative_to(context_path)),
                    options=image_definition_pb2.PipOptions(
                        index_url=layer.index_url,
                        extra_index_urls=layer.extra_index_urls,
                        pre=layer.pre,
                        extra_args=layer.extra_args,
                    ),
                    secret_mounts=secrets_from_request(layer.secret_mounts) if layer.secret_mounts else None,
                )
            )
            layers.append(wheel_layer)

        elif isinstance(layer, Requirements):
            dst_path = copy_files_to_context(layer.file, context_path)
            requirements_layer = image_definition_pb2.Layer(
                requirements=image_definition_pb2.Requirements(
                    file=str(dst_path.relative_to(context_path)),
                    options=image_definition_pb2.PipOptions(
                        index_url=layer.index_url,
                        extra_index_urls=layer.extra_index_urls,
                        pre=layer.pre,
                        extra_args=layer.extra_args,
                    ),
                    secret_mounts=secrets_from_request(layer.secret_mounts) if layer.secret_mounts else None,
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
                    options=image_definition_pb2.PipOptions(
                        index_url=layer.index_url,
                        extra_index_urls=layer.extra_index_urls,
                        pre=layer.pre,
                        extra_args=layer.extra_args,
                    ),
                    secret_mounts=secrets_from_request(layer.secret_mounts) if layer.secret_mounts else None,
                )
            )
            layers.append(pip_layer)
        elif isinstance(layer, UVProject):
            for line in layer.pyproject.read_text().splitlines():
                if "tool.uv.index" in line:
                    raise ValueError("External sources are not supported in pyproject.toml")
            shutil.copy2(layer.pyproject, context_path / layer.pyproject.name)

            uv_layer = image_definition_pb2.Layer(
                uv_project=image_definition_pb2.UVProject(
                    pyproject=str(layer.pyproject.name),
                    uvlock=str(layer.uvlock.name),
                )
            )
            layers.append(uv_layer)
        elif isinstance(layer, Commands):
            commands_layer = image_definition_pb2.Layer(
                commands=image_definition_pb2.Commands(cmd=list(layer.commands))
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
