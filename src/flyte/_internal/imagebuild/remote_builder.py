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
from flyteidl2.common import phase_pb2

import flyte
import flyte.errors
from flyte import Image, remote
from flyte._code_bundle._ignore import STANDARD_IGNORE_PATTERNS
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
from flyte._internal.imagebuild.utils import (
    copy_files_to_context,
    get_and_list_dockerignore,
    get_uv_project_editable_dependencies,
)
from flyte._internal.runtime.task_serde import get_security_context
from flyte._logging import logger
from flyte._secret import Secret
from flyte._status import status
from flyte.remote import ActionOutputs, Run

if TYPE_CHECKING:
    from flyteidl2.imagebuilder import definition_pb2 as image_definition_pb2

    from flyte._build import ImageBuild

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

        image_name = f"{repository.rsplit('/', maxsplit=1)[-1]}:{tag}"

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
            logger.debug(f"Image {resp.image.fqin} found in remote registry")
            return resp.image.fqin
        except Exception:
            status.info(f"Image {image_name} was not found or has expired")
            return None


class RemoteImageBuilder(ImageBuilder):
    def get_checkers(self) -> Optional[typing.List[typing.Type[ImageChecker]]]:
        """Return the image checker."""
        return [RemoteImageChecker]

    async def build_image(self, image: Image, dry_run: bool = False, wait: bool = True) -> "ImageBuild":
        from flyte._build import ImageBuild

        image_name = f"{image.name}:{image._final_tag}"
        spec, context = await _validate_configuration(image)

        start = datetime.now(timezone.utc)
        try:
            entity = await remote.Task.get(
                name=IMAGE_TASK_NAME,
                project=IMAGE_TASK_PROJECT,
                domain=IMAGE_TASK_DOMAIN,
                auto_version="latest",
            ).override.aio(secrets=_get_build_secrets_from_image(image))
        except flyte.errors.RemoteTaskNotFoundError:
            raise flyte.errors.ImageBuildError(
                "remote image builder is not enabled. Please contact Union support to enable it."
            )

        status.step("Submitting a new build...")
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

        status.step(f"Started build at: {run.url}")
        if not wait:
            # return the ImageBuild with the run object (uri will be None since build hasn't completed)
            return ImageBuild(uri=None, remote_run=run)

        status.step("Waiting for build to finish")
        await run.wait.aio(quiet=True)
        run_details = await run.details.aio()
        elapsed = str(datetime.now(timezone.utc) - start).split(".")[0]

        if run_details.action_details.raw_phase == phase_pb2.ACTION_PHASE_SUCCEEDED:
            status.success(f"Build completed in {elapsed}")
        else:
            raise flyte.errors.ImageBuildError(f"❌ Build failed in {elapsed} at {run.url}")

        outputs = await run_details.outputs()
        uri = _get_fully_qualified_image_name(outputs)
        return ImageBuild(uri=uri, remote_run=run)


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
            status.warn(
                f"Context size is {context_size / (1024 * 1024):.2f} MB, which is larger than 5 MB. "
                "Upload and build speed will be impacted."
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
    docker_ignore_patterns = get_and_list_dockerignore(image)

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
            dst_path = copy_files_to_context(layer.wheel_dir, context_path, [])
            wheel_layer = image_definition_pb2.Layer(
                python_wheels=image_definition_pb2.PythonWheels(
                    dir=str(dst_path.relative_to(context_path)),
                    options=pip_options,
                    secret_mounts=secret_mounts,
                ),
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
                if header.pyprojects:
                    layers.append(
                        image_definition_pb2.Layer(
                            apt_packages=image_definition_pb2.AptPackages(
                                packages=["git"],  # To get the version of the project.
                            ),
                        )
                    )

                    for pyproject in header.pyprojects:
                        pyproject_dst = copy_files_to_context(Path(pyproject), context_path, docker_ignore_patterns)
                        uv_lock_path = Path(pyproject) / "uv.lock"
                        uv_project_kwargs = {
                            "pyproject": str(pyproject_dst.relative_to(context_path)),
                            "options": pip_options,
                            "secret_mounts": secret_mounts,
                        }
                        if uv_lock_path.exists():
                            uv_project_kwargs["uvlock"] = str(
                                copy_files_to_context(uv_lock_path, context_path).relative_to(context_path)
                            )
                        uv_project_layer = image_definition_pb2.Layer(
                            uv_project=image_definition_pb2.UVProject(**uv_project_kwargs)
                        )
                        layers.append(uv_project_layer)

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
            pyproject_dst = copy_files_to_context(layer.pyproject, context_path)
            # Keep track of the directory containing the pyproject.toml file
            # this is what should be passed to the UVProject image definition proto as 'pyproject'
            pyproject_dir_dst = pyproject_dst.parent
            pyproject_dir_dsts = [pyproject_dir_dst]

            # Copy uv.lock itself (if provided)
            uvlock_dst = copy_files_to_context(layer.uvlock, context_path) if layer.uvlock is not None else None

            # Handle the project install mode
            match layer.project_install_mode:
                case "dependencies_only":
                    if pip_options.extra_args and ("--no-install-project" not in pip_options.extra_args):
                        pip_options.extra_args += " --no-install-project"
                    else:
                        pip_options.extra_args = "--no-install-project"
                    # Copy the editable dependencies defined under the [tool.uv.sources] in pyproject.toml
                    standard_ignore_patterns = STANDARD_IGNORE_PATTERNS.copy()
                    for editable_dep in get_uv_project_editable_dependencies(layer.pyproject.parent):
                        pyproject_dir_dsts.append(
                            copy_files_to_context(
                                editable_dep / "pyproject.toml",
                                context_path,
                                ignore_patterns=[*standard_ignore_patterns, *docker_ignore_patterns],
                            )
                        )
                case "install_project":
                    # Copy the entire project
                    pyproject_dir_dst = copy_files_to_context(
                        layer.pyproject.parent, context_path, docker_ignore_patterns
                    )
                case _:
                    raise ValueError(f"Invalid project install mode: {layer.project_install_mode}")

            uv_layer = image_definition_pb2.Layer(
                uv_project=image_definition_pb2.UVProject(
                    # NOTE: UVProject expects 'pyproject' to be the directory containing the pyproject.toml file
                    # whereas it expects 'uvlock' to be the path to the uv.lock file itself.
                    pyproject=str(pyproject_dir_dst.relative_to(context_path)),
                    uvlock=str(uvlock_dst.relative_to(context_path)) if uvlock_dst else None,
                    options=pip_options,
                    secret_mounts=secret_mounts,
                    # Source dir should be the common parent directory of all the pyproject.toml files
                    # we copied for this UV project layer.
                    source_dir=str(Path(os.path.commonpath(pyproject_dir_dsts)).relative_to(context_path)),
                )
            )
            layers.append(uv_layer)
        elif isinstance(layer, PoetryProject):
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
            dst_path = copy_files_to_context(layer.src, context_path, docker_ignore_patterns)

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
    seen_secrets: typing.Set[typing.Tuple[typing.Optional[str], str]] = set()
    DEFAULT_SECRET_DIR = Path("/etc/flyte/secrets")
    for layer in image._layers:
        if isinstance(layer, (PipOption, Commands, AptPackages)) and layer.secret_mounts is not None:
            for secret_mount in layer.secret_mounts:
                # Mount all the image secrets to a default directory that will be passed to the BuildKit server.
                if isinstance(secret_mount, Secret):
                    secret_id = (secret_mount.group, secret_mount.key)
                    if secret_id not in seen_secrets:
                        seen_secrets.add(secret_id)
                        secrets.append(Secret(key=secret_mount.key, group=secret_mount.group, mount=DEFAULT_SECRET_DIR))
                elif isinstance(secret_mount, str):
                    secret_id = (None, secret_mount)
                    if secret_id not in seen_secrets:
                        seen_secrets.add(secret_id)
                        secrets.append(Secret(key=secret_mount, mount=DEFAULT_SECRET_DIR))
                else:
                    raise ValueError(f"Unsupported secret_mount type: {type(secret_mount)}")

    image_registry_secret = image._image_registry_secret
    if image_registry_secret:
        secret_id = (image_registry_secret.group, image_registry_secret.key)
        if secret_id not in seen_secrets:
            seen_secrets.add(secret_id)
            secrets.append(
                Secret(key=image_registry_secret.key, group=image_registry_secret.group, mount=DEFAULT_SECRET_DIR)
            )
    return secrets
