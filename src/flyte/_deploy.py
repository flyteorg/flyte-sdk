from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import rich.repr

import flyte.errors
from flyte.models import SerializationContext
from flyte.syncify import syncify

from ._environment import Environment
from ._image import Image
from ._initialize import ensure_client, get_client, get_common_config, requires_initialization
from ._logging import logger
from ._task import TaskTemplate
from ._task_environment import TaskEnvironment

if TYPE_CHECKING:
    from flyte._protos.workflow import task_definition_pb2

    from ._code_bundle import CopyFiles
    from ._internal.imagebuild.image_builder import ImageCache


@rich.repr.auto
@dataclass
class DeploymentPlan:
    envs: Dict[str, Environment]
    version: Optional[str] = None


@rich.repr.auto
@dataclass
class Deployment:
    envs: Dict[str, Environment]
    deployed_tasks: List[task_definition_pb2.TaskSpec] | None = None

    def summary_repr(self) -> str:
        """
        Returns a summary representation of the deployment.
        """
        env_names = ", ".join(self.envs.keys())
        task_names_versions = ", ".join(
            f"{task.task_template.id.name} (v{task.task_template.id.version})" for task in self.deployed_tasks or []
        )
        return f"Deployment(envs=[{env_names}], tasks=[{task_names_versions}])"

    def task_repr(self) -> List[List[Tuple[str, str]]]:
        """
        Returns a detailed representation of the deployed tasks.
        """
        tuples = []
        if self.deployed_tasks:
            for task in self.deployed_tasks:
                tuples.append(
                    [
                        ("name", task.task_template.id.name),
                        ("version", task.task_template.id.version),
                    ]
                )
        return tuples

    def env_repr(self) -> List[List[Tuple[str, str]]]:
        """
        Returns a detailed representation of the deployed environments.
        """
        tuples = []
        for env_name, env in self.envs.items():
            tuples.append(
                [
                    ("environment", env_name),
                    ("image", env.image.uri if isinstance(env.image, Image) else env.image or ""),
                ]
            )
        return tuples


async def _deploy_task(
    task: TaskTemplate, serialization_context: SerializationContext, dryrun: bool = False
) -> task_definition_pb2.TaskSpec:
    """
    Deploy the given task.
    """
    ensure_client()
    from ._internal.runtime.convert import convert_upload_default_inputs
    from ._internal.runtime.task_serde import translate_task_to_wire
    from ._protos.workflow import task_definition_pb2, task_service_pb2

    image_uri = task.image.uri if isinstance(task.image, Image) else task.image

    if dryrun:
        return translate_task_to_wire(task, serialization_context)

    default_inputs = await convert_upload_default_inputs(task.interface)
    spec = translate_task_to_wire(task, serialization_context, default_inputs=default_inputs)

    msg = f"Deploying task {task.name}, with image {image_uri} version {serialization_context.version}"
    if spec.task_template.HasField("container") and spec.task_template.container.args:
        msg += f" from {spec.task_template.container.args[-3]}.{spec.task_template.container.args[-1]}"
    logger.info(msg)
    task_id = task_definition_pb2.TaskIdentifier(
        org=spec.task_template.id.org,
        project=spec.task_template.id.project,
        domain=spec.task_template.id.domain,
        version=spec.task_template.id.version,
        name=spec.task_template.id.name,
    )

    await get_client().task_service.DeployTask(task_service_pb2.DeployTaskRequest(task_id=task_id, spec=spec))
    logger.info(f"Deployed task {task.name} with version {task_id.version}")
    return spec


async def _build_image_bg(env_name: str, image: Image) -> Tuple[str, str]:
    """
    Build the image in the background and return the environment name and the built image.
    """
    from ._build import build

    logger.info(f"Building image {image.name} for environment {env_name}")
    return env_name, await build.aio(image)


async def _build_images(deployment: DeploymentPlan) -> ImageCache:
    """
    Build the images for the given deployment plan and update the environment with the built image.
    """
    from ._internal.imagebuild.image_builder import ImageCache

    images = []
    image_identifier_map = {}
    for env_name, env in deployment.envs.items():
        if not isinstance(env.image, str):
            logger.debug(f"Building Image for environment {env_name}, image: {env.image}")
            images.append(_build_image_bg(env_name, env.image))

        elif env.image == "auto" and "auto" not in image_identifier_map:
            auto_image = Image.from_debian_base()
            image_identifier_map["auto"] = auto_image.uri
    final_images = await asyncio.gather(*images)

    for env_name, image_uri in final_images:
        logger.warning(f"Built Image for environment {env_name}, image: {image_uri}")
        env = deployment.envs[env_name]
        if isinstance(env.image, Image):
            image_identifier_map[env.image.identifier] = image_uri

    return ImageCache(image_lookup=image_identifier_map)


@requires_initialization
async def apply(deployment: DeploymentPlan, copy_style: CopyFiles, dryrun: bool = False) -> Deployment:
    from ._code_bundle import build_code_bundle

    cfg = get_common_config()
    image_cache = await _build_images(deployment)

    version = deployment.version
    if copy_style == "none" and not version:
        raise flyte.errors.DeploymentError("Version must be set when copy_style is none")
    else:
        code_bundle = await build_code_bundle(from_dir=cfg.root_dir, dryrun=dryrun, copy_style=copy_style)
        version = version or code_bundle.computed_version
        # TODO we should update the version to include the image cache digest and code bundle digest. This is
        # to ensure that changes in image dependencies, cause an update to the deployment version.
        # TODO Also hash the environment and tasks to ensure that changes in the environment or tasks

    sc = SerializationContext(
        project=cfg.project,
        domain=cfg.domain,
        org=cfg.org,
        code_bundle=code_bundle,
        version=version,
        image_cache=image_cache,
        root_dir=cfg.root_dir,
    )

    tasks = []
    for env_name, env in deployment.envs.items():
        logger.info(f"Deploying environment {env_name}")
        # TODO Make this pluggable based on the environment type
        if isinstance(env, TaskEnvironment):
            for task in env.tasks.values():
                tasks.append(_deploy_task(task, dryrun=dryrun, serialization_context=sc))
    return Deployment(envs=deployment.envs, deployed_tasks=await asyncio.gather(*tasks))


def _recursive_discover(
    planned_envs: Dict[str, Environment], envs: Environment | List[Environment]
) -> Dict[str, Environment]:
    """
    Recursively deploy the environment and its dependencies, if not already deployed (present in env_tasks) and
    return the updated env_tasks.
    """
    if isinstance(envs, Environment):
        envs = [envs]
    for env in envs:
        # Skip if the environment is already planned
        if env.name in planned_envs:
            continue
        # Recursively discover dependent environments
        for dependent_env in env.depends_on:
            _recursive_discover(planned_envs, dependent_env)
        # Add the environment to the existing envs
        planned_envs[env.name] = env
    return planned_envs


def plan_deploy(*envs: Environment, version: Optional[str] = None) -> DeploymentPlan:
    if envs is None:
        return DeploymentPlan({})
    planned_envs = _recursive_discover({}, *envs)
    return DeploymentPlan(planned_envs, version=version)


@syncify
async def deploy(
    *envs: Environment,
    dryrun: bool = False,
    version: str | None = None,
    interactive_mode: bool | None = None,
    copy_style: CopyFiles = "loaded_modules",
) -> Deployment:
    """
    Deploy the given environment or list of environments.
    :param envs: Environment or list of environments to deploy.
    :param dryrun: dryrun mode, if True, the deployment will not be applied to the control plane.
    :param version: version of the deployment, if None, the version will be computed from the code bundle.
    TODO: Support for interactive_mode
    :param interactive_mode: Optional, can be forced to True or False.
       If not provided, it will be set based on the current environment. For example Jupyter notebooks are considered
         interactive mode, while scripts are not. This is used to determine how the code bundle is created.
    :param copy_style: Copy style to use when running the task

    :return: Deployment object containing the deployed environments and tasks.
    """
    if interactive_mode:
        raise NotImplementedError("Interactive mode not yet implemented for deployment")
    deployment = plan_deploy(*envs, version=version)
    return await apply(deployment, copy_style=copy_style, dryrun=dryrun)


@syncify
async def build_images(*envs: Environment) -> ImageCache:
    """
    Build the images for the given environments.
    :param envs: Environment or list of environments to build images for.
    :return: ImageCache containing the built images.
    """
    deployment = plan_deploy(*envs)
    return await _build_images(deployment)
