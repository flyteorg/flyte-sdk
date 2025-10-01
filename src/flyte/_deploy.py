from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, Set, Tuple, Type

import cloudpickle
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
    from flyteidl2.task import task_definition_pb2
    from flyteidl2.trigger import trigger_definition_pb2

    from ._code_bundle import CopyFiles
    from ._internal.imagebuild.image_builder import ImageCache


@rich.repr.auto
@dataclass
class DeploymentPlan:
    envs: Dict[str, Environment]
    version: Optional[str] = None


@rich.repr.auto
@dataclass
class DeploymentContext:
    """
    Context for deployment operations.
    """

    environment: Environment | TaskEnvironment
    serialization_context: SerializationContext
    dryrun: bool = False


@rich.repr.auto
@dataclass
class DeployedTask:
    deployed_task: task_definition_pb2.TaskSpec
    deployed_triggers: List[trigger_definition_pb2.TaskTrigger]

    def summary_repr(self) -> str:
        """
        Returns a summary representation of the deployed task.
        """
        return (
            f"DeployedTask(name={self.deployed_task.task_template.id.name}, "
            f"version={self.deployed_task.task_template.id.version})"
        )

    def table_repr(self) -> List[Tuple[str, ...]]:
        """
        Returns a table representation of the deployed task.
        """
        return [
            ("name", self.deployed_task.task_template.id.name),
            ("version", self.deployed_task.task_template.id.version),
            ("triggers", ",".join([t.name for t in self.deployed_triggers])),
        ]


@rich.repr.auto
@dataclass
class DeployedEnv:
    env: Environment
    deployed_entities: List[DeployedTask]

    def summary_repr(self) -> str:
        """
        Returns a summary representation of the deployment.
        """
        entities = ", ".join(f"{e.summary_repr()}" for e in self.deployed_entities or [])
        return f"Deployment(env=[{self.env.name}], entities=[{entities}])"

    def table_repr(self) -> List[List[Tuple[str, ...]]]:
        """
        Returns a detailed representation of the deployed tasks.
        """
        tuples = []
        if self.deployed_entities:
            for e in self.deployed_entities:
                tuples.append(e.table_repr())
        return tuples

    def env_repr(self) -> List[Tuple[str, ...]]:
        """
        Returns a detailed representation of the deployed environments.
        """
        env = self.env
        return [
            ("environment", env.name),
            ("image", env.image.uri if isinstance(env.image, Image) else env.image or ""),
        ]


@rich.repr.auto
@dataclass(frozen=True)
class Deployment:
    envs: Dict[str, DeployedEnv]

    def summary_repr(self) -> str:
        """
        Returns a summary representation of the deployment.
        """
        envs = ", ".join(f"{e.summary_repr()}" for e in self.envs.values() or [])
        return f"Deployment(envs=[{envs}])"

    def table_repr(self) -> List[List[Tuple[str, ...]]]:
        """
        Returns a detailed representation of the deployed tasks.
        """
        tuples = []
        for d in self.envs.values():
            tuples.extend(d.table_repr())
        return tuples

    def env_repr(self) -> List[List[Tuple[str, ...]]]:
        """
        Returns a detailed representation of the deployed environments.
        """
        tuples = []
        for d in self.envs.values():
            tuples.append(d.env_repr())
        return tuples


async def _deploy_task(
    task: TaskTemplate, serialization_context: SerializationContext, dryrun: bool = False
) -> DeployedTask:
    """
    Deploy the given task.
    """
    ensure_client()
    import grpc.aio

    from flyteidl2.task import task_definition_pb2, task_service_pb2

    from ._internal.runtime.convert import convert_upload_default_inputs
    from ._internal.runtime.task_serde import translate_task_to_wire
    from ._internal.runtime.trigger_serde import to_task_trigger

    image_uri = task.image.uri if isinstance(task.image, Image) else task.image

    try:
        if dryrun:
            return DeployedTask(translate_task_to_wire(task, serialization_context), [])

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

        deployable_triggers_coros = []
        for t in task.triggers:
            inputs = spec.task_template.interface.inputs
            default_inputs = spec.default_inputs
            deployable_triggers_coros.append(
                to_task_trigger(t=t, task_name=task.name, task_inputs=inputs, task_default_inputs=list(default_inputs))
            )

        deployable_triggers = await asyncio.gather(*deployable_triggers_coros)
        try:
            await get_client().task_service.DeployTask(
                task_service_pb2.DeployTaskRequest(
                    task_id=task_id,
                    spec=spec,
                    triggers=deployable_triggers,
                )
            )
            logger.info(f"Deployed task {task.name} with version {task_id.version}")
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                logger.info(f"Task {task.name} with image {image_uri} already exists, skipping deployment.")
                return DeployedTask(spec, deployable_triggers)
            raise

        return DeployedTask(spec, deployable_triggers)
    except Exception as e:
        logger.error(f"Failed to deploy task {task.name} with image {image_uri}: {e}")
        raise flyte.errors.DeploymentError(
            f"Failed to deploy task {task.name} file{task.source_file} with image {image_uri}, Error: {e!s}"
        ) from e


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
            images.append(_build_image_bg(env_name, auto_image))
    final_images = await asyncio.gather(*images)

    for env_name, image_uri in final_images:
        logger.warning(f"Built Image for environment {env_name}, image: {image_uri}")
        env = deployment.envs[env_name]
        image_identifier_map[env_name] = image_uri

    return ImageCache(image_lookup=image_identifier_map)


class Deployer(Protocol):
    """
    Protocol for deployment callables.
    """

    async def __call__(self, context: DeploymentContext) -> DeployedEnv:
        """
        Deploy the environment described in the context.

        Args:
            context: Deployment context containing environment, serialization context, and dryrun flag

        Returns:
            Deployment result
        """
        ...


async def _deploy_task_env(context: DeploymentContext) -> DeployedEnv:
    """
    Deploy the given task environment.
    """
    ensure_client()
    env = context.environment
    if not isinstance(env, TaskEnvironment):
        raise ValueError(f"Expected TaskEnvironment, got {type(env)}")

    task_coros = []
    for task in env.tasks.values():
        task_coros.append(_deploy_task(task, context.serialization_context, dryrun=context.dryrun))
    deployed_task_vals = await asyncio.gather(*task_coros)
    deployed_tasks = []
    for t in deployed_task_vals:
        deployed_tasks.append(t)
    return DeployedEnv(env=env, deployed_entities=deployed_tasks)


_ENVTYPE_REGISTRY: Dict[Type[Environment | TaskEnvironment], Deployer] = {
    TaskEnvironment: _deploy_task_env,
}


def register_deployer(env_type: Type[Environment | TaskEnvironment], deployer: Deployer) -> None:
    """
    Register a deployer for a specific environment type.

    Args:
        env_type: Type of environment this deployer handles
        deployer: Deployment callable that conforms to the Deployer protocol
    """
    _ENVTYPE_REGISTRY[env_type] = deployer


def get_deployer(env_type: Type[Environment | TaskEnvironment]) -> Deployer:
    """
    Get the registered deployer for an environment type.

    Args:
        env_type: Type of environment to get deployer for

    Returns:
        Deployer for the environment type, defaults to task environment deployer
    """
    v = _ENVTYPE_REGISTRY.get(env_type)
    if v is None:
        raise ValueError(f"No deployer registered for environment type {env_type}")
    return v


@requires_initialization
async def apply(deployment_plan: DeploymentPlan, copy_style: CopyFiles, dryrun: bool = False) -> Deployment:
    from ._code_bundle import build_code_bundle

    cfg = get_common_config()

    image_cache = await _build_images(deployment_plan)

    if copy_style == "none" and not deployment_plan.version:
        raise flyte.errors.DeploymentError("Version must be set when copy_style is none")
    else:
        code_bundle = await build_code_bundle(from_dir=cfg.root_dir, dryrun=dryrun, copy_style=copy_style)
        if deployment_plan.version:
            version = deployment_plan.version
        else:
            h = hashlib.md5()
            h.update(cloudpickle.dumps(deployment_plan.envs))
            h.update(code_bundle.computed_version.encode("utf-8"))
            h.update(cloudpickle.dumps(image_cache))
            version = h.hexdigest()

    sc = SerializationContext(
        project=cfg.project,
        domain=cfg.domain,
        org=cfg.org,
        code_bundle=code_bundle,
        version=version,
        image_cache=image_cache,
        root_dir=cfg.root_dir,
    )

    deployment_coros = []
    for env_name, env in deployment_plan.envs.items():
        logger.info(f"Deploying environment {env_name}")
        deployer = get_deployer(type(env))
        context = DeploymentContext(environment=env, serialization_context=sc, dryrun=dryrun)
        deployment_coros.append(deployer(context))
    deployed_envs = await asyncio.gather(*deployment_coros)
    envs = {}
    for d in deployed_envs:
        envs[d.env.name] = d

    return Deployment(envs)


def _recursive_discover(planned_envs: Dict[str, Environment], env: Environment) -> Dict[str, Environment]:
    """
    Recursively deploy the environment and its dependencies, if not already deployed (present in env_tasks) and
    return the updated env_tasks.
    """
    if env.name in planned_envs:
        if planned_envs[env.name] is not env:
            # Raise error if different TaskEnvironment objects have the same name
            raise ValueError(f"Duplicate environment name '{env.name}' found")
    # Add the environment to the existing envs
    planned_envs[env.name] = env

    # Recursively discover dependent environments
    for dependent_env in env.depends_on:
        _recursive_discover(planned_envs, dependent_env)
    return planned_envs


def plan_deploy(*envs: Environment, version: Optional[str] = None) -> List[DeploymentPlan]:
    if envs is None:
        return [DeploymentPlan({})]
    deployment_plans = []
    visited_envs: Set[str] = set()
    for env in envs:
        if env.name in visited_envs:
            raise ValueError(f"Duplicate environment name '{env.name}' found")
        planned_envs = _recursive_discover({}, env)
        deployment_plans.append(DeploymentPlan(planned_envs, version=version))
        visited_envs.update(planned_envs.keys())
    return deployment_plans


@syncify
async def deploy(
    *envs: Environment,
    dryrun: bool = False,
    version: str | None = None,
    interactive_mode: bool | None = None,
    copy_style: CopyFiles = "loaded_modules",
) -> List[Deployment]:
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
    deployment_plans = plan_deploy(*envs, version=version)
    deployments = []
    for deployment_plan in deployment_plans:
        deployments.append(apply(deployment_plan, copy_style=copy_style, dryrun=dryrun))
    return await asyncio.gather(*deployments)


@syncify
async def build_images(envs: Environment) -> ImageCache:
    """
    Build the images for the given environments.
    :param envs: Environment to build images for.
    :return: ImageCache containing the built images.
    """
    deployment = plan_deploy(envs)
    return await _build_images(deployment[0])
