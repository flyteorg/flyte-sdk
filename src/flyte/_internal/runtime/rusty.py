import sys
import time
from typing import Any, List, Tuple

from flyte._context import contextual_run
from flyte._internal.controllers import Controller
from flyte._internal.controllers import create_controller as _create_controller
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.runtime.entrypoints import download_code_bundle, load_pkl_task, load_task
from flyte._internal.runtime.taskrunner import extract_download_run_upload
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte.models import ActionID, Checkpoints, CodeBundle, RawDataPath


async def download_tgz(destination: str, version: str, tgz: str) -> CodeBundle:
    """
    Downloads and loads the task from the code bundle or resolver.
    :param tgz: The path to the task template in a tar.gz format.
    :param destination: The path to save the downloaded task template.
    :param version: The version of the task to load.
    :return: The CodeBundle object.
    """
    start_time = time.time()
    logger.info(
        f"[rusty] TASK_START: Downloading tgz code bundle from {tgz} to {destination}"
        f" with version {version} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    sys.path.insert(0, ".")

    code_bundle = CodeBundle(
        tgz=tgz,
        destination=destination,
        computed_version=version,
    )
    result = await download_code_bundle(code_bundle)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"[rusty] TASK_COMPLETE: Downloaded tgz code bundle to {destination} in {duration:.2f}s"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    return result


async def download_load_pkl(destination: str, version: str, pkl: str) -> Tuple[CodeBundle, TaskTemplate]:
    """
    Downloads and loads the task from the code bundle or resolver.
    :param pkl: The path to the task template in a pickle format.
    :param destination: The path to save the downloaded task template.
    :param version: The version of the task to load.
    :return: The CodeBundle object.
    """
    start_time = time.time()
    logger.info(
        f"[rusty] TASK_START: Downloading pkl code bundle from {pkl} to {destination}"
        f" with version {version} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    sys.path.insert(0, ".")

    code_bundle = CodeBundle(
        pkl=pkl,
        destination=destination,
        computed_version=version,
    )
    code_bundle = await download_code_bundle(code_bundle)
    task_template = load_pkl_task(code_bundle)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"[rusty] TASK_COMPLETE: Downloaded and loaded pkl code bundle to {destination}"
        f" in {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    return code_bundle, task_template


def load_task_from_code_bundle(resolver: str, resolver_args: List[str]) -> TaskTemplate:
    """
    Loads the task from the code bundle or resolver.
    :param resolver: The resolver to use to load the task.
    :param resolver_args: The arguments to pass to the resolver.
    :return: The loaded task template.
    """
    start_time = time.time()
    logger.info(
        f"[rusty] TASK_START: Loading task from code bundle {resolver} with args:"
        f" {resolver_args} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    task_template = load_task(resolver, *resolver_args)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"[rusty] TASK_COMPLETE: Loaded task '{task_template.name}' from code bundle in"
        f" {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    return task_template


async def create_controller(
    endpoint: str = "host.docker.internal:8090",
    insecure: bool = False,
    api_key: str | None = None,
) -> Controller:
    """
    Creates a controller instance for remote operations.
    :param endpoint:
    :param insecure:
    :param api_key:
    :return:
    """
    start_time = time.time()
    logger.info(
        f"[rusty] TASK_START: Creating controller with endpoint {endpoint}"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    from flyte._initialize import init

    # TODO Currently refrence tasks are not supported in Rusty.
    await init.aio()
    controller_kwargs: dict[str, Any] = {"insecure": insecure}
    if api_key:
        logger.info("[rusty] Using api key from environment")
        controller_kwargs["api_key"] = api_key
    else:
        controller_kwargs["endpoint"] = endpoint
        if "localhost" in endpoint or "docker" in endpoint:
            controller_kwargs["insecure"] = True
        logger.debug(f"[rusty] Using controller endpoint: {endpoint} with kwargs: {controller_kwargs}")

    controller = _create_controller(ct="remote", **controller_kwargs)

    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"[rusty] TASK_COMPLETE: Created controller for endpoint {endpoint}"
        f" in {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )

    return controller


async def run_task(
    task: TaskTemplate,
    controller: Controller,
    org: str,
    project: str,
    domain: str,
    run_name: str,
    name: str,
    raw_data_path: str,
    output_path: str,
    run_base_dir: str,
    version: str,
    image_cache: str | None = None,
    checkpoint_path: str | None = None,
    prev_checkpoint: str | None = None,
    code_bundle: CodeBundle | None = None,
    input_path: str | None = None,
):
    """
    Runs the task with the provided parameters.
    :param prev_checkpoint: Previous checkpoint path to resume from.
    :param checkpoint_path: Checkpoint path to save the current state.
    :param image_cache: Image cache to use for the task.
    :param name: Action name to run.
    :param run_name: Parent run name to use for the task.
    :param domain: domain to run the task in.
    :param project: project to run the task in.
    :param org: organization to run the task in.
    :param task: The task template to run.
    :param raw_data_path: The path to the raw data.
    :param output_path: The path to save the output.
    :param run_base_dir: The base directory for the run.
    :param version: The version of the task to run.
    :param controller: The controller to use for the task.
    :param code_bundle: Optional code bundle for the task.
    :param input_path: Optional input path for the task.
    :return: The loaded task template.
    """
    start_time = time.time()
    action_id = f"{org}/{project}/{domain}/{run_name}/{name}"

    logger.info(
        f"[rusty] TASK_EXECUTION_START: Running task '{task.name}' (action: {action_id})"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )
    logger.info(f"[rusty] TASK_CONFIG: version={version}, input_path={input_path}, output_path={output_path}")
    logger.info(f"[rusty] TASK_CONFIG: raw_data_path={raw_data_path}, run_base_dir={run_base_dir}")
    logger.info(f"[rusty] TASK_CONFIG: checkpoint_path={checkpoint_path}, prev_checkpoint={prev_checkpoint}")
    logger.info(f"[rusty] TASK_CONFIG: image_cache={image_cache}")

    try:
        await contextual_run(
            extract_download_run_upload,
            task,
            action=ActionID(name=name, org=org, project=project, domain=domain, run_name=run_name),
            version=version,
            controller=controller,
            raw_data_path=RawDataPath(path=raw_data_path),
            output_path=output_path,
            run_base_dir=run_base_dir,
            checkpoints=Checkpoints(prev_checkpoint_path=prev_checkpoint, checkpoint_path=checkpoint_path),
            code_bundle=code_bundle,
            input_path=input_path,
            image_cache=ImageCache.from_transport(image_cache) if image_cache else None,
        )

        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"[rusty] TASK_EXECUTION_SUCCESS: Task '{task.name}' (action: {action_id}) completed successfully"
            f" in {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        logger.info(f"[rusty] TASK_OUTPUTS_UPLOADED: Task outputs uploaded to '{output_path}' for task '{task.name}'")

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(
            f"[rusty] TASK_EXECUTION_FAILED: Task '{task.name}' (action: {action_id})"
            f" failed after {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        logger.error(f"[rusty] TASK_ERROR: {e!s}")
        raise


async def ping(name: str) -> str:
    """
    A simple hello world function to test the Rusty entrypoint.
    """
    print(f"Received ping request from {name} in Rusty!")
    return f"pong from Rusty to {name}!"


async def hello(name: str):
    """
    A simple hello world function to test the Rusty entrypoint.
    :param name: The name to greet.
    :return: A greeting message.
    """
    print(f"Received hello request in Rusty with name: {name}!")
