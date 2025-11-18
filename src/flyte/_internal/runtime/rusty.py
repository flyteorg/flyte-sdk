import asyncio
import time
from typing import List, Tuple

from flyte._context import contextual_run
from flyte._internal.controllers import Controller
from flyte._internal.controllers import create_controller as _create_controller
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.runtime.entrypoints import download_code_bundle, load_pkl_task, load_task
from flyte._internal.runtime.taskrunner import extract_download_run_upload
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte._utils import adjust_sys_path
from flyte.models import ActionID, Checkpoints, CodeBundle, PathRewrite, RawDataPath


async def download_tgz(destination: str, version: str, tgz: str) -> CodeBundle:
    """
    Downloads and loads the task from the code bundle or resolver.
    :param tgz: The path to the task template in a tar.gz format.
    :param destination: The path to save the downloaded task template.
    :param version: The version of the task to load.
    :return: The CodeBundle object.
    """
    logger.info(f"[rusty] Downloading tgz code bundle from {tgz} to {destination} with version {version}")
    adjust_sys_path()

    code_bundle = CodeBundle(
        tgz=tgz,
        destination=destination,
        computed_version=version,
    )
    return await download_code_bundle(code_bundle)


async def download_load_pkl(destination: str, version: str, pkl: str) -> Tuple[CodeBundle, TaskTemplate]:
    """
    Downloads and loads the task from the code bundle or resolver.
    :param pkl: The path to the task template in a pickle format.
    :param destination: The path to save the downloaded task template.
    :param version: The version of the task to load.
    :return: The CodeBundle object.
    """
    logger.info(f"[rusty] Downloading pkl code bundle from {pkl} to {destination} with version {version}")
    adjust_sys_path()

    code_bundle = CodeBundle(
        pkl=pkl,
        destination=destination,
        computed_version=version,
    )
    code_bundle = await download_code_bundle(code_bundle)
    return code_bundle, load_pkl_task(code_bundle)


def load_task_from_code_bundle(resolver: str, resolver_args: List[str]) -> TaskTemplate:
    """
    Loads the task from the code bundle or resolver.
    :param resolver: The resolver to use to load the task.
    :param resolver_args: The arguments to pass to the resolver.
    :return: The loaded task template.
    """
    logger.debug(f"[rusty] Loading task from code bundle {resolver} with args: {resolver_args}")
    return load_task(resolver, *resolver_args)


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
    logger.info(f"[rusty] Creating controller with endpoint {endpoint}")
    import flyte.errors
    from flyte._initialize import init_in_cluster

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(flyte.errors.silence_grpc_polling_error)

    # TODO Currently reference tasks are not supported in Rusty.
    controller_kwargs = await init_in_cluster.aio(api_key=api_key, endpoint=endpoint, insecure=insecure)
    return _create_controller(ct="remote", **controller_kwargs)


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
    path_rewrite_cfg: str | None = None,
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
    :param path_rewrite_cfg: Optional path rewrite configuration.
    :return: The loaded task template.
    """
    start_time = time.time()
    action_id = f"{org}/{project}/{domain}/{run_name}/{name}"

    logger.info(
        f"[rusty] Running task '{task.name}' (action: {action_id})"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    path_rewrite = PathRewrite.from_str(path_rewrite_cfg) if path_rewrite_cfg else None
    if path_rewrite:
        import flyte.storage as storage

        if not await storage.exists(path_rewrite.new_prefix):
            logger.error(
                f"[rusty] Path rewrite failed for path {path_rewrite.new_prefix}, "
                f"not found, reverting to original path {path_rewrite.old_prefix}"
            )
            path_rewrite = None
        else:
            logger.info(f"[rusty] Using path rewrite: {path_rewrite}")

    try:
        await contextual_run(
            extract_download_run_upload,
            task,
            action=ActionID(name=name, org=org, project=project, domain=domain, run_name=run_name),
            version=version,
            controller=controller,
            raw_data_path=RawDataPath(path=raw_data_path, path_rewrite=path_rewrite),
            output_path=output_path,
            run_base_dir=run_base_dir,
            checkpoints=Checkpoints(prev_checkpoint_path=prev_checkpoint, checkpoint_path=checkpoint_path),
            code_bundle=code_bundle,
            input_path=input_path,
            image_cache=ImageCache.from_transport(image_cache) if image_cache else None,
        )
    except Exception as e:
        logger.error(f"[rusty] Task failed: {e!s}")
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"[rusty] TASK_EXECUTION_END: Task '{task.name}' (action: {action_id})"
            f" done after {duration:.2f}s at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )


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
