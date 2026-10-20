import asyncio
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

from flyte._context import contextual_run
from flyte._internal.controllers import Controller
from flyte._internal.controllers import create_controller as _create_controller
from flyte._internal.image_cache import ImageCache
from flyte._internal.runtime.entrypoints import download_code_bundle, load_pkl_task, load_task
from flyte._internal.runtime.taskrunner import extract_download_run_upload
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte._utils import adjust_sys_path
from flyte.models import ActionID, CheckpointPaths, CodeBundle, PathRewrite, RawDataPath

_F_PATH_REWRITE = "_F_PATH_REWRITE"


async def download_tgz(destination: str, version: str, tgz: str) -> CodeBundle:
    """
    Downloads and loads the task from the code bundle or resolver.

    Args:
        tgz: The path to the task template in a tar.gz format.
        destination: The path to save the downloaded task template.
        version: The version of the task to load.

    Returns:
        The CodeBundle object.
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

    Args:
        pkl: The path to the task template in a pickle format.
        destination: The path to save the downloaded task template.
        version: The version of the task to load.

    Returns:
        The CodeBundle object.
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

    Args:
        resolver: The resolver to use to load the task.
        resolver_args: The arguments to pass to the resolver.

    Returns:
        The loaded task template.
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

    Args:
        endpoint:
        insecure:
        api_key:
    """
    logger.info(f"[rusty] Creating controller with endpoint {endpoint}")
    import flyte.errors
    from flyte._initialize import init_in_cluster

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(flyte.errors.silence_polling_error)

    # TODO Currently remote tasks are not supported in Rusty.
    controller_kwargs = await init_in_cluster.aio(api_key=api_key, endpoint=endpoint, insecure=insecure)
    return _create_controller(ct="remote", **controller_kwargs)


def _parse_attempt_number(attempt_number: int | str | None) -> int | None:
    """Normalize the per-invocation attempt number, returning None when it is missing or unusable."""
    if attempt_number is None or isinstance(attempt_number, bool):
        return None
    try:
        value = int(attempt_number)
    except (TypeError, ValueError):
        logger.warning(f"[rusty] Ignoring invalid attempt number {attempt_number!r}")
        return None
    if value < 0:
        logger.warning(f"[rusty] Ignoring negative attempt number {attempt_number!r}")
        return None
    return value


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
    # The actor worker parses the RFC3339 string (and the unsubstituted "{{.runStartTime}}"
    # placeholder → None) Rust-side and passes a datetime here; absent → None and TaskContext
    # fills in now(UTC).
    run_start_time: Optional[datetime] = None,
    # The 0-based retry attempt of this invocation. The actor worker receives it per task in the
    # FLYTE_ATTEMPT_NUMBER entry of the env vars that come with each assignment, and should pass that
    # value here. The worker's own process environment cannot be used instead: it is set once when the
    # replica starts, so every retry routed to a warm replica would otherwise read the first value.
    attempt_number: int | str | None = None,
):
    """
    Runs the task with the provided parameters.

    Args:
        prev_checkpoint: Previous checkpoint path to resume from.
        checkpoint_path: Checkpoint path to save the current state.
        image_cache: Image cache to use for the task.
        name: Action name to run.
        run_name: Parent run name to use for the task.
        domain: domain to run the task in.
        project: project to run the task in.
        org: organization to run the task in.
        task: The task template to run.
        raw_data_path: The path to the raw data.
        output_path: The path to save the output.
        run_base_dir: The base directory for the run.
        version: The version of the task to run.
        controller: The controller to use for the task.
        code_bundle: Optional code bundle for the task.
        input_path: Optional input path for the task.
        path_rewrite_cfg: Optional path rewrite configuration.
        run_start_time: Optional start time of the parent run.
        attempt_number: Optional 0-based retry attempt of this invocation, as an int or its string form.
            When absent, `TaskContext.attempt_number` falls back to the process environment.

    Returns:
        The loaded task template.
    """
    start_time = time.time()
    action_id = f"{org}/{project}/{domain}/{run_name}/{name}"

    logger.info(
        f"[rusty] Running task '{task.name}' (action: {action_id})"
        f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    if path_rewrite_cfg is None:
        # Workers may not forward the path-rewrite config as an argument; fall back
        # to the pod env var, same as the standard runtime entrypoint.
        path_rewrite_cfg = os.getenv(_F_PATH_REWRITE, None)

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

    retry_attempt = _parse_attempt_number(attempt_number)

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
            checkpoint_paths=CheckpointPaths(prev_checkpoint_path=prev_checkpoint, checkpoint_path=checkpoint_path),
            code_bundle=code_bundle,
            input_path=input_path,
            image_cache=ImageCache.from_transport(image_cache) if image_cache else None,
            run_start_time=run_start_time,
            retry_attempt=retry_attempt,
        )
    except asyncio.CancelledError as e:
        logger.error(f"[rusty] Task cancellation received: {e!s}")
        raise
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
        _flush_output_streams()


def _flush_output_streams() -> None:
    """
    Flush stdout, stderr and the flyte log handlers after a task invocation.

    The a0 entrypoint flushes these once, right before its process exits. A reusable container's worker
    process runs one invocation after another and never reaches that point, and when its output goes to a
    pipe Python buffers it in blocks, so print() output from a task could sit in the buffer indefinitely
    and never show up in that task's logs. Flushing at the end of every invocation gets it out while it
    still belongs to the task that wrote it. A stream that user code closed or replaced with something
    that cannot flush is skipped, since a logging hiccup must not change the outcome of the task.
    """
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


async def ping(name: str) -> str:
    """
    A simple hello world function to test the Rusty entrypoint.
    """
    print(f"Received ping request from {name} in Rusty!")
    return f"pong from Rusty to {name}!"


async def hello(name: str):
    """
    A simple hello world function to test the Rusty entrypoint.

    Args:
        name: The name to greet.

    Returns:
        A greeting message.
    """
    print(f"Received hello request in Rusty with name: {name}!")
