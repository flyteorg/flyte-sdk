"""
Flyte runtime module, this is the entrypoint script for the Flyte runtime.

Caution: Startup time for this module is very important, as it is the entrypoint for the Flyte runtime.
Refrain from importing any modules here. If you need to import any modules, do it inside the main function.
"""

import asyncio
import os
import sys
from typing import Any, List

import click

# Todo: work with pvditt to make these the names
# ACTION_NAME = "_U_ACTION_NAME"
# RUN_NAME = "_U_RUN_NAME"
# PROJECT_NAME = "_U_PROJECT_NAME"
# DOMAIN_NAME = "_U_DOMAIN_NAME"
# ORG_NAME = "_U_ORG_NAME"

ACTION_NAME = "ACTION_NAME"
RUN_NAME = "RUN_NAME"
PROJECT_NAME = "FLYTE_INTERNAL_TASK_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_TASK_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"
RUN_OUTPUT_BASE_DIR = "_U_RUN_BASE"
ENABLE_REF_TASKS = "_REF_TASKS"  # This is a temporary flag to enable reference tasks in the runtime.

# TODO: Remove this after proper auth is implemented
_UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"


@click.group()
def _pass_through():
    pass


@_pass_through.command("a0")
@click.option("--inputs", "-i", required=True)
@click.option("--outputs-path", "-o", required=True)
@click.option("--version", "-v", required=True)
@click.option("--run-base-dir", envvar=RUN_OUTPUT_BASE_DIR, required=True)
@click.option("--raw-data-path", "-r", required=False)
@click.option("--checkpoint-path", "-c", required=False)
@click.option("--prev-checkpoint", "-p", required=False)
@click.option("--name", envvar=ACTION_NAME, required=False)
@click.option("--run-name", envvar=RUN_NAME, required=False)
@click.option("--project", envvar=PROJECT_NAME, required=False)
@click.option("--domain", envvar=DOMAIN_NAME, required=False)
@click.option("--org", envvar=ORG_NAME, required=False)
@click.option("--image-cache", required=False)
@click.option("--tgz", required=False)
@click.option("--pkl", required=False)
@click.option("--dest", required=False)
@click.option("--resolver", required=False)
@click.argument(
    "resolver-args",
    type=click.UNPROCESSED,
    nargs=-1,
)
def main(
    run_name: str,
    name: str,
    project: str,
    domain: str,
    org: str,
    image_cache: str,
    version: str,
    inputs: str,
    run_base_dir: str,
    outputs_path: str,
    raw_data_path: str,
    checkpoint_path: str,
    prev_checkpoint: str,
    tgz: str,
    pkl: str,
    dest: str,
    resolver: str,
    resolver_args: List[str],
):
    sys.path.insert(0, ".")

    import faulthandler
    import signal

    import flyte
    import flyte._utils as utils
    from flyte._initialize import init
    from flyte._internal.controllers import create_controller
    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte._internal.runtime.entrypoints import load_and_run_task
    from flyte._logging import logger
    from flyte.models import ActionID, Checkpoints, CodeBundle, RawDataPath

    logger.info("Registering faulthandler for SIGUSR1")
    faulthandler.register(signal.SIGUSR1)

    logger.info(f"Initializing flyte runtime - version {flyte.__version__}")
    assert org, "Org is required for now"
    assert project, "Project is required"
    assert domain, "Domain is required"
    assert run_name, f"Run name is required {run_name}"
    assert name, f"Action name is required {name}"

    if run_name.startswith("{{"):
        run_name = os.getenv("RUN_NAME", "")
    if name.startswith("{{"):
        name = os.getenv("ACTION_NAME", "")

    # Figure out how to connect
    # This detection of api key is a hack for now.
    controller_kwargs: dict[str, Any] = {"insecure": False}
    if api_key := os.getenv(_UNION_EAGER_API_KEY_ENV_VAR):
        logger.info("Using api key from environment")
        controller_kwargs["api_key"] = api_key
    else:
        ep = os.environ.get(ENDPOINT_OVERRIDE, "host.docker.internal:8090")
        controller_kwargs["endpoint"] = ep
        if "localhost" in ep or "docker" in ep:
            controller_kwargs["insecure"] = True
        logger.debug(f"Using controller endpoint: {ep} with kwargs: {controller_kwargs}")

    bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
    enable_ref_tasks = os.getenv(ENABLE_REF_TASKS, "false").lower() in ("true", "1", "yes")
    # We init regular client here so that reference tasks can work
    # Current reference tasks will not work with remote controller, because we create 2 different
    # channels on different threads and this is not supported by grpcio or the auth system. It ends up leading
    # File "src/python/grpcio/grpc/_cython/_cygrpc/aio/completion_queue.pyx.pxi", line 147,
    # in grpc._cython.cygrpc.PollerCompletionQueue._handle_events
    # BlockingIOError: [Errno 11] Resource temporarily unavailable
    # TODO solution is to use a single channel for both controller and reference tasks, but this requires a refactor
    if enable_ref_tasks:
        logger.warning(
            "Reference tasks are enabled. This will initialize client and you will see a BlockIOError. "
            "This is harmless, but a nuisance. We are working on a fix."
        )
        init(org=org, project=project, domain=domain, **controller_kwargs)
    else:
        init()
    # Controller is created with the same kwargs as init, so that it can be used to run tasks
    controller = create_controller(ct="remote", **controller_kwargs)

    ic = ImageCache.from_transport(image_cache) if image_cache else None

    # Create a coroutine to load the task and run it
    task_coroutine = load_and_run_task(
        resolver=resolver,
        resolver_args=resolver_args,
        action=ActionID(name=name, run_name=run_name, project=project, domain=domain, org=org),
        raw_data_path=RawDataPath(path=raw_data_path),
        checkpoints=Checkpoints(checkpoint_path, prev_checkpoint),
        code_bundle=bundle,
        input_path=inputs,
        output_path=outputs_path,
        run_base_dir=run_base_dir,
        version=version,
        controller=controller,
        image_cache=ic,
    )
    # Create a coroutine to watch for errors
    controller_failure = controller.watch_for_errors()

    # Run both coroutines concurrently and wait for first to finish and cancel the other
    async def _run_and_stop():
        await utils.run_coros(controller_failure, task_coroutine)
        await controller.stop()

    asyncio.run(_run_and_stop())


if __name__ == "__main__":
    _pass_through()
