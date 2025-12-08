"""
Flyte runtime module, this is the entrypoint script for the Flyte runtime.

Caution: Startup time for this module is very important, as it is the entrypoint for the Flyte runtime.
Refrain from importing any modules here. If you need to import any modules, do it inside the main function.
"""

import asyncio
import os
import sys
from typing import List

import click

from flyte.models import PathRewrite

# Todo: work with pvditt to make these the names
# ACTION_NAME = "_U_ACTION_NAME"
# RUN_NAME = "_U_RUN_NAME"
# PROJECT_NAME = "_U_PROJECT_NAME"
# DOMAIN_NAME = "_U_DOMAIN_NAME"
# ORG_NAME = "_U_ORG_NAME"

ACTION_NAME = "ACTION_NAME"
RUN_NAME = "RUN_NAME"
PROJECT_NAME = "FLYTE_INTERNAL_EXECUTION_PROJECT"
DOMAIN_NAME = "FLYTE_INTERNAL_EXECUTION_DOMAIN"
ORG_NAME = "_U_ORG_NAME"
ENDPOINT_OVERRIDE = "_U_EP_OVERRIDE"
INSECURE_SKIP_VERIFY_OVERRIDE = "_U_INSECURE_SKIP_VERIFY"
RUN_OUTPUT_BASE_DIR = "_U_RUN_BASE"
FLYTE_ENABLE_VSCODE_KEY = "_F_E_VS"

_UNION_EAGER_API_KEY_ENV_VAR = "_UNION_EAGER_API_KEY"
_F_PATH_REWRITE = "_F_PATH_REWRITE"


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
@click.option("--debug", envvar=FLYTE_ENABLE_VSCODE_KEY, type=click.BOOL, required=False)
@click.option("--interactive-mode", type=click.BOOL, required=False)
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
@click.pass_context
def main(
    ctx: click.Context,
    run_name: str,
    name: str,
    project: str,
    domain: str,
    org: str,
    debug: bool,
    interactive_mode: bool,
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
    import flyte.errors
    import flyte.storage as storage
    from flyte._initialize import init_in_cluster
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

    logger.warning(f"Flyte runtime started for action {name} with run name {run_name}")

    if debug and name == "a0":
        from flyte._debug.vscode import _start_vscode_server

        asyncio.run(_start_vscode_server(ctx))

    controller_kwargs = init_in_cluster(org=org, project=project, domain=domain)
    bundle = None
    if tgz or pkl:
        bundle = CodeBundle(tgz=tgz, pkl=pkl, destination=dest, computed_version=version)
    # Controller is created with the same kwargs as init, so that it can be used to run tasks
    controller = create_controller(ct="remote", **controller_kwargs)

    ic = ImageCache.from_transport(image_cache) if image_cache else None

    path_rewrite_cfg = os.getenv(_F_PATH_REWRITE, None)
    path_rewrite = None
    if path_rewrite_cfg:
        potential_path_rewrite = PathRewrite.from_str(path_rewrite_cfg)
        if storage.exists_sync(potential_path_rewrite.new_prefix):
            path_rewrite = potential_path_rewrite
            logger.info(f"Path rewrite configured for {path_rewrite.new_prefix}")
        else:
            logger.error(
                f"Path rewrite failed for path {potential_path_rewrite.new_prefix}, "
                f"not found, reverting to original path {potential_path_rewrite.old_prefix}"
            )

    # Create a coroutine to load the task and run it
    task_coroutine = load_and_run_task(
        resolver=resolver,
        resolver_args=resolver_args,
        action=ActionID(name=name, run_name=run_name, project=project, domain=domain, org=org),
        raw_data_path=RawDataPath(path=raw_data_path, path_rewrite=path_rewrite),
        checkpoints=Checkpoints(checkpoint_path, prev_checkpoint),
        code_bundle=bundle,
        input_path=inputs,
        output_path=outputs_path,
        run_base_dir=run_base_dir,
        version=version,
        controller=controller,
        image_cache=ic,
        interactive_mode=interactive_mode or debug,
    )
    # Create a coroutine to watch for errors
    controller_failure = controller.watch_for_errors()

    # Run both coroutines concurrently and wait for first to finish and cancel the other
    async def _run_and_stop():
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(flyte.errors.silence_grpc_polling_error)
        try:
            await utils.run_coros(controller_failure, task_coroutine)
            await controller.stop()
        except flyte.errors.RuntimeSystemError as e:
            logger.error(f"Runtime system error: {e}")
            from flyte._internal.runtime.convert import convert_from_native_to_error
            from flyte._internal.runtime.io import upload_error

            logger.error(f"Flyte runtime failed for action {name} with run name {run_name}, error: {e}")
            err = convert_from_native_to_error(e)
            path = await upload_error(err.err, outputs_path)
            logger.error(f"Run {run_name} Action {name} failed with error: {err}. Uploaded error to {path}")
            await controller.stop()
            raise

    asyncio.run(_run_and_stop())
    logger.warning(f"Flyte runtime completed for action {name} with run name {run_name}")


if __name__ == "__main__":
    _pass_through()
