import importlib
import os
import traceback
from typing import List, Optional, Tuple, Type

import flyte.errors
from flyte._code_bundle import download_bundle
from flyte._context import contextual_run
from flyte._internal import Controller
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._logging import log, logger
from flyte._task import TaskTemplate
from flyte.models import ActionID, Checkpoints, CodeBundle, RawDataPath

from ..._utils import adjust_sys_path
from .convert import Error, Inputs, Outputs
from .taskrunner import (
    convert_and_run,
    extract_download_run_upload,
)


async def direct_dispatch(
    task: TaskTemplate,
    *,
    action: ActionID,
    raw_data_path: RawDataPath,
    controller: Controller,
    version: str,
    output_path: str,
    run_base_dir: str,
    checkpoints: Checkpoints | None = None,
    code_bundle: CodeBundle | None = None,
    inputs: Inputs | None = None,
) -> Tuple[Optional[Outputs], Optional[Error]]:
    """
    This method is used today by the local_controller and is positioned to be used by a rust core in the future.
    The caller, loads the task and invokes this method. This method is used to convert the inputs to native types,
    The reason for this is that the rust entrypoint will not have access to the python context, and
    will not be able to run the tasks in the context tree.
    """
    return await contextual_run(
        convert_and_run,
        task=task,
        inputs=inputs or Inputs.empty(),
        action=action,
        raw_data_path=raw_data_path,
        checkpoints=checkpoints,
        code_bundle=code_bundle,
        controller=controller,
        version=version,
        output_path=output_path,
        run_base_dir=run_base_dir,
    )


def load_class(qualified_name) -> Type:
    """
    Load a class from a qualified name. The qualified name should be in the format 'module.ClassName'.
    :param qualified_name: The qualified name of the class to load.
    :return: The class object.
    """
    module_name, class_name = qualified_name.rsplit(".", 1)  # Split module and class
    module = importlib.import_module(module_name)  # Import the module
    return getattr(module, class_name)  # Retrieve the class


def load_task(resolver: str, *resolver_args: str) -> TaskTemplate:
    """
    Load a task from a resolver. This is a placeholder function.

    :param resolver: The resolver to use to load the task.
    :param resolver_args: Arguments to pass to the resolver.
    :return: The loaded task.
    """
    resolver_class = load_class(resolver)
    resolver_instance = resolver_class()
    try:
        return resolver_instance.load_task(resolver_args)
    except ModuleNotFoundError as e:
        cwd = os.getcwd()
        files = []
        try:
            for root, dirs, filenames in os.walk(cwd):
                for name in dirs + filenames:
                    rel_path = os.path.relpath(os.path.join(root, name), cwd)
                    files.append(rel_path)
        except Exception as list_err:
            files = [f"(Failed to list directory: {list_err})"]

        msg = (
            "\n\nFull traceback:\n" + "".join(traceback.format_exc()) + f"\n[ImportError Diagnostics]\n"
            f"Module '{e.name}' not found in either the Python virtual environment or the current working directory.\n"
            f"Current working directory: {cwd}\n"
            f"Files found under current directory:\n" + "\n".join(f"  - {f}" for f in files)
        )
        raise ModuleNotFoundError(msg) from e


def load_pkl_task(code_bundle: CodeBundle) -> TaskTemplate:
    """
    Loads a task from a pickled code bundle.
    :param code_bundle: The code bundle to load the task from.
    :return: The loaded task template.
    """
    logger.debug(f"Loading task from pkl: {code_bundle.downloaded_path}")
    try:
        import gzip

        import cloudpickle

        with gzip.open(str(code_bundle.downloaded_path), "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        logger.exception(f"Failed to load pickled task from {code_bundle.downloaded_path}. Reason: {e!s}")
        raise


async def download_code_bundle(code_bundle: CodeBundle) -> CodeBundle:
    """
    Downloads the code bundle if it is not already downloaded.
    :param code_bundle: The code bundle to download.
    :return: The code bundle with the downloaded path.
    """
    adjust_sys_path()
    logger.debug(f"Downloading {code_bundle}")
    downloaded_path = await download_bundle(code_bundle)
    return code_bundle.with_downloaded_path(downloaded_path)


async def _download_and_load_task(
    code_bundle: CodeBundle | None, resolver: str | None = None, resolver_args: List[str] | None = None
) -> TaskTemplate:
    if code_bundle and (code_bundle.tgz or code_bundle.pkl):
        logger.debug(f"Downloading {code_bundle}")
        code_bundle = await download_code_bundle(code_bundle)
        if code_bundle.pkl:
            return load_pkl_task(code_bundle)

        if not resolver or not resolver_args:
            raise flyte.errors.RuntimeSystemError(
                "MalformedCommand", "Resolver and resolver args are required. for task"
            )
        logger.debug(
            f"Loading task from tgz: {code_bundle.downloaded_path}, resolver: {resolver}, args: {resolver_args}"
        )
        return load_task(resolver, *resolver_args)
    if not resolver or not resolver_args:
        raise flyte.errors.RuntimeSystemError("MalformedCommand", "Resolver and resolver args are required. for task")
    logger.debug(f"No code bundle provided, loading task from resolver: {resolver}, args: {resolver_args}")
    return load_task(resolver, *resolver_args)


@log
async def load_and_run_task(
    action: ActionID,
    raw_data_path: RawDataPath,
    output_path: str,
    run_base_dir: str,
    version: str,
    controller: Controller,
    resolver: str,
    resolver_args: List[str],
    checkpoints: Checkpoints | None = None,
    code_bundle: CodeBundle | None = None,
    input_path: str | None = None,
    image_cache: ImageCache | None = None,
    interactive_mode: bool = False,
):
    """
    This method is invoked from the runtime/CLI and is used to run a task. This creates the context tree,
    for the tasks to run in. It also handles the loading of the task.

    :param controller: Controller to use for the task.
    :param resolver: The resolver to use to load the task.
    :param resolver_args: The arguments to pass to the resolver.
    :param action: The ActionID to use for the task.
    :param raw_data_path: The raw data path to use for the task.
    :param output_path: The output path to use for the task.
    :param run_base_dir: Base output directory to pass down to child tasks.
    :param version: The version of the task to run.
    :param checkpoints: The checkpoints to use for the task.
    :param code_bundle: The code bundle to use for the task.
    :param input_path: The input path to use for the task.
    :param image_cache: Mappings of Image identifiers to image URIs.
    :param interactive_mode: Whether to run the task in interactive mode.
    """
    task = await _download_and_load_task(code_bundle, resolver, resolver_args)

    await contextual_run(
        extract_download_run_upload,
        task,
        action=action,
        version=version,
        controller=controller,
        raw_data_path=raw_data_path,
        output_path=output_path,
        run_base_dir=run_base_dir,
        checkpoints=checkpoints,
        code_bundle=code_bundle,
        input_path=input_path,
        image_cache=image_cache,
        interactive_mode=interactive_mode,
    )
