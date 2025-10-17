"""
This module is responsible for running tasks in the V2 runtime. All methods in this file should be
invoked within a context tree.
"""

import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import flyte.report
from flyte._context import internal_ctx
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._logging import log, logger
from flyte._task import TaskTemplate
from flyte.errors import CustomError, RuntimeSystemError, RuntimeUnknownError, RuntimeUserError
from flyte.models import ActionID, Checkpoints, CodeBundle, RawDataPath, TaskContext

from .. import Controller
from .convert import (
    Error,
    Inputs,
    Outputs,
    convert_from_native_to_error,
    convert_from_native_to_outputs,
    convert_inputs_to_native,
)
from .io import load_inputs, upload_error, upload_outputs


def replace_task_cli(args: List[str], inputs: Inputs, tmp_path: pathlib.Path, action: ActionID) -> List[str]:
    """
    This method can be used to run an task from the cli, if you have cli for the task. It will replace,
    all the args with the task args.

    The urun cli is of the format
    ```python
    ['urun', '--inputs', '{{.Inputs}}', '--outputs-path', '{{.Outputs}}', '--version', '',
     '--raw-data-path', '{{.rawOutputDataPrefix}}',
      '--checkpoint-path', '{{.checkpointOutputPrefix}}', '--prev-checkpoint', '{{.prevCheckpointPrefix}}',
       '--run-name', '{{.runName}}', '--name', '{{.actionName}}',
        '--tgz', 'some-path', '--dest', '.',
         '--resolver', 'flyte._internal.resolvers.default.DefaultTaskResolver', '--resolver-args',
          'mod', 'test_round_trip', 'instance', 'task1']
    ```
    We will replace, inputs, outputs, raw_data_path, checkpoint_path, prev_checkpoint, run_name, name
    with supplied values.

    :param args: urun command
    :param inputs: converted inputs to the task
    :param tmp_path: temporary path to use for the task
    :param action: run id to use for the task
    :return: modified args
    """
    # Iterate over all the args and replace the inputs, outputs, raw_data_path, checkpoint_path, prev_checkpoint,
    # root_name, run_name with the supplied values
    # first we will write the inputs to a file called inputs.pb
    inputs_path = tmp_path / "inputs.pb"
    with open(inputs_path, "wb") as f:
        f.write(inputs.proto_inputs.SerializeToString())
    # now modify the args
    args = list(args)  # copy first because it's a proto container
    for i, arg in enumerate(args):
        match arg:
            case "--inputs":
                args[i + 1] = str(inputs_path)
            case "--outputs-path":
                args[i + 1] = str(tmp_path)
            case "--raw-data-path":
                args[i + 1] = str(tmp_path / "raw_data_path")
            case "--checkpoint-path":
                args[i + 1] = str(tmp_path / "checkpoint_path")
            case "--prev-checkpoint":
                args[i + 1] = str(tmp_path / "prev_checkpoint")
            case "--run-name":
                args[i + 1] = action.run_name or ""
            case "--name":
                args[i + 1] = action.name
    insert_point = args.index("--raw-data-path")
    args.insert(insert_point, str(tmp_path))
    args.insert(insert_point, "--run-base-dir")
    return args


@log
async def run_task(
    tctx: TaskContext, controller: Controller, task: TaskTemplate, inputs: Dict[str, Any]
) -> Tuple[Any, Optional[Exception]]:
    try:
        logger.info(f"Parent task executing {tctx.action}")
        outputs = await task.execute(**inputs)
        logger.info(f"Parent task completed successfully, {tctx.action}")
        return outputs, None
    except RuntimeSystemError as e:
        logger.exception(f"Task failed with error: {e}")
        return {}, e
    except RuntimeUnknownError as e:
        logger.exception(f"Task failed with error: {e}")
        return {}, e
    except RuntimeUserError as e:
        logger.exception(f"Task failed with error: {e}")
        return {}, e
    except Exception as e:
        logger.exception(f"Task failed with error: {e}")
        return {}, CustomError.from_exception(e)
    finally:
        logger.info(f"Parent task finalized {tctx.action}")
        # reconstruct run id here
        await controller.finalize_parent_action(tctx.action)


async def convert_and_run(
    *,
    task: TaskTemplate,
    action: ActionID,
    controller: Controller,
    raw_data_path: RawDataPath,
    version: str,
    output_path: str,
    run_base_dir: str,
    inputs: Inputs = Inputs.empty(),
    input_path: str | None = None,
    checkpoints: Checkpoints | None = None,
    code_bundle: CodeBundle | None = None,
    image_cache: ImageCache | None = None,
    interactive_mode: bool = False,
) -> Tuple[Optional[Outputs], Optional[Error]]:
    """
    This method is used to convert the inputs to native types, and run the task. It assumes you are running
    in a context tree.
    """
    ctx = internal_ctx()

    # Load inputs first to get context
    if input_path:
        inputs = await load_inputs(input_path, path_rewrite_config=raw_data_path.path_rewrite)

    # Extract context from inputs
    custom_context = inputs.context if inputs else {}

    tctx = TaskContext(
        action=action,
        checkpoints=checkpoints,
        code_bundle=code_bundle,
        input_path=input_path,
        output_path=output_path,
        run_base_dir=run_base_dir,
        version=version,
        raw_data_path=raw_data_path,
        compiled_image_cache=image_cache,
        report=flyte.report.Report(name=action.name),
        mode="remote" if not ctx.data.task_context else ctx.data.task_context.mode,
        interactive_mode=interactive_mode,
        custom_context=custom_context,
    )

    with ctx.replace_task_context(tctx):
        inputs_kwargs = await convert_inputs_to_native(inputs, task.native_interface)
        out, err = await run_task(tctx=tctx, controller=controller, task=task, inputs=inputs_kwargs)
        if err is not None:
            return None, convert_from_native_to_error(err)
        if task.report:
            await flyte.report.flush.aio()
        return await convert_from_native_to_outputs(out, task.native_interface, task.name), None


async def extract_download_run_upload(
    task: TaskTemplate,
    *,
    action: ActionID,
    controller: Controller,
    raw_data_path: RawDataPath,
    output_path: str,
    run_base_dir: str,
    version: str,
    checkpoints: Checkpoints | None = None,
    code_bundle: CodeBundle | None = None,
    input_path: str | None = None,
    image_cache: ImageCache | None = None,
    interactive_mode: bool = False,
):
    """
    This method is invoked from the CLI (urun) and is used to run a task. This assumes that the context tree
    has already been created, and the task has been loaded. It also handles the loading of the task.
    """
    t = time.time()
    logger.warning(f"Task {action.name} started at {t}")
    outputs, err = await convert_and_run(
        task=task,
        input_path=input_path,
        action=action,
        controller=controller,
        raw_data_path=raw_data_path,
        output_path=output_path,
        run_base_dir=run_base_dir,
        version=version,
        checkpoints=checkpoints,
        code_bundle=code_bundle,
        image_cache=image_cache,
        interactive_mode=interactive_mode,
    )
    if err is not None:
        path = await upload_error(err.err, output_path)
        logger.error(f"Task {task.name} failed with error: {err}. Uploaded error to {path}")
        return
    if outputs is None:
        logger.info(f"Task {task.name} completed successfully, no outputs")
        return
    await upload_outputs(outputs, output_path) if output_path else None
    logger.warning(f"Task {task.name} completed successfully, uploaded outputs to {output_path} in {time.time() - t}s")
