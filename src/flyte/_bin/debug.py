

import click

from flyte._internal.runtime.convert import Inputs
from flyte._internal.runtime.io import load_inputs


@click.group()
def _debug():
    """Debug commands for Flyte."""


@_debug.command("step-through")
@click.option("--task-module-name", "-m", required=True, help="Name of the task module.")
@click.option("--task-name", "-t", required=True, help="Name of the task function.")
@click.option("--context-working-dir", "-w", required=True, help="Working directory for the task context.")
@click.option("--input_path", "-i", required=False, help="Path to the inputs file for the task.")
def step_through(task_module_name, task_name, context_working_dir, input_path):
    """
    Step through a Flyte task for debugging purposes.

    Args:
        task_module_name (str): Name of the Python module containing the task.
        task_name (str): Name of the task function within the module.
        context_working_dir (str): Directory path where the input file and module file are located.
        input_path (str): Path to the input file for the task.
    """
    from flyte._debug.utils import get_task_inputs
    from flyte._internal.runtime.convert import convert_inputs_to_native
    import asyncio

    inputs = asyncio.run(load_inputs(input_path)) if input_path else Inputs.empty()
    inputs_kwargs = asyncio.run(convert_inputs_to_native(inputs, task.native_interface))

    inputs = get_task_inputs(task_module_name, task_name, context_working_dir)
    print(f"Inputs for {task_name} in {task_module_name}: {inputs}")


@_debug.command("resume")
@click.option("--pid", "-m", type=int, required=True, help="PID of the vscode server.")
def resume(pid):
    """
    Resume a Flyte task for debugging purposes.

    Args:
        pid (int): PID of the vscode server.
    """
    import os
    import signal

    print("Terminating server and resuming task.")
    answer = input(
        "This operation will kill the server. All unsaved data will be lost, and you will no longer be able to connect to it. Do you really want to terminate? (Y/N): ").strip().upper()
    if answer == 'Y':
        os.kill(pid, signal.SIGTERM)
        print(f"The server has been terminated and the task has been resumed.")
    else:
        print("Operation canceled.")


if __name__ == "__main__":
    _debug()
