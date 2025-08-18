import click


@click.group()
def debug():
    """Debug commands for Flyte."""


@debug.command("step-through")
@click.option("--task-module-name", "-m", required=True, help="Name of the task module.")
@click.option("--task-name", "-t", required=True, help="Name of the task function.")
@click.option("--context-working-dir", "-w", required=True, help="Working directory for the task context.")
@click.option("--inputs-file", "-i", required=False, help="Path to the inputs file for the task.")
def step_through(task_module_name, task_name, context_working_dir, inputs_file):
    """
    Step through a Flyte task for debugging purposes.

    Args:
        task_module_name (str): Name of the Python module containing the task.
        task_name (str): Name of the task function within the module.
        context_working_dir (str): Directory path where the input file and module file are located.
    """
    from flyte._debug.utils import get_task_inputs

    inputs = get_task_inputs(task_module_name, task_name, context_working_dir)
    print(f"Inputs for {task_name} in {task_module_name}: {inputs}")


@debug.command("resume")
@click.option("--task-module-name", "-m", required=True, help="Name of the task module.")
@click.option("--task-name", "-t", required=True, help="Name of the task function.")
@click.option("--context-working-dir", "-w", required=True, help="Working directory for the task context.")
@click.option("--inputs-file", "-i", required=False, help="Path to the inputs file for the task.")
def resume(task_module_name, task_name, context_working_dir, inputs_file):
    """
    Resume a Flyte task for debugging purposes.

    Args:
        task_module_name (str): Name of the Python module containing the task.
        task_name (str): Name of the task function within the module.
        context_working_dir (str): Directory path where the input file and module file are located.
    """
    from flyte._debug.utils import get_task_inputs

    inputs = get_task_inputs(task_module_name, task_name, context_working_dir)
    print(f"Resuming {task_name} in {task_module_name} with inputs: {inputs}")
