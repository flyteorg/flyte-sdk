import pathlib
from typing import Tuple

from flyte._module import extract_obj_module
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate


def extract_task_module(task: TaskTemplate, /, source_dir: pathlib.Path) -> Tuple[str, str]:
    """
    Extract the task module from the task template.

    :param task: The task template to extract the module from.
    :param source_dir: The source directory to use for relative paths.
    :return: A tuple containing the entity name, module
    """
    if isinstance(task, AsyncFunctionTaskTemplate):
        entity_name = task.func.__name__
        entity_module_name, _ = extract_obj_module(task.func, source_dir)
        return entity_name, entity_module_name
    else:
        raise NotImplementedError(f"Task module {task.name} not implemented.")
