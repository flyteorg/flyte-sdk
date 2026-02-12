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
        entity_module_name, entity_module = extract_obj_module(task.func, source_dir)

        # CodeTaskTemplate uses a dummy lambda — find it by scanning the module
        if not entity_name.isidentifier():
            for attr in vars(entity_module):
                if getattr(entity_module, attr, None) is task:
                    return attr, entity_module_name
            raise ValueError(
                f"Task '{task.name}' not found as a module-level attribute in '{entity_module_name}'"
            )

        return entity_name, entity_module_name
    else:
        raise NotImplementedError(f"Task module {task.name} not implemented.")
