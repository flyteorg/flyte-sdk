import pathlib
from types import FunctionType
from typing import Tuple, cast

from flyte._module import extract_obj_module
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate


def extract_task_module(task: TaskTemplate, /, source_dir: pathlib.Path) -> Tuple[str, str]:
    """
    Extract the task module from the task template.

    Args:
        task: The task template to extract the module from.
        source_dir: The source directory to use for relative paths.

    Returns:
        A tuple containing the entity name, module
    """
    if isinstance(task, AsyncFunctionTaskTemplate):
        entity_name = cast(FunctionType, task.func).__name__
        entity_module_name, entity_module = extract_obj_module(task.func, source_dir)

        # CodeTaskTemplate uses a dummy lambda — find it by scanning the module.
        if not entity_name.isidentifier():
            for attr in vars(entity_module):
                if getattr(entity_module, attr, None) is task:
                    return attr, entity_module_name
            # The deploy may have loaded the module a second time (as a file, then by import), so the attribute
            # holds an equal task from the other load rather than this object: match it by name, but only when
            # exactly one attribute matches (an alias or a name collision would make the choice a guess).
            matches = [
                attr
                for attr, value in vars(entity_module).items()
                if isinstance(value, TaskTemplate) and value.name == task.name
            ]
            if len(matches) == 1:
                return matches[0], entity_module_name
            raise ValueError(f"Task '{task.name}' not found as a module-level attribute in '{entity_module_name}'")

        return entity_name, entity_module_name
    else:
        raise NotImplementedError(f"Task module {task.name} not implemented.")
