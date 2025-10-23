import inspect
import os
import pathlib
import sys
from typing import Tuple

from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate


def extract_task_module(task: TaskTemplate, /, source_dir: pathlib.Path | None = None) -> Tuple[str, str]:
    """
    Extract the task module from the task template.

    :param task: The task template to extract the module from.
    :param source_dir: The source directory to use for relative paths.
    :return: A tuple containing the entity name, module
    """
    entity_name = task.name
    if isinstance(task, AsyncFunctionTaskTemplate):
        entity_module = inspect.getmodule(task.func)
        if entity_module is None:
            raise ValueError(f"Task {entity_name} has no module.")

        fp = entity_module.__file__
        if fp is None:
            raise ValueError(f"Task {entity_name} has no module.")

        file_path = pathlib.Path(fp)
        try:
            # Get the relative path to the current directory
            # Will raise ValueError if the file is not in the source directory
            relative_path = file_path.relative_to(str(source_dir))

            if relative_path == pathlib.Path("."):
                entity_module_name = entity_module.__name__
            else:
                # Replace file separators with dots and remove the '.py' extension
                dotted_path = os.path.splitext(str(relative_path))[0].replace(os.sep, ".")
                entity_module_name = dotted_path
        except ValueError:
            # If source_dir is not provided or file is not in source_dir, fallback to module name
            # File is not relative to source_dir - check if it's an installed package
            file_path_str = str(file_path)
            if "site-packages" in file_path_str or "dist-packages" in file_path_str:
                # It's an installed package - use the module's __name__ directly
                # This will be importable via importlib.import_module()
                entity_module_name = entity_module.__name__
            else:
                # File is not in source_dir and not in site-packages - re-raise the error
                raise ValueError(
                    f"Task {entity_name} module file {file_path} is not relative to "
                    f"source directory {source_dir} and is not an installed package."
                )

        entity_name = task.func.__name__
    else:
        raise NotImplementedError(f"Task module {entity_name} not implemented.")

    if entity_module_name == "__main__":
        """
        This case is for the case in which the task is run from the main module.
        """
        fp = sys.modules["__main__"].__file__
        if fp is None:
            raise ValueError(f"Task {entity_name} has no module.")
        main_path = pathlib.Path(fp)
        entity_module_name = main_path.stem

    return entity_name, entity_module_name
