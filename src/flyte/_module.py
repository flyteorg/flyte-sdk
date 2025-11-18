import inspect
import os
import pathlib
import sys
from types import ModuleType
from typing import Tuple


def extract_obj_module(obj: object, /, source_dir: pathlib.Path | None = None) -> Tuple[str, ModuleType]:
    """
    Extract the module from the given object. If source_dir is provided, the module will be relative to the source_dir.

    Args:
        obj: The object to extract the module from.
        source_dir: The source directory to use for relative paths.

    Returns:
        The module name as a string.
    """
    if source_dir is None:
        raise ValueError("extract_obj_module: source_dir cannot be None - specify root-dir")
    # Get the module containing the object
    entity_module = inspect.getmodule(obj)
    if entity_module is None:
        obj_name = getattr(obj, "__name__", str(obj))
        raise ValueError(f"Object {obj_name} has no module.")

    fp = entity_module.__file__
    if fp is None:
        obj_name = getattr(obj, "__name__", str(obj))
        raise ValueError(f"Object {obj_name} has no module.")

    file_path = pathlib.Path(fp)
    try:
        # Get the relative path to the current directory
        # Will raise ValueError if the file is not in the source directory
        relative_path = file_path.relative_to(str(pathlib.Path(source_dir).absolute()))

        if relative_path == pathlib.Path("_internal/resolvers"):
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
            obj_name = getattr(obj, "__name__", str(obj))
            raise ValueError(
                f"Object {obj_name} module file {file_path} is not relative to "
                f"source directory {source_dir} and is not an installed package."
            )

    if entity_module_name == "__main__":
        """
        This case is for the case in which the object is run from the main module.
        """
        fp = sys.modules["__main__"].__file__
        if fp is None:
            obj_name = getattr(obj, "__name__", str(obj))
            raise ValueError(f"Object {obj_name} has no module.")
        main_path = pathlib.Path(fp)
        entity_module_name = main_path.stem

    return entity_module_name, entity_module
