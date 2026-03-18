import inspect
import os
import pathlib
import sys
from types import ModuleType
from typing import Tuple


def _find_package_root(file_path: pathlib.Path, source_dir: pathlib.Path) -> pathlib.Path:
    """Find the best Python package root for a file by checking sys.path.

    In src-layout projects (e.g., my_app/src/my_app/main.py), the importable
    package root is my_app/src/, not source_dir. This function looks for
    the deepest sys.path entry between source_dir and the file that still
    produces a valid module path (i.e., not just the bare filename).

    Example:
        source_dir = /repo
        file_path  = /repo/my_app/src/my_app/main.py
        sys.path   = [..., /repo/my_app/src, ...]
        returns      /repo/my_app/src  (so module = my_app.main)
    """
    file_parent = file_path.parent.resolve()
    best = source_dir
    for p in sys.path:
        if "site-packages" in p or "dist-packages" in p:
            continue
        candidate = pathlib.Path(p).resolve()
        # candidate must be: between source_dir and file_path, but not the
        # file's own directory (that would give just the filename, e.g. "main")
        if (
            candidate != source_dir
            and candidate != file_parent
            and candidate.is_relative_to(source_dir)
            and file_path.is_relative_to(candidate)
            and len(candidate.parts) > len(best.parts)
        ):
            best = candidate
    return best


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
        source_dir_abs = pathlib.Path(source_dir).absolute()
        relative_path = file_path.relative_to(str(source_dir_abs))

        if relative_path == pathlib.Path("_internal/resolvers"):
            entity_module_name = entity_module.__name__
        elif "site-packages" in str(file_path) or "dist-packages" in str(file_path):
            raise ValueError("Object from a library")
        else:
            package_root = _find_package_root(file_path, source_dir_abs)
            relative_path = file_path.relative_to(str(package_root))
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
