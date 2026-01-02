import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import flyte.errors
from flyte._constants import FLYTE_SYS_PATH
from flyte._logging import logger


def load_python_modules(
    path: Path, root_dir: Path, recursive: bool = False
) -> Tuple[List[ModuleType], List[Tuple[Path, str]]]:
    """
    Load all Python modules from a path and return list of loaded module names.

    :param path: File or directory path
    :param root_dir: Root directory to search for modules
    :param recursive: If True, load modules recursively from subdirectories
    :return: List of loaded module names, and list of file paths that failed to load
    """
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    loaded_modules = []
    failed_paths = []

    if path.is_file() and path.suffix == ".py":
        rel_path = path.resolve().relative_to(root_dir)
        mod = (".".join(rel_path.parts))[:-3]
        imported_module = importlib.import_module(mod)
        loaded_modules.append(imported_module)

    elif path.is_dir():
        # Directory case - find all Python files
        pattern = "**/*.py" if recursive else "*.py"
        python_files = list(path.glob(pattern))

        # Filter out __init__.py files
        python_files = [f for f in python_files if f.name != "__init__.py"]

        if not python_files:
            # If no .py files found, try importing as a module
            try:
                rel_path = path.resolve().relative_to(root_dir)
                mod = ".".join(rel_path.parts)
                imported_module = importlib.import_module(mod)
                loaded_modules.append(imported_module)
            except (ValueError, ModuleNotFoundError):
                pass
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("• {task.fields[current_file]}"),
            ) as progress:
                task = progress.add_task(f"Loading {len(python_files)} files", total=len(python_files), current_file="")

                for file_path in python_files:
                    progress.update(task, advance=1, current_file=file_path.name)

                    try:
                        rel_path = file_path.resolve().relative_to(root_dir)
                        mod = (".".join(rel_path.parts))[:-3]
                        imported_module = importlib.import_module(mod)
                        loaded_modules.append(imported_module)
                    except flyte.errors.ModuleLoadError as e:
                        failed_paths.append((file_path, str(e)))

                progress.update(task, current_file="[green]Done[/green]")

    return loaded_modules, failed_paths


def _load_module_from_file(file_path: Path) -> str | None:
    """
    Load a Python module from a file path.

    :param file_path: Path to the Python file
    :return: Module name if successfully loaded, None otherwise
    """
    try:
        # Use the file stem as module name
        module_name = file_path.stem

        # Load the module specification
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        # Create and execute the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        module_path = os.path.dirname(os.path.abspath(file_path))
        sys.path.append(module_path)
        spec.loader.exec_module(module)

        return module_name

    except Exception as e:
        raise flyte.errors.ModuleLoadError(f"Failed to load module from {file_path}: {e}") from e


def adjust_sys_path(additional_paths: List[str] | None = None):
    """
    Adjust sys.path to include local sys.path entries under the root directory.
    """
    if "." not in sys.path or os.getcwd() not in sys.path:
        sys.path.insert(0, ".")
        logger.info(f"Added {os.getcwd()} to sys.path")
    for p in os.environ.get(FLYTE_SYS_PATH, "").split(":"):
        if p and p not in sys.path:
            sys.path.insert(0, p)
            logger.info(f"Added {p} to sys.path")
    if additional_paths:
        for p in additional_paths:
            if p and p not in sys.path:
                sys.path.insert(0, p)
                logger.info(f"Added {p} to sys.path")
