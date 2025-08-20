import inspect
import os
import site
import sys
from pathlib import Path


INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR = "INTERNAL_APP_ENDPOINT_PATTERN"


def _extract_files_loaded_from_cwd(cwd: Path) -> list[str]:
    """Look for all files loaded in sys.modules that is also in cwd."""
    cwd = os.fspath(cwd.absolute())
    loaded_modules = list(sys.modules)

    # Do not include site packages and anything in sys.prefix
    invalid_dirs = [site.getusersitepackages(), *site.getsitepackages(), sys.prefix]

    files_loaded_from_cwd = []
    for module_name in loaded_modules:
        try:
            module_file_path = inspect.getfile(sys.modules[module_name])
        except Exception:
            continue

        absolute_file_path = os.path.abspath(module_file_path)
        if not os.path.commonpath([absolute_file_path, cwd]) == cwd:
            continue

        is_invalid = any(
            os.path.commonpath([absolute_file_path, invalid_dir]) == invalid_dir for invalid_dir in invalid_dirs
        )
        if is_invalid:
            continue

        files_loaded_from_cwd.append(absolute_file_path)

    return files_loaded_from_cwd
