import inspect
import os
import site
import sys
from contextlib import contextmanager
from pathlib import Path

from flytekit.image_spec import default_builder

from union.ucimage import _image_builder

INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR = "INTERNAL_APP_ENDPOINT_PATTERN"


def _return_empty_str():
    return ""


@contextmanager
def patch_get_flytekit_for_pypi():
    """Mock out get_flytekit_for_pypi to return an empty string."""
    try:
        orig_get_flytekit_for_pypi = default_builder.get_flytekit_for_pypi
        default_builder.get_flytekit_for_pypi = _return_empty_str
        yield
        default_builder.get_flytekit_for_pypi = orig_get_flytekit_for_pypi
    except AttributeError:
        # Catch this error just in case `flytekit` changes the location of
        # get_flytekit_for_pypi
        yield


@contextmanager
def patch_get_unionai_for_pypi():
    """Mock out get_unionai_for_pypi to return an empty string."""
    orig_get_unionai_for_pypi = _image_builder.get_unionai_for_pypi
    _image_builder.get_unionai_for_pypi = _return_empty_str
    yield
    _image_builder.get_unionai_for_pypi = orig_get_unionai_for_pypi


@contextmanager
def patch_get_flytekit_and_union_for_pypi():
    """Patches both unionai and flyekit"""
    with patch_get_flytekit_for_pypi(), patch_get_unionai_for_pypi():
        yield


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
