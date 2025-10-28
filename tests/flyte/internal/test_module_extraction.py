import importlib
import importlib.util
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from flyte._internal.resolvers._task_module import extract_task_module
from flyte._task import AsyncFunctionTaskTemplate


@pytest.fixture
def mock_task():
    task = MagicMock(spec=AsyncFunctionTaskTemplate)
    task.name = "sample_task"
    return task


def test_extract_task_module_success(mock_task, tmp_path):
    mock_func = MagicMock()
    mock_func.__name__ = "sample_func"
    mock_task.func = mock_func

    mock_module = MagicMock()
    mock_module.__name__ = "sample_module"
    mock_module.__file__ = str(tmp_path / "sample_module.py")

    os.makedirs(tmp_path / "subdir")
    path_to_file = tmp_path / "subdir" / "sample_module.py"
    path_to_file.touch()

    spec = importlib.util.spec_from_file_location("sample_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch("inspect.getmodule", return_value=module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "sample_func"
        assert module_name == "subdir.sample_module"


def test_extract_task_module_outside_source_dir(mock_task, tmp_path):
    mock_func = MagicMock()
    mock_func.__name__ = "sample_func"
    mock_task.func = mock_func

    mock_module = MagicMock()
    mock_module.__name__ = "sample_module"
    mock_module.__file__ = str(tmp_path / "sample_module.py")

    os.makedirs(tmp_path / "subdir")
    path_to_file = tmp_path / "subdir" / "sample_module.py"
    path_to_file.touch()

    spec = importlib.util.spec_from_file_location("sample_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch("inspect.getmodule", return_value=module):
        with pytest.raises(ValueError, match="is not relative to"):
            extract_task_module(mock_task, tmp_path / "other_dir")


def test_extract_task_module_no_module(mock_task):
    mock_func = MagicMock()
    mock_func.__name__ = "sample_func"
    mock_task.func = mock_func

    with patch("inspect.getmodule", return_value=None):
        with pytest.raises(ValueError, match="has no module"):
            extract_task_module(mock_task, pathlib.Path("."))


def test_extract_task_module_main_module(mock_task, tmp_path):
    mock_func = MagicMock()
    mock_func.__name__ = "sample_func"
    mock_task.func = mock_func

    main_module = MagicMock()
    main_module.__name__ = "__main__"
    main_module.__file__ = str(tmp_path / "main_script.py")

    with patch("inspect.getmodule", return_value=main_module), patch.dict(sys.modules, {"__main__": main_module}):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)
        assert entity_name == "sample_func"
        assert module_name == "main_script"


def test_extract_task_module_not_implemented():
    task = MagicMock()
    task.name = "non_async_task"
    with pytest.raises(NotImplementedError, match="not implemented"):
        extract_task_module(task, pathlib.Path("."))


def test_extract_task_module_installed_package_site_packages(mock_task, tmp_path):
    """Test extraction from an installed package in site-packages."""
    mock_func = MagicMock()
    mock_func.__name__ = "installed_func"
    mock_task.func = mock_func

    # Create a mock module that appears to be from site-packages
    mock_module = MagicMock()
    mock_module.__name__ = "my_package.submodule"
    mock_module.__file__ = "/usr/lib/python3.9/site-packages/my_package/submodule.py"

    with patch("inspect.getmodule", return_value=mock_module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "installed_func"
        assert module_name == "my_package.submodule"


def test_extract_task_module_installed_package_dist_packages(mock_task, tmp_path):
    """Test extraction from an installed package in dist-packages."""
    mock_func = MagicMock()
    mock_func.__name__ = "dist_func"
    mock_task.func = mock_func

    # Create a mock module that appears to be from dist-packages
    mock_module = MagicMock()
    mock_module.__name__ = "another_package.module"
    mock_module.__file__ = "/usr/local/lib/python3.9/dist-packages/another_package/module.py"

    with patch("inspect.getmodule", return_value=mock_module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "dist_func"
        assert module_name == "another_package.module"


def test_extract_task_module_installed_package_nested_module(mock_task, tmp_path):
    """Test extraction from a deeply nested installed package."""
    mock_func = MagicMock()
    mock_func.__name__ = "nested_func"
    mock_task.func = mock_func

    # Create a mock module with nested structure
    mock_module = MagicMock()
    mock_module.__name__ = "deep.package.sub.module"
    mock_module.__file__ = "/usr/lib/python3.9/site-packages/deep/package/sub/module.py"

    with patch("inspect.getmodule", return_value=mock_module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "nested_func"
        assert module_name == "deep.package.sub.module"


def test_extract_task_module_not_in_source_dir_not_installed(mock_task, tmp_path):
    """Test that non-installed modules outside source_dir raise ValueError."""
    mock_func = MagicMock()
    mock_func.__name__ = "external_func"
    mock_task.func = mock_func

    # Create a mock module that's neither in source_dir nor in site-packages
    mock_module = MagicMock()
    mock_module.__name__ = "external_module"
    mock_module.__file__ = "/some/other/path/external_module.py"

    with patch("inspect.getmodule", return_value=mock_module):
        with pytest.raises(ValueError, match="is not relative to"):
            extract_task_module(mock_task, tmp_path)


def test_extract_task_module_no_file_path(mock_task):
    """Test that modules without __file__ raise ValueError."""
    mock_func = MagicMock()
    mock_func.__name__ = "no_file_func"
    mock_task.func = mock_func

    mock_module = MagicMock()
    mock_module.__name__ = "no_file_module"
    mock_module.__file__ = None

    with patch("inspect.getmodule", return_value=mock_module):
        with pytest.raises(ValueError, match="has no module"):
            extract_task_module(mock_task, pathlib.Path("."))


def test_extract_task_module_current_directory(mock_task, tmp_path):
    """Test extraction when module is in the current directory (root of source_dir)."""
    mock_func = MagicMock()
    mock_func.__name__ = "root_func"
    mock_task.func = mock_func

    # Create a module file directly in source_dir
    path_to_file = tmp_path / "root_module.py"
    path_to_file.touch()

    spec = importlib.util.spec_from_file_location("root_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch("inspect.getmodule", return_value=module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "root_func"
        assert module_name == "root_module"


def test_extract_task_module_windows_path_installed(mock_task, tmp_path):
    """Test extraction from installed package with Windows-style paths."""
    mock_func = MagicMock()
    mock_func.__name__ = "windows_func"
    mock_task.func = mock_func

    # Create a mock module with Windows path
    mock_module = MagicMock()
    mock_module.__name__ = "win_package.module"
    mock_module.__file__ = "C:\\Python39\\Lib\\site-packages\\win_package\\module.py"

    with patch("inspect.getmodule", return_value=mock_module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "windows_func"
        assert module_name == "win_package.module"


def test_extract_task_module_deeply_nested_source_dir(mock_task, tmp_path):
    """Test extraction from deeply nested directories within source_dir."""
    mock_func = MagicMock()
    mock_func.__name__ = "deep_func"
    mock_task.func = mock_func

    # Create nested directory structure
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    os.makedirs(nested_dir)
    path_to_file = nested_dir / "deep_module.py"
    path_to_file.touch()

    spec = importlib.util.spec_from_file_location("deep_module", path_to_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with patch("inspect.getmodule", return_value=module):
        entity_name, module_name = extract_task_module(mock_task, tmp_path)

        assert entity_name == "deep_func"
        assert module_name == "level1.level2.level3.deep_module"


def test_extract_task_module_site_packages_case_sensitivity(mock_task, tmp_path):
    """Test that 'site-packages' detection is case-sensitive (lowercase)."""
    mock_func = MagicMock()
    mock_func.__name__ = "case_func"
    mock_task.func = mock_func

    # Test with uppercase - should NOT be treated as installed
    mock_module = MagicMock()
    mock_module.__name__ = "case_module"
    mock_module.__file__ = "/usr/lib/python3.9/SITE-PACKAGES/case_module.py"

    with patch("inspect.getmodule", return_value=mock_module):
        with pytest.raises(ValueError, match="is not relative to"):
            extract_task_module(mock_task, tmp_path)
