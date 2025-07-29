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
        with pytest.raises(ValueError, match="is not in the subpath of"):
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
