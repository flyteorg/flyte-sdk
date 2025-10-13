"""Test running mypy on task modules."""

import pathlib
import subprocess
import sys
from typing import List

import pytest

MODULES_DIR = pathlib.Path(__file__).parent / "modules"


def get_module_files() -> dict[str, List[pathlib.Path]]:
    """Get all Python files in the modules directory."""
    module_files = [f for f in MODULES_DIR.glob("*.py") if f.name != "__init__.py"]
    return {"argvalues": module_files, "ids": [f.name for f in module_files]}


@pytest.mark.parametrize("module_file", **get_module_files())
def test_mypy_on_module(module_file: pathlib.Path):
    """Test that mypy passes on each module file."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", str(module_file)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy failed for {module_file.name}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
