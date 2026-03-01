import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from flyte._utils.module_loader import load_python_modules


@pytest.fixture
def temp_project(tmp_path):
    """Create a temp project with a .venv directory and regular Python files.

    Structure mimics uv_workspace/albatross:
        project/
            packages/
                bird_feeder/
                    workflow.py
                    .venv/
                        installed.py
                        lib/python3.12/site-packages/somepkg/module.py
                other_pkg/
                    task.py
    """
    root = tmp_path / "project"
    root.mkdir()

    packages = root / "packages"
    packages.mkdir()

    # Regular Python files in subpackages
    bird_feeder = packages / "bird_feeder"
    bird_feeder.mkdir()
    (bird_feeder / "workflow.py").write_text("x = 1\n")

    other_pkg = packages / "other_pkg"
    other_pkg.mkdir()
    (other_pkg / "task.py").write_text("y = 2\n")

    # .venv directory inside a subpackage (should be ignored)
    venv = bird_feeder / ".venv"
    venv.mkdir()
    (venv / "installed.py").write_text("z = 3\n")
    venv_lib = venv / "lib" / "python3.12" / "site-packages" / "somepkg"
    venv_lib.mkdir(parents=True)
    (venv_lib / "module.py").write_text("w = 4\n")

    # __pycache__ directory (should also be ignored)
    pycache = bird_feeder / "__pycache__"
    pycache.mkdir()
    (pycache / "workflow.cpython-312.pyc").write_text("")

    return root


def _fake_import(mod_name):
    """Mock importlib.import_module that records the module name."""

    class FakeModule:
        __name__ = mod_name

    return FakeModule()


def test_load_python_modules_skips_venv(temp_project):
    """load_python_modules with recursive=True should skip .venv directories."""
    packages = temp_project / "packages"

    imported_modules = []

    def fake_import(mod_name):
        m = _fake_import(mod_name)
        imported_modules.append(mod_name)
        return m

    sys.path.insert(0, str(temp_project))
    try:
        with patch("importlib.import_module", side_effect=fake_import):
            modules, _ = load_python_modules(packages, root_dir=temp_project, recursive=True)

        assert len(modules) == 2
        module_names = set(imported_modules)
        assert any("workflow" in m for m in module_names)
        assert any("task" in m for m in module_names)

        # No .venv files should be loaded
        for m in module_names:
            assert ".venv" not in m
            assert "installed" not in m
            assert "somepkg" not in m
    finally:
        sys.path.remove(str(temp_project))


def test_load_python_modules_skips_venv_with_relative_paths(temp_project):
    """Same test but with relative paths, matching real CLI behavior (path=Path('packages'), root_dir=Path.cwd())."""
    imported_modules = []

    def fake_import(mod_name):
        m = _fake_import(mod_name)
        imported_modules.append(mod_name)
        return m

    old_cwd = os.getcwd()
    os.chdir(temp_project)
    sys.path.insert(0, str(temp_project))
    try:
        with patch("importlib.import_module", side_effect=fake_import):
            modules, _ = load_python_modules(Path("packages"), root_dir=Path.cwd(), recursive=True)

        assert len(modules) == 2
        module_names = set(imported_modules)
        assert any("workflow" in m for m in module_names)
        assert any("task" in m for m in module_names)

        for m in module_names:
            assert ".venv" not in m
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(temp_project))
