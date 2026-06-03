import os
import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest

import flyte.errors
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


def test_load_python_modules_file_outside_root_raises_click_exception(tmp_path):
    """When a single .py file is loaded but lives outside the configured root_dir, surface a
    click.ClickException with an actionable message instead of letting pathlib's
    ``ValueError: '...' is not in the subpath of '...'`` bubble up to Sentry as an SDK crash.

    Reproduces FLYTE-SDK-2T: user ran `flyte deploy` on a file under
    ``examples/basics/multi_status.py`` while the SDK had detected the root as
    ``flyte-sdk/src``, so ``Path.resolve().relative_to(root)`` blew up.
    """
    root_dir = tmp_path / "src"
    root_dir.mkdir()
    outside_dir = tmp_path / "examples"
    outside_dir.mkdir()
    outside_file = outside_dir / "workflow.py"
    outside_file.write_text("x = 1\n")

    with pytest.raises(click.ClickException) as excinfo:
        load_python_modules(outside_file, root_dir=root_dir, recursive=False)

    msg = excinfo.value.message
    assert "not inside the project root" in msg
    assert "--root-dir" in msg
    assert str(outside_file) in msg
    assert str(root_dir) in msg


def test_load_python_modules_single_file_wraps_module_not_found(tmp_path):
    """When a single .py file imports something not installed, surface a
    ModuleLoadError (a RuntimeUserError that the Sentry filter skips) instead
    of letting bare ModuleNotFoundError reach Sentry.

    Reproduces FLYTE-SDK-3T/3K/3R/3Q/3P/3M/3N/3J/3H/3E.
    """
    root = tmp_path / "project"
    root.mkdir()
    f = root / "workflow.py"
    f.write_text("x = 1\n")

    def boom(_mod):
        raise ModuleNotFoundError("No module named 'fold_to_ascii'")

    with patch("importlib.import_module", side_effect=boom):
        with pytest.raises(flyte.errors.ModuleLoadError) as excinfo:
            load_python_modules(f, root_dir=root, recursive=False)

    msg = str(excinfo.value)
    assert "workflow.py" in msg
    assert "ModuleNotFoundError" in msg
    assert "fold_to_ascii" in msg


def test_load_python_modules_dir_collects_import_errors_in_failed_paths(tmp_path):
    """When loading a directory of .py files, ImportError/ModuleNotFoundError on
    one file should be recorded in failed_paths so the deploy command can
    surface a clean error -- not propagate as an unhandled crash to Sentry."""
    root = tmp_path / "project"
    root.mkdir()
    ok = root / "ok.py"
    ok.write_text("x = 1\n")
    bad = root / "bad.py"
    bad.write_text("import nonexistent_pkg\n")

    def fake_import(mod_name):
        if mod_name == "bad":
            raise ModuleNotFoundError("No module named 'nonexistent_pkg'")

        class FakeMod:
            __name__ = mod_name

        return FakeMod()

    with patch("importlib.import_module", side_effect=fake_import):
        modules, failed = load_python_modules(root, root_dir=root, recursive=False)

    assert len(modules) == 1
    assert len(failed) == 1
    failed_path, failed_msg = failed[0]
    assert failed_path == bad
    assert "ModuleNotFoundError" in failed_msg
    assert "nonexistent_pkg" in failed_msg


@pytest.mark.parametrize(
    "exc_type,exc_kwargs",
    [
        (SyntaxError, {"msg": "bad syntax"}),
        (NameError, {"name": "x"}),
        (AttributeError, {}),
        (TypeError, {}),
        (ValueError, {}),
        (KeyError, {}),
    ],
)
def test_load_python_modules_single_file_wraps_user_code_errors(tmp_path, exc_type, exc_kwargs):
    """SyntaxError / NameError / etc. from user module top-level execution are
    also user-code bugs and should be surfaced via ModuleLoadError, not raw."""
    root = tmp_path / "project"
    root.mkdir()
    f = root / "workflow.py"
    f.write_text("x = 1\n")

    def boom(_mod):
        if exc_type is SyntaxError:
            raise SyntaxError("bad syntax")
        if exc_type is NameError:
            raise NameError("name 'gcp_adr' is not defined")
        raise exc_type("boom")

    with patch("importlib.import_module", side_effect=boom):
        with pytest.raises(flyte.errors.ModuleLoadError) as excinfo:
            load_python_modules(f, root_dir=root, recursive=False)

    assert exc_type.__name__ in str(excinfo.value)


def test_load_python_modules_single_file_wraps_missing_env_var_keyerror(tmp_path):
    """When a user module does ``os.environ["GCP_ADR"]`` at top level and the env
    var is missing, the resulting ``KeyError`` is a user config issue, not an
    SDK crash. It must surface as ``ModuleLoadError`` so the Sentry filter
    skips it (consistent with NameError/TypeError already wrapped).

    Reproduces FLYTE-SDK-3D.
    """
    root = tmp_path / "project"
    root.mkdir()
    f = root / "workflow.py"
    f.write_text('import os\nGCP_ADR = os.environ["NEVER_SET_GCP_ADR_XYZ"]\n')

    # Make sure the env var truly is missing.
    os.environ.pop("NEVER_SET_GCP_ADR_XYZ", None)

    sys.path.insert(0, str(root))
    try:
        with pytest.raises(flyte.errors.ModuleLoadError) as excinfo:
            load_python_modules(f, root_dir=root, recursive=False)
    finally:
        sys.path.remove(str(root))
        sys.modules.pop("workflow", None)

    msg = str(excinfo.value)
    assert "workflow.py" in msg
    assert "KeyError" in msg
    assert "NEVER_SET_GCP_ADR_XYZ" in msg
