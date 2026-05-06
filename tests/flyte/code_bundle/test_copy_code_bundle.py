import tempfile
from pathlib import Path
from unittest.mock import patch

from flyte._code_bundle._utils import copy_code_bundle_to_context


def test_copy_code_bundle_all_default_patterns():
    """Test 'all' copy_style with default ignore patterns (ignores __pycache__, .git)."""
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / "project"
        context_path = Path(tmp) / "context"
        root_dir.mkdir()
        context_path.mkdir()

        # Create source files
        (root_dir / "main.py").write_text("print('hello')")
        (root_dir / "lib").mkdir()
        (root_dir / "lib" / "util.py").write_text("def util(): pass")

        # Create files that should be ignored by default patterns
        (root_dir / "__pycache__").mkdir()
        (root_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00")
        (root_dir / ".git").mkdir()
        (root_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")

        dst = copy_code_bundle_to_context(root_dir, "all", context_path)

        # Normal files should be copied
        assert (dst / "main.py").exists()
        assert (dst / "main.py").read_text() == "print('hello')"
        assert (dst / "lib" / "util.py").exists()

        # Ignored files should NOT be copied
        assert not (dst / "__pycache__").exists()
        assert not (dst / ".git").exists()


def test_copy_code_bundle_all_custom_patterns():
    """Test 'all' copy_style with custom ignore patterns."""
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / "project"
        context_path = Path(tmp) / "context"
        root_dir.mkdir()
        context_path.mkdir()

        # Create source files
        (root_dir / "main.py").write_text("print('hello')")
        (root_dir / "data.csv").write_text("a,b,c")
        (root_dir / "notes.txt").write_text("notes")

        dst = copy_code_bundle_to_context(root_dir, "all", context_path, ignore_patterns=["*.csv", "*.txt"])

        # .py files should be present
        assert (dst / "main.py").exists()

        # .csv and .txt should be excluded by custom patterns
        assert not (dst / "data.csv").exists()
        assert not (dst / "notes.txt").exists()


def test_copy_code_bundle_loaded_modules():
    """Test 'loaded_modules' copy_style copies only the returned module files."""
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / "project"
        context_path = Path(tmp) / "context"
        root_dir.mkdir()
        context_path.mkdir()

        # Create source files
        mod_file = root_dir / "my_module.py"
        mod_file.write_text("x = 1")
        (root_dir / "other.py").write_text("y = 2")

        with patch(
            "flyte._code_bundle._utils.list_imported_modules_as_files",
            return_value=[str(mod_file.resolve())],
        ):
            dst = copy_code_bundle_to_context(root_dir, "loaded_modules", context_path)

        # Only the mocked module file should be copied
        assert (dst / "my_module.py").exists()
        assert (dst / "my_module.py").read_text() == "x = 1"

        # other.py was not in the list, so it should not be copied
        assert not (dst / "other.py").exists()


def test_copy_code_bundle_absolute_path_uses_flyte_abs_context():
    """Test that absolute root_dir results in _flyte_abs_context path convention."""
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / "project"
        context_path = Path(tmp) / "context"
        root_dir.mkdir()
        context_path.mkdir()

        (root_dir / "app.py").write_text("print('app')")

        # root_dir is absolute (as it's in a temp dir)
        assert root_dir.is_absolute()

        dst = copy_code_bundle_to_context(root_dir, "all", context_path)

        # Should be under _flyte_abs_context
        assert "_flyte_abs_context" in str(dst)
        assert (dst / "app.py").exists()
        assert (dst / "app.py").read_text() == "print('app')"


def test_copy_code_bundle_relative_path():
    """Test that relative root_dir does NOT use _flyte_abs_context."""
    with tempfile.TemporaryDirectory() as tmp:
        root_dir = Path(tmp) / "project"
        context_path = Path(tmp) / "context"
        root_dir.mkdir()
        context_path.mkdir()

        (root_dir / "app.py").write_text("print('app')")

        # Use a relative path
        rel_root = Path("project")

        # We need to actually use a relative path that exists; chdir into tmp
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            dst = copy_code_bundle_to_context(rel_root, "all", context_path)

            # Should NOT be under _flyte_abs_context
            assert "_flyte_abs_context" not in str(dst)
            assert (dst / "app.py").exists()
        finally:
            os.chdir(old_cwd)
