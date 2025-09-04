import pathlib
import tempfile
from unittest.mock import Mock

from flyte._code_bundle._utils import list_all_files


def test_list_all_files():
    """Test list_all_files function with a simple directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        test_dir = pathlib.Path(tmpdir)

        # Create subdirectories
        src_dir = test_dir / "src"
        src_dir.mkdir()

        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Create test files
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "README.md").write_text("# Test Project")
        (src_dir / "app.py").write_text("import os")
        (utils_dir / "helper.py").write_text("def helper(): pass")

        # Test without ignore_group
        files = list_all_files(test_dir, deref_symlinks=False)

        # Verify all files are found
        assert len(files) == 4

        # Convert to relative paths for easier comparison
        relative_files = [str(pathlib.Path(f).relative_to(test_dir)) for f in files]
        relative_files.sort()

        expected_files = ["README.md", "main.py", "src/app.py", "src/utils/helper.py"]

        assert relative_files == expected_files

        # Test with ignore_group (mock)
        mock_ignore_group = Mock()
        mock_ignore_group.is_ignored.return_value = False

        files_with_ignore = list_all_files(test_dir, deref_symlinks=False, ignore_group=mock_ignore_group)
        assert len(files_with_ignore) == 4
