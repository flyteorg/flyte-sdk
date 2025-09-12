import tempfile
from pathlib import Path

from flyte._code_bundle._ignore import IgnoreGroup, StandardIgnore


def test_ignore_group_list_ignored():
    """Test that IgnoreGroup.list_ignored() correctly identifies ignored files"""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)

        # Create test files and directories
        (root_path / "main.py").write_text("print('hello')")
        (root_path / "test.py").write_text("print('test')")
        (root_path / "config.json").write_text("{}")
        (root_path / "temp.log").write_text("log content")
        (root_path / "cache").mkdir()
        (root_path / "cache" / "temp.cache").write_text("cache data")
        (root_path / "src").mkdir()
        (root_path / "src" / "module.py").write_text("def func(): pass")
        (root_path / "src" / "temp.tmp").write_text("temp data")

        # Create IgnoreGroup with StandardIgnore that ignores .tmp and .cache files
        ignore_patterns = ["*.tmp", "*.cache", "temp.log"]
        ignore_group = IgnoreGroup(root_path, StandardIgnore)

        # Override the patterns for this test
        ignore_group.ignores[0].patterns = ignore_patterns

        # Debug: Let's check what files are actually being processed
        print(f"Root path: {root_path}")
        print(f"Ignore patterns: {ignore_patterns}")

        # Call list_ignored method
        ignored_files = ignore_group.list_ignored()
        print(f"Ignored files: {ignored_files}")

        # Let's also test individual file checks
        for file_path in root_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(root_path)
                is_ignored = ignore_group.is_ignored(file_path)
                print(f"File: {rel_path}, Ignored: {is_ignored}")

        # Verify the results - should be exactly 3 ignored files
        assert len(ignored_files) == 3, f"Expected 3 ignored files, got {len(ignored_files)}: {ignored_files}"

        # Convert to set for easier comparison
        ignored_set = set(ignored_files)

        # Check that the expected files are ignored
        assert "temp.log" in ignored_set, "temp.log should be ignored"
        assert "cache/temp.cache" in ignored_set, "cache/temp.cache should be ignored"
        assert "src/temp.tmp" in ignored_set, "src/temp.tmp should be ignored"

        # Check that non-ignored files are not in the list
        assert "main.py" not in ignored_set, "main.py should not be ignored"
        assert "test.py" not in ignored_set, "test.py should not be ignored"
        assert "config.json" not in ignored_set, "config.json should not be ignored"
        assert "src/module.py" not in ignored_set, "src/module.py should not be ignored"


def test_standard_ignore_valueerror_handling():
    """Test that StandardIgnore handles ValueError when path is not under root"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        
        # Create a file outside the root directory
        outside_file = Path(tmpdir).parent / "outside_file.txt"
        outside_file.write_text("outside content")
        
        # Create StandardIgnore instance
        ignore_patterns = ["*.txt"]
        standard_ignore = StandardIgnore(root_path, ignore_patterns)
        
        # Test that ValueError is handled gracefully
        # The path outside the root should not be ignored (since it's not under root)
        is_ignored = standard_ignore.is_ignored(outside_file)
        assert not is_ignored, "File outside root should not be ignored"
        
        # Clean up
        outside_file.unlink()
