import subprocess
import tempfile
from pathlib import Path

from flyte._code_bundle._ignore import GitIgnore, IgnoreGroup, StandardIgnore


def test_ignore_group_list_ignored():
    """Test that IgnoreGroup.list_ignored() correctly identifies ignored files"""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

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
        root_path = Path(tmpdir).resolve()

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


def test_gitignore_with_absolute_paths():
    """Test that GitIgnore correctly handles absolute paths by converting to relative paths"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create .gitignore
        gitignore_content = """
.venv/
*.pyc
__pycache__/
node_modules/
"""
        (root_path / ".gitignore").write_text(gitignore_content)

        # Create test files and directories
        (root_path / "main.py").write_text("print('hello')")
        (root_path / "test.pyc").write_text("bytecode")

        # Create .venv directory with files
        (root_path / ".venv").mkdir()
        (root_path / ".venv" / "bin").mkdir()
        (root_path / ".venv" / "bin" / "python").write_text("#!/usr/bin/env python")
        (root_path / ".venv" / "lib").mkdir()
        (root_path / ".venv" / "lib" / "site-packages").mkdir()
        (root_path / ".venv" / "lib" / "site-packages" / "package.py").write_text("code")

        # Create __pycache__ directory
        (root_path / "__pycache__").mkdir()
        (root_path / "__pycache__" / "main.cpython-312.pyc").write_text("bytecode")

        # Add .gitignore to git
        subprocess.run(["git", "add", ".gitignore"], cwd=root_path, capture_output=True, check=True)

        git_ignore = GitIgnore(root_path)

        # Test that files (use absolute path, as done by list_all_files) are ignored
        venv_file = (root_path / ".venv" / "bin" / "python").absolute()
        venv_nested_file = (root_path / ".venv" / "lib" / "site-packages" / "package.py").absolute()
        pyc_file = (root_path / "test.pyc").absolute()
        pycache_file = (root_path / "__pycache__" / "main.cpython-312.pyc").absolute()
        assert git_ignore.is_ignored(venv_file), ".venv/bin/python should be ignored"
        assert git_ignore.is_ignored(venv_nested_file), ".venv/lib/site-packages/package.py should be ignored"
        assert git_ignore.is_ignored(pyc_file), "test.pyc should be ignored"
        assert git_ignore.is_ignored(pycache_file), "__pycache__/main.cpython-312.pyc should be ignored"

        # main.py should not be ignored
        main_file = (root_path / "main.py").absolute()
        assert not git_ignore.is_ignored(main_file), "main.py should NOT be ignored"


def test_gitignore_with_flyteignore():
    """Test that GitIgnore respects both .gitignore and .flyteignore files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create .gitignore with some patterns
        gitignore_content = """
*.pyc
__pycache__/
.venv/
"""
        (root_path / ".gitignore").write_text(gitignore_content)

        # Create .flyteignore with different patterns
        flyteignore_content = """
*.log
temp/
secrets.json
"""
        (root_path / ".flyteignore").write_text(flyteignore_content)

        # Create test files
        (root_path / "main.py").write_text("print('hello')")
        (root_path / "test.pyc").write_text("bytecode")  # Should be ignored by .gitignore
        (root_path / "debug.log").write_text("logs")  # Should be ignored by .flyteignore
        (root_path / "secrets.json").write_text("{}")  # Should be ignored by .flyteignore
        (root_path / "config.json").write_text("{}")  # Should NOT be ignored

        # Create temp directory with files (ignored by .flyteignore)
        (root_path / "temp").mkdir()
        (root_path / "temp" / "data.txt").write_text("temp data")

        # Create .venv directory (ignored by .gitignore)
        (root_path / ".venv").mkdir()
        (root_path / ".venv" / "python").write_text("#!/usr/bin/env python")

        # Add ignore files to git
        subprocess.run(["git", "add", ".gitignore", ".flyteignore"], cwd=root_path, capture_output=True, check=True)

        git_ignore = GitIgnore(root_path)

        # Test files ignored by .gitignore
        assert git_ignore.is_ignored(root_path / "test.pyc"), "test.pyc should be ignored (from .gitignore)"
        assert git_ignore.is_ignored(root_path / ".venv" / "python"), ".venv/python should be ignored (from .gitignore)"

        # Test files ignored by .flyteignore
        assert git_ignore.is_ignored(root_path / "debug.log"), "debug.log should be ignored (from .flyteignore)"
        assert git_ignore.is_ignored(root_path / "secrets.json"), "secrets.json should be ignored (from .flyteignore)"
        assert git_ignore.is_ignored(root_path / "temp" / "data.txt"), (
            "temp/data.txt should be ignored (from .flyteignore)"
        )

        # Test files that should NOT be ignored
        assert not git_ignore.is_ignored(root_path / "main.py"), "main.py should NOT be ignored"
        assert not git_ignore.is_ignored(root_path / "config.json"), "config.json should NOT be ignored"


def test_gitignore_flyteignore_directory_patterns():
    """Test that directory patterns work correctly in both .gitignore and .flyteignore"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create .gitignore with directory pattern
        gitignore_content = """
build/
"""
        (root_path / ".gitignore").write_text(gitignore_content)

        # Create .flyteignore with directory pattern
        flyteignore_content = """
artifacts/
"""
        (root_path / ".flyteignore").write_text(flyteignore_content)

        # Create directories and files
        (root_path / "build").mkdir()
        (root_path / "build" / "output.txt").write_text("build output")
        (root_path / "artifacts").mkdir()
        (root_path / "artifacts" / "result.txt").write_text("artifact")
        (root_path / "src").mkdir()
        (root_path / "src" / "main.py").write_text("code")

        # Add ignore files to git
        subprocess.run(["git", "add", ".gitignore", ".flyteignore"], cwd=root_path, capture_output=True, check=True)

        git_ignore = GitIgnore(root_path)

        # Test directory patterns from .gitignore
        assert git_ignore.is_ignored(root_path / "build" / "output.txt"), (
            "build/output.txt should be ignored (from .gitignore)"
        )

        # Test directory patterns from .flyteignore
        assert git_ignore.is_ignored(root_path / "artifacts" / "result.txt"), (
            "artifacts/result.txt should be ignored (from .flyteignore)"
        )

        # Test that non-ignored directories still work
        assert not git_ignore.is_ignored(root_path / "src" / "main.py"), "src/main.py should NOT be ignored"


def test_gitignore_flyteignore_overlapping_patterns():
    """Test behavior when both .gitignore and .flyteignore have overlapping patterns"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create .gitignore and .flyteignore with overlapping patterns
        gitignore_content = """
*.log
"""
        (root_path / ".gitignore").write_text(gitignore_content)

        flyteignore_content = """
!*.log
debug/
"""
        (root_path / ".flyteignore").write_text(flyteignore_content)

        # Create test files
        (root_path / "app.log").write_text("logs")
        (root_path / "debug").mkdir()
        (root_path / "debug" / "trace.log").write_text("debug logs")

        # Add ignore files to git
        subprocess.run(["git", "add", ".gitignore", ".flyteignore"], cwd=root_path, capture_output=True, check=True)

        git_ignore = GitIgnore(root_path)

        # Files should be ignored regardless of which file defines the pattern
        assert not git_ignore.is_ignored(root_path / "app.log"), "app.log should not be ignored"
        assert git_ignore.is_ignored(root_path / "debug" / "trace.log"), "debug/trace.log should be ignored"


def test_gitignore_subdirectory_ignore_files():
    """Test that .gitignore and .flyteignore files in subdirectories are respected"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create root .gitignore
        gitignore_content = """
*.pyc
"""
        (root_path / ".gitignore").write_text(gitignore_content)

        # Create root .flyteignore
        flyteignore_content = """
*.log
"""
        (root_path / ".flyteignore").write_text(flyteignore_content)

        # Create subdirectory structure
        (root_path / "src").mkdir()
        (root_path / "src" / "app.py").write_text("code")
        (root_path / "src" / "test.pyc").write_text("bytecode")
        (root_path / "src" / "debug.log").write_text("logs")

        # Create .gitignore in subdirectory
        src_gitignore = """
*.tmp
"""
        (root_path / "src" / ".gitignore").write_text(src_gitignore)

        # Create .flyteignore in subdirectory
        src_flyteignore = """
*.cache
"""
        (root_path / "src" / ".flyteignore").write_text(src_flyteignore)

        # Create files that should be ignored by subdirectory ignore files
        (root_path / "src" / "temp.tmp").write_text("temp")
        (root_path / "src" / "data.cache").write_text("cache")

        # Create nested subdirectory
        (root_path / "src" / "module").mkdir()
        (root_path / "src" / "module" / "file.py").write_text("code")
        (root_path / "src" / "module" / "file.tmp").write_text("temp")
        (root_path / "src" / "module" / "file.cache").write_text("cache")
        (root_path / "src" / "module" / "file.pyc").write_text("bytecode")
        (root_path / "src" / "module" / "file.log").write_text("log")

        # Add all ignore files to git
        subprocess.run(
            ["git", "add", ".gitignore", ".flyteignore", "src/.gitignore", "src/.flyteignore"],
            cwd=root_path,
            capture_output=True,
            check=True,
        )

        git_ignore = GitIgnore(root_path)

        # Test files ignored by root .gitignore
        assert git_ignore.is_ignored(root_path / "src" / "test.pyc"), "src/test.pyc should be ignored (root .gitignore)"
        assert git_ignore.is_ignored(root_path / "src" / "module" / "file.pyc"), (
            "src/module/file.pyc should be ignored (root .gitignore)"
        )

        # Test files ignored by root .flyteignore
        assert git_ignore.is_ignored(root_path / "src" / "debug.log"), (
            "src/debug.log should be ignored (root .flyteignore)"
        )
        assert git_ignore.is_ignored(root_path / "src" / "module" / "file.log"), (
            "src/module/file.log should be ignored (root .flyteignore)"
        )

        # Test files ignored by subdirectory .gitignore
        assert git_ignore.is_ignored(root_path / "src" / "temp.tmp"), "src/temp.tmp should be ignored (src/.gitignore)"
        assert git_ignore.is_ignored(root_path / "src" / "module" / "file.tmp"), (
            "src/module/file.tmp should be ignored (src/.gitignore)"
        )

        # Test files ignored by subdirectory .flyteignore
        assert git_ignore.is_ignored(root_path / "src" / "data.cache"), (
            "src/data.cache should be ignored (src/.flyteignore)"
        )
        assert git_ignore.is_ignored(root_path / "src" / "module" / "file.cache"), (
            "src/module/file.cache should be ignored (src/.flyteignore)"
        )

        # Test files that should NOT be ignored
        assert not git_ignore.is_ignored(root_path / "src" / "app.py"), "src/app.py should NOT be ignored"
        assert not git_ignore.is_ignored(root_path / "src" / "module" / "file.py"), (
            "src/module/file.py should NOT be ignored"
        )


def test_gitignore_with_git_root_and_working_dir():
    """Test that ignore files in git repo root are respected when working from a subdirectory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=repo_root, capture_output=True, check=True)

        # Create .gitignore in git repo root
        repo_gitignore = """
*.secret
"""
        (repo_root / ".gitignore").write_text(repo_gitignore)

        # Create .flyteignore in git repo root
        repo_flyteignore = """
*.private
"""
        (repo_root / ".flyteignore").write_text(repo_flyteignore)

        # Create a subdirectory structure (this will be our working directory)
        examples_dir = repo_root / "examples" / "basics"
        examples_dir.mkdir(parents=True)

        # Create files in the examples directory
        (examples_dir / "main.py").write_text("code")
        (examples_dir / "config.secret").write_text("secret data")  # Should be ignored by repo root .gitignore
        (examples_dir / "data.private").write_text("private data")  # Should be ignored by repo root .flyteignore
        (examples_dir / "public.txt").write_text("public data")  # Should NOT be ignored

        # Create .gitignore in examples directory
        examples_gitignore = """
*.temp
"""
        (examples_dir / ".gitignore").write_text(examples_gitignore)

        # Create files that match the examples .gitignore
        (examples_dir / "work.temp").write_text("temp work")

        # Add ignore files to git
        subprocess.run(
            ["git", "add", ".gitignore", ".flyteignore", "examples/basics/.gitignore"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )

        # Initialize GitIgnore with examples_dir as root (not repo root)
        git_ignore = GitIgnore(examples_dir)

        # Verify git root was detected correctly
        assert git_ignore.git_root == repo_root, f"Git root should be {repo_root}, got {git_ignore.git_root}"

        # Test files ignored by repo root .gitignore
        assert git_ignore.is_ignored(examples_dir / "config.secret"), (
            "config.secret should be ignored (from repo root .gitignore)"
        )

        # Test files ignored by repo root .flyteignore
        assert git_ignore.is_ignored(examples_dir / "data.private"), (
            "data.private should be ignored (from repo root .flyteignore)"
        )

        # Test files ignored by local .gitignore
        assert git_ignore.is_ignored(examples_dir / "work.temp"), "work.temp should be ignored (from local .gitignore)"

        # Test files that should NOT be ignored
        assert not git_ignore.is_ignored(examples_dir / "main.py"), "main.py should NOT be ignored"
        assert not git_ignore.is_ignored(examples_dir / "public.txt"), "public.txt should NOT be ignored"
