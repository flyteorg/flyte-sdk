import subprocess
import tempfile
from pathlib import Path

from flyte._code_bundle._ignore import FlyteIgnore, GitIgnore, IgnoreGroup, StandardIgnore


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


def test_find_ignore_files_skips_standard_ignored_dirs():
    """Test that _find_ignore_files discovers ignore files in subdirectories but skips standard-ignored dirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir).resolve()

        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=root_path, capture_output=True, check=True)

        # Create root ignore files
        (root_path / ".gitignore").write_text("*.pyc\n")
        (root_path / ".flyteignore").write_text("*.log\n")

        # Create a subdirectory with ignore files (should be found)
        (root_path / "src").mkdir()
        (root_path / "src" / ".gitignore").write_text("*.tmp\n")
        (root_path / "src" / ".flyteignore").write_text("*.cache\n")

        # Create standard-ignored directories with ignore files (should be skipped)
        (root_path / ".venv").mkdir()
        (root_path / ".venv" / ".gitignore").write_text("should be skipped\n")

        (root_path / "__pycache__").mkdir()
        (root_path / "__pycache__" / ".gitignore").write_text("should be skipped\n")

        (root_path / "node_modules").mkdir()  # not in standard patterns but .git is
        (root_path / ".git_internal_test").mkdir()  # should NOT be pruned (not exact match)

        # Nested standard-ignored dir
        (root_path / "src" / ".mypy_cache").mkdir()
        (root_path / "src" / ".mypy_cache" / ".gitignore").write_text("should be skipped\n")

        subprocess.run(["git", "add", ".gitignore", ".flyteignore"], cwd=root_path, capture_output=True, check=True)

        git_ignore = GitIgnore(root_path)
        found = git_ignore.ignore_file_paths

        found_strs = {str(f) for f in found}

        # Root ignore files should be found
        assert str(root_path / ".gitignore") in found_strs
        assert str(root_path / ".flyteignore") in found_strs

        # Subdirectory ignore files should be found
        assert str(root_path / "src" / ".gitignore") in found_strs
        assert str(root_path / "src" / ".flyteignore") in found_strs

        # Standard-ignored directory ignore files should NOT be found
        assert str(root_path / ".venv" / ".gitignore") not in found_strs
        assert str(root_path / "__pycache__" / ".gitignore") not in found_strs
        assert str(root_path / "src" / ".mypy_cache" / ".gitignore") not in found_strs


# ---------------------------------------------------------------------------
# FlyteIgnore tests
# ---------------------------------------------------------------------------


def _git_commit_all(repo: Path) -> None:
    """Stage and commit all files in the repo."""
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True, check=True)


def test_flyteignore_excludes_tracked_file():
    """Tracked (committed) files listed in .flyteignore must be excluded by FlyteIgnore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "main.py").write_text("print('hello')")
        (root / "large_data.csv").write_text("a,b,c\n1,2,3")
        (root / ".flyteignore").write_text("large_data.csv\n")

        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "large_data.csv"), "tracked large_data.csv must be excluded"
        assert not ignore.is_ignored(root / "main.py"), "main.py must not be excluded"


def test_flyteignore_excludes_tracked_directory():
    """A directory pattern in .flyteignore must exclude all files inside it, even if tracked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "data").mkdir()
        (root / "data" / "file1.csv").write_text("col\n1")
        (root / "data" / "file2.csv").write_text("col\n2")
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("pass")
        (root / ".flyteignore").write_text("data/\n")

        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "data" / "file1.csv"), "data/file1.csv must be excluded"
        assert ignore.is_ignored(root / "data" / "file2.csv"), "data/file2.csv must be excluded"
        assert not ignore.is_ignored(root / "src" / "app.py"), "src/app.py must not be excluded"


def test_flyteignore_wildcard_pattern():
    """Wildcard patterns in .flyteignore must exclude all matching tracked files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "report.csv").write_text("x\n1")
        (root / "export.csv").write_text("y\n2")
        (root / "main.py").write_text("pass")
        (root / ".flyteignore").write_text("*.csv\n")

        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "report.csv"), "report.csv must be excluded by *.csv"
        assert ignore.is_ignored(root / "export.csv"), "export.csv must be excluded by *.csv"
        assert not ignore.is_ignored(root / "main.py"), "main.py must not be excluded"


def test_flyteignore_subdirectory_scope():
    """Patterns in src/.flyteignore apply only under src/, not at the repo root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "src").mkdir()
        (root / "src" / "secrets.json").write_text("{}")
        (root / "secrets.json").write_text("{}")  # root-level must NOT be excluded
        (root / "src" / ".flyteignore").write_text("secrets.json\n")

        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "src" / "secrets.json"), "src/secrets.json must be excluded"
        assert not ignore.is_ignored(root / "secrets.json"), "root secrets.json must NOT be excluded"


def test_flyteignore_works_without_git():
    """FlyteIgnore must work even when there is no git repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        # Intentionally no git init

        (root / "exclude_me.txt").write_text("sensitive")
        (root / "keep_me.py").write_text("pass")
        (root / ".flyteignore").write_text("exclude_me.txt\n")

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "exclude_me.txt"), "exclude_me.txt must be excluded without git"
        assert not ignore.is_ignored(root / "keep_me.py"), "keep_me.py must not be excluded"


def test_flyteignore_no_file_is_noop():
    """When no .flyteignore file exists, FlyteIgnore must exclude nothing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        (root / "main.py").write_text("pass")
        (root / "data.csv").write_text("a,b\n1,2")

        ignore = FlyteIgnore(root)
        assert not ignore.is_ignored(root / "main.py"), "main.py must not be excluded"
        assert not ignore.is_ignored(root / "data.csv"), "data.csv must not be excluded"


def test_flyteignore_bare_pattern_matches_at_any_depth():
    """Gitignore semantics: a bare pattern (no internal slash) matches at any depth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        # Three depths of the same filename, all tracked
        (root / "secrets.json").write_text("{}")
        (root / "subdir").mkdir()
        (root / "subdir" / "secrets.json").write_text("{}")
        (root / "a" / "b" / "c").mkdir(parents=True)
        (root / "a" / "b" / "c" / "secrets.json").write_text("{}")
        (root / "main.py").write_text("pass")

        (root / ".flyteignore").write_text("secrets.json\n")
        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "secrets.json")
        assert ignore.is_ignored(root / "subdir" / "secrets.json")
        assert ignore.is_ignored(root / "a" / "b" / "c" / "secrets.json")
        assert not ignore.is_ignored(root / "main.py")


def test_flyteignore_bare_directory_matches_at_any_depth():
    """Gitignore: bare 'data/' matches a directory named data anywhere."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "data").mkdir()
        (root / "data" / "x.csv").write_text("x")
        (root / "src" / "data").mkdir(parents=True)
        (root / "src" / "data" / "y.csv").write_text("y")
        (root / "src" / "keep.py").write_text("pass")
        (root / "a" / "b" / "data").mkdir(parents=True)
        (root / "a" / "b" / "data" / "z.csv").write_text("z")

        (root / ".flyteignore").write_text("data/\n")
        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "data" / "x.csv")
        assert ignore.is_ignored(root / "src" / "data" / "y.csv")
        assert ignore.is_ignored(root / "a" / "b" / "data" / "z.csv")
        assert not ignore.is_ignored(root / "src" / "keep.py")


def test_flyteignore_bare_wildcard_matches_at_any_depth():
    """Gitignore: bare '*.csv' matches CSVs at any depth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "top.csv").write_text("x")
        (root / "src").mkdir()
        (root / "src" / "deep.csv").write_text("y")
        (root / "src" / "keep.py").write_text("pass")

        (root / ".flyteignore").write_text("*.csv\n")
        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "top.csv")
        assert ignore.is_ignored(root / "src" / "deep.csv")
        assert not ignore.is_ignored(root / "src" / "keep.py")


def test_flyteignore_anchored_pattern_is_root_only():
    """Gitignore: leading '/' anchors a pattern to the .flyteignore's directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "secrets.json").write_text("{}")
        (root / "src").mkdir()
        (root / "src" / "secrets.json").write_text("{}")

        (root / ".flyteignore").write_text("/secrets.json\n")
        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "secrets.json"), "anchored pattern must match root"
        assert not ignore.is_ignored(root / "src" / "secrets.json"), "anchored pattern must NOT match nested"


def test_flyteignore_negation_preserved():
    """Negation patterns must continue to work after normalization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "debug.log").write_text("x")
        (root / "important.log").write_text("y")
        (root / "sub").mkdir()
        (root / "sub" / "debug.log").write_text("x")
        (root / "sub" / "important.log").write_text("y")

        (root / ".flyteignore").write_text("*.log\n!important.log\n")
        _git_commit_all(root)

        ignore = FlyteIgnore(root)
        assert ignore.is_ignored(root / "debug.log")
        assert ignore.is_ignored(root / "sub" / "debug.log")
        assert not ignore.is_ignored(root / "important.log")
        assert not ignore.is_ignored(root / "sub" / "important.log")


def test_flyteignore_in_default_ignores_excludes_tracked_file():
    """Integration: list_files_to_bundle must omit tracked files listed in .flyteignore."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        subprocess.run(["git", "init"], cwd=root, capture_output=True, check=True)

        (root / "app.py").write_text("pass")
        (root / "dataset.csv").write_text("x\n1\n2\n3")
        (root / ".flyteignore").write_text("dataset.csv\n")

        _git_commit_all(root)

        from flyte._code_bundle._ignore import FlyteIgnore, GitIgnore, StandardIgnore
        from flyte._code_bundle._packaging import list_files_to_bundle

        files, _ = list_files_to_bundle(root, False, GitIgnore, FlyteIgnore, StandardIgnore, copy_style="all")

        file_names = {Path(f).name for f in files}
        assert "app.py" in file_names, "app.py must be in the bundle"
        assert "dataset.csv" not in file_names, "tracked dataset.csv must be excluded from bundle"
