import tempfile
from pathlib import Path

import pytest

from flyte._code_bundle._ignore import STANDARD_IGNORE_PATTERNS
from flyte._internal.imagebuild.docker import PatternMatcher
from flyte._internal.imagebuild.utils import copy_files_to_context

# ========== Docker .dockerignore Syntax Coverage ==========


def test_dockerignore_syntax_comprehensive():
    """
    Test ALL Docker .dockerignore pattern syntax in one comprehensive test.
    Covers: *, **, ?, !, pattern order, case insensitivity, directory matching
    """
    pm = PatternMatcher(
        [
            # Basic wildcards
            "*.log",  # * matches any chars (not /)
            "test?.py",  # ? matches single char
            # ** patterns (match any depth)
            "**/*.pyc",  # Any .pyc file
            "**/node_modules",  # node_modules at any level
            "**/temp/**",  # Everything in any temp/ directory
            # Negation (!) - order matters!
            "secrets/*",  # Exclude all in secrets/
            "!secrets/public.key",  # But include public.key
            # Directory-only patterns
            "build/",  # Only if it's a directory
            # Exact matches
            ".env",  # Exact filename
            "Dockerfile.dev",  # Exact filename
        ]
    )

    # Test *
    assert pm.matches("debug.log")
    assert pm.matches("error.log")
    assert not pm.matches("subdir/app.log")  # * doesn't cross /

    # Test ?
    assert pm.matches("test1.py")
    assert not pm.matches("test12.py")  # ? is single char

    # Test **
    assert pm.matches("main.pyc")
    assert pm.matches("src/utils/helper.pyc")
    assert pm.matches("node_modules")
    assert pm.matches("frontend/node_modules/react")
    assert pm.matches("temp/cache.txt")
    assert pm.matches("build/temp/nested/file.txt")

    # Test negation (!) and order
    assert pm.matches("secrets/private.key")
    assert not pm.matches("secrets/public.key")  # Negated!

    # Test exact matches
    assert pm.matches(".env")
    assert pm.matches("Dockerfile.dev")
    assert not pm.matches("subdir/.env")  # Exact match is root only

    # Test case insensitivity
    assert pm.matches("DEBUG.LOG")
    assert pm.matches("Error.Log")


def test_pattern_order_and_negation():
    """
    Test that pattern order matters (last matching pattern wins).
    This is critical Docker .dockerignore behavior.
    """
    pm = PatternMatcher(
        [
            "*.txt",  # 1. Exclude all .txt
            "!important.txt",  # 2. Include important.txt
            "temp.txt",  # 3. Exclude temp.txt again
            "!**/*.txt",  # 4. Include all .txt in subdirs
        ]
    )

    assert not pm.matches("random.txt")  # Matches rule 4
    assert not pm.matches("important.txt")  # Rule 2 overrides
    assert not pm.matches("src/data.txt")  # Rule 4 overrides (in subdir)
    assert not pm.matches("file.py")  # No rules match


def test_real_world_python_project():
    """
    Test realistic Python project .dockerignore patterns.
    This is the PRIMARY use case.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create realistic Python project structure
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "requirements.txt").write_text("flask")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.pyc").write_text("compiled")
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "lib").mkdir()
        (tmp_path / ".venv" / "lib" / "site-packages").write_text("packages")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("app")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_app.py").write_text("test")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git")
        (tmp_path / "debug.log").write_text("log")
        (tmp_path / ".env").write_text("secrets")

        pm = PatternMatcher(
            [
                "**/__pycache__",
                "**/__pycache__/**",
                "**/*.pyc",
                "**/*.pyo",
                "**/.venv",
                "**/.venv/**",
                "**/venv",
                "**/venv/**",
                "**/.git",
                "**/.git/**",
                "**/*.log",
                "**/.env",
            ]
        )

        files = set(pm.walk(tmp_dir))

        # Should include source code
        assert "main.py" in files
        assert "requirements.txt" in files
        assert "src/app.py" in files
        assert "tests/test_app.py" in files

        # Should exclude build artifacts and secrets
        assert "__pycache__" not in files
        assert "__pycache__/main.pyc" not in files
        assert ".venv" not in files
        assert ".venv/lib/site-packages" not in files
        assert ".git" not in files
        assert ".git/config" not in files
        assert "debug.log" not in files
        assert ".env" not in files


# ========== Integration Tests ==========


def test_copy_files_with_dockerignore():
    """
    Test the full integration: copy_files_to_context with .dockerignore patterns.
    This is what actually runs in production.
    """
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as ctx_dir:
        src_path = Path(src_dir)
        ctx_path = Path(ctx_dir)

        # Create Python project
        (src_path / "main.py").write_text("main")
        (src_path / "utils.py").write_text("utils")
        (src_path / "test.pyc").write_text("compiled")
        (src_path / "__pycache__").mkdir()
        (src_path / "__pycache__" / "main.pyc").write_text("compiled")
        (src_path / ".venv").mkdir()
        (src_path / ".venv" / "lib.py").write_text("lib")
        (src_path / "secret.log").write_text("log")
        (src_path / "data.csv").write_text("data")
        (src_path / "src").mkdir()
        (src_path / "src" / "app.py").write_text("app")

        # Copy with custom + default patterns
        dst = copy_files_to_context(src_path, ctx_path, ignore_patterns=["**/*.log", "**/*.csv"] + STANDARD_IGNORE_PATTERNS)

        # Should copy Python source
        assert (dst / "main.py").exists()
        assert (dst / "utils.py").exists()
        assert (dst / "src" / "app.py").exists()

        # Should ignore (default patterns)
        assert not (dst / "test.pyc").exists()
        assert not (dst / "__pycache__").exists()
        assert not (dst / ".venv").exists()

        # Should ignore (custom patterns)
        assert not (dst / "secret.log").exists()
        assert not (dst / "data.csv").exists()


def test_copy_single_file_and_absolute_path():
    """Test edge cases: single file and absolute path handling"""
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as ctx_dir:
        src_path = Path(src_dir)
        ctx_path = Path(ctx_dir)

        # Test 1: Single file (no patterns applied)
        single_file = src_path / "requirements.txt"
        single_file.write_text("flask==2.0.0")
        dst_file = copy_files_to_context(single_file, ctx_path)

        assert dst_file.exists()
        assert dst_file.read_text() == "flask==2.0.0"

        # Test 2: Absolute path gets special treatment
        abs_dir = (src_path / "project").resolve()
        abs_dir.mkdir()
        (abs_dir / "main.py").write_text("main")

        dst_abs = copy_files_to_context(abs_dir, ctx_path)

        assert "_flyte_abs_context" in str(dst_abs)
        assert (dst_abs / "main.py").exists()


# ========== Async Handler Test ==========


@pytest.mark.asyncio
async def test_copy_config_handler_integration():
    """
    Test that CopyConfigHandler properly uses PatternMatcher.
    This verifies the entire pipeline works end-to-end.
    """
    from flyte._image import CopyConfig
    from flyte._internal.imagebuild.docker_builder import CopyConfigHandler

    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as ctx_dir:
        src_path = Path(src_dir)
        ctx_path = Path(ctx_dir)

        # Create files
        (src_path / "config.yaml").write_text("config")
        (src_path / "app.py").write_text("app")
        (src_path / "test.pyc").write_text("compiled")
        (src_path / "__pycache__").mkdir()
        (src_path / "__pycache__" / "app.pyc").write_text("compiled")

        layer = CopyConfig(path_type=1, src=src_path, dst="/app")
        dockerfile = ""

        result = await CopyConfigHandler.handle(
            layer, ctx_path, dockerfile, docker_ignore_patterns=["**/*.pyc", "**/__pycache__"]
        )

        # Verify Dockerfile generated
        assert "COPY" in result
        assert "/app" in result

        # Verify only correct files copied
        yaml_files = list(ctx_path.rglob("*.yaml"))
        py_files = list(ctx_path.rglob("app.py"))
        pyc_files = list(ctx_path.rglob("*.pyc"))

        assert len(yaml_files) == 1  # config.yaml copied
        assert len(py_files) == 1  # app.py copied
        assert len(pyc_files) == 0  # .pyc files ignored


# ========== Docker Syntax Edge Cases ==========


def test_docker_syntax_edge_cases():
    """Test less common but valid Docker .dockerignore syntax"""
    pm = PatternMatcher(
        [
            # Comment handling (should be stripped by file parser, but test anyway)
            "*.tmp",
            # Multiple wildcards
            "**/*test*/**",
            # Root-only patterns (no leading **)
            "node_modules",
            # Complex negations
            "logs/**",
            "!logs/important/**",
        ]
    )

    assert pm.matches("file.tmp")
    assert pm.matches("src/mytest/file.txt")
    assert pm.matches("node_modules")
    assert pm.matches("logs/debug.log")
    assert not pm.matches("logs/important/keep.log")


def test_empty_and_special_cases():
    """Test empty patterns and special handling"""
    # Empty patterns
    pm_empty = PatternMatcher([])
    assert not pm_empty.matches("anything.txt")

    # Whitespace in patterns (should be normalized)
    pm_spaces = PatternMatcher(["  *.log  ", "\t*.tmp\t"])
    assert pm_spaces.matches("debug.log")
    assert pm_spaces.matches("temp.tmp")


def test_dockerignore_patterns():
    """
    Comprehensive .dockerignore pattern test - all critical behaviors in one place.

    Tests:
    - dir/* matches recursively (via parent matching)
    - Negation and pattern order (last match wins)
    - Depth-specific patterns
    - Real-world integration
    """

    # 1. dir/* matches ALL contents (not just direct children)
    pm1 = PatternMatcher(["temp/*"])
    assert pm1.matches("temp/file.txt")  # Direct
    assert pm1.matches("temp/a/b/deep.txt")  # Nested (parent 'temp/a' matches)
    assert not pm1.matches("other/file.txt")

    # 2. Negation overrides (last match wins)
    pm2 = PatternMatcher(
        [
            "secrets/**",
            "!secrets/public",
            "!secrets/public/**",
        ]
    )
    assert pm2.matches("secrets/key.txt")  # Excluded
    assert not pm2.matches("secrets/public/cert.pem")  # Kept via negation

    # 3. Depth-specific: **/*/*.log = at least 1 dir before .log
    pm3 = PatternMatcher(["**/*/*.log"])
    assert not pm3.matches("app.log")  # Root level
    assert pm3.matches("logs/app.log")  # 1+ dirs deep
    assert pm3.matches("a/b/c/app.log")

    # 4. Real-world: exclude build/* but keep build/config/
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        (tmp / "src").mkdir()
        (tmp / "src" / "main.py").write_text("src")

        (tmp / "build").mkdir()
        (tmp / "build" / "bundle.js").write_text("build")
        (tmp / "build" / "config").mkdir()
        (tmp / "build" / "config" / "env.json").write_text("config")

        pm4 = PatternMatcher(
            [
                "build/*",
                "!build/config",
                "!build/config/**",
            ]
        )

        files = set(pm4.walk(tmp_dir))

        assert "src/main.py" in files
        assert "build/config/env.json" in files  # Kept ✅
        assert "build/bundle.js" not in files  # Excluded ❌
