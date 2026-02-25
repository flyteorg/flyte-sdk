import tarfile
import tempfile
from pathlib import Path

import pytest

from flyte._code_bundle._include import FlyteInclude
from flyte._code_bundle.bundle import build_code_bundle


def test_flyte_include_no_file():
    """When no .flyteinclude file exists, has_patterns is False and patterns is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        fi = FlyteInclude(root)
        assert fi.has_patterns is False
        assert fi.patterns == []


def test_flyte_include_with_patterns():
    """Parses paths, glob patterns, comments, and blank lines correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        (root / ".flyteinclude").write_text(
            "# this is a comment\n"
            "\n"
            "lib1/\n"
            "  project2/  \n"  # leading/trailing whitespace stripped
            "src/**/*.py\n"
            "# another comment\n"
            "pyproject.toml\n"
        )
        fi = FlyteInclude(root)
        assert fi.has_patterns is True
        assert fi.patterns == ["lib1/", "project2/", "src/**/*.py", "pyproject.toml"]


def test_flyte_include_empty_file():
    """An empty .flyteinclude (or all comments/blanks) yields no patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()
        (root / ".flyteinclude").write_text("# just a comment\n\n")
        fi = FlyteInclude(root)
        assert fi.has_patterns is False
        assert fi.patterns == []


# ---------------------------------------------------------------------------
# Integration tests — full bundle flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_uses_flyteinclude():
    """When .flyteinclude is present, build_code_bundle() bundles only the listed paths
    and ignores everything else in the root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        # Simulate a monorepo: lib1/ and project/ should be included, lib2/ should not.
        (root / "lib1").mkdir()
        (root / "lib1" / "utils.py").write_text("def helper(): pass")
        (root / "lib1" / "models.py").write_text("class Model: pass")

        (root / "lib2").mkdir()
        (root / "lib2" / "other.py").write_text("def other(): pass")

        (root / "project").mkdir()
        (root / "project" / "workflow.py").write_text("import lib1")

        (root / ".flyteinclude").write_text("lib1/\nproject/\n")

        bundle_out = root / "bundle_out"
        bundle_out.mkdir()

        bundle = await build_code_bundle(root, dryrun=True, copy_bundle_to=bundle_out)

        rel_files = {str(Path(f).relative_to(root)) for f in bundle.files}

        assert "lib1/utils.py" in rel_files
        assert "lib1/models.py" in rel_files
        assert "project/workflow.py" in rel_files
        # lib2 was not listed in .flyteinclude — must be absent
        assert "lib2/other.py" not in rel_files

        # Verify the tarball on disk matches: extract and check member names
        assert bundle.tgz is not None
        with tarfile.open(bundle.tgz, "r:gz") as tar:
            names = set(tar.getnames())
        assert "lib1/utils.py" in names
        assert "project/workflow.py" in names
        assert "lib2/other.py" not in names


@pytest.mark.asyncio
async def test_build_code_bundle_without_flyteinclude_walks_normally():
    """Without .flyteinclude, build_code_bundle() falls back to the normal ignore-based
    walk and includes all non-ignored files under the root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir).resolve()

        (root / "lib1").mkdir()
        (root / "lib1" / "utils.py").write_text("def helper(): pass")

        (root / "lib2").mkdir()
        (root / "lib2" / "other.py").write_text("def other(): pass")

        # No .flyteinclude — all files should be bundled
        bundle_out = root / "bundle_out"
        bundle_out.mkdir()

        bundle = await build_code_bundle(root, dryrun=True, copy_bundle_to=bundle_out, copy_style="all")

        rel_files = {str(Path(f).relative_to(root)) for f in bundle.files}

        assert "lib1/utils.py" in rel_files
        assert "lib2/other.py" in rel_files
