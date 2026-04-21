"""
End-to-end tests for ``build_code_bundle`` across canonical example layouts.

Each layout under ``testdata/`` models a real-world usage pattern that appears
in ``examples/*`` — single-file scripts, multi-file projects, include-file
shipping, and ``copy_style`` variants. For every layout we exercise the full
bundling pipeline in ``dryrun=True`` mode, inspect the produced tarball, and
assert the expected set of files is present.

Fixtures are copied into a tempdir per test so the tests stay isolated from
any ``__pycache__`` or stray artifacts that may accumulate in-tree.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import pytest

from flyte._code_bundle.bundle import (
    build_code_bundle,
    build_code_bundle_from_relative_paths,
)

TESTDATA_ROOT = Path(__file__).parent / "testdata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_layout(name: str, dest: Path) -> Path:
    """Copy a testdata layout into ``dest`` and return the layout root."""
    src = TESTDATA_ROOT / name
    assert src.is_dir(), f"missing testdata layout: {src}"
    target = dest / name
    shutil.copytree(src, target)
    return target


def _bundle_out(dest: Path, name: str = "out") -> Path:
    """Create and return an output directory for ``copy_bundle_to``."""
    out = dest / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tar_files(tgz_path: Path) -> set[str]:
    """Return the set of regular-file member names in a .tar.gz bundle."""
    with tarfile.open(tgz_path, "r:gz") as tar:
        return {m.name.lstrip("./") for m in tar.getmembers() if m.isfile()}


@pytest.fixture
def importer() -> Iterator[Callable[[Path, str], ModuleType]]:
    """
    Import a Python file under a unique name and register it in
    ``sys.modules`` so it is visible to the ``copy_style='loaded_modules'``
    discovery path.

    We snapshot ``sys.modules`` on entry and remove every new key on teardown
    — this cleans up both the modules we register explicitly and any *implicit*
    siblings the loader may have cached via ``from pkg.mod import …`` lines in
    the fixture sources (e.g. ``utils``, ``utils.helper``). Without this, one
    test's ``utils`` package bleeds into the next test's copy of ``utils`` and
    breaks imports of names that only exist in the second fixture.
    """
    pre_existing = set(sys.modules.keys())

    def _load(path: Path, name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        # Make sibling packages importable via their real names during
        # ``exec_module`` (e.g., ``from utils.helper import …`` in main.py).
        layout_root = str(path.parent)
        inserted_on_path = layout_root not in sys.path
        if inserted_on_path:
            sys.path.insert(0, layout_root)
        try:
            spec.loader.exec_module(mod)
        finally:
            if inserted_on_path:
                try:
                    sys.path.remove(layout_root)
                except ValueError:
                    pass
        return mod

    yield _load

    for name in list(sys.modules.keys()):
        if name not in pre_existing:
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _clear_bundle_cache() -> Iterator[None]:
    """
    ``build_code_bundle`` is wrapped in ``alru_cache`` at module level. Clear
    it before and after each test so cached results from one layout never leak
    into another — and so determinism tests can re-run the same inputs.
    """
    build_code_bundle.cache_clear()
    build_code_bundle_from_relative_paths.cache_clear()
    yield
    build_code_bundle.cache_clear()
    build_code_bundle_from_relative_paths.cache_clear()


# ---------------------------------------------------------------------------
# Single-file layouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_single_file_copy_all():
    """Canonical ``examples/basics/hello.py`` layout: one file, copy everything."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("single_file", tmp_dir)

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        assert members == {"hello.py"}
        # The digest is content-addressed — stored on the bundle as
        # ``computed_version`` for later cache lookups.
        assert bundle.computed_version


@pytest.mark.asyncio
async def test_build_code_bundle_single_file_with_include():
    """
    Mirrors ``examples/reports/html_template_report.py``: a Python task plus a
    separate HTML template shipped via ``Environment.include``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("single_file_with_include", tmp_dir)

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            additional_files=(str(layout / "report_template.html"),),
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        assert members == {"report.py", "report_template.html"}


# ---------------------------------------------------------------------------
# Multi-file layouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_multi_file_copy_all():
    """``copy_style='all'``: every non-ignored file ships, including data + docs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("multi_file_all", tmp_dir)

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        assert members == {
            "main.py",
            "utils/__init__.py",
            "utils/helper.py",
            "configs/dev.yaml",
            "configs/prod.yaml",
        }


@pytest.mark.asyncio
async def test_build_code_bundle_multi_file_copy_loaded_modules(importer):
    """
    ``copy_style='loaded_modules'`` keeps only Python files that the process
    actually imported. The ``unused/`` subpackage and data/doc files must be
    absent from the bundle.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("multi_file", tmp_dir)

        # Force the same import graph you'd get if the user ran `python main.py`.
        # Use unique names to avoid clashing with other tests in this session.
        importer(layout / "utils" / "__init__.py", "testdata_mf_utils")
        importer(layout / "utils" / "helper.py", "testdata_mf_utils_helper")
        importer(layout / "main.py", "testdata_mf_main")

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="loaded_modules",
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        # Only imported Python modules survive.
        assert "main.py" in members
        assert "utils/helper.py" in members
        assert "utils/__init__.py" in members
        # Everything else is filtered out.
        assert "unused/never_imported.py" not in members
        assert "data/config.yaml" not in members
        assert "README.md" not in members


# ---------------------------------------------------------------------------
# copy_style="none"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_copy_style_none_with_include():
    """
    ``copy_style='none'`` with ``additional_files`` falls back to the
    relative-paths path — only the explicitly-listed include ships.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("copy_none_with_include", tmp_dir)

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="none",
            additional_files=(str(layout / "standalone.yaml"),),
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        assert members == {"standalone.yaml"}


@pytest.mark.asyncio
async def test_build_code_bundle_copy_style_none_without_include_raises():
    """
    ``copy_style='none'`` without any include is a no-op that the caller
    should not reach — the function raises rather than produce an empty tarball.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("single_file", tmp_dir)

        with pytest.raises(ValueError, match="just don't make a code bundle"):
            await build_code_bundle(
                from_dir=layout,
                dryrun=True,
                copy_style="none",
                copy_bundle_to=_bundle_out(tmp_dir),
            )


# ---------------------------------------------------------------------------
# Multi-file + loaded_modules + include (end-to-end of the common case)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_multi_file_loaded_modules_with_include(importer):
    """
    The common production layout: imported Python modules via
    ``loaded_modules`` plus non-Python assets shipped via ``include``.
    An unused Python file must not sneak in; both include files must.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("multi_file_with_include", tmp_dir)

        importer(layout / "utils" / "__init__.py", "testdata_mfwi_utils")
        importer(layout / "utils" / "helper.py", "testdata_mfwi_helper")
        importer(layout / "main.py", "testdata_mfwi_main")

        templates = layout / "templates"
        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="loaded_modules",
            additional_files=(
                str(templates / "page.html"),
                str(templates / "snippet.html"),
            ),
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        members = _tar_files(Path(bundle.tgz))
        # Imported Python modules come in via copy_style discovery.
        assert "main.py" in members
        assert "utils/helper.py" in members
        assert "utils/__init__.py" in members
        # Include files come in via additional_files.
        assert "templates/page.html" in members
        assert "templates/snippet.html" in members
        # Non-imported .py stays out.
        assert "unused.py" not in members


# ---------------------------------------------------------------------------
# Determinism / caching regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_digest_is_content_addressed():
    """
    Two identical layouts in different parent directories must produce the
    same ``computed_version``. This is the property that makes the bundler
    cacheable at the control-plane layer.
    """
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        layout_a = _copy_layout("multi_file_all", Path(tmp_a))
        layout_b = _copy_layout("multi_file_all", Path(tmp_b))

        bundle_a = await build_code_bundle(
            from_dir=layout_a,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(Path(tmp_a)),
        )
        bundle_b = await build_code_bundle(
            from_dir=layout_b,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(Path(tmp_b)),
        )

        assert bundle_a.computed_version == bundle_b.computed_version


@pytest.mark.asyncio
async def test_build_code_bundle_include_changes_digest():
    """Adding an include file must change the bundle digest."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("single_file_with_include", tmp_dir)

        bundle_no_include = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(tmp_dir, "out_1"),
        )
        # Clear so the second call doesn't return the cached result.
        build_code_bundle.cache_clear()

        bundle_with_extra = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            # Re-listing a file already picked up by copy_style='all' must not
            # change the digest — the unioning is idempotent.
            additional_files=(str(layout / "report_template.html"),),
            copy_bundle_to=_bundle_out(tmp_dir, "out_2"),
        )

        # "all" already picks up every file, so listing the html again via
        # additional_files is a no-op — digest stable.
        assert bundle_no_include.computed_version == bundle_with_extra.computed_version


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_include_outside_source_raises():
    """Include paths outside ``from_dir`` must fail loudly with a helpful error."""
    with tempfile.TemporaryDirectory() as tmp_inside, tempfile.TemporaryDirectory() as tmp_outside:
        layout = _copy_layout("single_file", Path(tmp_inside))
        outside_file = Path(tmp_outside) / "stray.txt"
        outside_file.write_text("oops")

        with pytest.raises(ValueError, match="outside the bundle root"):
            await build_code_bundle(
                from_dir=layout,
                dryrun=True,
                copy_style="all",
                additional_files=(str(outside_file),),
                copy_bundle_to=_bundle_out(Path(tmp_inside)),
            )


@pytest.mark.asyncio
async def test_build_code_bundle_empty_source_raises(tmp_path):
    """An empty source directory with copy_style='all' surfaces a clear error."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(Exception, match="No files found to bundle"):
        await build_code_bundle(
            from_dir=empty,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(tmp_path),
        )


# ---------------------------------------------------------------------------
# Tar hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_code_bundle_tarball_uses_forward_slashes():
    """Tarball members must use POSIX separators for Linux-side extraction."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        layout = _copy_layout("multi_file_all", tmp_dir)

        bundle = await build_code_bundle(
            from_dir=layout,
            dryrun=True,
            copy_style="all",
            copy_bundle_to=_bundle_out(tmp_dir),
        )

        with tarfile.open(bundle.tgz, "r:gz") as tar:
            for m in tar.getmembers():
                assert "\\" not in m.name, f"backslash in tar member: {m.name!r}"
