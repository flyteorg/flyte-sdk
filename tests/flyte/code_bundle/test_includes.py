"""
Tests for `Environment.include` across the unified code-bundling path.

Covers:
- `collect_env_include_files` resolves relative paths against each env's
  declaring file and dedupes.
- `ls_files(..., additional_files=...)` unions include files into the
  copy_style discovery and rehashes deterministically.
- `build_code_bundle(..., additional_files=...)` returns a tarball whose
  `files` list covers both the copy_style discovery and the extras.
- Include paths outside the bundle root produce a clear error.
- `copy_style='none'` with include files falls back to a relative-paths bundle.
"""

from __future__ import annotations

import pathlib
import tarfile
import tempfile
from pathlib import Path

import pytest

import flyte
from flyte._code_bundle._includes import collect_env_include_files
from flyte._code_bundle._utils import ls_files
from flyte._code_bundle.bundle import build_code_bundle


def _make_env_at(tmp_dir: Path, name: str, include: tuple[str, ...]) -> flyte.TaskEnvironment:
    """Instantiate a TaskEnvironment while pretending to be `tmp_dir/env.py`."""
    env = flyte.TaskEnvironment(name=name, include=include)
    env._declaring_file = str(tmp_dir / "env.py")
    return env


def test_collect_env_include_files_relative_to_declaring_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "a.html").write_text("A")
        (tmp_dir / "b.html").write_text("B")

        env = _make_env_at(tmp_dir, "relative_env", include=("a.html", "b.html"))

        resolved = collect_env_include_files([env])
        assert set(resolved) == {str((tmp_dir / "a.html").resolve()), str((tmp_dir / "b.html").resolve())}


def test_collect_env_include_files_dedupes_across_envs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "shared.html").write_text("S")

        env_one = _make_env_at(tmp_dir, "dedup_a", include=("shared.html",))
        env_two = _make_env_at(tmp_dir, "dedup_b", include=("shared.html",))

        resolved = collect_env_include_files([env_one, env_two])
        assert resolved == (str((tmp_dir / "shared.html").resolve()),)


def test_collect_env_include_files_absolute_passes_through():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        absolute = tmp_dir / "abs.txt"
        absolute.write_text("X")

        env = _make_env_at(tmp_dir, "absolute_env", include=(str(absolute),))

        resolved = collect_env_include_files([env])
        assert resolved == (str(absolute),)


def test_ls_files_unions_additional_files_and_rehashes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "a.py").write_text("print('a')")
        (tmp_dir / "template.html").write_text("<html></html>")

        files_a, digest_a = ls_files(tmp_dir, copy_file_detection="all", deref_symlinks=False)
        files_b, digest_b = ls_files(
            tmp_dir,
            copy_file_detection="all",
            deref_symlinks=False,
            additional_files=[str(tmp_dir / "template.html")],
        )

        # Both discoveries should include template.html via "all"; adding it again
        # through additional_files must not duplicate or change the hash.
        assert files_a == files_b
        assert digest_a == digest_b


def test_ls_files_rejects_path_outside_source():
    with tempfile.TemporaryDirectory() as outside:
        outside_file = Path(outside) / "outside.txt"
        outside_file.write_text("nope")

        with tempfile.TemporaryDirectory() as inside:
            with pytest.raises(ValueError, match="outside the bundle root"):
                ls_files(
                    Path(inside),
                    copy_file_detection="all",
                    deref_symlinks=False,
                    additional_files=[str(outside_file)],
                )


@pytest.mark.asyncio
async def test_build_code_bundle_includes_additional_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "script.py").write_text("print('x')")
        html = tmp_dir / "template.html"
        html.write_text("<html></html>")

        bundle = await build_code_bundle(
            from_dir=tmp_dir,
            dryrun=True,
            copy_style="all",
            additional_files=(str(html),),
        )

        # Resulting tarball should carry both files — the include file proves
        # the unified path is wired up.
        tgz_path = pathlib.Path(bundle.tgz)
        assert tgz_path.exists(), f"expected bundle at {tgz_path}"

        with tarfile.open(tgz_path, "r:gz") as tar:
            members = {m.name.lstrip("./") for m in tar.getmembers() if m.isfile()}

        assert "template.html" in members
        assert "script.py" in members


@pytest.mark.asyncio
async def test_build_code_bundle_copy_style_none_with_additional_files():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out_dir:
        tmp_dir = Path(tmp)
        html = tmp_dir / "only.html"
        html.write_text("<html></html>")

        bundle = await build_code_bundle(
            from_dir=tmp_dir,
            dryrun=True,
            copy_style="none",
            additional_files=(str(html),),
            copy_bundle_to=Path(out_dir),
        )

        tgz_path = pathlib.Path(bundle.tgz)
        assert tgz_path.exists()
        with tarfile.open(tgz_path, "r:gz") as tar:
            members = {m.name.lstrip("./") for m in tar.getmembers() if m.isfile()}

        assert "only.html" in members


def test_environment_include_coerces_list_to_tuple():
    env = flyte.TaskEnvironment(name="coerce_list", include=["one.html", "two.html"])
    assert env.include == ("one.html", "two.html")


def test_environment_include_default_is_empty_tuple():
    env = flyte.TaskEnvironment(name="default_empty")
    assert env.include == ()


def test_environment_include_rejects_bare_string():
    with pytest.raises(TypeError, match="sequence of str paths"):
        flyte.TaskEnvironment(name="bad_input", include="not_a_sequence.html")
