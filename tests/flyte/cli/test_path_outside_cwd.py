"""Tests for addressing a task/app file by a path that lies outside the current directory."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from flyte.cli._common import FileGroup, relative_to_cwd


@pytest.fixture()
def outside(tmp_path, monkeypatch):
    """A source directory and a *separate* working directory, neither inside the other."""
    src = tmp_path / "project"
    src.mkdir()
    (src / "task.py").write_text("x = 1\n")
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return src


def test_relative_to_cwd_shortens_a_path_inside_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inside = Path(os.getcwd()) / "pkg" / "task.py"
    assert relative_to_cwd(inside) == Path("pkg/task.py")


def test_relative_to_cwd_keeps_a_path_outside_the_current_directory(outside):
    """The bug: `Path.relative_to` raises for a path with no relative spelling under cwd."""
    absolute = (outside / "task.py").resolve()
    assert relative_to_cwd(absolute) == absolute


def test_relative_to_cwd_keeps_a_relative_path(outside):
    """A relative path is already as short as it gets; `is_relative_to` always rejects it."""
    assert relative_to_cwd(Path("../project/task.py")) == Path("../project/task.py")


def test_file_group_lists_outside_paths_as_absolute(outside):
    """FileGroup deliberately emits absolute paths for files it cannot spell relatively.

    This is what feeds the per-file groups below, so the two halves have to agree: whatever
    `files` emits must be something the group constructors accept.
    """
    files = FileGroup(name="g", directory=outside).files
    assert files == [os.fspath((outside / "task.py"))]


def test_file_group_shortens_paths_inside_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.py").write_text("x = 1\n")
    assert FileGroup(name="g", directory=tmp_path).files == ["task.py"]


def test_task_per_file_group_accepts_a_path_outside_the_current_directory(outside):
    """`flyte run /abs/path/outside/cwd/task.py` used to die with 'is not in the subpath of'."""
    from flyte.cli._run import RunArguments, TaskPerFileGroup

    target = outside / "task.py"
    grp = TaskPerFileGroup(filename=target, run_args=RunArguments(), name=str(target))
    assert grp.filename == target


def test_app_per_file_group_accepts_a_path_outside_the_current_directory(outside):
    """Same defect on the `flyte serve` side."""
    from flyte.cli._serve import AppPerFileGroup, ServeArguments

    target = outside / "task.py"
    grp = AppPerFileGroup(filename=target, serve_args=ServeArguments(), name=str(target))
    assert grp.filename == target
