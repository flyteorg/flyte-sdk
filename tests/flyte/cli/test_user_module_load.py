"""Tests for converting user-module exec errors into click.ClickException."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import click
import pytest

from flyte.cli._common import ObjectsPerFileGroup


class _DummyGroup(ObjectsPerFileGroup):
    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        return {k: v for k, v in vars(module).items() if not k.startswith("_")}

    def _get_command_for_obj(self, ctx, obj_name, obj):  # pragma: no cover - unused
        raise NotImplementedError


def _make_group(tmp_path: Path, body: str) -> _DummyGroup:
    p = tmp_path / "user_mod.py"
    p.write_text(body)
    return _DummyGroup(filename=p, name="dummy")


def test_import_error_in_user_module_becomes_click_exception(tmp_path):
    grp = _make_group(tmp_path, "from definitely_not_a_module import nope\n")
    with pytest.raises(click.ClickException) as excinfo:
        _ = grp.objs
    assert "Failed to load" in excinfo.value.message
    assert "ModuleNotFoundError" in excinfo.value.message or "ImportError" in excinfo.value.message


def test_value_error_in_user_module_becomes_click_exception(tmp_path):
    grp = _make_group(tmp_path, "raise ValueError('bad user config')\n")
    with pytest.raises(click.ClickException) as excinfo:
        _ = grp.objs
    assert "Failed to load" in excinfo.value.message
    assert "ValueError" in excinfo.value.message
    assert "bad user config" in excinfo.value.message


def test_syntax_error_in_user_module_becomes_click_exception(tmp_path):
    grp = _make_group(tmp_path, "def broken(:\n")
    with pytest.raises(click.ClickException) as excinfo:
        _ = grp.objs
    assert "Failed to load" in excinfo.value.message
    assert "SyntaxError" in excinfo.value.message


def test_click_exception_in_user_module_is_passed_through(tmp_path):
    grp = _make_group(
        tmp_path,
        "import rich_click as click\nraise click.ClickException('explicit user-facing error')\n",
    )
    with pytest.raises(click.ClickException) as excinfo:
        _ = grp.objs
    # The original ClickException is preserved, not wrapped.
    assert excinfo.value.message == "explicit user-facing error"


def test_unexpected_exception_is_not_swallowed(tmp_path):
    grp = _make_group(tmp_path, "raise RuntimeError('genuinely unexpected')\n")
    with pytest.raises(RuntimeError):
        _ = grp.objs
