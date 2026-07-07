"""Tests for remote TUI settings persistence."""

from __future__ import annotations

import json

import pytest

from flyte.cli._tui._remote import _settings


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "remote-tui-settings.json"
    monkeypatch.setattr(_settings, "SETTINGS_PATH", path)
    return path


def test_record_and_get_recent_projects(settings_file):
    _settings.record_recent_project("/cfg/a.yaml", "proj-b")
    _settings.record_recent_project("/cfg/a.yaml", "proj-a")
    _settings.record_recent_project("/cfg/a.yaml", "proj-b")

    assert _settings.get_recent_projects("/cfg/a.yaml") == ["proj-b", "proj-a"]


def test_recent_projects_scoped_by_config_key(settings_file):
    _settings.record_recent_project("/cfg/a.yaml", "only-a")
    _settings.record_recent_project("/cfg/b.yaml", "only-b")

    assert _settings.get_recent_projects("/cfg/a.yaml") == ["only-a"]
    assert _settings.get_recent_projects("/cfg/b.yaml") == ["only-b"]


def test_resolve_config_key_explicit_path(tmp_path):
    cfg = tmp_path / "my-config.yaml"
    cfg.write_text("task:\n  domain: dev\n", encoding="utf-8")
    key = _settings.resolve_config_key(str(cfg))
    assert key == str(cfg.resolve())


def test_settings_written_under_flyte_dir(settings_file):
    _settings.record_recent_project("default", "p1")
    data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert data["configs"]["default"]["recent_projects"] == ["p1"]
