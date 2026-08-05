"""Persist remote TUI preferences under ``~/.flyte/remote-tui-settings.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".flyte" / "remote-tui-settings.json"
MAX_RECENT_PROJECTS = 8


def resolve_config_key(config: str | Path | None) -> str:
    """Stable settings key for the active Flyte config file."""
    if config is not None:
        return str(Path(config).expanduser().resolve())
    from flyte.config._reader import resolve_config_path

    resolved = resolve_config_path()
    if resolved is not None:
        return str(resolved.expanduser().resolve())
    return "default"


def _load_raw() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {"configs": {}}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"configs": {}}
    if not isinstance(data, dict):
        return {"configs": {}}
    configs = data.get("configs")
    if not isinstance(configs, dict):
        data["configs"] = {}
    return data


def _save_raw(data: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def get_recent_projects(config_key: str) -> list[str]:
    """Return recent project IDs for *config_key*, most recent first."""
    data = _load_raw()
    entry = data.get("configs", {}).get(config_key, {})
    if not isinstance(entry, dict):
        return []
    recent = entry.get("recent_projects", [])
    if not isinstance(recent, list):
        return []
    return [str(p) for p in recent if p]


def record_recent_project(config_key: str, project_id: str) -> None:
    """Move *project_id* to the front of recents for *config_key*."""
    data = _load_raw()
    configs = data.setdefault("configs", {})
    entry = configs.setdefault(config_key, {})
    if not isinstance(entry, dict):
        entry = {}
        configs[config_key] = entry
    recent = [p for p in entry.get("recent_projects", []) if p != project_id]
    if not isinstance(recent, list):
        recent = []
    recent.insert(0, project_id)
    entry["recent_projects"] = recent[:MAX_RECENT_PROJECTS]
    _save_raw(data)
