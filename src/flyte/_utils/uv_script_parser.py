import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import toml


@dataclass
class ToolUVConfig:
    exclude_newer: Optional[str] = None


@dataclass
class UVScriptMetadata:
    requires_python: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tool: Optional[Dict[str, ToolUVConfig]] = None


def _extract_uv_metadata_block(text: str) -> str | None:
    pattern = re.compile(r"# /// script\s*(.*?)# ///", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    lines = [line.lstrip("# ").rstrip() for line in match.group(1).splitlines()]
    return "\n".join(lines)


def parse_uv_script_file(path: pathlib.Path) -> UVScriptMetadata:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    raw_header = _extract_uv_metadata_block(text)
    if raw_header is None:
        raise ValueError("No uv metadata block found")

    try:
        data = toml.loads(raw_header)
    except toml.TomlDecodeError as e:
        raise ValueError(f"Invalid TOML in metadata block: {e}")

    tool_data = data.get("tool", {}).get("uv", {})
    return UVScriptMetadata(
        requires_python=data.get("requires-python"),
        dependencies=data.get("dependencies", []),
        tool={"uv": ToolUVConfig(exclude_newer=tool_data.get("exclude-newer"))} if tool_data else None,
    )
