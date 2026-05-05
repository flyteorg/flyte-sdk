from __future__ import annotations

import base64
import html
from enum import Enum
from pathlib import PurePath
from typing import Any, TypeAlias

import flyte.report
import yaml
from flyte.syncify import syncify

from omegaconf import DictConfig, ListConfig

from .codec import (
    KIND_DICT,
    KIND_ENUM,
    KIND_LIST,
    KIND_MISSING,
    KIND_PATH,
    KIND_TUPLE,
    PAYLOAD_KIND,
    PAYLOAD_MARKER,
    PAYLOAD_NAME,
    PAYLOAD_VALUE,
    PAYLOAD_VALUES,
    serialize_omegaconf,
)

OmegaConfContainer: TypeAlias = DictConfig | ListConfig

_REPORT_CSS = """
<style>
  .omegaconf-report {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0 0 1rem;
  }
  .omegaconf-report h2 {
    font-size: 1rem;
    font-weight: 650;
    margin: 0 0 0.5rem;
  }
  .omegaconf-report pre {
    background: #f7f8fa;
    border: 1px solid #d9dee7;
    border-radius: 6px;
    color: #172033;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.875rem;
    line-height: 1.5;
    margin: 0;
    overflow-x: auto;
    padding: 0.875rem 1rem;
    white-space: pre;
  }
</style>
""".strip()


def to_yaml(config: OmegaConfContainer, *, sort_keys: bool = False) -> str:
    """Render an OmegaConf container as clean YAML for humans."""
    plain_value = _payload_to_plain_value(serialize_omegaconf(config))
    return yaml.safe_dump(plain_value, sort_keys=sort_keys)


def to_html(
    config: OmegaConfContainer,
    *,
    title: str = "OmegaConf config",
    sort_keys: bool = False,
) -> str:
    """Render an OmegaConf container as escaped HTML suitable for a Flyte report."""
    yaml_text = to_yaml(config, sort_keys=sort_keys)
    return (
        _REPORT_CSS + "\n"
        '<section class="omegaconf-report">'
        f"<h2>{html.escape(title)}</h2>"
        f"<pre><code>{html.escape(yaml_text)}</code></pre>"
        "</section>"
    )


@syncify
async def log_yaml(
    config: OmegaConfContainer,
    *,
    title: str = "OmegaConf config",
    tab: str = "OmegaConf",
    sort_keys: bool = False,
    do_flush: bool = True,
) -> None:
    """Append a YAML rendering of an OmegaConf container to a Flyte report tab."""
    flyte.report.get_tab(tab).log(to_html(config, title=title, sort_keys=sort_keys))
    if do_flush:
        await flyte.report.flush.aio()


@syncify
async def replace_yaml(
    config: OmegaConfContainer,
    *,
    title: str = "OmegaConf config",
    tab: str = "OmegaConf",
    sort_keys: bool = False,
    do_flush: bool = False,
) -> None:
    """Replace a Flyte report tab with a YAML rendering of an OmegaConf container."""
    flyte.report.get_tab(tab).replace(to_html(config, title=title, sort_keys=sort_keys))
    if do_flush:
        await flyte.report.flush.aio()


def _payload_to_plain_value(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_payload_to_plain_value(item) for item in payload]

    if not isinstance(payload, dict):
        return _plain_leaf(payload)

    if payload.get(PAYLOAD_MARKER) is not True:
        return {key: _payload_to_plain_value(value) for key, value in payload.items()}

    kind = payload[PAYLOAD_KIND]

    if kind == KIND_MISSING:
        return "???"

    if kind == KIND_ENUM:
        return payload.get(PAYLOAD_NAME, payload.get(PAYLOAD_VALUE))

    if kind == KIND_PATH:
        return payload[PAYLOAD_VALUE]

    if kind == KIND_TUPLE:
        return [_payload_to_plain_value(item) for item in payload[PAYLOAD_VALUES]]

    if kind == KIND_LIST:
        return [_payload_to_plain_value(item) for item in payload[PAYLOAD_VALUES]]

    if kind == KIND_DICT:
        return {key: _payload_to_plain_value(value) for key, value in payload[PAYLOAD_VALUES].items()}

    return {key: _payload_to_plain_value(value) for key, value in payload.items()}


def _plain_leaf(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(value).decode("ascii")
    if isinstance(value, tuple):
        return [_plain_leaf(item) for item in value]
    return value
