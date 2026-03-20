from __future__ import annotations

from typing import Any, Protocol


class PanderaReportRenderer(Protocol):
    def to_html(self, title: str, data: Any, schema: Any, error: Exception | None = None) -> str: ...
