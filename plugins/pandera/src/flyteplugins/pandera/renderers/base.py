from __future__ import annotations

from typing import Any, Protocol

from pandera.errors import SchemaErrors


class PanderaReportRenderer(Protocol):
    def to_html(self, title: str, data: Any, schema: Any, error: SchemaErrors | None = None) -> str: ...
