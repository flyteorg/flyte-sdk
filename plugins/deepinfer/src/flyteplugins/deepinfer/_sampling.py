"""Sampling and structured-output options for `DeepInfer.generate` (internal).

DeepInfer constrains decoding through `SamplingParams(so_args={"schema_type": ...,
"schema": ...})`, with `schema_type` one of `regex`, `json` (a JSON-serialized schema
string) or `ebnf`. `build_sampling` turns `generate`'s keyword arguments into that dict
plus a parser for the decoded text, without importing DeepInfer.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional


def _identity(text: str) -> Any:
    return text


def build_sampling(
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    json_schema: Any = None,
    regex: Optional[str] = None,
    grammar: Optional[str] = None,
    **extra: Any,
) -> tuple[dict[str, Any], Callable[[str], Any]]:
    """Return `(SamplingParams kwargs, result parser)`.

    Only the options that are set are included, so everything else falls back to
    DeepInfer's own `SamplingParams` defaults rather than a copy that could drift.
    """
    constraints = [
        name for name, value in (("json_schema", json_schema), ("regex", regex), ("grammar", grammar)) if value
    ]
    if len(constraints) > 1:
        raise ValueError(f"Set at most one of json_schema, regex and grammar (got {', '.join(constraints)})")

    params: dict[str, Any] = {
        k: v
        for k, v in {"temperature": temperature, "top_p": top_p, "top_k": top_k, "max_tokens": max_tokens}.items()
        if v is not None
    }
    params.update(extra)

    parse: Callable[[str], Any] = _identity
    if regex:
        params["so_args"] = {"schema_type": "regex", "schema": regex}
    elif grammar:
        params["so_args"] = {"schema_type": "ebnf", "schema": grammar}
    elif json_schema is not None:
        schema, parse = _json_schema(json_schema)
        params["so_args"] = {"schema_type": "json", "schema": schema}
    return params, parse


def _json_schema(spec: Any) -> tuple[str, Callable[[str], Any]]:
    """A schema string for DeepInfer plus the matching parser."""
    model_json_schema = getattr(spec, "model_json_schema", None)
    if isinstance(spec, type) and model_json_schema is not None:  # a Pydantic model class
        return json.dumps(model_json_schema()), getattr(spec, "model_validate_json")
    if isinstance(spec, dict):
        return json.dumps(spec), json.loads
    if isinstance(spec, str):
        return spec, json.loads
    raise TypeError(f"json_schema must be a Pydantic model class, a dict, or a JSON string; got {type(spec).__name__}")
