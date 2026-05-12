"""
Helper functions for the HITL plugin.
"""

from __future__ import annotations

import os
from typing import Any, Type


def _get_type_name(data_type: Type) -> str:
    """Get a string name for a type that can be used in the form."""
    if data_type is int:
        return "int"
    elif data_type is float:
        return "float"
    elif data_type is bool:
        return "bool"
    elif data_type is str:
        return "str"
    else:
        # For complex types, default to string (JSON serialized)
        return "str"


def _convert_value(value: str, data_type: str) -> Any:
    """Convert a string value to the specified data type."""
    if data_type == "int":
        return int(value)
    elif data_type == "float":
        return float(value)
    elif data_type == "bool":
        return value.lower() in ("true", "1", "yes")
    else:
        # Default to string
        return value


def _get_hitl_base_path() -> str:
    """Get the base path for HITL requests in object storage."""
    return "hitl-requests"


def _get_request_path(request_id: str) -> str:
    """Get the storage path for a HITL request."""
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    if ctx.has_raw_data:
        base = ctx.raw_data.path
    elif raw_data_path_env_var := os.getenv("RAW_DATA_PATH"):
        base = raw_data_path_env_var
    else:
        # Fallback for local development
        base = "/tmp/flyte/hitl"
    return f"{base}/{_get_hitl_base_path()}/{request_id}/request.json"


def _get_response_path(request_id: str) -> str:
    """Get the storage path for a HITL response."""
    from flyte._context import internal_ctx

    ctx = internal_ctx()
    if ctx.has_raw_data:
        base = ctx.raw_data.path
    elif raw_data_path_env_var := os.getenv("RAW_DATA_PATH"):
        base = raw_data_path_env_var
    else:
        # Fallback for local development
        base = "/tmp/flyte/hitl"
    return f"{base}/{_get_hitl_base_path()}/{request_id}/response.json"
