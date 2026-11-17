"""The parts of connectrpc that differ between 0.9/0.10 and 0.11, so the SDK runs on either.

connectrpc 0.11 switched its default Protobuf implementation from google.protobuf to protobuf-py and
turned the `RequestContext` accessors into properties. flyteidl2's messages are google.protobuf ones.
"""

from __future__ import annotations

from typing import Any


def codec_kwargs() -> dict[str, Any]:
    """Client constructor arguments that make it marshal google.protobuf messages."""
    try:
        from connectrpc.compat import google_protobuf_binary_codec
    except ImportError:
        # Before 0.11 google.protobuf is the default codec, and connectrpc.compat does not exist.
        return {}
    return {"codec": google_protobuf_binary_codec()}


def request_headers(ctx: Any) -> Any:
    """The mutable request headers of a `RequestContext`: a method before 0.11, a property since."""
    headers = ctx.request_headers
    return headers() if callable(headers) else headers


def detail_type_name(detail: Any) -> str:
    """The fully-qualified message name of a `ConnectError` detail: an `Any` before 0.11, an `ErrorDetail` since."""
    # `ErrorDetail.type_name` strips only the type.googleapis.com/ prefix, so any other one is still on it.
    type_name = getattr(detail, "type_name", None)
    if type_name is None:
        type_name = detail.type_url
    return type_name.rsplit("/", 1)[-1]
