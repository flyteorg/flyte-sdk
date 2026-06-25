"""Compatibility helpers for the connectrpc ``RequestContext`` API.

The dependency pin is ``connectrpc>=0.9.0,<1.0.0``. In connectrpc < 0.11 the
``RequestContext`` accessors (``request_headers`` and friends) are plain
methods, so callers write ``ctx.request_headers()``. In connectrpc >= 0.11 the
same accessors became ``@property`` attributes, so ``ctx.request_headers`` is
already a ``Headers`` object and calling it raises
``TypeError: 'Headers' object is not callable``.

Because the pin spans both APIs, resolve the accessor defensively instead of
assuming either form.
"""

from __future__ import annotations

from typing import Any


def request_headers(ctx: Any) -> Any:
    """Return the request headers for a connectrpc ``RequestContext``.

    Works across connectrpc < 0.11 (``request_headers`` is a method) and
    connectrpc >= 0.11 (``request_headers`` is a property).
    """
    headers = ctx.request_headers
    return headers() if callable(headers) else headers
