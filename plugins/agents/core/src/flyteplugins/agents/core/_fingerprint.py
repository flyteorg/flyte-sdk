"""Deterministic request fingerprints for keying durable steps."""

from __future__ import annotations

import hashlib
import json
import typing


def fingerprint(payload: typing.Mapping[str, typing.Any]) -> str:
    """A deterministic sha256 hex of a request ``payload``.

    Must be a pure function of the semantic request so replays line up. Pass a
    mapping of the request's identifying fields (e.g. system prompt, input
    items, model, tool names) — not callables or live objects. Anything not
    natively JSON-serializable is coerced with ``str``.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def jsonable(obj: typing.Any) -> typing.Any:
    """Best-effort conversion of an SDK object to something JSON-dumpable."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    for attr in ("to_json_dict", "model_dump"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:  # pragma: no cover - defensive
                pass
    return str(obj)
