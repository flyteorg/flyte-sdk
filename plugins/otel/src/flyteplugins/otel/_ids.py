"""Deriving a stable OpenTelemetry trace id from a Flyte run.

A durable run can span several containers: it crashes, resumes, retries, and each of those
is a fresh process with a fresh OpenTelemetry SDK. Left alone, every one of them mints its
own random trace id and the backend ends up holding several unrelated traces for what the
user thinks of as a single agent run.

Deriving the trace id from the run identifier fixes that without any coordination. Every
process computes the same 16 bytes from values it already has, so spans recorded before a
crash and spans recorded after the resume land in the same trace even though neither
process ever spoke to the other.

Only the trace id is derived. Span ids stay random, which keeps each attempt a distinct
subtree under the shared trace rather than a set of colliding ids, and means a resumed run
shows its earlier attempts alongside the one that finally succeeded.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flyte.models import ActionID

__all__ = ["format_trace_id", "trace_id_for_run"]

# Namespace prefix so these digests cannot collide with any other use of the same inputs.
_NAMESPACE = b"flyte.otel.run.v1"

# W3C trace ids are 16 bytes and must not be all zero.
_TRACE_ID_BYTES = 16


def trace_id_for_run(action: "ActionID") -> int:
    """The trace id shared by every span in this run, across attempts and containers.

    Derived from the fully qualified run identity rather than the run name alone, so two
    runs that happen to share a name in different projects or domains stay distinct.
    """
    parts = (
        action.org or "",
        action.project or "",
        action.domain or "",
        action.run_name or action.name,
    )
    digest = hashlib.blake2b(
        b"\x00".join(part.encode("utf-8") for part in parts),
        digest_size=_TRACE_ID_BYTES,
        person=_NAMESPACE[:16],
    ).digest()

    # A zero trace id is invalid per the spec and would be silently dropped by exporters.
    return int.from_bytes(digest, "big") or 1


def format_trace_id(trace_id: int) -> str:
    """Render a trace id the way backends display it: 32 lowercase hex characters."""
    return format(trace_id, "032x")
