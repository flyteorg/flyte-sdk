"""The queue that uploads made in the current context are destined for.

A run's code bundle and inputs must land in storage the run's cluster can read. Queues can map to
cluster pools on different clouds (e.g. the project default on GCP, another queue on Azure), so
uploads are routed to the queue's cluster rather than the project/domain default. The runner sets
the queue around submission with `upload_queue(...)`; the dataproxy routing and the code bundle
cache read it with `current_upload_queue()`.

Like any ContextVar it follows `await`s and tasks created inside the block, but not work handed to
another thread (e.g. a sync `flyte.remote.upload_file(...)` call, which syncify runs on its own
loop thread). Inside the block, upload with the `.aio` variants.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_upload_queue: ContextVar[str | None] = ContextVar("_upload_queue", default=None)


@contextmanager
def upload_queue(queue: str | None) -> Iterator[None]:
    """Route uploads made inside this block to `queue`'s cluster storage. A falsy queue keeps the
    project/domain default routing."""
    token = _upload_queue.set(queue or None)
    try:
        yield
    finally:
        _upload_queue.reset(token)


def current_upload_queue() -> str | None:
    return _upload_queue.get()
